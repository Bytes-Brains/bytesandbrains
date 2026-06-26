# Authoring Components

The library-writer + app-extension guide for the `bytesandbrains`
framework. Cross-references:

- [`ROLES.md`](ROLES.md) — role surface reference.
- [`IR_AND_DSL.md`](IR_AND_DSL.md) — DSL → `ModelProto` mapping.
- [`ENGINE.md`](ENGINE.md) — Engine state machine + dispatch.
- [`ANALYSIS.md`](ANALYSIS.md) — 18-pass compiler pipeline.

---

## 1. The three-phase pipeline

Every program walks the same path:

1. **Author** — write a `Module` that records the program shape
   through a `Graph` recorder. `Module::build()` returns one
   `bb_ir::proto::onnx::ModelProto`.
2. **Compile** — `Compiler::new().bind_<role>::<T>("slot")…
   .compile(model)` runs the canonical 18-pass pipeline (see
   `CANONICAL_PASS_NAMES` in `bb-compiler`), stamps the compilation
   passport + binding table onto the model, and returns one
   compiled `ModelProto`.
3. **Install** — `bytesandbrains::install::install(peer_id,
   addresses, compiled, targets: &[&str], Config::new())` verifies
   the passport, reads the binding table for each target,
   deduplicates shared slot bindings across targets, constructs
   each bound concrete via the inventory, and returns a `Node`
   ready to `poll()`. Single-target installs pass `&["MyModule"]`;
   peers hosting multiple partitions (Client + Server on the same
   Node) pass `&["Client", "Server"]` and the install path shares
   one `ComponentRef` per slot the targets jointly declare.

### Minimal end-to-end

The simplest installable program — record a `search` call against
a generic `Index` placeholder, bind a concrete index + a CPU
backend, install, poll.

```rust
use std::task::{Context, Waker};
use bytesandbrains::compiler::Compiler;
use bytesandbrains::framework::Address;
use bytesandbrains::graph::Graph;
use bytesandbrains::ids::PeerId;
use bytesandbrains::install::install;
use bytesandbrains::ops::backends::cpu::CpuBackend;
use bytesandbrains::placeholders::Index;
use bytesandbrains::{Config, Module};

// HnswIndex is the running example from §2 below.
struct SearchApp { index: Index }

impl Module for SearchApp {
    fn name(&self) -> &str { "SearchApp" }
    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let _ = self.index.search(g, query, 3);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Author
    let app = SearchApp { index: Index };
    let model = app.build()?;

    // Compile — bind each slot to a concrete type.
    let compiled = Compiler::new()
        .bind_index::<HnswIndex>("primary_index")
        .bind_backend::<CpuBackend>("compute")
        .compile(model)?;

    // Install
    let target = compiled.functions[0].name.clone();
    let mut node = install(
        PeerId::from(1u64),
        vec![Address::empty()],
        compiled,
        &[target.as_str()],
        Config::new(),
    )?;

    // Drive
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let _ = node.poll(&mut cx);
    Ok(())
}
```

`Module::body(g)` is the only method an author writes. Inputs come
from `g.input("name")`; values are dropped into outputs via
`g.output("name", value)` (local sink) or
`g.net_out("name", peers, value)` (network sink). The body returns
`()`.

---

## 2. Implementing a Contract trait

Concrete components implement one or more user-facing Contract
traits from `bytesandbrains::contracts`:

| Role           | Contract trait                       |
|----------------|--------------------------------------|
| Index          | `bb::Index`                          |
| Aggregator     | `bb::Aggregator`                     |
| Model          | `bb::Model`                          |
| Codec          | `bb::Codec`                          |
| DataSource     | `bb::DataSource`                     |
| PeerSelector   | `bb::PeerSelector`                   |
| Backend        | `bb::Backend`                        |

### Memory boundaries

Components NEVER see allocation failures from wire / app
ingress. The framework's boundary callers
(`Engine::decode_typed_fill`, `Node::deliver_event`,
`Node::invoke`, `CompletionSink::complete`) cap, charge against
`NodeConfig::ingress_byte_budget`, and fallibly reserve
framework-owned storage BEFORE handing payloads to a Contract
method. An allocation failure surfaces as
`InfraEvent::WireReceiveError::AllocationFailed` (wire) or
`InfraEvent::AppIngressError::AllocationFailed` (app) — the
offending bytes drop at the boundary and the Contract method
never runs. See [WIRE.md §Principle 1a](WIRE.md#principle-1a-external-byte-payloads-cross-as-u8)
+ [ENGINE.md §6.7](ENGINE.md#67-engine-boundary-fallibility-line)
for the boundary contract.

**Inside a Contract method, normal Rust allocation patterns
apply.** Components are designed to play by the runtime
contract: a `Vec::push` that runs out of memory is a process
abort, not a framework-handled failure. Components needing
graceful degradation under memory pressure handle that inside
their own implementation.

**Backend authors own tensor-materialisation budget.** Wire-bytes
land inside `Backend::materialize_from_wire(type_hash, bytes: Vec<u8>)`
(`bb-runtime/src/contracts/backend.rs:497`); the framework
already charged `bytes.len()` against the ingress byte budget
and moved ownership of the `Vec<u8>` into the call. The backend
chooses the materialisation strategy (zero-copy adoption via
`ArrayD::from_shape_vec`, pool-pulled buffer + copy in,
fresh-allocate). Returning `Err` drops the fill, releases the
charge, and emits `WireReceiveError::BackendMaterializeFailed`
on the bus. See [ROLES.md §Backend-owned tensor memory](ROLES.md#backend-owned-tensor-memory)
for the lifecycle.

Each Contract is a method-per-op surface that returns a typed
`ContractResponse<R, E>`:

- `ContractResponse::Now(Ok(value))` — result is ready inline. The
  framework returns
  `DispatchResult::Immediate(vec![(port, Box::new(value) as Box<dyn SlotValue>)])`,
  boxes `value` straight into the slot table (no bincode at this
  boundary; downstream ops downcast via `as_any`), and skips the
  park/ingress cycle.
- `ContractResponse::Later` — the impl retained the
  `CompletionHandle` (sent it to a worker thread, queued a remote
  RPC). The framework returns `DispatchResult::Async(cmd_id)` and
  parks the dispatched op until the impl calls
  `handle.complete(result)` from off-thread.

### Worked example: an async Index

`bb::Index` declares three methods: `add`, `search`, `remove`.
This impl shells out to a worker thread that owns the actual HNSW
data; each method ships work to the worker and returns `Later`
while keeping the handle alive:

```rust
use std::sync::{mpsc, Arc, Mutex};
use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::contracts::Index as IndexContract;
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
struct HnswIndex {
    capacity: u32,
    #[serde(skip)]
    tx: Arc<Mutex<Option<mpsc::Sender<WorkItem>>>>,
}

enum WorkItem {
    Add    { vec: Vec<f32>,   completion: CompletionHandle<u64, HnswError> },
    Search { query: Vec<f32>, k: u32,
             completion: CompletionHandle<Vec<(u64, f32)>, HnswError> },
    Remove { completion: CompletionHandle<(), HnswError> },
}

impl IndexContract for HnswIndex {
    type Vector = [f32];    // TYPE_TENSOR_F32 — f32-native index
    type Error = HnswError;

    fn add(
        &mut self,
        vec: &Self::Vector,
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error> {
        self.send(WorkItem::Add { vec: vec.to_vec(), completion });
        ContractResponse::Later
    }

    fn search(
        &self,
        query: &Self::Vector,
        k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error> {
        self.send(WorkItem::Search { query: query.to_vec(), k, completion });
        ContractResponse::Later
    }

    fn remove(
        &mut self,
        _id: u64,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        self.send(WorkItem::Remove { completion });
        ContractResponse::Later
    }
}
```

The worker thread owns the HNSW state and pushes `Completion`
results back onto the engine's ingress queue. The next call to
`node.poll(cx)` drains the ingress and resumes every parked op.
The complete example lives at
[`examples/custom_index_hnsw.rs`](../examples/custom_index_hnsw.rs).

### Authoring a trainable Index

`bb::Index` ships a default-no-op `train(samples)` body so
fixed-structure indexes (flat, HNSW with a streaming insert
path) compile without writing anything. Indexes whose accuracy
depends on a fitted partition or codebook override `train` and
keep the fitted state on the struct. IVF stores its centroids
there; PQ stores its per-sub-vector codebooks. The signature
mirrors the other `Index` methods exactly so a `Later` worker
hand-off composes with the same `CompletionHandle` machinery
the rest of the surface uses.

```rust
#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
struct IvfIndex {
    n_centroids: u32,
    #[serde(skip)]
    centroids: Vec<Vec<f32>>,
}

impl IndexContract for IvfIndex {
    type Vector = [f32];
    type Error = IvfError;

    fn train(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        samples: &[&Self::Vector],
        _completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        // Fit coarse k-means over `samples`; persist the result
        // on `self.centroids` so subsequent `add` / `search`
        // calls can route through the trained partition.
        self.centroids = kmeans(samples, self.n_centroids as usize);
        ContractResponse::Now(Ok(()))
    }

    fn add(&mut self, ctx, vec, completion)         { /* assign to nearest centroid */ }
    fn search(&self, ctx, query, k, completion)     { /* probe top-N centroids */ }
    fn remove(&mut self, ctx, id, completion)       { /* … */ }
}
```

Record the training call inside `Module::bootstrap` so the
engine drains it before body-phase `add` / `search` ops fire.
The DSL surface is `IndexSlot::train(g, samples) -> Output`;
the returned `TYPE_TRIGGER` handle can also feed a `bb.barrier`
when an author needs body-phase ordering without the bootstrap
phase.

```rust
impl Module for VectorStore {
    fn bootstrap(&self, g: &mut Graph) {
        let training_corpus = self.data.next_batch(g).0;
        let _ = self.index.train(g, training_corpus);
    }
    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let _ = self.index.search(g, query, 10);
    }
}
```

The host kicks the recorded bootstrap explicitly:

```rust
let mut node = bytesandbrains::install(
    peer_id, vec![Address::empty()], compiled,
    &["VectorStore"], Config::new(),
)?;

// Host-driven: install does not auto-fire. Drive the bootstrap
// queue to quiescence before the body poll loop runs.
let _ = node.run_bootstrap(BootstrapTarget::All)?;

// Body-phase work after bootstrap drains.
node.deliver_event("VectorStore", "query", &query_bytes)?;
```

The per-component `is_op_locked` gate
(`bb-runtime/src/engine/core.rs:1762-1806`) holds body-phase
`add` / `search` ops behind the bootstrap call's touch set —
the closure of every `ComponentRef` the bootstrap body reaches.
Disjoint components keep firing during the bootstrap. The body
phase resumes once the bootstrap's in-flight entry retires.

A heavy training pass that should not block the engine thread
ships the samples to a worker exactly like `add` does — capture
`completion` on a `WorkItem`, return `ContractResponse::Later`,
call `completion.complete(Ok(()))` from the worker when the
fitted state lands. The bootstrap's `pending_async` entry
clears, the in-flight entry retires, and the body-phase ops
fire.

`bb::Codec::train` follows the identical pattern: PQ codecs
keep `codebooks: Vec<Vec<[f32; D]>>` on the struct and fit
them inside `train(samples)`; an affine int8 quantizer
computes `(scale, zero_point)` from the corpus and stores
them on `self`.

### Authoring a Component-level bootstrap

When the one-shot setup needs Rust code rather than recorded
graph ops, override `bb::Bootstrap` alongside the primary
Contract. Every `#[derive(bb::Concrete)]` type already
participates in the Component bootstrap dispatch path via the
trait's no-op default — override to allocate pools, mmap state,
prime calibration caches, or dial seed peers.

```rust
use bytesandbrains::contracts::bootstrap::{Bootstrap, BootstrapCtx};

#[derive(bb_derive::Concrete, bb_derive::Backend)]
#[bootstrap_override]
struct PinnedHostBackend {
    pool: HostBufferPool,
}

impl Bootstrap for PinnedHostBackend {
    type Error = AllocError;

    fn bootstrap(&mut self, _ctx: &mut BootstrapCtx) -> Result<(), AllocError> {
        // One-shot pinned-buffer pool allocation. Body-phase
        // kernels read through `self.pool`.
        self.pool.prime(/* config */)?;
        Ok(())
    }
}
```

`#[bootstrap_override]` on the struct
(`bb-derive/src/parse.rs:36-48`) suppresses the derive's default
no-op impl so the hand-written one does not collide. The derive
still emits the `BootstrapDispatcherRegistration` inventory
entry, so `install()` wires the dispatcher
(`src/install.rs:451-466`) without naming the type at the call
site.

The host fires the override explicitly through the slot the
binding chain bound the concrete onto:

```rust
node.run_bootstrap(BootstrapTarget::Slots(&["compute"]))?;
// or batch several slots in one call:
node.run_bootstrap(BootstrapTarget::Slots(&["compute", "primary_index"]))?;
```

Prefer Component bootstrap over Module bootstrap when:

- The setup is **Rust-side state** — buffer pools, file handles,
  mmap regions, kernel caches — that no body op needs to *see*
  as a graph value.
- The setup is **per-instance** — one Component, one
  initialization — rather than per-Module-target. The slot
  granularity matches the resource lifetime exactly.
- The setup needs the broader Rust runtime — system calls,
  filesystem access, GPU context creation — and recording it
  inside `Module::bootstrap` would force a placeholder syscall
  that ultimately reaches Component code anyway.

Prefer Module bootstrap when the setup must compose with the
graph: an `Index::train(samples)` call needs a `DataSource` to
produce the samples, and the recording in `Module::bootstrap`
expresses that composition naturally. Module bootstrap also
wins when the setup spans several Components — record one
`Module::bootstrap` body that orchestrates them, rather than
dispatching N Component bootstraps and re-implementing the
composition outside the IR.

### Sync-only Contracts

When a Contract method's work is purely in-memory, return
`ContractResponse::Now(Ok(value))` and ignore the handle:

```rust
impl IndexContract for CountingIndex {
    type Vector = [f32];    // TYPE_TENSOR_F32
    type Error = bytesandbrains::bus::OpError;

    fn search(
        &self,
        _query: &Self::Vector,
        _k: u32,
        _c: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error> {
        ContractResponse::Now(Ok(Vec::new()))
    }
    // …
}
```

### Declaring storage type

Every tensor-carrying Contract has one `Storage`-bound associated
type that declares where in the tensor-type tree the concrete sits.
The compiler reads `Storage::TYPE` at bind time to stamp port
denotations; the type solver then refuses unbridged mismatches.

**Specialized leaf** — a HNSW that only understands f32 and computes
distances natively:

```rust
impl Index for FlatHnsw {
    type Vector = [f32];   // → TYPE_TENSOR_F32 (leaf)
    // …
}
```

**Algorithm-class generic** — a HNSW that outsources all numeric ops
to a bound `Backend`. The non-leaf `AnyTensor` position means any
tensor subtype unifies into it:

```rust
#[derive(bb::Concrete, bb::Index)]
#[depends(backend = "compute")]
struct GenericHnsw { /* graph state */ }

impl Index for GenericHnsw {
    type Vector = AnyTensor;   // → TYPE_TENSOR (non-leaf)
    fn add(&mut self, v: &AnyTensor, completion, ctx) {
        let backend = ctx.dependency::<CpuBackend>("compute");
        // delegate numeric ops to the bound backend
        …
    }
}
```

**Custom packed type** — a library maker introducing a new dtype
registers a `TypeNode` and `impl Storage`:

```rust
pub struct Int4Packed(pub Vec<u8>);
impl bb_ir::types::Storage for Int4Packed {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_I4;
}

impl Index for Int4Index {
    type Vector = Int4Packed;   // → TYPE_TENSOR_I4 (user-registered leaf)
}
```

**Wiring a `Codec`** — if an upstream output type doesn't unify with a
downstream port type the compiler reports an `IncompatibleStorageOnEdge`
error with a hint ("Insert a `Codec<In=…, Out=…>` on the edge."). The
author wires a `Codec` node explicitly:

```rust
// f32 → u8 affine quantizer
#[derive(bb::Concrete, bb::Codec)]
struct Int8AffineQuantizer { scale: f32, zero_point: i32 }

impl Codec for Int8AffineQuantizer {
    type In  = [f32];           // TYPE_TENSOR_F32
    type Out = [u8];            // TYPE_TENSOR_U8
    type Error = QuantizeError;
    fn train(&mut self, samples, …)  { /* calibrate scale + zero_point */ }
    fn encode(&self, x, …)           { /* affine quantize f32 → u8 */ }
    fn decode(&self, y, …)           { /* affine dequantize u8 → f32 */ }
}
```

---

## 3. The derives

A concrete component carries two kinds of derives:

```rust
#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
struct HnswIndex { /* … */ }
```

- **`#[derive(bb::Concrete)]`** — emits the universal triple every
  concrete needs:
  - `impl ConcreteComponent` with `TYPE_NAME`,
    `PACKAGE = Application`, `Config = ()`, `Error =
    Infallible`, `new(&()) → Self::default()`, plus bincode-backed
    `serialize` / `restore`.
  - `impl AnyComponent` (the erased component plumbing).
  - `inventory::submit!{ ConcreteComponentEntry { … } }` — submits
    a `(TYPE_NAME, construct_fn, serialize_fn, restore_fn,
    package, dependencies)` carrier to the global registry. The
    install path looks the entry up by `TYPE_NAME`.
- **`#[derive(bb::<Role>)]`** — one per role the concrete plays
  (`Index`, `Aggregator`, `Model`, `Codec`, `DataSource`,
  `PeerSelector`, `Backend`). Each emits the bridge from the
  user-facing Contract trait the author implemented to the engine-
  internal `<Role>Runtime` trait the engine dispatches through. The
  `<Role>Runtime` traits live in `bb-runtime/src/roles/` and are
  framework-internal — library authors do not implement them by
  hand. The derive also submits an
  `inventory::submit!{ DispatcherRegistration { … } }` carrier
  binding `(TYPE_NAME, role)` to the engine-side register fn.

`#[derive(bb::Concrete)]` requires the struct to carry
`Default + Serialize + Deserialize`. Components that need a
non-trivial `Config` or a fallible constructor write
`impl ConcreteComponent for Self { … }` by hand instead.

---

## 4. Declaring sibling dependencies

A concrete declares the slots it reads at dispatch time through a
`#[depends(…)]` attribute on the struct:

```rust
#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
#[depends(backend = "compute")]
struct CountingIndex { bias: u32 }
```

The `Concrete` derive parses the attribute into
`ConcreteComponent::DEPENDENCIES`. The compiler walks every bound
component's `DEPENDENCIES` slice during
`resolve_component_dependencies` and refuses to compile a model
where a declared slot isn't bound to a sibling concrete of the
right role — surfacing `CompileError::UnboundDependency` /
`DependencyRoleMismatch`. The compiler also stamps
`ai.bytesandbrains.dep.Backend = "compute"` onto every NodeProto
the declaring concrete contributes.

At dispatch time, reach the bound sibling through the engine's
`RuntimeResourceRef`. Contract methods that take a `ctx` argument
(`PeerSelector::select`, `PeerSelector::sample`,
`PeerSelector::current_view`) call `ctx.dependency::<T>(slot)`
directly. Contracts that don't carry `ctx` (Index, Aggregator,
Model, Codec, DataSource, Backend) read sibling slots via
the engine's per-component `RuntimeResourceRef` surfaced through
the role bridge:

```rust
let backend = ctx.dependency::<CpuBackend>("compute")
    .expect("compute slot is verified at compile time");
// backend.matmul(…), backend.add(…), …
```

The `expect` is sound because the compiler refused to install a
configuration where the slot was missing. See
[`examples/component_with_dependency.rs`](../examples/component_with_dependency.rs).

Roles available inside `#[depends(...)]`: `index`, `aggregator`,
`model`, `codec`, `data_source`, `peer_selector`, `backend`,
`protocol`.

---

## 5. Registering custom atomic ops + protocols

Two declarative macros handle the cases the Contract surface
doesn't cover.

### `bb::register_op!{}`

When a custom op doesn't fit a role Contract — a syscall-style
side-effect, a domain-specific primitive — register an invoke fn
directly:

```rust
bytesandbrains::register_op! {
    domain: "myapp.ops",
    op_type: "Foo",
    invoke: invoke_foo,
}

fn invoke_foo(/* canonical engine-side invoke signature */) { /* … */ }
```

The macro emits one `inventory::submit!{ OpRegistration { … } }`.
The engine consumes the registry during install + routes every
NodeProto whose `(domain, op_type)` matches to `invoke_foo`.

### `bb::register_protocol!{}`

When a custom networked protocol fits the Protocol role's atomic
opset model (Kademlia-style FindNode / Ping / …), the macro writes
the struct + serde + `ConcreteComponent` + `AnyComponent` +
`ProtocolRuntime` + `atomic_opset` + `dispatch_atomic` +
inventory submission in one block:

```rust
bytesandbrains::register_protocol! {
    struct Kademlia { routing_table: Vec<u64>, k: usize }
    domain: "bb-kademlia.kademlia.atomic"
    version: 1
    ops {
        FindNode,
        Ping,
    }
}
```

The macro emits the universal triple plus a `ProtocolRuntime` impl
whose `dispatch_atomic` returns `DispatchResult::Immediate(vec![])`
for each declared op — the author fills in the bodies by hand
after the macro expands (or hand-writes the trait if the per-op
arms need real logic). There is no `#[derive(bb::Protocol)]`; the
macro IS the protocol authoring surface.

### Bundling typed values into one composite Output

When a Module needs to ship more than one typed value across a
single `net_out` port, `Graph::bundle` + `Graph::unbundle` pack N
typed Outputs into one `CompositeValue` envelope and decompose
them back on the receiver. The pattern:

```rust
fn body(&self, g: &mut Graph) {
    let params = ModelSlot.params(g);          // TYPE_TENSOR_F32
    let owner  = Output::new("owner".into(), &TYPE_PEER_ID);
    let payload = g.bundle(&[params, owner]);  // one composite Output
    g.net_out("update", peers, payload);

    let received = g.lookup_output("update").expect("port registered");
    let parts = g.unbundle(received, &[&TYPE_TENSOR_F32, &TYPE_PEER_ID]);
    let recovered_params = parts[0].clone();
    let recovered_owner  = parts[1].clone();
    // … both downstream consumers downcast directly to their
    // concrete carrier; no `BytesValue` indirection.
}
```

In-process forwarding pays one `SlotValue::clone_boxed` per child
(`bb-runtime/src/syscall/values.rs:87-93`) — no bincode encode,
no decode. The wire codec on `CompositeValue`
(`bb-runtime/src/syscall/values.rs:114-165`) only fires when the
envelope crosses a Node boundary, at which point each child is
serialised through its own `SlotValue::to_wire_bytes` and the
receiver materialises typed carriers via `wire_decoder_registry()`
(`bb-ir/src/slot_value.rs:199-212`).

Authoring rule for ship-able value carriers: every type that may
ride a `CompositeValue` child slot OR a wire payload registers
itself once via `bb_ir::slot_value::register_type_node!`. The macro
emits both the lattice TypeId binding and the decoder for the
cross-Node receive path; one invocation per carrier, no manual
`SlotValue` impl required:

```rust
use bb_ir::slot_value::register_type_node;
use bb_ir::types::TYPE_TENSOR_F32;

register_type_node!(MyTensorF32, &TYPE_TENSOR_F32);
```

The blanket `SlotValue` impl
(`bb-ir/src/slot_value.rs:262-285`) requires `Clone + Serialize +
DeserializeOwned + Send + Sync + 'static`. Unregistered carriers
ride in-process fine but cannot be decoded after a wire hop on a
peer that did not see the `register_type_node!` line; the decoder
registry surfaces a typed `SlotValueError::UnknownTypeHash` when
the receiver lacks the binding.

---

## 6. Inventory + DCE

The framework wires concrete components, role dispatchers, and
custom ops through a single mechanism: `inventory::submit!`. Each
derive + each `register_op!{}` + each `register_protocol!{}`
expansion emits one or more `submit!{}` blocks. The global registry
collects every submitted entry at process start.

**Rust DCE strips submissions whose containing crate is never
referenced by a function symbol.** This is the framework's binary-
size story: components in unreferenced crates pay nothing. It's
also a footgun — a transitively-depended-on `rlib` whose entries
are never touched by user code will have its `submit!{}` blocks
DCE'd along with the rest of the object file.

The framework solves this for its own components: `bb-ops` exports
`link_force()`, a function that takes one `black_box(fn as usize)`
per inventory-bearing module. The first call inside
`bb::install::install` invokes `bb_ops::link_force()`, anchoring
every `bb-ops` object file across the rlib boundary. User crates
that ship their own components and worry about DCE follow the
same pattern: export a `link_force()` that black-boxes one fn
pointer per `submit!{}`-bearing module + call it from user code.

---

## 7. The install path

`bytesandbrains::install::install` is the single Node construction
entry point. It takes:

```rust
pub fn install(
    peer_id: PeerId,
    addresses: Vec<Address>,
    model: ModelProto,
    targets: &[&str],
    config: Config,
) -> Result<Node, InstallError>;
```

The install path
(`src/install.rs:235-338`):

1. Calls `bb_ops::link_force()` to anchor the framework's
   inventory carriers across the rlib boundary.
2. **Rejects empty `targets`** — `targets: &[]` returns
   `InstallError::EmptyTargets` before any work
   (`src/install.rs:245-247`).
3. **Verifies the compilation passport** — checks the model
   carries `metadata_props["ai.bytesandbrains.compiled"] = "v1"`.
   Without it: `InstallError::NotCompiled`. Mismatched version:
   `InstallError::IncompatibleCompiledVersion`.
4. **Resolves each target name** — for every entry in `targets`,
   looks up the function against `model.functions[]` (exact match
   wins; falls back to the compiler's content-hash-suffixed
   `<target>#<hash>` form) and collects the resolved name.
   Missing: `InstallError::UnknownTarget { available }`
   (`src/install.rs:251-259`).
5. **Parses the binding table per target** — walks
   `metadata_props["ai.bytesandbrains.binding.<resolved>.<slot>"]
   = "<role>|<TYPE_NAME>|<slot_id|-1>"` for every entry matching
   each resolved target.
6. **Dedups slot bindings across targets** — groups every target's
   bindings by slot name in first-seen call order; any slot whose
   contributors disagree on `(TYPE_NAME, role)` fails with
   `InstallError::SlotBindingConflict { slot, conflicts }`. The
   `Display` impl enumerates every contributor in call order so
   the error message points at every target that named the
   conflicting slot (`src/install.rs:524-571`).
7. **Constructs each bound concrete exactly once** via the
   inventory — looks up `TYPE_NAME` in the global registry,
   downcasts the per-slot config to the concrete's `Config`
   associated type, calls the inventory's `construct_fn`. Each
   slot allocates one `ComponentRef`, shared across every target
   that names the slot. `Config::with(slot, value)` supplies
   values for non-`()` configs; slots whose concrete declares
   `type Config = ()` get `&()` automatically.
8. **Registers the resulting instance** with the engine, stamps
   the slot binding, and registers every role's dispatcher.
9. **Installs every target** as an entry-point graph (marked
   `is_entry_point = true` so top-level outputs surface as
   `EngineStep::AppEvent`); carries every other function in the
   function library so cross-Module calls resolve at dispatch.
10. **Shares the `Arc<ModelProto>` across targets.** `Node::set_model`
    wraps the proto once; `Node::register_module` runs per target,
    so the host can call `Node::deliver_event(target, input,
    bytes)` for any registered target name and the proto bytes
    live on the Node exactly once.
11. **Returns the Node** with the engine resolved + ready to
    `poll(cx) -> Poll<Vec<EngineStep>>`.

### Single-target installs

The slice form covers the single-target case verbatim:

```rust
let target = compiled.functions[0].name.clone();
let node = install(
    peer_id,
    vec![Address::empty()],
    compiled,
    &[target.as_str()],
    Config::new(),
)?;
```

One target → one entry in `Node::module_index` → one bootstrap
queued (if the target declares one) → unchanged observable
behaviour. Pre-1.0 means there is no `install_single` shim; callers
update the single call site.

### Multi-role on one Node — host both partitions of a federated round

A peer hosting both `Client` and `Server` partitions from one
compile passes both names in install order:

```rust
let mut node = install(
    PeerId::from(1u64),
    vec![Address::empty()],
    compiled,
    &["Client", "Server"],
    Config::new()
        .with("backend", CpuConfig::default())
        .with("aggregator", FedAvgConfig::default()),
)?;

// Drive the host-facing entry points per target.
node.deliver_event("Client", "train_step", &client_bytes)?;
node.deliver_event("Server", "aggregate",  &server_bytes)?;
```

Mechanics:

- The compiler emitted `Client` and `Server` as sibling partitions
  of the same `ModelProto`; both stamp
  `binding.<target>.backend = "Backend|CpuBackend|<slot_id>"` and
  `binding.<target>.aggregator = "Aggregator|FedAvg|<slot_id>"` so
  the dedup walk converges on one `CpuBackend` instance + one
  `FedAvg` instance shared by both partitions
  (`src/install.rs:267-324,524-571`).
- Bootstrap functions queue in slice order. Install records
  every `module_phase = "bootstrap"` FunctionProto on
  `BootstrapState::install_order` without arming the queue; the
  host calls `node.run_bootstrap(BootstrapTarget::All)` to drive
  every queued bootstrap to quiescence in slice order — `Client`'s
  bootstrap fires first, runs to quiescence, emits its
  `EngineStep::BootstrapComplete`, then `Server`'s bootstrap
  fires (`bb-runtime/src/engine/core.rs:1211-1239`,
  `bb-runtime/src/engine/bootstrap.rs:256-296`). Single-target
  installs are the length-1 case of the same path.
- Each target registers as its own `Node::module_index` entry, so
  `deliver_event("Client", ...)` and `deliver_event("Server", ...)`
  route to different entry-point graphs. Top-level outputs surface
  as `EngineStep::AppEvent { module_name: "Client" | "Server",
  output_name, value }` so the host distinguishes which partition
  produced an event.
- Targets that need distinct concrete instances for the same role
  (e.g. one `FedAvg` for `Client`, a separate `FedAvg` for
  `Server`) wire two slot names at compile time
  (`bind_aggregator::<FedAvg>("client_agg")` +
  `bind_aggregator::<FedAvg>("server_agg")`) — two slots, two
  `ComponentRef`s, no sharing.

### Supplying per-slot config

```rust
let cfg = Config::new()
    .with("primary_index", HnswConfig { dim: 4, capacity: 64 })
    .with("compute", CpuConfig::default());

let node = install(peer_id, addresses, compiled, &targets, cfg)?;
```

`Config::with` downcasts the value to the bound concrete's
`Config` associated type at install. Type mismatches surface as
`InstallError::ConfigTypeMismatch { slot, type_name, detail }`.
Per-slot configs apply to the single shared `ComponentRef` for that
slot — every target jointly declaring the slot sees the same
configured instance.

---

## 8. Putting it together

Library authors ship one crate per concrete (`bb-kademlia`,
`bb-faiss`, `bb-burn-backend`) carrying the `Contract` impl + the
`#[derive(bb::Concrete)]` + `#[derive(bb::<Role>)]` derives.
Application code depends on the publishable crates:

- `bytesandbrains` — facade re-exporting every surface
- `bb-ir` — IR foundation (proto, wire envelope, ids)
- `bb-ops` — every concrete component the framework ships
- `bb-dsl` — `Module` trait + `Graph` recorder + Contract re-exports
- `bb-compiler` — 18-pass compiler pipeline
- `bb-runtime` — Engine, Node, Contract traits, role dispatchers
- `bb-derive` — proc-macros

…depends on whichever component crates it needs, authors a
`Module`, compiles it with the right `bind_<role>::<T>` chain, and
hands the result to `install`. The same workflow covers a
single-process integration test, a federated training run, and an
embedded sensor.
