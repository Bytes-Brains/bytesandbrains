# Contract dispatch — the canonical user-facing surface

The Contract trait family
(`bb_dsl::contracts::{Index, Aggregator, Model, Codec,
DataSource, PeerSelector, Backend, Protocol}`) is the
**canonical** authoring surface a library maker writes when
shipping a concrete component. The framework's `bb-derive`
proc-macros bridge each Contract impl to the framework-internal
`<Role>Runtime` trait the runtime dispatches through (specified in
[ROLES.md](ROLES.md)); library makers do **not** implement
`<Role>Runtime` directly.

Every Contract method takes the same trio:

1. `ctx: &mut RuntimeResourceRef<'_>` — the per-dispatch runtime
   surface. Contract impls reach declared `#[depends(...)]`
   siblings through `ctx.dependency::<T>("<slot>")`, walk the
   address book through `ctx.peers.addresses`, plan delayed
   work through `ctx.time`, and mint completion handles through
   `ctx.open_completion::<R, E>()`.
2. The op's typed inputs (the same primitives the role-DSL
   placeholder records).
3. `completion: CompletionHandle<R, Self::Error>` — the handle
   the impl retains when it needs to complete the op off-thread.

The method returns `ContractResponse<R, Self::Error>`:

- `ContractResponse::Now(Ok(value))` — result is ready inline. The
  framework returns
  `DispatchResult::Immediate(vec![(port, Box::new(value) as Box<dyn SlotValue>)])`
  — `value` lands in the slot table as `Box<dyn SlotValue>` with no
  serialization at this boundary (downstream ops downcast via
  `as_any`; bincode fires only at the wire boundary in
  `SlotValue::to_wire_bytes`, in `CompletionHandle::complete` for
  cross-thread delivery, and at snapshot save/restore) — and skips
  the park / ingress-drain cycle. The `CompletionHandle` is ignored.
- `ContractResponse::Later` — the impl retained the handle (sent it
  to a worker thread, spawned a tokio task, queued a remote RPC).
  The framework returns `DispatchResult::Async(cmd_id)` and parks
  the dispatched op until the user calls
  `handle.complete(result)` from off-thread, at which point
  `IngressEvent::Completion` lands on the Node's ingress queue and
  the engine unparks the op.

The derive emits `dispatch_atomic` arms that:

1. Downcast each input `&dyn SlotValue` to the expected primitive
   (typed-failure `OpError` on mismatch — never silent default).
2. Open a `CompletionHandle` via `ctx.open_completion::<R, E>()`.
3. Call the user's Contract method, forwarding `ctx`, the typed
   inputs, and the handle.
4. Translate the returned `ContractResponse<R, E>` into the
   matching `DispatchResult`.

The same dispatch path runs whether the op fires from a `body`
function or a `bootstrap` function — bootstrap is a regular
`FunctionProto` whose body ops the engine seeds onto the frontier
under a fresh `ExecId` when the host kicks the queue via
`Node::run_bootstrap(BootstrapTarget::*)`. The per-component
`is_op_locked` gate
(`bb-runtime/src/engine/core.rs:1762-1806`) parks body-phase ops
that touch any in-flight bootstrap's `ComponentRef` touch set
until the bootstrap drains. Contract methods see identical
`RuntimeResourceRef` semantics in either phase; `Later`
completions land through the same ingress queue.

The dispatch path also does not change in the multi-target install
case. `bb::install(.., targets: &[&str], ..)` deduplicates slot
bindings across targets so each slot has exactly one `ComponentRef`
on the Node, no matter how many targets named the slot
(`src/install.rs:524-571`). A Contract method dispatched from
target `A`'s body and a Contract method dispatched from target `B`'s
body route to the **same** `ComponentRef` when the slot binding
matches; the impl sees one `&mut self` (or `&self`) sequence of
calls across both partitions, in cycle-order, on the engine
thread. Mutable state the Contract impl keeps on the struct is
shared by every target sharing the slot — by design. Authors who
need per-target instances bind two distinct slot names at compile
time (see [ROLES.md "Role bindings shared across install targets"](ROLES.md#role-bindings-shared-across-install-targets)).

## Reaching declared dependencies via `ctx`

`ctx.dependency::<T>("<slot>")` is the canonical way a Contract
impl reaches a sibling concrete declared via `#[depends(<role> =
"<slot>")]`. The accessor downcasts the bound `ErasedComponent` to
`&T` and returns `Result<&T, DependencyError>`. The
`resolve_component_dependencies` compiler pass verifies at compile
time that every declared `#[depends(...)]` slot maps to a bound
concrete of the right role, so a successful `Compiler::compile`
implies an `Ok` here; `.expect("compiler verified ...")` is the
intended call site.

```rust
impl Aggregator for FedAvg<B> {
    fn aggregate(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        _completion: CompletionHandle<(Box<Self::Element>, Self::Metadata), Self::Error>,
    ) -> ContractResponse<(Box<Self::Element>, Self::Metadata), Self::Error> {
        let backend = ctx
            .dependency::<B>("backend")
            .expect("resolve_component_dependencies verified `backend` slot");
        let scaled = backend.mul(&tensor, &weight)?;
        // ...
    }
}
```

The same pattern flows through every role: an Index whose
distance kernel composes onto a `Backend`, a Codec whose
calibration tensors land on a device-side `Backend`, a Model
whose checkpoint snapshots stream through a `Backend`'s tensor
serializer, etc.

## Architecture

### `CompletionHandle<R, E>`

Defined in `bb-runtime/src/completion.rs`. Carries:

- `cmd_id: CommandId` — the engine parked the dispatched op behind
  this id.
- `sink: Arc<dyn CompletionSink>` — abstract completion delivery.
  `bb-runtime` implements `CompletionSink` on `IngressQueue`; the
  user's Contract method never sees the concrete sink type.

The handle is `Send + Sync` so it crosses thread boundaries freely.

### `CompletionSink` trait

```rust
pub trait CompletionSink: Send + Sync {
    fn complete(&self, cmd_id: CommandId, result_bytes: &[u8]);
    fn fail(&self, cmd_id: CommandId, detail: &str);
}
```

(`bb-runtime/src/completion.rs`.) The `&[u8]` / `&str` signature
follows [WIRE.md §Principle 1a](WIRE.md#principle-1a-external-byte-payloads-cross-as-u8):
result bytes cross the application-ingress boundary as borrowed
slices and the framework copies them into a framework-owned
`Vec<u8>` inside the `IngressQueue` impl
(`bb-runtime/src/runtime.rs:29-97`).

The bytes are bincode + serde encoding of the user's `R` type. The
receiving end (`Engine::handle_completion`) wraps each owned
`Vec<u8>` as a `BytesValue` and writes it to the parked op's output
sites.

`complete`'s implementation runs `NodeConfig::max_completion_result_bytes`
cap, charges against `Engine::ingress_byte_budget`, and
`try_reserve_exact`s framework-owned storage. Cap / alloc
failures publish
`InfraEvent::AppIngressError { source: Completion { command },
byte_count, kind: PerItemCapExceeded | AllocationFailed }`
and drop the result; the parked op times out naturally on the
host side. `fail`'s `detail` is truncated to
`COMPLETION_DETAIL_CAP` (4 KiB) at a UTF-8 character boundary
rather than rejected so the host's display message always
lands.

**Dispatch is allocation-free.** Once a Contract method runs
(inline `ContractResponse::Now` or deferred via the handle), the
framework does not allocate ingress storage on the dispatch
path — the engine already charged for the inbound bytes at the
boundary. Completion-side budgets are accounted at ingress
(when `CompletionSink::complete` arrives) so a parked op's
result-byte reservation happens once, not per dispatch
iteration.

### `RuntimeResourceRef::open_completion`

`bb-runtime` exposes:

```rust
pub fn open_completion<R, E>(&mut self) -> CompletionHandle<R, E>
where R: serde::Serialize, E: std::fmt::Display
```

Mints a fresh `CommandId` + clones the engine's ingress `Arc` into
the handle. The dispatcher captures the cmd_id before passing the
handle to the user's method.

### Trait shape

Every Contract method follows the same skeleton — take `ctx` as
the first positional, then the op's typed inputs, then a
`CompletionHandle` for deferred completion, and return a
`ContractResponse` so inline results skip the park cycle:

```rust
pub trait Index: Send + Sync {
    type Vector: ?Sized + bb_ir::types::Storage;
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    fn add(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        vec: &Self::Vector,
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error>;

    fn search(
        &self,
        ctx: &mut RuntimeResourceRef<'_>,
        query: &Self::Vector,
        k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error>;

    fn remove(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        id: u64,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    fn train(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        samples: &[&Self::Vector],
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> { ContractResponse::Now(Ok(())) }
}
```

### Train dispatch path

`Index::train` and `Codec::train` route through the same
`dispatch_atomic` arm as their sibling ops: `IndexRuntime` /
`CodecRuntime` each declare a `Train` entry in their opset
alongside `Add` / `Search` / `Remove` (Index) and `Encode` /
`Decode` (Codec). The derive's `dispatch_atomic` body downcasts
the `samples` input slot to `Vec<Box<[f32]>>` (matching the
`type Vector = [f32]` / `type In = [f32]` carrier the framework
ships), rebuilds it as `&[&[f32]]` borrow-style, opens a
`CompletionHandle<(), Self::Error>` via
`ctx.open_completion::<(), E>()`, and forwards into the
Contract's `train` body. The translation from
`ContractResponse<(), E>` to `DispatchResult` is identical to
the other arms (`Now(Ok)` collapses to `Immediate(vec![])`,
`Later` carries the `cmd_id`). Impls with non-`[f32]` vectors
(e.g. `type Vector = AnyTensor`) hand-roll `IndexRuntime` /
`CodecRuntime` directly; the derive's `[f32]` downcast follows
the same convention as `add` / `encode`.

Authors that need body-phase ordering against training place the
`IndexSlot::train` / `CodecSlot::train` call inside
`Module::bootstrap`. The per-component `is_op_locked` gate parks
every body op whose touched `ComponentRef` falls in the
bootstrap's touch set, so `add` and `search` ops never observe
an untrained index. The host kicks the recorded bootstrap with
`node.run_bootstrap(BootstrapTarget::All)` before the body poll
loop.

The error type's `Display` impl is what surfaces if the user calls
`completion.complete(Err(e))` from the deferred path, or if the
inline path returns `ContractResponse::Now(Err(e))`.

## Bootstrap is just another Contract method

`bb::Bootstrap::bootstrap(&mut self, &mut BootstrapCtx)`
(`bb-runtime/src/contracts/bootstrap.rs:54-67`) is dispatched
through the same per-T downcast bridge every other Contract
method uses. `#[derive(bb::Concrete)]`
(`bb-derive/src/roles.rs:21-79`) emits one of two impls per
struct:

1. **Default no-op.** A blanket `impl Bootstrap for #struct
   { type Error = Infallible; }` so a Concrete with no
   manual override participates in the Component bootstrap
   dispatch path without boilerplate.
2. **Override-respecting.** The derive sees
   `#[bootstrap_override]` (`bb-derive/src/parse.rs:36-48`)
   on the struct and skips the default impl so a hand-written
   `impl Bootstrap for X { ... }` does not collide.

In both cases the derive submits a
`BootstrapDispatcherRegistration` inventory entry
(`bb-runtime/src/registry.rs:170-208`) carrying a per-T
registration callback that invokes
`Engine::register_bootstrap_dispatcher::<T>()` against the
engine. The install path
(`src/install.rs:451-466`) walks the inventory and registers
every Concrete's dispatcher before the first Component
bootstrap fires — no per-call type lookup.

At dispatch time, `Engine::dispatch_component_bootstrap`
(`bb-runtime/src/engine/core.rs:1405-1432`) follows the same
take-restore pattern `dispatch_atomic` uses:

1. `take_component(cref)` moves the boxed concrete out of the
   `Option<Box<dyn ErasedComponent>>` slot.
2. The erased `&mut dyn Any` looks up its `TypeId` in
   `bootstrap_dispatchers`.
3. The matching `BootstrapDispatchFn`
   (`bb-runtime/src/engine/invoke.rs:1019-1050`) downcasts
   `&mut dyn Any → &mut T`, builds the `BootstrapCtx`,
   invokes `T::bootstrap(&mut ctx)`, and translates
   `Result<(), T::Error>` into the matching `DispatchResult`
   (`Ok(())` → `Immediate(vec![])`; `Err(e)` carries the
   `Display`ed error).
4. `restore_component(cref, taken)` returns the concrete to
   its slot before the next dispatch fires.

The translation makes `Bootstrap` indistinguishable from any
other Contract method on the engine side — `dispatch_atomic`,
the Role dispatchers, and the Bootstrap dispatcher all run
through the same erased-Any + TypeId-keyed lookup pattern. The
only Bootstrap-specific bookkeeping is the engine's
`bootstrap.in_flight` lock set, which the per-component
`is_op_locked` gate consults to park body ops while the
override runs.

## Per-role status

The Contract trait shape and derive bridge ship for every role.
`bb-derive/src/roles.rs::emit_role_derive` emits the typed
`dispatch_atomic` translation from `ContractResponse` to
`DispatchResult` per role; `bb-runtime/src/engine/core.rs` registers
the per-role dispatchers via `register_dispatchers_for`.

| Role | Contract trait | Derive bridge | Notes |
|---|---|---|---|
| Index | shipped | shipped | Reference implementation; see `examples/custom_index_hnsw.rs` |
| Aggregator | shipped | shipped | Typed Contribution + AggregatorEvent |
| Model | shipped | shipped | Includes `Forward`/`Backward`/`ApplyDelta`/`LoadParameters`/`ComputeLoss`/`Params` |
| Codec | shipped | shipped | Typed downcast arms for `Vec<Vec<f32>>` / `Vec<u8>` |
| DataSource | shipped | shipped | `NextBatch` returns typed payload via `ContractResponse` |
| PeerSelector | shipped | shipped | `SelectParams` enum surface; see [`ROLES.md`](ROLES.md) |
| Protocol | shipped | shipped | Carries `dispatch_atomic` directly for user-defined opsets |
| Backend | shipped | shipped | `ctx`-exempt per-op + `execute` + `dispatch` surfaces — see below |

### Backend ctx exemption

`Backend`'s user-facing surface (30 per-op primitives plus
`execute` plus `dispatch`) is the **only** Contract surface whose
methods do not take `ctx`. The chained pattern every other
Contract uses —

```rust
let backend = ctx.dependency::<B>("backend")?;   // &B borrowed via ctx
backend.mul(&tensor, &weight)?;                  // call on &B
```

— is incompatible with `Backend::mul` taking `&mut ctx`: the
second call would re-borrow `ctx` while `backend` still holds an
immutable borrow of it (E0502). Backend is the terminal dep in
the injection chain (a leaf), not a composition seam, so the
exemption costs nothing — kernels stay pure tensor functions
(`add(a, b) → c`, `matmul(a, b) → c`). The derive's
`BackendRuntime::dispatch_atomic` bridge still receives `ctx` and
threads `current_node_attributes` + `current_node_metadata` into a
`BackendAttrs<'_>` for the `execute` override; the per-op
Contract surface stays ctx-free.

## Worker-thread pattern

`examples/custom_index_hnsw.rs` demonstrates the canonical pattern
for compute-heavy or non-blocking Contract impls:

```rust
struct HnswIndex {
    tx: mpsc::Sender<WorkItem>,
}

impl Index for HnswIndex {
    type Vector = [f32];
    type Error = HnswError;

    fn add(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        vec: &Self::Vector,
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error> {
        // Capture vec + completion handle on a work item; ship to worker.
        self.tx
            .send(WorkItem::Add { vec: vec.to_vec(), completion })
            .ok();
        ContractResponse::Later
    }
    // ...
}

// Worker thread, separate from the engine thread:
fn worker(rx: mpsc::Receiver<WorkItem>) {
    while let Ok(item) = rx.recv() {
        match item {
            WorkItem::Add { vec, completion } => {
                let id = expensive_index_op(&vec);
                completion.complete(Ok(id));  // pushes IngressEvent::Completion
            }
        }
    }
}
```

The `CompletionHandle` is `Send + Sync` (via the `Arc<dyn CompletionSink>`),
so it crosses the channel cleanly. The engine sees the completion
when it next drains its ingress queue — no shared state between the
engine and worker beyond the ingress queue itself.
