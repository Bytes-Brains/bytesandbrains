# API_DESIGN.md — application ↔ library interface

This document covers every surface where an application interacts with
the BytesAndBrains library: how applications **author** graphs (the
DSL), how they **extend** the library with custom components, how
they **construct** Nodes, and how the binary stays small as the
catalog grows. It pairs with `IR_AND_DSL.md` (the canonical IR
mapping) and `ENGINE.md` (the runtime + engine spec).

Three guiding principles, all carried over from earlier conversation:

1. **The public API surface is plain structs and plain functions —
   no `<T>` syntax visible in signatures users write or read, no
   proc-macros required of user code, no derive magic.** Internal
   generics inside framework types are fine because users never
   write the parameters; e.g. `DataSource<D: Dataset>` is OK
   because the user calls `DataSource::new(dataset, 32)` and hands
   the result to `with_data_source(source)` — no `<T>` appears.
   Once this rule holds, language bindings are mechanical:
   wrappers wrap the public surface; we don't design for FFI
   specifically.
2. **The Node API is small and uniform.** A single chain shape. No
   special-case methods. No magic accessors.
3. **The library scales by composition, not by accretion.** The core
   crate ships the contract; integration crates ship the impls.
   Binaries pay for what they use.
4. **Every role is extensible by the same pattern.** Backend, Model,
   Index, Aggregator, Codec, DataSource, PeerSelector, Wire,
   Protocol — all extended by identical mechanics. The framework
   ships the contracts; applications implement them. A user adding
   a new database-backed Index follows the same six steps as a user
   adding a new Burn-backed Model.

---

## Part 1 — The component duality (frontend ↔ runtime impl)

Every component in the framework has **two correlated faces**:

- **The DSL frontend.** A Rust struct whose methods record `NodeProto`s
  into a `Graph`. Pure graph mutation. No runtime behavior. Same
  shape Rust → Python.
- **The runtime impl.** A trait impl that the BB engine dispatches
  through at execution time. Owns the real state (weights, sockets,
  DB handles, etc.).

Two forms of correlation are shipped:

### Form 1 — Generic placeholder (no impl shipped)

The framework ships unit-struct placeholders for each role under
`bytesandbrains::placeholders` (re-exported at the crate root):

```rust
pub struct Backend;
pub struct Model;
pub struct Index;
pub struct Aggregator;
pub struct Codec;
pub struct DataSource;
pub struct PeerSelector;
```

Each carries the DSL method set for its role. Each records
NodeProtos under the right opset (`ai.onnx` for Backend, the matching
`ai.bytesandbrains.role.*` for others). **No runtime impl is bundled
in the core crate** — these are placeholders. At load time the user
must supply a `bb::Backend` / `bb::Model` / etc. Contract binding for
the slot via the chained Node API; the `#[derive(bb::<Role>)]`
proc-macro emits the framework-internal runtime bridge.

This is the "run on any conformant runtime" form. The graph can be
authored once and executed under different backend impls — Burn for
training, ONNX Runtime for inference, custom for production.

### Form 2 — Concrete impl (placeholder DSL + Contract impl together)

A concrete impl is a struct that ships **both** faces. Same DSL
method names as its placeholder counterpart; additionally implements
the `bb::Backend` Contract. The `#[derive(bb::Backend)]` proc-macro
generates the framework-internal `BackendRuntime` impl:

```rust
// In `bb-burn` (integration crate):
#[derive(bb_derive::Concrete, bb_derive::Backend)]
pub struct BurnBackend { config: BurnConfig, /* state */ }

impl bb::Backend for BurnBackend {
    type Tensor = burn::Tensor;
    type Error = BurnError;

    // 30 per-op primitives the Contract requires. Per-op methods
    // default to wrapping a single-node GraphProto and calling
    // `execute` — backends that compile graphs natively override
    // `execute` only.
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error> { /* ... */ }
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor)
        -> Result<Self::Tensor, Self::Error> { /* ... */ }
    // ... full ai.onnx v1 catalog ...
}
```

When the user writes `self.burn.matmul(g, a, b)` in their Module's
`graph()`, the recorded NodeProto:
- Uses `domain = "ai.onnx"` (same as if it had been `Backend`).
- Tags with `metadata_props["ai.bytesandbrains.slot"] = "burn"` (the
  Module field name).
- Tags with `metadata_props["ai.bytesandbrains.concrete_type"] =
  "burn_integration::BurnBackend"` so the framework's deserializer
  registry knows how to reconstruct from snapshot.

The construction config (`BurnConfig`) is serialized into the
parent `FunctionProto.attribute_proto[<slot>].s` on the first DSL
call. At load, the framework looks up the registered deserializer
for `BurnBackend` and rehydrates.

### Why both forms

- **Form 1 (placeholder)** lets a Module declare "I need any
  Backend" — a portable graph that runs on any conformant runtime.
- **Form 2 (concrete)** lets a Module pin a specific implementation
  with its config — useful when the user has chosen the backend
  (e.g. a training pipeline that needs Burn's autograd) or when the
  binding is application-specific (e.g. `MySqlLiteIndex` wrapping a
  particular database file).

A single Module can mix both:

```rust
struct DistillationModule {
    backend: Backend,         // generic — any backend
    teacher: BurnModel,       // concrete — pinned to Burn
    student: BurnModel,       // concrete — also Burn, distinct slot
    db: MySqlLiteIndex,       // concrete — application-specific
}
```

This is the "everything in one Module struct" pattern from the
canonical example. Each DSL method calls
`g.register_concrete::<Self>(self)` (for concrete impls) or
`g.register_generic(...)` (for unit-struct placeholders) at its top,
populating identity metadata on every recorded NodeProto. The
framework distinguishes generic vs concrete by the metadata key
that's present (`concrete_type` vs `required_trait`) — see Part 3.

---

## Part 2 — The component duality at the IR level

How the duality lands in the IR is already specified by
`IR_AND_DSL.md`:

| Form | FunctionProto representation |
|---|---|
| Generic placeholder | `FunctionProto.attribute: repeated string` — required, no default; binding supplied at load |
| Concrete impl | `FunctionProto.attribute_proto: repeated AttributeProto` — defaulted; payload (.s bytes / .t TensorProto / etc.) carries construction state |

Multi-instance per role (`teacher` + `student` both `BurnModel`) is
naturally supported: two distinct entries in `attribute_proto` with
distinct names but the same `concrete_type`. The framework's slot
disambiguator is the attribute name.

---

## Part 3 — The Module trait

The trait has two user methods (`name` + `body`) and three
framework-provided defaults (`op` + `call` + `build`). The body
records into the supplied `Graph` and returns `()`; outputs are
declared by name through `Graph::output` or `Graph::net_out`.

```rust
pub trait Module {
    /// Short identifier — becomes `FunctionProto.name`.
    fn name(&self) -> &str;

    /// User-implemented recording logic. Declare inputs through
    /// `g.input("name")`. Emit outputs through
    /// `g.output(name, value)` (local sink) or
    /// `g.net_out(name, peers, value)` (network sink). Compose child
    /// Modules through `self.child.call().input(...).build(g)`.
    fn body(&self, g: &mut Graph);

    /// Framework-supplied entry: opens a function scope named
    /// `self.name()` and records `body()` into it. Top-level wraps
    /// fold the body straight into `functions[0]`; nested calls
    /// produce a sub-`FunctionProto` plus a CALL NodeProto. Do not
    /// override.
    fn op(&self, g: &mut Graph, bindings: &[(String, Output)])
        -> Vec<(String, String)>
    {
        g.with_function(self.name(), bindings, |g| self.body(g))
    }

    /// Open a fluent call-site that inlines `self`'s body into the
    /// caller's `Graph`. See §4 for the chain shape.
    fn call(&self) -> ModuleCall<'_, Self> { /* … */ }

    /// Walk the composition tree and emit ONE pre-compile
    /// `ModelProto` whose `functions[0]` is the top-level body and
    /// `functions[1..]` are every composed sub-Module (one entry per
    /// unique `name()` reached during recording — the same Module
    /// composed into multiple parents shares its FunctionProto).
    /// Partitioning is downstream in `Compiler::compile`; the bare
    /// `ModelProto` from `build()` is not directly installable.
    fn build(self) -> Result<ModelProto, BuildError>
    where Self: Sized {
        // default impl in bb-dsl/src/module.rs
    }
}
```

`BuildError` (in `bb-dsl/src/module.rs`) surfaces the two
recorder-level failures: `EmptyModule` (the body recorded no nodes
and declared no sub-functions) and `MissingOutputPort { name }` (a
`Graph::output` reference to a port no producer emits). Compile-time
type / dispatch failures surface from the compiler instead.

### Outputs declared by name

A Module body sinks data through the Graph's two output methods.
Both register a named port in the active `FunctionProto.output`
list; both are idempotent on `(scope, name)`.

```rust
impl Graph {
    /// Local output: emits a `PassThrough` NodeProto that re-names
    /// `value`'s producer to `name`, then records `name` in the
    /// active function's `output[]` + `value_info[]`.
    pub fn output(&mut self, name: &str, value: Output);

    /// Network output: emits a `wire.Send` NodeProto with
    /// `(value, peers)` as inputs and `(name, request_id)` as
    /// outputs, records `name` in the active function's `output[]`,
    /// and lets the compiler's `partition_by_wire_ops` /
    /// `synthesize_wire_recvs` passes carve the partition boundary
    /// + synthesize the matching `wire.Recv` on each consumer-side
    /// partition. `net_out` IS the network boundary; there is no
    /// separate `Graph::wire`, `net::send`, or `net::recv` surface.
    pub fn net_out(&mut self, name: &str, peers: Output, value: Output);
}
```

The Module struct's embedded component fields (placeholders or
concretes) record NodeProtos whose `metadata_props` carry slot /
instance identity stamps. The compiler's `resolve_slots` pass
matches each stamp to a `Compiler::bind_<role>::<T>(slot)` entry
and stamps the binding table onto the model.

### Graph-driven metadata propagation

DSL methods on embedded fields call the Graph's registration helper
at their top so each emitted NodeProto carries identity metadata:

```rust
impl Graph {
    /// Identity capture for generic placeholders (`Backend`,
    /// `Index`, etc.). Idempotent on `(TypeId, instance pointer)`:
    /// mints a `slot_id` on first encounter, appends
    /// `"__slot_<slot_id>"` to the active function's `attribute`
    /// list, and returns the cached id for subsequent calls.
    pub fn register_generic<T: 'static>(
        &mut self,
        instance: &T,
        required_trait: &'static str,
    ) -> u32;

    /// Declare a named Module input. Returns an `Output` handle the
    /// body threads through DSL calls. Inside a `with_function`
    /// scope the input lands on the active sub-function's
    /// `input[]` / `value_info[]`; at the top level it lands on the
    /// root function.
    pub fn input(&mut self, name: &str) -> Output;

    /// Allocate a fresh value-name. Monotonic counter; format `"v<n>"`.
    pub fn next_site_name(&mut self) -> String;

    /// Push a NodeProto into the active function. Called by DSL
    /// methods after stamping captured metadata.
    pub fn push_node(&mut self, node: NodeProto);
}
```

Concrete-component identity travels through the compiler's
`BindingSpec` (built from `Compiler::bind_<role>::<T>(slot)`
declarations) rather than through Graph-side capture — the
recorder doesn't need to know which concrete fills a slot. The
`Module`'s placeholder fields produce slot-tagged NodeProtos; the
compiler matches `slot` to a binding entry; install instantiates the
concrete via the inventory's `construct_fn` per §5.

`ConcreteComponent` is the framework's polymorphism contract — every
concrete impl declares `TYPE_NAME`, `PACKAGE`, `Config`, `Error`,
and the `new` / `serialize` / `restore` triad. Part 6 covers the
trait in detail; the `bb_dsl::concrete::ConcreteComponent` path
re-exports the canonical definition from `bb-runtime::concrete`.

### Sub-Module composition — the `call()` builder

A sub-Module is invoked through its `call()` builder. The fluent
chain binds named formals to caller-scope `Output` handles, builds
the body into the caller's Graph, and hands back a `ModuleOutputs`
handle that resolves the sub-Module's declared output ports by
name:

```rust
impl Module for OuterModule {
    fn name(&self) -> &str { "outer" }
    fn body(&self, g: &mut Graph) {
        let q = g.input("query");
        let coord = self
            .coordinator
            .call()
            .input("incoming", q)
            .build(g);
        let grad = coord.output("aggregated_grad");
        g.output("grad", grad);
    }
}
```

`g.with_function(name, bindings, body)` stamps every NodeProto
recorded inside with
`ai.bytesandbrains.module_instance = "outer_coordinator"` (joined
by `_`). That hierarchy is what the compiler's partition-naming
pass uses to label each emitted partition's root function after
wire ops slice the graph.

### Inline vs. hoist (sub-Module bodies in the IR)

Sub-Modules with no wire ops can be hoisted into standalone
`FunctionProto` entries in `ModelProto.functions[]` and referenced
through CALL NodeProtos — exactly how ONNX function calls are meant
to be used. The hoist saves space when the same sub-Module is
composed multiple times.

Sub-Modules that contain a wire op are inlined into their parent's
body so `partition_by_wire_ops` can slice across them. Wire ops are
partition boundaries; a partition can end mid-way through a
sub-Module's body.

The compiler's `inline_for_partition` pass makes this decision; the
user just calls `child.call().input(...).build(g)` and the
framework chooses.

---

## Part 4 — The Graph wrapper and `ModuleCall`

`Graph` is the recording context threaded through `Module::body()`.
User code never constructs `Graph` directly — `Module::build()`
allocates a fresh `Graph` before calling `self.op(&mut g, &[])`. DSL
methods on components (Backend, Aggregator, etc.) take `&mut Graph`
because that's the surface they write NodeProtos to; the body just
threads the Graph it received.

`Graph` is a thin wrapper around an in-progress `FunctionProto` plus
the bookkeeping needed for nested `with_function` scopes. The ONNX
`FunctionProto` IS the IR — semantically meaningful BB attributes
ride on proto fields, not on parallel Rust shadow stores. See
[`IR_AND_DSL.md`](IR_AND_DSL.md) for the concept-to-proto mapping.

The Rust wrapper carries what the proto can't represent:

- `instance_for_pointer` — pointer-identity dedup for generic
  placeholder fields, keyed by `(TypeId, *const ())` so two
  zero-sized placeholders with different types resolve to distinct
  slots.
- `site_counter` — monotonic counter that mints `"v<n>"` value
  names through `next_site_name`.
- `module_scope` / `sub_functions` / `recording_target` —
  composition-hierarchy stack and FunctionProto registry maintained
  across nested `with_function` calls.
- `named_output_types` — keyed `(target_idx, name)` map for
  idempotent `output` / `net_out` registration.
- `formal_binding_stack` — formal→actual handle map maintained
  during a `ModuleCall::build` so the body's `g.input(name)`
  lookups resolve to the caller-side handle.
- `pending_errors` — recorder-time `BuildError` accumulator drained
  by `Module::build`.

The methods DSL authors and Module bodies call:

```rust
impl Graph {
    /// Declare a Module input by name. Returns an `Output` handle
    /// the body threads through DSL calls. Inside a nested
    /// `with_function` scope the input lands on the active
    /// sub-function only; at the top level it lands on the root
    /// function.
    pub fn input(&mut self, name: &str) -> Output;

    /// Register `name` as a local output port: emits a
    /// `PassThrough` NodeProto renaming `value`'s producer to
    /// `name` and adds `name` to the active function's `output[]`.
    /// Idempotent on `(scope, name)`.
    pub fn output(&mut self, name: &str, value: Output);

    /// Register `name` as a network output port: emits a
    /// `wire.Send` NodeProto with `(value, peers)` inputs and
    /// adds `name` to the active function's `output[]`. The
    /// compiler partitions the graph at this boundary and
    /// synthesizes the matching `wire.Recv` on each consumer-side
    /// partition. `net_out` IS the network output — no separate
    /// `Graph::wire`, `net::send`, or `net::recv` surface.
    pub fn net_out(&mut self, name: &str, peers: Output, value: Output);

    /// Look up an output port registered earlier in the body.
    /// Returns `None` when neither `output` nor an enclosing scope
    /// has registered the port; callers report
    /// `BuildError::MissingOutputPort`.
    pub fn lookup_output(&self, name: &str) -> Option<Output>;

    /// Identity capture for generic placeholders. Idempotent on
    /// `(TypeId, instance pointer)`; mints a `slot_id` on first
    /// encounter and appends `"__slot_<slot_id>"` to the active
    /// function's `attribute` list.
    pub fn register_generic<T: 'static>(
        &mut self,
        instance: &T,
        required_trait: &'static str,
    ) -> u32;

    /// Allocate a fresh value-name (`"v<n>"`).
    pub fn next_site_name(&mut self) -> String;

    /// Declare a `ValueInfoProto` for `name` carrying the supplied
    /// `TypeNode` on the active recording target. Idempotent.
    pub fn declare_value_info(
        &mut self,
        name: &str,
        type_node: &'static TypeNode,
    );

    /// Push a NodeProto into the active function. Called by DSL
    /// methods after stamping captured metadata. The current
    /// composition-hierarchy chain (if any) is prefixed onto the
    /// node's `ai.bytesandbrains.module_instance` stamp.
    pub fn push_node(&mut self, node: NodeProto);

    /// Push a typed recorder error onto the pending-errors queue.
    /// `Module::build()` drains the queue and surfaces the first
    /// entry as the build's typed failure.
    pub fn record_build_error(&mut self, err: BuildError);
}
```

The internal `with_function` primitive (driven by `Module::op`'s
default impl) opens a scope, records the body into the active
`FunctionProto`, and emits the CALL NodeProto in the outer scope.
Top-level wraps fold the body into `functions[0]` instead of
creating a separate sub-function + CALL. User code does not call
`with_function` directly — `Module::op` and `ModuleCall::build`
own that primitive.

### Sub-Module composition: `ModuleCall` + `ModuleOutputs`

`Module::call()` returns a `ModuleCall` builder. The builder binds
named formal inputs, calls `Module::op` to record the body into the
caller's Graph, and hands back a `ModuleOutputs` that resolves each
declared output port by name:

```rust
pub struct ModuleCall<'a, M: ?Sized + Module> { /* … */ }

impl<M: ?Sized + Module> ModuleCall<'_, M> {
    /// Bind a formal input name to a caller-side `Output` handle.
    pub fn input(self, name: &'static str, handle: Output) -> Self;

    /// Inline the sub-Module's body into `g`. Emits a CALL
    /// NodeProto in the current scope whose `input[]` is the bound
    /// actuals and whose `output[]` is freshly-minted outer-scope
    /// names — one per output port the body declared.
    pub fn build(self, g: &mut Graph) -> ModuleOutputs<'_>;
}

pub struct ModuleOutputs<'a> { /* … */ }

impl ModuleOutputs<'_> {
    /// Pull the named output the inlined body registered via
    /// `g.output(name, handle)` or `g.net_out(name, peers, handle)`.
    /// Resolves against the CALL NodeProto's outer-scope output
    /// names for sub-function composition; falls back to the
    /// parent-scope `lookup_output` for top-level wraps.
    pub fn output(&self, name: &'static str) -> Output;
}
```

A canonical composition site:

```rust
impl Module for Outer {
    fn name(&self) -> &str { "Outer" }
    fn body(&self, g: &mut Graph) {
        let q = g.input("query");
        let coord = self
            .coordinator
            .call()
            .input("incoming", q)
            .build(g);
        let grad = coord.output("aggregated_grad");
        g.output("grad", grad);
    }
}
```

This is the single supported composition surface. The `call() →
input() → build() → output()` chain is what the architecture commits
to; previous `op(g, &[inputs])`-style direct invocation is gone.

### Port declarations on derived Modules

Ports are declared exclusively through the body-recorded canonical
pattern. The body calls `g.input(name)` to register a named input,
`g.output(name, value)` to register a local sink, and
`g.net_out(name, peers, value)` to register a network sink;
`g.lookup_output(name)` retrieves a previously-registered output
inside the same body. There is no struct-level
`#[port_in("name")]` / `#[port_out("name")]` /
`#[port_in_net("name")]` / `#[port_out_net("name")]` attribute set
and no `#[derive(bb::Module)]` — every Module is hand-written
against the `Module` trait and every port surfaces through the
four `Graph` recorder methods.

```rust
impl Module for MyModule {
    fn name(&self) -> &str { "MyModule" }
    fn body(&self, g: &mut Graph) {
        let query = g.input("query");        // local input
        let peers = g.input("peers");        // network peer set
        let result = self.backend.matmul(g, query, /* … */);
        g.output("result", result);          // local output
        g.net_out("broadcast", peers, result); // network output
    }
}
```

The compiler's `verify_network_at_boundary` pass walks the recorded
NodeProtos and matches each wire op against the port the body
registered through `g.net_out(...)`; mismatches surface as
`CompileError`s at compile time.

### DSL method body shape

DSL methods on placeholder unit structs (`Backend`, `Index`, …
shipped from `bytesandbrains::placeholders`) record NodeProtos
under their role's domain. The canonical shape:

```rust
impl Backend {
    pub fn matmul(&self, g: &mut Graph, a: Output, b: Output) -> Output {
        let slot_id = g.register_generic(self, "BackendRuntime");
        let out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "MatMul".into(),
            domain:  "ai.onnx".into(),
            input:   vec![a.name.clone(), b.name.clone()],
            output:  vec![out_name.clone()],
            metadata_props: vec![
                kv("ai.bytesandbrains.required_trait", "BackendRuntime"),
                kv("ai.bytesandbrains.slot_id",        &slot_id.to_string()),
            ],
            ..Default::default()
        });
        Output::new(out_name, &TYPE_TENSOR_F32)
    }
}
```

The same shape covers every placeholder method. Concrete-impl
identity is supplied later: `Compiler::bind_<role>::<T>(slot)`
records the `(slot, role, TYPE_NAME)` triple, the `resolve_slots`
pass stamps the binding table onto the model, and `bb::install`
constructs the concrete via the inventory's `construct_fn`. The
recorder does not capture concrete instances.

`kv`, `attr_int`, `attr_float`, `attr_ints`, `attr_graph`, and
`attr_tensor` are exported from `bb_dsl::graph` for DSL-method
authors writing NodeProto bodies by hand.

---

## Part 5 — The Node lifecycle: author → compile → install

The framework commits to a three-phase contract:

1. **AUTHOR.** `Module::build()` walks the composition tree and
   returns ONE pre-compile `ModelProto`. The bare proto is not
   installable; it carries no compilation passport.
2. **COMPILE.** `Compiler::new().bind_<role>::<T>(slot)…
   .compile(model)` records the author's concrete-binding choices,
   runs the 18-pass compilation pipeline, stamps the compilation
   passport (`ai.bytesandbrains.compiled = "v1"`) + per-target
   binding table onto the model, and returns a single installable
   `ModelProto`.
3. **INSTALL.** `bb::install(peer_id, addr, compiled, target,
   config)` verifies the passport, locates the function named
   `target` (exact match or `target#<hash>`), parses the binding
   metadata for that target, constructs each declared concrete via
   the inventory's `construct_fn`, registers the instances, and
   installs the target function as the engine's root graph.

The canonical end-to-end shape:

```rust
let module = MyAppModule::new();          // author

let compiled: ModelProto = bb::Compiler::new()
    .bind_backend::<CpuBackend>("compute")
    .bind_index::<HnswIndex>("primary")
    .bind_data_source::<MyLoader>("loader")
    .compile(module.build()?)?;            // compile

let mut node = bb::install(                // install
    my_peer_id,
    my_addr,
    compiled,
    "MyAppModule",                         // target function name
    bb::Config::new()
        .with("loader", MyLoaderConfig { batch_size: 32 }),
)?;

let waker = std::task::Waker::noop();
let mut cx = std::task::Context::from_waker(waker);
loop {
    match node.poll(&mut cx) {
        std::task::Poll::Ready(steps) => { /* drain steps */ }
        std::task::Poll::Pending => break,
    }
}
```

### 5.1 `Module::build()`

```rust
fn build(self) -> Result<ModelProto, BuildError> where Self: Sized;
```

Allocates a fresh `Graph`, invokes `self.op(&mut g, &[])`, drains
the recorder's `BuildError` queue (surfacing the first entry on
failure), and packages the recorded root function plus every nested
sub-function into a single `ModelProto`. The returned model has
empty `metadata_props` — install rejects it with
`InstallError::NotCompiled` until it passes through `Compiler`.

### 5.2 `Compiler`

```rust
impl Compiler {
    pub fn new() -> Self;
    pub fn default() -> Self;

    // Bind chain — author-chosen slot name → concrete type. Each
    // method is generic over `T: ConcreteComponent + <Role>Runtime`
    // so the type system enforces "this concrete actually
    // implements the role you're binding it under" at the bind
    // site.
    pub fn bind_backend       <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + BackendRuntime;
    pub fn bind_model         <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + ModelRuntime;
    pub fn bind_index         <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + IndexRuntime;
    pub fn bind_aggregator    <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + AggregatorRuntime;
    pub fn bind_codec         <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + CodecRuntime;
    pub fn bind_data_source   <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + DataSourceRuntime;
    pub fn bind_peer_selector <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + PeerSelectorRuntime;
    pub fn bind_protocol      <T>(self, slot: impl Into<String>) -> Self where T: ConcreteComponent + ProtocolRuntime;

    // Pipeline shaping.
    pub fn push_back_stage<S: CompilerStage + 'static>(self, stage: S) -> Self;
    pub fn push_front_stage<S: CompilerStage + 'static>(self, stage: S) -> Self;
    pub fn insert_stage<S: CompilerStage + 'static>(self, index: usize, stage: S) -> Self;
    pub fn without_stage(self, name: &str) -> Self;
    pub fn with_target_version(self, version: u32) -> Self;
    pub fn with_per_hop_budget_ns(self, budget_ns: u64) -> Self;
    pub fn with_permissive_types(self) -> Self;

    // Final emission.
    pub fn compile(self, model: ModelProto) -> Result<ModelProto, CompileError>;

    // Inspection helper — emits partitions without stamping the
    // passport or validating bindings. Tests only.
    pub fn compile_partitions(&self, model: ModelProto)
        -> Result<Vec<ModelProto>, CompileError>;
}
```

`compile` returns one `ModelProto` whose `functions[]` contains
every partition root (named by the composition hierarchy plus a
content-hash suffix for snapshot stability) and whose
`metadata_props` carry:

- `ai.bytesandbrains.compiled = "v1"` — the install passport.
- One `ai.bytesandbrains.binding.<target>.<slot> =
  "<role>|<TYPE_NAME>|<slot_id|-1>"` entry per
  `(target_function, slot)` pair. `install` parses these for the
  chosen target to recover the per-slot binding table.

The pipeline runs:

1. The 17 canonical passes listed by
   `bb_compiler::CANONICAL_PASS_NAMES` (in order:
   `inline_for_partition`, `derive_wire_deadlines`, `validate`,
   `expand_ops`, `type_solver`, `infer_peer_classes`,
   `synthesize_wire_recvs`, `partition_by_wire_ops`,
   `resolve_slots`, `analyze_wire_edges`, `insert_dedup_gate_rx`,
   `insert_peer_health_gate_rx`, `insert_backoff_gate_rx`,
   `insert_peer_health_gate_tx`, `insert_backoff_gate_tx`,
   `insert_async_deadlines`, `validate_runtime_complete`).
2. Any user-supplied stages registered via `push_*_stage` /
   `insert_stage`, in declared order, against each emitted
   partition.
3. Component-dependency resolution (`resolve_component_dependencies`),
   slot-binding validation (`validate_all_slots_bound`), passport
   stamping, and partition merging.

`CompileError` surfaces every compiler failure: `EmptyFunctionTable`,
`IrVersionMismatch`, `UnboundSlot`, `UnboundDependency`,
`DependencyRoleMismatch`, the per-pass validation errors, and
`Internal { detail }` for user-stage failures.

### 5.3 `bb::install`

```rust
pub fn install(
    peer_id: PeerId,
    addr: Address,
    model: ModelProto,
    target: &str,
    config: Config,
) -> Result<Node, InstallError>;
```

`install`:

1. Anchors the `bb-ops` inventory submissions so DCE-trimmed
   carriers stay linked.
2. Verifies the compilation passport — `InstallError::NotCompiled`
   when absent, `InstallError::IncompatibleCompiledVersion` on
   mismatch.
3. Finds the function whose `name` matches `target` (exact match
   first, then `target#<hash>` substring match). Missing target
   surfaces `InstallError::UnknownTarget { target, available }`.
4. Parses `ai.bytesandbrains.binding.<target>.<slot>` entries into
   a list of `(slot, type_name, slot_id)` triples for the resolved
   target.
5. For each binding, looks up the concrete's
   `inventory::submit!`-registered entry by `TYPE_NAME`. Missing
   registration surfaces `InstallError::UnregisteredConcrete`.
6. Supplies the per-slot config (defaulting to `&()`) to the
   inventory's `construct_fn`, which downcasts to the concrete's
   `Config` associated type and calls `T::new(&config)`. Downcast
   failures surface `InstallError::ConfigTypeMismatch`; the
   concrete's `T::new` errors surface
   `InstallError::ConstructionFailed`. Concretes that decline a
   config surface `InstallError::MissingConfig`.
7. Registers the materialized concrete with the engine, binds the
   `slot` name + `slot_id` to its `ComponentRef`, stamps role
   metadata, and registers a `ComponentHandle` for snapshot/restore.
8. Installs the resolved target function as the engine's root
   graph and the other functions as the function library. Runs
   `resolve_dispatch` so every NodeProto's `OpDispatch::*` variant
   is stamped before the first `poll`.
9. Registers the target as a valid `IngressEvent::Invoke` /
   `deliver_event` destination on the Node.

### 5.4 `Config`

```rust
pub struct Config { /* … */ }

impl Config {
    pub fn new() -> Self;
    pub fn with<C: Any + 'static>(self, slot: impl Into<String>, config: C) -> Self;
}
```

`Config::new().with(slot, value)` attaches typed configs that the
inventory's `construct_fn` downcasts to the bound concrete's
`Config` associated type. Concretes whose `type Config = ()` need
no entry — install supplies `&()` automatically. The `Any + 'static`
bound is the entire compile-time enforcement; runtime mismatch
surfaces `InstallError::ConfigTypeMismatch` with the slot, type
name, and downcast detail.

### 5.5 The `Node` runtime surface

`bb::install` returns a `Node` ready to drive. The public surface
the host uses:

```rust
impl Node {
    /// Returns the queue handle the host uses to push
    /// `IngressEvent::Invoke` / `Inbound` / `Completion` etc.
    pub fn ingress_handle(&self) -> IngressQueueRef;

    /// Poll the engine. Returns `Ready(Vec<EngineStep>)` after each
    /// cycle; an empty vec means the engine reached quiescence.
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Vec<EngineStep>>;

    /// Push an inbound `WireEnvelope` directly (transport adapters).
    pub fn deliver_inbound(&mut self, env: WireEnvelope) -> Result<(), DeliveryError>;

    /// Push a typed in-process event onto the engine bus.
    pub fn deliver_event(/* … */) -> Result<(), DeliveryError>;

    /// Invoke an installed module by name.
    pub fn invoke(/* … */) -> Result<ExecId, DeliveryError>;

    /// Capture the snapshottable state once the bus is drained.
    pub fn snapshot(&self) -> Result<NodeSnapshot, SnapshotError>;

    /// Restore from a `NodeSnapshot` produced earlier.
    pub fn restore(&mut self, snap: NodeSnapshot) -> Result<(), RestoreError>;

    /// Override the default `NodeConfig` (cycle caps, async caps,
    /// outbound caps). Must be called before driving `poll`.
    pub fn with_config(self, cfg: NodeConfig) -> Self;
}
```

`Node`'s constructor is `pub(crate)` — only `bb::install` reaches it.
End users never name `Node::new`; the canonical path is
`bb::install(peer, addrs, compiled, &[target], Config::new())`,
which surfaces the compilation passport, binding-table parse, and
concrete construction as a typed `InstallError` instead of leaving
the Node in a half-built state. There is no chained builder for
binding concretes at the `Node` level — bindings are resolved
during `install` from the compiler's metadata.

### 5.6 `InstallError` taxonomy

```rust
pub enum InstallError {
    NotCompiled,
    IncompatibleCompiledVersion { got: String, expected: &'static str },
    UnknownTarget               { target: String, available: Vec<String> },
    InvalidBindingTable         { key: String, detail: String },
    UnregisteredConcrete        { type_name: String },
    MissingConfig               { slot: String, type_name: String },
    ConfigTypeMismatch          { slot: String, type_name: String, detail: String },
    ConstructionFailed          { slot: String, type_name: String, detail: String },
}
```

Each variant carries enough context to be actionable: missing
target lists the available function names, missing concrete names
the `TYPE_NAME` the artifact references, missing config names the
slot + concrete pair so the host knows which `Config::with` call
to add.

---

## Part 6 — Concrete-type registration via the `ConcreteComponent` trait

The framework's polymorphism contract is `ConcreteComponent`. Every
concrete component implements it:

```rust
pub trait ConcreteComponent: ErasedComponent + Sized {
    /// Stable identifier — used as the dispatch key, the snapshot
    /// matching key, and the metadata stamped onto NodeProtos.
    /// Convention: `<crate>::<TypeName>`.
    const TYPE_NAME: &'static str;

    /// Origin tag — defaults to `Application`. Library writers
    /// override to `Framework`.
    const PACKAGE: ComponentPackage = ComponentPackage::Application;

    /// Sibling components this concrete depends on at compile time.
    /// The `bb::Concrete` derive populates this from
    /// `#[bb::depends(<role> = "<slot>", …)]` attributes; the
    /// compiler's `resolve_component_dependencies` pass uses it to
    /// verify every declared slot is bound to a sibling concrete of
    /// the right role.
    const DEPENDENCIES: &'static [DependencyDecl] = &[];

    /// Per-instance configuration `install` passes to `Self::new`.
    /// Use `()` for stateless concretes; use a small `pub struct`
    /// for parameterized concretes (HnswIndex dimensions, etc.).
    type Config: Any + 'static;

    /// Per-impl error surfaced by `Self::new`. Use
    /// `std::convert::Infallible` when construction can't fail.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Construct a fresh instance from per-deployment config.
    /// `install` calls this for every BindingSpec slot.
    fn new(config: &Self::Config) -> Result<Self, Self::Error>;

    /// Capture current state to bytes. Called at `Node::snapshot`
    /// to persist runtime state.
    fn serialize(&self) -> Vec<u8>;

    /// Reconstruct from snapshot bytes. Called at
    /// `Node::restore_from_snapshot`.
    fn restore(bytes: &[u8]) -> Result<Self, RestoreError>;
}

pub enum ComponentPackage { Framework, Application }
```

`#[derive(bb::Concrete)]` implements the trait for you: it uses
`bincode` round-trips for `serialize` / `restore`, sets
`type Config = Self` (configs feed directly into the construction
clone), sets `type Error = Infallible`, and reads
`#[bb::depends(...)]` attributes for `DEPENDENCIES`. The derive
also emits the `inventory::submit!` block that registers the type
in the global registry — the inventory's `construct_fn` is what
`bb::install` calls to materialize fresh instances from a binding
table entry.

```rust
#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
struct MyFancyIndex {
    capacity: u32,
}
```

That's the whole registration mechanism. The accompanying
`#[derive(bb::<Role>)]` (e.g. `bb_derive::Index`) emits the
framework-internal `<Role>Runtime` bridge between the user's
Contract trait impl (`impl bb::contracts::Index for MyFancyIndex`)
and the engine's dispatch machinery, plus the per-type dispatcher
registration the engine wires up at install.

### Three install touchpoints

1. `bb::install` parses the model's binding metadata, finds each
   slot's `TYPE_NAME` in the global inventory, downcasts the
   user-supplied `&dyn Any` config to `&Self::Config`, and calls
   `T::new(&config)`. Construction errors surface as
   `InstallError::ConstructionFailed`.
2. The resulting `Box<dyn ErasedComponent>` is registered with the
   engine under a fresh `ComponentRef`. The slot name + slot id
   are bound to that `ComponentRef` so dispatch can route role-op
   NodeProtos to the right instance.
3. A `ComponentHandle` carrying the type name, instance id,
   `serialize_fn`, and `restore_fn` is appended to the Node's
   handle list. `Node::snapshot` walks the list to capture each
   instance's bytes; `Node::restore` re-materializes via the same
   handles.

### Why this shape

1. **Compile-time enforcement.** `Compiler::bind_<role>::<T>(slot)`
   is bounded `T: ConcreteComponent + <Role>Runtime`. Forgetting
   either trait is a compile error at the bind site.
2. **Binary size.** Concretes register through `inventory::submit!`
   blocks emitted by the derive. Linker DCE drops the registration
   if the binary never references the concrete from its
   construction code (the `bb-ops` facade's `link_force()` function
   anchors the framework-shipped components for `bb::install` to
   find).
3. **Auditable.** `Node`'s linked-component list reports every
   instance by `(TYPE_NAME, instance_id, package)`. The `PACKAGE`
   const distinguishes framework-shipped from user-shipped.

For the long-form walkthrough with a multi-role networked component
running example, see [AUTHORING_COMPONENTS.md](AUTHORING_COMPONENTS.md).

---

## Part 7 — Binary size strategy

If every op and every component ships in the framework, the binary
grows linearly with the catalog. The workspace dodges this with a
layered crate structure: the framework crates ship abstractions, the
`bb-ops` crate ships every framework-shipped concrete, and
downstream integration crates are free to ship their own concretes
under the same pattern.

### Workspace layout

The published workspace has seven crates: the facade
`bytesandbrains` plus six members.

```
bytesandbrains   # facade — re-exports the workspace's public surface
                 #   feature gates: cpu-backend (default), simd,
                 #   tracing-otel, test-components
bb-ir            # foundation: prost-generated proto, wire envelope,
                 #   peer_class, TypeNode, slot_value, the wire-bound
                 #   IDs (PeerId, RequestId, OpsetId, ComponentTag)
bb-dsl           # Module trait, Graph recorder, Output handle,
                 #   syscalls; re-exports contracts + ConcreteComponent
                 #   from bb-runtime for authoring ergonomics
bb-derive        # proc-macros: #[derive(bb::Concrete)],
                 #   #[derive(bb::<Role>)], register_op!,
                 #   register_protocol!
bb-compiler      # 18-pass compilation pipeline, Compiler driver,
                 #   CompilerStage trait, CompileError taxonomy
bb-runtime       # Engine, Node, framework primitives, Contract
                 #   traits (bb::contracts::*), framework-internal
                 #   <Role>Runtime traits (bb::roles::*),
                 #   inventory registry, dispatch types, snapshot
bb-ops           # every framework-shipped concrete: syscall + wire
                 #   ops, the reference cpu Backend, role concretes
                 #   (aggregators, peer selectors, protocols, …)
```

### Downstream integration crates

A user's application lists `bytesandbrains` plus the integration
crates they need; each integration crate ships its own concrete
components paired with the right `#[derive(bb::<Role>)]`:

```toml
[dependencies]
bytesandbrains = "0.3"
# Hypothetical third-party integrations following the same pattern:
my-burn-backend  = "0.1"   # ships a Burn-backed Backend concrete
my-faiss-index   = "0.1"   # ships a FAISS-backed Index concrete
```

Each integration crate's `inventory::submit!` block (emitted by the
derive) registers its concretes in the global registry; the
binary's `bb::install` call resolves the model's binding table
against that registry at runtime. Linker DCE drops components that
neither the user code nor the artifact references.

### Feature flags on the facade

```toml
[features]
default          = ["cpu-backend"]
cpu-backend      = ["bb-ops/cpu-backend"]
tracing-otel     = ["bb-runtime/tracing-otel"]
test-components  = ["bb-ops/test-components", "bb-runtime/test-components"]
```

`cpu-backend` controls the pure-Rust reference CPU backend at
`bytesandbrains::ops::backends::cpu`. `tracing-otel` enables the
opt-in OTLP subscriber constructor at
`bytesandbrains::telemetry::otel`. `test-components` enables the
fixtures the integration tests and examples rely on.

A no-default-features build of `bytesandbrains` gets the minimum
runtime + IR + Contract / role traits. Concretes opt in by listing
the matching feature or pulling a downstream integration crate.

### What lives where

- Authoring surface (`Module`, `Graph`, `Output`, `BuildError`,
  `ConcreteComponent`, Contract traits, placeholder unit structs):
  `bb-dsl` + `bb-runtime`, re-exported through the facade.
- Compile pipeline (`Compiler`, `CompilerStage`, `CompileError`,
  `CANONICAL_PASS_NAMES`): `bb-compiler`, re-exported as
  `bytesandbrains::compiler` and `bytesandbrains::Compiler`.
- Runtime (`Node`, `NodeConfig`, `EngineStep`, `IngressEvent`,
  `IngressQueueRef`, `NodeSnapshot`, `InstallError`): `bb-runtime`
  (with `install` defined in the facade `src/install.rs`).
- Framework concretes (`syscall` ops, `wire` ops, the reference
  CPU backend, framework-shipped role concretes): `bb-ops`,
  exported through `bytesandbrains::ops::*`.

---

## Part 8 — Extension points for application developers

Every role in the framework is extended by the same pattern. A
user adding a custom database-backed `Index`, a custom Burn-backed
`Model`, a custom gossip `PeerSelector`, a custom HTTP `Backend`,
or a custom Kademlia `Protocol` writes the same five items, with
only the role's contract trait changing.

### 8.1 The universal extension recipe

To extend any role, ship a struct that supplies:

1. **A struct.** Holds configuration + runtime state.
2. **A serde derive on the struct.** `#[derive(Serialize,
   Deserialize)]` plus `#[derive(bb::Concrete)]` gives the
   framework the `serialize` / `restore` / `new` triad for free —
   the derive's `new(config)` constructs via
   `Config::clone() → Self`, `serialize` round-trips through
   `bincode`, and `restore` decodes the bytes back.
3. **A `#[derive(bb::<Role>)]`.** Generates the framework-internal
   bridge between the user-facing Contract trait and the engine-side
   `<Role>Runtime`. The derive also registers the type with the
   inventory so `bb::install` can construct fresh instances.
4. **The Contract trait impl** — one of `bb::Index`, `bb::Backend`,
   `bb::Model`, `bb::Aggregator`, `bb::Codec`, `bb::DataSource`,
   `bb::PeerSelector`, or `bb::Protocol`. Defines the role's method
   surface using `CompletionHandle` + `ContractResponse` so async
   work can return `ContractResponse::Later` and complete out-of-band.
5. **DSL surface.** Module authors record the role's ops by calling
   methods on the placeholder unit struct
   (`bytesandbrains::placeholders::Index`, etc.). User code that
   wants to expose the same methods directly on its concrete type
   defines inherent methods following the canonical NodeProto shape
   from Part 4 — the binding mechanism kicks in at `Compiler::bind_*`
   regardless.

Item (2) covers `ConcreteComponent`; item (3) covers the role
plumbing; together they're the entire registration mechanism. User
code writes no global registry calls and no `inventory::submit!`
blocks — the derives emit the submissions, and DCE drops
unreferenced concretes at link time.

### 8.2 Role contracts at a glance

The framework's eight extensible roles. Library makers implement
the user-facing Contract trait in the "Contract" column and add
the matching `#[derive(bb::<Role>)]`; the engine-side
`<Role>Runtime` trait is framework-internal and shouldn't appear
in user code.

| Role         | Contract              | Framework trait        | Placeholder    | NodeProto `domain`               |
|--------------|-----------------------|------------------------|----------------|----------------------------------|
| Backend      | `bb::contracts::Backend`      | `BackendRuntime`       | `Backend`      | `ai.onnx`                        |
| Model        | `bb::contracts::Model`        | `ModelRuntime`         | `Model`        | `ai.bytesandbrains.role.model`        |
| Index        | `bb::contracts::Index`        | `IndexRuntime`         | `Index`        | `ai.bytesandbrains.role.index`        |
| Aggregator   | `bb::contracts::Aggregator`   | `AggregatorRuntime`    | `Aggregator`   | `ai.bytesandbrains.role.aggregator`   |
| Codec        | `bb::contracts::Codec`        | `CodecRuntime`         | `Codec`        | `ai.bytesandbrains.role.codec`        |
| DataSource   | `bb::contracts::DataSource`   | `DataSourceRuntime`    | `DataSource`   | `ai.bytesandbrains.role.data_source`  |
| PeerSelector | `bb::contracts::PeerSelector` | `PeerSelectorRuntime`  | `PeerSelector` | `ai.bytesandbrains.role.peer_selector`|
| Protocol     | `bb::contracts::Protocol`     | `ProtocolRuntime`      | (user-shipped) | user-chosen (e.g. `user.kademlia`)    |

Wire isn't a role — it's engine-native infrastructure exposed
through the `Graph::net_out` recorder method. `wire.Send` /
`wire.Recv` NodeProtos are emitted by the framework; user code
never reaches for them directly.

Each row is one role. Pick a row, follow the five steps, ship the
crate. The framework dispatches a user impl identically to any
framework-shipped impl of the same role.

### 8.3 A worked example — custom `Index` extension

`Index` is shown as a representative; the recipe is identical for
any other role.

```rust
use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::contracts::Index as IndexContract;
use bytesandbrains::graph::Graph;
use bytesandbrains::module::Module;
use bytesandbrains::placeholders::Index;
use bytesandbrains::{install, Compiler, Config};
use bytesandbrains::framework::Address;
use bytesandbrains::ids::PeerId;
use serde::{Deserialize, Serialize};

// 1 + 2 + 3 — struct, serde derive, role derive. `bb::Concrete`
//     gives Self::Config = Self (clone-construct), serialize via
//     bincode, restore via bincode. `bb::Index` emits the
//     `IndexRuntime` bridge + inventory submission.
#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
struct MyFancyIndex {
    capacity: u32,
}

// 4 — Contract trait impl. CompletionHandle lets long-running work
//     return ContractResponse::Later and complete out-of-band; the
//     engine resumes the suspended op when the handle's
//     `complete(...)` is called.
impl IndexContract for MyFancyIndex {
    type Error = std::convert::Infallible;

    fn add(
        &mut self,
        vec: &[f32],
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error> {
        // synchronous path: return Now(id)
        completion.complete(Ok(/* id */ 0));
        ContractResponse::Later
    }

    fn search(
        &self,
        query: &[f32],
        k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error> {
        completion.complete(Ok(Vec::new()));
        ContractResponse::Later
    }

    fn remove(
        &mut self,
        _id: u64,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        completion.complete(Ok(()));
        ContractResponse::Later
    }
}

// 5 — Module that exercises the Index. The body records
//     `self.index.search(...)` against the placeholder; the
//     compiler's resolve_slots pass matches the recorded slot to
//     the `bind_index::<MyFancyIndex>(...)` declaration.
struct SearchApp {
    index: Index,
}

impl Module for SearchApp {
    fn name(&self) -> &str { "SearchApp" }
    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let _ = self.index.search(g, query, 3);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = SearchApp { index: Index };

    let compiled = Compiler::new()
        .bind_index::<MyFancyIndex>("primary")
        .compile(app.build()?)?;

    let mut node = install(
        PeerId::from(1u64),
        Address::empty(),
        compiled,
        "SearchApp",
        Config::new(),
    )?;

    // node.ingress_handle().push(IngressEvent::Invoke { … });
    // loop { node.poll(&mut cx); }
    Ok(())
}
```

The full async-with-worker-thread version lives in
`examples/custom_index_hnsw.rs`.

**To extend a different role**: swap `bb::contracts::Index` for the
target role's Contract trait, swap `bb_derive::Index` for the
target's role derive, and replace the method set with the target
role's. The DSL placeholder field type changes accordingly
(`bytesandbrains::placeholders::Backend` for a Backend, etc.).

### 8.4 The extension import surface

Extension authors import directly from the canonical paths the
facade re-exports:

```rust
use bytesandbrains::contracts::{
    Aggregator, Backend, Codec, DataSource, Index, Model, PeerSelector,
    // Protocol lives in bytesandbrains::contracts::protocol
};
use bytesandbrains::completion::{CompletionHandle, ContractResponse, CompletionSink};
use bytesandbrains::concrete::{ConcreteComponent, ComponentHandle, ComponentPackage, RestoreError};
use bytesandbrains::placeholders;          // Backend, Index, Model, …
use bytesandbrains::module::{Module, BuildError};
use bytesandbrains::graph::{Graph, kv, attr_int, attr_float, attr_ints};
use bytesandbrains::output::Output;
use bytesandbrains::proto::onnx::{NodeProto, FunctionProto, ModelProto};
use bytesandbrains::types::{TypeNode, TYPE_TENSOR_F32, TYPE_PEER_ID};
use bytesandbrains::{Compiler, Config, install};
use bb_derive::{Concrete, Index, Backend, Model, Aggregator, Codec,
                DataSource, PeerSelector, Protocol};
```

The facade crate's top-level `pub use` chain (in `src/lib.rs`)
re-exports the same items so `use bytesandbrains::*;` works for
ergonomic prototyping.

### 8.5 Dataset / Sampler / DataSource

Data-loading helpers ship as concrete `DataSource` implementations
under `bytesandbrains::ops::*` (see the `bb-ops` crate's
data-source components). Application authors implement
`bb::contracts::DataSource` directly when they need a custom data
pipeline — the framework treats the role identically to any other
extensible component.

---

## Part 9 — Language bindings

We don't design for FFI specifically — the no-public-generics rule
(Principle 1) is sufficient. Once every public struct and function
in the framework's API has signatures with no `<T>` syntax, a
binding generator can wrap them mechanically. The Python /
TypeScript / C++ / Swift binding crates each re-expose the same
chain shape, same method names, same constructors. The framework's
job is to keep the public surface generic-free; the binding crates
do straightforward wrapping work over that.

The framework crate's structs can carry internal generics
(`DataSource<D: Dataset>` is fine because the user never writes the
`<D>` — they call `DataSource::new(dataset, batch_size)` and pass
the result to `with_data_source(source)`). What matters is what the
user sees in the call chain, not what's inside the types.

---

## Part 10 — Pulling it all together: worked example

A complete application that wires up a custom Index and uses it in
a Module. The pattern follows `examples/custom_index_hnsw.rs`.

```rust
use std::task::{Context, Waker};

use serde::{Deserialize, Serialize};

use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::contracts::Index as IndexContract;
use bytesandbrains::framework::Address;
use bytesandbrains::graph::Graph;
use bytesandbrains::ids::PeerId;
use bytesandbrains::ingress::IngressEvent;
use bytesandbrains::module::Module;
use bytesandbrains::placeholders::Index;
use bytesandbrains::{install, Compiler, Config};

// User's concrete Index. `bb::Concrete` derives the ConcreteComponent
// trait (serialize / restore via bincode, new via clone-construct);
// `bb::Index` emits the IndexRuntime bridge + inventory submission.
#[derive(Default, Clone, Serialize, Deserialize,
         bb_derive::Concrete, bb_derive::Index)]
struct MyFancyIndex {
    capacity: u32,
}

impl IndexContract for MyFancyIndex {
    type Error = std::convert::Infallible;

    fn add(
        &mut self,
        _vec: &[f32],
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error> {
        completion.complete(Ok(0));
        ContractResponse::Later
    }

    fn search(
        &self,
        _query: &[f32],
        _k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error> {
        completion.complete(Ok(Vec::new()));
        ContractResponse::Later
    }

    fn remove(
        &mut self,
        _id: u64,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        completion.complete(Ok(()));
        ContractResponse::Later
    }
}

// User's Module. The `Index` placeholder field records the role op;
// the compiler matches the recorded slot to the bind declaration.
struct EmbeddingPipeline {
    index: Index,
}

impl Module for EmbeddingPipeline {
    fn name(&self) -> &str { "EmbeddingPipeline" }

    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let _ = self.index.search(g, query, 3);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = EmbeddingPipeline { index: Index };

    // AUTHOR: record program shape.
    let model = app.build()?;

    // COMPILE: bind concretes + run the 18-pass compiler.
    let compiled = Compiler::new()
        .bind_index::<MyFancyIndex>("primary")
        .compile(model)?;

    // INSTALL: construct concretes from the inventory, bring the
    // Node up, and route the target function as the engine's root
    // graph.
    let mut node = install(
        PeerId::from(1u64),
        Address::empty(),
        compiled,
        "EmbeddingPipeline",
        Config::new(),
    )?;

    // Drive it.
    let query_bytes = bincode::serialize::<Vec<f32>>(&vec![1.0, 2.0, 3.0, 4.0])?;
    node.ingress_handle().push(IngressEvent::Invoke {
        module_name: "EmbeddingPipeline".into(),
        inputs: vec![("query".into(), query_bytes)],
        exec_id: bytesandbrains::ids::ExecId::from(0u64),
    });

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match node.poll(&mut cx) {
            std::task::Poll::Ready(steps) => {
                if steps.is_empty() {
                    break;
                }
            }
            std::task::Poll::Pending => break,
        }
    }

    Ok(())
}
```

The binary contains:

- The `bytesandbrains` facade plus the workspace crates it
  re-exports (`bb-ir`, `bb-dsl`, `bb-compiler`, `bb-runtime`,
  `bb-derive`, `bb-ops`).
- User code for `MyFancyIndex` + `EmbeddingPipeline`.

No integration crates the user didn't reference are pulled in —
DCE drops the unused branches of `bb-ops`.

---

## Part 11 — Open questions and tradeoffs

A few decisions in this design need calibration with you. None
block; flagging for review.

### 11.1 The dual-method-set duplication

Both `Backend` (placeholder) and a concrete backend like
`CpuBackend` have the same method names (`matmul`, `add`, etc.).
The framework ships the placeholder's method bodies once; concrete
impls supply runtime behavior through their `BackendRuntime` impl
(generated by `#[derive(bb::Backend)]`) and a Contract trait impl
(`impl bb::contracts::Backend for CpuBackend`). The DSL surface
and the runtime surface stay in sync because both sides reference
the same `ai.onnx` op catalog.

### 11.2 Binary size — feature gates vs separate crates

I propose separate crates for every concrete impl. An alternative is
one big `bytesandbrains` crate with feature flags. Separate crates
are cleaner (explicit dependencies, no `[features]` matrix to
maintain) but require more git/cargo overhead. I lean separate
crates; the framework crate stays small and integration crates
evolve independently.

---

## Part 12 — Talking back

Issues raised during design, and where the implemented surface
lands on each:

> "structs that are exposed to the api each struct owns it methods that mutate the a dsl"

Each component is a struct; its DSL methods take `&mut Graph` and
push NodeProtos. The placeholder unit structs in
`bytesandbrains::placeholders` carry the recording-only methods; a
concrete impl supplies the runtime behavior through its Contract
trait impl (`bb::contracts::Index`, `bb::contracts::Backend`, …).

> "we have the generic versions, but also the actual impl very op as well. meaning every component has a correlated frontend."

Placeholders (unit structs, framework-shipped) and concrete impls
(framework- or app-shipped, paired with `#[derive(bb::<Role>)]`)
share the same role contract. The placeholder records into the
graph; the bound concrete supplies dispatch at install.

> "the version where we make a struct impl Module and just run data through it"

The `Module` trait is minimal: `name()` + `body(&mut Graph)`. The
body declares inputs through `g.input("name")` and emits outputs
through `g.output(name, value)` (local) or
`g.net_out(name, peers, value)` (network). The body returns `()`.
Sub-Module composition uses the `child.call().input(...).build(g)`
chain. Cross-Module data flow inside a single body happens through
ordinary `Output` threading; cross-Node flow happens through
`net_out` after the compiler partitions at wire boundaries.

> "Is there a way we can unify the Module and Graph objects"

Module is the unit, Graph is internal recording machinery. The user
implements `body()`; the framework provides default `op()` (used
for sub-Module composition) and `build()` (the top-level entry).
`Module::build()` returns ONE pre-compile `ModelProto`; partitioning
is downstream in `Compiler::compile`. The Node lifecycle is
`module.build() → Compiler::new().bind_*(...).compile(model) →
bb::install(peer, addr, model, target, config)`. User code never
constructs a Graph directly; it receives one as a parameter to
`body()`.

> "we need generator functions to instantiate the real components on the node"

Each concrete impl implements `ConcreteComponent` (TYPE_NAME,
Config, Error, new, serialize, restore) and registers itself
through `#[derive(bb::Concrete)]`'s emitted `inventory::submit!`
block. `bb::install` looks the type up in the inventory, supplies
the per-slot config from `Config::new().with(...)`, and calls
`T::new(&config)` to materialize the instance. Snapshot/restore
reuses the same captured fn pointers.

> "i'm worried that the binary will not be able to be small, as it will have to grow with every op or component we ever have"

Part 7's binary-size strategy: integration crates are independently
versioned and the user's `Cargo.toml` pulls in only what they need.
The framework crates ship placeholders + runtime traits + engine.
Concrete implementations live in `bb-ops` (and downstream
integration crates) with each module gated by its `inventory`
submission so DCE drops unreferenced entries.

> "the infra for creating a index, or model, or dataloader (the core traits roles that have ops) very easily for the user in rust"

The extension recipe (Part 8) is five mechanical items: struct,
serde derives, `bb::Concrete` derive, `bb::<Role>` derive, Contract
trait impl. The role contracts live in
`bytesandbrains::contracts`; the placeholder unit structs live in
`bytesandbrains::placeholders`; the `Compiler::bind_<role>::<T>(slot)`
chain ties recorded slots to concrete types. See
AUTHORING_COMPONENTS.md for the long-form walkthrough.

> "applications to extend dispatch seamlessly with their own database etc"

A user's `MyFancyIndex` impl wraps their database, implements
`bb::contracts::Index`, derives `bb::Concrete` + `bb::Index`, and
binds at compile-time through
`Compiler::new().bind_index::<MyFancyIndex>("primary")`. No
proc-macros at the user's call site beyond the derives, no
generics in the install chain, no special-casing, no manual
registry call.

> "the Node api also needs to be very simple here. Like we talked about. Ie no generics or anything"

The user-facing Node lifecycle (Part 5) has zero visible generics
at the call sites application code writes. `Compiler::bind_*::<T>`
makes `T` an explicit type parameter at the bind site (which the
type system uses to enforce role conformance), but the install
call (`bb::install(peer, addr, model, target, config)`) and the
runtime calls (`node.poll(...)`, `node.ingress_handle()`, …) take
plain values.

> "Dont worry about FFI all we need is no generics and its trivial to make language bindings."

Principle 1 (Part 1) states the rule directly: the public API
surface is plain structs and plain functions. Internal generics
inside framework types are explicitly allowed
(`Compiler::bind_index::<T>` is fine because `T` is named at the
bind site and erased by the time the artifact ships). Part 9 stops
short of speccing FFI mechanics — once the no-public-generics rule
holds, binding generators wrap the surface mechanically. The
public API is designed; bindings fall out.

> "the extensions can be for any of the roles. IE they can make a new any core role they just need to match the contract"

Principle 4 (Part 1) and Part 8 lead with this: every role is
extensible by the same five-item pattern. Part 8.2 is a single
table showing all eight extensible roles (Backend, Model, Index,
Aggregator, Codec, DataSource, PeerSelector, Protocol) with
their Contract trait + framework-internal `<Role>Runtime` +
placeholder + NodeProto domain. Part 8.3 walks through the recipe
for one role (Index); any other role is mechanically identical —
swap the Contract trait, swap the `bb::<Role>` derive, swap the
method set. Extension authors import directly from
`bytesandbrains::contracts`, `bytesandbrains::completion`,
`bytesandbrains::placeholders`, `bytesandbrains::Compiler`, and
`bytesandbrains::install`.

> "the graph itself already holds all the concrete components we need" / "we literally only need the op function and nothing else"

The Module trait is `name()` + `body()`. The placeholder field on
the Module struct records role ops into the graph; the compiler's
`resolve_slots` pass matches the slot stamps to the
`Compiler::bind_<role>::<T>(slot)` declarations; install
materializes the bound concretes from the inventory. The graph
carries every slot tag the compiler needs.

> "construction time objects are just throwaway, the Node materializes fresh from state bytes"

The recorder doesn't capture concrete instances at all — the
Module struct holds placeholder unit-struct fields. Concrete
instances are constructed at install from the per-slot `Config`
the user passes to `bb::install`. Snapshot/restore reuses the
same `serialize` / `restore` fn pointers captured in the
`ComponentHandle` for each installed instance.

> "we want compile time enforcement, no macros at the user boundary, polymorphism analog of a C++ base class with a virtual generator"

`ConcreteComponent` is the polymorphism contract. The
`Compiler::bind_<role>::<T>(slot)` chain bounds its parameter
`T: ConcreteComponent + <Role>Runtime`, so the compile error
surfaces at the bind site if either is missing. Library writers
use `#[derive(bb::Concrete)]` + `#[derive(bb::<Role>)]` for
ergonomic codegen; the derives expand to the same trait impls a
hand-written component would supply.

> "nodes should only get graphs at construction time"

The compiled `ModelProto` is supplied once to `bb::install`.
Modules are bound at Node construction; their installation
persists for the Node's lifetime. There is no method on `Node`
that swaps in a new module post-install.

---

That's the full API design. See
[AUTHORING_COMPONENTS.md](AUTHORING_COMPONENTS.md) for the long-form
walkthrough using a multi-role networked component as the running
example.
