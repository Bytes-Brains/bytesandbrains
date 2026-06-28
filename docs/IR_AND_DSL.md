# IR_AND_DSL.md — BytesAndBrains as an ONNX extension

A focused mapping of every BytesAndBrains (BB) concept to its canonical
representation in the ONNX intermediate representation (IR). BB does
not invent a parallel schema; it adds three vendor opsets, a
type-denotation namespace, and a small set of `metadata_props` keys —
nothing more. The ModelProto is the BB program.

See `crates/bytesandbrains-old/src/proto/onnx-ml.proto` for the
canonical schema this document references throughout. Field numbers
and message-member names cited here exist verbatim in that file.

---

## Part 1 — Thesis

ONNX defines three things: an extensible computation graph model,
standard data types, and built-in operators. BB extends the graph
model by registering vendor opsets in the `ai.bytesandbrains.*`
domain, extends the data types via `TypeProto.Opaque`, and supplies
its own built-in operators that complement (rather than replace)
`ai.onnx`. A BB Node is conceptually an ONNX runtime that:

- Dispatches `ai.onnx` ops to a bound backend (Burn, ONNX Runtime,
  TFLite, custom).
- Dispatches `ai.bytesandbrains.syscall` ops to the framework's
  built-in scheduler / bus / lifecycle machinery.
- Dispatches `ai.bytesandbrains.wire` ops to the bound wire runtime,
  surfacing typed envelopes for transport.
- Dispatches `ai.bytesandbrains.role.*` ops in one of two modes —
  graph-inlined or opaque Rust call — chosen per op by the bound
  runtime impl.

Everything BB needs to load, validate, snapshot, restore, and execute
a graph lives in canonical ONNX messages: `FunctionProto` for module
composition, `AttributeProto` for slot attributes and inlined
sub-graphs, `TypeProto.Opaque` for vendor types, `GraphProto.initializer`
for weights, `ModelProto.opset_import` for version negotiation,
`NodeProto.metadata_props` for the very few extension annotations the
framework requires. No parallel schema, no out-of-band binding tables,
no metadata-as-bytes workarounds.

**Role ops dispatch atomically**: every role-DSL call site emits a
NodeProto stamped with `(required_trait, slot_id)` metadata. The
engine routes by `(domain, op_type, instance)` against the
per-Node atomic dispatch table to the bound impl's
`dispatch_atomic`. Backends + Index + Aggregator + Model +
Codec + DataSource + PeerSelector + Protocol all share the
same atomic-dispatch contract; the role-method-as-subgraph splice
path is reserved for non-default-path overrides at higher
abstraction levels.

---

## Part 2 — Concept-to-proto mapping

A single dense reference for every BB concept's canonical ONNX home.
Each row cites the ONNX message and field the concept rides on.

| BB concept | ONNX representation |
|---|---|
| **Node program** | `ModelProto` (proto §444) |
| **Module** | `FunctionProto` (proto §933) with `(domain, name, overload)` identity, registered in `ModelProto.functions` (proto §516) |
| **Module body** | `FunctionProto.node: repeated NodeProto` (proto §963) |
| **Module bootstrap** | A sibling `FunctionProto` named `"<module>__bootstrap"`, stamped `metadata_props["ai.bytesandbrains.module_phase"] = "bootstrap"`. Recorded by `Module::bootstrap(&self, g)` next to `Module::body`. Install registers it on `BootstrapState::install_order` without arming the queue; the host kicks via `Node::run_bootstrap(&[BootstrapInput, ...])` — empty slice drives every install-order target with empty inputs, non-empty slice drives the named Module targets and stages each request's input formals. The engine seeds bootstrap bodies onto the frontier under a fresh `ExecId`; the per-component `is_op_locked` gate parks body ops touching any in-flight bootstrap's `ComponentRef` touch set until the bootstrap drains. See ENGINE.md §6.8. |
| **Module typed I/O** | `FunctionProto.input/output: repeated string` (proto §949–950) + `FunctionProto.value_info: repeated ValueInfoProto` (proto §994) |
| **Sub-module call** | `NodeProto { op_type: <function_name>, domain: <function_domain> }` in parent body; the runtime resolves `(domain, op_type)` against `ModelProto.functions` per ONNX's standard model-local-function rule (proto §502–516) |
| **Generic component placeholder** (`Backend`, `Model`, …) | `FunctionProto.attribute: repeated string` (proto §954) — required attribute name; the framework requires a binding at load |
| **Concrete component impl** (`BurnModel(configs)`, …) | `FunctionProto.attribute_proto: repeated AttributeProto` (proto §960) — attribute with a default value; the `AttributeProto` carries the impl's construction config in `.s` (bytes), `.t` (TensorProto for embedded weights/state), `.g` (sub-graph if construction itself is graph-shaped), or `.tp` (TypeProto for type-parameterised impls) |
| **Component slot identity** | The string name in `FunctionProto.attribute` / `attribute_proto.name` (e.g. `"backend"`, `"model"`, `"teacher"`, `"student"`) |
| **Per-node slot binding** | `NodeProto.metadata_props["ai.bytesandbrains.slot"] = "<function_attr_name>"` — points at the FunctionProto attribute that owns this node |
| **Tensor (live runtime value)** | `SlotValue` trait (Rust runtime, out-of-IR); on-wire / on-disk = `TensorProto` (proto §602) |
| **Tensor type declaration** | `TypeProto.Tensor { elem_type, shape: TensorShapeProto }` (proto §824) on `ValueInfoProto.type` (proto §204) |
| **Tensor memory ownership** | Backend-owned. `Backend::Tensor` is an `Arc`-shared handle around a backend-managed buffer (e.g. `CpuTensor(Arc<CpuBackendBuffer>)` at `bb-ops/src/backends/cpu/tensor.rs:44-65`); `Clone` is `Arc::clone`. Wire-receive of a tensor slot routes through `Backend::materialize_from_wire(type_hash, bytes: Vec<u8>) -> Result<Self::Tensor, _>` (`bb-runtime/src/contracts/backend.rs:497-522`) — the framework moves `fill.payload` into the call by value, the backend chooses pool / fresh / zero-copy adoption. Engine wraps the result in `BackendTensorCarrier` (`bb-runtime/src/slot_value.rs:43-174`) for slot residency. See [ROLES.md §Backend-owned tensor memory](ROLES.md#backend-owned-tensor-memory). |
| **Model weights / parameters** | `GraphProto.initializer: repeated TensorProto` (proto §570) — named tensors referenced by `NodeProto.input` |
| **Sparse weights** | `GraphProto.sparse_initializer: repeated SparseTensorProto` (proto §573) |
| **BB scalar types** (`Trigger`, `PeerId`, `RequestId`, `WireRequestId`, `CommandId`, `Timestamp`, `EventKind`, `CorrelationToken`) | `TypeProto.Opaque { domain: "ai.bytesandbrains", name: "<TypeName>" }` (proto §867) |
| **BB collection types** (`Vec<PeerId>`, `ResponseBatch`) | `TypeProto.Sequence { elem_type: TypeProto }` (proto §833) wrapping the canonical element type |
| **Opset declaration** | `ModelProto.opset_import: repeated OperatorSetIdProto` (proto §457) and `FunctionProto.opset_import` (proto §980) — each entry is `OperatorSetIdProto { domain, version }` (proto §915) |
| **Sub-graph carried on an op** (If/Loop branches; future role-method bodies) | `AttributeProto.g: GraphProto` (proto §182) for single sub-graph, `AttributeProto.graphs: repeated GraphProto` (proto §192) for multiple |
| **Op constant config** | `NodeProto.attribute: repeated AttributeProto` (proto §234) — typed via `AttributeProto.type` enum (proto §138), payload in one of `.f/.i/.s/.t/.g/.tp/.sparse_tensor` or their repeated variants |
| **Symbolic shape dimension** | `TensorShapeProto.Dimension { dim_param: "batch" }` (proto §807) |
| **Standard dimension semantics** | `TensorShapeProto.Dimension.denotation` (proto §814) — e.g. `"DATA_BATCH"`, `"DATA_CHANNEL"` |
| **Standard type semantics** | `TypeProto.denotation` (proto §909) — string-keyed standard semantic description |
| **Cross-Node type-identity hash** (the per-Node decoder dispatch key) | Computed at runtime from the value's `TypeProto.denotation` + the version from the relevant `OperatorSetIdProto`. Not stored in ONNX — the receiving Node computes and looks up. |
| **Quantization config** (Codec codebooks, scale/zero-point) | `GraphProto.quantization_annotation: repeated TensorAnnotation` (proto §590) |
| **Multi-device sharding hints** | `NodeProto.device_configurations: repeated NodeDeviceConfigurationProto` (proto §243); model-level config in `ModelProto.configuration` (proto §520) |
| **Training step semantics** (optional ONNX-native path) | `ModelProto.training_info: repeated TrainingInfoProto` (proto §498). BB defaults to recording training as plain ops in the inference graph; ONNX-Runtime-compatible export is opt-in (sets `update_binding`s for the trainable initializers). |
| **Role-op original-op trace** | `NodeProto.metadata_props["ai.bytesandbrains.original_op"]` — telemetry tag carrying the source `<role>:<op>` for trace-back. Routing is by `(domain, op_type, instance)` lookup in the per-Node atomic dispatch table. |
| **Module-instance identity** (for descriptive partition naming) | `NodeProto.metadata_props["ai.bytesandbrains.module_instance"]` — the composition-hierarchy chain (`<parent>_<child>_<grandchild>`) stamped by `Graph::with_module(name, |g| { ... })` scope helpers. The partition pass uses this only to *name* each wire-op-bounded partition; it is NOT the partition boundary itself (wire ops are). Distinct from the per-component `instance` key below. |
| **Concrete component type tag** | `NodeProto.metadata_props["ai.bytesandbrains.concrete_type"]` — the `ConcreteComponent::TYPE_NAME` of the component whose DSL method recorded this op. Absent for ops emitted by generic placeholders. Stamped at DSL recording time via `Graph::register_concrete::<T>(&T)`. |
| **Per-op instance disambiguator** | `NodeProto.metadata_props["ai.bytesandbrains.instance"]` — monotonic integer assigned at DSL recording time from `Graph`'s pointer-identity index. Multiple DSL calls from the same `&instance` share an `instance` value; two distinct concrete instances of the same TYPE_NAME get different values. The `partition_by_wire_ops` pass propagates it through every NodeProto it splices, merges, or moves. Distinct from `module_instance`. |
| **Generic placeholder slot tag** | `NodeProto.metadata_props["ai.bytesandbrains.required_trait"]` + `["ai.bytesandbrains.slot_id"]` — stamped at DSL recording time via `Graph::register_generic(ptr, trait)`. Identifies a slot that must be filled at Node.build() via the user's `with_<role>(impl)` chain call. |
| **Snapshot** | `ModelProto` bytes of the resolved-state graph (every slot already filled, every `attribute_proto` populated) PLUS framework-side `TransientSnapshot` (out-of-ONNX) for in-flight engine state |
| **Wire envelope** | NOT in ONNX. Lives in `proto/bb_envelope.proto`. Payload may carry ONNX-shaped values (TensorProto bytes, Opaque-typed bincode) but the envelope itself is the transport plane, separate from the IR. |
| **Function bodies (hoisted sub-Modules + backend subgraphs)** | `ModelProto.functions[]` per ONNX (proto §516). The Node holds ONE canonical `ModelProto` — every registered Module's main partition function + every hoisted/collapsed sub-function is one entry in `functions[]`, deduped by `(domain, name, overload)` at register time (linker ODR check). |
| **Function call** | A plain `NodeProto` whose `(op_type, domain, overload)` matches a registered `FunctionProto`'s `(name, domain, overload)`. Per the ONNX spec, this is the canonical call mechanism. No special call op_type — same NodeProto shape as any other. |
| **Hoisted sub-Module domain** | `ai.bytesandbrains.module`. `FunctionProto.name` is `Hoist_<chain>_<body_hash>` where `<chain>` is the joined `with_module` scope chain and `<body_hash>` is a hex hash over the canonicalized body (positional formals `__hoist_in_<i>`, `__hoist_out_<j>`, `__hoist_v_<n>`). Identical bodies — whether from N invocations in one Module or one body shared across N registered Modules — converge on the same name and dedupe at link time. |
| **Function-call overload convention** | Always empty string. Multi-instance disambiguation rides on the function `name` (`<type>#<instance>` for concrete bindings, the full scope chain for hoist), so `overload` is unused. |

That's the entire BB-to-ONNX mapping in a single page. Everything that
follows elaborates on these rows; nothing introduces a row not in this
table.

---

## Part 3 — Generic vs concrete components via FunctionProto attributes

ONNX `FunctionProto` already has the exact distinction BB needs
between "slot to be filled at load" and "slot with a default already
specified":

- `FunctionProto.attribute: repeated string` (proto §954) — names of
  attributes the function REQUIRES from its caller. No default. A
  caller that does not supply one is malformed.
- `FunctionProto.attribute_proto: repeated AttributeProto` (proto §960)
  — attributes with a default `AttributeProto` payload. The caller MAY
  override; if not, the default is used.

BB uses these two lists, with NO INVENTION of a parallel schema, to
distinguish generic placeholders from concrete impls:

### Generic placeholder slot — required attribute, no default

```
Module struct in Rust:
  struct MyModule {
      backend: Backend,   // unit-struct placeholder
      // …
  }

Recorded in FunctionProto for MyModule:
  function.attribute = ["backend"]
  // "backend" appears in `function.attribute` but NOT in
  // `function.attribute_proto`. Required, no default.
```

At load:

- The framework walks `function.attribute`. Each entry is a slot
  needing a runtime impl. The user supplies bindings via the chained
  Node API (`with_backend(impl)`); the framework verifies the bound
  impl satisfies the trait implied by the slot's name + opset.
- If any required attribute lacks a binding, load fails with
  `LoadError::UnboundGenericSlot { slot_name }`.

### Concrete impl slot — defaulted attribute carrying construction config

```
Module struct in Rust:
  struct MyModule {
      model: BurnModel,    // concrete with configs
      // …
  }
  let m = MyModule {
      model: BurnModel::new(config_0, config_1),
      …
  };

Recorded in FunctionProto for MyModule:
  function.attribute_proto = [
      AttributeProto {
          name: "model",
          type: STRING,        // or TENSOR / GRAPH / TYPE_PROTO,
                               // depending on the impl's construction shape
          s: <serialized BurnModel construction state, bincode bytes>,
          metadata_props: [
              ("ai.bytesandbrains.concrete_type", "burn_integration::BurnModel"),
          ],
      },
      …
  ]
```

The `AttributeProto` is fully expressive:

- `.s: bytes` — opaque serialized state (bincode/serde) for impls
  whose construction is "give me these bytes and I'll deserialize"
- `.t: TensorProto` — for impls whose construction is "give me these
  weights" (e.g. a `LoadedMlp` initialized from a TensorProto)
- `.g: GraphProto` — for impls whose construction itself is graph-
  shaped (e.g. a "model defined by this ONNX function")
- `.tp: TypeProto` — for impls parameterised by type metadata
- `.metadata_props["ai.bytesandbrains.concrete_type"]` — the Rust type
  identifier (or Python class name); the framework looks up the
  registered deserializer for that type and reconstructs the impl

At load:

- The framework walks `function.attribute_proto`. Each entry is a
  concrete slot with construction state baked in. The framework looks
  up the registered deserializer for the type (registered at process
  startup via `Engine::register_concrete_type<T>()`) and instantiates
  the impl from the AttributeProto.
- If the deserializer is not registered for a `concrete_type`, load
  fails with `LoadError::UnregisteredConcreteType { type_name }`.

### Multi-instance per role

`function.attribute_proto` carries multiple entries with distinct
names. A Module with two `BurnModel` fields — one named `teacher`, one
named `student` — produces two entries in `attribute_proto` with names
`"teacher"` and `"student"`, both with `concrete_type =
"burn_integration::BurnModel"` but distinct construction bytes (and
therefore distinct deserialized instances at load). The NodeProtos
emitted by teacher.forward(…) carry
`metadata_props["ai.bytesandbrains.slot"] = "teacher"`; student.forward
emissions carry `slot = "student"`. Disambiguation is by attribute
name throughout.

### What this gives us

- **The framework needs no `components()` accessor on the Module
  trait**: the slot list is exactly `function.attribute +
  function.attribute_proto`. The Rust struct fields are the authoring
  surface; the FunctionProto's attribute lists are the runtime
  surface; both describe the same thing through the DSL recording.
- **Cross-language**: Python's `onnx` library knows FunctionProto
  natively. Python-side BB walks the same attribute lists. A
  ModelProto produced from Rust loads in Python (or vice versa) with
  identical slot resolution semantics.
- **Snapshot is free**: every concrete impl's construction state is
  already in the ModelProto's FunctionProto. A ModelProto round-trip
  is a snapshot round-trip.

---

## Part 4 — TypeProto.Opaque for BB-domain types

ONNX provides `TypeProto.Opaque { domain, name }` (proto §867) for
vendor-defined types whose internal layout only the vendor
understands. This is exactly the right fit for every non-tensor BB
type. We register every BB scalar / non-tensor type as an Opaque
under the `ai.bytesandbrains` domain:

```
Trigger          → Opaque { domain: "ai.bytesandbrains", name: "Trigger" }
PeerId           → Opaque { domain: "ai.bytesandbrains", name: "PeerId" }
Address          → Opaque { domain: "ai.bytesandbrains", name: "Multiaddress" }
RequestId        → Opaque { domain: "ai.bytesandbrains", name: "RequestId" }
WireRequestId    → Opaque { domain: "ai.bytesandbrains", name: "WireRequestId" }
CommandId        → Opaque { domain: "ai.bytesandbrains", name: "CommandId" }
Timestamp        → Opaque { domain: "ai.bytesandbrains", name: "Timestamp" }
EventKind        → Opaque { domain: "ai.bytesandbrains", name: "EventKind" }
CorrelationToken → Opaque { domain: "ai.bytesandbrains", name: "CorrelationToken" }
ResponseBatch    → Opaque { domain: "ai.bytesandbrains", name: "ResponseBatch" }
```

Collection types compose canonically:

```
Vec<PeerId>      → Sequence { elem_type: Opaque { ai.bytesandbrains, PeerId } }
Vec<Address>     → Opaque { domain: "ai.bytesandbrains", name: "address_vec" }
```

`Vec<Address>` rides on the concrete leaf `TYPE_ADDRESS_VEC`
(`bb-ir/src/types/builtins.rs:306-318`) rather than a generic
`Sequence` wrapper so the wire-hash (`0x0303`) distinguishes it
from a single `TYPE_MULTIADDRESS` on the wire. The carrier is
`AddressVecValue` (`bb-runtime/src/syscall/values.rs:67-68`),
populated by `AddressBook::Lookup` outputs and by `wire.Send`'s
`src_peer_addresses` envelope stamp.

Tensor types stay canonical:

```
Dense<f32>       → Tensor { elem_type: FLOAT,   shape: [dynamic] }
Dense<f64>       → Tensor { elem_type: DOUBLE,  shape: [dynamic] }
Dense<i32>       → Tensor { elem_type: INT32,   shape: [dynamic] }
Dense<i64>       → Tensor { elem_type: INT64,   shape: [dynamic] }
```

For consumers that need a stable cross-process type-identity key (the
per-Node decoder dispatch hash):

```
hash = compute_wire_hash(opaque.name, opset_version)
     = FNV-1a-64 of format!("{}@{}", opaque.name, opset_version)
```

The hash is NOT stored anywhere in ONNX. The sender computes it from
its outgoing value's TypeProto + the opset version declared in
`opset_import`; the receiver computes the same hash from its loaded
type expectation; both end up at the same `u64` and route to the
same decoder. Pure function of stable inputs; no registry needed.

### Why Opaque (not Tensor) for scalars

A naive mapping might cast `PeerId` (a `Multihash<64>`) as
`Tensor { elem_type: UINT8, shape: [N] }`. This works for byte-
level round-trip but loses type identity at the schema level:
every other byte-vector scalar in the graph collapses to the same
ONNX type, defeating the framework's typed-input-port validation.
`Opaque { domain, name }` keeps each BB scalar distinct in the IR
and in the eyes of every ONNX consumer.

---

## Part 5 — Opset catalogs

### Part 5a — `ai.bytesandbrains.syscall v1`

Framework primitives. Domain: `ai.bytesandbrains.syscall`. Version:
`1`. Dispatch: all stateless framework dispatch (the
`DispatchEntry::Stateless` variant in [ENGINE.md §8.1](ENGINE.md));
each op runs in-engine via the built-in framework Components.

| op_type | inputs | outputs | attributes | semantics |
|---|---|---|---|---|
| `Pulse` | – | `trigger: Opaque<Trigger>` | – | One-shot at bootstrap |
| `OnTrigger` | `trigger: Opaque<Trigger>` | `trigger: Opaque<Trigger>` | – | Re-fires on each input arrival |
| `Threshold` | `inputs: variadic` | `trigger: Opaque<Trigger>` | `n: int` | Fires after N inputs arrive |
| `Interval` | – | `tick: Opaque<Timestamp>` | `period_ns: int` | Periodic timer |
| `EventSource` | – | `event: Opaque<EventKind>` | `kind: int` | Fires on bus event of given kind |
| `After` | `trigger: Opaque<Trigger>` | `trigger: Opaque<Trigger>` | `delay_ns: int` | Delays trigger |
| `Limit.Acquire` | `trigger: Opaque<Trigger>` | `trigger: Opaque<Trigger>` | `name: string, n: int` | Semaphore acquire |
| `Limit.Release` | `trigger: Opaque<Trigger>` | – | `name: string` | Semaphore release |
| `Any` | `inputs: variadic` | `value: <first-arrival type>` | `group: string` | First-arrival group |
| `Gate` | `value: any, trigger: Opaque<Trigger>` | `value: any` | – | Host-controlled gate |
| `Serialize.Enqueue` | `value: any` | `trigger: Opaque<Trigger>` | `queue: string` | FIFO enqueue |
| `Serialize.Dequeue` | `trigger: Opaque<Trigger>` | `value: any` | `queue: string` | FIFO dequeue |
| `CorrelateTag` | `trigger: Opaque<Trigger>` | `token: Opaque<CorrelationToken>` | – | Mints a fresh correlation token |
| `Hold.Stash` | `value: any` | – | `slot: string` | Buffers value |
| `Hold.Flush` | `trigger: Opaque<Trigger>` | `value: any` | `slot: string` | Releases held value |
| `AppEmit` | `value: any` | – | `name: string` | Surfaces `EngineStep::AppEvent { topic: name }` to host |
| `AppNotify` | `trigger: Opaque<Trigger>` | – | `name: string` | Marker `EngineStep::AppEvent` |
| `Record` | `value: any` | – | `name: string` | Push to per-Node ring buffer |
| `IncrMetric` | `trigger: Opaque<Trigger>` | – | `name: string, delta: int` | Counter increment |
| `LifecyclePhase` | – | `trigger: Opaque<Trigger>` | `phase: int` (Shutdown=1, Snapshot=2) | Fires on `Engine::fire_lifecycle(phase)`. Bootstrap is not a lifecycle phase — see `Module::bootstrap` below. |
| `GateDispatch` | `value: any` | `value: any` | (compiler-inserted) | Edge-gate inserted by augmentation pass |
| `MintDispatch` | `trigger: Opaque<Trigger>` | `token: Opaque<CorrelationToken>` | (compiler-inserted) | Token mint inserted by augmentation pass |
| `GateManyDispatch` | `value: any, gates: variadic` | `value: any` | (compiler-inserted) | Multi-edge gate |
| `Clock` | `trigger: Opaque<Trigger>` | `now: Opaque<Timestamp>` | – | Reads system clock |
| `RngU64` | `trigger: Opaque<Trigger>` | `value: u64` | – | PRNG output |
| `Sleep` | `trigger: Opaque<Trigger>` | `trigger: Opaque<Trigger>` | `duration_ns: int` | Async timer |
| `DeadlineMatch` | `then: Opaque<Trigger>, timeout: Opaque<Trigger>` | `winner: Opaque<Trigger>` | – | First-to-fire selector |
| `PassThrough` | `value: any` | `value: any` | – | Identity |
| `Tee` | `value: any` | `outputs: variadic` | `fanout: int` | Duplicate input N ways |
| `Constant` | – | `value: any` | `value: AttributeProto` | Emit a constant at boot (value carried in the attribute) |

All syscall ops are framework-internal stateless dispatch (the
`DispatchEntry::Stateless` variant in ENGINE.md §8.1; not routed
through the atomic dispatch table). They run on the framework's
built-in dispatch through `RuntimeResourceRef`'s scheduler /
event_source / bus / outbound_queue.

### Engine event channels — function signature vs. AppEmit

There are TWO complementary paths that surface `EngineStep::AppEvent`
to the host; both coexist and both produce the same step variant.

**(a) Top-level Module's function signature.** The Module the host
binds with `bb::install(peer, addrs, compiled, &[target], Config::new())`
exposes its `function.input` ports as ingress trigger sites and its
`function.output` ports as engine-observable result sites. When a
value lands at one of those output sites AND no downstream consumer
in the function reads it, the engine emits
`EngineStep::AppEvent { topic: <output port name> }`. Sub-Module
outputs do NOT take this path — their outputs always have a
downstream consumer in the parent's body.

**(b) Explicit syscall ops.** `AppEmit` / `AppNotify` can be placed
anywhere in the graph, including inside deeply nested sub-Modules.
They fire mid-cycle, push to `framework.pending_app_events`, and
Phase 8 drains them into `EngineStep::AppEvent`. Use this channel
for intermittent reporting / progress events that don't fit the
single-final-output shape.

### Part 5a.1 — `ai.bytesandbrains.address_book v1`

DAG-mutable `AddressBook` ops. Domain: `ai.bytesandbrains.address_book`.
Version: `1`. Dispatch: custom ops registered via `bb::register_op!`
in `bb-ops/src/syscalls/peers/`; the engine routes through the
shared atomic dispatch path. Carriers: `TYPE_PEER_ID`,
`TYPE_MULTIADDRESS`, `TYPE_ADDRESS_VEC`.

| op_type | inputs | outputs | attributes | semantics |
|---|---|---|---|---|
| `Insert` | `peer: PeerId, address: Multiaddress` | – | – | New peer → `add_peer(peer, vec![addr])`; known peer → `register_address(peer, addr)`. Errors on empty list / `Full`. (`bb-ops/src/syscalls/peers/insert.rs`) |
| `InsertMany` | `peer: PeerId, addresses: AddressVec` | – | – | New peer → `add_peer(peer, addrs)`; known peer → one `register_address` per address. Errors on empty input / `Full`. (`bb-ops/src/syscalls/peers/insert_many.rs:33-67`) |
| `Lookup` | `peer: PeerId` | `addresses: AddressVec` | – | Full ordered slice via `AddressBook::lookup`. Errors on unknown peer / empty list. (`bb-ops/src/syscalls/peers/lookup.rs:29-49`) |

The `AddressVec` output type lands on `TYPE_ADDRESS_VEC`
(`ai.bytesandbrains.address_vec`, wire-hash `0x0303`,
`bb-ir/src/types/builtins.rs:306-318`). The receiver-side merge
inside `Engine::poll` (`bb-runtime/src/engine/poll.rs:1005-1062`)
calls the underlying `AddressBook` methods directly rather than
recording syscalls — the syscall surface exists for discovery
protocols that compile address propagation into a graph.

DSL helpers live at `bb-dsl/src/syscalls.rs:55-83`
(`address_book_insert_many`, `address_book_lookup`); the
single-address `Insert` path is runtime-internal.

### Part 5b — `ai.bytesandbrains.wire v1`

Network endpoint ops. Domain: `ai.bytesandbrains.wire`. Version: `1`.
Dispatch: the engine registers `Send` and `Recv` as stateless
syscalls at construction (`src/syscall/wire.rs`). There is no
`WireRuntime` binding — wire is engine-native infrastructure.

| op_type | inputs | outputs | attributes | semantics |
|---|---|---|---|---|
| `Send` | `data: any, dest: Address` (multiaddr) | – | – | Fire-and-forget broadcast. N typed `data` inputs are packed as N `SlotFill`s in one envelope to `dest`. |
| `SendReqBatched` | `data: any, dest: Address` | `req_id: Opaque<RequestId>, responses: Opaque<ResponseBatch>` | – | Batched request/response; `responses` fires ONCE when cohort completes |
| `SendResp` | `data: any, dest: Address, req_id: Opaque<RequestId>` | – | – | Reply to an inbound request |
| `Recv` | – | `trigger: Opaque<Trigger>, payload: any` | `payload_type: TypeProto (via attribute_proto.tp)` | Declare inbound type acceptance. The Recv's `NodeSiteId` becomes the routable destination; senders construct `/site/<id>` suffixes for it. Inbound payload bytes materialise into a typed `SlotValue` via the shared `wire_decoder_registry` per [WIRE.md §5.4](WIRE.md#typed-receive) — the same registry the `CompositeValue` codec consults, symmetric with Bundle's wire encode. |
| `RecvReq` | – | `trigger: Opaque<Trigger>, payload: any, req_id: Opaque<RequestId>` | `payload_type: TypeProto` | Declare inbound request acceptance |
| `RecvRespBatched` | `req_id: Opaque<RequestId>` | `trigger: Opaque<Trigger>, responses: Sequence<any>` | – | Receiver-side batched-response collector |

Per [ADDRESSING.md](ADDRESSING.md), `dest` is a multiaddr (Address)
not a `PeerId` — it encodes both the transport target and the per-slot
suffix that identifies the destination Recv site or component op.

**Correlation modeling.** Every inbound/outbound wire NodeProto
carries `metadata_props["ai.bytesandbrains.wire_correlation"]` with
one of `"none"`, `"request"`, `"response"`. The wire envelope's
proto-level `WireCorrelation` field (in `bb_envelope.proto`, not
ONNX) is the runtime echo of this static annotation.

**TriggerOnly classification.** Cross-Node edges carry
`metadata_props["ai.bytesandbrains.wire_transport"]` with `"data"` or
`"trigger_only"`. Set by the compiler's partition pass after walking
consumer types; the engine reads it at send-time to skip payload
encoding for trigger-only fills.

**Validator pairing.** Every `SendReqBatched` node MUST be paired with
exactly one `SendResp` node whose `req_id` input traces back to the
`SendReqBatched`'s `req_id` output. Unpaired requests fail validation
(`ValidationError::UnpairedWireRequest`).

**Streaming variants are intentionally absent.** Use `SendReqBatched`
with cohort sizing for fanout patterns.

**Allocation path (Send + Recv).** `Send` invokes
`SlotValue::to_wire_bytes` (bincode for the framework-carrier
shape; `BackendTensorCarrier::wire_encode_fn` for the
backend-mediated shape) and builds a `SlotFill { dest_suffix,
payload: Vec<u8>, trigger_only }`. The `Vec<u8>` is
framework-owned for the lifetime of the outbound envelope. `Recv`
delivers via `decode_typed_fill`
(`bb-runtime/src/engine/poll.rs:996-1083`): the framework charges
`fill.payload.len()` against
`NodeConfig::ingress_byte_budget`, branches on whether the
destination slot binds a `Backend` role
(`Engine::slot_id_to_role_ref` —
`bb-runtime/src/engine/core.rs:236`), and either `mem::take`s
the bytes into `Backend::materialize_from_wire` (tensor path,
zero memcpy on the framework side) or runs the global
`wire_decoder_registry` decoder against `&fill.payload`
(framework-carrier path). Per-fill failures (`AllocationFailed`,
`BudgetExceeded`, `BackendMaterializeFailed`, `TypeMismatch`,
`UnknownTypeHash`, `DecodeFailed`) surface as
`InfraEvent::WireReceiveError` and continue iterating sibling
fills (partial-delivery semantics). See [WIRE.md §5.4](WIRE.md#54-wire-eligibility-and-typed-receive)
for the full failure-mode catalog.

### Part 5c — `ai.bytesandbrains.role.* v1`

Six role opsets, one per role trait. Domains:
`ai.bytesandbrains.role.index`, `ai.bytesandbrains.role.model`,
`ai.bytesandbrains.role.aggregator`,
`ai.bytesandbrains.role.compressor`,
`ai.bytesandbrains.role.data_loader`,
`ai.bytesandbrains.role.peer_selector`. Version: `1` for all.

#### Part 5c.1 — Role-op dispatch: graph-returning trait methods + atomic-op opsets

Every `ai.bytesandbrains.role.*` op enters the IR as a NodeProto
stamped with `(required_trait, slot_id)` metadata. The engine
routes by `(domain, op_type, instance)` against the per-Node atomic
dispatch table to the bound impl's `dispatch_atomic`. Role methods
ARE the contract surface; there is no separate "role method returns
a GraphProto" path in the production pipeline.

##### Atomic-op opset (current pipeline)

Each `<Role>Runtime::atomic_opset()` declares the impl's per-op
domain + the typed input/output shape of each op. The DSL records
NodeProtos under that domain; the engine resolves
`(domain, op_type, instance)` → bound impl → `dispatch_atomic` at
install time. This is the canonical path for everything role-shaped:
Index ops, Aggregator ops, Backend per-op kernels, Model forward /
backward / step, Codec encode / decode, DataSource next_batch,
PeerSelector sample / current_view, Protocol custom opsets.

##### Future — role-method-returns-graph

The architecture reserves space for `<Role>Runtime::<method>` to
return a `Result<GraphProto, Self::Error>` so the compiler can
splice the body into the parent graph (enabling backend-portable
role definitions that decompose into `ai.onnx v1` math). The
splicing pipeline is not in the production compiler today; the
extension is future work for `ai.onnx`-decomposable roles.

##### Mixing per op

A single `ModelRuntime` impl freely mixes both shapes per op:

```rust
impl ModelRuntime for BurnModel {
    fn forward(&self) -> Result<GraphProto, Self::Error> {
        // Shape 1 — decomposable Gemm + ReLU + Gemm body.
        Ok(self.build_forward_graph_ai_onnx())
    }
    fn backward(&self) -> Result<GraphProto, Self::Error> {
        // Shape 2 — single atomic node referencing this impl's opset.
        Ok(single_node_graph(
            "bb-burn.BurnModel.atomic", "Backward", &["grad"], &["cmd"]
        ))
    }
    fn step(&self) -> Result<GraphProto, Self::Error> {
        // Shape 2 — optimizer state mutation; can't be a graph.
        Ok(single_node_graph(
            "bb-burn.BurnModel.atomic", "Step", &["grads"], &["cmd"]
        ))
    }

    fn atomic_opset(&self) -> AtomicOpsetDecl { /* registers Backward, Step, … */ }
    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
    ) -> Result<DispatchResult, Self::Error> {
        match op_type {
            "Backward" => /* run autograd backward; return CommandId */,
            "Step"     => /* mutate optimizer state; return CommandId */,
            _ => unreachable!(),
        }
    }
}
```

The trait's role methods are called once per Node load by the
compiler; `dispatch_atomic` is called repeatedly at execution. See
[ROLES.md §2](ROLES.md) for the full runtime-trait contract.

##### Cross-runtime portability

- **Shape-1 bodies are backend-portable.** A Module whose
  `model.forward` returns a Shape-1 body runs on any `bb::Backend`
  Contract impl declaring the opsets the body uses. Swap Burn for
  ONNX Runtime without changing the Module.
- **Shape-2 bodies pin to a specific impl.** A Module whose
  `model.backward` returns a Shape-2 body referencing
  `bb-burn.BurnModel.atomic::Backward` only runs on `BurnModel`'s
  `ModelRuntime` impl (or another impl declaring the same atomic
  opset). The IR carries the requirement ("`bb-burn.BurnModel.atomic`
  must be bound in the atomic dispatch table"); the binding answers
  with the registered impl.

##### NodeProto schema

Each role-op NodeProto stays under the impl's atomic opset:

```
NodeProto {
    op_type: "Backward",
    domain: "bb-burn.BurnModel.atomic",
    input: [<input_value_name>],
    output: [<output_value_name>],
    metadata_props: [
        ("ai.bytesandbrains.concrete_type", "bb-burn::BurnModel"),
        ("ai.bytesandbrains.instance",      "0"),
        ("ai.bytesandbrains.original_op",
         "ai.bytesandbrains.role.model::Backward"),
    ],
}
```

Routing is by `(domain, op_type, instance)` lookup in the per-Node
atomic dispatch table. The `original_op` metadata is retained for
telemetry and trace-back.

#### Part 5c.2 — Op-by-op tables

Each role op below has fixed inputs / outputs / attributes — the
contract the runtime trait's role method (`<Role>Runtime::<op>`) must
match in the GraphProto it returns. The **canonical body** column
indicates the typical shape (Shape 1 = decomposable; Shape 2 =
single-atomic per §5c.1). Concrete impls are free to choose either
shape per op — the contract is the IO signature, not the body.

##### `ai.bytesandbrains.role.index v1`

| op_type | inputs | outputs | attributes | canonical body |
|---|---|---|---|---|
| `Add` | `vec: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (stateful) |
| `Search` | `query: Tensor` | `results: Sequence<Tuple<Tensor, FLOAT>>` | `k: int` | Shape 2 (typically); Shape 1 for in-memory flat indexes |
| `Remove` | `id: Tensor (UINT64)` | `cmd: Opaque<CommandId>` | – | Shape 2 (stateful) |

##### `ai.bytesandbrains.role.model v1`

| op_type | inputs | outputs | attributes | canonical body |
|---|---|---|---|---|
| `Forward` | `input: Tensor` | `output: Tensor` | – | Shape 1 (decomposable; fuses with surrounding `ai.onnx` math) |
| `Backward` | `grad: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (autograd internals) |
| `Step` | `grads: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (optimizer state mutation) |
| `Evaluate` | `input: Tensor, target: Tensor` | `loss: Tensor` | – | Shape 1 (decomposable) |
| `ApplyDelta` | `delta: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (parameter mutation) |
| `LoadParameters` | `params: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (parameter mutation) |
| `Params` | – | `params: Tensor` | – | Shape 2 (snapshot read) |

##### `ai.bytesandbrains.role.aggregator v1`

| op_type | inputs | outputs | attributes | canonical body |
|---|---|---|---|---|
| `Contribute` | `contribution: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (buffer write) |
| `Aggregate` | `trigger: Opaque<Trigger>` | `result: Tensor` | – | Shape 1 (mean / weighted-sum / replace expressible in `ai.onnx`) |
| `CurrentTensor` | `trigger: Opaque<Trigger>` | `tensor: Tensor` | – | Shape 2 (state read) |

##### `ai.bytesandbrains.role.compressor v1`

| op_type | inputs | outputs | attributes | canonical body |
|---|---|---|---|---|
| `TrainCodebook` | `training: Tensor` | `cmd: Opaque<CommandId>` | – | Shape 2 (codebook mutation) |
| `Compress` | `t: Tensor` | `code: Tensor` | – | Shape 2 (impl-specific nearest-codeword search) |
| `Decompress` | `code: Tensor` | `t: Tensor` | – | Shape 1 (`ai.onnx::Gather` over the codebook) |

##### `ai.bytesandbrains.role.data_loader v1`

| op_type | inputs | outputs | attributes | canonical body |
|---|---|---|---|---|
| `NextBatch` | – | `batch: Tensor, labels: Optional<Tensor>` | – | Shape 2 (data source has side effects) |
| `Reset` | `trigger: Opaque<Trigger>` | `trigger: Opaque<Trigger>` | – | Shape 2 |
| `OnDataLoaded` | – | `trigger: Opaque<Trigger>` | – | Shape 2 |

##### `ai.bytesandbrains.role.peer_selector v1`

| op_type | inputs | outputs | attributes | canonical body |
|---|---|---|---|---|
| `Sample` | – | `peers: Sequence<Opaque<PeerId>>` | `n: int` | Shape 2 (state-dependent sampling) |
| `CurrentView` | – | `view: Sequence<Opaque<PeerId>>` | – | Shape 2 (state read) |

### Part 5d — `ai.onnx v1` (the minimum-viable required subset)

A `bb::Backend` Contract impl declaring `ai.onnx v1` MUST support
these 51 op types. Semantics are canonical ONNX; backends executing
them follow the standard ONNX spec. BB does NOT redefine semantics —
it only specifies the required subset for compatibility.

**Arithmetic:** `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Sqrt`,
`Exp`, `Log`, `Pow`.

**Linear algebra:** `MatMul`, `Gemm`, `Dot`.

**Activations:** `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyRelu`,
`Gelu`.

**Shape / structural:** `Reshape`, `Transpose`, `Concat`, `Split`,
`Slice`, `Squeeze`, `Unsqueeze`, `Identity`, `Cast`.

**Reductions:** `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`.

**Comparison:** `Equal`, `Greater`, `Less`.

**Normalization:** `BatchNormalization`, `LayerNormalization`.

**Conv / Pool:** `Conv`, `MaxPool`, `AveragePool`, `GlobalAveragePool`.

**Creation:** `Zeros`, `Ones`, `Constant`.

**Indexing:** `Gather`, `Scatter`.

**Control flow:** `If`, `Loop`.

Backends supporting a superset (e.g. ONNX Runtime, Burn) trivially
pass the load pre-flight check. Backends supporting a subset fail
`LoadError::UnsupportedOps` listing the missing op_types — surfaced
before any execution.

---

## Part 6 — DSL → NodeProto records

Every DSL method materializes into one or more NodeProtos. The
pattern is mechanical:

```rust
// DSL call:
self.backend.matmul(g, a, b)

// Recorded NodeProto (for a call on a concrete ConcreteComponent impl):
NodeProto {
    op_type: "MatMul",
    domain: "ai.onnx",
    input: vec![a.name.clone(), b.name.clone()],
    output: vec![g.next_site_name()],
    attribute: vec![],
    metadata_props: vec![
        StringStringEntryProto {
            key: "ai.bytesandbrains.concrete_type".into(),
            value: "bb-burn::BurnBackend".into(),  // = T::TYPE_NAME
        },
        StringStringEntryProto {
            key: "ai.bytesandbrains.instance".into(),
            value: "0".into(),  // = instance_id from Graph::register_concrete
        },
    ],
    name: "",
    doc_string: "",
    overload: "",
    device_configurations: vec![],
}
```

The DSL's contract:

- Method name maps to `op_type` via standard CamelCase (`matmul` →
  `"MatMul"`, `recv_req` → `"RecvReq"`, `forward` → `"Forward"`,
  `next_batch` → `"NextBatch"`).
- The component handle's opset (looked up from the trait it
  satisfies) maps to `domain`.
- `Output` arguments contribute their `name` strings to `input`.
- Newly-created sites get fresh names via `Graph::next_site_name()`,
  populated into `output` and returned as new `Output` handles.
- Op-specific config arguments populate `attribute` as
  `AttributeProto`s — `axis: i64` → `AttributeProto { type: INT, i:
  axis, name: "axis" }`, etc.
- Identity metadata goes into the `concrete_type` + `instance` keys
  (for ConcreteComponent impls) or the `required_trait` + `slot_id`
  keys (for generic placeholder unit structs). The DSL method
  calls `g.register_concrete::<Self>(self)` or
  `g.register_generic(self as *const _, REQUIRED_TRAIT)` at the top of
  its body; the Graph tracks pointer-identity and assigns the
  per-instance id or per-slot id, returning the values for the DSL
  method to stamp into the NodeProto.

The Output return shape mirrors the canonical ONNX op signature:

- One `Output` for single-output ops.
- A `(Output, Output, …)` tuple for multi-output ops.
- The output's `TypeNode` is statically known from the DSL method
  signature; the Graph populates `value_info` with the
  matching `ValueInfoProto.type` for downstream type checking.

There is no implicit type erasure: every Output carries its
canonical `TypeProto.denotation` so the validator can match
producer/consumer types.

---

## Part 7 — Graph identity, opset_import, version negotiation

A loaded `ModelProto` declares its opsets in `opset_import`:

```
model.opset_import = [
    OperatorSetIdProto { domain: "ai.onnx",                      version: 17 },
    OperatorSetIdProto { domain: "ai.bytesandbrains.syscall",    version: 1 },
    OperatorSetIdProto { domain: "ai.bytesandbrains.wire",       version: 1 },
    OperatorSetIdProto { domain: "ai.bytesandbrains.role.model", version: 1 },
    OperatorSetIdProto { domain: "ai.bytesandbrains.role.aggregator", version: 1 },
]
```

A `FunctionProto` body carries its own `opset_import` declaring the
opsets its inlined nodes use. This lets sub-modules import additional
opsets the parent doesn't directly use.

**Per ONNX semantics, when multiple opsets declare the same op_type,
the runtime binds against the HIGHEST version in the imported sets.**
BB follows this rule verbatim. A backend supporting `ai.onnx v17` but
graph importing `ai.onnx v18` runs with v18 semantics for any v18-
defined ops; v17-stable ops use v17 semantics.

**Pre-flight check at load.** The framework walks `opset_import` and
verifies the bound runtime impls cover each opset's required ops:

- For `ai.onnx v<n>`: the bound backend's `supported_ops()` covers
  every `ai.onnx` op_type appearing in the graph.
- For `ai.bytesandbrains.role.<role> v<n>`: the bound role runtime's
  `supported_ops()` covers every role op_type appearing.
- For `ai.bytesandbrains.syscall v<n>`: framework-built-in; no binding
  required (always supported).
- For `ai.bytesandbrains.wire v<n>`: framework-built-in; the engine
  registers `Send` and `Recv` as stateless syscalls, no binding
  required (always supported).

Failure produces `LoadError::IncompatibleRuntime { opset, missing_ops }`.

---

## Part 8 — Wire envelope (out-of-IR, coherent with it)

The wire envelope is **transport, not IR**. It lives in a separate
proto file (`proto/bb_envelope.proto`) and is NOT part of the ONNX
schema. The envelope's job is to carry an opaque payload between
Nodes; the IR (loaded ModelProtos on both ends) defines what the
payload means.

Envelope schema (per [ADDRESSING.md](ADDRESSING.md) — addresses
route themselves):

```proto
syntax = "proto3";
package bb.core;

enum CorrelationKind { NONE = 0; REQUEST = 1; RESPONSE = 2; }

message WireCorrelation {
  CorrelationKind kind = 1;
  uint64 wire_req_id = 2;
}

message WireEnvelope {
  repeated bytes dest_peer_addresses = 1;  // resolved address list from
                                           // AddressBook::lookup(peer);
                                           // transport picks one entry.
                                           // Lookup miss → no envelope
                                           // (EngineStep::PeerResolveFailed
                                           // surfaces instead).
  repeated SlotFill fills = 2;             // batched fills
  WireCorrelation correlation = 3;         // request/response pairing
  // ... fields 4-7: deadline propagation + RTT piggyback + ...
  repeated bytes src_peer_addresses = 8;   // sender's local-address bag
                                           // (snapshot of `ctx.local_addresses()`
                                           // at send time); receiver merges
                                           // into AddressBook entry for the
                                           // sender. Capped at decode time
                                           // via EnvelopeCaps.
}

message SlotFill {
  bytes dest_suffix = 1;   // per-slot multiaddr suffix (intra-node):
                           //   /site/<NodeSiteId>           — data plane
                           //   /component/<cref>/op/<name>  — control plane
  bytes payload = 2;       // wire-encoded bytes; empty when trigger_only
  bool  trigger_only = 3;
}
```

Peer routing is the resolved
`dest_peer_addresses: repeated bytes` (the wire syscall populates
it from `AddressBook::lookup(peer)`; the transport adapter picks
one entry by capability); intra-node routing is each fill's
`dest_suffix`. Receivers parse the suffix segments to dispatch (see
[ADDRESSING.md](ADDRESSING.md) for the canonical reference,
including the DAG-mutable `peers/` syscall ops + `PeerResolveFailed`
lifecycle event).
- `SlotFill.dest_suffix` ending in `/site/<NodeSiteId>` identifies
  the slot inside the receiver's installed graph. The slot's
  declared `TypeNode` (looked up from `ValueInfoProto.type` via the
  installed graph's `site_names` map) tells the receiver which
  decoder to use.
- `SlotFill.dest_suffix` ending in `/component/<ComponentRef>/op/<name>`
  routes directly to `components[cref].dispatch_atomic(name, ...)`
  for control-plane components. The component owns its payload
  encoding.

The envelope plane and the IR plane never collide: the envelope is
how Nodes exchange bytes; the IR (graphs + type-meta + addresses)
is what makes those bytes meaningful on the receiver side.

---

## Part 9 — Worked example: canonical SplitLearning Module

Source Rust:

```rust
struct SplitLearning {
    backend: Backend,                            // generic
    network_server: NoBarrierOneShot,            // concrete
    network_client: BarrierNetworkReqResp,       // concrete
    model: BurnModel,                            // concrete
    codec: ProductQuantization,                  // concrete
    gossip: Cyclone,                             // concrete
    aggregator: WeightAggregator,                // concrete
}

impl Module for SplitLearning {
    fn name(&self) -> &str { "SplitLearning" }

    fn op(&self, g: &mut Graph, _inputs: &[Output]) -> Vec<Output> {
        let (_t1, enc_in) = self.network_server.recv(g);
        let dec_in = self.codec.decompress(g, enc_in);
        let dec_out = self.model.forward(g, dec_in);
        let enc_out = self.codec.compress(g, dec_out);

        let peers = self.gossip.sample(g, 5);
        let (req_id, _ack) = self.network_client.send_req_batched(g, enc_out, peers);

        let (_t2, batched_grads) = self.network_client.recv_responses(g, req_id);
        let dec_grads = self.codec.decompress(g, batched_grads);
        let avg_grad = self.aggregator.aggregate(g, dec_grads);
        let _ = self.model.step(g, avg_grad);
        let _ = self.model.backward(g, avg_grad);
        vec![]  // no top-level outputs
    }
}

// Application entry point.
let modules = SplitLearning { /* ... */ }.build()?;
```

Produced `ModelProto`:

```proto
ModelProto {
  ir_version: 12,
  producer_name: "bytesandbrains",
  producer_version: "0.9.0",
  domain: "user.app",
  model_version: 1,
  opset_import: [
    {domain: "ai.onnx",                         version: 17},
    {domain: "ai.bytesandbrains.syscall",       version: 1},
    {domain: "ai.bytesandbrains.wire",          version: 1},
    {domain: "ai.bytesandbrains.role.model",    version: 1},
    {domain: "ai.bytesandbrains.role.aggregator", version: 1},
    {domain: "ai.bytesandbrains.role.compressor",  version: 1},
    {domain: "ai.bytesandbrains.role.peer_selector", version: 1},
  ],
  graph: GraphProto {
    name: "SplitLearning",
    node: [
      NodeProto {
        op_type: "SplitLearning",
        domain: "user.app",
        // The top-level graph just calls the SplitLearning function:
        input: [],
        output: [],
        metadata_props: [{
          key: "ai.bytesandbrains.module_instance",
          value: "SplitLearning#0",
        }],
      },
    ],
  },
  functions: [
    FunctionProto {
      name: "SplitLearning",
      domain: "user.app",
      // Generic placeholders (required, no default):
      attribute: ["backend"],
      // Concrete impls (defaulted; payload carries construction config):
      attribute_proto: [
        AttributeProto {
          name: "network_server",
          type: STRING,
          s: <bincode: NoBarrierOneShot construction state>,
          metadata_props: [{
            key: "ai.bytesandbrains.concrete_type",
            value: "framework_wire::NoBarrierOneShot",
          }],
        },
        AttributeProto {
          name: "network_client",
          type: STRING,
          s: <bincode: BarrierNetworkReqResp construction state>,
          metadata_props: [{
            key: "ai.bytesandbrains.concrete_type",
            value: "framework_wire::BarrierNetworkReqResp",
          }],
        },
        AttributeProto {
          name: "model",
          type: STRING,
          s: <bincode: BurnModel construction state + weights references>,
          metadata_props: [{
            key: "ai.bytesandbrains.concrete_type",
            value: "burn_integration::BurnModel",
          }],
        },
        AttributeProto {
          name: "codec",
          type: STRING,
          s: <bincode: ProductQuantization{M, N} state>,
          metadata_props: [{
            key: "ai.bytesandbrains.concrete_type",
            value: "framework_compressors::ProductQuantization",
          }],
        },
        AttributeProto {
          name: "gossip",
          type: STRING,
          s: <bincode: Cyclone{C, H, S} state>,
          metadata_props: [{
            key: "ai.bytesandbrains.concrete_type",
            value: "framework_peer_selector::Cyclone",
          }],
        },
        AttributeProto {
          name: "aggregator",
          type: STRING,
          s: <bincode: WeightAggregator state>,
          metadata_props: [{
            key: "ai.bytesandbrains.concrete_type",
            value: "framework_aggregators::WeightAggregator",
          }],
        },
      ],
      input: [],   // SplitLearning takes no graph inputs at top level
      output: [],  // and produces no top-level outputs (effects via wire + step)
      opset_import: [...],  // mirrors ModelProto.opset_import
      node: [
        NodeProto {
          op_type: "Recv",
          domain: "ai.bytesandbrains.wire",
          input: [],
          output: ["site_1", "site_2"],  // trigger, encoded_input
          attribute: [
            AttributeProto {
              name: "payload_type",
              type: TYPE_PROTO,
              tp: TypeProto.Tensor { elem_type: FLOAT, shape: <dynamic> },
            },
          ],
          metadata_props: [
            {key: "ai.bytesandbrains.concrete_type", value: "framework_wire::NoBarrierOneShot"},
            {key: "ai.bytesandbrains.instance",      value: "0"},
          ],
        },
        NodeProto {
          op_type: "Decompress",
          domain: "ai.bytesandbrains.role.compressor",
          input: ["site_2"],
          output: ["site_3"],
          metadata_props: [
            {key: "ai.bytesandbrains.concrete_type", value: "framework_compressors::ProductQuantization"},
            {key: "ai.bytesandbrains.instance",      value: "0"},
          ],
        },
        NodeProto {
          op_type: "Forward",
          domain: "ai.bytesandbrains.role.model",
          input: ["site_3"],
          output: ["site_4"],
          metadata_props: [
            {key: "ai.bytesandbrains.concrete_type", value: "burn_integration::BurnModel"},
            {key: "ai.bytesandbrains.instance",      value: "0"},
          ],
        },
        NodeProto {
          op_type: "Compress",
          domain: "ai.bytesandbrains.role.compressor",
          input: ["site_4"],
          output: ["site_5"],
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "framework_compressors::ProductQuantization"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "Sample",
          domain: "ai.bytesandbrains.role.peer_selector",
          input: [],
          output: ["site_6"],
          attribute: [
            AttributeProto { name: "n", type: INT, i: 5 },
          ],
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "framework_peer_selector::Cyclone"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "SendReqBatched",
          domain: "ai.bytesandbrains.wire",
          input: ["site_5", "site_6"],   // encoded_output, peers
          output: ["site_7", "site_8"],  // req_id, responses
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "framework_wire::BarrierNetworkReqResp"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "RecvRespBatched",
          domain: "ai.bytesandbrains.wire",
          input: ["site_7"],
          output: ["site_9", "site_10"],  // trigger, batched_grads
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "framework_wire::BarrierNetworkReqResp"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "Decompress",
          domain: "ai.bytesandbrains.role.compressor",
          input: ["site_10"],
          output: ["site_11"],
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "framework_compressors::ProductQuantization"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "Aggregate",
          domain: "ai.bytesandbrains.role.aggregator",
          input: ["site_11"],
          output: ["site_12"],
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "framework_aggregators::WeightAggregator"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "Step",
          domain: "ai.bytesandbrains.role.model",
          input: ["site_12"],
          output: ["site_13"],
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "burn_integration::BurnModel"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
        NodeProto {
          op_type: "Backward",
          domain: "ai.bytesandbrains.role.model",
          input: ["site_12"],
          output: ["site_14"],
          metadata_props: [{key: "ai.bytesandbrains.concrete_type", value: "burn_integration::BurnModel"},
                           {key: "ai.bytesandbrains.instance", value: "0"}],
        },
      ],
      value_info: [
        ValueInfoProto {
          name: "site_2",
          type: TypeProto.Tensor { elem_type: FLOAT, shape: <dynamic> },
        },
        // ... one per intermediate value, optional but useful for validation
      ],
    },
  ],
}
```

Everything the framework needs to load, validate, snapshot, and
execute this graph is in the `ModelProto`. The Rust struct declared
the components; the DSL methods recorded the `FunctionProto.attribute`
+ `attribute_proto` + `node` lists; the concrete impls' construction
state is baked into `attribute_proto.s`. At load:

1. The framework walks `function.attribute = ["backend"]` — the user
   must supply a `bb::Backend` Contract binding via the chained Node
   API; `#[derive(bb::Backend)]` generates the runtime bridge.
2. The framework walks `function.attribute_proto` — for each entry,
   look up the registered deserializer for `concrete_type`,
   instantiate from `.s` (or `.t` / `.g` / `.tp`).
3. The compiler runs: validates the recorded NodeProtos, infers
   peer classes, partitions by wire ops, and inserts the deadline /
   dedup / backoff / peer-health gate ops on every wire path. Role
   NodeProtos stay atomic-opset entries and dispatch at runtime via
   the per-Node atomic dispatch table.
4. Pre-flight: every used op_type in every opset has a covering
   binding. Failure surfaces typed errors before any execution.

---

## Part 10 — What ONNX gives us free

By riding inside canonical ONNX messages, BB inherits without code:

- **Netron, onnxruntime, Burn's loader, TFLite's converter, the
  Python `onnx` package** all read framework graphs natively. The
  vendor opsets show as namespaced ops; the rest is just ONNX.
- **Snapshot = ModelProto bytes.** Any ONNX-aware tool opens a BB
  snapshot. Diffing, lineage analysis, visualization — all free.
- **FunctionProto-based composition is how ONNX itself models reusable
  graphs.** Inlining, parameter substitution, multi-instance — all
  spec-defined behaviors.
- **`opset_import` solves version negotiation.** The same mechanism
  used between PyTorch and ONNX Runtime works for BB graphs across
  Nodes and across framework versions.
- **`GraphProto.initializer`-based weights round-trip** without any
  serializer code on our side. A `BurnModel` whose construction state
  references initializer names exports a graph any ONNX consumer can
  load with weights intact.
- **`TensorAnnotation` for quantization** is canonical. PQ codebooks,
  scale/zero-point pairs, etc. live where every consumer expects them.
- **`TypeProto.Opaque`** is the right primitive for our domain types
  without inventing custom representations. Python's `onnx` library
  knows Opaque types as opaque (it preserves the `domain` + `name`
  without trying to interpret them); the framework's deserializer
  registry interprets them where they need interpretation.
- **Role-op bodies decompose into shared opsets or terminate at a
  single atomic-op NodeProto**, mirroring ONNX's standard-op vs
  vendor-extension distinction. Same conceptual shape applied at the
  role boundary. Toolchain knowledge transfers directly.

---

## Part 11 — The Rust-dispatch boundary (closing principle)

> **Graph decomposition stops at Rust dispatch.**
>
> Every Op in a loaded ModelProto is either (a) graph-expressible —
> its body is recoverable as a sub-`GraphProto` and the compiler may
> inline it; or (b) Rust-dispatched — the BB engine calls a Rust
> function the bound runtime supplied, and from that point the op is
> opaque to the IR. There is no third mode.
>
> This is what makes a BB Node an ONNX runtime: standard ops
> (`ai.onnx`) and graph-expressible role ops are dispatched through
> normal graph-execution machinery; opaque role ops + framework
> primitives (`ai.bytesandbrains.syscall`, `ai.bytesandbrains.wire`)
> are the vendor-specified dispatch surface.

Everything above the Rust dispatch boundary is graph-traversable:
inlineable, collapsible, partitionable, snapshottable, exportable to
any ONNX consumer. Everything below is opaque: dispatched only by the
bound runtime's Rust function, never further decomposed.

A backend's `execute_subgraph` is also a Rust-dispatch terminal: once
the BB engine hands the GraphProto to the backend, what the backend
does internally (JIT compile, fuse kernels, dispatch to ONNX Runtime,
hand to a GPU) is invisible to the IR. The IR's contract is
`(inputs, GraphProto) -> outputs`; the implementation is the
vendor's.

This invariant is what lets BB cleanly compose: graphs flow through
graphs (composition, inlining, collapse, partition); Rust runs Rust
(backends, role impls, framework primitives). The two layers don't
leak into each other. The transition is explicit and observable in
the IR via the `(domain, op_type)` pair of each NodeProto: anything
under a registered atomic-op opset is Rust dispatch, anything else is
graph-level composition.

## Update — M-phase additions

This section reflects the M1–M11 + Phase D landings.

### Module ports

Ports are declared in the Module body recording surface: `g.input(name)`
for local inputs, `g.output(name, value)` for local outputs,
`g.net_out(port, peers, value)` for network outputs, and
`g.lookup_output(port)` to pull a value the compiler has wired in from
a network input. The compiler infers the port set from the recorded
body.

### Module::bootstrap recording

`Module::bootstrap(&self, g: &mut Graph)` is the author entry
point for pre-body initialization. The trait method defaults to
no-op; authors override it next to `Module::body` and compose
their initialization graph the same way they compose `Module::body`.
Components contribute ops (`GlobalRegistryClient::Announce`,
`Index::train`, `BackendSlot::prime`, …) that the Module's
bootstrap orchestrates.

```rust
impl Module for VectorStore {
    fn bootstrap(&self, g: &mut Graph) {
        // Stage initial inputs via `g.input(name)` — same recorder
        // call body uses for top-level formals. Each input
        // becomes a declared formal on the emitted
        // `"<module>__bootstrap"` FunctionProto, addressable from
        // the host via `BootstrapInput::inputs`.
        let seed_corpus = g.input("seed_corpus");
        let _ = self.index.train(g, seed_corpus);
    }

    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let _ = self.index.search(g, query, 10);
    }
}
```

`Module::build()` emits the bootstrap recording as a sibling
`FunctionProto` named `"<module>__bootstrap"` stamped with
`metadata_props["ai.bytesandbrains.module_phase"] = "bootstrap"`
(see Part 2). The bootstrap function's `function.input` list is
the recorder's seen `g.input(name)` calls inside the bootstrap
recording, in order.

DSL helpers like `bytesandbrains::constant(g, label, type_node,
data_type)` (typed bootstrap-stage constants the compiler folds
through `expand_constant`) and `bytesandbrains::announce(g,
server_peer)` (`GlobalRegistryClient::Announce` recording with
auto-supplied local addresses + heartbeat throttle) keep common
bootstrap patterns one call wide.

The host stages bytes for each declared formal through
`Node::run_bootstrap`:

```rust
node.run_bootstrap(&[bb::engine::BootstrapInput {
    target: "VectorStore",
    inputs: &[("seed_corpus", corpus_bytes.as_slice())],
}])?;
node.poll(cx); // drives the bootstrap body to quiescence
```

The engine validates `inputs` against the target's declared
formals at the boundary: `UnknownInput` rejects extras,
`MissingInput` rejects gaps, `UnknownTarget` rejects unknown
names — all before any bytes stage. Validated requests follow the
Principle 1a copy (`try_charge → try_reserve_exact →
extend_from_slice`) and the framework-owned `BytesValue` carriers
land in the bootstrap's slot table entries at the body's fresh
`ExecId`. Caller's borrowed `&[u8]` slices may drop the moment
`run_bootstrap` returns.

A bootstrap that takes no formals records zero `g.input` calls;
the host kicks every install-order target with
`node.run_bootstrap(&[])`.

### Composition API

The canonical composition shape:

```rust
let cell_out = self.cell.call()
    .input("query", q)
    .input("incoming_grad", grad)
    .build(g);                              // returns ModuleOutputs<'_>
let response = cell_out.output("response"); // by name, not position
```

### Network primitives at module boundaries

`g.net_out(name, peers, value)` is the single-slot network-sink
primitive on the recorder. It emits a `wire.Send` NodeProto
and registers `name` as a network-typed output port on the
current function. `peers` must be a `Vec<PeerId>` output;
`value` is the payload. The compiler's `partition_by_wire_ops`
cuts the graph at `wire.Send` boundaries; `synthesize_wire_recvs`
materializes the matching `wire.Recv` NodeProto on every
consumer-side partition that reads the named port. `wire.Recv`
is compiler-synthesized and does not appear in user-authored
Module bodies. The receive site's type is inferred by the
TypeSolver from the matching `wire.Send` payload type.

### Composition: bundling typed Outputs

`g.bundle(parts: &[Output]) → Output` packs N typed Outputs into
ONE composite Output for transmission through a single port; the
matching `g.unbundle(composite, &[&TypeNode, …]) → Vec<Output>`
decomposes the envelope back into N typed children on the
receiver. The composite envelope rides `TYPE_COMPOSITE` (a new
concrete leaf under `Any`); the wire infrastructure already
supports any wire-eligible value through `wire.Send`, so the
single composite hop reuses the existing `net_out` machinery
verbatim.

The recorded NodeProto shapes:

- **Bundle** (`domain = "ai.bytesandbrains.composite"`,
  `op_type = "Bundle"`): variable-arity input `[parts[0].name,
  parts[1].name, …, parts[N-1].name]`; single output port carrying
  the assembled `CompositeValue`. Stamps
  `ai.bytesandbrains.composite.child_count` (INT) and
  `ai.bytesandbrains.composite.child_types` (comma-joined
  TypeNode denotations).
- **Unbundle** (same domain, `op_type = "Unbundle"`): single input
  `[composite.name]`; N outputs named `child_0..child_{N-1}` with
  `ValueInfoProto.denotation` stamped from the corresponding
  `part_types[i].denotation`. Each child output is the original
  concrete `SlotValue` carrier the sender bundled (`PeerIdValue`,
  `CpuTensor`, …), not a `BytesValue`. Downstream consumers
  downcast directly via
  `as_any().downcast_ref::<T>()` against the declared denotation.

#### Type-fidelity story

`CompositeValue` is in-process typed: its `children` field carries
`Vec<Box<dyn SlotValue>>` (`bb-runtime/src/syscall/values.rs:80-85`),
not a `Vec<(u64, Vec<u8>)>` bag. Bundle's invoke clones each input
via `SlotValue::clone_boxed`
(`bb-ops/src/syscalls/composite/bundle.rs:43-46`); Unbundle's
invoke emits each child via `clone_boxed`
(`bb-ops/src/syscalls/composite/unbundle.rs:61-64`). In-process
forwarding pays one `clone_boxed` per child — no bincode encode,
no decode, no opaque `BytesValue` hop.

At the wire boundary `SlotValue::to_wire_bytes` invokes
`CompositeValue`'s hand-rolled `Serialize`
(`bb-runtime/src/syscall/values.rs:114-131`), which encodes each
child as a `(type_hash, child.to_wire_bytes())` tuple. The
receiver's `Deserialize`
(`bb-runtime/src/syscall/values.rs:133-165`) reads each
`(type_hash, bytes)` pair, looks the hash up in
`wire_decoder_registry()` (`bb-ir/src/slot_value.rs:199-212`), and
materialises a typed `Box<dyn SlotValue>` carrier — so Unbundle
on the receiver downcasts to `T` even after a cross-Node hop.

The decoder registry is populated automatically by every
`register_type_node!(MyValue, &TYPE_X)` invocation
(`bb-ir/src/slot_value.rs:237-256`); a peer running a build that
does not know a given carrier's `type_hash` surfaces a typed
`SlotValueError::DecodeFailed` on receive rather than crashing.

The intended pattern: pack `(params, metadata)` once with
`g.bundle`, ship through a single `net_out`, unpack on the
receiver with `g.unbundle`. Single-port DAG semantics hold
because the bundle/unbundle pair traverses one Output between
peers; `synthesize_wire_recvs` keeps its single-port cross-
partition resolution.

Empty `parts` (Bundle) or empty `part_types` (Unbundle) panic at
recording time — composition of zero values has no semantic
meaning and is almost certainly an author bug.

### PeerSelector + SelectParams

`bb::contracts::PeerSelector::select(ctx, params, completion)` is
the generic peer-selection surface (see [ROLES.md](ROLES.md) for
the canonical `ctx` / `completion` shape every Contract method
follows). `SelectParams` carries:

- `Random { n }` — sample N peers uniformly.
- `NearKey { key, n }` — closest N peers under the selector's
  metric.
- `All` — every peer in the current view.

Concrete impls handle the variants they support and fail the
unsupported ones via `ContractResponse::Now(Err(...))`. Built-in
selectors: `GlobalRegistryServer` (centralized peer registry), `ConstantView` (fixed peer list).

### Wire op cardinality

`extract_dest_peers` accepts ONLY `PeerIdVecValue` at position 1.

### RecordedModule.module_tree

Every recorded module carries a `module_tree: Vec<ModuleTreeNode>`
with port declarations + parent/child relationships. The
`partition_by_module_boundary` pass walks this tree and emits
one partition per module + a NetworkEdge per matching
`g.net_out` → `g.lookup_output` pair.

### Multi-target compile + entry-point semantics

`Compiler::compile(module) → ModelProto` emits a single
`ModelProto` whose `functions[]` carries **every partition** produced
by `partition_by_wire_ops`. One compile call → one proto, regardless
of partition count. A federated module that partitions into `Client`
+ `Server` emits both as sibling `FunctionProto`s under
`model.functions`; sub-Module bodies and the synthesized helpers
(gate carriers, lifecycle containers) ride alongside in the same
list. The compilation passport (`ai.bytesandbrains.compiled = "v1"`)
+ per-target binding metadata
(`ai.bytesandbrains.binding.<target>.<slot> =
"<role>|<TYPE_NAME>|<slot_id>"`) stamp onto `model.metadata_props`
keyed by partition name, so the same proto carries every target's
binding spec without colliding.

`bb::install(peer_id, addresses, model, targets: &[&str], config)`
(`src/install.rs:235-338`) takes an ordered slice of target names
and installs **all** of them onto one Node. The host picks which
partitions live on each peer by passing different `targets` slices
to `install` on different peers; the proto is the same artifact
across the deployment. A peer hosting both halves of a federated
round receives `&["Client", "Server"]`; a single-Node demo passes
`&["MyModule"]`. The order is observable: bootstrap functions fire
in slice order — `BootstrapState::install_order`
(`bb-runtime/src/engine/bootstrap.rs:256-296`) is the append-only
queue the seeder walks front-to-back. See ENGINE.md §6.8.

Per-target lookup uses exact-match against `model.functions[].name`
first, then falls back to the compiler's content-hash suffix
(`<target>#<hash>`) — the partition pass stamps the hash so two
modules emitting partitions named `Client` from different
authoring crates don't collide
(`src/install.rs:356-373`).

The compiled `ModelProto` is shareable across targets at the Node
layer: `bb::install` wraps it in `Arc<ModelProto>` once via
`Node::set_model` and shares the handle across every
`Node::register_module` call so the proto bytes live on the Node
exactly once
(`src/install.rs:332-335`, `bb-runtime/src/node/mod.rs:55-65`,
`530-548`).
