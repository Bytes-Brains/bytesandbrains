# COMPILER.md — compilation pipeline

The compiler is the pre-runtime pipeline that takes a recorded
`ModelProto` and produces an engine-ready form. It validates,
mutates, dissects, inlines, collapses, and partitions the IR until
what remains is a set of per-Node sub-graphs the Engine can install
directly. Each pass is a pure function on `GraphProto` (or
`FunctionProto`); the orchestrator composes them in a single
canonical order; failure at any pass surfaces a typed error and
leaves prior state untouched.

This document specifies every pass, the order they run, the IR
invariants each one establishes, and what the compiler's output
looks like by the time the Engine sees it.

Pairs with IR_AND_DSL.md (what the IR is), ENGINE.md (the runtime),
and ENGINE.md (what consumes the compiler's output).

---

## Part 1 — Overview

**The compiler runs inside `Compiler::compile()`.** A user-authored
Module's `op(&self, g: &mut Graph, inputs: &[Output]) -> Vec<Output>`
records a `ModelProto` whose `functions` list contains FunctionProtos
for every Module + sub-Module (sub-Modules just call their own `op()`
from inside the parent's `op()` body). Op nodes are recorded as the
user wrote them, tagged with slot metadata, typed via `value_info`
and `TypeProto` declarations.

That recorded ModelProto is not directly executable. Compilation
runs in three phases:

1. **`Module::build() → ModelProto`** records the program shape (no
   compiler work — pure recording).
2. **`Compiler::new().bind_<role>::<T>("slot").compile(module) → ModelProto`**
   runs the 17 canonical passes documented below and returns a single
   `ModelProto` whose `functions[]` carries every partition (root
   plus hoisted sub-Modules plus backend subgraphs), with the
   compilation passport and binding metadata stamped onto
   `metadata_props`.
3. **`bb::install(peer_id, addresses, compiled, targets: &[&str], Config::new())`**
   picks every function whose name matches an entry in `targets`,
   installs each as a host-facing entry-point graph, and brings
   the Node up. Single-target installs pass `&["MyModule"]`; peers
   hosting more than one partition (e.g. a federated peer running
   both `Client` and `Server` from the same compile) pass each
   partition name in slice order.

Single-Node Modules collapse to one partition. Multi-Node (federated-
style) Modules produce one partition per BB Node; the host picks
which partition(s) live on each peer by passing the corresponding
`targets` slice to `bb::install`. The same compiled `ModelProto`
can serve a peer hosting only `Client`, a peer hosting only
`Server`, AND a peer hosting both — different `targets` slices,
same proto bytes.

The pipeline covers:

- **Wire-op surfacing** — sub-Module bodies that contain wire ops
  surface to the root function so dissection sees them.
- **Deadline stamping** — every `wire.Send` carries a static
  `deadline_ns` budget before validation runs.
- **Structural validation** — does it parse, does it reference
  declared opsets correctly, do all wire requests pair with
  responses?
- **Variant expansion** — does each Op have its variant choice
  materialized?
- **Auto-edge insertion** — do framework-mediated implicit edges
  (timers, lifecycle markers, sub-module IO crossing) exist
  explicitly?
- **Type resolution + peer-class inference** — every value
  resolves to a concrete `TypeNode`; values carrying a `PeerClass`
  denotation pick up explicit stamps.
- **Wire-receiver synthesis** — every `wire.Send` gets a paired
  `wire.Recv` on each consumer scope chain.
- **Multi-Node dissection** — partition the recorded function at
  wire ops; each partition becomes one installable target.
- **Slot resolution** — match each NodeProto's slot metadata to a
  `bind_<role>` declaration from the `Compiler`.
- **Wire-edge analysis** — classify cross-partition edges
  (Data vs TriggerOnly), group by destination for batching.
- **Gate insertion** — splice dedup-rx, peer-health-rx/tx, and
  backoff-rx/tx gates adjacent to wire ops.
- **Async-deadline stamping** — every async-suspending NodeProto
  carries a runtime deadline.
- **Runtime-complete validation** — final pre-flight: every slot
  bound, every opset covered, every wire pair matched.

These run as a fixed pipeline. The order is non-arbitrary —
explained in §3.

---

## Part 2 — Terminology

Two distinct uses of "node" appear in this doc; pin them down:

- **BB Node** — a runtime instance. A process, peer, or simulator
  instance. Hosts an Engine. Communicates with other BB Nodes via
  `WireEnvelope`s. Capitalized.
- **NodeProto / op** — a single op in an ONNX `GraphProto.node[]`
  list. The IR-level unit. Lowercased as "op" in this doc to avoid
  collision.

"BB Node instance" or "Node role" refers to which BB Node the user
intends an op to run on. The partition pass dissects multi-instance
graphs into per-Node sub-graphs.

---

## Part 3 — The canonical pipeline

The pipeline runs in the order declared by
[`bb_compiler::CANONICAL_PASS_NAMES`][canon]. Each pass has a
corresponding `bb-compiler/src/<pass>.rs` source file. Passes are
keyed by name so `Compiler::without_stage("name")` can disable
canonical passes for test scenarios.

[canon]: ../bb-compiler/src/runner.rs

**Pre-pipeline passes — run before `run_pipeline`:**

| Pass | Purpose | Reads |
|---|---|---|
| `refine_polymorphic_value_info` | Narrows `TYPE_TENSOR` placeholder denotations on Contract-method NodeProtos to each bound concrete's actual `Storage::TYPE`. | `BindingSpec` (requires binding context — cannot live inside `run_pipeline`) |
| `validate_bootstrap_composition` | Walks the `<root>__bootstrap` call graph: every CALL into a `module_phase = "bootstrap"` child resolves to a `FunctionProto` in the model, and the bootstrap composition tree is a DAG. Surfaces `CompileError::BootstrapCompositionGap` / `BootstrapCompositionCycle`. |  full top-level `ModelProto` (needs every bootstrap FunctionProto in `functions[]` — runs before partitioning strips them per-target). |

`refine_polymorphic_value_info` runs in `Compiler::compile()` BEFORE
`run_pipeline` (and therefore before `type_solver` at pass 6) so the
solver walks the narrowed denotations, not the placeholder ones. It
needs access to `BindingSpec`, which `run_pipeline` does not receive.
Source: `bb-compiler/src/refine_polymorphic_value_info.rs`.

`validate_bootstrap_composition` runs at the top of
`run_pipeline_with_options` — after `inline_for_partition` +
`derive_wire_deadlines` settle the function table but before any
per-target processing — so the bootstrap call graph is checked while
every Module's `"<name>__bootstrap"` FunctionProto still sits in
`temp.functions`. Body-phase passes (partitioning, slot resolution,
gate insertion) never see the bootstrap functions; they remain in
`shared_functions` and ride attached to each emitted installable so
the engine seeds the bootstrap call at install time. Source:
`bb-compiler/src/validate_bootstrap_composition.rs`.

### Touch-set computation

The engine's per-component body-op gate (see
[ENGINE.md §6.8.4](ENGINE.md#684-per-component-gate--is_op_locked))
needs the closure of every `ComponentRef` each Module
bootstrap's body reaches. This is the bootstrap's **touch set**.
Computation lives on the engine — not in the compiler — because
slot-id → `ComponentRef` binding only resolves at install time
when concretes instantiate.

`Engine::compute_touch_set(function_key)`
(`bb-runtime/src/engine/core.rs:1145-1196`) walks the bootstrap
function body once after install populates
`self.functions` + `self.slot_id_to_cref`. For each NodeProto in
the body:

1. **Direct touch.** Read the NodeProto's
   `metadata_props["ai.bytesandbrains.slot_id"]` stamp. If
   present, look up `slot_id → ComponentRef` via
   `Engine::slot_id_to_cref` and add the cref to the touch set.
2. **Transitive touch via FunctionCall.** Build the callee
   `FunctionKey` from `(domain, op_type, overload)`. When the
   callee resolves a sibling FunctionProto in `self.functions`,
   recurse on the callee body. A `visited_keys: HashSet<FunctionKey>`
   defends against cycles (Module A bootstrap → Module B body →
   Module A body via FunctionCalls).

The result stamps onto `BootstrapState::module_bootstraps[name].touch_set`
(`bb-runtime/src/engine/core.rs:1126-1131`) before any bootstrap
fires. At gate time `is_op_locked` reads this pre-stamped set
in O(1) — no per-call body walks.

Install defers the stamp to **after** the install loop completes
(`bb-runtime/src/engine/core.rs:1121-1131`) so forward-referenced
FunctionCalls (callees declared later in `functions`) resolve
against the fully populated registry. The touch-set computation
itself uses only data the engine already owns; the compiler's
contract is to leave every `slot_id` stamp + every FunctionCall
domain/op_type intact through `validate_bootstrap_composition`
and `resolve_slots`.

**Canonical pipeline passes — run by `run_pipeline`:**

| # | Pass | Purpose |
|---|---|---|
| 1 | `inline_for_partition` | Surface wire ops from sub-Modules to the root function so partitioning sees them. |
| 2 | `derive_wire_deadlines` | Stamp each wire.Send's static `deadline_ns = chain_depth × per_hop_budget_ns`. Missing `chain_depth` defaults to 1. |
| 3 | `validate` | Structural sanity check on the recorded program (typed inputs, well-formed graph). Reject bad input before any mutation. |
| 4 | `expand_ops` | Materialize op-variant defaults (attribute fill-in). |
| 5 | `type_solver` | Resolve TypeNode constraints. Strict by default — every unresolved value surfaces `BuildError::UnresolvedType`. |
| 6 | `infer_peer_classes` | Stamp `peer_class` metadata on every value based on the placeholder it flows from. |
| 7 | `synthesize_wire_recvs` | For every user-authored `wire.Send`, insert a synthesized `wire.Recv` on each consumer's scope chain. |
| 8 | `partition_by_wire_ops` | Slice the recorded function at wire ops via reachability. Each partition becomes one installable target. |
| 9 | `resolve_slots` | Match each `(required_trait, slot_id)` placeholder to a binding from `Compiler::bind_<role>::<T>(slot)`. |
| 10 | `analyze_wire_edges` | Per-partition: classify cross-partition edges (Data vs TriggerOnly); group by `(producer, destination)` for batching. |
| 11 | `insert_dedup_gate_rx` | Insert per-peer dedup gates upstream of every wire.Recv. |
| 12 | `insert_peer_health_gate_rx` | Insert peer-health gates upstream of every wire.Recv. |
| 13 | `insert_backoff_gate_rx` | Insert backoff gates upstream of every wire.Recv. |
| 14 | `insert_peer_health_gate_tx` | Insert peer-health gates upstream of every wire.Send. |
| 15 | `insert_backoff_gate_tx` | Insert backoff gates upstream of every wire.Send. |
| 16 | `insert_async_deadlines` | Stamp `deadline_ns` on async-suspending atomic op carriers. |
| 17 | `validate_runtime_complete` | Pre-flight check: every slot bound, every opset covered, every wire pair matched. Failure → `BuildError`. |

`Compiler::compile()` returns ONE `ModelProto` whose `functions[]`
carries every partition produced by `partition_by_wire_ops`. The
compilation passport (`bb.compiled = "v1"`) + per-target binding
metadata (`bb.binding.<target>.<slot> = "<role>|<TYPE_NAME>|<slot_id>"`)
are stamped onto `model.metadata_props` by
`stamp_compilation_metadata` (which runs after the canonical
pipeline; not a member of `CANONICAL_PASS_NAMES` because it's
unconditional).

### 3.1 Why this order

- **`refine_polymorphic_value_info` before `run_pipeline`** so the
  type solver sees narrowed, concrete `Storage::TYPE` denotations on
  every Contract-method port instead of the polymorphic `TYPE_TENSOR`
  placeholder the DSL recorder stamps. The pass needs `BindingSpec`
  access (unavailable inside `run_pipeline`), so it runs as a
  pre-pipeline step in `Compiler::compile()`.
- **Inline-for-partition first** so wire ops authored inside nested
  sub-Module bodies surface to the root function before downstream
  passes look at them. Partitioning, deadlining, and type solving
  all assume a flat top-level view of the wire surface.
- **Derive wire deadlines next** so every `wire.Send` carries a
  static `deadline_ns` budget before validation walks the graph;
  later gates and async-deadline stamping reference it.
- **Validate before any mutation** so structural errors surface
  against the user's recorded IR, not against partially mutated
  output.
- **Expand op variants before edge / type / peer-class passes** so
  attribute defaults are present when those passes read NodeProto
  metadata.
- **Type-solve before peer-class inference** so every value has a
  resolved `TypeNode` when peer-class flow analysis walks it.
- **Infer peer classes before synthesizing wire receivers** so each
  synthesized `wire.Recv` lands on a scope chain whose values carry
  consistent peer-class metadata.
- **Synthesize wire receivers before partitioning** so partition
  reachability sees both halves of every wire pair.
- **Partition before slot resolution** — different BB Nodes may
  bind different concrete impls for the same role; resolving slots
  globally would mis-bind partitions. The partition pass produces
  per-target FunctionProtos that `resolve_slots` walks
  independently.
- **Wire-edge analysis after partition + slot resolution** because
  classification (Data vs TriggerOnly) and batching grouping
  require per-partition consumer information.
- **Gate insertion (Rx then Tx) after wire-edge analysis** so each
  gate is inserted against final wire-transport classifications.
  Dedup-Rx runs first because peer-health and backoff gates
  consume the deduped event stream.
- **Async-deadline stamping after every structural mutation** so
  deadlines are computed against the final NodeProto layout.
- **Runtime-complete validation last** because the check requires
  every prior invariant to hold.

### 3.2 Pass purity

Every pass is a pure function: input is a graph, output is a
modified graph + diagnostics. No global state, no side effects, no
hidden context. This lets the orchestrator compose them
deterministically, lets tests run any pass in isolation, and lets
the snapshot/restore machinery skip the compiler entirely on
restore (snapshots store the post-analysis form).

---

## Part 4 — Pass 3: validate

Reject malformed input before any mutation. Validation errors are
the user's bugs; reporting them at this gate gives the cleanest
error surface.

```rust
pub fn validate(graph: &GraphProto) -> Result<(), ValidationError>;
```

### 4.1 Rules

1. **Op type known.** Every `NodeProto.op_type + domain` is either
   in the framework's reserved opsets (`ai.bytesandbrains.*`,
   `ai.onnx`) OR contributed by a Component via the inventory
   registry (every `#[derive(bb::<Role>)]` / `bb::register_op!`
   declaration extends the opset set).
2. **Inputs reachable.** Every `NodeProto.input` value name is
   produced by some upstream op OR appears in `GraphProto.input`.
   No dangling consumers.
3. **Outputs unique.** Every value name written by some op's
   output list is unique within the graph. No two ops write the
   same value.
4. **Type declarations present.** Every `GraphProto.input` has a
   matching `ValueInfoProto.type`. No implicit-type inputs.
5. **Slot metadata well-formed.** Role-domain NodeProtos carry
   either `(concrete_type, instance)` or
   `(required_trait, slot_id)` metadata; the pass rejects malformed
   role bindings with `MalformedSlotMetadata`.
6. **No cycles.** The graph is a DAG. Cycle detection runs via
   topological sort; cycles → `CyclicGraph`.
7. **Opset versions imported.** Every `(domain, op_type)` used in
   the graph corresponds to an `OperatorSetIdProto` in
   `ModelProto.opset_import`. Missing → `OpsetNotImported`.

### 4.2 Failure modes

```rust
pub enum ValidationError {
    UnknownOp { node_name: String, op_type: String, domain: String },
    DanglingInput { node_name: String, input_name: String },
    DuplicateOutput { value_name: String, node_a: String, node_b: String },
    MissingTypeInfo { input_name: String },
    MalformedSlotMetadata { node_name: String, detail: String },
    CyclicGraph { involves: Vec<String> },
    OpsetNotImported { domain: String, version_used: i64 },
}
```

Wire pairing is enforced by `synthesize_wire_recvs` (Pass 7) and
`validate_runtime_complete` (Pass 17); `validate` itself only
checks structural well-formedness against the recorded function.

### 4.3 What validation does NOT check

- Slot resolution — that's `resolve_slots` (Pass 9), which runs
  once binding metadata is bound to each partition.
- Whether the bound backend supports every op — that's
  `validate_runtime_complete` (Pass 17).
- Semantic correctness of role-method bodies — the framework
  cannot enforce equivalence between a Component's atomic op
  signatures and the runtime behavior the impl exhibits at
  `Backend::execute` / contract dispatch.

---

## Part 5 — Pass 4: expand_ops

Materialize op-variant choices. The DSL records the user's intent;
some Ops have variants the framework chooses based on context.
Examples:

- A `Wire::send_req_batched` whose `peers` input is provably
  size-1 expands to a `WireSendOneShot` variant (single-peer, no
  cohort tracking).
- A `Constant` Op declared in a sub-module gets its value attribute
  materialized from the FunctionProto's `attribute_proto` default.
- A `RecvRespBatched`'s cohort_n is inferred from the connected
  `SendReqBatched`'s peer count (if statically known).

### 5.1 Signature

```rust
pub fn expand_ops(graph: &mut GraphProto) -> Result<(), CompileError>;
```

### 5.2 The expansion table

A static `lookup_expansion(domain, op_type) → Option<ExpandFn>`
match dispatches each NodeProto to its registered expander. Each
`ExpandFn` reads the NodeProto and rewrites attribute defaults in
place.

```rust
pub type ExpandFn = fn(&mut NodeProto) -> Result<(), CompileError>;
```

The pass walks every NodeProto in `graph.node`, looks up the
expander for its `(domain, op_type)` pair, and invokes it on the
mutable NodeProto. Unhandled `(domain, op_type)` pairs are no-ops.

### 5.3 Idempotence

Every `ExpandFn` is idempotent: running expand_ops twice produces
the same output as running it once. Lets the pipeline re-run if
needed (e.g. after a mutation pass introduces fresh ops).

---

## Part 6 — Sub-Module composition happens at authoring time

There is no compiler-side sub-Module inlining pass. When a parent
Module's `op()` body calls `child.op(g, inputs)`, the child runs
inline against the same `Graph` immediately — its NodeProtos land
directly in the parent's recording. By the time the compiler sees
the recorded function, the hierarchy is already flat.

The composition hierarchy is preserved in
`ai.bytesandbrains.module_instance` metadata: each
`Graph::with_function(name, bindings, body)` scope pushes its
`name` onto Graph's internal `module_scope` stack and `push_node`
stamps every NodeProto with the joined chain (e.g.
`"parent_child_grandchild"`). The chain is informational —
partitioning groups by `infer_peer_classes`' `HOME_CLASS_KEY`
stamp (§8.3), not by `module_instance`.

Sub-Modules without wire ops collapse into the same partition as
their parent. Sub-Modules whose body contains a wire op surface to
the root function via `inline_for_partition` (Pass 1) so the
partition pass can slice across them.

---

## Part 8 — Pass 8: partition_by_wire_ops (DISSECTION)

**The user-requested dissection pass.** Slices the recorded function
into per-BB-Node sub-graphs by walking dataflow and stopping at
wire ops. Wire ops are the partition boundary; everything else is
reachability.

### 8.1 Wire ops are the partition boundary

Every NodeProto under domain `ai.bytesandbrains.wire` is a wire op.
The opset has six members per [IR_AND_DSL.md §5d](IR_AND_DSL.md):
`Send`, `SendReqBatched`, `SendResp` (Send-flavored — the data
leaves the partition outbound), and `Recv`, `RecvReq`,
`RecvRespBatched` (Recv-flavored — the data enters the partition
inbound). Users place wire ops explicitly in their Module's `op()`
body; the compiler never synthesizes them.

### 8.2 Reachability rule

Two non-wire NodeProtos A and B belong to the same partition iff
there exists a dataflow path from A to B (or B to A) via value-name
edges that does NOT pass through a wire op. Wire ops break the
dataflow graph into pieces.

Wire ops attach to the partition on their data side:

- A `Send` / `SendReqBatched` / `SendResp` op attaches to the
  partition of its data-input producers (the outbound side).
- A `Recv` / `RecvReq` / `RecvRespBatched` op attaches to the
  partition of its data-output consumers (the inbound side).

### 8.3 Partition naming

`infer_peer_classes` (Pass 6) stamps every NodeProto with a
`HOME_CLASS_KEY` metadata value pointing at the BB-Node class the
op runs on. `partition_by_wire_ops` does a direct group-by on
that stamp — the dataflow shape already defines class membership,
so neither union-find nor `module_instance` LCP naming is needed.
Nodes lacking a home stamp (hand-built fixtures, legacy
single-Node Modules) fall through to the canonical
`SELF_CLASS` ("self").

Examples:

| Composition | Partition names |
|---|---|
| Single-Node `DistributedRetrieval` (no wire ops) | `["self"]` |
| Federated `FederatedRetrieval { client, server }` with one wire op pair | `["client", "server"]` |
| Federated training with explicit roles via placeholder type denotations | one partition per inferred peer class |

### 8.4 Pass signature

```rust
pub fn partition_by_wire_ops(
    graph: &GraphProto,
) -> Result<NetworkAnalysis, CompileError>;

pub struct NetworkAnalysis {
    /// One entry per BB-Node partition keyed by peer-class name.
    pub per_role: BTreeMap<String, GraphProto>,
    /// One entry per Send/Recv pair, populated by `discover_wire_edges`.
    pub wire_edges: Vec<WireEdge>,
}

pub struct WireEdge {
    pub producer_role: String,
    pub consumer_role: String,
    pub value_name: String,
    pub send_node: NodeProto,
    pub recv_node: NodeProto,
}
```

The `wire_edges` field carries one entry per matched Send/Recv pair
(matched via the `__send_sentinel_<idx>` output rename
`synthesize_wire_recvs` stamps on every Send plus the
`SYNTHESIZED_FROM_KEY = <idx>` back-reference on every synthesized
Recv). Sends without a paired Recv (fire-and-forget) are skipped.
`analyze_wire_edges` (Pass 10) reads this list to classify each
edge's transport.

### 8.5 Algorithm

1. Walk every NodeProto. Read its `HOME_CLASS_KEY` metadata stamp
   (or fall back to `SELF_CLASS` for unstamped nodes).
2. Append the node to `per_role[home_class]`.
3. For each per-role sub-graph, scan the contained NodeProto
   `input`/`output` value names and copy across the matching
   `GraphProto.input` / `GraphProto.value_info` entries plus the
   intersection of `GraphProto.output` and locally-produced values.
4. Walk every Send NodeProto's `__send_sentinel_<idx>` output and
   match it against the synthesized Recv carrying
   `SYNTHESIZED_FROM_KEY = <idx>`; build a `WireEdge` for each
   matched pair.

### 8.6 The single-Node case

When `infer_peer_classes` stamps every NodeProto with the same
`HOME_CLASS_KEY`, partitioning produces a single entry under that
class name. A Module with no peer-class metadata at all collapses
into `per_role[SELF_CLASS]`.

### 8.7 Why this happens before slot resolution

Two reasons:

1. **Role bindings are per-target.** Different BB Nodes may bind
   different concrete impls for the same role (one target binds
   `BurnBackend`, another binds `CandleBackend`). Resolving slots
   globally before partitioning would mis-bind partitions.
2. **Cross-Node edges traverse wire ops.** Partitioning first
   ensures each per-Node sub-graph contains only the role ops that
   actually run on that Node. `resolve_slots` (Pass 9) then walks
   each partition with that partition's binding metadata.

---

## Part 9 — Pass 10: analyze_wire_edges

For each per-Node sub-graph, classify every wire edge as Data or
TriggerOnly + group by destination for batching.

### 9.1 Per-edge classification

```rust
pub fn analyze_wire_edges(
    sub_graph: &mut GraphProto,
    wire_edges: &[WireEdge],
) -> Result<(), CompileError>;
```

For each wire edge, walk the consumer's downstream:

- If every downstream consumer of the edge's value declares its
  input port type as `bb.trigger` → mark the edge `trigger_only`.
- Otherwise → mark `data`.

The classification is stored on the Wire send + recv NodeProto via
`metadata_props["ai.bytesandbrains.wire_transport"] = "data" |
"trigger_only"`.

At runtime, trigger-only sends ship zero-byte payloads (per
WIRE.md §8); the receiver synthesizes Trigger. Data sends ship
full encoded bytes.

### 9.2 Per-cycle batching grouping

For each `(producer_role, destination_peer)`, collect all wire
sends whose source ops are in the same cycle scope (i.e. firing
together as part of the same DAG burst). Tag them with a shared
`batch_group_id` so the runtime's outbound batcher knows to pack
them into one envelope.

### 9.3 Destination-address stamping (per ADDRESSING.md)

For every cross-Node edge, the pass also stamps the consumer-side
recv output value name onto the producer's Send NodeProto:

```
metadata_props["ai.bytesandbrains.dest_site_name.<input>"] = recv_value_name
```

The name is symbolic at analysis time — `NodeSiteId`s aren't
allocated until install. Node's install path resolves each
`dest_site_name.<input>` entry against the global `site_names` map
(spanning every installed ModelProto's graph) and rewrites it as a
canonical `ai.bytesandbrains.dest_suffix.<input>` AttributeProto
carrying `Address::empty().site(NodeSiteId).to_bytes()`. The wire
syscall's `collect_fills` reads each fill's suffix from this
attribute at dispatch time (see [WIRE.md](WIRE.md) Part 5).

The helper API ships alongside the pass:
- `dest_suffix_attr(node, input_name) → Option<&[u8]>` — read the
  resolved Address bytes off a Send NodeProto.
- `dest_suffix_attribute(input_name, bytes) → AttributeProto` —
  build the attribute from canonical Address bytes (used by
  Node's resolver).

### 9.4 Output

The sub-graph now has fully-annotated wire boundary nodes. The
runtime knows for each send: is it data or trigger? which batch
does it belong to? where does the payload land
(`dest_site_name.<input>`, resolved at install to a multiaddr
suffix)? All static info.

---

## Part 10 — Passes 1, 2, 6, 7, 8: setup + flow analysis

Each pass has a corresponding source file under `bb-compiler/src/`
plus a sibling `*_tests.rs`. The summaries here cover what each
pass mutates and what invariant it establishes; the source file is
the authoritative description.

### 10.1 Pass 1: `inline_for_partition`

Source: `bb-compiler/src/inline_for_partition.rs`.

Walks the root function and inlines every nested sub-Module body
whose recorded NodeProtos contain a wire op. Pure sub-Modules
(no wire ops in any descendant) stay as call NodeProtos and ride
the partition main intact. The pass leaves a flat top-level view
of the wire surface for `partition_by_wire_ops` to slice on.

### 10.2 Pass 2: `derive_wire_deadlines`

Source: `bb-compiler/src/derive_wire_deadlines.rs`.

For every `wire.Send` NodeProto, stamps
`metadata_props["ai.bytesandbrains.deadline_ns"] = chain_depth ×
per_hop_budget_ns`. `chain_depth` is read from the NodeProto's
existing metadata; missing values default to 1. `per_hop_budget_ns`
is supplied via `Compiler::with_per_hop_budget_ns(ns)`. The stamp
is consumed by `insert_async_deadlines` (Pass 16) and by runtime
wire scheduling.

### 10.3 Pass 5: `type_solver`

Source: `bb-compiler/src/type_solver.rs`.

Bipartite worklist that walks `ValueInfoProto`s and NodeProto
input/output type slots, resolving every value to a concrete
`TypeNode`. Strict by default — every unresolved value surfaces
`CompileError::UnresolvedType { value }`. Constraint conflicts
surface `CompileError::TypeConstraintFailed`. Relaxed mode is
opt-in via `Compiler::with_strict_types(false)` and skips the
final unresolved-value check.

The pass uses `bb-ir`'s `TypeNode` lattice (Tensor element types,
PeerClass denotations, Trigger, tuple shapes). Downstream passes
(`infer_peer_classes`, `analyze_wire_edges`) assume every
`ValueInfoProto.type` carries a resolved TypeNode.

### 10.4 Pass 6: `infer_peer_classes`

Source: `bb-compiler/src/infer_peer_classes.rs`.

Walks dataflow from declared peer-class roots (`PeerClass`-typed
placeholders, peer-sampling Op outputs) and stamps
`metadata_props["ai.bytesandbrains.peer_class"]` on every value
that carries a `PeerClass` denotation. Lets the wire-edge compiler
classify cross-Node edges by peer scope, and lets the runtime
build per-class delivery tables at install.

### 10.5 Pass 7: `synthesize_wire_recvs`

Source: `bb-compiler/src/synthesize_wire_recvs.rs`.

For every user-authored `wire.Send` NodeProto, walks the consumer
scope chain (the `module_instance` lineage of every value-name
consumer downstream of the send's payload) and inserts a
synthesized `wire.Recv` NodeProto at each consumer's entry point.
The synthesized Recv carries the same wire type, peer class, and
deadline stamping as the source Send. Partitioning sees both
halves of every wire pair after this pass.

### 10.6 `stamp_compilation_metadata`: slot-binding stamping

Source: `bb-compiler/src/stamp_compilation_metadata.rs`.

The final pass that turns each per-partition `ModelProto` into a
complete install artifact. Two surfaces:

- **Compilation passport** — `bb.compiled = "v1"`
  (`COMPILED_KEY`) on the model + one
  `bb.binding.<target>.<slot> = "<role>|<TYPE_NAME>|<slot_id>"`
  entry per `BindingSlot`. Install reads these to construct each
  bound component's `inventory` entry and bind it to the slot.
- **`recv_site_to_slot_id` metadata** — for every `wire.Recv`
  NodeProto whose payload output flows into a role NodeProto's
  input, the pass stamps `RECV_SLOT_ID_KEY` on the Recv node's
  `metadata_props` carrying the consumer role's `slot_id`
  (`bb-compiler/src/stamp_compilation_metadata.rs:82-130`).
  Install (`src/install.rs`) reads this to populate
  `GraphSlot::recv_site_to_slot_id` so `decode_typed_fill` can
  cross from data-plane identity (`NodeSiteId`) to binding
  identity (`slot_id`) and route backend-bound tensor fills
  through `Backend::materialize_from_wire`. Recv nodes whose
  payload does not flow into a role NodeProto are left
  unstamped (framework-carrier path).

The pass is partition-local and idempotent. The stamp key
`ai.bytesandbrains.recv_site.<node_site_id>` is orthogonal to
the existing `ai.bytesandbrains.binding.<target>.<slot>`
namespace.

---

## Part 11 — Pass 9: resolve_slots

Source: `bb-compiler/src/resolve_slots.rs`.

Walks every NodeProto under a role domain
(`ai.bytesandbrains.role.<role>`) and collects, per role, the
distinct `concrete_type` providers (NodeProtos stamped with
`ai.bytesandbrains.concrete_type`) and the distinct
`(required_trait, slot_id)` generic providers. If both kinds
appear under the same role, the pass rejects the recording with
`CompileError::AmbiguousRole { role, concrete_type, generic_slot_id }`
— a single role binding must be either concrete or generic, not
both.

```rust
pub fn resolve_slots(function: &FunctionProto) -> Result<(), CompileError>;
```

Slot-existence and decoder-availability checks live in
`validate_runtime_complete` (Pass 17), where the full binding
spec is in scope.

---

## Part 12 — Passes 12-16: gate insertion

Source: five sibling files under `bb-compiler/src/`:
`insert_dedup_gate_rx.rs`, `insert_peer_health_gate_rx.rs`,
`insert_backoff_gate_rx.rs`, `insert_peer_health_gate_tx.rs`,
`insert_backoff_gate_tx.rs`. Each pass walks the partition main +
inserts a single class of framework-mediated gate NodeProtos
adjacent to wire ops; together they form the policy enforcement
layer between the transport and the user-authored DAG.

| Pass | Position | Purpose |
|---|---|---|
| 12 `insert_dedup_gate_rx` | upstream of every `wire.Recv` | drops duplicate per-peer deliveries (replay protection); keyed by `(producer_peer, wire_id)`. |
| 13 `insert_peer_health_gate_rx` | upstream of every `wire.Recv` | suppresses delivery when the source peer's health snapshot is below threshold. |
| 14 `insert_backoff_gate_rx` | upstream of every `wire.Recv` | applies exponential backoff against repeatedly-misbehaving peers on the receive side. |
| 15 `insert_peer_health_gate_tx` | upstream of every `wire.Send` | suppresses outbound sends to peers below the health threshold. |
| 16 `insert_backoff_gate_tx` | upstream of every `wire.Send` | applies exponential backoff against repeatedly-failing peers on the send side. |

Gates are NodeProtos under `ai.bytesandbrains.gate`. Each gate
consults a runtime table (`PeerHealth`, `BackoffState`,
`DedupCache`) provided by the engine; reconfiguring a gate's
threshold or window happens at runtime via the gate Component's
contract methods. Rx gates run before Tx gates so the inbound
event stream is deduped + health-filtered before any outbound
reaction fires.

`bb-compiler/src/rx_chain.rs` (used by the three Rx passes)
encapsulates the shared "walk every Recv, insert N carriers in
order" splice; `bb-compiler/src/gate_contract.rs` holds the gate
Component contract the engine binds at install.

---

## Part 13 — Passes 17, 18: deadline stamping + pre-flight

### 13.1 Pass 16: `insert_async_deadlines`

Source: `bb-compiler/src/insert_async_deadlines.rs`.

Walks every NodeProto that suspends asynchronously (wire-batch
carriers, async syscall carriers, dispatch carriers for Components
that return `CommandId` from their atomic ops) and stamps
`metadata_props["ai.bytesandbrains.deadline_ns"]` from the
upstream `derive_wire_deadlines` chain plus a per-Op extra budget
declared on the Op's metadata. The engine reads the stamp when it
enrolls the operation in the deadline wheel.

### 13.2 Pass 17: `validate_runtime_complete`

Source: `bb-compiler/src/validate_runtime_complete.rs`.

Final structural completeness check before compilation completes.
Walks each per-partition sub-graph and verifies the compiler-
required ops are present alongside the ops they serve:

- A peer-routed `wire.Send` (Send carrying a `peer` attribute) is
  accompanied by `PeerHealthGateTx` + `BackoffGateTx` NodeProtos.
- A peer-routed `wire.Recv` is accompanied by `DedupGateRx` +
  `PeerHealthGateRx` + `BackoffGateRx`.
- Any NodeProto stamped with `deadline_ns` has a `DeadlineCheck`
  upstream in the partition.
- Every gate contract registered in `bb-compiler/src/gate_contract.rs`'s
  inventory passes its `assert_inserted(sub_graph)` check.

Sends and Recvs without an explicit `peer` attribute (those routed
via the address book using `dest_target` metadata) bypass the
per-peer gate requirement.

Failures surface as `CompileError::Internal { detail }` (legacy
hand-written checks) or `CompileError::RuntimeIncomplete { missing }`
(contract-driven checks). `verify_no_dangling_calls`
(`bb-compiler/src/verify_no_dangling_calls.rs`) is invoked between
the mutation passes via the runner's seam checks — every `Call*`
NodeProto must point at a FunctionProto present in
`model.functions[]`.

---

## Part 14 — Validation gates between passes

Each pass writes invariants the next pass assumes. The runner
invokes a handful of cheap structural checks (the `verify` family in
`bb-ir`) between pipeline seams so a malformed `ModelProto` from
one stage fails LOUDLY at the seam instead of producing a confusing
error several passes downstream.

| After pass | Invariant |
|---|---|
| 1 `inline_for_partition` | Wire ops from sub-Modules surface to the root function |
| 2 `derive_wire_deadlines` | Every `wire.Send` carries a non-zero `deadline_ns` |
| 3 `validate` | Graph is well-formed (op types known, inputs reachable, outputs unique, wire pairs match, slot metadata well-formed, DAG, opsets imported) |
| 4 `expand_ops` | Every Op variant is materialized with full attributes |
| 5 `type_solver` | Every `ValueInfoProto.type` carries a resolved `TypeNode` |
| 6 `infer_peer_classes` | Every value with a `PeerClass` denotation carries `metadata_props["ai.bytesandbrains.peer_class"]` |
| 7 `synthesize_wire_recvs` | Every `wire.Send` has a paired `wire.Recv` on every consumer scope chain |
| 8 `partition_by_wire_ops` | One sub-graph per wire-op-bounded partition, named by longest common `module_instance` prefix |
| 9 `resolve_slots` | Every slot referenced by a NodeProto is bound (concrete or generic) in the partition's binding spec |
| 10 `analyze_wire_edges` | Every wire send + recv has `wire_transport ∈ {data, trigger_only}` and a `batch_group_id` |
| 11-15 gate insertion | Every wire op has its full gate chain (dedup-rx, peer-health-rx/tx, backoff-rx/tx) attached |
| 16 `insert_async_deadlines` | Every async-suspending NodeProto carries a `deadline_ns` |
| 17 `validate_runtime_complete` | Every slot bound; every opset covered; every wire pair paired; every deadline stamp parses |

`bb-ir::verify::{types, wire_pairs, function_calls}` are invoked
between mutation passes (see `bb-compiler/src/runner.rs`). The
`verify_no_dangling_calls` pass (a sibling of the canonical list)
runs after each splice that introduces `Call*` NodeProtos.

---

## Part 15 — Pipeline output: `ModelProto`

`Compiler::compile()` returns a single `ModelProto` whose
`functions[]` carries every partition. The bare `ModelProto` IS the
compiled artifact — there is no wrapper struct.

The proto definition is the upstream ONNX schema at
[`proto/onnx-ml.proto`](../proto/onnx-ml.proto); `prost-build`
generates the Rust types in `bb-ir::proto::onnx`. The compiled form
carries:

- `functions[0..n]` — one `FunctionProto` per partition main,
  plus any sub-Module bodies recorded via `Graph::with_function`
  and the synthesized helpers (gate carriers, lifecycle
  containers) the compiler spliced in. Each partition main's
  `name` matches a string the host passes in the `targets` slice
  to `bb::install`. **Any number of partition mains can serve as
  entry-point Modules**: the compiler emits all of them as sibling
  `FunctionProto`s and the install path picks which ones to expose
  on each Node via the `targets` slice. A federated module that
  partitions into `Client` + `Server` ships both as
  `model.functions` entries; the same proto installs as Client-only
  on one peer, Server-only on another, or both on a peer that
  hosts the whole round in-process.
- `opset_import[]` — every `(domain, version)` referenced by any
  NodeProto in any function.
- `metadata_props[]` — the compilation passport
  (`bb.compiled = "v1"`), per-target binding triples
  (`bb.binding.<target>.<slot> = "<role>|<TYPE_NAME>|<slot_id>"`)
  emitted **once per partition** so the install path can read each
  target's bindings independently, and any compiler telemetry
  attached by `stamp_compilation_metadata`.
- `graph` (the top-level `GraphProto`) — left empty; the partition
  bodies live entirely in `functions[]`.

The post-analysis `ModelProto` round-trips through `prost`
serialization with no extra wrapper layer, so it composes naturally
into outer Modules (a built `ModelProto` is itself a `Module` whose
`op()` replays the stored function into a parent graph).

`bb::install(peer_id, addresses, compiled, targets: &[&str],
Config::new())` walks `compiled.functions[]` once per entry in
`targets`, reads each target's binding metadata from
`metadata_props`, deduplicates slot bindings across targets (a slot
named by multiple targets resolves to one `ComponentRef` shared by
every target's dispatch path), and constructs the
per-deployment component instances using their inventory-registered
`construct_fn`s. Different BB Nodes pick different `targets` slices
from the same compiled `ModelProto`. The compiler itself is
unchanged — partitioning, binding-table stamping, gate insertion,
and runtime-complete validation all run per partition exactly as
before; multi-target install is a property of `install.rs`, not the
compiler.

---

## Part 16 — Telemetry + observability

Each pass opens a `tracing` span with timing + statistics:

```
compiler.inline_for_partition { graph: "ServerModule", surfaced: 4 }
  ↳ took 0.2ms
compiler.validate { graph: "ServerModule", nodes: 47, errors: 0 }
  ↳ took 0.3ms
compiler.expand_ops { graph: "ServerModule", expanded: 12 }
  ↳ took 0.5ms
compiler.type_solver { resolved: 32, unresolved: 0 }
  ↳ took 0.4ms
compiler.partition_by_wire_ops { partitions: ["federated_retrieval_client",
                                              "federated_retrieval_server"],
                                wire_ops: 4 }
  ↳ took 1.1ms
compiler.analyze_wire_edges { data: 3, trigger_only: 1 }
  ↳ took 0.7ms
compiler.validate_runtime_complete { unbound: 0, covered: true }
  ↳ took 0.4ms
```

Counters keyed by pass name (each pass emits at most a handful):

- `compiler.surfaced_wire_ops` (Pass 1)
- `compiler.deadlines_stamped` (Pass 2)
- `compiler.validation_errors` (Pass 3)
- `compiler.expansion_count` (Pass 4)
- `compiler.types_resolved` / `compiler.types_unresolved` (Pass 5)
- `compiler.peer_classes_stamped` (Pass 6)
- `compiler.wire_recvs_synthesized` (Pass 7)
- `compiler.partition_count` (Pass 8)
- `compiler.slots_resolved` / `compiler.slots_unbound` (Pass 9)
- `compiler.wire_edges_data` / `compiler.wire_edges_trigger_only` (Pass 10)
- `compiler.gates_inserted` (Passes 11-15)
- `compiler.async_deadlines_stamped` (Pass 16)
- `compiler.preflight_errors` (Pass 17)

Build-time tracking lets ops teams diagnose which pass is the
latency bottleneck. The compiler typically runs in <10 ms for
production-sized graphs (≤ 500 ops); larger graphs scale roughly
linearly per pass.

---

## Part 17 — Worked example: SplitLearning through the pipeline

A canonical end-to-end run showing what each canonical pass
produces. The user code reads:

```rust
let compiled: ModelProto = Compiler::new()
    .bind_backend::<BurnBackend>("compute")
    .bind_codec::<ProductQuantizer>("codec")
    .bind_aggregator::<WeightAggregator>("agg")
    .compile(SplitLearning::new(/* config */))?;

let node = bb::install(
    peer_id,
    vec![Address::empty()],
    compiled,
    &["SplitLearning"],
    Config::new().with("compute", burn_config),
)?;
```

### Input

The canonical SplitLearning Module from
[IR_AND_DSL.md §10](IR_AND_DSL.md). The Module's
`op(&self, g: &mut Graph, inputs: &[Output]) -> Vec<Output>` records
11 NodeProtos covering recv → decode → forward → encode →
sample peers → send_req → recv_responses → decode → aggregate
→ apply_delta → backward.

Single top-level Module; no sub-Modules in this example. (Sub-Module
composition calls `child.op(g, inputs)` from inside the parent's
`op()`; the pipeline below handles either shape.) The example is
single-Node; federated variants surface multiple partitions at
Pass 8.

### Pass 1 — inline_for_partition

No sub-Module bodies in the top-level SplitLearning function.
Surfaced wire ops: 0.

### Pass 2 — derive_wire_deadlines

Two wire ops carry chain depth metadata. With
`per_hop_budget_ns = 50_000_000`, each `wire.Send` is stamped
`deadline_ns = 50_000_000`.

### Pass 3 — validate

All 11 NodeProtos have known `(op_type, domain)` pairs. All inputs
reachable. Wire pairing satisfied (`SendReqBatched` + `SendResp`).
All types declared. No cycles. Errors: 0.

### Pass 4 — expand_ops

Three NodeProtos materialize default attributes (e.g. `Threshold.n`
from the FunctionProto's `attribute_proto[NUM_CONCURRENT]`).

### Pass 5 — type_solver

32 value sites resolved to concrete `TypeNode`s (Tensor element
types, PeerClass denotations). Unresolved: 0.

### Pass 5 — infer_peer_classes

Three `wire.Send` payload values carry `PeerClass::SampledGossip`.
Peer-class stamps: 3.

### Pass 6 — synthesize_wire_recvs

Two `wire.Recv` NodeProtos synthesized — one each on the consumer
scope chains downstream of the two user-authored `wire.Send`s.

### Pass 7 — partition_by_wire_ops

Reachability collapses everything into one partition. Common
`module_instance` prefix: `SplitLearning`. Partition count: 1.

### Pass 8 — resolve_slots

- 1 generic slot (`compute`) → bound to `BurnBackend`.
- 6 concrete slots → registered components present.

Unbound: 0.

### Pass 9 — analyze_wire_edges

Each wire send/recv pair gets `wire_transport = "data"` (payloads
are `Tensor`, not `Trigger`). No batching opportunity — one wire
send per cycle. `dest_site_name` stamps land on each producer.

### Passes 11-15 — gate insertion

Per `wire.Recv`, the pipeline inserts dedup-rx, peer-health-rx,
and backoff-rx gates (3 gates per Recv × 1 Recv = 3). Per
`wire.Send`, peer-health-tx and backoff-tx gates (2 × 1 = 2).
Total gates inserted: 5.

### Pass 15 — insert_async_deadlines

Async-suspending NodeProtos (the two wire batch carriers plus any
Component atomic ops that return a `CommandId`) carry `deadline_ns`
stamps derived from the Pass 2 budgets plus per-Op extras.

### Pass 16 — validate_runtime_complete

- Peer-routed `wire.Send` accompanied by `PeerHealthGateTx` +
  `BackoffGateTx`.
- Peer-routed `wire.Recv` accompanied by `DedupGateRx` +
  `PeerHealthGateRx` + `BackoffGateRx`.
- Each NodeProto carrying `deadline_ns` has an upstream
  `DeadlineCheck`.
- Every inventory-registered gate contract's `assert_inserted`
  check passes.

Errors: 0.

### Output

`Compiler::compile()` returns one `ModelProto` whose `functions[0]`
is the `SplitLearning` partition main and whose `functions[1..]`
carry sub-Module bodies recorded via `Graph::with_function`. The
compilation passport (`bb.compiled = "v1"`) and binding triples
are stamped onto `metadata_props`.

`bb::install(peer_id, addr, compiled, "SplitLearning", cfg)` walks
`compiled.functions[]`, picks the function named `SplitLearning`,
materializes each declared component via its inventory-registered
`construct_fn`, and brings the Node up. At runtime each role op
dispatches via the bound Component's contract methods, the bound
`BurnBackend` handles every `ai.onnx` op via `Backend::execute`,
and each wire send/recv ships one envelope.

---

## Part 18 — Pass-level testability

Each pass is a pure function on IR. Tests live as
`bb-compiler/src/<pass>_tests.rs`:

- Each test constructs a minimal IR fragment.
- Calls the pass directly.
- Asserts on the output IR shape + metadata + diagnostics.

This makes the compiler the most testable layer in the framework:
no engine state, no runtime threading, no async, just IR in → IR
out + assertions on the difference.

Cross-pass tests live as `bb-compiler/src/driver_tests.rs` and
`bb-compiler/src/runner.rs` integration tests that exercise the
full pipeline composition against canonical examples.

---

That's the pipeline. Seventeen canonical passes producing one
`ModelProto` ready for `bb::install` to bind to a Node. Each pass
pure, well-ordered, explicitly diagnosable. Dissection (Pass 8)
runs before per-target slot resolution (Pass 9) so different BB
Nodes can resolve different bindings against the same compiled
artifact. Every mutation is observable; every error is typed; the
final invariants are checked once before the Engine starts
executing.
