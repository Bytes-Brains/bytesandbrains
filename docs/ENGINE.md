# ENGINE.md — runtime engine specification

The engine is the heart of a BytesAndBrains Node: it drives every
poll cycle, owns every slot value, dispatches every Op, routes every
inbound envelope, schedules every timer, completes every async
CommandId, and emits every EngineStep. This document specifies the
engine's full execution model — concurrency model, state
organization, queue architecture, entry-point taxonomy, multi-
instance graph execution, app-interaction surface, event emission,
tracing, and snapshot semantics. Pairs with IR_AND_DSL.md (the IR)
and API_DESIGN.md (the user-facing API).

## Part 1 — Overview

A Node hosts an Engine. The Engine owns:

- Zero or more **registered graphs**, each carrying a `FunctionProto`
  body + bound runtime components.
- A **dispatch table** mapping each Op type to an invoke fn pointer.
- A **slot table** holding intermediate value state, keyed by
  `(NodeSiteId, ExecId)` so concurrent executions of the same graph
  don't clobber each other.
- A **frontier** — the in-cycle queue of Ops ready to fire.
- An **ingress queue** — the thread-safe inbox where external events
  accumulate between cycles.
- **Pending-async tracking** — `CommandId → suspended-Op` map for
  long-running operations.
- **Per-Node framework components** — scheduler, bus, request
  tracker, outbound queue, etc.
- A **wake-up waker** stash for async `poll(cx)` integration.

A single Node is **single-threaded at the engine layer.** The Engine
holds `&mut self` for an entire poll cycle; Op invocations are
sequential. External threads only interact with the Engine through
the ingress queue, which is the single thread-safe boundary.

The poll cycle:

```
node.poll(cx)
  ├─ drain ingress → seed frontier + write slots + complete commands
  │                   (each envelope's fills materialise into typed
  │                    SlotValues via wire_decoder_registry; per-fill
  │                    failure emits InfraEvent::WireReceiveError +
  │                    EngineStep::WireReceiveFailed and continues
  │                    iterating sibling fills — partial-delivery
  │                    semantics; see WIRE.md §5.4.)
  ├─ walk frontier (DAG cascade)
  ├─ route bus events → seed frontier → walk
  ├─ poll matured timers → fire entries → walk
  ├─ drain pending_completions (from in-cycle hooks) → walk
  ├─ drain outbound_queue → emit SendEnvelope steps
  └─ if nothing to do: stash waker, return Poll::Pending
                  else: return Poll::Ready(steps)
```

Each cycle is atomic with respect to external observation: the host
sees a vector of `EngineStep`s the cycle produced, then has a chance
to inject more work before the next poll.

---

## Part 2 — The two-queue architecture

The engine deliberately separates **in-cycle scheduling** (fast,
single-threaded) from **cross-thread event accumulation** (thread-
safe, slower per-push). Two distinct data structures, two distinct
concurrency models.

### 2.1 The frontier — VecDeque, single-threaded

```rust
pub struct Engine {
    /// In-cycle DAG-walking queue. Holds (OpRef, ExecId) pairs
    /// that are ready to fire. Mutated only by the engine while
    /// it holds `&mut self`; never shared across threads.
    frontier: VecDeque<(OpRef, ExecId)>,
    // ...
}
```

The frontier is the engine's working queue while walking a cascade.
When an Op fires + writes values into output sites, the engine
checks each site's downstream consumers; consumers whose inputs all
became ready get pushed onto the frontier. The engine then pops the
next entry + invokes it. Loop until frontier is empty.

**Choice of `VecDeque`:**
- O(1) push_back + pop_front + len.
- No locking, no synchronization, no atomics.
- No allocation per push after warm-up.
- Pure performance — the frontier is touched millions of times per
  cycle in heavy workloads.

**Why not a heap / priority queue?** Op execution order within a
cycle is FIFO by spec: if Op A is scheduled before Op B and both
become ready in the same cycle, A fires first. Priority-based
scheduling would change observable semantics; we stick to FIFO.

**Why not Rayon / parallel execution?** Op invocations may borrow
mutable references to shared framework components via
`RuntimeResourceRef`. Parallel invocation would require splitting
those borrows, which makes the Op author's job harder for marginal
gain. The atomic unit of parallelism in BB is the Node, not the Op
— launch multiple Nodes if you need parallel work.

### 2.2 The ingress queue — lockless MPMC + AtomicWaker

The ingress is built on two third-party crates, both lock-free, both
runtime-independent (no tokio / smol / async-std dependency), both
under permissive `MIT OR Apache-2.0` licenses:

| Crate | Version | License | Role |
|---|---|---|---|
| [`concurrent-queue`](https://crates.io/crates/concurrent-queue) | `2` | MIT OR Apache-2.0 | The lock-free MPMC queue |
| [`atomic-waker`](https://crates.io/crates/atomic-waker) | `1` | MIT OR Apache-2.0 | The lock-free `Waker` slot |

```rust
use concurrent_queue::{ConcurrentQueue, PushError};
use atomic_waker::AtomicWaker;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::Waker;

pub struct IngressQueue {
    /// Lock-free bounded MPMC queue. Multiple external threads push;
    /// the single engine thread drains.
    queue: ConcurrentQueue<IngressEvent>,
    /// Lock-free waker slot. The most recent register() replaces the
    /// previous; wake() retrieves and fires.
    waker: AtomicWaker,
    /// Cumulative count of events dropped due to capacity overflow.
    dropped_overflow: AtomicU64,
}

pub enum IngressEvent {
    /// An inbound envelope from transport, attributed to a source peer.
    /// `src_observed_address` carries the dialer endpoint the adapter
    /// actually saw (NAT-translated reflexive address) so the Phase 1
    /// AddressBook merge composes claimed + observed. `None` means the
    /// transport could not surface a reflexive address.
    /// (`bb-runtime/src/ingress.rs:47-60`.)
    EnvelopeFrom {
        src_peer: PeerId,
        envelope: WireEnvelope,
        src_observed_address: Option<Address>,
    },
    /// A host-pushed unconnected-input value targeting a registered
    /// Module's named input.
    AppEvent {
        module_name: String,
        input_name: String,
        value_bytes: Vec<u8>,
    },
    /// A timer maturity signal pushed by the host clock task.
    TimerMatured { at_ns: u64 },
    /// An explicit Module invocation triggered by the host.
    Invoke { module_name: String, inputs: Vec<(String, Vec<u8>)> },
    /// A pending-completion that originated off-thread (e.g.
    /// a non-Engine task fulfilling a CommandId).
    Completion { cmd_id: CommandId, results: Vec<Vec<u8>> },
    /// A control signal (clear / shutdown notification / etc.).
    Control(ControlSignal),
}

impl IngressQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: ConcurrentQueue::bounded(capacity),
            waker: AtomicWaker::new(),
            dropped_overflow: AtomicU64::new(0),
        }
    }

    /// Push an event. Returns Err(event) if the queue is at capacity
    /// — the caller decides whether to retry, drop, or escalate.
    pub fn push(&self, event: IngressEvent) -> Result<(), IngressEvent> {
        match self.queue.push(event) {
            Ok(()) => {
                self.waker.wake();
                Ok(())
            }
            Err(PushError::Full(ev)) => {
                self.dropped_overflow.fetch_add(1, Ordering::Relaxed);
                Err(ev)
            }
            Err(PushError::Closed(ev)) => Err(ev),
        }
    }

    /// Drain every queued event into a vector. Called by the engine
    /// at the start of each poll cycle.
    pub fn drain_all(&self) -> Vec<IngressEvent> {
        let mut out = Vec::with_capacity(self.queue.len());
        while let Ok(event) = self.queue.pop() {
            out.push(event);
        }
        out
    }

    /// Register the waker the engine wants fired on the next push.
    /// Called by the engine when poll(cx) returns Pending.
    pub fn register_waker(&self, waker: &Waker) {
        self.waker.register(waker);
    }

    pub fn len(&self) -> usize { self.queue.len() }
    pub fn capacity(&self) -> Option<usize> { self.queue.capacity() }
    pub fn dropped_count(&self) -> u64 {
        self.dropped_overflow.load(Ordering::Relaxed)
    }
}
```

**No locks anywhere.** `ConcurrentQueue` is built on atomic operations
+ hazard pointers; `AtomicWaker` is built on atomic state +
`AcqRel` ordering. Both are extensively tested in production via
their parent ecosystems (`concurrent-queue` is the foundation under
`async-channel`; `atomic-waker` is the foundation under `futures` /
`smol`).

**Runtime independence.** Neither crate pulls in tokio / smol / any
specific async runtime. They work with `std::task::Waker` directly,
which is what async runtimes hand the framework via the `Context<'_>`
passed to `poll(cx)`. The Node remains sans-IO; the host's executor
provides the waker via its own `Context`; everything wires up
through `std::task` primitives.

**Why MPMC even though we have a single consumer.** `ConcurrentQueue`
ships only an MPMC API; single-consumer optimization would require
a separate crate (`crossbeam-queue::SegQueue` is MPMC too; the
SPSC choice is `rtrb` but it's bounded-only with different
semantics). The extra cost is negligible: single-consumer drain
just calls `pop()` in a loop the same way an MPMC consumer would.

**Push cost.** A successful `ConcurrentQueue::push` is one CAS on
the queue's tail index + one atomic store of the slot value + one
`AtomicWaker::wake` (which is one atomic swap). Total per-push:
~30 ns on x86_64, ~50 ns on ARM. No allocation. No memory barrier
beyond the atomic operations.

**Drain cost.** `ConcurrentQueue::pop` is one CAS on the head index
+ one load. Bulk drain of N events: ~30 ns * N. The drained vector
allocation is the dominant cost for large batches; reuse the
allocation across cycles to amortize.

**Capacity + backpressure.** Bounded capacity; defaults to
`bus_capacity * 4`. Overflowing pushes return `Err(event)` so the
producer can decide: retry later, drop with a metric, escalate as
back-pressure. The framework's transport adapters drop + increment
`IncrMetric("ingress.dropped")`. Dropping is preferred to blocking:
a sans-IO framework never blocks the producer thread.

**Alternative crate choice.** If you'd rather use the crossbeam
ecosystem,
[`crossbeam-queue::ArrayQueue`](https://crates.io/crates/crossbeam-queue)
fits the same role with near-identical performance characteristics
(also MIT OR Apache-2.0, also lock-free, also runtime-independent).
The API differs slightly: `push(t) -> Result<(), T>`,
`pop() -> Option<T>`, no `try_iter`. We pick `concurrent-queue`
for its smaller dependency footprint (no `crossbeam-utils`
transitive) but the choice is reversible without changing the
public engine API.

**Snapshot semantics.** The ingress contents ARE snapshotted (so
in-flight envelopes survive restart): `ConcurrentQueue` doesn't
expose iteration over a live queue, but at snapshot time we
drain into a `Vec<IngressEvent>` via the same `drain_all()` the
engine uses, serialize, and store as part of `TransientSnapshot`.
The `AtomicWaker` is NOT snapshotted (runtime-only state).

### 2.2.1 Cargo.toml entry

```toml
[dependencies]
concurrent-queue = { version = "2", default-features = false }
atomic-waker     = { version = "1", default-features = false }
```

`default-features = false` keeps `no_std` compatibility in scope
for future embedded Node deployments — both crates support
`no_std + alloc` with their default features disabled.

### 2.3 Why two queues

| Aspect | Frontier | Ingress |
|---|---|---|
| Concurrency | Single-threaded (engine only) | Multi-producer, single-consumer |
| Data structure | `VecDeque<(OpRef, ExecId)>` | `Mutex<VecDeque<IngressEvent>>` |
| Per-op cost | ~10 ns | ~100 ns (lock + push) |
| Pop pattern | Single, in a tight loop | Bulk drain at cycle start |
| Capacity | Unbounded (growable Vec) | Bounded, drops on overflow |
| Snapshot | Yes (part of TransientSnapshot) | Yes |
| Waker | N/A | Stored alongside; fired on push |

The split lets the hot path (DAG walking) stay lock-free + cache-hot
while the cold path (cross-thread event injection) pays only for
itself.

### 2.4 What else lives where

- **Outbound queue** (`Vec<WireEnvelope>` to be drained as
  `SendEnvelope` steps) — single-threaded, on the Engine, drained at
  the end of each cycle. Not an ingress; outbound.
- **Bus** (`TypedBus`) — the in-cycle event channel between Ops.
  Single-threaded. Drained between cycle phases.
- **Pending_completions** (the queue `RuntimeResourceRef` exposes for
  `complete_command` inside a hook) — single-threaded, lives on the
  current `RuntimeResourceRef` instance, drained after each hook
  returns. Not the ingress; the ingress is for completions arriving
  from off-thread.
- **Pending_async** (`HashMap<CommandId, PendingAsync>` of suspended
  Ops) — single-threaded, on the Engine. Looked up by
  `handle_completion`.
- **Scheduler** (timer heap) — single-threaded, on the framework
  Components bundle. Polled at the timer phase of each cycle.

---

## Part 3 — Engine state organization

### 3.1 The runtime linker model

The Node holds ONE canonical `ModelProto`. `Node::ensure_ready` is
the linker — every registered `ModelProto.functions[]` entry
gets merged with ODR (One Definition Rule) dedup at register time:
two Modules carrying identical FunctionProto bodies under the same
`(domain, name, overload)` share one installed body. Conflicting
bodies surface as `LoadError::FunctionDefinitionConflict`. Recursive
function references (forbidden by ONNX) raise
`LoadError::RecursiveFunctionCall`.

This makes the Node a symbol table over the function library. A
NodeProto whose `(domain, op_type, overload)` matches a registered
FunctionProto's `(domain, name, overload)` is a function call site;
multiple call sites — from any combination of registered Modules —
share the same body OpRefs in `Engine.graphs` (embedded-memory win).

### 3.2 Engine fields

```rust
pub struct Engine {
    // ─── Graph storage ─────────────────────────────────────────
    /// Installed function bodies, one per FunctionProto in the
    /// linked `Node.model.functions[]`. Keyed by a derived string
    /// (`<domain>::<name>` or `<domain>::<name>#<overload>`).
    /// The map IS the symbol table — call NodeProtos resolve here.
    graphs: HashMap<GraphName, InstalledGraph>,

    /// Symbol-table index keyed by the canonical
    /// `(domain, name, overload)`. Populated by
    /// `Engine::install_function_library` alongside `graphs`. Used
    /// at install for dispatch resolution and at snapshot/restore
    /// time to roundtrip the function library.
    functions: HashMap<(String, String, String), FunctionProto>,

    // ─── Dispatch ──────────────────────────────────────────────
    /// Op TypeId → invoke fn pointer. Populated at install.
    dispatch_table: HashMap<TypeId, DispatchEntry>,

    /// Per-Node decoder table (the §14 hash-keyed map).
    decoders: HashMap<u64, DecodeFn>,

    /// Per-Node atomic dispatch table — the single index of every
    /// `(domain, op_type, instance)` that routes to a bound runtime
    /// impl's `dispatch_atomic`. Populated from each registered impl's
    /// `atomic_opset()` declaration at Node::ready() time.
    /// The `instance` slot is the per-Node instance_id assigned during
    /// recording (see IR_AND_DSL.md §2 + COMPILER.md §10.5); it
    /// distinguishes multiple instances of the same concrete type.
    /// See ROLES.md §2 for the universal contract.

    // ─── Component storage ─────────────────────────────────────
    /// Bound runtime impls indexed by `ComponentRef::as_u32() as usize`.
    /// The `Option<...>` wrapper lets `invoke_atomic` take the
    /// dispatching component out via `mem::take` so a live
    /// `ComponentsView` can borrow `&self.components` for cross-
    /// component reads while the dispatch closure runs.
    components: Vec<Option<Box<dyn ErasedComponent>>>,

    /// The Node's own `PeerId`. Threaded into every
    /// `RuntimeResourceRef` so Components can identify themselves
    /// in outbound envelopes.
    self_peer: PeerId,

    /// Per-Node framework primitive bundle (scheduler, peer_gate,
    /// request_tracker, backoff_table, inbound_dedup, address_book,
    /// outbound_queue, event_source, etc.).
    framework: FrameworkComponents,

    /// The per-Node typed event bus.
    bus: TypedBus,

    // ─── Execution state ───────────────────────────────────────
    /// Per-poll execution-state bundle: frontier, slot table,
    /// per-execution liveness, parked async ops + in-cycle
    /// completions, function-call invocation frames, and the
    /// monotonic ID allocator. See `crate::exec_state::ExecState`.
    exec: ExecState,

    /// Reverse index from fused `binding_id` → `ComponentRef`.
    /// Populated at install; read by dispatch resolution to bind
    /// NodeProtos that reference a backend by `binding_id`.
    binding_id_index: HashMap<String, ComponentRef>,

    /// Per-event-kind bus subscriptions, keyed by destination
    /// `NodeSiteId`. Built at install from every `EventSource`
    /// syscall op. Phase 3 of `poll` writes a `TriggerValue` to
    /// each subscribed site at a fresh `ExecId` and pushes
    /// downstream consumers — matching wire delivery semantics
    /// per `docs/ADDRESSING.md`.
    event_subscriptions: HashMap<String, Vec<NodeSiteId>>,

    /// Per-LifecyclePhase op enrollments. Phase 7 of `poll`
    /// pushes every enrolled op when `Engine::fire_lifecycle`
    /// queues the phase.
    lifecycle_table: HashMap<String, Vec<OpRef>>,

    // ─── Async + cross-thread ──────────────────────────────────
    /// Thread-safe inbox for external events. Producers may push
    /// from any thread; the engine drains serially.
    ingress: Arc<IngressQueue>,

    /// Lifecycle phases queued for Phase 7 firing.
    fired_phases: Vec<String>,

    // ─── Dispatcher registries (per-Engine) ────────────────────
    /// Concrete-type role dispatchers, keyed by `TypeId::of::<T>()`.
    /// Consulted at install when resolve_dispatch needs the
    /// per-type closure for a newly registered component.
    role_dispatchers: HashMap<TypeId, RoleDispatcher>,

    // ─── Slot registry (atomic dispatch) ───────────────────────
    /// Author-chosen-slot-name → `ComponentRef` registry. Every
    /// install path populates it; every dispatch path reads
    /// through it. `ComponentsView::for_slot` consults this for
    /// sibling-component access.
    slots: HashMap<String, ComponentRef>,

    /// Parallel index: compiler-assigned `slot_id` →
    /// `ComponentRef`. Populated alongside `slots` at install.
    /// `resolve_dispatch` reads a role NodeProto's
    /// `ai.bytesandbrains.slot_id` metadata, looks up
    /// `slot_id_to_cref`, and stamps `OpDispatch::Atomic`.
    slot_id_to_cref: HashMap<u32, ComponentRef>,

    /// Per-component declared roles. Reported by
    /// `Engine::roles_for` for introspection.
    component_roles: HashMap<ComponentRef, HashSet<ComponentRole>>,

    // ─── Production-safety caps ────────────────────────────────
    /// Per-`NodeConfig.cycle_op_budget`. When set, `poll` yields
    /// after this many invocations and emits
    /// `EngineStep::CycleBudgetExceeded`.
    cycle_op_budget: Option<usize>,

    /// Per-`NodeConfig.max_pending_async`. When at cap, an Op
    /// returning `DispatchResult::Async(_)` fails synchronously.
    max_pending_async: Option<usize>,

    /// Total in-flight ingress bytes the engine holds across the
    /// ingress queue + slot table at any instant. Boundary callers
    /// (wire-decode in `decode_typed_fill`, `Node::deliver_event`,
    /// `Node::invoke`, `CompletionSink::complete`) charge before
    /// installing payloads; slot-table overwrite / eviction
    /// releases via `SlotValue::charged_bytes`. Capped at
    /// `NodeConfig::ingress_byte_budget`
    /// (`bb-runtime/src/engine/core.rs:264-274,540-552`).
    ingress_bytes_in_flight: usize,

    /// Slot id → `(role, ComponentRef)` map populated at install
    /// time. `decode_typed_fill` reads this to discover the backend
    /// bound to a tensor slot and route through the
    /// backend-mediated branch
    /// (`bb-runtime/src/engine/core.rs:236,811-827`).
    slot_id_to_role_ref: HashMap<u32, (ComponentRole, ComponentRef)>,

    /// `PhantomData<*const ()>` makes `Engine` neither `Send` nor
    /// `Sync` — the single-threaded sans-IO contract is enforced
    /// at compile time. `Arc<IngressQueue>` is still `Send+Sync`,
    /// so producers can push from other threads.
    _not_send: PhantomData<*const ()>,
}
```

Per-graph state lives in `GraphSlot`:

```rust
pub struct GraphSlot {
    /// The graph identity (name).
    name: String,
    /// The function defining the graph body.
    function: FunctionProto,
    /// Producer site → downstream consumer ops.
    consumers: HashMap<NodeSiteId, Vec<OpRef>>,
    /// Site name → NodeSiteId allocation.
    site_names: HashMap<String, NodeSiteId>,
    /// `OpDispatch` per NodeProto in `function.node[]`. Stamped
    /// at `resolve_dispatch` time; runtime invoke is one
    /// positional probe (`op_dispatch[node_idx]`).
    op_dispatch: Vec<OpDispatch>,
    /// Whether this graph is the entry-point root of an install
    /// target. Only entry-point graphs surface their top-level
    /// outputs as `EngineStep::AppEvent`.
    is_entry_point: bool,
}
```

---

## Part 4 — Concurrent modules on one Node

A Node may host many modules simultaneously. Common patterns:

- A server module + a client module on the same Node (asymmetric
  workload).
- A training module + an inference module on the same Node, both
  bound to the same `CpuBackend` slot or to distinct slots.
- A control-plane Module driving discovery + a data-plane Module
  driving training, both reachable on the same Node.

The engine treats each module as **independent**. Each module
authored against the DSL contributes one `FunctionProto` to the
compiler's emitted `ModelProto`; the install path promotes the
named `target` function to an entry-point `GraphSlot` and pushes
every other function into the function library so the
`function_call` syscall can resolve cross-module invocations.

- Each `GraphSlot` has its own `consumers` map and `site_names`
  allocation.
- All graphs share `framework`, `bus`, `slot_table`,
  `dispatch_table`, and the engine's `next_exec_id` counter
  (per-Node scope). Wire envelopes route by multiaddr per
  [ADDRESSING.md](ADDRESSING.md) — no routing_table HashMap.

NodeSiteIds are allocated globally across all modules — the
`NodeSiteId::allocate()` atomic counter ensures no two modules share
a site even by accident. OpRefs are also globally allocated.

### 4.1 Module isolation

- **Slot table**: per-(NodeSiteId, ExecId) keying. Two modules' Ops
  write to different NodeSiteIds; no cross-contamination.
- **Frontier**: a single VecDeque holds Ops from all modules. The
  engine doesn't distinguish — an Op is an Op. This lets the engine
  treat cross-module cascades the same as within-module cascades.
- **Components**: modules may share a single concrete (the bind
  chain points two slots at the same `bind_<role>::<T>(...)` call,
  one ComponentRef serves both) or use distinct concretes (two
  separate bind calls, two ComponentRefs). The engine's per-Node
  slot registry (`Engine::bind_slot`) maps each slot name to the
  serving ComponentRef.
- **Bus**: shared per-Node. Cross-module signaling possible (training
  module emits an event; inference module subscribes).

### 4.2 Cross-module data flow

The wire opset is the only sanctioned cross-module data path:

```
Module A: ... → Wire::Send(data, peers=[self]) → ...
Module B: ... ← Wire::Recv(payload_type) ← ...
```

A Wire::Send targeted at `peers = [self_peer_id]` produces a
WireEnvelope routed to the same Node's deliver_inbound, then to the
matching Wire::Recv on Module B. This is the same machinery used for
cross-Node communication; Nodes can self-deliver to bridge between
their own modules.

Direct cross-module slot writes are forbidden (each module's slots
are isolated by site allocation).

### 4.3 Per-module backend / role bindings

The three-phase contract — AUTHOR → COMPILE → INSTALL — binds
runtime components at compile time and supplies per-deployment
configs at install time:

```rust
let model = MyModule.build()?;

let compiled = bb::Compiler::new()
    .bind_backend::<CpuBackend>("compute")
    .compile(model)?;

let node = bb::install(
    peer_id,
    vec![Address::empty()],     // ordered local-address bag
    compiled,
    &["MyModule"],              // one entry per target function
    bb::Config::new(),
)?;
```

`Compiler::bind_<role>::<T>(slot)` records each slot's concrete
into the artifact's binding spec; the binding metadata stamps onto
`model.metadata_props` as `ai.bytesandbrains.binding.<target>.<slot>
= "<role>|<TYPE_NAME>|<slot_id|-1>"`. `bb::install` verifies the
compilation passport, picks every function named in `targets` out
of `model.functions[]`, walks each target's binding entries,
deduplicates shared slot bindings across targets (one
`ComponentRef` per slot — see §4.4 below), constructs each concrete
exactly once via its inventory-registered `construct_fn`, and
registers every target as a host-facing module. When an Op fires,
the engine resolves its slot through `Engine::bind_slot` and
dispatches to the ComponentRef registered at install time.

A single Node hosts heterogeneous modules through compile-time
binding decisions: two modules in the same `ModelProto` can each
bind a different `Backend` concrete at their own slot, and the
install path constructs both. `Config::new().with("slot", cfg)`
ships per-slot user configs to slots whose concrete's
`Config` associated type is not `()`.

### 4.4 Multi-target install + shared ComponentRef invariant

`bb::install(peer_id, addresses, model, targets: &[&str], config)`
(`src/install.rs:235-338`) accepts an **ordered slice** of target
function names. A federated peer hosting both `Client` and
`Server` partitions passes `&["Client", "Server"]`; single-Node
demos pass `&["MyModule"]`. Empty `targets` is rejected at the
boundary with `InstallError::EmptyTargets`
(`src/install.rs:149-151,245-247`).

**Install path** (`src/install.rs:251-338`):

1. **Resolve each target name** to a `FunctionProto` via
   `find_target` — exact match wins; content-hash suffix
   (`<target>#<hash>`) is the fallback so two partitions named
   `Client` from different compiles don't collide
   (`src/install.rs:356-373`).
2. **Parse per-target bindings** from
   `model.metadata_props["ai.bytesandbrains.binding.<resolved>.<slot>"]`
   (`src/install.rs:393-433`).
3. **Dedup across targets.**
   `dedupe_bindings_across_targets`
   (`src/install.rs:524-571`) walks every target's
   `ResolvedBinding` list, groups by slot name in first-seen
   call order, and asserts every contributor for a given slot
   agrees on `(TYPE_NAME, role)`. Disagreement surfaces
   `InstallError::SlotBindingConflict { slot, conflicts: Vec<(target,
   type_name, role)> }` enumerating every contributor for the
   conflicting slot (`src/install.rs:142-148,203-212`). The
   compiler-assigned `slot_id` carries first-wins because the
   stamping pass guarantees the same id across contributors.
4. **One ComponentRef per slot.** The install loop iterates the
   deduplicated `UnifiedBinding`s in first-seen order
   (`src/install.rs:267-324`); each slot allocates exactly one
   `ComponentRef`, registers the instance once, and binds the
   slot id once. Multi-target dispatches into the same slot
   route to the same `ComponentRef` — the engine's slot tables
   (`Engine::slots`, `Engine::slot_id_to_cref`,
   `Engine::slot_id_to_role_ref`,
   `bb-runtime/src/engine/core.rs:203-221`) carry one entry per
   slot, not per target.
5. **Install every target as an entry-point graph + register as a
   module.** `install_targets`
   (`src/install.rs:469-500`) marks each resolved target's
   `GraphSlot::is_entry_point = true` so top-level outputs surface
   as `EngineStep::AppEvent`. Every other function (sub-Module
   bodies, gate carriers, lifecycle containers) lands in the
   function library so cross-target FunctionCalls resolve at
   dispatch. `Node::set_model(model)` wraps the proto in
   `Arc<ModelProto>` once; `Node::register_module(target)` runs
   per resolved target name, sharing the `Arc`
   (`src/install.rs:332-335`,
   `bb-runtime/src/node/mod.rs:55-65,530-548`).

**Backward-compat note.** Pre-1.0 means no compatibility shim:
callers update from `install(.., "name", ..)` to
`install(.., &["name"], ..)` directly. Single-target installs are
the slice-length-1 case of the multi-target path; observable
behaviour (one bootstrap, one entry-point graph, one module
registration) is unchanged.

### 4.5 AppEvent routing per target

Every resolved target name registers as an entry in
`Node::module_index` (`bb-runtime/src/node/mod.rs:55,541-547`). The
host-pushed entry points route by target name:

- `Node::deliver_event(target: &str, input: &str, &[u8])`
  (`bb-runtime/src/node/mod.rs:813`) — looks up `target` against
  `module_index`; unknown name returns
  `DeliveryError::UnknownModule`.
- `Node::invoke(target: &str, &[(&str, &[u8])])`
  (`bb-runtime/src/node/mod.rs:892`) — same lookup, batched-input
  variant.

A peer hosting `&["Client", "Server"]` calls
`node.deliver_event("Client", "incoming_grad", &bytes)` to route
into the Client partition's `incoming_grad` port and
`node.deliver_event("Server", "tick", &bytes)` to route into the
Server partition's `tick` port. The same `Arc<ModelProto>` backs
both module-index entries; the input-name → `NodeSiteId` lookup
walks the entry-point graph identified by `target`. Multi-target
installs surface each partition's top-level outputs as
`EngineStep::AppEvent { module_name: <target>, output_name, value }`
so the host distinguishes which partition produced a given event.

---

## Part 5 — Multiple instances of the same module simultaneously

A single registered graph can have **multiple executions in flight
at the same time**. This matters for:

- Server-style graphs receiving concurrent inbound requests (each
  request triggers a fresh execution).
- Training graphs running parallel data-parallel rounds.
- Graphs triggered by both timer ticks and explicit invocations,
  where the timer fires before the previous tick's execution
  completes.

The discriminator is `ExecId`. Every entry point creates a fresh
`ExecId`; the slot_table is keyed by `(NodeSiteId, ExecId)`; the
engine tracks per-execution liveness via `execution_state`.

### 5.1 ExecId allocation

```rust
pub struct ExecId(u64);

impl ExecId {
    pub fn allocate() -> Self { /* global atomic increment */ }
}
```

ExecIds are globally unique within a Node. Each entry point that
seeds the frontier (see §6) allocates a fresh ExecId for the
execution it kicks off. The ExecId propagates through the entire
cascade: when an Op's output writes a slot, the slot is keyed
`(NodeSiteId, ExecId)`. When a downstream consumer's input is
ready, the engine pushes `(consumer_op_ref, exec_id)` onto the
frontier, propagating the same ExecId.

### 5.2 Slot table partitioning

```rust
slot_table: HashMap<(NodeSiteId, ExecId), Option<Box<dyn SlotValue>>>
```

Two concurrent executions of the same graph write to the same
NodeSiteIds but different ExecIds → no clobbering. The same Op's
output exists multiple times in the slot table, one per in-flight
execution.

### 5.3 Execution liveness + GC

```rust
pub struct ExecutionState {
    /// Which graph this execution belongs to.
    graph: GraphName,
    /// How many output sites have been written.
    sites_written: u64,
    /// Total output sites expected (so we know when done).
    sites_expected: u64,
    /// Set of OpRefs currently in pending_async for this
    /// execution.
    suspended_ops: HashSet<CommandId>,
}
```

When an execution finishes (no more frontier entries with this
ExecId, no pending_async entries), the engine garbage-collects:
removes the `ExecutionState` + every `(NodeSiteId, ExecId)` slot
table entry for the completed ExecId.

### 5.4 Cascade chaining preserves ExecId

When Op A (running with ExecId X) writes output → consumer Op B
becomes ready → engine pushes `(B, X)`. Same ExecId. The cascade is
self-contained per execution.

Cross-execution interactions only happen through:
- The bus (events broadcast to all subscribers regardless of
  ExecId).
- Component state (e.g. an Index's storage is shared across
  executions; the Index's mutable internal state is racy by
  design — the framework doesn't isolate it per ExecId).
- Wire envelopes (different executions may send to different
  peers; on inbound delivery a new ExecId is allocated for the
  responding cascade).

### 5.5 Concurrent-execution caps

The engine accepts unbounded concurrent executions by default.
Hosts that need to bound concurrency use the `Limit.Acquire /
Release` syscall ops (§22 in ENGINE.md): place a `Limit.Acquire(n)`
at the entry point of the graph; releases happen automatically when
the execution completes (the runtime detects when the matching
`Limit.Release` Op fires or the execution's `ExecutionState` is
GC'd).

---

## Part 6 — Entry points

A graph starts executing when an entry point fires + seeds the
frontier. The framework recognizes **five entry-point types**:

### 6.1 Lifecycle phase entry

`Engine::fire_lifecycle(phase)` looks up every Op whose
`Op::lifecycle_phase()` returns `Some(phase)` and pushes it onto the
frontier with a fresh ExecId per Op. Used for:

- `LifecyclePhase::Snapshot` — fires before `Node::snapshot()` so
  components can flush.
- `LifecyclePhase::Shutdown` — fires at Node shutdown.

Bootstrap is **not** a lifecycle phase — see §6.8 for the host-
driven bootstrap entry point.

Custom phases can be defined by extension components if they
register custom `LifecyclePhase` values; the engine looks them up
the same way.

### 6.2 Inbound envelope entry

`Node::deliver_inbound(src_peer, bytes)` is the entry point for
network-triggered execution. The engine ignores
`envelope.dest_peer_addresses` on the inbound side (that field is
solely for the sender's transport adapter to pick an outbound
endpoint). Before iterating `envelope.fills`, the ingress merge
runs:

- **Sender-claimed addresses.** `envelope.src_peer_addresses` decodes
  into a `Vec<Address>` and merges into the receiver's `AddressBook`
  entry for `src_peer` via `merge_src_peer_addresses` at
  `bb-runtime/src/engine/poll.rs:1013-1039`. Empty list is a no-op
  (sender chose not to advertise). The skip-on-unchanged guard
  compares the decoded slice to the existing entry via slice equality
  and elides the rewrite when they match — without this the receiver
  would update the book once per envelope.
- **Transport-observed address.** When
  `IngressEvent::EnvelopeFrom.src_observed_address` is `Some(addr)`,
  `merge_src_observed_address` at
  `bb-runtime/src/engine/poll.rs:1049-1062` containment-checks against
  the existing entry and skips the write when the address is already
  present. Missing entry bootstraps a fresh one via `add_peer`.

Claimed addresses merge first so the entry exists for the
observed-address step; observed wins for the NAT-translated case the
sender's snapshot cannot know. Both paths swallow cap errors so an
adversarial advertiser cannot abort delivery by tripping the
AddressBook cap.

The dispatch step then iterates `envelope.fills`, parses each fill's
`dest_suffix` multiaddr per [ADDRESSING.md](ADDRESSING.md), and
dispatches:

- Suffix ends in `/site/<NodeSiteId>` → data-plane path. The
  `decode_typed_fill` step
  (`bb-runtime/src/engine/poll.rs:996-1083`) pre-charges
  `fill.payload.len()` against `Engine::ingress_byte_budget`,
  branches on backend binding via `Engine::slot_id_to_role_ref`
  (`bb-runtime/src/engine/core.rs:236,811-827`) — backend-mediated
  fills move bytes into `Backend::materialize_from_wire` via
  `mem::take`; framework-carrier fills run the global
  `wire_decoder_registry` decoder against `&fill.payload` —
  writes the typed `SlotValue` at a fresh `ExecId`, and pushes
  consumers from `installed_graph.consumers[site]` onto the
  frontier. Per-fill failures emit
  `InfraEvent::WireReceiveError` + matching
  `EngineStep::WireReceiveFailed`
  (`bb-runtime/src/engine/poll.rs:1208-` etc.) and continue
  iterating sibling fills (partial-delivery semantics —
  failures NEVER short-circuit the envelope).
- Suffix ends in `/component/<ComponentRef>/op/<name>` →
  `ProtocolRuntime::dispatch_atomic` on the addressed component
  with synthesized `payload` + `correlation` inputs. The protocol
  decides when to surface work via
  `ctx.complete_command(cmd_id, ...)`, which the engine drains into
  the frontier. Control-plane fills charge against the same byte
  budget as data-plane fills; the payload bytes remain
  framework-owned `Vec<u8>` until the protocol component consumes
  them.
- Malformed/unrecognized suffix → silent drop.

Each inbound envelope creates a fresh ExecId for its responding
cascade.

### 6.3 Timer maturity entry

`Engine::poll_timers(now_ns)` walks the scheduler heap; matured
entries fire:

- `TimerKind::Completion(cmd_id)` — resumes a suspended Op by
  calling `handle_completion(cmd_id, [Trigger])`. Uses the
  suspended Op's original ExecId.
- `TimerKind::EnqueueOp(op_ref)` — pushes `(op_ref, fresh_exec_id)`
  onto the frontier. New execution.
- `TimerKind::Component { component_ref, kind }` — calls
  `ProtocolRuntime::on_timer(kind, ctx)` on the bound component.
  The component decides whether to surface work.

Timers can be host-pushed via the ingress queue's `TimerMatured`
variant (so a host-side clock task firing at `at_ns` can wake the
engine without polling).

### 6.4 App event entry

`Node::deliver_event(module: &str, input: &str, value_bytes: &[u8])`
(`bb-runtime/src/node/mod.rs:765-835`) is how the host pushes
external values into a registered Module's unconnected input. Per
Principle 1a (see [WIRE.md §Principle 1a](WIRE.md#principle-1a-external-byte-payloads-cross-as-u8))
the call takes a borrowed slice; the framework copies the bytes
into a framework-owned `Vec<u8>` inside the call. The engine:

1. Looks up `module_name`; returns `DeliveryError::UnknownModule`
   when missing.
2. Caps `value_bytes.len()` against
   `NodeConfig::max_app_event_bytes`
   (`bb-runtime/src/node/config.rs:167`). On overflow returns
   `DeliveryError::OversizePayload { byte_count, cap }` and
   publishes
   `InfraEvent::AppIngressError { source: AppEvent { module, input },
   byte_count, kind: PerItemCapExceeded { cap } }` on the bus.
3. Charges `byte_count` against
   `Engine::ingress_byte_budget`
   (`bb-runtime/src/engine/core.rs:540-552`). On overflow returns
   `DeliveryError::BudgetExceeded { byte_count, budget_remaining }`
   and publishes `AppIngressError { kind: BudgetExceeded {
   budget_remaining } }`.
4. `crate::fallible::try_reserve_exact` reserves a framework-owned
   `Vec<u8>`. On `TryReserveError` releases the budget charge,
   returns `DeliveryError::AllocationFailed { byte_count,
   reason: HeapExhausted }`, and publishes the matching
   `AppIngressError { kind: AllocationFailed { reason } }`.
5. `extend_from_slice` copies the caller's bytes in; the slice is
   never retained past the call.
6. Pushes `IngressEvent::AppEvent { module_name, input_name,
   value_bytes }` onto the ingress queue.
7. Engine consumes the event in Phase 1 of the next `poll`: wraps
   the typed value, allocates a fresh `ExecId`, writes to the
   input's `NodeSiteId`, and pushes the slot's consumers onto the
   frontier.

Multiple `deliver_event` calls for the same input each create fresh
ExecIds — fully concurrent executions of the same module driven by
host-pushed data.

### 6.5 Explicit invocation entry

`Node::invoke(module: &str, inputs: &[(&str, &[u8])])`
(`bb-runtime/src/node/mod.rs:845-952`) batches several inputs into
a single `ExecId`. Same borrowed-slice contract as `deliver_event`
per Principle 1a; caps split into `NodeConfig::max_invoke_inputs`
(count) and `NodeConfig::max_invoke_bytes` (cumulative payload
sum). The byte budget charge runs once against the cumulative sum;
per-input `try_reserve_exact` failures release the full sum
(no half-committed state in the slot table).

```rust
let exec_id = node.invoke(
    "TrainingModule",
    &[
        ("batch",  batch_bytes.as_slice()),
        ("target", target_bytes.as_slice()),
    ],
)?;
```

Returns the allocated `ExecId` so the host can correlate the
emitted `EngineStep::AppEvent`s with their originating invocation.
Failure surface mirrors `deliver_event`:
`DeliveryError::{TooManyInputs, OversizePayload, BudgetExceeded,
AllocationFailed, UnknownModule, IngressClosed}` synchronously
plus the matching `AppIngressError { source: Invoke { module,
input_count }, ... }` bus emission.

### 6.6 Async completion entry

`CompletionSink::complete(&self, cmd_id, result_bytes: &[u8])`
and `fail(&self, cmd_id, detail: &str)`
(`bb-runtime/src/runtime.rs:29-97`) are the borrowed-slice entry
points async-completing components use to wake a parked
`CommandId`. Same `Principle 1a` shape:
`NodeConfig::max_completion_result_bytes`
(`bb-runtime/src/node/config.rs:195`) caps `result_bytes.len()`;
oversize / alloc failure publishes
`AppIngressError { source: Completion { command }, byte_count,
kind: PerItemCapExceeded | AllocationFailed }` and drops. The
parked op times out naturally on the host side — same surface as
a missing completion. `detail` is truncated to
`COMPLETION_DETAIL_CAP` (4 KiB) at a UTF-8 character boundary
rather than rejected so the host's display message always
lands.

### 6.7 Engine boundary fallibility line

External boundaries entering the engine — wire ingress
(`Node::deliver_inbound` / transport adapter), application
ingress (`deliver_event`, `invoke`), and async-completion ingress
(`CompletionSink::complete` / `fail`) — use fallible allocation
(`try_reserve_exact`) AND a budget guard. On allocation failure
or budget exceedance, the engine emits a typed `InfraEvent`
(`WireReceiveError::AllocationFailed`, `::BudgetExceeded`, or
the parallel `AppIngressError` variants), DROPS the offending
bytes, and continues processing other envelopes / events.

The engine NEVER panics, NEVER aborts, NEVER stalls at the
boundary. Inside the engine (component contracts, role traits,
dispatch) normal Rust patterns apply — components are designed
to play by the runtime contract. The single fallibility line
is the boundary itself.

### 6.8 Host-driven bootstrap entry

Bootstrap is the framework's pre-body initialization phase.
Install records every `module_phase = "bootstrap"` FunctionProto
(authored via `Module::bootstrap`, see
[IR_AND_DSL.md §2](IR_AND_DSL.md#part-2--concept-to-proto-mapping))
and every Component-level `Bootstrap` impl
(authored via the `bb::Bootstrap` Contract, see
[ROLES.md §Bootstrap](ROLES.md#part-11--bbbootstrap))
onto the engine without arming the queue. **The host owns when
bootstrap fires** — `Engine::poll` no longer auto-seeds
(`bb-runtime/src/engine/poll.rs:170-180`); the body-op gate stays
dormant until the host calls `Node::run_bootstrap(target)`.

#### 6.8.1 `BootstrapState` — single owner of bootstrap fields

Engine consolidates every bootstrap field into a single struct
(`bb-runtime/src/engine/bootstrap.rs:244-299`):

```rust
pub(crate) struct BootstrapState {
    /// Per-target Module bootstrap metadata. Stamped at install
    /// when a `module_phase = bootstrap` FunctionProto lands.
    module_bootstraps: HashMap<String, ModuleBootstrap>,

    /// Per-slot Component bootstrap metadata. Populated by the
    /// `BootstrapDispatcherRegistration` install walk.
    component_bootstraps: HashMap<String, ComponentBootstrap>,

    /// Append-only sequence of Module bootstrap target names in
    /// install order. The seeder walks front-to-back.
    install_order: Vec<String>,

    /// Host-supplied bootstrap input staging requests parked
    /// awaiting a conflict-free slot.
    pending_requests: VecDeque<OwnedBootstrapRequest>,

    /// Currently executing bootstraps. Vec shape supports
    /// concurrent disjoint Component bootstraps.
    in_flight: Vec<InFlightBootstrap>,

    /// Validated + staged bootstraps ready to fire once
    /// `in_flight` drains (overlap promotion path).
    waiting: VecDeque<QueuedBootstrap>,

    /// Seed pointer into `install_order`. Bumps each time
    /// `maybe_complete_bootstrap` observes a phase drained.
    next_idx: usize,

    /// Coarse "queue still has work" flag the body-op gate
    /// consults to skip the bootstrap path on idle cycles.
    pending: bool,
}
```

Replaces the prior `bootstrap_function_keys` /
`bootstrap_next_idx` / `bootstrap_pending` / `bootstrap_exec_id`
quartet. Every read + write of bootstrap state goes through
`BootstrapState`; the engine borrows it as one field.

#### 6.8.2 Host kick — single entry point + `BootstrapTarget`

One public method on `Node` drives every bootstrap path; the
caller selects what fires through a `BootstrapTarget` enum
(`bb-runtime/src/node/mod.rs`):

```rust
pub enum BootstrapTarget<'a> {
    /// Drive every install-order Module bootstrap target on this Node.
    All,
    /// Drive specific Module bootstrap targets by name (with empty inputs).
    ModuleNames(&'a [&'a str]),
    /// Drive Module bootstrap targets with explicit inputs.
    ModuleRequests(&'a [BootstrapRequest<'a>]),
    /// Drive Component bootstraps by slot name.
    Slots(&'a [&'a str]),
}

impl Node {
    /// Drive bootstrap targets to completion. Returns the full
    /// `Vec<EngineStep>` the bootstrap path emitted; idempotent on
    /// a Node whose queue already drained (returns an empty Vec).
    pub fn run_bootstrap(
        &mut self,
        target: BootstrapTarget<'_>,
    ) -> Result<Vec<EngineStep>, BootstrapError>;

    /// Inspect bootstrap state without triggering anything.
    pub fn bootstrap_status(&self) -> BootstrapStatus;
}
```

Per-variant semantics:

- **`BootstrapTarget::All`** — arm the install-order queue, seed
  the first target, drive `Engine::poll` in a loop until every
  queued bootstrap drains (or one suspends on async). The
  canonical "kick the install-order queue" call after
  `bb::install`.
- **`BootstrapTarget::ModuleNames(&["A", "B"])`** — batch entry
  for Module bootstraps that take no input formals; each name
  stages an empty-input request before firing.
- **`BootstrapTarget::ModuleRequests(&[BootstrapRequest])`** —
  batch entry for Module bootstraps with input formals. Each
  `BootstrapRequest { target: &str, inputs: &[(&str, &[u8])] }`
  stages owned-form bytes into the target's input sites via the
  F5 immediate-fire path. Validates atomically up-front
  (unknown target / duplicate batch entry surfaces
  `BootstrapError::{UnknownTarget, AlreadyTransitivelyQueued}`
  before any staging happens).
- **`BootstrapTarget::Slots(&["compute"])`** — fire Component
  bootstraps by slot name. Resolves `slot → ComponentRef` via
  the `bootstrap.component_bootstraps` registry the install
  walk populated; dispatches through the registered `Bootstrap`
  dispatcher.

`enqueue_bootstrap_request` is `pub(crate)` only — the engine's
input-staging helper, not a user-facing surface. Authors compose
multi-input bootstraps through `BootstrapTarget::ModuleRequests`,
which runs the Principle 1a copy on each `(input_name, &[u8])`
pair before pushing the body's `OpRef`s onto the frontier.

#### 6.8.3 Component-level invocation path

Component bootstraps fire as a synthetic single-op dispatch.
`Engine::fire_component_bootstrap`
(`bb-runtime/src/engine/core.rs:1319-1397`) resolves
`slot → ComponentRef`, allocates a fresh `ExecId`, locks the
`{cref}` touch set on `bootstrap.in_flight`, and invokes
`Bootstrap::bootstrap(&mut ctx)` through the per-T dispatcher
registry (`bb-runtime/src/engine/invoke.rs:1019-1050`). The
`#[derive(bb::Concrete)]` macro emits the dispatcher registration
automatically via the `BootstrapDispatcherRegistration` inventory
carrier (`bb-runtime/src/registry.rs:170-208`,
`bb-derive/src/roles.rs:46-79`); see
[CONTRACT_DISPATCH.md](CONTRACT_DISPATCH.md#bootstrap-is-just-another-contract-method)
for the bridge.

Synchronous `DispatchResult::Immediate(_)` retires the in-flight
entry inline via `BootstrapState::on_bootstrap_drained` and fires
any promoted waiters whose touch set no longer conflicts
(`bb-runtime/src/engine/core.rs:1356-1366`).
`DispatchResult::Async(cmd_id)` parks the ExecId on
`pending_async` under a synthetic `OpRef`
(`OpRef::pack(u32::MAX, 0)`) so the regular `handle_completion`
path drives the eventual drain
(`bb-runtime/src/engine/core.rs:1368-1386`).

#### 6.8.4 Per-component gate — `is_op_locked`

The body-op gate (`Engine::is_op_locked`,
`bb-runtime/src/engine/core.rs:1762-1806`) parks any body op
whose touched `ComponentRef` falls inside some in-flight
bootstrap's `touch_set`. Disjoint components keep firing.

Resolution order (per the docstring at
`bb-runtime/src/engine/core.rs:1743-1761`):

1. No in-flight bootstraps → fire (gate dormant).
2. `exec_id` descends from some in-flight bootstrap's ExecId
   via the `pending_calls.parent_exec_id` chain → fire. The
   bootstrap body itself and its sub-FunctionCalls invoke ops
   freely.
3. Resolve the touched `ComponentRef` from the op's NodeProto via
   `SLOT_ID_KEY → slot_id_to_cref`. If the touched cref is in
   some in-flight bootstrap's `touch_set` → park. Stateless
   syscalls (no slot_id stamp, no role binding) fire because
   they reach no component.

`pop_frontier_fireable` (`bb-runtime/src/engine/core.rs:2096-2110`)
scans the frontier for the first entry the gate accepts; parked
ops stay on the frontier until the in-flight set drops.

The touch set is the closure of every `ComponentRef` referenced
by the bootstrap function body (slot-id NodeProtos + transitive
FunctionCalls), computed once at install time by
`Engine::compute_touch_set`
(`bb-runtime/src/engine/core.rs:1145-1196`) — see
[COMPILER.md §Touch-set computation](COMPILER.md#touch-set-computation).

#### 6.8.5 Conflict queue + concurrent in-flight bootstraps

`BootstrapState::process_pending_requests`
(`bb-runtime/src/engine/bootstrap.rs:459-497`) drains the parked
request queue once per poll. For each request the engine looks
up the target's touch set and compares against every currently
in-flight bootstrap's `touch_set`. Disjoint targets surface as
`ReadyBootstrap`s the engine seeds immediately; overlapping ones
park as `QueuedBootstrap`s in `bootstrap.waiting`.

`BootstrapState::on_bootstrap_drained`
(`bb-runtime/src/engine/bootstrap.rs:499-522`) retires the
in-flight entry by ExecId, then walks `waiting` once and promotes
any waiter whose touch set no longer conflicts.
`Engine::maybe_complete_bootstrap`
(`bb-runtime/src/engine/core.rs:1824-1887`) is the call site —
it runs after every drain phase, advances `next_idx`, and fires
promoted waiters in-cycle so the host sees every
`BootstrapComplete` step and the body's first ops in a single
poll when the budget permits.

#### 6.8.6 Lifecycle status

`Node::bootstrap_status()` returns
`BootstrapStatus::{Idle, Running, WaitingForInput}`
(`bb-runtime/src/engine/bootstrap.rs:114-128`) without advancing
any queue. `Running` means at least one entry occupies
`bootstrap.in_flight`; `WaitingForInput` means the install-order
queue still has unseeded targets or host-staged requests sit on
`pending_requests` / `waiting`; `Idle` otherwise. The host
consults this when deciding whether to call `run_bootstrap` again
with `ModuleRequests` to stage more inputs, or to move on to the
body poll loop.

The engine surfaces per-target completions as
`EngineStep::BootstrapComplete` and async suspensions as
`EngineStep::WaitingOnBootstrap`
(`bb-runtime/src/engine/step.rs:60-75`,
`bb-runtime/src/engine/poll.rs:450-470`). Each target's
bootstrap emits one `BootstrapComplete` step in install order
before the next seeds.

---

## Part 7 — The poll cycle in detail

```rust
impl Node {
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Vec<EngineStep>> {
        let mut steps = Vec::new();
        let mut any_work = false;

        // Phase 1 — Drain ingress.
        let ingress_events = self.engine.ingress.drain_all();
        if !ingress_events.is_empty() {
            any_work = true;
            for event in ingress_events {
                steps.extend(self.process_ingress_event(event));
            }
        }

        // Phase 2 — Drain frontier (initial pass).
        if !self.engine.frontier.is_empty() {
            any_work = true;
            steps.extend(self.engine.drain_frontier());
        }

        // Phase 3 — Route bus events to subscribed sites.
        // For each NodeEvent, write a `TriggerValue` to each
        // subscribed `NodeSiteId` at a fresh `ExecId` and push the
        // site's downstream consumers — uniform with the wire
        // delivery model per `docs/ADDRESSING.md`.
        if !self.engine.bus.is_empty() {
            any_work = true;
            steps.extend(self.engine.route_bus_events());
        }

        // Phase 4 — Poll matured timers.
        let now = self.now_ns();
        if self.engine.framework.scheduler.has_matured(now) {
            any_work = true;
            steps.extend(self.engine.poll_timers(now));
        }

        // Phase 5 — Drain any pending_completions accumulated by
        //           hooks during the above phases.
        if !self.engine.pending_completions.is_empty() {
            any_work = true;
            steps.extend(self.engine.drain_pending_completions());
        }

        // Phase 6 — Final frontier drain (cascades from phases 3-5).
        if !self.engine.frontier.is_empty() {
            any_work = true;
            steps.extend(self.engine.drain_frontier());
        }

        // Phase 7 — Drain outbound queue.
        for envelope in self.engine.framework.outbound_queue.drain_all() {
            any_work = true;
            steps.push(EngineStep::SendEnvelope(envelope));
        }

        // Phase 8 — Decide Pending vs Ready.
        if any_work {
            Poll::Ready(steps)
        } else {
            // Stash the waker. The ingress queue's push will fire it
            // when new work arrives.
            *self.engine.waker.lock() = Some(cx.waker().clone());
            // Defensive: also stash on the scheduler so timer
            // maturity wakes us.
            self.engine.framework.scheduler.set_waker(cx.waker().clone());
            Poll::Pending
        }
    }
}
```

### 7.1 Ingress event processing

Each `IngressEvent` translates to engine work:

```rust
fn process_ingress_event(&mut self, event: IngressEvent) -> Vec<EngineStep> {
    match event {
        IngressEvent::Envelope(env) => {
            self.deliver_inbound_internal(env)
        }
        IngressEvent::AppEvent { module_name, input_name, value_bytes } => {
            self.deliver_event_internal(&module_name, &input_name, value_bytes)
        }
        IngressEvent::TimerMatured { at_ns } => {
            self.engine.poll_timers(at_ns)
        }
        IngressEvent::Invoke { module_name, inputs } => {
            self.invoke_internal(&module_name, inputs)
        }
        IngressEvent::Completion { cmd_id, results } => {
            let decoded = self.decode_completion_results(cmd_id, results)?;
            self.engine.handle_completion(cmd_id, decoded)
        }
        IngressEvent::Control(sig) => {
            self.handle_control(sig)
        }
    }
}
```

### 7.2 The internal deliver_inbound / deliver_event

`Node::deliver_inbound(envelope)` and `Node::deliver_event(...)` are
public APIs the host calls. They push onto the ingress AND
immediately wake the poll loop. The actual processing happens in
the next poll cycle's Phase 1.

This indirection matters: pushing through the ingress lets the host
call `deliver_inbound` from a transport task running on a different
thread than the engine. The engine never sees concurrent mutation;
it processes events in cycle order.

For synchronous in-thread use (e.g. tests calling
`node.deliver_inbound(env)` then `node.poll(cx)`), the indirection
costs nothing — the next poll drains the ingress immediately.

### 7.3 Phase ordering rationale

- **Ingress first** so external work has priority over recurring
  internal work.
- **Frontier between phases** so each phase's cascades complete
  before the next phase opens.
- **Timers after frontier** so a long cascade doesn't starve them.
- **Pending_completions before final frontier** so completions from
  hooks fire in the same cycle.
- **Outbound last** so all in-cycle Op fires get a chance to push
  outbound envelopes before the host ships them.

### 7.4 Worst-case work per cycle

Unbounded by default. The host SHOULD cap cycle duration via a
fairness mechanism if needed: stop draining frontier after N ops,
re-poll. The framework ships
`NodeConfig::with_cycle_op_budget(usize)`; once N Ops have fired in
a cycle, the engine stashes a "more work pending" signal +
returns Poll::Ready early so the host can interleave other tasks.
Default budget is `usize::MAX` (no limit).

---

## Part 8 — Op invocation lifecycle

### 8.1 invoke_one

Dispatch is **pre-stamped at install time** by
`Engine::resolve_dispatch`. Each `InstalledGraph` carries
`op_dispatch: Vec<OpDispatch>` parallel to `function.node[]`.
Runtime invoke is one indirect probe — no HashMap lookups on the
hot path. `OpDispatch` has four variants:

| Variant | Set when | Runtime action |
|---|---|---|
| `Stateless(fn)` | NodeProto's `(domain, op_type)` matches a registered syscall in `syscall_index`. | Call the stateless invoke fn. |
| `Atomic { component_ref }` | NodeProto matches `(domain, op_type, instance)` in `atomic_dispatch`. | Route to `components[cref].dispatch_atomic`. |
| `FunctionCall { target, input_rename, output_rename }` | NodeProto's `(domain, op_type, overload)` matches a registered FunctionProto AND domain is `ai.bytesandbrains.module`. | Splice the body's OpRefs onto the frontier with renames; see §8.4. |
| `Unresolved` | install couldn't classify. | Fail the op at invoke time with `unresolved dispatch for <domain>::<op_type>`. |

```rust
fn invoke_one(&mut self, op_ref: OpRef, exec_id: ExecId) -> EngineStep {
    let _span = tracing::info_span!("invoke", ?op_ref, ?exec_id).entered();

    // 1. Pre-stamped per-OpRef dispatch — one indirect probe.
    let (graph, idx) = self.graph_and_index_for(op_ref)?;
    let dispatch = graph.op_dispatch[idx].clone();

    // 2. Build typed input map (renamed per call context if we're
    //    executing inside a FunctionCall scope; see §8.4).
    let inputs = self.build_typed_inputs(op_ref, exec_id)?;
    let mut ctx = RuntimeResourceRef::new(&mut self.framework, &mut self.bus, op_ref);

    let result: OpResult = match dispatch {
        OpDispatch::Stateless(invoke_fn) => invoke_fn(&op_box, &mut ctx, &inputs),
        OpDispatch::Atomic { component_ref } => {
            let runtime = self.components.get_mut(&component_ref)?;
            runtime.dispatch_atomic(&entry.op_type, &inputs, &mut ctx)?
                .into_op_result()
        }
        OpDispatch::FunctionCall { target, input_rename, output_rename } => {
            // See §8.4.
            self.invoke_function_call(op_ref, exec_id, &target, &input_rename, &output_rename)
        }
        OpDispatch::Unresolved => self.fail_op(
            op_ref,
            exec_id,
            &format!("unresolved dispatch for {}::{}", node.domain, node.op_type),
        ),
    };

    // 5. Drain pending_completions accumulated during the call.
    self.pending_completions.extend(ctx.pending_completions.drain(..));

    // 6. Interpret OpResult.
    match result {
        OpResult::Sync(values) => {
            self.write_outputs(op_ref, exec_id, values);
            EngineStep::OpCompleted { op_ref, exec_id, sites_written: ... }
        }
        OpResult::Async(cmd_id) => {
            self.pending_async.insert(cmd_id, PendingAsync {
                op_ref, exec_id,
                output_sites: self.op_outputs(op_ref),
            });
            EngineStep::AsyncSuspended { op_ref, exec_id, cmd_id }
        }
        OpResult::Trigger => {
            EngineStep::OpTrigger { op_ref, exec_id }
        }
        OpResult::Sink => {
            EngineStep::OpSink { op_ref, exec_id }
        }
        OpResult::Failed(err) => {
            self.bus.publish(NodeEvent::Infra(InfraEvent::OpFailure {
                op_ref, error: err.clone(),
            }));
            EngineStep::OpFailed { op_ref, exec_id, error: err }
        }
        OpResult::SyncPartial(opts) => {
            self.write_outputs_partial(op_ref, exec_id, opts);
            EngineStep::OpCompleted { op_ref, exec_id, sites_written: ... }
        }
    }
}
```

### 8.2 Output write + consumer pushing

```rust
fn write_outputs(&mut self, op_ref: OpRef, exec_id: ExecId, values: Vec<Box<dyn SlotValue>>) {
    let output_sites = self.op_outputs(op_ref);
    for (site, value) in output_sites.iter().zip(values) {
        self.slot_table.insert((*site, exec_id), Some(value));
    }
    self.bump_execution_state(exec_id, output_sites.len());

    // Push consumers whose inputs are now all ready.
    for site in output_sites {
        if let Some(consumers) = self.consumers.get(site) {
            for &consumer_op in consumers {
                if self.all_inputs_ready(consumer_op, exec_id) {
                    self.frontier.push_back((consumer_op, exec_id));
                }
            }
        }
    }
}
```

### 8.3 all_inputs_ready check

```rust
fn all_inputs_ready(&self, op_ref: OpRef, exec_id: ExecId) -> bool {
    let entry = &self.op_entry(op_ref);
    let port_readiness = entry.op.input_readiness();
    let mut any_of_groups: HashMap<&str, bool> = HashMap::new();

    for (i, input) in entry.op.input_ports().iter().enumerate() {
        let has_value = self.slot_table
            .get(&(input.site, exec_id))
            .map(|v| v.is_some())
            .unwrap_or(false);

        match port_readiness[i] {
            InputReadiness::Required => {
                if !has_value { return false; }
            }
            InputReadiness::AnyOf(group) => {
                let entry = any_of_groups.entry(group).or_insert(false);
                if has_value { *entry = true; }
            }
        }
    }

    any_of_groups.values().all(|&filled| filled)
}
```

`Required` inputs MUST all be populated. `AnyOf(group)` inputs in
the same group form an OR — at least one must be populated. Multiple
distinct `AnyOf` groups are AND'd.

### 8.4 Function-call splice (`OpDispatch::FunctionCall`)

When `op_dispatch[idx]` is `FunctionCall { target, input_rename,
output_rename }`, the engine performs a flat-frontier splice:

1. **Resolve target.** `engine.graphs[graph_name_for(target)]` is
   the body's installed graph. Its `op_refs` are shared across every
   call site that targets this function — one allocation regardless
   of N call sites.
2. **Allocate body ExecId.** A fresh `ExecId::allocate()` isolates
   this call's slot_table entries from concurrent calls of the same
   body.
3. **Set up the call context.** `pending_calls[body_exec_id] =
   CallContext { parent_exec_id, input_rename, output_forwarding }`.
   `input_rename` maps each formal parameter name → caller-side
   `(graph_name, NodeSiteId)`. `output_forwarding` maps each body
   output `NodeSiteId` → caller-side `NodeSiteId`.
4. **Push the body's OpRefs** onto the frontier at `body_exec_id`.
5. **Input read.** When a body node fires, `build_typed_inputs`
   checks `pending_calls[exec_id]`. For inputs matching a formal
   parameter name, it reads from the caller's slot at
   `parent_exec_id`; for body-internal values it reads from the
   body's slot at `body_exec_id`. No value copying — values stay
   in the caller's scope and are looked up by alias.
6. **Output write hook.** When `write_outputs` writes a body output
   site, it also writes the value to the corresponding caller-side
   site at `parent_exec_id` per `output_forwarding`, and re-runs
   `push_ready_consumers` for the caller's downstream.
7. **Cleanup.** When all body outputs have been forwarded, the
   `pending_calls` entry is removed.

This preserves call-frame semantics without recursion in the engine
and without a scope stack — embedded-friendly.

**Concrete implementation** — `src/engine/invoke.rs::invoke_function_call`:
- `CallContext` (per `src/engine/call_context.rs`) holds
  `parent_exec_id`, `target`, `input_aliases`, `output_forwarding`,
  `outputs_remaining`. Stored on `Engine.pending_calls: HashMap<ExecId,
  CallContext>` keyed by the body's fresh `ExecId`.
- `resolve_input_pairs` (invoke.rs:629) is alias-aware: when
  `pending_calls.contains_key(&exec_id)`, formal-name inputs route
  through `input_aliases` to the caller-side `NodeSiteId` read at
  `parent_exec_id`. Zero-copy — body nodes read the caller's slot
  directly.
- `forward_outputs_to_caller` (invoke.rs:715) runs inside
  `write_outputs` after every body write: for each just-written
  body site present in `output_forwarding`, the value is moved to
  the matching caller site at `parent_exec_id` (MOVE not clone —
  `SlotValue` isn't `Clone`; function outputs are one-shot so
  the move is semantically correct), `push_ready_consumers` runs
  for the caller's downstream, and `surface_top_level_outputs`
  fires for entry-point AppEvent semantics. `outputs_remaining`
  decrements; the `pending_calls` entry is removed at zero.
- The zero-output corner case is collapsed at call-setup time so
  no leaked entries accumulate.

Covered by `src/engine/invoke_function_call_tests.rs` (6 tests).

### 8.5 Backend-subgraph dispatch — future

Reserved for a future compiler pass that fuses contiguous Backend-
Contract atomic ops into a single subgraph dispatched whole-graph
through `Backend::execute`. The dispatch variant
(`OpDispatch::BackendSubgraph`), the compiler pass
(`collapse_backend_subgraphs`), and the runtime entry
(`Engine::invoke_backend_subgraph`) are not in the production
pipeline today; the per-op surface on the Backend Contract handles
the whole dispatch story.

**Concrete implementation** — `src/engine/invoke.rs::invoke_backend_subgraph`:
- `BackendCompute` is not dyn-safe (associated `Tensor` + `Error`
  types). The engine uses the same dispatcher-registry pattern as
  `ProtocolRuntime`: `register_backend_subgraph_dispatcher::<T>()`
  captures a per-concrete-type closure into a process-global
  `BACKEND_SUBGRAPH_DISPATCHERS` (`invoke.rs:1002`). At dispatch
  time the engine upcasts the bound component to `&dyn Any`, walks
  the registry, and the matching closure does
  `as_any().downcast_ref::<T>()`, downcasts each
  `&dyn SlotValue` input to `&T::Tensor` via `as_any()`, calls
  `T::backend_subgraph(body, inputs)`, and boxes outputs back to
  `Box<dyn SlotValue>` via the blanket
  `impl<T: Tensor> SlotValue for T`.
- `Engine.binding_id_index: HashMap<String, ComponentRef>`
  (core.rs:166) is populated by `Node::ensure_ready` at
  install time (`node.rs:644,649`) — concrete bindings register
  as `<type_name>#<instance>`, generics as `<trait>#<slot_id>`.
  `lookup_backend_for_call` (core.rs:454) reads the call
  NodeProto's `binding_id` metadata and returns the bound
  `ComponentRef`.
- `function_to_graph_view` is duplicated locally in
  `invoke.rs:859` (avoids a cross-module API change vs. the
  compiler's private helper).

Covered by `src/engine/invoke_backend_subgraph_tests.rs` (4 tests).

---

## Part 9 — CommandId + async completion

### 9.1 Async suspension

An Op returning `OpResult::Async(cmd_id)` indicates it kicked off
long-running work + will fulfill the cmd_id later. The engine
records the suspension:

```rust
pub struct PendingAsync {
    pub op_ref: OpRef,
    pub exec_id: ExecId,
    /// The output sites this Op declared — populated when the
    /// CommandId completes with values.
    pub output_sites: Vec<NodeSiteId>,
}
```

The Op's downstream consumers do NOT fire on `OpResult::Async`; they
wait for `handle_completion(cmd_id, values)`.

### 9.2 handle_completion

```rust
fn handle_completion(
    &mut self,
    cmd_id: CommandId,
    values: Vec<Box<dyn SlotValue>>,
) -> Vec<EngineStep> {
    let pending = self.pending_async.remove(&cmd_id)?;
    let mut steps = vec![];

    // Write values to the suspended Op's output sites.
    for (site, value) in pending.output_sites.iter().zip(values) {
        self.slot_table.insert((*site, pending.exec_id), Some(value));
    }

    // Push consumers.
    for site in &pending.output_sites {
        if let Some(consumers) = self.consumers.get(site) {
            for &consumer_op in consumers {
                if self.all_inputs_ready(consumer_op, pending.exec_id) {
                    self.frontier.push_back((consumer_op, pending.exec_id));
                }
            }
        }
    }

    steps.extend(self.drain_frontier());
    steps
}
```

### 9.3 Where completions originate

- **In-cycle from a `ProtocolRuntime` hook**: the protocol's
  `dispatch_atomic` or `on_timer` calls
  `ctx.complete_command(cmd_id, values)`; the values land in
  `RuntimeResourceRef::pending_completions`; the engine drains
  after the hook returns (Phase 5 of the poll cycle).
- **From the scheduler**: a `TimerKind::Completion(cmd_id)` matures
  → `handle_completion(cmd_id, [Trigger])` (one-shot deferred ops
  like Sleep).
- **From off-thread via ingress**: an external task fulfilling a
  CommandId pushes `IngressEvent::Completion { cmd_id, results }`
  onto the ingress. Phase 1 of the next cycle decodes + delivers.

### 9.4 Lost completions on snapshot/restore

When a Node is snapshotted with pending_async entries, those
suspensions are preserved. On restore, the entries are still
in `pending_async`. The host's responsibility is to re-trigger the
external work that was supposed to fulfill them; the framework
cannot recreate the off-thread tasks.

For `TimerKind::Completion` suspensions, the scheduler snapshots
the timer queue → restore replays it → the timer matures normally
post-restore.

---

## Part 10 — `RuntimeResourceRef` bridging

Every Op invoke receives a `RuntimeResourceRef<'a>`:

```rust
pub struct RuntimeResourceRef<'a> {
    pub peer_gate: &'a mut PeerGate,
    pub backoff_table: &'a mut BackoffTable,
    pub request_tracker: &'a mut RequestTracker,
    pub inbound_dedup: &'a mut InboundDedup,
    pub address_book: &'a mut AddressBook,
    pub peer_address_book: &'a mut PeerAddressBook,
    pub outbound_queue: &'a mut OutboundQueue,
    pub event_source: &'a mut EventSource,
    pub scheduler: &'a mut Scheduler,
    pub bus: &'a mut TypedBus,
    pub current_op_ref: OpRef,
    pub pending_completions: Vec<PendingCompletion>,
}
```

### 10.1 Split-borrow construction

The engine constructs the ref by split-borrowing the framework
bundle + the bus:

```rust
let mut ctx = RuntimeResourceRef {
    peer_gate: &mut self.framework.peer_gate,
    backoff_table: &mut self.framework.backoff_table,
    request_tracker: &mut self.framework.request_tracker,
    // ...
    bus: &mut self.bus,
    current_op_ref: op_ref,
    pending_completions: Vec::new(),
};
```

Each field is a distinct `&mut`, so the borrow checker enforces
exclusivity across fields. An Op may touch any subset.

### 10.2 ctx.complete_command

```rust
impl<'a> RuntimeResourceRef<'a> {
    pub fn complete_command(
        &mut self,
        cmd_id: CommandId,
        results: Vec<Box<dyn SlotValue>>,
    ) {
        self.pending_completions.push(PendingCompletion { cmd_id, results });
    }
}
```

Used by `ProtocolRuntime` hooks. The engine drains
`pending_completions` after the hook returns + invokes
`handle_completion` for each entry.

### 10.3 Lifetime guarantees

The ref's lifetime is bound to the engine's cycle: `'a` matches the
cycle's `&mut self` borrow of `Node`. Ops cannot store the ref past
invoke return. Async Ops that need to do work later must do it via
the bus or via CommandId completion routed back through the engine.

---

## Part 11 — Wake semantics

### 11.1 The Waker stash

When `poll(cx)` would return `Poll::Pending`, the engine stashes
`cx.waker().clone()` in:

- `IngressQueue::waker` — fires when an ingress push happens.
- `Scheduler::waker` — fires when a timer matures.

When a stash already exists, the new waker REPLACES it (matches
`std::task::Waker` semantics: only one waker active at a time).

### 11.2 Wake triggers

The waker fires (the host's executor re-polls the Node):

- `IngressQueue::push(event)` — any external thread pushing an
  event. Covers `deliver_inbound`, `deliver_event`, `invoke`,
  off-thread `Completion`, `TimerMatured`.
- `Scheduler::on_matured()` — when a timer would mature. The host's
  clock task SHOULD wake the engine when it knows a timer at
  `at_ns` should fire (so the engine doesn't have to poll the
  scheduler reactively).
- An `EngineStep::SendEnvelope` consumer (e.g. the transport
  adapter) finishing transmission and pushing back an ingress
  event (the typical request/response flow).

### 11.3 Self-wake on persistent work

If `poll()` returns `Poll::Ready(steps)` while the frontier or
ingress is non-empty (because the cycle hit the op budget), the
host's executor re-polls on its own — no waker needed.

---

## Part 12 — App interaction surface

### 12.1 What the host can do

```rust
impl Node {
    // — Construct —
    // Nodes are constructed exclusively via the install path:
    //   bb::install(peer_id, addresses, compiled_model, targets, Config::new())
    // `Node::new` is `pub(crate)` — not a user-facing constructor.

    // — Drive —
    pub fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Vec<EngineStep>>;

    // — Drive bootstrap —
    pub fn run_bootstrap(&mut self, target: BootstrapTarget<'_>)
        -> Result<Vec<EngineStep>, BootstrapError>;
    pub fn bootstrap_status(&self) -> BootstrapStatus;

    // — Push external work —
    pub fn deliver_inbound(&mut self, envelope: WireEnvelope) -> Result<(), DeliveryError>;
    pub fn deliver_event(&mut self, module: &str, input: &str, bytes: &[u8])
        -> Result<(), DeliveryError>;
    pub fn invoke(&mut self, module: &str, inputs: &[(&str, &[u8])])
        -> Result<ExecId, DeliveryError>;

    // — Snapshot/restore —
    pub fn snapshot(&self) -> NodeSnapshot;
    pub fn restore(&mut self, snapshot: NodeSnapshot) -> Result<(), RestoreError>;
    pub fn clear(&mut self);
    pub fn incarnation(&self) -> u64;

    // — Introspection (read-only) —
    pub fn loaded_modules(&self) -> Vec<&str>;
    pub fn linked_components(&self) -> Vec<&ComponentHandle>;
    pub fn peer_id(&self) -> PeerId;
    pub fn execution_state(&self, exec_id: ExecId) -> Option<&ExecutionState>;
    pub fn pending_async_count(&self) -> usize;
    pub fn ingress_handle(&self) -> Arc<IngressQueue>;  // for off-thread injectors
}
```

Modules are installed exclusively via `bb::install` (see
[AUTHORING_COMPONENTS.md §7](AUTHORING_COMPONENTS.md#7-the-install-path)).
The set of installed modules is fixed for the Node's lifetime;
`loaded_modules()` lets the host introspect what was registered, and
`linked_components()` lets it inspect every owned component by
`(TYPE_NAME, instance_id, package)`.

### 12.2 The `ingress_handle()` pattern

A transport adapter on a separate thread takes a clone of
`ingress_handle()` and pushes envelopes directly:

```rust
let ingress = node.ingress_handle();
std::thread::spawn(move || {
    while let Some(envelope) = transport.recv() {
        ingress.push(IngressEvent::Envelope(envelope));
    }
});
```

Same for clock tasks pushing `TimerMatured`, host-side data
producers pushing `AppEvent`, etc. The single `Arc<IngressQueue>`
is the framework's only thread-safe seam.

### 12.3 What the host observes via EngineStep

```rust
pub enum EngineStep {
    /// One Op fired synchronously + wrote its output sites.
    OpCompleted { op_ref: OpRef, exec_id: ExecId, sites_written: Vec<NodeSiteId> },

    /// One Op returned OpResult::Async; engine now tracks the cmd_id.
    AsyncSuspended { op_ref: OpRef, exec_id: ExecId, cmd_id: CommandId },

    /// Trigger-only fire — no value.
    OpTrigger { op_ref: OpRef, exec_id: ExecId },

    /// Explicit sink — Op deliberately produced no downstream effect.
    OpSink { op_ref: OpRef, exec_id: ExecId },

    /// Op failed.
    OpFailed { op_ref: OpRef, exec_id: ExecId, error: OpError },

    /// One outbound envelope to ship. `WireEnvelope.dest_peer_addresses`
    /// is the resolved ordered list `AddressBook::lookup(peer)`
    /// returned at dispatch time; the transport picks one entry.
    SendEnvelope(WireEnvelope),

    /// `wire::Send` could not resolve its destination peer's
    /// addresses (peer unknown to AddressBook, address list empty,
    /// or input didn't carry a parseable PeerId). No envelope
    /// shipped. Mirror of `InfraEvent::PeerResolveFailure` on the
    /// bus — same telemetry family as PeerBlocked/PeerDown/PeerUp.
    /// See ADDRESSING.md §Peer resolution failure.
    PeerResolveFailed {
        peer: Option<PeerId>,
        op_ref: OpRef,
        exec_id: ExecId,
    },

    /// An unconnected Module output emitted a value to the host.
    AppEvent {
        module_name: String,
        output_name: String,
        value: Vec<u8>,
    },

    /// One IncrMetric / Record syscall ran (host telemetry).
    Telemetry(TelemetryEvent),
}
```

Host responsibilities:
- `SendEnvelope(env)` → ship via transport (pick one entry from
  `env.dest_peer_addresses` based on networking capabilities).
- `PeerResolveFailed { peer, op_ref, exec_id }` → log + surface to
  app (the producer DAG executed but no bytes left the Node).
- `AppEvent { ... }` → consume in the host's logic (training loop,
  evaluation, UI updates).
- `OpFailed` → log + decide retry / shutdown.
- `Telemetry` → forward to metrics backend if desired.
- `OpCompleted` / `AsyncSuspended` / `OpTrigger` / `OpSink` →
  primarily for tracing / debug; production hosts ignore.

---

## Part 13 — Event emission

The framework distinguishes **three event-emission channels**:

### 13.1 The bus (in-cycle, intra-Node)

`runtime_ref.bus.publish(NodeEvent::...)` — published events route
to subscribed Components on the next `route_bus_events` phase.
Used for cross-Component signaling within a Node:

```rust
pub enum NodeEvent {
    Infra(InfraEvent),    // framework-emitted
    Wire(WireEvent),       // wire-component-emitted
    App(AppEvent),         // app-Op-emitted
}

pub enum InfraEvent {
    WireResponseLanded { cmd_id: CommandId },
    OpFailure { op_ref: OpRef, error: OpError },
    /// Routable telemetry mirror of EngineStep::PeerResolveFailed.
    /// Surfaces via the bus so subscribers can monitor peer-resolution
    /// failures alongside PeerBlocked / PeerDown / PeerUp.
    PeerResolveFailure { peer: Option<PeerId>, op_ref: OpRef },
}
```

Each `EventSource` syscall op declares its `kind` attribute at
DSL recording time. Node install resolves the op's output
value name to a `NodeSiteId` and calls
`engine.register_event_subscription(kind, site)`. Phase 3 writes a
`TriggerValue` to each subscribed site and pushes the site's
downstream consumers (uniform with the multiaddr wire-delivery
model per [ADDRESSING.md](ADDRESSING.md) — bus and wire fills land
identically). The bus is bounded (`bus_capacity` from NodeConfig);
overflow drops oldest + bumps the `bus.dropped` counter.

### 13.2 AppEvent (cross-boundary, Node → host)

Surfaces via `EngineStep::AppEvent { module_name, topic }`. Two
channels feed this variant; both are observable by the host on the
same `poll()` result:

**(a) The top-level Module's function signature.** When the
host registers a `ModelProto` with `Node::with_module`, that
module's `model.functions[0]` becomes the program. Its declared
`function.input` ports are the engine's ingress trigger sites; its
declared `function.output` ports are the egress event sites. When
a value lands at one of those output sites AND no downstream
consumer in the function reads it, the engine pushes an
`EngineStep::AppEvent { topic: <output port name> }` on the
host's poll result. This is the "function signature is the engine
I/O contract" path — the simple case for top-level Modules whose
result flow naturally ends at a top-level output.

**(b) Explicit `AppEmit` / `AppNotify` syscall ops.** Anywhere
in the graph — including inside deeply nested sub-Modules — the
user can place an `AppEmit` or `AppNotify` op that fires
mid-cycle, pushes to `framework.pending_app_events`, and is
drained as `EngineStep::AppEvent` in Phase 8. Use this for
intermittent reporting / multi-stage computations whose progress
the host wants to observe without waiting for a final output.

Both channels coexist by design and both produce the same
`EngineStep::AppEvent` variant. Pick the one that fits: simple
final-result patterns use the function signature; mid-cycle
emissions use explicit syscall ops.

### 13.3 Tracing spans (observability)

Every engine entry point + every Op invoke opens a `tracing` span.
Spans are correlated by `ExecId` so the host can see exactly which
execution each event belongs to.

```
node.poll
├─ engine.drain_frontier (exec_id=42)
│   ├─ invoke (op_ref=7, op_type=MatMul, slot=backend)
│   │   └─ backend.execute_subgraph (component_id=BurnBackend)
│   ├─ invoke (op_ref=8, op_type=Add, slot=backend)
│   └─ ...
├─ engine.route_bus_events
└─ engine.poll_timers
```

Filter mutation via `bytesandbrains::tracing_runtime::set_filter`
lets the host adjust verbosity at runtime without restart.

---

## Part 14 — Tracing + observability

### 14.1 Span hierarchy

- `node.poll` — root span per cycle. Fields: `cycle_seq`,
  `incarnation`.
- `engine.<phase>` — child per cycle phase
  (`drain_frontier`, `route_bus_events`, `poll_timers`,
  `handle_completion`, `drain_pending_completions`,
  `fire_lifecycle`).
- `invoke` — per Op invocation. Fields: `op_ref`, `exec_id`,
  `op_type`, `domain`, `concrete_type` (when present), `instance`.
- `deliver_inbound` — per inbound envelope. Fields:
  `dest_peer_addresses_count` (transport-side info only; engine
  doesn't route on it inbound), `fill_count`, `correlation`.
- `deliver_fill` — per `SlotFill` dispatch. Fields: `dest_suffix`,
  `kind` (`data` / `control`), `site_id` (when data plane),
  `component_ref` + `op_type` (when control plane).
- `deliver_event` — per host event push. Fields: `graph`, `module`,
  `input`.

### 14.2 ExecId correlation

Every span emits an `exec_id` field. Aggregators (OTel, Honeycomb,
etc.) group by `exec_id` to surface the full lifecycle of one
graph execution — useful for debugging stuck executions, latency
breakdowns, cross-Node correlation when wire ops cross peers
carrying the originating `exec_id` in `metadata_props`.

### 14.3 Counters + metrics

`IncrMetric(name)` syscall ops bump per-Node `AtomicU64` counters.
The framework auto-tracks:
- `frontier.depth_peak` — max frontier size per cycle.
- `ingress.dropped` — events dropped on overflow.
- `bus.dropped` — events dropped on overflow.
- `cycle.duration_ns` — wall-clock per cycle (host-visible via
  AppEvent if the host wires it).
- `pending_async.count` — current suspensions.

Hosts pull counters via `Node::counter(name) -> u64` or via the
emitted `EngineStep::Telemetry` stream.

### 14.4 Tracing-runtime filter mutation

```rust
bytesandbrains::tracing_runtime::install_default();
bytesandbrains::tracing_runtime::set_filter("bb::engine=debug,bb::wire=info")?;
let current = bytesandbrains::tracing_runtime::current_filter();
```

Backed by `tracing_subscriber::reload::Handle`; mutation is
process-global. The framework's FFI exposes
`bb_tracing_set_filter(directive, len)` for cross-language tracing
control.

---

## Part 15 — Snapshot / restore

### 15.1 What's snapshotted

```rust
pub struct NodeSnapshot {
    pub incarnation: u64,
    pub config: NodeConfig,
    pub graphs: Vec<NamedGraphSnapshot>,           // post-resolution ModelProto bytes
    pub components: Vec<NamedComponentSnapshot>,    // per-impl opaque bytes (all runtime impls — Backend, Model, Index, Protocol, …)
    pub atomic_dispatch: Vec<AtomicDispatchEntry>,
    pub routing_table: Vec<RouteEntry>,        // deprecated — always empty;
                                               // wire envelopes route by
                                               // multiaddr per ADDRESSING.md
    pub wire_types: Vec<&'static str>,
    pub transient: TransientSnapshot,
}

pub struct TransientSnapshot {
    pub frontier: Vec<(OpRef, ExecId)>,
    pub slot_table: HashMap<(NodeSiteId, ExecId), Option<Vec<u8>>>,
    pub pending_async: HashMap<CommandId, PendingAsyncSnapshot>,
    pub execution_state: HashMap<ExecId, ExecutionStateSnapshot>,
    pub framework: FrameworkSnapshot,
    pub bus: TypedBusSnapshot,
    pub ingress: Vec<IngressEvent>,                  // in-flight inbound work
    pub wire_states: HashMap<ComponentRef, WireStateSnapshot>,
    pub pending_completions: Vec<PendingCompletion>, // mid-cycle hooks' output
}

pub struct FrameworkSnapshot {
    pub counters:           HashMap<String, u64>,
    pub fired_phases:       Vec<String>,
    // Multiaddr-keyed peer registries per docs/ADDRESSING.md.
    // Address bytes use the canonical Address::to_bytes encoding.
    // PeerId stored as u64 — works for identity-coded test peers
    // via `PeerId::as_u64_test()`. Real production multihash
    // PeerIds collapse to 0; snapshot proto widening to `bytes`
    // is a v1.1 follow-up.
    pub address_book:       Vec<(u64, Vec<u8>)>,             // (PeerId, Address bytes)
    pub peer_address_book:  Vec<(u64, Vec<u8>, Vec<u8>)>,    // (PeerId, identity, Address bytes)
}

pub struct TypedBusSnapshot {
    // Bus subscriptions keyed by EventKind → NodeSiteIds. Restored
    // verbatim; Phase 3 writes a TriggerValue to each subscribed
    // site (uniform with wire delivery — see ADDRESSING.md).
    pub event_subscriptions: HashMap<String, Vec<u64>>,
}
```

The slot table snapshots every `(NodeSiteId, ExecId)` entry —
including in-flight executions. The frontier snapshots
in-cycle scheduling. The ingress snapshots not-yet-processed
external events. After restore + first poll, the Node resumes
exactly where it left off.

### 15.2 What's NOT snapshotted

- The `Waker` stash (purely runtime state).
- The dispatch table (rebuilt from registered concrete types +
  installed graphs).
- Tracing subscriber state.
- Atomic counter values (covered by component snapshots if a
  component owns them; framework counters are reset).
- `Engine::ingress_bytes_in_flight` (derived state — recomputed
  at restore from `SlotValue::charged_bytes` on each live
  slot-table entry; the counter resumes accounting as slots
  drain through normal overwrite / eviction).

### 15.3 Restore semantics

```rust
fn restore(&mut self, snapshot: NodeSnapshot) -> Result<(), RestoreError> {
    // 1. Validate snapshot is compatible (incarnation, wire_types
    //    cover loaded denotations, routing_table matches bound
    //    components).
    // 2. Reinstall graphs as resolved (skip slot-resolution —
    //    they're already resolved).
    // 3. Restore each component's state.
    // 4. Restore transient state (frontier, slot_table,
    //    pending_async, execution_state, ingress contents).
    // 5. Bump incarnation.
    // 6. The Node is operationally identical to the
    //    snapshotted state.
}
```

After restore, the first `poll()` cycle drains the restored ingress
(processing any in-flight inbound events that were waiting at
snapshot time) and continues normal execution.

---

## Part 16 — Backpressure + capacity

| Capacity | Default | Overflow behavior |
|---|---|---|
| `ingress.capacity` | `bus_capacity * 4` | Drop event + `IncrMetric("ingress.dropped")` |
| `bus.capacity` | 1024 | Drop oldest + `IncrMetric("bus.dropped")` |
| `outbound_queue.capacity` | unbounded | – |
| `frontier.capacity` | unbounded | – |
| `pending_async.capacity` | unbounded | – |
| `slot_table.capacity` | unbounded | GC'd on execution complete |
| `ingress_byte_budget` | 256 MiB / edge 8 MiB | Drop offending payload + `InfraEvent::WireReceiveError::BudgetExceeded` or `AppIngressError::BudgetExceeded` |
| `max_app_event_bytes` | 1 MiB / edge 64 KiB | Reject + `DeliveryError::OversizePayload` + `AppIngressError::PerItemCapExceeded` |
| `max_invoke_inputs` / `max_invoke_bytes` | 100 / 10 MiB; edge 16 / 256 KiB | Reject + matching `DeliveryError` + `AppIngressError` |
| `max_completion_result_bytes` | 4 MiB / edge 64 KiB | Drop result + `AppIngressError::PerItemCapExceeded`; parked op times out naturally |

`NodeConfig::ingress_byte_budget` + the per-source caps form a
two-level defence: the per-source cap rejects oversize single
payloads, the budget caps cumulative in-flight bytes. Boundary
callers charge before installing; slot-table writers release on
overwrite / eviction via `SlotValue::charged_bytes`
(`bb-runtime/src/engine/core.rs:540-552`). Hot-path cost is one
saturating-add + one comparison per fill admission, below the
prost decode that follows. Edge preset
(`NodeConfig::edge()` —
`bb-runtime/src/node/config.rs:226-235`) tightens every cap for
8 MiB-class devices.

Frontier + slot_table grow with concurrent execution count. The
host bounds execution concurrency via `Limit.Acquire/Release`
syscall ops at graph entry points. Without such limits, a
heavily-loaded Node may grow unboundedly — that's a graph design
issue, not a framework limitation.

---

## Part 17 — Performance notes

### 17.1 Data structure choices

- **Frontier**: `VecDeque<(OpRef, ExecId)>`. No locking needed
  (single-threaded). Capacity grows geometrically; reuse across
  cycles.
- **Slot table**: `HashMap<(NodeSiteId, ExecId), ...>` — for low
  concurrent-execution counts (≤ 32), a custom `SmallMap` outperforms
  `HashMap`; the framework's `SmallMap` switches at size 32.
- **Ingress**: `concurrent-queue::ConcurrentQueue<IngressEvent>` +
  `atomic-waker::AtomicWaker`. Lock-free, runtime-independent,
  `MIT OR Apache-2.0`. ~30 ns per push (one CAS + atomic store +
  waker swap). See §2.2.
- **Dispatch table**: `HashMap<TypeId, DispatchEntry>` with FxHash
  (TypeIds hash well already).
- **Consumers map**: `HashMap<NodeSiteId, SmallVec<[OpRef; 4]>>` —
  most sites have ≤ 4 consumers; `SmallVec` avoids heap allocation.

### 17.2 Allocation budget

- One `RuntimeResourceRef` per invoke — stack-allocated, no heap.
- `Box<dyn SlotValue>` per slot write — heap, but values are
  typically small (Trigger = zero-byte; primitives 8-byte;
  SlotValue = `Arc`-shared so cheap clone).
- `Vec<EngineStep>` per poll — heap; reuse across cycles via
  buffer pooling.

### 17.3 Hot path cost

Per Op invocation, no I/O, no async:
- Frontier pop: ~5 ns
- TypeId lookup: ~30 ns
- TypedInputMap build: ~50 ns per input port
- Invoke dispatch (vtable call): ~5 ns
- Output write: ~30 ns per site
- Consumer push: ~10 ns per ready consumer

Engine overhead ≈ 200 ns per Op (excluding the Op body's own work).
A million-Op cascade fits in 200 ms of pure engine overhead;
practical workloads spend the bulk of their time inside backends
and runtime impls, not the engine.

---

## Part 18 — Worked example: tracing a poll cycle end-to-end

A canonical request/response flow on a server Node:

```
0. external transport thread:
     receives bytes from socket
     decodes WireEnvelope (dest_peer_addresses=[/* picked by sender's
                            transport from the resolved list */],
                          correlation=Request(42),
                          fills=[SlotFill { dest_suffix=/site/<site_recv_value>,
                                            payload=<u64 bytes>,
                                            trigger_only=false }])
     pushes IngressEvent::EnvelopeFrom { src_peer, envelope } onto
       node.ingress_handle() (src_peer comes from the transport's
       connection identity; the framework keys peers by PeerId.
       The engine ignores dest_peer_addresses on inbound — it's
       solely the sender's transport-selection hint)
     ingress.waker.wake() fires
1. host executor re-polls the Node.
2. node.poll(cx):
   - Phase 1: drain_ingress → 1 event
     - process Envelope:
       - iterate env.fills (1 fill)
       - Address::from_bytes(fill.dest_suffix) → /site/<site_recv_value>
       - data-plane path: site_id resolved
       - allocate ExecId(100)
       - write BytesValue(<u64 bytes>) to slot
         (site_recv_value, 100)
       - push downstream consumers from installed_graph.consumers[site]
         (op_ref=6 = WireSendResp, all-inputs-ready check will fire it)
   - Phase 2: drain_frontier
     - pop (6, 100): WireSendRespOp
       - invoke: takes payload + req_id from inputs
       - resolves original sender's PeerId via the AddressBook —
         packs `dest_peer_addresses = AddressBook::lookup(peer)` (or
         emits PeerResolveFailed if the entry is gone)
       - constructs response WireEnvelope with
         dest_peer_addresses=[/* resolved list */],
         correlation=Response(42)
       - pushes onto outbound_queue
       - returns Sync(vec![])
       - emits EngineStep::OpCompleted { op=6, exec=100 }
   - Phase 3: drain_bus → no events.
   - Phase 4: poll_timers → no maturity.
   - Phase 5: drain_pending_completions → none.
   - Phase 6: final frontier → empty.
   - Phase 7: outbound_queue drains → 1 envelope →
     emits EngineStep::SendEnvelope(response_env).
   - Execution(100) reaches completion → GC slot_table entries.
   - any_work = true → Poll::Ready([
       OpCompleted { op=6, exec=100 },
       SendEnvelope(response_env),
     ])
3. host receives 2 steps.
   - 1 OpCompleted → tracing.
   - 1 SendEnvelope → hands to transport.
4. transport reads `response_env.dest_peer_addresses` (an ordered
   list of candidate transport endpoints) and ships to whichever one
   matches its networking capabilities (IPv4, QUIC, relay, etc.).
5. node.poll(cx) again → no work → Poll::Pending,
   waker stashed. Engine idles.
6. ... time passes ...
7. next inbound bytes from peer → restart at step 0.
```

A complete request/response cycle: 2 Ops fired, 1 envelope shipped,
~500 ns of engine overhead, plus whatever the Op bodies + transport
spent on real work. Concurrent executions of the same graph add
fresh `ExecId`s; the engine's slot table partitions by
`(NodeSiteId, ExecId)` keeps them isolated; cleanup is automatic on
execution completion.

---

That's the engine. Two queues separating concurrency models, per-
ExecId slot partitioning enabling concurrent graph instances, a
clear entry-point taxonomy, a single chained poll cycle that
processes everything in a predictable order, full app interaction
surface, and observability via per-ExecId trace correlation. The
host drives the engine through `poll(cx)`; everything else falls
out of the structure.
