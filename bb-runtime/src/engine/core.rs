//! The `Engine` struct + test-only constructor + registration
//! accessors. Hot-path poll cycle lives in `engine::poll`.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use crate::bus::TypedBus;
use crate::component::ErasedComponent;
use crate::engine::dispatch_entry::{FunctionKey, OpDispatch, StatelessInvokeFn};
use crate::engine::graph_slot::GraphSlot;
use crate::engine::invoke::{make_protocol_dispatcher, RoleDispatcher};
use crate::framework::FrameworkComponents;
use crate::ids::{CommandId, ComponentRef, ExecId, NodeSiteId, OpRef};
use crate::ingress::IngressQueue;
use crate::slot_value::SlotValue;
use bb_ir::proto::onnx::{FunctionProto, NodeProto};

/// Point-in-time hot-path counters for dashboards + saturation
/// detection. Not synchronized against an in-flight poll cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineStats {
    /// `(OpRef, ExecId)` pairs on the frontier. Climbing → poll
    /// loop falling behind ingress.
    pub frontier_len: usize,
    /// Events queued on the typed bus awaiting routing.
    pub bus_len: usize,
    /// Suspended async commands. Approaching
    /// `NodeConfig.max_pending_async` → cap pressure.
    pub pending_async: usize,
    /// `(NodeSiteId, ExecId)` entries holding a value.
    pub slot_table_occupied: usize,
    /// Approximate MPMC ingress depth.
    pub ingress_depth: usize,
    /// Envelopes queued for next outbound drain.
    pub outbound_queue_depth: usize,
    /// Event kinds with at least one subscriber.
    pub event_subscriptions: usize,
    /// Number of registered components.
    pub registered_components: usize,
    /// Number of installed graphs.
    pub graph_slots: usize,
}

/// Single-Node engine state. Built via `Node::ensure_ready`;
/// tests use `Engine::new()` and seed fields directly.
pub struct Engine {
    // --- Graph storage ---------------------------------------------
    /// Installed graphs in insertion order. `OpRef::pack` puts the
    /// slot index in the high 32 bits so `dispatch_for` resolves an
    /// op via two direct array accesses (no HashMap probe).
    pub(crate) graphs: Vec<GraphSlot>,

    /// Graph name → `graphs[]` index. Lookup table for name-keyed
    /// resolution (function-call targets, `Engine::graph(name)`
    /// accessors). The index doubles as the value packed into every
    /// `OpRef`.
    pub(crate) graph_index: HashMap<String, u32>,

    /// Sub-Module function registry by `(domain, name, overload)`.
    /// Populated by `Node` at install time.
    pub(crate) functions: HashMap<(String, String, String), FunctionProto>,

    // --- Dispatch --------------------------------------------------
    /// Unified syscall dispatch table — one lookup per op by
    /// `(domain, op_type)` to its stateless invoke fn pointer.
    /// Populated by `register_syscall` and `register_all_framework_syscalls`
    /// at Engine construction; `resolve_dispatch` reads this to stamp
    /// `OpDispatch::Stateless` for matching NodeProtos.
    pub syscall_table: HashMap<(String, String), StatelessInvokeFn>,

    // --- Component storage -----------------------------------------
    /// All bound runtime impls indexed by `ComponentRef.as_usize()`.
    /// The `Option<...>` wrapper lets `invoke_atomic` take the
    /// dispatching component out of the Vec via `mem::take` so a
    /// live [`crate::runtime::ComponentsView`] can borrow
    /// `&self.components` for cross-component reads while the
    /// dispatch closure runs - the dispatching slot is restored
    /// after the closure returns.
    pub(crate) components: Vec<Option<Box<dyn ErasedComponent>>>,

    /// The Node's own `PeerId` (the one passed to `Node::new`).
    /// Threaded into every `RuntimeResourceRef` so Components can
    /// identify themselves in outbound envelopes.
    pub self_peer: crate::ids::PeerId,

    /// Per-Node framework primitive bundle (scheduler, peer_gate,
    /// request_tracker, backoff_table, inbound_dedup, address_book).
    /// Exposed as `pub` so cross-crate test fixtures in `bb-ops` can
    /// construct `RuntimeResourceRef` from an Engine's framework
    /// primitives. Engine internals proper stay encapsulated; this is
    /// the dependency-injection boundary.
    pub framework: FrameworkComponents,

    /// The per-Node typed event bus.
    pub bus: TypedBus,

    // --- Execution state -------------------------------------------
    /// Per-poll execution-state bundle: frontier, slot table, per-
    /// execution liveness, parked async ops + in-cycle completions,
    /// function-call invocation frames, inbound-envelope context
    /// map, the timer scheduler, and the monotonic ID allocator.
    /// See [`crate::exec_state::ExecState`].
    pub exec: crate::exec_state::ExecState,

    /// Reverse index from a fused `binding_id` (e.g.
    /// `"BurnBackend#0"`) to the bound component's `ComponentRef`.
    /// Populated by `Node` at install time; read by dispatch
    /// resolution to bind NodeProtos that reference a backend by
    /// binding_id.
    pub(crate) binding_id_index: HashMap<String, ComponentRef>,

    /// Per-`event_kind` subscription map. Each entry names the
    /// `NodeSiteId`(s) of `EventSource` ops listening for that kind.
    /// The bus-routing pass writes a `TriggerValue` into each
    /// subscribed site at a fresh `ExecId` and pushes the site's
    /// downstream consumers — matching the wire delivery semantics
    /// per `docs/ADDRESSING.md` so bus + wire share one model.
    pub(crate) event_subscriptions: HashMap<String, Vec<NodeSiteId>>,

    /// Per-LifecyclePhase Op enrollments. `pub` for cross-crate
    /// `bb-ops` test fixtures that exercise the LifecyclePhase
    /// syscall.
    pub lifecycle_table: HashMap<String, Vec<OpRef>>,

    // --- Async + cross-thread --------------------------------------
    /// Thread-safe inbox for external events. Producers (transport
    /// adapters, host invocations) may push from any thread; the
    /// Engine drains serially from its own (single) thread. Engine
    /// holds an `Arc` so it can hand clones to producers without
    /// surrendering ownership.
    pub(crate) ingress: Arc<IngressQueue>,

    /// Lifecycle phases queued for firing by `Engine::fire_lifecycle`
    /// on the next poll.
    pub(crate) fired_phases: Vec<String>,

    /// Snapshot of the ingress queue depth at the start of the
    /// engine's ingress drain. Used by the backpressure detection
    /// hook in `process_ingress_event` to compare against the
    /// configured high-water mark. Refreshed every poll cycle.
    pub(crate) phase1_pre_drain_depth: usize,

    // --- Bootstrap ---------------------------------------------------
    /// Consolidated bootstrap state — every field the install path,
    /// poll seeder, and body-op gate read goes through here. See
    /// [`crate::engine::bootstrap::BootstrapState`] for field roles.
    pub(crate) bootstrap: crate::engine::bootstrap::BootstrapState,

    // --- Dispatcher registries (per-Engine) ------------------------
    /// Concrete-type `ProtocolRuntime` dispatchers registered against
    /// this Engine, indexed by `TypeId::of::<T>()`. Populated by
    /// [`Engine::register_protocol_dispatcher`] at setup; consulted by
    /// `invoke_atomic` via direct HashMap lookup (no linear scan).
    pub(crate) role_dispatchers: HashMap<std::any::TypeId, RoleDispatcher>,

    // --- Generic slot registry -------------------------------------
    /// Slot-name → `ComponentRef` registry. Generic over component
    /// role: indexes EVERY bound Component (backends, indexes,
    /// models, peer selectors, custom) by binding slot name
    /// (defaults to the field name; overridable via
    /// `#[bb::slot("custom")]`). Components reach declared
    /// dependencies through this map at dispatch time via
    /// [`crate::runtime::ComponentsView::for_slot`].
    ///
    /// This is the canonical lookup table. Every install path
    /// populates it; every dispatch path reads through it. No
    /// per-role specialization above the slot abstraction.
    pub(crate) slots: HashMap<String, ComponentRef>,

    /// Parallel index: compiler-assigned slot id (the value of
    /// `ai.bytesandbrains.slot_id` stamped on role NodeProtos by the
    /// placeholders pass) → `ComponentRef`. Populated alongside
    /// [`Self::slots`] at install time. `resolve_dispatch` reads
    /// the role NodeProto's `slot_id`, looks it up here, and stamps
    /// `OpDispatch::Atomic` against the resolved component. Single
    /// source of truth: install populates both indexes from the same
    /// `bb.binding.<target>.<slot>` metadata.
    pub(crate) slot_id_to_cref: HashMap<u32, ComponentRef>,

    /// Parallel index: compiler-assigned `slot_id` → `(role,
    /// ComponentRef)`. Populated by [`Self::bind_slot_id_with_role`]
    /// at install time from the same `binding.<target>.<slot>`
    /// metadata that drives `slot_id_to_cref`; retains the
    /// `ComponentRole` so `decode_typed_fill` can decide between the
    /// framework-carrier path and the backend-mediated tensor path.
    pub(crate) slot_id_to_role_ref: HashMap<u32, (crate::registry::ComponentRole, ComponentRef)>,

    // --- Component role introspection ------------------------------
    /// Per-component set of declared roles, sourced from
    /// `inventory::iter::<ComponentRoleBinding>` keyed by
    /// `T::TYPE_NAME`. Populated by `Node::ensure_ready` after
    /// component registration; reported by [`Engine::roles_for`] for
    /// introspection (engine tests, host tooling).
    pub(crate) component_roles:
        HashMap<ComponentRef, std::collections::HashSet<crate::registry::ComponentRole>>,

    // --- Production-safety caps ------------------------------------
    /// Soft per-poll-cycle op-invocation budget per
    /// `NodeConfig.cycle_op_budget`. When set, `Engine::poll` yields
    /// after this many invocations and surfaces
    /// `EngineStep::CycleBudgetExceeded { ops_invoked }` so the host
    /// can re-poll. `None` disables the budget.
    pub(crate) cycle_op_budget: Option<usize>,

    /// Cap on the number of in-flight `pending_async` entries per
    /// `NodeConfig.max_pending_async`. When at cap, an Op returning
    /// `DispatchResult::Async(_)` fails synchronously via the
    /// existing `OpFailed` path. `None` disables the cap.
    pub(crate) max_pending_async: Option<usize>,

    /// Cumulative cap on in-flight ingress bytes held across the
    /// ingress queue + slot table + pending async completion
    /// buffers. Sourced from `NodeConfig::ingress_byte_budget`.
    /// Boundary callers call [`Self::try_charge`] before installing
    /// a payload; the slot-table writer calls [`Self::release`] on
    /// overwrite / eviction.
    pub(crate) ingress_byte_budget: usize,

    /// Live count of ingress bytes the engine currently holds.
    /// Incremented by [`Self::try_charge`] on successful admission;
    /// decremented by [`Self::release`] on slot-table overwrite /
    /// eviction / drop. The budget guard surfaces this as a
    /// snapshot via [`Self::ingress_bytes_in_flight`].
    pub(crate) ingress_bytes_in_flight: usize,

    // --- Single-threaded anchor ------------------------------------
    /// `PhantomData<*const ()>` makes `Engine` neither `Send` nor
    /// `Sync` - the single-threaded sans-IO contract is enforced at
    /// compile time. Producers can still push to `ingress` from other
    /// threads because the `Arc<IngressQueue>` handle is independently
    /// `Send + Sync`.
    _not_send: PhantomData<*const ()>,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Construct an empty engine with the default ingress capacity.
    /// `Node::new` wraps this with `self_peer`, framework syscalls,
    /// and config caps applied. For non-default `bus_capacity` use
    /// [`Self::with_bus_capacity`] so the ingress queue sizes to
    /// `bus_capacity * 4` per ENGINE.md §2.2.
    pub fn new() -> Self {
        Self::with_bus_capacity(crate::node::DEFAULT_BUS_CAPACITY)
    }

    /// Construct an empty engine whose ingress queue holds up to
    /// `bus_capacity * 4` events (the ENGINE.md §2.2 ratio that
    /// reserves headroom for async completions, app events, and
    /// inbound envelopes between poll cycles).
    pub fn with_bus_capacity(bus_capacity: usize) -> Self {
        Self {
            graphs: Vec::new(),
            graph_index: HashMap::new(),
            functions: HashMap::new(),
            syscall_table: HashMap::new(),
            slot_id_to_cref: HashMap::new(),
            slot_id_to_role_ref: HashMap::new(),
            components: Vec::new(),
            self_peer: crate::ids::PeerId::from(0u64),
            framework: FrameworkComponents::new(),
            bus: TypedBus::new(),
            exec: crate::exec_state::ExecState::new(),
            binding_id_index: HashMap::new(),
            event_subscriptions: HashMap::new(),
            lifecycle_table: HashMap::new(),
            ingress: Arc::new(IngressQueue::with_capacity(bus_capacity.saturating_mul(4))),
            fired_phases: Vec::new(),
            phase1_pre_drain_depth: 0,
            bootstrap: crate::engine::bootstrap::BootstrapState::new(),
            role_dispatchers: HashMap::new(),
            slots: HashMap::new(),
            component_roles: HashMap::new(),
            cycle_op_budget: crate::node::DEFAULT_CYCLE_OP_BUDGET,
            max_pending_async: crate::node::DEFAULT_MAX_PENDING_ASYNC,
            ingress_byte_budget: crate::node::DEFAULT_INGRESS_BYTE_BUDGET,
            ingress_bytes_in_flight: 0,
            _not_send: PhantomData,
        }
    }

    /// wipe restorable engine state ahead of a
    /// `Node::restore` call, leaving the install-time-stamped
    /// surfaces (`graphs`, `functions`, `dispatch_table`,
    /// `atomic_dispatch`, `components`, `self_peer`,
    /// `syscall_index`, `role_dispatchers`, `binding_id_index`,
    /// `lifecycle_table`, `event_subscriptions`,
    /// `cycle_op_budget`, `max_pending_async`) intact. The Node
    /// re-applies the snapshot's framework state, ID counters, and
    /// pending async/completion queues on top of the cleared
    /// state, so the post-restore Engine is the same install
    /// re-seeded with the snapshot's restorable transient state.
    ///
    /// Restorable surfaces explicitly cleared:
    /// - `frontier`, `slot_table`, `execution_state`,
    ///   `pending_async`, `pending_completions`, `pending_calls`,
    ///   `fired_phases`
    /// - `framework` (FrameworkComponents reseeds from snapshot)
    /// - `bus` (re-establishes subscriptions from snapshot)
    /// - `ingress` queue (fresh; in-flight inbound is the host's
    ///   responsibility to redeliver)
    pub fn clear_for_restore(&mut self) {
        self.exec.frontier.clear();
        self.exec.slot_table.clear();
        self.exec.execution_state.clear();
        self.exec.pending_async.clear();
        self.exec.pending_completions.clear();
        self.exec.pending_calls.clear();
        self.fired_phases.clear();
        // Slot-table clear above dropped every charged carrier;
        // reset the counter so the restored snapshot doesn't inherit
        // an in-flight balance the new state doesn't own.
        self.ingress_bytes_in_flight = 0;
        // Restore deliberately suppresses bootstrap re-runs: the
        // restored Node already executed its bootstrap call before
        // the snapshot, and replaying would re-seed the address
        // book, re-fire the first Announce, etc.
        // `install_order` + `module_bootstraps` stay populated for
        // introspection (multi-target installs surface every queued
        // target via [`Self::bootstrap_function_keys`]); `pending`,
        // `current_exec_id`, and `next_idx` reset so
        // `Node::run_bootstrap` is a no-op on a restored Node —
        // bumping the index to the end of `install_order` keeps the
        // seeder from re-firing if the host nonetheless polls.
        self.bootstrap.clear_for_restore();
        self.framework = FrameworkComponents::new();
        self.bus = TypedBus::new();
        self.ingress = Arc::new(IngressQueue::new());
        // ID counters reset to 0; the restore path re-applies the
        // snapshot's persisted values so post-restore IDs continue
        // from where the pre-snapshot Node left off ().
        self.exec.ids.next_exec_id = 0;
        self.exec.ids.next_command_id = 0;
    }

    /// Mint a fresh `ExecId`. Replaces the prior static counter
    /// in `src/ids.rs` so allocation runs single-threaded under
    /// the engine's borrow discipline.
    pub fn allocate_exec_id(&mut self) -> ExecId {
        let id = self.exec.ids.next_exec_id;
        self.exec.ids.next_exec_id = self
            .exec
            .ids
            .next_exec_id
            .checked_add(1)
            .expect("ExecId counter overflow");
        ExecId::from(id)
    }

    /// Mint a fresh `CommandId`. Used by async-suspending syscall
    /// ops via `RuntimeResourceRef::next_command_id`.
    pub fn allocate_command_id(&mut self) -> CommandId {
        let id = self.exec.ids.next_command_id;
        self.exec.ids.next_command_id = self
            .exec
            .ids
            .next_command_id
            .checked_add(1)
            .expect("CommandId counter overflow");
        CommandId::from(id)
    }

    /// Mint a fresh `NodeSiteId`. Used by graph installation; sites
    /// must be globally unique across installed graphs.
    pub fn allocate_node_site_id(&mut self) -> NodeSiteId {
        let id = self.exec.ids.next_node_site_id;
        self.exec.ids.next_node_site_id = self
            .exec
            .ids
            .next_node_site_id
            .checked_add(1)
            .expect("NodeSiteId counter overflow");
        NodeSiteId::from(id)
    }

    /// Drop slot_table and execution_state entries belonging to
    /// executions that have completed. An execution is complete
    /// when it has no frontier entries, no pending_async entries,
    /// and no pending_calls entry pointing at its `ExecId`. Called
    /// at the end of every `poll()` cycle so a long-running Node
    /// keeps a bounded slot_table.
    pub(crate) fn gc_completed_executions(&mut self) {
        if self.exec.execution_state.is_empty() {
            return;
        }
        let mut live: std::collections::HashSet<ExecId> =
            std::collections::HashSet::with_capacity(self.exec.execution_state.len());
        for (_, exec_id) in &self.exec.frontier {
            live.insert(*exec_id);
        }
        for p in self.exec.pending_async.values() {
            live.insert(p.exec_id);
        }
        for exec_id in self.exec.pending_calls.keys() {
            live.insert(*exec_id);
        }
        let dead: Vec<ExecId> = self
            .exec
            .execution_state
            .keys()
            .copied()
            .filter(|e| !live.contains(e))
            .collect();
        if dead.is_empty() {
            return;
        }
        let dead_set: std::collections::HashSet<ExecId> = dead.iter().copied().collect();
        // Walk doomed slot entries once to release any charged
        // ingress bytes the slot-table writer admitted, then drop
        // the entries. `retain` would let us mutate-in-place, but
        // it borrows the table mutably for the entire walk; the
        // explicit collect-then-remove pattern lets us drain
        // `charged_bytes()` from each prior carrier first.
        let doomed_keys: Vec<(NodeSiteId, ExecId)> = self
            .exec
            .slot_table
            .iter()
            .filter_map(|(key, _)| dead_set.contains(&key.1).then_some(*key))
            .collect();
        for key in doomed_keys {
            // `clear_slot` releases the prior carrier's
            // `charged_bytes()` against `ingress_bytes_in_flight`.
            // Non-ingress carriers report 0 — release is a no-op.
            let _ = self.clear_slot(key.0, key.1);
        }
        for exec_id in &dead {
            self.exec.execution_state.remove(exec_id);
        }
    }

    /// Apply production-safety caps from a `NodeConfig`. Called by
    /// `Node::ensure_ready` after constructing the Engine; tests can
    /// invoke directly to exercise specific cap values.
    pub fn apply_config_caps(&mut self, config: &crate::node::NodeConfig) {
        self.cycle_op_budget = config.cycle_op_budget;
        self.max_pending_async = config.max_pending_async;
        self.ingress_byte_budget = config.ingress_byte_budget;
        self.framework
            .outbound_queue
            .set_cap(config.max_outbound_queue);
        self.bus.set_cap(Some(config.bus_capacity));
        // The off-thread `CompletionSink::complete` path consults the
        // ingress queue itself for its per-item cap; reseed the
        // atomic so sinks created before this call see the configured
        // value on their next push.
        self.ingress
            .set_completion_result_cap(config.max_completion_result_bytes);
        // Reseed the BackpressureTracker with the configured knobs.
        // `apply_config_caps` is the canonical entry the host calls
        // before the first poll, so a fresh tracker reflecting the
        // resolved knobs is the only state observers see.
        self.framework.peer_state.backpressure = crate::framework::BackpressureTracker::with_config(
            config.backpressure_high_water_pct,
            config.backpressure_k_before_silent,
            config.backpressure_min_notice_interval_ns,
        );
    }

    /// Live count of ingress bytes the engine currently holds across
    /// its ingress queue + slot table + pending async completion
    /// buffers. Updated by every successful charge / release pair on
    /// the ingress paths. Surfaced for observability (operator
    /// dashboards) and assertions.
    pub fn ingress_bytes_in_flight(&self) -> usize {
        self.ingress_bytes_in_flight
    }

    /// Configured cap on cumulative in-flight ingress bytes,
    /// sourced from `NodeConfig::ingress_byte_budget`. Constant
    /// between `apply_config_caps` calls.
    pub fn ingress_byte_budget(&self) -> usize {
        self.ingress_byte_budget
    }

    /// Pre-admission budget guard for an ingress payload of length
    /// `bytes`. On success the bytes are added to
    /// `ingress_bytes_in_flight` and the caller may install the
    /// resulting carrier into the slot table or pending-completion
    /// queue. On overflow the counter is left unchanged and the
    /// caller drops the payload, emitting the appropriate
    /// `BudgetExceeded` `InfraEvent`.
    ///
    /// One saturating-add + one comparison; below the cost of the
    /// prost decode that typically follows.
    pub(crate) fn try_charge(&mut self, bytes: usize) -> Result<(), BudgetExceededReason> {
        let after = self.ingress_bytes_in_flight.saturating_add(bytes);
        if after > self.ingress_byte_budget {
            return Err(BudgetExceededReason {
                byte_count: bytes,
                budget_remaining: self
                    .ingress_byte_budget
                    .saturating_sub(self.ingress_bytes_in_flight),
            });
        }
        self.ingress_bytes_in_flight = after;
        Ok(())
    }

    /// Decrement `ingress_bytes_in_flight` after a charged payload
    /// leaves engine state — slot-table overwrite, slot clear,
    /// eviction, or in-cycle drop. `saturating_sub` defends against
    /// a release path that arrives without a paired charge (e.g. a
    /// snapshot replay reseeding the slot table before any wire
    /// traffic).
    pub(crate) fn release(&mut self, bytes: usize) {
        self.ingress_bytes_in_flight = self.ingress_bytes_in_flight.saturating_sub(bytes);
    }

    /// Write a value into the slot table at `(site, exec_id)`,
    /// releasing the prior occupant's `charged_bytes` against
    /// `ingress_bytes_in_flight`. The incoming carrier's
    /// `charged_bytes` is NOT re-added — admission callers
    /// (`decode_typed_fill`, `deliver_event`, etc.) have already run
    /// `try_charge` against the wire-byte budget. This helper is the
    /// slot-table-side bookkeeping that closes the loop on overwrite.
    ///
    /// Returns the prior boxed value (if any) so the caller can
    /// run additional teardown.
    pub(crate) fn slot_write(
        &mut self,
        site: NodeSiteId,
        exec_id: ExecId,
        value: Box<dyn SlotValue>,
    ) -> Option<Box<dyn SlotValue>> {
        let key = (site, exec_id);
        let prior = self.exec.slot_table.insert(key, Some(value));
        // `prior` is `Option<Option<Box<dyn SlotValue>>>`: outer None
        // means the slot was untouched; inner None means the slot
        // existed but was empty (cleared previously). Release only
        // when the prior carrier was alive.
        match prior {
            Some(Some(prior_box)) => {
                self.ingress_bytes_in_flight = self
                    .ingress_bytes_in_flight
                    .saturating_sub(prior_box.charged_bytes());
                Some(prior_box)
            }
            _ => None,
        }
    }

    /// Remove the slot at `(site, exec_id)`, releasing any
    /// `charged_bytes` the prior carrier was holding against
    /// `ingress_bytes_in_flight`. Returns the removed carrier so
    /// downstream paths (graph reset, GC) can pass it onward.
    pub(crate) fn clear_slot(
        &mut self,
        site: NodeSiteId,
        exec_id: ExecId,
    ) -> Option<Box<dyn SlotValue>> {
        let key = (site, exec_id);
        match self.exec.slot_table.remove(&key) {
            Some(Some(prior_box)) => {
                self.ingress_bytes_in_flight = self
                    .ingress_bytes_in_flight
                    .saturating_sub(prior_box.charged_bytes());
                Some(prior_box)
            }
            _ => None,
        }
    }

    /// Snapshot of the engine's hot-path state, sized for cheap
    /// reads on every poll cycle (no allocation). Production
    /// observability: operators see saturation building up before
    /// the process locks up.
    pub fn engine_stats(&self) -> EngineStats {
        EngineStats {
            frontier_len: self.exec.frontier.len(),
            bus_len: self.bus.len(),
            pending_async: self.exec.pending_async.len(),
            slot_table_occupied: self.exec.slot_table.len(),
            ingress_depth: self.ingress.len(),
            outbound_queue_depth: self.framework.outbound_queue.len(),
            event_subscriptions: self.event_subscriptions.len(),
            registered_components: self.components.len(),
            graph_slots: self.graphs.len(),
        }
    }

    /// Register a `ProtocolRuntime` dispatcher for the concrete
    /// component type `T`. Call once per `T` after constructing the
    /// Engine - typically alongside `register_component` for any
    /// component whose `dispatch_atomic` you want to drive. Indexed
    /// by `TypeId::of::<T>()` so dispatch is one HashMap lookup,
    /// not a linear scan across the registry.
    pub fn register_protocol_dispatcher<T: crate::roles::ProtocolRuntime + 'static>(&mut self)
    where
        T::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers
            .insert(type_id, make_protocol_dispatcher::<T>());
    }

    /// Register a role dispatcher keyed by `TypeId::of::<T>()` for a
    /// concrete `IndexRuntime` impl. Lets `Node::with_index(&value)`
    /// wire atomic dispatch even when `T` does not implement
    /// `ProtocolRuntime`. Calling this twice for the same `T`
    /// silently overwrites; the dispatcher is idempotent because
    /// `T::dispatch_atomic` is the only consumer.
    pub fn register_index_dispatcher<T: crate::roles::IndexRuntime + 'static>(&mut self)
    where
        <T as crate::roles::IndexRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers
            .insert(type_id, crate::engine::invoke::make_index_dispatcher::<T>());
    }

    /// Register an `AggregatorRuntime` dispatcher. See
    /// [`Engine::register_index_dispatcher`] for the rationale.
    pub fn register_aggregator_dispatcher<T: crate::roles::AggregatorRuntime + 'static>(&mut self)
    where
        <T as crate::roles::AggregatorRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers.insert(
            type_id,
            crate::engine::invoke::make_aggregator_dispatcher::<T>(),
        );
    }

    /// Register a `ModelRuntime` dispatcher. See
    /// [`Engine::register_index_dispatcher`] for the rationale.
    pub fn register_model_dispatcher<T: crate::roles::ModelRuntime + 'static>(&mut self)
    where
        <T as crate::roles::ModelRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers
            .insert(type_id, crate::engine::invoke::make_model_dispatcher::<T>());
    }

    /// Register a `CodecRuntime` dispatcher. See
    /// [`Engine::register_index_dispatcher`] for the rationale.
    pub fn register_codec_dispatcher<T: crate::roles::CodecRuntime + 'static>(&mut self)
    where
        <T as crate::roles::CodecRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers
            .insert(type_id, crate::engine::invoke::make_codec_dispatcher::<T>());
    }

    /// Register a `DataSourceRuntime` dispatcher. See
    /// [`Engine::register_index_dispatcher`] for the rationale.
    pub fn register_data_source_dispatcher<T: crate::roles::DataSourceRuntime + 'static>(&mut self)
    where
        <T as crate::roles::DataSourceRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers.insert(
            type_id,
            crate::engine::invoke::make_data_source_dispatcher::<T>(),
        );
    }

    /// Register a `PeerSelectorRuntime` dispatcher. See
    /// [`Engine::register_index_dispatcher`] for the rationale.
    pub fn register_peer_selector_dispatcher<T: crate::roles::PeerSelectorRuntime + 'static>(
        &mut self,
    ) where
        <T as crate::roles::PeerSelectorRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers.insert(
            type_id,
            crate::engine::invoke::make_peer_selector_dispatcher::<T>(),
        );
    }

    /// Register a `BackendRuntime` dispatcher. The `Backend` Contract
    /// trait's per-atomic-op surface dispatches through this entry,
    /// emitted from `#[derive(bb::Backend)]`.
    pub fn register_backend_dispatcher<T: crate::roles::BackendRuntime + 'static>(&mut self)
    where
        <T as crate::roles::BackendRuntime>::Error: std::fmt::Display,
    {
        let type_id = std::any::TypeId::of::<T>();
        self.role_dispatchers.insert(
            type_id,
            crate::engine::invoke::make_backend_dispatcher::<T>(),
        );
    }

    /// Record the inventory-declared roles for a registered
    /// component. `Node::ensure_ready` calls this once per
    /// `ComponentRef` after registration, passing the set computed
    /// from `crate::registry::roles_for_component(T::TYPE_NAME)`.
    pub fn set_component_roles(
        &mut self,
        cref: ComponentRef,
        roles: std::collections::HashSet<crate::registry::ComponentRole>,
    ) {
        self.component_roles.insert(cref, roles);
    }

    /// Return the inventory-declared roles for a registered
    /// component, or an empty set if the component wasn't registered
    /// through a derive emitting `ComponentRoleBinding` entries.
    /// Introspection surface for engine tests + host tooling.
    pub fn roles_for(
        &self,
        cref: ComponentRef,
    ) -> std::collections::HashSet<crate::registry::ComponentRole> {
        self.component_roles.get(&cref).cloned().unwrap_or_default()
    }

    /// Register a `binding_id → ComponentRef` mapping. Called by
    /// `Node::ensure_ready` after binding resolution so the bound-
    /// backend lookup can resolve a NodeProto's `binding_id`
    /// metadata against an installed component.
    pub fn register_binding_id(&mut self, binding_id: String, cref: ComponentRef) {
        self.binding_id_index.insert(binding_id, cref);
    }

    /// Look up the `ComponentRef` bound at the given slot name -
    /// the GENERIC dependency-resolution accessor. Components reach
    /// declared dependencies through this method (typically via
    /// [`crate::runtime::ComponentsView::for_slot`] at dispatch
    /// time). Returns `None` when no slot of that name is bound.
    pub fn slot(&self, slot: &str) -> Option<ComponentRef> {
        self.slots.get(slot).copied()
    }

    /// Bind a `ComponentRef` at the given slot name. The
    /// `install` facade in the `bytesandbrains` crate calls this
    /// from the T8 chain; in-crate callers use it from the
    /// transitional `Node::with_backend(slot, &b)` path. Returns
    /// the previous binding if any.
    pub fn bind_slot(&mut self, slot: String, cref: ComponentRef) -> Option<ComponentRef> {
        self.slots.insert(slot, cref)
    }

    /// Register the compiler-assigned `slot_id` → `ComponentRef`
    /// mapping. Called by `bb::install()` alongside
    /// [`Self::bind_slot`]; the pair is read by
    /// [`Self::resolve_dispatch`] when stamping
    /// `OpDispatch::Atomic` against a role NodeProto's
    /// `ai.bytesandbrains.slot_id` metadata. Returns the previous
    /// binding, if any.
    pub fn bind_slot_id(&mut self, slot_id: u32, cref: ComponentRef) -> Option<ComponentRef> {
        self.slot_id_to_cref.insert(slot_id, cref)
    }

    /// Register the compiler-assigned `slot_id` → `(role,
    /// ComponentRef)` mapping. Called by `bb::install()` alongside
    /// [`Self::bind_slot_id`]; the role is required so
    /// `decode_typed_fill` can branch between framework-carrier
    /// decode (`Codec`, `Index`, …) and backend-mediated tensor
    /// materialisation (`Backend`). Returns the previous binding, if
    /// any.
    pub fn bind_slot_id_with_role(
        &mut self,
        slot_id: u32,
        role: crate::registry::ComponentRole,
        cref: ComponentRef,
    ) -> Option<(crate::registry::ComponentRole, ComponentRef)> {
        self.slot_id_to_role_ref.insert(slot_id, (role, cref))
    }

    /// Look up the `(role, ComponentRef)` bound at a compiler-assigned
    /// `slot_id`. Used by `decode_typed_fill` to discover whether an
    /// inbound wire payload routes through a backend.
    pub fn role_ref_for_slot_id(
        &self,
        slot_id: u32,
    ) -> Option<(crate::registry::ComponentRole, ComponentRef)> {
        self.slot_id_to_role_ref.get(&slot_id).copied()
    }

    /// Iterate every `(slot_name, ComponentRef)` pair currently
    /// bound. Surfaces the registry to introspection tools + the
    /// compiler's resolve-dependencies pass.
    pub fn slots_iter(&self) -> impl Iterator<Item = (&str, ComponentRef)> {
        self.slots.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// Subscribe a `NodeSiteId` to bus events of `event_kind` (the
    /// discriminator returned by [`crate::bus::NodeEvent::kind`]).
    /// The bus-routing pass writes a `TriggerValue` to each
    /// subscribed site at a fresh `ExecId` and pushes the site's
    /// downstream consumers onto the frontier — uniform with wire
    /// delivery semantics per `docs/ADDRESSING.md`.
    ///
    /// `Node` calls this at install time for every
    /// `EventSource` syscall op, passing the op's output `NodeSiteId`.
    pub fn register_event_subscription(&mut self, event_kind: &str, site: NodeSiteId) {
        let entry = self
            .event_subscriptions
            .entry(event_kind.to_string())
            .or_default();
        if !entry.contains(&site) {
            entry.push(site);
        }
    }

    /// Cheap clone of the shared `IngressQueue` handle. Test
    /// harnesses + transport adapters push `IngressEvent`s through
    /// this surface.
    pub fn ingress_queue_handle(&self) -> Arc<IngressQueue> {
        Arc::clone(&self.ingress)
    }

    /// Queue a lifecycle phase for firing on the next `poll()`
    /// call. The framework emits `EngineStep::LifecycleFired { phase }`
    /// for each queued phase and also pushes every `LifecyclePhase`
    /// op enrolled under that phase name (via
    /// [`Engine::register_lifecycle_op`]) onto the frontier with a
    /// fresh `ExecId`.
    pub fn fire_lifecycle(&mut self, phase: &str) {
        self.fired_phases.push(phase.to_string());
    }

    /// Enroll `op_ref` under `phase` per IR_AND_DSL.md §5a.
    /// Idempotent - the same `(phase, OpRef)` pair never enrolls
    /// twice. `Node` calls this at install time after parsing
    /// each `LifecyclePhase` NodeProto's `phase` attribute.
    pub fn register_lifecycle_op(&mut self, phase: &str, op_ref: OpRef) {
        let entry = self.lifecycle_table.entry(phase.to_string()).or_default();
        if !entry.contains(&op_ref) {
            entry.push(op_ref);
        }
    }

    /// Register a stateless syscall op. Captures `TypeId::of::<T>()`
    /// into both `dispatch_table` (TypeId → invoke fn) and
    /// `syscall_index` ((domain, op_type) → TypeId) so
    /// `resolve_dispatch` can stamp `OpDispatch::Stateless`.
    ///
    /// Register a stateless syscall by its `(domain, op_type)` key.
    pub fn register_syscall(&mut self, domain: &str, op_type: &str, invoke_fn: StatelessInvokeFn) {
        self.syscall_table
            .insert((domain.to_string(), op_type.to_string()), invoke_fn);
    }

    /// Install every framework syscall shipped via inventory by
    /// `bb-ops`. Each registration carries its own
    /// `(domain, op_type)` + invoke pointer; the engine stamps them
    /// into `syscall_table`.
    pub fn register_all_framework_syscalls(&mut self) {
        for reg in crate::registry::framework_syscalls() {
            self.syscall_table.insert(
                (reg.domain.to_string(), reg.op_type.to_string()),
                reg.invoke,
            );
        }
    }

    /// Test-only installer. Inserts a fresh `GraphSlot` keyed by
    /// `name` with empty per-node tables but with `op_dispatch`
    /// pre-filled with `Unresolved` so subsequent `resolve_dispatch`
    /// can stamp dispatch decisions. Use [`Engine::install_graph`]
    /// for the canonical path that walks the FunctionProto.
    #[cfg(any(test, feature = "test-components"))]
    pub fn install_graph_for_test(
        &mut self,
        name: String,
        function: FunctionProto,
    ) -> &mut GraphSlot {
        let mut g = GraphSlot::new_for_test(name.clone(), function);
        g.op_dispatch = (0..g.function.node.len())
            .map(|_| crate::engine::dispatch_entry::OpDispatch::Unresolved)
            .collect();
        let idx = self.push_graph_slot(name, g);
        &mut self.graphs[idx as usize]
    }

    /// Push a `GraphSlot` onto the storage Vec and register its
    /// name → index entry. Returns the assigned `graph_idx` (the
    /// value that gets packed into `OpRef`). Same name twice
    /// overwrites the existing slot but keeps the original
    /// `graph_idx` — preserving OpRef stability across re-install.
    pub(crate) fn push_graph_slot(&mut self, name: String, slot: GraphSlot) -> u32 {
        if let Some(&idx) = self.graph_index.get(&name) {
            self.graphs[idx as usize] = slot;
            return idx;
        }
        let idx = self.graphs.len() as u32;
        self.graph_index.insert(name, idx);
        self.graphs.push(slot);
        idx
    }

    /// Resolve a graph by name. Returns `None` when the name
    /// isn't registered. Equivalent to `self.graphs.get(name)` on
    /// the prior HashMap-keyed shape.
    pub fn graph(&self, name: &str) -> Option<&GraphSlot> {
        let idx = *self.graph_index.get(name)?;
        self.graphs.get(idx as usize)
    }

    /// Resolve a graph by name for mutation. `None` when the name
    /// isn't registered.
    pub fn graph_mut(&mut self, name: &str) -> Option<&mut GraphSlot> {
        let idx = *self.graph_index.get(name)?;
        self.graphs.get_mut(idx as usize)
    }

    /// `true` when a graph with this name is installed.
    pub fn has_graph(&self, name: &str) -> bool {
        self.graph_index.contains_key(name)
    }

    /// Resolve a graph's positional index by name. Used by paths
    /// that need to compute `OpRef::pack(idx, node_idx)` from a
    /// graph name (function-call site resolution, etc.).
    pub fn graph_idx(&self, name: &str) -> Option<u32> {
        self.graph_index.get(name).copied()
    }

    /// Build an `OpRef` for the `node_idx`-th NodeProto of a graph
    /// identified by name. Test-only convenience for tests that
    /// used to fish the OpRef out of `GraphSlot.op_index`; with
    /// positional `OpRef::pack(graph_idx, node_idx)` the lookup is
    /// trivial.
    #[cfg(any(test, feature = "test-components"))]
    pub fn op_ref_at(&self, graph_name: &str, node_idx: u32) -> Option<OpRef> {
        let gi = self.graph_idx(graph_name)?;
        let g = self.graphs.get(gi as usize)?;
        if (node_idx as usize) < g.function.node.len() {
            Some(OpRef::pack(gi, node_idx))
        } else {
            None
        }
    }

    /// Iterate every installed `GraphSlot` in install order.
    pub fn graphs_iter(&self) -> impl Iterator<Item = &GraphSlot> {
        self.graphs.iter()
    }

    /// Iterate every (`name`, `&GraphSlot`) pair in install order.
    pub fn graphs_named(&self) -> impl Iterator<Item = (&str, &GraphSlot)> {
        // graph_index maps name -> idx; rebuild idx -> name for the
        // walk so the iteration order matches the storage Vec.
        let mut by_idx: Vec<(u32, &str)> = self
            .graph_index
            .iter()
            .map(|(n, i)| (*i, n.as_str()))
            .collect();
        by_idx.sort_by_key(|&(i, _)| i);
        by_idx
            .into_iter()
            .filter_map(move |(i, n)| self.graphs.get(i as usize).map(|g| (n, g)))
    }

    /// Canonical install path: builds an
    /// [`GraphSlot`] from the FunctionProto (allocating
    /// `OpRef`s + `NodeSiteId`s for every node + produced value) and
    /// inserts it under `name`.
    ///
    /// Used by [`crate::node::Node::ready`] for each
    /// `ModelProto.functions[0]`. Returns a mutable reference
    /// for any subsequent setup (slot_bindings, local_event_subs).
    pub fn install_graph(&mut self, name: String, function: FunctionProto) -> &mut GraphSlot {
        let graph_idx = self.graphs.len() as u32;
        let mut g = GraphSlot::from_function(
            name.clone(),
            function,
            graph_idx,
            &mut self.exec.ids.next_node_site_id,
        );
        // Entry-point graphs (installed via `install_graph`, not
        // `install_function_library`) get a `NodeSiteId` registered
        // for every function input so `Engine::deliver_app_event`
        // can seed the input via ingress. Body functions used in
        // `OpDispatch::FunctionCall` deliberately route through
        // `input_aliases` and must NOT get input sites; that path
        // installs through `install_function_library` instead.
        register_function_input_sites(&mut g, &mut self.exec.ids.next_node_site_id, graph_idx);
        let idx = self.push_graph_slot(name, g);
        &mut self.graphs[idx as usize]
    }

    /// Runtime-linker install: walk `model.functions[]` and install
    /// each FunctionProto as an `GraphSlot` keyed by its
    /// canonical `(domain, name, overload)`-derived string. Also
    /// populates the symbol-table index `functions` keyed on the same
    /// tuple, so call NodeProtos can be resolved at dispatch time.
    ///
    /// `entry_point_keys` lists the `FunctionKey`s for the registered
    /// Modules' main partition functions - those graphs get
    /// `is_entry_point = true` (their top-level outputs surface as
    /// `EngineStep::AppEvent`; sub-function bodies do not).
    ///
    /// A function stamped `MODULE_PHASE_KEY = "bootstrap"` registers
    /// its `FunctionKey` with the engine's bootstrap state (appends
    /// to `install_order`, populates `module_bootstraps`) without
    /// arming `pending` — install is pure. The host arms the queue by
    /// calling [`crate::node::Node::run_bootstrap`], which fans out
    /// each install-order target serially and emits one
    /// `BootstrapComplete` step per drained phase; multi-target
    /// installs surface their targets in slice order without further
    /// host action.
    ///
    /// Idempotent under ODR (same key + same body) - silently skips
    /// reinstall. Caller (Node linker) is responsible for the
    /// byte-equality check before calling.
    pub fn install_function_library(
        &mut self,
        functions: &[FunctionProto],
        entry_point_keys: &[FunctionKey],
    ) {
        let entry_set: std::collections::HashSet<&FunctionKey> = entry_point_keys.iter().collect();
        for f in functions {
            let key: FunctionKey = (f.domain.clone(), f.name.clone(), f.overload.clone());
            let graph_name = graph_name_for(&key);
            if self.has_graph(&graph_name) {
                continue;
            }
            let graph_idx = self.graphs.len() as u32;
            let mut g = GraphSlot::from_function(
                graph_name.clone(),
                f.clone(),
                graph_idx,
                &mut self.exec.ids.next_node_site_id,
            );
            // Only entry-point functions surface their outputs as
            // AppEvents. Sub-function bodies' outputs are forwarded
            // via output_forwarding at call sites.
            g.is_entry_point = entry_set.contains(&key);
            if !g.is_entry_point {
                g.top_level_outputs.clear();
            }
            let is_bootstrap = bb_ir::keys::read_function_module_phase(f)
                .is_some_and(|p| p == bb_ir::keys::MODULE_PHASE_BOOTSTRAP);
            // Bootstrap functions seed their inputs through the
            // host-driven staging path
            // (`Node::run_bootstrap(&[BootstrapInput])`) rather than
            // via a FunctionCall splice. Mint a `NodeSiteId` per
            // declared input formal so the staging path can address
            // the slot via `(NodeSiteId, body_exec_id)` and the body
            // ops can resolve their input names through
            // `resolve_site_name`.
            if is_bootstrap {
                register_function_input_sites(
                    &mut g,
                    &mut self.exec.ids.next_node_site_id,
                    graph_idx,
                );
            }
            self.push_graph_slot(graph_name, g);
            self.functions.insert(key.clone(), f.clone());
            if is_bootstrap {
                // Register the bootstrap target. A multi-target
                // install registers one bootstrap per target (in
                // the order [`crate::install::install`] iterates the
                // user-supplied `targets` slice). Seeding drains
                // `install_order` front-to-back so each target's
                // bootstrap fires in slice order.
                self.bootstrap.register_module(key);
            }
        }
    }

    /// Allocate the next bootstrap call's body `ExecId` and push every
    /// body OpRef of the front of
    /// [`crate::engine::bootstrap::BootstrapState::install_order`]
    /// onto the frontier. Returns `true` when a bootstrap call was
    /// seeded; `false` when the engine has no remaining bootstrap
    /// functions or the previous call is still in flight.
    ///
    /// Host-driven: `Node::run_bootstrap(&[])` invokes this once after
    /// arming `bootstrap.pending`; the poll cascade reseeds via
    /// `maybe_complete_bootstrap` after each phase drains so multi-
    /// target installs surface one `BootstrapComplete` per target in
    /// install order without further host action. Install itself no
    /// longer arms `pending`.
    pub(crate) fn seed_bootstrap_call(&mut self) -> bool {
        if self.bootstrap.current_exec_id.is_some() {
            return false;
        }
        if self.bootstrap.next_idx >= self.bootstrap.install_order.len() {
            // No further bootstraps to seed; the gate clears on the
            // next `maybe_complete_bootstrap` pass.
            self.bootstrap.pending = false;
            return false;
        }
        let target_name = self.bootstrap.install_order[self.bootstrap.next_idx].clone();
        let Some(meta) = self.bootstrap.module_bootstraps.get(&target_name) else {
            // Defensive skip: install_order names a target whose
            // metadata is missing. Advance past the stale entry
            // rather than wedging.
            self.bootstrap.next_idx += 1;
            return false;
        };
        let key = meta.function_key.clone();
        if self.fire_module_bootstrap(target_name, &key).is_none() {
            self.bootstrap.next_idx += 1;
            return false;
        }
        true
    }

    /// Seed a Module bootstrap body onto the frontier under a fresh
    /// ExecId and record its ExecId in `bootstrap.current_exec_id`.
    /// Returns the body ExecId on success or `None` when the graph
    /// name is missing (defensive — install populates it).
    fn fire_module_bootstrap(&mut self, target_name: String, key: &FunctionKey) -> Option<ExecId> {
        let graph_name = graph_name_for(key);
        let graph_idx = self.graph_idx(&graph_name)?;
        let body_exec_id = self.allocate_exec_id();
        let node_count = self
            .graphs
            .get(graph_idx as usize)
            .map(|g| g.function.node.len())
            .unwrap_or(0);
        // Bootstrap takes no input formals and produces no outputs,
        // so no CallContext lives in `pending_calls`; the body-op
        // gate identifies bootstrap-descendant ExecIds by either
        // direct match against the current bootstrap ExecId or chain
        // walk through descendant FunctionCall CallContexts.
        // Quiescence resolves through `maybe_complete_bootstrap` once
        // every descendant frontier + pending_async entry clears.
        self.bootstrap
            .mark_module_in_flight(target_name, body_exec_id);
        for node_idx in 0..node_count as u32 {
            let op_ref = OpRef::pack(graph_idx, node_idx);
            self.exec.frontier.push_back((op_ref, body_exec_id));
        }
        Some(body_exec_id)
    }

    /// Stage one [`crate::engine::BootstrapInput`] against its target's
    /// declared formal inputs, copy the bytes via Principle 1a, and
    /// seed the body onto the frontier. Helper called by
    /// [`Self::run_bootstrap`] per non-empty target.
    fn enqueue_module_bootstrap(
        &mut self,
        request: crate::engine::bootstrap::BootstrapInput<'_>,
    ) -> Result<(), crate::errors::BootstrapError> {
        // 1. Resolve target → function_key → graph_idx.
        let meta = self
            .bootstrap
            .module_bootstraps
            .get(request.target)
            .ok_or_else(|| crate::errors::BootstrapError::UnknownTarget {
                target_name: request.target.to_string(),
                available: self.bootstrap.install_order.clone(),
            })?;
        let function_key = meta.function_key.clone();
        let graph_name = graph_name_for(&function_key);
        let graph_idx = self.graph_idx(&graph_name).ok_or_else(|| {
            crate::errors::BootstrapError::UnknownTarget {
                target_name: request.target.to_string(),
                available: self.bootstrap.install_order.clone(),
            }
        })?;
        let graph = &self.graphs[graph_idx as usize];

        // 2. Read declared input formals from the GraphSlot.
        let declared: Vec<String> = graph.function.input.clone();

        // 3. Validate. UnknownInput fires first (the supplied set is
        // the host's authoritative request shape; surfacing extras
        // before missing ones gives clearer diagnostics on typos).
        for (input_name, _) in request.inputs {
            if !declared.iter().any(|d| d == input_name) {
                return Err(crate::errors::BootstrapError::UnknownInput {
                    target_name: request.target.to_string(),
                    input_name: input_name.to_string(),
                    declared: declared.clone(),
                });
            }
        }
        for formal in &declared {
            if !request.inputs.iter().any(|(name, _)| *name == formal) {
                return Err(crate::errors::BootstrapError::MissingInput {
                    target_name: request.target.to_string(),
                    input_name: formal.clone(),
                });
            }
        }

        // Resolve each formal to its NodeSiteId before allocating the
        // ExecId — a missing site at this stage is an install
        // invariant violation, but defending against it keeps the
        // error path total.
        let mut sites: Vec<(crate::ids::NodeSiteId, &[u8])> =
            Vec::with_capacity(request.inputs.len());
        for (input_name, bytes) in request.inputs {
            let Some(&site) = graph.site_names.get(*input_name) else {
                return Err(crate::errors::BootstrapError::UnknownInput {
                    target_name: request.target.to_string(),
                    input_name: input_name.to_string(),
                    declared: declared.clone(),
                });
            };
            sites.push((site, *bytes));
        }

        // 4. Allocate body ExecId. Done after validation so a rejected
        // request does not consume an ExecId counter slot.
        let body_exec_id = self.allocate_exec_id();

        // 5. Per-input charge + Principle 1a copy. Track total
        // admitted bytes so a mid-loop failure releases the full
        // charge in one shot.
        let mut admitted: usize = 0;
        for (site, bytes) in &sites {
            let byte_count = bytes.len();
            if let Err(reason) = self.try_charge(byte_count) {
                self.release(admitted);
                return Err(crate::errors::BootstrapError::AllocationFailed {
                    target_name: request.target.to_string(),
                    byte_count,
                    budget_remaining: reason.budget_remaining,
                });
            }
            admitted = admitted.saturating_add(byte_count);
            let mut owned: Vec<u8> = Vec::new();
            if crate::fallible::try_reserve_exact(&mut owned, byte_count).is_err() {
                self.release(admitted);
                return Err(crate::errors::BootstrapError::AllocationFailed {
                    target_name: request.target.to_string(),
                    byte_count,
                    budget_remaining: self
                        .ingress_byte_budget
                        .saturating_sub(self.ingress_bytes_in_flight),
                });
            }
            owned.extend_from_slice(bytes);
            let value: Box<dyn crate::slot_value::SlotValue> =
                Box::new(crate::syscall::values::BytesValue(owned));
            self.slot_write(*site, body_exec_id, value);
        }

        // 6. Mark the Module bootstrap in-flight and push every body
        // OpRef onto the frontier. The body-op gate (`is_op_locked`)
        // recognises the freshly seeded ExecId via `current_exec_id`
        // so descendant ops keep firing.
        self.bootstrap.pending = true;
        self.bootstrap
            .mark_module_in_flight(request.target.to_string(), body_exec_id);
        let node_count = self.graphs[graph_idx as usize].function.node.len();
        for node_idx in 0..node_count as u32 {
            let op_ref = OpRef::pack(graph_idx, node_idx);
            self.exec.frontier.push_back((op_ref, body_exec_id));
        }
        Ok(())
    }

    /// Flat host-facing bootstrap entry point. Empty slice fires the
    /// install-order kick (arming + seeding every queued target);
    /// non-empty slice stages each [`BootstrapInput`] in slice order
    /// using the same validation + Principle 1a copy + frontier seed
    /// as `enqueue_module_bootstrap`.
    pub fn run_bootstrap(
        &mut self,
        targets: &[crate::engine::bootstrap::BootstrapInput<'_>],
    ) -> Result<bool, crate::errors::BootstrapError> {
        if targets.is_empty() {
            if !self.bootstrap.arm_install_order() {
                return Ok(false);
            }
            return Ok(self.seed_bootstrap_call());
        }
        for req in targets {
            self.enqueue_module_bootstrap(crate::engine::bootstrap::BootstrapInput {
                target: req.target,
                inputs: req.inputs,
            })?;
        }
        Ok(true)
    }

    /// `(domain, name, overload)` of the first bootstrap function
    /// recorded at install time, or `None` when no
    /// `module_phase = "bootstrap"` FunctionProto reached the function
    /// library. Stable across `clear_for_restore` (which preserves
    /// install-order metadata but bumps `next_idx` past every queued
    /// target so the restored Node does not re-fire bootstraps it
    /// already ran). For the full ordered list (multi-target installs
    /// queue one key per target), use [`Self::bootstrap_function_keys`].
    pub fn bootstrap_function_key(&self) -> Option<FunctionKey> {
        self.bootstrap.first_function_key().cloned()
    }

    /// All bootstrap function keys the engine has queued, in install
    /// order. Multi-target installs append one entry per target via
    /// [`Self::install_function_library`]; the seeder fires each in
    /// slice order. Stable across `clear_for_restore` for snapshot
    /// introspection.
    pub fn bootstrap_function_keys(&self) -> Vec<FunctionKey> {
        self.bootstrap.function_keys()
    }

    /// `true` while a bootstrap call is outstanding. Armed by
    /// [`Self::run_bootstrap`]; cleared once every queued phase
    /// drains.
    pub fn bootstrap_pending(&self) -> bool {
        self.bootstrap.pending
    }

    /// Lifecycle status for the host-facing
    /// [`crate::node::Node::bootstrap_status`] accessor. `Idle` when no
    /// bootstrap is queued or in-flight; `Running` when a bootstrap
    /// body is currently in-flight; `WaitingForInput` when the
    /// install-order queue still has unseeded targets but no body is
    /// active yet (the host must drive the queue to advance).
    pub fn bootstrap_status(&self) -> crate::engine::bootstrap::BootstrapStatus {
        if self.bootstrap.current_exec_id.is_some() {
            return crate::engine::bootstrap::BootstrapStatus::Running;
        }
        if self.bootstrap.pending && self.bootstrap.next_idx < self.bootstrap.install_order.len() {
            return crate::engine::bootstrap::BootstrapStatus::WaitingForInput;
        }
        crate::engine::bootstrap::BootstrapStatus::Idle
    }

    /// `true` when `target_name` is already on the install-order Module
    /// bootstrap queue. Used by `Node::run_bootstrap` validation.
    pub fn module_bootstrap_registered(&self, target_name: &str) -> bool {
        self.bootstrap.module_bootstraps.contains_key(target_name)
    }

    /// Snapshot of every registered Module bootstrap target name in
    /// install order. Returned by `BootstrapError::UnknownTarget` so
    /// callers see the legal set.
    pub fn module_bootstrap_target_names(&self) -> Vec<String> {
        self.bootstrap.install_order.clone()
    }

    /// Body-op gate. Returns `true` when the op must park because a
    /// bootstrap body is in-flight and the op's ExecId is not a
    /// descendant of the bootstrap ExecId; `false` when the op is
    /// fireable.
    ///
    /// Resolution order:
    /// 1. `bootstrap.pending` clear → fire (gate dormant).
    /// 2. `exec_id` descends from the in-flight bootstrap ExecId via
    ///    the `pending_calls.parent_exec_id` chain → fire. Bootstrap
    ///    body + its sub-FunctionCalls keep firing while the body
    ///    runs.
    /// 3. Otherwise → park. The collapsed gate denies every body-op
    ///    until the bootstrap drains.
    pub(crate) fn is_op_locked(&self, _op_ref: OpRef, exec_id: ExecId) -> bool {
        if !self.bootstrap.pending {
            return false;
        }
        let Some(boot_exec) = self.bootstrap.current_exec_id else {
            return false;
        };
        // 2. Bootstrap-descendant exec ids fire freely. Walk the
        // call chain once and short-circuit if the in-flight ExecId
        // matches anywhere along the chain.
        let mut current = exec_id;
        loop {
            if current == boot_exec {
                return false;
            }
            match self.exec.pending_calls.get(&current) {
                Some(call) => current = call.parent_exec_id,
                None => break,
            }
        }
        true
    }

    /// Inspect engine state and pop the in-flight bootstrap key once
    /// every bootstrap-descendant frontier entry and `pending_async`
    /// entry has cleared. Returns `true` when one phase just drained
    /// (i.e. the caller poll cycle should append a `BootstrapComplete`
    /// step for that phase). With remaining queued keys, the next
    /// `seed_bootstrap_call` advances to the following target;
    /// `bootstrap.pending` flips off only after the last key drains.
    /// Called after each drain phase + the ingress completion drain.
    pub(crate) fn maybe_complete_bootstrap(&mut self) -> bool {
        if !self.bootstrap.pending {
            return false;
        }
        let Some(boot_exec) = self.bootstrap.current_exec_id else {
            return false;
        };
        let descendant = |engine: &Engine, mut exec_id: ExecId| -> bool {
            loop {
                if exec_id == boot_exec {
                    return true;
                }
                match engine.exec.pending_calls.get(&exec_id) {
                    Some(call) => exec_id = call.parent_exec_id,
                    None => return false,
                }
            }
        };
        if self
            .exec
            .frontier
            .iter()
            .any(|(_, exec_id)| descendant(self, *exec_id))
        {
            return false;
        }
        if self
            .exec
            .pending_async
            .values()
            .any(|p| descendant(self, p.exec_id))
        {
            return false;
        }
        // The in-flight phase drained. Advance the install_order
        // pointer (the post-kick cascade) and retire the in-flight
        // ExecId. `bootstrap.pending` clears once the cursor reaches
        // the end of `install_order` AND no in-flight body remains.
        self.bootstrap.next_idx += 1;
        self.bootstrap.clear_in_flight();
        if self.bootstrap.next_idx >= self.bootstrap.install_order.len()
            && self.bootstrap.current_exec_id.is_none()
        {
            self.bootstrap.pending = false;
        }
        true
    }

    /// Resolve every NodeProto's dispatch kind into `op_dispatch[]`
    /// per ENGINE.md §8.1 + §8.4. Run after install completes (so all
    /// symbols are in `self.functions`) but before the first poll.
    ///
    /// Walk order: each GraphSlot in turn. For each NodeProto:
    /// - syscall (`syscall_index` hit) → `Stateless`
    /// - call to function in `self.functions` with domain
    ///   `ai.bytesandbrains.module` → `FunctionCall` with the target
    ///   key + input/output rename pairs from the call's value names
    ///   zipped against the target function's formals.
    /// - call to function with domain `ai.bytesandbrains.framework`
    ///   starting with `BackendSubgraph_` → `BackendSubgraph` with
    ///   bound backend resolved via `BINDING_ID_KEY` metadata against
    ///   the atomic-dispatch table.
    /// - else atomic dispatch by `(domain, op_type, instance)` →
    ///   `Atomic`.
    ///
    /// Returns the number of nodes that remained `Unresolved`. Caller
    /// should fail build if non-zero.
    pub fn resolve_dispatch(&mut self) -> usize {
        // Snapshot per-graph node lists so we don't hold a borrow on
        // self.graphs while reading other tables. Indices are the
        // positional graph_idx values packed into OpRefs.
        let graph_count = self.graphs.len();
        let mut unresolved = 0;
        for graph_idx in 0..graph_count {
            let (function_domain, nodes): (String, Vec<NodeProto>) = {
                let g = &self.graphs[graph_idx];
                (g.function.domain.clone(), g.function.node.clone())
            };
            // BackendSubgraph bodies (domain == ai.bytesandbrains.framework)
            // are handed wholesale to the bound Backend Contract impl per
            // ENGINE.md §8.5; their interior NodeProtos are never invoked
            // individually by the engine, so resolve_dispatch leaves the
            // body's op_dispatch as Unresolved without counting it as a
            // failure. Mirror the same skip already applied by
            // Node's unsupported-ops pre-flight in `src/node.rs`.
            if function_domain == "ai.bytesandbrains.framework" {
                let mut dispatch: Vec<OpDispatch> = Vec::with_capacity(nodes.len());
                for _ in &nodes {
                    dispatch.push(OpDispatch::Unresolved);
                }
                self.graphs[graph_idx].op_dispatch = dispatch;
                continue;
            }
            let mut dispatch: Vec<OpDispatch> = Vec::with_capacity(nodes.len());
            for node in &nodes {
                let resolved = self.resolve_one(node);
                if matches!(resolved, OpDispatch::Unresolved) {
                    unresolved += 1;
                }
                dispatch.push(resolved);
            }
            self.graphs[graph_idx].op_dispatch = dispatch;
        }
        unresolved
    }

    fn resolve_one(&self, node: &NodeProto) -> OpDispatch {
        // 1) Syscall path — single lookup by (domain, op_type).
        if let Some(&fn_ptr) = self
            .syscall_table
            .get(&(node.domain.clone(), node.op_type.clone()))
        {
            return OpDispatch::Stateless(fn_ptr);
        }
        // `crate::registry` for custom ops registered via
        // `bb::register_op!{}`. DCE strips unreferenced entries,
        // so a binary that doesn't `use` a library's op never
        // pulls it into the link image.
        if let Some(reg) = crate::registry::find_op(&node.domain, &node.op_type) {
            return OpDispatch::Stateless(reg.invoke);
        }
        // 2) Function-call paths via the symbol table.
        let key: FunctionKey = (
            node.domain.clone(),
            node.op_type.clone(),
            node.overload.clone(),
        );
        if let Some(target_fn) = self.functions.get(&key) {
            if node.domain == "ai.bytesandbrains.module" {
                let input_rename: Rc<[(String, String)]> = node
                    .input
                    .iter()
                    .zip(target_fn.input.iter())
                    .map(|(caller, formal)| (caller.clone(), formal.clone()))
                    .collect();
                let output_rename: Rc<[(String, String)]> = target_fn
                    .output
                    .iter()
                    .zip(node.output.iter())
                    .map(|(formal, caller)| (formal.clone(), caller.clone()))
                    .collect();
                return OpDispatch::FunctionCall {
                    target: key,
                    input_rename,
                    output_rename,
                };
            }
        }
        // 3) Atomic role path. The placeholders pass stamps
        // `ai.bytesandbrains.slot_id` on every role NodeProto; install
        // populates `slot_id_to_cref` from the model's binding metadata.
        // Single lookup chain: NodeProto.slot_id → ComponentRef →
        // bound role dispatcher closure.
        let slot_id = node
            .metadata_props
            .iter()
            .find(|p| p.key == bb_ir::keys::SLOT_ID_KEY)
            .and_then(|p| p.value.parse::<u32>().ok());
        if let Some(slot_id) = slot_id {
            if let Some(&cref) = self.slot_id_to_cref.get(&slot_id) {
                if let Some(dispatch_fn) = self.dispatch_fn_for_component(cref) {
                    return OpDispatch::Atomic {
                        component_ref: cref,
                        dispatch_fn,
                    };
                }
            }
        }
        OpDispatch::Unresolved
    }

    /// Look up the install-time-stamped `ProtocolDispatchFn` for a
    /// registered component by its `TypeId`. `resolve_dispatch`
    /// embeds the result into `OpDispatch::Atomic { dispatch_fn }`
    /// so runtime invoke skips the per-op TypeId probe.
    fn dispatch_fn_for_component(
        &self,
        cref: ComponentRef,
    ) -> Option<crate::engine::invoke::ProtocolDispatchFn> {
        let component = self.component(cref)?;
        let any: &dyn std::any::Any = component;
        let tid = (*any).type_id();
        self.role_dispatchers.get(&tid).map(|d| d.dispatch)
    }

    /// Resolve a registered component by `ComponentRef`.
    /// `None` when no component lives at that index, or when the
    /// slot was `mem::take`-ed out during dispatch (the caller
    /// dispatching component is invisible to itself).
    pub fn component(&self, cref: ComponentRef) -> Option<&dyn ErasedComponent> {
        self.components.get(cref.as_u32() as usize)?.as_deref()
    }

    /// Resolve a registered component by `ComponentRef` for
    /// mutation. Same null semantics as [`Self::component`].
    pub fn component_mut(&mut self, cref: ComponentRef) -> Option<&mut Box<dyn ErasedComponent>> {
        self.components.get_mut(cref.as_u32() as usize)?.as_mut()
    }

    /// Take the component at `cref` out of the registry, leaving
    /// `None` in its slot. Returns `None` if the slot was empty (or
    /// out of range). Paired with [`Self::restore_component`] in
    /// `invoke_atomic` so a [`crate::runtime::ComponentsView`] can
    /// borrow the rest of `engine.components` while the dispatching
    /// component is held exclusively.
    pub(crate) fn take_component(
        &mut self,
        cref: ComponentRef,
    ) -> Option<Box<dyn ErasedComponent>> {
        self.components.get_mut(cref.as_u32() as usize)?.take()
    }

    /// Put a component back into the registry slot it was taken
    /// from via [`Self::take_component`]. The slot index must be
    /// within range (i.e. the cref came from a prior registration).
    pub(crate) fn restore_component(
        &mut self,
        cref: ComponentRef,
        component: Box<dyn ErasedComponent>,
    ) {
        let idx = cref.as_u32() as usize;
        if let Some(slot) = self.components.get_mut(idx) {
            *slot = Some(component);
        }
    }

    /// test-only registrar. Stores a bound component impl
    /// at `cref`. Grows the underlying Vec to fit the index, filling
    /// holes with `None` so out-of-order registration works.
    pub fn register_component(&mut self, cref: ComponentRef, component: Box<dyn ErasedComponent>) {
        let idx = cref.as_u32() as usize;
        if self.components.len() <= idx {
            self.components.resize_with(idx + 1, || None);
        }
        self.components[idx] = Some(component);
    }

    /// Push an `(OpRef, ExecId)` onto the frontier.
    pub fn push_frontier(&mut self, op_ref: OpRef, exec_id: ExecId) {
        self.exec.frontier.push_back((op_ref, exec_id));
    }

    /// Pop the next `(OpRef, ExecId)` off the frontier. Used by the
    /// poll cycle's drain phases.
    pub fn pop_frontier(&mut self) -> Option<(OpRef, ExecId)> {
        self.exec.frontier.pop_front()
    }

    /// Pop the next fireable `(OpRef, ExecId)` off the frontier,
    /// honouring the per-component body-op gate. With no in-flight
    /// bootstrap the front of the queue fires unconditionally. With
    /// one or more in-flight bootstraps the gate parks any op
    /// touching a locked `ComponentRef`; this scan picks the first
    /// unparked entry so disjoint Components can keep firing while
    /// bootstrap runs against an unrelated slot.
    pub(crate) fn pop_frontier_fireable(&mut self) -> Option<(OpRef, ExecId)> {
        if !self.bootstrap.pending {
            return self.exec.frontier.pop_front();
        }
        let idx = self
            .exec
            .frontier
            .iter()
            .position(|(op_ref, exec_id)| !self.is_op_locked(*op_ref, *exec_id))?;
        self.exec.frontier.remove(idx)
    }

    /// Snapshot of the `(NodeSiteId, ExecId)` keys currently in the
    /// slot table. Test-only - used to assert wire-envelope delivery
    /// lands at the right site without exposing the full
    /// `Box<dyn SlotValue>` payload type.
    pub fn slot_table_keys(&self) -> Vec<(NodeSiteId, ExecId)> {
        self.exec.slot_table.keys().copied().collect()
    }

    /// Iterate every `((NodeSiteId, ExecId), Option<&dyn SlotValue>)`
    /// pair currently in the slot table. Test-only.
    pub fn slot_table_iter(
        &self,
    ) -> impl Iterator<Item = (&(NodeSiteId, ExecId), Option<&dyn SlotValue>)> {
        self.exec
            .slot_table
            .iter()
            .map(|(k, v)| (k, v.as_ref().map(|b| b.as_ref())))
    }

    /// Read a slot value by `(NodeSiteId, ExecId)`. Returns `None`
    /// if the slot is empty or not yet allocated.
    pub fn slot_at(&self, site: NodeSiteId, exec_id: ExecId) -> Option<&dyn SlotValue> {
        self.exec
            .slot_table
            .get(&(site, exec_id))
            .and_then(|s| s.as_ref())
            .map(|b| b.as_ref())
    }
}

/// Register a `NodeSiteId` for every function input name on
/// `graph` so `Engine::deliver_app_event` can seed the input
/// via ingress. Pre-existing entries (where a node output
/// happens to share a name with a function input - rare but
/// possible) are left alone. After the input sites are minted,
/// re-walk the function's node inputs and add consumer entries
/// for any node consuming an input - `GraphSlot::from_function`
/// already populated consumers for node outputs but skipped
/// inputs because they weren't in `site_names` yet.
fn register_function_input_sites(
    graph: &mut crate::engine::graph_slot::GraphSlot,
    next_node_site_id: &mut u64,
    graph_idx: u32,
) {
    for input_name in graph.function.input.clone().iter() {
        if input_name.is_empty() {
            continue;
        }
        graph
            .site_names
            .entry(input_name.clone())
            .or_insert_with(|| {
                let r = NodeSiteId::from(*next_node_site_id);
                *next_node_site_id = next_node_site_id.saturating_add(1);
                r
            });
    }
    // Backfill consumers for nodes whose inputs reference function
    // inputs we just minted sites for. Positional OpRefs make every
    // node's ref `OpRef::pack(graph_idx, node_idx)`.
    let nodes_inputs: Vec<(OpRef, Vec<String>)> = graph
        .function
        .node
        .iter()
        .enumerate()
        .map(|(idx, node)| (OpRef::pack(graph_idx, idx as u32), node.input.clone()))
        .collect();
    for (op_ref, inputs) in nodes_inputs {
        for input in inputs {
            if input.is_empty() {
                continue;
            }
            if let Some(&site) = graph.site_names.get(&input) {
                let entry = graph.consumers.entry(site).or_default();
                if !entry.contains(&op_ref) {
                    entry.push(op_ref);
                }
            }
        }
    }
}

/// Failure case returned by `Engine::try_charge` when admitting
/// `byte_count` more bytes against `ingress_byte_budget` would
/// exceed the cap. `budget_remaining` is the cap minus the live
/// `ingress_bytes_in_flight` at the time of the rejection — the
/// caller embeds both into the resulting `WireReceiveError` or
/// `AppIngressError` so subscribers see the magnitude of the
/// rejection without re-querying the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetExceededReason {
    /// Bytes the caller tried to admit.
    pub byte_count: usize,
    /// Bytes still available under `ingress_byte_budget` at the time
    /// of the rejection.
    pub budget_remaining: usize,
}

/// Stable graph-name key for `Engine.graphs` derived from a
/// `FunctionKey`. Joins the tuple parts so two distinct keys produce
/// distinct strings, preserving the symbol-table semantics.
pub(crate) fn graph_name_for(key: &FunctionKey) -> String {
    let (domain, name, overload) = key;
    if overload.is_empty() {
        format!("{domain}::{name}")
    } else {
        format!("{domain}::{name}#{overload}")
    }
}


#[cfg(test)]
#[path = "core_multi_bootstrap_tests.rs"]
mod multi_bootstrap_tests;

#[cfg(test)]
#[path = "core_op_locked_tests.rs"]
mod op_locked_tests;

#[cfg(test)]
#[path = "core_bootstrap_input_tests.rs"]
mod bootstrap_input_tests;
