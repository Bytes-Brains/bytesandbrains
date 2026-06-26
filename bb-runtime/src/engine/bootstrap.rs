//! Engine bootstrap state — consolidated owner of every bootstrap
//! field the engine reads during install + poll.
//!
//! Replaces the scattered `Engine::bootstrap_*` fields
//! (`bootstrap_function_keys`, `bootstrap_next_idx`,
//! `bootstrap_pending`, `bootstrap_exec_id`). Every read + write goes
//! through `BootstrapState` so the host-driven bootstrap redesign
//! (per
//! `docs/internal/superpowers/specs/2026-06-25-host-driven-bootstrap.md`)
//! can extend the surface with per-target input staging,
//! Component-level bootstraps, and a pending-request queue without
//! re-threading shape changes through call sites.
//!
//! Host-driven (F4): install records targets on `install_order` +
//! `module_bootstraps` but leaves `pending` disarmed. The host arms
//! the queue via [`crate::node::Node::run_bootstrap`] (which calls
//! [`Self::arm_install_order`] + `Engine::seed_bootstrap_call`) or
//! by staging a `BootstrapRequest` (which arms via
//! [`Self::enqueue_request`]).
//!
//! ## Field roles
//!
//! - `install_order` — append-only target-name sequence the install
//!   path stamps when a `module_phase = bootstrap` FunctionProto
//!   lands. Drives the seeder front-to-back so multi-target installs
//!   surface one BootstrapComplete per target in the order the host
//!   supplied to `bytesandbrains::install`.
//! - `module_bootstraps` — per-target metadata (function key + touch
//!   set) keyed on the target name. The touch set is the closure of
//!   every `ComponentRef` referenced by the bootstrap function body
//!   (slot-id NodeProtos + transitive FunctionCalls); the install
//!   path stamps it via `Engine::compute_touch_set` so the
//!   host-driven driver can pre-acquire bound components without
//!   re-walking the program at run time.
//! - `component_bootstraps` — per-slot Component bootstrap registry.
//!   Empty today; F5 populates when Component bootstrap registration
//!   lands.
//! - `pending_requests` — host-supplied `BootstrapRequest`s awaiting
//!   validation + staging. Empty today; F3 populates from
//!   `Node::run_bootstrap` (`BootstrapTarget::ModuleRequests`).
//! - `in_flight` — currently executing bootstraps (Module or
//!   Component). Today at most one entry — the currently seeded
//!   module bootstrap. The Vec shape readies the host-driven path
//!   for concurrent Component bootstraps.
//! - `waiting` — validated + staged `QueuedBootstrap`s ready to fire
//!   once the in-flight set drains. Empty today; F4 populates.
//! - `next_idx` — seed pointer into `install_order`. Bumps each time
//!   `Engine::maybe_complete_bootstrap` observes a phase drained.
//!   `Engine::seed_bootstrap_call` reads it to pick the next target
//!   once the host kicks the queue.
//! - `pending` — coarse "queue still has work" flag the body-op gate
//!   + `maybe_complete_bootstrap` consult. Armed by
//!     `arm_install_order` (host kick) or `enqueue_request` (staged
//!     `BootstrapRequest`), cleared once every queued phase drains.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::engine::dispatch_entry::FunctionKey;
use crate::ids::{ComponentRef, ExecId};

/// Per-target Module bootstrap metadata. Stamped into the engine
/// bootstrap state when the install path sees a
/// `module_phase = bootstrap` FunctionProto.
#[derive(Clone, Debug)]
pub struct ModuleBootstrap {
    /// Canonical `(domain, name, overload)` key of the bootstrap
    /// FunctionProto. Used to look up the GraphSlot at seed time.
    pub function_key: FunctionKey,

    /// Closure of every `ComponentRef` referenced by the bootstrap
    /// body (slot-id NodeProtos + every transitively-called
    /// FunctionProto's slot-id NodeProtos). Computed by the engine
    /// at install time; the host-driven driver consults it to pre-
    /// acquire bound components without re-walking the program.
    pub touch_set: HashSet<ComponentRef>,
}

/// Per-slot Component bootstrap metadata. Stamped into the engine
/// bootstrap state when a Component registers a `Bootstrap`
/// Contract impl. Empty today; F5 wires the registration path.
#[derive(Clone, Debug)]
pub struct ComponentBootstrap {
    /// Component reference the bootstrap dispatches against.
    pub cref: ComponentRef,
}

/// Discriminator for the two bootstrap dispatch kinds the engine
/// drives. Module bootstraps splice into the FunctionCall path under
/// a fresh ExecId; Component bootstraps invoke a Contract method on
/// the bound runtime impl directly.
#[derive(Clone, Debug)]
pub enum BootstrapKind {
    /// Module bootstrap — `target` names the FunctionProto whose
    /// body the engine seeds onto the frontier.
    Module {
        /// Target function name (matches an entry in the engine's
        /// `install_order` queue).
        target: String,
    },

    /// Component bootstrap — `slot` names the binding slot whose
    /// bound Component the engine invokes.
    Component {
        /// Slot name (matches a key in the engine's
        /// `component_bootstraps` registry).
        slot: String,
    },
}

/// Host-facing bootstrap lifecycle status. Returned by
/// `Node::bootstrap_status` so the caller can decide whether to keep
/// polling or surface a "wait for input" prompt. F3 fills the
/// observable surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapStatus {
    /// No bootstrap queued or in-flight — `Node::poll` runs the
    /// body phase freely.
    Idle,

    /// Bootstrap is queued + executing. Body-phase ops park until
    /// the queue drains.
    Running,

    /// Bootstrap is queued but waiting on host-supplied input
    /// formals. The host must call `Node::run_bootstrap`
    /// (`BootstrapTarget::ModuleRequests` or `Slots`) to advance.
    WaitingForInput,
}

/// Host-supplied bootstrap input staging request — borrowed shape per
/// the F5 host-driven bootstrap spec
/// (`docs/internal/superpowers/specs/2026-06-25-host-driven-bootstrap.md`
/// §3.1). The host hands the engine a `target` name plus an ordered
/// `(input_name, value_bytes)` slice; the engine validates against the
/// target's declared formal input ports and runs the Principle 1a copy
/// (cap-check → `try_reserve_exact` → `extend_from_slice`) so the
/// caller's borrowed buffers can drop the moment
/// [`crate::engine::core::Engine::enqueue_bootstrap_request`] returns.
///
/// Owned-form storage on the engine's internal conflict queue lives in
/// [`OwnedBootstrapRequest`]; the borrowed form here is the host-facing
/// request shape only.
pub struct BootstrapRequest<'a> {
    /// Target function name. Must match an entry in the engine's
    /// `install_order` queue (i.e. a Module whose `bootstrap`
    /// FunctionProto landed via `install_function_library`).
    pub target: &'a str,

    /// Ordered `(input_name, value_bytes)` pairs. Validated against
    /// the target's declared input formals; missing required inputs
    /// surface as `BootstrapError::MissingInput` and unknown ones as
    /// `BootstrapError::UnknownInput` before any staging happens.
    pub inputs: &'a [(&'a str, &'a [u8])],
}

/// Owned-form bootstrap input request used by the engine's internal
/// conflict-queue path (`BootstrapState::pending_requests`). The host-
/// facing borrowed [`BootstrapRequest`] is the canonical staging shape;
/// this owned mirror exists so the conflict queue can keep parked
/// requests alive across poll cycles. Kept internal-ish to the engine —
/// the [`BootstrapState::enqueue_request`] consumer is exercised by
/// sibling tests only.
#[derive(Clone, Debug)]
pub struct OwnedBootstrapRequest {
    /// Target function name. Same semantics as
    /// [`BootstrapRequest::target`].
    pub target_name: String,

    /// Ordered `(input_name, value_bytes)` pairs. Same semantics as
    /// [`BootstrapRequest::inputs`] but with owned strings + buffers
    /// so the queue survives poll cycles.
    pub inputs: Vec<(String, Vec<u8>)>,
}

/// Currently executing bootstrap. The host-driven seeder
/// (`Engine::seed_bootstrap_call`) pops one Module target per phase
/// and pushes it here; the Vec shape supports concurrent disjoint
/// Component bootstraps fired through the conflict-queue path.
#[derive(Clone, Debug)]
pub struct InFlightBootstrap {
    /// Kind discriminator — Module vs Component.
    pub kind: BootstrapKind,

    /// ExecId allocated for the bootstrap's body. The body-op gate
    /// walks the `parent_exec_id` chain on `pending_calls` and
    /// fires an op when the chain terminates at this ExecId.
    pub exec_id: ExecId,

    /// `ComponentRef` closure this bootstrap locks against body-phase
    /// ops. Populated from `ModuleBootstrap.touch_set` (Module) or the
    /// single bound `ComponentRef` (Component) at fire time. The
    /// body-op gate (`Engine::is_op_locked`) parks any body op whose
    /// touched `ComponentRef` falls in this set so disjoint Components
    /// can keep firing while bootstrap runs.
    pub touch_set: HashSet<ComponentRef>,

    /// Names already covered by staged input values. Empty today;
    /// F4 populates from `BootstrapRequest::inputs` as the host
    /// supplies formals.
    pub staged_inputs: HashSet<String>,
}

/// Validated + staged bootstrap that cleared the touch-set conflict
/// check and is ready for the engine to assign an ExecId + push its
/// body onto the frontier. The conflict queue
/// (`BootstrapState::process_pending_requests` /
/// `on_bootstrap_drained`) emits `ReadyBootstrap`s back to the engine
/// instead of mutating in-flight state directly so the seed step
/// (ExecId allocation, frontier population) stays on the engine
/// side where the OpRef tables live.
#[derive(Clone, Debug)]
pub struct ReadyBootstrap {
    /// Kind discriminator — Module vs Component.
    pub kind: BootstrapKind,

    /// `ComponentRef` closure the engine should record on the new
    /// `InFlightBootstrap` so the gate locks the right slots.
    pub touch_set: HashSet<ComponentRef>,

    /// Names already covered by staged input values (carried over
    /// from the originating `QueuedBootstrap`). Empty today;
    /// `Node::run_bootstrap` (`BootstrapTarget::ModuleRequests`) populates once F4 lands.
    pub staged_inputs: HashSet<String>,
}

/// Validated + staged bootstrap awaiting an in-flight slot. F4
/// drains `waiting` once `in_flight` clears. Empty today.
#[derive(Clone, Debug)]
pub struct QueuedBootstrap {
    /// Kind discriminator — Module vs Component.
    pub kind: BootstrapKind,

    /// `ComponentRef` closure this queued bootstrap will lock when it
    /// promotes to in-flight. Mirrors `InFlightBootstrap.touch_set`
    /// so the conflict-queue check (F3 Commit 2) can compare a
    /// waiter's touch set against currently-in-flight touch sets.
    pub touch_set: HashSet<ComponentRef>,

    /// Names already covered by staged input values. Filled by
    /// `Node::run_bootstrap` (`BootstrapTarget::ModuleRequests`) once F4 lands.
    pub staged_inputs: HashSet<String>,
}

/// Engine-owned bootstrap state. One field on `Engine`, replacing
/// the four `bootstrap_*` fields the prior shape carried.
pub(crate) struct BootstrapState {
    /// Per-target Module bootstrap metadata. Populated by the
    /// install path when a `module_phase = bootstrap` FunctionProto
    /// lands.
    pub(crate) module_bootstraps: HashMap<String, ModuleBootstrap>,

    /// Per-slot Component bootstrap metadata. Empty today; F5 wires
    /// the registration path.
    pub(crate) component_bootstraps: HashMap<String, ComponentBootstrap>,

    /// Append-only sequence of Module bootstrap target names in
    /// install order. The seeder walks front-to-back so multi-
    /// target installs surface one BootstrapComplete per target.
    /// Append-only across a Node's lifetime; the seeder advances
    /// `next_idx` rather than mutating the Vec so introspection
    /// keeps reporting every queued target across phases.
    pub(crate) install_order: Vec<String>,

    /// Host-supplied bootstrap input staging requests awaiting
    /// validation + staging. Empty today; F3 populates from
    /// `Node::run_bootstrap` (`BootstrapTarget::ModuleRequests`). The owned-form mirror of
    /// [`BootstrapRequest`] keeps parked entries alive across poll
    /// cycles; the engine's F5 immediate-fire entry point
    /// ([`crate::engine::core::Engine::enqueue_bootstrap_request`])
    /// bypasses this queue and stages directly.
    pub(crate) pending_requests: VecDeque<OwnedBootstrapRequest>,

    /// Currently executing bootstraps. The host-driven seeder
    /// (`Engine::seed_bootstrap_call`) records one Module target per
    /// phase; the conflict-queue path can fire additional disjoint
    /// Component bootstraps so multiple entries coexist when
    /// `pending_requests` fans out.
    pub(crate) in_flight: Vec<InFlightBootstrap>,

    /// Validated + staged bootstraps ready to fire once `in_flight`
    /// drains. Empty today; F4 populates.
    pub(crate) waiting: VecDeque<QueuedBootstrap>,

    /// Seed pointer into `install_order`. Bumps each time
    /// [`crate::engine::core::Engine::maybe_complete_bootstrap`]
    /// observes a phase drained. The host kick via
    /// [`Self::arm_install_order`] rewinds — or rather, picks up at —
    /// this offset so re-arming a partially-drained queue continues
    /// from the next unseeded target.
    pub(crate) next_idx: usize,

    /// Coarse "queue still has work" flag. Armed by
    /// [`Self::arm_install_order`] (host kick) or
    /// [`Self::enqueue_request`] (staged input), cleared by
    /// `maybe_complete_bootstrap` after the last queued phase drains.
    /// `Engine::poll` consults it to skip the bootstrap path entirely
    /// on cycles with no queued work.
    pub(crate) pending: bool,
}

impl BootstrapState {
    /// Construct an empty bootstrap state — every map / Vec /
    /// VecDeque empty, `next_idx = 0`, `pending = false`.
    pub(crate) fn new() -> Self {
        Self {
            module_bootstraps: HashMap::new(),
            component_bootstraps: HashMap::new(),
            install_order: Vec::new(),
            pending_requests: VecDeque::new(),
            in_flight: Vec::new(),
            waiting: VecDeque::new(),
            next_idx: 0,
            pending: false,
        }
    }

    /// Reset transient fields ahead of a `Node::restore` call.
    /// `install_order` and `module_bootstraps` stay populated (the
    /// install path stamps both at register time, and restore
    /// preserves install metadata); `pending`, `in_flight`,
    /// `pending_requests`, `waiting`, and `next_idx` reset so the
    /// restored Node does not re-fire bootstraps it already ran.
    pub(crate) fn clear_for_restore(&mut self) {
        self.pending = false;
        self.in_flight.clear();
        self.pending_requests.clear();
        self.waiting.clear();
        self.next_idx = self.install_order.len();
    }

    /// First queued Module bootstrap's function key, or `None` when
    /// the queue is empty. Mirrors the prior `bootstrap_function_key()`
    /// accessor.
    pub(crate) fn first_function_key(&self) -> Option<&FunctionKey> {
        let name = self.install_order.first()?;
        self.module_bootstraps.get(name).map(|m| &m.function_key)
    }

    /// All queued Module bootstrap function keys in install order.
    /// Mirrors the prior `bootstrap_function_keys()` accessor.
    /// Allocates a fresh Vec each call — the caller uses this for
    /// introspection (snapshot dumps, host-side asserts), not the
    /// hot path.
    pub(crate) fn function_keys(&self) -> Vec<FunctionKey> {
        self.install_order
            .iter()
            .filter_map(|name| {
                self.module_bootstraps
                    .get(name)
                    .map(|m| m.function_key.clone())
            })
            .collect()
    }

    /// ExecId of the currently in-flight Module bootstrap. Mirrors
    /// the prior `bootstrap_exec_id` field. Returns `None` when no
    /// Module bootstrap is in-flight; if multiple in-flight entries
    /// exist (future Component bootstrap path), returns the first
    /// Module-kind ExecId.
    pub(crate) fn module_exec_id(&self) -> Option<ExecId> {
        self.in_flight.iter().find_map(|b| match b.kind {
            BootstrapKind::Module { .. } => Some(b.exec_id),
            BootstrapKind::Component { .. } => None,
        })
    }

    /// Record a Module bootstrap target. Appends to `install_order`,
    /// inserts `module_bootstraps[name]`. Idempotent per target name —
    /// re-registering the same name updates the metadata in place
    /// without re-appending to `install_order`.
    ///
    /// Does not arm `pending`. Host-driven F4 model: install records
    /// the bootstrap, the host calls [`crate::node::Node::run_bootstrap`]
    /// (or stages a `BootstrapRequest`) to actually fire it. `Engine::poll`
    /// no longer auto-seeds the queue on first call.
    pub(crate) fn register_module(&mut self, function_key: FunctionKey) {
        let name = function_key.1.clone();
        let entry = self
            .module_bootstraps
            .entry(name.clone())
            .or_insert_with(|| ModuleBootstrap {
                function_key: function_key.clone(),
                touch_set: HashSet::new(),
            });
        // Refresh the function_key in case install path passes a
        // different (domain, overload) for the same name — the most
        // recent install wins.
        entry.function_key = function_key;
        if !self.install_order.iter().any(|n| n == &name) {
            self.install_order.push(name);
        }
    }

    /// Arm `pending` and rewind `next_idx` to the first unseeded target
    /// so [`crate::engine::core::Engine::seed_bootstrap_call`] picks up
    /// where the previous drain left off. The host calls this through
    /// `Node::run_bootstrap` to kick the install-time bootstrap queue;
    /// returns `false` when no install-order target remains (idempotent
    /// on a fully drained Node).
    pub(crate) fn arm_install_order(&mut self) -> bool {
        if self.next_idx >= self.install_order.len() {
            return false;
        }
        self.pending = true;
        true
    }

    /// Mark a Module bootstrap target as in-flight. Pushes an
    /// `InFlightBootstrap` with kind `Module { target }` carrying
    /// the supplied ExecId + the `ComponentRef` closure the body-op
    /// gate consults to park overlapping body ops.
    pub(crate) fn mark_module_in_flight(
        &mut self,
        target: String,
        exec_id: ExecId,
        touch_set: HashSet<ComponentRef>,
    ) {
        self.in_flight.push(InFlightBootstrap {
            kind: BootstrapKind::Module { target },
            exec_id,
            touch_set,
            staged_inputs: HashSet::new(),
        });
    }

    /// Look up a Component bootstrap by slot name. F5 populates
    /// `component_bootstraps` from the Bootstrap Contract
    /// registration path; today the map stays empty so this lookup
    /// always returns `None`. Wired now so the field has a non-test
    /// reader and the F5 dispatch path doesn't need to expand the
    /// accessor surface — it just calls this method.
    pub(crate) fn component_bootstrap(&self, slot: &str) -> Option<&ComponentBootstrap> {
        self.component_bootstraps.get(slot)
    }

    /// Push a host-supplied [`OwnedBootstrapRequest`] onto the pending
    /// queue + arm `pending` so the body gate stays parked until the
    /// staged work drains. Direct entry-point exists for the engine's
    /// internal conflict-queue tests; production callers use
    /// [`crate::engine::core::Engine::enqueue_bootstrap_request`]
    /// (F5 immediate-fire) which validates + stages without going
    /// through the parked queue.
    #[cfg(test)]
    pub(crate) fn enqueue_request(&mut self, req: OwnedBootstrapRequest) {
        self.pending_requests.push_back(req);
        self.pending = true;
    }

    /// Drain `pending_requests` once. For each request, resolve its
    /// target's `ComponentRef` touch set (Module bootstraps look it
    /// up via `module_bootstraps`); compare against every currently
    /// in-flight bootstrap's touch set. On disjoint → return a
    /// [`ReadyBootstrap`] for the engine to seed. On overlap →
    /// enqueue as [`QueuedBootstrap`] in `waiting` and let
    /// [`Self::on_bootstrap_drained`] promote it later. Requests
    /// whose target is unknown drop silently — caller-side
    /// validation is responsible for surfacing the error before the
    /// request reaches the queue.
    pub(crate) fn process_pending_requests(&mut self) -> Vec<ReadyBootstrap> {
        let mut ready = Vec::new();
        while let Some(req) = self.pending_requests.pop_front() {
            // Look up the touch set for this target. Module
            // bootstraps cache it on `module_bootstraps`. Unknown
            // targets drop here — the host-facing API
            // (`Node::run_bootstrap` (`BootstrapTarget::ModuleRequests`)) returns
            // `BootstrapError::UnknownTarget` before enqueueing, so
            // reaching this branch indicates a stale request and we
            // skip rather than wedge.
            let Some(meta) = self.module_bootstraps.get(&req.target_name) else {
                continue;
            };
            let touch_set = meta.touch_set.clone();
            let kind = BootstrapKind::Module {
                target: req.target_name.clone(),
            };
            // Staged-inputs placeholder — F4 populates from
            // `req.inputs` once input staging lands. For Commit 2
            // the conflict-queue path is the only consumer and it
            // ignores the names.
            let staged_inputs: HashSet<String> =
                req.inputs.iter().map(|(name, _)| name.clone()).collect();
            if Self::overlaps_any_in_flight(&self.in_flight, &touch_set) {
                self.waiting.push_back(QueuedBootstrap {
                    kind,
                    touch_set,
                    staged_inputs,
                });
            } else {
                ready.push(ReadyBootstrap {
                    kind,
                    touch_set,
                    staged_inputs,
                });
            }
        }
        ready
    }

    /// Drop the in-flight bootstrap whose body ExecId matches
    /// `exec_id`, then walk `waiting` once and promote any waiter
    /// whose touch set no longer conflicts with the remaining
    /// in-flight set. Returns promoted waiters in queue order so
    /// the engine can seed them. The engine calls this from
    /// `maybe_complete_bootstrap` after a phase drains.
    pub(crate) fn on_bootstrap_drained(&mut self, exec_id: ExecId) -> Vec<ReadyBootstrap> {
        self.in_flight.retain(|b| b.exec_id != exec_id);
        let mut promoted = Vec::new();
        let mut remaining = VecDeque::new();
        while let Some(waiter) = self.waiting.pop_front() {
            if Self::overlaps_any_in_flight(&self.in_flight, &waiter.touch_set) {
                remaining.push_back(waiter);
            } else {
                promoted.push(ReadyBootstrap {
                    kind: waiter.kind,
                    touch_set: waiter.touch_set,
                    staged_inputs: waiter.staged_inputs,
                });
            }
        }
        self.waiting = remaining;
        promoted
    }

    /// `true` when `touch_set` overlaps any in-flight bootstrap's
    /// touch set. Empty touch sets never conflict — Module
    /// bootstraps whose body touches no Component can fan out
    /// alongside any in-flight target.
    fn overlaps_any_in_flight(
        in_flight: &[InFlightBootstrap],
        touch_set: &HashSet<ComponentRef>,
    ) -> bool {
        if touch_set.is_empty() {
            return false;
        }
        in_flight
            .iter()
            .any(|b| !b.touch_set.is_disjoint(touch_set))
    }
}

