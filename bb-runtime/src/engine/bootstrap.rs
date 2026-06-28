//! Engine bootstrap state — consolidated owner of every bootstrap
//! field the engine reads during install + poll.
//!
//! Host-driven: install records targets on `install_order` +
//! `module_bootstraps` but leaves `pending` disarmed. The host arms
//! the queue via [`crate::node::Node::run_bootstrap`] (empty slice)
//! to drive every install-order target, or supplies one or more
//! [`BootstrapInput`]s to fire named targets with staged inputs.
//!
//! ## Field roles
//!
//! - `install_order` — append-only target-name sequence the install
//!   path stamps when a `module_phase = bootstrap` FunctionProto
//!   lands. Drives the seeder front-to-back so multi-target installs
//!   surface one BootstrapComplete per target in the order the host
//!   supplied to `bytesandbrains::install`.
//! - `module_bootstraps` — per-target metadata (function key) keyed
//!   on the target name. Install stamps each entry as the
//!   `module_phase = bootstrap` FunctionProto lands; the engine
//!   reads it to resolve target → FunctionKey at seed time.
//! - `current_exec_id` — `Some(ExecId)` while a bootstrap body is
//!   in-flight; `None` otherwise. Single-target body gate — the
//!   collapsed shape replaces the prior per-Component touch-set
//!   conflict queue.
//! - `next_idx` — seed pointer into `install_order`. Bumps each time
//!   `Engine::maybe_complete_bootstrap` observes a phase drained.
//!   `Engine::seed_bootstrap_call` reads it to pick the next target
//!   once the host kicks the queue (empty-slice `run_bootstrap`).
//! - `pending` — single body-op gate. Armed by
//!   [`Self::arm_install_order`] (empty-slice host kick) or by the
//!   engine's per-target staging path (non-empty `run_bootstrap`).
//!   Cleared once every queued phase drains.

use std::collections::HashMap;

use crate::engine::dispatch_entry::FunctionKey;
use crate::ids::ExecId;

/// Per-target Module bootstrap metadata. Stamped into the engine
/// bootstrap state when the install path sees a
/// `module_phase = bootstrap` FunctionProto.
#[derive(Clone, Debug)]
pub struct ModuleBootstrap {
    /// Canonical `(domain, name, overload)` key of the bootstrap
    /// FunctionProto. Used to look up the GraphSlot at seed time.
    pub function_key: FunctionKey,
}

/// Discriminator carried over from earlier shapes. Module is the
/// only kind today; the enum is kept so external introspection that
/// matched on `BootstrapKind::Module` keeps compiling.
#[derive(Clone, Debug)]
pub enum BootstrapKind {
    /// Module bootstrap — `target` names the FunctionProto whose
    /// body the engine seeds onto the frontier.
    Module {
        /// Target function name (matches an entry in the engine's
        /// `install_order` queue).
        target: String,
    },
}

/// Host-facing bootstrap lifecycle status. Returned by
/// `Node::bootstrap_status` so the caller can decide whether to keep
/// polling or surface a "wait for input" prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapStatus {
    /// No bootstrap queued or in-flight — `Node::poll` runs the
    /// body phase freely.
    Idle,

    /// Bootstrap is queued + executing. Body-phase ops park until
    /// the queue drains.
    Running,

    /// Bootstrap is queued but waiting on host-supplied input
    /// formals. The host must call `Node::run_bootstrap` with a
    /// non-empty `BootstrapInput` slice to advance.
    WaitingForInput,
}

/// Host-supplied bootstrap input staging request — borrowed shape.
/// The host hands the engine a `target` name plus an ordered
/// `(input_name, value_bytes)` slice; the engine validates against the
/// target's declared formal input ports and runs the Principle 1a copy
/// (cap-check → `try_reserve_exact` → `extend_from_slice`) so the
/// caller's borrowed buffers can drop the moment
/// [`crate::node::Node::run_bootstrap`] returns.
pub struct BootstrapInput<'a> {
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

/// Owned-form mirror of [`BootstrapInput`]. Kept around because
/// callers (and tests) may want an owned shape, but the new flat
/// `run_bootstrap` path stages each entry directly without parking
/// it on a queue.
#[derive(Clone, Debug)]
pub struct OwnedBootstrapInput {
    /// Target function name. Same semantics as
    /// [`BootstrapInput::target`].
    pub target: String,

    /// Ordered `(input_name, value_bytes)` pairs. Same semantics as
    /// [`BootstrapInput::inputs`] but with owned strings + buffers.
    pub inputs: Vec<(String, Vec<u8>)>,
}

/// Engine-owned bootstrap state.
pub(crate) struct BootstrapState {
    /// Per-target Module bootstrap metadata. Populated by the
    /// install path when a `module_phase = bootstrap` FunctionProto
    /// lands.
    pub(crate) module_bootstraps: HashMap<String, ModuleBootstrap>,

    /// Append-only sequence of Module bootstrap target names in
    /// install order. The seeder walks front-to-back so multi-
    /// target installs surface one BootstrapComplete per target.
    /// Append-only across a Node's lifetime; the seeder advances
    /// `next_idx` rather than mutating the Vec so introspection
    /// keeps reporting every queued target across phases.
    pub(crate) install_order: Vec<String>,

    /// Currently in-flight bootstrap body ExecId. Single-slot;
    /// the post-collapse design fires bootstraps sequentially so
    /// at most one body is alive at a time.
    pub(crate) current_exec_id: Option<ExecId>,

    /// Seed pointer into `install_order`. Bumps each time
    /// [`crate::engine::core::Engine::maybe_complete_bootstrap`]
    /// observes a phase drained. The host kick via
    /// [`Self::arm_install_order`] picks up at this offset so a
    /// partially-drained queue continues from the next unseeded
    /// target.
    pub(crate) next_idx: usize,

    /// Coarse "queue still has work" flag. Armed by
    /// [`Self::arm_install_order`] (host kick) or by the engine's
    /// per-target staging path; cleared by `maybe_complete_bootstrap`
    /// after the last queued phase drains. `Engine::poll` consults
    /// it to gate body-phase ops.
    pub(crate) pending: bool,
}

impl BootstrapState {
    /// Construct an empty bootstrap state — every map / Vec empty,
    /// `next_idx = 0`, `pending = false`.
    pub(crate) fn new() -> Self {
        Self {
            module_bootstraps: HashMap::new(),
            install_order: Vec::new(),
            current_exec_id: None,
            next_idx: 0,
            pending: false,
        }
    }

    /// Reset transient fields ahead of a `Node::restore` call.
    /// `install_order` and `module_bootstraps` stay populated (the
    /// install path stamps both at register time, and restore
    /// preserves install metadata); `pending`, `current_exec_id`,
    /// and `next_idx` reset so the restored Node does not re-fire
    /// bootstraps it already ran.
    pub(crate) fn clear_for_restore(&mut self) {
        self.pending = false;
        self.current_exec_id = None;
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

    /// Record a Module bootstrap target. Appends to `install_order`,
    /// inserts `module_bootstraps[name]`. Idempotent per target name —
    /// re-registering the same name updates the metadata in place
    /// without re-appending to `install_order`.
    ///
    /// Does not arm `pending`. Host-driven model: install records
    /// the bootstrap, the host calls [`crate::node::Node::run_bootstrap`]
    /// to actually fire it. `Engine::poll` no longer auto-seeds the
    /// queue on first call.
    pub(crate) fn register_module(&mut self, function_key: FunctionKey) {
        let name = function_key.1.clone();
        let entry = self
            .module_bootstraps
            .entry(name.clone())
            .or_insert_with(|| ModuleBootstrap {
                function_key: function_key.clone(),
            });
        // Refresh the function_key in case install path passes a
        // different (domain, overload) for the same name — the most
        // recent install wins.
        entry.function_key = function_key;
        if !self.install_order.iter().any(|n| n == &name) {
            self.install_order.push(name);
        }
    }

    /// Arm `pending` so [`crate::engine::core::Engine::seed_bootstrap_call`]
    /// picks up where the previous drain left off. The host calls
    /// this through `Node::run_bootstrap` (empty slice) to kick the
    /// install-time bootstrap queue; returns `false` when no
    /// install-order target remains (idempotent on a fully drained
    /// Node).
    pub(crate) fn arm_install_order(&mut self) -> bool {
        if self.next_idx >= self.install_order.len() {
            return false;
        }
        self.pending = true;
        true
    }

    /// Mark a Module bootstrap target as in-flight by recording its
    /// body ExecId in `current_exec_id`. The body-op gate consults
    /// this single ExecId via the descendant-chain walk.
    pub(crate) fn mark_module_in_flight(&mut self, _target: String, exec_id: ExecId) {
        self.current_exec_id = Some(exec_id);
    }

    /// Drop the currently in-flight bootstrap body ExecId.
    /// `maybe_complete_bootstrap` calls this once the phase drains.
    pub(crate) fn clear_in_flight(&mut self) {
        self.current_exec_id = None;
    }
}

