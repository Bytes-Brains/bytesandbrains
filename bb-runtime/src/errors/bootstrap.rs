//! `BootstrapError`
//!
//! Returned by [`crate::node::Node::run_bootstrap`] when input
//! staging or target selection violates the contract per
//! `docs/internal/superpowers/specs/2026-06-25-host-driven-bootstrap.md`
//! §3.2.
//!
//! Each variant carries every datum a host needs to recover (the
//! offending name plus the declared set, the unknown slot id plus
//! the available ids, …) so callers can surface human-readable
//! prompts without re-introspecting the engine.

/// Errors surfaced by host-facing bootstrap staging methods on
/// `Node`. F3 lands the Node API + validation logic; this commit
/// defines the error taxonomy + display only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootstrapError {
    /// `run_bootstrap(BootstrapTarget::ModuleRequests|ModuleNames|Slots)`
    /// named a target the engine has no bootstrap registration for.
    /// Carries the queue snapshot so callers can present the legal set.
    UnknownTarget {
        /// Target name the caller supplied.
        target_name: String,
        /// Target names currently queued, in install order.
        available: Vec<String>,
    },

    /// `run_bootstrap(BootstrapTarget::Slots(...))` named a slot id
    /// that does not exist on the target's input site map. Carries
    /// the legal ids so callers can correct the mapping.
    UnknownSlot {
        /// Slot id the caller supplied.
        slot_id: u32,
        /// Slot ids declared on the target's input site map.
        available: Vec<u32>,
    },

    /// The host called `run_bootstrap` with a target whose inputs are
    /// already staged + queued. The host must drive the pending
    /// request through to completion (or cancel it) before
    /// re-targeting the same name.
    AlreadyTransitivelyQueued {
        /// Target name that was already queued.
        target_name: String,
    },

    /// `run_bootstrap(BootstrapTarget::ModuleRequests(...))` named an
    /// input the target does not declare as a formal. Carries the
    /// declared set so callers can correct the request.
    UnknownInput {
        /// Target name the request targeted.
        target_name: String,
        /// Input name the caller supplied.
        input_name: String,
        /// Input names declared as formals on the target.
        declared: Vec<String>,
    },

    /// `run_bootstrap(BootstrapTarget::ModuleRequests(...))` is
    /// missing a required formal input. Validation fails atomically —
    /// no inputs stage when this fires.
    MissingInput {
        /// Target name the request targeted.
        target_name: String,
        /// Input name declared as a formal but not supplied.
        input_name: String,
    },

    /// Bootstrap input staging hit the engine's `ingress_byte_budget`
    /// cap or the `try_reserve_exact` seam returned `TryReserveError`.
    /// Carries the offending input's byte count + the remaining
    /// budget at the point of rejection so the host can decide
    /// whether to back off, shrink the payload, or raise the cap.
    /// Any per-input charges that landed earlier in the same
    /// request are released before the engine surfaces this error —
    /// the bootstrap state stays untouched.
    AllocationFailed {
        /// Target name the request targeted.
        target_name: String,
        /// Bytes the staging step tried to admit when the cap or
        /// allocator rejected the request.
        byte_count: usize,
        /// Bytes still available under `ingress_byte_budget` at the
        /// point of the rejection.
        budget_remaining: usize,
    },
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTarget {
                target_name,
                available,
            } => write!(
                f,
                "unknown bootstrap target '{target_name}'; available: {available:?}",
            ),
            Self::UnknownSlot { slot_id, available } => write!(
                f,
                "unknown bootstrap slot id {slot_id}; available: {available:?}",
            ),
            Self::AlreadyTransitivelyQueued { target_name } => write!(
                f,
                "bootstrap target '{target_name}' already has a queued input request",
            ),
            Self::UnknownInput {
                target_name,
                input_name,
                declared,
            } => write!(
                f,
                "target '{target_name}' has no input '{input_name}'; declared: {declared:?}",
            ),
            Self::MissingInput {
                target_name,
                input_name,
            } => write!(
                f,
                "target '{target_name}' missing required input '{input_name}'",
            ),
            Self::AllocationFailed {
                target_name,
                byte_count,
                budget_remaining,
            } => write!(
                f,
                "target '{target_name}' input staging refused {byte_count}B (budget remaining {budget_remaining}B)",
            ),
        }
    }
}

impl std::error::Error for BootstrapError {}

