//! `PendingAsync` + `ExecutionState` - engine's async-suspension
//! bookkeeping + execution liveness
//! tracking per §5.3.

use crate::ids::{ExecId, NodeSiteId, OpRef};

/// Bookkeeping for an Op suspended on a `CommandId` per
/// `docs/ENGINE.md` §9.1. Stored in `Engine.pending_async`,
/// keyed by `CommandId`.
pub struct PendingAsync {
    /// The Op that's suspended.
    pub op_ref: OpRef,
    /// The execution this suspension belongs to.
    pub exec_id: ExecId,
    /// Output sites the Op declared. Populated when the CommandId
    /// completes with values.
    pub output_sites: Vec<NodeSiteId>,
    /// Absolute deadline (`scheduler.now_ns()` clock) past which
    /// the suspension expires. `None` means "no engine-side
    /// deadline" - the Op runs until the transport reports
    /// completion or failure. Phase 5 of the poll cycle scans
    /// pending suspensions each tick and fails any whose deadline
    /// has passed via the existing `OpFailed` path.
    pub deadline_ns: Option<u64>,
}

/// Per-execution liveness tracker
/// Stored in `Engine.execution_state`, keyed by `ExecId`. /// minimum-viable: just the output counter for GC bookkeeping;
/// may extend.
#[derive(Default)]
pub struct ExecutionState {
    /// How many output sites this execution has filled.
    pub outputs_written: u32,
}
