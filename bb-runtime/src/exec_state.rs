//! `ExecState` — the engine's per-poll execution-state bundle.
//!
//! Groups the fields that walk the DAG cascade together: the
//! frontier work queue, the slot table, per-execution liveness,
//! parked async ops, in-cycle completions, function-call
//! invocation frames, and the monotonic ID allocator.
//!
//! `Engine` carries one `exec: ExecState` field rather than nine
//! sibling fields. The architectural distinction matches the poll
//! cycle's phases: Phase 2/6 drains `exec.frontier`; Phase 5 matches
//! `exec.pending_completions` against `exec.pending_async`.
//!
//! The timer scheduler and inbound-envelope context map live on
//! [`crate::framework::FrameworkComponents`] alongside the other
//! syscall-driven primitives (`HoldTable`, `SerializeQueue`,
//! `RecordBuffer`, `EventSource`). Syscalls (`Sleep`, `Interval`,
//! `Pulse`) drive the scheduler; RX gates + `wire.Send` forwarding
//! drive inbound context lookup. Treating them as framework
//! primitives matches their consumers.

use std::collections::{HashMap, VecDeque};

use crate::engine::call_context::CallContext;
use crate::engine::pending_async::{ExecutionState, PendingAsync};
use crate::ids::{CommandId, ExecId, NodeSiteId, OpRef};
use crate::runtime::PendingCompletion;
use crate::slot_value::SlotValue;

/// Monotonic per-Node ID source. Single-threaded reads/writes so the
/// ingress queue stays the only cross-thread sync primitive.
///
/// `OpRef` has no counter - it packs `(graph_idx, node_idx)`
/// positionally at install time; see [`crate::ids::OpRef::pack`].
#[derive(Default)]
pub struct IdAllocator {
    /// Source of fresh `ExecId`s minted at every entry point.
    pub next_exec_id: u64,
    /// Source of fresh `CommandId`s for async dispatch suspensions.
    pub next_command_id: u64,
    /// Source of fresh `NodeSiteId`s for graph installation.
    pub next_node_site_id: u64,
}

/// Per-poll execution-state bundle held on `Engine`.
pub struct ExecState {
    /// In-cycle DAG-walking queue: ops ready to fire now, paired
    /// with the execution they belong to.
    pub frontier: VecDeque<(OpRef, ExecId)>,
    /// Per-execution slot storage keyed by `(NodeSiteId, ExecId)`.
    pub slot_table: HashMap<(NodeSiteId, ExecId), Option<Box<dyn SlotValue>>>,
    /// Per-execution liveness tracker (output counter used for GC
    /// bookkeeping when an execution finishes).
    pub execution_state: HashMap<ExecId, ExecutionState>,
    /// Suspended ops awaiting `CommandId` completion. Phase 5 drains
    /// matches against `pending_completions`.
    pub pending_async: HashMap<CommandId, PendingAsync>,
    /// Completions captured during a dispatch hook. Drained by
    /// Phase 5 and matched against `pending_async`.
    pub pending_completions: Vec<PendingCompletion>,
    /// Active function-call invocations keyed by the body's
    /// `ExecId`. Populated when `OpDispatch::FunctionCall` fires;
    /// removed as outputs are forwarded back to the caller.
    pub pending_calls: HashMap<ExecId, CallContext>,
    /// Monotonic ID source for `ExecId` / `CommandId` /
    /// `NodeSiteId` / `OpRef`.
    pub ids: IdAllocator,
}

impl Default for ExecState {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecState {
    /// Construct a fresh `ExecState` with empty queues and zero
    /// counters.
    pub fn new() -> Self {
        Self {
            frontier: VecDeque::new(),
            slot_table: HashMap::new(),
            execution_state: HashMap::new(),
            pending_async: HashMap::new(),
            pending_completions: Vec::new(),
            pending_calls: HashMap::new(),
            ids: IdAllocator::default(),
        }
    }
}
