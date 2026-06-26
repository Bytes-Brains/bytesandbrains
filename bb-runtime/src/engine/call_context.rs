//! Per-function-call invocation context.
//!
//! Stored on `Engine.pending_calls[body_exec_id]` when an
//! `OpDispatch::FunctionCall` fires. Encodes the runtime aliasing
//! the body's nodes need to read inputs from the caller's scope
//! (zero-copy) and the output forwarding map that lands body outputs
//! back at the caller's slots when they complete.

use std::collections::HashMap;

use crate::engine::dispatch_entry::FunctionKey;
use crate::ids::{ExecId, NodeSiteId};

/// Per-call invocation context, keyed by the body's fresh
/// `ExecId` in `Engine.pending_calls`.
#[derive(Debug)]
pub struct CallContext {
    /// The caller's `ExecId` - where input slots live and where
    /// output forwarding writes back.
    pub parent_exec_id: ExecId,

    /// The called function's symbol-table key. Stamped onto the
    /// `engine.function_call` tracing span so traces attribute body
    /// activity to the calling function.
    pub target: FunctionKey,

    /// Formal parameter name → caller-side `NodeSiteId`. Body
    /// nodes that consume a formal input look up the alias here
    /// and read from `slot_table[(alias_site, parent_exec_id)]`.
    /// No value copy - body reads from caller's slot directly.
    pub input_aliases: HashMap<String, NodeSiteId>,

    /// Body-side `NodeSiteId` → caller-side `NodeSiteId`. When
    /// `write_outputs` writes to a body output site, the value is
    /// also moved to the matching caller site at
    /// `parent_exec_id`, and `push_ready_consumers` is re-run for
    /// the caller's downstream.
    pub output_forwarding: HashMap<NodeSiteId, NodeSiteId>,

    /// Decremented each time an output is forwarded; the entry
    /// is dropped from `pending_calls` when this reaches zero.
    pub outputs_remaining: usize,
}
