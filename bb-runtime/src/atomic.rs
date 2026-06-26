//! Engine-side atomic dispatch result. Catalog types
//! (`AtomicOpsetDecl`, `AtomicOpDecl`, `AtomicOpKind`) live in
//! `bb_ir::atomic`; re-exported here for one-import access.

use crate::ids::CommandId;
use bb_ir::slot_value::SlotValue;

pub use bb_ir::atomic::{AtomicOpDecl, AtomicOpKind, AtomicOpsetDecl};

/// Return type of `<Role>Runtime::dispatch_atomic`.
pub enum DispatchResult {
    /// Outputs ready synchronously.
    Immediate(Vec<(String, Box<dyn SlotValue>)>),
    /// Impl will call `ctx.complete_command(cmd_id, outputs)` later.
    Async(CommandId),
}

impl std::fmt::Debug for DispatchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Immediate(outputs) => f
                .debug_tuple("Immediate")
                .field(&format!("<{} outputs>", outputs.len()))
                .finish(),
            Self::Async(cmd_id) => f.debug_tuple("Async").field(cmd_id).finish(),
        }
    }
}

