//! Atomic-op opset declaration types. Returned from
//! `<Role>Runtime::atomic_opset()`; engine-side `DispatchResult`
//! lives in `bb_runtime::atomic` because it carries `CommandId`.

use crate::types::TypeNode;
use crate::types::TypeRelation;

/// Atomic-op opset owned by a `<Role>Runtime` impl. Merged into the
/// per-Node `(domain, op_type, instance) → ComponentRef` table at
/// `Node::ready()` time.
#[derive(Clone, Copy, Debug)]
pub struct AtomicOpsetDecl {
    /// Per-impl namespace. Convention: `<crate>.<TypeName>.atomic`.
    pub domain: &'static str,

    /// Major version. Bumped when the op set changes meaningfully.
    pub version: i64,

    /// Op_types this impl handles via `dispatch_atomic`.
    pub ops: &'static [AtomicOpDecl],
}

/// One atomic-op declaration inside an `AtomicOpsetDecl`.
#[derive(Debug)]
pub struct AtomicOpDecl {
    /// Op_type string. Used as the `(domain, op_type, instance)`
    /// dispatch key.
    pub name: &'static str,

    /// Input slot names + their `TypeNode`. The engine validates
    /// that each `dispatch_atomic` call's inputs match.
    pub inputs: &'static [(&'static str, &'static TypeNode)],

    /// Output slot names + their `TypeNode`.
    pub outputs: &'static [(&'static str, &'static TypeNode)],

    /// Sync or async completion semantics.
    pub kind: AtomicOpKind,

    /// Type relations the TypeSolver instantiates. Empty for ops
    /// whose `inputs/outputs` already pin concrete types; populated
    /// for polymorphic ops (Add, MatMul, Reshape).
    pub type_relations: &'static [TypeRelation],
}

/// Sync vs. async completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOpKind {
    /// Outputs returned from `dispatch_atomic` directly.
    Immediate,
    /// Outputs arrive via `ctx.complete_command(cmd_id, ...)`.
    Async,
}

