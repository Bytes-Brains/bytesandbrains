//! Type relations on ops.
//!
//! Every op declares its type contract via composable
//! [`TypeRelation`]s — a small set of high-coverage predicates
//! (`SameElementType`, `Elementwise`, `BroadcastShape`,
//! `ReduceOver`) plus a `Custom` escape hatch. The compiler's
//! TypeSolver walks the graph, instantiates each relation as a
//! constraint node, and resolves every value's TypeNode via a
//! bipartite worklist (TVM Relay shape).
//!
//! Coverage strategy: a library of trait predicates (MLIR pattern)
//! handles ~90% of ops. The remaining ~10% (Reshape, Gather, Concat,
//! anything with structural type effects) use `Custom`. Adding a
//! new op = declaring its `type_relations` in `atomic_opset()`.

use super::TypeNode;

/// Reference to a port position on an op's input/output list.
/// Indices into `AtomicOpDecl.inputs` / `outputs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortRef {
    /// `inputs[index]` on the surrounding op.
    Input(u8),
    /// `outputs[index]` on the surrounding op.
    Output(u8),
}

/// Outcome of running a relation against the solver's current type
/// nodes. The solver's worklist treats each variant differently:
/// `Refined` requeues dependents, `Satisfied` removes the relation,
/// `Defer` parks it for later, `Failed` aborts the solve.
#[derive(Debug)]
pub enum RelationResult {
    /// Made progress narrowing one or more type variables. Requeue
    /// any relations sharing those types.
    Refined,
    /// Constraint fully satisfied. Remove from the worklist.
    Satisfied,
    /// Insufficient information today. Come back when something
    /// else refines the participating types.
    Defer,
    /// Hard contradiction. The solver propagates the diagnostic
    /// back as a build error.
    Failed(&'static str),
}

/// One type relation declared on an op. The TypeSolver instantiates
/// each as a constraint node linked to its participating type
/// variables via back-edges.
#[derive(Debug)]
pub enum TypeRelation {
    /// All listed ports share the SAME concrete TypeNode. Implements
    /// Julia's "diagonal variable" rule - a port declared `Tensor`
    /// that participates in `SameType([in0, in1, out0])` collapses
    /// to ONE element type across all three positions, regardless
    /// of the bound's permissiveness.
    SameType(&'static [PortRef]),

    /// All listed Tensor-typed ports share the same ELEMENT type.
    /// Shapes may differ (broadcasting is a separate concern).
    /// `Add(x: Tensor, y: Tensor) -> Tensor` uses this.
    SameElementType(&'static [PortRef]),

    /// The output is the broadcast of two tensor inputs. Composes
    /// with `SameElementType` to express `Add` / `Mul` / `Sub` /
    /// `Div` fully.
    BroadcastShape {
        /// First broadcast operand.
        in0: PortRef,
        /// Second broadcast operand.
        in1: PortRef,
        /// Output (shape = broadcast(in0.shape, in1.shape)).
        out: PortRef,
    },

    /// Output preserves the input's TypeNode entirely. Used by
    /// element-wise unary ops (`Sqrt`, `Neg`, `Abs`, `Relu`, etc.):
    /// shape preserved, element type preserved.
    Elementwise {
        /// Input.
        input: PortRef,
        /// Output.
        output: PortRef,
    },

    /// Output is a reduction over the input: same element type,
    /// reduced shape (driven by op attributes like `axes`).
    /// `ReduceSum` / `ReduceMean` / `ReduceMax` use this.
    ReduceOver {
        /// Input tensor being reduced.
        input: PortRef,
        /// Output tensor (lower rank or same rank with size-1 axes).
        output: PortRef,
    },

    /// Escape hatch for ops that don't fit a predicate. The custom
    /// function receives the current TypeNodes for participating
    /// ports and returns a [`RelationResult`].
    ///
    /// Use sparingly - `Reshape`, `Gather`, `Concat`, `Cast`, and
    /// any op with attribute-driven type changes need this.
    Custom {
        /// Stable identifier for diagnostics.
        name: &'static str,
        /// Solver entry point. Receives the participating ports'
        /// current resolutions (`Option<&TypeNode>`); narrows them
        /// or returns `Failed`.
        run: fn(&CustomRelationCtx<'_>) -> RelationResult,
    },
}

/// Context passed to a `Custom` relation's `run` function. Borrows
/// from the solver; exposes a read-only view of each participating
/// port's current type resolution. Concrete shape lands when the
/// TypeSolver (T4) materializes.
#[derive(Debug)]
pub struct CustomRelationCtx<'a> {
    /// Solver-allocated handles for the ports this relation touches,
    /// paired with their current best-known TypeNode (None = still
    /// unresolved).
    pub ports: &'a [(PortRef, Option<&'static TypeNode>)],
}
