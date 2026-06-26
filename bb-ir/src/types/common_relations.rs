//! Named `TypeRelation` slice constants for the common op shapes.
//!
//! Sharing one set of canonical slices across `bb-ops::backends`,
//! `bb-derive`, and any third-party crate keeps op declarations
//! one-liners and lets the TypeSolver work with stable
//! pointer-equal constraint pools (the solver indexes relations by
//! their `&'static` address). New shapes graft in alongside these
//! without breaking the convention.

use super::{relations::TypeRelation, PortRef};

/// `[]` — explicit "no constraint" slot. Use this on ops whose type
/// is attribute-driven (`Cast`), variadic (`Concat`, `Split`,
/// `Gemm`), structural (`Reshape`'s output shape comes from an
/// attribute), or whose I/O types are heterogeneous and lack a
/// shared element-type bound (most role placeholders).
pub static NO_RELATIONS: &[TypeRelation] = &[];

/// Unary element-wise: `output.TypeNode == input.TypeNode`. Shape
/// preserved, element type preserved. Use for `Neg`, `Abs`, `Sqrt`,
/// `Exp`, `Log`, `Identity`, `Relu`, `Sigmoid`, `Tanh`, `Softmax`,
/// `LeakyRelu`, `Gelu`, `GlobalAveragePool`.
pub static ELEMENTWISE: &[TypeRelation] = &[TypeRelation::Elementwise {
    input: PortRef::Input(0),
    output: PortRef::Output(0),
}];

/// `input[0].ElementType == output[0].ElementType` without
/// constraining shape. Use for ops that preserve element type but
/// reshape (`Reshape`, `Transpose`, `Slice`, `Squeeze`, `Unsqueeze`).
pub static UNARY_SAME_ELEMENT: &[TypeRelation] = &[TypeRelation::SameElementType(&[
    PortRef::Input(0),
    PortRef::Output(0),
])];

/// Binary broadcast arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Pow`.
/// Inputs + output share element type AND the output's shape is
/// the broadcast of the two inputs.
pub static BROADCAST_BINARY: &[TypeRelation] = &[
    TypeRelation::SameElementType(&[PortRef::Input(0), PortRef::Input(1), PortRef::Output(0)]),
    TypeRelation::BroadcastShape {
        in0: PortRef::Input(0),
        in1: PortRef::Input(1),
        out: PortRef::Output(0),
    },
];

/// `MatMul` / `Dot` — both inputs and the output share an element
/// type; shape is matmul-specific and stays unconstrained until a
/// `Custom` shape relation lands.
pub static MATMUL_BINARY: &[TypeRelation] = &[TypeRelation::SameElementType(&[
    PortRef::Input(0),
    PortRef::Input(1),
    PortRef::Output(0),
])];

/// `ReduceSum` / `ReduceMean` / `ReduceMax` / `ReduceMin` — same
/// element type, reduced shape driven by `axes` / `keepdims`
/// attributes.
pub static REDUCE_AXIS: &[TypeRelation] = &[TypeRelation::ReduceOver {
    input: PortRef::Input(0),
    output: PortRef::Output(0),
}];
