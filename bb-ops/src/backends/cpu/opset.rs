//! Opset declaration for the `CpuBackend`.
//!
//! The backend's `atomic_opset` mirrors
//! `bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS` exactly — the
//! 30-op primitive floor every `Backend` impl must declare. The
//! ops that aren't primitives but ARE backed by ndarray kernels in
//! this crate (Relu, Sigmoid, Tanh, Softmax, LeakyRelu, Gelu, Dot,
//! Zeros, Ones, GlobalAveragePool) get listed via
//! `extension_opsets()`. Lying entries the prior 49-op declaration
//! carried (BatchNorm, LayerNorm, Conv, MaxPool, AveragePool,
//! Scatter, If, Loop) are dropped — they have no kernel.
//!
//! `BackendSubgraph` is the framework's collapse-carrier op; it
//! lives in `ai.bytesandbrains.framework` and routes through
//! `invoke_backend_subgraph`, not this opset.
//!
//! Each entry carries `type_relations` so the TypeSolver narrows
//! the participating values' TypeNodes. The canonical relation
//! slices live in `bb_ir::types::common_relations`.

use bb_ir::types::{
    common_relations::{
        BROADCAST_BINARY, ELEMENTWISE, MATMUL_BINARY, NO_RELATIONS, REDUCE_AXIS, UNARY_SAME_ELEMENT,
    },
    relations::TypeRelation,
};
use bb_runtime::atomic::{AtomicOpDecl, AtomicOpKind, AtomicOpsetDecl};

/// `ai.onnx` opset domain — primitives + extension ops.
pub const ONNX_DOMAIN: &str = "ai.onnx";

/// Primitive-floor opset version.
pub const ONNX_VERSION: i64 = 1;

/// Backend-shipped extension version. Separate from the primitive
/// floor so a future opset bump on either side stays independent.
pub const EXTENSION_VERSION: i64 = 1;

/// Opset domain for the activations + creation + indexing extras
/// the CpuBackend ships. Same canonical `ai.onnx` namespace; the
/// distinct *opset* (different version) keeps the floor + extras
/// inspectable as separate declarations.
pub const EXTENSION_DOMAIN: &str = "ai.onnx";

/// 30-entry primitive-floor opset returned by
/// `BackendRuntime::atomic_opset`. Matches
/// `bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS` element-for-
/// element.
pub static PRIMITIVE_OPS: &[AtomicOpDecl] = &[
    // Arithmetic (6) — broadcast binary on same element type.
    op("Add", BROADCAST_BINARY),
    op("Sub", BROADCAST_BINARY),
    op("Mul", BROADCAST_BINARY),
    op("Div", BROADCAST_BINARY),
    op("Neg", ELEMENTWISE),
    op("Abs", ELEMENTWISE),
    // Math (4) — Sqrt/Exp/Log are elementwise; Pow is broadcast binary.
    op("Sqrt", ELEMENTWISE),
    op("Pow", BROADCAST_BINARY),
    op("Exp", ELEMENTWISE),
    op("Log", ELEMENTWISE),
    // Linear algebra (1) — same element type; shape is matmul-specific.
    op("MatMul", MATMUL_BINARY),
    // Reductions (4).
    op("ReduceSum", REDUCE_AXIS),
    op("ReduceMean", REDUCE_AXIS),
    op("ReduceMax", REDUCE_AXIS),
    op("ReduceMin", REDUCE_AXIS),
    // Shape (9) — Reshape/Transpose/Slice/Squeeze/Unsqueeze preserve
    // element type, change shape; Identity is a true pass-through;
    // Concat/Split are variadic + Cast is attribute-driven, all left
    // unconstrained until a Custom relation lands.
    op("Reshape", UNARY_SAME_ELEMENT),
    op("Transpose", UNARY_SAME_ELEMENT),
    op("Concat", NO_RELATIONS),
    op("Slice", UNARY_SAME_ELEMENT),
    op("Split", NO_RELATIONS),
    op("Squeeze", UNARY_SAME_ELEMENT),
    op("Unsqueeze", UNARY_SAME_ELEMENT),
    op("Identity", ELEMENTWISE),
    op("Cast", NO_RELATIONS),
    // Comparison (3) — element-wise inputs share type; output is
    // bool. Leave unconstrained until the lattice ships a `bool`
    // tensor leaf.
    op("Equal", NO_RELATIONS),
    op("Greater", NO_RELATIONS),
    op("Less", NO_RELATIONS),
    // Conditional (1).
    op("Where", NO_RELATIONS),
    // Creation (1) — value comes from an embedded `TensorProto`
    // attribute, so the type is attribute-driven.
    op("Constant", NO_RELATIONS),
    // Indexing (1) — Gather mixes tensor + index types.
    op("Gather", NO_RELATIONS),
];

/// Non-primitive ops the CpuBackend backs with ndarray kernels.
/// Surfaces via `BackendRuntime::extension_opsets()` so the
/// install-time check classifies them correctly (they're NOT in
/// the primitive floor; users who bind a different backend may
/// not get them).
pub static EXTENSION_OPS: &[AtomicOpDecl] = &[
    // Activations — pure element-wise; element type + shape
    // preserved.
    op("Relu", ELEMENTWISE),
    op("Sigmoid", ELEMENTWISE),
    op("Tanh", ELEMENTWISE),
    op("Softmax", ELEMENTWISE),
    op("LeakyRelu", ELEMENTWISE),
    op("Gelu", ELEMENTWISE),
    // Linear algebra extras — same element type across operands;
    // shape is matmul-specific. `Gemm` takes an optional bias
    // (3-input variadic) so its element-type relation is captured
    // by the 2-operand `MATMUL_BINARY` and the optional `c` falls
    // out of the constraint until a Custom relation lands.
    op("Dot", MATMUL_BINARY),
    op("Gemm", MATMUL_BINARY),
    // Creation extras — attribute-driven shape; element type
    // determined by the `dtype` attribute (defaulted to f32 today).
    op("Zeros", NO_RELATIONS),
    op("Ones", NO_RELATIONS),
    // Pooling — element type preserved; spatial dims collapse, but
    // the element-type constraint holds via `ELEMENTWISE`.
    op("GlobalAveragePool", ELEMENTWISE),
];

/// Primitive-floor opset declaration. Returned by
/// `CpuBackend::atomic_opset()`.
pub const ONNX_V1_OPSET: AtomicOpsetDecl = AtomicOpsetDecl {
    domain: ONNX_DOMAIN,
    version: ONNX_VERSION,
    ops: PRIMITIVE_OPS,
};

/// Extension opset declaration. Returned alongside the primitive
/// floor by `CpuBackend::extension_opsets()`.
pub const EXTENSION_OPSET: AtomicOpsetDecl = AtomicOpsetDecl {
    domain: EXTENSION_DOMAIN,
    version: EXTENSION_VERSION,
    ops: EXTENSION_OPS,
};

const fn op(name: &'static str, type_relations: &'static [TypeRelation]) -> AtomicOpDecl {
    AtomicOpDecl {
        name,
        inputs: &[],
        outputs: &[],
        kind: AtomicOpKind::Immediate,
        type_relations,
    }
}

