//! The framework's curated floor of primitive tensor ops.
//!
//! Every `Backend` impl MUST declare an `atomic_opset()` whose ops
//! list contains every entry in [`TENSOR_PRIMITIVES_OPS`]. Ops not
//! in this set (Relu, Sigmoid, Tanh, Softmax, LeakyRelu, Gelu,
//! Conv, MaxPool, AveragePool, BatchNormalization,
//! LayerNormalization, …) are **extensions**: a backend MAY
//! support them via `extension_opsets()`; a graph using them
//! either binds to a backend that declares them OR (future work)
//! a lowering pass decomposes them into primitives.
//!
//! Naming rationale — the op-types here live in the `ai.onnx`
//! domain because that's where `Add`, `MatMul`, `Reshape`, etc.
//! are canonically named. The framework deliberately avoids
//! `ONNX_V1_*` / `onnx_v1` identifiers anywhere so users don't
//! read this floor as a claim to implement the ONNX v1
//! specification — the floor is OUR curation of primitives, not
//! the formal ONNX v1 catalog.

use crate::atomic::AtomicOpsetDecl;

/// Canonical opset domain for the primitive tensor ops. Same
/// string the upstream ONNX project uses for its op-type catalog.
pub const TENSOR_PRIMITIVES_DOMAIN: &str = "ai.onnx";

/// Version of the framework's primitive-tensor floor. Bumped when
/// the set changes meaningfully.
pub const TENSOR_PRIMITIVES_VERSION: i64 = 1;

/// 30 primitive tensor ops every `Backend` impl MUST declare.
///
/// Categories: arithmetic (6) + math (4) + linear algebra (1) +
/// reductions (4) + shape (9) + comparison (3) + conditional (1)
/// + creation (1) + indexing (1) = 30.
pub const TENSOR_PRIMITIVES_OPS: &[&str] = &[
    // Arithmetic (6)
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Neg",
    "Abs",
    // Math (4)
    "Sqrt",
    "Pow",
    "Exp",
    "Log",
    // Linear algebra (1)
    "MatMul",
    // Reductions (4)
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "ReduceMin",
    // Shape (9)
    "Reshape",
    "Transpose",
    "Concat",
    "Slice",
    "Split",
    "Squeeze",
    "Unsqueeze",
    "Identity",
    "Cast",
    // Comparison (3)
    "Equal",
    "Greater",
    "Less",
    // Conditional (1)
    "Where",
    // Creation (1)
    "Constant",
    // Indexing (1)
    "Gather",
];

/// Result of [`opset_covers_primitives`] when a backend's
/// declared opset is missing one or more entries from the floor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingPrimitives {
    /// The backend's `atomic_opset().domain`.
    pub backend_domain: &'static str,
    /// The backend's `atomic_opset().version`.
    pub backend_version: i64,
    /// Primitive op names absent from the backend's opset.
    /// Reported in [`TENSOR_PRIMITIVES_OPS`] declaration order.
    pub missing: Vec<&'static str>,
}

impl std::fmt::Display for MissingPrimitives {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "backend opset {}@v{} missing {} primitive op(s): {}",
            self.backend_domain,
            self.backend_version,
            self.missing.len(),
            self.missing.join(", "),
        )
    }
}

impl std::error::Error for MissingPrimitives {}

/// Confirm `opset` declares every primitive in
/// [`TENSOR_PRIMITIVES_OPS`]. Returns the list of missing names so
/// the caller surfaces a typed error instead of a flat boolean.
/// Ops in opsets with a non-`TENSOR_PRIMITIVES_DOMAIN` domain
/// don't count toward the check — primitives are sourced from the
/// canonical `ai.onnx` namespace.
pub fn opset_covers_primitives(opset: &AtomicOpsetDecl) -> Result<(), MissingPrimitives> {
    // Collect the backend's declared op names from the primitives
    // domain. Backends layer their non-primitive ops via
    // `extension_opsets()`; those aren't relevant here.
    let declared: std::collections::HashSet<&str> = if opset.domain == TENSOR_PRIMITIVES_DOMAIN {
        opset.ops.iter().map(|o| o.name).collect()
    } else {
        std::collections::HashSet::new()
    };

    let missing: Vec<&'static str> = TENSOR_PRIMITIVES_OPS
        .iter()
        .copied()
        .filter(|name| !declared.contains(*name))
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(MissingPrimitives {
            backend_domain: opset.domain,
            backend_version: opset.version,
            missing,
        })
    }
}

