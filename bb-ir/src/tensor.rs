//! Tensor + Scalar abstractions. The framework crate ships no
//! concrete tensor type; backends implement these traits over their
//! own storage of choice.
//!
//! - `Scalar` - every scalar projects to `f32` for cross-backend
//!   interop. Framework ships universal impls for `f32` + `f64`;
//!   backends add others.
//! - `Tensor` - the contract every backend's concrete tensor type
//!   implements: shape + total length + canonical ONNX
//!   `TensorProto` round-trip.

use crate::proto::onnx::TensorProto;

// ---------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------

/// A scalar value usable in tensors. Every scalar projects to
/// `f32` for cross-backend interop.
///
/// Backends pick their own concrete scalar set; the framework
/// ships universal primitive impls for `f32` + `f64` only.
/// Concrete backends may add impls for `i32`, `i64`, `u32`, `u64`,
/// `bool`, `f16` (bf16, fp8 variants), etc.
pub trait Scalar: Copy + Send + Sync + 'static {
    /// Projection to `f32`. Lossy for wider types (f64, i64);
    /// faithful for f32 + smaller integers.
    fn to_f32(&self) -> f32;
}

impl Scalar for f32 {
    fn to_f32(&self) -> f32 {
        *self
    }
}

impl Scalar for f64 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

// ---------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------

/// The contract every backend's concrete tensor type implements.
///
/// Backend impls land in integration crates (e.g. `bb-cpu-onnx`).
/// The framework crate ships no concrete tensor type - the
/// `Tensor` trait IS the contract.
///
/// The serde + `Clone` bounds are what make every concrete tensor
/// type a [`crate::slot_value::SlotValue`] via the universal
/// blanket - tensors ride slots, wire envelopes, and snapshots
/// through the same bincode encoding path as every other value.
pub trait Tensor:
    Clone
    + std::fmt::Debug
    + std::fmt::Display
    + Send
    + Sync
    + 'static
    + serde::Serialize
    + serde::de::DeserializeOwned
{
    /// The scalar element type this tensor holds.
    type Scalar: Scalar;

    /// Tensor shape. ONNX-compatible signed-dim convention; `-1`
    /// for dynamic dims.
    fn dims(&self) -> &[i64];

    /// Total element count across all dims. For dynamic-dim
    /// tensors callers must resolve concrete dims before consulting.
    fn len(&self) -> usize;

    /// `true` when the tensor holds zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize to canonical ONNX `TensorProto`. The result is
    /// portable across backends declaring the same scalar type.
    fn to_proto(&self) -> TensorProto;

    /// Deserialize from canonical ONNX `TensorProto`. Returns an
    /// error if the proto's `elem_type` / shape doesn't match
    /// `Self`'s expectations.
    fn from_proto(proto: TensorProto) -> Result<Self, TensorSerializationError>;
}

/// Errors surfaced by `Tensor::from_proto`.
#[derive(Debug)]
pub enum TensorSerializationError {
    /// Proto's elem_type didn't match the impl's expected scalar.
    ElementTypeMismatch {
        /// What the impl expected (ONNX `DataType` enum value).
        expected: i32,
        /// What the proto held.
        found: i32,
    },
    /// Proto's shape couldn't be interpreted as the impl's tensor
    /// layout (e.g. byte-count mismatch, malformed dim list).
    ShapeError(String),
    /// Impl-specific deserialization failure.
    Custom(String),
}

impl std::fmt::Display for TensorSerializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElementTypeMismatch { expected, found } => {
                write!(
                    f,
                    "tensor elem_type mismatch: expected {expected}, found {found}"
                )
            }
            Self::ShapeError(m) => write!(f, "tensor shape error: {m}"),
            Self::Custom(m) => write!(f, "tensor serialization failure: {m}"),
        }
    }
}

impl std::error::Error for TensorSerializationError {}

