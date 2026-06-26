//! `Storage` — static link between a Rust storage type and its IR
//! `TypeNode`. Library makers declare *where in the polymorphism tree*
//! their concrete sits by picking the `Storage` impl for their
//! associated type.

use super::builtins::{
    TYPE_SCALAR_F16, TYPE_SCALAR_F32, TYPE_SCALAR_F64, TYPE_SCALAR_I32, TYPE_SCALAR_U8,
    TYPE_TENSOR, TYPE_TENSOR_BF16, TYPE_TENSOR_BOOL, TYPE_TENSOR_F16, TYPE_TENSOR_F32,
    TYPE_TENSOR_F64, TYPE_TENSOR_I16, TYPE_TENSOR_I32, TYPE_TENSOR_I64, TYPE_TENSOR_I8,
    TYPE_TENSOR_U16, TYPE_TENSOR_U32, TYPE_TENSOR_U64, TYPE_TENSOR_U8,
};
use super::TypeNode;

/// Static link between a Rust storage type and its IR `TypeNode`.
///
/// Library makers declare what wire format a backend / index / model
/// natively understands. The framework reads `Storage::TYPE` at
/// recording time (and at slot binding time) to stamp every Contract
/// port's `value_info` — the compiler's type solver then walks the
/// graph and refuses any mismatch.
pub trait Storage: Send + Sync + 'static {
    /// Position-in-tree declaration. The `TypeNode` static this
    /// constant points at decides what other storage types unify with
    /// this one during the type-solver walk.
    const TYPE: &'static TypeNode;
}

// --- Tensor (slice) leaf impls --------------------------------------

impl Storage for [f32] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_F32;
}
impl Storage for [f64] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_F64;
}
// Half-precision float storage uses `half::f16` (canonical Rust newtype
// around the binary16 bit pattern). The earlier polymorphic-storage
// design conflated `[u16]` with packed half, but that prevented u16
// from being a first-class unsigned-integer tensor type. Both are
// distinct positions in the polymorphism tree now.
impl Storage for [half::f16] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_F16;
}
impl Storage for [half::bf16] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_BF16;
}
impl Storage for [u8] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_U8;
}
impl Storage for [u16] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_U16;
}
impl Storage for [u32] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_U32;
}
impl Storage for [u64] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_U64;
}
impl Storage for [i8] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_I8;
}
impl Storage for [i16] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_I16;
}
impl Storage for [i32] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_I32;
}
impl Storage for [i64] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_I64;
}
impl Storage for [bool] {
    const TYPE: &'static TypeNode = &TYPE_TENSOR_BOOL;
}

// --- Scalar leaf impls ----------------------------------------------

impl Storage for f32 {
    const TYPE: &'static TypeNode = &TYPE_SCALAR_F32;
}
impl Storage for f64 {
    const TYPE: &'static TypeNode = &TYPE_SCALAR_F64;
}
// Scalar half stays as `u16` (single-element bit pattern) for now —
// scalar variants don't yet have their own `half::f16` impl because
// scalar polymorphism isn't as load-bearing as tensor polymorphism.
impl Storage for u16 {
    const TYPE: &'static TypeNode = &TYPE_SCALAR_F16;
}
impl Storage for u8 {
    const TYPE: &'static TypeNode = &TYPE_SCALAR_U8;
}
impl Storage for i32 {
    const TYPE: &'static TypeNode = &TYPE_SCALAR_I32;
}

// --- Generic-position storage ---------------------------------------

/// Concrete-erased tensor. Stores raw bytes plus a runtime-known
/// dtype + shape. Compute-outsourcing concretes (an index that
/// delegates distance math to a bound Backend) declare
/// `type Vector = AnyTensor` — `Storage::TYPE = &TYPE_TENSOR` puts
/// the value at a non-leaf position in the tree, so any tensor
/// subtype unifies into it.
#[derive(Clone, Debug)]
pub struct AnyTensor {
    /// Raw little-endian bytes of the tensor payload, packed per `dtype`.
    pub bytes: Vec<u8>,
    /// Runtime dtype. Pair with `Self::shape` for full interpretation.
    pub dtype: Dtype,
    /// Per-axis shape. `shape.iter().product::<usize>()` × dtype-size
    /// is expected to equal `bytes.len()`.
    pub shape: Vec<usize>,
}

impl Storage for AnyTensor {
    const TYPE: &'static TypeNode = &TYPE_TENSOR;
}

/// Runtime dtype tag for `AnyTensor` and any caller that needs to
/// dispatch over the framework's known tensor dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit IEEE 754 float — corresponds to `TYPE_TENSOR_F32`.
    F32,
    /// 64-bit IEEE 754 float — corresponds to `TYPE_TENSOR_F64`.
    F64,
    /// 16-bit IEEE 754 half — corresponds to `TYPE_TENSOR_F16`. Stored as `half::f16`.
    F16,
    /// 16-bit bfloat (Google's brain-float) — corresponds to `TYPE_TENSOR_BF16`. Stored as `half::bf16`.
    BF16,
    /// 8-bit unsigned — corresponds to `TYPE_TENSOR_U8`.
    U8,
    /// 16-bit unsigned — corresponds to `TYPE_TENSOR_U16`.
    U16,
    /// 32-bit unsigned — corresponds to `TYPE_TENSOR_U32`.
    U32,
    /// 64-bit unsigned — corresponds to `TYPE_TENSOR_U64`.
    U64,
    /// 8-bit signed — corresponds to `TYPE_TENSOR_I8` (edge quantization).
    I8,
    /// 16-bit signed — corresponds to `TYPE_TENSOR_I16`.
    I16,
    /// 32-bit signed — corresponds to `TYPE_TENSOR_I32`.
    I32,
    /// 64-bit signed — corresponds to `TYPE_TENSOR_I64` (indices, embeddings).
    I64,
    /// 1-bit boolean — corresponds to `TYPE_TENSOR_BOOL` (masks, predicates).
    Bool,
}

impl Dtype {
    /// Return the framework `TypeNode` static for this dtype.
    pub fn type_node(self) -> &'static TypeNode {
        match self {
            Dtype::F32 => &TYPE_TENSOR_F32,
            Dtype::F64 => &TYPE_TENSOR_F64,
            Dtype::F16 => &TYPE_TENSOR_F16,
            Dtype::BF16 => &TYPE_TENSOR_BF16,
            Dtype::U8 => &TYPE_TENSOR_U8,
            Dtype::U16 => &TYPE_TENSOR_U16,
            Dtype::U32 => &TYPE_TENSOR_U32,
            Dtype::U64 => &TYPE_TENSOR_U64,
            Dtype::I8 => &TYPE_TENSOR_I8,
            Dtype::I16 => &TYPE_TENSOR_I16,
            Dtype::I32 => &TYPE_TENSOR_I32,
            Dtype::I64 => &TYPE_TENSOR_I64,
            Dtype::Bool => &TYPE_TENSOR_BOOL,
        }
    }
}
