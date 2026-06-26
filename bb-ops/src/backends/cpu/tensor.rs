//! `CpuTensor` — `Arc`-shared handle to a `CpuBackendBuffer` holding
//! an `ndarray::ArrayD<f32>`. Cloning the handle is an `Arc::clone`
//! (O(1) refcount bump); the underlying buffer is owned by the
//! backend and may be pooled or fresh-allocated by `CpuBackend`.
//!
//! Storage is `ndarray::ArrayD<f32>`: heap-dynamic rank, row-major,
//! with broadcasting + axis-walking primitives ndarray already
//! provides. Hot kernels downcast to `Ix2` / `Ix3` via
//! `.into_dimensionality::<...>()` for typed-dim performance and
//! return to `IxDyn` at the boundary.
//!
//! Phase C scope: f32 only. Backend-side extensions for f64 / i32 /
//! i64 / bool land via the optional `extension_opsets()`
//! declaration when those types are exercised.

use std::sync::Arc;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use bb_ir::proto::onnx::TensorProto;
use bb_ir::tensor::{Tensor, TensorSerializationError};
use bb_ir::types::TYPE_TENSOR_F32;
use bb_ir::{register_charged_bytes, register_type_node};
use bb_runtime::slot_value::SlotValue;
use ndarray::{ArrayD, IxDyn};

register_type_node!(CpuTensor, &TYPE_TENSOR_F32);
// Backend-mediated wire receive stamps the byte charge into the
// buffer; the slot-table writer reads it back through the default
// SlotValue::charged_bytes body so the engine can release the
// admission charge on overwrite / eviction.
register_charged_bytes!(CpuTensor, |t: &CpuTensor| t.0.charged_bytes);

/// ONNX `DataType::FLOAT` numeric tag.
pub const ONNX_FLOAT: i32 = 1;

/// Backend-owned buffer behind a [`CpuTensor`] handle. Holds the
/// `ndarray` storage plus the byte count charged against the
/// `NodeConfig::ingress_byte_budget` at materialization time so the
/// slot-table writer can release the charge on overwrite / eviction.
#[derive(Debug)]
pub struct CpuBackendBuffer {
    /// f32 storage, heap-dynamic rank, row-major.
    pub(crate) data: ArrayD<f32>,
    /// i64-typed shape cache (ONNX convention). Always equals
    /// `data.shape().iter().map(|&n| n as i64).collect()`.
    pub(crate) dims_i64: Vec<i64>,
    /// Wire-byte count charged at the ingress boundary; carriers
    /// holding this buffer surface this through
    /// `SlotValue::charged_bytes` for budget release on slot
    /// overwrite. Zero for tensors that did not arrive via the wire
    /// (kernel outputs, test fixtures).
    pub(crate) charged_bytes: usize,
}

/// f32-dense CPU-resident tensor handle. `Arc`-shared so intra-Node
/// clones (FedAvg's per-peer buffer insert, slot-table writes, etc.)
/// are refcount bumps rather than `Vec<f32>` deep copies. The
/// underlying [`CpuBackendBuffer`] is owned by the backend, which is
/// free to pool / reuse / free the storage at a later milestone
/// without API churn (the handle shape stays identical).
#[derive(Clone, Debug)]
pub struct CpuTensor(pub(crate) Arc<CpuBackendBuffer>);

/// Errors `CpuTensor::new_checked` and `from_proto` may return.
#[derive(Debug)]
pub enum CpuTensorError {
    /// The product of dims doesn't match `data.len()`. Surfaces
    /// from `new_checked` and the proto deserialization boundary.
    ShapeMismatch {
        /// Expected element count (product of dims).
        expected: usize,
        /// Observed element count.
        got: usize,
    },
}

impl std::fmt::Display for CpuTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => write!(
                f,
                "CpuTensor shape mismatch: dims product {expected} ≠ data.len {got}",
            ),
        }
    }
}

impl std::error::Error for CpuTensorError {}

impl CpuTensor {
    /// Wrap an existing `ArrayD<f32>` in a fresh backend buffer.
    /// `charged_bytes = 0` — kernel outputs and test fixtures don't
    /// arrive via the wire and therefore don't hold an ingress
    /// charge. The wire path uses the crate-private
    /// `from_wire_buffer` helper from `CpuBackend::materialize_from_wire`.
    pub fn from_array(data: ArrayD<f32>) -> Self {
        let dims_i64 = data.shape().iter().map(|&n| n as i64).collect();
        Self(Arc::new(CpuBackendBuffer {
            data,
            dims_i64,
            charged_bytes: 0,
        }))
    }

    /// Construct from ONNX-signed shape + flat row-major data via
    /// `ndarray::ArrayD::from_shape_vec`. Equivalent to
    /// [`Self::new`] but spelled out for callers preferring the
    /// builder-style name.
    pub fn from_vec(shape: Vec<i64>, data: Vec<f32>) -> Self {
        Self::new(shape, data)
    }

    /// Borrow the underlying ndarray.
    pub fn as_array(&self) -> &ArrayD<f32> {
        &self.0.data
    }

    /// Clone the backend buffer into an owned `ArrayD<f32>`. The
    /// `Arc` shape means the buffer cannot be unwrapped in-place
    /// (other handles may share it), so this always pays the
    /// `ndarray` deep copy. Test-only callers needing flat data
    /// should prefer [`Self::flat_data`].
    pub fn into_array(self) -> ArrayD<f32> {
        self.0.data.clone()
    }

    /// Test-helper that returns the cached i64 shape. Real callers
    /// use the [`Tensor::dims`] trait method or
    /// [`Self::as_array`]`.shape()`.
    #[doc(hidden)]
    pub fn dims_vec(&self) -> &[i64] {
        &self.0.dims_i64
    }

    /// Test-helper that materializes a flat row-major copy. Real
    /// callers iterate `self.as_array()` directly to avoid the
    /// allocation.
    #[doc(hidden)]
    pub fn flat_data(&self) -> Vec<f32> {
        self.0.data.iter().copied().collect()
    }

    /// Construct from ONNX-signed shape + flat row-major data.
    /// Panics if the dims product doesn't match the data length —
    /// callers needing strict checking use [`Self::new_checked`].
    pub fn new(dims: Vec<i64>, data: Vec<f32>) -> Self {
        let shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("CpuTensor::new shape × data mismatch");
        Self::from_array(array)
    }

    /// Construct + validate `dims_product(&dims) == data.len()`.
    pub fn new_checked(dims: Vec<i64>, data: Vec<f32>) -> Result<Self, CpuTensorError> {
        let expected = dims_product(&dims);
        if expected != data.len() {
            return Err(CpuTensorError::ShapeMismatch {
                expected,
                got: data.len(),
            });
        }
        let shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|_| CpuTensorError::ShapeMismatch { expected, got: 0 })?;
        Ok(Self::from_array(array))
    }

    /// Construct a zero-filled tensor with the given shape.
    pub fn zeros(dims: Vec<i64>) -> Self {
        let shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        Self::from_array(ArrayD::zeros(IxDyn(&shape)))
    }

    /// Construct a ones-filled tensor with the given shape.
    pub fn ones(dims: Vec<i64>) -> Self {
        let shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
        Self::from_array(ArrayD::ones(IxDyn(&shape)))
    }

    /// Observe the underlying `Arc<CpuBackendBuffer>` strong-refcount.
    /// One strong holder means the caller holds the only handle to
    /// the buffer; future pooling implementations (`v2`) read this to
    /// decide whether to return the buffer to the pool on drop. Tests
    /// use this to prove the wire-decode path lands a single carrier
    /// in the slot table with no spurious clones.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Wrap a kernel-supplied `ArrayD<f32>` plus a wire-byte charge
    /// in a fresh backend buffer. Used by
    /// `CpuBackend::materialize_from_wire` so the resulting tensor
    /// carries the charge that the slot-table writer releases on
    /// eviction.
    pub(crate) fn from_wire_buffer(data: ArrayD<f32>, charged_bytes: usize) -> Self {
        let dims_i64 = data.shape().iter().map(|&n| n as i64).collect();
        Self(Arc::new(CpuBackendBuffer {
            data,
            dims_i64,
            charged_bytes,
        }))
    }
}

fn dims_product(dims: &[i64]) -> usize {
    dims.iter().map(|d| (*d).max(0) as usize).product()
}

impl std::fmt::Display for CpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CpuTensor(dims={:?}, len={})",
            self.0.data.shape(),
            self.0.data.len(),
        )
    }
}

impl Tensor for CpuTensor {
    type Scalar = f32;

    fn dims(&self) -> &[i64] {
        &self.0.dims_i64
    }

    fn len(&self) -> usize {
        self.0.data.len()
    }

    fn to_proto(&self) -> TensorProto {
        let dims: Vec<i64> = self.0.data.shape().iter().map(|&n| n as i64).collect();
        let float_data: Vec<f32> = self.0.data.iter().copied().collect();
        TensorProto {
            dims,
            data_type: ONNX_FLOAT,
            float_data,
            ..Default::default()
        }
    }

    fn from_proto(proto: TensorProto) -> Result<Self, TensorSerializationError> {
        if proto.data_type != ONNX_FLOAT {
            return Err(TensorSerializationError::ElementTypeMismatch {
                expected: ONNX_FLOAT,
                found: proto.data_type,
            });
        }
        // Prefer `float_data`; fall back to raw bytes if shipping
        // tools encoded via `raw_data` (4-byte little-endian floats).
        let data = if !proto.float_data.is_empty() {
            proto.float_data
        } else if !proto.raw_data.is_empty() {
            if proto.raw_data.len() % 4 != 0 {
                return Err(TensorSerializationError::ShapeError(format!(
                    "raw_data length {} not divisible by 4",
                    proto.raw_data.len(),
                )));
            }
            let mut out = Vec::with_capacity(proto.raw_data.len() / 4);
            for chunk in proto.raw_data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out
        } else {
            Vec::new()
        };
        let expected = dims_product(&proto.dims);
        if expected != data.len() {
            return Err(TensorSerializationError::ShapeError(format!(
                "dims product {expected} doesn't match data len {len}",
                len = data.len()
            )));
        }
        let shape: Vec<usize> = proto.dims.iter().map(|&d| d.max(0) as usize).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
            TensorSerializationError::ShapeError(format!("ndarray::from_shape_vec: {e}"))
        })?;
        Ok(Self::from_array(array))
    }
}

// Hand-written Serialize / Deserialize: skip the `Arc` indirection
// on the wire (a remote receiver wants the buffer's contents, not
// the local refcount cell) and skip `charged_bytes` (a snapshot
// replay restarts ingress accounting from zero). Deserialization
// fresh-allocates a `CpuBackendBuffer` wrapped in a brand-new `Arc`.

impl Serialize for CpuTensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("CpuTensor", 2)?;
        s.serialize_field("data", &self.0.data)?;
        s.serialize_field("dims_i64", &self.0.dims_i64)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for CpuTensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Data,
            DimsI64,
        }

        struct CpuTensorVisitor;

        impl<'de> Visitor<'de> for CpuTensorVisitor {
            type Value = CpuTensor;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("struct CpuTensor")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let data: ArrayD<f32> = seq.next_element()?.ok_or_else(|| {
                    de::Error::invalid_length(0, &"struct CpuTensor with 2 fields")
                })?;
                let dims_i64: Vec<i64> = seq.next_element()?.ok_or_else(|| {
                    de::Error::invalid_length(1, &"struct CpuTensor with 2 fields")
                })?;
                Ok(CpuTensor(Arc::new(CpuBackendBuffer {
                    data,
                    dims_i64,
                    charged_bytes: 0,
                })))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut data: Option<ArrayD<f32>> = None;
                let mut dims_i64: Option<Vec<i64>> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        Field::DimsI64 => {
                            if dims_i64.is_some() {
                                return Err(de::Error::duplicate_field("dims_i64"));
                            }
                            dims_i64 = Some(map.next_value()?);
                        }
                    }
                }
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;
                let dims_i64 = dims_i64.ok_or_else(|| de::Error::missing_field("dims_i64"))?;
                Ok(CpuTensor(Arc::new(CpuBackendBuffer {
                    data,
                    dims_i64,
                    charged_bytes: 0,
                })))
            }
        }

        const FIELDS: &[&str] = &["data", "dims_i64"];
        deserializer.deserialize_struct("CpuTensor", FIELDS, CpuTensorVisitor)
    }
}

// `Tensor` implies the framework's blanket `SlotValue` impl so
// `CpuTensor` can be passed by-ref into `dispatch_atomic` inputs.
// The blanket lives in `bytesandbrains::tensor`; this re-export
// reminds readers that the contract is satisfied.
const _: fn() = || {
    fn _check<T: SlotValue>() {}
    _check::<CpuTensor>();
};

