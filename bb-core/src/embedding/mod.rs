pub mod f32_l2;
pub mod f32_cosine;

use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Rem, Sub},
};

pub use f32_cosine::*;
pub use f32_l2::*;

// Helper traits for conditional proto conversion bounds.
// When proto feature is enabled, these require the actual conversion traits.
// When disabled, they're blanket-implemented for all types.

#[cfg(feature = "proto")]
mod proto_bounds {
    pub trait DistanceProtoConvert:
        Into<crate::proto::DistanceProto>
        + TryFrom<crate::proto::DistanceProto, Error = crate::proto::ProtoConversionError>
    {
    }

    impl<T> DistanceProtoConvert for T where
        T: Into<crate::proto::DistanceProto>
            + TryFrom<crate::proto::DistanceProto, Error = crate::proto::ProtoConversionError>
    {
    }

    pub trait EmbeddingProtoConvert:
        Into<crate::proto::TensorProto>
        + TryFrom<crate::proto::TensorProto, Error = crate::proto::ProtoConversionError>
    {
    }

    impl<T> EmbeddingProtoConvert for T where
        T: Into<crate::proto::TensorProto>
            + TryFrom<crate::proto::TensorProto, Error = crate::proto::ProtoConversionError>
    {
    }
}

#[cfg(not(feature = "proto"))]
mod proto_bounds {
    pub trait DistanceProtoConvert {}
    impl<T> DistanceProtoConvert for T {}

    pub trait EmbeddingProtoConvert {}
    impl<T> EmbeddingProtoConvert for T {}
}

use proto_bounds::{DistanceProtoConvert, EmbeddingProtoConvert};

/// A totally-ordered distance metric value.
pub trait Distance:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Eq
    + Ord
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + fmt::Debug
    + From<f32>
    + Into<f32>
    + Hash
    + DistanceProtoConvert
{
    fn next_up(&self) -> Self;
    fn zero() -> Self;
    fn max_value() -> Self;
    fn min_value() -> Self;
}

/// A fixed-dimensionality embedding vector.
pub trait Embedding:
    'static + Clone + Hash + PartialEq + Eq + fmt::Debug + fmt::Display + EmbeddingProtoConvert
{
    type Scalar: 'static + Copy + Clone + Send + Sync;

    fn length() -> usize;
    fn as_slice(&self) -> &[Self::Scalar];
    fn from_slice(data: &[Self::Scalar]) -> Self;
    fn zeros() -> Self;
}

/// An embedding space defines the embedding type, distance metric, and space properties.
pub trait EmbeddingSpace: Clone + PartialEq + Eq + fmt::Debug + Send + Sync {
    type EmbeddingData: Embedding;
    type DistanceValue: Distance;
    /// Precomputed state for efficient distance queries.
    type Prepared: Clone;

    fn space_id(&self) -> &'static str;

    /// Compute distance between two embeddings.
    fn distance(&self, lhs: &Self::EmbeddingData, rhs: &Self::EmbeddingData) -> Self::DistanceValue;

    /// Prepare a query embedding for efficient repeated distance computations.
    fn prepare(&self, embedding: &Self::EmbeddingData) -> Self::Prepared;

    /// Compute distance using prepared query state.
    fn distance_prepared(
        &self,
        prepared: &Self::Prepared,
        target: &Self::EmbeddingData,
    ) -> Self::DistanceValue;

    fn length() -> usize;

    /// Maps a finite distance range to an infinite range (e.g., tan(pi * x / 4))
    fn infinite_mapping(native_distance: &Self::DistanceValue) -> f32;

    /// Compute the space's distance metric on raw scalar slices of arbitrary length.
    ///
    /// This is the same distance function as `distance()` but operates on raw `f32` slices
    /// rather than typed `EmbeddingData`. Used when the caller has subvectors or
    /// centroid slices that don't match the full embedding dimension.
    fn slice_distance(a: &[f32], b: &[f32]) -> f32;

    fn create_embedding(data: Vec<<Self::EmbeddingData as Embedding>::Scalar>) -> Self::EmbeddingData {
        Self::EmbeddingData::from_slice(&data)
    }

    fn create_distance(dist: f32) -> Self::DistanceValue {
        Self::DistanceValue::from(dist)
    }

    fn zero_vector() -> Self::EmbeddingData {
        Self::EmbeddingData::zeros()
    }

    fn zero_distance() -> Self::DistanceValue {
        Self::DistanceValue::zero()
    }
}

/// A total-ordered f32 distance wrapper. NaN is treated as greater than all
/// normal values, and NaN == NaN for consistency in ordered collections.
#[derive(Debug, Clone, Copy)]
pub struct F32Distance(pub f32);

impl F32Distance {
    pub fn new(val: f32) -> Self {
        F32Distance(val)
    }
    pub fn value(&self) -> f32 {
        self.0
    }
}

impl PartialEq for F32Distance {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_nan() && other.0.is_nan() {
            true
        } else {
            self.0 == other.0
        }
    }
}

impl Eq for F32Distance {}

impl PartialOrd for F32Distance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for F32Distance {
    fn cmp(&self, other: &F32Distance) -> std::cmp::Ordering {
        if self.0.is_nan() && other.0.is_nan() {
            std::cmp::Ordering::Equal
        } else if self.0.is_nan() {
            std::cmp::Ordering::Greater
        } else if other.0.is_nan() {
            std::cmp::Ordering::Less
        } else {
            self.0.partial_cmp(&other.0).unwrap()
        }
    }
}

impl Add for F32Distance {
    type Output = F32Distance;
    fn add(self, other: F32Distance) -> F32Distance {
        F32Distance(self.0 + other.0)
    }
}

impl Sub for F32Distance {
    type Output = F32Distance;
    fn sub(self, other: F32Distance) -> F32Distance {
        F32Distance(self.0 - other.0)
    }
}

impl Mul for F32Distance {
    type Output = F32Distance;
    fn mul(self, other: F32Distance) -> F32Distance {
        F32Distance(self.0 * other.0)
    }
}

impl Div for F32Distance {
    type Output = F32Distance;
    fn div(self, other: F32Distance) -> F32Distance {
        F32Distance(self.0 / other.0)
    }
}

impl Rem for F32Distance {
    type Output = F32Distance;
    fn rem(self, other: F32Distance) -> F32Distance {
        F32Distance(self.0 % other.0)
    }
}

impl From<f32> for F32Distance {
    fn from(value: f32) -> Self {
        F32Distance(value)
    }
}

impl From<F32Distance> for f32 {
    fn from(distance: F32Distance) -> f32 {
        distance.0
    }
}

impl Hash for F32Distance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Distance for F32Distance {
    fn next_up(&self) -> Self {
        F32Distance(self.0.next_up())
    }
    fn zero() -> Self {
        F32Distance(0.0)
    }
    fn max_value() -> Self {
        F32Distance(f32::MAX)
    }
    fn min_value() -> Self {
        F32Distance(f32::MIN)
    }
}

/// A const-generic fixed-size f32 embedding vector.
#[derive(Clone, PartialEq, Debug)]
pub struct F32Embedding<const L: usize>(pub [f32; L]);

impl<const L: usize> fmt::Display for F32Embedding<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F32Embedding({:?})", self.0)
    }
}

impl<const L: usize> Hash for F32Embedding<L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &value in &self.0 {
            state.write(&value.to_le_bytes());
        }
    }
}

impl<const L: usize> Eq for F32Embedding<L> {}

impl<const L: usize> Embedding for F32Embedding<L> {
    type Scalar = f32;

    fn length() -> usize {
        L
    }

    fn as_slice(&self) -> &[Self::Scalar] {
        &self.0
    }

    fn from_slice(data: &[Self::Scalar]) -> Self {
        let mut array = [0.0; L];
        let copy_len = data.len().min(L);
        array[..copy_len].copy_from_slice(&data[..copy_len]);
        F32Embedding(array)
    }

    fn zeros() -> Self {
        F32Embedding([0.0; L])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── F32Distance tests ───────────────────────────────────────────────

    #[test]
    fn test_f32_distance_creation() {
        let dist = F32Distance(5.0);
        assert_eq!(dist.value(), 5.0);
    }

    #[test]
    fn test_f32_distance_arithmetic() {
        let a = F32Distance(10.0);
        let b = F32Distance(3.0);

        assert_eq!(a + b, F32Distance(13.0));
        assert_eq!(a - b, F32Distance(7.0));
        assert_eq!(a * b, F32Distance(30.0));
        assert_eq!(a / b, F32Distance(10.0 / 3.0));
        assert_eq!(a % b, F32Distance(1.0));
    }

    #[test]
    fn test_f32_distance_ordering() {
        let a = F32Distance(1.0);
        let b = F32Distance(2.0);
        let c = F32Distance(1.0);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
        assert!(a <= c);
        assert!(a >= c);
    }

    #[test]
    fn test_f32_distance_nan_ordering() {
        let nan = F32Distance(f32::NAN);
        let num = F32Distance(5.0);
        let nan2 = F32Distance(f32::NAN);

        assert!(nan > num);
        assert_eq!(nan.cmp(&nan2), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_f32_distance_conversions() {
        let dist: F32Distance = 42.5f32.into();
        assert_eq!(dist.value(), 42.5);

        let float_val: f32 = dist.into();
        assert_eq!(float_val, 42.5);
    }

    #[test]
    fn test_distance_trait_methods() {
        assert_eq!(F32Distance::zero(), F32Distance(0.0));
        assert_eq!(F32Distance::max_value(), F32Distance(f32::MAX));
        assert_eq!(F32Distance::min_value(), F32Distance(f32::MIN));

        let dist = F32Distance(1.0);
        let next = dist.next_up();
        assert!(next.value() > dist.value());
    }


    #[test]
    fn test_f32_embedding_creation() {
        let embedding = F32Embedding::<3>::zeros();
        assert_eq!(embedding.as_slice(), &[0.0, 0.0, 0.0]);
        assert_eq!(F32Embedding::<3>::length(), 3);
    }

    #[test]
    fn test_f32_embedding_from_slice() {
        let data = [1.0, 2.0, 3.0];
        let embedding = F32Embedding::<3>::from_slice(&data);
        assert_eq!(embedding.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_f32_embedding_from_partial_slice() {
        let data = [1.0, 2.0];
        let embedding = F32Embedding::<3>::from_slice(&data);
        assert_eq!(embedding.as_slice(), &[1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_f32_embedding_from_oversized_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let embedding = F32Embedding::<3>::from_slice(&data);
        assert_eq!(embedding.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_f32_embedding_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let embedding1 = F32Embedding::<2>([1.0, 2.0]);
        let embedding2 = F32Embedding::<2>([1.0, 2.0]);
        let embedding3 = F32Embedding::<2>([2.0, 1.0]);

        map.insert(embedding1.clone(), "value1");
        map.insert(embedding3, "value3");

        assert_eq!(map.get(&embedding2), Some(&"value1"));
        assert_eq!(map.len(), 2);
    }
}
