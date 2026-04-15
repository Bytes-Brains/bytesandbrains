use std::fmt;
use std::hash::{Hash, Hasher};

use bb_core::embedding::Embedding;

/// Encoded representation from TurboQuant scalar quantization.
///
/// Stores D quantization indices (one per coordinate in the rotated space),
/// each as a single byte supporting up to 8 bits per coordinate.
///
/// - D: embedding dimension (must match the TurboQuant's embedding space)
///
/// Total storage: D bytes. For nbits 1-4, each index fits in a u8.
/// Bit-packing for sub-byte widths can be added as a future optimization.
#[derive(Clone, PartialEq, Eq)]
pub struct TQCode<const D: usize> {
    pub indices: [u8; D],
}

impl<const D: usize> TQCode<D> {
    /// Create a zeroed code.
    pub fn zeros() -> Self {
        Self { indices: [0u8; D] }
    }

    /// Create from a slice of indices.
    pub fn from_indices(data: &[u8]) -> Self {
        let mut indices = [0u8; D];
        let copy_len = data.len().min(D);
        indices[..copy_len].copy_from_slice(&data[..copy_len]);
        Self { indices }
    }

    /// Get index at coordinate j.
    #[inline]
    pub fn get(&self, j: usize) -> u8 {
        self.indices[j]
    }

    /// Set index at coordinate j.
    #[inline]
    pub fn set(&mut self, j: usize, value: u8) {
        self.indices[j] = value;
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        D
    }
}

impl<const D: usize> fmt::Debug for TQCode<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TQCode")
            .field("D", &D)
            .field("indices[..8]", &&self.indices[..D.min(8)])
            .finish()
    }
}

impl<const D: usize> fmt::Display for TQCode<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TQCode<{}>({}B)", D, D)
    }
}

impl<const D: usize> Hash for TQCode<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.indices.hash(state);
    }
}

impl<const D: usize> Embedding for TQCode<D> {
    type Scalar = u8;

    fn length() -> usize {
        D
    }

    fn as_slice(&self) -> &[Self::Scalar] {
        &self.indices
    }

    fn from_slice(data: &[Self::Scalar]) -> Self {
        Self::from_indices(data)
    }

    fn zeros() -> Self {
        Self::zeros()
    }
}

#[cfg(feature = "proto")]
impl<const D: usize> From<TQCode<D>> for bb_core::proto::TensorProto {
    fn from(code: TQCode<D>) -> Self {
        bb_core::proto::TensorProto {
            dims: vec![D as i64],
            data_type: bb_core::proto::DATA_TYPE_UINT8,
            raw_data: code.indices.to_vec(),
            ..Default::default()
        }
    }
}

#[cfg(feature = "proto")]
impl<const D: usize> TryFrom<bb_core::proto::TensorProto> for TQCode<D> {
    type Error = bb_core::proto::ProtoConversionError;

    fn try_from(proto: bb_core::proto::TensorProto) -> Result<Self, Self::Error> {
        use bb_core::proto::{ProtoConversionError, DATA_TYPE_UINT8};

        if proto.data_type != DATA_TYPE_UINT8 {
            return Err(ProtoConversionError::InvalidDataType {
                expected: DATA_TYPE_UINT8,
                actual: proto.data_type,
            });
        }

        if proto.raw_data.len() != D {
            return Err(ProtoConversionError::ConversionFailed(format!(
                "Expected {} bytes, got {}",
                D,
                proto.raw_data.len()
            )));
        }

        Ok(Self::from_indices(&proto.raw_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tq_code_creation() {
        let code = TQCode::<8>::from_indices(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(code.get(0), 1);
        assert_eq!(code.get(7), 8);
        assert_eq!(code.dim(), 8);
    }

    #[test]
    fn test_tq_code_zeros() {
        let code = TQCode::<4>::zeros();
        assert_eq!(code.indices, [0, 0, 0, 0]);
    }

    #[test]
    fn test_tq_code_set() {
        let mut code = TQCode::<4>::zeros();
        code.set(2, 15);
        assert_eq!(code.get(2), 15);
        assert_eq!(code.get(0), 0);
    }

    #[test]
    fn test_tq_code_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        let code1 = TQCode::<4>::from_indices(&[1, 2, 3, 4]);
        let code2 = TQCode::<4>::from_indices(&[1, 2, 3, 4]);
        let code3 = TQCode::<4>::from_indices(&[4, 3, 2, 1]);

        map.insert(code1.clone(), "a");
        map.insert(code3, "b");

        assert_eq!(map.get(&code2), Some(&"a"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_tq_code_embedding_trait() {
        assert_eq!(TQCode::<8>::length(), 8);

        let code = TQCode::<4>::from_indices(&[10, 20, 30, 40]);
        assert_eq!(code.as_slice(), &[10, 20, 30, 40]);
    }

    #[test]
    fn test_tq_code_partial_slice() {
        // from_slice with shorter data pads with zeros
        let code = TQCode::<4>::from_indices(&[1, 2]);
        assert_eq!(code.indices, [1, 2, 0, 0]);
    }
}
