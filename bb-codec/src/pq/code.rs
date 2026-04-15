use std::fmt;
use std::hash::{Hash, Hasher};

use bb_core::embedding::Embedding;

/// Bytes needed for NBITS (compile-time ceil(NBITS/8))
pub const fn bytes_for_nbits(nbits: usize) -> usize {
    (nbits + 7) / 8
}

/// Encoded representation from Product Quantization.
///
/// Stores M centroid indices, each using ceil(NBITS/8) bytes (byte-aligned).
///
/// - M: number of subquantizers
/// - NBITS: bits per centroid index
///
/// Examples:
/// - `PQCode<8, 8>`: 8 subquantizers, 256 centroids each, 8 bytes total
/// - `PQCode<16, 10>`: 16 subquantizers, 1024 centroids each, 32 bytes total
#[derive(Clone, PartialEq, Eq)]
pub struct PQCode<const M: usize, const NBITS: usize>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    /// Raw byte storage: M * ceil(NBITS/8) bytes, stored as [[u8; B]; M]
    pub codes: [[u8; bytes_for_nbits(NBITS)]; M],
}

impl<const M: usize, const NBITS: usize> PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    /// Number of centroids per subquantizer (2^NBITS)
    pub const KSUB: usize = 1 << NBITS;

    /// Bytes per centroid index
    pub const BYTES_PER_CODE: usize = bytes_for_nbits(NBITS);

    /// Total bytes for all M codes
    pub const TOTAL_BYTES: usize = M * Self::BYTES_PER_CODE;

    /// Create a new PQCode from raw byte array
    pub fn new(codes: [[u8; bytes_for_nbits(NBITS)]; M]) -> Self {
        Self { codes }
    }

    /// Get centroid index at position m as u32
    pub fn get(&self, m: usize) -> u32 {
        let mut value = 0u32;
        for (i, &byte) in self.codes[m].iter().enumerate() {
            value |= (byte as u32) << (i * 8);
        }
        value
    }

    /// Set centroid index at position m
    pub fn set(&mut self, m: usize, value: u32) {
        for i in 0..Self::BYTES_PER_CODE {
            self.codes[m][i] = ((value >> (i * 8)) & 0xFF) as u8;
        }
    }

    /// Number of subquantizers
    pub fn m(&self) -> usize {
        M
    }

    /// Create a zeroed PQCode
    pub fn zeros() -> Self {
        Self { codes: [[0u8; bytes_for_nbits(NBITS)]; M] }
    }

    /// Create from individual centroid indices
    pub fn from_indices(indices: &[u32]) -> Self {
        let mut code = Self::zeros();
        for (m, &idx) in indices.iter().take(M).enumerate() {
            code.set(m, idx);
        }
        code
    }
}

impl<const M: usize, const NBITS: usize> fmt::Debug for PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indices: Vec<u32> = (0..M).map(|m| self.get(m)).collect();
        f.debug_struct("PQCode")
            .field("M", &M)
            .field("NBITS", &NBITS)
            .field("indices", &indices)
            .finish()
    }
}

impl<const M: usize, const NBITS: usize> fmt::Display for PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indices: Vec<u32> = (0..M).map(|m| self.get(m)).collect();
        write!(f, "PQCode<{}, {}>({:?})", M, NBITS, indices)
    }
}

impl<const M: usize, const NBITS: usize> Hash for PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.codes.hash(state);
    }
}

impl<const M: usize, const NBITS: usize> PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    /// Compile-time assertion: verify [[u8; B]; M] has same size as [u8; M*B]
    /// This ensures our unsafe transmutation in Embedding impl is sound.
    const _SIZE_CHECK: () = assert!(
        std::mem::size_of::<[[u8; bytes_for_nbits(NBITS)]; M]>() == M * bytes_for_nbits(NBITS)
    );
}

impl<const M: usize, const NBITS: usize> Embedding for PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    type Scalar = u8;

    fn length() -> usize {
        Self::TOTAL_BYTES
    }

    fn as_slice(&self) -> &[Self::Scalar] {
        // Safe: [[u8; B]; M] has same layout as [u8; M*B] (verified by _SIZE_CHECK)
        let _ = Self::_SIZE_CHECK;
        unsafe {
            std::slice::from_raw_parts(
                self.codes.as_ptr() as *const u8,
                Self::TOTAL_BYTES
            )
        }
    }

    fn from_slice(data: &[Self::Scalar]) -> Self {
        let _ = Self::_SIZE_CHECK;
        let mut codes = [[0u8; bytes_for_nbits(NBITS)]; M];
        let total_bytes = M * bytes_for_nbits(NBITS);
        let copy_len = data.len().min(total_bytes);
        // Safe: [[u8; B]; M] has same layout as [u8; M*B] (verified by _SIZE_CHECK)
        let flat = unsafe {
            std::slice::from_raw_parts_mut(
                codes.as_mut_ptr() as *mut u8,
                total_bytes
            )
        };
        flat[..copy_len].copy_from_slice(&data[..copy_len]);
        Self { codes }
    }

    fn zeros() -> Self {
        Self::zeros()
    }
}

#[cfg(feature = "proto")]
impl<const M: usize, const NBITS: usize> From<PQCode<M, NBITS>> for bb_core::proto::TensorProto
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn from(code: PQCode<M, NBITS>) -> Self {
        bb_core::proto::TensorProto {
            dims: vec![M as i64, PQCode::<M, NBITS>::BYTES_PER_CODE as i64],
            data_type: bb_core::proto::DATA_TYPE_UINT8,
            raw_data: code.as_slice().to_vec(),
            ..Default::default()
        }
    }
}

#[cfg(feature = "proto")]
impl<const M: usize, const NBITS: usize> TryFrom<bb_core::proto::TensorProto> for PQCode<M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    type Error = bb_core::proto::ProtoConversionError;

    fn try_from(proto: bb_core::proto::TensorProto) -> Result<Self, Self::Error> {
        use bb_core::proto::{ProtoConversionError, DATA_TYPE_UINT8};

        if proto.data_type != DATA_TYPE_UINT8 {
            return Err(ProtoConversionError::InvalidDataType {
                expected: DATA_TYPE_UINT8,
                actual: proto.data_type,
            });
        }

        let expected_dims = vec![M as i64, Self::BYTES_PER_CODE as i64];
        if proto.dims != expected_dims {
            return Err(ProtoConversionError::InvalidTensorShape {
                expected: expected_dims,
                actual: proto.dims,
            });
        }

        if proto.raw_data.len() != Self::TOTAL_BYTES {
            return Err(ProtoConversionError::ConversionFailed(format!(
                "Expected {} bytes in TensorProto raw_data, got {}",
                Self::TOTAL_BYTES,
                proto.raw_data.len()
            )));
        }

        Ok(Self::from_slice(&proto.raw_data))
    }
}

/// Backwards-compatible alias for common case (nbits=8)
pub type PQCode8<const M: usize> = PQCode<M, 8>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_code_creation_nbits8() {
        let code = PQCode::<4, 8>::from_indices(&[1, 2, 3, 4]);
        assert_eq!(code.get(0), 1);
        assert_eq!(code.get(1), 2);
        assert_eq!(code.get(2), 3);
        assert_eq!(code.get(3), 4);
        assert_eq!(code.m(), 4);
    }

    #[test]
    fn test_pq_code_creation_nbits10() {
        // 10 bits = 2 bytes per code, up to 1024 centroids
        let code = PQCode::<4, 10>::from_indices(&[500, 1000, 100, 1023]);
        assert_eq!(code.get(0), 500);
        assert_eq!(code.get(1), 1000);
        assert_eq!(code.get(2), 100);
        assert_eq!(code.get(3), 1023);
    }

    #[test]
    fn test_pq_code_creation_nbits16() {
        // 16 bits = 2 bytes per code, up to 65536 centroids
        let code = PQCode::<2, 16>::from_indices(&[65535, 32768]);
        assert_eq!(code.get(0), 65535);
        assert_eq!(code.get(1), 32768);
    }

    #[test]
    fn test_bytes_for_nbits() {
        assert_eq!(bytes_for_nbits(1), 1);
        assert_eq!(bytes_for_nbits(4), 1);
        assert_eq!(bytes_for_nbits(8), 1);
        assert_eq!(bytes_for_nbits(9), 2);
        assert_eq!(bytes_for_nbits(10), 2);
        assert_eq!(bytes_for_nbits(16), 2);
        assert_eq!(bytes_for_nbits(17), 3);
        assert_eq!(bytes_for_nbits(24), 3);
    }

    #[test]
    fn test_pq_code_total_bytes() {
        assert_eq!(PQCode::<8, 8>::TOTAL_BYTES, 8);
        assert_eq!(PQCode::<8, 10>::TOTAL_BYTES, 16);
        assert_eq!(PQCode::<16, 8>::TOTAL_BYTES, 16);
        assert_eq!(PQCode::<16, 10>::TOTAL_BYTES, 32);
    }

    #[test]
    fn test_pq_code_embedding_trait() {
        assert_eq!(PQCode::<8, 8>::length(), 8);
        assert_eq!(PQCode::<8, 10>::length(), 16);

        let code = PQCode::<4, 8>::from_indices(&[5, 6, 7, 8]);
        assert_eq!(code.as_slice(), &[5, 6, 7, 8]);

        let zeros = PQCode::<4, 8>::zeros();
        assert_eq!(zeros.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_pq_code_embedding_trait_nbits10() {
        // nbits=10 uses 2 bytes per code
        let code = PQCode::<2, 10>::from_indices(&[500, 1000]);
        let slice = code.as_slice();
        assert_eq!(slice.len(), 4); // 2 codes * 2 bytes each

        // Verify little-endian encoding
        // 500 = 0x01F4 -> [0xF4, 0x01]
        assert_eq!(slice[0], 0xF4);
        assert_eq!(slice[1], 0x01);
        // 1000 = 0x03E8 -> [0xE8, 0x03]
        assert_eq!(slice[2], 0xE8);
        assert_eq!(slice[3], 0x03);
    }

    #[test]
    fn test_pq_code_hash() {
        use std::collections::HashMap;
        let mut map = HashMap::new();

        let code1 = PQCode::<3, 8>::from_indices(&[1, 2, 3]);
        let code2 = PQCode::<3, 8>::from_indices(&[1, 2, 3]);
        let code3 = PQCode::<3, 8>::from_indices(&[3, 2, 1]);

        map.insert(code1.clone(), "value1");
        map.insert(code3, "value3");

        assert_eq!(map.get(&code2), Some(&"value1"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_pqcode8_alias() {
        // PQCode8<M> should be equivalent to PQCode<M, 8>
        let code1: PQCode8<4> = PQCode::from_indices(&[1, 2, 3, 4]);
        let code2: PQCode<4, 8> = PQCode::from_indices(&[1, 2, 3, 4]);
        assert_eq!(code1, code2);
    }
}
