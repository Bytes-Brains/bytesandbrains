mod code;
mod distance;
mod sdc;

pub use code::{PQCode, PQCode8, bytes_for_nbits};
pub use distance::PQDistanceTable;
pub use sdc::SDCTable;

use std::fmt;
use std::sync::Arc;

use bb_core::{
    Codec,
    embedding::{Embedding, EmbeddingSpace},
    index::{OpId, OpRef},
};
use bb_ml::KMeans;

/// Product Quantizer for vector compression and fast distance computation.
///
/// Product Quantization splits a D-dimensional vector into M subvectors,
/// learns a codebook for each subspace via k-means, and encodes each
/// subvector as the index of its nearest centroid.
///
/// Const generics:
/// - M: number of subquantizers
/// - NBITS: bits per centroid index (determines storage and centroid count)
///
/// The embedding dimension must be divisible by M.
///
/// After training, ProductQuantizer implements `EmbeddingSpace` with
/// `EmbeddingData = PQCode<M, NBITS>`, allowing direct use with FlatIndex
/// and other structures.
///
/// This implementation supports:
/// - Training via k-means++ on each subspace
/// - Encoding: vector -> PQCode<M, NBITS>
/// - Decoding: PQCode<M, NBITS> -> reconstructed vector
/// - ADC: Asymmetric Distance Computation via precomputed distance tables
/// - SDC: Symmetric Distance Computation via precomputed centroid-to-centroid distances
///
/// Note: PQ internally uses L2 distance for subspace quantization.
#[derive(Clone)]
pub struct ProductQuantizer<S: EmbeddingSpace, const M: usize, const NBITS: usize>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    space: S,
    dsub: usize,
    d: usize,
    /// Shared codebook data — Arc so cloning PQ across actors is cheap (~pointer copy).
    centroids: Arc<Vec<f32>>,
    sdc_table: Option<Arc<SDCTable<M, NBITS>>>,
    trained: bool,
    next_op_id: u64,
    /// Reusable buffer for subvector operations to avoid per-call allocations
    subvec_buffer: Vec<f32>,
}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> fmt::Debug for ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProductQuantizer")
            .field("M", &M)
            .field("NBITS", &NBITS)
            .field("ksub", &(1usize << NBITS))
            .field("dsub", &self.dsub)
            .field("trained", &self.trained)
            .finish()
    }
}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> PartialEq for ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{
    fn eq(&self, other: &Self) -> bool {
        self.dsub == other.dsub
            && self.trained == other.trained
            && self.centroids == other.centroids
    }
}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> Eq for ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
{}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
    <S::EmbeddingData as Embedding>::Scalar: Into<f32> + From<f32>,
{
    /// Number of centroids per subspace (2^NBITS)
    pub const KSUB: usize = 1 << NBITS;

    /// Create a new Product Quantizer.
    ///
    /// # Arguments
    /// * `space` - The embedding space
    ///
    /// # Panics
    /// Panics if dimension is not divisible by M.
    pub fn new(space: S) -> Self {
        let d = S::EmbeddingData::length();
        assert!(
            d % M == 0,
            "dimension {} must be divisible by M={}",
            d,
            M
        );
        let dsub = d / M;

        Self {
            space,
            dsub,
            d,
            centroids: Arc::new(Vec::new()),
            sdc_table: None,
            trained: false,
            next_op_id: 1,
            // Pre-allocate buffer for subvector operations
            subvec_buffer: vec![0.0; dsub],
        }
    }

    /// Get a reference to the underlying embedding space.
    pub fn space(&self) -> &S {
        &self.space
    }

    fn alloc_op_id(&mut self) -> OpId {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;
        id
    }

    /// Number of subquantizers.
    pub fn m(&self) -> usize {
        M
    }

    /// Number of centroids per subspace.
    pub fn ksub(&self) -> usize {
        Self::KSUB
    }

    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Find the nearest centroid in a given subspace.
    /// The subvector must already be in self.subvec_buffer.
    fn find_nearest_centroid(&self, subspace: usize) -> usize {
        let ksub = Self::KSUB;
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for k in 0..ksub {
            let centroid_offset = (subspace * ksub + k) * self.dsub;
            let centroid = &self.centroids[centroid_offset..centroid_offset + self.dsub];

            let dist: f32 = self.subvec_buffer
                .iter()
                .zip(centroid.iter())
                .map(|(&s, &c)| {
                    let diff = s - c;
                    diff * diff
                })
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_idx = k;
            }
        }

        best_idx
    }

    /// Copy a subvector from the embedding slice into the reusable buffer.
    fn fill_subvec_buffer(&mut self, slice: &[<S::EmbeddingData as Embedding>::Scalar], subspace: usize) {
        let start = subspace * self.dsub;
        for i in 0..self.dsub {
            self.subvec_buffer[i] = slice[start + i].into();
        }
    }

    /// Encode a single embedding to a PQ code.
    pub fn encode_embedding(&mut self, embedding: &S::EmbeddingData) -> PQCode<M, NBITS> {
        assert!(self.trained, "codec must be trained before encoding");

        let slice = embedding.as_slice();
        let mut code = PQCode::<M, NBITS>::zeros();

        for subspace in 0..M {
            self.fill_subvec_buffer(slice, subspace);
            let nearest = self.find_nearest_centroid(subspace);
            code.set(subspace, nearest as u32);
        }

        code
    }

    /// Decode a PQ code to a reconstructed embedding.
    pub fn decode_code(&self, code: &PQCode<M, NBITS>) -> S::EmbeddingData {
        assert!(self.trained, "codec must be trained before decoding");

        let ksub = Self::KSUB;
        let mut result = vec![0.0f32; self.d];

        for m in 0..M {
            let c = code.get(m) as usize;
            let centroid_offset = (m * ksub + c) * self.dsub;
            let centroid = &self.centroids[centroid_offset..centroid_offset + self.dsub];

            let start = m * self.dsub;
            result[start..start + self.dsub].copy_from_slice(centroid);
        }

        let scalars: Vec<<S::EmbeddingData as Embedding>::Scalar> =
            result.into_iter().map(|x| x.into()).collect();
        S::EmbeddingData::from_slice(&scalars)
    }

    /// Train the quantizer on a dataset using k-means++.
    pub fn train_on(&mut self, data: &[S::EmbeddingData]) {
        assert!(!data.is_empty(), "training data cannot be empty");

        let ksub = Self::KSUB;
        assert!(
            data.len() >= ksub,
            "need at least {} data points (ksub), got {}",
            ksub,
            data.len()
        );

        // Allocate flat centroid storage: M * ksub * dsub
        let mut centroids = vec![0.0; M * ksub * self.dsub];

        // Train each subspace independently
        for subspace in 0..M {
            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> = data
                .iter()
                .map(|emb| {
                    let slice = emb.as_slice();
                    let start = subspace * self.dsub;
                    let end = start + self.dsub;
                    slice[start..end].iter().map(|&s| s.into()).collect()
                })
                .collect();

            // Run k-means++ on this subspace
            let kmeans = KMeans::fit(&subvectors, ksub, 25);

            // Copy centroids to flat storage
            for (k, centroid) in kmeans.centroids.iter().enumerate() {
                let offset = (subspace * ksub + k) * self.dsub;
                centroids[offset..offset + self.dsub].copy_from_slice(centroid);
            }
        }

        // Build SDC table using the embedding space's distance metric
        let sdc = SDCTable::from_centroids_with_distance(
            &centroids,
            self.dsub,
            S::slice_distance,
        );
        self.centroids = Arc::new(centroids);
        self.sdc_table = Some(Arc::new(sdc));
        self.trained = true;
    }

    /// Build a distance table for ADC (Asymmetric Distance Computation).
    pub fn build_distance_table(&mut self, query: &S::EmbeddingData) -> PQDistanceTable<S, M, NBITS> {
        assert!(self.trained, "codec must be trained before distance computation");

        let ksub = Self::KSUB;
        let query_slice = query.as_slice();
        let mut table = Vec::with_capacity(M * ksub);

        for subspace in 0..M {
            self.fill_subvec_buffer(query_slice, subspace);

            for k in 0..ksub {
                let centroid_offset = (subspace * ksub + k) * self.dsub;
                let centroid = &self.centroids[centroid_offset..centroid_offset + self.dsub];

                let dist = S::slice_distance(&self.subvec_buffer, centroid);

                table.push(S::DistanceValue::from(dist));
            }
        }

        PQDistanceTable::new(table, ksub)
    }

    /// Get the SDC table for symmetric distance computation.
    ///
    /// Returns None if the quantizer is not trained.
    pub fn sdc_table(&self) -> Option<&SDCTable<M, NBITS>> {
        self.sdc_table.as_deref()
    }
}

/// Eager operation reference for local/synchronous operations.
///
/// This is a simple wrapper for synchronous operations that complete immediately.
/// The result is stored and returned on first call to `finish()`. Subsequent calls
/// will return a clone of the original error (if the operation failed) or an
/// `AlreadyFinished` error (if the operation succeeded).
pub struct EagerOpRef<T, E> {
    id: OpId,
    result: Option<Result<T, E>>,
    /// Cached error for repeated finish() calls
    cached_error: Option<E>,
}

impl<T, E: Clone> EagerOpRef<T, E> {
    pub fn ok(id: OpId, value: T) -> Self {
        Self {
            id,
            result: Some(Ok(value)),
            cached_error: None,
        }
    }

    pub fn err(id: OpId, error: E) -> Self {
        Self {
            id,
            result: Some(Err(error.clone())),
            cached_error: Some(error),
        }
    }
}

/// Trait for error types that can represent "already finished" state.
pub trait FinishableError: Clone {
    /// Create an error indicating the operation was already finished.
    fn already_finished() -> Self;
}

impl FinishableError for PQError {
    fn already_finished() -> Self {
        PQError::AlreadyFinished
    }
}

impl<T, E: FinishableError> OpRef for EagerOpRef<T, E> {
    type Info = ();
    type Stats = ();
    type Result = T;
    type Error = E;

    fn id(&self) -> &OpId {
        &self.id
    }

    fn info(&self) -> Option<Self::Info> {
        Some(())
    }

    fn stats(&self) -> Option<Self::Stats> {
        Some(())
    }

    fn is_finished(&self) -> bool {
        true
    }

    fn finish(&mut self) -> Result<Self::Result, Self::Error> {
        match self.result.take() {
            Some(result) => result,
            None => {
                // finish() called again - return cached error or AlreadyFinished
                Err(self.cached_error.clone().unwrap_or_else(E::already_finished))
            }
        }
    }
}

/// Error type for PQ codec operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PQError {
    /// Codec has not been trained yet.
    NotTrained,
    /// OpRef::finish() was called more than once.
    AlreadyFinished,
    /// Serialization or deserialization error.
    #[cfg(feature = "codec")]
    SerializationError(String),
}

impl std::fmt::Display for PQError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PQError::NotTrained => write!(f, "codec not trained"),
            PQError::AlreadyFinished => write!(f, "operation already finished"),
            #[cfg(feature = "codec")]
            PQError::SerializationError(e) => write!(f, "serialization error: {}", e),
        }
    }
}

impl std::error::Error for PQError {}

impl<S: EmbeddingSpace + Default, const M: usize, const NBITS: usize> Default
    for ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
    <S::EmbeddingData as Embedding>::Scalar: Into<f32> + From<f32>,
{
    fn default() -> Self {
        Self::new(S::default())
    }
}

// =========================================================================
// Codebook Serialization
// =========================================================================

/// Serializable representation of a trained PQ codebook.
/// Contains all state needed to reconstruct a ProductQuantizer.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PQCodebook {
    pub centroids: Vec<f32>,
    pub sdc_table: Vec<f32>,
    pub sdc_ksub: usize,
    pub dsub: usize,
    pub d: usize,
    pub m: usize,
    pub nbits: usize,
}

#[cfg(feature = "codec")]
impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
    <S::EmbeddingData as Embedding>::Scalar: Into<f32> + From<f32>,
{
    /// Save the trained codebook to a file.
    pub fn save_codebook(&self, path: &std::path::Path) -> Result<(), PQError> {
        if !self.trained {
            return Err(PQError::NotTrained);
        }
        let sdc = self.sdc_table.as_ref().ok_or(PQError::NotTrained)?;
        let codebook = PQCodebook {
            centroids: (*self.centroids).clone(),
            sdc_table: sdc.table_data().to_vec(),
            sdc_ksub: sdc.ksub(),
            dsub: self.dsub,
            d: self.d,
            m: M,
            nbits: NBITS,
        };
        let encoded = bincode::serialize(&codebook)
            .map_err(|e| PQError::SerializationError(e.to_string()))?;
        std::fs::write(path, encoded)
            .map_err(|e| PQError::SerializationError(e.to_string()))?;
        Ok(())
    }

    /// Load a trained codebook from a file and reconstruct the ProductQuantizer.
    pub fn load_codebook(path: &std::path::Path, space: S) -> Result<Self, PQError> {
        let data = std::fs::read(path)
            .map_err(|e| PQError::SerializationError(e.to_string()))?;
        let codebook: PQCodebook = bincode::deserialize(&data)
            .map_err(|e| PQError::SerializationError(e.to_string()))?;

        if codebook.m != M || codebook.nbits != NBITS {
            return Err(PQError::SerializationError(format!(
                "Codebook M={}, NBITS={} does not match expected M={}, NBITS={}",
                codebook.m, codebook.nbits, M, NBITS
            )));
        }

        let sdc_table = SDCTable::from_raw(codebook.sdc_table, codebook.sdc_ksub);

        Ok(Self {
            space,
            dsub: codebook.dsub,
            d: codebook.d,
            centroids: Arc::new(codebook.centroids),
            sdc_table: Some(Arc::new(sdc_table)),
            trained: true,
            next_op_id: 0,
            subvec_buffer: vec![0.0; codebook.dsub],
        })
    }
}

impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> Codec<S> for ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
    <S::EmbeddingData as Embedding>::Scalar: Into<f32> + From<f32>,
{
    type Encoded = PQCode<M, NBITS>;
    type EncodeRef<'b> = EagerOpRef<PQCode<M, NBITS>, PQError> where Self: 'b;
    type DecodeRef<'b> = EagerOpRef<S::EmbeddingData, PQError> where Self: 'b;
    type TrainRef<'b> = EagerOpRef<(), PQError> where Self: 'b;
    type ObserveRef<'b> = EagerOpRef<(), PQError> where Self: 'b;

    fn encode(&mut self, embedding: &S::EmbeddingData) -> Self::EncodeRef<'_> {
        let id = OpId(0);
        if !self.trained {
            return EagerOpRef::err(id, PQError::NotTrained);
        }
        EagerOpRef::ok(id, self.encode_embedding(embedding))
    }

    fn encode_batch(&mut self, embeddings: &[S::EmbeddingData]) -> Vec<Self::EncodeRef<'_>> {
        embeddings.iter().map(|e| {
            let id = OpId(0);
            if !self.trained {
                return EagerOpRef::err(id, PQError::NotTrained);
            }
            EagerOpRef::ok(id, self.encode_embedding(e))
        }).collect()
    }

    fn decode(&self, encoded: &Self::Encoded) -> Self::DecodeRef<'_> {
        let id = OpId(0);
        if !self.trained {
            return EagerOpRef::err(id, PQError::NotTrained);
        }
        EagerOpRef::ok(id, self.decode_code(encoded))
    }

    fn decode_batch(&self, encoded: &[Self::Encoded]) -> Vec<Self::DecodeRef<'_>> {
        encoded.iter().map(|e| self.decode(e)).collect()
    }

    fn code_size(&self) -> Option<usize> {
        Some(PQCode::<M, NBITS>::TOTAL_BYTES)
    }

    fn train(&mut self, embeddings: &[S::EmbeddingData]) -> Self::TrainRef<'_> {
        let id = self.alloc_op_id();
        self.train_on(embeddings);
        EagerOpRef::ok(id, ())
    }

    fn observe(&mut self, _embedding: &S::EmbeddingData) -> Self::ObserveRef<'_> {
        // Online training not implemented - no-op
        EagerOpRef::ok(self.alloc_op_id(), ())
    }

    fn observe_batch(&mut self, embeddings: &[S::EmbeddingData]) -> Vec<Self::ObserveRef<'_>> {
        embeddings.iter().map(|_| self.observe(&S::EmbeddingData::zeros())).collect()
    }

    fn is_trained(&self) -> bool {
        self.trained
    }
}

// ProductQuantizer implements EmbeddingSpace directly for PQCode<M, NBITS>
impl<S: EmbeddingSpace, const M: usize, const NBITS: usize> EmbeddingSpace for ProductQuantizer<S, M, NBITS>
where
    [(); bytes_for_nbits(NBITS)]:,
    <S::EmbeddingData as Embedding>::Scalar: Into<f32> + From<f32>,
{
    type EmbeddingData = PQCode<M, NBITS>;
    type DistanceValue = S::DistanceValue;
    type Prepared = PQCode<M, NBITS>;

    fn space_id(&self) -> &'static str {
        "pq"
    }

    fn distance(&self, lhs: &Self::EmbeddingData, rhs: &Self::EmbeddingData) -> Self::DistanceValue {
        let sdc = self.sdc_table.as_ref().expect("ProductQuantizer must be trained before computing distances");
        S::DistanceValue::from(sdc.distance(lhs, rhs))
    }

    fn prepare(&self, embedding: &Self::EmbeddingData) -> Self::Prepared {
        embedding.clone()
    }

    fn distance_prepared(
        &self,
        prepared: &Self::Prepared,
        target: &Self::EmbeddingData,
    ) -> Self::DistanceValue {
        self.distance(prepared, target)
    }

    fn length() -> usize {
        PQCode::<M, NBITS>::TOTAL_BYTES
    }

    fn slice_distance(a: &[f32], b: &[f32]) -> f32 {
        S::slice_distance(a, b)
    }

    fn infinite_mapping(native_distance: &Self::DistanceValue) -> f32 {
        S::infinite_mapping(native_distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bb_core::embedding::{F32Embedding, F32L2Space};

    type Space = F32L2Space<8>;

    fn make_test_vectors(n: usize) -> Vec<F32Embedding<8>> {
        (0..n)
            .map(|i| {
                let val = i as f32;
                F32Embedding([val, val + 0.1, val + 0.2, val + 0.3, val + 0.4, val + 0.5, val + 0.6, val + 0.7])
            })
            .collect()
    }

    #[test]
    fn test_pq_creation() {
        let space = F32L2Space::<8>;
        let pq = ProductQuantizer::<Space, 2, 8>::new(space);
        assert_eq!(pq.m(), 2);
        assert_eq!(pq.ksub(), 256);
        assert_eq!(pq.dsub(), 4);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_pq_creation_nbits2() {
        let space = F32L2Space::<8>;
        let pq = ProductQuantizer::<Space, 2, 2>::new(space);
        assert_eq!(pq.m(), 2);
        assert_eq!(pq.ksub(), 4);  // 2^2 = 4
        assert_eq!(pq.dsub(), 4);
        assert!(!pq.is_trained());
    }

    #[test]
    fn test_pq_train_and_encode() {
        let space = F32L2Space::<8>;
        let mut pq = ProductQuantizer::<Space, 2, 2>::new(space); // 2 bits = 4 centroids
        let data = make_test_vectors(100);

        pq.train_on(&data);
        assert!(pq.is_trained());

        let code = pq.encode_embedding(&data[0]);
        assert_eq!(code.m(), 2);
    }

    #[test]
    fn test_pq_encode_decode() {
        let space = F32L2Space::<8>;
        // Use 4 bits = 16 centroids for better reconstruction
        let mut pq = ProductQuantizer::<Space, 2, 4>::new(space);
        let data = make_test_vectors(100);

        pq.train_on(&data);

        let original = &data[50];
        let code = pq.encode_embedding(original);
        let decoded = pq.decode_code(&code);

        // Decoded should be close to original (within quantization error)
        let orig_slice = original.as_slice();
        let dec_slice = decoded.as_slice();

        for i in 0..8 {
            let diff = (orig_slice[i] - dec_slice[i]).abs();
            assert!(diff < 10.0, "dimension {} differs by {}", i, diff);
        }
    }

    #[test]
    fn test_pq_adc_distance() {
        let space = F32L2Space::<8>;
        let mut pq = ProductQuantizer::<Space, 2, 2>::new(space);
        let data = make_test_vectors(100);

        pq.train_on(&data);

        let query = &data[0];
        let target = pq.encode_embedding(&data[1]);

        let table = pq.build_distance_table(query);
        let dist = table.distance(&target);

        // Distance should be positive for different vectors
        assert!(dist.value() >= 0.0);
    }

    #[test]
    fn test_pq_sdc() {
        let space = F32L2Space::<8>;
        let mut pq = ProductQuantizer::<Space, 2, 2>::new(space);
        let data = make_test_vectors(100);

        pq.train_on(&data);

        // Encode embeddings first before borrowing sdc table
        let code1 = pq.encode_embedding(&data[0]);
        let code2 = pq.encode_embedding(&data[1]);

        let sdc = pq.sdc_table().expect("should have SDC table after training");

        // Same code should have distance 0
        assert_eq!(sdc.distance(&code1, &code1), 0.0);

        // Different codes should have positive distance
        let dist = sdc.distance(&code1, &code2);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_pq_embedding_space() {
        let space = F32L2Space::<8>;
        let mut pq = ProductQuantizer::<Space, 2, 2>::new(space);
        let data = make_test_vectors(100);

        pq.train_on(&data);

        let code1 = pq.encode_embedding(&data[0]);
        let code2 = pq.encode_embedding(&data[1]);

        // Test EmbeddingSpace interface directly on ProductQuantizer
        let dist = pq.distance(&code1, &code2);
        assert!(dist.value() >= 0.0);

        // Same code should have distance 0
        assert_eq!(pq.distance(&code1, &code1).value(), 0.0);

        // Test prepare/distance_prepared
        let prepared = pq.prepare(&code1);
        let dist_prepared = pq.distance_prepared(&prepared, &code2);
        assert_eq!(dist, dist_prepared);
    }

    #[test]
    fn test_codec_trait() {
        let space = F32L2Space::<8>;
        let mut pq = ProductQuantizer::<Space, 2, 2>::new(space);
        let data = make_test_vectors(100);

        let mut train_ref = pq.train(&data);
        assert!(train_ref.is_finished());
        train_ref.finish().unwrap();

        assert!(pq.is_trained());

        let mut encode_ref = pq.encode(&data[0]);
        assert!(encode_ref.is_finished());
        let code = encode_ref.finish().unwrap();

        let mut decode_ref = pq.decode(&code);
        assert!(decode_ref.is_finished());
        let _decoded = decode_ref.finish().unwrap();
    }

    #[test]
    fn test_code_size() {
        // M=4, NBITS=8 -> 4 bytes per code (1 byte each)
        let pq = ProductQuantizer::<Space, 4, 8>::new(F32L2Space::<8>);
        assert_eq!(pq.code_size(), Some(4));

        // M=4, NBITS=10 -> 8 bytes per code (2 bytes each)
        let pq2 = ProductQuantizer::<Space, 4, 10>::new(F32L2Space::<8>);
        assert_eq!(pq2.code_size(), Some(8));
    }

    #[test]
    #[ignore] // Slow: k-means with 1024 centroids takes ~200s in debug mode
    fn test_pq_nbits10() {
        // Test with nbits=10 (1024 centroids per subspace)
        // This requires more training data
        let space = F32L2Space::<8>;
        let mut pq = ProductQuantizer::<Space, 2, 10>::new(space);

        // Need at least 1024 vectors for training
        let data: Vec<F32Embedding<8>> = (0..2000)
            .map(|i| {
                let val = (i as f32) * 0.01;
                F32Embedding([val, val + 0.1, val + 0.2, val + 0.3, val + 0.4, val + 0.5, val + 0.6, val + 0.7])
            })
            .collect();

        pq.train_on(&data);
        assert!(pq.is_trained());
        assert_eq!(pq.ksub(), 1024);

        let code = pq.encode_embedding(&data[500]);
        // Verify centroid indices can be > 255
        let idx0 = code.get(0);
        let idx1 = code.get(1);
        assert!(idx0 < 1024);
        assert!(idx1 < 1024);
    }
}
