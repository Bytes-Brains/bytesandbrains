use bb_core::{
    Index,
    embedding::EmbeddingSpace,
    index::{OpId, OpRef},
};

/// A simple flat index that stores raw embeddings.
///
/// This index performs brute-force linear scan for similarity search.
/// It borrows a reference to the embedding space for distance computation.
pub struct FlatIndex<'a, S: EmbeddingSpace, V: Clone> {
    space: &'a S,
    embeddings: Vec<S::EmbeddingData>,
    values: Vec<V>,
    next_op_id: u64,
}

impl<'a, S: EmbeddingSpace, V: Clone> FlatIndex<'a, S, V> {
    pub fn new(space: &'a S) -> Self {
        Self {
            space,
            embeddings: Vec::new(),
            values: Vec::new(),
            next_op_id: 1,
        }
    }

    fn alloc_op_id(&mut self) -> OpId {
        let id = OpId(self.next_op_id);
        self.next_op_id += 1;
        id
    }

    /// Add an embedding with its associated value.
    pub fn add_embedding(&mut self, embedding: S::EmbeddingData, value: V) {
        self.embeddings.push(embedding);
        self.values.push(value);
    }

    /// Get a reference to all stored embeddings.
    pub fn embeddings(&self) -> &[S::EmbeddingData] {
        &self.embeddings
    }

    /// Search for the k nearest neighbors.
    pub fn search_knn(&self, query: &S::EmbeddingData, k: usize) -> Vec<(V, S::DistanceValue)> {
        let prepared = self.space.prepare(query);

        let mut candidates: Vec<_> = self
            .embeddings
            .iter()
            .zip(self.values.iter())
            .map(|(emb, val)| {
                let dist = self.space.distance_prepared(&prepared, emb);
                (val.clone(), dist)
            })
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1));
        candidates.truncate(k);
        candidates
    }
}

/// Search parameters for FlatIndex.
#[derive(Clone, Debug, Default)]
pub struct FlatSearchParams {
    pub k: usize,
}

/// Add parameters for FlatIndex.
#[derive(Clone, Debug, Default)]
pub struct FlatAddParams;

/// Remove parameters for FlatIndex.
#[derive(Clone, Debug, Default)]
pub struct FlatRemoveParams;

/// Train parameters for FlatIndex.
#[derive(Clone, Debug, Default)]
pub struct FlatTrainParams;

/// Search result for FlatIndex.
pub struct FlatSearchResult<V: Clone, D> {
    pub neighbors: Vec<(V, D)>,
}

/// Eager operation reference for local/synchronous operations.
pub struct EagerOpRef<T, E> {
    id: OpId,
    result: Option<Result<T, E>>,
}

impl<T, E> EagerOpRef<T, E> {
    pub fn ok(id: OpId, value: T) -> Self {
        Self {
            id,
            result: Some(Ok(value)),
        }
    }
}

impl<T, E: Clone> OpRef for EagerOpRef<T, E> {
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
        self.result.take().expect("finish called twice")
    }
}

/// Error type for flat index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlatIndexError {
    NotFound,
}

impl std::fmt::Display for FlatIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlatIndexError::NotFound => write!(f, "not found"),
        }
    }
}

impl std::error::Error for FlatIndexError {}

impl<'a, S: EmbeddingSpace, V: Clone> Index<S> for FlatIndex<'a, S, V> {
    type Value = V;
    type SearchType = FlatSearchParams;
    type AddType = FlatAddParams;
    type RemoveType = FlatRemoveParams;
    type TrainType = FlatTrainParams;

    type SearchRef<'b> = EagerOpRef<FlatSearchResult<V, S::DistanceValue>, FlatIndexError> where Self: 'b;
    type AddRef<'b> = EagerOpRef<(), FlatIndexError> where Self: 'b;
    type RemoveRef<'b> = EagerOpRef<(), FlatIndexError> where Self: 'b;
    type TrainRef<'b> = EagerOpRef<(), FlatIndexError> where Self: 'b;
    type ObserveRef<'b> = EagerOpRef<(), FlatIndexError> where Self: 'b;

    fn search(
        &mut self,
        search_embedding: &S::EmbeddingData,
        search_type: &Self::SearchType,
    ) -> Self::SearchRef<'_> {
        let id = self.alloc_op_id();
        let neighbors = self.search_knn(search_embedding, search_type.k);
        EagerOpRef::ok(id, FlatSearchResult { neighbors })
    }

    fn add(
        &mut self,
        embedding: &S::EmbeddingData,
        value: Self::Value,
        _add_type: &Self::AddType,
    ) -> Self::AddRef<'_> {
        let id = self.alloc_op_id();
        self.add_embedding(embedding.clone(), value);
        EagerOpRef::ok(id, ())
    }

    fn remove(
        &mut self,
        _embedding: &S::EmbeddingData,
        _remove_type: &Self::RemoveType,
    ) -> Self::RemoveRef<'_> {
        // Linear scan removal not implemented
        let id = self.alloc_op_id();
        EagerOpRef::ok(id, ())
    }

    fn train(
        &mut self,
        _data: &[S::EmbeddingData],
        _train_type: &Self::TrainType,
    ) -> Self::TrainRef<'_> {
        // Flat index doesn't need training
        EagerOpRef::ok(self.alloc_op_id(), ())
    }

    fn observe(&mut self, _embedding: &S::EmbeddingData) -> Self::ObserveRef<'_> {
        EagerOpRef::ok(self.alloc_op_id(), ())
    }

    fn reset(&mut self) {
        self.embeddings.clear();
        self.values.clear();
    }

    fn len(&self) -> usize {
        self.embeddings.len()
    }

    fn is_trained(&self) -> bool {
        true
    }

    fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bb_core::embedding::{F32Embedding, F32L2Space};

    type Space = F32L2Space<4>;

    #[test]
    fn test_flat_index_add_search() {
        let space = F32L2Space::<4>;
        let mut index = FlatIndex::new(&space);

        // Add some vectors
        index.add_embedding(F32Embedding([0.0, 0.0, 0.0, 0.0]), 0);
        index.add_embedding(F32Embedding([1.0, 1.0, 1.0, 1.0]), 1);
        index.add_embedding(F32Embedding([2.0, 2.0, 2.0, 2.0]), 2);

        assert_eq!(index.len(), 3);

        // Search for nearest to origin
        let results = index.search_knn(&F32Embedding([0.0, 0.0, 0.0, 0.0]), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Closest should be index 0
        assert_eq!(results[1].0, 1); // Second closest should be index 1
    }

    #[test]
    fn test_flat_index_trait() {
        let space = F32L2Space::<4>;
        let mut index = FlatIndex::new(&space);

        // Use Index trait methods
        let mut add_ref = index.add(
            &F32Embedding([1.0, 2.0, 3.0, 4.0]),
            42,
            &FlatAddParams,
        );
        assert!(add_ref.is_finished());
        add_ref.finish().unwrap();

        let mut search_ref = index.search(
            &F32Embedding([1.0, 2.0, 3.0, 4.0]),
            &FlatSearchParams { k: 1 },
        );
        assert!(search_ref.is_finished());
        let result = search_ref.finish().unwrap();

        assert_eq!(result.neighbors.len(), 1);
        assert_eq!(result.neighbors[0].0, 42);
    }

    #[test]
    fn test_flat_index_empty() {
        let space = F32L2Space::<4>;
        let index: FlatIndex<Space, usize> = FlatIndex::new(&space);
        assert!(index.is_empty());
        assert!(index.is_trained());
    }

    #[test]
    fn test_flat_index_reset() {
        let space = F32L2Space::<4>;
        let mut index = FlatIndex::new(&space);
        index.add_embedding(F32Embedding([0.0, 0.0, 0.0, 0.0]), 0);
        assert_eq!(index.len(), 1);

        index.reset();
        assert!(index.is_empty());
    }
}
