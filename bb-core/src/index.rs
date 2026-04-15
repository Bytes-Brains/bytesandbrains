use crate::embedding::EmbeddingSpace;
pub use crate::op::{OpId, OpRef, NoopOpRef};

/// A vector index for storage and similarity search.
///
/// `Index` is generic over an [`EmbeddingSpace`] `S` that encodes the metric
/// type and dimensionality at the type level, so there are no runtime
/// dimension or metric fields.
///
/// Every mutating or query operation returns an [`OpRef`] handle rather than
/// a bare result. This lets the same trait describe both in-process indices
/// (where the handle completes synchronously) and networked indices (where
/// the handle tracks a remote RPC).
///
/// ## Associated type families
///
/// | Parameter type | Operation handle (GAT) | Operations |
/// |---|---|---|
/// | [`SearchType`](Index::SearchType) | [`SearchRef`](Index::SearchRef) | [`search`](Index::search) |
/// | [`AddType`](Index::AddType) | [`AddRef`](Index::AddRef) | [`add`](Index::add) |
/// | [`RemoveType`](Index::RemoveType) | [`RemoveRef`](Index::RemoveRef) | [`remove`](Index::remove) |
/// | [`TrainType`](Index::TrainType) | [`TrainRef`](Index::TrainRef) | [`train`](Index::train) |
/// | — | [`ObserveRef`](Index::ObserveRef) | [`observe`](Index::observe) |
///
/// ## `EmbeddingSpace` and `Value`
///
/// The embedding space `S` determines the vector representation
/// (`S::EmbeddingData`) and distance metric. The associated [`Value`](Index::Value)
/// type holds arbitrary data stored alongside each embedding; indices that
/// don't need payloads can set `type Value = ()`.
pub trait Index<S: EmbeddingSpace> {
    /// Arbitrary data stored alongside each embedding.
    type Value: Clone;

    /// Parameters that configure a search (e.g. k for k-NN, radius, hybrid
    /// config).
    type SearchType;

    /// Parameters that configure an add operation.
    type AddType;

    /// Parameters that configure a remove operation.
    type RemoveType;

    /// Parameters that configure a train operation.
    type TrainType;

    /// Handle to an in-flight search operation.
    type SearchRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight add operation.
    type AddRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight remove operation.
    type RemoveRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight train operation.
    type TrainRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight observe operation.
    type ObserveRef<'a>: OpRef where Self: 'a;

    /// Execute a single similarity search against the index.
    fn search(
        &mut self,
        search_embedding: &S::EmbeddingData,
        search_type: &Self::SearchType,
    ) -> Self::SearchRef<'_>;

    /// Insert an embedding and its associated value.
    fn add(
        &mut self,
        embedding: &S::EmbeddingData,
        value: Self::Value,
        add_type: &Self::AddType,
    ) -> Self::AddRef<'_>;

    /// Remove an embedding from the index.
    fn remove(
        &mut self,
        embedding: &S::EmbeddingData,
        remove_type: &Self::RemoveType,
    ) -> Self::RemoveRef<'_>;

    /// Train the index on a representative dataset.
    ///
    /// Flat indices should make this a no-op and return `true` from
    /// [`is_trained`](Index::is_trained).
    fn train(
        &mut self,
        data: &[S::EmbeddingData],
        train_type: &Self::TrainType,
    ) -> Self::TrainRef<'_>;

    /// Incrementally update the index from a single observation without full
    /// retraining.
    fn observe(
        &mut self,
        embedding: &S::EmbeddingData,
    ) -> Self::ObserveRef<'_>;

    /// Remove all vectors from the index.
    fn reset(&mut self);

    /// The number of vectors currently stored.
    fn len(&self) -> usize;

    /// Whether the index has been trained and is ready for use.
    fn is_trained(&self) -> bool;

    /// Whether the index contains no vectors.
    fn is_empty(&self) -> bool;
}

/// No-op index implementation for `()`.
///
/// Used as the default type parameter when a protocol needs no local storage.
impl<S: EmbeddingSpace> Index<S> for () {
    type Value = ();
    type SearchType = ();
    type AddType = ();
    type RemoveType = ();
    type TrainType = ();
    type SearchRef<'a> = NoopOpRef where Self: 'a;
    type AddRef<'a> = NoopOpRef where Self: 'a;
    type RemoveRef<'a> = NoopOpRef where Self: 'a;
    type TrainRef<'a> = NoopOpRef where Self: 'a;
    type ObserveRef<'a> = NoopOpRef where Self: 'a;

    fn search(&mut self, _: &S::EmbeddingData, _: &()) -> NoopOpRef { NoopOpRef::new(0) }
    fn add(&mut self, _: &S::EmbeddingData, _: (), _: &()) -> NoopOpRef { NoopOpRef::new(0) }
    fn remove(&mut self, _: &S::EmbeddingData, _: &()) -> NoopOpRef { NoopOpRef::new(0) }
    fn train(&mut self, _: &[S::EmbeddingData], _: &()) -> NoopOpRef { NoopOpRef::new(0) }
    fn observe(&mut self, _: &S::EmbeddingData) -> NoopOpRef { NoopOpRef::new(0) }
    fn reset(&mut self) {}
    fn len(&self) -> usize { 0 }
    fn is_trained(&self) -> bool { true }
    fn is_empty(&self) -> bool { true }
}
