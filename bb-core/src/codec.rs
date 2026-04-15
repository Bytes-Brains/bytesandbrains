use crate::embedding::EmbeddingSpace;
use crate::index::OpRef;

/// Encode, decode, and optionally train a compressed embedding representation.
///
/// `Codec` is the data-plane counterpart to [`Index`](crate::Index): an index
/// stores and retrieves vectors, while a codec compresses them. If a codec also
/// participates in gossip, the gossip protocol wraps the codec and implements
/// [`OverlayProtocol`](crate::OverlayProtocol) separately — separation of
/// data-plane from control-plane.
///
/// Every operation returns an [`OpRef`] handle rather than a bare result,
/// mirroring the pattern used by [`Index`](crate::Index). This lets the same
/// trait describe both in-process codecs (where the handle completes
/// synchronously) and networked codecs (where the handle tracks a remote RPC).
///
/// Like [`Index`](crate::Index), `Codec` includes training and observation
/// methods directly. Codecs that don't need training should make
/// [`train`](Codec::train) and [`observe`](Codec::observe) no-ops and return
/// `true` from [`is_trained`](Codec::is_trained).
///
/// ## Associated type families
///
/// | Associated type | Operation handle (GAT) | Operations |
/// |---|---|---|
/// | [`Encoded`](Codec::Encoded) | [`EncodeRef`](Codec::EncodeRef) | [`encode`](Codec::encode), [`encode_batch`](Codec::encode_batch) |
/// | — | [`DecodeRef`](Codec::DecodeRef) | [`decode`](Codec::decode), [`decode_batch`](Codec::decode_batch) |
/// | — | [`TrainRef`](Codec::TrainRef) | [`train`](Codec::train) |
/// | — | [`ObserveRef`](Codec::ObserveRef) | [`observe`](Codec::observe), [`observe_batch`](Codec::observe_batch) |
pub trait Codec<S: EmbeddingSpace> {
    /// The compressed representation produced by [`encode`](Codec::encode).
    type Encoded: Clone;

    /// Handle to an in-flight encode operation.
    type EncodeRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight decode operation.
    type DecodeRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight train operation.
    type TrainRef<'a>: OpRef where Self: 'a;

    /// Handle to an in-flight observe operation.
    type ObserveRef<'a>: OpRef where Self: 'a;

    /// Compress a single embedding.
    fn encode(&mut self, embedding: &S::EmbeddingData) -> Self::EncodeRef<'_>;

    /// Compress a batch of embeddings.
    fn encode_batch(&mut self, embeddings: &[S::EmbeddingData]) -> Vec<Self::EncodeRef<'_>>;

    /// Reconstruct an embedding from its compressed form.
    fn decode(&self, encoded: &Self::Encoded) -> Self::DecodeRef<'_>;

    /// Reconstruct a batch of embeddings from their compressed forms.
    fn decode_batch(&self, encoded: &[Self::Encoded]) -> Vec<Self::DecodeRef<'_>>;

    /// Returns the fixed byte-size of an encoded vector, if the encoding is
    /// fixed-size. Variable-length encodings return `None`.
    fn code_size(&self) -> Option<usize>;

    /// Batch training from a set of embeddings.
    ///
    /// Codecs that don't learn from data should make this a no-op and return
    /// `true` from [`is_trained`](Codec::is_trained).
    fn train(&mut self, embeddings: &[S::EmbeddingData]) -> Self::TrainRef<'_>;

    /// Online/incremental update from a single observation.
    fn observe(&mut self, embedding: &S::EmbeddingData) -> Self::ObserveRef<'_>;

    /// Online/incremental update from a batch of observations.
    fn observe_batch(&mut self, embeddings: &[S::EmbeddingData]) -> Vec<Self::ObserveRef<'_>>;

    /// Whether the codec has been trained and is ready to encode/decode.
    fn is_trained(&self) -> bool;
}
