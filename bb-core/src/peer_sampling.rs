use crate::index::OpRef;

/// Capability trait for protocols that can sample peers from their view.
///
/// The selection strategy is passed as a parameter to each selection call,
/// making the trait stateless with respect to selection mode.
pub trait PeerSampling {
    /// The peer type exposed by this protocol's view.
    type Peer: Clone;

    /// Immutable reference to the current peer view.
    type PeerView<'a> where Self: 'a;

    /// The type used to choose between peer selection strategies.
    type SamplingMode;

    /// Handle to an in-flight peer selection operation.
    type SelectPeerRef<'a>: OpRef where Self: 'a;

    /// Return an immutable reference to the current peer view.
    fn view(&self) -> Self::PeerView<'_>;

    /// Number of peers currently in the view.
    fn view_len(&self) -> usize;

    /// Select a single peer using the given selection strategy.
    fn select_peer(&mut self, mode: &Self::SamplingMode) -> Self::SelectPeerRef<'_>;

    /// Return all peers known to this sampler for broadcast operations.
    ///
    /// Unlike [`select_peer`](PeerSampling::select_peer) which returns a single peer,
    /// this returns all peers in the view. Useful for protocols that need to
    /// send messages to multiple peers (e.g., gossip push-pull).
    ///
    /// The caller is responsible for limiting fan-out if needed.
    fn broadcast(&self) -> Vec<Self::Peer>;
}
