use crate::{address::Address, index::OpRef, peer::Peer, peer_id::PeerId};

/// The result of a protocol step: events emitted and messages to send.
///
/// Callers must process the returned `Step` — it contains outbound messages
/// that must be delivered and events that must be handled by the application.
#[must_use = "the returned Step contains messages and events that must be processed"]
pub struct Step<P: OverlayProtocol + ?Sized> {
    pub events: Vec<P::Event>,
    pub messages: Vec<OutMessage<P>>,
}

impl<P: OverlayProtocol + ?Sized> Step<P> {
    pub fn new() -> Self {
        Step {
            events: Vec::new(),
            messages: Vec::new(),
        }
    }

    pub fn with_event(mut self, event: P::Event) -> Self {
        self.events.push(event);
        self
    }

    pub fn with_message(mut self, message: OutMessage<P>) -> Self {
        self.messages.push(message);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty() && self.messages.is_empty()
    }

    /// Extend this step with events and messages from another step.
    pub fn extend(&mut self, other: Step<P>) {
        self.events.extend(other.events);
        self.messages.extend(other.messages);
    }
}

impl<P: OverlayProtocol + ?Sized> Default for Step<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// An outgoing protocol message addressed to a specific peer.
pub struct OutMessage<P: OverlayProtocol + ?Sized> {
    pub destination: Peer<P::Address>,
    pub message: P::Message,
}

/// The core behavior trait for overlay protocols.
///
/// Generalizes the poll()/message() -> Step pattern. Applicable to any
/// overlay protocol (gossip peer sampling, etc.).
pub trait OverlayProtocol {
    type Address: Address;
    type Message;
    type Event;
    /// The peer type managed by this overlay.
    type Peer: Clone;
    /// Configuration for bootstrap operations.
    type BootstrapConfig;
    /// Handle to an in-flight bootstrap operation.
    type BootstrapRef<'a>: OpRef
    where
        Self: 'a;

    /// Unique protocol identifier for message dispatch (e.g., "/bb/gossip/1.0.0")
    const PROTOCOL_ID: &'static str;

    /// Advance internal timers, check liveness, progress queries.
    fn poll(&mut self) -> Step<Self>;

    /// Handle an incoming protocol message from another peer.
    fn on_message(&mut self, from: Peer<Self::Address>, msg: Self::Message) -> Step<Self>;

    /// The local peer identity.
    fn local_peer_id(&self) -> &PeerId;

    /// Bootstrap the overlay by discovering peers in the network.
    ///
    /// Returns an [`OpRef`] handle to track the bootstrap operation. The caller
    /// should poll [`is_finished`](OpRef::is_finished) or wait for events via
    /// [`poll`](OverlayProtocol::poll), then call [`finish`](OpRef::finish) to
    /// retrieve the result.
    fn bootstrap(&mut self, config: Self::BootstrapConfig) -> Self::BootstrapRef<'_>;

    /// Called when a connection to a peer fails or is lost.
    ///
    /// The overlay should fail all pending requests to this peer and clean up state.
    /// Returns a Step with any resulting events (e.g., peer removal, query failures).
    fn on_connection_failed(&mut self, peer_id: &PeerId) -> Step<Self>;

    /// Add a peer to the overlay's view.
    fn add_peer(&mut self, peer: Self::Peer) -> Step<Self>;

    /// Remove a peer from the overlay's view by ID.
    fn remove_peer(&mut self, peer_id: &PeerId) -> Option<Self::Peer>;

    /// Associate an address with a known peer.
    fn add_address(&mut self, peer_id: &PeerId, address: Self::Address);

    /// Remove an address from a peer.
    ///
    /// If this was the peer's last address, removes the peer entirely and returns it.
    fn remove_address(&mut self, peer_id: &PeerId, address: &Self::Address) -> Option<Self::Peer>;
}
