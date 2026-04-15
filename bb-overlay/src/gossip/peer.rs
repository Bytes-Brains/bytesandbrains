use std::fmt;

use bb_core::address::AddressBook;
use bb_core::{Address, Peer, PeerId};

#[cfg(feature = "proto")]
use bb_core::proto::ProtoConversionError;
#[cfg(feature = "proto")]
use super::proto::gossip_proto::GossipPeerProto;

/// Trait for peer types used in the gossip framework.
///
/// Each gossip strategy defines its own peer type (e.g., `AgePeer` for
/// randomized gossip, or an embedding-based peer for proximity overlays).
/// This trait provides the framework access to the underlying `Peer<A>`.
///
/// When the `proto` feature is enabled, implementors must also provide
/// `Into<GossipPeerProto>` and `TryFrom<GossipPeerProto>` so the gossip
/// message serialization can work generically over any peer type.
#[cfg(not(feature = "proto"))]
pub trait GossipPeerType<A: Address>: Clone + fmt::Debug {
    fn peer(&self) -> &Peer<A>;
    fn peer_mut(&mut self) -> &mut Peer<A>;

    fn peer_id(&self) -> PeerId {
        self.peer().peer_id
    }

    /// Construct a peer from an address (used for the local peer).
    fn from_address(address: A) -> Self;
}

#[cfg(feature = "proto")]
pub trait GossipPeerType<A: Address>:
    Clone
    + fmt::Debug
    + Into<GossipPeerProto>
    + TryFrom<GossipPeerProto, Error = ProtoConversionError>
{
    fn peer(&self) -> &Peer<A>;
    fn peer_mut(&mut self) -> &mut Peer<A>;

    fn peer_id(&self) -> PeerId {
        self.peer().peer_id
    }

    /// Construct a peer from an address (used for the local peer).
    fn from_address(address: A) -> Self;
}

/// A peer with an age counter, used in randomized gossip.
///
/// The `age` field tracks how many exchange rounds this peer has survived
/// in the view. Higher age means the entry is staler. The `TailSelector`
/// uses this to prefer older peers for exchange, promoting freshness.
#[derive(Clone, Debug)]
pub struct AgePeer<A: Address> {
    pub peer: Peer<A>,
    pub age: u32,
}

impl<A: Address> AgePeer<A> {
    pub fn new(address: A, age: u32) -> Self {
        let peer_id = PeerId::from_data(&address.to_string());
        let addresses = AddressBook::new(address, 5);
        Self {
            peer: Peer::new(peer_id, addresses),
            age,
        }
    }
}

impl<A: Address> GossipPeerType<A> for AgePeer<A> {
    fn peer(&self) -> &Peer<A> {
        &self.peer
    }

    fn peer_mut(&mut self) -> &mut Peer<A> {
        &mut self.peer
    }

    fn from_address(address: A) -> Self {
        Self::new(address, 0)
    }
}
