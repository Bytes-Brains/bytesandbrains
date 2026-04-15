use crate::{
    address::{Address, AddressBook},
    peer_id::PeerId,
};

/// A generic peer identity: PeerId + a page of network addresses.
///
/// This is the base peer type used across all protocols. Protocol-specific
/// peer types that extend this with additional state live in their respective crates.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Peer<A: Address> {
    pub peer_id: PeerId,
    pub addresses: AddressBook<A>,
}

impl<A: Address> Peer<A> {
    pub fn new(peer_id: PeerId, addresses: AddressBook<A>) -> Self {
        Self { peer_id, addresses }
    }
}

impl<A: Address> std::fmt::Display for Peer<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Peer({}, {})", self.peer_id, self.addresses.first().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_creation() {
        let id = PeerId::from_data("test-peer");
        let addrs = AddressBook::new("127.0.0.1:8080".to_string(), 5);
        let peer = Peer::new(id, addrs);
        assert_eq!(peer.peer_id, id);
        assert_eq!(peer.addresses.len(), 1);
    }

    #[test]
    fn test_peer_clone() {
        let id = PeerId::from_data("test-peer");
        let addrs = AddressBook::new("127.0.0.1:8080".to_string(), 5);
        let peer1 = Peer::new(id, addrs);
        let peer2 = peer1.clone();
        assert_eq!(peer1, peer2);
    }

    #[test]
    fn test_peer_equality() {
        let id = PeerId::from_data("same");
        let addrs1 = AddressBook::new("addr1".to_string(), 5);
        let addrs2 = AddressBook::new("addr2".to_string(), 5);
        let peer1 = Peer::new(id, addrs1);
        let peer2 = Peer::new(id, addrs2);
        // Different addresses means not equal
        assert_ne!(peer1, peer2);
    }

    #[test]
    fn test_peer_hash_as_key() {
        use std::collections::HashMap;
        let id = PeerId::from_data("peer1");
        let addrs = AddressBook::new("addr".to_string(), 5);
        let peer = Peer::new(id, addrs);
        let mut map = HashMap::new();
        map.insert(peer.clone(), 42);
        assert_eq!(map.get(&peer), Some(&42));
    }
}
