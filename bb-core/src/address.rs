use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::str::FromStr;

use lru::LruCache;

/// Trait for peer addresses — blanket implemented for any type that is
/// `Hash + Eq + Clone + ToString + FromStr`.
///
/// This abstracts over `String`, `SocketAddr`, `Multiaddr`, or any custom address type.
pub trait Address: Hash + Eq + Clone + ToString + FromStr + fmt::Debug {}

impl<T: Hash + Eq + Clone + ToString + FromStr + fmt::Debug> Address for T {}

/// A bounded, deduplicated collection of addresses for a peer.
///
/// Backed by an LRU cache: all operations (insert, lookup, promote, evict)
/// are O(1). Most recently seen addresses are promoted automatically.
/// When capacity is exceeded, the least recently seen entry is evicted.
pub struct AddressBook<A: Address> {
    cache: LruCache<A, ()>,
    max_size: usize,
}

impl<A: Address> Clone for AddressBook<A> {
    fn clone(&self) -> Self {
        let mut cache = LruCache::new(NonZeroUsize::new(self.max_size).unwrap());
        for (addr, _) in self.cache.iter().rev() {
            cache.put(addr.clone(), ());
        }
        Self {
            cache,
            max_size: self.max_size,
        }
    }
}

impl<A: Address> PartialEq for AddressBook<A> {
    fn eq(&self, other: &Self) -> bool {
        if self.cache.len() != other.cache.len() {
            return false;
        }
        self.cache
            .iter()
            .zip(other.cache.iter())
            .all(|((a, _), (b, _))| a == b)
    }
}

impl<A: Address> Eq for AddressBook<A> {}

impl<A: Address> Hash for AddressBook<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (addr, _) in self.cache.iter() {
            addr.hash(state);
        }
    }
}

impl<A: Address> AddressBook<A> {
    pub fn new(first_addr: A, max_size: usize) -> AddressBook<A> {
        assert!(max_size > 0, "AddressBook max_size must be > 0");
        let mut cache = LruCache::new(NonZeroUsize::new(max_size).unwrap());
        cache.put(first_addr, ());
        AddressBook { cache, max_size }
    }

    /// Record that we've seen this address. Returns true if it was newly added.
    /// If already present, promotes it to most recent. If at capacity,
    /// evicts the least recently seen entry.
    pub fn seen(&mut self, addr: A) -> bool {
        self.cache.put(addr, ()).is_none()
    }

    pub fn get(&self, index: usize) -> Option<&A> {
        self.cache.iter().nth(index).map(|(addr, _)| addr)
    }

    /// Returns the most recently seen address.
    pub fn first(&self) -> &A {
        self.cache
            .iter()
            .next()
            .map(|(addr, _)| addr)
            .expect("AddressBook is never empty")
    }

    /// Iterates from most recently seen to least recently seen.
    pub fn iter(&self) -> impl Iterator<Item = &A> {
        self.cache.iter().map(|(addr, _)| addr)
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Consumes the page and returns addresses in MRU-first order.
    pub fn into_vec(self) -> Vec<A> {
        self.cache.into_iter().map(|(addr, _)| addr).collect()
    }

    pub fn contains(&self, addr: &A) -> bool {
        self.cache.contains(addr)
    }

    pub fn remove(&mut self, addr: &A) -> bool {
        if self.cache.len() <= 1 {
            return false;
        }
        self.cache.pop(addr).is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Merge addresses from another page into this one.
    /// New addresses go in as least recent. Returns number added.
    pub fn concat(&mut self, other: &AddressBook<A>) -> usize {
        let mut added = 0;
        // Iterate other in LRU-first order so we don't disturb our MRU ordering
        for (addr, _) in other.cache.iter().rev() {
            if !self.cache.contains(addr) && self.cache.len() < self.max_size {
                // push without promoting existing — just insert at LRU end
                self.cache.push(addr.clone(), ());
                added += 1;
            }
        }
        added
    }
}

impl<A: Address + fmt::Debug> fmt::Debug for AddressBook<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let addrs: Vec<&A> = self.cache.iter().map(|(a, _)| a).collect();
        f.debug_struct("AddressBook")
            .field("addrs", &addrs)
            .field("max_size", &self.max_size)
            .finish()
    }
}

impl<A: Address> fmt::Display for AddressBook<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let addrs: Vec<String> = self.iter().map(|a| a.to_string()).collect();
        write!(f, "[{}] (max {})", addrs.join(", "), self.max_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let page = AddressBook::new("addr1".to_string(), 5);
        assert_eq!(page.len(), 1);
        assert_eq!(page.first(), "addr1");
        assert_eq!(page.max_size(), 5);
    }

    #[test]
    fn test_seen_new_address() {
        let mut page = AddressBook::new("addr1".to_string(), 5);
        assert!(page.seen("addr2".to_string()));
        assert_eq!(page.len(), 2);
        assert_eq!(page.first(), "addr2");
    }

    #[test]
    fn test_seen_existing_promotes() {
        let mut page = AddressBook::new("addr1".to_string(), 5);
        page.seen("addr2".to_string());
        // addr2 is at front, addr1 at back
        assert!(!page.seen("addr1".to_string()));
        // addr1 promoted to front
        assert_eq!(page.first(), "addr1");
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut page = AddressBook::new("addr1".to_string(), 2);
        page.seen("addr2".to_string());
        assert_eq!(page.len(), 2);
        page.seen("addr3".to_string());
        assert_eq!(page.len(), 2);
        // addr1 should be evicted (least recently seen)
        assert!(!page.contains(&"addr1".to_string()));
        assert!(page.contains(&"addr3".to_string()));
    }

    #[test]
    fn test_contains() {
        let mut page = AddressBook::new("addr1".to_string(), 5);
        assert!(page.contains(&"addr1".to_string()));
        assert!(!page.contains(&"addr2".to_string()));
        page.seen("addr2".to_string());
        assert!(page.contains(&"addr2".to_string()));
    }

    #[test]
    fn test_remove() {
        let mut page = AddressBook::new("addr1".to_string(), 5);
        page.seen("addr2".to_string());
        assert!(page.remove(&"addr1".to_string()));
        assert!(!page.contains(&"addr1".to_string()));
        assert!(!page.remove(&"addr1".to_string()));
    }

    #[test]
    fn test_remove_last_address_refused() {
        let mut page = AddressBook::new("addr1".to_string(), 5);
        assert!(!page.remove(&"addr1".to_string()));
        assert_eq!(page.len(), 1);
        assert!(page.contains(&"addr1".to_string()));
    }

    #[test]
    fn test_concat() {
        let mut page1 = AddressBook::new("a".to_string(), 5);
        page1.seen("b".to_string());

        let mut page2 = AddressBook::new("c".to_string(), 5);
        page2.seen("d".to_string());

        let added = page1.concat(&page2);
        assert_eq!(added, 2);
        assert_eq!(page1.len(), 4);
    }

    #[test]
    fn test_concat_respects_capacity() {
        let mut page1 = AddressBook::new("a".to_string(), 3);
        page1.seen("b".to_string());

        let mut page2 = AddressBook::new("c".to_string(), 5);
        page2.seen("d".to_string());
        page2.seen("e".to_string());

        let added = page1.concat(&page2);
        assert_eq!(added, 1); // Only room for 1 more
        assert_eq!(page1.len(), 3);
    }

    #[test]
    fn test_into_vec() {
        let mut page = AddressBook::new("a".to_string(), 5);
        page.seen("b".to_string());
        let v = page.into_vec();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_equality() {
        let mut page1 = AddressBook::new("a".to_string(), 5);
        page1.seen("b".to_string());
        let mut page2 = AddressBook::new("a".to_string(), 5);
        page2.seen("b".to_string());
        assert_eq!(page1, page2);
    }

    #[test]
    fn test_socket_addr_as_address() {
        use std::net::SocketAddr;
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
        let page = AddressBook::new(addr, 5);
        assert_eq!(page.len(), 1);
    }
}
