use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::{Duration, Instant};

use crate::PeerId;

/// Unique identifier for an individual request within a query.
/// Unlike QueryId which identifies a query operation, RequestId identifies
/// a specific request-response exchange with a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RequestId(pub u64);

impl RequestId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Composite key for tracking pending requests.
/// Correlates responses by (peer_id, request_id) rather than by content.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RequestKey {
    pub peer_id: PeerId,
    pub request_id: RequestId,
}

impl RequestKey {
    pub fn new(peer_id: PeerId, request_id: RequestId) -> Self {
        Self { peer_id, request_id }
    }
}

#[derive(Debug, Clone)]
pub struct PendingRequestManager<K, V> {
    requests: HashMap<K, V>,
    timeout_queue: VecDeque<TimeoutEntry<K>>,
    default_timeout: Duration,
}

#[derive(Debug, Clone)]
struct TimeoutEntry<K> {
    key: K,
    timeout: Instant,
}

pub enum InsertResult<V> {
    Inserted,
    Replaced(V),
}

impl<K, V> PendingRequestManager<K, V>
where
    K: Clone + Eq + Hash,
{
    pub fn new(default_timeout: Duration) -> Self {
        Self {
            requests: HashMap::new(),
            timeout_queue: VecDeque::new(),
            default_timeout,
        }
    }

    pub fn insert(&mut self, key: K, data: V) -> InsertResult<V> {
        if self.requests.contains_key(&key) {
            let old_val = self.requests.insert(key.clone(), data).unwrap();
            self.timeout_queue.push_back(TimeoutEntry {
                key,
                timeout: Instant::now() + self.default_timeout,
            });
            return InsertResult::Replaced(old_val);
        }

        let timeout = Instant::now() + self.default_timeout;

        self.timeout_queue.push_back(TimeoutEntry {
            key: key.clone(),
            timeout,
        });

        self.requests.insert(key, data);
        InsertResult::Inserted
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.requests.remove(key)
    }

    pub fn process_timeouts(&mut self) -> Vec<(K, V)> {
        let now = Instant::now();
        let mut timed_out = Vec::new();

        while let Some(front) = self.timeout_queue.front() {
            if front.timeout > now {
                break;
            }

            let entry = self.timeout_queue.pop_front().unwrap();

            if let Some(data) = self.requests.remove(&entry.key) {
                timed_out.push((entry.key, data));
            }
        }

        timed_out
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.requests.contains_key(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.requests.keys()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.requests.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.requests.get_mut(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.requests.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.requests.iter_mut()
    }
}

impl<'a, K, V> IntoIterator for &'a PendingRequestManager<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.requests.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut PendingRequestManager<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = std::collections::hash_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.requests.iter_mut()
    }
}

/// Tracks pending request-response exchanges by (PeerId, RequestId).
///
/// This enables correct response correlation even when peers change their
/// embedding (drift), since we match by who responded, not where they claim to be.
///
/// The value type `V` can be an enum to support multiple request types:
/// ```ignore
/// enum RequestData {
///     Knn { search_embedding: Embedding, k: usize },
///     Kfn { search_embedding: Embedding, k: usize },
///     Ping,
/// }
/// let tracker: RequestTracker<RequestData> = RequestTracker::new(timeout);
/// ```
#[derive(Debug, Clone)]
pub struct RequestTracker<V> {
    inner: PendingRequestManager<RequestKey, V>,
    next_request_id: u64,
}

impl<V> RequestTracker<V>
where
    V: Clone,
{
    pub fn new(default_timeout: Duration) -> Self {
        Self {
            inner: PendingRequestManager::new(default_timeout),
            next_request_id: 1, // Start at 1 like libp2p
        }
    }

    /// Insert a new pending request, returning the assigned RequestId.
    pub fn insert(&mut self, peer_id: PeerId, data: V) -> RequestId {
        let request_id = RequestId::new(self.next_request_id);
        self.next_request_id += 1;
        let key = RequestKey::new(peer_id, request_id);
        self.inner.insert(key, data);
        request_id
    }

    /// Remove a pending request by peer_id and request_id.
    pub fn remove(&mut self, peer_id: &PeerId, request_id: &RequestId) -> Option<V> {
        let key = RequestKey { peer_id: peer_id.clone(), request_id: *request_id };
        self.inner.remove(&key)
    }

    /// Check if a request is pending for this peer.
    pub fn is_pending(&self, peer_id: &PeerId, request_id: &RequestId) -> bool {
        let key = RequestKey { peer_id: peer_id.clone(), request_id: *request_id };
        self.inner.contains_key(&key)
    }

    /// Process timeouts, returning timed-out (RequestKey, V) pairs.
    pub fn process_timeouts(&mut self) -> Vec<(RequestKey, V)> {
        self.inner.process_timeouts()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over all pending requests.
    pub fn iter(&self) -> impl Iterator<Item = (&RequestKey, &V)> {
        self.inner.iter()
    }

    /// Remove all pending requests for a specific peer.
    /// Returns all removed (RequestKey, V) pairs.
    pub fn remove_all_for_peer(&mut self, peer_id: &PeerId) -> Vec<(RequestKey, V)> {
        self.inner
            .requests
            .extract_if(|key, _| &key.peer_id == peer_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_new_manager() {
        let manager: PendingRequestManager<u32, String> =
            PendingRequestManager::new(Duration::from_millis(100));

        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
    }

    #[test]
    fn test_insert_and_contains() {
        let mut manager = PendingRequestManager::new(Duration::from_millis(100));

        let result = manager.insert(1, "first".to_string());
        assert!(matches!(result, InsertResult::Inserted));
        assert_eq!(manager.len(), 1);
        assert!(manager.contains_key(&1));
        assert!(!manager.contains_key(&2));
    }

    #[test]
    fn test_insert_duplicate_key() {
        let mut manager = PendingRequestManager::new(Duration::from_millis(100));

        let result1 = manager.insert(1, "first".to_string());
        assert!(matches!(result1, InsertResult::Inserted));

        let result2 = manager.insert(1, "second".to_string());
        match result2 {
            InsertResult::Replaced(returned_value) => assert_eq!(returned_value, "first"),
            _ => panic!("Expected Replaced variant"),
        }

        assert_eq!(manager.len(), 1);
        assert!(manager.contains_key(&1));

        let removed = manager.remove(&1);
        assert_eq!(removed, Some("second".to_string()));
    }

    #[test]
    fn test_remove() {
        let mut manager = PendingRequestManager::new(Duration::from_millis(100));

        manager.insert(1, "test".to_string());
        assert_eq!(manager.len(), 1);

        let removed = manager.remove(&1);
        assert_eq!(removed, Some("test".to_string()));
        assert_eq!(manager.len(), 0);
        assert!(!manager.contains_key(&1));

        let removed2 = manager.remove(&2);
        assert_eq!(removed2, None);
    }

    #[test]
    fn test_timeout_processing() {
        let mut manager = PendingRequestManager::new(Duration::from_millis(50));

        manager.insert(1, "first".to_string());
        manager.insert(2, "second".to_string());

        thread::sleep(Duration::from_millis(60));

        let timed_out = manager.process_timeouts();
        assert_eq!(timed_out.len(), 2);
        assert_eq!(manager.len(), 0);

        let mut keys: Vec<_> = timed_out.iter().map(|(k, _)| *k).collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2]);
    }

    #[test]
    fn test_no_timeouts() {
        let mut manager = PendingRequestManager::new(Duration::from_millis(100));

        manager.insert(1, "test".to_string());

        let timed_out = manager.process_timeouts();
        assert_eq!(timed_out.len(), 0);
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_manual_remove_before_timeout() {
        let mut manager = PendingRequestManager::new(Duration::from_millis(50));

        manager.insert(1, "test".to_string());

        let removed = manager.remove(&1);
        assert_eq!(removed, Some("test".to_string()));

        thread::sleep(Duration::from_millis(60));

        let timed_out = manager.process_timeouts();
        assert_eq!(timed_out.len(), 0);
        assert_eq!(manager.len(), 0);
    }
}
