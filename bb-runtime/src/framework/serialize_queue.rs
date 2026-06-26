//! `SerializeQueue` - named-FIFO map for `Serialize.Enqueue` /
//! `Serialize.Dequeue` syscalls.
//!
//! every per-key FIFO is bounded with a drop counter
//! so a hostile or buggy producer cannot grow the queue unbounded.
//! `enqueue` drops the OLDEST entry on overflow (FIFO eviction)
//! and ticks the per-key drop counter. The default cap is
//! [`SerializeQueue::DEFAULT_PER_QUEUE_CAP`]; production deployments
//! override via `NodeConfig` (see ).

use std::collections::{HashMap, VecDeque};

/// Named-FIFO map with per-queue bounded capacity.
pub struct SerializeQueue {
    queues: HashMap<String, VecDeque<Vec<u8>>>,
    drops: HashMap<String, u64>,
    per_queue_cap: usize,
}

impl Default for SerializeQueue {
    fn default() -> Self {
        Self {
            queues: HashMap::new(),
            drops: HashMap::new(),
            per_queue_cap: Self::DEFAULT_PER_QUEUE_CAP,
        }
    }
}

impl SerializeQueue {
    /// Default per-named-queue capacity. 4096 entries balances
    /// "deep enough to survive a transient publisher burst" against
    /// "small enough to bound the worst-case memory footprint of a
    /// single misbehaving key."
    pub const DEFAULT_PER_QUEUE_CAP: usize = 4096;

    /// Construct a fresh empty map with the default per-queue cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a custom per-queue cap. Useful for edge
    /// deployments tightening the bound below the default.
    pub fn with_per_queue_cap(per_queue_cap: usize) -> Self {
        Self {
            queues: HashMap::new(),
            drops: HashMap::new(),
            per_queue_cap,
        }
    }

    /// Enqueue `bytes` into the named FIFO. If the FIFO is already
    /// at capacity, the OLDEST entry is dropped (FIFO eviction)
    /// and the per-key drop counter ticks.
    pub fn enqueue(&mut self, name: &str, bytes: Vec<u8>) {
        let queue = self.queues.entry(name.to_string()).or_default();
        if queue.len() >= self.per_queue_cap {
            queue.pop_front();
            *self.drops.entry(name.to_string()).or_default() += 1;
        }
        queue.push_back(bytes);
    }

    /// Dequeue from the head of the named FIFO. Returns `None` if
    /// empty.
    pub fn dequeue(&mut self, name: &str) -> Option<Vec<u8>> {
        self.queues.get_mut(name).and_then(|q| q.pop_front())
    }

    /// Length of the named FIFO (0 if missing).
    pub fn len(&self, name: &str) -> usize {
        self.queues.get(name).map(|q| q.len()).unwrap_or(0)
    }

    /// Cumulative drop count for the named FIFO. Observability for
    /// queue-overflow conditions.
    pub fn drops(&self, name: &str) -> u64 {
        self.drops.get(name).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enqueue_dequeue_roundtrip() {
        let mut q = SerializeQueue::new();
        q.enqueue("foo", b"a".to_vec());
        q.enqueue("foo", b"b".to_vec());
        assert_eq!(q.len("foo"), 2);
        assert_eq!(q.dequeue("foo"), Some(b"a".to_vec()));
        assert_eq!(q.dequeue("foo"), Some(b"b".to_vec()));
        assert_eq!(q.dequeue("foo"), None);
    }

    #[test]
    fn enqueue_at_cap_drops_oldest_and_ticks_counter() {
        // bounded queue: at cap, the oldest entry
        // gets evicted and the drop counter increments.
        let mut q = SerializeQueue::with_per_queue_cap(2);
        q.enqueue("foo", b"a".to_vec());
        q.enqueue("foo", b"b".to_vec());
        assert_eq!(q.len("foo"), 2);
        assert_eq!(q.drops("foo"), 0);
        q.enqueue("foo", b"c".to_vec());
        assert_eq!(q.len("foo"), 2);
        assert_eq!(q.drops("foo"), 1);
        // Oldest ("a") is gone; "b" + "c" remain in FIFO order.
        assert_eq!(q.dequeue("foo"), Some(b"b".to_vec()));
        assert_eq!(q.dequeue("foo"), Some(b"c".to_vec()));
    }
}
