//! `InboundDedup` - sliding-window seen-message tracker per
//! ENGINE.md §10.4.
//!
//! Recv-flavoured wire ops consult this set to discard envelopes
//! that repeat a recently-seen message hash. Storage is bounded -
//! when the window fills, the oldest entry is evicted.
//!
//! lands a simple LRU-by-insertion-order implementation;
use std::collections::{HashSet, VecDeque};

/// Default rolling window (8192 distinct message hashes). Picked to
/// be roughly 2× the default ingress capacity so duplicates from
/// concurrent peers don't collide with recent legit traffic.
pub const DEFAULT_WINDOW_SIZE: usize = 8192;

/// Sliding-window seen-message tracker.
pub struct InboundDedup {
    seen: HashSet<u64>,
    order: VecDeque<u64>,
    capacity: usize,
}

impl Default for InboundDedup {
    fn default() -> Self {
        Self::new()
    }
}

impl InboundDedup {
    /// Construct with the default window size.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_WINDOW_SIZE)
    }

    /// Construct with a custom window size. Sizes below 1 are
    /// clamped to 1.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            seen: HashSet::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Has this hash been seen within the current window?
    pub fn contains(&self, hash: u64) -> bool {
        self.seen.contains(&hash)
    }

    /// Mark `hash` as seen. Returns `true` if the entry was
    /// already present (a duplicate). On insertion the oldest
    /// entry is evicted if the window is full.
    pub fn record(&mut self, hash: u64) -> bool {
        if !self.seen.insert(hash) {
            return true;
        }
        self.order.push_back(hash);
        if self.order.len() > self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.seen.remove(&evicted);
            }
        }
        false
    }

    /// Clear the entire window.
    pub fn clear(&mut self) {
        self.seen.clear();
        self.order.clear();
    }

    /// Current window size (number of distinct entries).
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// `true` when no messages have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Configured window capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

