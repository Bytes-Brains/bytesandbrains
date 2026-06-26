//! `RecordBuffer` - bounded per-name ring buffer used by the
//! `Record` syscall.
//!
//! bounded TWO ways: each per-name ring has a fixed
//! capacity (oldest entry evicted at cap), AND the table caps the
//! number of distinct ring names with a drop counter. New ring
//! names that arrive past `max_rings` are dropped with the counter
//! ticking; existing rings always accept records (oldest-first
//! evicted) up to the per-ring cap.

use std::collections::{HashMap, VecDeque};

/// Default per-name capacity.
const DEFAULT_CAPACITY: usize = 1024;
/// Default distinct-name cap.
const DEFAULT_MAX_RINGS: usize = 1024;

/// Per-name ring buffer with two bounds.
pub struct RecordBuffer {
    rings: HashMap<String, VecDeque<Vec<u8>>>,
    capacity: usize,
    max_rings: usize,
    name_drops: u64,
}

impl Default for RecordBuffer {
    fn default() -> Self {
        Self {
            rings: HashMap::new(),
            capacity: DEFAULT_CAPACITY,
            max_rings: DEFAULT_MAX_RINGS,
            name_drops: 0,
        }
    }
}

impl RecordBuffer {
    /// Construct with the default capacities.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with explicit per-ring and distinct-name caps.
    pub fn with_caps(per_ring: usize, max_rings: usize) -> Self {
        Self {
            rings: HashMap::new(),
            capacity: per_ring,
            max_rings,
            name_drops: 0,
        }
    }

    /// Record `bytes` under the named ring; oldest entry evicted
    /// when capacity reached. New ring names past `max_rings` are
    /// dropped with the `name_drops` counter ticking.
    pub fn record(&mut self, name: &str, bytes: Vec<u8>) {
        if !self.rings.contains_key(name) && self.rings.len() >= self.max_rings {
            self.name_drops += 1;
            return;
        }
        let cap = self.capacity;
        let ring = self.rings.entry(name.to_string()).or_default();
        if ring.len() >= cap {
            ring.pop_front();
        }
        ring.push_back(bytes);
    }

    /// Read the current contents of the named ring.
    pub fn snapshot(&self, name: &str) -> Vec<Vec<u8>> {
        self.rings
            .get(name)
            .map(|r| r.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Cumulative drop count for unknown-name overflows.
    pub fn name_drops(&self) -> u64 {
        self.name_drops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_snapshot_round_trip() {
        let mut b = RecordBuffer::new();
        b.record("m", b"a".to_vec());
        b.record("m", b"b".to_vec());
        let snap = b.snapshot("m");
        assert_eq!(snap, vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[test]
    fn record_overflow_drops_new_name_and_ticks_counter() {
        let mut b = RecordBuffer::with_caps(8, 2);
        b.record("a", b"1".to_vec());
        b.record("b", b"2".to_vec());
        assert_eq!(b.name_drops(), 0);
        // Third distinct name at cap: dropped.
        b.record("c", b"3".to_vec());
        assert_eq!(b.name_drops(), 1);
        assert_eq!(b.snapshot("c"), Vec::<Vec<u8>>::new());
        // Existing rings still accept records at cap.
        b.record("a", b"more".to_vec());
        assert_eq!(b.snapshot("a"), vec![b"1".to_vec(), b"more".to_vec()]);
    }
}
