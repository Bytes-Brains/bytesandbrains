//! `HoldTable` - slot-named value buffer for `Hold.Stash` /
//! `Hold.Flush` syscalls.
//!
//! bounded by max-distinct-slot-names with a drop
//! counter, so a producer that mints arbitrary slot names cannot
//! grow the table unbounded. When the table is at capacity and an
//! attempt to stash a fresh slot name arrives, the new entry is
//! dropped and the drop counter ticks. Existing slots' values
//! can always be overwritten (stash on an existing key never
//! evicts).

use std::collections::HashMap;

/// Named-slot value buffer with bounded distinct-slot count.
pub struct HoldTable {
    slots: HashMap<String, Vec<u8>>,
    max_slots: usize,
    drops: u64,
}

impl Default for HoldTable {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            max_slots: Self::DEFAULT_MAX_SLOTS,
            drops: 0,
        }
    }
}

impl HoldTable {
    /// Default cap on distinct slot names. 1024 covers any
    /// reasonable user workload while bounding the worst-case
    /// memory footprint of an unknown-key flood.
    pub const DEFAULT_MAX_SLOTS: usize = 1024;

    /// Construct a fresh empty table with the default cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a custom max distinct-slot cap.
    pub fn with_max_slots(max_slots: usize) -> Self {
        Self {
            slots: HashMap::new(),
            max_slots,
            drops: 0,
        }
    }

    /// Stash bytes into the named slot. Overwrites any previous
    /// value (existing slot names ALWAYS replace cleanly). New
    /// slot names that would exceed the table's configured cap are dropped
    /// and the drop counter ticks.
    pub fn stash(&mut self, slot: &str, bytes: Vec<u8>) {
        if self.slots.contains_key(slot) || self.slots.len() < self.max_slots {
            self.slots.insert(slot.to_string(), bytes);
        } else {
            self.drops += 1;
        }
    }

    /// Take the stashed value from the named slot, if present.
    pub fn flush(&mut self, slot: &str) -> Option<Vec<u8>> {
        self.slots.remove(slot)
    }

    /// Cumulative drop count for new-slot-on-full attempts.
    pub fn drops(&self) -> u64 {
        self.drops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stash_flush_roundtrip() {
        let mut t = HoldTable::new();
        t.stash("a", vec![1, 2, 3]);
        assert_eq!(t.flush("a"), Some(vec![1, 2, 3]));
        assert_eq!(t.flush("a"), None);
    }

    #[test]
    fn stash_overflow_drops_new_slot_and_ticks_counter() {
        let mut t = HoldTable::with_max_slots(2);
        t.stash("a", vec![1]);
        t.stash("b", vec![2]);
        assert_eq!(t.drops(), 0);
        // Third distinct slot at cap: dropped.
        t.stash("c", vec![3]);
        assert_eq!(t.drops(), 1);
        assert_eq!(t.flush("c"), None);
        // Existing slot overwrite still works at cap.
        t.stash("a", vec![9]);
        assert_eq!(t.flush("a"), Some(vec![9]));
    }
}
