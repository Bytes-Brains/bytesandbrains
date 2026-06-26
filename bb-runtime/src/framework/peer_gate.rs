//! `PeerGate` - per-name concurrency limiter used by `Limit.Acquire`
//! / `Limit.Release` syscalls.
use std::collections::HashMap;

/// Per-name concurrency limiter.
#[derive(Default)]
pub struct PeerGate {
    inflight: HashMap<String, u32>,
}

impl PeerGate {
    /// Construct fresh.
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to acquire one permit. Returns `false` if the current
    /// `inflight` count for `name` is at or above `n`, else
    /// increments the counter and returns `true`.
    pub fn acquire(&mut self, name: &str, n: u32) -> bool {
        let current = self.inflight.entry(name.to_string()).or_insert(0);
        if *current >= n {
            return false;
        }
        *current += 1;
        true
    }

    /// Release one permit.
    pub fn release(&mut self, name: &str) {
        if let Some(c) = self.inflight.get_mut(name) {
            if *c > 0 {
                *c -= 1;
            }
        }
    }

    /// Read current inflight count for the named gate.
    pub fn inflight(&self, name: &str) -> u32 {
        self.inflight.get(name).copied().unwrap_or(0)
    }
}

