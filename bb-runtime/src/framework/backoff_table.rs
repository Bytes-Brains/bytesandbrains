//! `BackoffTable` - per-peer exponential backoff state used by wire
//! syscalls + transport adapters per ENGINE.md §10.2.
//!
//! Each peer tracks consecutive failures, the timestamp of the most
//! recent attempt, and the computed next-retry time. The backoff
//! schedule is exponential with a configurable base and cap:
//!
//! ```text
//! delay(n) = min(BASE_NS * 2^n, MAX_DELAY_NS)
//! ```
//!
//! On success, `clear(peer)` resets state. On failure,
//! `record_failure(peer, now_ns)` bumps the attempt counter +
//! schedules the next retry. `should_retry(peer, now_ns)` reports
//! whether the cooldown has elapsed (peers never seen before are
//! always allowed to retry).

use std::collections::HashMap;

use crate::ids::PeerId;

/// Default base delay (10 ms) for the first retry after a failure.
pub const DEFAULT_BASE_NS: u64 = 10_000_000;

/// Default cap (60 s). Stops the doubling from running away on long
/// outages.
pub const DEFAULT_MAX_DELAY_NS: u64 = 60_000_000_000;

/// Per-peer backoff bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackoffState {
    /// Number of consecutive failures.
    pub attempts: u32,
    /// `now_ns` recorded at the most recent `record_failure` call.
    pub last_attempt_ns: u64,
    /// Earliest `now_ns` at which the next retry is permitted.
    pub next_retry_ns: u64,
}

/// Per-peer exponential backoff table.
pub struct BackoffTable {
    states: HashMap<PeerId, BackoffState>,
    base_ns: u64,
    max_delay_ns: u64,
}

impl Default for BackoffTable {
    fn default() -> Self {
        Self::new()
    }
}

impl BackoffTable {
    /// Construct a fresh table using the spec's default schedule
    /// (10 ms base, 60 s cap).
    pub fn new() -> Self {
        Self::with_schedule(DEFAULT_BASE_NS, DEFAULT_MAX_DELAY_NS)
    }

    /// Construct a table with a custom exponential schedule.
    pub fn with_schedule(base_ns: u64, max_delay_ns: u64) -> Self {
        Self {
            states: HashMap::new(),
            base_ns: base_ns.max(1),
            max_delay_ns: max_delay_ns.max(1),
        }
    }

    /// Record a failure for `peer` observed at `now_ns`. Increments
    /// the attempt counter + schedules the next retry.
    pub fn record_failure(&mut self, peer: PeerId, now_ns: u64) {
        let attempts = self
            .states
            .get(&peer)
            .map(|s| s.attempts.saturating_add(1))
            .unwrap_or(1);
        let next_retry_ns = now_ns.saturating_add(self.delay_for(attempts));
        self.states.insert(
            peer,
            BackoffState {
                attempts,
                last_attempt_ns: now_ns,
                next_retry_ns,
            },
        );
    }

    /// Record a remotely-advised back-off for `peer` per the
    /// backpressure protocol at
    /// `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
    /// §5.2. Unlike [`Self::record_failure`], the advisory sets
    /// `next_retry_ns = now_ns + min_backoff_ns` directly instead of
    /// applying the local exponential schedule - the receiver knows
    /// best how long the sender should pause.
    ///
    /// Bumps the attempt counter so a subsequent local
    /// `record_failure` resumes the exponential schedule at the
    /// next step. Caps `min_backoff_ns` at `max_delay_ns` so a
    /// pathological remote advisory cannot pin the peer indefinitely.
    pub fn record_remote_advisory(&mut self, peer: PeerId, now_ns: u64, min_backoff_ns: u64) {
        let attempts = self
            .states
            .get(&peer)
            .map(|s| s.attempts.saturating_add(1))
            .unwrap_or(1);
        let capped = min_backoff_ns.min(self.max_delay_ns);
        let next_retry_ns = now_ns.saturating_add(capped);
        self.states.insert(
            peer,
            BackoffState {
                attempts,
                last_attempt_ns: now_ns,
                next_retry_ns,
            },
        );
    }

    /// Record a success for `peer`; clears any tracked failure
    /// state. Subsequent `should_retry` returns `true` immediately.
    pub fn record_success(&mut self, peer: PeerId) {
        self.states.remove(&peer);
    }

    /// Whether a retry to `peer` is permitted at `now_ns`. Peers
    /// with no recorded failures retry immediately.
    pub fn should_retry(&self, peer: PeerId, now_ns: u64) -> bool {
        match self.states.get(&peer) {
            None => true,
            Some(state) => now_ns >= state.next_retry_ns,
        }
    }

    /// Inspect the recorded state for `peer`. Returns `None` when no
    /// failures have been recorded.
    pub fn state(&self, peer: PeerId) -> Option<BackoffState> {
        self.states.get(&peer).copied()
    }

    /// Number of peers currently being tracked.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether any peer is currently backing off.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Iterate `(PeerId, BackoffState)` for snapshot capture.
    /// .
    pub fn iter(&self) -> impl Iterator<Item = (PeerId, BackoffState)> + '_ {
        self.states.iter().map(|(p, s)| (*p, *s))
    }

    /// Restore one peer's backoff state directly. Used by
    /// `Node::restore` to re-seed peers from a
    /// `FrameworkSnapshot::backoff_table` entry without going
    /// through the failure-counting path.
    pub fn restore_state(&mut self, peer: PeerId, state: BackoffState) {
        self.states.insert(peer, state);
    }

    fn delay_for(&self, attempts: u32) -> u64 {
        // 2^(n-1) backoff (attempts >= 1 after first failure). Clamp
        // shift so we never overflow before applying the max cap.
        let shift = attempts.saturating_sub(1).min(63);
        let factor = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
        self.base_ns.saturating_mul(factor).min(self.max_delay_ns)
    }
}

