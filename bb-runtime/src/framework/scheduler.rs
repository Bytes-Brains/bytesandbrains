//! `Scheduler` - sorted timer heap.
//!
//! ships the real `BinaryHeap`-backed scheduler that drives
//! `Interval` / `After` / `Sleep` / `TimerKind::Completion` syscall
//! ops. `has_matured(now_ns)` reports whether any timer has matured
//! by the supplied time; `poll_matured(now_ns)` drains every
//! matured timer in age order.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::ids::CommandId;

/// What a matured timer signals covers Sleep / Interval /
/// After / Completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimerKind {
    /// One-shot sleep maturity - fulfills `CommandId`.
    Sleep(CommandId),
    /// Periodic timer - re-armed after maturity. Carries the
    /// owning Op's site name as a u64 key for the matured-timer
    /// routing.
    Interval {
        /// Period in nanoseconds.
        period_ns: u64,
        /// Stable id for the owning Op (caller-supplied u64).
        key: u64,
    },
    /// One-shot delayed Trigger emission.
    After {
        /// Stable id for the owning Op.
        key: u64,
    },
    /// External-completion shim (used by `Sleep`-like ops that
    /// re-route through `handle_completion`).
    Completion(CommandId),
}

/// Internal heap entry. `BinaryHeap` is a max-heap by default;
/// we reverse the ordering on `maturity_ns` so the earliest
/// timer is at the top.
#[derive(Debug)]
struct Entry {
    maturity_ns: u64,
    kind: TimerKind,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.maturity_ns.cmp(&self.maturity_ns)
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.maturity_ns == other.maturity_ns
    }
}

impl Eq for Entry {}

/// Default cap on the timer heap. A runaway Interval / After /
/// Pulse / Sleep call site would otherwise grow `heap` without
/// bound and drag a long-running Node into OOM.
pub const DEFAULT_TIMER_HEAP_CAP: usize = 65_536;

/// Sorted timer heap.
pub struct Scheduler {
    heap: BinaryHeap<Entry>,
    now_ns: u64,
    /// Maximum permitted heap depth. `schedule` drops the latest
    /// timer (with a `tracing::warn`) once `heap.len() >= cap`.
    /// Operators tune via [`Self::set_cap`].
    cap: usize,
    /// Count of timers dropped due to cap. Exposed for telemetry.
    dropped: u64,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self {
            heap: BinaryHeap::new(),
            now_ns: 0,
            cap: DEFAULT_TIMER_HEAP_CAP,
            dropped: 0,
        }
    }
}

impl Scheduler {
    /// Construct a fresh scheduler with `now_ns = 0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the timer heap cap. Production paths size from
    /// `NodeConfig` if the host advertises a different bound.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap.max(1);
    }

    /// Number of timers dropped on overflow since construction.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Advance the scheduler's notion of current time.
    pub fn set_now(&mut self, now_ns: u64) {
        self.now_ns = now_ns;
    }

    /// Read the current time.
    pub fn now_ns(&self) -> u64 {
        self.now_ns
    }

    /// Schedule a timer. Drops with a `tracing::warn` once the
    /// heap reaches its cap so a runaway Interval/After loop
    /// can't grow the heap to OOM.
    pub fn schedule(&mut self, maturity_ns: u64, kind: TimerKind) {
        if self.heap.len() >= self.cap {
            self.dropped = self.dropped.saturating_add(1);
            tracing::warn!(
                cap = self.cap,
                dropped_total = self.dropped,
                ?kind,
                "Scheduler: timer dropped, heap at cap",
            );
            return;
        }
        self.heap.push(Entry { maturity_ns, kind });
    }

    /// Whether any timer has matured by `now_ns`.
    pub fn has_matured(&self, now_ns: u64) -> bool {
        self.heap.peek().is_some_and(|e| e.maturity_ns <= now_ns)
    }

    /// Drain every timer whose `maturity_ns <= now_ns` in age order.
    pub fn poll_matured(&mut self, now_ns: u64) -> Vec<TimerKind> {
        let mut out = Vec::new();
        while let Some(entry) = self.heap.peek() {
            if entry.maturity_ns > now_ns {
                break;
            }
            out.push(self.heap.pop().expect("just peeked").kind);
        }
        out
    }

    /// Number of pending (un-matured + matured) timers.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod cap_tests {
    use super::*;
    use crate::ids::CommandId;

    #[test]
    fn schedule_drops_at_cap() {
        let mut s = Scheduler::new();
        s.set_cap(3);
        for i in 0..3 {
            s.schedule(i, TimerKind::Sleep(CommandId::from(i)));
        }
        assert_eq!(s.len(), 3);
        s.schedule(100, TimerKind::Sleep(CommandId::from(100)));
        assert_eq!(s.len(), 3, "4th schedule dropped");
        assert_eq!(s.dropped(), 1);
    }

    #[test]
    fn schedule_resumes_when_heap_drains_below_cap() {
        let mut s = Scheduler::new();
        s.set_cap(2);
        s.schedule(0, TimerKind::Sleep(CommandId::from(0)));
        s.schedule(1, TimerKind::Sleep(CommandId::from(1)));
        let _matured = s.poll_matured(0);
        s.schedule(2, TimerKind::Sleep(CommandId::from(2)));
        assert_eq!(s.dropped(), 0);
    }
}

