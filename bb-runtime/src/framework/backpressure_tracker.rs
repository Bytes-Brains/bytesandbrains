//! `BackpressureTracker` - per-peer receiver-side state for the
//! typed-overload-signal protocol per
//! `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`.
//!
//! The framework owns this primitive; the engine consults it at
//! Phase 1 of the poll cycle when ingress depth crosses the
//! high-water mark or when `RttTracker::scan_phi` flips a sender to
//! `Suspect`. Each consultation either yields a
//! [`Decision::EmitNotice`] (the receiver will emit a typed
//! `BackoffNotice` envelope back to the sender) or a
//! [`Decision::Suppress`] (duplicate-suppression window still
//! open, silent-drop mode active, or another reason the receiver
//! should not act).
//!
//! ## Composition
//!
//! Per `peer_state.rs`, the tracker joins the existing per-peer
//! state cluster (`gate`, `governor`, `backoff`) as a sibling
//! field. Component authors and engine sites reach it through
//! `ctx.peers.backpressure` / `framework.peer_state.backpressure`,
//! matching the existing access pattern.
//!
//! The tracker is receiver-side state. The matching sender-side
//! state lives in the existing `BackoffTable` - on receipt of a
//! `BackoffNotice`, the sender updates `backoff` so the existing
//! `BackoffGateTx` consultation already gates the next outbound
//! send. No new sender-side primitive is required.
//!
//! ## K-then-silent semantics
//!
//! Each emitted notice bumps `notices_sent` for the peer. The
//! per-peer counter resets when the sender's `RttTracker::scan_phi`
//! flips back to `Live` (matching the existing recovery surface).
//! Once `notices_sent` exceeds `notice_threshold_k` without
//! recovery, [`Decision::EmitNotice`] is no longer returned;
//! [`Decision::SilentDrop`] is returned instead. The engine's Phase
//! 1 envelope router drops envelopes from a silent-drop peer
//! without further notice emission. The first silent-drop
//! transition surfaces as `InfraEvent::SilentDropActive` on the
//! bus; subsequent envelopes from the same peer surface no further
//! event until the peer recovers.

use std::collections::HashMap;

use crate::ids::PeerId;

/// Default high-water mark percentage. Matches the spec default
/// in §6: ingress queue depth >= 75% of capacity triggers a
/// `BackoffCause::QueueFull` notice.
pub const DEFAULT_HIGH_WATER_PCT: u8 = 75;

/// Default K (notices-without-slowdown before silent-drop).
/// Matches `RttEma::is_warm`'s "evidence sufficient to act"
/// threshold of 3 samples per
/// `bb-runtime/src/framework/rtt_tracker.rs:126-128`.
pub const DEFAULT_K_BEFORE_SILENT: u32 = 3;

/// Default minimum interval between successive notices to the
/// same peer (1 second). Acts as a hard lower bound on the
/// duplicate-suppression window so a flood of inbound envelopes
/// from one peer produces at most one notice per second even when
/// the receiver lacks a tighter per-cause `min_backoff_ns` hint.
pub const DEFAULT_MIN_NOTICE_INTERVAL_NS: u64 = 1_000_000_000;

/// Why the receiver is requesting a back-off.
///
/// Mirrors the wire-protocol `BackoffCause` payload field; lives
/// on the framework side so the engine + bus events reference
/// the same enum the wire op encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackoffCause {
    /// `IngressQueue` depth crossed the high-water mark.
    QueueFull,
    /// `PhiAccrualState` marked the sender as `Suspect` (sender is
    /// too fast for this receiver's processing rate).
    PhiAccrual,
    /// A Component returned a typed reject (e.g. role rate-limit
    /// such as an aggregator already filled its per-round window).
    ExplicitDrop,
}

/// Per-peer back-pressure bookkeeping.
#[derive(Clone, Copy, Debug, Default)]
pub struct BackpressureEntry {
    /// Total notices emitted to this peer since the last reset.
    /// Reset to 0 when the peer's φ collapses back below `Suspect`.
    pub notices_sent: u32,
    /// `now_ns` recorded at the most recent notice emission.
    /// `0` if none have been emitted yet. Used together with
    /// `last_min_backoff_ns` to suppress redundant notices inside
    /// the previously-quoted back-off window.
    pub last_notice_at_ns: u64,
    /// `min_backoff_ns` quoted on the most recent notice. The
    /// duplicate-suppression check skips emission while
    /// `now_ns < last_notice_at_ns + last_min_backoff_ns`. `0`
    /// when no notice has been emitted yet.
    pub last_min_backoff_ns: u64,
    /// `true` once `notices_sent` >= K without observed recovery.
    /// Phase 1 of `Engine::poll` drops envelopes from this peer
    /// while set.
    pub silent_drop_active: bool,
}

/// Decision returned by [`BackpressureTracker::observe_overload`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Receiver should emit a `BackoffNotice` envelope back to the
    /// sender. Carries the `min_backoff_ns` the receiver chose for
    /// the notice (either propagated from the caller or sized
    /// proportional to the cause).
    EmitNotice {
        /// Minimum back-off the notice will quote.
        min_backoff_ns: u64,
        /// Why the notice is being emitted.
        cause: BackoffCause,
    },
    /// Receiver should not emit (duplicate-suppression window is
    /// still open, or the cause was already covered by a recent
    /// notice).
    Suppress,
    /// Receiver should drop the inbound envelope without emitting a
    /// notice. Returned once the K-without-recovery threshold has
    /// been crossed.
    SilentDrop,
}

/// Per-peer receiver-side back-pressure state.
///
/// Sibling field on `PeerState` per
/// `bb-runtime/src/framework/peer_state.rs`. The tracker is
/// receiver-state-only; sender-side back-off lives in the existing
/// `BackoffTable`.
pub struct BackpressureTracker {
    entries: HashMap<PeerId, BackpressureEntry>,
    high_water_mark_pct: u8,
    notice_threshold_k: u32,
    min_notice_interval_ns: u64,
}

impl Default for BackpressureTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BackpressureTracker {
    /// Construct a fresh tracker with the spec's defaults
    /// (high-water = 75%, K = 3, min-notice-interval = 1 second).
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_HIGH_WATER_PCT,
            DEFAULT_K_BEFORE_SILENT,
            DEFAULT_MIN_NOTICE_INTERVAL_NS,
        )
    }

    /// Construct a tracker with custom config values. `high_water_mark_pct`
    /// is clamped to `1..=100`; `notice_threshold_k` is clamped to
    /// at least 1; `min_notice_interval_ns` is clamped to at least 1.
    pub fn with_config(
        high_water_mark_pct: u8,
        notice_threshold_k: u32,
        min_notice_interval_ns: u64,
    ) -> Self {
        Self {
            entries: HashMap::new(),
            high_water_mark_pct: high_water_mark_pct.clamp(1, 100),
            notice_threshold_k: notice_threshold_k.max(1),
            min_notice_interval_ns: min_notice_interval_ns.max(1),
        }
    }

    /// High-water mark threshold as a percentage.
    pub fn high_water_mark_pct(&self) -> u8 {
        self.high_water_mark_pct
    }

    /// Whether the supplied queue depth (`len`) crosses the
    /// configured high-water mark for the supplied capacity.
    pub fn is_over_high_water(&self, len: usize, capacity: usize) -> bool {
        if capacity == 0 {
            return false;
        }
        // Use 128-bit arithmetic to avoid overflow on
        // pathologically large capacities; cast back is safe
        // because `100 * usize::MAX` fits in u128.
        let lhs = (len as u128).saturating_mul(100);
        let rhs = (capacity as u128).saturating_mul(self.high_water_mark_pct as u128);
        lhs >= rhs
    }

    /// K threshold (notices-without-recovery before silent-drop).
    pub fn notice_threshold_k(&self) -> u32 {
        self.notice_threshold_k
    }

    /// Minimum interval enforced between successive notices to the
    /// same peer.
    pub fn min_notice_interval_ns(&self) -> u64 {
        self.min_notice_interval_ns
    }

    /// Whether the peer is currently in silent-drop mode.
    pub fn is_silent_drop_active(&self, peer: PeerId) -> bool {
        self.entries
            .get(&peer)
            .is_some_and(|e| e.silent_drop_active)
    }

    /// Inspect the recorded entry for `peer`. Returns `None` when
    /// no overload event has been observed for this peer yet.
    pub fn entry(&self, peer: PeerId) -> Option<BackpressureEntry> {
        self.entries.get(&peer).copied()
    }

    /// Iterate `(PeerId, BackpressureEntry)` for snapshot capture.
    pub fn iter(&self) -> impl Iterator<Item = (PeerId, BackpressureEntry)> + '_ {
        self.entries.iter().map(|(p, e)| (*p, *e))
    }

    /// Number of peers currently tracked.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any peer has been tracked.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Observe an overload condition for `peer` at `now_ns`.
    ///
    /// Returns:
    /// - `Decision::SilentDrop` if the peer is already in
    ///   silent-drop mode. The caller drops the envelope; no
    ///   notice is emitted.
    /// - `Decision::Suppress` if a recent notice's quoted back-off
    ///   window has not yet elapsed (duplicate suppression).
    /// - `Decision::EmitNotice` if the caller should emit a notice.
    ///   The tracker increments `notices_sent` + records the
    ///   emission timestamp + back-off. If this push crosses the
    ///   K threshold, the entry transitions to `silent_drop_active`
    ///   in the *next* observation - the current decision still
    ///   emits the K-th notice so the sender gets the final
    ///   warning before silent drop kicks in.
    ///
    /// `min_backoff_ns` is the back-off duration the caller intends
    /// to quote on the notice. The tracker uses it for the
    /// duplicate-suppression window. A 0 value collapses to the
    /// configured `min_notice_interval_ns` floor.
    pub fn observe_overload(
        &mut self,
        peer: PeerId,
        cause: BackoffCause,
        min_backoff_ns: u64,
        now_ns: u64,
    ) -> Decision {
        // Capture the prior emit-state so we can decide between
        // Suppress (early-return without mutation) and EmitNotice
        // (which mutates) before the per-entry borrow extends.
        let prior = self.entries.get(&peer).copied().unwrap_or_default();

        if prior.silent_drop_active {
            return Decision::SilentDrop;
        }

        let effective_min = min_backoff_ns.max(self.min_notice_interval_ns);

        // Duplicate suppression: skip emission while inside the
        // previously-quoted back-off window.
        if prior.last_notice_at_ns != 0
            && now_ns
                < prior
                    .last_notice_at_ns
                    .saturating_add(prior.last_min_backoff_ns)
        {
            return Decision::Suppress;
        }

        // Emit. Bump the counter + record the emission window.
        let new_notices = prior.notices_sent.saturating_add(1);
        let silent_drop_active = new_notices >= self.notice_threshold_k;
        self.entries.insert(
            peer,
            BackpressureEntry {
                notices_sent: new_notices,
                last_notice_at_ns: now_ns,
                last_min_backoff_ns: effective_min,
                silent_drop_active,
            },
        );
        Decision::EmitNotice {
            min_backoff_ns: effective_min,
            cause,
        }
    }

    /// Record that the sender has recovered (e.g., φ-accrual
    /// transitioned back to `Live`). Resets the per-peer counter
    /// and clears `silent_drop_active`. The next `observe_overload`
    /// for the peer starts fresh.
    pub fn record_recovery(&mut self, peer: PeerId) {
        self.entries.remove(&peer);
    }

    /// Whether `peer` is currently inside its duplicate-suppression
    /// window. Used by tests + introspection.
    pub fn in_suppression_window(&self, peer: PeerId, now_ns: u64) -> bool {
        let Some(entry) = self.entries.get(&peer) else {
            return false;
        };
        if entry.last_notice_at_ns == 0 {
            return false;
        }
        now_ns
            < entry
                .last_notice_at_ns
                .saturating_add(entry.last_min_backoff_ns)
    }
}

