//! `PeerGovernor` - the single source of truth for peer policy
//! and health tracking per .
//!
//! The framework owns this Component; it's consulted at both
//! delivery boundaries:
//!
//! - **Inbound **: Phase 1 of `Engine::poll` calls
//!   [`PeerGovernor::check_inbound`] for every
//!   `IngressEvent::EnvelopeFrom { src_peer, .. }`. Blocked /
//!   non-allowlisted peers are dropped before any slot is written.
//!
//! - **Outbound **: the compiler pass
//!   `bb-compiler/src/insert_peer_health_gates.rs` inserts a
//!   `PeerHealthGate` syscall op upstream of every `wire::Send`.
//!   The gate's `dispatch_atomic` calls
//!   [`PeerGovernor::check_outbound`].
//!
//! Health state (per-peer consecutive failures + last-seen) is
//! updated by Send-completion callbacks so Component authors stop
//! touching [`BackoffTable`] by hand - the compiler wires the
//! tracking in for every wire send automatically.
//!
//! [`BackoffTable`]: crate::framework::BackoffTable

use std::collections::{HashMap, HashSet};

use crate::ids::PeerId;

/// Default number of consecutive `wire::Send` failures before a
/// peer is marked as down. `PeerGovernor::record_failure` emits
/// the lifecycle transition; the engine's Phase 8 surfaces it as
/// `EngineStep::PeerDown`.
pub const DEFAULT_FAILURE_THRESHOLD: u32 = 5;

/// Why a peer can't receive an envelope, surfaced both inbound
/// (drop) and outbound (gate failure).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlockReason {
    /// Peer is explicitly in the blocklist.
    Blocklisted,
    /// An allowlist is configured and the peer isn't on it.
    NotAllowlisted,
    /// Peer is currently in failure-driven cooldown.
    Cooldown {
        /// `scheduler.now_ns()` past which the peer becomes
        /// eligible again.
        retry_ns: u64,
    },
}

/// Per-peer health snapshot. Read by `Node::peer_health()` for
/// operator introspection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PeerHealth {
    /// Number of consecutive `record_failure` calls since the
    /// last `record_success`.
    pub consecutive_failures: u32,
    /// `scheduler.now_ns()` at the last `record_success` /
    /// `record_failure`. `0` if neither has been called.
    pub last_event_ns: u64,
    /// True once `consecutive_failures` has reached the
    /// configured threshold without an intervening success;
    /// remains true until a success clears the streak.
    pub down: bool,
}

/// Outcome of a `check_inbound` / `check_outbound` consultation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Delivery / send is permitted.
    Allow,
    /// Delivery / send is denied; `reason` carries why.
    Deny(BlockReason),
}

/// Side-effect of recording a failure or success - the engine
/// translates these into `EngineStep` variants in Phase 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleTransition {
    /// No observable state change.
    None,
    /// Peer just crossed below the failure threshold.
    WentDown,
    /// Peer just came back up after a failure streak.
    CameUp,
}

/// Per-peer policy + health state owner.
///
/// Snapshot/restore: full state is captured in
/// `FrameworkSnapshot` (work). Pre-Stage-5, restore
/// rebuilds the governor from scratch - blocklist + allowlist
/// settings on `NodeConfig` are re-applied at construction time.
pub struct PeerGovernor {
    blocklist: HashSet<PeerId>,
    allowlist: Option<HashSet<PeerId>>,
    health: HashMap<PeerId, PeerHealth>,
    failure_threshold: u32,
}

impl Default for PeerGovernor {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerGovernor {
    /// Construct a fresh governor with no blocklist, no
    /// allowlist, and the default failure threshold.
    pub fn new() -> Self {
        Self {
            blocklist: HashSet::new(),
            allowlist: None,
            health: HashMap::new(),
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
        }
    }

    /// Configure the failure threshold (consecutive failures
    /// before a peer is marked down). Default
    /// [`DEFAULT_FAILURE_THRESHOLD`].
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold.max(1);
        self
    }

    /// Add a peer to the blocklist. Subsequent inbound envelopes
    /// from this peer are dropped; subsequent outbound sends fail.
    pub fn block(&mut self, peer: PeerId) {
        self.blocklist.insert(peer);
    }

    /// Remove a peer from the blocklist.
    pub fn unblock(&mut self, peer: PeerId) {
        self.blocklist.remove(&peer);
    }

    /// Set (or clear) the allowlist. When `Some`, only peers in
    /// the allowlist may communicate; everyone else is denied.
    /// `None` means "open" (default).
    pub fn set_allowlist(&mut self, allowlist: Option<HashSet<PeerId>>) {
        self.allowlist = allowlist;
    }

    /// Per-peer health snapshot read; returns `None` if no
    /// success/failure has ever been recorded for `peer`.
    pub fn peer_health(&self, peer: PeerId) -> Option<PeerHealth> {
        self.health.get(&peer).copied()
    }

    /// True when `peer` is currently marked down.
    pub fn is_down(&self, peer: PeerId) -> bool {
        self.health.get(&peer).is_some_and(|h| h.down)
    }

    /// Inbound consultation - called from the engine's Phase 1
    /// envelope router before any slot is written.
    pub fn check_inbound(&self, peer: PeerId) -> Decision {
        if self.blocklist.contains(&peer) {
            return Decision::Deny(BlockReason::Blocklisted);
        }
        if let Some(allow) = &self.allowlist {
            if !allow.contains(&peer) {
                return Decision::Deny(BlockReason::NotAllowlisted);
            }
        }
        Decision::Allow
    }

    /// Outbound consultation - called from the compiler-inserted
    /// `PeerHealthGate` syscall op upstream of every `wire::Send`.
    /// Returns `Deny(Cooldown { retry_ns })` when the peer is in
    /// failure cooldown, prompting the gate to reschedule.
    pub fn check_outbound(
        &self,
        peer: PeerId,
        backoff: &super::BackoffTable,
        now_ns: u64,
    ) -> Decision {
        if self.blocklist.contains(&peer) {
            return Decision::Deny(BlockReason::Blocklisted);
        }
        if let Some(allow) = &self.allowlist {
            if !allow.contains(&peer) {
                return Decision::Deny(BlockReason::NotAllowlisted);
            }
        }
        if !backoff.should_retry(peer, now_ns) {
            let retry_ns = backoff
                .state(peer)
                .map(|s| s.next_retry_ns)
                .unwrap_or(now_ns);
            return Decision::Deny(BlockReason::Cooldown { retry_ns });
        }
        Decision::Allow
    }

    /// Record a successful exchange with `peer` at `now_ns`.
    /// Resets the consecutive-failure counter; clears `down`.
    /// Returns the lifecycle transition the engine should
    /// surface as an `EngineStep::PeerUp`.
    pub fn record_success(&mut self, peer: PeerId, now_ns: u64) -> LifecycleTransition {
        let was_down = self.health.get(&peer).map(|h| h.down).unwrap_or(false);
        self.health.insert(
            peer,
            PeerHealth {
                consecutive_failures: 0,
                last_event_ns: now_ns,
                down: false,
            },
        );
        if was_down {
            LifecycleTransition::CameUp
        } else {
            LifecycleTransition::None
        }
    }

    /// Record a failure for `peer` at `now_ns`. Returns
    /// `WentDown` when the failure pushes the streak across the
    /// threshold.
    pub fn record_failure(&mut self, peer: PeerId, now_ns: u64) -> LifecycleTransition {
        let prev = self.health.get(&peer).copied().unwrap_or_default();
        let consecutive_failures = prev.consecutive_failures.saturating_add(1);
        let just_went_down = !prev.down && consecutive_failures >= self.failure_threshold;
        self.health.insert(
            peer,
            PeerHealth {
                consecutive_failures,
                last_event_ns: now_ns,
                down: prev.down || just_went_down,
            },
        );
        if just_went_down {
            LifecycleTransition::WentDown
        } else {
            LifecycleTransition::None
        }
    }

    /// Number of distinct peers with health state recorded.
    pub fn tracked_peers(&self) -> usize {
        self.health.len()
    }

    /// Read-only view of the blocklist. Used by snapshot capture.
    pub fn blocklist(&self) -> &HashSet<PeerId> {
        &self.blocklist
    }

    /// Read-only view of the allowlist.
    pub fn allowlist(&self) -> Option<&HashSet<PeerId>> {
        self.allowlist.as_ref()
    }

    /// Iterate `(PeerId, PeerHealth)` entries for snapshot capture.
    pub fn iter_health(&self) -> impl Iterator<Item = (PeerId, PeerHealth)> + '_ {
        self.health.iter().map(|(p, h)| (*p, *h))
    }

    /// Current failure threshold (consecutive failures to mark a
    /// peer down). Used by snapshot capture.
    pub fn failure_threshold(&self) -> u32 {
        self.failure_threshold
    }

    /// Replace a peer's health state directly. Used by
    /// `Node::restore` to re-seed health from a snapshot entry.
    pub fn restore_health(&mut self, peer: PeerId, health: PeerHealth) {
        self.health.insert(peer, health);
    }

    /// Overwrite the failure threshold. Used by `Node::restore`.
    pub fn set_failure_threshold(&mut self, threshold: u32) {
        self.failure_threshold = threshold.max(1);
    }
}

