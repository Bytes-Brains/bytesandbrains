//! `PeerState` — the framework's consolidated per-peer state.
//!
//! The engine's four peer-related primitives — `PeerGate` (named
//! concurrency limiter), `PeerGovernor` (policy + health source of
//! truth), `BackoffTable` (per-peer exponential backoff), and
//! `BackpressureTracker` (receiver-side overload state for the
//! backpressure protocol per
//! `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`)
//! — cluster under one struct. Component authors reach them through
//! `ctx.peer_state.{gate, governor, backoff, backpressure}` instead
//! of four sibling fields on the framework bundle.

use crate::framework::{BackoffTable, BackpressureTracker, PeerGate, PeerGovernor};

/// Consolidated per-peer state.
///
/// Each sub-field retains its existing API; the consolidation is
/// purely organisational so `FrameworkComponents` carries one peer
/// field instead of four. Future work can merge the sub-fields
/// into a single per-peer entry table.
pub struct PeerState {
    /// Named concurrency limiter consulted by `Limit.Acquire` /
    /// `Limit.Release` syscalls.
    pub gate: PeerGate,
    /// Single source of truth for peer policy (blocklist /
    /// allowlist) + per-peer health tracking. Consulted by inbound
    /// + outbound peer-health gates.
    pub governor: PeerGovernor,
    /// Per-peer exponential backoff schedule. Consulted by the
    /// outbound governor at `check_outbound` time.
    pub backoff: BackoffTable,
    /// Receiver-side per-peer back-pressure state. Tracks notices
    /// emitted to each sender, the duplicate-suppression window,
    /// and the K-then-silent fallback. Consulted at Phase 1 of the
    /// engine poll cycle when ingress overload is detected.
    pub backpressure: BackpressureTracker,
}

impl Default for PeerState {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerState {
    /// Construct a fresh consolidated peer-state bundle.
    pub fn new() -> Self {
        Self {
            gate: PeerGate::new(),
            governor: PeerGovernor::new(),
            backoff: BackoffTable::new(),
            backpressure: BackpressureTracker::new(),
        }
    }
}
