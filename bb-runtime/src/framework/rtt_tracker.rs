//! for adaptive deadlines on every wire round-trip.
//!
//! Sits alongside [`super::address_book::AddressBook`] in the
//! framework. Keyed by [`crate::ids::NodeSiteId`] so a single
//! physical peer hosting two logical sites (a fast ping handler +
//! an async GPU compute handler) keeps independent EMAs.
//!
//! ## Hierarchical fallback for `estimate_budget_ns`
//!
//! When the engine needs a deadline for a Send to a site, it walks
//! these tiers in order, stopping at the first warm hit:
//!
//! 1. Per-edge stats for `(site, chain_id, hop_index)` - exact
//!    match in this chain context.
//! 2. Per-site aggregate Jacobson - every round-trip to this site
//!    feeds this EMA regardless of context.
//! 3. Per-chain prior - refines the global "what's typical for the
//!    kind of topology this chain represents" based on any peer
//!    that's carried chain traffic before.
//! 4. Global prior - every round-trip in the runtime feeds this
//!    EMA with a small learning rate.
//! 5. Static `NodeConfig.per_hop_budget_ns` fallback.
//!
//! ## Reverse-path piggyback (consumed in Phase 3e-iii)
//!
//! On response landing, the runtime parses [`EdgeRttReport`] entries
//! that downstream sites attached. Each report becomes a
//! `reported_outgoing` entry on the caller's per-site
//! [`RttTrackerEntry`] so multi-hop chain budgets can compose from
//! one address-book entry per direct neighbor.

use std::collections::HashMap;

use crate::ids::NodeSiteId;

/// Stable identifier for a chain. Hash of the compiler-stamped
/// comma-separated `chain_targets` string. Producer and consumer
/// derive the same value from the same string, so cross-site EMAs
/// align without exchanging the raw chain composition.
pub type ChainId = u64;

/// Hash the compiler's `ai.bytesandbrains.wire.chain_targets`
/// metadata value into a stable [`ChainId`]. The hash is FNV-1a
/// over the raw bytes - fast, no allocations, deterministic across
/// runs.
pub fn chain_id_from_targets(chain_targets: &str) -> ChainId {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in chain_targets.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Per-(chain, hop) refinement key.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EdgeKey {
    /// Hash of the compiler-stamped `chain_targets` CSV.
    pub chain_id: ChainId,
    /// Zero-based hop position in the chain.
    pub hop_index: u8,
}

/// Jacobson/Karels RTT EMA: smoothed SRTT + smoothed RTTVAR with
/// sample-count tracking. Mirrors RFC 6298 §2 with α = 1/8 and
/// β = 1/4.
///
/// The deadline-derivation formula `SRTT + 4·RTTVAR` follows the
/// canonical Karn/Partridge recommendation for retransmission
/// timeout. [`Self::is_warm`] gates the per-tier fallback so cold
/// EMAs (fewer than three samples) fall through to coarser priors.
#[derive(Clone, Copy, Debug, Default)]
pub struct RttEma {
    /// Smoothed round-trip time, nanoseconds.
    pub srtt_ns: u64,
    /// Smoothed round-trip-time variance, nanoseconds.
    pub rttvar_ns: u64,
    /// Count of samples observed.
    pub sample_count: u64,
}

impl RttEma {
    /// Observe a round-trip sample. Updates SRTT + RTTVAR using
    /// Jacobson's α = 1/8 / β = 1/4 weights.
    pub fn observe(&mut self, sample_ns: u64) {
        self.observe_with_alpha_beta(sample_ns, 3, 2);
    }

    /// Observe a round-trip sample with a smaller learning rate.
    /// Used by the global prior to dampen the influence of any one
    /// peer's bursty samples on the cross-runtime estimate.
    ///
    /// `alpha_shift` = log2(1/α), `beta_shift` = log2(1/β). Values
    /// of (3, 2) match the Jacobson recommendation (α=1/8, β=1/4);
    /// larger shifts (e.g. (6, 5)) make the EMA more conservative.
    pub fn observe_with_alpha_beta(&mut self, sample_ns: u64, alpha_shift: u8, beta_shift: u8) {
        if self.sample_count == 0 {
            // RFC 6298 §2.2: first sample initializes SRTT = sample
            // and RTTVAR = sample / 2.
            self.srtt_ns = sample_ns;
            self.rttvar_ns = sample_ns >> 1;
        } else {
            // RTTVAR ← (1 − β)·RTTVAR + β·|SRTT − sample|
            let delta = sample_ns.abs_diff(self.srtt_ns);
            let beta_div = 1u64 << beta_shift;
            self.rttvar_ns =
                self.rttvar_ns - (self.rttvar_ns >> beta_shift) + (delta >> beta_shift);
            // SRTT ← (1 − α)·SRTT + α·sample
            let alpha_div = 1u64 << alpha_shift;
            self.srtt_ns =
                self.srtt_ns - (self.srtt_ns >> alpha_shift) + (sample_ns >> alpha_shift);
            let _ = beta_div;
            let _ = alpha_div;
        }
        self.sample_count = self.sample_count.saturating_add(1);
    }

    /// Recommended budget: `SRTT + 4·RTTVAR` per RFC 6298 §2.3.
    pub fn budget_ns(&self) -> u64 {
        self.srtt_ns
            .saturating_add(self.rttvar_ns.saturating_mul(4))
    }

    /// "Warm" once we have three samples - gates the fallback
    /// hierarchy so very-fresh EMAs don't poison budgets.
    pub fn is_warm(&self) -> bool {
        self.sample_count >= 3
    }
}

/// φ-accrual failure detector per direct chain neighbor. Heartbeat
/// = any wire round-trip in the last window; rising φ indicates the
/// peer is silent relative to its historical inter-arrival
/// distribution.
///
/// Implementation per Hayashibara et al. - exponential
/// approximation of the empirical inter-arrival distribution. The
/// threshold defaults to 8 (Cassandra / Akka conservative).
#[derive(Clone, Debug)]
pub struct PhiAccrualState {
    /// Recent inter-arrival times of heartbeats, nanoseconds.
    ///
    /// `VecDeque` so eviction at capacity is
    /// O(1) `pop_front` instead of O(n) `Vec::remove(0)` memmove.
    /// At `history_capacity = 1000` and per-heartbeat ingest, the
    /// quadratic cost dominated when φ-accrual ran on dozens of
    /// peers in steady state.
    pub inter_arrival_history: std::collections::VecDeque<u64>,
    /// Capacity of the rolling history.
    pub history_capacity: usize,
    /// Suspicion threshold; φ > this → peer is suspect.
    pub threshold_phi: f64,
    /// Hard-down threshold; φ > this → peer is down.
    pub down_phi: f64,
}

impl Default for PhiAccrualState {
    fn default() -> Self {
        Self {
            inter_arrival_history: std::collections::VecDeque::new(),
            history_capacity: 1000,
            threshold_phi: 8.0,
            down_phi: 16.0,
        }
    }
}

impl PhiAccrualState {
    /// Record a heartbeat at `now_ns` given the prior heartbeat at
    /// `last_seen_at_ns`. The inter-arrival time enters the rolling
    /// history (oldest sample evicted at capacity).
    pub fn record_heartbeat(&mut self, now_ns: u64, last_seen_at_ns: u64) {
        if last_seen_at_ns == 0 {
            return;
        }
        let delta = now_ns.saturating_sub(last_seen_at_ns);
        if self.inter_arrival_history.len() == self.history_capacity {
            self.inter_arrival_history.pop_front();
        }
        self.inter_arrival_history.push_back(delta);
    }

    /// Compute current suspicion level φ. Returns 0.0 when no
    /// history is available (i.e., no heartbeats yet - the peer is
    /// assumed alive on the first contact).
    pub fn phi(&self, now_ns: u64, last_seen_at_ns: u64) -> f64 {
        if self.inter_arrival_history.is_empty() || last_seen_at_ns == 0 {
            return 0.0;
        }
        let elapsed = now_ns.saturating_sub(last_seen_at_ns) as f64;
        let sum: f64 = self.inter_arrival_history.iter().map(|&x| x as f64).sum();
        let mean = sum / self.inter_arrival_history.len() as f64;
        if mean <= 0.0 {
            return 0.0;
        }
        // Exponential approximation: φ = -log10(P_later(elapsed)) =
        // elapsed / (mean · ln(10)). Hayashibara §5.1.
        elapsed / (mean * std::f64::consts::LN_10)
    }

    /// `true` once φ crosses [`Self::threshold_phi`].
    pub fn is_suspect(&self, now_ns: u64, last_seen_at_ns: u64) -> bool {
        self.phi(now_ns, last_seen_at_ns) > self.threshold_phi
    }

    /// `true` once φ crosses [`Self::down_phi`] (hard fail).
    pub fn is_down(&self, now_ns: u64, last_seen_at_ns: u64) -> bool {
        self.phi(now_ns, last_seen_at_ns) > self.down_phi
    }
}

/// One AddressBook-side entry per logical site we've ever observed.
#[derive(Default)]
pub struct RttTrackerEntry {
    /// Aggregate Jacobson over ALL round-trips to this site, any
    /// context. Fed by data plane, control plane, handshakes,
    /// anything using `Engine::wire_send_tracked`.
    pub site_stats: RttEma,

    /// Per-(chain, hop) refinement specific to this site.
    pub per_edge_stats: HashMap<EdgeKey, RttEma>,

    /// Reverse-path piggyback: this site told us about ITS outgoing
    /// edges in chains, indexed by (next-hop, chain_id). Lets a
    /// chain originator compose a multi-hop budget from one entry
    /// per direct neighbor.
    pub reported_outgoing: HashMap<(NodeSiteId, ChainId), RttEma>,

    /// φ-accrual per direct neighbor.
    pub phi_accrual: PhiAccrualState,

    /// Timestamp of the most recent wire round-trip with this site,
    /// nanoseconds since the engine clock epoch.
    pub last_seen_at_ns: u64,

    /// Timestamp of the most recent EMA update.
    pub last_updated_at_ns: u64,

    /// -v - last φ state surfaced
    /// by [`RttTracker::scan_phi`]. The scan emits a transition only
    /// when the state changes so subscribers don't get a `Suspect`
    /// event every poll cycle while the site stays silent.
    pub last_phi_state: PhiState,
}

/// Runtime-owned RTT tracker.
///
/// Sits alongside [`super::address_book::AddressBook`] in the
/// framework. Every wire round-trip the engine observes (any
/// protocol, any chain context) feeds [`Self::observe_round_trip`];
/// every outgoing send queries [`Self::estimate_budget_ns`] for
/// its deadline.
#[derive(Default)]
pub struct RttTracker {
    entries: HashMap<NodeSiteId, RttTrackerEntry>,
    /// Per-chain aggregate. Survives peer churn - even if every
    /// peer hosting chain X gets evicted, future peers joining the
    /// chain pick up this prior as their first-guess budget.
    chain_priors: HashMap<ChainId, RttEma>,
    /// Final fallback before the static `NodeConfig` default.
    global_prior: RttEma,
}

/// Optional chain context the engine threads to
/// [`RttTracker::estimate_budget_ns`] and
/// [`RttTracker::observe_round_trip`].
#[derive(Clone, Copy, Debug)]
pub struct ChainContext {
    /// Hash of the compiler-stamped `chain_targets` CSV.
    pub chain_id: ChainId,
    /// Zero-based hop position in the chain.
    pub hop_index: u8,
}

impl RttTracker {
    /// Fresh, empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Hierarchical fallback: per-edge → per-site → per-chain →
    /// global → static. First warm tier wins.
    pub fn estimate_budget_ns(
        &self,
        site: NodeSiteId,
        chain: Option<ChainContext>,
        static_default_ns: u64,
    ) -> u64 {
        // Tier 1: per-edge exact match in this chain context.
        if let (Some(ctx), Some(entry)) = (chain, self.entries.get(&site)) {
            let key = EdgeKey {
                chain_id: ctx.chain_id,
                hop_index: ctx.hop_index,
            };
            if let Some(ema) = entry.per_edge_stats.get(&key) {
                if ema.is_warm() {
                    return ema.budget_ns();
                }
            }
        }
        // Tier 2: per-site aggregate.
        if let Some(entry) = self.entries.get(&site) {
            if entry.site_stats.is_warm() {
                return entry.site_stats.budget_ns();
            }
        }
        // Tier 3: per-chain prior.
        if let Some(ctx) = chain {
            if let Some(prior) = self.chain_priors.get(&ctx.chain_id) {
                if prior.is_warm() {
                    return prior.budget_ns();
                }
            }
        }
        // Tier 4: global prior.
        if self.global_prior.is_warm() {
            return self.global_prior.budget_ns();
        }
        // Tier 5: static default.
        static_default_ns
    }

    /// Feed a round-trip sample. Updates per-site Jacobson EMA
    /// always; per-edge + per-chain when chain context is present;
    /// global prior always (with a smaller learning rate).
    pub fn observe_round_trip(
        &mut self,
        site: NodeSiteId,
        chain: Option<ChainContext>,
        elapsed_ns: u64,
        now_ns: u64,
    ) {
        let entry = self.entries.entry(site).or_default();
        // Per-site aggregate.
        entry.site_stats.observe(elapsed_ns);
        entry.last_updated_at_ns = now_ns;
        // φ-accrual heartbeat
        entry
            .phi_accrual
            .record_heartbeat(now_ns, entry.last_seen_at_ns);
        entry.last_seen_at_ns = now_ns;

        // Per-edge + per-chain refinement.
        if let Some(ctx) = chain {
            let key = EdgeKey {
                chain_id: ctx.chain_id,
                hop_index: ctx.hop_index,
            };
            entry
                .per_edge_stats
                .entry(key)
                .or_default()
                .observe(elapsed_ns);
            self.chain_priors
                .entry(ctx.chain_id)
                .or_default()
                .observe(elapsed_ns);
        }

        // Global prior - small learning rate so noisy samples don't
        // dominate.
        self.global_prior.observe_with_alpha_beta(elapsed_ns, 6, 5);
    }

    /// Ingest a reverse-path piggyback report - a downstream site
    /// telling us about ITS outgoing edge to `next_hop` in chain
    /// `chain_id`.
    pub fn ingest_reported_outgoing(
        &mut self,
        from_site: NodeSiteId,
        next_hop: NodeSiteId,
        chain_id: ChainId,
        srtt_ns: u64,
        rttvar_ns: u64,
        sample_count: u64,
    ) {
        let entry = self.entries.entry(from_site).or_default();
        let report = entry
            .reported_outgoing
            .entry((next_hop, chain_id))
            .or_default();
        report.srtt_ns = srtt_ns;
        report.rttvar_ns = rttvar_ns;
        report.sample_count = sample_count;
    }

    /// Read-only access to a per-site entry. Returns `None` when no
    /// round-trip with the site has been observed.
    pub fn entry(&self, site: NodeSiteId) -> Option<&RttTrackerEntry> {
        self.entries.get(&site)
    }

    /// Read-only access to the per-chain prior.
    pub fn chain_prior(&self, chain_id: ChainId) -> Option<&RttEma> {
        self.chain_priors.get(&chain_id)
    }

    /// Read-only access to the global prior.
    pub fn global_prior(&self) -> &RttEma {
        &self.global_prior
    }

    /// Snapshot of every site currently tracked.
    pub fn tracked_sites(&self) -> impl Iterator<Item = NodeSiteId> + '_ {
        self.entries.keys().copied()
    }

    /// -v - scan φ-accrual states
    /// at the current engine clock and surface state transitions.
    /// Returns one entry per tracked site whose suspicion level
    /// changed since the last scan: `PhiTransition::Suspect`,
    /// `Down`, or `Live` (after a previous `Suspect`/`Down` resolves).
    ///
    /// The tracker keeps a per-site `last_phi_state` ratchet so
    /// repeat scans don't re-emit the same event every poll cycle.
    pub fn scan_phi(&mut self, now_ns: u64) -> Vec<PhiTransition> {
        let mut transitions = Vec::new();
        for (&site, entry) in self.entries.iter_mut() {
            let phi = entry.phi_accrual.phi(now_ns, entry.last_seen_at_ns);
            let new_state = if entry.phi_accrual.is_down(now_ns, entry.last_seen_at_ns) {
                PhiState::Down
            } else if entry.phi_accrual.is_suspect(now_ns, entry.last_seen_at_ns) {
                PhiState::Suspect
            } else {
                PhiState::Live
            };
            if new_state != entry.last_phi_state {
                transitions.push(match new_state {
                    PhiState::Live => PhiTransition::Live { site },
                    PhiState::Suspect => PhiTransition::Suspect { site, phi },
                    PhiState::Down => PhiTransition::Down { site, phi },
                });
                entry.last_phi_state = new_state;
            }
        }
        transitions
    }
}

/// Discrete states tracked by [`RttTracker::scan_phi`] for each
/// per-site φ-accrual detector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PhiState {
    /// Default - peer is healthy.
    #[default]
    Live,
    /// φ crossed the suspect threshold.
    Suspect,
    /// φ crossed the hard-down threshold.
    Down,
}

/// State transitions surfaced by [`RttTracker::scan_phi`]. The
/// engine maps each entry onto a bus
/// [`crate::bus::InfraEvent::PeerSuspect`] /
/// [`crate::bus::InfraEvent::PeerDown`] /
/// [`crate::bus::InfraEvent::PeerLive`].
#[derive(Clone, Copy, Debug)]
pub enum PhiTransition {
    /// Site recovered (φ collapsed below suspect threshold).
    Live {
        /// Per-Node site whose φ-accrual detector dropped back below the suspect threshold.
        site: NodeSiteId,
    },
    /// Site crossed the suspect threshold.
    Suspect {
        /// Per-Node site whose φ-accrual detector crossed into the suspect band.
        site: NodeSiteId,
        /// φ value at the moment the suspect threshold was crossed.
        phi: f64,
    },
    /// Site crossed the hard-down threshold.
    Down {
        /// Per-Node site whose φ-accrual detector crossed into the hard-down band.
        site: NodeSiteId,
        /// φ value at the moment the hard-down threshold was crossed.
        phi: f64,
    },
}

