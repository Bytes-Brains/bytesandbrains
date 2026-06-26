//! `RequestTracker` - mint correlation tokens for `CorrelateTag` and
//! track in-flight wire requests.
//!
//! 7 token minter with an `in_flight` map keyed by `wire_req_id`.
//! `register_in_flight` records the dispatch-time clock + target
//! site + (optional) chain context; `observe_response` pops the
//! entry on the matching response landing and surfaces the data the
//! [`crate::framework::rtt_tracker::RttTracker`] needs to update its
//! EMAs.

use std::collections::HashMap;

use crate::ids::{CommandId, NodeSiteId};

use crate::framework::rtt_tracker::ChainContext;

/// Bookkeeping for a wire round-trip in flight.
#[derive(Clone, Copy, Debug)]
pub struct InFlightSend {
    /// Engine-clock timestamp when the send was dispatched.
    pub started_at_ns: u64,
    /// Engine-clock timestamp at which this entry should be evicted
    /// as timed-out. Zero means "no TTL — never evict on age".
    pub expires_at_ns: u64,
    /// Destination logical site, derived from the resolved PeerId.
    pub target_site: NodeSiteId,
    /// Optional compiler-stamped chain context.
    pub chain: Option<ChainContext>,
    /// `CommandId` of the originator's local op parked waiting for
    /// the chain's response. When `drain_stale` evicts on TTL, the
    /// engine routes a `WireTimeout` failure through this CommandId
    /// so the parked continuation unsticks. `None` for entries
    /// registered outside an async-suspension context (e.g.
    /// fire-and-forget Sends).
    pub parked_op: Option<crate::ids::CommandId>,
}

/// Round-trip sample surfaced by [`RequestTracker::observe_response`].
#[derive(Clone, Copy, Debug)]
pub struct RoundTripSample {
    /// Destination site for the round trip.
    pub target_site: NodeSiteId,
    /// Chain context if one was recorded at dispatch.
    pub chain: Option<ChainContext>,
    /// Elapsed wall-clock time, nanoseconds.
    pub elapsed_ns: u64,
}

/// Monotonically-increasing token minter + in-flight tracker.
#[derive(Default)]
pub struct RequestTracker {
    next_token: u64,
    in_flight: HashMap<u64, InFlightSend>,
}

impl RequestTracker {
    /// Construct fresh with token counter at 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a fresh correlation token (used by `CorrelateTag` +
    /// `wire::Send` to identify the round trip).
    pub fn mint_token(&mut self) -> CommandId {
        let token = self.next_token;
        self.next_token = self.next_token.saturating_add(1);
        CommandId::from(token)
    }

    /// Record an outbound wire round-trip's dispatch-time
    /// bookkeeping. `wire_req_id` is the token returned by
    /// [`Self::mint_token`]; reused on the response side to match.
    /// `ttl_ns` is the maximum age the entry will survive before
    /// the tracker considers it stale (typed as
    /// `NonZeroU64` so callers cannot accidentally register an
    /// entry that never expires, which was the only failure mode
    /// the `RequestTracker.in_flight` map had as an unbounded
    /// resource). `parked_op` is the CommandId of the originator's
    /// local op parked waiting for the chain response; `None` for
    /// entries registered outside an async-suspension context.
    pub fn register_in_flight(
        &mut self,
        wire_req_id: u64,
        started_at_ns: u64,
        target_site: NodeSiteId,
        chain: Option<ChainContext>,
        ttl_ns: std::num::NonZeroU64,
        parked_op: Option<crate::ids::CommandId>,
    ) {
        let expires_at_ns = started_at_ns.saturating_add(ttl_ns.get());
        self.in_flight.insert(
            wire_req_id,
            InFlightSend {
                started_at_ns,
                expires_at_ns,
                target_site,
                chain,
                parked_op,
            },
        );
    }

    /// Pop the in-flight entry for a landing response and surface
    /// the elapsed-time sample the RTT tracker should observe.
    /// Returns `None` when no matching in-flight entry exists (the
    /// response is unsolicited, duplicate, or fired before the
    /// tracker was populated).
    pub fn observe_response(&mut self, wire_req_id: u64, now_ns: u64) -> Option<RoundTripSample> {
        let entry = self.in_flight.remove(&wire_req_id)?;
        let elapsed_ns = now_ns.saturating_sub(entry.started_at_ns);
        Some(RoundTripSample {
            target_site: entry.target_site,
            chain: entry.chain,
            elapsed_ns,
        })
    }

    /// Read-only peek at an in-flight entry without consuming it.
    pub fn peek(&self, wire_req_id: u64) -> Option<&InFlightSend> {
        self.in_flight.get(&wire_req_id)
    }

    /// Count of currently in-flight round trips.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Drain in-flight entries whose per-entry `expires_at_ns` has
    /// passed (entries with `expires_at_ns == 0` never expire on
    /// age). Returns the dropped entries so the engine can emit
    /// `EngineStep::WireTimeout` and fail any parked CommandId.
    pub fn drain_stale(&mut self, now_ns: u64) -> Vec<(u64, InFlightSend)> {
        let drained: Vec<(u64, InFlightSend)> = self
            .in_flight
            .iter()
            .filter(|(_, entry)| entry.expires_at_ns > 0 && entry.expires_at_ns <= now_ns)
            .map(|(k, v)| (*k, *v))
            .collect();
        for (k, _) in &drained {
            self.in_flight.remove(k);
        }
        drained
    }
}

