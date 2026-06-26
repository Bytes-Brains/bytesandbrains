//! 9 framework primitives the engine bundles into the
//! `RuntimeResourceRef` for every `dispatch_atomic` call per
//! `docs/ENGINE.md` §10 + `docs/internal/IMPLEMENTATION_PLAN.md` //! lines 770-779.
//!
//! ships real impls for Scheduler / PeerGate /
//! RequestTracker + adds 3 new primitives (serialize_queue /
//! hold_table / record_buffer). Other primitives stay minimal until
pub mod address_book;
pub mod backoff_table;
pub mod backpressure_notice;
pub mod backpressure_tracker;
pub mod event_source;
pub mod hold_table;
pub mod inbound_dedup;
pub mod outbound_queue;
pub mod peer_gate;
pub mod peer_governor;
pub mod peer_state;
pub mod record_buffer;
pub mod request_tracker;
pub mod rng;
pub mod rtt_tracker;
pub mod scheduler;
pub mod serialize_queue;

pub use address_book::{
    Address, AddressBook, AddressBookError, AddressError, Multiaddress, Protocol,
};
pub use backoff_table::BackoffTable;
pub use backpressure_notice::{
    backoff_notice_type_hash, build_backoff_notice_envelope, BackoffCauseWire,
    BackoffNoticePayload, BACKPRESSURE_DOMAIN,
};
pub use backpressure_tracker::{
    BackoffCause, BackpressureEntry, BackpressureTracker, Decision as BackpressureDecision,
    DEFAULT_HIGH_WATER_PCT, DEFAULT_K_BEFORE_SILENT, DEFAULT_MIN_NOTICE_INTERVAL_NS,
};
pub use event_source::EventSource;
pub use hold_table::HoldTable;
pub use inbound_dedup::InboundDedup;
pub use outbound_queue::OutboundQueue;
pub use peer_gate::PeerGate;
pub use peer_governor::{
    BlockReason, Decision, LifecycleTransition, PeerGovernor, PeerHealth, DEFAULT_FAILURE_THRESHOLD,
};
pub use peer_state::PeerState;
pub use record_buffer::RecordBuffer;
pub use request_tracker::RequestTracker;
pub use rng::{CounterRng, GetrandomU64, RngU64Source};
pub use scheduler::{Scheduler, TimerKind};
pub use serialize_queue::SerializeQueue;

/// Bundle of framework primitives held on the `Engine` per
// Re-export AppEvent for FrameworkComponents.pending_app_events.
use crate::bus::AppEvent;

/// `docs/ENGINE.md` §3 bundle of framework primitives. Split-borrowed
/// into each `dispatch_atomic` call's `RuntimeResourceRef`.
pub struct FrameworkComponents {
    /// Sorted timer heap. `Sleep`/`Interval`/`Pulse` syscalls
    /// schedule entries here; Phase 4 of the poll cycle drains
    /// matured timers and re-fires their consumer ops.
    pub scheduler: Scheduler,
    /// Consolidated per-peer state: named concurrency gate, policy +
    /// health governor, and exponential backoff. Component authors
    /// reach the three sub-primitives through `peer_state.{gate,
    /// governor, backoff}`.
    pub peer_state: PeerState,
    /// In-flight wire-request → CommandId map + token minter.
    pub request_tracker: RequestTracker,
    /// Sliding-window seen-message tracker.
    pub inbound_dedup: InboundDedup,
    /// `PeerId → Address` mapping.
    pub address_book: AddressBook,
    /// Per-NodeSiteId adaptive RTT tracker driving deadline
    /// derivation for every wire round-trip the engine observes. Fed
    /// by `Engine::wire_send_tracked` on send + by the response path
    /// on completion.
    pub rtt_tracker: rtt_tracker::RttTracker,
    /// FIFO of wire envelopes ready to ship.
    pub outbound_queue: OutboundQueue,
    /// Registered `EventKind → ComponentTag` subscriptions.
    pub event_source: EventSource,
    /// Named-FIFO map for Serialize.Enqueue / Dequeue.
    pub serialize_queue: SerializeQueue,
    /// Named-slot value buffer for Hold.Stash / Flush.
    pub hold_table: HoldTable,
    /// Per-name bounded ring buffer for Record.
    pub record_buffer: RecordBuffer,
    /// App events pending Phase 8 emission.
    pub pending_app_events: Vec<AppEvent>,
    /// Per-Node counters bumped by `IncrMetric` syscalls.
    pub counters: std::collections::HashMap<String, u64>,
    /// `u64` RNG source used by the `RngU64` syscall.
    pub rng: Box<dyn RngU64Source>,
    /// Per-`group` first-arrival latch for the `Any` syscall. Once a
    /// group fires, subsequent arrivals are absorbed without
    /// re-firing. Cleared on snapshot restore via the framework
    /// reset.
    pub any_fired_groups: std::collections::HashSet<String>,
    /// Per-`(OpRef, ExecId)` latch for the `DeadlineMatch` syscall.
    /// First invocation per execution determines the winner (`then`
    /// if non-empty, otherwise `timeout`); subsequent invocations
    /// inside the same execution are absorbed. New executions start
    /// fresh — the latch is keyed by ExecId, not just OpRef, so a
    /// DeadlineMatch op fires once per logical execution rather
    /// than once per Node lifetime.
    pub deadline_match_fired: std::collections::HashSet<(u64, u64)>,
    /// Peer-resolution failures captured during `wire::Send`
    /// dispatch when the destination `PeerId` either isn't in the
    /// `AddressBook` or maps to an empty address list. The engine
    /// drains this in Phase 8 and surfaces each entry as both a
    /// `EngineStep::PeerResolveFailed` and a bus
    /// `InfraEvent::PeerResolveFailure`. Per
    /// `docs/ADDRESSING.md`.
    pub pending_peer_resolve_failures: Vec<(Option<crate::ids::PeerId>, crate::ids::OpRef)>,
    /// Per-`ExecId` inbound envelope context. Populated by
    /// `Engine::route_envelope` when a wire envelope arrives; read by
    /// RX gates (`PeerHealthGateRx`, `BackoffGateRx`) for src-peer
    /// filtering and by `wire.Send` for in-chain correlation token
    /// reuse + Dapper-style elapsed-time accounting.
    pub inbound_contexts: std::collections::HashMap<crate::ids::ExecId, InboundContext>,
}

/// Per-`ExecId` context captured at inbound envelope delivery.
/// Replaces the four parallel `envelope_*` HashMaps with one struct
/// of optional fields. Components access this through
/// `RuntimeResourceRef::inbound`.
#[derive(Clone, Debug, Default)]
pub struct InboundContext {
    /// Source peer of the inbound envelope, if known. RX gates
    /// (`PeerHealthGateRx`, `BackoffGateRx`) filter on this.
    pub src_peer: Option<crate::ids::PeerId>,
    /// Inbound wire-correlation token. `wire.Send` reuses this when
    /// forwarding inside a chain instead of minting a fresh one.
    /// `None` when the envelope was not part of a request/response
    /// chain.
    pub wire_req_id: Option<u64>,
    /// Arrival timestamp (engine ns). `wire.Send` subtracts this from
    /// `now_ns` for Dapper-style elapsed-time accounting.
    pub arrival_ns: Option<u64>,
    /// Remaining deadline propagated by the sender. `wire.Send` carries
    /// this forward (minus elapsed) instead of re-estimating from RTT.
    pub remaining_deadline_ns: Option<u64>,
}

impl FrameworkComponents {
    /// Construct a fresh bundle.
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            peer_state: PeerState::new(),
            request_tracker: RequestTracker::new(),
            inbound_dedup: InboundDedup::new(),
            address_book: AddressBook::new(),
            rtt_tracker: rtt_tracker::RttTracker::new(),
            outbound_queue: OutboundQueue::new(),
            event_source: EventSource::new(),
            serialize_queue: SerializeQueue::new(),
            hold_table: HoldTable::new(),
            record_buffer: RecordBuffer::new(),
            pending_app_events: Vec::new(),
            counters: std::collections::HashMap::new(),
            rng: Box::new(GetrandomU64::new()),
            any_fired_groups: std::collections::HashSet::new(),
            deadline_match_fired: std::collections::HashSet::new(),
            pending_peer_resolve_failures: Vec::new(),
            inbound_contexts: std::collections::HashMap::new(),
        }
    }
}

impl Default for FrameworkComponents {
    fn default() -> Self {
        Self::new()
    }
}
