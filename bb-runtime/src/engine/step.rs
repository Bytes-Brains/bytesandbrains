//! `EngineStep` - observable output of `Engine::poll` per
//! `docs/ENGINE.md` §10 + `docs/internal/IMPLEMENTATION_PLAN.md` //! lines 751-753.

use crate::bus::{OpError, WireReceiveErrorKind};
use crate::envelope::WireEnvelope;
use crate::framework::BlockReason;
use crate::ids::{CommandId, ExecId, NodeSiteId, OpRef, PeerId};

/// One step of work the engine performed during a poll cycle.
/// `Engine::poll` returns `Vec<EngineStep>` capturing every event
/// the host can observe.
#[derive(Clone, Debug)]
pub enum EngineStep {
    /// An Op completed successfully + wrote `sites_written` slots.
    OpCompleted {
        /// The completed Op.
        op_ref: OpRef,
        /// The execution it belonged to.
        exec_id: ExecId,
        /// Output sites the Op wrote.
        sites_written: Vec<NodeSiteId>,
    },

    /// An Op suspended on a `CommandId` awaiting completion.
    AsyncSuspended {
        /// The suspended Op.
        op_ref: OpRef,
        /// The execution it belonged to.
        exec_id: ExecId,
        /// The CommandId the Op returned.
        cmd_id: CommandId,
    },

    /// An outbound envelope is ready to ship.
    SendEnvelope(WireEnvelope),

    /// An app-facing event was published. Carries the topic name
    /// (the `function.output` value name, or the `AppEmit` topic for
    /// mid-cycle emissions) plus the serialized value bytes. When the
    /// emitter is a `Notify`-style call with no payload, `value_bytes`
    /// is empty.
    AppEvent {
        /// Module that emitted the event.
        module_name: String,
        /// Topic name - typically a `function.output` name for
        /// top-level surface, or an explicit topic for `AppEmit`.
        topic: String,
        /// Serialized payload - the slot value at the emission site,
        /// encoded via the type's `WireType::to_wire_bytes` path.
        /// Empty for marker-only notifications.
        value_bytes: Vec<u8>,
    },

    /// A lifecycle phase fired. fills in the per-phase
    /// payload.
    LifecycleFired {
        /// Phase name (e.g. `"Bootstrap"`, `"PreShutdown"`).
        phase: String,
    },

    /// The single bootstrap FunctionCall the engine seeded at install
    /// completion drained to quiescence. Body ops fire on the same
    /// poll cycle once this step is emitted. Bootstrap is a one-shot
    /// per Node lifetime; a restored Node does not re-emit (its
    /// bootstrap pass already ran pre-snapshot).
    BootstrapComplete,

    /// At least one bootstrap-phase op returned `DispatchResult::Async`
    /// and the engine is waiting on its completion before activating
    /// body ops. The host drives the completion via the ingress and
    /// re-invokes `Node::run_bootstrap`.
    WaitingOnBootstrap,

    /// An Op failed. The error is also published on the bus as
    /// `InfraEvent::OpFailure`.
    OpFailed {
        /// The failed Op.
        op_ref: OpRef,
        /// The execution it belonged to.
        exec_id: ExecId,
        /// The failure detail.
        error: OpError,
    },

    /// `cycle_op_budget` was hit during a `poll()`. The engine
    /// yielded mid-cascade; the host should poll again to drain
    /// the remaining frontier. Emitted at most once per poll.
    CycleBudgetExceeded {
        /// Number of op-invocations the cycle issued before
        /// yielding (== `cycle_op_budget`).
        ops_invoked: usize,
    },

    /// `max_outbound_queue` was hit since the previous poll;
    /// `count` envelopes were FIFO-dropped to make room. Emitted
    /// at most once per poll.
    OutboundDropped {
        /// Number of envelopes dropped since the last poll.
        count: usize,
    },

    /// An inbound wire envelope's payload could not be decoded.
    /// The envelope's slot fill was dropped; this step lets the
    /// host observe the drop. Carries the same context as the
    /// matching `InfraEvent::WireDecodeFailure` on the bus.
    WireDecodeFailed {
        /// Wire-type hash the envelope advertised (0 if the
        /// failure occurred before the hash could be read).
        hash: u64,
        /// Length of the offending payload, in bytes.
        payload_size: usize,
        /// Human-readable failure detail.
        detail: String,
    },

    /// An inbound wire fill failed the typed-decode step. Mirrors
    /// the bus's [`crate::bus::InfraEvent::WireReceiveError`] so
    /// the host poll() caller observes the per-fill failure
    /// without subscribing to the bus. Other fills in the same
    /// envelope still deliver (partial-delivery semantics).
    WireReceiveFailed {
        /// Sender of the failing envelope, if known.
        src_peer: Option<PeerId>,
        /// Position of the failing fill within the envelope
        /// (0-based).
        fill_index: u32,
        /// The `type_hash` the sender stamped on the fill.
        actual_hash: u64,
        /// Length of the offending payload, in bytes.
        payload_size: usize,
        /// Which failure mode fired.
        kind: WireReceiveErrorKind,
    },

    /// A registered in-flight request entry was evicted by the
    /// engine's per-poll `RequestTracker::drain_stale` sweep because
    /// its per-entry TTL elapsed without a matching response. The
    /// originator's local DAG continuation parked behind
    /// `parked_op` (if `Some`) is failed with "chain timeout" via
    /// the same path async-suspension completions take.
    WireTimeout {
        /// The chain correlation token that timed out.
        wire_req_id: u64,
        /// Destination site the request was dispatched to.
        target_site: crate::ids::NodeSiteId,
        /// Engine-clock timestamp when the originating Send fired.
        started_at_ns: u64,
        /// `CommandId` of the originator's parked local op, if the
        /// request was registered with one.
        parked_op: Option<crate::ids::CommandId>,
    },

    /// An inbound envelope from `peer` was dropped by the
    /// [`crate::framework::PeerGovernor`] before any slot was
    /// written. path -
    /// the "first contact with IP" check the user flagged.
    PeerBlocked {
        /// The peer whose envelope was rejected.
        peer: PeerId,
        /// Why the envelope was rejected.
        reason: BlockReason,
    },

    /// A peer crossed below the failure threshold and is now
    /// marked down. Emitted at most once per transition.
    PeerDown {
        /// The peer that went down.
        peer: PeerId,
    },

    /// A peer recovered after a failure streak.
    PeerUp {
        /// The peer that came back up.
        peer: PeerId,
    },

    /// `wire::Send` could not resolve its destination peer's
    /// addresses against the framework's
    /// [`crate::framework::AddressBook`]. Either the peer is
    /// unknown, its address list is empty, or the Send op's `peer`
    /// input didn't carry a valid `PeerId`. The Send op produces
    /// no envelope; the host application reacts via this event.
    /// Mirrors `InfraEvent::PeerResolveFailure` on the bus -
    /// telemetry-tap parity with the     /// `PeerBlocked`/`PeerDown`/`PeerUp` family.
    PeerResolveFailed {
        /// The peer whose addresses could not be resolved. `None`
        /// when the Send op had no parseable `peer` input.
        peer: Option<PeerId>,
        /// The Send op that failed to resolve.
        op_ref: OpRef,
        /// Execution this Send belonged to.
        exec_id: ExecId,
    },
}
