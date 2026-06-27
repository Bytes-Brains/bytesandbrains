//! Typed in-Node event bus. Cross-Component signaling per
//! `docs/ENGINE.md` §13.1.
//!
//! Components publish events via the bus; the engine's bus-event
//! routing pass delivers them to subscribed Components.

use std::collections::VecDeque;

use crate::ids::{CommandId, ComponentRef, OpRef, PeerId};

/// All events that may flow through the in-Node bus.
#[derive(Clone, Debug)]
pub enum NodeEvent {
    /// Framework-emitted infrastructure events.
    Infra(InfraEvent),
    /// App-Op-emitted events.
    App(AppEvent),
}

impl NodeEvent {
    /// String discriminator used by [`crate::engine::Engine`]'s
    /// bus-routing lookup against
    /// `event_subscriptions: HashMap<String, Vec<NodeSiteId>>`.
    ///
    /// `Infra` events map to the variant name (e.g.
    /// `"OpFailure"`); `App` events use the user-supplied topic so
    /// `EventSource` subscribers can target a specific channel.
    pub fn kind(&self) -> &str {
        match self {
            NodeEvent::Infra(InfraEvent::WireResponseLanded { .. }) => "WireResponseLanded",
            NodeEvent::Infra(InfraEvent::OpFailure { .. }) => "OpFailure",
            NodeEvent::Infra(InfraEvent::WireDecodeFailure { .. }) => "WireDecodeFailure",
            NodeEvent::Infra(InfraEvent::WireReceiveError { .. }) => "WireReceiveError",
            NodeEvent::Infra(InfraEvent::AppIngressError { .. }) => "AppIngressError",
            NodeEvent::Infra(InfraEvent::BusOverflow { .. }) => "BusOverflow",
            NodeEvent::Infra(InfraEvent::PeerResolveFailure { .. }) => "PeerResolveFailure",
            NodeEvent::Infra(InfraEvent::PeerSuspect { .. }) => "PeerSuspect",
            NodeEvent::Infra(InfraEvent::PeerDown { .. }) => "PeerDown",
            NodeEvent::Infra(InfraEvent::PeerLive { .. }) => "PeerLive",
            NodeEvent::Infra(InfraEvent::BackoffNoticeSent { .. }) => "BackoffNoticeSent",
            NodeEvent::Infra(InfraEvent::SilentDropActive { .. }) => "SilentDropActive",
            NodeEvent::App(AppEvent::Emit { name, .. }) => name.as_str(),
            NodeEvent::App(AppEvent::Notify { name }) => name.as_str(),
        }
    }
}

/// Framework-emitted infrastructure events per
/// `docs/ENGINE.md` §13.1.
#[derive(Clone, Debug)]
pub enum InfraEvent {
    /// A previously-suspended wire-request's response landed.
    WireResponseLanded {
        /// The `CommandId` that the wire-request was suspended on.
        cmd_id: CommandId,
    },
    /// An Op invocation failed.
    OpFailure {
        /// The Op that failed.
        op_ref: OpRef,
        /// The failure detail.
        error: OpError,
    },
    /// An inbound wire envelope's payload could not be decoded -
    /// the payload's wire-type hash didn't resolve, the bytes were
    /// malformed, or the destination address parsing failed. The
    /// engine drops the envelope's slot fill rather than writing
    /// garbage into a slot; this event lets the host observe the
    /// drop. Emitted by the inbound envelope router.
    WireDecodeFailure {
        /// Wire-type hash that the envelope advertised (0 if the
        /// failure occurred before the hash could be read).
        hash: u64,
        /// Length of the offending payload, in bytes.
        payload_size: usize,
        /// Human-readable failure detail.
        detail: String,
    },
    /// Per-fill failure on the wire-receive typed-decode path.
    /// Distinct from [`InfraEvent::WireDecodeFailure`]
    /// (envelope-level: malformed `dest_suffix` / header) -
    /// `WireReceiveError` fires after the envelope has parsed and
    /// an individual fill reached the decoder-registry lookup +
    /// typed materialisation step. Other fills in the same
    /// envelope continue to deliver (partial-delivery semantics).
    WireReceiveError {
        /// Sender of the failing envelope, if the wire layer was
        /// able to identify them.
        src_peer: Option<PeerId>,
        /// Position of the failing fill within the envelope
        /// (0-based). Other fills in the same envelope are still
        /// delivered.
        fill_index: u32,
        /// The `type_hash` the sender stamped on the fill.
        actual_hash: u64,
        /// Bytes that did not deliver. Tracked for telemetry, NOT
        /// for fallback decode - degrading to `BytesValue` is
        /// exactly the silent type-loss path this surface closes.
        payload_size: usize,
        /// Which failure mode fired.
        kind: WireReceiveErrorKind,
    },
    /// Application-side ingress failure - host pushed an
    /// `AppEvent` / `Invoke` / async completion whose payload could
    /// not enter engine state because allocation failed, the
    /// engine-wide ingress byte budget was exhausted, or a per-item
    /// cap rejected the request at the boundary. The offending
    /// bytes are dropped; the engine continues processing other
    /// ingress work. Audience: host / SDK author watching their
    /// own push errors (distinct from wire-side
    /// [`InfraEvent::WireReceiveError`]).
    AppIngressError {
        /// Which application-side entry point raised the failure
        /// and the identity it carries (module/input name for
        /// `AppEvent` / `Invoke`, `CommandId` for an async
        /// completion).
        source: AppIngressSource,
        /// Bytes the boundary was asked to admit.
        byte_count: usize,
        /// Which failure mode fired.
        kind: AppIngressErrorKind,
    },
    /// The typed bus dropped `count` oldest events to make room for
    /// newer publishes when `NodeConfig.bus_capacity` was hit.
    /// Emitted by the bus-routing pass if any drops accumulated
    /// since the last poll.
    BusOverflow {
        /// Number of events FIFO-dropped since the last poll.
        count: usize,
    },
    /// Routable telemetry mirror of
    /// [`crate::engine::EngineStep::PeerResolveFailed`].
    /// Surfaces via the bus to subscribers so dashboards can
    /// monitor peer-resolution failures alongside `PeerBlocked` /
    /// `PeerDown` / `PeerUp` from .
    PeerResolveFailure {
        /// The peer whose addresses could not be resolved. `None`
        /// when the failing Send op had no parseable `peer` input.
        peer: Option<PeerId>,
        /// The Send op that failed to resolve.
        op_ref: OpRef,
    },
    /// φ-accrual failure detector crossed the suspect threshold
    /// for the named logical site. Components (gossip overlays,
    /// peer-sampling services, deadline planners) subscribe to react.
    PeerSuspect {
        /// Suspect logical site.
        site: crate::ids::NodeSiteId,
        /// Current φ value (informational; subscribers can ignore).
        phi: f64,
    },
    /// φ-accrual failure detector crossed the hard-down threshold
    /// for the named logical site.
    PeerDown {
        /// Down logical site.
        site: crate::ids::NodeSiteId,
        /// Current φ value.
        phi: f64,
    },
    /// φ collapsed back below the suspect threshold after a
    /// `PeerSuspect` or `PeerDown` was emitted. Lets subscribers
    /// reinstate the peer.
    PeerLive {
        /// Recovered logical site.
        site: crate::ids::NodeSiteId,
    },
    /// A `BackoffNotice` envelope was emitted to `peer`. Surfaces
    /// the local overload decision on the bus so ops dashboards +
    /// Component authors who want to react to local overload can
    /// subscribe.
    BackoffNoticeSent {
        /// The sender the receiver asked to slow down.
        peer: PeerId,
        /// Why the receiver requested back-off.
        cause: crate::framework::BackoffCause,
        /// `min_backoff_ns` quoted on the notice.
        min_backoff_ns: u64,
    },
    /// `peer` crossed the K-notices-without-recovery threshold;
    /// subsequent envelopes from that peer are dropped silently at
    /// the inbound boundary until the peer recovers. Emitted once
    /// per silent-drop transition; the recovery path reuses the
    /// existing `PeerLive` event.
    SilentDropActive {
        /// The sender now in silent-drop mode.
        peer: PeerId,
    },
}

/// Sub-kind discriminator for [`InfraEvent::WireReceiveError`].
/// One top-level variant + an enum sub-kind keeps the bus-topic
/// count down and lets subscribers route on the variant while
/// matching the sub-kind out of the variant fields - the
/// `PeerSuspect`/`PeerDown`/`PeerLive` triple pattern is for
/// distinct lifecycle events; these three share lifecycle (one
/// fill, one decode step) and audience (wire-payload integrity).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireReceiveErrorKind {
    /// No decoder is registered for `actual_hash`. The sender
    /// shipped a value whose concrete type is unknown to this
    /// Node's inventory - either a version skew (sender has a
    /// carrier the receiver hasn't compiled in) or a malicious /
    /// fuzzed envelope.
    UnknownTypeHash,
    /// Destination slot carries a compile-time wire-type
    /// assertion (`expected_hash`) and the fill's `actual_hash`
    /// does not match.
    TypeMismatch {
        /// The `type_hash` the destination slot's compile-time
        /// metadata declared.
        expected_hash: u64,
    },
    /// Decoder ran and returned `Err` - the bytes were not a
    /// valid encoding of the advertised type.
    DecodeFailed {
        /// Human-readable underlying error from the registered
        /// decoder (typically a `bincode::Error::to_string`).
        error_summary: String,
    },
    /// The framework-owned scratch buffer could not be reserved
    /// before decode - heap allocation failed or a per-item cap
    /// rejected the request. Emitted by the
    /// `Engine::decode_typed_fill` boundary on `Vec::try_reserve_exact`
    /// failure or before the prost decode runs when the fill's
    /// payload length exceeds `EnvelopeCaps::max_per_fill_bytes`.
    AllocationFailed {
        /// Number of bytes the boundary tried to reserve.
        byte_count: usize,
        /// Why the reservation failed.
        reason: AllocFailReason,
    },
    /// Admitting this fill's payload would push the engine over
    /// `NodeConfig::ingress_byte_budget`. The fill is dropped; the
    /// envelope's other fills continue to deliver.
    BudgetExceeded {
        /// Bytes the fill would have added to the in-flight budget.
        byte_count: usize,
        /// Bytes still available under `NodeConfig::ingress_byte_budget`
        /// at the time of the rejection.
        budget_remaining: usize,
    },
    /// The destination slot is bound to a `Backend` role and the
    /// backend's `materialize_from_wire` impl returned `Err`. The
    /// engine drops the fill, releases the byte charge, and emits
    /// this event so operators can see which backend rejected
    /// inbound payloads. Distinct from
    /// [`WireReceiveErrorKind::DecodeFailed`] (framework-side
    /// registry decoder failure).
    BackendMaterializeFailed {
        /// `ComponentRef` of the destination slot's bound backend.
        backend_ref: ComponentRef,
        /// Short `Display` of the backend's typed error.
        backend_error_summary: String,
    },
}

/// Why a fallible-allocation boundary refused to admit bytes into
/// engine state. Carried by [`WireReceiveErrorKind::AllocationFailed`]
/// and [`AppIngressErrorKind::AllocationFailed`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocFailReason {
    /// `Vec::try_reserve_exact` (or equivalent fallible-allocator
    /// call) returned `TryReserveError`. The host's allocator has
    /// no headroom for the request.
    HeapExhausted,
    /// A caller-side per-item cap (e.g.
    /// `EnvelopeCaps::max_per_fill_bytes`,
    /// `NodeConfig::max_app_event_bytes`,
    /// `NodeConfig::max_invoke_bytes`) rejected the request before
    /// any allocation was attempted.
    PerItemCapExceeded {
        /// The cap value the boundary enforced.
        cap: usize,
    },
}

/// Application-side entry point that raised an
/// [`InfraEvent::AppIngressError`]. Carries the identity the host
/// referenced when the failure occurred so subscribers can correlate
/// the bus event with the original push call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppIngressSource {
    /// `Node::deliver_event(module, input, value_bytes)` failed at
    /// the boundary.
    AppEvent {
        /// Target module name (the host's `module` argument).
        module: String,
        /// Target input slot name on that module.
        input: String,
    },
    /// `Node::invoke(module, inputs)` failed at the boundary.
    Invoke {
        /// Target module name.
        module: String,
        /// Number of `(name, bytes)` inputs the host attempted to
        /// admit. Lets a subscriber distinguish a cap-by-count
        /// rejection (`max_invoke_inputs`) from a
        /// cap-by-bytes rejection (`max_invoke_bytes`).
        input_count: usize,
    },
    /// `CompletionSink::complete(cmd, ...)` or `::fail(cmd, ...)`
    /// failed at the boundary. The `CommandId` identifies the
    /// pending async operation the host was attempting to settle.
    Completion {
        /// The pending command the completion targeted.
        command: CommandId,
    },
}

/// Sub-kind discriminator for [`InfraEvent::AppIngressError`]. One
/// top-level variant + sub-kind keeps the bus-topic count down -
/// subscribers route on the variant name and match the sub-kind out
/// of the event's fields when they need to distinguish allocation
/// failure from budget exhaustion from a per-item cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppIngressErrorKind {
    /// Fallible reservation of the engine-side ingress buffer
    /// returned `TryReserveError` or hit a per-item cap before
    /// allocation was attempted.
    AllocationFailed {
        /// Why the reservation failed.
        reason: AllocFailReason,
    },
    /// Admitting this payload would push the engine over
    /// `NodeConfig::ingress_byte_budget`.
    BudgetExceeded {
        /// Bytes still available under `NodeConfig::ingress_byte_budget`
        /// at the time of the rejection.
        budget_remaining: usize,
    },
    /// A per-item cap (`max_app_event_bytes`, `max_invoke_inputs`,
    /// `max_invoke_bytes`, or `max_completion_result_bytes`) rejected
    /// the payload before any allocation was attempted. The host's
    /// synchronous `deliver_event` / `invoke` call returns
    /// `DeliveryError::OversizePayload` in addition to this bus
    /// emission.
    PerItemCapExceeded {
        /// The cap value the boundary enforced.
        cap: usize,
    },
}

/// App-Op-emitted events surfaced as `EngineStep::AppEvent`.
///
/// Application code constructs these via [`AppEvent::emit`] /
/// [`AppEvent::notify`] which reject framework-reserved topic
/// prefixes (`bb.`, `ai.bytesandbrains.`) so the framework's own
/// `InfraEvent` topic namespace can never be impersonated by user
/// publish calls (closes the trust-boundary requirement
/// in §Trust boundary). The struct variants stay constructible
/// directly for framework-internal forwarding paths that are not
/// crossing the application boundary.
#[derive(Clone, Debug)]
pub enum AppEvent {
    /// `AppEmit(name, value_bytes)` - full app event carrying bytes.
    Emit {
        /// Event topic.
        name: String,
        /// Encoded value payload.
        value_bytes: Vec<u8>,
    },
    /// `AppNotify(name)` - marker-only notification.
    Notify {
        /// Event topic.
        name: String,
    },
}

/// Reserved topic prefix #1 — framework-emitted infra topics.
const RESERVED_PREFIX_BB: &str = "bb.";

/// Reserved topic prefix #2 — framework-namespaced metadata keys.
const RESERVED_PREFIX_FRAMEWORK: &str = "ai.bytesandbrains.";

/// Reasons an [`AppEvent::emit`] / [`AppEvent::notify`] construction
/// can fail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppEventError {
    /// Topic begins with one of the framework-reserved prefixes
    /// (`bb.` or `ai.bytesandbrains.`) — only the framework's own
    /// `InfraEvent` may publish under those prefixes.
    ReservedPrefix {
        /// The offending topic the application tried to publish.
        topic: String,
        /// Which prefix it collided with.
        prefix: &'static str,
    },
}

impl std::fmt::Display for AppEventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReservedPrefix { topic, prefix } => write!(
                f,
                "AppEvent topic `{topic}` collides with the framework-reserved prefix `{prefix}`",
            ),
        }
    }
}

impl std::error::Error for AppEventError {}

impl AppEvent {
    /// Construct an [`AppEvent::Emit`], rejecting framework-reserved
    /// topic prefixes. The trust boundary `AppEvent::new` referenced
    /// in `docs-plan/CORRECTED_ARCHITECTURE.md` §Trust boundary.
    pub fn emit(name: impl Into<String>, value_bytes: Vec<u8>) -> Result<Self, AppEventError> {
        let name = name.into();
        check_reserved(&name)?;
        Ok(Self::Emit { name, value_bytes })
    }

    /// Construct an [`AppEvent::Notify`], rejecting framework-
    /// reserved topic prefixes. See [`Self::emit`].
    pub fn notify(name: impl Into<String>) -> Result<Self, AppEventError> {
        let name = name.into();
        check_reserved(&name)?;
        Ok(Self::Notify { name })
    }
}

fn check_reserved(topic: &str) -> Result<(), AppEventError> {
    for prefix in [RESERVED_PREFIX_BB, RESERVED_PREFIX_FRAMEWORK] {
        if topic.starts_with(prefix) {
            return Err(AppEventError::ReservedPrefix {
                topic: topic.to_string(),
                prefix,
            });
        }
    }
    Ok(())
}

/// Op invocation failure kind. a stable categorical
/// label consumers match on for retry / report / drop policy
/// decisions without parsing the freeform `OpError::detail`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpErrorKind {
    /// Input slot value didn't match the expected concrete type
    /// (typed downcast failed in a dispatch arm).
    TypeMismatch,
    /// A required slot was absent at dispatch time.
    MissingSlot,
    /// The dispatched op has no registered handler.
    NotRegistered,
    /// The op handler ran but its work failed (numeric, IO,
    /// inventory, etc.). The detail string carries specifics.
    ExecutionFailed,
    /// An off-thread completion handle returned an error (the
    /// user's Contract method's `Error` type produced this via
    /// `ContractResponse::Now(Err)` or `CompletionHandle::fail`).
    RemoteFailed,
    /// The op's deadline elapsed before completion landed.
    Timeout,
    /// Adversarial / malformed input from a peer.
    BadInput,
    /// Peer-health / backoff gate held the op (transient retry-
    /// eligible failure).
    Cooldown,
    /// Catch-all for failures that don't fit a more specific kind.
    /// Default for call sites that only carry a `detail` string.
    Other,
}

impl Default for OpErrorKind {
    fn default() -> Self {
        Self::Other
    }
}

/// Op invocation failure detail. Surfaced by the engine when
/// `dispatch_atomic` returns `Err` or the dispatch table lookup
/// misses. Three-field shape: `kind` is a stable categorical label,
/// `reason` is a `&'static str` (e.g. `"blocklisted"`, `"cooldown"`)
/// callers can match on, `detail` is a free-form human-readable
/// description.
#[derive(Clone, Debug, Default)]
pub struct OpError {
    /// Categorical kind. Default `Other`.
    pub kind: OpErrorKind,
    /// Stable diagnostic label (e.g. `"blocklisted"`, `"cooldown"`,
    /// `"nak"`) consumers match on. Default `""`.
    pub reason: &'static str,
    /// Human-readable failure detail.
    pub detail: String,
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.reason.is_empty() {
            write!(f, "op error[{:?}]: {}", self.kind, self.detail)
        } else {
            write!(
                f,
                "op error[{:?} reason={}]: {}",
                self.kind, self.reason, self.detail,
            )
        }
    }
}

impl std::error::Error for OpError {}

/// Typed in-Node event bus. Carries published events from one
/// poll cycle to the next; the engine's bus-routing pass drains
/// the queue and routes to subscribed `NodeSiteId`s.
///
/// Bounded by `NodeConfig.bus_capacity` (default 1024). When a
/// publish would exceed the cap, the oldest event is FIFO-dropped
/// and a counter increments. The routing pass reads the counter
/// and emits `InfraEvent::BusOverflow { count }` so the host sees
/// the loss.
#[derive(Default)]
pub struct TypedBus {
    queue: VecDeque<NodeEvent>,
    cap: Option<usize>,
    dropped_since_last_drain: usize,
}

impl TypedBus {
    /// Construct a fresh empty bus with no cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a fresh empty bus with the given FIFO-drop cap.
    pub fn with_cap(cap: Option<usize>) -> Self {
        Self {
            queue: VecDeque::new(),
            cap,
            dropped_since_last_drain: 0,
        }
    }

    /// Set the FIFO-drop cap. `None` removes the cap.
    pub fn set_cap(&mut self, cap: Option<usize>) {
        self.cap = cap;
    }

    /// Publish an event. The engine's bus-routing pass delivers
    /// published events to subscribed Components in the next poll
    /// cycle.
    pub fn publish(&mut self, event: NodeEvent) {
        if let Some(cap) = self.cap {
            while self.queue.len() >= cap {
                self.queue.pop_front();
                self.dropped_since_last_drain += 1;
            }
        }
        self.queue.push_back(event);
    }

    /// Drain all queued events. Called by the engine's bus-routing pass.
    pub fn drain(&mut self) -> Vec<NodeEvent> {
        self.queue.drain(..).collect()
    }

    /// Read + reset the count of FIFO-dropped events since the last
    /// call. Returns 0 when no drops occurred.
    pub fn take_dropped_count(&mut self) -> usize {
        std::mem::take(&mut self.dropped_since_last_drain)
    }

    /// `true` when the bus has no queued events.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Number of queued events.
    pub fn len(&self) -> usize {
        self.queue.len()
    }
}

