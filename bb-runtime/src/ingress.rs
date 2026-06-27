//! Lock-free MPMC ingress queue.
//!
//! External tasks (transport, host invocations, off-thread
//! completions) push `IngressEvent`s onto the queue; the engine
//! drains them on its next poll. Lock-free via `concurrent-queue`
//! v2; the engine sleeps on an `AtomicWaker` until a producer
//! wakes it.
//!
//! `Arc<IngressQueue>` is shared between the engine and any number
//! of external producer tasks running on different threads.
//!
//! Per ENGINE.md §2.2 + §16: the queue is BOUNDED with default
//! capacity `bus_capacity * 4` (= 4096 when bus_capacity uses the
//! spec default of 1024). On overflow, `push` returns
//! `Err(IngressEvent)` so the transport adapter can choose to
//! retry, drop with a metric, or escalate as back-pressure to its
//! upstream. The `dropped_overflow` counter tracks total overflow
//! drops surfaced via `dropped_overflow()`.

use std::ops::Deref;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::Waker;

use atomic_waker::AtomicWaker;
use concurrent_queue::{ConcurrentQueue, PushError};

use crate::bus::{AppIngressErrorKind, AppIngressSource};
use crate::ids::CommandId;

/// Per-`fail()` detail-string hard cap. Truncated rather than rejected
/// so the host's `Display`-rendered failure message always lands,
/// even when oversized.
pub const COMPLETION_DETAIL_CAP: usize = 4 * 1024;

/// Default bus capacity per ENGINE.md §16; the ingress queue size
/// defaults to 4× this value.
const DEFAULT_BUS_CAPACITY: usize = 1024;

/// Default ingress capacity: `bus_capacity * 4` per ENGINE.md §2.2.
pub const DEFAULT_INGRESS_CAPACITY: usize = DEFAULT_BUS_CAPACITY * 4;

/// External-event variants pushed to the ingress queue per
/// `docs/ENGINE.md` §6 entry points.
#[derive(Debug)]
pub enum IngressEvent {
    /// Inbound wire envelope from the transport layer, attributed
    /// to a source peer. The engine calls
    /// `PeerGovernor::check_inbound(src_peer)` on ingress; blocked or
    /// non-allowlisted peers are dropped before any slot is written,
    /// surfacing as `EngineStep::PeerBlocked`.
    EnvelopeFrom {
        /// Peer the envelope arrived from.
        src_peer: crate::ids::PeerId,
        /// The envelope payload.
        envelope: crate::envelope::WireEnvelope,
        /// Transport-observed source address, when the adapter can
        /// supply it (e.g. NAT-translated remote endpoint, dialer's
        /// observed multiaddr). The receiver merges this into its
        /// AddressBook entry for `src_peer` so reflexive-address
        /// discovery composes with the sender-claimed
        /// `envelope.src_peer_addresses` list. `None` means the
        /// transport didn't surface an observed address.
        src_observed_address: Option<crate::framework::Address>,
    },

    /// Host pushed an app event onto a Module input.
    AppEvent {
        /// Target Module's name.
        module_name: String,
        /// Module input port name.
        input_name: String,
        /// Encoded value payload.
        value_bytes: Vec<u8>,
    },

    /// External timer maturity signal (used when an off-thread
    /// scheduler drives the engine).
    TimerMatured {
        /// Maturity timestamp (nanoseconds).
        at_ns: u64,
    },

    /// Explicit Module invocation from host.
    Invoke {
        /// Target Module's name.
        module_name: String,
        /// `(input_name, value_bytes)` pairs.
        inputs: Vec<(String, Vec<u8>)>,
        /// `ExecId` allocated by `Node::invoke` so the host can
        /// correlate `EngineStep::AppEvent` / `OpCompleted` /
        /// `AsyncSuspended` outputs back to the originating call.
        exec_id: crate::ids::ExecId,
    },

    /// External (off-thread) async completion landing back at the
    /// engine.
    Completion {
        /// The `CommandId` being fulfilled.
        cmd_id: CommandId,
        /// Encoded output payloads.
        results: Vec<Vec<u8>>,
    },

    /// Async completion FAILURE landing back at the engine.
    /// Distinct from `Completion`: `CompletionSink::fail` mints
    /// this variant directly so `handle_completion_failed` can
    /// route to the typed `OpFailed` surface — the host sees a
    /// real error, not a success-bytes masquerade.
    CompletionFailed {
        /// The `CommandId` whose await failed.
        cmd_id: CommandId,
        /// Human-readable failure detail; the runtime wraps it
        /// into `bus::OpError` on the engine side.
        detail: String,
    },

    /// Transport-side send-outcome failure surfaced
    /// by an adapter (libp2p, sim, etc.) when the network NAKed an
    /// outbound envelope or its delivery deadline elapsed without
    /// an ACK. Distinct from `CompletionFailed` (which covers
    /// off-thread compute completion); this variant covers
    /// transport-layer delivery failure.
    SendFailed {
        /// The wire request id of the failed outbound envelope.
        wire_req_id: u64,
        /// The destination peer that NAKed or timed out (raw
        /// multihash bytes so the engine can reconstruct
        /// `PeerId::from_bytes(&peer)`).
        peer: Vec<u8>,
        /// Stable diagnostic label (e.g. `"nak"`, `"timeout"`,
        /// `"network_unreachable"`). Adapters pick from a fixed
        /// vocabulary so consumers can match on the label.
        reason: &'static str,
    },

    /// Off-thread application-ingress failure (currently only
    /// `CompletionSink::complete` exceeding the per-completion result
    /// cap). The engine drains this variant and publishes a matching
    /// `InfraEvent::AppIngressError` on the bus so subscribers see the
    /// rejection. The synchronous `Node::deliver_event` / `Node::invoke`
    /// path publishes directly with `&mut bus` access; this variant is
    /// the cross-thread bridge for sinks that don't hold a bus
    /// reference. The Component observes an async-op timeout in place
    /// of the dropped completion.
    AppIngressError {
        /// Which application-side entry point raised the failure.
        source: AppIngressSource,
        /// Bytes the boundary was asked to admit.
        byte_count: usize,
        /// Which failure mode fired.
        kind: AppIngressErrorKind,
    },
}

impl IngressEvent {
    /// Construct an `EnvelopeFrom` for the in-process router common
    /// case where the transport carries no NAT and the observed
    /// address is the sender's PeerId-tagged multiaddr. Test buses
    /// and the in-process router call this so observed-address
    /// propagation exercises the same merge path as a real
    /// transport's reflexive surface.
    pub fn from_in_process(
        src_peer: crate::ids::PeerId,
        envelope: crate::envelope::WireEnvelope,
    ) -> Self {
        Self::EnvelopeFrom {
            src_peer,
            envelope,
            src_observed_address: Some(crate::framework::Address::empty().p2p(src_peer)),
        }
    }
}

/// Lock-free MPMC ingress queue + waker. Multiple external
/// producers may `push` concurrently; the engine's single consumer
/// drains via `drain_all` on each poll cycle.
pub struct IngressQueue {
    queue: ConcurrentQueue<IngressEvent>,
    waker: AtomicWaker,
    dropped_overflow: AtomicU64,
    /// Per-`CompletionSink::complete` result cap sourced from
    /// `NodeConfig::max_completion_result_bytes` via
    /// `apply_config_caps`. Defaults to `usize::MAX` (no cap) so
    /// constructions outside the `Node::new` → `apply_config_caps`
    /// path (test fixtures, snapshot reseed) behave like the
    /// pre-cap world.
    completion_result_cap: AtomicUsize,
}

impl IngressQueue {
    /// Construct a fresh ingress queue with the default capacity
    /// ([`DEFAULT_INGRESS_CAPACITY`]).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_INGRESS_CAPACITY)
    }

    /// Construct a fresh ingress queue with the supplied bounded
    /// capacity. Per ENGINE.md §2.2 the canonical sizing is
    /// `bus_capacity * 4`; pass the host's chosen bus_capacity
    /// multiplied by 4 to match.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            queue: ConcurrentQueue::bounded(capacity),
            waker: AtomicWaker::new(),
            dropped_overflow: AtomicU64::new(0),
            completion_result_cap: AtomicUsize::new(usize::MAX),
        }
    }

    /// Install the per-`CompletionSink::complete` result cap.
    /// `Engine::apply_config_caps` calls this from
    /// `NodeConfig::max_completion_result_bytes` so off-thread
    /// completions see the configured cap without the sink needing a
    /// reference to `NodeConfig`.
    pub(crate) fn set_completion_result_cap(&self, cap: usize) {
        self.completion_result_cap.store(cap, Ordering::Relaxed);
    }

    /// Per-`complete()` result-byte cap. Defaults to `usize::MAX`
    /// when not configured; `apply_config_caps` reseeds it from
    /// `NodeConfig::max_completion_result_bytes`.
    pub fn completion_result_cap(&self) -> usize {
        self.completion_result_cap.load(Ordering::Relaxed)
    }

    /// Push an event. On success returns `Ok(())` and wakes the
    /// engine if it's sleeping. On a full queue the event comes
    /// back in `Err(_)` and the `dropped_overflow` counter is
    /// incremented; transport adapters decide whether to retry,
    /// drop with a metric, or escalate as back-pressure. The
    /// `IngressEvent` Err variant is large (carries a
    /// `WireEnvelope` with multihash PeerIds); transport adapters
    /// already box or re-queue, so the cost lives at the boundary.
    #[allow(clippy::result_large_err)]
    pub fn push(&self, event: IngressEvent) -> Result<(), IngressEvent> {
        match self.queue.push(event) {
            Ok(()) => {
                self.waker.wake();
                Ok(())
            }
            Err(PushError::Full(ev)) => {
                self.dropped_overflow.fetch_add(1, Ordering::Relaxed);
                Err(ev)
            }
            Err(PushError::Closed(ev)) => Err(ev),
        }
    }

    /// Drain all available events. Called by the engine on each
    /// poll cycle's ingress drain.
    ///
    /// Pre-reserves capacity for the bounded queue's full length so the
    /// drain Vec grows once at construction, not in `O(log n)`
    /// reallocations as events pop. The queue itself caps inflight at
    /// `self.capacity()`; the drain is bounded by the same cap, so the
    /// upfront reservation is the exact-fit answer.
    pub fn drain_all(&self) -> Vec<IngressEvent> {
        let mut out = Vec::with_capacity(self.queue.capacity().unwrap_or(0));
        while let Ok(event) = self.queue.pop() {
            out.push(event);
        }
        out
    }

    /// Register the engine's waker so future pushes can wake it.
    pub fn register_waker(&self, waker: &Waker) {
        self.waker.register(waker);
    }

    /// `true` when the queue currently holds no events.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Approximate current queue depth. The underlying
    /// `concurrent-queue` returns an approximate `len` for the
    /// MPMC case; introspection callers should treat
    /// this as a snapshot, not a real-time invariant.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Bounded capacity supplied at construction. `concurrent-queue`
    /// guarantees `Some(cap)` for bounded queues, so unwrapping is
    /// safe for the framework's path that never builds an unbounded
    /// ingress queue.
    pub fn capacity(&self) -> usize {
        self.queue.capacity().unwrap_or(usize::MAX)
    }

    /// Total events dropped due to the queue being full since this
    /// queue was constructed. Telemetry hook for transport adapters
    /// + Node introspection.
    pub fn dropped_overflow(&self) -> u64 {
        self.dropped_overflow.load(Ordering::Relaxed)
    }
}

impl Default for IngressQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Cheap-clone handle to the shared [`IngressQueue`] surfaced by
/// [`crate::node::Node::ingress_handle`].
/// Behaves identically to `Arc<IngressQueue>` via `Deref` so
/// callers can `.push(IngressEvent::...)` directly. The newtype
/// wrapper isolates the public API from the underlying smart-pointer
/// choice.
#[derive(Clone)]
pub struct IngressQueueRef(Arc<IngressQueue>);

impl IngressQueueRef {
    /// Wrap an existing `Arc<IngressQueue>`. Used by `Node` after
    /// borrowing from the inner engine.
    pub fn new(queue: Arc<IngressQueue>) -> Self {
        Self(queue)
    }
}

impl IngressQueueRef {
    /// Borrow the underlying `Arc<IngressQueue>`. Used by transport
    /// adapters and in-process test buses that need to share the
    /// queue across threads — both pin a per-Node queue handle and
    /// push events as the transport receives them.
    pub fn arc(&self) -> &Arc<IngressQueue> {
        &self.0
    }
}

impl Deref for IngressQueueRef {
    type Target = IngressQueue;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for IngressQueueRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IngressQueueRef")
            .field("len", &self.len())
            .field("dropped_overflow", &self.dropped_overflow())
            .finish()
    }
}

