//! Runtime resource handle threaded into every `dispatch_atomic`
//! call.
//!
//! The engine constructs a `RuntimeResourceRef` by split-borrowing
//! the framework-primitive bundle + the bus before each
//! `dispatch_atomic` call. Each field is a distinct `&mut`, so the
//! borrow checker enforces field-level exclusivity — an Op may
//! touch any subset.
//!
//! Async-completing impls call `ctx.complete_command(cmd_id,
//! results)`; the engine drains `pending_completions` after the
//! hook returns and routes them through `handle_completion`.

use std::sync::Arc;

use crate::bus::{AllocFailReason, AppIngressErrorKind, AppIngressSource, NodeEvent, TypedBus};
use crate::completion::{CompletionHandle, CompletionSink};
use crate::framework::rtt_tracker::{chain_id_from_targets, ChainContext};
use crate::framework::{
    rtt_tracker::RttTracker, AddressBook, BackoffTable, BackpressureTracker, EventSource,
    HoldTable, InboundDedup, OutboundQueue, PeerGate, PeerGovernor, RecordBuffer, RequestTracker,
    RngU64Source, Scheduler, SerializeQueue,
};
use crate::ids::{CommandId, OpRef};
use crate::ingress::{IngressEvent, IngressQueue, COMPLETION_DETAIL_CAP};
use crate::slot_value::SlotValue;

impl CompletionSink for IngressQueue {
    fn complete(&self, cmd_id: CommandId, result_bytes: &[u8]) {
        // Per Principle 1a: `result_bytes` is borrowed from the
        // caller's stack/transport buffer. Cap-check, fallibly
        // reserve framework-owned storage, then copy. The owned
        // `Vec<u8>` rides into the engine via the `Completion`
        // variant; cap / alloc failures publish an `AppIngressError`
        // sibling and drop the result (the parked op times out
        // naturally — same surface as a missing completion).
        let byte_count = result_bytes.len();
        let cap = self.completion_result_cap();
        if byte_count > cap {
            let _ = self.push(IngressEvent::AppIngressError {
                source: AppIngressSource::Completion { command: cmd_id },
                byte_count,
                kind: AppIngressErrorKind::PerItemCapExceeded { cap },
            });
            return;
        }
        let mut owned: Vec<u8> = Vec::new();
        if crate::fallible::try_reserve_exact(&mut owned, byte_count).is_err() {
            let _ = self.push(IngressEvent::AppIngressError {
                source: AppIngressSource::Completion { command: cmd_id },
                byte_count,
                kind: AppIngressErrorKind::AllocationFailed {
                    reason: AllocFailReason::HeapExhausted,
                },
            });
            return;
        }
        owned.extend_from_slice(result_bytes);
        let _ = self.push(IngressEvent::Completion {
            cmd_id,
            results: vec![owned],
        });
    }

    fn fail(&self, cmd_id: CommandId, detail: &str) {
        // Push the typed `CompletionFailed` variant so the parked
        // op fails through `handle_completion_failed` → typed
        // `OpFailed`, not via success-bytes masquerading as a
        // completion.
        //
        // The detail string is truncated rather than rejected at
        // `COMPLETION_DETAIL_CAP`: a `Display`-rendered failure must
        // always land so the component sees a real failure instead
        // of a missing completion masquerading as a timeout.
        let truncated = if detail.len() > COMPLETION_DETAIL_CAP {
            let mut end = COMPLETION_DETAIL_CAP;
            while end > 0 && !detail.is_char_boundary(end) {
                end -= 1;
            }
            &detail[..end]
        } else {
            detail
        };
        let owned: String = truncated.to_string();
        let _ = self.push(IngressEvent::CompletionFailed {
            cmd_id,
            detail: owned,
        });
    }
}

/// Per-peer state borrowed mutably during dispatch.
pub struct PeerCtx<'a> {
    /// Per-peer concurrency limiter.
    pub gate: &'a mut PeerGate,
    /// Per-peer exponential backoff state.
    pub backoff: &'a mut BackoffTable,
    /// Peer policy + health source-of-truth.
    pub governor: &'a mut PeerGovernor,
    /// `PeerId → (Vec<Address>, ref_count)` registry.
    pub addresses: &'a mut AddressBook,
    /// Receiver-side back-pressure tracker. RX gates + ingress
    /// detection sites consult this to record overload and decide
    /// between emitting a typed `BackoffNotice` envelope or silently
    /// dropping.
    pub backpressure: &'a mut BackpressureTracker,
}

/// Network/transport state borrowed mutably during dispatch.
pub struct NetCtx<'a> {
    /// FIFO of wire envelopes ready to ship on the next outbound drain.
    pub outbound: &'a mut OutboundQueue,
    /// Per-NodeSiteId adaptive RTT tracker.
    pub rtt: &'a mut RttTracker,
    /// In-flight wire-request → CommandId map.
    pub requests: &'a mut RequestTracker,
    /// Sliding-window seen-message tracker.
    pub dedup: &'a mut InboundDedup,
    /// Peer-resolution failures captured during this poll cycle.
    pub pending_peer_resolve_failures: &'a mut Vec<(Option<crate::ids::PeerId>, crate::ids::OpRef)>,
}

/// Time / scheduling state borrowed mutably during dispatch.
pub struct TimeCtx<'a> {
    /// Sorted timer heap.
    pub scheduler: &'a mut Scheduler,
}

/// Syscall-side state (storage, RNG, latches, app-event drain).
pub struct SyscallCtx<'a> {
    /// Named-FIFO map for `Serialize.Enqueue` / `Dequeue`.
    pub serialize_queue: &'a mut SerializeQueue,
    /// Named-slot value buffer for `Hold.Stash` / `Flush`.
    pub hold_table: &'a mut HoldTable,
    /// Per-name bounded ring buffer for `Record`.
    pub record_buffer: &'a mut RecordBuffer,
    /// Registered `EventKind → ComponentTag` subscriptions.
    pub event_source: &'a mut EventSource,
    /// Per-Node counters bumped by `IncrMetric`.
    pub counters: &'a mut std::collections::HashMap<String, u64>,
    /// Per-group first-arrival latch for the `Any` syscall.
    pub any_fired_groups: &'a mut std::collections::HashSet<String>,
    /// Per-`(OpRef, ExecId)` latch for the `DeadlineMatch` syscall.
    pub deadline_match_fired: &'a mut std::collections::HashSet<(u64, u64)>,
    /// `u64` RNG source for the `RngU64` syscall.
    pub rng: &'a mut dyn RngU64Source,
    /// App events pending emission on the next poll's outbound drain.
    pub pending_app_events: &'a mut Vec<crate::bus::AppEvent>,
}

/// Inbound-envelope context captured at delivery time and threaded
/// into every op dispatched as part of the cascade.
#[derive(Clone, Copy, Debug, Default)]
pub struct InboundCtx {
    /// The `PeerId` of the inbound envelope's transport-reported
    /// source. `None` outside the inbound delivery path.
    pub src_peer: Option<crate::ids::PeerId>,
    /// The inbound envelope's `wire_req_id` correlation token.
    pub wire_req_id: Option<u64>,
    /// Engine-clock timestamp the envelope arrived at.
    pub arrival_ns: Option<u64>,
    /// Remaining deadline budget propagated by the sender.
    pub remaining_deadline_ns: Option<u64>,
}

/// State scoped to the currently-dispatching op (NodeProto-level
/// metadata, the op's identity, completion drain, command-id mint).
pub struct CurrentCallCtx<'a> {
    /// The `OpRef` of the Op currently being dispatched.
    pub op_ref: OpRef,
    /// The `ExecId` this dispatch belongs to. Syscalls that latch
    /// per-execution (`DeadlineMatch`, `Any`) key on
    /// `(op_ref, exec_id)` so a fresh execution starts unlatched.
    pub exec_id: crate::ids::ExecId,
    /// The Node's own `PeerId`.
    pub self_peer: crate::ids::PeerId,
    /// Attributes of the NodeProto being dispatched.
    pub node_attributes: &'a [bb_ir::proto::onnx::AttributeProto],
    /// Metadata_props of the NodeProto being dispatched.
    pub node_metadata: &'a [bb_ir::proto::onnx::StringStringEntryProto],
    /// Inbound-envelope context (all four `inbound_*` fields).
    pub inbound: InboundCtx,
    /// Completions captured during this dispatch.
    pub pending_completions: Vec<PendingCompletion>,
    /// Engine's monotonic CommandId source.
    pub next_command_id: &'a mut u64,
}

/// Engine-resource handle threaded into every `dispatch_atomic`
/// call. Per `docs/ENGINE.md` §10. Fields are grouped by concern -
/// `peers`/`net`/`time`/`syscall` carry the framework primitive
/// references; `current` carries per-op state; `bus`/`ingress`
/// stay top-level; `components` exposes the cross-component
/// read-only surface.
pub struct RuntimeResourceRef<'a> {
    /// Per-peer state (gate, backoff, governor, addresses).
    pub peers: PeerCtx<'a>,
    /// Network/transport state (outbound queue, RTT, requests, dedup).
    pub net: NetCtx<'a>,
    /// Time / scheduling state.
    pub time: TimeCtx<'a>,
    /// Syscall-side state (storage, RNG, latches, app-event drain).
    pub syscall: SyscallCtx<'a>,
    /// The in-Node typed event bus.
    pub bus: &'a mut TypedBus,
    /// Shared handle to the Node's ingress queue.
    pub ingress: Arc<IngressQueue>,
    /// Read-only view onto sibling components registered on the Node.
    pub components: ComponentsView<'a>,
    /// State scoped to the currently-dispatching op.
    pub current: CurrentCallCtx<'a>,
}

/// Read-only view onto sibling components registered on the Node.
/// Constructed by the engine at dispatch time from
/// `&engine.components` + `&engine.slots`. The Vec is indexed by
/// `ComponentRef.as_u32() as usize`; the slot the currently-
/// dispatching component lives in is `None` for the duration of
/// the dispatch (take-and-restore in `invoke_atomic`), so callers
/// can't accidentally re-enter themselves.
#[derive(Default)]
pub struct ComponentsView<'a> {
    /// All registered components indexed by `ComponentRef.as_u32()`,
    /// borrowed from `engine.components`. `None` outside engine
    /// context (test setups bypassing the registry).
    pub instances: Option<&'a [Option<Box<dyn crate::component::ErasedComponent>>]>,
    /// Author-chosen-slot-name → `ComponentRef` map, borrowed from
    /// `engine.slots`. THE canonical dependency-resolution surface.
    /// `None` outside engine context.
    pub slots: Option<&'a std::collections::HashMap<String, crate::ids::ComponentRef>>,
}

impl ComponentsView<'_> {
    /// Look up the component bound at `slot_name`. The generic
    /// dependency-resolution surface — Components reach their
    /// declared dependencies through this accessor. Returns `None`
    /// when no slot of that name is bound or when the view has no
    /// engine context.
    pub fn for_slot(&self, slot_name: &str) -> Option<&dyn crate::component::ErasedComponent> {
        let slots = self.slots?;
        let instances = self.instances?;
        let cref = slots.get(slot_name)?;
        let idx = cref.as_u32() as usize;
        instances.get(idx)?.as_deref()
    }

    /// Look up the component bound at `slot_name` AND downcast it
    /// to `&T`. The typed counterpart to [`Self::for_slot`].
    /// Returns `None` when the slot is unbound OR when the bound
    /// concrete is not a `T`.
    ///
    /// In production this is reached through
    /// [`crate::runtime::RuntimeResourceRef::dependency`], which
    /// wraps the lookup in [`DependencyError`] variants for typed
    /// error reporting.
    pub fn for_slot_as<T: 'static>(&self, slot_name: &str) -> Option<&T> {
        let erased = self.for_slot(slot_name)?;
        let any: &dyn std::any::Any = erased;
        any.downcast_ref::<T>()
    }
}

/// Errors surfaced by [`RuntimeResourceRef::dependency`].
///
/// In production the dependency is verified at compile time by
/// `resolve_component_dependencies`, so a runtime miss represents
/// a framework invariant breach (e.g. someone bypassed
/// `Node::install` to register a custom slot mapping). The error
/// is exposed instead of panicking so test fixtures + introspection
/// tooling can probe the surface without aborting the process.
#[derive(Debug)]
pub enum DependencyError {
    /// No component is bound at the requested slot.
    NotBound {
        /// The slot name the caller requested.
        slot: String,
    },
    /// A component IS bound but downcasting to the requested
    /// type failed - the slot holds a different concrete than
    /// the caller expected.
    TypeMismatch {
        /// The slot name the caller requested.
        slot: String,
        /// `std::any::type_name` of the expected type.
        expected: &'static str,
    },
}

impl std::fmt::Display for DependencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotBound { slot } => write!(f, "no component bound at slot `{slot}`"),
            Self::TypeMismatch { slot, expected } => {
                write!(f, "component at slot `{slot}` is not a `{expected}`",)
            }
        }
    }
}

impl std::error::Error for DependencyError {}

impl RuntimeResourceRef<'_> {
    /// Ordered local-address bag for this Node. Reads the
    /// AddressBook entry keyed by `self.current.self_peer`; returns
    /// an empty slice when no local addresses are registered.
    /// Wire ops + identity-bearing protocol replies (Announce,
    /// Handshake) stamp this onto their outbound envelopes so
    /// receivers can dial back on every reachable interface.
    pub fn local_addresses(&self) -> &[crate::framework::Address] {
        self.peers
            .addresses
            .lookup(self.current.self_peer)
            .unwrap_or(&[])
    }

    /// Typed accessor for an author-declared dependency. Resolves
    /// `slot_name` against the engine's generic slot registry and
    /// downcasts the bound `ErasedComponent` to `&T`.
    ///
    /// In production the resolution is guaranteed to succeed -
    /// `resolve_component_dependencies` verifies at compile time
    /// that every `#[depends(<role> = "<slot>")]` declaration
    /// matches a bound concrete of the right role. A miss here is
    /// either a test fixture bypassing the compiler pipeline or a
    /// framework invariant breach.
    ///
    /// ```ignore
    /// // Inside a Component's dispatch_atomic / Contract impl:
    /// let backend = ctx
    ///     .dependency::<MyCpuBackend>("compute")
    ///     .expect("compiler verified");
    /// let result = backend.matmul(&lhs, &rhs)?;
    /// ```
    pub fn dependency<T: 'static>(&self, slot_name: &str) -> Result<&T, DependencyError> {
        if self.components.for_slot(slot_name).is_none() {
            return Err(DependencyError::NotBound {
                slot: slot_name.to_string(),
            });
        }
        self.components
            .for_slot_as::<T>(slot_name)
            .ok_or_else(|| DependencyError::TypeMismatch {
                slot: slot_name.to_string(),
                expected: std::any::type_name::<T>(),
            })
    }

    /// -ii - convenience helper
    /// that walks the hierarchical fallback in
    /// [`crate::framework::rtt_tracker::RttTracker`] to pick the
    /// effective deadline for a wire round-trip.
    ///
    /// `chain_id` + `hop_index` come from the compiler-stamped
    /// `chain_targets` / `chain_depth` metadata on the current
    /// NodeProto (read from
    /// [`Self::current_node_metadata`] via
    /// [`Self::read_chain_context`]); pass `None` for control-plane
    /// sends that have no chain context.
    pub fn estimate_wire_budget_ns(
        &self,
        target: crate::ids::NodeSiteId,
        chain: Option<crate::framework::rtt_tracker::ChainContext>,
        static_default_ns: u64,
    ) -> u64 {
        self.net
            .rtt
            .estimate_budget_ns(target, chain, static_default_ns)
    }

    /// Read the compiler-stamped `chain_targets` + chain hop
    /// (encoded in `chain_depth` metadata) off the current NodeProto
    /// and convert them into an [`crate::framework::rtt_tracker::ChainContext`].
    /// Returns `None` when no chain metadata is present (the Send is
    /// a fire-and-forget escape hatch or a control-plane round-trip).
    pub fn read_chain_context(&self) -> Option<crate::framework::rtt_tracker::ChainContext> {
        let mut chain_targets: Option<&str> = None;
        let mut hop_index: u8 = 0;
        for prop in self.current.node_metadata {
            match prop.key.as_str() {
                "ai.bytesandbrains.wire.chain_targets" => {
                    chain_targets = Some(prop.value.as_str());
                }
                "ai.bytesandbrains.wire.chain_hop_index" => {
                    if let Ok(h) = prop.value.parse::<u8>() {
                        hop_index = h;
                    }
                }
                _ => {}
            }
        }
        chain_targets.map(|targets| ChainContext {
            chain_id: chain_id_from_targets(targets),
            hop_index,
        })
    }

    /// Record a wire round-trip sample into the RTT tracker. Called
    /// on response landing (after the matching `WireResponseLanded`
    /// event surfaces the elapsed time) so all the hierarchical-
    /// fallback EMA tiers stay current.
    pub fn observe_wire_round_trip(
        &mut self,
        target: crate::ids::NodeSiteId,
        chain: Option<crate::framework::rtt_tracker::ChainContext>,
        elapsed_ns: u64,
        now_ns: u64,
    ) {
        self.net
            .rtt
            .observe_round_trip(target, chain, elapsed_ns, now_ns);
    }

    /// Mint a fresh `CommandId` via the engine's monotonic counter.
    /// Used by async-suspending syscalls (`After`, `Sleep`,
    /// `BootstrapDispatch`).
    pub fn allocate_command_id(&mut self) -> CommandId {
        let id = *self.current.next_command_id;
        *self.current.next_command_id = self.current.next_command_id.saturating_add(1);
        CommandId::from(id)
    }

    /// Record a CommandId completion for the engine to drain after
    /// `dispatch_atomic` returns. Used by `ProtocolRuntime` impls +
    /// any role impl that returned `DispatchResult::Async`. Invoking
    /// this in the same call lets the consumer fire in the same poll
    /// cycle via the engine's catch-up drain.
    pub fn complete_command(
        &mut self,
        cmd_id: CommandId,
        results: Vec<(String, Box<dyn SlotValue>)>,
    ) {
        self.current
            .pending_completions
            .push(PendingCompletion { cmd_id, results });
    }

    /// Convenience for publishing events to the in-Node bus.
    pub fn publish_bus(&mut self, event: NodeEvent) {
        self.bus.publish(event);
    }

    /// Open a completion handle for an async Contract method. The
    /// caller receives a fresh [`CommandId`] + a shared
    /// [`CompletionSink`] backed by the Node's ingress queue. The
    /// user's Contract method holds the handle past the dispatch
    /// return and calls [`CompletionHandle::complete`] when work
    /// finishes. The dispatch arm returns
    /// `DispatchResult::Async(handle.cmd_id())` so the engine parks
    /// the op until the completion lands.
    pub fn open_completion<R, E>(&mut self) -> CompletionHandle<R, E>
    where
        R: serde::Serialize,
        E: std::fmt::Display,
    {
        let cmd_id = self.allocate_command_id();
        let sink: Arc<dyn CompletionSink> = self.ingress.clone();
        CompletionHandle::new(cmd_id, sink)
    }
}

/// Captured async-completion payload. The engine drains these from
/// the post-dispatch `RuntimeResourceRef` and routes them through
/// `Engine::handle_completion`.
pub struct PendingCompletion {
    /// The `CommandId` being fulfilled.
    pub cmd_id: CommandId,
    /// `(name, value)` pairs to write to the suspended Op's output
    /// sites.
    pub results: Vec<(String, Box<dyn SlotValue>)>,
}

/// Component-scheduled timer kind. Used by `ProtocolRuntime::on_timer`:
/// protocol impls schedule timers via `ctx.time.scheduler` and receive
/// the matured timer back via this newtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComponentTimerKind(pub u32);

impl ComponentTimerKind {
    /// Construct from an explicit kind id.
    pub const fn new(kind: u32) -> Self {
        Self(kind)
    }

    /// Inner value accessor.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

