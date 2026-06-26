//! Construction-time configuration for a `Node`.
//!
//! The local peer identity is required at construction; every
//! other field carries a production-conservative default
//! (cycle-op budget, in-flight async cap, outbound-queue cap,
//! bus capacity).

use crate::framework::{
    DEFAULT_HIGH_WATER_PCT, DEFAULT_K_BEFORE_SILENT, DEFAULT_MIN_NOTICE_INTERVAL_NS,
};
use crate::ids::PeerId;

use crate::envelope::EnvelopeCaps;

/// Default per-cycle Op budget. `Some(1000)` yields voluntarily
/// after 1000 op-invocations per poll; an emit of
/// `EngineStep::CycleBudgetExceeded` tells the host the engine
/// paused so other work can run. `None` disables the budget guard.
pub const DEFAULT_CYCLE_OP_BUDGET: Option<usize> = Some(1000);

/// Default cap on the number of in-flight `DispatchResult::Async`
/// commands. `Some(10_000)` rejects further async dispatches with
/// `OpError("pending-async limit exceeded")` once the cap is hit -
/// protects against a runaway component returning `Async(_)` in a
/// tight loop. `None` disables the cap.
pub const DEFAULT_MAX_PENDING_ASYNC: Option<usize> = Some(10_000);

/// Default cap on the outbound envelope queue depth. `Some(10_000)`
/// drops the oldest envelope when the cap is hit (FIFO drop) and
/// emits `EngineStep::OutboundDropped`. `None` disables the cap.
pub const DEFAULT_MAX_OUTBOUND_QUEUE: Option<usize> = Some(10_000);

/// Default bus capacity.
pub const DEFAULT_BUS_CAPACITY: usize = 1024;

/// Default per-target-boundary-hop
/// budget for sizing async deadlines on wire.Send NodeProtos.
/// 100 ms in nanoseconds. The compiler's `analyze_wire_dependencies`
/// stamps a static `chain_depth` count on each Send; the engine
/// multiplies by this budget at deadline-stamp time so a Send whose
/// downstream chain reaches `N` target boundaries gets `N *
/// per_hop_budget_ns` to respond.
pub const DEFAULT_PER_HOP_BUDGET_NS: u64 = 100_000_000;

/// Default cap on total in-flight ingress bytes the engine may hold
/// across the ingress queue, slot table, and pending async
/// completion buffers at any instant. 256 MiB. Wire and application
/// boundaries `try_charge` against this cap before admitting a
/// payload; overflow emits the appropriate `BudgetExceeded`
/// `InfraEvent` and drops the offending bytes.
pub const DEFAULT_INGRESS_BYTE_BUDGET: usize = 256 * 1024 * 1024;

/// Default per-`AppEvent` payload cap consulted by
/// `Node::deliver_event`. 1 MiB. Oversize payloads return
/// `DeliveryError::OversizePayload` synchronously AND emit
/// `AppIngressError { kind: PerItemCapExceeded }` on the bus.
pub const DEFAULT_MAX_APP_EVENT_BYTES: usize = 1024 * 1024;

/// Default per-`Invoke` input-count cap. 100 inputs. Caller-side
/// guard against pathological `invoke()` calls; the cap rejects
/// before any per-input allocation runs.
pub const DEFAULT_MAX_INVOKE_INPUTS: usize = 100;

/// Default per-`Invoke` cumulative payload cap. 10 MiB. Sum of
/// every `(name, bytes)` entry's payload length.
pub const DEFAULT_MAX_INVOKE_BYTES: usize = 10 * 1024 * 1024;

/// Default per-`CompletionHandle` result-payload cap. 4 MiB. Larger
/// completions emit `AppIngressError { kind: PerItemCapExceeded }`
/// and the component sees an async-op timeout in place of the
/// dropped completion.
pub const DEFAULT_MAX_COMPLETION_RESULT_BYTES: usize = 4 * 1024 * 1024;

/// Edge preset for [`NodeConfig::ingress_byte_budget`]. 8 MiB.
pub const EDGE_INGRESS_BYTE_BUDGET: usize = 8 * 1024 * 1024;

/// Edge preset for [`NodeConfig::max_app_event_bytes`]. 64 KiB.
pub const EDGE_MAX_APP_EVENT_BYTES: usize = 64 * 1024;

/// Edge preset for [`NodeConfig::max_invoke_inputs`]. 16 inputs.
pub const EDGE_MAX_INVOKE_INPUTS: usize = 16;

/// Edge preset for [`NodeConfig::max_invoke_bytes`]. 256 KiB.
pub const EDGE_MAX_INVOKE_BYTES: usize = 256 * 1024;

/// Edge preset for [`NodeConfig::max_completion_result_bytes`].
/// 64 KiB.
pub const EDGE_MAX_COMPLETION_RESULT_BYTES: usize = 64 * 1024;

/// Construction-time configuration for a `Node`.
#[derive(Clone, Debug)]
pub struct NodeConfig {
    /// Local peer identity. Required at construction - the Node
    /// holds its own identity from the moment it exists.
    pub peer_id: PeerId,

    /// Soft per-poll-cycle budget - the engine voluntarily yields
    /// after N op-invocations to honor caller-side backpressure.
    /// `None` disables the budget guard.
    pub cycle_op_budget: Option<usize>,

    /// Cap on the number of in-flight `DispatchResult::Async`
    /// commands. When the cap is hit, further async dispatches
    /// fail synchronously with `OpError("pending-async limit
    /// exceeded")`. `None` disables the cap.
    pub max_pending_async: Option<usize>,

    /// Cap on the outbound envelope queue depth. When the cap is
    /// hit, the oldest envelope is dropped (FIFO) and a count
    /// surfaces via `EngineStep::OutboundDropped`. `None` disables
    /// the cap.
    pub max_outbound_queue: Option<usize>,

    /// Bus capacity. Overflow drops the oldest event + bumps a
    /// counter.
    pub bus_capacity: usize,

    /// Per-target-boundary budget in nanoseconds. The Engine's
    /// deadline-stamping path multiplies this by the static
    /// `chain_depth` metadata on each outbound `wire.Send` to size
    /// the call's deadline. Sized to represent the worst-case
    /// round-trip cost of one network boundary; chains crossing
    /// multiple boundaries pay the multiplier.
    pub per_hop_budget_ns: u64,

    /// inbound envelope decode caps the
    /// [`crate::envelope::EnvelopeCodec::decode_capped`] consults
    /// on every inbound buffer. Production defaults match the
    /// design's "16 MiB / 256 / 4 MiB / 4 KiB" recommendation;
    /// edge deployments use [`EnvelopeCaps::edge()`] for tighter
    /// bounds (256 KiB / 16 / 64 KiB / 512). Custom caps via
    /// [`Self::with_envelope_caps`].
    pub envelope_caps: EnvelopeCaps,

    /// Receiver-side back-pressure high-water mark, as a percentage
    /// of the ingress queue capacity. Once ingress depth reaches
    /// this fraction, the framework emits a `BackoffNotice` envelope
    /// to each contributing sender per the backpressure protocol
    /// design at
    /// `docs/internal/superpowers/specs/2026-06-23-backpressure-runtime.md`
    /// §6. Default [`DEFAULT_HIGH_WATER_PCT`] (75).
    pub backpressure_high_water_pct: u8,

    /// K = notices-without-recovery before the receiver transitions
    /// the sender to silent-drop mode. Default
    /// [`DEFAULT_K_BEFORE_SILENT`] (3) - matches the
    /// `RttEma::is_warm` threshold at
    /// `bb-runtime/src/framework/rtt_tracker.rs:126-128`.
    pub backpressure_k_before_silent: u32,

    /// Minimum interval enforced between successive `BackoffNotice`
    /// emissions to the same sender. Acts as a hard lower bound on
    /// the duplicate-suppression window so a flood of inbound
    /// envelopes from one peer produces at most one notice per
    /// interval. Default [`DEFAULT_MIN_NOTICE_INTERVAL_NS`]
    /// (1 second).
    pub backpressure_min_notice_interval_ns: u64,

    /// Total in-flight ingress bytes the engine may hold across the
    /// ingress queue + slot table + pending async completion buffers
    /// at any instant. Wire and application ingress boundaries
    /// `try_charge` against this cap before installing a payload; on
    /// overflow the offending bytes are dropped and the appropriate
    /// `BudgetExceeded` `InfraEvent` is emitted. Default
    /// [`DEFAULT_INGRESS_BYTE_BUDGET`] (256 MiB); edge preset
    /// [`EDGE_INGRESS_BYTE_BUDGET`] (8 MiB).
    pub ingress_byte_budget: usize,

    /// Per-`AppEvent` payload cap (host-driven push). Default
    /// [`DEFAULT_MAX_APP_EVENT_BYTES`] (1 MiB); edge preset
    /// [`EDGE_MAX_APP_EVENT_BYTES`] (64 KiB). On overflow,
    /// `Node::deliver_event` returns `DeliveryError::OversizePayload`
    /// AND emits `AppIngressError { kind: PerItemCapExceeded }`.
    pub max_app_event_bytes: usize,

    /// Per-`Invoke` input-count cap. Default
    /// [`DEFAULT_MAX_INVOKE_INPUTS`] (100); edge preset
    /// [`EDGE_MAX_INVOKE_INPUTS`] (16). Caps the number of
    /// `(name, bytes)` pairs an `invoke()` call may carry before any
    /// per-input allocation runs.
    pub max_invoke_inputs: usize,

    /// Per-`Invoke` cumulative payload cap. Default
    /// [`DEFAULT_MAX_INVOKE_BYTES`] (10 MiB); edge preset
    /// [`EDGE_MAX_INVOKE_BYTES`] (256 KiB). Sum of every input's
    /// payload length crossing the boundary.
    pub max_invoke_bytes: usize,

    /// Per-`CompletionHandle` result-payload cap. Default
    /// [`DEFAULT_MAX_COMPLETION_RESULT_BYTES`] (4 MiB); edge preset
    /// [`EDGE_MAX_COMPLETION_RESULT_BYTES`] (64 KiB). The detail
    /// string on `fail()` is independently capped at 4 KiB (truncated
    /// rather than rejected). Component sees an async-op timeout in
    /// place of a dropped completion.
    pub max_completion_result_bytes: usize,
}

impl NodeConfig {
    /// Construct with the given local peer identity and
    /// production-conservative defaults for every other field.
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            cycle_op_budget: DEFAULT_CYCLE_OP_BUDGET,
            max_pending_async: DEFAULT_MAX_PENDING_ASYNC,
            max_outbound_queue: DEFAULT_MAX_OUTBOUND_QUEUE,
            bus_capacity: DEFAULT_BUS_CAPACITY,
            per_hop_budget_ns: DEFAULT_PER_HOP_BUDGET_NS,
            envelope_caps: EnvelopeCaps::default(),
            backpressure_high_water_pct: DEFAULT_HIGH_WATER_PCT,
            backpressure_k_before_silent: DEFAULT_K_BEFORE_SILENT,
            backpressure_min_notice_interval_ns: DEFAULT_MIN_NOTICE_INTERVAL_NS,
            ingress_byte_budget: DEFAULT_INGRESS_BYTE_BUDGET,
            max_app_event_bytes: DEFAULT_MAX_APP_EVENT_BYTES,
            max_invoke_inputs: DEFAULT_MAX_INVOKE_INPUTS,
            max_invoke_bytes: DEFAULT_MAX_INVOKE_BYTES,
            max_completion_result_bytes: DEFAULT_MAX_COMPLETION_RESULT_BYTES,
        }
    }

    /// Convenience constructor with the tighter edge-device presets
    /// applied to every cap: envelope caps (256 KiB / 16 / 64 KiB /
    /// 512), ingress budget (8 MiB), per-`AppEvent` (64 KiB),
    /// per-`Invoke` (16 inputs / 256 KiB), and per-`Completion`
    /// (64 KiB).
    pub fn new_edge(peer_id: PeerId) -> Self {
        Self {
            envelope_caps: EnvelopeCaps::edge(),
            ingress_byte_budget: EDGE_INGRESS_BYTE_BUDGET,
            max_app_event_bytes: EDGE_MAX_APP_EVENT_BYTES,
            max_invoke_inputs: EDGE_MAX_INVOKE_INPUTS,
            max_invoke_bytes: EDGE_MAX_INVOKE_BYTES,
            max_completion_result_bytes: EDGE_MAX_COMPLETION_RESULT_BYTES,
            ..Self::new(peer_id)
        }
    }

    /// Override the inbound envelope decode caps.
    pub fn with_envelope_caps(mut self, caps: EnvelopeCaps) -> Self {
        self.envelope_caps = caps;
        self
    }

    /// Cap how many op-invocations a single `Node::poll()` may issue
    /// before voluntarily yielding back to the host.
    pub fn with_cycle_op_budget(mut self, budget: usize) -> Self {
        self.cycle_op_budget = Some(budget);
        self
    }

    /// Disable the per-cycle op budget - let `poll()` drain the
    /// frontier to quiescence in a single cycle.
    pub fn without_cycle_op_budget(mut self) -> Self {
        self.cycle_op_budget = None;
        self
    }

    /// Cap how many in-flight `DispatchResult::Async` commands the
    /// engine will hold at once.
    pub fn with_max_pending_async(mut self, cap: usize) -> Self {
        self.max_pending_async = Some(cap);
        self
    }

    /// Disable the pending-async cap - the engine accepts any
    /// number of in-flight commands.
    pub fn without_max_pending_async(mut self) -> Self {
        self.max_pending_async = None;
        self
    }

    /// Cap the outbound envelope queue depth.
    pub fn with_max_outbound_queue(mut self, cap: usize) -> Self {
        self.max_outbound_queue = Some(cap);
        self
    }

    /// Disable the outbound queue cap.
    pub fn without_max_outbound_queue(mut self) -> Self {
        self.max_outbound_queue = None;
        self
    }

    /// Override the bus capacity (default
    /// [`DEFAULT_BUS_CAPACITY`]).
    pub fn with_bus_capacity(mut self, capacity: usize) -> Self {
        self.bus_capacity = capacity;
        self
    }

    /// Override the per-target-boundary budget in nanoseconds
    /// (default [`DEFAULT_PER_HOP_BUDGET_NS`]). Deployments with
    /// known-slow links can dial this up; tightly-coupled
    /// deployments can dial it down.
    pub fn with_per_hop_budget_ns(mut self, budget_ns: u64) -> Self {
        self.per_hop_budget_ns = budget_ns;
        self
    }

    /// Override the receiver-side back-pressure high-water mark
    /// percentage (default [`DEFAULT_HIGH_WATER_PCT`]). Clamped to
    /// `1..=100` by the `BackpressureTracker` constructor.
    pub fn with_backpressure_high_water_pct(mut self, pct: u8) -> Self {
        self.backpressure_high_water_pct = pct;
        self
    }

    /// Override K (notices-without-recovery before silent-drop;
    /// default [`DEFAULT_K_BEFORE_SILENT`]). Clamped to at least 1
    /// by the `BackpressureTracker` constructor.
    pub fn with_backpressure_k_before_silent(mut self, k: u32) -> Self {
        self.backpressure_k_before_silent = k;
        self
    }

    /// Override the minimum interval between successive notices to
    /// the same peer (default [`DEFAULT_MIN_NOTICE_INTERVAL_NS`]).
    /// Clamped to at least 1 by the `BackpressureTracker`
    /// constructor.
    pub fn with_backpressure_min_notice_interval_ns(mut self, interval_ns: u64) -> Self {
        self.backpressure_min_notice_interval_ns = interval_ns;
        self
    }

    /// Override the cumulative ingress byte budget.
    pub fn with_ingress_byte_budget(mut self, bytes: usize) -> Self {
        self.ingress_byte_budget = bytes;
        self
    }

    /// Override the per-`AppEvent` payload cap.
    pub fn with_max_app_event_bytes(mut self, bytes: usize) -> Self {
        self.max_app_event_bytes = bytes;
        self
    }

    /// Override the per-`Invoke` input-count cap.
    pub fn with_max_invoke_inputs(mut self, count: usize) -> Self {
        self.max_invoke_inputs = count;
        self
    }

    /// Override the per-`Invoke` cumulative payload cap.
    pub fn with_max_invoke_bytes(mut self, bytes: usize) -> Self {
        self.max_invoke_bytes = bytes;
        self
    }

    /// Override the per-`CompletionHandle` result cap.
    pub fn with_max_completion_result_bytes(mut self, bytes: usize) -> Self {
        self.max_completion_result_bytes = bytes;
        self
    }
}

