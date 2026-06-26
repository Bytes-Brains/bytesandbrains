//! `TransientSnapshot` — the runtime ephemeral state surfaced for
//! `Node::snapshot` / `Node::restore`.
//!
//! Stable framework state (counters, lifecycle phases, address book,
//! peer governor, backoff table) round-trips today via the populated
//! `framework` + `bus` fields. The remaining fields (`frontier`,
//! `slot_table`, `pending_async`, `execution_state`, `ingress`,
//! `wire_states`, `pending_completions`) exist on the struct so the
//! shape matches the future in-flight execution snapshot but are not
//! yet populated by `Node::snapshot`; restored Nodes start from a
//! fresh frontier.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Runtime ephemeral state per ENGINE.md §15.1.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TransientSnapshot {
    /// In-cycle DAG-walking queue.
    pub frontier: Vec<(u64, u64)>,
    /// `(NodeSiteId, ExecId) → Option<serialized slot bytes>`.
    /// `None` means "slot allocated but empty".
    pub slot_table: HashMap<(u64, u64), Option<Vec<u8>>>,
    /// Suspended Ops awaiting CommandId completion.
    pub pending_async: HashMap<u64, PendingAsyncSnapshot>,
    /// Per-execution liveness state.
    pub execution_state: HashMap<u64, ExecutionStateSnapshot>,
    /// Framework-primitive state (counters, backoff_table,
    /// inbound_dedup, etc.) snapshotted at quiesce. Per
    /// ENGINE.md §15.1 line 1402.
    pub framework: FrameworkSnapshot,
    /// Typed-bus state - subscription table + any queued events
    /// that survive the cycle boundary. Per ENGINE.md §15.1 line
    /// 1403.
    pub bus: TypedBusSnapshot,
    /// In-flight ingress events.
    pub ingress: Vec<IngressEventSnapshot>,
    /// Per-component wire-state. Currently empty; macro
    /// populates as components grow per-wire state.
    pub wire_states: HashMap<u32, Vec<u8>>,
    /// Mid-cycle pending completions surfaced by `ProtocolRuntime`
    /// hooks via `ctx.complete_command(...)`. Phase 5 drains these
    /// post-dispatch per ENGINE.md §15.1 line 1406.
    pub pending_completions: Vec<PendingCompletionSnapshot>,
}

/// Serializable view of `FrameworkComponents` per ENGINE.md §16.
/// Captures counters, queued lifecycle phases, the multiaddr-keyed
/// peer registries per `docs/ADDRESSING.md`, plus the
/// `PeerGovernor` + `BackoffTable` policy/health state introduced
/// in  - so a restored Node remembers
/// blocklisted peers, allowlist policy, and in-flight backoff
/// cooldowns across restarts (no thundering herd on cold start).
///
/// Other framework primitives (peer_gate inflight counts) stay
/// transient - they're meaningful only within a single poll cycle.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrameworkSnapshot {
    /// Counters from `Engine.counters` keyed by metric name.
    pub counters: HashMap<String, u64>,
    /// Lifecycle phases queued for next `poll()` per
    /// `Engine.fired_phases`.
    pub fired_phases: Vec<String>,
    /// `AddressBook` entries - ordered address list + reference
    /// count per peer (see `AddressBookEntrySnapshot`).
    pub address_book: Vec<AddressBookEntrySnapshot>,
    /// `PeerGovernor` policy + health.
    #[serde(default)]
    pub peer_governor: PeerGovernorSnapshot,
    /// Per-peer `BackoffTable` state.
    #[serde(default)]
    pub backoff_table: Vec<BackoffEntry>,
    /// Pending outbound envelopes that didn't make it to a Phase 8
    /// drain before snapshotting. Each entry carries the
    /// `redelivered` flag so the host's transport adapter can
    /// decide whether to retry or drop after restore.
    #[serde(default)]
    pub pending_outbound: Vec<PendingOutboundEntry>,
    /// Canonical multihash bytes for the Node's `PeerId`. Restore
    /// reconstructs the PeerId via `PeerId::from_bytes(&peer_id_bytes)`,
    /// round-tripping every multihash code (identity, sha2-256,
    /// blake2b, ...) without information loss.
    #[serde(default)]
    pub peer_id_bytes: Vec<u8>,
    /// Engine ID counter persistence. The previous
    /// snapshot dropped `next_command_id` / `next_exec_id`, so a
    /// restored Node would mint ID 0 again — colliding with any
    /// in-flight command/exec the pre-snapshot Node had issued.
    #[serde(default)]
    pub next_command_id: u64,
    /// Same for ExecIds.
    #[serde(default)]
    pub next_exec_id: u64,
    /// Snapshot schema version (incarnation distinct).
    /// Bumped when the FrameworkSnapshot shape changes in a way
    /// older code cannot soundly restore from.
    #[serde(default = "default_spec_version")]
    pub spec_version: u32,
}

fn default_spec_version() -> u32 {
    CURRENT_SNAPSHOT_SPEC_VERSION
}

/// Current snapshot spec version this build can soundly restore.
/// Bumped when the `FrameworkSnapshot` shape changes in a way
/// older code cannot replay (e.g. field-encoding change, removed
/// invariant). Restore rejects snapshots stamped with any other
/// version.
pub const CURRENT_SNAPSHOT_SPEC_VERSION: u32 = 1;

/// One peer's `AddressBook` entry: ordered list of `Address`
/// byte vectors + reference count. Preserves the multi-address +
/// ref-counted shape across snapshot/restore.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AddressBookEntrySnapshot {
    /// Canonical multihash bytes — `PeerId::to_bytes()`.
    pub peer_id: Vec<u8>,
    /// Ordered address list - each entry is `Address::to_bytes()`.
    pub addresses: Vec<Vec<u8>>,
    /// Reference count owned by overlay protocols / transport /
    /// the application. Preserved across restore so peers stay
    /// alive at their proper grip count.
    pub ref_count: u64,
}

/// Serializable view of `PeerGovernor`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PeerGovernorSnapshot {
    /// Blocklist entries, each `PeerId::to_bytes()`.
    pub blocklist: Vec<Vec<u8>>,
    /// `None` ⇒ open policy. `Some(vec)` ⇒ only the listed peers
    /// (`PeerId::to_bytes()`) may communicate.
    pub allowlist: Option<Vec<Vec<u8>>>,
    /// `(PeerId::to_bytes(), consecutive_failures, last_event_ns,
    /// down)` per peer.
    pub health: Vec<(Vec<u8>, u32, u64, bool)>,
    /// Failure threshold (consecutive failures to mark a peer
    /// down).
    pub failure_threshold: u32,
}

/// Serializable view of one peer's `BackoffState`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackoffEntry {
    /// Canonical multihash bytes — `PeerId::to_bytes()`.
    pub peer: Vec<u8>,
    /// Consecutive failures.
    pub attempts: u32,
    /// `now_ns` at most recent failure.
    pub last_attempt_ns: u64,
    /// Earliest `now_ns` at which a retry is permitted.
    pub next_retry_ns: u64,
}

/// One outbound envelope that hadn't been shipped yet when the
/// snapshot was taken. The `redelivered` flag tells the transport
/// adapter "I've seen this before, decide whether to ship again."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingOutboundEntry {
    /// Prost-serialized `WireEnvelope` bytes.
    pub envelope_bytes: Vec<u8>,
    /// `true` once a previous snapshot/restore cycle already
    /// surfaced this envelope.
    pub redelivered: bool,
}

/// Serializable view of `TypedBus` per ENGINE.md §16. Captures the
/// subscription table - `(event_kind → Vec<NodeSiteId.0>)` matching
/// the multiaddr-routed delivery model in `docs/ADDRESSING.md`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TypedBusSnapshot {
    /// Event-kind → subscriber `NodeSiteId.0` map. Mirrors
    /// `Engine.event_subscriptions` keyed by string discriminator.
    pub event_subscriptions: HashMap<String, Vec<u64>>,
}

/// Serializable view of `PendingCompletion` per ENGINE.md §10.2.
/// The opaque `results` payload is serialized via the same wire
/// path as ordinary slot values - SlotValue implementors carry
/// proto-mirroring; non-tensor `WireValue`s round-trip as raw bytes
/// here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingCompletionSnapshot {
    /// `CommandId.0` being fulfilled.
    pub cmd_id: u64,
    /// `(slot-name, serialized payload)` pairs to write to the
    /// suspended Op's output sites.
    pub results: Vec<(String, Vec<u8>)>,
}

/// Serializable view of `PendingAsync`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingAsyncSnapshot {
    /// `OpRef.0` of the suspended Op.
    pub op_ref: u64,
    /// `ExecId.0` of the suspended execution.
    pub exec_id: u64,
    /// Captured output sites as `NodeSiteId.0` values.
    pub output_sites: Vec<u64>,
    /// Engine-side deadline (`scheduler.now_ns()` clock) past
    /// which the suspension expires. `None` ⇒ no engine deadline.
    /// .
    #[serde(default)]
    pub deadline_ns: Option<u64>,
}

/// Serializable view of `ExecutionState`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExecutionStateSnapshot {
    /// Number of outputs written so far.
    pub outputs_written: u32,
}

/// Serializable view of `IngressEvent`. Only the variants that can
/// realistically survive a snapshot boundary are recorded; Waker /
/// Control variants are dropped on snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IngressEventSnapshot {
    /// Inbound envelope (encoded as raw bytes - the routing table
    /// re-routes on restore).
    Envelope(Vec<u8>),
    /// App-event delivery.
    AppEvent {
        /// Module name.
        module_name: String,
        /// Input port name.
        input_name: String,
        /// Encoded payload bytes.
        value_bytes: Vec<u8>,
    },
    /// Module invocation.
    Invoke {
        /// Module name.
        module_name: String,
        /// Input port + value-bytes pairs.
        inputs: Vec<(String, Vec<u8>)>,
    },
    /// Timer maturity signal.
    TimerMatured {
        /// Maturity timestamp (nanoseconds).
        at_ns: u64,
    },
    /// Async completion landing back at the engine.
    Completion {
        /// `CommandId.0` being fulfilled.
        cmd_id: u64,
        /// Encoded result payloads.
        results: Vec<Vec<u8>>,
    },
}

