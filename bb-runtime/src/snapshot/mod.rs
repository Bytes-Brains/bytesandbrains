//! `NodeSnapshot`
//!
//! Captures the snapshottable state of a running Node so a fresh
//! Node can be reconstructed via `Node::restore(snap)` and resume
//! execution.
//!
//! **scope** - ships the Rust struct surface. The 11 proto
//! mirror messages outlined in
//! `docs/internal/IMPLEMENTATION_PLAN.md`  are deferred to the
//! cross-stage audit; on-the-wire snapshot transfer is not a
//! acceptance-gate requirement and the in-memory Rust
//! surface is sufficient for the round-trip semantics. The fields
//! line up 1:1 with the proto-spec field shapes so the future
//! addition is mechanical.

use serde::{Deserialize, Serialize};

use crate::concrete::ComponentPackage;
use crate::node::NodeConfig;

pub mod transient;
pub use transient::TransientSnapshot;

/// Top-level snapshot per ENGINE.md §15.1.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeSnapshot {
    /// Monotonically bumped on every `Node::restore`. Used by hosts
    /// to detect "is this the same Node since N seconds ago?".
    pub incarnation: u64,

    /// The NodeConfig captured at snapshot time. NodeConfig must
    /// match exactly at restore time.
    pub config: NodeConfigSnapshot,

    /// Installed graphs (post-analysis FunctionProto bytes).
    pub graphs: Vec<NamedGraphSnapshot>,

    /// Per-component serialized state.
    pub components: Vec<NamedComponentSnapshot>,

    /// In-flight transient state (frontier, slot_table, …).
    pub transient: TransientSnapshot,
}

/// Stable serializable view of `NodeConfig` (the runtime struct has
/// fields that aren't 1:1 serde-friendly).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NodeConfigSnapshot {
    /// Canonical multihash bytes for `NodeConfig.peer_id`.
    /// Empty when the snapshot was taken without a configured peer.
    #[serde(default)]
    pub peer_id: Vec<u8>,
    /// `NodeConfig.cycle_op_budget`.
    pub cycle_op_budget: Option<usize>,
    /// `NodeConfig.max_pending_async`.
    #[serde(default)]
    pub max_pending_async: Option<usize>,
    /// `NodeConfig.max_outbound_queue`.
    #[serde(default)]
    pub max_outbound_queue: Option<usize>,
    /// `NodeConfig.bus_capacity`.
    pub bus_capacity: usize,
}

impl From<&NodeConfig> for NodeConfigSnapshot {
    fn from(c: &NodeConfig) -> Self {
        Self {
            peer_id: c.peer_id.to_bytes(),
            cycle_op_budget: c.cycle_op_budget,
            max_pending_async: c.max_pending_async,
            max_outbound_queue: c.max_outbound_queue,
            bus_capacity: c.bus_capacity,
        }
    }
}

/// One installed graph's snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedGraphSnapshot {
    /// Graph name (key in `engine.graphs`).
    pub name: String,
    /// Prost-serialized `FunctionProto` bytes.
    pub function_proto_bytes: Vec<u8>,
}

/// One component's snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamedComponentSnapshot {
    /// `ConcreteComponent::TYPE_NAME`.
    pub type_name: String,
    /// Per-Node `instance_id` used in NodeProto metadata.
    pub instance_id: u32,
    /// `ConcreteComponent::PACKAGE`.
    pub package: ComponentPackage,
    /// Captured state bytes from `ConcreteComponent::serialize`.
    pub state_bytes: Vec<u8>,
}

impl NodeSnapshot {
    /// Bincode-encode the snapshot for on-disk persistence or
    /// inter-process transfer. ships this as the only
    /// transport; may add the prost-bytes equivalent.
    pub fn encode(&self) -> Vec<u8> {
        bincode::serialize(self).expect("NodeSnapshot serde is infallible for valid types")
    }

    /// Decode a snapshot from bincode bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

