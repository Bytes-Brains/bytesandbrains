//! Canonical Send / Recv NodeProto shape — the contract every
//! wire-IR-touching pass + runtime gate agrees on.
//!
//! Wire ops carry a specific NodeProto shape. Centralizing the
//! contract here keeps the DSL emitters, compiler passes, and
//! runtime gates from drifting on string literals or attribute
//! layouts.
//!
//! This module is the **single declarative description** of that
//! shape. The DSL emits to it; the compiler passes mutate within
//! it; the runtime consumes it. [`crate::verify::wire_shape`]
//! checks a `ModelProto` against the contract.
//!
//! ## The shape
//!
//! ### Send
//!
//! ```text
//! NodeProto {
//!     op_type: "Send",
//!     domain: "ai.bytesandbrains.wire",
//!     input: [payload_0, payload_1, ..., payload_{N-1}, peer],
//!     output: [handle],
//!     attribute: [
//!         (name: "peer", type: BYTES, t: <PeerId.to_bytes() multihash>),
//!         (name: "dest_suffix.{name}", type: BYTES, t: <multiaddr-bytes>),
//!         (name: "deadline_ns", type: INT, i: <i64-ns>),  // optional
//!     ],
//!     metadata_props: [
//!         ("ai.bytesandbrains.wire.wire_id", "<u64>"),
//!         ("ai.bytesandbrains.wire_transport", "data" | "trigger_only"),
//!         ("ai.bytesandbrains.batch_group_id", "<u32>"),
//!         ("ai.bytesandbrains.dest_site_name.{name}", "<recv-site-name>"),
//!     ],
//! }
//! ```
//!
//! ### Recv
//!
//! ```text
//! NodeProto {
//!     op_type: "Recv",
//!     domain: "ai.bytesandbrains.wire",
//!     input: [],
//!     output: [received_0, received_1, ..., received_{N-1}, sender],
//!     metadata_props: [
//!         ("ai.bytesandbrains.wire.wire_id", "<u64 matching paired Send>"),
//!         ("ai.bytesandbrains.wire_transport", "data" | "trigger_only"),
//!     ],
//! }
//! ```
//!
//! `wire_id` pairs the Send/Recv halves across the cut.
//! `wire_transport` tells the runtime whether each fill carries a
//! payload (`data`) or is firing-signal-only (`trigger_only`).
//!
//! ## Key invariants the contract pins
//!
//! - **`ATTR_PEER` is bytes, not i64.** The peer attribute on a
//!   Send carries the PeerId's canonical multihash byte form
//!   (`PeerId::to_bytes()`), NOT a `u64`-collapsed identity hash.
//!   The runtime gates parse via
//!   `PeerId::from_bytes(&attr.t)`.
//!
//! - **`wire_id` is the pairing token.** The DSL `Graph::wire`
//!   mints a monotonic u64 and stamps it on BOTH halves; the
//!   compiler's `discover_wire_edges` pair Send/Recv by it.
//!
//! - **`wire_transport` lives on the NodeProto itself.**
//!   `analyze_wire_edges` stamps it onto
//!   `partition.functions[0].node[i].metadata_props` in place, so
//!   downstream passes and the runtime read a single source of
//!   truth.
//!
//! - **`SlotFill.type_hash` is populated from the sender side's
//!   `T::HASH`.** Receivers dispatch wire bytes via
//!   `if envelope.fill.type_hash == T::HASH { T::deserialize(&fill.payload) }`.

/// Wire-op domain. All Send / Recv NodeProtos live under here.
pub const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";

/// op_type for a Send node.
pub const OP_SEND: &str = "Send";

/// op_type for a Recv node.
pub const OP_RECV: &str = "Recv";

/// Attribute key on a Send NodeProto carrying the destination peer
/// as **multihash bytes** (`PeerId::to_bytes()`), i.e. the
/// `attribute.t` (bytes) field, not `attribute.i` (i64).
///
/// Aliased from [`crate::syscall_ids::ATTR_PEER`] so consumers can
/// reach for `bb_ir::wire_shape::ATTR_PEER` or `bb_ir::keys::ATTR_PEER`
/// — they're the same key, kept in one place.
pub use crate::syscall_ids::ATTR_PEER;

/// Attribute key prefix for per-fill multiaddr destination
/// suffixes. Full key is `format!("{DEST_SUFFIX_ATTR_PREFIX}{slot_name}")`.
pub use crate::keys::DEST_SUFFIX_ATTR_PREFIX;

/// Attribute key for the optional static deadline (in nanoseconds
/// since the reference clock epoch) stamped by
/// `insert_async_deadlines`.
pub use crate::syscall_ids::ATTR_DEADLINE_NS;

/// `metadata_props` key carrying the wire-pairing token.
pub use crate::keys::WIRE_ID_KEY;

/// `metadata_props` key carrying the data-vs-trigger-only
/// classification.
pub use crate::keys::WIRE_TRANSPORT_KEY;

use crate::proto::onnx::{attribute_proto, AttributeProto, StringStringEntryProto};

/// Value of [`WIRE_TRANSPORT_KEY`] for full-payload edges.
pub use crate::keys::WIRE_TRANSPORT_DATA;

/// Value of [`WIRE_TRANSPORT_KEY`] for trigger-only edges.
pub use crate::keys::WIRE_TRANSPORT_TRIGGER_ONLY;

/// `metadata_props` key prefix for per-fill recv-site names. Full
/// key is `format!("{DEST_SITE_NAME_PREFIX}{slot_name}")`.
pub use crate::keys::DEST_SITE_NAME_PREFIX;

/// Return `true` if the NodeProto is a `wire.Send`.
pub fn is_send(node: &crate::proto::onnx::NodeProto) -> bool {
    node.op_type == OP_SEND && node.domain == WIRE_DOMAIN
}

/// Return `true` if the NodeProto is a `wire.Recv`.
pub fn is_recv(node: &crate::proto::onnx::NodeProto) -> bool {
    node.op_type == OP_RECV && node.domain == WIRE_DOMAIN
}

/// Read the wire_id metadata stamp from a Send or Recv node.
/// Returns `None` if the key is absent or non-numeric.
pub fn read_wire_id(node: &crate::proto::onnx::NodeProto) -> Option<u64> {
    node.metadata_props
        .iter()
        .find(|p| p.key == WIRE_ID_KEY)
        .and_then(|p| p.value.parse::<u64>().ok())
}

/// Read the destination peer's multihash bytes from a Send / gate
/// NodeProto. Returns `None` if the attribute is absent or carries
/// no byte content. The byte payload lives on `attribute.s` per the
/// ONNX convention for raw bytes (paired with
/// `AttributeType::String`); callers reconstruct the PeerId via
/// `PeerId::from_bytes(read_peer_bytes(node)?)`.
pub fn read_peer_bytes(node: &crate::proto::onnx::NodeProto) -> Option<&[u8]> {
    let attr = node.attribute.iter().find(|a| a.name == ATTR_PEER)?;
    if attr.s.is_empty() {
        None
    } else {
        Some(&attr.s)
    }
}

/// Stamp the destination peer onto a Send / gate NodeProto's
/// `attribute.s` (bytes) using the canonical multihash form. Used
/// by the compiler's gate-insertion passes and any pass synthesizing
/// new Send NodeProtos.
pub fn stamp_peer_bytes(node: &mut crate::proto::onnx::NodeProto, peer_bytes: Vec<u8>) {
    let attr_type = attribute_proto::AttributeType::String as i32;
    if let Some(existing) = node.attribute.iter_mut().find(|a| a.name == ATTR_PEER) {
        existing.s = peer_bytes;
        existing.r#type = attr_type;
        existing.i = 0;
    } else {
        node.attribute.push(AttributeProto {
            name: ATTR_PEER.to_string(),
            s: peer_bytes,
            r#type: attr_type,
            ..Default::default()
        });
    }
}

/// Stamp the [`WIRE_TRANSPORT_KEY`] classification onto a wire-op
/// NodeProto's `metadata_props` (idempotent: replaces an existing
/// value).
pub fn stamp_wire_transport(node: &mut crate::proto::onnx::NodeProto, transport: &str) {
    if let Some(entry) = node
        .metadata_props
        .iter_mut()
        .find(|p| p.key == WIRE_TRANSPORT_KEY)
    {
        entry.value = transport.to_string();
    } else {
        node.metadata_props.push(StringStringEntryProto {
            key: WIRE_TRANSPORT_KEY.to_string(),
            value: transport.to_string(),
        });
    }
}

