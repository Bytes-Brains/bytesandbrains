//! Pass 5 - `partition_by_wire_ops`. Slice the recorded function
//! into per-BB-Node sub-graphs by wire-op reachability per
//! `docs/COMPILER.md` §8.
//!
//! Wire ops (`Send`, `SendReqBatched`, `SendResp`, `Recv`, `RecvReq`,
//! `RecvRespBatched` under domain `ai.bytesandbrains.wire`) are the
//! partition boundary. Two NodeProtos belong to the same partition
//! iff there is a dataflow path between them that does NOT cross a
//! wire op.
//!
//! Each partition is named from the longest common
//! `ai.bytesandbrains.module_instance` prefix across its non-wire
//! nodes (the chain stamped by `Graph::with_module`). Partitions
//! whose nodes share no common prefix fall back to `@default`.
//! Wire ops attach to the partition on their data side: Send-flavored
//! ops join the partition of their data-input producers, Recv-flavored
//! ops join the partition of their data-output consumers.
//!
//! Wire ops are user-authored `Send` NodeProtos and compiler-
//! synthesized `Recv` NodeProtos. The downstream `analyze_wire_edges`
//! pass classifies each cross-partition edge directly on each
//! per-role sub-graph; `wire_edges` carries the resulting per-edge
//! metadata.

use std::collections::{BTreeMap, HashMap};

use crate::error::CompileError;
use crate::synthesize_wire_recvs::SYNTHESIZED_FROM_KEY;
use bb_ir::peer_class::{home_class_of_node, SELF_CLASS};
use bb_ir::proto::onnx::{GraphProto, NodeProto, ValueInfoProto};

/// Wire-op domain - every NodeProto with this domain is a wire op.
/// `wire.Send` is the only user-authored op; `wire.Recv` is
/// synthesized by [`super::synthesize_wire_recvs::synthesize_wire_recvs`].
pub const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";

/// Send-flavored wire op_types. Single user-facing `Send` op per the
/// wire-collapse design - see [`crate::syscall::wire`]. Used by
/// downstream passes to detect send NodeProtos by `op_type`.
pub const SEND_OP_TYPES: &[&str] = &["Send"];

/// Recv-flavored wire op_types. `Recv` is synthesized by
/// [`super::synthesize_wire_recvs::synthesize_wire_recvs`] - never user-authored.
pub const RECV_OP_TYPES: &[&str] = &["Recv"];

/// Output of the partition pass: per-role sub-graphs + cross-role
/// edges. Each cross-partition edge appears as a user-authored
/// `wire::Send` / synthesized `wire::Recv` NodeProto inside the
/// per-role sub-graphs; `wire_edges` carries the matching
/// per-edge metadata (producer/consumer roles, transport kind).
#[derive(Debug, Default)]
pub struct NetworkAnalysis {
    /// One entry per BB-Node partition. Single-Node Modules yield
    /// one entry; federated Modules yield one per wire-op-bounded
    /// partition.
    pub per_role: BTreeMap<String, GraphProto>,

    /// Cross-role edges paired by sender index - one entry per
    /// Send/Recv pair discovered after `synthesize_wire_recvs`.
    /// [`super::analyze_wire_edges::analyze_wire_edges`] reads this
    /// to classify each edge's transport and assign batch ids.
    pub wire_edges: Vec<WireEdge>,
}

/// A directional cross-partition edge - produced by the compiler
/// when a wire op pair is identified. Populated by
/// [`super::analyze_wire_edges::analyze_wire_edges`].
#[derive(Debug)]
pub struct WireEdge {
    /// Origin BB-Node role.
    pub producer_role: String,

    /// Destination BB-Node role.
    pub consumer_role: String,

    /// The value-name crossing the edge (the producer-side output
    /// name).
    pub value_name: String,

    /// Producer-side `Send`-flavored NodeProto.
    pub send_node: NodeProto,

    /// Consumer-side `Recv`-flavored NodeProto.
    pub recv_node: NodeProto,
}

/// Partition the graph by inferred home class. Pure per
/// COMPILER.md §3.2.
///
/// After [`super::infer_peer_classes`] has stamped every NodeProto
/// with `HOME_CLASS_KEY`, partitioning is a direct group-by on that
/// key - the dataflow shape already defines class membership, so
/// neither union-find nor `module_instance` LCP naming is needed.
/// Nodes lacking a home stamp (hand-built fixtures, legacy single-
/// Node Modules) fall through to
/// [`SELF_CLASS`](super::peer_class::SELF_CLASS).
pub fn partition_by_wire_ops(graph: &GraphProto) -> Result<NetworkAnalysis, CompileError> {
    let mut per_role: BTreeMap<String, GraphProto> = BTreeMap::new();
    for node in &graph.node {
        let class = home_class_of_node(node)
            .map(str::to_string)
            .unwrap_or_else(|| SELF_CLASS.to_string());
        per_role.entry(class).or_default().node.push(node.clone());
    }

    // Copy each role's referenced graph.input + value_info.
    let value_info_by_name: HashMap<&str, &ValueInfoProto> = graph
        .value_info
        .iter()
        .map(|v| (v.name.as_str(), v))
        .collect();
    let input_by_name: HashMap<&str, &ValueInfoProto> =
        graph.input.iter().map(|v| (v.name.as_str(), v)).collect();

    // forward each role's relevant slice of
    // `graph.output` so the post-partition validator + downstream
    // passes (analyze_wire_edges, the gate-rx inserters) see the
    // same "this value crosses the boundary" hints the recorder
    // stamped on the original ModelProto.
    let output_by_name: HashMap<&str, &ValueInfoProto> =
        graph.output.iter().map(|v| (v.name.as_str(), v)).collect();

    for sub in per_role.values_mut() {
        let mut referenced: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for node in &sub.node {
            for inp in &node.input {
                if !inp.is_empty() {
                    referenced.insert(inp.clone());
                }
            }
        }
        for name in &referenced {
            if let Some(&vi) = input_by_name.get(name.as_str()) {
                sub.input.push(vi.clone());
            }
            if let Some(&vi) = value_info_by_name.get(name.as_str()) {
                sub.value_info.push(vi.clone());
            }
        }

        // Carry forward every top-level output produced by this
        // sub_graph's nodes. The set of producer-side outputs is
        // exactly the intersection of `graph.output` names and
        // values emitted by the sub_graph's nodes.
        let mut produced_here: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for node in &sub.node {
            for out in &node.output {
                if !out.is_empty() {
                    produced_here.insert(out.clone());
                }
            }
        }
        for name in &produced_here {
            if let Some(&vi) = output_by_name.get(name.as_str()) {
                sub.output.push(vi.clone());
            }
        }
    }

    let wire_edges = discover_wire_edges(graph);

    Ok(NetworkAnalysis {
        per_role,
        wire_edges,
    })
}

/// Pair Send and synthesized Recv NodeProtos into one [`WireEdge`]
/// per cross-partition data flow. Sends carry an
/// `output[0] = "<data>__send_sentinel_<idx>"` rename from
/// [`super::synthesize_wire_recvs`]; synthesized Recvs carry
/// `SYNTHESIZED_FROM_KEY = <idx>` metadata pointing back at the same
/// Send. Matching pairs become wire edges. Sends without a paired
/// Recv (fire-and-forget) are skipped.
fn discover_wire_edges(graph: &GraphProto) -> Vec<WireEdge> {
    let mut send_by_idx: HashMap<usize, &NodeProto> = HashMap::new();
    for node in &graph.node {
        if node.domain != WIRE_DOMAIN || !SEND_OP_TYPES.contains(&node.op_type.as_str()) {
            continue;
        }
        if let Some(idx) = parse_send_sentinel_idx(node) {
            send_by_idx.insert(idx, node);
        }
    }

    let mut edges = Vec::new();
    for recv in &graph.node {
        if recv.domain != WIRE_DOMAIN || !RECV_OP_TYPES.contains(&recv.op_type.as_str()) {
            continue;
        }
        let Some(send_idx) = recv
            .metadata_props
            .iter()
            .find(|p| p.key == SYNTHESIZED_FROM_KEY)
            .and_then(|p| p.value.parse::<usize>().ok())
        else {
            continue;
        };
        let Some(send) = send_by_idx.get(&send_idx) else {
            continue;
        };
        let producer_role = home_class_of_node(send)
            .map(str::to_string)
            .unwrap_or_else(|| SELF_CLASS.to_string());
        let consumer_role = home_class_of_node(recv)
            .map(str::to_string)
            .unwrap_or_else(|| SELF_CLASS.to_string());
        let value_name = recv.output.first().cloned().unwrap_or_default();
        edges.push(WireEdge {
            producer_role,
            consumer_role,
            value_name,
            send_node: (*send).clone(),
            recv_node: recv.clone(),
        });
    }
    edges
}

/// Strip the `__send_sentinel_<idx>` suffix from a Send's first
/// output and return the index. Mirrors the rename
/// [`super::synthesize_wire_recvs`] applies to every Send with a
/// downstream consumer.
fn parse_send_sentinel_idx(send: &NodeProto) -> Option<usize> {
    let first = send.output.first()?;
    let marker = "__send_sentinel_";
    let pos = first.rfind(marker)?;
    first[pos + marker.len()..].parse().ok()
}

