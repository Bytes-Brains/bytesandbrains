//! `infer_peer_classes` - stamp every NodeProto with the **class of
//! Node** it runs on.
//!
//! Runs in [`runner::run_pipeline`](super::runner::run_pipeline)
//! after `expand_ops` and before `synthesize_wire_recvs`. The result feeds
//! [`partition_by_wire_ops`](super::partition_by_wire_ops::partition_by_wire_ops) - partitions
//! are now defined by `home_class`, not by `module_instance` chains.
//!
//! ## Algorithm
//!
//! 1. Seed every function input's `home` to [`SELF_CLASS`].
//! 2. Walk nodes in declaration order. For each NodeProto:
//!    - `wire.Send` re-homes its `data` output to the **destination
//!      class** (taken from the peer input's `peer_class` tag).
//!      The send itself runs on its payload's home class; the
//!      `handle` output stays with the sender. Self-send case
//!      (`dest_class == payload_home`) is just a value of `dest_class`.
//!    - Every other op inherits its home from its data inputs. All
//!      data inputs (i.e. inputs that aren't PEER_ID values) must
//!      agree on a home; otherwise [`CompileError::CrossClassDataflow`].
//!      Peer-id inputs are **ambient** - they don't constrain the
//!      consuming op's home class.
//! 3. The home is stamped on the NodeProto as [`HOME_CLASS_KEY`]
//!    metadata for downstream passes.
//!
//! ## Self-send semantics
//!
//! When a `wire.Send`'s destination class equals its sender's home
//! class, both the send and the synthesized recv land in the same
//! partition at the partition pass. The runtime side is N physical
//! instances of one class talking to each other (e.g. gossip peers).

use std::collections::HashMap;

use crate::error::CompileError;
use crate::partition_by_wire_ops::WIRE_DOMAIN;
use bb_ir::peer_class::{
    home_class_of_node, peer_class_of_node, peer_class_of_value_info, HOME_CLASS_KEY,
    PEER_CLASS_KEY, SELF_CLASS,
};
use bb_ir::proto::onnx::{type_proto, GraphProto, StringStringEntryProto, TypeProto};

/// Walk `graph.node` and stamp `HOME_CLASS_KEY` on each NodeProto.
/// Pure.
pub fn infer_peer_classes(graph: &mut GraphProto) -> Result<(), CompileError> {
    // Compile-time peer-class trace: for every wire.Send peer
    // input, walk backward through allow-listed pass-through ops
    // (Identity, Slice, Gather, Concat, Squeeze, Unsqueeze). Graph
    // inputs reached along that walk get the `peer_class =
    // <input_name>` stamp; non-pass-through producers stop the
    // trace (their own peer_class metadata, if any, drives routing).
    stamp_peer_class_on_inputs_feeding_wire_sends(graph);

    // value_name → home class.
    let mut home: HashMap<String, String> = HashMap::new();

    // wire_id → destination class for the matched Send. Populated
    // when each Send is processed; consulted when the paired Recv
    // is processed (Recv outputs + home_class lift to the same
    // destination class so the partitioner cuts cleanly).
    let mut wire_id_to_dest_class: HashMap<String, String> = HashMap::new();

    // Pre-scan function inputs: every input is on @self; PEER_ID
    // inputs additionally seed `peer_class[input_name] = <class>` so
    // a `wire.Send` reading that input can find its destination class.
    let mut peer_class_of_value: HashMap<String, String> = HashMap::new();
    let mut peer_id_value_names: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    for vi in &graph.input {
        home.insert(vi.name.clone(), SELF_CLASS.to_string());
        if value_info_is_peer_id(vi) {
            peer_id_value_names.insert(vi.name.clone());
        }
        if let Some(class) = peer_class_of_value_info(vi) {
            peer_class_of_value.insert(vi.name.clone(), class.to_string());
        }
    }
    for vi in &graph.value_info {
        if value_info_is_peer_id(vi) {
            peer_id_value_names.insert(vi.name.clone());
        }
        if let Some(class) = peer_class_of_value_info(vi) {
            peer_class_of_value
                .entry(vi.name.clone())
                .or_insert_with(|| class.to_string());
        }
    }

    // Walk nodes in declaration order. The runner only feeds us
    // topologically ordered functions; we don't re-sort.
    for node in graph.node.iter_mut() {
        // Skip nodes that already carry an inferred home (idempotent
        // re-runs return the same stamps).
        if home_class_of_node(node).is_some() {
            continue;
        }

        // Record dynamically-produced peer outputs (peer-sampling,
        // gossip neighbor selection) BEFORE handling the node so a
        // wire.Send referencing one of these outputs finds it.
        if let Some(class) = peer_class_of_node(node) {
            for out in &node.output {
                if !out.is_empty() {
                    peer_class_of_value
                        .entry(out.clone())
                        .or_insert_with(|| class.to_string());
                }
            }
        }

        let is_wire_send = node.domain == WIRE_DOMAIN && node.op_type == "Send";
        let is_wire_recv = node.domain == WIRE_DOMAIN && node.op_type == "Recv";
        if is_wire_send {
            // wire.Send signature is (payload_0, ..., payload_{N-1}, peer):
            // the peer is the LAST input, payloads precede it.
            // Reading the last input lets multi-input wires (hierarchical
            // FedAvg, GlobalRegistry Announce, gossip disseminate) infer
            // the right destination class.
            //
            // Fallback to `@default` when the peer source carries no class
            // annotation so naming downstream stays stable.
            let payload_name = node.input.first().cloned().unwrap_or_default();
            let peer_input = node.input.last().cloned().unwrap_or_default();
            let payload_home = home
                .get(&payload_name)
                .cloned()
                .unwrap_or_else(|| SELF_CLASS.to_string());
            let dest_class = peer_class_of_value
                .get(&peer_input)
                .cloned()
                .unwrap_or_else(|| "@default".to_string());

            // Record wire_id → dest_class so the paired Recv lifts
            // its outputs into the same partition.
            if let Some(wire_id) = read_wire_id(node) {
                wire_id_to_dest_class.insert(wire_id, dest_class.clone());
            }

            // Send output arity disambiguates the shape:
            //   len==1 → [handle]; output[0] stays with the sender.
            //   len>=2 → [data, handle]; output[0] is the data lifted
            //            to dest_class (carried by the paired Recv on
            //            the single-output variant).
            if let Some(first_out) = node.output.first() {
                if !first_out.is_empty() {
                    let class = if node.output.len() >= 2 {
                        dest_class.clone()
                    } else {
                        payload_home.clone()
                    };
                    home.insert(first_out.clone(), class);
                }
            }
            if let Some(handle_out) = node.output.get(1) {
                if !handle_out.is_empty() {
                    home.insert(handle_out.clone(), payload_home.clone());
                }
            }
            stamp_home(node, &payload_home);
            continue;
        }
        if is_wire_recv {
            // wire.Recv carries no graph inputs; its outputs flow
            // into downstream user ops on the destination class.
            // Match the destination class via the wire_id metadata
            // the DSL stamped on both halves of the pair.
            let dest_class = read_wire_id(node)
                .and_then(|wid| wire_id_to_dest_class.get(&wid).cloned())
                .unwrap_or_else(|| SELF_CLASS.to_string());
            for out in &node.output {
                if !out.is_empty() {
                    home.insert(out.clone(), dest_class.clone());
                }
            }
            stamp_home(node, &dest_class);
            continue;
        }

        // Non-send ops: collect data-input homes. peer_id inputs are
        // ambient routing metadata, not dataflow - they don't
        // constrain home.
        let mut input_homes: Vec<String> = Vec::new();
        for input in &node.input {
            if input.is_empty() {
                continue;
            }
            if peer_id_value_names.contains(input) {
                continue;
            }
            if let Some(h) = home.get(input) {
                input_homes.push(h.clone());
            }
        }
        // Dedup while preserving order so the error message points at
        // the first conflict, not a sorted permutation.
        input_homes.dedup();
        let node_home = match input_homes.len() {
            0 => SELF_CLASS.to_string(),
            1 => input_homes.remove(0),
            _ => {
                return Err(CompileError::CrossClassDataflow {
                    node_name: node.name.clone(),
                    home_a: input_homes[0].clone(),
                    home_b: input_homes[1].clone(),
                });
            }
        };
        for out in &node.output {
            if !out.is_empty() {
                home.insert(out.clone(), node_home.clone());
            }
        }
        stamp_home(node, &node_home);
    }

    Ok(())
}

/// Walk `wire.Send` ops; for each peer-slot input value, trace
/// backward through allow-listed pass-through ops until reaching
/// either a graph input (stamp it) or a non-pass-through producer
/// (the producing op's `peer_class` metadata, if any, drives the
/// destination class downstream — no input-side stamp needed).
///
/// Peer values commonly flow through structural ops (`Identity`,
/// `Slice`, `Gather`, `Squeeze`, `Unsqueeze`, `Concat`) before
/// reaching a `wire.Send`'s peer slot — picking the first N peers
/// of a view or concatenating two peer subsets. The trace tolerates
/// those so the graph-input source still gets stamped.
fn stamp_peer_class_on_inputs_feeding_wire_sends(graph: &mut GraphProto) {
    let producers = build_producer_map(graph);

    let mut input_roots: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();

    for node in &graph.node {
        if node.domain != WIRE_DOMAIN || node.op_type != "Send" {
            continue;
        }
        let Some(peer_input) = node.input.last() else {
            continue;
        };
        if peer_input.is_empty() {
            continue;
        }
        trace_peer_source(
            peer_input,
            &producers,
            &graph.node,
            &mut input_roots,
            &mut visited,
        );
    }

    if input_roots.is_empty() {
        return;
    }

    for vi in graph.input.iter_mut().chain(graph.value_info.iter_mut()) {
        if !input_roots.contains(&vi.name) {
            continue;
        }
        let already = vi.metadata_props.iter().any(|p| p.key == PEER_CLASS_KEY);
        if !already {
            vi.metadata_props.push(StringStringEntryProto {
                key: PEER_CLASS_KEY.to_string(),
                value: vi.name.clone(),
            });
        }
    }
}

/// Trace a value name backward through producers, collecting any
/// graph-input ancestors reached via the allow-listed pass-through
/// ops. Non-pass-through producers terminate the walk (their
/// output may still carry `peer_class` metadata; the main pass
/// reads that separately via [`peer_class_of_node`]).
fn trace_peer_source(
    name: &str,
    producers: &HashMap<String, usize>,
    nodes: &[bb_ir::proto::onnx::NodeProto],
    input_roots: &mut std::collections::HashSet<String>,
    visited: &mut std::collections::HashSet<String>,
) {
    if !visited.insert(name.to_string()) {
        return;
    }
    if let Some(&idx) = producers.get(name) {
        let producer = &nodes[idx];
        if !is_peer_pass_through(producer) {
            return;
        }
        for input in &producer.input {
            if input.is_empty() {
                continue;
            }
            trace_peer_source(input, producers, nodes, input_roots, visited);
        }
        return;
    }
    // Not produced by any node in this graph — it's a graph input
    // (or a function arg). Mark it for stamping.
    input_roots.insert(name.to_string());
}

/// Build `value_name → producing_node_index` over `graph.node`.
/// Empty output names are skipped.
fn build_producer_map(graph: &GraphProto) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (i, node) in graph.node.iter().enumerate() {
        for out in &node.output {
            if out.is_empty() {
                continue;
            }
            m.insert(out.clone(), i);
        }
    }
    m
}

/// Conservative allow-list of ops whose output preserves the
/// peer-class semantics of their inputs. Adding to this list is a
/// deliberate act: a new entry says "if this op's input is a graph
/// input feeding a `wire.Send`'s peer slot, the graph input itself
/// is the peer source." Ops that produce peer values from non-peer
/// inputs (e.g. `PeerSelector::Sample`) are NOT pass-throughs —
/// their `peer_class` metadata already drives destination routing.
fn is_peer_pass_through(node: &bb_ir::proto::onnx::NodeProto) -> bool {
    matches!(
        (node.domain.as_str(), node.op_type.as_str()),
        ("ai.onnx", "Identity")
            | ("ai.onnx", "Slice")
            | ("ai.onnx", "Gather")
            | ("ai.onnx", "Concat")
            | ("ai.onnx", "Squeeze")
            | ("ai.onnx", "Unsqueeze")
    )
}

/// Pull the [`bb_ir::keys::WIRE_ID_KEY`] metadata stamp from a wire
/// op NodeProto. Used to pair Send and Recv NodeProtos the DSL
/// `Graph::wire` emits together.
fn read_wire_id(node: &bb_ir::proto::onnx::NodeProto) -> Option<String> {
    node.metadata_props
        .iter()
        .find(|p| p.key == bb_ir::keys::WIRE_ID_KEY)
        .map(|p| p.value.clone())
}

/// Stamp the `HOME_CLASS_KEY` metadata onto a NodeProto, replacing
/// any existing stamp (idempotent re-runs preserve the same value).
fn stamp_home(node: &mut bb_ir::proto::onnx::NodeProto, home: &str) {
    if let Some(existing) = node
        .metadata_props
        .iter_mut()
        .find(|p| p.key == HOME_CLASS_KEY)
    {
        existing.value = home.to_string();
        return;
    }
    node.metadata_props.push(StringStringEntryProto {
        key: HOME_CLASS_KEY.to_string(),
        value: home.to_string(),
    });
}

/// Returns `true` when a ValueInfoProto carries the `peer_class`
/// metadata stamp from `Graph::input(name, &TYPE_PEER_ID)`. We use the
/// presence of `PEER_CLASS_KEY` as the signal rather than the TypeNode
/// denotation, because the compiler doesn't have access to the
/// `TypeNode` static after the graph crosses the recording boundary.
///
/// Accept both `bb.peer_id` (single recipient) and
/// `bb.peer_id_vec` (broadcast multi-peer recipient) denotations
/// so peer-vec values don't get misclassified as non-peer data.
fn value_info_is_peer_id(vi: &bb_ir::proto::onnx::ValueInfoProto) -> bool {
    if vi.metadata_props.iter().any(|p| p.key == PEER_CLASS_KEY) {
        return true;
    }
    // Fall back to the type's denotation for plain `Output<PeerId>`
    // / `Output<Vec<PeerId>>` values that didn't go through
    // `Graph::input` (hand-built fixtures, replayed ModelProto
    // bodies).
    matches!(&vi.r#type, Some(TypeProto { value: Some(type_proto::Value::TensorType(_)), denotation, .. })
        if denotation == "bb.peer_id" || denotation == "bb.peer_id_vec")
}

