//! `analyze_wire_edges` — classify each cross-Node edge as `data`
//! or `trigger_only` and group sends in the same cycle scope for
//! batching.

use std::collections::{BTreeMap, HashMap};

use crate::error::CompileError;
use crate::partition_by_wire_ops::WireEdge;
use bb_ir::proto::onnx::{GraphProto, NodeProto, StringStringEntryProto};

// IR-level metadata keys + helpers live in `bb_ir::keys` — single
// source of truth across DSL → compiler → runtime. This pass uses
// them directly.
pub use bb_ir::keys::{
    dest_suffix_attribute, BATCH_GROUP_KEY, DEST_SITE_NAME_PREFIX, DEST_SUFFIX_ATTR_PREFIX,
    TRIGGER_DENOTATION, WIRE_TRANSPORT_KEY,
};

/// Per-edge classification + per-cycle batching. Pure.
///
/// Writes the classification metadata directly onto the matching
/// `sub_graph.node` NodeProtos (Send + Recv pairs). The
/// `wire_edges` slice drives iteration — identifying which edges
/// exist and pairing producer/consumer roles — but the pass treats
/// it as read-only: the `WireEdge.send_node` / `WireEdge.recv_node`
/// clones are discarded by downstream passes, so writing them is
/// a no-op.
pub fn analyze_wire_edges(
    sub_graph: &mut GraphProto,
    wire_edges: &[WireEdge],
) -> Result<(), CompileError> {
    let denotation_by_name: HashMap<&str, &str> = sub_graph
        .value_info
        .iter()
        .chain(sub_graph.input.iter())
        .chain(sub_graph.output.iter())
        .filter_map(|v| {
            let denot = v.r#type.as_ref()?.denotation.as_str();
            if denot.is_empty() {
                None
            } else {
                Some((v.name.as_str(), denot))
            }
        })
        .collect();

    let mut batch_groups: BTreeMap<(String, String), u32> = BTreeMap::new();
    let mut next_batch_id: u32 = 0;

    // Classification rule: if EVERY
    // downstream consumer's input-port type is `bb.trigger`, mark
    // the edge `trigger_only`; otherwise `data`. We walk the
    // sub-graph's nodes to find each consumer of the edge value,
    // then resolve that consumer's input-port type via the
    // value_info denotation map. Empty consumer set defaults to
    // `data` (conservative - preserves payload bytes for
    // out-of-graph receivers).
    for edge in wire_edges {
        let consumer_port_denots: Vec<&str> =
            consumer_input_denotations(&denotation_by_name, sub_graph, &edge.value_name);

        let transport = if consumer_port_denots.is_empty() {
            // No in-sub-graph consumer found - fall back to the
            // edge value's declared denotation (preserves prior
            // behavior for edges that exit the sub-graph entirely).
            match denotation_by_name.get(edge.value_name.as_str()) {
                Some(d) if *d == TRIGGER_DENOTATION => "trigger_only",
                _ => "data",
            }
        } else if consumer_port_denots
            .iter()
            .all(|d| *d == TRIGGER_DENOTATION)
        {
            "trigger_only"
        } else {
            "data"
        };

        let key = (edge.producer_role.clone(), edge.consumer_role.clone());
        let batch_id = *batch_groups.entry(key).or_insert_with(|| {
            let id = next_batch_id;
            next_batch_id += 1;
            id
        });
        let batch_str = batch_id.to_string();

        // Stamp the deferred recv-site name on the producer Send
        // NodeProto. Node's install path resolves each entry
        // to a `NodeSiteId` against the consumer's installed graph
        // and rewrites the Send NodeProto with a canonical
        // `dest_suffix.<input>` Address-bytes attribute.
        let recv_site_name = edge
            .recv_node
            .output
            .first()
            .cloned()
            .unwrap_or_else(|| edge.value_name.clone());
        let dest_site_key = format!("{DEST_SITE_NAME_PREFIX}{}", edge.value_name);

        for node in sub_graph.node.iter_mut() {
            let matches_value = node.output.iter().any(|o| o == &edge.value_name);
            if !matches_value {
                continue;
            }
            if node.op_type == "Send" {
                set_metadata(&mut node.metadata_props, WIRE_TRANSPORT_KEY, transport);
                set_metadata(&mut node.metadata_props, BATCH_GROUP_KEY, &batch_str);
                set_metadata(&mut node.metadata_props, &dest_site_key, &recv_site_name);
            } else if node.op_type == "Recv" {
                set_metadata(&mut node.metadata_props, WIRE_TRANSPORT_KEY, transport);
                set_metadata(&mut node.metadata_props, BATCH_GROUP_KEY, &batch_str);
            }
        }
    }

    Ok(())
}

/// Look up a per-input `dest_suffix.<name>` attribute on the given
/// NodeProto. Returns the canonical Address byte encoding stamped by
/// Node's install-time resolver. Used by the wire syscall to
/// populate each `SlotFill.dest_suffix` at dispatch time.
pub fn dest_suffix_attr<'a>(node: &'a NodeProto, input_name: &str) -> Option<&'a [u8]> {
    let key = format!("{DEST_SUFFIX_ATTR_PREFIX}{input_name}");
    node.attribute
        .iter()
        .find(|a| a.name == key)
        .map(|a| a.s.as_slice())
}

/// Walk every NodeProto in `sub_graph` and, for each one that
/// consumes `value_name` on any of its input ports, return the
/// per-port denotation as declared in `denotation_by_name`. Empty
/// when no consumer references the value. (ONNX typically declares
/// one type per value name, but the per-port walk lets us be
/// explicit about §9.1's "every downstream consumer" rule.)
fn consumer_input_denotations<'a>(
    denotation_by_name: &HashMap<&'a str, &'a str>,
    sub_graph: &'a GraphProto,
    value_name: &str,
) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::new();
    for node in &sub_graph.node {
        for input in &node.input {
            if input == value_name {
                if let Some(d) = denotation_by_name.get(input.as_str()) {
                    out.push(*d);
                } else {
                    // Consumer with no declared type → default to
                    // data so we keep the payload bytes available.
                    out.push("");
                }
            }
        }
    }
    out
}

fn set_metadata(props: &mut Vec<StringStringEntryProto>, key: &str, value: &str) {
    if let Some(existing) = props.iter_mut().find(|p| p.key == key) {
        existing.value = value.to_string();
        return;
    }
    props.push(StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    });
}

