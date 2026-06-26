//! Compiler pass - pair every synthesized `wire::Recv` op with a
//! downstream `DedupGateRx`. First in the RX gate chain (cheap drops
//! for replays).
//!
//! The gate consumes the Recv's data output, hashes the wire bytes,
//! and consults [`crate::framework::InboundDedup`]. On dup it returns
//! an `OpError` whose `detail` carries `duplicate`; on first-arrival
//! the value forwards polymorphically.
//!
//! Idempotent. Updates the Recv's
//! `RX_CHAIN_HEAD_KEY` metadata so subsequent RX passes attach to the
//! gate's output rather than the Recv's.

use crate::error::CompileError;
use crate::rx_chain::{rx_chain_head, set_rx_chain_head};
use bb_ir::proto::onnx::{GraphProto, NodeProto, StringStringEntryProto};
use bb_ir::syscall_ids::{OP_DEDUP_GATE_RX as GATE_OP_TYPE, SYSCALL_DOMAIN as GATE_DOMAIN};

/// Idempotence stamp on the gated Recv.
pub const GATED_KEY: &str = "ai.bytesandbrains.dedup_rx_gated";

const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";
const WIRE_RECV_OP: &str = "Recv";

/// Insert a `DedupGateRx` after every synthesized `wire::Recv`.
pub fn insert_dedup_gate_rx(sub_graph: &mut GraphProto) -> Result<(), CompileError> {
    let recv_indices: Vec<usize> = sub_graph
        .node
        .iter()
        .enumerate()
        .filter_map(|(i, n)| (n.domain == WIRE_DOMAIN && n.op_type == WIRE_RECV_OP).then_some(i))
        .collect();

    let mut new_gates: Vec<NodeProto> = Vec::new();

    for recv_idx in recv_indices {
        if metadata_value(&sub_graph.node[recv_idx], GATED_KEY).is_some() {
            continue;
        }
        let recv_name = sub_graph.node[recv_idx].name.clone();
        let head = rx_chain_head(&sub_graph.node[recv_idx]);
        let new_head = format!("{recv_name}#dedup_rx_out");

        new_gates.push(build_gate_node(&recv_name, &head, &new_head));

        // Rewire consumers that read from the prior head.
        rewire_consumers(sub_graph, recv_idx, &head, &new_head);

        set_metadata(
            &mut sub_graph.node[recv_idx].metadata_props,
            GATED_KEY,
            "true",
        );
        set_rx_chain_head(&mut sub_graph.node[recv_idx], &new_head);
    }

    sub_graph.node.extend(new_gates);
    Ok(())
}

fn build_gate_node(source_name: &str, input: &str, output: &str) -> NodeProto {
    NodeProto {
        op_type: GATE_OP_TYPE.to_string(),
        domain: GATE_DOMAIN.to_string(),
        name: format!("DedupGateRx@{source_name}"),
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        metadata_props: vec![StringStringEntryProto {
            key: "ai.bytesandbrains.dedup_rx_source".to_string(),
            value: source_name.to_string(),
        }],
        ..Default::default()
    }
}

fn rewire_consumers(sub_graph: &mut GraphProto, recv_idx: usize, old_name: &str, new_name: &str) {
    for (idx, node) in sub_graph.node.iter_mut().enumerate() {
        if idx == recv_idx {
            continue;
        }
        for inp in node.input.iter_mut() {
            if inp == old_name {
                *inp = new_name.to_string();
            }
        }
    }
}

fn metadata_value(node: &NodeProto, key: &str) -> Option<String> {
    node.metadata_props
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.clone())
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

