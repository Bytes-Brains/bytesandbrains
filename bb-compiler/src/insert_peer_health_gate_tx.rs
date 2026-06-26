//! Compiler pass - pair every `wire::Send` op with an upstream
//! `PeerHealthGateTx` syscall. The gate consults
//! [`crate::framework::PeerGovernor`] before the Send fires;
//! blocked / cooldowned peers fail the send through the existing
//! `OpFailed` path without the Send queueing an envelope.
//!

use crate::error::CompileError;
use bb_ir::proto::onnx::{GraphProto, NodeProto, StringStringEntryProto};
use bb_ir::syscall_ids::{OP_PEER_HEALTH_GATE_TX as GATE_OP_TYPE, SYSCALL_DOMAIN as GATE_DOMAIN};
use bb_ir::wire_shape;

/// Idempotence stamp on the gated Send.
pub const GATED_KEY: &str = "ai.bytesandbrains.peer_health_tx_gated";

const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";
const WIRE_SEND_OP: &str = "Send";

/// Insert a `PeerHealthGateTx` upstream of every `wire::Send` node
/// that carries a destination `peer` attribute.
pub fn insert_peer_health_gate_tx(sub_graph: &mut GraphProto) -> Result<(), CompileError> {
    let mut gates: Vec<NodeProto> = Vec::new();

    for node in sub_graph.node.iter_mut() {
        if node.domain != WIRE_DOMAIN || node.op_type != WIRE_SEND_OP {
            continue;
        }
        if metadata_value(node, GATED_KEY).is_some() {
            continue;
        }
        let Some(peer) = read_peer(node) else {
            continue;
        };
        let Some(gated_input) = node.input.first().cloned() else {
            continue;
        };

        let gate_output = format!("{}#peer_health_tx_gated", node.name);
        gates.push(build_gate_node(
            &node.name,
            &gated_input,
            &gate_output,
            &peer,
        ));

        node.input[0] = gate_output;
        set_metadata(&mut node.metadata_props, GATED_KEY, "true");
    }

    sub_graph.node.extend(gates);
    Ok(())
}

fn build_gate_node(
    source_name: &str,
    trigger_input: &str,
    gate_output: &str,
    peer_bytes: &[u8],
) -> NodeProto {
    // stamp ATTR_PEER as multihash bytes
    // (`attribute.s`) via wire_shape::stamp_peer_bytes.
    let mut node = NodeProto {
        op_type: GATE_OP_TYPE.to_string(),
        domain: GATE_DOMAIN.to_string(),
        name: format!("PeerHealthGateTx@{source_name}"),
        input: vec![trigger_input.to_string()],
        output: vec![gate_output.to_string()],
        metadata_props: vec![
            StringStringEntryProto {
                key: "ai.bytesandbrains.peer_health_tx_source".to_string(),
                value: source_name.to_string(),
            },
            StringStringEntryProto {
                key: GATED_KEY.to_string(),
                value: "true".to_string(),
            },
        ],
        ..Default::default()
    };
    wire_shape::stamp_peer_bytes(&mut node, peer_bytes.to_vec());
    node
}

/// Read the destination peer bytes from the Send.
fn read_peer(node: &NodeProto) -> Option<Vec<u8>> {
    wire_shape::read_peer_bytes(node).map(|bytes| bytes.to_vec())
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

