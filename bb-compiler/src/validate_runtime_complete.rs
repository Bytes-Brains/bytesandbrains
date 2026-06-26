//! Final structural completeness check.
//!
//! Walks each per-partition installed graph and asserts every wire
//! op is paired with the gate chain the engine expects and every
//! async op carries a `DeadlineCheck`. Catches missing
//! compiler-inserted ops before the graph reaches `Node::ensure_ready`.

use bb_ir::proto::onnx::GraphProto;

use crate::error::CompileError;
use crate::partition_by_wire_ops::WIRE_DOMAIN;

const SYSCALL_DOMAIN: &str = "ai.bytesandbrains.syscall";
const SEND_OP: &str = "Send";
const RECV_OP: &str = "Recv";

const PEER_HEALTH_GATE_TX_OP: &str = "PeerHealthGateTx";
const BACKOFF_GATE_TX_OP: &str = "BackoffGateTx";
const DEDUP_GATE_RX_OP: &str = "DedupGateRx";
const PEER_HEALTH_GATE_RX_OP: &str = "PeerHealthGateRx";
const BACKOFF_GATE_RX_OP: &str = "BackoffGateRx";
const DEADLINE_CHECK_OP: &str = "DeadlineCheck";

const DEADLINE_NS_ATTR: &str = "deadline_ns";

const PEER_ATTR: &str = "peer";

/// Validate that every compiler-required gate / lifecycle op is
/// present alongside the ops it serves. Returns `Ok(())` on success,
/// `Err` describing the first missing piece otherwise.
///
/// Sends/Recvs that carry an explicit `peer` attribute (peer-specific
/// routing) are gated by the canonical TX/RX gate chains. Sends/Recvs
/// without `peer` route via the runtime address book using
/// `dest_target` metadata and do not require the per-peer gates.
pub fn validate_runtime_complete(sub_graph: &GraphProto) -> Result<(), CompileError> {
    let nodes = &sub_graph.node;
    let has_op = |op_type: &str, domain: &str| -> bool {
        nodes
            .iter()
            .any(|n| n.op_type == op_type && n.domain == domain)
    };

    // Peer-routed Sends need the TX gate chain.
    let has_peer_send = nodes.iter().any(|n| {
        n.domain == WIRE_DOMAIN
            && n.op_type == SEND_OP
            && n.attribute.iter().any(|a| a.name == PEER_ATTR)
    });
    if has_peer_send {
        if !has_op(PEER_HEALTH_GATE_TX_OP, SYSCALL_DOMAIN) {
            return Err(CompileError::Internal {
                detail: format!(
                    "validate_runtime_complete: partition `{}` has a peer-routed wire.Send but no PeerHealthGateTx",
                    sub_graph.name,
                ),
            });
        }
        if !has_op(BACKOFF_GATE_TX_OP, SYSCALL_DOMAIN) {
            return Err(CompileError::Internal {
                detail: format!(
                    "validate_runtime_complete: partition `{}` has a peer-routed wire.Send but no BackoffGateTx",
                    sub_graph.name,
                ),
            });
        }
    }

    // Peer-routed Recvs need the RX gate chain.
    let has_peer_recv = nodes.iter().any(|n| {
        n.domain == WIRE_DOMAIN
            && n.op_type == RECV_OP
            && n.attribute.iter().any(|a| a.name == PEER_ATTR)
    });
    if has_peer_recv {
        if !has_op(DEDUP_GATE_RX_OP, SYSCALL_DOMAIN) {
            return Err(CompileError::Internal {
                detail: format!(
                    "validate_runtime_complete: partition `{}` has a peer-routed wire.Recv but no DedupGateRx",
                    sub_graph.name,
                ),
            });
        }
        if !has_op(PEER_HEALTH_GATE_RX_OP, SYSCALL_DOMAIN) {
            return Err(CompileError::Internal {
                detail: format!(
                    "validate_runtime_complete: partition `{}` has a peer-routed wire.Recv but no PeerHealthGateRx",
                    sub_graph.name,
                ),
            });
        }
        if !has_op(BACKOFF_GATE_RX_OP, SYSCALL_DOMAIN) {
            return Err(CompileError::Internal {
                detail: format!(
                    "validate_runtime_complete: partition `{}` has a peer-routed wire.Recv but no BackoffGateRx",
                    sub_graph.name,
                ),
            });
        }
    }

    // Every NodeProto carrying `deadline_ns` needs DeadlineCheck.
    let any_deadline = nodes
        .iter()
        .any(|n| n.attribute.iter().any(|a| a.name == DEADLINE_NS_ATTR));
    if any_deadline && !has_op(DEADLINE_CHECK_OP, SYSCALL_DOMAIN) {
        return Err(CompileError::Internal {
            detail: format!(
                "validate_runtime_complete: partition `{}` carries deadline_ns but no DeadlineCheck",
                sub_graph.name,
            ),
        });
    }

    // every gate-insertion pass publishes a
    // `GateContract` via inventory; iterate them so that adding a
    // new gate is "ship the inserting pass + register its
    // contract" rather than "edit `validate_runtime_complete`"
    // (closes `chief:S12`). Failures surface as
    // `CompileError::RuntimeIncomplete { missing }`.
    for reg in crate::gate_contract::contracts() {
        reg.contract.assert_inserted(sub_graph)?;
    }

    Ok(())
}

