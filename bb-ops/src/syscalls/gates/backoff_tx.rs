//! `BackoffGateTx` syscall op - gate inserted by the compiler pass
//! `bb-compiler/src/insert_backoff_gate_tx.rs` between
//! `PeerHealthGateTx` and `wire::Send`. Consults
//! [`bb_runtime::framework::BackoffTable`] and drops the send if the
//! destination peer is still in cooldown.
//!
//! The destination peer rides as `ATTR_PEER` on the gate NodeProto
//! as multihash bytes on `attribute.s` (per
//! [`bb_ir::wire_shape::ATTR_PEER`] / [`bb_ir::wire_shape::read_peer_bytes`]).

use bb_ir::proto::onnx::NodeProto;
use bb_ir::wire_shape;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::ids::PeerId;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TriggerValue;

/// Marker struct for `register_syscall::<BackoffGateTxOp>`.
pub struct BackoffGateTxOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "BackoffGateTx";
/// Re-export the canonical ATTR_PEER constant from `bb_ir::wire_shape`.
pub const ATTR_PEER: &str = wire_shape::ATTR_PEER;

/// Invoke fn - consults `BackoffTable::should_retry` for the
/// destination peer. Emits a `TriggerValue` when the peer is retry-
/// able; fails with a `cooldown` reason when still in the backoff
/// window.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let peer = read_peer_attr(node).ok_or_else(|| OpError {
        detail: format!("BackoffGateTx missing required `{ATTR_PEER}` attribute"),
        ..Default::default()
    })?;
    let now_ns = ctx.time.scheduler.now_ns();

    if ctx.peers.backoff.should_retry(peer, now_ns) {
        Ok(DispatchResult::Immediate(vec![(
            "trigger".to_string(),
            Box::new(TriggerValue),
        )]))
    } else {
        Err(OpError {
            detail: format!("BackoffGateTx held send to peer {peer:?}: reason=cooldown"),
            ..Default::default()
        })
    }
}

/// Read the gate's destination peer from `ATTR_PEER` (multihash
/// bytes on `attribute.s`).
fn read_peer_attr(node: &NodeProto) -> Option<PeerId> {
    wire_shape::read_peer_bytes(node).and_then(|bytes| PeerId::from_bytes(bytes).ok())
}


use bb_runtime::registry::OpRegistration as _BbOpsSyscallReg;

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: OP_TYPE,
        invoke,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}
