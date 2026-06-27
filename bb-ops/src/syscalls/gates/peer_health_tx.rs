//! `PeerHealthGateTx` syscall op - gate inserted by the compiler
//! pass `bb-compiler/src/insert_peer_health_gate_tx.rs` upstream of
//! every `wire::Send`. Consults [`bb_runtime::framework::PeerGovernor`]
//! and emits a `TriggerValue` on Allow; the downstream `wire::Send`
//! fires only on Allow. On Deny the op returns an `OpError` whose
//! `detail` carries a stable label (`blocklisted` / `not_allowlisted`
//! / `cooldown`).
//!
//! The peer identity rides as `ATTR_PEER` on the gate NodeProto
//! as multihash bytes on `attribute.s`.

use bb_ir::proto::onnx::NodeProto;
use bb_ir::wire_shape;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::framework::{BlockReason, Decision};
use bb_runtime::ids::PeerId;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TriggerValue;

/// Marker struct for `register_syscall::<PeerHealthGateTxOp>`.
pub struct PeerHealthGateTxOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "PeerHealthGateTx";
/// Re-export the canonical ATTR_PEER constant from `bb_ir::wire_shape`.
pub const ATTR_PEER: &str = wire_shape::ATTR_PEER;

/// Invoke fn - consults `PeerGovernor::check_outbound` and emits a
/// `TriggerValue` on Allow, fails with a stable `reason` label on
/// any Deny variant.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let peer = read_peer_attr(node).ok_or_else(|| OpError {
        detail: format!("PeerHealthGateTx missing required `{ATTR_PEER}` attribute"),
        ..Default::default()
    })?;
    let now_ns = ctx.time.scheduler.now_ns();

    match ctx
        .peers
        .governor
        .check_outbound(peer, ctx.peers.backoff, now_ns)
    {
        Decision::Allow => Ok(DispatchResult::Immediate(vec![(
            "trigger".to_string(),
            Box::new(TriggerValue),
        )])),
        Decision::Deny(reason) => Err(OpError {
            detail: format!(
                "PeerHealthGateTx denied send to peer {peer:?}: reason={}",
                reason_label(&reason),
            ),
            ..Default::default()
        }),
    }
}

/// Reduce a `BlockReason` to a stable diagnostic string used in
/// `OpError.detail`. Consumers match on the label to react.
pub fn reason_label(reason: &BlockReason) -> &'static str {
    match reason {
        BlockReason::Blocklisted => "blocklisted",
        BlockReason::NotAllowlisted => "not_allowlisted",
        BlockReason::Cooldown { .. } => "cooldown",
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
