//! `PeerHealthGateRx` syscall op - gate inserted by the compiler
//! pass `bb-compiler/src/insert_peer_health_gate_rx.rs` between every
//! synthesized `wire::Recv` and its downstream consumers.
//!
//! Consults [`bb_runtime::framework::PeerGovernor`]::check_inbound for
//! the envelope's source peer and forwards the input value
//! polymorphically on Allow. On Deny the op returns an `OpError`
//! whose `detail` carries a stable label.
//!
//! The source peer is read from `RuntimeResourceRef::envelope_src_peer`
//! which the engine populates per inbound envelope (RX gates never
//! consult a NodeProto attribute for peer identity; the runtime
//! delivers the inbound src_peer directly).

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::framework::{BlockReason, Decision};
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

/// Marker struct for `register_syscall::<PeerHealthGateRxOp>`.
pub struct PeerHealthGateRxOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "PeerHealthGateRx";

/// Invoke fn - consults `PeerGovernor::check_inbound` and forwards
/// the input on Allow; fails with a stable `reason` label on Deny.
/// The forwarded value preserves its concrete type via `clone_boxed`.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let (_, value) = inputs.first().ok_or_else(|| OpError {
        detail: "PeerHealthGateRx requires one input".into(),
        ..Default::default()
    })?;

    let Some(src_peer) = ctx.current.inbound.src_peer else {
        return Err(OpError {
            detail: "PeerHealthGateRx: no envelope source peer in runtime context".into(),
            ..Default::default()
        });
    };

    match ctx.peers.governor.check_inbound(src_peer) {
        Decision::Allow => Ok(DispatchResult::Immediate(vec![(
            "value".to_string(),
            value.clone_boxed(),
        )])),
        Decision::Deny(reason) => Err(OpError {
            detail: format!(
                "PeerHealthGateRx denied envelope from peer {src_peer:?}: reason={}",
                reason_label(&reason),
            ),
            ..Default::default()
        }),
    }
}

/// Reduce a `BlockReason` to a stable diagnostic string used in
/// `OpError.detail`.
pub fn reason_label(reason: &BlockReason) -> &'static str {
    match reason {
        BlockReason::Blocklisted => "blocklisted",
        BlockReason::NotAllowlisted => "not_allowlisted",
        BlockReason::Cooldown { .. } => "cooldown",
    }
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
