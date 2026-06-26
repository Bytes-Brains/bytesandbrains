//! `BackoffGateRx` syscall op - gate inserted by the compiler pass
//! `bb-compiler/src/insert_backoff_gate_rx.rs` in the RX chain.
//! Consults [`bb_runtime::framework::BackoffTable`] for the inbound
//! envelope's source peer and forwards the input value on retry-
//! eligible; drops with a `cooldown` reason when the peer is still
//! in backoff.
//!
//! The source peer is read from `RuntimeResourceRef::envelope_src_peer`
//! populated by the engine per inbound envelope (RX gates never
//! consult a NodeProto attribute for peer identity; the runtime
//! delivers the inbound src_peer directly).

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

/// Marker struct for `register_syscall::<BackoffGateRxOp>`.
pub struct BackoffGateRxOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "BackoffGateRx";

/// Invoke fn - consults `BackoffTable::should_retry` for the
/// envelope's source peer and forwards the input on retry-eligible.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let (_, value) = inputs.first().ok_or_else(|| OpError {
        detail: "BackoffGateRx requires one input".into(),
        ..Default::default()
    })?;

    let Some(src_peer) = ctx.current.inbound.src_peer else {
        return Err(OpError {
            detail: "BackoffGateRx: no envelope source peer in runtime context".into(),
            ..Default::default()
        });
    };

    let now_ns = ctx.time.scheduler.now_ns();
    if ctx.peers.backoff.should_retry(src_peer, now_ns) {
        Ok(DispatchResult::Immediate(vec![(
            "value".to_string(),
            value.clone_boxed(),
        )]))
    } else {
        Err(OpError {
            detail: format!(
                "BackoffGateRx dropped envelope from peer {src_peer:?}: reason=cooldown"
            ),
            ..Default::default()
        })
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
