//! `DeadlineMatch` syscall - first-arrival selector between a
//! `then` trigger and a `timeout` trigger.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TriggerValue;

/// Marker struct.
pub struct DeadlineMatchOp;

/// Op domain.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "DeadlineMatch";

/// Invoke fn - emits a single `winner` Trigger per IR_AND_DSL.md
/// §5a. First-to-fire semantics: whichever of `then` / `timeout`
/// has a non-empty slot becomes the winner. Once a winner is
/// determined, the per-Op latch absorbs subsequent invocations so
/// downstream consumers don't double-fire.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    if inputs.is_empty() {
        return Err(OpError {
            detail: "DeadlineMatch requires at least one input".to_string(),
            ..Default::default()
        });
    }
    // Latch on first fire per execution. Keyed by (OpRef, ExecId)
    // so the same op fires once per logical execution rather than
    // once per Node lifetime.
    let latch_key = (ctx.current.op_ref.as_u64(), ctx.current.exec_id.as_u64());
    if !ctx.syscall.deadline_match_fired.insert(latch_key) {
        return Ok(DispatchResult::Immediate(vec![]));
    }
    Ok(DispatchResult::Immediate(vec![(
        "winner".to_string(),
        Box::new(TriggerValue),
    )]))
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
