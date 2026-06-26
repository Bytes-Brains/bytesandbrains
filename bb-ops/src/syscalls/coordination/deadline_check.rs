//! `DeadlineCheck` syscall op - a clock-driven gate that passes a
//! trigger through if the current time is before its `deadline_ns`
//! attribute, fails with `"deadline exceeded"` otherwise.
//!
//! Authors don't typically record this op directly; the compiler
//! pass `bb-compiler/src/insert_async_deadlines.rs` inserts one
//! upstream of every Async-shaped op that carries a `deadline_ns`
//! attribute (Pattern C in the
//! design). This is the pre-suspension gate; the engine's
//! Phase-5 scan over `PendingAsync.deadline_ns` is the
//! post-suspension timer.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::TriggerValue;

/// Marker struct for `register_syscall::<DeadlineCheckOp>`.
pub struct DeadlineCheckOp;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.syscall";
/// Op type name.
pub const OP_TYPE: &str = "DeadlineCheck";
/// Attribute name carrying the absolute deadline in nanoseconds on
/// the engine's `scheduler.now_ns()` clock.
pub const ATTR_DEADLINE_NS: &str = "deadline_ns";

/// Invoke fn - if `now_ns < deadline_ns`, emit a `TriggerValue`;
/// otherwise return `OpError("deadline exceeded")` which surfaces
/// as `EngineStep::OpFailed`.
pub fn invoke(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let deadline_ns = node
        .attribute
        .iter()
        .find(|a| a.name == ATTR_DEADLINE_NS)
        .map(|a| a.i as u64)
        .ok_or_else(|| OpError {
            detail: format!("DeadlineCheck missing required `{ATTR_DEADLINE_NS}` attribute"),
            ..Default::default()
        })?;
    let now_ns = ctx.time.scheduler.now_ns();
    if now_ns >= deadline_ns {
        return Err(OpError {
            detail: "deadline exceeded".to_string(),
            ..Default::default()
        });
    }
    Ok(DispatchResult::Immediate(vec![(
        "trigger".to_string(),
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
