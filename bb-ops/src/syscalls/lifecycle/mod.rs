//! Lifecycle + bootstrap ops.
//!
//! - `LifecyclePhase(phase: int) → Trigger` - fires when
//!   `Engine::fire_lifecycle(phase)` is called.
//! - `BootstrapDispatch() → cmd` - mints a CommandId.
//! - `BootstrapOutput(cmd) → Trigger` - awaits completion of the
//!   matching CommandId.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::ids::CommandId;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::{CommandIdValue, TriggerValue};

const DOMAIN: &str = "ai.bytesandbrains.syscall";

/// Marker struct for dispatch_table TypeId keying.
pub struct LifecyclePhaseOp;

/// `LifecyclePhase(phase: string) → Trigger`. Phase-gated firing is
/// enforced by the engine: only ops enrolled in
/// `Engine.lifecycle_table[phase]` are pushed onto the frontier when
/// `fire_lifecycle(phase)` runs, so this body only needs to emit the
/// trigger when invoked. `Node` is the named consumer
/// of `Engine::register_lifecycle_op`, parsing each node's `phase`
/// attribute at install time.
pub fn invoke_lifecycle_phase(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    Ok(DispatchResult::Immediate(vec![(
        "trigger".to_string(),
        Box::new(TriggerValue),
    )]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct BootstrapDispatchOp;

/// `BootstrapDispatch() → cmd`.
pub fn invoke_bootstrap_dispatch(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let cmd = ctx.allocate_command_id();
    Ok(DispatchResult::Immediate(vec![(
        "cmd".to_string(),
        Box::new(CommandIdValue(bb_runtime::ids::CommandId::from(
            cmd.as_u64(),
        ))),
    )]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct BootstrapOutputOp;

/// `BootstrapOutput(cmd) → Trigger`. Awaits the matching
/// `BootstrapDispatch`'s `CommandId`. Reads the upstream `cmd`
/// input (a `CommandIdValue`) and returns
/// `DispatchResult::Async(cmd_id)` - the engine parks the op in
/// `pending_async[cmd_id]` until the host completes the command
/// via the ingress queue.
pub fn invoke_bootstrap_output(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let cmd_handle = inputs
        .iter()
        .find(|(name, _)| *name == "cmd")
        .map(|(_, h)| *h)
        .ok_or_else(|| OpError {
            detail: "BootstrapOutput: missing `cmd` input".into(),
            ..Default::default()
        })?;
    let cmd = cmd_handle
        .as_any()
        .downcast_ref::<CommandIdValue>()
        .ok_or_else(|| OpError {
            detail: "BootstrapOutput: `cmd` input is not a CommandIdValue".into(),
            ..Default::default()
        })?;
    Ok(DispatchResult::Async(CommandId::new(cmd.0.as_u64())))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

/// Linker-anchor - see `bb_ops::link_force` for details.
pub fn link_force() {
    use std::hint::black_box;
    black_box(invoke_lifecycle_phase as usize);
    black_box(invoke_bootstrap_dispatch as usize);
    black_box(invoke_bootstrap_output as usize);
}

use bb_runtime::registry::OpRegistration as _BbOpsSyscallReg;

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "LifecyclePhase",
        invoke: invoke_lifecycle_phase,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "BootstrapDispatch",
        invoke: invoke_bootstrap_dispatch,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "BootstrapOutput",
        invoke: invoke_bootstrap_output,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}
