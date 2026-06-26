//! App-emit + telemetry ops - AppEmit, AppNotify, Record,
//! IncrMetric. //!
//! Spec: Sub-C in `docs/IR_AND_DSL.md` §5a.
//!
//! `AppEmit` / `AppNotify` push onto `ctx.syscall.pending_app_events`;
//! the Engine's Phase 8 drains them into `EngineStep::AppEvent`.
//! `Record` writes to `ctx.syscall.record_buffer`; `IncrMetric` bumps
//! `ctx.syscall.counters`.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::{AppEvent, OpError, OpErrorKind};
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

const DOMAIN: &str = "ai.bytesandbrains.syscall";

fn read_name(node: &NodeProto) -> String {
    node.attribute
        .iter()
        .find(|a| a.name == "name")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default()
}

/// Marker struct for dispatch_table TypeId keying.
pub struct AppEmitOp;

/// `AppEmit(value, name: string) → Sink`. Expects a `BytesValue`
/// input - the emitter is a byte-level surface, so callers wrap
/// their payload in `BytesValue` upstream.
///
/// The user-supplied `name` is validated against the framework's
/// reserved-topic prefixes (`bb.`, `ai.bytesandbrains.`) via
/// [`AppEvent::emit`] — a collision surfaces as
/// `OpError::BadInput` so the engine doesn't ferry the impersonating
/// publish to subscribers.
pub fn invoke_app_emit(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let name = read_name(node);
    let value_bytes =
        crate::syscalls::first_input_optional_bytes("AppEmit", inputs)?.unwrap_or_default();
    let event = AppEvent::emit(name, value_bytes).map_err(|e| OpError {
        kind: OpErrorKind::BadInput,
        reason: "reserved_topic_prefix",
        detail: e.to_string(),
    })?;
    ctx.syscall.pending_app_events.push(event);
    Ok(DispatchResult::Immediate(vec![]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct AppNotifyOp;

/// `AppNotify(trigger, name: string) → Sink`. See [`invoke_app_emit`]
/// for the reserved-prefix validation rule that applies here too.
pub fn invoke_app_notify(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let name = read_name(node);
    let event = AppEvent::notify(name).map_err(|e| OpError {
        kind: OpErrorKind::BadInput,
        reason: "reserved_topic_prefix",
        detail: e.to_string(),
    })?;
    ctx.syscall.pending_app_events.push(event);
    Ok(DispatchResult::Immediate(vec![]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct RecordOp;

/// `Record(value, name: string) → Sink`. Expects a `BytesValue`
/// input - the record buffer stores byte payloads.
pub fn invoke_record(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let name = read_name(node);
    let bytes = crate::syscalls::first_input_optional_bytes("Record", inputs)?.unwrap_or_default();
    ctx.syscall.record_buffer.record(&name, bytes);
    Ok(DispatchResult::Immediate(vec![]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct IncrMetricOp;

/// `IncrMetric(trigger, name: string, delta: int) → Sink`.
pub fn invoke_incr_metric(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let name = read_name(node);
    let delta = node
        .attribute
        .iter()
        .find(|a| a.name == "delta")
        .map(|a| a.i)
        .unwrap_or(1) as u64;
    *ctx.syscall.counters.entry(name).or_insert(0) += delta;
    Ok(DispatchResult::Immediate(vec![]))
}

/// Linker-anchor - see `bb_ops::link_force` for details.
pub fn link_force() {
    use std::hint::black_box;
    black_box(invoke_app_emit as usize);
    black_box(invoke_app_notify as usize);
    black_box(invoke_record as usize);
    black_box(invoke_incr_metric as usize);
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use bb_runtime::registry::OpRegistration as _BbOpsSyscallReg;

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "AppEmit",
        invoke: invoke_app_emit,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "AppNotify",
        invoke: invoke_app_notify,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Record",
        invoke: invoke_record,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "IncrMetric",
        invoke: invoke_incr_metric,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}
