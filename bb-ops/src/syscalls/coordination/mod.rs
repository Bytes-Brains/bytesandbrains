//! Coordination ops - Limit, Any, Gate, Serialize, CorrelateTag,
//! Hold, DeadlineCheck. + .
//!
//! Spec: Sub-B in `docs/IR_AND_DSL.md` §5a.
//!
//! The original 9 ops (Limit/Any/Gate/Serialize/CorrelateTag/Hold)
//! live inline in this file because each body is small and follows
//! the same template. `DeadlineCheck` lives in its own
//! sub-module because it's compiler-inserted  and the
//! compiler pass needs to reference its `OP_TYPE` + attribute name
//! constants - keeping them in their own module gives a clean
//! re-export surface.

pub mod deadline_check;

use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;

/// Linker-anchor - see `bb_ops::link_force` for details.
pub fn link_force() {
    use std::hint::black_box;
    black_box(invoke_limit_acquire as usize);
    black_box(invoke_limit_release as usize);
    black_box(invoke_any as usize);
    black_box(invoke_gate as usize);
    black_box(invoke_serialize_enqueue as usize);
    black_box(invoke_serialize_dequeue as usize);
    black_box(invoke_correlate_tag as usize);
    black_box(invoke_hold_stash as usize);
    black_box(invoke_hold_flush as usize);
    black_box(deadline_check::invoke as usize);
}
use bb_ir::proto::onnx::NodeProto;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::{BytesValue, CorrelationTokenValue, TriggerValue};

const DOMAIN: &str = "ai.bytesandbrains.syscall";

// --- Limit.Acquire / Limit.Release ---------------------------------

/// Marker struct for dispatch_table TypeId keying.
pub struct LimitAcquireOp;

/// `Limit.Acquire(trigger, name: string, n: int) → Trigger | Sink`.
pub fn invoke_limit_acquire(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let name = node
        .attribute
        .iter()
        .find(|a| a.name == "name")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    let n = node
        .attribute
        .iter()
        .find(|a| a.name == "n")
        .map(|a| a.i as u32)
        .unwrap_or(1);
    if ctx.peers.gate.acquire(&name, n) {
        Ok(DispatchResult::Immediate(vec![(
            "trigger".to_string(),
            Box::new(TriggerValue),
        )]))
    } else {
        Ok(DispatchResult::Immediate(vec![]))
    }
}

/// Marker struct for dispatch_table TypeId keying.
pub struct LimitReleaseOp;

/// `Limit.Release(trigger, name: string) → Sink`.
pub fn invoke_limit_release(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let name = node
        .attribute
        .iter()
        .find(|a| a.name == "name")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    ctx.peers.gate.release(&name);
    Ok(DispatchResult::Immediate(vec![]))
}

// --- Any -----------------------------------------------------------

/// Marker struct for dispatch_table TypeId keying.
pub struct AnyOp;

/// `Any(inputs: variadic, group: string) → value` per IR_AND_DSL.md
/// §5a. First-arrival semantics: the first input to arrive in a
/// group emits its value; subsequent arrivals within the same
/// `group` are absorbed (Immediate `vec![]`) so downstream
/// consumers don't re-fire.
pub fn invoke_any(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let Some((_, first)) = inputs.first() else {
        return Ok(DispatchResult::Immediate(vec![]));
    };
    let group = node
        .attribute
        .iter()
        .find(|a| a.name == "group")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    // Empty group attribute → degrade to legacy "always fire" mode
    // for tests that don't supply a `group` attr. Non-empty: gate
    // on the latch.
    if !group.is_empty() && !ctx.syscall.any_fired_groups.insert(group) {
        // Already fired this group - absorb.
        return Ok(DispatchResult::Immediate(vec![]));
    }
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        first.clone_boxed(),
    )]))
}

// --- Gate ----------------------------------------------------------

/// Marker struct for dispatch_table TypeId keying.
pub struct GateOp;

/// `Gate(value, trigger) → value`. Releases the value once trigger
/// arrives (both inputs are required by all_inputs_ready before
/// the engine invokes us).
pub fn invoke_gate(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let Some((_, value)) = inputs.first() else {
        return Err(OpError {
            detail: "Gate requires value input".to_string(),
            ..Default::default()
        });
    };
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        value.clone_boxed(),
    )]))
}

// --- Serialize.Enqueue / Dequeue -----------------------------------

/// Marker struct for dispatch_table TypeId keying.
pub struct SerializeEnqueueOp;

/// `Serialize.Enqueue(value, queue: string) → Trigger`.
pub fn invoke_serialize_enqueue(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let queue = node
        .attribute
        .iter()
        .find(|a| a.name == "queue")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    let bytes = crate::syscalls::first_input_optional_bytes("Serialize.Enqueue", inputs)?
        .unwrap_or_default();
    ctx.syscall.serialize_queue.enqueue(&queue, bytes);
    Ok(DispatchResult::Immediate(vec![(
        "trigger".to_string(),
        Box::new(TriggerValue),
    )]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct SerializeDequeueOp;

/// `Serialize.Dequeue(trigger, queue: string) → value`.
pub fn invoke_serialize_dequeue(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let queue = node
        .attribute
        .iter()
        .find(|a| a.name == "queue")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    let Some(bytes) = ctx.syscall.serialize_queue.dequeue(&queue) else {
        return Ok(DispatchResult::Immediate(vec![]));
    };
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        Box::new(BytesValue(bytes)),
    )]))
}

// --- CorrelateTag --------------------------------------------------

/// Marker struct for dispatch_table TypeId keying.
pub struct CorrelateTagOp;

/// `CorrelateTag(trigger) → token`.
pub fn invoke_correlate_tag(
    _node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let token = ctx.net.requests.mint_token();
    Ok(DispatchResult::Immediate(vec![(
        "token".to_string(),
        Box::new(CorrelationTokenValue(token.as_u64())),
    )]))
}

// --- Hold.Stash / Hold.Flush ---------------------------------------

/// Marker struct for dispatch_table TypeId keying.
pub struct HoldStashOp;

/// `Hold.Stash(value, slot: string) → Sink`.
pub fn invoke_hold_stash(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let slot = node
        .attribute
        .iter()
        .find(|a| a.name == "slot")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    let bytes =
        crate::syscalls::first_input_optional_bytes("Hold.Stash", inputs)?.unwrap_or_default();
    ctx.syscall.hold_table.stash(&slot, bytes);
    Ok(DispatchResult::Immediate(vec![]))
}

/// Marker struct for dispatch_table TypeId keying.
pub struct HoldFlushOp;

/// `Hold.Flush(trigger, slot: string) → value`.
pub fn invoke_hold_flush(
    node: &NodeProto,
    _inputs: &[(&str, &dyn SlotValue)],
    ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let slot = node
        .attribute
        .iter()
        .find(|a| a.name == "slot")
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_default();
    let Some(bytes) = ctx.syscall.hold_table.flush(&slot) else {
        return Ok(DispatchResult::Immediate(vec![]));
    };
    Ok(DispatchResult::Immediate(vec![(
        "value".to_string(),
        Box::new(BytesValue(bytes)),
    )]))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

use bb_runtime::registry::OpRegistration as _BbOpsSyscallReg;

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Limit.Acquire",
        invoke: invoke_limit_acquire,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Limit.Release",
        invoke: invoke_limit_release,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Any",
        invoke: invoke_any,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Gate",
        invoke: invoke_gate,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Serialize.Enqueue",
        invoke: invoke_serialize_enqueue,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Serialize.Dequeue",
        invoke: invoke_serialize_dequeue,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "CorrelateTag",
        invoke: invoke_correlate_tag,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Hold.Stash",
        invoke: invoke_hold_stash,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}

inventory::submit! {
    _BbOpsSyscallReg {
        domain: DOMAIN,
        op_type: "Hold.Flush",
        invoke: invoke_hold_flush,
        kind: bb_runtime::registry::RegistrationKind::Syscall,
    }
}
