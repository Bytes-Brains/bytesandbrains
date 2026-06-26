//! `composite.Bundle` - pack N typed slot values into one
//! [`CompositeValue`].
//!
//! Variable-arity input convention: the DSL recorder names the i'th
//! payload `child_{i}` (positional naming because input ordering carries
//! the semantics, not the names). The op clones each input via
//! `SlotValue::clone_boxed` into the envelope, preserving the concrete
//! carrier type for in-process Unbundle. Wire-boundary encoding is
//! deferred to `CompositeValue::to_wire_bytes`.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::CompositeValue;

/// `(domain, op_type)` registration key. Shared with `TYPE_COMPOSITE`'s
/// canonical denotation root.
pub const DOMAIN: &str = "ai.bytesandbrains.composite";
/// Op type name.
pub const OP_TYPE: &str = "Bundle";
/// Output port carrying the assembled [`CompositeValue`].
pub const PORT_BUNDLE: &str = "bundle";

/// Invoke fn - clone each `child_{i}` input into the envelope via
/// `SlotValue::clone_boxed` and emit a single `CompositeValue` on the
/// `bundle` output port. Bind-site port ordering survives because the
/// DSL recorder stamps inputs in positional order; the op preserves
/// that order in the resulting `Vec`.
pub fn invoke(
    _node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    if inputs.is_empty() {
        return Err(OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "bundle_no_children",
            detail: "composite.Bundle: at least one child input required".into(),
        });
    }
    let mut children: Vec<Box<dyn SlotValue>> = Vec::with_capacity(inputs.len());
    for (_slot_name, value) in inputs {
        children.push(value.clone_boxed());
    }
    Ok(DispatchResult::Immediate(vec![(
        PORT_BUNDLE.to_string(),
        Box::new(CompositeValue { children }) as Box<dyn SlotValue>,
    )]))
}


bb_derive::register_op! {
    domain: "ai.bytesandbrains.composite",
    op_type: "Bundle",
    invoke: invoke,
}
