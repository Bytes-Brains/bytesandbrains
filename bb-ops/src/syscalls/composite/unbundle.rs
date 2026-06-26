//! `composite.Unbundle` - decompose a [`CompositeValue`] into N
//! per-child outputs.
//!
//! Reads two attributes the DSL recorder stamps:
//!
//! - `ai.bytesandbrains.composite.child_count` (INT) - declared number
//!   of child slots, validated against the incoming envelope's length.
//! - `ai.bytesandbrains.composite.child_types` (STRING) - comma-separated
//!   TypeNode denotations naming each child's declared output type. The
//!   compiler's TypeSolver narrows downstream consumer inputs through
//!   these denotations regardless of the in-process carrier shape.
//!
//! Each output port is named `child_{i}` to match the `Bundle` input
//! convention. The op emits each child as its original concrete
//! `SlotValue` carrier (`clone_boxed` of the stored child) - downstream
//! consumers downcast directly to `T` instead of bincode-decoding a
//! `BytesValue` against the declared denotation.

use bb_ir::proto::onnx::NodeProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;
use bb_runtime::syscall::values::CompositeValue;

/// `(domain, op_type)` registration key.
pub const DOMAIN: &str = "ai.bytesandbrains.composite";
/// Op type name.
pub const OP_TYPE: &str = "Unbundle";
/// Input port carrying the [`CompositeValue`].
pub const PORT_BUNDLE: &str = "bundle";
/// Attribute name (INT) declaring the expected child count.
pub const ATTR_CHILD_COUNT: &str = "ai.bytesandbrains.composite.child_count";
/// Attribute name (STRING) carrying the comma-separated TypeNode
/// denotations per child.
pub const ATTR_CHILD_TYPES: &str = "ai.bytesandbrains.composite.child_types";

/// Invoke fn - validate the incoming [`CompositeValue`] against the
/// declared child count, emit each typed child on `child_{i}` via
/// `SlotValue::clone_boxed`. Downstream consumers downcast directly
/// to the concrete type the graph contract guarantees at each child
/// site.
pub fn invoke(
    node: &NodeProto,
    inputs: &[(&str, &dyn SlotValue)],
    _ctx: &mut RuntimeResourceRef<'_>,
) -> Result<DispatchResult, OpError> {
    let composite = downcast_composite(inputs)?;
    let declared = declared_child_count(node)?;
    if composite.children.len() != declared {
        return Err(OpError {
            kind: OpErrorKind::ExecutionFailed,
            reason: "unbundle_child_count_mismatch",
            detail: format!(
                "composite.Unbundle: envelope carries {} children, declared {}",
                composite.children.len(),
                declared
            ),
        });
    }
    let mut outs: Vec<(String, Box<dyn SlotValue>)> = Vec::with_capacity(declared);
    for (i, child) in composite.children.iter().enumerate() {
        outs.push((format!("child_{i}"), child.clone_boxed()));
    }
    Ok(DispatchResult::Immediate(outs))
}

fn downcast_composite<'a>(
    inputs: &'a [(&str, &dyn SlotValue)],
) -> Result<&'a CompositeValue, OpError> {
    let (_, value) = inputs
        .iter()
        .find(|(n, _)| *n == PORT_BUNDLE)
        .ok_or_else(|| OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "missing_bundle",
            detail: "composite.Unbundle: required input `bundle` is absent".into(),
        })?;
    value
        .as_any()
        .downcast_ref::<CompositeValue>()
        .ok_or_else(|| OpError {
            kind: OpErrorKind::TypeMismatch,
            reason: "expected_composite",
            detail: "composite.Unbundle: input `bundle` is not a CompositeValue".into(),
        })
}

fn declared_child_count(node: &NodeProto) -> Result<usize, OpError> {
    let attr = node
        .attribute
        .iter()
        .find(|a| a.name == ATTR_CHILD_COUNT)
        .ok_or_else(|| OpError {
            kind: OpErrorKind::MissingSlot,
            reason: "missing_child_count",
            detail: format!("composite.Unbundle: missing `{ATTR_CHILD_COUNT}` attribute"),
        })?;
    if attr.i < 1 {
        return Err(OpError {
            kind: OpErrorKind::ExecutionFailed,
            reason: "child_count_below_one",
            detail: format!(
                "composite.Unbundle: `{ATTR_CHILD_COUNT}` must be >= 1, got {}",
                attr.i
            ),
        });
    }
    Ok(attr.i as usize)
}


bb_derive::register_op! {
    domain: "ai.bytesandbrains.composite",
    op_type: "Unbundle",
    invoke: invoke,
}
