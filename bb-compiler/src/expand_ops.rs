//! `expand_ops` — materialize op-variant choices.
//!
//! Each `(domain, op_type)` that needs expansion has a matching arm
//! in `lookup_expansion` returning the `ExpandFn` to apply. Plain
//! `match` because the compiler runs at build time on a single
//! thread — a table behind a sync primitive would be overkill, and
//! a `match` makes the catalog trivially auditable. All expansions
//! stamp `EXPANDED_KEY = "true"` for idempotence.

use crate::error::CompileError;
use bb_ir::proto::onnx::{
    attribute_proto::AttributeType, AttributeProto, GraphProto, NodeProto, StringStringEntryProto,
};

/// Idempotence stamp key.
pub const EXPANDED_KEY: &str = "ai.bytesandbrains.expanded";

const SYSCALL_DOMAIN: &str = "ai.bytesandbrains.syscall";

/// Default Interval period when the attribute is missing
/// (1 second in nanoseconds).
const INTERVAL_DEFAULT_PERIOD_NS: i64 = 1_000_000_000;

/// Per-op expansion function. Mutates the node in place; returns
/// `Err` only on malformed input the compiler can't recover from.
pub type ExpandFn = fn(&mut NodeProto) -> Result<(), CompileError>;

/// Resolve a `(domain, op_type)` to its expansion function, or
/// `None` when no expansion applies (most ops fall here - the
/// pass is a no-op for them).
fn lookup_expansion(domain: &str, op_type: &str) -> Option<ExpandFn> {
    match (domain, op_type) {
        (SYSCALL_DOMAIN, "Interval") => Some(expand_interval),
        (SYSCALL_DOMAIN, "Constant") => Some(expand_constant),
        _ => None,
    }
}

/// Expand ops in-place per the static expansion registry. Pure.
pub fn expand_ops(graph: &mut GraphProto) -> Result<(), CompileError> {
    for node in graph.node.iter_mut() {
        if metadata_value(node, EXPANDED_KEY).is_some() {
            continue;
        }
        let Some(expand_fn) = lookup_expansion(&node.domain, &node.op_type) else {
            continue;
        };
        expand_fn(node)?;
        set_metadata(&mut node.metadata_props, EXPANDED_KEY, "true");
    }
    Ok(())
}

fn expand_interval(node: &mut NodeProto) -> Result<(), CompileError> {
    if node.attribute.iter().any(|a| a.name == "period_ns") {
        return Ok(());
    }
    node.attribute.push(AttributeProto {
        name: "period_ns".to_string(),
        r#type: AttributeType::Int as i32,
        i: INTERVAL_DEFAULT_PERIOD_NS,
        ..Default::default()
    });
    Ok(())
}

/// `Constant` expansion per COMPILER.md §5.2: every `Constant` node
/// MUST carry a `value` attribute of type `TENSOR`. The expansion
/// validates that requirement so downstream dispatch never sees a
/// mis-shaped `Constant`. Nodes that already carry a non-empty
/// `value` attribute are accepted; everything else is rejected with
/// `CompileError::ExpansionFailed`.
fn expand_constant(node: &mut NodeProto) -> Result<(), CompileError> {
    let value_attr = node.attribute.iter().find(|a| a.name == "value");
    let Some(attr) = value_attr else {
        return Err(CompileError::ExpansionFailed {
            domain: node.domain.clone(),
            op_type: node.op_type.clone(),
            reason: "Constant node is missing the required `value` attribute".into(),
        });
    };
    if attr.r#type != AttributeType::Tensor as i32 {
        return Err(CompileError::ExpansionFailed {
            domain: node.domain.clone(),
            op_type: node.op_type.clone(),
            reason: format!(
                "Constant `value` attribute must be TENSOR (got type tag {})",
                attr.r#type
            ),
        });
    }
    if attr.t.is_none() {
        return Err(CompileError::ExpansionFailed {
            domain: node.domain.clone(),
            op_type: node.op_type.clone(),
            reason: "Constant `value` attribute carries no TensorProto payload".into(),
        });
    }
    Ok(())
}

fn metadata_value(node: &NodeProto, key: &str) -> Option<String> {
    node.metadata_props
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.clone())
}

fn set_metadata(props: &mut Vec<StringStringEntryProto>, key: &str, value: &str) {
    if let Some(existing) = props.iter_mut().find(|p| p.key == key) {
        existing.value = value.to_string();
        return;
    }
    props.push(StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    });
}

