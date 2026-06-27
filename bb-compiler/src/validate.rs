//! Structural sanity check. Reject malformed input before any
//! other pass mutates it. Pure function over `GraphProto`.
//!
//! Implemented rules:
//!
//! - Rule 1 — op type known (framework reserved opsets + `ai.onnx`).
//! - Rule 2 — inputs reachable.
//! - Rule 3 — outputs unique.
//! - Rule 5 — every graph input has a matching `ValueInfoProto`.
//! - Rule 6 — role-domain NodeProtos carry canonical metadata.
//! - Rule 7 — no cycles.

use std::collections::{HashMap, HashSet};

use crate::error::ValidationError;
use bb_ir::proto::onnx::GraphProto;

/// Validate the recorded graph. Pure.
pub fn validate(graph: &GraphProto) -> Result<(), ValidationError> {
    rule_1_known_op(graph)?;
    rule_2_inputs_reachable(graph)?;
    rule_3_outputs_unique(graph)?;
    rule_5_type_declarations_present(graph)?;
    rule_6_slot_metadata_well_formed(graph)?;
    rule_7_no_cycles(graph)?;
    Ok(())
}

/// Reserved framework opset prefixes
const RESERVED_OPSET_PREFIXES: &[&str] = &["ai.bytesandbrains", "ai.onnx"];

fn is_reserved_opset(domain: &str) -> bool {
    RESERVED_OPSET_PREFIXES
        .iter()
        .any(|p| domain == *p || domain.starts_with(&format!("{p}.")))
}

/// Rule 1 - every `(domain, op_type)` belongs to a known opset.
fn rule_1_known_op(graph: &GraphProto) -> Result<(), ValidationError> {
    for node in &graph.node {
        if !is_reserved_opset(&node.domain) {
            return Err(ValidationError::UnknownOp {
                node_name: node.name.clone(),
                op_type: node.op_type.clone(),
                domain: node.domain.clone(),
            });
        }
    }
    Ok(())
}

/// Rule 2 - every input value name is produced upstream or appears
/// in `graph.input`.
fn rule_2_inputs_reachable(graph: &GraphProto) -> Result<(), ValidationError> {
    let mut produced: HashSet<&str> = HashSet::new();
    for input in &graph.input {
        produced.insert(input.name.as_str());
    }
    // First scan all node outputs so we don't reject forward refs
    // within a DAG-valid topological order before we've collected
    // them - rule 7 separately enforces acyclicity.
    for node in &graph.node {
        for out in &node.output {
            produced.insert(out.as_str());
        }
    }
    for node in &graph.node {
        for inp in &node.input {
            if inp.is_empty() {
                // ONNX permits empty input slots (e.g. optional Conv
                // bias) - skip rather than flag.
                continue;
            }
            if !produced.contains(inp.as_str()) {
                return Err(ValidationError::DanglingInput {
                    node_name: node.name.clone(),
                    input_name: inp.clone(),
                });
            }
        }
    }
    Ok(())
}

/// Rule 3 - every output value name is written by at most one op.
fn rule_3_outputs_unique(graph: &GraphProto) -> Result<(), ValidationError> {
    let mut writers: HashMap<&str, &str> = HashMap::new();
    for node in &graph.node {
        for out in &node.output {
            if out.is_empty() {
                continue;
            }
            if let Some(&prev) = writers.get(out.as_str()) {
                return Err(ValidationError::DuplicateOutput {
                    value_name: out.clone(),
                    node_a: prev.to_string(),
                    node_b: node.name.clone(),
                });
            }
            writers.insert(out.as_str(), node.name.as_str());
        }
    }
    Ok(())
}

/// Rule 5 - every `graph.input` has a matching `ValueInfoProto.type`.
fn rule_5_type_declarations_present(graph: &GraphProto) -> Result<(), ValidationError> {
    for input in &graph.input {
        if input.r#type.is_none() {
            return Err(ValidationError::MissingTypeInfo {
                input_name: input.name.clone(),
            });
        }
    }
    Ok(())
}

/// Rule 6 - every role-domain NodeProto carries the canonical
/// metadata keys.
///
/// For `domain` starting with `"ai.bytesandbrains.role."`:
/// - EITHER `(concrete_type, instance)` BOTH present, OR
/// - `(required_trait, slot_id)` BOTH present.
fn rule_6_slot_metadata_well_formed(graph: &GraphProto) -> Result<(), ValidationError> {
    for node in &graph.node {
        if !node.domain.starts_with("ai.bytesandbrains.role.") {
            continue;
        }
        let has_concrete = meta_has(node, "ai.bytesandbrains.concrete_type")
            && meta_has(node, "ai.bytesandbrains.instance");
        let has_generic = meta_has(node, "ai.bytesandbrains.required_trait")
            && meta_has(node, "ai.bytesandbrains.slot_id");
        if !has_concrete && !has_generic {
            return Err(ValidationError::MalformedSlotMetadata {
                node_name: node.name.clone(),
                detail: format!(
                    "role-domain NodeProto {} lacks both (concrete_type, instance) and (required_trait, slot_id) metadata",
                    node.op_type,
                ),
            });
        }
    }
    Ok(())
}

fn meta_has(node: &bb_ir::proto::onnx::NodeProto, key: &str) -> bool {
    node.metadata_props.iter().any(|p| p.key == key)
}

/// Rule 7 - no cycles. Kahn's algorithm over the producer-consumer
/// DAG.
fn rule_7_no_cycles(graph: &GraphProto) -> Result<(), ValidationError> {
    // Build producer map: value_name → producing NodeProto index.
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (idx, node) in graph.node.iter().enumerate() {
        for out in &node.output {
            producer.insert(out.as_str(), idx);
        }
    }
    // Build edges: for each node, find each input's producing node →
    // adjacency.
    let n = graph.node.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (idx, node) in graph.node.iter().enumerate() {
        for inp in &node.input {
            if let Some(&p) = producer.get(inp.as_str()) {
                if p != idx {
                    adj[p].push(idx);
                    in_degree[idx] += 1;
                }
            }
        }
    }
    // Kahn's: drain zero-in-degree nodes.
    let mut queue: std::collections::VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, d)| **d == 0)
        .map(|(i, _)| i)
        .collect();
    let mut visited = 0;
    while let Some(idx) = queue.pop_front() {
        visited += 1;
        for &next in &adj[idx] {
            in_degree[next] -= 1;
            if in_degree[next] == 0 {
                queue.push_back(next);
            }
        }
    }
    if visited != n {
        let involves: Vec<String> = graph
            .node
            .iter()
            .enumerate()
            .filter(|(i, _)| in_degree[*i] > 0)
            .map(|(_, n)| n.name.clone())
            .collect();
        return Err(ValidationError::CyclicGraph { involves });
    }
    Ok(())
}

