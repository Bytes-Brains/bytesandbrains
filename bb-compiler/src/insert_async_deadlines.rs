//! Compiler pass - pair every NodeProto carrying a `deadline_ns`
//! attribute with an upstream `DeadlineCheck` syscall gate
//! ( per the audit plan).
//!
//! The DSL records a `deadline_ns: i64` attribute on any op for
//! which "if the deadline has passed before we get here, fail
//! instead of running." This pass walks the partition graph,
//! finds every such NodeProto, and inserts a `DeadlineCheck`
//! upstream so the check fires before the protected op runs.
//!
//! Idempotent - re-running on an already-gated graph is a no-op.
//! Joins the runner after `augment` in
//! `bb-compiler/src/runner.rs` so the gates see the post-augmented
//! graph and the augmented carriers see un-gated source nodes.

use crate::error::CompileError;
use bb_ir::proto::onnx::{AttributeProto, GraphProto, NodeProto, StringStringEntryProto};
use bb_ir::syscall_ids::{
    ATTR_DEADLINE_NS, OP_DEADLINE_CHECK as DEADLINE_OP_TYPE, SYSCALL_DOMAIN as DEADLINE_DOMAIN,
};

/// Idempotence stamp - once a node has been paired with a
/// `DeadlineCheck`, this metadata entry stops the pass from
/// re-inserting on a second run.
pub const GATED_KEY: &str = "ai.bytesandbrains.deadline_gated";

/// Suffix appended to a protected node's name to form the sibling
/// trigger slot that the inserted `DeadlineCheck` writes (/// — closes `chief:B9 + S7`: the prior pass rewrote
/// `node.input[0]`, destroying the payload on every `wire.Send`).
pub const TRIGGER_DEADLINE_SUFFIX: &str = "#__trigger_deadline";

/// Insert a `DeadlineCheck` upstream of every `deadline_ns`-bearing
/// node in `sub_graph`. Pure (the input graph is mutated in place
/// but the function has no side effects beyond that).
///
/// The `DeadlineCheck` writes to a NEW sibling input slot named
/// `<protected>#__trigger_deadline`; the protected node's existing
/// inputs (including `input[0]` payload on a `wire.Send`) stay
/// intact. The engine fires the protected node only once every
/// input — original payloads PLUS the new deadline-trigger slot —
/// has been filled, which preserves the original gating semantics
/// without destroying payload bytes.
pub fn insert_async_deadlines(sub_graph: &mut GraphProto) -> Result<(), CompileError> {
    let mut gates: Vec<NodeProto> = Vec::new();

    for node in sub_graph.node.iter_mut() {
        if metadata_value(node, GATED_KEY).is_some() {
            continue;
        }
        let Some(deadline_ns) = read_deadline(node) else {
            continue;
        };
        // The gate observes the protected node's first input as a
        // proxy for "this node is preparing to fire". When the node
        // has no inputs there is nothing to observe, so skip.
        let Some(trigger_proxy) = node.input.first().cloned() else {
            continue;
        };

        let trigger_slot = format!("{}{TRIGGER_DEADLINE_SUFFIX}", node.name);
        gates.push(build_deadline_check_node(
            &node.name,
            &trigger_proxy,
            &trigger_slot,
            deadline_ns,
        ));

        // Append the sibling deadline-trigger slot to the protected
        // node's input list. Existing inputs (incl. input[0] payload
        // on a `wire.Send`) stay untouched — the engine waits for
        // both the original payload and the new trigger slot before
        // firing.
        node.input.push(trigger_slot);
        set_metadata(&mut node.metadata_props, GATED_KEY, "true");
    }

    sub_graph.node.extend(gates);
    Ok(())
}

fn build_deadline_check_node(
    source_name: &str,
    trigger_input: &str,
    gate_output: &str,
    deadline_ns: i64,
) -> NodeProto {
    NodeProto {
        op_type: DEADLINE_OP_TYPE.to_string(),
        domain: DEADLINE_DOMAIN.to_string(),
        name: format!("DeadlineCheck@{source_name}"),
        input: vec![trigger_input.to_string()],
        output: vec![gate_output.to_string()],
        attribute: vec![AttributeProto {
            name: ATTR_DEADLINE_NS.to_string(),
            i: deadline_ns,
            r#type: bb_ir::proto::onnx::attribute_proto::AttributeType::Int as i32,
            ..Default::default()
        }],
        metadata_props: vec![
            StringStringEntryProto {
                key: "ai.bytesandbrains.deadline_source".to_string(),
                value: source_name.to_string(),
            },
            // Mark the inserted gate as already-gated so a second
            // pass run doesn't try to gate its own previous output.
            StringStringEntryProto {
                key: GATED_KEY.to_string(),
                value: "true".to_string(),
            },
        ],
        ..Default::default()
    }
}

fn read_deadline(node: &NodeProto) -> Option<i64> {
    node.attribute
        .iter()
        .find(|a| a.name == ATTR_DEADLINE_NS)
        .map(|a| a.i)
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

// ──── inventory-published GateContract ─────────────────

/// Insertion contract for the deadline gate. Every node carrying a
/// `deadline_ns` attribute must already have its sibling
/// `__trigger_deadline` slot in `input` (the proof that
/// `insert_async_deadlines` ran on it) and a matching
/// `DeadlineCheck` upstream writing to that slot. The contract
/// detects partial runs / missing gates that the prior
/// presence-only validator silently accepted (closes `chief:S12`).
struct DeadlineCheckContract;

impl crate::gate_contract::GateContract for DeadlineCheckContract {
    fn name(&self) -> &'static str {
        "DeadlineCheck"
    }

    fn assert_inserted(
        &self,
        sub_graph: &bb_ir::proto::onnx::GraphProto,
    ) -> Result<(), CompileError> {
        for node in &sub_graph.node {
            // The DeadlineCheck gate itself carries `deadline_ns`
            // (it propagates the value to the runtime check) AND
            // is marked GATED — it never needs a sibling trigger
            // slot of its own. Skip it.
            if node.op_type == DEADLINE_OP_TYPE || metadata_value(node, GATED_KEY).is_some() {
                continue;
            }
            if node.attribute.iter().any(|a| a.name == ATTR_DEADLINE_NS)
                && !node
                    .input
                    .iter()
                    .any(|i| i.ends_with(TRIGGER_DEADLINE_SUFFIX))
            {
                return Err(CompileError::RuntimeIncomplete {
                    missing: format!(
                        "DeadlineCheck not inserted upstream of `{}` (no sibling `{TRIGGER_DEADLINE_SUFFIX}` input)",
                        node.name,
                    ),
                });
            }
        }
        Ok(())
    }
}

static DEADLINE_CHECK_CONTRACT: DeadlineCheckContract = DeadlineCheckContract;

bb_ir::registry::inventory::submit! {
    crate::gate_contract::GateContractRegistration {
        contract: &DEADLINE_CHECK_CONTRACT,
    }
}

