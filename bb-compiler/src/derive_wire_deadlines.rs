//!
//! Stamps each `wire.Send`'s static deadline from the (per_hop_budget_ns ×
//! chain_depth) formula, defaulting to chain_depth = 1 when no
//! `chain_depth` on every target-boundary `wire.Send`. For each
//! such Send, computes the static deadline as
//! `chain_depth * per_hop_budget_ns` and stamps it as a
//! `deadline_ns: i64` attribute on the NodeProto. The existing
//! [`mod@super::insert_async_deadlines`] pass then inserts a
//! `DeadlineCheck` gate upstream so the deadline is enforced at
//! runtime.
//!
//! The per-hop budget comes from the [`crate::Compiler`]'s
//! `per_hop_budget_ns` field (default
//! [`bb_ir::syscall_ids::DEFAULT_PER_HOP_BUDGET_NS`]). Build-time
//! CI matrices that produce deployment bundles with different
//! latency assumptions can override via
//! `Compiler::with_per_hop_budget_ns(ns)`.
//!
//! Runtime override note: the runtime side
//! ([`crate::node::config::NodeConfig::per_hop_budget_ns`]) is the
//! source of truth at delivery time. When `chain_depth` metadata
//! survives on the NodeProto, the engine can replace the static
//! deadline with `chain_depth * NodeConfig.per_hop_budget_ns` at
//! dispatch time. The compiler pass exists so single-Node
//! deployments + tests get a sane baseline deadline without
//! requiring runtime fixup; multi-Node deployments with bespoke
//! latency profiles rely on the runtime override.

use crate::error::CompileError;
use crate::partition_by_wire_ops::WIRE_DOMAIN;
use bb_ir::keys::CHAIN_DEPTH_KEY;
use bb_ir::proto::onnx::{attribute_proto, AttributeProto, ModelProto};
use bb_ir::syscall_ids::ATTR_DEADLINE_NS;

const SEND_OP: &str = "Send";

/// Walk every `wire.Send` NodeProto with `chain_depth` metadata and
/// stamp a `deadline_ns: i64` attribute. Idempotent - re-runs
/// overwrite the previous attribute in place.
///
/// Returns the count of stamps applied.
pub fn derive_wire_deadlines(
    model: &mut ModelProto,
    per_hop_budget_ns: u64,
) -> Result<usize, CompileError> {
    let mut stamp_count = 0usize;
    for func in model.functions.iter_mut() {
        for node in func.node.iter_mut() {
            if node.domain != WIRE_DOMAIN || node.op_type != SEND_OP {
                continue;
            }
            // Missing chain_depth defaults to 1 (single-hop) — covers
            // every wire.Send the compiler hasn't paired with a
            // multi-hop chain. Multi-hop chains stamp explicit
            // chain_depth metadata via a later pass; both branches feed
            // the same `chain_depth * per_hop_budget_ns` formula.
            let chain_depth = read_chain_depth(node).unwrap_or(1);
            let deadline_ns = (chain_depth as i64).saturating_mul(per_hop_budget_ns as i64);
            upsert_deadline(node, deadline_ns);
            stamp_count += 1;
        }
    }
    Ok(stamp_count)
}

fn read_chain_depth(node: &bb_ir::proto::onnx::NodeProto) -> Option<u64> {
    node.metadata_props
        .iter()
        .find(|p| p.key == CHAIN_DEPTH_KEY)
        .and_then(|p| p.value.parse().ok())
}

fn upsert_deadline(node: &mut bb_ir::proto::onnx::NodeProto, deadline_ns: i64) {
    if let Some(existing) = node
        .attribute
        .iter_mut()
        .find(|a| a.name == ATTR_DEADLINE_NS)
    {
        existing.i = deadline_ns;
        return;
    }
    node.attribute.push(AttributeProto {
        name: ATTR_DEADLINE_NS.to_string(),
        i: deadline_ns,
        r#type: attribute_proto::AttributeType::Int as i32,
        ..Default::default()
    });
}

