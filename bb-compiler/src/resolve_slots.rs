//! Pass 11 - `resolve_slots` + pre-flight. Final check before a
//! ModelProto is emitted
//!
//! implements **only the AmbiguousRole detection** per
//! `docs/internal/IMPLEMENTATION_PLAN.md` :
//!
//! For each role domain (`ai.bytesandbrains.role.<role>`), collect
//! the `concrete_type` providers AND the `(required_trait, slot_id)`
//! providers visible in the NodeProtos. If BOTH kinds appear under
//! the same role → reject with `CompileError::AmbiguousRole`.
//!
//! The runner (`runner.rs`) maps the `AmbiguousRole` variant to
//! `BuildError::AmbiguousRole` for the public Module::build() surface
//! per the plan.
//!
//! Other §14 checks (slot binding existence, decoder availability,
//! opset coverage) require bindings + wire types;
//! they're deferred.

use std::collections::BTreeMap;

use crate::error::CompileError;
use bb_ir::proto::onnx::FunctionProto;

/// Resolve slots + run pre-flight. Pure per ANALYSIS.md §3.2.
pub fn resolve_slots(function: &FunctionProto) -> Result<(), CompileError> {
    // Per role: collect distinct concrete_type providers + distinct
    // (required_trait, slot_id) generic providers.
    let mut role_providers: BTreeMap<String, RoleProviders> = BTreeMap::new();

    for node in &function.node {
        if !node.domain.starts_with("ai.bytesandbrains.role.") {
            continue;
        }
        let providers = role_providers.entry(node.domain.clone()).or_default();
        if let Some(concrete) = meta_value(node, "ai.bytesandbrains.concrete_type") {
            providers.concrete_types.insert(concrete.to_string());
        }
        if let (Some(rt), Some(sid)) = (
            meta_value(node, "ai.bytesandbrains.required_trait"),
            meta_value(node, "ai.bytesandbrains.slot_id"),
        ) {
            if let Ok(id) = sid.parse::<u32>() {
                providers.generic_slots.insert(id, rt.to_string());
            }
        }
    }

    for (role, providers) in role_providers {
        if !providers.concrete_types.is_empty() && !providers.generic_slots.is_empty() {
            // First concrete + first generic slot are surfaced as
            // canonical witnesses. The pass is deterministic because
            // BTreeMap iteration is ordered.
            let concrete_type = providers
                .concrete_types
                .iter()
                .next()
                .cloned()
                .unwrap_or_default();
            let (slot_id, _trait_name) = providers
                .generic_slots
                .iter()
                .next()
                .map(|(k, v)| (*k, v.clone()))
                .unwrap_or_default();
            return Err(CompileError::AmbiguousRole {
                role,
                concrete_type,
                generic_slot_id: slot_id,
            });
        }
    }

    Ok(())
}

#[derive(Default)]
struct RoleProviders {
    concrete_types: std::collections::BTreeSet<String>,
    generic_slots: BTreeMap<u32, String>,
}

fn meta_value<'a>(node: &'a bb_ir::proto::onnx::NodeProto, key: &str) -> Option<&'a str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == key)
        .map(|p| p.value.as_str())
}

