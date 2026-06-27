//! `resolve_component_dependencies` pass.
//!
//! Walks every bound concrete in a [`CompiledArtifact`]'s
//! `BindingSpec`, reads the concrete's declared
//! [`DependencyDecl`]s through the inventory carrier, and verifies
//! each required slot is bound to a concrete whose role matches the
//! dependency's required role. On success, every NodeProto
//! contributed by a concrete (identified by its `concrete_type`
//! metadata) gets its declared deps stamped via
//! [`bb_ir::keys::stamp_dependency_metadata`] so downstream passes +
//! the runtime can recover the wiring from the IR alone.
//!
//! Surfaces:
//! - [`CompileError::UnboundDependency`] when the required slot has
//!   no binding.
//! - [`CompileError::DependencyRoleMismatch`] when the slot is bound
//!   to a concrete whose role set does not include the required role.

use bb_ir::component::DependencyDecl;
use bb_ir::keys::{stamp_dependency_metadata, CONCRETE_TYPE_KEY};
use bb_ir::proto::onnx::ModelProto;
use bb_ir::registry::find_concrete_component;

use crate::artifact::{BindingSlot, BindingSpec};
use crate::error::CompileError;

/// Run the pass over the artifact's spec + IR. On success, mutates
/// every concrete-bearing NodeProto in `models` to carry its
/// declared dep metadata.
pub(crate) fn resolve_component_dependencies(
    spec: &BindingSpec,
    models: &mut [ModelProto],
) -> Result<(), CompileError> {
    for slot in &spec.slots {
        let concrete_type = slot.concrete_type_name.as_str();
        if concrete_type.is_empty() {
            // Generic placeholder slot — no concrete bound, no deps
            // to verify. The Compiler::bind chain in T8 fills these.
            continue;
        }
        let entry = match find_concrete_component(concrete_type) {
            Some(e) => e,
            None => {
                // Concrete isn't in this binary's inventory —
                // `validate_runtime_complete` surfaces it.
                // Skip so this pass stays a pure dep-graph check.
                continue;
            }
        };
        verify_deps(slot, entry.dependencies, spec)?;
    }

    stamp_dep_metadata_across_models(spec, models);
    Ok(())
}

fn verify_deps(
    component_slot: &BindingSlot,
    deps: &[DependencyDecl],
    spec: &BindingSpec,
) -> Result<(), CompileError> {
    for dep in deps {
        let target = spec
            .get(dep.slot)
            .ok_or_else(|| CompileError::UnboundDependency {
                component: component_slot.concrete_type_name.clone(),
                bound_at_slot: component_slot.slot.clone(),
                required_role: dep.role.to_string(),
                required_slot: dep.slot.to_string(),
            })?;
        if !role_matches(&target.role, dep.role) {
            return Err(CompileError::DependencyRoleMismatch {
                component: component_slot.concrete_type_name.clone(),
                bound_at_slot: component_slot.slot.clone(),
                required_role: dep.role.to_string(),
                required_slot: dep.slot.to_string(),
                provided_role: target.role.clone(),
            });
        }
    }
    Ok(())
}

/// `BindingSlot.role` historically uses the engine-side trait name
/// (e.g. `"BackendRuntime"`); `DependencyDecl.role` uses the
/// canonical Contract role name (e.g. `"Backend"`). Normalize both
/// to the bare PascalCase role identifier before comparison.
fn role_matches(provided: &str, required: &str) -> bool {
    normalize_role(provided) == normalize_role(required)
}

fn normalize_role(role: &str) -> &str {
    role.strip_suffix("Runtime").unwrap_or(role)
}

fn stamp_dep_metadata_across_models(spec: &BindingSpec, models: &mut [ModelProto]) {
    for model in models {
        // Walk the graph + each function for any NodeProto whose
        // `concrete_type` metadata names a bound concrete; stamp
        // the concrete's declared deps onto its metadata_props.
        if let Some(graph) = model.graph.as_mut() {
            for node in &mut graph.node {
                stamp_for_node(spec, node);
            }
        }
        for function in &mut model.functions {
            for node in &mut function.node {
                stamp_for_node(spec, node);
            }
        }
    }
}

fn stamp_for_node(spec: &BindingSpec, node: &mut bb_ir::proto::onnx::NodeProto) {
    let Some(concrete_type) = node
        .metadata_props
        .iter()
        .find(|e| e.key == CONCRETE_TYPE_KEY)
        .map(|e| e.value.clone())
    else {
        return;
    };
    let Some(_bound_slot) = spec
        .slots
        .iter()
        .find(|s| s.concrete_type_name == concrete_type)
    else {
        return;
    };
    let Some(entry) = find_concrete_component(&concrete_type) else {
        return;
    };
    if entry.dependencies.is_empty() {
        return;
    }
    stamp_dependency_metadata(node, entry.dependencies);
}

