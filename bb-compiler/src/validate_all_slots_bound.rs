//! `validate_all_slots_bound` — Phase B of the chosen-path install
//! migration. Walks the artifact's `BindingSpec` + IR + every
//! bound concrete's `DEPENDENCIES`; surfaces a typed
//! [`CompileError::UnboundSlot`] when the compiler hasn't been given
//! enough information to construct every component the runtime
//! needs.
//!
//! Three classes of unbound slot the pass catches:
//!
//! 1. **Direct placeholder unbound** — the IR carries NodeProtos
//!    stamped with `(required_trait, slot_id)` for a role the user
//!    never `.bind_<role>::<T>("…")`'d. The Module body uses a
//!    placeholder of that role but the bind chain doesn't supply
//!    one. Surfaces [`SlotSource::DirectPlaceholder`].
//!
//! 2. **Dependency unbound** — a bound concrete declares
//!    `#[depends(<role> = "<slot>")]` for a slot the bind chain
//!    didn't include. The chain says `.bind_index::<CountingIndex>("primary_index")`
//!    and `CountingIndex` declares `#[depends(backend = "compute")]`
//!    but no `.bind_backend::<X>("compute")` follows. Surfaces
//!    [`SlotSource::DependencyOf`].
//!
//! 3. **Dependency role mismatch** — handled by the prior
//!    `resolve_component_dependencies` pass via
//!    [`CompileError::DependencyRoleMismatch`]. This pass complements
//!    that one without duplicating its work.
//!
//! Runs after `resolve_component_dependencies` so dep stamping has
//! already happened; lets the install path treat
//! `binding_spec.slots` as the complete inventory of slots it must
//! construct.

use std::collections::HashSet;

use bb_ir::proto::onnx::ModelProto;
use bb_ir::registry::find_concrete_component;

use crate::artifact::BindingSpec;
use crate::error::{CompileError, SlotSource};

/// Stable PascalCase role names — used to map the recorder's
/// `required_trait` metadata (e.g. `"BackendRuntime"`) back to the
/// canonical Contract role identifier (`"Backend"`). The
/// `<Role>Runtime` → `<Role>` strip mirrors the convention
/// `resolve_component_dependencies::normalize_role` uses.
fn canonical_role(required_trait: &str) -> &str {
    required_trait
        .strip_suffix("Runtime")
        .unwrap_or(required_trait)
}

/// Stable IR metadata key the recorder stamps on every NodeProto
/// emitted via a generic placeholder's DSL method. Mirrors the
/// constant the placeholder bodies use in `bb-ops/src/placeholders`.
const REQUIRED_TRAIT_KEY: &str = "ai.bytesandbrains.required_trait";

/// Run the pass. Returns `Ok(())` when every role referenced by the
/// IR has at least one matching `BindingSlot` AND every bound
/// concrete's declared dependencies are themselves bound to a
/// matching role.
pub(crate) fn validate_all_slots_bound(
    spec: &BindingSpec,
    models: &[ModelProto],
) -> Result<(), CompileError> {
    validate_direct_placeholders(spec, models)?;
    validate_dependency_slots(spec)?;
    Ok(())
}

/// Walk every NodeProto across every function; collect distinct
/// `required_trait` strings; for each, ensure
/// `BindingSpec.slots` carries at least one entry of that role with
/// a non-empty `concrete_type_name`.
fn validate_direct_placeholders(
    spec: &BindingSpec,
    models: &[ModelProto],
) -> Result<(), CompileError> {
    let bound_roles: HashSet<String> = spec
        .slots
        .iter()
        .filter(|s| !s.concrete_type_name.is_empty())
        .map(|s| canonical_role(&s.role).to_string())
        .collect();

    let mut required_roles: Vec<String> = Vec::new();
    for model in models {
        if let Some(graph) = &model.graph {
            for node in &graph.node {
                if let Some(role) = required_trait_of_node(node) {
                    let canon = canonical_role(role).to_string();
                    if !required_roles.contains(&canon) {
                        required_roles.push(canon);
                    }
                }
            }
        }
        for function in &model.functions {
            for node in &function.node {
                if let Some(role) = required_trait_of_node(node) {
                    let canon = canonical_role(role).to_string();
                    if !required_roles.contains(&canon) {
                        required_roles.push(canon);
                    }
                }
            }
        }
    }

    for role in required_roles {
        if !bound_roles.contains(&role) {
            return Err(CompileError::UnboundSlot {
                role,
                source: SlotSource::DirectPlaceholder,
            });
        }
    }
    Ok(())
}

/// Walk every bound concrete; for each declared dep, ensure
/// `BindingSpec.slots` has an entry at `dep.slot` with the right
/// role. Complements `resolve_component_dependencies` by surfacing
/// the new `UnboundSlot { source: DependencyOf }` shape (the
/// existing pass keeps emitting `UnboundDependency` for the
/// migration window).
fn validate_dependency_slots(spec: &BindingSpec) -> Result<(), CompileError> {
    for slot in &spec.slots {
        if slot.concrete_type_name.is_empty() {
            continue;
        }
        let Some(entry) = find_concrete_component(&slot.concrete_type_name) else {
            // The earlier `validate_runtime_complete` pass surfaces
            // unregistered concretes as `RuntimeIncomplete`; skip
            // here so the dep pass remains a pure dep-graph check.
            continue;
        };
        for dep in entry.dependencies {
            let Some(target) = spec.get(dep.slot) else {
                return Err(CompileError::UnboundSlot {
                    role: dep.role.to_string(),
                    source: SlotSource::DependencyOf {
                        component: slot.concrete_type_name.clone(),
                        bound_at_slot: slot.slot.clone(),
                        required_slot: dep.slot.to_string(),
                    },
                });
            };
            if target.concrete_type_name.is_empty() {
                return Err(CompileError::UnboundSlot {
                    role: dep.role.to_string(),
                    source: SlotSource::DependencyOf {
                        component: slot.concrete_type_name.clone(),
                        bound_at_slot: slot.slot.clone(),
                        required_slot: dep.slot.to_string(),
                    },
                });
            }
        }
    }
    Ok(())
}

fn required_trait_of_node(node: &bb_ir::proto::onnx::NodeProto) -> Option<&str> {
    node.metadata_props
        .iter()
        .find(|p| p.key == REQUIRED_TRAIT_KEY)
        .map(|p| p.value.as_str())
}

