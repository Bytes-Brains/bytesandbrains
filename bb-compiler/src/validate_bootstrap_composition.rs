//! Structural check for the bootstrap-as-function composition tree.
//!
//! Every Module's bootstrap recording emits its own FunctionProto
//! stamped `MODULE_PHASE_BOOTSTRAP`. A parent Module that composes
//! children via `ModuleCall::bootstrap(g)` records a CALL NodeProto
//! into its own bootstrap pointing at the child's
//! `"<child>__bootstrap"` FunctionProto. This pass walks the call
//! graph rooted at the target's bootstrap and surfaces:
//!
//! - [`CompileError::BootstrapCompositionGap`] when a Call points at a
//!   bootstrap function name that has no matching FunctionProto in the
//!   model — the engine's FunctionCall dispatch would refuse to seed a
//!   CallContext for a missing target.
//! - [`CompileError::BootstrapCompositionCycle`] when the function-call
//!   graph is not a DAG. Bootstrap completion is one-shot; a cycle in
//!   the composition tree wedges the engine in `bootstrap_pending`
//!   forever.
//!
//! Runs at the top level of the compile pipeline — before inlining or
//! partitioning can rearrange the function table — so the error
//! surfaces against the recorded composition shape the author wrote.

use std::collections::{HashMap, HashSet};

use bb_ir::keys::{read_function_module_phase, MODULE_PHASE_BOOTSTRAP};
use bb_ir::proto::onnx::ModelProto;

use crate::error::CompileError;

/// CALL NodeProto domain produced by `Graph::with_function`. Same
/// constant as `verify_no_dangling_calls` / `inline_for_partition` use;
/// duplicated here so this pass stays self-contained.
const MODULE_CALL_DOMAIN: &str = "ai.bytesandbrains.module";

/// Walk the bootstrap-call graph rooted at `<target>__bootstrap` and
/// verify every reachable Call resolves to an existing FunctionProto.
///
/// `target_name` is the root function's name (i.e. the body half of
/// the top-level Module). Models whose root Module has no bootstrap
/// recording are no-ops — the trait's default `fn bootstrap(&self,
/// _g: &mut Graph) {}` emits an empty bootstrap that
/// `Module::build` drops on the floor.
pub fn validate_bootstrap_composition(
    model: &ModelProto,
    target_name: &str,
) -> Result<(), CompileError> {
    let by_name: HashMap<&str, &bb_ir::proto::onnx::FunctionProto> = model
        .functions
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect();

    let root_bootstrap = format!("{target_name}__bootstrap");
    let Some(root_fn) = by_name.get(root_bootstrap.as_str()) else {
        return Ok(());
    };
    if read_function_module_phase(root_fn) != Some(MODULE_PHASE_BOOTSTRAP) {
        // Name collides with the bootstrap convention but the phase
        // stamp is missing — treat as not-a-bootstrap rather than
        // misclassify.
        return Ok(());
    }

    // Tri-coloured DFS over the bootstrap call graph. `gray` carries
    // the active recursion stack so a back-edge surfaces the
    // participating function names in cycle order.
    let mut black: HashSet<String> = HashSet::new();
    let mut gray: Vec<String> = Vec::new();
    walk(&root_bootstrap, &by_name, &mut gray, &mut black)
}

fn walk(
    name: &str,
    by_name: &HashMap<&str, &bb_ir::proto::onnx::FunctionProto>,
    gray: &mut Vec<String>,
    black: &mut HashSet<String>,
) -> Result<(), CompileError> {
    if black.contains(name) {
        return Ok(());
    }
    if let Some(pos) = gray.iter().position(|n| n == name) {
        let mut involves: Vec<String> = gray[pos..].to_vec();
        involves.push(name.to_string());
        return Err(CompileError::BootstrapCompositionCycle { involves });
    }

    let Some(function) = by_name.get(name) else {
        // Caller is responsible for confirming `name` resolves before
        // calling `walk`; an unresolved name here means a child Call
        // pointed at a missing bootstrap.
        return Err(CompileError::BootstrapCompositionGap {
            caller: gray.last().cloned().unwrap_or_else(|| name.to_string()),
            target: name.to_string(),
        });
    };

    gray.push(name.to_string());
    for node in &function.node {
        if node.domain != MODULE_CALL_DOMAIN {
            continue;
        }
        let target = node.op_type.as_str();
        if target.is_empty() {
            continue;
        }
        if !by_name.contains_key(target) {
            return Err(CompileError::BootstrapCompositionGap {
                caller: name.to_string(),
                target: target.to_string(),
            });
        }
        // Only descend into callees that themselves carry the
        // bootstrap-phase stamp. A Call from a bootstrap into a body
        // function is a recording bug the DSL never emits, but
        // ignoring non-bootstrap callees here keeps the walk
        // bootstrap-scoped — body-only call cycles surface in
        // `verify_no_dangling_calls` / the per-graph cycle pass.
        let callee = by_name[target];
        if read_function_module_phase(callee) == Some(MODULE_PHASE_BOOTSTRAP) {
            walk(target, by_name, gray, black)?;
        }
    }
    gray.pop();
    black.insert(name.to_string());
    Ok(())
}

