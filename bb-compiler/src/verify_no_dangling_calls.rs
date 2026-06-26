//! verifier.
//!
//! After `wrap_as_model` assembles a `ModelProto`, every entry in
//! `model.functions[1..]` is supposed to be a sub-function referenced
//! by a CALL NodeProto chain rooted at `model.functions[0]`. This
//! verifier double-checks the structural invariant Phase 2's
//! `Graph::with_module` is supposed to maintain - a DSL bug that
//! pushes an orphan sub-function (or emits a CALL chasing a missing
//! function) would be caught here instead of failing opaquely at
//! runtime CALL dispatch.
//!
//! The verifier runs as a final step inside
//! [`crate::run_pipeline_with_options`] (right after the
//! `wrap_as_model` call). It's deliberately structural: it does NOT
//! traverse into hoisted bodies' inputs / outputs / metadata, only
//! the CALL graph. Hoisted FunctionProtos with no CALL referencing
//! them are still functions in `model.functions[]`; if a future pass
//! adds reachability-pruning, this verifier promotes from
//! "double-check the DSL emitted a consistent table" to "enforce no
//! orphans". The hard error is therefore reserved for "CALL chases a
//! function the table doesn't contain" today.

use std::collections::{HashMap, HashSet};

use crate::error::CompileError;
use bb_ir::proto::onnx::ModelProto;

/// The `domain` value every `with_module`-emitted CALL NodeProto
/// carries.
const MODULE_CALL_DOMAIN: &str = "ai.bytesandbrains.module";

/// Walks the CALL graph rooted at `model.functions[0]` and errors
/// when any CALL's `op_type` doesn't match a `FunctionProto` in the
/// table. Returns the set of reachable function names so callers can
/// audit orphan entries if they want to.
pub fn verify_no_dangling_calls(model: &ModelProto) -> Result<HashSet<String>, CompileError> {
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (idx, f) in model.functions.iter().enumerate() {
        by_name.insert(f.name.as_str(), idx);
    }

    let Some(root) = model.functions.first() else {
        return Ok(HashSet::new());
    };

    let mut reached: HashSet<String> = HashSet::new();
    reached.insert(root.name.clone());

    let mut frontier: Vec<usize> = vec![0];
    while let Some(idx) = frontier.pop() {
        let f = &model.functions[idx];
        for node in &f.node {
            if node.domain != MODULE_CALL_DOMAIN {
                continue;
            }
            let callee_name = node.op_type.as_str();
            let Some(&callee_idx) = by_name.get(callee_name) else {
                return Err(CompileError::Internal {
                    detail: format!(
                        "function `{}` emits CALL chasing missing function `{}`",
                        f.name, callee_name,
                    ),
                });
            };
            if reached.insert(callee_name.to_string()) {
                frontier.push(callee_idx);
            }
        }
    }
    Ok(reached)
}

