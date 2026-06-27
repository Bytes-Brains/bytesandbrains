//! Structural verifier: every entry in `model.functions[1..]` must
//! be reachable from `model.functions[0]` via the CALL chain. Errors
//! when a CALL chases a function name absent from `model.functions`.
//! Orphan entries (no CALL referencing them) are tolerated today; a
//! later reachability-pruning pass may tighten this.

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

