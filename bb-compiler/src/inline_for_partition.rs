//! Selective function inlining for the partition stage.
//!
//! Modules don't drive network boundaries any more — wire ops do.
//! Three classes of functions get inlined at every CALL site before
//! [`crate::partition_by_wire_ops`] runs:
//!
//! 1. **Wire-touching functions** — any function whose transitive
//!    closure contains an `ai.bytesandbrains.wire` op. Wire ops must
//!    live at the top level so the partitioner's reachability walk
//!    cuts the graph cleanly; a wire hidden behind a CALL fragments
//!    the partition boundary.
//!
//! 2. **Pure-ONNX functions** — any function whose transitive closure
//!    is entirely `ai.onnx.*`. Inlining surfaces each NodeProto at
//!    the top level so the engine can route per-op against the
//!    bound `Backend`'s Contract methods (`add`, `matmul`, …)
//!    without an intervening CALL indirection.
//!
//! 3. **Single-call functions** — any function CALLed from exactly
//!    one site across the whole call graph. Keeping it as a
//!    FunctionProto saves no memory (the body would appear once
//!    either way) but adds an indirection at dispatch time, so we
//!    inline it eagerly.
//!
//! Functions called from multiple sites (and not in classes 1 or 2)
//! survive as `FunctionProto` — the body appears once, callers
//! reference it via CALL nodes.
//!
//! The root function (`model.functions[0]`) is always preserved
//! regardless of its body's classification — it's the entry point
//! the compiler partitions on.

use std::collections::{HashMap, HashSet};

use crate::error::CompileError;
use bb_ir::proto::onnx::{FunctionProto, ModelProto, NodeProto};

const MODULE_CALL_DOMAIN: &str = "ai.bytesandbrains.module";
const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";
const ONNX_DOMAIN: &str = "ai.onnx";

/// Inline every wire-touching or pure-ONNX function at every CALL
/// site. Iterates to a fixed point because inlining a wire-touching
/// callee may itself reveal a new wire-touching caller. Returns the
/// total number of CALL replacements performed.
pub fn inline_for_partition(model: &mut ModelProto) -> Result<usize, CompileError> {
    let root_name = model.functions.first().map(|f| f.name.clone());
    let mut total_inlines: usize = 0;
    let mut next_unique: u64 = 0;

    loop {
        let inlinable = classify_inlinable(model, root_name.as_deref());
        if inlinable.is_empty() {
            break;
        }
        let order = reverse_topo_order(model, &inlinable);

        for name in order {
            // Snapshot the body — we splice copies of it into each
            // CALL site.
            let body = match model.functions.iter().find(|f| f.name == name) {
                Some(f) => f.clone(),
                None => continue,
            };

            for caller in model.functions.iter_mut() {
                if caller.name == name {
                    continue;
                }
                let mut rewritten: Vec<NodeProto> = Vec::with_capacity(caller.node.len());
                let mut inlined_value_info: Vec<bb_ir::proto::onnx::ValueInfoProto> = Vec::new();
                for node in caller.node.iter() {
                    if node.domain == MODULE_CALL_DOMAIN && node.op_type == name {
                        let (nodes, value_info) = inline_one_call(&body, node, &mut next_unique);
                        rewritten.extend(nodes);
                        inlined_value_info.extend(value_info);
                        total_inlines += 1;
                    } else {
                        rewritten.push(node.clone());
                    }
                }
                caller.node = rewritten;
                // Inlined sub-function value_info entries ride into
                // the caller so strict-types-by-default sees a
                // declared denotation for every renamed intermediate.
                for vi in inlined_value_info {
                    if !caller.value_info.iter().any(|v| v.name == vi.name) {
                        caller.value_info.push(vi);
                    }
                }
            }
        }

        // Drop the inlined functions from the table — their bodies
        // now live at every former call site.
        model.functions.retain(|f| !inlinable.contains(&f.name));
    }

    Ok(total_inlines)
}

/// Classify every non-root function as inlinable or kept. A
/// function is inlinable iff it's wire-touching, pure-ONNX, or
/// called from exactly one site.
fn classify_inlinable(model: &ModelProto, root_name: Option<&str>) -> HashSet<String> {
    let wire_touching = wire_closure(model);
    let pure_onnx = pure_onnx_closure(model);
    let call_counts = count_call_sites(model);

    let mut result = HashSet::new();
    for f in &model.functions {
        if root_name == Some(f.name.as_str()) {
            continue;
        }
        let single_call = call_counts.get(&f.name).copied() == Some(1);
        if wire_touching.contains(&f.name) || pure_onnx.contains(&f.name) || single_call {
            result.insert(f.name.clone());
        }
    }
    result
}

/// Count CALL sites referencing each function across the model.
/// Keyed by callee function name.
fn count_call_sites(model: &ModelProto) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for f in &model.functions {
        for node in &f.node {
            if node.domain == MODULE_CALL_DOMAIN {
                *counts.entry(node.op_type.clone()).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Functions whose transitive closure contains any wire op. Computed
/// by starting at functions with a direct wire op in their body and
/// propagating up the call graph (any caller of a wire-touching
/// function is itself wire-touching).
fn wire_closure(model: &ModelProto) -> HashSet<String> {
    let mut closure: HashSet<String> = model
        .functions
        .iter()
        .filter(|f| f.node.iter().any(|n| n.domain == WIRE_DOMAIN))
        .map(|f| f.name.clone())
        .collect();

    loop {
        let mut changed = false;
        for f in &model.functions {
            if closure.contains(&f.name) {
                continue;
            }
            if f.node
                .iter()
                .any(|n| n.domain == MODULE_CALL_DOMAIN && closure.contains(&n.op_type))
            {
                closure.insert(f.name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    closure
}

/// Functions whose transitive closure is entirely `ai.onnx.*`.
/// Computed by iterating: a function is pure-ONNX if every node in
/// its body is either (a) a CALL to another pure-ONNX function or
/// (b) a direct `ai.onnx` op.
fn pure_onnx_closure(model: &ModelProto) -> HashSet<String> {
    let mut closure: HashSet<String> = HashSet::new();
    loop {
        let mut changed = false;
        for f in &model.functions {
            if closure.contains(&f.name) {
                continue;
            }
            // Empty body counts as pure-ONNX vacuously, but only
            // matters for synthetic cases — real ONNX functions have
            // at least one op.
            let all_ok = !f.node.is_empty()
                && f.node.iter().all(|n| {
                    if n.domain == MODULE_CALL_DOMAIN {
                        closure.contains(&n.op_type)
                    } else {
                        n.domain == ONNX_DOMAIN
                    }
                });
            if all_ok {
                closure.insert(f.name.clone());
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    closure
}

/// Reverse-topological order over the inlinable subset. Leaves of
/// the call graph (functions whose bodies contain no CALLs to other
/// inlinable functions) come first so each inline operation sees a
/// body that no longer references other inlinables.
fn reverse_topo_order(model: &ModelProto, inlinable: &HashSet<String>) -> Vec<String> {
    let inlinable_idx: HashMap<String, usize> = model
        .functions
        .iter()
        .enumerate()
        .filter(|(_, f)| inlinable.contains(&f.name))
        .map(|(i, f)| (f.name.clone(), i))
        .collect();

    let mut visited: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::new();

    fn visit(
        name: &str,
        model: &ModelProto,
        inlinable_idx: &HashMap<String, usize>,
        visited: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if !visited.insert(name.to_string()) {
            return;
        }
        let Some(&idx) = inlinable_idx.get(name) else {
            return;
        };
        let f = &model.functions[idx];
        for node in &f.node {
            if node.domain == MODULE_CALL_DOMAIN && inlinable_idx.contains_key(&node.op_type) {
                visit(&node.op_type, model, inlinable_idx, visited, order);
            }
        }
        order.push(name.to_string());
    }

    let names: Vec<String> = inlinable_idx.keys().cloned().collect();
    for name in &names {
        visit(name, model, &inlinable_idx, &mut visited, &mut order);
    }
    order
}

/// Splice one inlined copy of `body` in place of `call`. Intermediate
/// value names get a unique suffix to avoid collisions across
/// multiple inlines; formal-input and body-output names are rewritten
/// to the CALL's actual arg/output names so downstream caller-side
/// consumers still resolve.
fn inline_one_call(
    body: &FunctionProto,
    call: &NodeProto,
    next_unique: &mut u64,
) -> (Vec<NodeProto>, Vec<bb_ir::proto::onnx::ValueInfoProto>) {
    let unique = *next_unique;
    *next_unique = next_unique.saturating_add(1);

    let mut rename: HashMap<String, String> = HashMap::new();
    for (i, formal) in body.input.iter().enumerate() {
        if let Some(actual) = call.input.get(i) {
            rename.insert(formal.clone(), actual.clone());
        }
    }
    for (i, body_out) in body.output.iter().enumerate() {
        if let Some(call_out) = call.output.get(i) {
            rename.insert(body_out.clone(), call_out.clone());
        }
    }

    let mut rename_value = |name: &str| -> String {
        if name.is_empty() {
            return String::new();
        }
        if let Some(renamed) = rename.get(name) {
            return renamed.clone();
        }
        let fresh = format!("{name}#inl{unique}");
        rename.insert(name.to_string(), fresh.clone());
        fresh
    };

    let mut out: Vec<NodeProto> = Vec::with_capacity(body.node.len());
    for node in &body.node {
        let mut cloned = node.clone();
        for input in cloned.input.iter_mut() {
            *input = rename_value(input);
        }
        for output in cloned.output.iter_mut() {
            *output = rename_value(output);
        }
        out.push(cloned);
    }

    // Copy value_info entries from the body into the caller under
    // the renamed names so denotations ride through inlining.
    let value_info: Vec<bb_ir::proto::onnx::ValueInfoProto> = body
        .value_info
        .iter()
        .filter_map(|vi| {
            let new_name = rename.get(&vi.name).cloned()?;
            let mut renamed = vi.clone();
            renamed.name = new_name;
            Some(renamed)
        })
        .collect();

    (out, value_info)
}

