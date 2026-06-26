//! Pipeline orchestrator. Runs canonical passes in order and emits
//! one `ModelProto` per partition; `functions[0]` is the partition
//! main, `functions[1..]` are hoisted sub-Module bodies. Driven
//! from [`crate::driver::Compiler::compile`].

use std::collections::HashSet;

use crate::error::CompileError;
use crate::infer_peer_classes::infer_peer_classes;
use crate::{
    analyze_wire_edges, derive_wire_deadlines, expand_ops, inline_for_partition,
    insert_async_deadlines, insert_backoff_gate_rx, insert_backoff_gate_tx, insert_dedup_gate_rx,
    insert_peer_health_gate_rx, insert_peer_health_gate_tx, partition_by_wire_ops, resolve_slots,
    synthesize_wire_recvs, validate, validate_bootstrap_composition, validate_runtime_complete,
    verify_no_dangling_calls,
};
use bb_dsl::recorded::RecordedModule;
use bb_ir::proto::onnx::{FunctionProto, GraphProto, ModelProto};

/// Canonical pass names in pipeline order. Each pass assumes the
/// prior passes' invariants.
pub const CANONICAL_PASS_NAMES: &[&str] = &[
    "inline_for_partition",
    "derive_wire_deadlines",
    "validate",
    "expand_ops",
    "type_solver",
    "infer_peer_classes",
    "synthesize_wire_recvs",
    "partition_by_wire_ops",
    "resolve_slots",
    "analyze_wire_edges",
    "insert_dedup_gate_rx",
    "insert_peer_health_gate_rx",
    "insert_backoff_gate_rx",
    "insert_peer_health_gate_tx",
    "insert_backoff_gate_tx",
    "insert_async_deadlines",
    "validate_runtime_complete",
];

/// Run the pipeline; canonical passes fire only when named in
/// `enabled`. Wire ops in the root drive partitioning after
/// `inline_for_partition` surfaces them.
pub(crate) fn run_pipeline_with_options(
    recorded: RecordedModule,
    _module_name: String,
    enabled: &HashSet<String>,
    per_hop_budget_ns: u64,
    strict_types: bool,
) -> Result<Vec<ModelProto>, CompileError> {
    let RecordedModule {
        function,
        sub_functions: dsl_sub_functions,
    } = recorded;

    let on = |name: &str| enabled.contains(name);

    let mut temp = ModelProto::default();
    temp.functions.push(function);
    temp.functions.extend(dsl_sub_functions);
    if on("inline_for_partition") {
        inline_for_partition(&mut temp)?;
    }
    if on("derive_wire_deadlines") {
        derive_wire_deadlines(&mut temp, per_hop_budget_ns)?;
    }

    // Seam verifiers catch malformed phase output at the source.
    bb_ir::verify::types(&temp).map_err(|e| CompileError::Internal {
        detail: format!("verify::types failed at frontend seam: {e}"),
    })?;
    bb_ir::verify::function_calls(&temp).map_err(|e| CompileError::Internal {
        detail: format!("verify::function_calls failed at frontend seam: {e}"),
    })?;

    let mut models: Vec<ModelProto> = Vec::new();

    // After inline_for_partition, functions[0] is the root and the
    // only "target" — partition_by_wire_ops will slice it into one
    // installable per connected component. Surviving non-inlined
    // FunctionProtos are shared and attach to each emitted ModelProto.
    let shared_functions: Vec<FunctionProto> = temp.functions.iter().skip(1).cloned().collect();
    let root = temp
        .functions
        .first()
        .ok_or_else(|| CompileError::Internal {
            detail: "compiler received an empty function table".into(),
        })?;
    let root_name = root.name.clone();

    // Bootstrap-as-function: validate the bootstrap composition tree
    // before partitioning starts. Runs against the full top-level model
    // view (where the `<root>__bootstrap` FunctionProto + every child
    // bootstrap still live in `temp.functions`); per-partition passes
    // never see the bootstrap function table.
    validate_bootstrap_composition(&temp, &root_name)?;

    let root = temp
        .functions
        .into_iter()
        .next()
        .expect("non-empty checked above");
    let target_models = process_target(root, &root_name, enabled, &shared_functions, strict_types)?;
    models.extend(target_models);

    if models.is_empty() {
        return Err(CompileError::Internal {
            detail: "compiler produced no partitions - recorded function was empty".into(),
        });
    }
    Ok(models)
}

/// Run the per-target pipeline: validate, structural transforms,
/// partition by wire ops, then per-partition transforms, emitting
/// one `ModelProto` per home-class partition the target produces.
/// `shared_functions` are FunctionProtos that survived
/// `inline_for_partition` and are carried into every emitted
/// ModelProto so CALL nodes still resolve.
fn process_target(
    mut target_function: FunctionProto,
    target_name: &str,
    enabled: &HashSet<String>,
    shared_functions: &[FunctionProto],
    strict_types: bool,
) -> Result<Vec<ModelProto>, CompileError> {
    let on = |name: &str| enabled.contains(name);

    if on("validate") {
        let view = function_to_graph_view(&target_function);
        validate(&view).map_err(CompileError::Validation)?;
    }

    let view = function_to_graph_view(&target_function);
    let mut graph = view;
    if on("expand_ops") {
        expand_ops(&mut graph)?;
    }
    // Track whether the type_solver pass ran so `check_wire_edge_types`
    // (which needs a TypeSolution) is only invoked when the solver did run.
    let type_solver_ran = if on("type_solver") {
        run_type_solver(&mut graph, strict_types)?;
        true
    } else {
        false
    };
    if on("infer_peer_classes") {
        infer_peer_classes(&mut graph)?;
    }
    if on("synthesize_wire_recvs") {
        synthesize_wire_recvs(&mut graph)?;
        // check_wire_edge_types runs HERE — after synthesis — so the
        // Recv nodes exist in the graph and SYNTHESIZED_FROM_KEY carries
        // the Send's original value NAME (not a node index), which the
        // TypeSolution can look up via `type_of`. The minted recv-output
        // value names got `stamped_value_info` entries during synthesis;
        // a fresh solver run picks them up so the comparison is accurate.
        if type_solver_ran {
            let fresh_solution = run_type_solver(&mut graph, strict_types)?;
            check_wire_edge_types(&graph, &fresh_solution)?;
        }
    }
    target_function = merge_graph_into_function(target_function, graph);

    let view = function_to_graph_view(&target_function);
    let mut analysis = partition_by_wire_ops(&view)?;

    if on("resolve_slots") {
        resolve_slots(&target_function)?;
    }

    // Call `analyze_wire_edges` ONCE per partition map, before the
    // per-role loop walks each sub_graph for the rest of the
    // pipeline. The pass reads denotations from the sub_graph
    // value_info and stamps classification metadata directly onto
    // the matching `sub_graph.node` Send/Recv NodeProtos —
    // `analysis.wire_edges` is consumed read-only to drive the
    // iteration.
    if on("analyze_wire_edges") {
        for sub_graph in analysis.per_role.values_mut() {
            analyze_wire_edges(sub_graph, &analysis.wire_edges)?;
        }
    }

    let mut models: Vec<ModelProto> = Vec::new();
    for (role, mut sub_graph) in analysis.per_role {
        let hoisted: Vec<FunctionProto> = Vec::new();

        if on("insert_dedup_gate_rx") {
            insert_dedup_gate_rx(&mut sub_graph)?;
        }
        if on("insert_peer_health_gate_rx") {
            insert_peer_health_gate_rx(&mut sub_graph)?;
        }
        if on("insert_backoff_gate_rx") {
            insert_backoff_gate_rx(&mut sub_graph)?;
        }
        if on("insert_peer_health_gate_tx") {
            insert_peer_health_gate_tx(&mut sub_graph)?;
        }
        if on("insert_backoff_gate_tx") {
            insert_backoff_gate_tx(&mut sub_graph)?;
        }
        if on("insert_async_deadlines") {
            insert_async_deadlines(&mut sub_graph)?;
        }

        if on("validate_runtime_complete") {
            validate_runtime_complete(&sub_graph)?;
        }

        let (composite_name, mut partition_function) =
            split_partition(&target_function, role.clone(), &sub_graph, target_name);
        partition_function.name = composite_name;
        let mut all_hoisted = hoisted;
        all_hoisted.extend(shared_functions.iter().cloned());
        let model = wrap_as_model(partition_function, all_hoisted);
        verify_no_dangling_calls(&model)?;
        models.push(model);
    }

    Ok(models)
}

use bb_ir::proto::function_to_graph_view;

fn merge_graph_into_function(mut function: FunctionProto, graph: GraphProto) -> FunctionProto {
    function.node = graph.node;
    function
}

fn split_partition(
    base: &FunctionProto,
    role: String,
    sub_graph: &GraphProto,
    module_name: &str,
) -> (String, FunctionProto) {
    let mut function = base.clone();
    function.node = sub_graph.node.clone();
    let base_name = if role == "@default" || role == bb_ir::peer_class::SELF_CLASS {
        module_name.to_string()
    } else if role == module_name || role.starts_with(&format!("{module_name}_")) {
        role
    } else {
        format!("{module_name}_{role}")
    };
    // widen the partition composite-name suffix from
    // 8 hex chars (`u32` mask) to 16 hex chars (full `u64`) so the
    // birthday-collision bound on partition naming goes from
    // ~65 k partitions (50 % collision at 2^16) to ~4 G partitions
    // (50 % collision at 2^32). Closes the narrowing flagged in
    // docs-plan/sections/Compiler.md §Determinism.
    //
    // Two partitions with the same base name (same role on
    // different builds) still collide if the content is identical
    // (correct — they ARE the same partition); a body edit shifts
    // the hash so snapshots from the old version don't accidentally
    // restore into the new partition.
    let content_hash = crate::function_dedup::hash_node_bodies(&sub_graph.node);
    let composite_name = format!("{base_name}#{content_hash:016x}");
    (composite_name, function)
}

fn wrap_as_model(function: FunctionProto, hoisted: Vec<FunctionProto>) -> ModelProto {
    let mut functions = Vec::with_capacity(1 + hoisted.len());
    functions.push(function);
    functions.extend(hoisted);
    ModelProto {
        functions,
        ..Default::default()
    }
}

/// Run the TypeSolver pass over `graph`, narrowing each value's
/// type from `TYPE_BYTES` to the most specific TypeNode
/// that connected ops' `type_relations` declarations permit.
///
/// Permissive-by-default: values whose constraints don't fully
/// resolve stay at `TYPE_ANY`. Set `strict = true` to surface
/// unresolved values as a typed `BuildError::UnresolvedType` —
/// useful for users who want a hard guarantee at compile time
/// that every input has a concrete TypeNode.
///
/// Op `(domain, op_type)` → `AtomicOpDecl` lookup walks every
/// registered opset (placeholders, syscalls, backends, custom
/// `bb::register_op!{}` entries) via the existing inventory
/// channel.
/// Run the TypeSolver pass over `graph` and return the resolved
/// `TypeSolution`. The solution is also applied to `graph.value_info`
/// in place.
///
/// `check_wire_edge_types` is NOT called here — it must run AFTER
/// `synthesize_wire_recvs` so that Recv nodes exist in the graph and
/// the minted recv-output value names are present in the solution.
/// Callers should call `run_type_solver` again (or
/// `check_wire_edge_types` directly with a fresh solution) after
/// synthesis to perform the wire-edge check.
fn run_type_solver(
    graph: &mut GraphProto,
    strict: bool,
) -> Result<crate::type_solver::TypeSolution, CompileError> {
    // No production component currently emits
    // `inventory::submit!{ OpSignatureRegistration { ... } }`, so the
    // solver always falls through to value_info round-trip. The
    // `decl_for_op` closure stays in the signature so the recorder
    // can plug in a real lookup once denotations are stamped
    // recorder-side under #20.
    let decl_for_op = |_: &str, _: &str| -> Option<&'static bb_ir::atomic::AtomicOpDecl> { None };

    let mut solver = crate::type_solver::TypeSolver::from_graph(graph, decl_for_op)
        .map_err(CompileError::from)?;
    solver.seed_from_value_info(graph);
    let solution = if strict {
        solver.solve_strict().map_err(CompileError::from)?
    } else {
        solver.solve().map_err(CompileError::from)?
    };
    crate::type_solver::TypeSolver::apply_solution_to_value_info(graph, &solution);
    Ok(solution)
}

/// Walk `graph.node` for synthesized `wire.Recv` NodeProtos. For each
/// Recv that carries `SYNTHESIZED_FROM_KEY` metadata, compare the
/// resolved type of the send-side value against the resolved type of
/// the recv-side value. If both are concrete and different, no `Codec`
/// bridge was wired on the edge and the author must insert one.
///
/// Exposed as `pub(crate)` so `type_solver_tests` can exercise the
/// check in isolation against hand-built graphs.
pub(crate) fn check_wire_edge_types(
    graph: &GraphProto,
    solution: &crate::type_solver::TypeSolution,
) -> Result<(), CompileError> {
    const SYNTHESIZED_FROM_KEY: &str = "ai.bytesandbrains.synthesized_from_send";
    const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";

    for node in &graph.node {
        if node.domain != WIRE_DOMAIN || node.op_type != "Recv" {
            continue;
        }
        // Look up the send-side value name from the SYNTHESIZED_FROM_KEY
        // metadata that `synthesize_wire_recvs` stamps on every Recv.
        let Some(src_val) = node
            .metadata_props
            .iter()
            .find(|p| p.key == SYNTHESIZED_FROM_KEY)
            .map(|p| p.value.as_str())
        else {
            continue;
        };
        // The recv-side value name is the Recv's first output.
        let Some(dst_val) = node.output.first().filter(|s| !s.is_empty()) else {
            continue;
        };

        let Some(actual_node) = solution.type_of(src_val) else {
            continue;
        };
        let Some(expected_node) = solution.type_of(dst_val) else {
            continue;
        };

        if !actual_node.is_concrete() || !expected_node.is_concrete() {
            continue;
        }
        if std::ptr::eq(actual_node, expected_node) {
            continue;
        }

        return Err(CompileError::IncompatibleStorageOnEdge {
            src: src_val.to_string(),
            dst: dst_val.to_string(),
            expected_id: expected_node.id,
            actual_id: actual_node.id,
        });
    }
    Ok(())
}
