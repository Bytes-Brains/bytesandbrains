//! `CpuBackend::execute_graph` — walk a `GraphProto` body (the kind
//! the compiler's `collapse_backend_subgraphs` pass emits inside
//! every `BackendSubgraph_*` `FunctionProto`) and run each
//! NodeProto through the existing kernel dispatch.
//!
//! No fancy scheduling — ONNX guarantees `GraphProto.node` is
//! already topologically ordered, so a linear walk suffices.

use std::collections::HashMap;

use bb_ir::proto::onnx::GraphProto;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::slot_value::SlotValue;

use crate::backends::cpu::{ops, CpuBackend, CpuTensor};

/// Failures `execute_graph` may surface alongside the kernel-level
/// `OpError`s already routed through the existing dispatch.
#[derive(Debug)]
pub enum BackendError {
    /// A node input name isn't in the value env. The graph either
    /// uses a value the caller didn't bind OR an upstream node
    /// failed to populate one of its declared outputs.
    MissingInput {
        /// Name of the missing value.
        name: String,
        /// op_type of the consuming node, for diagnostics.
        op_type: String,
    },

    /// A node output value isn't a `CpuTensor`. The CpuBackend's
    /// graph walker only handles f32 tensors; any other `SlotValue`
    /// kind from a custom kernel rejects.
    OutputNotTensor {
        /// op_type that produced the offending value.
        op_type: String,
    },

    /// The kernel itself returned an `OpError`. Wraps the error so
    /// callers see which op surfaced it.
    KernelFailed {
        /// op_type that failed.
        op_type: String,
        /// Underlying kernel error.
        source: OpError,
    },

    /// Bridged failure from the framework's default `Backend`
    /// walker (malformed graph fed to `Backend::execute`).
    DefaultWalker(bb_runtime::contracts::backend_default_walk::BackendWalkError),
}

impl From<bb_runtime::contracts::backend_default_walk::BackendWalkError> for BackendError {
    fn from(value: bb_runtime::contracts::backend_default_walk::BackendWalkError) -> Self {
        Self::DefaultWalker(value)
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingInput { name, op_type } => {
                write!(f, "execute_graph: input `{name}` missing for `{op_type}`",)
            }
            Self::OutputNotTensor { op_type } => write!(
                f,
                "execute_graph: `{op_type}` produced a non-CpuTensor output",
            ),
            Self::KernelFailed { op_type, source } => {
                write!(f, "execute_graph: `{op_type}` kernel failed: {source}",)
            }
            Self::DefaultWalker(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Run every `NodeProto` in `graph.node` in order, threading a
/// `HashMap<String, CpuTensor>` value env. Returns the subset of
/// `env` named in `graph.output`.
///
/// Pure over the `(graph, inputs)` pair — no engine context
/// required. Each node's `node.attribute` is the kernel's
/// attribute source; the dispatch path is identical to the
/// `BackendRuntime::dispatch_atomic` per-op path except that
/// attributes come from the NodeProto directly instead of through
/// `RuntimeResourceRef::current_node_attributes`.
pub fn execute_graph(
    backend: &CpuBackend,
    graph: &GraphProto,
    inputs: HashMap<String, CpuTensor>,
) -> Result<HashMap<String, CpuTensor>, BackendError> {
    let mut env: HashMap<String, CpuTensor> = inputs;

    for node in &graph.node {
        let mut input_refs: Vec<(&str, &dyn SlotValue)> = Vec::with_capacity(node.input.len());
        for name in &node.input {
            if name.is_empty() {
                continue;
            }
            let tensor = env.get(name).ok_or_else(|| BackendError::MissingInput {
                name: name.clone(),
                op_type: node.op_type.clone(),
            })?;
            input_refs.push((name.as_str(), tensor as &dyn SlotValue));
        }

        let result =
            ops::dispatch(backend, &node.op_type, &input_refs, &node.attribute).map_err(|e| {
                BackendError::KernelFailed {
                    op_type: node.op_type.clone(),
                    source: e,
                }
            })?;

        let outputs = match result {
            DispatchResult::Immediate(outs) => outs,
            DispatchResult::Async(_) => {
                return Err(BackendError::KernelFailed {
                    op_type: node.op_type.clone(),
                    source: OpError {
                        detail: format!(
                            "{op}: async dispatch unsupported inside execute_graph",
                            op = node.op_type,
                        ),
                        ..Default::default()
                    },
                });
            }
        };

        // Map each (kernel-named) output positionally to the
        // NodeProto's `node.output[i]` name. Kernels label outputs
        // `"C"` / `"Y"` / `"out_0"` etc., which don't have to match
        // the consumer-side value name the graph references.
        for (i, (_kernel_name, boxed)) in outputs.into_iter().enumerate() {
            let Some(graph_name) = node.output.get(i) else {
                // Kernel produced more outputs than the graph
                // declares — drop the extra silently (the consumer
                // doesn't reference it).
                continue;
            };
            if graph_name.is_empty() {
                continue;
            }
            // Consume the boxed kernel output into the env without
            // cloning. `into_any_boxed` repackages `Box<dyn SlotValue>`
            // as `Box<dyn Any>` so `Box::downcast` lands the concrete
            // tensor by move.
            let any = boxed.into_any_boxed();
            let tensor: Box<CpuTensor> =
                any.downcast::<CpuTensor>()
                    .map_err(|_| BackendError::OutputNotTensor {
                        op_type: node.op_type.clone(),
                    })?;
            env.insert(graph_name.clone(), *tensor);
        }
    }

    // Return the subset named in graph.output. Missing names drop
    // silently — callers explicitly ask for them by walking
    // `graph.output` themselves if they need full coverage.
    let mut out: HashMap<String, CpuTensor> = HashMap::new();
    for vi in &graph.output {
        if let Some(t) = env.remove(&vi.name) {
            out.insert(vi.name.clone(), t);
        }
    }
    Ok(out)
}

