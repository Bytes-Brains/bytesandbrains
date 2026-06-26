//! Shared default impls for the `Backend` Contract trait.
//!
//! The Contract surface is *thirty typed per-op methods*
//! ([`super::Backend::add`], [`super::Backend::matmul`], …) plus
//! [`super::Backend::execute`] (`&GraphProto, HashMap` →
//! `HashMap`). Each side has a default body that calls into the
//! other so backend authors override whichever side is natural:
//!
//! - **Override per-op methods** (e.g. `CpuBackend` over ndarray):
//!   each Contract method is a direct kernel call. The default
//!   `execute` walks the graph node-by-node and dispatches through
//!   the overridden per-op methods.
//!
//! - **Override `execute`** (e.g. a graph-compiling backend like
//!   Burn): the whole `GraphProto` body is handed to the native
//!   execution engine. The per-op defaults wrap a single-node
//!   `GraphProto` and call back into `execute`.
//!
//! Pathological case: a backend that overrides *neither* side
//! stack-overflows — every `add` call walks into a single-node
//! graph that calls `add` again. Document loudly on the trait.
//!
//! This module also encodes the attribute conventions every
//! `BackendSubgraph_*` carrier uses for primitive ops with
//! attributes (`ReduceSum.axes`, `Reshape.shape`, `Cast.to`, …).
//! Per-op defaults call [`ints_attr`] / [`int_attr`] / [`tensor_attr`]
//! to encode; the walker calls `attr_ints` / `attr_int` /
//! `attr_tensor` to decode. ONNX-style names are preserved
//! (`axes`, `keepdims`, `shape`, `perm`, `axis`, `to`, `value`).

use std::collections::HashMap;

use bb_ir::proto::onnx::{AttributeProto, GraphProto, NodeProto, TensorProto, ValueInfoProto};

use super::backend::Backend;

const SINGLE_OP_OUTPUT_NAME: &str = "__bb_default_walk_output";

/// Failures the default walker surfaces when handed a malformed
/// `GraphProto` body or when a `Backend::execute` impl violates
/// its output-name contract. Required `From` bound on
/// [`Backend::Error`] makes the walker fail with a typed error
/// instead of `panic!`-ing on peer-supplied or buggy input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendWalkError {
    /// A `NodeProto` references an input value not in the running
    /// environment. The graph either uses a value the caller didn't
    /// bind OR an upstream node failed to populate one of its
    /// declared outputs.
    MissingInput {
        /// `op_type` of the consuming node.
        op_type: String,
        /// Name of the missing input value.
        input_name: String,
    },
    /// A per-op method produced a different number of outputs than
    /// the consuming `NodeProto` declares. Indicates a compiler
    /// or graph-builder bug.
    OutputArityMismatch {
        /// `op_type` of the node.
        op_type: String,
        /// Number of outputs the per-op method produced.
        produced: usize,
        /// Number of outputs the graph declares.
        declared: usize,
    },
    /// `op_type` is not one of [`bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS`].
    /// Backends that use the default per-op walker must keep graphs
    /// to primitives only. The wire path can hit this when an
    /// adversarial peer ships a `BackendSubgraph_*` carrier whose
    /// body references an extension op.
    UnknownOpType(String),
    /// A `Backend::execute` impl returned successfully but did not
    /// populate the declared output `output_name` in the result
    /// map. Indicates a Backend impl bug — `execute` MUST honor
    /// the graph's output names.
    MissingExecuteOutput {
        /// `op_type` of the single-node graph that was executed.
        op_type: String,
        /// Output name the graph declared but `execute` omitted.
        output_name: String,
    },
    /// Default [`crate::contracts::Backend::materialize_from_wire`]
    /// failed before reaching the backend: no registered decoder,
    /// the decoder rejected the bytes, or the decoded carrier was
    /// not assignable to `Self::Tensor`. Backends overriding
    /// `materialize_from_wire` never surface this — they emit their
    /// own typed error.
    WireMaterializeFailed {
        /// `type_hash` the inbound `SlotFill` advertised.
        type_hash: u64,
        /// Why the default path could not produce a `Self::Tensor`.
        reason: String,
    },
}

impl std::fmt::Display for BackendWalkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingInput { op_type, input_name } => write!(
                f,
                "Backend default walker: `{op_type}` references input `{input_name}` not in the value env",
            ),
            Self::OutputArityMismatch { op_type, produced, declared } => write!(
                f,
                "Backend default walker: per-op `{op_type}` produced {produced} outputs but graph declares {declared}",
            ),
            Self::UnknownOpType(op_type) => write!(
                f,
                "Backend default walker: op_type `{op_type}` is not in TENSOR_PRIMITIVES_OPS",
            ),
            Self::MissingExecuteOutput { op_type, output_name } => write!(
                f,
                "Backend::execute (op_type `{op_type}`) did not produce its declared output `{output_name}`",
            ),
            Self::WireMaterializeFailed { type_hash, reason } => write!(
                f,
                "Backend default materialize_from_wire (type_hash {type_hash:#018x}): {reason}",
            ),
        }
    }
}

impl std::error::Error for BackendWalkError {}

/// Default body for every per-op method on [`Backend`] — wraps the
/// op in a one-node `GraphProto` and routes through
/// [`Backend::execute`]. Backends that override `execute` natively
/// (graph-compiling backends) get the per-op surface free; backends
/// that override the per-op methods directly bypass this helper
/// entirely.
///
/// `attributes` carries the ONNX-style per-op attributes the typed
/// per-op signatures encode (`ReduceSum`'s `axes` + `keepdims`,
/// `Reshape`'s `shape`, etc.). The walker decodes them back via
/// `attr_ints` / `attr_int` / `attr_tensor` before calling
/// the typed per-op method on the backend.
pub fn execute_single<B: Backend + ?Sized>(
    backend: &B,
    op_type: &str,
    inputs: &[&B::Tensor],
    attributes: Vec<AttributeProto>,
) -> Result<B::Tensor, B::Error> {
    let input_names: Vec<String> = (0..inputs.len())
        .map(|i| format!("__bb_default_walk_in_{i}"))
        .collect();

    let node = NodeProto {
        op_type: op_type.to_string(),
        input: input_names.clone(),
        output: vec![SINGLE_OP_OUTPUT_NAME.to_string()],
        attribute: attributes,
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node],
        output: vec![ValueInfoProto {
            name: SINGLE_OP_OUTPUT_NAME.to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let input_map: HashMap<String, B::Tensor> = input_names
        .into_iter()
        .zip(inputs.iter().map(|t| (*t).clone()))
        .collect();

    let mut output_map = backend.execute(
        &graph,
        input_map,
        super::backend::BackendAttrs {
            current_node_attributes: &[],
            current_node_metadata: &[],
        },
    )?;
    let result = output_map.remove(SINGLE_OP_OUTPUT_NAME).ok_or_else(|| {
        BackendWalkError::MissingExecuteOutput {
            op_type: op_type.to_string(),
            output_name: SINGLE_OP_OUTPUT_NAME.to_string(),
        }
    })?;
    Ok(result)
}

/// Default body for multi-output per-op methods ([`Backend::split`]
/// today). Builds a one-node `GraphProto` with `output_count`
/// positionally-named outputs, calls [`Backend::execute`], and
/// extracts the outputs in declared order.
///
/// If `output_count == 0`, returns an empty `Vec` without invoking
/// `execute` — multi-output ops with zero outputs are degenerate
/// and the engine never produces such carriers.
pub fn execute_multi<B: Backend + ?Sized>(
    backend: &B,
    op_type: &str,
    inputs: &[&B::Tensor],
    attributes: Vec<AttributeProto>,
    output_count: usize,
) -> Result<Vec<B::Tensor>, B::Error> {
    if output_count == 0 {
        return Ok(Vec::new());
    }

    let input_names: Vec<String> = (0..inputs.len())
        .map(|i| format!("__bb_default_walk_in_{i}"))
        .collect();
    let output_names: Vec<String> = (0..output_count)
        .map(|i| format!("__bb_default_walk_out_{i}"))
        .collect();

    let node = NodeProto {
        op_type: op_type.to_string(),
        input: input_names.clone(),
        output: output_names.clone(),
        attribute: attributes,
        ..Default::default()
    };
    let graph = GraphProto {
        node: vec![node],
        output: output_names
            .iter()
            .map(|n| ValueInfoProto {
                name: n.clone(),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };

    let input_map: HashMap<String, B::Tensor> = input_names
        .into_iter()
        .zip(inputs.iter().map(|t| (*t).clone()))
        .collect();

    let mut output_map = backend.execute(
        &graph,
        input_map,
        super::backend::BackendAttrs {
            current_node_attributes: &[],
            current_node_metadata: &[],
        },
    )?;
    output_names
        .into_iter()
        .map(|n| {
            output_map.remove(&n).ok_or_else(|| {
                BackendWalkError::MissingExecuteOutput {
                    op_type: op_type.to_string(),
                    output_name: n,
                }
                .into()
            })
        })
        .collect()
}

/// Default body for [`Backend::execute`] — walks `graph.node` in
/// topological order, dispatching each through the typed per-op
/// methods on `backend`. The implementation is a tight linear scan:
/// ONNX guarantees `graph.node` is topologically ordered, so no
/// petgraph / explicit ordering is needed.
///
/// Op-types must be in [`bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS`].
/// A graph containing extension ops (Relu, Conv, …) needs either a
/// backend that overrides `execute` natively OR a lowering pass
/// (future work) decomposing the extensions into primitives.
pub fn execute_graph_via_per_op<B: Backend + ?Sized>(
    backend: &B,
    graph: &GraphProto,
    inputs: HashMap<String, B::Tensor>,
) -> Result<HashMap<String, B::Tensor>, B::Error> {
    let mut env: HashMap<String, B::Tensor> = inputs;

    for node in &graph.node {
        let input_tensors: Vec<&B::Tensor> = node
            .input
            .iter()
            .filter(|n| !n.is_empty())
            .map(|n| {
                env.get(n).ok_or_else(|| BackendWalkError::MissingInput {
                    op_type: node.op_type.clone(),
                    input_name: n.clone(),
                })
            })
            .collect::<Result<Vec<&B::Tensor>, BackendWalkError>>()
            .map_err(B::Error::from)?;

        let outputs = dispatch_per_op(backend, &node.op_type, &input_tensors, &node.attribute)?;

        for (i, name) in node.output.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let Some(tensor) = outputs.get(i) else {
                return Err(BackendWalkError::OutputArityMismatch {
                    op_type: node.op_type.clone(),
                    produced: outputs.len(),
                    declared: node.output.len(),
                }
                .into());
            };
            env.insert(name.clone(), tensor.clone());
        }
    }

    let mut result: HashMap<String, B::Tensor> = HashMap::new();
    for vi in &graph.output {
        if let Some(t) = env.remove(&vi.name) {
            result.insert(vi.name.clone(), t);
        }
    }
    Ok(result)
}

/// Dispatch a single `NodeProto` (whose `op_type` MUST be one of
/// the 30 `TENSOR_PRIMITIVES_OPS`) through the appropriate typed
/// per-op method on `backend`. Returns a `Vec` to handle multi-
/// output primitives (`Split`).
fn dispatch_per_op<B: Backend + ?Sized>(
    backend: &B,
    op_type: &str,
    inputs: &[&B::Tensor],
    attrs: &[AttributeProto],
) -> Result<Vec<B::Tensor>, B::Error> {
    let single = |t: B::Tensor| Ok(vec![t]);
    match op_type {
        // Arithmetic
        "Add" => single(backend.add(inputs[0], inputs[1])?),
        "Sub" => single(backend.sub(inputs[0], inputs[1])?),
        "Mul" => single(backend.mul(inputs[0], inputs[1])?),
        "Div" => single(backend.div(inputs[0], inputs[1])?),
        "Neg" => single(backend.neg(inputs[0])?),
        "Abs" => single(backend.abs(inputs[0])?),
        // Math
        "Sqrt" => single(backend.sqrt(inputs[0])?),
        "Pow" => single(backend.pow(inputs[0], inputs[1])?),
        "Exp" => single(backend.exp(inputs[0])?),
        "Log" => single(backend.log(inputs[0])?),
        // Linear algebra
        "MatMul" => single(backend.matmul(inputs[0], inputs[1])?),
        // Reductions
        "ReduceSum" => single(backend.reduce_sum(
            inputs[0],
            &attr_ints(attrs, "axes"),
            attr_int(attrs, "keepdims", 1) != 0,
        )?),
        "ReduceMean" => single(backend.reduce_mean(
            inputs[0],
            &attr_ints(attrs, "axes"),
            attr_int(attrs, "keepdims", 1) != 0,
        )?),
        "ReduceMax" => single(backend.reduce_max(
            inputs[0],
            &attr_ints(attrs, "axes"),
            attr_int(attrs, "keepdims", 1) != 0,
        )?),
        "ReduceMin" => single(backend.reduce_min(
            inputs[0],
            &attr_ints(attrs, "axes"),
            attr_int(attrs, "keepdims", 1) != 0,
        )?),
        // Shape
        "Reshape" => single(backend.reshape(inputs[0], &attr_ints(attrs, "shape"))?),
        "Transpose" => single(backend.transpose(inputs[0], &attr_ints(attrs, "perm"))?),
        "Concat" => single(backend.concat(inputs, attr_int(attrs, "axis", 0))?),
        "Slice" => single(backend.slice(
            inputs[0],
            &attr_ints(attrs, "starts"),
            &attr_ints(attrs, "ends"),
            &attr_ints(attrs, "axes"),
            &attr_ints(attrs, "steps"),
        )?),
        "Split" => Ok(backend.split(
            inputs[0],
            attr_int(attrs, "axis", 0),
            &attr_ints(attrs, "split"),
        )?),
        "Squeeze" => single(backend.squeeze(inputs[0], &attr_ints(attrs, "axes"))?),
        "Unsqueeze" => single(backend.unsqueeze(inputs[0], &attr_ints(attrs, "axes"))?),
        "Identity" => single(backend.identity(inputs[0])?),
        "Cast" => single(backend.cast(inputs[0], attr_int(attrs, "to", 1) as i32)?),
        // Comparison
        "Equal" => single(backend.equal(inputs[0], inputs[1])?),
        "Greater" => single(backend.greater(inputs[0], inputs[1])?),
        "Less" => single(backend.less(inputs[0], inputs[1])?),
        // Conditional
        "Where" => single(backend.r#where(inputs[0], inputs[1], inputs[2])?),
        // Creation
        "Constant" => single(backend.constant(attr_tensor(attrs, "value").unwrap_or_default())?),
        // Indexing
        "Gather" => single(backend.gather(inputs[0], inputs[1], attr_int(attrs, "axis", 0))?),
        other => Err(BackendWalkError::UnknownOpType(other.to_string()).into()),
    }
}

// ──────────────────────────────────────────────────────────────────
// Attribute encoders — called from the Contract per-op defaults.
// ──────────────────────────────────────────────────────────────────

/// Build an `AttributeProto` of type `INT` (per ONNX
/// [`AttributeType::Int`]). Used by the Contract per-op defaults
/// for scalar `i64` attributes (`axis`, `to`, `keepdims`).
pub fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: bb_ir::proto::onnx::attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

/// Build an `AttributeProto` of type `INTS`. Used for vector
/// attributes (`axes`, `shape`, `perm`, `starts`, `ends`, `steps`,
/// `split`).
pub fn ints_attr(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: bb_ir::proto::onnx::attribute_proto::AttributeType::Ints as i32,
        ints: values.to_vec(),
        ..Default::default()
    }
}

/// Build an `AttributeProto` of type `TENSOR`. Used for
/// `Constant`'s `value` attribute.
pub fn tensor_attr(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: bb_ir::proto::onnx::attribute_proto::AttributeType::Tensor as i32,
        t: Some(tensor),
        ..Default::default()
    }
}

// ──────────────────────────────────────────────────────────────────
// Attribute decoders — called from the walker.
// ──────────────────────────────────────────────────────────────────

fn attr_int(attrs: &[AttributeProto], name: &str, default: i64) -> i64 {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.i)
        .unwrap_or(default)
}

fn attr_ints(attrs: &[AttributeProto], name: &str) -> Vec<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
        .unwrap_or_default()
}

fn attr_tensor(attrs: &[AttributeProto], name: &str) -> Option<TensorProto> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| a.t.clone())
}

