//! Per-op kernel dispatch for the `CpuBackend`.
//!
//! The 51-op `ai.onnx v1` catalog is grouped into category modules.
//! This module routes each declared op to its category-specific
//! kernel; categories that don't yet have implementations return a
//! clear "not yet implemented" error.
//!
//! **Attribute access limitation.** `BackendRuntime::dispatch_atomic`
//! receives `(op_type, inputs, attrs)` - no NodeProto, no attribute
//! map. Ops whose semantics depend on attributes (Reshape's `dims`,
//! Softmax's `axis`, Conv's `kernel_shape`, etc.) cannot be fully
//! implemented within this dispatch shape. They report
//! `RequiresAttributes` so callers can route them through a
//! follow-up channel.

mod elementwise;
mod linalg;
mod shape;

use bb_ir::proto::onnx::{attribute_proto::AttributeType, AttributeProto};
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::slot_value::SlotValue;

use crate::backends::cpu::CpuBackend;
use crate::backends::cpu::CpuTensor;

/// Find an attribute by name in the supplied `attrs` slice.
fn find_attr<'a>(attrs: &'a [AttributeProto], name: &str) -> Option<&'a AttributeProto> {
    attrs.iter().find(|a| a.name == name)
}

fn need_int_attr(op: &str, attrs: &[AttributeProto], name: &str) -> Result<i64, OpError> {
    let a = find_attr(attrs, name).ok_or_else(|| OpError {
        detail: format!("{op}: missing `{name}` attribute"),
        ..Default::default()
    })?;
    if a.r#type != AttributeType::Int as i32 {
        return Err(OpError {
            detail: format!("{op}: `{name}` attribute must be INT"),
            ..Default::default()
        });
    }
    Ok(a.i)
}

fn need_ints_attr(op: &str, attrs: &[AttributeProto], name: &str) -> Result<Vec<i64>, OpError> {
    let a = find_attr(attrs, name).ok_or_else(|| OpError {
        detail: format!("{op}: missing `{name}` attribute"),
        ..Default::default()
    })?;
    if a.r#type != AttributeType::Ints as i32 {
        return Err(OpError {
            detail: format!("{op}: `{name}` attribute must be INTS"),
            ..Default::default()
        });
    }
    Ok(a.ints.clone())
}

fn opt_float_attr(attrs: &[AttributeProto], name: &str, default: f32) -> f32 {
    find_attr(attrs, name)
        .filter(|a| a.r#type == AttributeType::Float as i32)
        .map(|a| a.f)
        .unwrap_or(default)
}

fn opt_int_attr(attrs: &[AttributeProto], name: &str, default: i64) -> i64 {
    find_attr(attrs, name)
        .filter(|a| a.r#type == AttributeType::Int as i32)
        .map(|a| a.i)
        .unwrap_or(default)
}

/// Downcast a `&dyn SlotValue` to `&CpuTensor`, raising `OpError`
/// with a clear message on type mismatch.
fn as_cpu_tensor<'a>(op: &str, role: &str, h: &'a dyn SlotValue) -> Result<&'a CpuTensor, OpError> {
    h.as_any()
        .downcast_ref::<CpuTensor>()
        .ok_or_else(|| OpError {
            detail: format!("{op}: {role} is not a CpuTensor"),
            ..Default::default()
        })
}

fn need_two_inputs<'a>(
    op: &str,
    inputs: &'a [(&str, &dyn SlotValue)],
) -> Result<(&'a CpuTensor, &'a CpuTensor), OpError> {
    if inputs.len() < 2 {
        return Err(OpError {
            detail: format!("{op}: requires two inputs, got {}", inputs.len()),
            ..Default::default()
        });
    }
    let a = as_cpu_tensor(op, "input 0", inputs[0].1)?;
    let b = as_cpu_tensor(op, "input 1", inputs[1].1)?;
    Ok((a, b))
}

fn need_one_input<'a>(
    op: &str,
    inputs: &'a [(&str, &dyn SlotValue)],
) -> Result<&'a CpuTensor, OpError> {
    if inputs.is_empty() {
        return Err(OpError {
            detail: format!("{op}: requires one input, got 0"),
            ..Default::default()
        });
    }
    as_cpu_tensor(op, "input 0", inputs[0].1)
}

fn out(name: &str, tensor: CpuTensor) -> DispatchResult {
    DispatchResult::Immediate(vec![(name.to_string(), Box::new(tensor))])
}

/// Route `op_type` to the matching kernel. `attrs` is the
/// NodeProto's attribute slice — ops whose semantics depend on
/// attributes (Reshape's `dims`, Softmax's `axis`, Gemm's
/// `alpha`/`beta`, …) read it directly. The framework's per-op
/// dispatch path threads `ctx.current.node_attributes` here; the
/// `execute_graph` walker threads `node.attribute` per node.
pub fn dispatch(
    backend: &CpuBackend,
    op_type: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<DispatchResult, OpError> {
    match op_type {
        // --- Element-wise binary -----------------------------------
        "Add" => Ok(out("C", elementwise::add(backend, op_type, inputs)?)),
        "Sub" => Ok(out("C", elementwise::sub(backend, op_type, inputs)?)),
        "Mul" => Ok(out("C", elementwise::mul(backend, op_type, inputs)?)),
        "Div" => Ok(out("C", elementwise::div(backend, op_type, inputs)?)),
        "Pow" => Ok(out("C", elementwise::pow(backend, op_type, inputs)?)),

        // --- Element-wise unary ------------------------------------
        "Neg" => Ok(out("Y", elementwise::neg(backend, op_type, inputs)?)),
        "Abs" => Ok(out("Y", elementwise::abs(backend, op_type, inputs)?)),
        "Sqrt" => Ok(out("Y", elementwise::sqrt(backend, op_type, inputs)?)),
        "Exp" => Ok(out("Y", elementwise::exp(backend, op_type, inputs)?)),
        "Log" => Ok(out("Y", elementwise::log(backend, op_type, inputs)?)),

        // --- Activations -------------------------------------------
        "Relu" => Ok(out("Y", elementwise::relu(backend, op_type, inputs)?)),
        "Sigmoid" => Ok(out("Y", elementwise::sigmoid(backend, op_type, inputs)?)),
        "Tanh" => Ok(out("Y", elementwise::tanh(backend, op_type, inputs)?)),
        "Gelu" => Ok(out("Y", elementwise::gelu(backend, op_type, inputs)?)),
        "Identity" => Ok(out("Y", elementwise::identity(backend, op_type, inputs)?)),
        "Softmax" => Ok(out("Y", shape::softmax(backend, op_type, inputs, attrs)?)),
        "LeakyRelu" => Ok(out(
            "Y",
            shape::leaky_relu(backend, op_type, inputs, attrs)?,
        )),

        // --- Element-wise comparison -------------------------------
        "Equal" => Ok(out("C", elementwise::equal(backend, op_type, inputs)?)),
        "Greater" => Ok(out("C", elementwise::greater(backend, op_type, inputs)?)),
        "Less" => Ok(out("C", elementwise::less(backend, op_type, inputs)?)),

        // --- Linear algebra ----------------------------------------
        "MatMul" => Ok(out("Y", linalg::matmul(backend, op_type, inputs)?)),
        "Dot" => Ok(out("Y", linalg::dot(backend, op_type, inputs)?)),
        "Gemm" => Ok(out("Y", shape::gemm(backend, op_type, inputs, attrs)?)),

        // --- Reductions --------------------------------------------
        "ReduceSum" => Ok(out("Y", linalg::reduce_sum(backend, op_type, inputs)?)),
        "ReduceMean" => Ok(out("Y", linalg::reduce_mean(backend, op_type, inputs)?)),
        "ReduceMax" => Ok(out("Y", linalg::reduce_max(backend, op_type, inputs)?)),
        "ReduceMin" => Ok(out("Y", linalg::reduce_min(backend, op_type, inputs)?)),

        // --- Shape / structural ------------------------------------
        "Reshape" => Ok(out("Y", shape::reshape(backend, op_type, inputs, attrs)?)),
        "Transpose" => Ok(out("Y", shape::transpose(backend, op_type, inputs, attrs)?)),
        "Concat" => Ok(out("Y", shape::concat(backend, op_type, inputs, attrs)?)),
        "Squeeze" => Ok(out("Y", shape::squeeze(backend, op_type, inputs, attrs)?)),
        "Unsqueeze" => Ok(out("Y", shape::unsqueeze(backend, op_type, inputs, attrs)?)),
        "Cast" => Ok(out("Y", shape::cast(backend, op_type, inputs, attrs)?)),
        "Slice" => Ok(out("Y", shape::slice(backend, op_type, inputs, attrs)?)),
        "Split" => Ok(shape::split(backend, op_type, inputs, attrs)?),

        // --- Indexing ----------------------------------------------
        "Gather" => Ok(out("Y", shape::gather(backend, op_type, inputs, attrs)?)),

        // --- Pooling without attributes ----------------------------
        "GlobalAveragePool" => Ok(out(
            "Y",
            linalg::global_average_pool(backend, op_type, inputs)?,
        )),

        // --- Creation ----------------------------------------------
        "Zeros" => Ok(out("Y", shape::zeros(backend, op_type, attrs)?)),
        "Ones" => Ok(out("Y", shape::ones(backend, op_type, attrs)?)),
        "Constant" => Ok(out("Y", shape::constant(backend, op_type, attrs)?)),

        // `BackendSubgraph` is routed by the engine's
        // `invoke_backend_subgraph` path, not this dispatch.
        // BatchNorm / LayerNorm / Conv / MaxPool / AveragePool /
        // Scatter / If / Loop are NOT in this backend's declared
        // opset — `Node::ready`'s per-`BackendSubgraph` check
        // catches them at install time. They surface here only as
        // a defensive fallthrough.
        other => Err(OpError {
            detail: format!("CpuBackend: unsupported op_type '{other}'"),
            ..Default::default()
        }),
    }
}

