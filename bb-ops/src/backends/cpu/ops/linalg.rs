//! Linear-algebra + reduction + pooling kernels for the
//! `CpuBackend`. ndarray's `Dot`, `sum_axis`, `mean_axis`, and
//! `fold_axis` carry the math; each kernel is a thin wrapper that
//! routes its output array through `CpuBackend::wrap_array` so the
//! backend owns the allocation hook.

use bb_runtime::bus::OpError;
use bb_runtime::slot_value::SlotValue;
use ndarray::{Array, ArrayD, Axis, Ix2, IxDyn};

use crate::backends::cpu::{CpuBackend, CpuTensor};

use super::{need_one_input, need_two_inputs};

// --- MatMul -------------------------------------------------------
//
// ONNX semantics: standard matrix multiplication. For 2-D inputs
// `A: [M, K]`, `B: [K, N]` → `Y: [M, N]`. Higher-rank tensors batch
// over the leading dims; v0 ships the 2-D case only.

pub fn matmul(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    let (a, b) = need_two_inputs(op, inputs)?;
    let a_dims = a.as_array().shape().to_vec();
    let b_dims = b.as_array().shape().to_vec();
    if a_dims.len() != 2 || b_dims.len() != 2 {
        return Err(OpError {
            detail: format!("{op}: only 2-D inputs supported (got {a_dims:?} and {b_dims:?})"),
            ..Default::default()
        });
    }
    if a_dims[1] != b_dims[0] {
        return Err(OpError {
            detail: format!(
                "{op}: inner-dim mismatch (A has K={}, B has K={})",
                a_dims[1], b_dims[0],
            ),
            ..Default::default()
        });
    }
    let a2 = a
        .as_array()
        .view()
        .into_dimensionality::<Ix2>()
        .expect("rank-2 enforced above");
    let b2 = b
        .as_array()
        .view()
        .into_dimensionality::<Ix2>()
        .expect("rank-2 enforced above");
    Ok(backend.wrap_array(a2.dot(&b2).into_dyn()))
}

// --- Dot ----------------------------------------------------------
//
// Reduces along the LAST axis. For 1-D inputs returns a scalar
// (rank-0 tensor); for higher-rank inputs the contracted axis is
// dropped.

pub fn dot(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    let (a, b) = need_two_inputs(op, inputs)?;
    if a.as_array().shape() != b.as_array().shape() {
        return Err(OpError {
            detail: format!(
                "{op}: shape mismatch ({:?} vs {:?})",
                a.as_array().shape(),
                b.as_array().shape(),
            ),
            ..Default::default()
        });
    }
    if a.as_array().shape().is_empty() {
        return Err(OpError {
            detail: format!("{op}: rank-0 inputs not supported"),
            ..Default::default()
        });
    }
    let last_axis = a.as_array().ndim() - 1;
    let product: ArrayD<f32> = a.as_array() * b.as_array();
    let summed: ArrayD<f32> = product.sum_axis(Axis(last_axis));
    Ok(backend.wrap_array(summed))
}

// --- Reductions (default: all dims, keepdims=false) ---------------
//
// ONNX `ReduceSum` / `ReduceMean` / `ReduceMax` / `ReduceMin`
// without explicit `axes` reduce across every dim. We collapse to
// a rank-0 tensor matching that default.

fn reduce(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    finalize: impl Fn(&ArrayD<f32>) -> f32,
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    if t.as_array().is_empty() {
        return Err(OpError {
            detail: format!("{op}: empty input"),
            ..Default::default()
        });
    }
    let scalar = finalize(t.as_array());
    let out: ArrayD<f32> = Array::from_elem(IxDyn(&[]), scalar);
    Ok(backend.wrap_array(out))
}

pub fn reduce_sum(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    reduce(backend, op, inputs, |a| a.sum())
}

pub fn reduce_mean(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    reduce(backend, op, inputs, |a| a.mean().unwrap_or(0.0))
}

pub fn reduce_max(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    // NaN-propagating: any NaN element poisons the result.
    reduce(backend, op, inputs, |a| {
        a.iter().copied().fold(f32::NEG_INFINITY, |acc, v| {
            if acc.is_nan() || v.is_nan() {
                f32::NAN
            } else {
                acc.max(v)
            }
        })
    })
}

pub fn reduce_min(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    reduce(backend, op, inputs, |a| {
        a.iter().copied().fold(f32::INFINITY, |acc, v| {
            if acc.is_nan() || v.is_nan() {
                f32::NAN
            } else {
                acc.min(v)
            }
        })
    })
}

// --- GlobalAveragePool ---------------------------------------------
//
// Reduces across the spatial dimensions, collapsing them to length-1.
// Input shape `[N, C, H, W, ...]` → output `[N, C, 1, 1, ...]`.

pub fn global_average_pool(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let shape: Vec<usize> = t.as_array().shape().to_vec();
    if shape.len() < 3 {
        return Err(OpError {
            detail: format!("{op}: input must have rank >= 3, got {shape:?}"),
            ..Default::default()
        });
    }
    let spatial: usize = shape[2..].iter().product();
    if spatial == 0 {
        return Err(OpError {
            detail: format!("{op}: spatial dims must be non-zero"),
            ..Default::default()
        });
    }
    // Reduce along every spatial axis (axis 2 .. ndim) by mean.
    // ndarray's `mean_axis` drops the reduced axis; we re-insert
    // length-1 axes to match ONNX's output shape.
    let mut working: ArrayD<f32> = t.as_array().clone();
    for _ in 2..shape.len() {
        working = working.mean_axis(Axis(2)).expect("non-empty axis");
    }
    // working now has shape [N, C]; insert (shape.len() - 2)
    // length-1 axes at the tail.
    for axis in 2..shape.len() {
        working.insert_axis_inplace(Axis(axis));
    }
    Ok(backend.wrap_array(working))
}

