//! Shape / structural / attribute-required kernels for the
//! `CpuBackend`. Most kernels are ~5-line wrappers around ndarray's
//! shape APIs (`into_shape_with_order`, `permuted_axes`,
//! `concatenate`, `select`, `slice_each_axis`). Output construction
//! routes through `CpuBackend::wrap_array` so the backend owns the
//! allocation hook.

use bb_ir::proto::onnx::{attribute_proto::AttributeType, AttributeProto, TensorProto};
use bb_ir::tensor::Tensor;
use bb_runtime::atomic::DispatchResult;
use bb_runtime::bus::OpError;
use bb_runtime::slot_value::SlotValue;
use ndarray::{ArrayD, ArrayViewD, Axis, Ix2, IxDyn, Slice};

use crate::backends::cpu::{CpuBackend, CpuTensor};

use super::{
    find_attr, need_int_attr, need_ints_attr, need_one_input, need_two_inputs, opt_float_attr,
    opt_int_attr,
};

fn dims_product(dims: &[i64]) -> usize {
    dims.iter().map(|d| (*d).max(0) as usize).product()
}

// --- Softmax ------------------------------------------------------
//
// `axis` attribute (default -1) selects the reduction axis. Numerical
// stability via the max-shift trick: subtract the per-axis max
// before exponentiation.

pub fn softmax(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let rank = t.as_array().ndim() as i64;
    let axis = opt_int_attr(attrs, "axis", -1);
    let axis = if axis < 0 { axis + rank } else { axis };
    if axis < 0 || axis >= rank {
        return Err(OpError {
            detail: format!("{op}: axis {axis} out of bounds for rank {rank}"),
            ..Default::default()
        });
    }
    let axis = Axis(axis as usize);
    let max: ArrayD<f32> = t
        .as_array()
        .fold_axis(axis, f32::NEG_INFINITY, |&m, &v| m.max(v));
    let shifted: ArrayD<f32> = t.as_array() - &max.insert_axis(axis);
    let exp: ArrayD<f32> = shifted.mapv(f32::exp);
    let sum: ArrayD<f32> = exp.sum_axis(axis).insert_axis(axis);
    Ok(backend.wrap_array(exp / sum))
}

// --- LeakyRelu ----------------------------------------------------

pub fn leaky_relu(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let alpha = opt_float_attr(attrs, "alpha", 0.01);
    Ok(backend.wrap_array(t.as_array().mapv(|x| if x >= 0.0 { x } else { alpha * x })))
}

// --- Gemm: alpha * (op(A) * op(B)) + beta * C ----------------------

pub fn gemm(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    if inputs.len() < 2 {
        return Err(OpError {
            detail: format!("{op}: requires at least 2 inputs"),
            ..Default::default()
        });
    }
    let a = super::as_cpu_tensor(op, "A", inputs[0].1)?;
    let b = super::as_cpu_tensor(op, "B", inputs[1].1)?;
    let c = if inputs.len() >= 3 {
        Some(super::as_cpu_tensor(op, "C", inputs[2].1)?)
    } else {
        None
    };
    let alpha = opt_float_attr(attrs, "alpha", 1.0);
    let beta = opt_float_attr(attrs, "beta", 1.0);
    let trans_a = opt_int_attr(attrs, "transA", 0) != 0;
    let trans_b = opt_int_attr(attrs, "transB", 0) != 0;
    if a.as_array().ndim() != 2 || b.as_array().ndim() != 2 {
        return Err(OpError {
            detail: format!("{op}: A and B must be 2-D"),
            ..Default::default()
        });
    }
    let a2 = a
        .as_array()
        .view()
        .into_dimensionality::<Ix2>()
        .expect("rank-2 enforced");
    let b2 = b
        .as_array()
        .view()
        .into_dimensionality::<Ix2>()
        .expect("rank-2 enforced");
    let a_op = if trans_a { a2.t() } else { a2.view() };
    let b_op = if trans_b { b2.t() } else { b2.view() };
    if a_op.shape()[1] != b_op.shape()[0] {
        return Err(OpError {
            detail: format!(
                "{op}: inner-dim mismatch ({} vs {})",
                a_op.shape()[1],
                b_op.shape()[0],
            ),
            ..Default::default()
        });
    }
    let mut out: ndarray::Array2<f32> = a_op.dot(&b_op);
    out *= alpha;
    if let Some(c) = c {
        let cv = c.as_array();
        let out_shape = out.shape().to_vec();
        if cv.shape() == out_shape.as_slice() {
            let c2 = cv
                .view()
                .into_dimensionality::<Ix2>()
                .map_err(|e| OpError {
                    detail: format!("{op}: C ndim convert: {e}"),
                    ..Default::default()
                })?;
            out.scaled_add(beta, &c2);
        } else if cv.ndim() == 1 && cv.shape()[0] == out_shape[1] {
            // Broadcast C as a row.
            for (i, row) in out.rows_mut().into_iter().enumerate() {
                let _ = i;
                let mut row = row;
                row.zip_mut_with(
                    &cv.view()
                        .into_dimensionality::<ndarray::Ix1>()
                        .expect("rank-1 enforced"),
                    |o, &v| {
                        *o += beta * v;
                    },
                );
            }
        } else {
            return Err(OpError {
                detail: format!(
                    "{op}: C shape {:?} doesn't match output {:?} or its column dim",
                    cv.shape(),
                    out_shape,
                ),
                ..Default::default()
            });
        }
    }
    Ok(backend.wrap_array(out.into_dyn()))
}

// --- Reshape ------------------------------------------------------
//
// `dims` attribute (INTS) specifies the new shape. -1 in dims means
// "infer from total element count".

pub fn reshape(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let mut dims = need_ints_attr(op, attrs, "dims")?;
    let known: usize = dims
        .iter()
        .filter(|d| **d >= 0)
        .map(|d| *d as usize)
        .product();
    let neg_one_count = dims.iter().filter(|d| **d < 0).count();
    if neg_one_count > 1 {
        return Err(OpError {
            detail: format!("{op}: dims may contain at most one -1"),
            ..Default::default()
        });
    }
    let elem_count = t.as_array().len();
    if neg_one_count == 1 {
        if known == 0 {
            return Err(OpError {
                detail: format!("{op}: cannot infer -1 when other dims include 0"),
                ..Default::default()
            });
        }
        let inferred = elem_count / known;
        for d in dims.iter_mut() {
            if *d < 0 {
                *d = inferred as i64;
            }
        }
    }
    let total = dims_product(&dims);
    if total != elem_count {
        return Err(OpError {
            detail: format!(
                "{op}: dims product {total} doesn't match input element count {elem_count}",
            ),
            ..Default::default()
        });
    }
    let new_shape: Vec<usize> = dims.iter().map(|&d| d.max(0) as usize).collect();
    let reshaped = t
        .as_array()
        .clone()
        .into_shape_with_order(IxDyn(&new_shape))
        .map_err(|e| OpError {
            detail: format!("{op}: ndarray reshape failed: {e}"),
            ..Default::default()
        })?;
    Ok(backend.wrap_array(reshaped))
}

// --- Transpose ----------------------------------------------------

pub fn transpose(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let rank = t.as_array().ndim();
    let perm: Vec<usize> = match find_attr(attrs, "perm") {
        Some(a) if a.r#type == AttributeType::Ints as i32 => {
            a.ints.iter().map(|&i| i as usize).collect()
        }
        _ => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return Err(OpError {
            detail: format!("{op}: perm length {} doesn't match rank {rank}", perm.len()),
            ..Default::default()
        });
    }
    let transposed = t.as_array().clone().permuted_axes(IxDyn(&perm));
    // `.permuted_axes` returns a view-backed array with non-standard
    // strides; materialize to dense layout so downstream ops can
    // assume row-major contiguity.
    let mut dense: ArrayD<f32> = ArrayD::zeros(transposed.raw_dim());
    dense.assign(&transposed);
    Ok(backend.wrap_array(dense))
}

// --- Concat -------------------------------------------------------

pub fn concat(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    if inputs.is_empty() {
        return Err(OpError {
            detail: format!("{op}: requires at least one input"),
            ..Default::default()
        });
    }
    let mut tensors: Vec<&CpuTensor> = Vec::with_capacity(inputs.len());
    for (i, (_, h)) in inputs.iter().enumerate() {
        tensors.push(super::as_cpu_tensor(op, &format!("input {i}"), *h)?);
    }
    let rank = tensors[0].as_array().ndim();
    let axis_raw = need_int_attr(op, attrs, "axis")?;
    let axis = if axis_raw < 0 {
        (axis_raw + rank as i64) as usize
    } else {
        axis_raw as usize
    };
    if axis >= rank {
        return Err(OpError {
            detail: format!("{op}: axis {axis} out of bounds"),
            ..Default::default()
        });
    }
    let views: Vec<ArrayViewD<f32>> = tensors.iter().map(|t| t.as_array().view()).collect();
    let view_slice: Vec<ArrayViewD<f32>> = views.iter().map(|v| v.view()).collect();
    let out = ndarray::concatenate(Axis(axis), &view_slice).map_err(|e| OpError {
        detail: format!("{op}: ndarray concatenate failed: {e}"),
        ..Default::default()
    })?;
    Ok(backend.wrap_array(out))
}

// --- Squeeze / Unsqueeze ------------------------------------------

pub fn squeeze(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let rank = t.as_array().ndim();
    let axes: Option<Vec<usize>> = find_attr(attrs, "axes")
        .filter(|a| a.r#type == AttributeType::Ints as i32)
        .map(|a| {
            a.ints
                .iter()
                .map(|i| {
                    if *i < 0 {
                        (i + rank as i64) as usize
                    } else {
                        *i as usize
                    }
                })
                .collect()
        });
    let target_axes: Vec<usize> = match axes {
        Some(axes) => axes,
        None => t
            .as_array()
            .shape()
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 1)
            .map(|(i, _)| i)
            .collect(),
    };
    // Remove axes in descending order so earlier indices stay valid.
    let mut working = t.as_array().clone();
    let mut sorted = target_axes.clone();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    for axis in sorted {
        if working.shape()[axis] != 1 {
            return Err(OpError {
                detail: format!(
                    "{op}: cannot squeeze axis {axis} of size {}",
                    working.shape()[axis],
                ),
                ..Default::default()
            });
        }
        working = working.remove_axis(Axis(axis));
    }
    Ok(backend.wrap_array(working))
}

pub fn unsqueeze(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let mut axes = need_ints_attr(op, attrs, "axes")?;
    axes.sort();
    let mut working = t.as_array().clone();
    for axis in axes {
        working.insert_axis_inplace(Axis(axis as usize));
    }
    Ok(backend.wrap_array(working))
}

// --- Cast ---------------------------------------------------------
//
// CpuBackend is f32-only - Cast is a shape-preserving identity. The
// `to` attribute is recorded so callers can detect the op shape
// even though we don't change the runtime element type.

pub fn cast(
    _backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let _to = need_int_attr(op, attrs, "to")?;
    // Arc clone — no buffer copy.
    Ok(t.clone())
}

// --- Slice --------------------------------------------------------
//
// `starts: INTS, ends: INTS` over consecutive leading axes (axes
// attribute defaults to 0..starts.len()). Steps default to 1.

pub fn slice(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    let starts = need_ints_attr(op, attrs, "starts")?;
    let ends = need_ints_attr(op, attrs, "ends")?;
    if starts.len() != ends.len() {
        return Err(OpError {
            detail: format!("{op}: starts/ends length mismatch"),
            ..Default::default()
        });
    }
    let rank = t.as_array().ndim();
    if starts.len() > rank {
        return Err(OpError {
            detail: format!("{op}: starts/ends rank exceeds input rank"),
            ..Default::default()
        });
    }
    let mut sliced = t.as_array().view();
    for (axis, (&s, &e)) in starts.iter().zip(ends.iter()).enumerate() {
        let dim = sliced.shape()[axis] as i64;
        let s = s.max(0).min(dim) as isize;
        let e = e.max(0).min(dim) as isize;
        sliced.slice_axis_inplace(Axis(axis), Slice::new(s, Some(e), 1));
    }
    // Materialize the view to an owned dense array.
    let owned: ArrayD<f32> = sliced.to_owned();
    Ok(backend.wrap_array(owned))
}

// --- Split --------------------------------------------------------

pub fn split(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<DispatchResult, OpError> {
    let t = need_one_input(op, inputs)?;
    let axis_raw = opt_int_attr(attrs, "axis", 0);
    let rank = t.as_array().ndim() as i64;
    let axis_idx = if axis_raw < 0 {
        axis_raw + rank
    } else {
        axis_raw
    } as usize;
    let split_sizes: Vec<usize> = need_ints_attr(op, attrs, "split")?
        .into_iter()
        .map(|d| d.max(0) as usize)
        .collect();
    let axis_dim = t.as_array().shape()[axis_idx];
    if split_sizes.iter().sum::<usize>() != axis_dim {
        return Err(OpError {
            detail: format!(
                "{op}: split sizes sum to {} but axis dim is {axis_dim}",
                split_sizes.iter().sum::<usize>(),
            ),
            ..Default::default()
        });
    }
    let mut outputs: Vec<(String, Box<dyn SlotValue>)> = Vec::with_capacity(split_sizes.len());
    let mut consumed: isize = 0;
    for (idx, size) in split_sizes.iter().enumerate() {
        let lo = consumed;
        let hi = consumed + *size as isize;
        let part = t
            .as_array()
            .slice_axis(Axis(axis_idx), Slice::new(lo, Some(hi), 1))
            .to_owned();
        outputs.push((format!("out_{idx}"), Box::new(backend.wrap_array(part))));
        consumed = hi;
    }
    Ok(DispatchResult::Immediate(outputs))
}

// --- Gather -------------------------------------------------------
//
// `axis` attribute selects which dim is indexed. Indices come from
// the second input. v0 supports 1-D indices.

pub fn gather(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let (data, idx) = need_two_inputs(op, inputs)?;
    let axis_raw = opt_int_attr(attrs, "axis", 0);
    let rank = data.as_array().ndim() as i64;
    let axis = if axis_raw < 0 {
        axis_raw + rank
    } else {
        axis_raw
    } as usize;
    if axis >= data.as_array().ndim() {
        return Err(OpError {
            detail: format!("{op}: axis out of bounds"),
            ..Default::default()
        });
    }
    if idx.as_array().ndim() != 1 {
        return Err(OpError {
            detail: format!("{op}: only 1-D indices supported"),
            ..Default::default()
        });
    }
    let dim = data.as_array().shape()[axis];
    let indices: Vec<usize> = idx
        .as_array()
        .iter()
        .map(|f| (*f as i64).max(0) as usize)
        .collect();
    for &i in &indices {
        if i >= dim {
            return Err(OpError {
                detail: format!("{op}: index {i} out of range {dim}"),
                ..Default::default()
            });
        }
    }
    // `select` picks the requested indices along the axis; returns an
    // owned array.
    Ok(backend.wrap_array(data.as_array().select(Axis(axis), &indices)))
}

// --- Creation: Zeros / Ones / Constant ----------------------------

pub fn zeros(
    backend: &CpuBackend,
    op: &str,
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let dims = need_ints_attr(op, attrs, "dims")?;
    Ok(backend.alloc_tensor(dims))
}

pub fn ones(
    _backend: &CpuBackend,
    op: &str,
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let dims = need_ints_attr(op, attrs, "dims")?;
    Ok(CpuTensor::ones(dims))
}

pub fn constant(
    _backend: &CpuBackend,
    op: &str,
    attrs: &[AttributeProto],
) -> Result<CpuTensor, OpError> {
    let attr = find_attr(attrs, "value").ok_or_else(|| OpError {
        detail: format!("{op}: missing `value` attribute"),
        ..Default::default()
    })?;
    if attr.r#type != AttributeType::Tensor as i32 {
        return Err(OpError {
            detail: format!("{op}: `value` attribute must be TENSOR"),
            ..Default::default()
        });
    }
    let proto: TensorProto = attr.t.clone().ok_or_else(|| OpError {
        detail: format!("{op}: `value` attribute has no TensorProto"),
        ..Default::default()
    })?;
    CpuTensor::from_proto(proto).map_err(|e| OpError {
        detail: format!("{op}: TensorProto decode failed: {e:?}"),
        ..Default::default()
    })
}

