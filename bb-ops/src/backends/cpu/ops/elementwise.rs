//! Element-wise binary, unary, comparison, and activation kernels
//! for the `CpuBackend`. Each kernel produces an output `ArrayD<f32>`
//! and routes the wrap through `CpuBackend::wrap_array` so a future
//! pool / arena hook composes at a single backend-owned point.

use bb_runtime::bus::OpError;
use bb_runtime::slot_value::SlotValue;
use ndarray::ArrayD;
use ndarray::Zip;

use crate::backends::cpu::{CpuBackend, CpuTensor};

use super::{need_one_input, need_two_inputs};

fn ensure_same_shape(op: &str, a: &CpuTensor, b: &CpuTensor) -> Result<(), OpError> {
    if a.as_array().shape() != b.as_array().shape() {
        return Err(OpError {
            detail: format!(
                "{op}: shape mismatch ({:?} vs {:?}); broadcasting not yet implemented",
                a.as_array().shape(),
                b.as_array().shape(),
            ),
            ..Default::default()
        });
    }
    Ok(())
}

fn binary_map(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    f: impl Fn(f32, f32) -> f32,
) -> Result<CpuTensor, OpError> {
    let (a, b) = need_two_inputs(op, inputs)?;
    ensure_same_shape(op, a, b)?;
    let mut out: ArrayD<f32> = ArrayD::zeros(a.as_array().raw_dim());
    Zip::from(&mut out)
        .and(a.as_array())
        .and(b.as_array())
        .for_each(|o, &x, &y| *o = f(x, y));
    Ok(backend.wrap_array(out))
}

fn unary_map(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
    f: impl Fn(f32) -> f32,
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    Ok(backend.wrap_array(t.as_array().mapv(f)))
}

// --- Binary arithmetic ---------------------------------------------

pub fn add(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| a + b)
}

pub fn sub(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| a - b)
}

pub fn mul(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| a * b)
}

pub fn div(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| a / b)
}

pub fn pow(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| a.powf(b))
}

// --- Unary arithmetic ----------------------------------------------

pub fn neg(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| -x)
}

pub fn abs(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| x.abs())
}

pub fn sqrt(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| x.sqrt())
}

pub fn exp(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| x.exp())
}

pub fn log(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| x.ln())
}

// --- Activations ---------------------------------------------------

pub fn relu(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    // NaN-propagating Relu — IEEE 754's `max(NaN, 0)` returns 0,
    // silently swallowing the poisoned element. Preserving NaN
    // matches PyTorch / NumPy semantics so downstream consumers
    // see the upstream error.
    unary_map(backend, op, inputs, |x| {
        if x.is_nan() {
            f32::NAN
        } else {
            x.max(0.0)
        }
    })
}

pub fn sigmoid(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| 1.0 / (1.0 + (-x).exp()))
}

pub fn tanh(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    unary_map(backend, op, inputs, |x| x.tanh())
}

/// Gelu via the tanh approximation (Hendrycks & Gimpel 2016):
///   `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
pub fn gelu(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    let coeff = (2.0_f32 / std::f32::consts::PI).sqrt();
    unary_map(backend, op, inputs, |x| {
        0.5 * x * (1.0 + (coeff * (x + 0.044715 * x * x * x)).tanh())
    })
}

pub fn identity(
    _backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    let t = need_one_input(op, inputs)?;
    // Arc clone — no buffer copy.
    Ok(t.clone())
}

// --- Comparison ----------------------------------------------------
//
// ONNX `Equal` / `Greater` / `Less` output a BOOL tensor. 's
// CpuBackend is f32-only, so we encode the boolean as 1.0 / 0.0 —
// downstream consumers can `Cast(BOOL)` once that op lands.

pub fn equal(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| if a == b { 1.0 } else { 0.0 })
}

pub fn greater(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| if a > b { 1.0 } else { 0.0 })
}

pub fn less(
    backend: &CpuBackend,
    op: &str,
    inputs: &[(&str, &dyn SlotValue)],
) -> Result<CpuTensor, OpError> {
    binary_map(backend, op, inputs, |a, b| if a < b { 1.0 } else { 0.0 })
}

