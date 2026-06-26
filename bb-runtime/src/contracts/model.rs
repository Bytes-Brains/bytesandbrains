//! `bb::Model` — Contract trait for ML models.
//!
//! Each method takes the engine's `&mut RuntimeResourceRef<'_>` ctx
//! plus a [`CompletionHandle`] AND returns [`ContractResponse`]. See
//! [`crate::contracts::index`] for the sync (Now) vs async (Later)
//! semantics.
//!
//! ## Associated type: `Tensor`
//!
//! One associated type covers input tensors, output tensors, parameter
//! vectors, gradients, and deltas. Mixed-precision (e.g. f32 input +
//! f16 weights + f32 output) is handled by wiring [`Codec`] nodes
//! around the model in the Module body — not by multiplying associated
//! types per port.
//!
//! [`Codec`]: crate::contracts::Codec

use crate::completion::{CompletionHandle, ContractResponse};
use crate::runtime::RuntimeResourceRef;

/// User-facing Contract trait for an ML model.
pub trait Model: Send + Sync {
    /// Tensor storage type. One associated type covers
    /// input/output/params/grad/delta. Implement as `[f32]` for
    /// flat f32 tensors.
    type Tensor: ?Sized + bb_ir::types::Storage;
    /// Library-maker-defined error type.
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    /// Forward pass: `input → output`. `ctx` is the per-dispatch
    /// runtime surface; impls reach their declared `#[depends(...)]`
    /// siblings through [`RuntimeResourceRef::dependency`].
    fn forward(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        input: &Self::Tensor,
        completion: CompletionHandle<Box<Self::Tensor>, Self::Error>,
    ) -> ContractResponse<Box<Self::Tensor>, Self::Error>;

    /// Load parameters wholesale.
    fn load_parameters(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        params: &Self::Tensor,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    /// Backward pass: accumulate gradients given upstream gradient.
    fn backward(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        grad: &Self::Tensor,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    /// Apply a parameter delta in-place.
    fn apply_delta(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        delta: &Self::Tensor,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    /// Compute loss: `(input, target) → scalar score`. Returns `f32`
    /// regardless of the tensor element type — loss is always a
    /// framework-fixed scalar.
    fn compute_loss(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        input: &Self::Tensor,
        target: &Self::Tensor,
        completion: CompletionHandle<f32, Self::Error>,
    ) -> ContractResponse<f32, Self::Error>;

    /// Snapshot the current parameter tensor (owned — async
    /// serialization needs owned values).
    fn params(
        &self,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<Box<Self::Tensor>, Self::Error>,
    ) -> ContractResponse<Box<Self::Tensor>, Self::Error>;
}
