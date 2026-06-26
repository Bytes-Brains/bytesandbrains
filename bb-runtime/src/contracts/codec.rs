//! `bb::Codec` — bidirectional storage-type bridge.
//!
//! Each method takes the engine's `&mut RuntimeResourceRef<'_>` ctx
//! plus a [`CompletionHandle`] AND returns [`ContractResponse`]. See
//! [`crate::contracts::index`] for the sync (Now) vs async (Later)
//! semantics.
//!
//! `ctx` exposes `ctx.dependency::<T>("<slot>")` so a codec impl can
//! reach any concrete declared in `#[depends(...)]` (e.g. a Backend
//! that materializes a quantizer's calibration tensors on-device).
//!
//! Library makers implement `Codec` when they want to bridge two
//! positions in the storage tree (f32 ↔ u8 quantization, f32 ↔ f16
//! half-precision lift, f32 ↔ opaque-bytes compression, …).

use crate::completion::{CompletionHandle, ContractResponse};
use crate::runtime::RuntimeResourceRef;

/// User-facing Contract trait for a typed in/out storage codec.
pub trait Codec: Send + Sync {
    /// Input storage. Position-in-tree declaration via [`bb_ir::types::Storage`].
    type In: ?Sized + bb_ir::types::Storage;

    /// Output storage. Different position in the tree from `In`
    /// (otherwise the codec is identity and the author should remove it).
    type Out: ?Sized + bb_ir::types::Storage;

    /// Library-maker-defined error type.
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    /// Optional training pass — quantizers need scale/zero-point
    /// calibration, PQ codebooks need k-means, plain dtype casts
    /// (f32 ↔ f16) skip this. Default returns `Now(Ok(()))`.
    fn train(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _samples: &[&Self::In],
        _completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }

    /// `In → Out`. `ctx` is the per-dispatch runtime surface; impls
    /// reach their declared `#[depends(...)]` siblings through
    /// [`RuntimeResourceRef::dependency`].
    fn encode(
        &self,
        ctx: &mut RuntimeResourceRef<'_>,
        input: &Self::In,
        completion: CompletionHandle<Box<Self::Out>, Self::Error>,
    ) -> ContractResponse<Box<Self::Out>, Self::Error>;

    /// `Out → In`. Lossy codecs implement the best-effort inverse.
    /// `ctx` is the per-dispatch runtime surface; impls reach their
    /// declared `#[depends(...)]` siblings through
    /// [`RuntimeResourceRef::dependency`].
    fn decode(
        &self,
        ctx: &mut RuntimeResourceRef<'_>,
        encoded: &Self::Out,
        completion: CompletionHandle<Box<Self::In>, Self::Error>,
    ) -> ContractResponse<Box<Self::In>, Self::Error>;
}

