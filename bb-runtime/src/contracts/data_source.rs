//! `bb::DataSource` — Contract trait for data loaders.
//!
//! Each method takes the engine's `&mut RuntimeResourceRef<'_>` ctx
//! plus a [`CompletionHandle`] AND returns [`ContractResponse`]. See
//! [`crate::contracts::index`] for the sync (Now) vs async (Later)
//! semantics.
//!
//! `ctx` exposes `ctx.dependency::<T>("<slot>")` for the source to
//! reach any concrete declared in `#[depends(...)]` (e.g. a Backend
//! to materialize the batch tensor on-device).
//!
//! ## Associated type: `Sample`
//!
//! `Sample` is the element type for both the batch tensor and the
//! optional labels tensor returned by [`DataSource::next_batch`].
//! Implement as `[f32]` for flat f32 sample batches. The second
//! slot (`Box<Self::Sample>`) holds labels; unsupervised sources
//! return a zero-length boxed slice.

use crate::completion::{CompletionHandle, ContractResponse};
use crate::runtime::RuntimeResourceRef;

/// User-facing Contract trait for a data source / data loader.
pub trait DataSource: Send + Sync {
    /// Sample storage type. One associated type covers both the batch
    /// tensor and the optional labels tensor. Implement as `[f32]` for
    /// flat f32 sample batches.
    type Sample: ?Sized + bb_ir::types::Storage;
    /// Library-maker-defined error type.
    type Error: std::error::Error + std::fmt::Display + Send + Sync + 'static;

    /// Fetch the next batch. Returns `(batch, labels)` as boxed
    /// `Self::Sample` slices; the second slot is zero-length for
    /// unsupervised sources.
    // ?Sized + Storage DST tuple `(Box<Self::Sample>, Box<Self::Sample>)` in CompletionHandle's generic param trips the type-complexity lint; unavoidable for DST-bound associated type.
    #[allow(clippy::type_complexity)]
    fn next_batch(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<(Box<Self::Sample>, Box<Self::Sample>), Self::Error>,
    ) -> ContractResponse<(Box<Self::Sample>, Box<Self::Sample>), Self::Error>;

    /// Reset to the beginning of the data stream.
    fn reset(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;

    /// One-shot notification signalling the source's data has
    /// finished loading (e.g. dataset download complete).
    fn on_data_loaded(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error>;
}
