//! The universal `SlotValue` trait — every value flowing through slot
//! sites (DSL outputs, wire payloads, syscall returns, role-method
//! returns) implements it via the blanket `impl<T: Tensor> SlotValue`
//! and per-primitive impls.
//!
//! Canonical home for `bb-runtime`. Also hosts engine-side carriers
//! that step outside the serde-driven blanket impl path (today:
//! [`BackendTensorCarrier`], which holds a type-erased backend-owned
//! tensor handle).

use std::any::Any;

pub use bb_ir::slot_value::*;

use crate::ids::ComponentRef;

/// Engine-internal `SlotValue` wrapping a backend-native tensor
/// behind a type-erased handle. Built by the engine's wire-decode
/// path (`decode_typed_fill` backend-mediated branch) from the
/// `Backend::materialize_from_wire` result; downstream graph ops
/// downcast `inner` to the backend's `Self::Tensor` to read the
/// value.
///
/// Lifecycle:
///
/// 1. Engine reads an inbound tensor `SlotFill`, charges its bytes
///    against `NodeConfig::ingress_byte_budget`, and hands the
///    `Vec<u8>` to the backend bound to the destination slot via
///    [`crate::roles::BackendRuntime::materialize_from_wire`].
/// 2. Backend returns a `Box<dyn SlotValue>` containing this
///    carrier; the engine installs it in the slot table.
/// 3. On slot overwrite / eviction, the writer reads
///    [`SlotValue::charged_bytes`] (returns `self.charged_bytes`)
///    and releases the budget against the engine counter.
///
/// `clone_fn` + `wire_encode_fn` carry the per-`T::Tensor` clone /
/// re-encode shape so the carrier supports the universal
/// `SlotValue` contract without a `Clone` / `Serialize` bound on
/// the type-erased `inner`. The `#[derive(bb::Backend)]` derive
/// captures `T` at the call site and stores these fn pointers in
/// the carrier; intra-Node clones (`clone_boxed`) and re-encodes
/// (`to_wire_bytes`) route through them.
pub struct BackendTensorCarrier {
    /// Backend's native tensor, type-erased. Internally Arc-shared
    /// (the backend's `Tensor` impl chooses the strategy — see
    /// `CpuTensor(Arc<CpuBackendBuffer>)`); cloning the carrier
    /// invokes `clone_fn` which calls the typed `Clone` impl, which
    /// is an `Arc::clone` for pooling-friendly backends.
    pub(crate) inner: Box<dyn Any + Send + Sync>,
    /// Per-`T` clone bridge. Reads the erased `inner` as `&T` and
    /// returns a `Box<dyn Any>` over a fresh `T::clone()`. Captured
    /// at materialize-time so the carrier stays dyn-safe without a
    /// `Clone` bound on `dyn Any`.
    pub(crate) clone_fn: fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync>,
    /// Per-`T` wire-encode bridge. Reads the erased `inner` as `&T`
    /// and returns the bincode payload bytes. Mirrors the blanket
    /// `SlotValue::to_wire_bytes` impl for `T: Serialize` but lives
    /// here so the carrier can re-encode through the same path the
    /// sender used.
    pub(crate) wire_encode_fn: fn(&(dyn Any + Send + Sync)) -> Result<Vec<u8>, SlotValueError>,
    /// Wire-type hash this carrier originated from. Receivers
    /// validate downcast targets and re-encode against this; senders
    /// stamp it into outbound `SlotFill.type_hash`.
    pub(crate) type_hash: u64,
    /// Bytes admitted against `NodeConfig::ingress_byte_budget` at
    /// receive time. The slot-table writer releases these on
    /// overwrite / eviction via [`SlotValue::charged_bytes`].
    pub(crate) charged_bytes: usize,
    /// `ComponentRef` of the backend that produced this carrier.
    /// `decode_typed_fill` stamps the source backend so future
    /// re-encode / forwarding paths can route through the same
    /// backend instance.
    pub(crate) backend_ref: ComponentRef,
}

impl BackendTensorCarrier {
    /// Construct a carrier from the backend's already-typed
    /// `Self::Tensor`. The `#[derive(bb::Backend)]` materialize
    /// bridge is the canonical caller; the constructor is `pub` so
    /// derive expansions in downstream crates can call it, but the
    /// engine-side fields (`charged_bytes`, `backend_ref`) get
    /// stamped via [`Self::stamp_engine_fields`] immediately after
    /// the bridge returns so authoring code never holds a carrier
    /// with stale accounting.
    pub fn from_typed<T>(
        tensor: T,
        type_hash: u64,
        charged_bytes: usize,
        backend_ref: ComponentRef,
    ) -> Self
    where
        T: Any + Send + Sync + Clone + serde::Serialize + 'static,
    {
        Self {
            inner: Box::new(tensor),
            clone_fn: |any| {
                let t: &T = any.downcast_ref::<T>().expect("inner is T by construction");
                Box::new(t.clone())
            },
            wire_encode_fn: |any| {
                let t: &T = any.downcast_ref::<T>().expect("inner is T by construction");
                bincode::serialize(t).map_err(|e| SlotValueError::EncodeFailed(Box::new(e)))
            },
            type_hash,
            charged_bytes,
            backend_ref,
        }
    }

    /// Borrow the carrier's wire-type hash. Used by the wire-encode
    /// path and by tests that assert a fill's type discriminator
    /// round-trips through the carrier.
    pub fn type_hash(&self) -> u64 {
        self.type_hash
    }

    /// Borrow the producing backend's `ComponentRef`. Used by
    /// re-encode + introspection.
    pub fn backend_ref(&self) -> ComponentRef {
        self.backend_ref
    }

    /// Downcast the type-erased inner tensor to the backend's
    /// concrete `Self::Tensor`. Engine consumers reach the tensor
    /// through this accessor; the inner field stays
    /// `pub(crate)` so external code can't dodge the downcast +
    /// type-hash validation step.
    pub fn downcast_inner<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for BackendTensorCarrier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendTensorCarrier")
            .field("type_hash", &format_args!("{:#018x}", self.type_hash))
            .field("charged_bytes", &self.charged_bytes)
            .field("backend_ref", &self.backend_ref)
            .finish()
    }
}

impl SlotValue for BackendTensorCarrier {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any_boxed(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }

    fn clone_boxed(&self) -> Box<dyn SlotValue> {
        Box::new(Self {
            inner: (self.clone_fn)(&*self.inner),
            clone_fn: self.clone_fn,
            wire_encode_fn: self.wire_encode_fn,
            type_hash: self.type_hash,
            charged_bytes: self.charged_bytes,
            backend_ref: self.backend_ref,
        })
    }

    fn to_wire_bytes(&self) -> Result<Vec<u8>, SlotValueError> {
        (self.wire_encode_fn)(&*self.inner)
    }

    fn type_hash(&self) -> u64 {
        self.type_hash
    }

    fn charged_bytes(&self) -> usize {
        self.charged_bytes
    }
}

/// Typed error surfaced by
/// [`crate::roles::BackendRuntime::materialize_from_wire`]. The
/// derive bridge converts the backend's typed
/// `<T as crate::contracts::Backend>::Error` to this through
/// `Display`; the engine maps it onto
/// [`crate::bus::WireReceiveErrorKind::BackendMaterializeFailed`].
#[derive(Debug, Clone)]
pub struct BackendMaterializeError {
    /// Short `Display` of the backend's typed error.
    pub summary: String,
}

impl std::fmt::Display for BackendMaterializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Backend::materialize_from_wire: {}", self.summary)
    }
}

impl std::error::Error for BackendMaterializeError {}

