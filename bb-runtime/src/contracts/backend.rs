//! `bb::Backend` — Contract trait for tensor compute backends.
//!
//! The Contract has THREE surfaces, exposed side-by-side:
//!
//! 1. **One typed method per mandatory primitive op** (the 30
//!    entries in [`bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS`]):
//!    `add`, `mul`, `matmul`, `reduce_sum`, `reshape`, …. Components
//!    reach these inline through `ctx.backends` for short-form
//!    tensor math — an Index's distance kernel calls
//!    `backend.matmul(&query, &vectors)?` instead of hand-rolling
//!    a loop.
//!
//! 2. **One method to execute a subgraph**:
//!    `execute(&GraphProto, HashMap<String, Tensor>, BackendAttrs) →
//!    HashMap<String, Tensor>`. Backends that prefer whole-graph
//!    dispatch override this entry point; the per-op defaults call
//!    through to it via a one-node `GraphProto`.
//!
//! 3. **One dispatch entry point for `BackendSubgraph` carriers**:
//!    `dispatch(&GraphProto, inputs, attrs, completion) →
//!    ContractResponse`. The engine calls this for every
//!    `BackendSubgraph` carrier op. The default falls through to
//!    `execute` synchronously. Backends with per-subgraph caching,
//!    JIT compilation, or async device execution override `dispatch`
//!    to return `ContractResponse::Later` while device work runs.
//!
//! ### How the two sides compose
//!
//! Default impls in [`crate::contracts::backend_default_walk`]
//! bridge the surfaces so a backend author overrides only the side
//! that's natural for their target:
//!
//! - **CpuBackend** overrides the 30 per-op methods directly
//!   (`add` runs ndarray's `Add` impl, `matmul` runs `dot`, …).
//!   It does NOT override `execute` — the default walker uses
//!   the overridden per-op methods.
//!
//! - **A Burn-style backend** overrides `execute` natively (Burn
//!   compiles the whole `GraphProto` to its own IR + runs once).
//!   It does NOT override per-op methods — they default-wrap a
//!   one-node `GraphProto` and call `execute`.
//!
//! Backends overriding *neither* side stack-overflow on the first
//! call: every per-op default wraps into `execute`, whose default
//! walks back to per-op, ad infinitum. Backends MUST override at
//! least one side.
//!
//! ### Extension ops
//!
//! Activation functions (Relu, Sigmoid, Softmax), pooling (MaxPool,
//! AveragePool), normalization (BatchNormalization, LayerNorm),
//! Conv, and so on are NOT on the Contract surface. They're
//! extensions — a backend MAY declare them via
//! [`crate::roles::BackendRuntime::extension_opsets`] and handle
//! them through its own `execute` override; OR a future lowering
//! pass decomposes them into primitives so the Contract surface
//! covers any graph.

use std::collections::HashMap;

use bb_ir::proto::onnx::{AttributeProto, GraphProto, StringStringEntryProto, TensorProto};

use crate::completion::{CompletionHandle, ContractResponse};
use crate::contracts::backend_default_walk;

/// Per-call NodeProto context surfaced to `Backend::execute` so
/// kernels overriding the whole-graph path see the original call
/// site's attributes + metadata alongside the body. Per-op
/// methods (which receive their attributes positionally as typed
/// args) don't need this struct.
pub struct BackendAttrs<'a> {
    /// Attribute list from the call NodeProto's `attribute` field.
    pub current_node_attributes: &'a [AttributeProto],
    /// `metadata_props` from the call NodeProto.
    pub current_node_metadata: &'a [StringStringEntryProto],
}

/// User-facing Contract trait for a tensor compute backend.
///
/// The `Tensor` associated type lets backends dispatch over their
/// native storage (`Dense<f32>`, an `ndarray::ArrayD<f32>`, an
/// opaque GPU handle, …); the framework round-trips through the
/// producer/consumer `SlotValue` carriers via the derive bridge
/// in [`crate::roles::BackendRuntime`].
///
/// `Self::Tensor: Clone` is required because the per-op default
/// impls clone tensors into a temporary `HashMap<String, _>` to
/// feed [`Backend::execute`]. Backends overriding the per-op
/// methods directly never invoke this clone; backends overriding
/// `execute` natively pay one clone per per-op call. ndarray's
/// `ArrayD<f32>` clones the shape + bumps an internal refcount —
/// a few-hundred-nanosecond cost, not a memcpy.
pub trait Backend: Send + Sync {
    /// Library-maker-defined error type. The
    /// `From<BackendWalkError>` bound lets the default per-op /
    /// `execute_graph_via_per_op` walker surface graph-validation
    /// failures as typed errors instead of panicking on
    /// peer-supplied or malformed `GraphProto` bodies.
    type Error: std::error::Error
        + std::fmt::Display
        + Send
        + Sync
        + From<crate::contracts::backend_default_walk::BackendWalkError>
        + 'static;

    /// Native tensor representation.
    type Tensor: Clone + Send + Sync + 'static + bb_ir::types::Storage;

    // ──────────────────────────────────────────────────────────
    // Per-op surface — one method per primitive in
    // `TENSOR_PRIMITIVES_OPS`. Each default wraps a one-node
    // `GraphProto` and calls `execute`.
    // ──────────────────────────────────────────────────────────

    // ─── Arithmetic (6) ───────────────────────────────────────

    /// Element-wise `a + b` with NumPy broadcasting.
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Add", &[a, b], Vec::new())
    }
    /// Element-wise `a - b` with NumPy broadcasting.
    fn sub(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Sub", &[a, b], Vec::new())
    }
    /// Element-wise `a * b` with NumPy broadcasting.
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Mul", &[a, b], Vec::new())
    }
    /// Element-wise `a / b` with NumPy broadcasting.
    fn div(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Div", &[a, b], Vec::new())
    }
    /// Element-wise unary negation.
    fn neg(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Neg", &[a], Vec::new())
    }
    /// Element-wise absolute value.
    fn abs(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Abs", &[a], Vec::new())
    }

    // ─── Math (4) ─────────────────────────────────────────────

    /// Element-wise square root.
    fn sqrt(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Sqrt", &[a], Vec::new())
    }
    /// Element-wise `a ** b` with NumPy broadcasting.
    fn pow(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Pow", &[a, b], Vec::new())
    }
    /// Element-wise natural exponential.
    fn exp(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Exp", &[a], Vec::new())
    }
    /// Element-wise natural logarithm.
    fn log(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Log", &[a], Vec::new())
    }

    // ─── Linear algebra (1) ───────────────────────────────────

    /// Matrix multiplication (NumPy semantics: 2-D × 2-D + batched
    /// higher-rank broadcasting).
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "MatMul", &[a, b], Vec::new())
    }

    // ─── Reductions (4) ───────────────────────────────────────

    /// Sum-reduce `a` along `axes`. `keepdims = true` preserves
    /// the reduced dims as length-1.
    fn reduce_sum(
        &self,
        a: &Self::Tensor,
        axes: &[i64],
        keepdims: bool,
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "ReduceSum",
            &[a],
            vec![
                backend_default_walk::ints_attr("axes", axes),
                backend_default_walk::int_attr("keepdims", keepdims as i64),
            ],
        )
    }
    /// Mean-reduce `a` along `axes`.
    fn reduce_mean(
        &self,
        a: &Self::Tensor,
        axes: &[i64],
        keepdims: bool,
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "ReduceMean",
            &[a],
            vec![
                backend_default_walk::ints_attr("axes", axes),
                backend_default_walk::int_attr("keepdims", keepdims as i64),
            ],
        )
    }
    /// Max-reduce `a` along `axes`.
    fn reduce_max(
        &self,
        a: &Self::Tensor,
        axes: &[i64],
        keepdims: bool,
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "ReduceMax",
            &[a],
            vec![
                backend_default_walk::ints_attr("axes", axes),
                backend_default_walk::int_attr("keepdims", keepdims as i64),
            ],
        )
    }
    /// Min-reduce `a` along `axes`.
    fn reduce_min(
        &self,
        a: &Self::Tensor,
        axes: &[i64],
        keepdims: bool,
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "ReduceMin",
            &[a],
            vec![
                backend_default_walk::ints_attr("axes", axes),
                backend_default_walk::int_attr("keepdims", keepdims as i64),
            ],
        )
    }

    // ─── Shape (9) ────────────────────────────────────────────

    /// Reshape `a` to the given dims. Total element count must
    /// match.
    fn reshape(&self, a: &Self::Tensor, shape: &[i64]) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Reshape",
            &[a],
            vec![backend_default_walk::ints_attr("shape", shape)],
        )
    }
    /// Transpose axes. Empty `perm` reverses all dims.
    fn transpose(&self, a: &Self::Tensor, perm: &[i64]) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Transpose",
            &[a],
            vec![backend_default_walk::ints_attr("perm", perm)],
        )
    }
    /// Concatenate `inputs` along `axis`.
    fn concat(&self, inputs: &[&Self::Tensor], axis: i64) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Concat",
            inputs,
            vec![backend_default_walk::int_attr("axis", axis)],
        )
    }
    /// NumPy-style slice. Empty `axes` defaults to all dims;
    /// empty `steps` defaults to 1 per axis.
    fn slice(
        &self,
        a: &Self::Tensor,
        starts: &[i64],
        ends: &[i64],
        axes: &[i64],
        steps: &[i64],
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Slice",
            &[a],
            vec![
                backend_default_walk::ints_attr("starts", starts),
                backend_default_walk::ints_attr("ends", ends),
                backend_default_walk::ints_attr("axes", axes),
                backend_default_walk::ints_attr("steps", steps),
            ],
        )
    }
    /// Split `a` along `axis` into parts of the given `sizes`.
    /// Empty `sizes` means equal-sized splits (count comes from
    /// the consumer side downstream).
    fn split(
        &self,
        a: &Self::Tensor,
        axis: i64,
        sizes: &[i64],
    ) -> Result<Vec<Self::Tensor>, Self::Error> {
        // `Split` is the only primitive returning multiple
        // tensors. We can't use `execute_single`'s single-output
        // path — instead we wrap into a graph that names each
        // output positionally and extract them.
        backend_default_walk::execute_multi(
            self,
            "Split",
            &[a],
            vec![
                backend_default_walk::int_attr("axis", axis),
                backend_default_walk::ints_attr("split", sizes),
            ],
            sizes.len(),
        )
    }
    /// Remove dimensions of size 1. Empty `axes` removes all
    /// size-1 dims.
    fn squeeze(&self, a: &Self::Tensor, axes: &[i64]) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Squeeze",
            &[a],
            vec![backend_default_walk::ints_attr("axes", axes)],
        )
    }
    /// Insert dimensions of size 1 at the given axes.
    fn unsqueeze(&self, a: &Self::Tensor, axes: &[i64]) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Unsqueeze",
            &[a],
            vec![backend_default_walk::ints_attr("axes", axes)],
        )
    }
    /// Identity / clone — pass-through useful for graph rewrites.
    fn identity(&self, a: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Identity", &[a], Vec::new())
    }
    /// Cast to the given ONNX `DataType` enum value (matches
    /// `bb_ir::proto::onnx::tensor_proto::DataType`).
    fn cast(&self, a: &Self::Tensor, dtype: i32) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Cast",
            &[a],
            vec![backend_default_walk::int_attr("to", dtype as i64)],
        )
    }

    // ─── Comparison (3) ───────────────────────────────────────

    /// Element-wise `a == b`. Result is boolean-typed.
    fn equal(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Equal", &[a, b], Vec::new())
    }
    /// Element-wise `a > b`. Result is boolean-typed.
    fn greater(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Greater", &[a, b], Vec::new())
    }
    /// Element-wise `a < b`. Result is boolean-typed.
    fn less(&self, a: &Self::Tensor, b: &Self::Tensor) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Less", &[a, b], Vec::new())
    }

    // ─── Conditional (1) ──────────────────────────────────────

    /// Element-wise ternary: `where cond { t } else { f }`.
    /// Named `r#where` to dodge the reserved Rust keyword.
    fn r#where(
        &self,
        cond: &Self::Tensor,
        t: &Self::Tensor,
        f: &Self::Tensor,
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(self, "Where", &[cond, t, f], Vec::new())
    }

    // ─── Creation (1) ─────────────────────────────────────────

    /// Materialize a constant from an ONNX `TensorProto`. The
    /// `value` attribute on the ONNX `Constant` op carries the
    /// data; rank, dtype, raw bytes all come from the proto.
    fn constant(&self, value: TensorProto) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Constant",
            &[],
            vec![backend_default_walk::tensor_attr("value", value)],
        )
    }

    // ─── Indexing (1) ─────────────────────────────────────────

    /// Gather slices of `data` along `axis` indexed by `indices`.
    fn gather(
        &self,
        data: &Self::Tensor,
        indices: &Self::Tensor,
        axis: i64,
    ) -> Result<Self::Tensor, Self::Error> {
        backend_default_walk::execute_single(
            self,
            "Gather",
            &[data, indices],
            vec![backend_default_walk::int_attr("axis", axis)],
        )
    }

    // ──────────────────────────────────────────────────────────
    // Whole-graph surface — default walks `graph.node` and
    // dispatches each through the typed per-op methods above.
    // ──────────────────────────────────────────────────────────

    /// Execute every NodeProto in `graph.node` against the value
    /// env `inputs`. Returns the subset of values named in
    /// `graph.output`.
    ///
    /// `graph.node` is topologically ordered per the ONNX spec,
    /// so the default walker (a linear scan) suffices for any
    /// `GraphProto` whose ops are all in
    /// [`bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS`]. A
    /// backend overriding this method may detect fused patterns,
    /// compile to GPU, or any other strategy.
    fn execute(
        &self,
        graph: &GraphProto,
        inputs: HashMap<String, Self::Tensor>,
        _attrs: BackendAttrs<'_>,
    ) -> Result<HashMap<String, Self::Tensor>, Self::Error> {
        backend_default_walk::execute_graph_via_per_op(self, graph, inputs)
    }

    /// Dispatch a `BackendSubgraph` carrier — the engine-facing entry
    /// point for whole-subgraph execution.
    ///
    /// The default falls through to [`Self::execute`] synchronously,
    /// keeping existing backends' behaviour identical. Backends with
    /// per-subgraph caching, JIT compilation, or async device execution
    /// override this to:
    ///
    /// - Cache the compiled subgraph by identity (e.g. graph name or
    ///   hash).
    /// - Return [`ContractResponse::Later`] and retain `completion`
    ///   while the device runs. The engine schedules other work;
    ///   the backend completes the handle from whatever runtime it
    ///   uses — `std::thread`, tokio task, custom event loop, single-
    ///   thread no-std loop.
    /// - Fall through to [`Self::execute`] on compile failure or
    ///   unsupported op.
    ///
    /// The `completion` parameter in the default impl is intentionally
    /// discarded (`let _ = completion`) because [`ContractResponse::Now`]
    /// does not retain the handle. This is correct — only overriders
    /// that return [`ContractResponse::Later`] must hold it.
    fn dispatch(
        &self,
        graph: &GraphProto,
        inputs: HashMap<String, Self::Tensor>,
        attrs: BackendAttrs<'_>,
        completion: CompletionHandle<HashMap<String, Self::Tensor>, Self::Error>,
    ) -> ContractResponse<HashMap<String, Self::Tensor>, Self::Error> {
        let _ = completion; // default doesn't retain it; signature stays for opt-in overriders
        ContractResponse::Now(self.execute(graph, inputs, attrs))
    }

    /// Materialise an inbound tensor `SlotFill` into this backend's
    /// native tensor representation.
    ///
    /// The framework has already (a) capped `bytes.len()` against the
    /// envelope's `EnvelopeCaps::max_per_fill_bytes`, (b) charged the
    /// length against `NodeConfig::ingress_byte_budget`, and (c) moved
    /// ownership of the wire bytes into this call. The backend may
    /// adopt the `Vec<u8>` directly (zero-copy via
    /// `ArrayD::from_shape_vec` when alignment permits), pull a buffer
    /// from a pool and copy in, or allocate fresh. The framework will
    /// not touch `bytes` after this call returns.
    ///
    /// The default delegates to the global wire-decoder registry: it
    /// looks up the decoder for `type_hash`, runs it on the bytes,
    /// then downcasts the resulting boxed `SlotValue` to `Self::Tensor`
    /// via the registry's `Box<dyn Any>` repackaging. Backends that
    /// have not implemented tensor pooling continue to work through
    /// this path; backends that override pay the registry hop only at
    /// override time.
    ///
    /// On `Err`, the engine drops the fill, releases the byte charge,
    /// and emits `WireReceiveError { kind: BackendMaterializeFailed }`.
    ///
    /// Ownership note: `bytes: Vec<u8>` by value (not `&[u8]` or
    /// `Cow`). This is the framework→backend handoff, NOT an external
    /// boundary — the backend lives inside the framework ecosystem
    /// and plays by the runtime contract. Principle 1a (ephemeral
    /// borrowed slices at external boundaries) does not apply here:
    /// the framework copied or owned the bytes already, and a backend
    /// that wants to adopt them (zero-copy) needs ownership.
    fn materialize_from_wire(
        &self,
        type_hash: u64,
        bytes: Vec<u8>,
    ) -> Result<Self::Tensor, Self::Error> {
        use crate::contracts::backend_default_walk::BackendWalkError;
        let decoder = bb_ir::slot_value::wire_decoder_registry()
            .get(&type_hash)
            .copied()
            .ok_or_else(|| BackendWalkError::WireMaterializeFailed {
                type_hash,
                reason: "no decoder registered for type_hash".into(),
            })?;
        let boxed = decoder(&bytes).map_err(|e| BackendWalkError::WireMaterializeFailed {
            type_hash,
            reason: e.to_string(),
        })?;
        let any = boxed.into_any_boxed();
        any.downcast::<Self::Tensor>().map(|b| *b).map_err(|_| {
            BackendWalkError::WireMaterializeFailed {
                type_hash,
                reason: "decoded carrier is not Self::Tensor".into(),
            }
            .into()
        })
    }
}
