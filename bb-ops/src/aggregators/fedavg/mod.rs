//! `FedAvg<B>` — federated-averaging aggregator. Composes the
//! reduction from the bound backend's `Mul` + `Add` primitives so
//! the 30-op floor stays unchanged. Aggregate emits the cumulative
//! `num_samples` for hierarchical weighting.
//!
//! Trust model: contributions are assumed finite. NaN/Inf
//! propagates per IEEE 754 and poisons the round; defenses belong
//! at the contribution boundary (signed Codec, attesting
//! PeerSelector, secure-aggregation protocol).

use std::any::Any;
use std::collections::BTreeMap;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

#[cfg(feature = "cpu-backend")]
use bb_ir::component::ErasedComponent;
use bb_ir::component::{AnyComponent, DependencyDecl, RestoreError};
use bb_ir::ids::PeerId;
use bb_ir::proto::onnx::TensorProto;
use bb_ir::tensor::Tensor;
use bb_ir::types::common_relations::NO_RELATIONS;
use bb_runtime::atomic::{AtomicOpDecl, AtomicOpKind, AtomicOpsetDecl, DispatchResult};
use bb_runtime::bus::{OpError, OpErrorKind};
use bb_runtime::completion::{CompletionHandle, ContractResponse};
use bb_runtime::concrete::{ComponentPackage, ConcreteComponent};
use bb_runtime::contracts::{Aggregator as AggregatorContract, Backend};
use bb_runtime::roles::AggregatorRuntime;
use bb_runtime::runtime::RuntimeResourceRef;
use bb_runtime::slot_value::SlotValue;

/// Sample-count metadata for FedAvg contributions / aggregates.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FedAvgMeta {
    /// Single-layer: local batch size. Hierarchical: subtree sum,
    /// used to weight this aggregate's contribution at parent level.
    pub num_samples: u64,
}

/// ONNX `DataType::FLOAT`.
const ONNX_FLOAT: i32 = 1;

/// `<B::Tensor as Storage>::TYPE`. Inventory-gated under
/// `cpu-backend` to avoid unused monomorphizations.
#[cfg(feature = "cpu-backend")]
fn fedavg_element_type<B: Backend>() -> &'static bb_ir::types::TypeNode {
    <B::Tensor as bb_ir::types::Storage>::TYPE
}

/// FedAvg aggregator: weighted average where weights are
/// `num_samples / total_samples`. Generic over `B: Backend` so the
/// reduction composes from the backend's `Mul` + `Add` primitives
/// without bumping the 30-op floor.
///
/// The buffer is a `BTreeMap<PeerId, (B::Tensor, u64)>` so the
/// reduction walk has a deterministic iteration order (lexical by
/// peer id). The engine's `dispatch_atomic(&mut self, ...)` contract
/// gives the aggregator exclusive access for the duration of each
/// call. The buffer is `#[serde(skip)]` because snapshot captures
/// the aggregator's structural identity, not the per-round transient
/// contribution state.
#[derive(Debug, Serialize, Deserialize)]
pub struct FedAvg<B: Backend> {
    /// Per-round buffer keyed by source `PeerId`. Duplicate
    /// contributions from the same peer in the same round REPLACE
    /// the prior entry, so a buggy or malicious peer cannot double
    /// its weight by contributing twice. `BTreeMap` is the ordered
    /// peer-id walk the spec's determinism guarantee relies on (the
    /// reduction's f32 accumulation order is the BTree's lexical
    /// order, not a hash-map's runtime-randomized order).
    #[serde(skip)]
    buffer: BTreeMap<PeerId, (B::Tensor, u64)>,
    #[serde(skip)]
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for FedAvg<B> {
    fn default() -> Self {
        Self {
            buffer: BTreeMap::new(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Clone for FedAvg<B> {
    fn clone(&self) -> Self {
        // Cloning returns a fresh empty buffer — snapshots restore
        // via `restore` (the universal `ConcreteComponent` path),
        // not through `Clone`. Clone is only used to satisfy the
        // framework's `T: Clone` bounds.
        Self::default()
    }
}

impl<B: Backend> AggregatorContract for FedAvg<B>
where
    B: 'static,
    B::Tensor: Tensor,
{
    type Element = B::Tensor;
    type Error = OpError;
    type Metadata = FedAvgMeta;

    fn contribute(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        src: PeerId,
        tensor: &Self::Element,
        metadata: FedAvgMeta,
        _completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        // Reject zero-sample contributions: an `n=0` entry would
        // contribute zero weight to the reduction yet still
        // displace a real same-peer contribution from the buffer.
        if metadata.num_samples == 0 {
            return ContractResponse::Now(Err(OpError {
                detail: "FedAvg::contribute: num_samples = 0 — degenerate weight".into(),
                ..Default::default()
            }));
        }
        // Keying on src prevents a peer from doubling its weight by
        // contributing twice in one round: the second entry replaces
        // the first rather than landing alongside it.
        self.buffer
            .insert(src, (tensor.clone(), metadata.num_samples));
        ContractResponse::Now(Ok(()))
    }

    fn aggregate(
        &mut self,
        ctx: &mut RuntimeResourceRef<'_>,
        _completion: CompletionHandle<(Box<Self::Element>, FedAvgMeta), Self::Error>,
    ) -> ContractResponse<(Box<Self::Element>, FedAvgMeta), Self::Error> {
        let backend = match ctx.dependency::<B>("backend") {
            Ok(b) => b,
            Err(e) => {
                return ContractResponse::Now(Err(OpError {
                    detail: format!("FedAvg::aggregate: backend lookup failed: {e}"),
                    ..Default::default()
                }));
            }
        };

        let entries: Vec<(B::Tensor, u64)> =
            std::mem::take(&mut self.buffer).into_values().collect();
        if entries.is_empty() {
            return ContractResponse::Now(Err(OpError {
                detail: "FedAvg::aggregate: empty buffer — no contributions to reduce".into(),
                ..Default::default()
            }));
        }

        let total_samples: u64 = entries.iter().map(|(_, n)| *n).sum();
        if total_samples == 0 {
            return ContractResponse::Now(Err(OpError {
                detail: "FedAvg::aggregate: total_samples = 0".into(),
                ..Default::default()
            }));
        }
        let total_f = total_samples as f32;

        // Determine the canonical output shape from the first
        // contribution; later contributions of mismatched shape will
        // be rejected by the backend's elementwise kernel. The shape
        // also drives the per-peer weight tensor's construction —
        // `CpuBackend`'s `Mul` requires same-shape inputs (full
        // NumPy broadcasting isn't implemented yet), so the weight
        // is materialized at the canonical shape rather than as a
        // length-1 broadcast scalar.
        let canonical_dims: Vec<i64> = entries[0].0.dims().to_vec();
        let canonical_len: usize = canonical_dims
            .iter()
            .map(|d| (*d).max(0) as usize)
            .product();

        let mut acc: Option<B::Tensor> = None;
        for (tensor, n) in &entries {
            let w = (*n as f32) / total_f;
            let weight_proto = TensorProto {
                data_type: ONNX_FLOAT,
                dims: canonical_dims.clone(),
                float_data: vec![w; canonical_len],
                ..Default::default()
            };
            let weight = match backend.constant(weight_proto) {
                Ok(t) => t,
                Err(e) => {
                    return ContractResponse::Now(Err(OpError {
                        detail: format!("FedAvg::aggregate: backend.constant failed: {e}"),
                        ..Default::default()
                    }));
                }
            };
            let scaled = match backend.mul(tensor, &weight) {
                Ok(t) => t,
                Err(e) => {
                    return ContractResponse::Now(Err(OpError {
                        detail: format!("FedAvg::aggregate: backend.mul failed: {e}"),
                        ..Default::default()
                    }));
                }
            };
            acc = Some(match acc {
                None => scaled,
                Some(prev) => match backend.add(&prev, &scaled) {
                    Ok(t) => t,
                    Err(e) => {
                        return ContractResponse::Now(Err(OpError {
                            detail: format!("FedAvg::aggregate: backend.add failed: {e}"),
                            ..Default::default()
                        }));
                    }
                },
            });
        }

        let params = acc.expect("entries non-empty implies acc populated");
        ContractResponse::Now(Ok((
            Box::new(params),
            FedAvgMeta {
                num_samples: total_samples,
            },
        )))
    }
}

// ─── Manual ConcreteComponent + AnyComponent + role plumbing ──────
//
// `bb_derive::Aggregator` does not handle generic structs; the
// inventory submissions below cover every monomorphization the
// framework needs (currently `FedAvg<CpuBackend>` when the
// `cpu-backend` feature is on). Generic-impl support in the derive
// is out of scope for the dep-injection milestone.

impl<B: Backend> ConcreteComponent for FedAvg<B>
where
    B: 'static + Default,
{
    const TYPE_NAME: &'static str = "FedAvg";
    const PACKAGE: ComponentPackage = ComponentPackage::Framework;
    const DEPENDENCIES: &'static [DependencyDecl] = &[DependencyDecl {
        role: "Backend",
        slot: "backend",
    }];

    type Config = ();
    type Error = std::convert::Infallible;

    fn new(_: &Self::Config) -> Result<Self, Self::Error> {
        Ok(Self::default())
    }

    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("FedAvg serialize — bincode infallible on Default state")
    }

    fn restore(bytes: &[u8]) -> Result<Self, RestoreError> {
        bincode::deserialize(bytes).map_err(RestoreError::Malformed)
    }
}

impl<B: Backend + 'static> AnyComponent for FedAvg<B> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Atomic opset for the `FedAvg<B>` aggregator. Names align with
/// the canonical Aggregator role surface
/// (`emit_aggregator_arms` in `bb-derive`).
static FEDAVG_ATOMIC_OPS: &[AtomicOpDecl] = &[
    AtomicOpDecl {
        name: "Contribute",
        inputs: &[],
        outputs: &[],
        kind: AtomicOpKind::Immediate,
        type_relations: NO_RELATIONS,
    },
    AtomicOpDecl {
        name: "Aggregate",
        inputs: &[],
        outputs: &[],
        kind: AtomicOpKind::Immediate,
        type_relations: NO_RELATIONS,
    },
];

impl<B> AggregatorRuntime for FedAvg<B>
where
    B: Backend + 'static + Default,
    B::Tensor: Tensor,
{
    type Error = OpError;

    fn atomic_opset(&self) -> AtomicOpsetDecl {
        AtomicOpsetDecl {
            domain: "ai.bytesandbrains.role.aggregator",
            version: 1,
            ops: FEDAVG_ATOMIC_OPS,
        }
    }

    fn dispatch_atomic(
        &mut self,
        op_type: &str,
        inputs: &[(&str, &dyn SlotValue)],
        ctx: &mut RuntimeResourceRef<'_>,
    ) -> Result<DispatchResult, Self::Error> {
        match op_type {
            "Contribute" => {
                // Borrow the boxed tensor through the SlotValue ref;
                // `contribute` takes `&Self::Element` so no owned copy
                // is needed at the dispatch boundary. The downstream
                // buffer insertion in `contribute` is the single
                // remaining tensor copy per contribution.
                let tensor_ref: &B::Tensor = match inputs
                    .first()
                    .and_then(|(_, v)| v.as_any().downcast_ref::<Box<B::Tensor>>())
                {
                    Some(b) => b,
                    None => {
                        return Err(OpError {
                            kind: OpErrorKind::TypeMismatch,
                            reason: "input_type_mismatch",
                            detail: format!(
                                "FedAvg::Contribute input 0 expected `Box<{}>`",
                                std::any::type_name::<B::Tensor>(),
                            ),
                        });
                    }
                };
                let metadata = match inputs
                    .get(1)
                    .and_then(|(_, v)| v.as_any().downcast_ref::<FedAvgMeta>())
                {
                    Some(m) => m.clone(),
                    None => {
                        return Err(OpError {
                            kind: OpErrorKind::TypeMismatch,
                            reason: "input_type_mismatch",
                            detail: "FedAvg::Contribute input 1 expected `FedAvgMeta`".into(),
                        });
                    }
                };
                let src = match ctx.current.inbound.src_peer {
                    Some(p) => p,
                    None => {
                        return Err(OpError {
                            detail: "FedAvg::Contribute: envelope_src_peer is None — wire envelope did not carry src_peer multihash bytes".into(),
                            ..Default::default()
                        });
                    }
                };
                let completion = ctx.open_completion::<(), OpError>();
                let cmd_id = completion.cmd_id();
                match <Self as AggregatorContract>::contribute(
                    self, ctx, src, tensor_ref, metadata, completion,
                ) {
                    ContractResponse::Now(Ok(())) => Ok(DispatchResult::Immediate(Vec::new())),
                    ContractResponse::Now(Err(e)) => Err(OpError {
                        detail: format!("{e}"),
                        ..Default::default()
                    }),
                    ContractResponse::Later => Ok(DispatchResult::Async(cmd_id)),
                }
            }
            "Aggregate" => {
                let completion = ctx.open_completion::<(Box<B::Tensor>, FedAvgMeta), OpError>();
                let cmd_id = completion.cmd_id();
                match <Self as AggregatorContract>::aggregate(self, ctx, completion) {
                    ContractResponse::Now(Ok((params, metadata))) => {
                        Ok(DispatchResult::Immediate(vec![
                            ("params".to_string(), Box::new(params) as Box<dyn SlotValue>),
                            (
                                "metadata".to_string(),
                                Box::new(metadata) as Box<dyn SlotValue>,
                            ),
                        ]))
                    }
                    ContractResponse::Now(Err(e)) => Err(OpError {
                        detail: format!("{e}"),
                        ..Default::default()
                    }),
                    ContractResponse::Later => Ok(DispatchResult::Async(cmd_id)),
                }
            }
            other => Err(OpError {
                detail: format!("FedAvg::dispatch_atomic: unknown op_type `{other}`"),
                ..Default::default()
            }),
        }
    }
}

// ─── Inventory submissions — `FedAvg<CpuBackend>` monomorphization ─
//
// Inventory carriers can only register concrete monomorphizations;
// each supported backend submits its own block (the `cpu-backend`
// feature is the only one shipping today).

#[cfg(feature = "cpu-backend")]
type FedAvgCpu = FedAvg<crate::backends::cpu::CpuBackend>;

#[cfg(feature = "cpu-backend")]
#[doc(hidden)]
fn __fedavg_cpu_serialize(erased: &dyn ErasedComponent) -> Vec<u8> {
    let any: &dyn Any = erased;
    let concrete: &FedAvgCpu = any
        .downcast_ref::<FedAvgCpu>()
        .expect("inventory downcast: FedAvg<CpuBackend>");
    <FedAvgCpu as ConcreteComponent>::serialize(concrete)
}

#[cfg(feature = "cpu-backend")]
#[doc(hidden)]
fn __fedavg_cpu_restore(bytes: &[u8]) -> Result<Box<dyn ErasedComponent>, RestoreError> {
    <FedAvgCpu as ConcreteComponent>::restore(bytes)
        .map(|v| Box::new(v) as Box<dyn ErasedComponent>)
}

#[cfg(feature = "cpu-backend")]
#[doc(hidden)]
fn __fedavg_cpu_construct(
    cfg: &dyn Any,
) -> Result<Box<dyn ErasedComponent>, bb_runtime::concrete::ConstructError> {
    let typed = cfg
        .downcast_ref::<()>()
        .ok_or_else(|| bb_runtime::concrete::ConstructError {
            type_name: "FedAvg",
            detail: "config type mismatch: expected `()`".into(),
        })?;
    <FedAvgCpu as ConcreteComponent>::new(typed)
        .map(|v| Box::new(v) as Box<dyn ErasedComponent>)
        .map_err(|e| bb_runtime::concrete::ConstructError {
            type_name: "FedAvg",
            detail: format!("{e}"),
        })
}

#[cfg(feature = "cpu-backend")]
#[doc(hidden)]
fn __fedavg_cpu_element_type_node() -> &'static bb_ir::types::TypeNode {
    fedavg_element_type::<crate::backends::cpu::CpuBackend>()
}

#[cfg(feature = "cpu-backend")]
inventory::submit! {
    bb_runtime::registry::ConcreteComponentRegistration {
        type_name: "FedAvg",
        package: ComponentPackage::Framework,
        serialize_fn: __fedavg_cpu_serialize,
        restore_fn: __fedavg_cpu_restore,
        construct_fn: __fedavg_cpu_construct,
        dependencies: <FedAvgCpu as ConcreteComponent>::DEPENDENCIES,
    }
}

#[cfg(feature = "cpu-backend")]
inventory::submit! {
    bb_runtime::registry::ComponentRoleBinding {
        type_name: "FedAvg",
        role: bb_runtime::registry::ComponentRole::Aggregator,
    }
}

#[cfg(feature = "cpu-backend")]
inventory::submit! {
    bb_runtime::registry::DispatcherRegistration {
        type_name: "FedAvg",
        role: bb_runtime::registry::ComponentRole::Aggregator,
        register_fn: |engine: &mut bb_runtime::engine::Engine| {
            engine.register_aggregator_dispatcher::<FedAvgCpu>();
        },
    }
}

#[cfg(feature = "cpu-backend")]
inventory::submit! {
    bb_runtime::registry::StorageTypeEntry {
        concrete_type_name: <FedAvgCpu as ConcreteComponent>::TYPE_NAME,
        role_runtime: "AggregatorRuntime",
        port: "element",
        type_node_fn: __fedavg_cpu_element_type_node,
    }
}

