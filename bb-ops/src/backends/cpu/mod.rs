//! Pure-Rust reference CPU backend. Implements `bb::Backend` for
//! the `ai.onnx v1` 51-op subset over ndarray. Gated by the
//! `cpu-backend` feature.

pub mod ops;
pub mod opset;
pub mod tensor;

use bb_derive::Backend;
use serde::{Deserialize, Serialize};

use bb_dsl::concrete::{ComponentPackage, ConcreteComponent, RestoreError};
use bb_runtime::component::AnyComponent;

pub use opset::{ONNX_DOMAIN, ONNX_V1_OPSET, ONNX_VERSION};
pub use tensor::{CpuTensor, CpuTensorError};

pub mod graph_walker;
pub use graph_walker::{execute_graph, BackendError};

/// Reference CPU backend. Dispatches its opset onto ndarray
/// kernels; storage is `ArrayD<f32>` end-to-end.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Backend)]
pub struct CpuBackend;

impl CpuBackend {
    /// Construct a fresh backend.
    pub fn new() -> Self {
        Self
    }

    /// Allocate a zero-initialised tensor with the given shape.
    /// Single allocation seam for future pool / arena strategies.
    pub fn alloc_tensor(&self, shape: Vec<i64>) -> CpuTensor {
        CpuTensor::zeros(shape)
    }

    /// Wrap a kernel-produced `ArrayD<f32>` as a `CpuTensor`.
    /// Single wrapping seam for future pool / arena strategies.
    pub fn wrap_array(&self, array: ndarray::ArrayD<f32>) -> CpuTensor {
        CpuTensor::from_array(array)
    }
}

impl AnyComponent for CpuBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl ConcreteComponent for CpuBackend {
    const TYPE_NAME: &'static str = "bytesandbrains::backends::cpu::CpuBackend";
    const PACKAGE: ComponentPackage = ComponentPackage::Framework;

    type Config = ();
    type Error = std::convert::Infallible;

    fn new(_config: &Self::Config) -> Result<Self, Self::Error> {
        Ok(Self)
    }

    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).expect("CpuBackend serde is infallible")
    }

    fn restore(bytes: &[u8]) -> Result<Self, RestoreError> {
        bincode::deserialize(bytes).map_err(RestoreError::Malformed)
    }
}

// Hand-rolled equivalent of `#[derive(bb::Concrete)]` so the
// manual `ConcreteComponent` impl above continues to drive inventory
// registration.
#[doc(hidden)]
fn __cpu_backend_serialize(erased: &dyn bb_ir::component::ErasedComponent) -> Vec<u8> {
    let any: &dyn std::any::Any = erased;
    let concrete: &CpuBackend = any
        .downcast_ref::<CpuBackend>()
        .expect("inventory downcast: CpuBackend by TYPE_NAME");
    <CpuBackend as ConcreteComponent>::serialize(concrete)
}
#[doc(hidden)]
fn __cpu_backend_restore(
    bytes: &[u8],
) -> Result<Box<dyn bb_ir::component::ErasedComponent>, RestoreError> {
    <CpuBackend as ConcreteComponent>::restore(bytes)
        .map(|v| Box::new(v) as Box<dyn bb_ir::component::ErasedComponent>)
}
#[doc(hidden)]
fn __cpu_backend_construct(
    cfg: &dyn std::any::Any,
) -> Result<Box<dyn bb_ir::component::ErasedComponent>, bb_ir::component::ConstructError> {
    let _typed: &() = cfg
        .downcast_ref::<()>()
        .ok_or_else(|| bb_ir::component::ConstructError {
            type_name: "bytesandbrains::backends::cpu::CpuBackend",
            detail: format!(
                "config type mismatch: expected `()`, got `{:?}`",
                cfg.type_id(),
            ),
        })?;
    <CpuBackend as ConcreteComponent>::new(_typed)
        .map(|v| Box::new(v) as Box<dyn bb_ir::component::ErasedComponent>)
        .map_err(|e| bb_ir::component::ConstructError {
            type_name: "bytesandbrains::backends::cpu::CpuBackend",
            detail: format!("{e}"),
        })
}
bb_ir::registry::inventory::submit! {
    bb_ir::registry::ConcreteComponentRegistration {
        type_name: "bytesandbrains::backends::cpu::CpuBackend",
        package: ComponentPackage::Framework,
        serialize_fn: __cpu_backend_serialize,
        restore_fn: __cpu_backend_restore,
        construct_fn: __cpu_backend_construct,
        dependencies: &[],
    }
}

/// `CpuTensor` → `TYPE_TENSOR_F32` in the polymorphism tree.
impl bb_ir::types::Storage for CpuTensor {
    const TYPE: &'static bb_ir::types::TypeNode = &bb_ir::types::TYPE_TENSOR_F32;
}

/// Thread-local invocation counter for `CpuBackend::execute`, gated
/// under `test-components`.
#[cfg(any(test, feature = "test-components"))]
mod dispatch_counter {
    use std::cell::Cell;
    thread_local! {
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }
    pub fn bump() {
        COUNT.with(|c| c.set(c.get().wrapping_add(1)));
    }
    /// Read the current per-thread invocation count.
    pub fn read() -> usize {
        COUNT.with(|c| c.get())
    }
    /// Reset the per-thread invocation count to zero.
    pub fn reset() {
        COUNT.with(|c| c.set(0));
    }
}

#[cfg(any(test, feature = "test-components"))]
pub use dispatch_counter::{read as dispatch_count, reset as reset_dispatch_count};

/// `bb::Backend` Contract impl. Overrides `execute` to run through
/// `graph_walker::execute_graph` rather than the default per-op
/// walker.
impl bb_runtime::contracts::Backend for CpuBackend {
    type Error = graph_walker::BackendError;
    type Tensor = CpuTensor;

    fn execute(
        &self,
        graph: &bb_ir::proto::onnx::GraphProto,
        inputs: std::collections::HashMap<String, Self::Tensor>,
        _attrs: bb_runtime::contracts::backend::BackendAttrs<'_>,
    ) -> Result<std::collections::HashMap<String, Self::Tensor>, Self::Error> {
        #[cfg(any(test, feature = "test-components"))]
        dispatch_counter::bump();
        graph_walker::execute_graph(self, graph, inputs)
    }

    /// Decode wire bytes inside the backend so the `CpuTensor`
    /// carries the ingress byte charge for slot-table release on
    /// overwrite. v1 uses bincode.
    fn materialize_from_wire(
        &self,
        type_hash: u64,
        bytes: Vec<u8>,
    ) -> Result<Self::Tensor, Self::Error> {
        use bb_runtime::contracts::backend_default_walk::BackendWalkError;
        let expected_hash = bb_ir::slot_value::type_hash_of::<CpuTensor>();
        if type_hash != expected_hash {
            return Err(graph_walker::BackendError::DefaultWalker(
                BackendWalkError::WireMaterializeFailed {
                    type_hash,
                    reason: format!(
                        "expected CpuTensor type_hash {expected_hash:#018x}, got {type_hash:#018x}",
                    ),
                },
            ));
        }
        let charged_bytes = bytes.len();
        // Re-wrap with the carried charge so the slot-table writer
        // can release the budget on overwrite.
        let wire: CpuTensor = bincode::deserialize(&bytes).map_err(|e| {
            graph_walker::BackendError::DefaultWalker(BackendWalkError::WireMaterializeFailed {
                type_hash,
                reason: format!("bincode decode: {e}"),
            })
        })?;
        // One copy out of the discarded handle; the wire path pays
        // this until pooling lands.
        Ok(CpuTensor::from_wire_buffer(
            wire.0.data.clone(),
            charged_bytes,
        ))
    }
}

