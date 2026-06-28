//! # bytesandbrains
//!
//! Composable building blocks for decentralized + federated machine
//! learning. Facade crate re-exporting `bb-ir`, `bb-dsl`,
//! `bb-compiler`, `bb-runtime`, `bb-ops`, and `bb-derive`. See
//! [`docs/`](https://github.com/Bytes-Brains/private/tree/main/docs)
//! for the canonical spec.

#![cfg_attr(not(test), warn(missing_docs))]

// bb-derive emits `::bytesandbrains::…` paths; this enables that
// resolution inside the facade crate itself.
extern crate self as bytesandbrains;

// --- bb-ir foundation re-exports ---------------------------------

/// Prost-generated proto bindings (ONNX + `bb.core`).
pub use bb_ir::proto;

/// Polymorphic type system — hierarchical `TypeNode` tree
/// resolved at compile time by the bb-compiler's TypeSolver.
pub use bb_ir::types;

/// Wire codec contract + canonical `TypeNode` statics
/// (`TYPE_TENSOR_F32`, `TYPE_TRIGGER`, `TYPE_PEER_ID`, …).
pub use bb_ir::wire;

/// Tensor and Scalar abstractions. Backends implement.
pub use bb_ir::tensor;

/// Universal slot value trait. Re-exports `bb_ir::slot_value` plus
/// engine-side carriers (`BackendTensorCarrier`) that bypass the
/// serde-driven blanket impl.
pub use bb_runtime::slot_value;

/// `ErasedComponent` + `AnyComponent` traits + foundation
/// polymorphism plumbing (`ComponentPackage`, `RestoreError`,
/// `SerializeFn`, `RestoreFn`).
pub use bb_ir::component;

/// Stable `(domain, op_type)` + attribute-name string constants for
/// every framework syscall - the IR contract between bb-compiler +
/// bb-runtime.
pub use bb_ir::syscall_ids;

/// Peer-class metadata stamps (`SELF_CLASS`, `PEER_CLASS_KEY`,
/// `HOME_CLASS_KEY`).
pub use bb_ir::peer_class;

/// Framework attribute-key string constants (`BACKEND_SUBGRAPH_BODY_ATTR`,
/// `WIRE_TRANSPORT_KEY`, …). Re-exported so proc-macro emission
/// reaches them without users depending on `bb-ir` directly.
pub use bb_ir::keys;

/// `bincode` re-export for proc-macro emission sites
/// (`#[derive(bb::Concrete)]` serialize/restore).
pub use bb_ir::bincode;

/// `inventory` re-export for proc-macro emission sites
/// (`#[derive(bb::<Role>)]`, `register_op!`, `register_protocol!`).
pub use bb_ir::inventory;

// --- bb-runtime engine-internal type re-exports ------------------

/// Identity newtypes. Engine-internal IDs (`NodeSiteId`, `OpRef`,
/// `ExecId`, `CommandId`, `ComponentRef`) live here; wire/compiler
/// IDs (`PeerId`, `RequestId`, `OpsetId`, `ComponentTag`) come from
/// `bb-ir` re-exported through this module.
pub use bb_runtime::ids;

/// Atomic dispatch declaration types + `DispatchResult`. Catalog
/// declarations stay in `bb-ir`; `DispatchResult` carries
/// `CommandId` and so lives here.
pub use bb_runtime::atomic;

/// `CompletionHandle` for async Contract methods.
pub use bb_runtime::completion;

// --- bb-dsl authoring re-exports ---------------------------------

/// The `Module` trait. `Module::build()` returns a recorded
/// `ModelProto` that `Compiler` consumes.
pub use bb_dsl::module;

/// `Graph` recording context.
pub use bb_dsl::graph;

/// Non-generic `Output` handle threaded through DSL method chains.
pub use bb_dsl::output;

/// User-facing Contract traits (`Index`, `Aggregator`, `Model`,
/// `Codec`, `DataSource`, `PeerSelector`, `Backend`,
/// `Protocol`).
pub use bb_dsl::contracts;

/// `ConcreteComponent` polymorphism contract + the
/// `ComponentHandle` fn-pointer-capture wrapper.
pub use bb_dsl::concrete;

/// Role-method dispatch slot placeholders (`BackendSlot`,
/// `IndexSlot`, `ModelSlot`, …). Module fields are typed against
/// these; the compiler binds them to concrete implementations.
pub use bb_ops::placeholders;

/// Concrete `Aggregator` implementations the framework ships
/// (e.g. `FedAvg`).
pub use bb_ops::aggregators;

/// Concrete `Protocol` implementations the framework ships
/// (e.g. `GlobalRegistryClient`, `GlobalRegistryServer`).
pub use bb_ops::protocols;

/// Concrete `Backend` implementations the framework ships
/// (`CpuBackend` + the `execute_graph` walker that drives a
/// `GraphProto` through it).
pub use bb_ops::backends;

/// `RecordedModule` — the DSL → Compiler hand-off produced by
/// `Graph::finish()`.
pub use bb_dsl::recorded;

// --- bb-compiler pipeline re-exports -----------------------------

/// The compiler driver, `CompilerStage` trait, canonical pass list.
pub use bb_compiler as compiler;

// --- bb-runtime engine re-exports --------------------------------

/// Typed in-Node event bus.
pub use bb_runtime::bus;

/// The sans-IO Engine state machine.
pub use bb_runtime::engine;

/// Per-poll execution-state bundle owned by `Engine.exec`.
pub use bb_runtime::exec_state;

/// `WireEnvelope` codec; per-type decoders register through
/// `bb_ir::slot_value::register_type_node!`.
pub use bb_runtime::envelope;

/// Public error taxonomies.
pub use bb_runtime::errors;

/// Framework primitives bundled into every `RuntimeResourceRef`.
pub use bb_runtime::framework;

/// Lock-free MPMC ingress queue.
pub use bb_runtime::ingress;

/// Public `Node` + lazy build chain.
pub use bb_runtime::node;

/// Global inventory-collected registry for custom ops + concrete
/// components. Proc-macros emit `::bytesandbrains::registry::*`
/// paths into this namespace.
pub use bb_runtime::registry;

/// The `<Role>Runtime` role traits.
pub use bb_runtime::roles;

/// Runtime resource handle + `ComponentTimerKind`.
pub use bb_runtime::runtime;

/// `NodeSnapshot`.
pub use bb_runtime::snapshot;

/// `ai.bytesandbrains.syscall v1` opset (runtime-side
/// dispatch + DSL-side helpers).
pub use bb_runtime::syscall;

/// Optional OpenTelemetry layer constructors for the engine's
/// `tracing::` spans.
pub use bb_runtime::telemetry;

/// Test-only components, gated behind the `test-components` feature.

/// Thread-local `try_reserve_exact` fault-injection seam. Exposed
/// under `test-components` so integration tests in `tests/` drive
/// the same seam as crate-internal sibling tests.
pub use bb_runtime::fallible;

/// Concrete components the framework ships (syscalls, wire
/// transport, backends, role implementations, protocols). See the
/// `bb_ops` crate docs for the per-component authoring contract.
pub use bb_ops as ops;

// --- bb-derive macro re-exports ----------------------------------

/// `register_op!{}` proc-macro re-exported from `bb-derive`.
pub use bb_derive::register_op;
/// `register_protocol!{}` proc-macro re-exported from `bb-derive`.
pub use bb_derive::register_protocol;

/// `#[derive(bb::Aggregator)]` — bridges a Contract impl to
/// `AggregatorRuntime`.
pub use bb_derive::Aggregator;
/// `#[derive(bb::Backend)]` — bridges a Contract impl to
/// `BackendRuntime` and emits the `ai.onnx v1` opset declaration.
pub use bb_derive::Backend;
/// `#[derive(bb::Codec)]` — bridges a Contract impl to
/// `CodecRuntime`.
pub use bb_derive::Codec;
/// `#[derive(bb::Concrete)]` — emits the `ConcreteComponent`
/// + `AnyComponent` bridge and the `inventory::submit!` entry.
pub use bb_derive::Concrete;
/// `#[derive(bb::DataSource)]` — bridges a Contract impl to
/// `DataSourceRuntime`.
pub use bb_derive::DataSource;
/// `#[derive(bb::Index)]` — bridges a Contract impl to
/// `IndexRuntime`.
pub use bb_derive::Index;
/// `#[derive(bb::Model)]` — bridges a Contract impl to
/// `ModelRuntime`.
pub use bb_derive::Model;
/// `#[derive(bb::PeerSelector)]` — bridges a Contract impl to
/// `PeerSelectorRuntime`.
pub use bb_derive::PeerSelector;

// --- Convenience top-level re-exports ----------------------------

// Authoring surface.
pub use bb_dsl::Graph;
pub use bb_dsl::Module;
pub use bb_dsl::Output;
pub use bb_runtime::node::derivation::GenericSlotSpec;

// Module recording-time error.
pub use bb_dsl::BuildError;

// Node lifecycle.
pub use bb_runtime::engine::BootstrapInput;
pub use bb_runtime::engine::{EngineStats, EngineStep};
pub use bb_runtime::ingress::{IngressEvent, IngressQueue, IngressQueueRef};
pub use bb_runtime::node::{Node, NodeConfig};

// Compile + install entry points.
pub mod install;
pub use bb_compiler::Compiler;
pub use install::{install, Config, InstallError};

// Identity + addressing.
pub use bb_ir::ids::PeerId;
pub use bb_runtime::framework::Address;

// User-facing Contract traits.
pub use bb_dsl::contracts::{Aggregator, Backend, Codec, DataSource, Index, Model, PeerSelector};

// Library-level syscall sugar.
pub use bb_dsl::syscalls::{
    address_book_insert_many, address_book_lookup, announce, constant, gate_dispatch, pass_through,
};

// Value-type metadata.
pub use bb_ir::types::{TypeNode, TYPE_PEER_ID, TYPE_TENSOR_F32};

// Errors.
pub use bb_runtime::errors::{DeliveryError, RestoreError};

// Snapshot.
pub use bb_runtime::snapshot::{NodeSnapshot, TransientSnapshot};

/// Canonical authoring surface in a single import.
///
/// `use bytesandbrains::prelude::*;` covers `Module` + `Graph`,
/// `install` + `Compiler`, the Contract traits + matching derives,
/// `Node` + lifecycle types, and identity types. Reach into
/// sub-modules (`ops`, `placeholders`, `protocols`, `contracts`,
/// `completion`, …) for surfaces the prelude omits.
pub mod prelude {
    pub use crate::{install, Compiler, Config, InstallError};
    pub use crate::{Address, PeerId};
    pub use crate::{BuildError, Graph, Module, Output};
    pub use crate::{EngineStats, EngineStep, IngressEvent, Node, NodeConfig};
    // Trait + matching #[derive(bb::<Role>)] share a name (type vs.
    // macro namespace). `Concrete` is derive-only.
    pub use crate::completion::{CompletionHandle, ContractResponse};
    pub use crate::{Aggregator, Backend, Codec, Concrete, DataSource, Index, Model, PeerSelector};
}
