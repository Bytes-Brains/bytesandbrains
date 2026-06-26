#![warn(missing_docs)]
//! `bb-runtime` — sans-IO engine. Hosts Node + Engine + framework
//! primitives + role-runtime traits + syscall registration +
//! snapshot. Consumes compiled `ModelProto`s from `bb-compiler`.
//! Concrete components live in `bb-ops`.

#![allow(rustdoc::broken_intra_doc_links)]

// bb-derive emits `::bytesandbrains::*` paths; resolve them locally.
extern crate self as bytesandbrains;

/// `ConcreteComponent` polymorphism contract + the
/// `ComponentHandle` fn-pointer-capture wrapper.
pub mod concrete;

/// `ExecState` — per-poll execution-state bundle (frontier,
/// slot table, pending state, scheduler, inbound contexts,
/// monotonic ID allocator). Owned by `Engine` as one field.
pub mod exec_state;

/// Engine identifier types — `PeerId`, `NodeSiteId`, `OpRef`,
/// `ComponentRef`, `ExecId`, `CommandId`, `RequestId`, `OpsetId`,
/// `ComponentTag`.
pub mod ids;

/// `AtomicOpsetDecl`, `AtomicOpDecl`, `AtomicOpKind`, `DispatchResult`.
pub mod atomic;

/// The universal `SlotValue` trait.
pub mod slot_value;

/// `CompletionHandle`, `CompletionSink`, `ContractResponse`.
pub mod completion;

/// `AnyComponent`, `ErasedComponent`, `ComponentPackage`, `RestoreError`.
pub mod component;

/// User-facing Contract traits (`Index`, `Backend`, `Aggregator`, …).
pub mod contracts;

/// Typed in-Node event bus.
pub mod bus;

/// The sans-IO Engine state machine.
pub mod engine;

/// `WireEnvelope` codec; per-type decoders register through
/// `bb_ir::slot_value::register_type_node!`.
pub mod envelope;

/// Public error taxonomies.
pub mod errors;

/// `try_reserve_exact` wrapper at ingress boundaries so allocator
/// failures surface as typed events. `fallible::testing` is a
/// stub-allocator seam under `test-components`.
#[cfg(any(test, feature = "test-components"))]
pub mod fallible;

#[cfg(not(any(test, feature = "test-components")))]
pub(crate) mod fallible;

/// Framework primitives bundled into every `RuntimeResourceRef`.
pub mod framework;

/// Lock-free MPMC ingress queue.
pub mod ingress;

/// Public `Node` + lazy build chain.
pub mod node;

/// Global inventory-collected registry for custom ops.
pub mod registry;

/// The `<Role>Runtime` role traits.
pub mod roles;

/// Runtime resource handle + `ComponentTimerKind`.
pub mod runtime;

/// `NodeSnapshot`.
pub mod snapshot;

/// Foundation `SlotValue` impls - `PeerIdValue`, `WireReqIdValue`,
/// `TriggerValue`, `BytesValue`, `CommandIdValue`.
pub mod syscall;

/// Optional OpenTelemetry layer constructors for the engine's
/// `tracing::` spans.
pub mod telemetry;
