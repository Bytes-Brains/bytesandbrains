#![warn(missing_docs)]
#![allow(rustdoc::broken_intra_doc_links)]

//! The compilation pipeline. Each pass is a pure function over
//! `GraphProto` / `FunctionProto`. Composition flattens at
//! authoring time, so no compiler-side sub-Module inlining pass
//! exists. `partition_by_wire_ops` slices at wire ops via
//! reachability + names partitions by longest common
//! `module_instance` prefix.

pub mod analyze_wire_edges;
pub(crate) mod artifact;
pub mod derive_wire_deadlines;
pub mod driver;
pub mod error;
pub mod expand_ops;
pub(crate) mod function_dedup;
pub mod gate_contract;
pub mod infer_peer_classes;
pub mod inline_for_partition;
pub mod insert_async_deadlines;
pub mod insert_backoff_gate_rx;
pub mod insert_backoff_gate_tx;
pub mod insert_dedup_gate_rx;
pub mod insert_peer_health_gate_rx;
pub mod insert_peer_health_gate_tx;
pub mod partition_by_wire_ops;
pub mod refine_polymorphic_value_info;
pub mod resolve_component_dependencies;
pub mod resolve_slots;
pub mod runner;
pub mod rx_chain;
pub mod stamp_compilation_metadata;
pub mod synthesize_wire_recvs;
pub mod type_solver;
pub mod validate;
pub mod validate_all_slots_bound;
pub mod validate_bootstrap_composition;
pub mod validate_runtime_complete;
pub mod verify_no_dangling_calls;

pub use analyze_wire_edges::analyze_wire_edges;
pub use derive_wire_deadlines::derive_wire_deadlines;
pub use driver::{Compiler, CompilerStage, PassError};
pub use error::{CompileError, SlotSource, ValidationError};
pub use expand_ops::expand_ops;
pub use inline_for_partition::inline_for_partition;
pub use insert_async_deadlines::insert_async_deadlines;
pub use insert_backoff_gate_rx::insert_backoff_gate_rx;
pub use insert_backoff_gate_tx::insert_backoff_gate_tx;
pub use insert_dedup_gate_rx::insert_dedup_gate_rx;
pub use insert_peer_health_gate_rx::insert_peer_health_gate_rx;
pub use insert_peer_health_gate_tx::insert_peer_health_gate_tx;
pub use partition_by_wire_ops::{partition_by_wire_ops, NetworkAnalysis, WireEdge};
pub use resolve_slots::resolve_slots;
pub use runner::CANONICAL_PASS_NAMES;
pub use stamp_compilation_metadata::stamp_for_test;
pub use synthesize_wire_recvs::synthesize_wire_recvs;
pub use type_solver::{TypeError, TypeSolution, TypeSolver};
pub use validate::validate;
pub use validate_bootstrap_composition::validate_bootstrap_composition;
pub use validate_runtime_complete::validate_runtime_complete;
pub use verify_no_dangling_calls::verify_no_dangling_calls;
