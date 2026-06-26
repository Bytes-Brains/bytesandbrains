//! Framework-wide version constants.
//!
//! Four orthogonal version axes are checked at the install seam
//! between DSL → Compiler → Runtime, per
//! `docs-plan/CORRECTED_ARCHITECTURE.md` §Versioning:
//!
//! 1. [`FRAMEWORK_IR_VERSION`] — stamped by DSL into
//!    `ModelProto.metadata_props["ai.bytesandbrains.ir_version"]`;
//!    checked once at `Compiler::with_target_version` entry.
//! 2. `WireEnvelope.schema_version` — lives on the wire envelope
//!    (`bb-runtime`); checked by `EnvelopeCodec::decode_capped`.
//! 3. `NodeSnapshot.spec_version` — lives on the snapshot
//!    (`bb-runtime`); checked by `Node::restore`.
//! 4. [`crate::syscall_ids::SYSCALL_OPSET_VERSION`] (and sibling
//!    domain-versioned opsets) — stamped into
//!    `GraphProto.opset_import`; the runtime registers
//!    `atomic_dispatch` keyed by `(domain, op_type, version)`.

/// The current framework IR contract version. Bump when the
/// `ModelProto` invariants the compiler depends on change in a way
/// that older DSL emits cannot satisfy.
///
/// Stamped by `Module::build` into the canonical key
/// `"ai.bytesandbrains.ir_version"` on `ModelProto.metadata_props`.
pub const FRAMEWORK_IR_VERSION: u32 = 1;

/// Canonical metadata-props key under which
/// [`FRAMEWORK_IR_VERSION`] is stamped on each `ModelProto`.
pub const FRAMEWORK_IR_VERSION_KEY: &str = "ai.bytesandbrains.ir_version";
