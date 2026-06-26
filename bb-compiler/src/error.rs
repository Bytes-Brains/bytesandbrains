//! Compiler error taxonomies. `ValidationError` is exclusive to
//! `validate`; `CompileError` covers everything else and wraps
//! `ValidationError` via `From`.

/// Errors from `validate` (pass 1). One variant per
/// `docs/COMPILER.md` §4.1 rule.
#[derive(Debug)]
pub enum ValidationError {
    /// Rule 1 - unknown `(op_type, domain)` pair.
    UnknownOp {
        /// `NodeProto.name`.
        node_name: String,
        /// `NodeProto.op_type`.
        op_type: String,
        /// `NodeProto.domain`.
        domain: String,
    },

    /// Rule 2 - an input value name has no producer.
    DanglingInput {
        /// `NodeProto.name`.
        node_name: String,
        /// The dangling input value name.
        input_name: String,
    },

    /// Rule 3 - two ops claim to produce the same output value name.
    DuplicateOutput {
        /// The duplicated value name.
        value_name: String,
        /// First producer (`NodeProto.name`).
        node_a: String,
        /// Second producer (`NodeProto.name`).
        node_b: String,
    },

    /// Rule 5 - a function input has no matching `ValueInfoProto.type`.
    MissingTypeInfo {
        /// The input value name lacking a type.
        input_name: String,
    },

    /// Rule 6 - a role-domain NodeProto lacks the canonical metadata
    /// keys (`concrete_type` + `instance` OR `required_trait` +
    /// `slot_id`).
    MalformedSlotMetadata {
        /// The offending node's name.
        node_name: String,
        /// Human-readable detail.
        detail: String,
    },

    /// Rule 7 - the graph contains at least one cycle.
    CyclicGraph {
        /// Node names involved in the cycle.
        involves: Vec<String>,
    },

    /// Rule 8 - an op uses an opset that was not declared in
    /// `ModelProto.opset_import`.
    OpsetNotImported {
        /// The missing opset's domain.
        domain: String,
        /// The version the graph used.
        version_used: i64,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownOp {
                node_name,
                op_type,
                domain,
            } => write!(f, "unknown op {domain}::{op_type} at node {node_name}",),
            Self::DanglingInput {
                node_name,
                input_name,
            } => write!(f, "dangling input {input_name} at node {node_name}",),
            Self::DuplicateOutput {
                value_name,
                node_a,
                node_b,
            } => write!(
                f,
                "duplicate output {value_name} produced by both {node_a} and {node_b}",
            ),
            Self::MissingTypeInfo { input_name } => {
                write!(f, "missing type info for input {input_name}")
            }
            Self::MalformedSlotMetadata { node_name, detail } => {
                write!(f, "malformed slot metadata at {node_name}: {detail}")
            }
            Self::CyclicGraph { involves } => {
                write!(f, "graph contains a cycle involving {involves:?}")
            }
            Self::OpsetNotImported {
                domain,
                version_used,
            } => {
                write!(f, "opset {domain} v{version_used} used but not imported")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Errors surfaced by any compiler pass beyond `validate`.
#[derive(Debug)]
pub enum CompileError {
    /// Pass 1 - wrapped [`ValidationError`].
    Validation(ValidationError),

    /// Pass 2 - `expand_ops` failed for a specific op.
    ExpansionFailed {
        /// `NodeProto.op_type`.
        op_type: String,
        /// `NodeProto.domain`.
        domain: String,
        /// Human-readable detail.
        reason: String,
    },

    /// Pass 7 - `runtime.<op>()` returned an error during role-method
    /// inlining.
    RoleMethodFailed {
        /// Slot name (e.g. `"backend"`, `"model"`).
        slot: String,
        /// `NodeProto.op_type` that triggered the inlining.
        op_type: String,
        /// Source error reported by the runtime impl.
        source: String,
    },

    /// Pass 11 - a concrete impl satisfying role R coexists with a
    /// generic placeholder of role R in the same Module. The runner
    /// surfaces this as `BuildError::AmbiguousRole` ().
    AmbiguousRole {
        /// Role domain (e.g. `"ai.bytesandbrains.role.index"`).
        role: String,
        /// Concrete `TYPE_NAME` providing the role.
        concrete_type: String,
        /// Slot id of the conflicting generic placeholder.
        generic_slot_id: u32,
    },

    /// `infer_peer_classes` - a `wire.Send`'s peer input has no
    /// declared `peer_class` (neither on the input's
    /// `ValueInfoProto` nor on the producing NodeProto's metadata).
    /// The compiler can't decide which class of Node owns the send's
    /// data output.
    UnresolvedPeerClass {
        /// `NodeProto.name` of the offending send.
        node_name: String,
        /// Value name of the peer input lacking a class.
        peer_input: String,
    },

    /// `infer_peer_classes` - a non-wire NodeProto consumes two values
    /// from different home classes. Either the user threaded a value
    /// from one peer's partition into another's compute without a
    /// `wire.send` between them, or a frontend forgot to mark a peer
    /// input as ambient.
    CrossClassDataflow {
        /// `NodeProto.name` of the offending consumer.
        node_name: String,
        /// One of the conflicting home classes.
        home_a: String,
        /// The other conflicting home class.
        home_b: String,
    },

    /// the input `ModelProto`'s stamped
    /// `FRAMEWORK_IR_VERSION` doesn't match what this compiler was
    /// built to consume. Surfaced by
    /// [`crate::driver::Compiler::with_target_version`] / the
    /// driver-entry check before any pass runs.
    IrVersionMismatch {
        /// Version the compiler expects.
        expected: u32,
        /// Version the input model carries.
        got: u32,
    },

    /// a binding the compiler required (a generic-slot
    /// concrete impl, a peer attribute, etc.) was not present at
    /// the offending site. Replaces the catch-all `Internal` for
    /// the missing-binding failure mode so consumers can match on
    /// shape and surface actionable diagnostics.
    MissingBinding {
        /// Stable slot identifier (e.g. `"ATTR_PEER"`, `"backend"`,
        /// `"required_trait:IndexRuntime"`).
        slot: String,
        /// Where the requirement was raised (typically
        /// `NodeProto.name` or `(function_name, node_index)` as a
        /// composite string).
        site: String,
    },

    /// `ModelProto.functions` was empty when the
    /// compiler needed at least the root function the recorder
    /// produces from `Module::body`. Distinct from
    /// `Validation(ValidationError)` because it surfaces from the
    /// driver entry, not a pass body.
    EmptyFunctionTable,

    /// `validate_runtime_complete` found the compiled
    /// model is missing a runtime requirement (e.g. a NodeProto
    /// whose op is not registered, or a gate that should have been
    /// inserted but wasn't).
    RuntimeIncomplete {
        /// Human-readable description of what's missing.
        missing: String,
    },

    /// Catch-all for orchestrator-level failures (e.g. ill-formed
    /// recorded module). Carries enough detail to debug.
    Internal {
        /// Human-readable failure detail.
        detail: String,
    },

    /// `type_solver` - a [`TypeRelation`] reported `Failed` while
    /// running against the constraint network. The op's relations
    /// can't be satisfied together with the seeded inputs.
    TypeConstraintFailed {
        /// Op the failing relation was attached to (`domain::op_type`).
        op: String,
        /// Diagnostic detail from the relation.
        detail: String,
    },

    /// `type_solver` (strict mode) - a value slot reached fixpoint
    /// still bound to an abstract TypeNode. The graph is under-
    /// constrained; either a seed is missing or an op's
    /// `type_relations` declarations are insufficient.
    UnresolvedType {
        /// Value name that didn't narrow to a concrete leaf.
        value: String,
    },

    /// `resolve_component_dependencies` - a concrete component
    /// declared `#[depends(<role> = "<slot>")]` for a slot that
    /// has no binding in the compiled artifact's spec. The user
    /// supplied an `index` binding but forgot the `backend`
    /// binding the index needs.
    UnboundDependency {
        /// `TYPE_NAME` of the concrete with the unsatisfied dep.
        component: String,
        /// Slot the component was bound at.
        bound_at_slot: String,
        /// The role the dependency requires
        /// (e.g. `"Backend"`).
        required_role: String,
        /// The slot name the dep points at.
        required_slot: String,
    },

    /// A NodeProto references a port name the module didn't
    /// record in its body via `g.input` / `g.output` / `g.net_out`
    /// / `g.net_in`.
    UnknownPort {
        /// Module that references the bad port.
        module: String,
        /// The bad port name.
        port: String,
    },

    /// A declared port was neither read (input) nor written
    /// (output) inside its module's body.
    PortUnwired {
        /// Module declaring the unwired port.
        module: String,
        /// The unwired port name.
        port: String,
        /// `"Input"` or `"Output"`.
        direction: String,
    },

    /// A `wire.Send` / `wire.Recv` op was found that doesn't
    /// land on a module's declared network boundary. Internal
    /// wire ops are forbidden — every wire boundary must coincide
    /// with a `g.net_out` / `g.net_in` recording.
    NetworkOpNotAtBoundary {
        /// Module hosting the off-boundary wire op.
        module: String,
        /// Op identifier (NodeProto.name).
        op_id: String,
    },

    /// `refine_polymorphic_value_info` — a Contract-method NodeProto
    /// carries `ai.bytesandbrains.slot_id` metadata that is missing or
    /// not a valid `u32`.
    InvalidSlotId {
        /// `NodeProto.name` of the offending node.
        node: String,
        /// The raw metadata value that failed to parse.
        value: String,
    },

    /// `refine_polymorphic_value_info` — the slot_id on a
    /// Contract-method NodeProto does not correspond to any slot in
    /// the compiled artifact's `BindingSpec`. Indicates the binding
    /// chain is missing an entry for the role this node requires.
    UnknownSlotId {
        /// `NodeProto.name` of the offending node.
        node: String,
        /// The slot_id that was not found.
        slot_id: u32,
    },

    /// `refine_polymorphic_value_info` — a Contract-method NodeProto
    /// declares a `ai.bytesandbrains.required_trait` value that the
    /// pass does not recognise as a known role-runtime identifier.
    UnknownRoleRuntime {
        /// `NodeProto.name` of the offending node.
        node: String,
        /// The unrecognised role string.
        role: String,
    },

    /// `refine_polymorphic_value_info` — a `CodecRuntime` NodeProto
    /// is missing the `ai.bytesandbrains.codec.port` metadata entry
    /// that indicates whether this node is an encode (`"out"`) or
    /// decode (`"in"`) operation.
    MissingCodecPortMetadata {
        /// `NodeProto.name` of the offending node.
        node: String,
    },

    /// `refine_polymorphic_value_info` — a `CodecRuntime` NodeProto
    /// carries a `ai.bytesandbrains.codec.port` value that is neither
    /// `"in"` nor `"out"`.
    InvalidCodecPort {
        /// `NodeProto.name` of the offending node.
        node: String,
        /// The invalid port value.
        value: String,
    },

    /// `refine_polymorphic_value_info` — two or more slots in the
    /// `BindingSpec` share the same `role_runtime` identifier (e.g.
    /// two `.bind_index::<A>("local").bind_index::<B>("remote")`
    /// calls produce two slots both with `role = "IndexRuntime"`).
    /// The pass uses `lookup_by_role` which returns only the first
    /// match, so it would silently apply the wrong concrete's storage
    /// type to nodes belonging to the other slot. Until slot_id-keyed
    /// lookup is implemented this ambiguity is a hard error.
    AmbiguousRoleBinding {
        /// The role string shared by multiple slots.
        role: String,
        /// Author-chosen slot names that share the role.
        slot_names: Vec<String>,
    },

    /// `refine_polymorphic_value_info` — a NodeProto carries
    /// `ai.bytesandbrains.slot_id` metadata (marking it as a
    /// Contract-method node) but lacks the companion
    /// `ai.bytesandbrains.required_trait` metadata. The DSL recorder
    /// always stamps both; a missing `required_trait` indicates a
    /// malformed IR that would cause the pass to silently mis-route
    /// the refinement.
    MissingRequiredTraitMetadata {
        /// `NodeProto.name` of the offending node.
        node: String,
    },

    /// `resolve_component_dependencies` - the slot a component's
    /// dep points at is bound to a concrete whose declared role
    /// set does NOT include the required role. The user bound
    /// the right slot to the wrong KIND of concrete.
    DependencyRoleMismatch {
        /// `TYPE_NAME` of the concrete with the dep.
        component: String,
        /// Slot the component was bound at.
        bound_at_slot: String,
        /// The role the dep requires.
        required_role: String,
        /// The slot name the dep points at.
        required_slot: String,
        /// The role(s) the bound concrete at `required_slot`
        /// actually provides.
        provided_role: String,
    },

    /// `validate_all_slots_bound` - the compiled artifact has at
    /// least one slot the install path would need a concrete for
    /// that the bind chain didn't supply. Source identifies why
    /// the slot is required so the diagnostic can point the user
    /// at exactly which `.bind_<role>::<T>("…")` is missing.
    UnboundSlot {
        /// Author-chosen role identifier (PascalCase Contract
        /// role name, e.g. `"Backend"`, `"Index"`).
        role: String,
        /// Where the requirement comes from.
        source: SlotSource,
    },

    /// `validate_bootstrap_composition` — a CALL inside a bootstrap
    /// function points at a target name that has no matching
    /// FunctionProto. The most common cause is a parent Module's
    /// `bootstrap` recording calling `self.child.call().bootstrap(g)`
    /// without the child's bootstrap recording reaching the
    /// `Module::build` output (e.g. an empty `bootstrap` override that
    /// `build` drops on the floor).
    BootstrapCompositionGap {
        /// Bootstrap function whose body emits the orphan CALL.
        caller: String,
        /// Missing FunctionProto name the CALL points at.
        target: String,
    },

    /// `validate_bootstrap_composition` — the bootstrap function-call
    /// graph contains a cycle. Bootstrap is a one-shot drain; a cycle
    /// would wedge the engine in `bootstrap_pending` forever.
    BootstrapCompositionCycle {
        /// Function names traversed in the cycle, with the repeated
        /// node appearing at both ends so the path reads naturally.
        involves: Vec<String>,
    },

    /// `type_solver` — a wire edge carries a concrete storage type on
    /// the send side that does not match the concrete storage type
    /// declared on the receive side, and no `Codec` bridge is
    /// wired between them. Reading the hint: add a
    /// `Codec<In=<actual_id>, Out=<expected_id>>` node on the edge so
    /// the encoder/decoder pair converts between the two storage
    /// representations. Quantization methods are not substitutable
    /// casts — the author must choose the right Codec impl.
    IncompatibleStorageOnEdge {
        /// Value name produced by the upstream send-side node.
        src: String,
        /// Value name expected by the downstream receive-side node.
        dst: String,
        /// `TypeNode.id` string the receive side declares
        /// (e.g. `"tensor.u8"`).
        expected_id: &'static str,
        /// `TypeNode.id` string the send side resolved to
        /// (e.g. `"tensor.f32"`).
        actual_id: &'static str,
    },
}

/// Why a slot needs to be bound. Threaded into
/// [`CompileError::UnboundSlot`] so the diagnostic can point at
/// the originating location.
#[derive(Debug, Clone)]
pub enum SlotSource {
    /// At least one NodeProto in the IR references a role via
    /// `(required_trait, slot_id)` metadata but no `BindingSpec`
    /// entry of that role exists. The user forgot the
    /// `.bind_<role>::<T>("…")` for a placeholder field their
    /// Module body actually uses.
    DirectPlaceholder,

    /// The slot is required because the named concrete declares
    /// `#[depends(<role> = "<slot>")]`, but no matching binding
    /// exists. Diagnostic: *"backend slot 'compute' is required
    /// by CountingIndex bound at slot 'primary_index' but isn't
    /// bound."*
    DependencyOf {
        /// `TYPE_NAME` of the concrete with the dep.
        component: String,
        /// Slot the dep-declaring concrete was bound at.
        bound_at_slot: String,
        /// Slot name the dependency references.
        required_slot: String,
    },
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(e) => write!(f, "{e}"),
            Self::ExpansionFailed { op_type, domain, reason } => write!(
                f,
                "op expansion failed for {domain}::{op_type}: {reason}",
            ),
            Self::RoleMethodFailed { slot, op_type, source } => write!(
                f,
                "role method {op_type} (slot {slot}) failed: {source}",
            ),
            Self::AmbiguousRole { role, concrete_type, generic_slot_id } => write!(
                f,
                "ambiguous role {role}: both concrete {concrete_type} and generic slot {generic_slot_id} provide it",
            ),
            Self::UnresolvedPeerClass { node_name, peer_input } => write!(
                f,
                "wire.Send {node_name} has no declared peer_class for peer input {peer_input}",
            ),
            Self::CrossClassDataflow { node_name, home_a, home_b } => write!(
                f,
                "node {node_name} consumes values from {home_a} and {home_b} without a wire.send between them",
            ),
            Self::IrVersionMismatch { expected, got } => write!(
                f,
                "IR version mismatch: compiler expects v{expected}, model carries v{got}",
            ),
            Self::MissingBinding { slot, site } => write!(
                f,
                "missing binding for slot `{slot}` at {site}",
            ),
            Self::EmptyFunctionTable => f.write_str(
                "ModelProto.functions is empty — the recorder produced no FunctionProto",
            ),
            Self::RuntimeIncomplete { missing } => write!(
                f,
                "compiled model is not runtime-complete: missing {missing}",
            ),
            Self::Internal { detail } => write!(f, "compiler internal error: {detail}"),
            Self::TypeConstraintFailed { op, detail } => write!(
                f,
                "type constraint failed at {op}: {detail}",
            ),
            Self::UnresolvedType { value } => write!(
                f,
                "value `{value}` did not resolve to a concrete type",
            ),
            Self::UnboundDependency {
                component,
                bound_at_slot,
                required_role,
                required_slot,
            } => write!(
                f,
                "{component} (bound at slot `{bound_at_slot}`) requires a {required_role} \
                 at slot `{required_slot}`, but no such slot is bound",
            ),
            Self::UnboundSlot { role, source } => match source {
                SlotSource::DirectPlaceholder => write!(
                    f,
                    "no `.bind_<role>::<T>(\"…\")` supplied a {role} concrete; the Module body \
                     uses a {role} placeholder that the compiler must fill at install time",
                ),
                SlotSource::DependencyOf {
                    component,
                    bound_at_slot,
                    required_slot,
                } => write!(
                    f,
                    "{component} (bound at slot `{bound_at_slot}`) requires a {role} at slot \
                     `{required_slot}`, but the bind chain doesn't include it; add \
                     `.bind_{}::<...>(\"{required_slot}\")`",
                    role.to_ascii_lowercase(),
                ),
            },
            Self::DependencyRoleMismatch {
                component,
                bound_at_slot,
                required_role,
                required_slot,
                provided_role,
            } => write!(
                f,
                "{component} (bound at slot `{bound_at_slot}`) requires a {required_role} \
                 at slot `{required_slot}`, but the slot is bound to a {provided_role}",
            ),
            Self::UnknownPort { module, port } => write!(
                f,
                "module `{module}` references port `{port}` that it did not declare",
            ),
            Self::PortUnwired { module, port, direction } => write!(
                f,
                "module `{module}` port `{port}` ({direction}) is declared but not wired in the body",
            ),
            Self::NetworkOpNotAtBoundary { module, op_id } => write!(
                f,
                "module `{module}` op `{op_id}` is a wire.Send/Recv but is not at a declared network port",
            ),
            Self::InvalidSlotId { node, value } => write!(
                f,
                "node `{node}` has invalid or missing `ai.bytesandbrains.slot_id` metadata: `{value}`",
            ),
            Self::UnknownSlotId { node, slot_id } => write!(
                f,
                "node `{node}` references slot_id {slot_id} which has no corresponding binding in BindingSpec",
            ),
            Self::UnknownRoleRuntime { node, role } => write!(
                f,
                "node `{node}` declares unknown role runtime `{role}` in `ai.bytesandbrains.required_trait`",
            ),
            Self::MissingCodecPortMetadata { node } => write!(
                f,
                "codec node `{node}` is missing `ai.bytesandbrains.codec.port` metadata (expected `in` or `out`)",
            ),
            Self::InvalidCodecPort { node, value } => write!(
                f,
                "codec node `{node}` has invalid `ai.bytesandbrains.codec.port` value `{value}` (expected `in` or `out`)",
            ),
            Self::AmbiguousRoleBinding { role, slot_names } => write!(
                f,
                "multiple slots share role `{role}` (slots: {slot_names:?}); \
                 polymorphic refinement requires slot_id discriminator support (TODO follow-up)",
            ),
            Self::MissingRequiredTraitMetadata { node } => write!(
                f,
                "node `{node}` declares slot_id without required_trait metadata",
            ),
            Self::IncompatibleStorageOnEdge {
                src,
                dst,
                expected_id,
                actual_id,
            } => write!(
                f,
                "port `{dst}` expects {expected_id}; upstream `{src}` outputs {actual_id}. \
                 Insert a `Codec<In={actual_id}, Out={expected_id}>` on the edge.",
            ),
            Self::BootstrapCompositionGap { caller, target } => write!(
                f,
                "bootstrap function `{caller}` calls `{target}`, which has no FunctionProto in the model",
            ),
            Self::BootstrapCompositionCycle { involves } => write!(
                f,
                "bootstrap composition cycle: {}",
                involves.join(" → "),
            ),
        }
    }
}

impl From<crate::type_solver::TypeError> for CompileError {
    fn from(e: crate::type_solver::TypeError) -> Self {
        match e {
            crate::type_solver::TypeError::ConstraintFailed { op, detail } => {
                Self::TypeConstraintFailed { op, detail }
            }
            crate::type_solver::TypeError::UnresolvedType { value } => {
                Self::UnresolvedType { value }
            }
            crate::type_solver::TypeError::PortOutOfRange { op, port } => Self::Internal {
                detail: format!("port {port:?} out of range on op {op}"),
            },
        }
    }
}

impl std::error::Error for CompileError {}

impl From<ValidationError> for CompileError {
    fn from(e: ValidationError) -> Self {
        Self::Validation(e)
    }
}
