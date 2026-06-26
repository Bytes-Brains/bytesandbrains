//! Compiler-internal binding accumulator.
//!
//! `BindingSpec` is the typed slot-binding structure the
//! `Compiler::compile()` pipeline uses to thread bindings from
//! `Compiler::bind_<role>::<T>("slot")` calls through
//! `resolve_component_dependencies` + `validate_all_slots_bound`
//! and into the final `stamp_compilation_metadata` pass. It does
//! NOT cross the compile boundary — once the pass writes its
//! contents into `ModelProto.metadata_props`, the typed struct is
//! discarded.
//!
//! Per the proto-at-every-boundary commitment, the public surface
//! of `Compiler::compile()` is `Result<ModelProto, CompileError>`.
//! Install reads the binding entries back off the proto; no Rust
//! struct travels alongside the IR.

/// One slot in the binding accumulator. Pairs the author-chosen
/// slot name with the role-trait identifier and the bound concrete
/// type's `TYPE_NAME`. `storage_types` carries per-port
/// `TypeNode` statics extracted from the inventory at bind time.
#[derive(Clone, Debug)]
pub(crate) struct BindingSlot {
    /// Author-chosen slot name.
    pub slot: String,
    /// Role-trait name (e.g. `"BackendRuntime"`, `"IndexRuntime"`).
    pub role: String,
    /// `TYPE_NAME` of the bound concrete type.
    pub concrete_type_name: String,
    /// Per-port Storage `TypeNode` statics populated by
    /// `bind_concrete_with_storage` at the call site.
    /// Keyed by port label (`"vector"`, `"element"`, `"in"`, …).
    /// Empty for roles without a Storage-bound associated type
    /// (e.g. `PeerSelector`) and for slots constructed directly
    /// in tests (where the registry lookup hasn't run).
    ///
    /// Read by `refine_polymorphic_value_info` to narrow placeholder
    /// denotations to the bound concrete's `Storage::TYPE`.
    pub storage_types: Vec<(&'static str, &'static bb_ir::types::TypeNode)>,
}

impl BindingSlot {
    /// Return the `TypeNode` for the given port, or `None` when the
    /// port has no entry in `storage_types`. Used by
    /// `refine_polymorphic_value_info` to implement graceful
    /// degradation for hand-implemented concretes whose
    /// `storage_types` is empty.
    pub(crate) fn storage_type_opt(&self, port: &str) -> Option<&'static bb_ir::types::TypeNode> {
        self.storage_types
            .iter()
            .find(|(k, _)| *k == port)
            .map(|(_, t)| *t)
    }
}

/// Ordered list of slots the install path must populate.
///
/// Compiler-internal: built by `Compiler::compile()` from
/// `self.bindings`, consumed by the canonical-pipeline validation
/// passes + the stamp pass, then dropped.
#[derive(Clone, Debug, Default)]
pub(crate) struct BindingSpec {
    /// One entry per named slot.
    pub slots: Vec<BindingSlot>,
}

impl BindingSpec {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn push(
        &mut self,
        slot: impl Into<String>,
        role: impl Into<String>,
        concrete: impl Into<String>,
    ) {
        self.slots.push(BindingSlot {
            slot: slot.into(),
            role: role.into(),
            concrete_type_name: concrete.into(),
            storage_types: Vec::new(),
        });
    }

    /// Push a slot with pre-resolved per-port Storage `TypeNode`
    /// statics. Called by `Compiler::bind_concrete_with_storage`
    /// after the registry lookup at bind time.
    pub(crate) fn push_with_storage(
        &mut self,
        slot: impl Into<String>,
        role: impl Into<String>,
        concrete: impl Into<String>,
        storage_types: Vec<(&'static str, &'static bb_ir::types::TypeNode)>,
    ) {
        self.slots.push(BindingSlot {
            slot: slot.into(),
            role: role.into(),
            concrete_type_name: concrete.into(),
            storage_types,
        });
    }

    pub(crate) fn get(&self, slot: &str) -> Option<&BindingSlot> {
        self.slots.iter().find(|s| s.slot == slot)
    }

    /// Look up a slot by its role-runtime identifier. The `role`
    /// argument is the `ai.bytesandbrains.required_trait` value
    /// stamped on Contract-method NodeProtos by the DSL recorder
    /// (e.g. `"IndexRuntime"`, `"BackendRuntime"`).
    ///
    /// Returns the first slot whose `role` field matches. When multiple
    /// slots share the same role (uncommon) only the first is returned;
    /// downstream passes that need per-slot disambiguation should also
    /// compare `slot_id` from the IR.
    ///
    /// Downstream consumer: `refine_polymorphic_value_info` (Task 10).
    pub(crate) fn lookup_by_role(&self, role: &str) -> Option<&BindingSlot> {
        self.slots.iter().find(|s| s.role == role)
    }
}

