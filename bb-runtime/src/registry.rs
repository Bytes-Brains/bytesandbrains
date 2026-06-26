//! Global registries collected via `inventory`.
//!
//! lives in [`bb_ir::registry`] (foundation, shared by `bb-dsl` and
//! `bb-runtime`). This module re-exports it and adds the custom-op
//! registry, which depends on runtime-side types (`RuntimeResourceRef`,
//! `OpError`, `DispatchResult`) and therefore stays alongside the
//! engine.
//!
//! See the module-level docs in `bb_ir::registry` for the
//! concrete-component half's authoring patterns. The custom-op
//! registry follows the same `inventory::submit!` pattern; the
//! `bb::register_op!{}` declarative macro emits the submission for
//! library makers.

// Re-export the concrete-component registry so legacy
// `crate::registry::ConcreteComponentRegistration` paths (emitted by
// the bb-derive proc macros) keep resolving.
pub use bb_ir::component::DependencyDecl;
pub use bb_ir::registry::{
    concrete_components, find_concrete_component, inventory, ConcreteComponentRegistration,
};

// --- Unified op registration ---------------------------------------
//
// One `OpRegistration` covers every op the framework dispatches —
// user-shipped via `bb::register_op!` AND framework-shipped from
// `bb-ops`. `RegistrationKind::{Custom, Syscall}` discriminates
// which iterator surface returns the entry.

/// Type-erased dispatch fn for any op (custom OR syscall).
pub type OpInvokeFn = fn(
    &bb_ir::proto::onnx::NodeProto,
    &[(&str, &dyn crate::slot_value::SlotValue)],
    &mut crate::runtime::RuntimeResourceRef<'_>,
) -> Result<crate::atomic::DispatchResult, crate::bus::OpError>;

/// User-custom vs framework-syscall discriminator on
/// [`OpRegistration`]. Iterators surface entries filtered by kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistrationKind {
    /// User-shipped custom op (emitted by `bb::register_op!`).
    Custom,
    /// Framework-shipped syscall (emitted by every `bb-ops`
    /// component inventory submission).
    Syscall,
}

/// Single inventory entry covering every op the framework
/// dispatches. Library makers ship via `bb::register_op!`; bb-ops's
/// syscalls submit directly. Engine dispatch keys on
/// `(domain, op_type)` — no TypeId lookup.
pub struct OpRegistration {
    /// Op's `(domain, op_type)` key.
    pub domain: &'static str,
    /// Op type name.
    pub op_type: &'static str,
    /// Dispatch entry point.
    pub invoke: OpInvokeFn,
    /// Discriminator selecting which iterator surface returns this
    /// entry.
    pub kind: RegistrationKind,
}

inventory::collect!(OpRegistration);

/// Look up a Custom-kind op by its `(domain, op_type)` key. Used
/// by the engine's dispatch fallback for user-shipped ops outside
/// the role surface.
pub fn find_op(domain: &str, op_type: &str) -> Option<&'static OpRegistration> {
    inventory::iter::<OpRegistration>
        .into_iter()
        .find(|r| r.domain == domain && r.op_type == op_type && r.kind == RegistrationKind::Custom)
}

/// Iterate every Custom-kind op registration this binary links in.
pub fn ops() -> impl Iterator<Item = &'static OpRegistration> {
    inventory::iter::<OpRegistration>
        .into_iter()
        .filter(|r| r.kind == RegistrationKind::Custom)
}

/// Iterate every Syscall-kind registration. Consumed by
/// `Engine::register_all_framework_syscalls` at Node construction
/// time.
pub fn framework_syscalls() -> impl Iterator<Item = &'static OpRegistration> {
    inventory::iter::<OpRegistration>
        .into_iter()
        .filter(|r| r.kind == RegistrationKind::Syscall)
}

// --- Component-role registration ----------------------------------
//
// Every `#[derive(bb::<Role>)]` proc-macro emits one
// `ComponentRoleBinding` per `(component_type, role)`. The Engine
// merges them at install time so a single `ComponentRegistration`
// (the existing `ConcreteComponentRegistration` foundation entry +
// these per-role bindings) carries the full picture: state
// serialize/restore + every role the component implements.

/// One concrete framework role a component declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentRole {
    /// `Index` role — vector-index Contract.
    Index,
    /// `Aggregator` role — federated-aggregator Contract.
    Aggregator,
    /// `Model` role — ML-model Contract.
    Model,
    /// `Codec` role — bidirectional storage-type codec Contract.
    Codec,
    /// `DataSource` role — data-loader Contract.
    DataSource,
    /// `PeerSelector` role — peer-sampling Contract.
    PeerSelector,
    /// `Backend` role — tensor-compute Contract.
    Backend,
    /// `Protocol` role — custom-protocol Contract.
    Protocol,
}

/// Per-`(component_type, role)` inventory entry: each
/// `#[derive(bb::<Role>)]` proc-macro emits one of these alongside
/// the universal triple. `Node::ensure_ready` walks the channel +
/// joins by `type_name` to compute the role bitflags per
/// concrete component.
pub struct ComponentRoleBinding {
    /// `ConcreteComponent::TYPE_NAME` — the join key against
    /// `ConcreteComponentRegistration`.
    pub type_name: &'static str,
    /// The role the derive corresponds to.
    pub role: ComponentRole,
}

inventory::collect!(ComponentRoleBinding);

/// Per-`(component_type, role)` inventory carrying a per-T
/// dispatcher-registration fn pointer. Each role derive emits
/// one of these alongside [`ComponentRoleBinding`]; `install()`
/// walks the channel + calls `register_fn(engine)` for each
/// bound concrete so the engine learns about every role
/// dispatcher without the install path needing the typed `&T`.
pub struct DispatcherRegistration {
    /// `ConcreteComponent::TYPE_NAME` — the join key.
    pub type_name: &'static str,
    /// The role the registration corresponds to.
    pub role: ComponentRole,
    /// Per-T registration callback. The derive captures `T` at
    /// the emit site and the fn body calls
    /// `engine.register_<role>_dispatcher::<T>()`.
    pub register_fn: fn(&mut crate::engine::Engine),
}

inventory::collect!(DispatcherRegistration);

/// Look up the dispatcher-registration fn for
/// `(type_name, role)`. `install()` calls this for every binding
/// in the artifact's `BindingSpec`. Returns `None` when the role
/// derive isn't present in this binary's inventory (e.g. a
/// hand-implemented role trait without the derive).
pub fn dispatcher_for(
    type_name: &str,
    role: ComponentRole,
) -> Option<fn(&mut crate::engine::Engine)> {
    inventory::iter::<DispatcherRegistration>
        .into_iter()
        .find(|r| r.type_name == type_name && r.role == role)
        .map(|r| r.register_fn)
}

/// Per-concrete inventory carrying a per-T Bootstrap dispatcher-
/// registration fn pointer. The `#[derive(bb::Concrete)]` macro
/// emits one of these per registered concrete so the install path
/// can register every concrete's `Bootstrap` dispatcher without the
/// per-T downcast. Pairs with [`crate::engine::Engine::register_bootstrap_dispatcher`].
///
/// The derive's emitted impl falls back to the
/// [`crate::contracts::bootstrap::Bootstrap`] trait default (no-op),
/// so a Concrete with no manual `impl Bootstrap` still registers a
/// dispatcher entry that drains the Component bootstrap phase
/// cleanly. Authors override the default by hand-writing
/// `impl Bootstrap for X` alongside the derive — the derive's
/// `#[bb(bootstrap_override)]` attribute (parsed in
/// `bb-derive::parse`) suppresses the default-impl emission so the
/// two impls do not collide.
pub struct BootstrapDispatcherRegistration {
    /// `ConcreteComponent::TYPE_NAME` — the join key against
    /// `ConcreteComponentRegistration`.
    pub type_name: &'static str,
    /// Per-T registration callback. The derive captures `T` at
    /// the emit site and the fn body calls
    /// `engine.register_bootstrap_dispatcher::<T>()`.
    pub register_fn: fn(&mut crate::engine::Engine),
}

inventory::collect!(BootstrapDispatcherRegistration);

/// Look up the Bootstrap dispatcher-registration fn for `type_name`.
/// `install()` calls this for every registered concrete so the
/// Component bootstrap fire path can dispatch through the right impl.
/// Returns `None` when the concrete was registered without the
/// `#[derive(bb::Concrete)]` Bootstrap-bridge emission.
pub fn bootstrap_dispatcher_for(type_name: &str) -> Option<fn(&mut crate::engine::Engine)> {
    inventory::iter::<BootstrapDispatcherRegistration>
        .into_iter()
        .find(|r| r.type_name == type_name)
        .map(|r| r.register_fn)
}

/// Iterate every role binding for a given component type. Used by
/// `Node::ensure_ready` to compute the bitflags + by introspection
/// tools to discover what roles a struct implements.
pub fn roles_for_component(type_name: &str) -> impl Iterator<Item = ComponentRole> + use<'_> {
    inventory::iter::<ComponentRoleBinding>
        .into_iter()
        .filter(move |b| b.type_name == type_name)
        .map(|b| b.role)
}

// --- Storage-type registration ------------------------------------
//
// Every `#[derive(bb::<Role>)]` proc-macro emits one or two
// `StorageTypeEntry` structs (Codec emits two — one for In, one for
// Out) into the inventory carrier. The compiler binding step (Task 9)
// reads these to populate `BindingSlot.storage_types`. The fn-pointer
// slot is necessary because `Storage::TYPE` is a const on a generic
// associated type that can't be named at registry-definition time —
// the derive captures the monomorphized path in the fn body.

/// Per-concrete Storage-type entry registered by `#[derive(bb::<Role>)]`.
///
/// Keyed on the triple `(concrete_type_name, role_runtime, port)`.
/// `type_node_fn` returns the `Storage::TYPE` static for the
/// concrete's associated type at that port.
pub struct StorageTypeEntry {
    /// `ConcreteComponent::TYPE_NAME` — the join key against
    /// `ConcreteComponentRegistration`.
    pub concrete_type_name: &'static str,
    /// Engine-side runtime trait name (`"IndexRuntime"`, etc.).
    pub role_runtime: &'static str,
    /// Port name: `"vector"`, `"element"`, `"tensor"`, `"sample"`,
    /// `"in"`, or `"out"`.
    pub port: &'static str,
    /// Returns the `&'static TypeNode` for the role's Storage-bound
    /// associated type at this port. The derive captures the
    /// monomorphized path at emit time.
    pub type_node_fn: fn() -> &'static bb_ir::types::TypeNode,
}

inventory::collect!(StorageTypeEntry);

/// Iterate every registered Storage-type entry in this binary.
pub fn iter_storage_types() -> Vec<&'static StorageTypeEntry> {
    inventory::iter::<StorageTypeEntry>().collect()
}

/// Look up the `TypeNode` for a specific `(concrete_type_name, role_runtime, port)` triple.
///
/// Returns `None` when no matching entry exists in the binary's
/// inventory (e.g. the concrete's role derive is absent or the port
/// name is wrong).
pub fn lookup_storage_type(
    concrete_type_name: &str,
    role: &str,
    port: &str,
) -> Option<&'static bb_ir::types::TypeNode> {
    iter_storage_types()
        .into_iter()
        .find(|e| {
            e.concrete_type_name == concrete_type_name && e.role_runtime == role && e.port == port
        })
        .map(|e| (e.type_node_fn)())
}

