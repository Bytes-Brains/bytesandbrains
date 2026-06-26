//! Concrete-component inventory registry. Derive macros submit
//! per-type entries; the linker preserves them if the type is
//! referenced. Custom-op registry lives in `bb_runtime::registry`.

use crate::component::{ComponentPackage, ConstructFn, DependencyDecl, RestoreFn, SerializeFn};

/// Inventory entry for a concrete component, keyed by `TYPE_NAME`.
pub struct ConcreteComponentRegistration {
    /// `ConcreteComponent::TYPE_NAME`.
    pub type_name: &'static str,
    /// Package origin.
    pub package: ComponentPackage,
    /// Monomorphized `T::serialize`.
    pub serialize_fn: SerializeFn,
    /// Monomorphized `T::restore`.
    pub restore_fn: RestoreFn,
    /// Downcasts `&dyn Any` → `&T::Config` then calls `T::new`.
    pub construct_fn: ConstructFn,
    /// Mirror of `ConcreteComponent::DEPENDENCIES`.
    pub dependencies: &'static [DependencyDecl],
}

inventory::collect!(ConcreteComponentRegistration);

/// Look up a concrete by `TYPE_NAME`. `None` when unregistered in
/// this binary.
pub fn find_concrete_component(type_name: &str) -> Option<&'static ConcreteComponentRegistration> {
    inventory::iter::<ConcreteComponentRegistration>
        .into_iter()
        .find(|r| r.type_name == type_name)
}

/// Iterate every concrete-component registration this binary links.
pub fn concrete_components() -> impl Iterator<Item = &'static ConcreteComponentRegistration> {
    inventory::iter::<ConcreteComponentRegistration>.into_iter()
}

/// Re-export for derive-macro emit sites.
pub use inventory;
