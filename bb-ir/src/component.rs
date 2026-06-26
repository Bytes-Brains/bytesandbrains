//! Foundation polymorphism plumbing shared by `bb-dsl` (authoring)
//! and `bb-runtime` (dispatch).

use std::any::Any;

/// Marker for engine-owned component instances. Blanket impl covers
/// every `Any + Send + Sync`.
pub trait ErasedComponent: Any + Send + Sync {}

impl<T: Any + Send + Sync> ErasedComponent for T {}

/// Dyn-safe downcast surface. No blanket impl — `Box<dyn AnyComponent>`
/// would otherwise shadow per-type vtables. `bb::Concrete` derive
/// emits the impl.
pub trait AnyComponent: ErasedComponent {
    /// Downcast view - immutable.
    fn as_any(&self) -> &dyn Any;

    /// Downcast view - mutable.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Component-package origin tag. Surfaces in introspection +
/// telemetry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ComponentPackage {
    /// Shipped in the bytesandbrains crate family.
    Framework,

    /// Shipped by the application author. Default.
    Application,
}

/// Author-declared sibling dependency at a named slot. Compiler
/// verifies role match; runtime reaches it via
/// `RuntimeResourceRef::dependency::<T>(slot)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DependencyDecl {
    /// PascalCase role identifier. Plain string so `bb-ir` stays
    /// free of the `ComponentRole` enum.
    pub role: &'static str,

    /// Slot name in the compiler's binding spec.
    pub slot: &'static str,
}

/// Error variants surfaced by `ConcreteComponent::restore`.
#[derive(Debug)]
pub enum RestoreError {
    /// Deserialization failed at the framework's bincode boundary.
    Malformed(bincode::Error),

    /// Impl-specific restore failure.
    Custom(String),
}

impl std::fmt::Display for RestoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(e) => write!(f, "malformed component state bytes: {e}"),
            Self::Custom(m) => write!(f, "component restore failed: {m}"),
        }
    }
}

impl std::error::Error for RestoreError {}

/// Monomorphized `T::serialize` captured at derive-codegen time.
pub type SerializeFn = fn(&dyn ErasedComponent) -> Vec<u8>;

/// Monomorphized `T::restore`; used by snapshot/resume.
pub type RestoreFn = fn(&[u8]) -> Result<Box<dyn ErasedComponent>, RestoreError>;

/// Errors surfaced by `ConstructFn`.
#[derive(Debug)]
pub struct ConstructError {
    /// `TYPE_NAME` of the concrete being constructed.
    pub type_name: &'static str,
    /// Downcast miss or `T::new` error stringified.
    pub detail: String,
}

impl std::fmt::Display for ConstructError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "construct {}: {}", self.type_name, self.detail)
    }
}

impl std::error::Error for ConstructError {}

/// Per-type constructor. Downcasts `&dyn Any` → `&Config` and calls
/// `T::new`. Install looks it up by `TYPE_NAME`.
pub type ConstructFn = fn(&dyn Any) -> Result<Box<dyn ErasedComponent>, ConstructError>;

