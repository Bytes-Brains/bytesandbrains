//! `ConcreteComponent` polymorphism contract + `ComponentHandle`
//! fn-pointer-capture wrapper. See `docs/AUTHORING_COMPONENTS.md`
//! §4 + §9.

use std::any::Any;

use bb_ir::component::ErasedComponent;

// Foundation plumbing lives in `bb_ir::component` to keep `bb-dsl`
// and `bb-runtime` cycle-free; re-exported here for ergonomics.
pub use bb_ir::component::{
    ComponentPackage, ConstructError, ConstructFn, DependencyDecl, RestoreError, RestoreFn,
    SerializeFn,
};

/// Polymorphism contract. Implementing this trait IS the
/// registration mechanism — no global registry, no macro
/// required. Serialized state must be self-contained so `restore`
/// reconstructs without the original `Config`.
pub trait ConcreteComponent: ErasedComponent + Sized {
    /// Stable identifier. Convention: `<crate>::<TypeName>`.
    const TYPE_NAME: &'static str;

    /// Origin tag; defaults to `Application`.
    const PACKAGE: ComponentPackage = ComponentPackage::Application;

    /// Sibling components this depends on. Populated by the
    /// `bb::Concrete` derive from `#[bb::depends(...)]`.
    const DEPENDENCIES: &'static [DependencyDecl] = &[];

    /// Per-deployment config. Use `()` for stateless concretes.
    type Config: Any + 'static;

    /// Error from [`Self::new`]; use `Infallible` if construction
    /// can't fail.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Construct from `&Self::Config`. Install calls this once per
    /// slot.
    fn new(config: &Self::Config) -> Result<Self, Self::Error>;

    /// Serialize state to bytes, including config-derived fields.
    fn serialize(&self) -> Vec<u8>;

    /// Reconstruct from `serialize` output.
    fn restore(bytes: &[u8]) -> Result<Self, RestoreError>;
}

/// Owned wrapper that travels through the ModelProto → Node
/// pipeline. Carries captured fn pointers + state bytes.
pub struct ComponentHandle {
    /// Stable type identifier.
    pub type_name: &'static str,
    /// Origin tag.
    pub package: ComponentPackage,
    /// Per-Node instance disambiguator assigned by install.
    pub instance_id: u32,
    /// Monomorphized `T::serialize` via downcast.
    pub serialize_fn: SerializeFn,
    /// Monomorphized `T::restore`.
    pub restore_fn: RestoreFn,
    /// Serialized state captured at build or snapshot time.
    pub state_bytes: Vec<u8>,
}

impl std::fmt::Debug for ComponentHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentHandle")
            .field("type_name", &self.type_name)
            .field("package", &self.package)
            .field("instance_id", &self.instance_id)
            .field("state_bytes_len", &self.state_bytes.len())
            .finish_non_exhaustive()
    }
}

impl ComponentHandle {
    /// Materialize a fresh instance via `restore_fn(state_bytes)`.
    pub fn materialize(&self) -> Result<Box<dyn ErasedComponent>, RestoreError> {
        (self.restore_fn)(&self.state_bytes)
    }

    /// Capture state from a live `&dyn ErasedComponent`.
    pub fn capture_state(&self, instance: &dyn ErasedComponent) -> Vec<u8> {
        (self.serialize_fn)(instance)
    }

    /// Build from a live `&T: ConcreteComponent`. Captures
    /// monomorphized `serialize`/`restore` + freezes current bytes.
    pub fn from_concrete<T: ConcreteComponent>(instance: &T, instance_id: u32) -> Self {
        Self {
            type_name: T::TYPE_NAME,
            package: T::PACKAGE,
            instance_id,
            serialize_fn: |erased: &dyn ErasedComponent| -> Vec<u8> {
                let any: &dyn Any = erased;
                let concrete: &T = any
                    .downcast_ref::<T>()
                    .expect("ComponentHandle::serialize_fn called with mismatched type");
                concrete.serialize()
            },
            restore_fn: |bytes: &[u8]| -> Result<Box<dyn ErasedComponent>, RestoreError> {
                T::restore(bytes).map(|v| Box::new(v) as Box<dyn ErasedComponent>)
            },
            state_bytes: instance.serialize(),
        }
    }
}
