//! Protocol implementations - `<Role>Runtime` bridges for concrete
//! coordination protocols. Each lives in a sub-directory and
//! self-registers via `inventory::submit!` per the standard
//! component-authoring contract.



pub mod global_registry;
pub use global_registry::{
    GlobalRegistryClient, GlobalRegistryServer, GlobalRegistryServerConfig, Handshake,
    GLOBAL_REGISTRY_CLIENT_CREF, GLOBAL_REGISTRY_DOMAIN, GLOBAL_REGISTRY_SERVER_CREF,
};

pub mod constant_view;
pub use constant_view::{ConstantView, ConstantViewError};
