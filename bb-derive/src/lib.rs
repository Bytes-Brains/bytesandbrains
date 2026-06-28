//! # bb-derive
//!
//! Proc-macro support for `bytesandbrains`. Per-role derives
//! (`Index`, `Aggregator`, …) bridge Contract impls to the engine's
//! `<Role>Runtime` traits and emit the universal triple
//! (`ConcreteComponent` + `AnyComponent` + `inventory::submit!`).
//! `register_op!{}` registers a custom op; `register_protocol!{}`
//! emits a full Protocol implementation in one block.

#![cfg_attr(not(test), warn(missing_docs))]

extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

mod codegen_shared;
mod parse;
mod roles;

// --- Concrete + 8 Role derives -------------------------------------

/// `#[derive(bb::Concrete)]` — emits the universal
/// `ConcreteComponent + AnyComponent + inventory::submit!` triple.
/// Pair with one or more `#[derive(bb::<Role>)]` derives for
/// role-trait bridges.
///
/// `#[depends(<role> = "<slot>", ...)]` entries (role is one of
/// `index | aggregator | model | codec | data_source |
/// peer_selector | backend | protocol`) surface as
/// `ConcreteComponent::DEPENDENCIES` for compile-time slot
/// verification. Multiple attributes stack.
#[proc_macro_derive(Concrete, attributes(depends))]
#[allow(non_snake_case)]
pub fn Concrete(input: TokenStream) -> TokenStream {
    let derive_input = parse_macro_input!(input as DeriveInput);
    roles::emit_concrete_derive(&derive_input).into()
}

macro_rules! role_derive {
    ($pm_name:ident, $emit_fn:ident, $doc:expr) => {
        #[doc = $doc]
        #[proc_macro_derive($pm_name)]
        #[allow(non_snake_case)]
        pub fn $pm_name(input: TokenStream) -> TokenStream {
            let DeriveInput { ident, .. } = parse_macro_input!(input as DeriveInput);
            roles::$emit_fn(&ident).into()
        }
    };
}

role_derive!(
    Index,
    emit_index_derive,
    "Derive `bb::Index` bridge - emits `ConcreteComponent`, `AnyComponent`, `IndexRuntime`, and `inventory::submit!`."
);
role_derive!(
    Aggregator,
    emit_aggregator_derive,
    "Derive `bb::Aggregator` bridge."
);
role_derive!(Model, emit_model_derive, "Derive `bb::Model` bridge.");
role_derive!(
    Codec,
    emit_codec_derive,
    "Derive `bb::Codec` bridge - emits ConcreteComponent + CodecRuntime."
);
role_derive!(
    DataSource,
    emit_data_source_derive,
    "Derive `bb::DataSource` bridge."
);
role_derive!(
    PeerSelector,
    emit_peer_selector_derive,
    "Derive `bb::PeerSelector` bridge."
);
role_derive!(
    Backend,
    emit_backend_derive,
    "Derive `bb::Backend` bridge - emits the ai.onnx v1 51-op opset declaration; `dispatch_atomic` forwards into `Backend::execute`."
);

// --- register_op! --------------------------------------------------

/// Register a custom op with the global inventory.
///
/// ```ignore
/// bb::register_op! {
///     domain: "myapp.ops",
///     op_type: "Foo",
///     invoke: invoke_foo,
/// }
/// ```
#[proc_macro]
pub fn register_op(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as parse::RegisterOpInput);
    let domain = &parsed.domain;
    let op_type = &parsed.op_type;
    let invoke = &parsed.invoke;
    let out = quote! {
        ::bytesandbrains::inventory::submit! {
            ::bytesandbrains::registry::OpRegistration {
                domain: #domain,
                op_type: #op_type,
                invoke: #invoke,
                kind: ::bytesandbrains::registry::RegistrationKind::Custom,
            }
        }
    };
    out.into()
}

// --- register_protocol! --------------------------------------------

/// Declarative macro emitting a complete `ProtocolRuntime` impl
/// from a struct + opset + ops block.
///
/// ```ignore
/// bb::register_protocol! {
///     struct Kademlia { routing_table: Vec<u64>, k: usize }
///     domain: "bb-kademlia.kademlia.atomic"
///     version: 1
///     ops {
///         FindNode,
///         Ping,
///     }
/// }
/// ```
#[proc_macro]
pub fn register_protocol(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as parse::RegisterProtocolInput);
    parse::emit_register_protocol(&parsed).into()
}
