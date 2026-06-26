//! Shared codegen helpers used by every `#[derive(bb::<Role>)]`.
//! Emits the universal triple (`ConcreteComponent` + `AnyComponent`
//! + `inventory::submit!`) alongside the role-specific bridge.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

/// One parsed `#[depends(<role> = "<slot>", ...)]` entry. Strings
/// are the contract: `role` maps to `ComponentRole`; `slot` matches
/// the compiler's binding spec.
#[derive(Clone)]
pub(crate) struct ParsedDependency {
    /// PascalCase role string (`"Backend"`, `"Index"`, …).
    pub role: String,
    /// Author-chosen slot name.
    pub slot: String,
}

/// PascalCase → snake_case for inventory helper fn idents. Matches
/// the legacy `component!{}` helper naming to avoid collisions.
pub(crate) fn pascal_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i != 0 {
            out.push('_');
        }
        for lc in ch.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

/// `impl ConcreteComponent for #struct_ident` via bincode + Default.
/// Author supplies `Serialize + Deserialize + Default`; non-trivial
/// Config / new requires a hand-written impl instead of this derive.
pub(crate) fn emit_concrete_component_impl(
    struct_ident: &Ident,
    type_name: &str,
    deps: &[ParsedDependency],
) -> TokenStream {
    let dep_entries = emit_dependency_entries(deps);
    quote! {
        impl ::bytesandbrains::concrete::ConcreteComponent for #struct_ident {
            const TYPE_NAME: &'static str = #type_name;
            const PACKAGE: ::bytesandbrains::concrete::ComponentPackage =
                ::bytesandbrains::concrete::ComponentPackage::Application;
            const DEPENDENCIES: &'static [::bytesandbrains::component::DependencyDecl] =
                #dep_entries;

            type Config = ();
            type Error = ::std::convert::Infallible;

            fn new(_config: &Self::Config) -> ::std::result::Result<Self, Self::Error> {
                ::std::result::Result::Ok(<Self as ::std::default::Default>::default())
            }

            fn serialize(&self) -> ::std::vec::Vec<u8> {
                ::bytesandbrains::bincode::serialize(self).expect(concat!(
                    "ConcreteComponent::serialize on ", #type_name, " - serde infallible"
                ))
            }

            fn restore(
                bytes: &[u8],
            ) -> ::std::result::Result<Self, ::bytesandbrains::concrete::RestoreError> {
                ::bytesandbrains::bincode::deserialize(bytes)
                    .map_err(::bytesandbrains::concrete::RestoreError::Malformed)
            }
        }
    }
}

/// Slice literal for `ConcreteComponent::DEPENDENCIES`.
fn emit_dependency_entries(deps: &[ParsedDependency]) -> TokenStream {
    let items: Vec<TokenStream> = deps
        .iter()
        .map(|d| {
            let role = &d.role;
            let slot = &d.slot;
            quote! {
                ::bytesandbrains::component::DependencyDecl {
                    role: #role,
                    slot: #slot,
                }
            }
        })
        .collect();
    quote! { &[ #( #items ),* ] }
}

/// `impl AnyComponent for #struct_ident { … }`.
pub(crate) fn emit_any_component_impl(struct_ident: &Ident) -> TokenStream {
    quote! {
        impl ::bytesandbrains::component::AnyComponent for #struct_ident {
            fn as_any(&self) -> &dyn ::std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any { self }
        }
    }
}

/// `ConcreteComponentRegistration` inventory entry. Mirrors the
/// trait surface so `find_concrete_component(type_name)` recovers
/// the dep list without crossing trait boundaries.
pub(crate) fn emit_inventory_submit(struct_ident: &Ident, type_name: &str) -> TokenStream {
    let snake = pascal_to_snake(&struct_ident.to_string());
    let serialize_fn_ident = format_ident!("__bb_derive_inventory_serialize_{}", snake);
    let restore_fn_ident = format_ident!("__bb_derive_inventory_restore_{}", snake);
    let construct_fn_ident = format_ident!("__bb_derive_inventory_construct_{}", snake);
    quote! {
        #[doc(hidden)]
        fn #serialize_fn_ident(
            erased: &dyn ::bytesandbrains::component::ErasedComponent,
        ) -> ::std::vec::Vec<u8> {
            let any: &dyn ::std::any::Any = erased;
            let concrete: &#struct_ident = any
                .downcast_ref::<#struct_ident>()
                .expect("inventory downcast: TYPE_NAME-keyed registration");
            <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::serialize(concrete)
        }
        #[doc(hidden)]
        fn #restore_fn_ident(
            bytes: &[u8],
        ) -> ::std::result::Result<
            ::std::boxed::Box<dyn ::bytesandbrains::component::ErasedComponent>,
            ::bytesandbrains::concrete::RestoreError,
        > {
            <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::restore(bytes)
                .map(|v| ::std::boxed::Box::new(v)
                    as ::std::boxed::Box<dyn ::bytesandbrains::component::ErasedComponent>)
        }
        #[doc(hidden)]
        fn #construct_fn_ident(
            cfg: &dyn ::std::any::Any,
        ) -> ::std::result::Result<
            ::std::boxed::Box<dyn ::bytesandbrains::component::ErasedComponent>,
            ::bytesandbrains::concrete::ConstructError,
        > {
            type __Cfg = <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::Config;
            let typed = cfg.downcast_ref::<__Cfg>().ok_or_else(|| {
                ::bytesandbrains::concrete::ConstructError {
                    type_name: #type_name,
                    detail: ::std::format!(
                        "config type mismatch: expected `{}`, got `{:?}`",
                        ::std::any::type_name::<__Cfg>(),
                        cfg.type_id(),
                    ),
                }
            })?;
            <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::new(typed)
                .map(|v| ::std::boxed::Box::new(v)
                    as ::std::boxed::Box<dyn ::bytesandbrains::component::ErasedComponent>)
                .map_err(|e| ::bytesandbrains::concrete::ConstructError {
                    type_name: #type_name,
                    detail: ::std::format!("{e}"),
                })
        }
        ::bytesandbrains::inventory::submit! {
            ::bytesandbrains::registry::ConcreteComponentRegistration {
                type_name: #type_name,
                package: <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::PACKAGE,
                serialize_fn: #serialize_fn_ident,
                restore_fn: #restore_fn_ident,
                construct_fn: #construct_fn_ident,
                dependencies: <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::DEPENDENCIES,
            }
        }
    }
}

/// Compose the universal triple — every derive starts here. `deps`
/// threads to `ConcreteComponent::DEPENDENCIES` + the inventory.
pub(crate) fn emit_universal_triple(
    struct_ident: &Ident,
    type_name: &str,
    deps: &[ParsedDependency],
) -> TokenStream {
    let concrete = emit_concrete_component_impl(struct_ident, type_name, deps);
    let any = emit_any_component_impl(struct_ident);
    let inventory = emit_inventory_submit(struct_ident, type_name);
    quote! {
        #concrete
        #any
        #inventory
    }
}

/// Default `TYPE_NAME` is the bare struct ident; users wanting full
/// qualification override via `#[bb(type_name = "…")]`. We can't
/// reach `module_path!` at proc-macro time.
pub(crate) fn default_type_name(struct_ident: &Ident) -> String {
    struct_ident.to_string()
}
