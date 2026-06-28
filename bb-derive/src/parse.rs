//! `syn::Parse` impls for `register_op!{}` + `register_protocol!{}`
//! and [`parse_depends_attrs`] for the `#[depends(...)]` attribute
//! on `#[derive(bb::Concrete)]` structs.

use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::{braced, Attribute, Ident, ItemStruct, LitInt, LitStr, Path, Token};

use crate::codegen_shared::ParsedDependency;

/// snake_case role identifier → PascalCase `ComponentRole` string
/// stored in `DependencyDecl.role`.
fn role_ident_to_canonical(role: &Ident) -> std::result::Result<&'static str, syn::Error> {
    Ok(match role.to_string().as_str() {
        "index" => "Index",
        "aggregator" => "Aggregator",
        "model" => "Model",
        "codec" => "Codec",
        "data_source" => "DataSource",
        "peer_selector" => "PeerSelector",
        "backend" => "Backend",
        "protocol" => "Protocol",
        other => {
            return Err(syn::Error::new(
                role.span(),
                format!(
                    "unknown role `{other}` in `#[bb::depends(...)]` - \
                     expected one of: index, aggregator, model, codec, \
                     data_source, peer_selector, backend, protocol",
                ),
            ));
        }
    })
}

/// Parse every `#[depends(role = "slot", ...)]` attribute on a
/// struct. Multiple attributes stack; non-matching paths are skipped.
pub(crate) fn parse_depends_attrs(
    attrs: &[Attribute],
) -> std::result::Result<Vec<ParsedDependency>, syn::Error> {
    let mut deps = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("depends") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            let role_ident = meta
                .path
                .get_ident()
                .ok_or_else(|| meta.error("expected `<role> = \"slot\"`"))?
                .clone();
            let canonical = role_ident_to_canonical(&role_ident)?;
            let slot_lit: LitStr = meta.value()?.parse()?;
            deps.push(ParsedDependency {
                role: canonical.to_string(),
                slot: slot_lit.value(),
            });
            Ok(())
        })?;
    }
    Ok(deps)
}

// --- register_op! grammar ------------------------------------------

mod kw_register_op {
    syn::custom_keyword!(domain);
    syn::custom_keyword!(op_type);
    syn::custom_keyword!(invoke);
    syn::custom_keyword!(version);
    syn::custom_keyword!(ops);
}

/// Parsed form of `bb::register_op!{ domain: ..., op_type: ..., invoke: ... }`.
pub struct RegisterOpInput {
    pub domain: LitStr,
    pub op_type: LitStr,
    pub invoke: Path,
}

impl Parse for RegisterOpInput {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<kw_register_op::domain>()?;
        input.parse::<Token![:]>()?;
        let domain: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;

        input.parse::<kw_register_op::op_type>()?;
        input.parse::<Token![:]>()?;
        let op_type: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;

        input.parse::<kw_register_op::invoke>()?;
        input.parse::<Token![:]>()?;
        let invoke: Path = input.parse()?;
        let _ = input.parse::<Token![,]>();

        Ok(RegisterOpInput {
            domain,
            op_type,
            invoke,
        })
    }
}

// --- register_protocol! grammar ------------------------------------

pub struct RegisterProtocolInput {
    pub item_struct: ItemStruct,
    pub domain: LitStr,
    pub version: LitInt,
    pub ops: Vec<Ident>,
}

impl Parse for RegisterProtocolInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let item_struct: ItemStruct = input.parse()?;

        input.parse::<kw_register_op::domain>()?;
        input.parse::<Token![:]>()?;
        let domain: LitStr = input.parse()?;

        input.parse::<kw_register_op::version>()?;
        input.parse::<Token![:]>()?;
        let version: LitInt = input.parse()?;

        input.parse::<kw_register_op::ops>()?;
        let ops_brace;
        braced!(ops_brace in input);
        let ops_punct: syn::punctuated::Punctuated<Ident, Token![,]> =
            ops_brace.parse_terminated(Ident::parse, Token![,])?;
        let ops: Vec<Ident> = ops_punct.into_iter().collect();

        Ok(RegisterProtocolInput {
            item_struct,
            domain,
            version,
            ops,
        })
    }
}

pub fn emit_register_protocol(input: &RegisterProtocolInput) -> proc_macro2::TokenStream {
    let item_struct = &input.item_struct;
    let struct_ident = &item_struct.ident;
    let type_name = struct_ident.to_string();
    let domain_lit = &input.domain;
    let version_lit = &input.version;

    let op_decls: Vec<proc_macro2::TokenStream> = input
        .ops
        .iter()
        .map(|op| {
            let name_lit = op.to_string();
            quote! {
                ::bytesandbrains::atomic::AtomicOpDecl {
                    name: #name_lit,
                    inputs: &[],
                    outputs: &[],
                    kind: ::bytesandbrains::atomic::AtomicOpKind::Immediate,
            type_relations: &[],
                }
            }
        })
        .collect();

    let op_arms: Vec<proc_macro2::TokenStream> = input
        .ops
        .iter()
        .map(|op| {
            let name_lit = op.to_string();
            quote! {
                #name_lit => ::std::result::Result::Ok(
                    ::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()),
                ),
            }
        })
        .collect();

    let universal = crate::codegen_shared::emit_universal_triple(struct_ident, &type_name, &[]);

    quote! {
        #item_struct
        #universal

        impl ::bytesandbrains::roles::ProtocolRuntime for #struct_ident {
            type Error = ::bytesandbrains::bus::OpError;

            fn atomic_opset(&self) -> ::bytesandbrains::atomic::AtomicOpsetDecl {
                static OPS: &[::bytesandbrains::atomic::AtomicOpDecl] = &[
                    #( #op_decls ),*
                ];
                ::bytesandbrains::atomic::AtomicOpsetDecl {
                    domain: #domain_lit,
                    version: #version_lit,
                    ops: OPS,
                }
            }

            fn dispatch_atomic(
                &mut self,
                op_type: &str,
                _inputs: &[(&str, &dyn ::bytesandbrains::slot_value::SlotValue)],
                _ctx: &mut ::bytesandbrains::runtime::RuntimeResourceRef<'_>,
            ) -> ::std::result::Result<
                ::bytesandbrains::atomic::DispatchResult,
                Self::Error,
            > {
                match op_type {
                    #( #op_arms )*
                    other => ::std::result::Result::Err(
                        ::bytesandbrains::bus::OpError {
                            detail: format!("register_protocol!: unknown op `{}`", other), ..Default::default()
                        },
                    ),
                }
            }
        }
    }
}
