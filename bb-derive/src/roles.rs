//! Per-role derive codegen.
//!
//! Each role derive emits the universal triple (`ConcreteComponent` +
//! `AnyComponent` + `inventory::submit!`) plus a `<Role>Runtime` impl
//! whose opset names one `AtomicOpDecl` per Contract method (PascalCase).
//! See `docs/CONTRACT_DISPATCH.md` for the Contract ↔ Runtime bridge design.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident};

use crate::codegen_shared::{default_type_name, emit_universal_triple};

/// `#[derive(bb::Concrete)]` codegen entry — emits the universal
/// triple for `struct_ident`, parses `#[depends(...)]` entries into
/// `ConcreteComponent::DEPENDENCIES`, and threads through the
/// Bootstrap bridge (default `impl Bootstrap` unless
/// `#[bootstrap_override]` is present + the
/// `BootstrapDispatcherRegistration` inventory entry the install
/// path uses to wire the engine's Component bootstrap dispatcher).
pub(crate) fn emit_concrete_derive(input: &DeriveInput) -> TokenStream {
    let struct_ident = &input.ident;
    let type_name = default_type_name(struct_ident);
    let deps = match crate::parse::parse_depends_attrs(&input.attrs) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error(),
    };
    let universal = emit_universal_triple(struct_ident, &type_name, &deps);
    let bootstrap_override = crate::parse::has_bootstrap_override(&input.attrs);
    let bootstrap_bridge = emit_bootstrap_bridge(struct_ident, &type_name, bootstrap_override);
    quote! {
        #universal
        #bootstrap_bridge
    }
}

/// Default `Bootstrap` impl + inventory registration emitted by
/// `#[derive(bb::Concrete)]`. The impl uses the trait's no-op default
/// so a Concrete with no manual override participates in the
/// Component bootstrap dispatch path without boilerplate. The
/// inventory entry captures `T` so `install()` can call
/// `engine.register_bootstrap_dispatcher::<T>()` for every registered
/// concrete. `#[bootstrap_override]` on the struct suppresses the
/// impl emission so the author can hand-write
/// `impl Bootstrap for X { ... }` without colliding with the derive.
fn emit_bootstrap_bridge(
    struct_ident: &syn::Ident,
    type_name: &str,
    bootstrap_override: bool,
) -> TokenStream {
    let default_impl = if bootstrap_override {
        // User supplies their own `impl Bootstrap for #struct_ident`;
        // the dispatcher routes through it via the same TypeId lookup.
        TokenStream::new()
    } else {
        quote! {
            impl ::bytesandbrains::contracts::bootstrap::Bootstrap for #struct_ident {
                type Error = ::std::convert::Infallible;
            }
        }
    };
    quote! {
        #default_impl

        // Bootstrap dispatcher registration. The install path walks
        // `inventory::iter::<BootstrapDispatcherRegistration>` and
        // calls `register_fn(engine)` for each registered concrete,
        // wiring `engine.register_bootstrap_dispatcher::<#struct_ident>()`
        // before the first Component bootstrap fires.
        ::bytesandbrains::inventory::submit! {
            ::bytesandbrains::registry::BootstrapDispatcherRegistration {
                type_name: #type_name,
                register_fn: |engine: &mut ::bytesandbrains::engine::Engine| {
                    engine.register_bootstrap_dispatcher::<#struct_ident>();
                },
            }
        }
    }
}

#[derive(Clone, Copy)]
struct OpSpec {
    /// PascalCase op name (the IR-level identifier).
    name: &'static str,
    /// Path (as a token-stream literal) to a
    /// `&'static [TypeRelation]` slice. Defaults to
    /// `::bytesandbrains::types::NO_RELATIONS` via `OpSpec::leaf`.
    type_relations_path: &'static str,
}

impl OpSpec {
    /// No declared type relations; solver leaves values at seeded
    /// bounds. Use for heterogeneous or attribute-driven I/O.
    const fn leaf(name: &'static str) -> Self {
        Self {
            name,
            type_relations_path: "::bytesandbrains::types::NO_RELATIONS",
        }
    }

    /// Custom `&'static [TypeRelation]` path consumed verbatim by the
    /// per-role derive's `AtomicOpDecl` codegen.
    const fn with_relations(name: &'static str, type_relations_path: &'static str) -> Self {
        Self {
            name,
            type_relations_path,
        }
    }
}

/// How the role's atomic-op dispatch surface is shaped. The typed
/// discriminator lets `emit_role_derive` branch on dispatch shape
/// instead of matching on the runtime trait name.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoleDispatchKind {
    /// One `dispatch_atomic` arm per Contract method (Index,
    /// Aggregator, Model, Codec, DataSource, PeerSelector).
    MethodStyle,
    /// Single `Backend::execute(graph, inputs, ctx)` over the
    /// collapsed `BackendSubgraph` carrier.
    ExecuteOverGraph,
}

struct RoleSpec {
    /// Engine-side runtime trait under `bytesandbrains::roles::`.
    runtime: &'static str,
    /// Role domain, e.g. `"ai.bytesandbrains.role.index"`.
    role_domain: &'static str,
    /// Atomic ops the derive declares for this role.
    ops: &'static [OpSpec],
    dispatch_kind: RoleDispatchKind,
    /// Storage-bound associated types on the Contract trait. `None`
    /// when the role has none (e.g. PeerSelector).
    storage_ports: Option<&'static [StoragePort]>,
}

/// One Storage-bound associated type on a role's Contract trait.
struct StoragePort {
    /// The port label stored in `StorageTypeEntry.port`.
    port: &'static str,
    /// Rust path to the Contract trait (e.g.
    /// `"::bytesandbrains::contracts::Index"`). Used to build the
    /// `<T as Contract>::AssocType` path in the emitted fn.
    contract_path: &'static str,
    /// The associated type name on the Contract (e.g. `"Vector"`).
    assoc_type: &'static str,
}

const INDEX: RoleSpec = RoleSpec {
    runtime: "IndexRuntime",
    role_domain: "ai.bytesandbrains.role.index",
    // No shared element-type bound at the role surface — every op
    // mixes tensor + ID types. `Train` is opt-in at the Contract.
    ops: &[
        OpSpec::leaf("Add"),
        OpSpec::leaf("Search"),
        OpSpec::leaf("Remove"),
        OpSpec::leaf("Train"),
    ],
    dispatch_kind: RoleDispatchKind::MethodStyle,
    storage_ports: Some(&[StoragePort {
        port: "vector",
        contract_path: "::bytesandbrains::contracts::Index",
        assoc_type: "Vector",
    }]),
};
const AGGREGATOR: RoleSpec = RoleSpec {
    runtime: "AggregatorRuntime",
    role_domain: "ai.bytesandbrains.role.aggregator",
    // No shared element-type bound; metadata + reduction are impl-defined.
    ops: &[OpSpec::leaf("Contribute"), OpSpec::leaf("Aggregate")],
    dispatch_kind: RoleDispatchKind::MethodStyle,
    storage_ports: Some(&[StoragePort {
        port: "element",
        contract_path: "::bytesandbrains::contracts::Aggregator",
        assoc_type: "Element",
    }]),
};
const MODEL: RoleSpec = RoleSpec {
    runtime: "ModelRuntime",
    role_domain: "ai.bytesandbrains.role.model",
    ops: &[
        // Forward propagates the input's element type to the output so
        // downstream wire.Send narrows correctly under typed seeding.
        OpSpec::with_relations("Forward", "::bytesandbrains::types::UNARY_SAME_ELEMENT"),
        OpSpec::leaf("Backward"),
        OpSpec::leaf("ComputeLoss"),
        OpSpec::leaf("ApplyDelta"),
        OpSpec::leaf("LoadParameters"),
        OpSpec::leaf("Params"),
    ],
    dispatch_kind: RoleDispatchKind::MethodStyle,
    storage_ports: Some(&[StoragePort {
        port: "tensor",
        contract_path: "::bytesandbrains::contracts::Model",
        assoc_type: "Tensor",
    }]),
};
const CODEC: RoleSpec = RoleSpec {
    runtime: "CodecRuntime",
    role_domain: "ai.bytesandbrains.role.codec",
    // In/Out may differ in element type (f32 ↔ u8, …); no shared bound.
    ops: &[
        OpSpec::leaf("Train"),
        OpSpec::leaf("Encode"),
        OpSpec::leaf("Decode"),
    ],
    dispatch_kind: RoleDispatchKind::MethodStyle,
    // Codec has TWO Storage-bound ports.
    storage_ports: Some(&[
        StoragePort {
            port: "in",
            contract_path: "::bytesandbrains::contracts::Codec",
            assoc_type: "In",
        },
        StoragePort {
            port: "out",
            contract_path: "::bytesandbrains::contracts::Codec",
            assoc_type: "Out",
        },
    ]),
};
const DATA_SOURCE: RoleSpec = RoleSpec {
    runtime: "DataSourceRuntime",
    role_domain: "ai.bytesandbrains.role.data_source",
    // Dataset-specific element types; no role-surface constraint.
    ops: &[
        OpSpec::leaf("NextBatch"),
        OpSpec::leaf("Reset"),
        OpSpec::leaf("OnDataLoaded"),
    ],
    dispatch_kind: RoleDispatchKind::MethodStyle,
    storage_ports: Some(&[StoragePort {
        port: "sample",
        contract_path: "::bytesandbrains::contracts::DataSource",
        assoc_type: "Sample",
    }]),
};
const PEER_SELECTOR: RoleSpec = RoleSpec {
    runtime: "PeerSelectorRuntime",
    role_domain: "ai.bytesandbrains.role.peer_selector",
    ops: &[OpSpec::leaf("Sample"), OpSpec::leaf("CurrentView")],
    dispatch_kind: RoleDispatchKind::MethodStyle,
    storage_ports: None,
};

/// 30 primitive tensor ops every backend must implement. Mirrors
/// `bb_ir::tensor_primitives::TENSOR_PRIMITIVES_OPS`. ML formulas
/// (Relu, Conv, Gemm, …) live in `extension_opsets()`, not here.
const BACKEND_ONNX_OPS: &[OpSpec] = &[
    // Arithmetic (6).
    OpSpec::with_relations("Add", "::bytesandbrains::types::BROADCAST_BINARY"),
    OpSpec::with_relations("Sub", "::bytesandbrains::types::BROADCAST_BINARY"),
    OpSpec::with_relations("Mul", "::bytesandbrains::types::BROADCAST_BINARY"),
    OpSpec::with_relations("Div", "::bytesandbrains::types::BROADCAST_BINARY"),
    OpSpec::with_relations("Neg", "::bytesandbrains::types::ELEMENTWISE"),
    OpSpec::with_relations("Abs", "::bytesandbrains::types::ELEMENTWISE"),
    // Math (4).
    OpSpec::with_relations("Sqrt", "::bytesandbrains::types::ELEMENTWISE"),
    OpSpec::with_relations("Pow", "::bytesandbrains::types::BROADCAST_BINARY"),
    OpSpec::with_relations("Exp", "::bytesandbrains::types::ELEMENTWISE"),
    OpSpec::with_relations("Log", "::bytesandbrains::types::ELEMENTWISE"),
    // Linear algebra (1).
    OpSpec::with_relations("MatMul", "::bytesandbrains::types::MATMUL_BINARY"),
    // Reductions (4).
    OpSpec::with_relations("ReduceSum", "::bytesandbrains::types::REDUCE_AXIS"),
    OpSpec::with_relations("ReduceMean", "::bytesandbrains::types::REDUCE_AXIS"),
    OpSpec::with_relations("ReduceMax", "::bytesandbrains::types::REDUCE_AXIS"),
    OpSpec::with_relations("ReduceMin", "::bytesandbrains::types::REDUCE_AXIS"),
    // Shape (9).
    OpSpec::with_relations("Reshape", "::bytesandbrains::types::UNARY_SAME_ELEMENT"),
    OpSpec::with_relations("Transpose", "::bytesandbrains::types::UNARY_SAME_ELEMENT"),
    OpSpec::leaf("Concat"),
    OpSpec::with_relations("Slice", "::bytesandbrains::types::UNARY_SAME_ELEMENT"),
    OpSpec::leaf("Split"),
    OpSpec::with_relations("Squeeze", "::bytesandbrains::types::UNARY_SAME_ELEMENT"),
    OpSpec::with_relations("Unsqueeze", "::bytesandbrains::types::UNARY_SAME_ELEMENT"),
    OpSpec::with_relations("Identity", "::bytesandbrains::types::ELEMENTWISE"),
    OpSpec::leaf("Cast"),
    // Comparison (3) — output bool; leaf until a bool-tensor lattice lands.
    OpSpec::leaf("Equal"),
    OpSpec::leaf("Greater"),
    OpSpec::leaf("Less"),
    // Conditional (1).
    OpSpec::leaf("Where"),
    // Creation (1).
    OpSpec::leaf("Constant"),
    // Indexing (1).
    OpSpec::leaf("Gather"),
];
const BACKEND: RoleSpec = RoleSpec {
    runtime: "BackendRuntime",
    role_domain: "ai.onnx",
    ops: BACKEND_ONNX_OPS,
    dispatch_kind: RoleDispatchKind::ExecuteOverGraph,
    storage_ports: Some(&[StoragePort {
        port: "tensor",
        contract_path: "::bytesandbrains::contracts::Backend",
        assoc_type: "Tensor",
    }]),
};

/// `RoleSpec.runtime` → `ComponentRole` enum variant identifier.
fn role_enum_variant(runtime: &str) -> &'static str {
    match runtime {
        "IndexRuntime" => "Index",
        "AggregatorRuntime" => "Aggregator",
        "ModelRuntime" => "Model",
        "CodecRuntime" => "Codec",
        "DataSourceRuntime" => "DataSource",
        "PeerSelectorRuntime" => "PeerSelector",
        "BackendRuntime" => "Backend",
        "ProtocolRuntime" => "Protocol",
        _ => "Protocol",
    }
}

/// Generate the per-role derive expansion for a struct.
fn emit_role_derive(struct_ident: &Ident, role: &RoleSpec) -> TokenStream {
    let domain_lit = role.role_domain;

    // Shared `&'static [TypeRelation]` slices keep the solver's
    // constraint nodes pointer-equal across role + backend decls.
    let op_decls: Vec<TokenStream> = role
        .ops
        .iter()
        .map(|op| {
            let name_lit = op.name;
            let relations_path: syn::Path =
                syn::parse_str(op.type_relations_path).expect("valid type_relations path");
            quote! {
                ::bytesandbrains::atomic::AtomicOpDecl {
                    name: #name_lit,
                    inputs: &[],
                    outputs: &[],
                    kind: ::bytesandbrains::atomic::AtomicOpKind::Immediate,
                    type_relations: #relations_path,
                }
            }
        })
        .collect();

    let runtime_ident: syn::Ident = syn::Ident::new(role.runtime, proc_macro2::Span::call_site());

    // Backend extension_opsets defaults to empty. materialize_from_wire
    // bridges Contract → BackendTensorCarrier; engine stamps the
    // accounting fields after the bridge returns.
    let role_method_stubs: TokenStream = match role.dispatch_kind {
        RoleDispatchKind::ExecuteOverGraph => match role.runtime {
            "BackendRuntime" => quote! {
                fn extension_opsets(&self) -> ::std::vec::Vec<::bytesandbrains::atomic::AtomicOpsetDecl> {
                    ::std::vec::Vec::new()
                }

                fn materialize_from_wire(
                    &self,
                    type_hash: u64,
                    bytes: ::std::vec::Vec<u8>,
                ) -> ::std::result::Result<
                    ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>,
                    ::bytesandbrains::slot_value::BackendMaterializeError,
                > {
                    let tensor = <Self as ::bytesandbrains::contracts::Backend>::materialize_from_wire(
                        self, type_hash, bytes,
                    )
                    .map_err(|e| ::bytesandbrains::slot_value::BackendMaterializeError {
                        summary: ::std::format!("{e}"),
                    })?;
                    let carrier = ::bytesandbrains::slot_value::BackendTensorCarrier::from_typed::<
                        <Self as ::bytesandbrains::contracts::Backend>::Tensor,
                    >(
                        tensor,
                        type_hash,
                        // Engine stamps charged_bytes + backend_ref
                        // immediately after this call returns.
                        0,
                        ::bytesandbrains::ids::ComponentRef::from(0u32),
                    );
                    ::std::result::Result::Ok(::std::boxed::Box::new(carrier))
                }
            },
            _ => quote! {},
        },
        RoleDispatchKind::MethodStyle => quote! {},
    };

    // dispatch_atomic forwards each op to its Contract method;
    // unknown op_types fall through to the catch-all OpError.
    let unknown_msg = format!(
        "{}::dispatch_atomic: unknown op_type for {{}} dispatching on `{{}}`",
        role.runtime
    );
    let known_arms: Vec<TokenStream> = match role.dispatch_kind {
        RoleDispatchKind::MethodStyle => match role.runtime {
            "IndexRuntime" => emit_index_arms(),
            "AggregatorRuntime" => emit_aggregator_arms(),
            "ModelRuntime" => emit_model_arms(),
            "DataSourceRuntime" => emit_data_source_arms(),
            "CodecRuntime" => emit_codec_arms(),
            "PeerSelectorRuntime" => emit_peer_selector_arms(),
            _ => Vec::new(),
        },
        RoleDispatchKind::ExecuteOverGraph => {
            let mut arms = emit_backend_op_arms(role.ops);
            arms.push(emit_backend_subgraph_arm());
            arms
        }
    };

    let role_variant_ident: syn::Ident = syn::Ident::new(
        role_enum_variant(role.runtime),
        proc_macro2::Span::call_site(),
    );
    let type_name_lit = struct_ident.to_string();

    // Per-role dispatcher-registration call.
    let register_dispatcher_call = match role.runtime {
        "IndexRuntime" => quote! { engine.register_index_dispatcher::<#struct_ident>(); },
        "AggregatorRuntime" => quote! { engine.register_aggregator_dispatcher::<#struct_ident>(); },
        "ModelRuntime" => quote! { engine.register_model_dispatcher::<#struct_ident>(); },
        "CodecRuntime" => quote! { engine.register_codec_dispatcher::<#struct_ident>(); },
        "DataSourceRuntime" => {
            quote! { engine.register_data_source_dispatcher::<#struct_ident>(); }
        }
        "PeerSelectorRuntime" => {
            quote! { engine.register_peer_selector_dispatcher::<#struct_ident>(); }
        }
        "BackendRuntime" => {
            quote! { engine.register_backend_dispatcher::<#struct_ident>(); }
        }
        "ProtocolRuntime" => quote! { engine.register_protocol_dispatcher::<#struct_ident>(); },
        _ => quote! { /* unknown role - dispatcher registration noop */ },
    };

    // Per Storage-bound associated type: doc-hidden fn capturing the
    // monomorphized `Storage::TYPE` + one `inventory::submit!` entry.
    let storage_type_submissions: Vec<TokenStream> = match role.storage_ports {
        None => Vec::new(),
        Some(ports) => ports
            .iter()
            .map(|sp| {
                let port_lit = sp.port;
                let contract_path: syn::Path =
                    syn::parse_str(sp.contract_path).expect("valid contract path");
                let assoc_ident: syn::Ident =
                    syn::Ident::new(sp.assoc_type, proc_macro2::Span::call_site());
                let snake = crate::codegen_shared::pascal_to_snake(&struct_ident.to_string());
                let fn_ident = proc_macro2::Ident::new(
                    &format!(
                        "__bb_storage_type_{}_for_{}",
                        sp.port.replace('-', "_"),
                        snake,
                    ),
                    proc_macro2::Span::call_site(),
                );
                let runtime_lit = role.runtime;
                quote! {
                    #[doc(hidden)]
                    #[allow(non_snake_case)]
                    fn #fn_ident() -> &'static ::bytesandbrains::types::TypeNode {
                        <<#struct_ident as #contract_path>::#assoc_ident
                            as ::bytesandbrains::types::Storage>::TYPE
                    }
                    ::bytesandbrains::inventory::submit! {
                        ::bytesandbrains::registry::StorageTypeEntry {
                            // TYPE_NAME constant matches manually overridden
                            // values (e.g. CpuBackend's custom path).
                            concrete_type_name: <#struct_ident as ::bytesandbrains::concrete::ConcreteComponent>::TYPE_NAME,
                            role_runtime:       #runtime_lit,
                            port:               #port_lit,
                            type_node_fn:       #fn_ident,
                        }
                    }
                }
            })
            .collect(),
    };

    quote! {
        impl ::bytesandbrains::roles::#runtime_ident for #struct_ident {
            type Error = ::bytesandbrains::bus::OpError;

            #role_method_stubs

            fn atomic_opset(&self) -> ::bytesandbrains::atomic::AtomicOpsetDecl {
                static OPS: &[::bytesandbrains::atomic::AtomicOpDecl] = &[
                    #( #op_decls ),*
                ];
                ::bytesandbrains::atomic::AtomicOpsetDecl {
                    domain: #domain_lit,
                    version: 1,
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
                    #( #known_arms )*
                    other => ::std::result::Result::Err(
                        ::bytesandbrains::bus::OpError {
                            detail: format!(#unknown_msg, stringify!(#struct_ident), other), ..Default::default()
                        },
                    ),
                }
            }
        }

        // Roles this struct implements; queried by Node::ensure_ready
        // without scanning trait impls.
        ::bytesandbrains::inventory::submit! {
            ::bytesandbrains::registry::ComponentRoleBinding {
                type_name: #type_name_lit,
                role: ::bytesandbrains::registry::ComponentRole::#role_variant_ident,
            }
        }

        // Dispatcher-registration fn captures T so install() never
        // needs the typed &T itself.
        ::bytesandbrains::inventory::submit! {
            ::bytesandbrains::registry::DispatcherRegistration {
                type_name: #type_name_lit,
                role: ::bytesandbrains::registry::ComponentRole::#role_variant_ident,
                register_fn: |engine: &mut ::bytesandbrains::engine::Engine| {
                    #register_dispatcher_call
                },
            }
        }

        // One StorageTypeEntry per Storage-bound Contract assoc type.
        #( #storage_type_submissions )*
    }
}

// --- Per-role public entry points ----------------------------------

pub(crate) fn emit_index_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &INDEX)
}
pub(crate) fn emit_aggregator_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &AGGREGATOR)
}
pub(crate) fn emit_model_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &MODEL)
}
pub(crate) fn emit_codec_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &CODEC)
}
pub(crate) fn emit_data_source_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &DATA_SOURCE)
}
pub(crate) fn emit_peer_selector_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &PEER_SELECTOR)
}
pub(crate) fn emit_backend_derive(struct_ident: &Ident) -> TokenStream {
    emit_role_derive(struct_ident, &BACKEND)
}

// ─── Per-role `dispatch_atomic` arms ───────────────────────────────
//
// Per-op arm tokens for `dispatch_atomic`. Inputs are positional;
// the Contract method is called with a typed `CompletionHandle`;
// sync `Ok` → `Immediate`, async → `Async(cmd_id)`, errors →
// `OpError` via `format!("{e}")`.

fn emit_index_arms() -> Vec<TokenStream> {
    vec![
        quote! {
            "Add" => {
                // Downcast assumes Self::Vector derefs to &[f32]; for
                // other tensor shapes the user hand-impls IndexRuntime.
                let vec_bytes = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<u64, <Self as ::bytesandbrains::contracts::Index>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Index>::add(self, _ctx, &*vec_bytes, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("id".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Search" => {
                // Downcast assumes Self::Vector derefs to &[f32].
                let query_bytes = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let k_val = match _inputs.get(1).and_then(|(_, v)| v.as_any().downcast_ref::<u32>()) {
                    Some(__v) => *__v,
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 1 expected `{}`", stringify!(u32)),
                    }),
                };
                let completion = _ctx.open_completion::<::std::vec::Vec<(u64, f32)>, <Self as ::bytesandbrains::contracts::Index>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Index>::search(self, _ctx, &*query_bytes, k_val, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("results".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Remove" => {
                let id_val = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<u64>()) {
                    Some(__v) => *__v,
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(u64)),
                    }),
                };
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Index>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Index>::remove(self, _ctx, id_val, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Train" => {
                // Corpus as Vec<Box<[f32]>>; each sample derefs to
                // &[f32] = &Self::Vector for the Vector=[f32] case.
                let samples_boxed = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::vec::Vec<::std::boxed::Box<[f32]>>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::vec::Vec<::std::boxed::Box<[f32]>>)),
                    }),
                };
                let sample_refs: ::std::vec::Vec<&[f32]> = samples_boxed.iter().map(|b| b.as_ref()).collect();
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Index>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Index>::train(self, _ctx, &sample_refs, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
    ]
}

fn emit_aggregator_arms() -> Vec<TokenStream> {
    vec![
        quote! {
            "Contribute" => {
                // Downcast assumes Self::Element derefs to &[f32].
                let tensor = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                // Typed `Aggregator::Metadata` — zero serde.
                let metadata = match _inputs.get(1).and_then(|(_, v)| v.as_any().downcast_ref::<<Self as ::bytesandbrains::contracts::Aggregator>::Metadata>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 1 expected `{}`", stringify!(<Self as ::bytesandbrains::contracts::Aggregator>::Metadata)),
                    }),
                };
                // Missing src_peer → typed OpError. Fabricating
                // PeerId(0) misattributes contributions and breaks
                // dedup-by-src (security hole).
                let src = match _ctx.current.inbound.src_peer {
                    ::std::option::Option::Some(p) => p,
                    ::std::option::Option::None => {
                        return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                            detail: ::std::format!(
                                "AggregatorRuntime::Contribute: envelope_src_peer is None — wire envelope did not carry src_peer multihash bytes",
                            ), ..Default::default()
                        });
                    }
                };
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Aggregator>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Aggregator>::contribute(self, _ctx, src, &*tensor, metadata, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Aggregate" => {
                let completion = _ctx.open_completion::<
                    (::std::boxed::Box<[f32]>, <Self as ::bytesandbrains::contracts::Aggregator>::Metadata),
                    <Self as ::bytesandbrains::contracts::Aggregator>::Error
                >();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Aggregator>::aggregate(self, _ctx, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok((params, metadata))) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![
                                ("params".to_string(), ::std::boxed::Box::new(params) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>),
                                ("metadata".to_string(), ::std::boxed::Box::new(metadata) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>),
                            ],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
    ]
}

fn emit_model_arms() -> Vec<TokenStream> {
    // Tensor I/O is Box<[f32]>; derefs to &[f32] = &Self::Tensor for
    // the Tensor=[f32] case.
    vec![
        quote! {
            "Forward" => {
                let input = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<::std::boxed::Box<[f32]>, <Self as ::bytesandbrains::contracts::Model>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Model>::forward(self, _ctx, &*input, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("output".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}"), ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "LoadParameters" => {
                let params = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Model>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Model>::load_parameters(self, _ctx, &*params, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}"), ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Backward" => {
                let grad = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Model>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Model>::backward(self, _ctx, &*grad, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}"), ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "ApplyDelta" => {
                let delta = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Model>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Model>::apply_delta(self, _ctx, &*delta, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}"), ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "ComputeLoss" => {
                let input = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let target = match _inputs.get(1).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 1 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<f32, <Self as ::bytesandbrains::contracts::Model>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Model>::compute_loss(self, _ctx, &*input, &*target, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("loss".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}"), ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Params" => {
                let completion = _ctx.open_completion::<::std::boxed::Box<[f32]>, <Self as ::bytesandbrains::contracts::Model>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Model>::params(self, _ctx, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("params".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}"), ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
    ]
}

fn emit_data_source_arms() -> Vec<TokenStream> {
    // NextBatch → (batch, labels) as Box<[f32]>; labels zero-len for
    // unsupervised sources.
    vec![
        quote! {
            "NextBatch" => {
                let completion = _ctx.open_completion::<
                    (::std::boxed::Box<[f32]>, ::std::boxed::Box<[f32]>),
                    <Self as ::bytesandbrains::contracts::DataSource>::Error
                >();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::DataSource>::next_batch(self, _ctx, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok((batch, labels))) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![
                                ("batch".to_string(), ::std::boxed::Box::new(batch) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>),
                                ("labels".to_string(), ::std::boxed::Box::new(labels) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>),
                            ],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        emit_zero_in_unit_out("Reset", "reset", "::bytesandbrains::contracts::DataSource"),
        emit_zero_in_unit_out(
            "OnDataLoaded",
            "on_data_loaded",
            "::bytesandbrains::contracts::DataSource",
        ),
    ]
}

fn emit_codec_arms() -> Vec<TokenStream> {
    // Hardcoded Int8/zstd shape: train Vec<Box<[f32]>>, encode
    // Box<[f32]> → Box<[u8]>, decode Box<[u8]> → Box<[f32]>.
    // Polymorphic Self::In/Self::Out threading is not yet implemented.
    vec![
        quote! {
            "Train" => {
                let samples_boxed = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::vec::Vec<::std::boxed::Box<[f32]>>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::vec::Vec<::std::boxed::Box<[f32]>>)),
                    }),
                };
                let sample_refs: ::std::vec::Vec<&[f32]> = samples_boxed.iter().map(|b| b.as_ref()).collect();
                let completion = _ctx.open_completion::<(), <Self as ::bytesandbrains::contracts::Codec>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Codec>::train(self, _ctx, &sample_refs, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Encode" => {
                let input = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[f32]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[f32]>)),
                    }),
                };
                let completion = _ctx.open_completion::<::std::boxed::Box<[u8]>, <Self as ::bytesandbrains::contracts::Codec>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Codec>::encode(self, _ctx, &*input, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("output".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "Decode" => {
                let encoded = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<::std::boxed::Box<[u8]>>()) {
                    Some(__v) => __v.clone(),
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(::std::boxed::Box<[u8]>)),
                    }),
                };
                let completion = _ctx.open_completion::<::std::boxed::Box<[f32]>, <Self as ::bytesandbrains::contracts::Codec>::Error>();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::Codec>::decode(self, _ctx, &*encoded, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("output".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
    ]
}

/// Per-op `BackendRuntime::dispatch_atomic` arms. Each downcasts
/// inputs to `Self::Tensor`, builds a single-node `GraphProto`, and
/// calls `Backend::execute`. Uniform across native-graph backends
/// and per-op-override backends (both reach this surface via the
/// default execute walker).
fn emit_backend_op_arms(ops: &'static [OpSpec]) -> Vec<TokenStream> {
    ops.iter()
        .map(|op| {
            let name_lit = op.name;
            quote! {
                #name_lit => {
                    let mut __bb_env: ::std::collections::HashMap<
                        ::std::string::String,
                        <Self as ::bytesandbrains::contracts::Backend>::Tensor,
                    > = ::std::collections::HashMap::with_capacity(_inputs.len());
                    let mut __bb_input_names: ::std::vec::Vec<::std::string::String> =
                        ::std::vec::Vec::with_capacity(_inputs.len());
                    for (__bb_i, (_, __bb_sv)) in _inputs.iter().enumerate() {
                        let __bb_name = ::std::format!("__bb_dispatch_in_{}", __bb_i);
                        __bb_input_names.push(__bb_name.clone());
                        let __bb_tensor = match __bb_sv.as_any().downcast_ref::<
                            <Self as ::bytesandbrains::contracts::Backend>::Tensor,
                        >() {
                            ::std::option::Option::Some(__bb_t) => __bb_t.clone(),
                            ::std::option::Option::None => {
                                return ::std::result::Result::Err(
                                    ::bytesandbrains::bus::OpError {
                                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                                        reason: "input_type_mismatch",
                                        detail: ::std::format!(
                                            "dispatch_atomic input {} is not the backend's Tensor type for op `{}`",
                                            __bb_i,
                                            #name_lit,
                                        ),
                                    },
                                );
                            }
                        };
                        __bb_env.insert(__bb_name, __bb_tensor);
                    }
                    let __bb_node = ::bytesandbrains::proto::onnx::NodeProto {
                        op_type: ::std::string::String::from(#name_lit),
                        input: __bb_input_names,
                        output: ::std::vec![::std::string::String::from("__bb_dispatch_out")],
                        attribute: _ctx.current.node_attributes.to_vec(),
                        ..::core::default::Default::default()
                    };
                    let __bb_graph = ::bytesandbrains::proto::onnx::GraphProto {
                        node: ::std::vec![__bb_node],
                        output: ::std::vec![
                            ::bytesandbrains::proto::onnx::ValueInfoProto {
                                name: ::std::string::String::from("__bb_dispatch_out"),
                                ..::core::default::Default::default()
                            }
                        ],
                        ..::core::default::Default::default()
                    };
                    let __bb_attrs = ::bytesandbrains::contracts::backend::BackendAttrs {
                        current_node_attributes: _ctx.current.node_attributes,
                        current_node_metadata: _ctx.current.node_metadata,
                    };
                    let mut __bb_result = <Self as ::bytesandbrains::contracts::Backend>::execute(
                        self,
                        &__bb_graph,
                        __bb_env,
                        __bb_attrs,
                    ).map_err(|__bb_e| ::bytesandbrains::bus::OpError {
                        detail: ::std::format!("{__bb_e}"),
                        ..::core::default::Default::default()
                    })?;
                    let __bb_output = match __bb_result.remove("__bb_dispatch_out") {
                        ::std::option::Option::Some(__bb_t) => __bb_t,
                        ::std::option::Option::None => {
                            return ::std::result::Result::Err(
                                ::bytesandbrains::bus::OpError {
                                    detail: ::std::format!(
                                        "Backend::execute did not produce `__bb_dispatch_out` for op `{}`",
                                        #name_lit,
                                    ),
                                    ..::core::default::Default::default()
                                },
                            );
                        }
                    };
                    ::std::result::Result::Ok(
                        ::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec![
                            (
                                ::std::string::String::from("output"),
                                ::std::boxed::Box::new(__bb_output)
                                    as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>,
                            ),
                        ]),
                    )
                },
            }
        })
        .collect()
}

/// `"BackendSubgraph"` arm. Extracts the embedded `GraphProto` from
/// the carrier's `BACKEND_SUBGRAPH_BODY_ATTR` attribute, builds an
/// env keyed by the caller-supplied slot names, and calls
/// `Backend::dispatch`. Missing attribute → `ExecutionFailed` (the
/// compiler can't produce this; adversarial wire input can).
fn emit_backend_subgraph_arm() -> TokenStream {
    quote! {
        "BackendSubgraph" => {
            // Keyed by the slot names the carrier supplied.
            let mut __bb_subgraph_env: ::std::collections::HashMap<
                ::std::string::String,
                <Self as ::bytesandbrains::contracts::Backend>::Tensor,
            > = ::std::collections::HashMap::with_capacity(_inputs.len());
            for (__bb_i, (__bb_name, __bb_sv)) in _inputs.iter().enumerate() {
                let __bb_tensor = match __bb_sv.as_any().downcast_ref::<
                    <Self as ::bytesandbrains::contracts::Backend>::Tensor,
                >() {
                    ::std::option::Option::Some(__bb_t) => __bb_t.clone(),
                    ::std::option::Option::None => {
                        return ::std::result::Result::Err(
                            ::bytesandbrains::bus::OpError {
                                kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                                reason: "input_type_mismatch",
                                detail: ::std::format!(
                                    "dispatch_atomic BackendSubgraph input {} is not the backend's Tensor type",
                                    __bb_i,
                                ),
                            },
                        );
                    }
                };
                __bb_subgraph_env.insert((__bb_name).to_string(), __bb_tensor);
            }

            let __bb_graph = {
                let __bb_attr = _ctx.current.node_attributes.iter().find(|__a| {
                    __a.name == ::bytesandbrains::keys::BACKEND_SUBGRAPH_BODY_ATTR
                });
                match __bb_attr.and_then(|__a| __a.g.as_ref()) {
                    ::std::option::Option::Some(__bb_g) => __bb_g,
                    ::std::option::Option::None => {
                        return ::std::result::Result::Err(
                            ::bytesandbrains::bus::OpError {
                                kind: ::bytesandbrains::bus::OpErrorKind::ExecutionFailed,
                                reason: "missing_subgraph_body",
                                detail: ::std::string::String::from(
                                    "BackendSubgraph carrier is missing the `body` GraphProto attribute"
                                ),
                            },
                        );
                    }
                }
            };

            let __bb_attrs = ::bytesandbrains::contracts::backend::BackendAttrs {
                current_node_attributes: _ctx.current.node_attributes,
                current_node_metadata: _ctx.current.node_metadata,
            };

            let __bb_completion = _ctx.open_completion::<
                ::std::collections::HashMap<
                    ::std::string::String,
                    <Self as ::bytesandbrains::contracts::Backend>::Tensor,
                >,
                <Self as ::bytesandbrains::contracts::Backend>::Error,
            >();
            let __bb_cmd_id = __bb_completion.cmd_id();

            match <Self as ::bytesandbrains::contracts::Backend>::dispatch(
                self,
                __bb_graph,
                __bb_subgraph_env,
                __bb_attrs,
                __bb_completion,
            ) {
                ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(__bb_outputs)) => {
                    let __bb_slots: ::std::vec::Vec<(
                        ::std::string::String,
                        ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>,
                    )> = __bb_outputs
                        .into_iter()
                        .map(|(__bb_k, __bb_v)| {
                            (
                                __bb_k,
                                ::std::boxed::Box::new(__bb_v)
                                    as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>,
                            )
                        })
                        .collect();
                    ::std::result::Result::Ok(
                        ::bytesandbrains::atomic::DispatchResult::Immediate(__bb_slots),
                    )
                }
                ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(__bb_e)) => {
                    ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        detail: ::std::format!("{__bb_e}"),
                        ..::core::default::Default::default()
                    })
                }
                ::bytesandbrains::completion::ContractResponse::Later => {
                    ::std::result::Result::Ok(
                        ::bytesandbrains::atomic::DispatchResult::Async(__bb_cmd_id),
                    )
                }
            }
        },
    }
}

fn emit_peer_selector_arms() -> Vec<TokenStream> {
    vec![
        quote! {
            "Sample" => {
                let n = match _inputs.get(0).and_then(|(_, v)| v.as_any().downcast_ref::<u32>()) {
                    Some(__v) => *__v,
                    None => return ::std::result::Result::Err(::bytesandbrains::bus::OpError {
                        kind: ::bytesandbrains::bus::OpErrorKind::TypeMismatch,
                        reason: "input_type_mismatch",
                        detail: format!("input 0 expected `{}`", stringify!(u32)),
                    }),
                };
                let completion = _ctx.open_completion::<
                    ::std::vec::Vec<::bytesandbrains::ids::PeerId>,
                    <Self as ::bytesandbrains::contracts::PeerSelector>::Error
                >();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::PeerSelector>::sample(self, _ctx, n, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("peers".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
        quote! {
            "CurrentView" => {
                let completion = _ctx.open_completion::<
                    ::std::vec::Vec<::bytesandbrains::ids::PeerId>,
                    <Self as ::bytesandbrains::contracts::PeerSelector>::Error
                >();
                let cmd_id = completion.cmd_id();
                match <Self as ::bytesandbrains::contracts::PeerSelector>::current_view(self, _ctx, completion) {
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(value)) => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(
                            ::std::vec![("peers".to_string(), ::std::boxed::Box::new(value) as ::std::boxed::Box<dyn ::bytesandbrains::slot_value::SlotValue>)],
                        ))
                    }
                    ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                        ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                    }
                    ::bytesandbrains::completion::ContractResponse::Later => {
                        ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                    }
                }
            },
        },
    ]
}

/// Zero-input → `()` out (DataSource.reset / on_data_loaded).
fn emit_zero_in_unit_out(op_name: &str, method_name: &str, contract: &str) -> TokenStream {
    let op_lit = op_name;
    let method_ident = syn::Ident::new(method_name, proc_macro2::Span::call_site());
    let contract_path: syn::Path = syn::parse_str(contract).unwrap();
    quote! {
        #op_lit => {
            let completion = _ctx.open_completion::<(), <Self as #contract_path>::Error>();
            let cmd_id = completion.cmd_id();
            match <Self as #contract_path>::#method_ident(self, _ctx, completion) {
                ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Ok(_)) => {
                    ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Immediate(::std::vec::Vec::new()))
                }
                ::bytesandbrains::completion::ContractResponse::Now(::std::result::Result::Err(e)) => {
                    ::std::result::Result::Err(::bytesandbrains::bus::OpError { detail: format!("{e}") , ..Default::default() })
                }
                ::bytesandbrains::completion::ContractResponse::Later => {
                    ::std::result::Result::Ok(::bytesandbrains::atomic::DispatchResult::Async(cmd_id))
                }
            }
        },
    }
}
