//! `bb::install` — single Node construction entry point.
//!
//! `Compiler::compile(module)` produces a `ModelProto` carrying:
//! - `metadata_props["ai.bytesandbrains.compiled"] = "v1"` — passport;
//!   install fails closed without it.
//! - `metadata_props["ai.bytesandbrains.binding.<target>.<slot>"] =
//!   "<role>|<TYPE_NAME>|<slot_id|-1>"` — one per `(target, slot)`.
//! - `functions[]` — every partition's root function.
//!
//! `install(peer_id, addrs, model, targets, config)` verifies the
//! passport, parses binding entries for each target, deduplicates
//! shared bindings ([`InstallError::SlotBindingConflict`] on
//! type/role mismatch), constructs each concrete exactly once via
//! the inventory, and registers every target as an engine entry
//! point. Bootstrap functions queue for serial firing in slice
//! order.
//!
//! Slots whose concrete declares `type Config = ()` skip
//! `Config::with(...)` — install supplies `&()` automatically.

use std::any::Any;
use std::collections::{HashMap, HashSet};

use bb_ir::ids::PeerId;
use bb_ir::keys::{
    parse_binding_key, parse_binding_value, read_model_metadata, BINDING_KEY_PREFIX,
    COMPILED_CURRENT_VERSION, COMPILED_KEY,
};
use bb_ir::proto::onnx::{FunctionProto, ModelProto};
use bb_ir::registry::find_concrete_component;
use bb_runtime::concrete::ComponentHandle;
use bb_runtime::engine::dispatch_entry::FunctionKey;
use bb_runtime::framework::Address;
use bb_runtime::ids::ComponentRef;
use bb_runtime::node::Node;
use bb_runtime::registry::ComponentRole as R;
use bb_runtime::registry::{dispatcher_for, roles_for_component};

/// Per-deployment configuration supplied to [`install`]. Maps slot
/// name → typed config value, downcast to the bound concrete's
/// `<T as ConcreteComponent>::Config` before `T::new`. Slots with
/// `type Config = ()` skip `with(...)`.
pub struct Config {
    configs: HashMap<String, Box<dyn Any>>,
}

impl Config {
    /// Empty config bag.
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Attach a typed config to `slot`. Downcast failures surface as
    /// [`InstallError::ConfigTypeMismatch`].
    pub fn with<C: Any + 'static>(mut self, slot: impl Into<String>, config: C) -> Self {
        self.configs.insert(slot.into(), Box::new(config));
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors surfaced by [`install`].
#[derive(Debug)]
pub enum InstallError {
    /// No `ai.bytesandbrains.compiled` passport — caller installed a
    /// bare `Module::build()` output instead of compiled artifact.
    NotCompiled,

    /// Passport version mismatch; recompile against this framework.
    IncompatibleCompiledVersion {
        /// Passport value read off the model.
        got: String,
        /// Passport value this framework version accepts.
        expected: &'static str,
    },

    /// `target` names no function in `model.functions[]`.
    UnknownTarget {
        /// Target name the user passed.
        target: String,
        /// Functions the model carries (candidates).
        available: Vec<String>,
    },

    /// `binding.<target>.<slot>` metadata entry failed to parse.
    /// Indicates a hand-edited proto.
    InvalidBindingTable {
        /// Metadata key that failed to parse.
        key: String,
        /// Free-form parse error description.
        detail: String,
    },

    /// Binding table references a `TYPE_NAME` not registered by any
    /// `inventory::submit!` carrier in this binary. Link the type's
    /// crate (and its `link_force()` helper, if any).
    UnregisteredConcrete {
        /// The unrecognized `TYPE_NAME`.
        type_name: String,
    },

    /// Config missing for a slot whose concrete declares a non-unit
    /// `Config` associated type.
    MissingConfig {
        /// Slot the binding spec declares.
        slot: String,
        /// `TYPE_NAME` of the concrete the spec says fills this slot.
        type_name: String,
    },

    /// `Config::with(slot, ...)` value's runtime type doesn't match
    /// the bound concrete's `Config` associated type.
    ConfigTypeMismatch {
        /// Slot the binding spec declares.
        slot: String,
        /// `TYPE_NAME` of the concrete the spec says fills this slot.
        type_name: String,
        /// Free-form detail from `ConstructError.detail`.
        detail: String,
    },

    /// `T::new(&config)` returned an `Err` for this slot.
    ConstructionFailed {
        /// Slot the binding spec declares.
        slot: String,
        /// `TYPE_NAME` of the concrete that failed to construct.
        type_name: String,
        /// Stringified impl error.
        detail: String,
    },

    /// Two or more targets bind the same slot to different
    /// `(TYPE_NAME, role)` pairs. A shared slot must resolve to one
    /// `ComponentRef`, so the bindings must agree.
    SlotBindingConflict {
        /// Slot name where the conflicting bindings collided.
        slot: String,
        /// `(target_name, type_name, role)` in call order.
        conflicts: Vec<(String, String, String)>,
    },

    /// The `targets` slice was empty.
    EmptyTargets,
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotCompiled => write!(
                f,
                "install: ModelProto carries no `{COMPILED_KEY}` metadata stamp; \
                 only `bb_compiler::Compiler::compile()` output may be installed",
            ),
            Self::IncompatibleCompiledVersion { got, expected } => write!(
                f,
                "install: ModelProto was compiled against `{got}` but this framework \
                 requires `{expected}`; recompile",
            ),
            Self::UnknownTarget { target, available } => write!(
                f,
                "install: target function `{target}` not found in model.functions[]; \
                 available targets: {available:?}",
            ),
            Self::InvalidBindingTable { key, detail } => write!(
                f,
                "install: binding metadata entry `{key}` is malformed: {detail}",
            ),
            Self::UnregisteredConcrete { type_name } => write!(
                f,
                "install: artifact references `{type_name}` but no \
                 `inventory::submit!` carrier registers it in this binary",
            ),
            Self::MissingConfig { slot, type_name } => write!(
                f,
                "install: slot `{slot}` expected a config for `{type_name}` \
                 (its `Config` associated type is not `()`); add \
                 `Config::new().with(\"{slot}\", <config>)`",
            ),
            Self::ConfigTypeMismatch {
                slot,
                type_name,
                detail,
            } => write!(
                f,
                "install: slot `{slot}` received a config whose type does not \
                 match `{type_name}`'s `Config` associated type ({detail})",
            ),
            Self::ConstructionFailed {
                slot,
                type_name,
                detail,
            } => write!(
                f,
                "install: `{type_name}::new` for slot `{slot}` returned an error: {detail}",
            ),
            Self::SlotBindingConflict { slot, conflicts } => {
                write!(
                    f,
                    "install: slot `{slot}` has conflicting bindings across targets:",
                )?;
                for (target, type_name, role) in conflicts {
                    write!(f, "\n  target `{target}` → `{type_name}` (role `{role}`)")?;
                }
                Ok(())
            }
            Self::EmptyTargets => write!(
                f,
                "install: targets slice is empty; supply at least one target name",
            ),
        }
    }
}

impl std::error::Error for InstallError {}

/// Single Node construction entry point. Verifies the passport,
/// dedupes bindings across targets (rejecting `(TYPE_NAME, role)`
/// conflicts with [`InstallError::SlotBindingConflict`]), constructs
/// each concrete exactly once, and installs every target as a valid
/// `deliver_event` / `invoke` destination.
///
/// `addresses` registers against `peer_id` in the engine's
/// AddressBook; an empty vec skips self-registration. `targets`
/// names target functions; the compiler's content-hash suffix
/// (`<target>#<hash>`) is matched after exact-name. Bootstrap
/// functions are recorded in slice order on
/// `bootstrap_function_keys`; the host calls
/// [`Node::run_bootstrap`] to drive each target serially before
/// the body phase observes the first poll.
pub fn install(
    peer_id: PeerId,
    addresses: Vec<Address>,
    model: ModelProto,
    targets: &[&str],
    config: Config,
) -> Result<Node, InstallError> {
    // Anchor `inventory::submit!{}` blocks so linker DCE keeps them.
    bb_ops::link_force();

    if targets.is_empty() {
        return Err(InstallError::EmptyTargets);
    }

    verify_compilation_stamp(&model)?;

    let mut resolved_target_names: Vec<String> = Vec::with_capacity(targets.len());
    let mut per_target_bindings: Vec<Vec<ResolvedBinding>> = Vec::with_capacity(targets.len());
    for raw in targets {
        let target_function = find_target(&model, raw)?;
        let resolved_name = target_function.name.clone();
        let bindings = parse_target_bindings(&model, &resolved_name)?;
        resolved_target_names.push(resolved_name);
        per_target_bindings.push(bindings);
    }

    let unified = dedupe_bindings_across_targets(&resolved_target_names, &per_target_bindings)?;

    let mut node = Node::new(peer_id, addresses);
    let mut registered_dispatchers: HashSet<(&'static str, &'static str)> = HashSet::new();
    let unit_default: &dyn Any = &();

    for (idx, binding) in unified.iter().enumerate() {
        let next_cref = idx as u32;
        let entry = find_concrete_component(&binding.type_name).ok_or_else(|| {
            InstallError::UnregisteredConcrete {
                type_name: binding.type_name.clone(),
            }
        })?;

        // `type Config = ()` slots fall back to `&()`.
        let supplied: &dyn Any = config
            .configs
            .get(&binding.slot)
            .map(|b| b.as_ref())
            .unwrap_or(unit_default);

        let instance = (entry.construct_fn)(supplied).map_err(|e| {
            if e.detail.starts_with("config type mismatch:") {
                InstallError::ConfigTypeMismatch {
                    slot: binding.slot.clone(),
                    type_name: binding.type_name.clone(),
                    detail: e.detail,
                }
            } else {
                InstallError::ConstructionFailed {
                    slot: binding.slot.clone(),
                    type_name: binding.type_name.clone(),
                    detail: e.detail,
                }
            }
        })?;

        register_dispatchers_for(
            node.engine_install_handle(),
            entry.type_name,
            &mut registered_dispatchers,
        );
        let cref = ComponentRef::from(next_cref);
        let instance_id = next_cref;
        let engine = node.engine_install_handle();
        engine.register_component(cref, instance);
        engine.bind_slot(binding.slot.clone(), cref);
        if let Some(slot_id) = binding.slot_id {
            engine.bind_slot_id(slot_id, cref);
            if let Some(role) = parse_role(&binding.role) {
                engine.bind_slot_id_with_role(slot_id, role, cref);
            }
        }
        stamp_component_roles(engine, entry.type_name, cref);

        node.push_linked_component(ComponentHandle {
            type_name: entry.type_name,
            package: entry.package,
            instance_id,
            serialize_fn: entry.serialize_fn,
            restore_fn: entry.restore_fn,
            state_bytes: Vec::new(),
        });
    }

    // Targets land as entry-point graphs; other functions enter the
    // library so cross-module FunctionCalls resolve at dispatch.
    install_targets(node.engine_install_handle(), &model, &resolved_target_names);
    node.engine_install_handle().resolve_dispatch();

    // Multi-target installs share one `Arc<ModelProto>`.
    node.set_model(model);
    for resolved in &resolved_target_names {
        node.register_module(resolved.clone());
    }

    Ok(node)
}

/// Verify the compilation passport on a model.
fn verify_compilation_stamp(model: &ModelProto) -> Result<(), InstallError> {
    let Some(got) = read_model_metadata(model, COMPILED_KEY) else {
        return Err(InstallError::NotCompiled);
    };
    if got != COMPILED_CURRENT_VERSION {
        return Err(InstallError::IncompatibleCompiledVersion {
            got: got.to_string(),
            expected: COMPILED_CURRENT_VERSION,
        });
    }
    Ok(())
}

/// Resolve `target` against `model.functions[]`. Exact match wins;
/// otherwise accept `<target>#<hash>` (compiler's content-hash suffix).
fn find_target<'a>(model: &'a ModelProto, target: &str) -> Result<&'a FunctionProto, InstallError> {
    if let Some(exact) = model.functions.iter().find(|f| f.name == target) {
        return Ok(exact);
    }
    let prefix = format!("{target}#");
    if let Some(suffixed) = model.functions.iter().find(|f| f.name.starts_with(&prefix)) {
        return Ok(suffixed);
    }
    let available = model
        .functions
        .iter()
        .map(|f| f.name.clone())
        .collect::<Vec<_>>();
    Err(InstallError::UnknownTarget {
        target: target.to_string(),
        available,
    })
}

/// One parsed binding entry.
#[derive(Debug, Clone)]
struct ResolvedBinding {
    slot: String,
    type_name: String,
    /// Compiler-assigned slot id, or `None` for dep-only slots that
    /// don't appear on any role NodeProto. Feeds the engine's
    /// `slot_id_to_cref` lookup used by `resolve_dispatch`.
    slot_id: Option<u32>,
    /// Canonical role identifier (`"Backend"`, `"Index"`, …). Passed
    /// to [`bb_runtime::engine::Engine::bind_slot_id_with_role`] so
    /// `decode_typed_fill` can branch between framework-carrier
    /// decode and backend-mediated tensor materialisation.
    role: String,
}

/// Walk `model.metadata_props` for `binding.<target>.<slot>` entries
/// matching the resolved `target_name` (content-hash suffix included).
fn parse_target_bindings(
    model: &ModelProto,
    target_name: &str,
) -> Result<Vec<ResolvedBinding>, InstallError> {
    let mut out = Vec::new();
    for entry in &model.metadata_props {
        if !entry.key.starts_with(BINDING_KEY_PREFIX) {
            continue;
        }
        let Some((target, slot)) = parse_binding_key(&entry.key) else {
            return Err(InstallError::InvalidBindingTable {
                key: entry.key.clone(),
                detail: "key not in `ai.bytesandbrains.binding.<target>.<slot>` form".into(),
            });
        };
        if target != target_name {
            continue;
        }
        let Some((role, type_name, slot_id)) = parse_binding_value(&entry.value) else {
            return Err(InstallError::InvalidBindingTable {
                key: entry.key.clone(),
                detail: format!(
                    "value `{}` not in `<role>|<TYPE_NAME>|<slot_id|-1>` form",
                    entry.value
                ),
            });
        };
        let slot_id = if slot_id < 0 {
            None
        } else {
            Some(slot_id as u32)
        };
        out.push(ResolvedBinding {
            slot: slot.to_string(),
            type_name: type_name.to_string(),
            slot_id,
            role: role.to_string(),
        });
    }
    Ok(out)
}

fn register_dispatchers_for(
    engine: &mut bb_runtime::engine::Engine,
    type_name: &'static str,
    registered: &mut HashSet<(&'static str, &'static str)>,
) {
    for role in roles_for_component(type_name) {
        let key = (type_name, role_as_str(role));
        if !registered.insert(key) {
            continue;
        }
        if let Some(register_fn) = dispatcher_for(type_name, role) {
            register_fn(engine);
        }
    }
    // Bootstrap dispatcher — the `#[derive(bb::Concrete)]` derive
    // emits one `BootstrapDispatcherRegistration` per concrete so
    // every registered component participates in component
    // bootstrap without the install caller naming the type.
    // Idempotent under the `bootstrap` pseudo-role key so cascade
    // re-registration of the same concrete (multi-target installs)
    // does not double-register.
    let bootstrap_key = (type_name, "bootstrap");
    if registered.insert(bootstrap_key) {
        if let Some(register_fn) = bb_runtime::registry::bootstrap_dispatcher_for(type_name) {
            register_fn(engine);
        }
    }
}

fn stamp_component_roles(
    engine: &mut bb_runtime::engine::Engine,
    type_name: &str,
    cref: ComponentRef,
) {
    let roles: std::collections::HashSet<bb_runtime::registry::ComponentRole> =
        roles_for_component(type_name).collect();
    if !roles.is_empty() {
        engine.set_component_roles(cref, roles);
    }
}

/// Install targets as entry-point graphs and every other function
/// into the library. Entry-point keys are passed to
/// `install_function_library` so each target's GraphSlot lands with
/// `is_entry_point = true`. Bootstrap functions for each target land
/// on [`bb_runtime::engine::Engine::bootstrap_function_keys`] without
/// arming the engine; the host calls `Node::run_bootstrap` to fire
/// them serially in slice order before the body phase runs.
fn install_targets(
    engine: &mut bb_runtime::engine::Engine,
    model: &ModelProto,
    resolved_target_names: &[String],
) {
    let target_set: HashSet<&str> = resolved_target_names.iter().map(|s| s.as_str()).collect();
    let mut entry_point_keys: Vec<FunctionKey> = Vec::with_capacity(resolved_target_names.len());
    for resolved in resolved_target_names {
        let Some(entry) = model
            .functions
            .iter()
            .find(|f| f.name == *resolved)
            .cloned()
        else {
            continue;
        };
        entry_point_keys.push((
            entry.domain.clone(),
            entry.name.clone(),
            entry.overload.clone(),
        ));
        engine.install_graph(entry.name.clone(), entry);
    }

    let sub_functions: Vec<bb_ir::proto::onnx::FunctionProto> = model
        .functions
        .iter()
        .filter(|f| !target_set.contains(f.name.as_str()))
        .cloned()
        .collect();
    engine.install_function_library(&sub_functions, &entry_point_keys);
}

/// One slot's binding after cross-target dedup. Every entry gets
/// exactly one `ComponentRef`, shared across targets referencing the
/// same slot.
#[derive(Debug, Clone)]
struct UnifiedBinding {
    /// Slot name (defaults to field name; overridden by
    /// `#[bb::slot("custom")]`).
    slot: String,
    /// Inventory-registered `TYPE_NAME`. All contributing targets
    /// agreed on this value (else `SlotBindingConflict`).
    type_name: String,
    /// Compiler-assigned slot id, or `None` for dep-only slots not
    /// pinned to a role NodeProto. Compiler stamping guarantees the
    /// same id across contributors so first-wins is safe.
    slot_id: Option<u32>,
    /// Canonical role identifier (`"Backend"`, `"Index"`, …).
    role: String,
}

/// Walk per-target bindings, group by slot, dedup against
/// `(type_name, role)`. Returns one [`UnifiedBinding`] per slot in
/// first-seen order across the call-ordered targets.
fn dedupe_bindings_across_targets(
    target_names: &[String],
    per_target_bindings: &[Vec<ResolvedBinding>],
) -> Result<Vec<UnifiedBinding>, InstallError> {
    // First-seen ordering keeps `ComponentRef` allocation stable
    // across reruns of the same install.
    let mut order: Vec<String> = Vec::new();
    let mut by_slot: HashMap<String, UnifiedBinding> = HashMap::new();
    let mut contributors: HashMap<String, Vec<(String, String, String)>> = HashMap::new();

    for (target_idx, bindings) in per_target_bindings.iter().enumerate() {
        let target_name = &target_names[target_idx];
        for binding in bindings {
            contributors.entry(binding.slot.clone()).or_default().push((
                target_name.clone(),
                binding.type_name.clone(),
                binding.role.clone(),
            ));
            match by_slot.get(&binding.slot) {
                None => {
                    order.push(binding.slot.clone());
                    by_slot.insert(
                        binding.slot.clone(),
                        UnifiedBinding {
                            slot: binding.slot.clone(),
                            type_name: binding.type_name.clone(),
                            slot_id: binding.slot_id,
                            role: binding.role.clone(),
                        },
                    );
                }
                Some(existing) => {
                    if existing.type_name != binding.type_name || existing.role != binding.role {
                        return Err(InstallError::SlotBindingConflict {
                            slot: binding.slot.clone(),
                            conflicts: contributors.remove(&binding.slot).unwrap_or_default(),
                        });
                    }
                }
            }
        }
    }

    Ok(order
        .into_iter()
        .map(|slot| by_slot.remove(&slot).expect("slot inserted above"))
        .collect())
}

fn role_as_str(role: bb_runtime::registry::ComponentRole) -> &'static str {
    match role {
        R::Index => "Index",
        R::Aggregator => "Aggregator",
        R::Model => "Model",
        R::Codec => "Codec",
        R::DataSource => "DataSource",
        R::PeerSelector => "PeerSelector",
        R::Backend => "Backend",
        R::Protocol => "Protocol",
    }
}

/// Parse the role identifier carried in the binding value. Returns
/// `None` for unknown identifiers; callers skip
/// `bind_slot_id_with_role` for forward-compat with future roles.
fn parse_role(role: &str) -> Option<bb_runtime::registry::ComponentRole> {
    match role {
        "Index" => Some(R::Index),
        "Aggregator" => Some(R::Aggregator),
        "Model" => Some(R::Model),
        "Codec" => Some(R::Codec),
        "DataSource" => Some(R::DataSource),
        "PeerSelector" => Some(R::PeerSelector),
        "Backend" => Some(R::Backend),
        "Protocol" => Some(R::Protocol),
        _ => None,
    }
}

