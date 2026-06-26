//! `Compiler` — single compile entry point. Canonical pipeline
//! runs once per compile target; user stages fire after, once per
//! emitted partition.
//!
//! ```ignore
//! use bytesandbrains::compiler::Compiler;
//!
//! let model = my_module.build()?;
//!
//! let installables = Compiler::default()
//!     .push_back_stage(MyStage::new())
//!     .without_stage("optimize")
//!     .compile(model)?;
//! ```

use std::collections::HashSet;

use crate::artifact::BindingSpec;
use crate::error::CompileError;
use crate::refine_polymorphic_value_info::refine_polymorphic_value_info;
use crate::resolve_component_dependencies::resolve_component_dependencies;
use crate::runner::{run_pipeline_with_options, CANONICAL_PASS_NAMES};
use crate::validate_all_slots_bound::validate_all_slots_bound;
use bb_dsl::recorded::RecordedModule;
use bb_ir::proto::onnx::ModelProto;

/// Concatenate partition `functions[]` into one `ModelProto`.
/// First partition's non-functions fields win; `metadata_props`
/// concatenates (later entries shadow on duplicate keys).
fn merge_partitions_into_one(partitions: Vec<ModelProto>) -> Result<ModelProto, CompileError> {
    let mut iter = partitions.into_iter();
    let Some(mut head) = iter.next() else {
        return Ok(ModelProto::default());
    };
    for next in iter {
        // Content-hash suffixes make collisions vanishingly rare.
        for fn_b in &next.functions {
            if head.functions.iter().any(|fn_a| fn_a.name == fn_b.name) {
                return Err(CompileError::Internal {
                    detail: format!(
                        "duplicate function name after partition merge: {}",
                        fn_b.name
                    ),
                });
            }
        }
        head.functions.extend(next.functions);
        head.metadata_props.extend(next.metadata_props);
    }
    Ok(head)
}

/// Error variants a user-supplied stage may return.
#[derive(Debug)]
pub enum PassError {
    /// Free-form error message from the user's stage.
    Custom(String),
}

impl std::fmt::Display for PassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Custom(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for PassError {}

/// User-supplied compiler stage.
pub trait CompilerStage: Send + Sync {
    /// Unique stage identifier within a Compiler.
    fn name(&self) -> &'static str;

    /// Mutate the emitted partition model.
    fn run(&self, model: &mut ModelProto) -> Result<(), PassError>;
}

/// One author-declared binding from `Compiler::bind_<role>::<T>(slot)`.
#[derive(Clone, Debug)]
struct CompilerBinding {
    /// Slot name matching `#[depends(role = "<slot>")]`.
    slot: String,
    /// PascalCase role identifier.
    role: String,
    /// `<T as ConcreteComponent>::TYPE_NAME`.
    concrete_type_name: String,
}

/// The single compile entry point.
pub struct Compiler {
    pub(crate) canonical_disabled: HashSet<String>,
    pub(crate) stages: Vec<Box<dyn CompilerStage>>,
    pub(crate) per_hop_budget_ns: u64,
    /// IR version the compiler was built to consume.
    /// `run()` checks the input `ModelProto.metadata_props` carries
    /// the matching `FRAMEWORK_IR_VERSION` stamp before any pass
    /// runs; mismatch surfaces as `CompileError::IrVersionMismatch`.
    pub(crate) target_ir_version: u32,
    /// Bindings collected by `bind_<role>::<T>` builders.
    bindings: Vec<CompilerBinding>,
    /// Storage `TypeNode`s parallel to `bindings`.
    binding_storage: Vec<Vec<(&'static str, &'static bb_ir::types::TypeNode)>>,
    /// `true` = strict TypeSolver; `false` allows `TYPE_ANY`
    /// fall-through (for hand-authored test fixtures).
    pub(crate) strict_types: bool,
}

impl Default for Compiler {
    fn default() -> Self {
        Self {
            canonical_disabled: HashSet::new(),
            stages: Vec::new(),
            per_hop_budget_ns: bb_ir::syscall_ids::DEFAULT_PER_HOP_BUDGET_NS,
            target_ir_version: bb_ir::version::FRAMEWORK_IR_VERSION,
            bindings: Vec::new(),
            binding_storage: Vec::new(),
            strict_types: true,
        }
    }
}

impl Compiler {
    /// Fresh compiler.
    ///
    /// ```ignore
    /// let artifact = bb::Compiler::new()
    ///     .bind_backend::<CpuBackend>("compute")
    ///     .bind_index::<HnswIndex>("primary_index")
    ///     .compile(module)?;
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the IR contract version. Mismatch with the input
    /// model's `FRAMEWORK_IR_VERSION` stamp raises
    /// `CompileError::IrVersionMismatch`.
    pub fn with_target_version(mut self, version: u32) -> Self {
        self.target_ir_version = version;
        self
    }

    /// Override the per-hop budget in nanoseconds used by
    /// `derive_wire_deadlines` when stamping static deadlines on
    /// wire ops.
    pub fn with_per_hop_budget_ns(mut self, budget_ns: u64) -> Self {
        self.per_hop_budget_ns = budget_ns;
        self
    }

    /// Permissive type-solver mode for hand-authored NodeProtos.
    /// Unresolved values pass through as `TYPE_ANY`.
    pub fn with_permissive_types(mut self) -> Self {
        self.strict_types = false;
        self
    }

    /// Disable a canonical pass by name. No-op for non-canonical.
    pub fn without_stage(mut self, name: &str) -> Self {
        if CANONICAL_PASS_NAMES.iter().any(|n| *n == name) {
            self.canonical_disabled.insert(name.to_string());
            return self;
        }
        self.stages.retain(|s| s.name() != name);
        self
    }

    /// Insert a stage at the front of the user-stage list.
    pub fn push_front_stage<S: CompilerStage + 'static>(mut self, stage: S) -> Self {
        self.stages.insert(0, Box::new(stage));
        self
    }

    /// Insert a stage at the back of the user-stage list.
    pub fn push_back_stage<S: CompilerStage + 'static>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(stage));
        self
    }

    /// Insert a stage at the supplied index. Index clamped to
    /// `[0, stages.len()]`.
    pub fn insert_stage<S: CompilerStage + 'static>(mut self, index: usize, stage: S) -> Self {
        let idx = index.min(self.stages.len());
        self.stages.insert(idx, Box::new(stage));
        self
    }

    // ─── Binding chain ─────────────────────────────────────────────
    //
    // Each `bind_<role>::<T>(slot)` records a binding; the type bound
    // (`T: ConcreteComponent + <Role>Runtime`) enforces role match.

    /// Bind a `Backend`-role concrete at `slot`.
    pub fn bind_backend<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::BackendRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "BackendRuntime", &["tensor"])
    }

    /// Bind an `Index`-role concrete at `slot`.
    pub fn bind_index<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::IndexRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "IndexRuntime", &["vector"])
    }

    /// Bind a `Model`-role concrete at `slot`.
    pub fn bind_model<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::ModelRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "ModelRuntime", &["tensor"])
    }

    /// Bind an `Aggregator`-role concrete at `slot`.
    pub fn bind_aggregator<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::AggregatorRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "AggregatorRuntime", &["element"])
    }

    /// Bind a `Codec`-role concrete at `slot`.
    pub fn bind_codec<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::CodecRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "CodecRuntime", &["in", "out"])
    }

    /// Bind a `DataSource`-role concrete at `slot`.
    pub fn bind_data_source<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::DataSourceRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "DataSourceRuntime", &["sample"])
    }

    /// Bind a `PeerSelector`-role concrete at `slot`.
    pub fn bind_peer_selector<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::PeerSelectorRuntime,
    {
        // PeerSelector has no Storage-bound associated type.
        self.bind_concrete_with_storage::<T>(slot.into(), "PeerSelectorRuntime", &[])
    }

    /// Bind a `Protocol`-role concrete at `slot`.
    pub fn bind_protocol<T>(self, slot: impl Into<String>) -> Self
    where
        T: bb_runtime::concrete::ConcreteComponent + bb_runtime::roles::ProtocolRuntime,
    {
        self.bind_concrete_with_storage::<T>(slot.into(), "ProtocolRuntime", &[])
    }

    /// Look up per-port Storage `TypeNode`s and stamp them on
    /// the binding. Missing inventory entries (hand-rolled
    /// `<Role>Runtime` impls) silently omit the port; the type
    /// solver treats missing as "unconstrained."
    fn bind_concrete_with_storage<T: bb_runtime::concrete::ConcreteComponent>(
        mut self,
        slot: String,
        role_runtime: &'static str,
        port_names: &[&'static str],
    ) -> Self {
        let concrete_type_name = T::TYPE_NAME;
        let storage_types: Vec<(&'static str, &'static bb_ir::types::TypeNode)> = port_names
            .iter()
            .filter_map(|&port| {
                bb_runtime::registry::lookup_storage_type(concrete_type_name, role_runtime, port)
                    .map(|t| (port, t))
            })
            .collect();
        self.bindings.push(CompilerBinding {
            slot,
            role: role_runtime.to_string(),
            concrete_type_name: concrete_type_name.to_string(),
        });
        self.binding_storage.push(storage_types);
        self
    }

    /// Test-only `BindingSpec` materializer.
    #[cfg(test)]
    pub(crate) fn into_binding_spec(self) -> BindingSpec {
        let empty_storage: Vec<(&'static str, &'static bb_ir::types::TypeNode)> = Vec::new();
        let mut spec = BindingSpec::new();
        for (i, b) in self.bindings.into_iter().enumerate() {
            let storage = self
                .binding_storage
                .get(i)
                .cloned()
                .unwrap_or_else(|| empty_storage.clone());
            spec.push_with_storage(b.slot, b.role, b.concrete_type_name, storage);
        }
        spec
    }

    /// Run the canonical pipeline and emit one compiled `ModelProto`.
    /// Output carries `compiled` passport,
    /// `binding.<target>.<slot>` entries, and `functions[]`
    /// (one partition root per target). See `src/install.rs` for
    /// the install-side parse.
    pub fn compile(self, mut model: ModelProto) -> Result<ModelProto, CompileError> {
        // Build BindingSpec before refine so the type solver walks
        // the narrowed denotations, not the placeholders.
        let mut binding_spec = BindingSpec::new();
        let empty_storage: Vec<(&'static str, &'static bb_ir::types::TypeNode)> = Vec::new();
        for (i, b) in self.bindings.iter().enumerate() {
            let storage = self.binding_storage.get(i).unwrap_or(&empty_storage);
            binding_spec.push_with_storage(
                b.slot.clone(),
                b.role.clone(),
                b.concrete_type_name.clone(),
                storage.clone(),
            );
        }

        refine_polymorphic_value_info(&mut model, &binding_spec)?;

        let mut models = self.run_pipeline(model)?;

        // Stamp dep metadata; reject placeholders missing a binding.
        resolve_component_dependencies(&binding_spec, &mut models)?;
        validate_all_slots_bound(&binding_spec, &models)?;

        let mut targets_per_model: Vec<String> = Vec::with_capacity(models.len());
        for partition in &models {
            let target = partition
                .functions
                .first()
                .map(|f| f.name.clone())
                .unwrap_or_default();
            targets_per_model.push(target);
        }
        for (partition, target) in models.iter_mut().zip(targets_per_model.iter()) {
            crate::stamp_compilation_metadata::stamp_compilation_metadata(
                partition,
                &binding_spec,
                target,
            );
        }

        merge_partitions_into_one(models)
    }

    /// Inspection-only: pipeline output as `Vec<ModelProto>`
    /// without binding validation or passport stamping. Tests only;
    /// production paths use [`Self::compile`].
    pub fn compile_partitions(&self, model: ModelProto) -> Result<Vec<ModelProto>, CompileError> {
        self.run_pipeline(model)
    }

    fn run_pipeline(&self, model: ModelProto) -> Result<Vec<ModelProto>, CompileError> {
        if model.functions.is_empty() {
            return Err(CompileError::EmptyFunctionTable);
        }
        let stamped: Option<u32> = model
            .metadata_props
            .iter()
            .find(|p| p.key == bb_ir::version::FRAMEWORK_IR_VERSION_KEY)
            .and_then(|p| p.value.parse::<u32>().ok());
        if let Some(got) = stamped {
            if got != self.target_ir_version {
                return Err(CompileError::IrVersionMismatch {
                    expected: self.target_ir_version,
                    got,
                });
            }
        }
        let mut iter = model.functions.into_iter();
        let root = iter.next().expect("non-empty checked above");
        let module_name = root.name.clone();
        let sub_functions: Vec<bb_ir::proto::onnx::FunctionProto> = iter.collect();
        let recorded = RecordedModule {
            function: root,
            sub_functions,
        };

        let enabled: HashSet<String> = CANONICAL_PASS_NAMES
            .iter()
            .filter(|n| !self.canonical_disabled.contains(**n))
            .map(|s| s.to_string())
            .collect();

        let mut models = run_pipeline_with_options(
            recorded,
            module_name,
            &enabled,
            self.per_hop_budget_ns,
            self.strict_types,
        )?;

        for stage in &self.stages {
            for model in models.iter_mut() {
                stage.run(model).map_err(|e| CompileError::Internal {
                    detail: format!("compiler stage `{}` failed: {e}", stage.name()),
                })?;
            }
        }

        Ok(models)
    }
}

