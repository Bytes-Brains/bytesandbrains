//! The `Module` trait. `Module::build()` produces one pre-compile
//! `ModelProto` where `functions[0]` is the top-level body and
//! `functions[1..]` are composed sub-Modules (deduped by `name()`).

use bb_ir::proto::onnx::ModelProto;

use crate::graph::Graph;
use crate::output::Output;

/// Recording-time errors. Compile-time errors come from the compiler.
#[derive(Debug)]
pub enum BuildError {
    /// Body recorded zero NodeProtos.
    EmptyModule,

    /// `output()` referenced a port with no recorded producer.
    MissingOutputPort {
        /// The port name as supplied by the user.
        name: String,
    },
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyModule => write!(f, "Module::build: recorded body is empty"),
            Self::MissingOutputPort { name } => write!(
                f,
                "Module::build: output(`{name}`) referenced but no producer recorded",
            ),
        }
    }
}

impl std::error::Error for BuildError {}

/// Fluent call-site for inlining a sub-Module's body. Inlining
/// rather than `FunctionProto` calls so independent branches inside
/// the sub-Module run as soon as their inputs are ready (not
/// blocked on a single CALL barrier).
pub struct ModuleCall<'a, M: ?Sized + Module> {
    module: &'a M,
    bound_inputs: std::vec::Vec<(&'static str, crate::output::Output)>,
}

impl<M: ?Sized + Module> ModuleCall<'_, M> {
    /// Bind a named input to a value the caller already produced.
    pub fn input(mut self, name: &'static str, handle: crate::output::Output) -> Self {
        self.bound_inputs.push((name, handle));
        self
    }

    /// Record the sub-Module's body into `g`. Pull named outputs
    /// from the returned [`ModuleOutputs`].
    ///
    /// ```ignore
    /// let coord_out = self.coordinator.call().input("incoming", q).build(g);
    /// let grad = coord_out.output("aggregated_grad");
    /// ```
    pub fn build(self, g: &mut crate::graph::Graph) -> ModuleOutputs<'_> {
        let bindings: std::vec::Vec<(String, crate::output::Output)> = self
            .bound_inputs
            .iter()
            .map(|(name, h)| ((*name).to_string(), h.clone()))
            .collect();
        let outputs = self.module.op(g, &bindings);
        ModuleOutputs { graph: g, outputs }
    }

    /// Compose a child Module's bootstrap into the parent's
    /// bootstrap. Emits a CALL to `"<name>__bootstrap"`; body-phase
    /// ops gate until the child's CallContext drops.
    pub fn bootstrap(self, g: &mut crate::graph::Graph) -> ModuleOutputs<'_> {
        let bindings: std::vec::Vec<(String, crate::output::Output)> = self
            .bound_inputs
            .iter()
            .map(|(name, h)| ((*name).to_string(), h.clone()))
            .collect();
        let name = format!("{}__bootstrap", self.module.name());
        let outputs = g.with_function(&name, &bindings, |g| self.module.bootstrap(g));
        ModuleOutputs { graph: g, outputs }
    }
}

/// Named-output handle returned by [`ModuleCall::build`].
pub struct ModuleOutputs<'a> {
    graph: &'a mut crate::graph::Graph,
    /// `(child_port_name, parent_call_output_name)` pairs. Empty
    /// for top-level wraps (body `g.output` lands in parent scope).
    outputs: Vec<(String, String)>,
}

impl ModuleOutputs<'_> {
    /// Resolve a named output. Returns the CALL NodeProto's
    /// outer-scope output for sub-function calls, the parent-scope
    /// `lookup_output` for top-level wraps, or a sentinel that
    /// surfaces as `BuildError::MissingOutputPort` downstream.
    pub fn output(&self, name: &'static str) -> crate::output::Output {
        if let Some(call_out) = self
            .outputs
            .iter()
            .find(|(port, _)| port.as_str() == name)
            .map(|(_, call_name)| call_name.clone())
        {
            return crate::output::Output::new(call_out, &bb_ir::types::TYPE_BYTES);
        }
        self.graph.lookup_output(name).unwrap_or_else(|| {
            crate::output::Output::new(name.to_string(), &bb_ir::types::TYPE_BYTES)
        })
    }
}

/// Unit of composition. Implement `name()` + `body()`; framework
/// supplies `op()` and `build()`. Body declares inputs via
/// `g.input("name")` and emits via `g.output("name", value)` (local)
/// or `g.net_out("name", peers, value)` (network).
pub trait Module {
    /// Short stable identifier — becomes `FunctionProto.name`.
    fn name(&self) -> &str;

    /// Recording logic. Compose child Modules via
    /// `self.child.call().input(...).build(g).output(...)`.
    fn body(&self, g: &mut Graph);

    /// Setup recording, run once before the first `body` poll. May
    /// emit `ContractResponse::Later`; the engine drains every
    /// outstanding bootstrap completion before activating body ops.
    fn bootstrap(&self, _g: &mut Graph) {}

    /// Records `body()` into a function scope named `self.name()`.
    /// Emits a CALL in the outer target. Do not override.
    fn op(&self, g: &mut Graph, bindings: &[(String, Output)]) -> Vec<(String, String)> {
        g.with_function(self.name(), bindings, |g| self.body(g))
    }

    /// Open a fluent call-site that inlines `self`'s body.
    fn call(&self) -> ModuleCall<'_, Self> {
        ModuleCall {
            module: self,
            bound_inputs: std::vec::Vec::new(),
        }
    }

    /// Emit one pre-compile `ModelProto`. Body becomes
    /// `functions[0]` stamped with `module_phase = "body"`. If
    /// `bootstrap` recorded any ops it lands as a sibling
    /// `"<name>__bootstrap"` stamped with `"bootstrap"`.
    fn build(self) -> Result<ModelProto, BuildError>
    where
        Self: Sized,
    {
        let mut body_g = Graph::new();
        let bindings: Vec<(String, Output)> = Vec::new();
        let _ = self.op(&mut body_g, &bindings);
        let mut pending = body_g.take_pending_errors();
        if !pending.is_empty() {
            return Err(pending.remove(0));
        }
        let body_recorded = body_g.finish();
        if body_recorded.function.node.is_empty() && body_recorded.sub_functions.is_empty() {
            return Err(BuildError::EmptyModule);
        }

        let body_name = self.name().to_string();
        let mut boot_g = Graph::new();
        boot_g.with_function(&format!("{body_name}__bootstrap"), &[], |g| {
            self.bootstrap(g);
        });
        let mut boot_pending = boot_g.take_pending_errors();
        if !boot_pending.is_empty() {
            return Err(boot_pending.remove(0));
        }
        let boot_recorded = boot_g.finish();

        let mut functions = Vec::with_capacity(
            1 + body_recorded.sub_functions.len() + boot_recorded.sub_functions.len() + 1,
        );
        let mut body_fn = body_recorded.function;
        bb_ir::keys::stamp_function_module_phase(&mut body_fn, bb_ir::keys::MODULE_PHASE_BODY);
        functions.push(body_fn);
        functions.extend(body_recorded.sub_functions);
        if !boot_recorded.function.node.is_empty() {
            let mut boot_fn = boot_recorded.function;
            bb_ir::keys::stamp_function_module_phase(
                &mut boot_fn,
                bb_ir::keys::MODULE_PHASE_BOOTSTRAP,
            );
            functions.push(boot_fn);
            functions.extend(boot_recorded.sub_functions);
        }

        Ok(ModelProto {
            functions,
            ..Default::default()
        })
    }
}

