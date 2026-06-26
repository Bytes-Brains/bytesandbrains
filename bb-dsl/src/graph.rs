//! Recording context wrapping the in-progress `FunctionProto`. The
//! proto is the IR — semantic BB attributes ride on proto fields,
//! not a parallel Rust shadow store. See `docs/IR_AND_DSL.md` §2.
//!
//! Rust-side wrapper carries what the proto can't represent:
//! `instance_for_pointer` (pointer-identity dedup for generic
//! placeholders) and `site_counter` (output-name minting cache).
//!
//! `Module::build()` constructs `Graph` automatically; `Graph::new()`
//! is for acceptance tests.

use std::any::TypeId;
use std::collections::HashMap;

use crate::output::Output;
use bb_ir::proto::onnx::tensor_proto::DataType as DT;
use bb_ir::proto::onnx::{
    attribute_proto, type_proto, AttributeProto, FunctionProto, NodeProto, StringStringEntryProto,
    TensorShapeProto, TypeProto, ValueInfoProto,
};
use bb_ir::types::TypeNode;

use crate::recorded::RecordedModule;

/// Composition-hierarchy chain stamped by [`Graph::with_function`].
/// Read by the compiler's partition naming.
const MODULE_INSTANCE_KEY: &str = "ai.bytesandbrains.module_instance";

fn upsert_metadata(props: &mut Vec<StringStringEntryProto>, key: &str, value: &str) {
    if let Some(entry) = props.iter_mut().find(|p| p.key == key) {
        entry.value = value.to_string();
    } else {
        props.push(StringStringEntryProto {
            key: key.to_string(),
            value: value.to_string(),
        });
    }
}

/// Recording context every DSL method writes into.
pub struct Graph {
    /// The IR body.
    function: FunctionProto,

    /// Output-name counter.
    site_counter: u64,

    /// Pointer-identity dedup. `TypeId` discriminator avoids
    /// collapsing distinct ZST placeholders (all ZSTs share an
    /// address); same-type ZSTs still alias — documented in
    /// `docs/DEPLOYMENT.md`.
    instance_for_pointer: HashMap<(TypeId, *const ()), u32>,
    next_instance_id: u32,

    /// Active `with_function` scopes; joined chain stamped onto
    /// each NodeProto's `MODULE_INSTANCE_KEY`. Empty stack → default
    /// to `@default`.
    module_scope: Vec<String>,

    /// Nested-`with_function` FunctionProtos. The top-level
    /// `Module::op` wrap folds into the root `function` instead of
    /// creating an entry here.
    sub_functions: Vec<FunctionProto>,

    /// Recording-target stack. `None`/empty → root `function`.
    recording_target: Vec<Option<usize>>,

    /// `true` once any `with_function` has fired. The top-level
    /// wrap-at-depth-1 check uses this to keep the body in
    /// `function[0]` rather than synthesizing a CALL.
    has_seen_function: bool,

    /// Typed recorder errors; `Module::build` surfaces the first.
    pending_errors: Vec<crate::module::BuildError>,

    /// Recording-mode stack. `with_function` pushes `Sealed` so
    /// inner `input()` calls don't leak formals into the outer
    /// FunctionProto. Empty → `Open`.
    mode_stack: Vec<RecordingMode>,

    /// `formal_name → actual_handle` bindings pre-loaded by
    /// `ModuleCall::input`. `g.input("name")` consults the top of
    /// the stack first.
    formal_binding_stack: Vec<HashMap<String, Output>>,

    /// `(target_idx, name) → (handle, TypeNode)` for ports
    /// registered via `output()`. Idempotent per the docs-plan
    /// "single PassThrough producer per name" invariant.
    /// `usize::MAX` indexes the root function.
    named_output_types: HashMap<(usize, String), (Output, &'static TypeNode)>,
}

/// Open top-level vs. sealed nested-function recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordingMode {
    /// `input()` propagates to root + every active sub-function.
    Open,
    /// `input()` lands only on the immediate sub-function.
    Sealed,
}

impl Graph {
    /// Empty `Graph`. `Module::build()` wraps the body in
    /// `with_function(self.name(), ...)` automatically.
    pub fn new() -> Self {
        Self {
            function: FunctionProto::default(),
            site_counter: 0,
            instance_for_pointer: HashMap::new(),
            next_instance_id: 0,
            module_scope: Vec::new(),
            sub_functions: Vec::new(),
            recording_target: Vec::new(),
            has_seen_function: false,
            pending_errors: Vec::new(),
            mode_stack: Vec::new(),
            named_output_types: HashMap::new(),
            formal_binding_stack: Vec::new(),
        }
    }

    /// Current recording mode; `Open` when stack empty.
    fn current_mode(&self) -> RecordingMode {
        self.mode_stack
            .last()
            .copied()
            .unwrap_or(RecordingMode::Open)
    }

    /// Drain accumulated recorder errors.
    pub fn take_pending_errors(&mut self) -> Vec<crate::module::BuildError> {
        std::mem::take(&mut self.pending_errors)
    }

    /// Register a named output port. Idempotent — a second call for
    /// the same `(target_idx, name)` returns the prior handle.
    pub fn output(&mut self, name: &str, handle: Output) {
        let target_idx = self
            .recording_target
            .last()
            .and_then(|t| *t)
            .unwrap_or(usize::MAX);
        let key = (target_idx, name.to_string());
        if self.named_output_types.contains_key(&key) {
            return;
        }
        let type_node = handle.type_node;

        // PassThrough renames the producer's value to the port name
        // so it appears as a NodeProto output for downstream passes.
        self.push_node(NodeProto {
            op_type: bb_ir::syscall_ids::OP_PASS_THROUGH.into(),
            domain: bb_ir::syscall_ids::SYSCALL_DOMAIN.into(),
            input: vec![handle.name.clone()],
            output: vec![name.to_string()],
            ..Default::default()
        });

        let function: &mut FunctionProto = match target_idx {
            usize::MAX => &mut self.function,
            idx => &mut self.sub_functions[idx],
        };
        if function.output.iter().all(|n| n != name) {
            function.output.push(name.to_string());
            function
                .value_info
                .push(type_meta_to_value_info(name, type_node));
        }
        let registered = Output::new(name.to_string(), type_node);
        self.named_output_types.insert(key, (registered, type_node));
    }

    /// Emit a `wire.Send` to `peers` (a `Vec<PeerId>` at dispatch)
    /// and register `name` as a network output. The compiler's
    /// `partition_by_wire_ops` cuts here; `synthesize_wire_recvs`
    /// materializes the matching `wire.Recv`.
    pub fn net_out(&mut self, name: &str, peers: Output, value: Output) {
        let value_type = value.type_node;
        let port_name = name.to_string();
        let handle_name = self.next_site_name();

        let target_idx = self
            .recording_target
            .last()
            .and_then(|t| *t)
            .unwrap_or(usize::MAX);
        let key = (target_idx, port_name.clone());
        let already_registered = self.named_output_types.contains_key(&key);

        self.push_node(NodeProto {
            op_type: bb_ir::syscall_ids::OP_WIRE_SEND.into(),
            domain: bb_ir::syscall_ids::WIRE_DOMAIN.into(),
            input: vec![value.name.clone(), peers.name],
            output: vec![port_name.clone(), handle_name.clone()],
            ..Default::default()
        });
        self.declare_value_info(&port_name, value_type);
        self.declare_value_info(&handle_name, &bb_ir::types::TYPE_WIRE_REQ_ID);

        if !already_registered {
            let function: &mut FunctionProto = match target_idx {
                usize::MAX => &mut self.function,
                idx => &mut self.sub_functions[idx],
            };
            if function.output.iter().all(|n| n != &port_name) {
                function.output.push(port_name.clone());
            }
            let handle = Output::new(port_name.clone(), value_type);
            self.named_output_types.insert(key, (handle, value_type));
        }
    }

    /// Pack N typed Outputs into one composite. Receiver pairs
    /// with [`Self::unbundle`]. Composite envelope is
    /// [`bb_ir::types::TYPE_COMPOSITE`]. Panics on empty `parts`.
    pub fn bundle(&mut self, parts: &[Output]) -> Output {
        assert!(
            !parts.is_empty(),
            "Graph::bundle: parts slice is empty; need >= 1 child Output",
        );
        let bundle_name = self.next_site_name();
        let inputs: Vec<String> = parts.iter().map(|p| p.name.clone()).collect();

        let child_count = parts.len();
        let child_types = parts
            .iter()
            .map(|p| p.type_node.denotation)
            .collect::<Vec<_>>()
            .join(",");

        self.push_node(NodeProto {
            op_type: "Bundle".into(),
            domain: "ai.bytesandbrains.composite".into(),
            input: inputs,
            output: vec![bundle_name.clone()],
            attribute: vec![
                attr_int(
                    "ai.bytesandbrains.composite.child_count",
                    child_count as i64,
                ),
                attr_string("ai.bytesandbrains.composite.child_types", &child_types),
            ],
            ..Default::default()
        });
        self.declare_value_info(&bundle_name, &bb_ir::types::TYPE_COMPOSITE);
        Output::new(bundle_name, &bb_ir::types::TYPE_COMPOSITE)
    }

    /// Extract a composite Output back into its N child Outputs.
    /// `part_types` declares the expected child TypeNodes positionally;
    /// the runtime op validates the envelope's child count against the
    /// declared length and emits one `BytesValue`-shaped output per
    /// child, named `child_{i}` and typed against `part_types[i]` via
    /// the stamped `ValueInfoProto.denotation`. Downstream consumers
    /// decode against that denotation, matching the wire.Recv pattern.
    ///
    /// Panics with a recording-time error if `part_types` is empty —
    /// the matching `g.bundle` cannot have produced a zero-child
    /// envelope.
    pub fn unbundle(&mut self, composite: Output, part_types: &[&'static TypeNode]) -> Vec<Output> {
        assert!(
            !part_types.is_empty(),
            "Graph::unbundle: part_types slice is empty; need >= 1 declared child type",
        );
        let child_count = part_types.len();
        let port_names: Vec<String> = (0..child_count).map(|_| self.next_site_name()).collect();
        let child_types = part_types
            .iter()
            .map(|t| t.denotation)
            .collect::<Vec<_>>()
            .join(",");

        self.push_node(NodeProto {
            op_type: "Unbundle".into(),
            domain: "ai.bytesandbrains.composite".into(),
            input: vec![composite.name],
            output: port_names.clone(),
            attribute: vec![
                attr_int(
                    "ai.bytesandbrains.composite.child_count",
                    child_count as i64,
                ),
                attr_string("ai.bytesandbrains.composite.child_types", &child_types),
            ],
            ..Default::default()
        });
        for (port_name, type_node) in port_names.iter().zip(part_types.iter()) {
            self.declare_value_info(port_name, type_node);
        }
        port_names
            .into_iter()
            .zip(part_types.iter())
            .map(|(name, t)| Output::new(name, t))
            .collect()
    }

    /// Look up a previously-registered output port by name on the
    /// current recording target. Returns `None` when neither
    /// `output(name, ...)` nor an enclosing scope has registered
    /// the port — callers report `BuildError::MissingOutputPort`.
    pub fn lookup_output(&self, name: &str) -> Option<Output> {
        let target_idx = self
            .recording_target
            .last()
            .and_then(|t| *t)
            .unwrap_or(usize::MAX);
        self.named_output_types
            .get(&(target_idx, name.to_string()))
            .map(|(h, _)| h.clone())
    }

    /// Push a [`crate::module::BuildError`] onto the recorder's
    /// pending-errors queue. Used by methods that must keep their
    /// existing return shape (e.g. `Graph::wire` returns the typed
    /// output triple) but want a typed-error escape from a panic.
    pub fn record_build_error(&mut self, err: crate::module::BuildError) {
        self.pending_errors.push(err);
    }

    /// Mutable view of whichever FunctionProto the recorder is
    /// currently writing into. Either the root `function` or one of
    /// the `sub_functions` (per ).
    fn current_function_mut(&mut self) -> &mut FunctionProto {
        match self.recording_target.last() {
            Some(Some(idx)) => &mut self.sub_functions[*idx],
            _ => &mut self.function,
        }
    }

    /// Extract the recorded function body for the compiler to
    /// consume. Called by `Module::build()` after `module.op()`
    /// returns. The chosen-path install constructs concrete
    /// instances via the inventory's `construct_fn` at install
    /// time; the IR carries no instance state.
    pub fn finish(self) -> RecordedModule {
        RecordedModule {
            function: self.function,
            sub_functions: self.sub_functions,
        }
    }

    /// Pointer-identity-keyed slot allocation for generic
    /// placeholders. Appends `"__slot_<slot_id>"` to
    /// `FunctionProto.attribute` on first encounter.
    pub fn register_generic<T: 'static>(
        &mut self,
        instance: &T,
        _required_trait: &'static str,
    ) -> u32 {
        let key = (TypeId::of::<T>(), (instance as *const T).cast::<()>());
        if let Some(&id) = self.instance_for_pointer.get(&key) {
            return id;
        }
        let id = self.next_instance_id;
        self.next_instance_id += 1;
        self.instance_for_pointer.insert(key, id);
        self.current_function_mut()
            .attribute
            .push(format!("__slot_{id}"));
        id
    }

    /// Declare a Module input by name. Lands with
    /// [`bb_ir::types::TYPE_BYTES`] sentinel; the TypeSolver
    /// narrows it later. Propagates up the recording-target chain
    /// in `Open` mode so enclosing CALL NodeProtos stay
    /// referenceable.
    pub fn input(&mut self, name: &str) -> Output {
        // Promote the actual's type into the formal's value_info
        // when the fluent builder pre-loaded a binding.
        let bound_type = self
            .formal_binding_stack
            .last()
            .and_then(|m| m.get(name))
            .map(|h| h.type_node);

        let build_vi = |name: &str| match bound_type {
            Some(type_node) => type_meta_to_value_info(name, type_node),
            None => opaque_value_info(name),
        };

        // Sealed → write only the immediate sub-function. Open →
        // root + every active sub-function.
        let active_targets: Vec<Option<usize>> = match self.current_mode() {
            RecordingMode::Sealed => match self.recording_target.last() {
                Some(slot) => vec![*slot],
                None => Vec::new(),
            },
            RecordingMode::Open => self.recording_target.to_vec(),
        };

        let mut seen_root = false;
        let touch_root = matches!(self.current_mode(), RecordingMode::Open);
        for target in active_targets
            .iter()
            .chain(std::iter::once(&None).take(if touch_root { 1 } else { 0 }))
        {
            let function: &mut FunctionProto = match target {
                Some(idx) => &mut self.sub_functions[*idx],
                None => {
                    if seen_root {
                        continue;
                    }
                    seen_root = true;
                    &mut self.function
                }
            };
            if function.input.iter().all(|n| n != name) {
                function.input.push(name.to_string());
                function.value_info.push(build_vi(name));
            }
        }

        Output::new(name.to_string(), &bb_ir::types::TYPE_BYTES)
    }

    /// Allocate a fresh value-name. Monotonic counter; format
    /// `"v<n>"`.
    pub fn next_site_name(&mut self) -> String {
        let n = self.site_counter;
        self.site_counter += 1;
        format!("v{n}")
    }

    /// Stamp a `ValueInfoProto` for `name` on the current target.
    /// Idempotent. Recorders call this on every minted output.
    pub fn declare_value_info(&mut self, name: &str, type_node: &'static bb_ir::types::TypeNode) {
        let function = self.current_function_mut();
        if function.value_info.iter().any(|v| v.name == name) {
            return;
        }
        function
            .value_info
            .push(type_meta_to_value_info(name, type_node));
    }

    /// Push a NodeProto into the active target. Stamps
    /// `MODULE_INSTANCE_KEY` with the joined `with_function` chain;
    /// an existing stamp is prefixed (preserves replayed hierarchy).
    pub fn push_node(&mut self, mut node: NodeProto) {
        if !self.module_scope.is_empty() {
            let prefix = self.module_scope.join("_");
            let existing = node
                .metadata_props
                .iter()
                .find(|p| p.key == MODULE_INSTANCE_KEY)
                .map(|p| p.value.clone());
            let combined = match existing {
                Some(inner) if !inner.is_empty() => format!("{prefix}_{inner}"),
                _ => prefix,
            };
            upsert_metadata(&mut node.metadata_props, MODULE_INSTANCE_KEY, &combined);
        }
        self.current_function_mut().node.push(node);
    }

    /// Record `body` into a sub-FunctionProto named `name` and emit
    /// a CALL in the outer target. Top-level wraps fold the body
    /// into `function[0]` instead of synthesizing a CALL.
    ///
    /// `bindings` pre-loads formal→actual handles so
    /// `g.input(formal)` inside `body` returns the actual.
    ///
    /// Returns `(child_port_name, parent_call_output_name)` pairs
    /// for non-top-level wraps; empty for top-level (`g.output`
    /// registers directly in the parent scope).
    pub fn with_function<F>(
        &mut self,
        name: &str,
        bindings: &[(String, Output)],
        body: F,
    ) -> Vec<(String, String)>
    where
        F: FnOnce(&mut Graph),
    {
        // Top-level wrap iff first call AND root function untouched.
        let is_top_level_wrap = !self.has_seen_function
            && self.recording_target.is_empty()
            && self.function.node.is_empty()
            && self.function.input.is_empty()
            && self.function.attribute_proto.is_empty();

        self.has_seen_function = true;

        if is_top_level_wrap {
            // Body becomes the entry; no CALL emitted.
            self.function.name = name.to_string();
            self.module_scope.push(name.to_string());
            let depth = self.module_scope.len();
            body(self);
            debug_assert_eq!(
                self.module_scope.len(),
                depth,
                "with_function body must not mutate the scope stack",
            );
            self.module_scope.pop();
            return Vec::new();
        }

        // Existing same-name sub-function → record into a scratch
        // slot and discard, so the canonical FunctionProto stays
        // shared across parents.
        let target_idx = if let Some(idx) = self.sub_functions.iter().position(|f| f.name == name) {
            idx
        } else {
            let new_idx = self.sub_functions.len();
            self.sub_functions.push(FunctionProto {
                name: name.to_string(),
                ..Default::default()
            });
            new_idx
        };

        let is_duplicate = target_idx + 1 != self.sub_functions.len();
        let recording_idx = if is_duplicate {
            let scratch_idx = self.sub_functions.len();
            self.sub_functions.push(FunctionProto::default());
            scratch_idx
        } else {
            target_idx
        };

        // Bind formals so `g.input(formal)` returns the actual.
        let binding_map: HashMap<String, Output> = bindings
            .iter()
            .map(|(name, h)| (name.clone(), h.clone()))
            .collect();
        self.formal_binding_stack.push(binding_map);

        self.recording_target.push(Some(recording_idx));
        self.module_scope.push(name.to_string());
        // seal the recorder while the body runs so any
        // `input()` calls land only on this sub-function and don't
        // leak out into the root function above.
        self.mode_stack.push(RecordingMode::Sealed);
        let depth = self.module_scope.len();
        body(self);
        debug_assert_eq!(
            self.module_scope.len(),
            depth,
            "with_function body must not mutate the scope stack",
        );
        self.mode_stack.pop();
        self.module_scope.pop();
        self.recording_target.pop();
        self.formal_binding_stack.pop();

        // The body's declared outputs (via `g.output(name, value)` /
        // `g.net_out(name, peers, value)`) already populated the
        // sub-function's `output[]` + `value_info[]` lists. Snapshot
        // the recorded output names so the CALL NodeProto's
        // positional output slots match the sub-function's
        // declarations.
        let recorded_outputs: Vec<String> = self.sub_functions[recording_idx].output.clone();

        if is_duplicate {
            self.sub_functions.pop();
        }

        // Emit the CALL NodeProto in the parent scope. Input list is
        // the actuals' names (parent-scope), positionally aligned
        // with the sub-function's declared input ports; output list
        // is freshly minted outer-scope names — one per
        // sub-function output.
        let final_name = self.sub_functions[target_idx].name.clone();
        let call_inputs: Vec<String> = bindings.iter().map(|(_, h)| h.name.clone()).collect();
        let call_outputs: Vec<String> = (0..recorded_outputs.len())
            .map(|_| self.next_site_name())
            .collect();
        let call = NodeProto {
            op_type: final_name,
            domain: "ai.bytesandbrains.module".into(),
            input: call_inputs,
            output: call_outputs.clone(),
            ..Default::default()
        };
        self.push_node(call);

        recorded_outputs.into_iter().zip(call_outputs).collect()
    }

    /// Read-only view of the recorded `FunctionProto`. 's
    /// compiler + the acceptance tests read everything from here -
    /// the proto is the single source of truth.
    pub fn function(&self) -> &FunctionProto {
        &self.function
    }

    /// Read-only view of the nested `with_function` sub-functions
    /// (). Test-only accessor; the
    /// canonical hand-off is via `Graph::finish() -> RecordedModule`.
    #[cfg(test)]
    pub(crate) fn sub_functions_for_test(&self) -> &[FunctionProto] {
        &self.sub_functions
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Stage-5 canonical mapping `TypeNode → ValueInfoProto.type`. Used
/// by [`Graph::input`] so the compiler's `validate` Rule 5 finds a
/// type declaration on every Module input.
///
/// Tensor-typed denotations (`ai.bytesandbrains.tensor.*`) map to a
/// `TensorType` carrying the matching ONNX `DataType` elem_type;
/// everything else maps to an `OpaqueType` keyed by denotation +
/// the `ai.bytesandbrains` domain. 's `bb.wire v1` constellation
/// may refine the mapping further.
/// Build a `ValueInfoProto` whose `TypeProto` is the canonical
/// opaque placeholder. Used by the single-arg [`Graph::input`]
/// API where authors declare ports by name only and the
/// compiler's TypeSolver narrows the type from connected ops.
fn opaque_value_info(name: &str) -> bb_ir::proto::onnx::ValueInfoProto {
    type_meta_to_value_info(name, &bb_ir::types::TYPE_BYTES)
}

fn type_meta_to_value_info(
    name: &str,
    type_node: &'static TypeNode,
) -> bb_ir::proto::onnx::ValueInfoProto {
    let value = if let Some(elem_type) = tensor_elem_from_denotation(type_node.denotation) {
        // Without per-shape metadata on `TypeNode`, every recorded
        // tensor declares an unconstrained shape — downstream
        // type-checking treats absence as "broadcastable", and
        // per-instance shape is the host's responsibility.
        type_proto::Value::TensorType(type_proto::Tensor {
            elem_type,
            shape: Some(TensorShapeProto::default()),
        })
    } else {
        type_proto::Value::OpaqueType(type_proto::Opaque {
            domain: "ai.bytesandbrains".into(),
            name: type_node.denotation.into(),
        })
    };

    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(value),
            denotation: type_node.denotation.into(),
        }),
        ..Default::default()
    }
}

fn tensor_elem_from_denotation(denotation: &str) -> Option<i32> {
    Some(match denotation {
        "ai.bytesandbrains.tensor.f32" => DT::Float as i32,
        "ai.bytesandbrains.tensor.f64" => DT::Double as i32,
        "ai.bytesandbrains.tensor.i32" => DT::Int32 as i32,
        "ai.bytesandbrains.tensor.i64" => DT::Int64 as i32,
        "ai.bytesandbrains.tensor.bool" => DT::Bool as i32,
        _ if denotation.starts_with("ai.bytesandbrains.tensor.") => DT::Undefined as i32,
        _ => return None,
    })
}

/// Construct a `StringStringEntryProto` for `metadata_props` or
/// `attribute_proto.metadata_props`. Used by every DSL method body.
pub fn kv(key: &str, value: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    }
}

/// Construct an `AttributeProto` of type `INT` for `NodeProto.attribute`.
/// Used by DSL methods that pass scalar `i64` config (`axis`, `group`,
/// `to`, etc.).
pub fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

/// Construct an `AttributeProto` of type `FLOAT`. Used for `epsilon`,
/// `alpha`, `momentum`, etc.
pub fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Float as i32,
        f: value,
        ..Default::default()
    }
}

/// Construct an `AttributeProto` of type `INTS`. Used for shape /
/// axes / strides / kernel_shape / pads / dilations / perm vectors.
pub fn attr_ints(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Ints as i32,
        ints: values,
        ..Default::default()
    }
}

/// Construct an `AttributeProto` of type `GRAPH`. Used by `If` /
/// `Loop` body sub-graphs per `docs/IR_AND_DSL.md` Part 2 line 80.
pub fn attr_graph(name: &str, value: bb_ir::proto::onnx::GraphProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Graph as i32,
        g: Some(value),
        ..Default::default()
    }
}

/// Construct an `AttributeProto` of type `STRING`. The proto stores
/// strings as `s: Vec<u8>`; this helper hides the bytes encoding.
/// Used by ops carrying structured-string metadata
/// (e.g. comma-separated TypeNode denotation lists on `composite`
/// Bundle / Unbundle).
pub fn attr_string(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

/// Construct an `AttributeProto` of type `TENSOR`. Used by `Constant`
/// for embedded literal payloads.
pub fn attr_tensor(name: &str, value: bb_ir::proto::onnx::TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Tensor as i32,
        t: Some(value),
        ..Default::default()
    }
}

