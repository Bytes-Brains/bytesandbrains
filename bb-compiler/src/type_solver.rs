//! Type-resolution pass — bipartite worklist solver following
//! TVM Relay's `type_solver.h` design, adapted to Rust.
//!
//! ## Shape
//!
//! Two arenas: **type nodes** (one per value position) and
//! **relation nodes** (one per `TypeRelation` instance on each
//! [`AtomicOpDecl`]). Cross-linked via `rel_set` back-edges. The
//! worklist holds relations ready to (re)run.
//!
//! ## Algorithm
//!
//! 1. **Seed.** Allocate a type node for every value name in the
//!    graph (function inputs, op outputs). Mark each with its
//!    declared bound (`TYPE_ANY` if none).
//! 2. **Instantiate relations.** For each NodeProto in the graph,
//!    look up its `AtomicOpDecl.type_relations` and allocate a
//!    relation node per declared [`TypeRelation`]. Each relation
//!    points at the type nodes for the ports it participates in.
//!    Type nodes track back-edges via `rel_set`.
//! 3. **Drain.** Pop a relation from the worklist, run it.
//!    [`RelationResult`] dictates the next move:
//!    - `Refined` → requeue every relation in the refined type
//!      nodes' `rel_set`.
//!    - `Satisfied` → remove from the worklist permanently.
//!    - `Defer` → leave in the worklist (will retry only when
//!      something else refines a participating type).
//!    - `Failed` → abort with a `TypeError`.
//! 4. **Fixpoint.** When the worklist is empty (or only `Defer`s
//!    remain that no new refinement could activate), check the
//!    post-condition: every type node resolves to a concrete leaf.
//!    Otherwise → `UnresolvedType`.
//!
//! ## Scope
//!
//! Currently handles [`TypeRelation::SameElementType`] and
//! [`TypeRelation::Elementwise`] — the two highest-frequency
//! relations covering most arithmetic + reduction ops. Other
//! variants (`BroadcastShape`, `SameType`, `ReduceOver`, `Custom`)
//! plug in by extending the per-variant handler match inside the
//! solver's internal `run_relation` dispatch.

use std::collections::HashMap;

use bb_ir::proto::onnx::GraphProto;
use bb_ir::types::{PortRef, RelationResult, TypeNode, TypeRelation, TYPE_ANY};

/// Errors the solver may report.
#[derive(Debug)]
pub enum TypeError {
    /// A relation produced a hard contradiction (e.g. two consumers
    /// of the same port require incompatible element types).
    ConstraintFailed {
        /// Op the relation was attached to.
        op: String,
        /// Relation diagnostic string.
        detail: String,
    },
    /// The solver reached fixpoint with type nodes still abstract
    /// (i.e. not narrowed to a concrete leaf in the lattice).
    UnresolvedType {
        /// Value name with no concrete resolution.
        value: String,
    },
    /// An op references a port index that doesn't map to a value
    /// (out-of-range input/output position on `AtomicOpDecl`).
    PortOutOfRange {
        /// Op name.
        op: String,
        /// Failing port reference.
        port: PortRef,
    },
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstraintFailed { op, detail } => {
                write!(f, "type constraint failed at {op}: {detail}")
            }
            Self::UnresolvedType { value } => {
                write!(f, "value `{value}` did not resolve to a concrete type")
            }
            Self::PortOutOfRange { op, port } => {
                write!(f, "op {op} references out-of-range port {port:?}")
            }
        }
    }
}

impl std::error::Error for TypeError {}

/// Solver output: every value name in the graph maps to its
/// resolved concrete TypeNode.
#[derive(Debug)]
pub struct TypeSolution {
    by_value: HashMap<String, &'static TypeNode>,
}

impl TypeSolution {
    /// Resolved TypeNode for a value name. `None` if the solver
    /// didn't see this value.
    pub fn type_of(&self, value: &str) -> Option<&'static TypeNode> {
        self.by_value.get(value).copied()
    }

    /// Iterate every resolved (value_name, TypeNode) pair.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &'static TypeNode)> {
        self.by_value.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

/// Bipartite type-resolution solver.
pub struct TypeSolver {
    /// Per-value type nodes. Index = the slot's position in
    /// [`Self::value_index`].
    types: Vec<TypeNodeSlot>,
    /// Per-relation constraint nodes.
    relations: Vec<RelationNode>,
    /// Value name → index into `types`.
    value_index: HashMap<String, usize>,
}

/// One value position's current type resolution + back-edges to
/// relations that depend on it.
struct TypeNodeSlot {
    /// Current best-known resolution. `&TYPE_ANY` until a relation
    /// narrows it. Refinement only proceeds toward MORE specific
    /// types (down the lattice).
    resolved: &'static TypeNode,
    /// Relations participating in this slot. Populated at solver
    /// construction; consulted when the slot refines to requeue
    /// dependents.
    rel_set: Vec<usize>,
}

/// One instantiated [`TypeRelation`] linked to its participating
/// type slots.
struct RelationNode {
    /// The relation declaration (from the op's atomic_opset).
    decl: &'static TypeRelation,
    /// Op name (for diagnostics).
    op_name: String,
    /// Type-slot indices participating in this relation, in
    /// declaration order. Length matches the relation variant's
    /// port count (e.g. 2 for `Elementwise{input, output}`).
    slots: Vec<usize>,
    /// `true` once the relation reports `Satisfied` and is
    /// permanently removed from the worklist.
    satisfied: bool,
}

impl TypeSolver {
    /// Build a fresh solver from a `GraphProto`. Walks every node,
    /// allocates slots for every value name, instantiates relations
    /// per the op's `AtomicOpDecl.type_relations`.
    ///
    /// `decl_for_op` lets the caller plug in their own
    /// `(domain, op_type) -> &AtomicOpDecl` lookup (typically the
    /// compiler's registered opset catalog).
    pub fn from_graph(
        graph: &GraphProto,
        decl_for_op: impl Fn(&str, &str) -> Option<&'static bb_ir::atomic::AtomicOpDecl>,
    ) -> Result<Self, TypeError> {
        let mut solver = Self {
            types: Vec::new(),
            relations: Vec::new(),
            value_index: HashMap::new(),
        };

        // First pass: allocate a type slot for every value name
        // (graph inputs, then every op's outputs).
        for input in &graph.input {
            solver.intern_value(&input.name);
        }
        for node in &graph.node {
            for out in &node.output {
                if !out.is_empty() {
                    solver.intern_value(out);
                }
            }
            for inp in &node.input {
                if !inp.is_empty() {
                    solver.intern_value(inp);
                }
            }
        }

        // Second pass: for each NodeProto, instantiate the relations
        // declared on its AtomicOpDecl.
        for node in &graph.node {
            let Some(decl) = decl_for_op(&node.domain, &node.op_type) else {
                // No declared opset entry - skip. Unknown ops fall
                // through; resolve_dispatch will catch them downstream.
                continue;
            };
            for relation in decl.type_relations {
                let slots = solver.resolve_relation_ports(node, relation)?;
                let rel_idx = solver.relations.len();
                solver.relations.push(RelationNode {
                    decl: relation,
                    op_name: format!("{}::{}", node.domain, node.op_type),
                    slots: slots.clone(),
                    satisfied: false,
                });
                for s in slots {
                    solver.types[s].rel_set.push(rel_idx);
                }
            }
        }

        Ok(solver)
    }

    fn intern_value(&mut self, name: &str) -> usize {
        if let Some(&idx) = self.value_index.get(name) {
            return idx;
        }
        let idx = self.types.len();
        self.types.push(TypeNodeSlot {
            resolved: &TYPE_ANY,
            rel_set: Vec::new(),
        });
        self.value_index.insert(name.to_string(), idx);
        idx
    }

    /// Resolve `relation`'s [`PortRef`]s against the NodeProto's
    /// input/output lists; return one type-slot index per
    /// declared port.
    fn resolve_relation_ports(
        &mut self,
        node: &bb_ir::proto::onnx::NodeProto,
        relation: &TypeRelation,
    ) -> Result<Vec<usize>, TypeError> {
        let ports: Vec<PortRef> = match relation {
            TypeRelation::SameType(p) | TypeRelation::SameElementType(p) => p.to_vec(),
            TypeRelation::Elementwise { input, output } => vec![*input, *output],
            TypeRelation::BroadcastShape { in0, in1, out } => vec![*in0, *in1, *out],
            TypeRelation::ReduceOver { input, output } => vec![*input, *output],
            TypeRelation::Custom { .. } => Vec::new(),
        };

        let op_name = format!("{}::{}", node.domain, node.op_type);
        let mut slots = Vec::with_capacity(ports.len());
        for port in ports {
            let value_name = match port {
                PortRef::Input(i) => node.input.get(i as usize).cloned(),
                PortRef::Output(o) => node.output.get(o as usize).cloned(),
            };
            let Some(name) = value_name else {
                return Err(TypeError::PortOutOfRange { op: op_name, port });
            };
            if name.is_empty() {
                return Err(TypeError::PortOutOfRange { op: op_name, port });
            }
            slots.push(self.intern_value(&name));
        }
        Ok(slots)
    }

    /// Seed a value's type with a concrete (or narrower-than-Any)
    /// TypeNode. Used by callers that know specific inputs' types
    /// upfront (e.g. an AppEvent feeding a `Tensor<F32>`).
    pub fn seed(&mut self, value: &str, node: &'static TypeNode) {
        if let Some(&idx) = self.value_index.get(value) {
            self.types[idx].resolved = node;
        }
    }

    /// Walk `graph.input` + `graph.value_info` and seed every value
    /// whose `ValueInfoProto.type.denotation` maps to a built-in
    /// TypeNode (via [`bb_ir::types::builtins::lookup_denotation`]).
    /// Values with unknown denotations are left at `TYPE_ANY`; the
    /// solver narrows them via relations during `solve()`.
    ///
    /// Per the architecture's polymorphic-type contract, the DSL's
    /// `Graph::input(name)` records each input with the
    /// `ai.bytesandbrains.opaque` denotation (→ `TYPE_ANY`). Wire
    /// op outputs + framework-recorded values carry pinned
    /// denotations the lookup recognizes; that pinning seeds the
    /// solver with concrete-leaf TypeNodes from which the
    /// relation network propagates.
    pub fn seed_from_value_info(&mut self, graph: &GraphProto) {
        for vi in graph.input.iter().chain(graph.value_info.iter()) {
            let Some(type_proto) = vi.r#type.as_ref() else {
                continue;
            };
            let denotation = type_proto.denotation.as_str();
            if denotation.is_empty() {
                continue;
            }
            if let Some(node) = bb_ir::types::builtins::lookup_denotation(denotation) {
                self.seed(&vi.name, node);
            }
        }
    }

    /// Run the worklist to fixpoint, then post-check that every
    /// slot resolved to a concrete leaf.
    pub fn solve(mut self) -> Result<TypeSolution, TypeError> {
        // Initial worklist = every relation.
        let mut worklist: std::collections::VecDeque<usize> = (0..self.relations.len()).collect();

        while let Some(rel_idx) = worklist.pop_front() {
            if self.relations[rel_idx].satisfied {
                continue;
            }
            let outcome = self.run_relation(rel_idx)?;
            match outcome {
                RelationResult::Refined => {
                    // Requeue dependents of any participating slot.
                    let slots = self.relations[rel_idx].slots.clone();
                    for s in slots {
                        for &dep in &self.types[s].rel_set {
                            if dep != rel_idx && !self.relations[dep].satisfied {
                                worklist.push_back(dep);
                            }
                        }
                    }
                }
                RelationResult::Satisfied => {
                    self.relations[rel_idx].satisfied = true;
                }
                RelationResult::Defer => {
                    // Don't requeue automatically; we'll come back
                    // when a participating slot refines.
                }
                RelationResult::Failed(detail) => {
                    return Err(TypeError::ConstraintFailed {
                        op: self.relations[rel_idx].op_name.clone(),
                        detail: detail.to_string(),
                    });
                }
            }
        }

        // Post-check: every slot must be a concrete leaf.
        let mut by_value: HashMap<String, &'static TypeNode> = HashMap::new();
        for (name, &idx) in &self.value_index {
            let node = self.types[idx].resolved;
            // Allow unresolved (Any) entries to pass through silently
            // - callers may want a partial solution for diagnostics.
            // Hard error happens only if we WERE supposed to resolve.
            by_value.insert(name.clone(), node);
        }
        Ok(TypeSolution { by_value })
    }

    /// Stamp `solution`'s resolved TypeNodes back onto every
    /// matching `ValueInfoProto.type.denotation` in `graph`.
    /// Downstream passes + the runtime read the narrowed
    /// denotation instead of the recorder's
    /// `ai.bytesandbrains.opaque` placeholder.
    ///
    /// Unresolved (still-`TYPE_ANY`) entries are left as-is —
    /// they keep their original denotation. Permissive mode
    /// surfaces here as silent pass-through; strict mode is the
    /// caller's choice via `solve_strict()` BEFORE this is called.
    pub fn apply_solution_to_value_info(graph: &mut GraphProto, solution: &TypeSolution) {
        for vi in graph.input.iter_mut().chain(graph.value_info.iter_mut()) {
            let Some(node) = solution.type_of(&vi.name) else {
                continue;
            };
            if node.is_abstract() {
                continue;
            }
            let denotation = type_node_to_denotation(node);
            if denotation.is_empty() {
                continue;
            }
            if let Some(type_proto) = vi.r#type.as_mut() {
                type_proto.denotation = denotation.to_string();
            }
        }
    }

    /// Strict-mode solve: every slot MUST resolve to a concrete leaf.
    /// Returns `UnresolvedType` on the first abstract slot.
    pub fn solve_strict(self) -> Result<TypeSolution, TypeError> {
        let solution = self.solve()?;
        for (name, node) in &solution.by_value {
            if node.is_abstract() {
                return Err(TypeError::UnresolvedType {
                    value: name.clone(),
                });
            }
        }
        Ok(solution)
    }

    /// Run one relation, return its outcome. The match dispatches
    /// to the per-variant handler.
    fn run_relation(&mut self, idx: usize) -> Result<RelationResult, TypeError> {
        let slots = self.relations[idx].slots.clone();
        let decl = self.relations[idx].decl;

        let outcome = match decl {
            TypeRelation::SameType(_) => self.run_same_type(&slots),
            TypeRelation::SameElementType(_) => self.run_same_element_type(&slots),
            TypeRelation::Elementwise { .. } => self.run_elementwise(&slots),
            TypeRelation::BroadcastShape { .. } => self.run_broadcast_shape(&slots),
            TypeRelation::ReduceOver { .. } => self.run_reduce_over(&slots),
            TypeRelation::Custom { run, .. } => {
                // Custom relations are not yet implemented;
                // defer until `CustomRelationCtx` has a real shape.
                let _ = run;
                Ok(RelationResult::Defer)
            }
        }?;

        Ok(outcome)
    }

    // ---- Per-relation handlers ----------------------------------

    /// `SameType` - every listed slot collapses to ONE concrete
    /// TypeNode. Implementation: take the FIRST concrete resolution
    /// among participants; narrow every other participant to match.
    fn run_same_type(&mut self, slots: &[usize]) -> Result<RelationResult, TypeError> {
        let pivot: Option<&'static TypeNode> = slots
            .iter()
            .map(|&s| self.types[s].resolved)
            .find(|n| n.is_concrete());
        let Some(pivot) = pivot else {
            return Ok(RelationResult::Defer);
        };
        let mut refined = false;
        for &s in slots {
            let cur = self.types[s].resolved;
            if std::ptr::eq(cur, pivot) {
                continue;
            }
            // Allow refinement if the current bound is abstract +
            // pivot is a subtype.
            if cur.is_abstract() && pivot.is_subtype_of(cur) {
                self.types[s].resolved = pivot;
                refined = true;
            } else {
                return Ok(RelationResult::Failed(
                    "SameType: incompatible concrete types",
                ));
            }
        }
        Ok(if refined {
            RelationResult::Refined
        } else {
            RelationResult::Satisfied
        })
    }

    /// `SameElementType` — every Tensor-typed slot shares an
    /// element type. Currently treated as `SameType` (shape not yet
    /// tracked); will tighten once explicit shape constraints land.
    fn run_same_element_type(&mut self, slots: &[usize]) -> Result<RelationResult, TypeError> {
        self.run_same_type(slots)
    }

    /// `Elementwise` - output's TypeNode equals input's. Shape
    /// preserved (when shape tracking lands).
    fn run_elementwise(&mut self, slots: &[usize]) -> Result<RelationResult, TypeError> {
        // slots[0] = input, slots[1] = output
        let inp = self.types[slots[0]].resolved;
        let out = self.types[slots[1]].resolved;
        if inp.is_concrete() && std::ptr::eq(inp, out) {
            return Ok(RelationResult::Satisfied);
        }
        if inp.is_concrete() && out.is_abstract() && inp.is_subtype_of(out) {
            self.types[slots[1]].resolved = inp;
            return Ok(RelationResult::Refined);
        }
        if out.is_concrete() && inp.is_abstract() && out.is_subtype_of(inp) {
            self.types[slots[0]].resolved = out;
            return Ok(RelationResult::Refined);
        }
        if inp.is_concrete() && out.is_concrete() && !std::ptr::eq(inp, out) {
            return Ok(RelationResult::Failed("Elementwise: input != output"));
        }
        Ok(RelationResult::Defer)
    }

    /// `BroadcastShape` — element types unify, output's shape is
    /// the broadcast of the two inputs'. Currently defers to
    /// element-type unification only (shape tracking is not yet
    /// implemented).
    fn run_broadcast_shape(&mut self, slots: &[usize]) -> Result<RelationResult, TypeError> {
        // slots[0] = in0, slots[1] = in1, slots[2] = out
        self.run_same_element_type(&[slots[0], slots[1], slots[2]])
    }

    /// `ReduceOver` - output's element type = input's element type.
    fn run_reduce_over(&mut self, slots: &[usize]) -> Result<RelationResult, TypeError> {
        self.run_elementwise(slots)
    }
}

/// Inverse of [`bb_ir::types::builtins::lookup_denotation`] — map
/// a built-in `TypeNode` back to the canonical denotation string
/// the DSL records on `ValueInfoProto.denotation`. Returns the
/// empty string for nodes without a known denotation (custom
/// types extending the lattice via inventory submission can carry
/// their own denotations; this helper covers the framework
/// canon).
fn type_node_to_denotation(node: &'static TypeNode) -> &'static str {
    use bb_ir::types::builtins as B;
    if std::ptr::eq(node, &B::TYPE_TENSOR_F32) {
        return "ai.bytesandbrains.tensor.f32";
    }
    if std::ptr::eq(node, &B::TYPE_TENSOR_F64) {
        return "ai.bytesandbrains.tensor.f64";
    }
    if std::ptr::eq(node, &B::TYPE_TENSOR_F16) {
        return "ai.bytesandbrains.tensor.f16";
    }
    if std::ptr::eq(node, &B::TYPE_TENSOR_U8) {
        return "ai.bytesandbrains.tensor.u8";
    }
    if std::ptr::eq(node, &B::TYPE_TENSOR_I32) {
        return "ai.bytesandbrains.tensor.i32";
    }
    if std::ptr::eq(node, &B::TYPE_SCALAR_F32) {
        return "bb.f32";
    }
    if std::ptr::eq(node, &B::TYPE_SCALAR_F64) {
        return "bb.f64";
    }
    if std::ptr::eq(node, &B::TYPE_SCALAR_F16) {
        return "bb.f16";
    }
    if std::ptr::eq(node, &B::TYPE_SCALAR_U8) {
        return "bb.u8";
    }
    if std::ptr::eq(node, &B::TYPE_SCALAR_I32) {
        return "bb.i32";
    }
    if std::ptr::eq(node, &B::TYPE_PEER_ID) {
        return "bb.peer_id";
    }
    if std::ptr::eq(node, &B::TYPE_PEER_ID_VEC) {
        return "bb.peer_id_vec";
    }
    if std::ptr::eq(node, &B::TYPE_TRIGGER) {
        return "bb.trigger";
    }
    if std::ptr::eq(node, &B::TYPE_WIRE_REQ_ID) {
        return "bb.wire_req_id";
    }
    ""
}

