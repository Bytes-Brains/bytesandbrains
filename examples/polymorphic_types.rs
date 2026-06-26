//! Polymorphic type system end-to-end demo.
//!
//! Walks every piece of the type-system stack:
//!
//! 1. The lattice (`TypeNode` hierarchy + subtype queries)
//! 2. SlotValue runtime-type introspection
//! 3. AtomicOpDecl type relations
//! 4. Compiler TypeSolver resolving a polymorphic graph
//! 5. The CompiledArtifact + dispatch-index stamping
//!
//! Run with:
//! ```text
//! cargo run --example polymorphic_types
//! ```
//!
//! The example is purely illustrative — no Module, no Node. It
//! exercises the type-system surfaces directly so the output
//! reads like a guided tour of what the type system produces.

use bytesandbrains::ops::backends::cpu::{tensor::CpuTensor, CpuBackend};
use bytesandbrains::slot_value::SlotValue;
use bytesandbrains::syscall::values::{BytesValue, PeerIdValue, TriggerValue};
use bytesandbrains::types::{
    Lattice, PortRef, TypeRelation, TYPE_ANY, TYPE_PEER_ID, TYPE_SCALAR_F32, TYPE_TENSOR,
    TYPE_TENSOR_F32, TYPE_TENSOR_F64,
};

use bytesandbrains::atomic::{AtomicOpDecl, AtomicOpKind};
use bytesandbrains::compiler::TypeSolver;
use bytesandbrains::proto::onnx::{
    FunctionProto, GraphProto, ModelProto, NodeProto, ValueInfoProto,
};

fn header(label: &str) {
    println!("\n─── {label} ───────────────────────────────────────");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Polymorphic type system end-to-end demo");

    // ── 1. The lattice ───────────────────────────────────────────
    header("Step 1: TypeNode lattice");
    let lattice = Lattice::get();
    println!("  registered TypeNodes: {}", lattice.nodes().count(),);
    println!(
        "  TYPE_TENSOR_F32.is_subtype_of(TYPE_TENSOR) = {}",
        TYPE_TENSOR_F32.is_subtype_of(&TYPE_TENSOR),
    );
    println!(
        "  TYPE_TENSOR.is_subtype_of(TYPE_ANY) = {}",
        TYPE_TENSOR.is_subtype_of(&TYPE_ANY),
    );
    println!(
        "  TYPE_SCALAR_F32.is_subtype_of(TYPE_TENSOR) = {}",
        TYPE_SCALAR_F32.is_subtype_of(&TYPE_TENSOR),
    );
    println!(
        "  TYPE_PEER_ID.is_subtype_of(TYPE_ANY) = {}",
        TYPE_PEER_ID.is_subtype_of(&TYPE_ANY),
    );

    // ── 2. SlotValue runtime type tags ──────────────────────────
    header("Step 2: SlotValue::runtime_type");
    let values: Vec<(&str, Box<dyn SlotValue>)> = vec![
        ("TriggerValue", Box::new(TriggerValue)),
        (
            "PeerIdValue",
            Box::new(PeerIdValue(bytesandbrains::PeerId::from(7u64))),
        ),
        ("BytesValue", Box::new(BytesValue(b"hello".to_vec()))),
        (
            "CpuTensor",
            Box::new(CpuTensor::new(vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0])),
        ),
    ];
    for (name, value) in &values {
        let t = value.runtime_type();
        println!(
            "  {name:14} runtime_type = {} ({})",
            t.id,
            if t.is_concrete() {
                "concrete"
            } else {
                "abstract"
            }
        );
    }

    // ── 3. AtomicOpDecl with type relations ─────────────────────
    header("Step 3: AtomicOpDecl carrying TypeRelations");
    static ADD_DECL: AtomicOpDecl = AtomicOpDecl {
        name: "Add",
        inputs: &[],
        outputs: &[],
        kind: AtomicOpKind::Immediate,
        type_relations: &[TypeRelation::SameElementType(&[
            PortRef::Input(0),
            PortRef::Input(1),
            PortRef::Output(0),
        ])],
    };
    println!(
        "  Add declares {} relation(s); kind = {:?}",
        ADD_DECL.type_relations.len(),
        ADD_DECL.kind,
    );

    // ── 4. TypeSolver: seed an input, watch it propagate ────────
    header("Step 4: TypeSolver propagating through a graph");
    let graph = GraphProto {
        input: vec![value_info("x"), value_info("y")],
        node: vec![
            // z = Add(x, y)    - all share an element type
            NodeProto {
                op_type: "Add".into(),
                domain: "ai.onnx".into(),
                input: vec!["x".into(), "y".into()],
                output: vec!["z".into()],
                ..Default::default()
            },
            // w = Relu(z)       - elementwise; output mirrors input
            NodeProto {
                op_type: "Relu".into(),
                domain: "ai.onnx".into(),
                input: vec!["z".into()],
                output: vec!["w".into()],
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let decl_for_op = |domain: &str, op_type: &str| -> Option<&'static AtomicOpDecl> {
        match (domain, op_type) {
            ("ai.onnx", "Add") => Some(&ADD_DECL),
            ("ai.onnx", "Relu") => Some(&RELU_DECL),
            _ => None,
        }
    };
    let mut solver = TypeSolver::from_graph(&graph, decl_for_op)?;
    solver.seed("x", &TYPE_TENSOR_F32);
    let solution = solver.solve()?;
    println!("  seed: x = tensor.f32");
    for value in ["x", "y", "z", "w"] {
        let t = solution.type_of(value).expect("solver resolved");
        println!("  resolved: {value} → {}", t.id);
    }

    // Demonstrate the diagonal-variable rule: conflicting concretes fail.
    header("Step 4b: Conflicting seeds raise TypeConstraintFailed");
    let mut solver = TypeSolver::from_graph(&graph, decl_for_op)?;
    solver.seed("x", &TYPE_TENSOR_F32);
    solver.seed("y", &TYPE_TENSOR_F64);
    match solver.solve() {
        Err(e) => println!("  ✓ solver rejected mixed F32/F64: {e}"),
        Ok(_) => println!("  ✗ solver should have rejected"),
    }

    // ── 5. Proto-at-every-boundary: pass through stamp_for_test ──
    header("Step 5: stamp_for_test emits the compilation passport");
    let mut model = ModelProto {
        functions: vec![FunctionProto {
            name: "demo".into(),
            node: vec![
                NodeProto {
                    op_type: "Relu".into(),
                    domain: "ai.onnx".into(),
                    input: vec!["x".into()],
                    output: vec!["y".into()],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Add".into(),
                    domain: "ai.onnx".into(),
                    input: vec!["y".into(), "y".into()],
                    output: vec!["z".into()],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
        ..Default::default()
    };
    bytesandbrains::compiler::stamp_for_test(
        &mut model,
        &[("default_backend", "Backend", "CpuBackend")],
    );
    use bb_ir::keys::{binding_key, read_model_metadata, COMPILED_KEY};
    println!(
        "  bb.compiled stamp = {:?}",
        read_model_metadata(&model, COMPILED_KEY),
    );
    println!(
        "  binding for `default_backend` = {:?}",
        read_model_metadata(&model, &binding_key("demo", "default_backend")),
    );

    println!(
        "Demo printed type-system surfaces (lattice queries, SlotValue tags, \
         TypeSolver propagation, stamp metadata) — no Compiler::compile / \
         bb::install / Node::poll executed here; see tests/ for full E2E."
    );
    let _ = std::marker::PhantomData::<CpuTensor>;
    let _ = std::marker::PhantomData::<CpuBackend>;
    Ok(())
}

fn value_info(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

static RELU_DECL: AtomicOpDecl = AtomicOpDecl {
    name: "Relu",
    inputs: &[],
    outputs: &[],
    kind: AtomicOpKind::Immediate,
    type_relations: &[TypeRelation::Elementwise {
        input: PortRef::Input(0),
        output: PortRef::Output(0),
    }],
};

fn _bring_value_info_to_scope() {
    let _ = ValueInfoProto::default;
}
