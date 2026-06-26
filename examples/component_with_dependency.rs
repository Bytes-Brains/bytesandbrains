//! Worked example - a custom Component declaring + reaching a
//! sibling dependency through the canonical workflow:
//!
//!     Module  →  Compile  →  Install  →  Poll
//!
//! Authoring shape:
//!
//! - `#[depends(<role> = "<slot>")]` on a struct deriving
//!   `bb::Concrete` + a role derive records a `DependencyDecl` on
//!   `ConcreteComponent::DEPENDENCIES`. The compiler reads it in
//!   `resolve_component_dependencies` and refuses to install a
//!   Node whose `<slot>` isn't bound to the declared role.
//! - At dispatch, reach the bound sibling via
//!   `RuntimeResourceRef::dependency::<T>("<slot>")`. The lookup
//!   is total once `Compiler::compile` succeeds — `.expect(...)`
//!   is the intended call site.
//!
//! Run with:
//! ```text
//! cargo run --example component_with_dependency --features test-components
//! ```

use std::task::{Context, Waker};

use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::concrete::ConcreteComponent;
use bytesandbrains::contracts::Backend;
use bytesandbrains::ops::backends::cpu::{self, CpuBackend, CpuTensor};
use bytesandbrains::placeholders::IndexSlot;
use bytesandbrains::proto::onnx::TensorProto;
use bytesandbrains::runtime::RuntimeResourceRef;
use bytesandbrains::{
    install, Address, Compiler, Concrete, Config, Graph, Index, IngressEvent, Module, PeerId,
};
use serde::{Deserialize, Serialize};

// ─── 1. A custom Index that declares a Backend dependency ────────
//
// `#[depends(backend = "compute")]` lands as a
// `DependencyDecl { role: "Backend", slot: "compute" }` on
// `ConcreteComponent::DEPENDENCIES`. The compiler's
// `resolve_component_dependencies` pass verifies "compute" is
// bound at install time and stamps
// `ai.bytesandbrains.dep.Backend = "compute"` onto every NodeProto
// CountingIndex contributes.

#[derive(Clone, Default, Serialize, Deserialize, Concrete, Index)]
#[depends(backend = "compute")]
struct CountingIndex {
    bias: u32,
}

// ONNX `DataType::FLOAT` tag.
const ONNX_FLOAT: i32 = 1;

// Build a 1-D `CpuTensor` of length `len` filled with `value` via
// the backend's `constant` op. Used inside `search` to demonstrate
// that the resolved `&CpuBackend` is a live compute surface, not a
// placeholder.
fn cpu_constant(backend: &CpuBackend, value: f32, len: usize) -> CpuTensor {
    backend
        .constant(TensorProto {
            data_type: ONNX_FLOAT,
            dims: vec![len as i64],
            float_data: vec![value; len],
            ..Default::default()
        })
        .expect("CpuBackend::constant on a length-N f32 proto is infallible")
}

// `Index` Contract impl that composes a real backend call inside
// `search`: it materializes the query as a `CpuTensor`, sums in a
// bias vector via `backend.add`, and discards the result (the
// example is about wiring + dispatch, not retrieval semantics).
impl Index for CountingIndex {
    type Vector = [f32];
    type Error = bytesandbrains::bus::OpError;

    fn add(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _vec: &Self::Vector,
        _c: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error> {
        ContractResponse::Now(Ok(0))
    }

    fn search(
        &self,
        ctx: &mut RuntimeResourceRef<'_>,
        query: &Self::Vector,
        _k: u32,
        _c: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error> {
        let backend = ctx
            .dependency::<CpuBackend>("compute")
            .expect("compiler-verified `compute` slot resolves to a CpuBackend");
        let len = query.len().max(1);
        let q = cpu_constant(backend, query.first().copied().unwrap_or(0.0), len);
        let bias = cpu_constant(backend, self.bias as f32, len);
        let _sum = backend
            .add(&q, &bias)
            .expect("CpuBackend::add on equal-shape f32");
        ContractResponse::Now(Ok(Vec::new()))
    }

    fn remove(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _id: u64,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }
}

// ─── 2. A Module that records the index's Search op ──────────────

struct CountingApp {
    index: IndexSlot,
}

impl Module for CountingApp {
    fn name(&self) -> &str {
        "CountingApp"
    }
    fn body(&self, g: &mut Graph) {
        // Record a Search op so the compiler walks CountingIndex's
        // declared deps + verifies the "compute" slot is bound.
        let query = g.input("query");
        let _ = self.index.search(g, query, 5);
    }
}

// ─── 3. Putting it together ──────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Declared deps ───────────────────────────────────────────
    println!("─── Declared dependencies");
    let deps = <CountingIndex as ConcreteComponent>::DEPENDENCIES;
    println!("CountingIndex declared {} dependency:", deps.len());
    for d in deps {
        println!("  needs role `{}` at slot `{}`", d.role, d.slot,);
    }

    // ── Compile ─────────────────────────────────────────────────
    println!("\n─── Module → Compile");
    let app = CountingApp { index: IndexSlot };
    let model = app.build()?;
    println!(
        "  CountingApp.build() → {} function(s)",
        model.functions.len()
    );

    let compiled = Compiler::new()
        .bind_index::<CountingIndex>("primary_index")
        .bind_backend::<CpuBackend>("compute")
        .compile(model)?;
    println!(
        "  compile() → ModelProto: {} function(s)",
        compiled.functions.len(),
    );

    // ── Install ─────────────────────────────────────────────────
    println!("\n─── Install");
    // CountingIndex + CpuBackend both use `type Config = ()`
    // (Default-derive), so install constructs both via the
    // inventory `construct_fn` with `&()` — no `.with(...)` needed.
    let target = compiled.functions[0].name.clone();
    let mut node = install(
        PeerId::from(42u64),
        vec![Address::empty()],
        compiled,
        &[target.as_str()],
        Config::new(),
    )?;
    println!(
        "  Node installed. Engine slots: primary_index={:?}, compute={:?}",
        node.slot("primary_index").is_some(),
        node.slot("compute").is_some(),
    );

    // ── Drive: push query + poll ────────────────────────────────
    //
    // Push a query input via `IngressEvent::Invoke` so the recorded
    // `Index.search` op moves onto the ready queue. The bridge
    // dispatches into `CountingIndex::search`, where the real
    // `ctx.dependency::<CpuBackend>("compute")` lookup happens —
    // and where `backend.add(&query, &bias)` lands on the
    // `CpuBackend` install bound to the `"compute"` slot.
    println!("\n─── Drive: push query + poll");
    cpu::reset_dispatch_count();
    let query: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let query_bytes = bincode::serialize(&query.into_boxed_slice())?;
    let _ = node.ingress_handle().push(IngressEvent::Invoke {
        module_name: target.clone(),
        inputs: vec![("query".into(), query_bytes)],
        exec_id: bytesandbrains::ids::ExecId::from(0u64),
    });

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut total_steps = 0usize;
    for _ in 0..40 {
        match node.poll(&mut cx) {
            std::task::Poll::Ready(steps) => {
                if steps.is_empty() {
                    break;
                }
                total_steps += steps.len();
            }
            std::task::Poll::Pending => break,
        }
    }
    println!(
        "  {total_steps} EngineStep(s) drained, CpuBackend::execute fired {} time(s)",
        cpu::dispatch_count(),
    );
    assert!(
        total_steps >= 1,
        "expected at least one EngineStep to drain end-to-end; got {total_steps}",
    );
    println!("\n✓ Dependency-declared component installed + driven end-to-end.");
    Ok(())
}
