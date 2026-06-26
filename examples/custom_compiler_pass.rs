//! Custom compiler stage — register a user-supplied `CompilerStage`
//! that fires AFTER the canonical pipeline, once per emitted
//! partition.
//!
//! Demonstrates the canonical separation:
//!
//!   `Module::build()` → `ModelProto` → `Compiler::new().compile()` →
//!   `Vec<ModelProto>`
//!
//! Plus the extension surface: `Compiler::push_back_stage(MyStage)`.
//! The stage walks every `wire.Send` NodeProto in each emitted
//! partition and stamps a `tracing_id` metadata.
//!
//! Run with:
//! ```text
//! cargo run --example custom_compiler_pass
//! ```

use bytesandbrains::compiler::{CompilerStage, PassError};
use bytesandbrains::placeholders::BackendSlot;
use bytesandbrains::proto::onnx::{ModelProto, StringStringEntryProto};
use bytesandbrains::{Compiler, Graph, Module};

const WIRE_DOMAIN: &str = "ai.bytesandbrains.wire";
const SEND_OP: &str = "Send";
const TRACING_ID_KEY: &str = "example.tracing_id";

// ─── A two-Module composition ─────────────────────────────────────

struct B;

impl Module for B {
    fn name(&self) -> &str {
        "B"
    }
    fn body(&self, g: &mut Graph) {
        let payload = g.input("payload");
        let _ = BackendSlot.identity(g, payload);
    }
}

struct App {
    b: B,
}

impl Module for App {
    fn name(&self) -> &str {
        "App"
    }
    fn body(&self, g: &mut Graph) {
        let payload = g.input("payload");
        let peers = g.input("peers");
        // Explicit network boundary — this is the Send the stage stamps.
        g.net_out("payload_out", peers, payload);
        // Sub-Module composition: the call NodeProto routes to B's
        // FunctionProto; the compiler pairs the upstream Send with B's
        // `payload` input via the synth-Recv pass.
        let downstream = g.lookup_output("payload_out").expect("port registered");
        self.b.call().input("payload", downstream).build(g);
    }
}

// ─── A custom CompilerStage ───────────────────────────────────────

struct StampTracingIds {
    counter: std::sync::atomic::AtomicU32,
}

impl CompilerStage for StampTracingIds {
    fn name(&self) -> &'static str {
        "stamp_tracing_ids"
    }
    fn run(&self, model: &mut ModelProto) -> Result<(), PassError> {
        for func in &mut model.functions {
            for node in &mut func.node {
                if node.domain == WIRE_DOMAIN && node.op_type == SEND_OP {
                    let id = self
                        .counter
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    node.metadata_props.push(StringStringEntryProto {
                        key: TRACING_ID_KEY.into(),
                        value: format!("trace-{id}"),
                    });
                }
            }
        }
        Ok(())
    }
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = App { b: B };
    let recorded_model: ModelProto = app.build()?;
    println!(
        "Module::build() → ModelProto with {} function(s): {:?}",
        recorded_model.functions.len(),
        recorded_model
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
    );

    let stage = StampTracingIds {
        counter: std::sync::atomic::AtomicU32::new(0),
    };
    // The example's Module body uses the generic `BackendSlot`
    // placeholder; the compiler's `validate_all_slots_bound` pass
    // now requires every required role to have an explicit
    // `.bind_<role>::<T>("…")` so the install path can construct
    // an instance via the inventory `construct_fn`.
    let compiled = Compiler::new()
        .bind_backend::<bytesandbrains::ops::backends::cpu::CpuBackend>("compute")
        .push_back_stage(stage)
        .compile(recorded_model)?;

    println!(
        "Compiler::new().compile() → ModelProto with {} function(s)",
        compiled.functions.len(),
    );

    let mut stamped = 0usize;
    for f in &compiled.functions {
        let name = f.name.as_str();
        for node in &f.node {
            if node.domain == WIRE_DOMAIN && node.op_type == SEND_OP {
                let id = node
                    .metadata_props
                    .iter()
                    .find(|p| p.key == TRACING_ID_KEY)
                    .map(|p| p.value.as_str())
                    .unwrap_or("<missing>");
                println!("  target {name}: Send → {id}");
                stamped += 1;
            }
        }
    }

    assert!(
        stamped > 0,
        "expected at least one wire.Send to carry a tracing_id",
    );
    println!("✓ {stamped} wire.Send NodeProto(s) carry tracing_id metadata");
    Ok(())
}
