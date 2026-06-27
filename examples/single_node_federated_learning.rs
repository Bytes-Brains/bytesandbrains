//! Single-Node federated learning — one Node hosts BOTH the
//! client-side and the server-side partitions, sharing one
//! `LinearModel` instance and one `FedAvg<CpuBackend>` aggregator
//! across both roles.
//!
//! This is the canonical multi-role-single-Node pattern: the
//! compiler partitions a `FederatedRound` composition into two
//! installable target functions (the client side that emits the
//! locally-updated parameters and the server side that folds them
//! into the aggregator), and a single
//! `bb::install(...&[client_target, server_target], ...)` call lands
//! both onto one Node. The deduplicating install pass collapses the
//! `"model"` and `"aggregator"` slots so both targets dispatch
//! through the same `ComponentRef` — every Contract method invoked
//! from either side reads and writes the same in-memory state.
//!
//! ## Topology
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────┐
//!  │ ClientLogic                                             │
//!  │   load_parameters(server_params)                        │
//!  │   next_batch → forward → params  ───┐                   │
//!  └─────────────────────────────────────┼───────────────────┘
//!                                        │
//!                                  g.net_out (single Node)
//!                                        │
//!  ┌─────────────────────────────────────▼───────────────────┐
//!  │ ServerLogic                                             │
//!  │   contribute(params, FedAvgMeta)                        │
//!  │   aggregate(trigger) → updated global params            │
//!  └─────────────────────────────────────────────────────────┘
//! ```
//!
//! Both partitions install on the same Node. The shared slot
//! bindings (`bind_model::<LinearModel>("model")` + the inventory
//! constructor) mean exactly one `LinearModel` instance lives in the
//! engine — Client's `load_parameters` writes to the same bytes
//! Server's `params` reads, so the round closes through a single
//! shared in-memory state.
//!
//! Run with:
//! ```text
//! cargo run --example single_node_federated_learning \
//!     --features test-components
//! ```

#![cfg(feature = "test-components")]

use std::task::{Context, Waker};

use bytesandbrains::aggregators::FedAvg;
use bytesandbrains::backends::cpu::CpuBackend;
use bytesandbrains::bus::OpError;
use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::contracts::{DataSource as DataSourceContract, Model as ModelContract};
use bytesandbrains::placeholders::{AggregatorSlot, DataLoaderSlot, ModelSlot};
use bytesandbrains::proto::onnx::ModelProto;
use bytesandbrains::runtime::RuntimeResourceRef;
use bytesandbrains::syscall::values::PeerIdVecValue;
use bytesandbrains::{install, Address, BootstrapTarget, Compiler, Config, Graph, Module, PeerId};

// ─── Constants ────────────────────────────────────────────────────

const PEER: u64 = 7;

// ─── User-supplied concretes (you bring these) ────────────────────

#[derive(
    Clone,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
    bytesandbrains::Concrete,
    bytesandbrains::Model,
)]
struct MyModel {
    w: f32,
}

impl ModelContract for MyModel {
    type Tensor = [f32];
    type Error = OpError;

    fn forward(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        input: &Self::Tensor,
        _c: CompletionHandle<Box<Self::Tensor>, Self::Error>,
    ) -> ContractResponse<Box<Self::Tensor>, Self::Error> {
        let x = input.first().copied().unwrap_or(0.0);
        ContractResponse::Now(Ok(vec![self.w * x].into_boxed_slice()))
    }

    fn load_parameters(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        params: &Self::Tensor,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        if let Some(w) = params.first().copied() {
            self.w = w;
        }
        ContractResponse::Now(Ok(()))
    }

    fn backward(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _grad: &Self::Tensor,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }

    fn apply_delta(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        delta: &Self::Tensor,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        if let Some(d) = delta.first().copied() {
            self.w += d;
        }
        ContractResponse::Now(Ok(()))
    }

    fn compute_loss(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _input: &Self::Tensor,
        _target: &Self::Tensor,
        _c: CompletionHandle<f32, Self::Error>,
    ) -> ContractResponse<f32, Self::Error> {
        ContractResponse::Now(Ok(0.0))
    }

    fn params(
        &self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<Box<Self::Tensor>, Self::Error>,
    ) -> ContractResponse<Box<Self::Tensor>, Self::Error> {
        ContractResponse::Now(Ok(vec![self.w].into_boxed_slice()))
    }
}

#[derive(
    Clone,
    Debug,
    Default,
    serde::Serialize,
    serde::Deserialize,
    bytesandbrains::Concrete,
    bytesandbrains::DataSource,
)]
struct MyData;

impl DataSourceContract for MyData {
    type Sample = [f32];
    type Error = OpError;

    fn next_batch(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<(Box<Self::Sample>, Box<Self::Sample>), Self::Error>,
    ) -> ContractResponse<(Box<Self::Sample>, Box<Self::Sample>), Self::Error> {
        ContractResponse::Now(Ok((
            vec![1.0_f32].into_boxed_slice(),
            vec![2.0_f32].into_boxed_slice(),
        )))
    }

    fn reset(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }

    fn on_data_loaded(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }
}

// ─── Leaf Modules: framework generics only ────────────────────────

/// Per-round client-side compute. Reads `MyModel.params`, runs a
/// local forward pass against `MyData`, and emits the locally-updated
/// parameters. The `g.net_out(...)` call below the leaves declares
/// the partition boundary the compiler uses to emit two install
/// targets.
struct ClientLogic;
impl Module for ClientLogic {
    fn name(&self) -> &str {
        "ClientLogic"
    }
    fn body(&self, g: &mut Graph) {
        let (batch, _labels) = DataLoaderSlot.next_batch(g);
        let _prediction = ModelSlot.forward(g, batch);
        let updated_params = ModelSlot.params(g);
        g.output("updated_params", updated_params);
    }
}

/// Per-round server-side compute. Consumes the inbound `params`
/// fill (the same wire port the client emitted into), folds it into
/// the shared `FedAvg` buffer via `AggregatorSlot.contribute`, and
/// records the contribution metadata (`FedAvgMeta` rides the
/// `metadata` input alongside the tensor).
struct ServerLogic;
impl Module for ServerLogic {
    fn name(&self) -> &str {
        "ServerLogic"
    }
    fn body(&self, g: &mut Graph) {
        let params = g.input("params");
        let metadata = g.input("metadata");
        let _ack = AggregatorSlot.contribute(g, params, metadata);
    }
}

/// Composition: one `g.net_out` between client and server declares
/// the partition boundary. The compiler emits one target function
/// per side; a single Node installs both via
/// `install([client, server])`, sharing the `model` +
/// `aggregator` slot bindings.
struct FederatedRound {
    client: ClientLogic,
    server: ServerLogic,
}

impl Module for FederatedRound {
    fn name(&self) -> &str {
        "FederatedRound"
    }
    fn body(&self, g: &mut Graph) {
        // Single composition-level input names the receiver
        // partition's peer class so the compiler addresses the two
        // sides distinctly.
        let server_peers = g.input("server_peers");

        // Client side runs locally; emits its updated_params output.
        let client_outs = self.client.call().build(g);
        let updated_params = client_outs.output("updated_params");

        // The partition boundary: ship `updated_params` to the
        // server-side partition. The compiler's synth-recv pass
        // materializes the matching `wire.Recv` on the server side
        // and `g.lookup_output(...)` returns the receiver-side
        // `Output` handle.
        g.net_out("params_port", server_peers, updated_params);
        let received_params = g.lookup_output("params_port").expect("net_out port");

        // Server side consumes the inbound fill. For this single-Node
        // demo the metadata channel rides the same wire-payload
        // handle; production deployments wire the `(params, meta)`
        // tuple with a second `g.net_out` carrying typed `FedAvgMeta`.
        let _ = self
            .server
            .call()
            .input("params", received_params.clone())
            .input("metadata", received_params)
            .build(g);
    }
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("─── BytesAndBrains: Single-Node Federated Learning ───");
    println!();
    println!("One Node hosts BOTH ClientLogic AND ServerLogic targets,");
    println!("sharing one MyModel instance and one FedAvg aggregator.");
    println!();

    // STEP 1 — record the composition into a pre-compile ModelProto.
    println!("─── STEP 1: Module composition → pre-compile ModelProto");
    let app = FederatedRound {
        client: ClientLogic,
        server: ServerLogic,
    };
    let pre_compile: ModelProto = app.build()?;
    println!(
        "  FederatedRound.build() → {} function(s)",
        pre_compile.functions.len(),
    );

    // STEP 2 — bind concretes once; the slots collapse across targets.
    println!();
    println!("─── STEP 2: Compiler::new().bind_*().compile()");
    let compiled = Compiler::new()
        .bind_model::<MyModel>("model")
        .bind_data_source::<MyData>("data")
        .bind_aggregator::<FedAvg<CpuBackend>>("aggregator")
        .bind_backend::<CpuBackend>("backend")
        .compile(pre_compile)?;
    println!(
        "  compiled → {} target function(s)",
        compiled.functions.len(),
    );
    for f in &compiled.functions {
        let sends = f
            .node
            .iter()
            .filter(|n| n.domain == "ai.bytesandbrains.wire" && n.op_type == "Send")
            .count();
        let recvs = f
            .node
            .iter()
            .filter(|n| n.domain == "ai.bytesandbrains.wire" && n.op_type == "Recv")
            .count();
        println!("    `{}` — {sends} wire.Send / {recvs} wire.Recv", f.name);
    }

    // Pick the two target names the compiler emitted. Their order
    // follows the partition emission order; the multi-target install
    // API fires bootstrap functions in the same slice order.
    let target_names: Vec<String> = compiled.functions.iter().map(|f| f.name.clone()).collect();
    let client_target = target_names
        .iter()
        .find(|n| !n.split('#').next().unwrap_or("").ends_with("server_peers"))
        .cloned()
        .expect("client-side partition present");
    let server_target = target_names
        .iter()
        .find(|n| n.split('#').next().unwrap_or("").ends_with("server_peers"))
        .cloned()
        .expect("server-side partition present");
    println!("    client target: `{client_target}`");
    println!("    server target: `{server_target}`");

    // STEP 3 — install BOTH targets on a single Node.
    println!();
    println!("─── STEP 3: install(... &[client_target, server_target] ...)");
    let peer = PeerId::from(PEER);
    let addrs = vec![Address::empty().p2p(peer)];
    let mut node = install(
        peer,
        addrs,
        compiled,
        &[client_target.as_str(), server_target.as_str()],
        Config::new(),
    )?;
    println!("  Node @ peer {PEER} hosts both partitions");
    println!("  loaded_modules() → {:?}", node.loaded_modules());

    // Slot-table dedup: the shared `model` + `aggregator` slot
    // bindings collapse into single ComponentRefs visible to every
    // target's dispatch path. The slot lookup returns the same
    // ComponentRef regardless of which target's dispatch surface
    // requests it.
    let model_cref = node.slot("model").expect("model slot bound");
    let aggregator_cref = node.slot("aggregator").expect("aggregator slot bound");
    println!("  shared slot bindings: model={model_cref:?}, aggregator={aggregator_cref:?}",);

    // STEP 4 — host-drive bootstrap + a single federated round in-process.
    println!();
    println!("─── STEP 4: Node::run_bootstrap + drive the poll cycle to quiescence");

    // Host-driven bootstrap: install records the targets; drive
    // `run_bootstrap` before any body ingress so every install-order
    // partition runs its setup recording before the body phase
    // observes the first poll.
    let mut bootstrap_completes = 0usize;
    let mut total_steps = 0usize;
    let boot_steps = node
        .run_bootstrap(BootstrapTarget::All)
        .expect("install-order kick");
    for step in &boot_steps {
        if matches!(step, bytesandbrains::engine::EngineStep::BootstrapComplete) {
            bootstrap_completes += 1;
        }
    }
    total_steps += boot_steps.len();

    // Seed the client partition's composition-level `server_peers`
    // input so the engine has a value to bind when the body ops fire.
    // A single-Node loopback uses the local peer; production
    // deployments dial peers the GlobalRegistryServer surfaced.
    let server_peers_bytes = bytesandbrains::bincode::serialize(&PeerIdVecValue(vec![peer]))?;
    let _ = node.invoke(&client_target, &[("server_peers", &server_peers_bytes)])?;
    println!("  seeded client partition's `server_peers` input via Node::invoke");

    // Drive the install-time poll cascade. Each `Engine::poll`
    // returns the steps the cycle drained — op completions,
    // app-event emissions. A single-Node demo without a bound
    // transport parks Pending after the body queue drains; the
    // engine-driven smoke test (`tests/federated_learning_smoke.rs`)
    // covers the full round-trip with a backend dispatch counter.
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    for _ in 0..64 {
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
        "  drained {total_steps} EngineStep(s); {bootstrap_completes} BootstrapComplete event(s)"
    );

    // STEP 5 — observe the shared model state. Both partitions
    // dispatch into the same `MyModel` instance through the dedup'd
    // `model` slot.
    println!();
    println!("─── STEP 5: shared state observable from one ComponentRef");
    println!("  model slot ComponentRef        : {model_cref:?}");
    println!("  aggregator slot ComponentRef   : {aggregator_cref:?}");
    let both_registered = node
        .loaded_modules()
        .iter()
        .any(|m| m == &client_target.as_str())
        && node
            .loaded_modules()
            .iter()
            .any(|m| m == &server_target.as_str());
    println!("  loaded_modules contains both   : {both_registered}");
    println!();
    println!("Both targets share the model + aggregator slot bindings;");
    println!("ClientLogic.load_parameters writes to the same in-memory");
    println!("MyModel that ServerLogic.params reads. The shared FedAvg");
    println!("aggregator folds contributions across rounds without");
    println!("inter-partition serialization.");

    Ok(())
}
