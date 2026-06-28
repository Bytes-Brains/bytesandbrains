//! Composition pattern with real DSL generics: `DataLoader`,
//! `Model`, `Aggregator`.
//!
//! Three leaf `Module`s use the framework's role placeholders for
//! their compute — none spells out a wire op. `FedNetwork` composes
//! them with explicit `g.net_out(...)` calls. Wires are the
//! partition boundary; the compiler emits one installable per
//! logical node.
//!
//! Each placeholder needs a concrete `<Role>Runtime` impl, bound
//! through the canonical compile chain:
//!
//! ```ignore
//! let artifact = bb::Compiler::new()
//!     .bind_data_source::<StubLoader>("loader")
//!     .bind_model::<StubModel>("model")
//!     .bind_aggregator::<StubAggregator>("aggregator")
//!     .compile(model)?;
//! let node = bb::install(peer_id, addr, artifact, components)?;
//! ```
//!
//! The impls below are minimal — they satisfy each Contract trait +
//! the `bb_derive::<Role>` bridges that emit the universal
//! `ConcreteComponent + AnyComponent + inventory::submit!` triple.
//!
//! Mirrors the canonical federated-learning loop:
//! `DataLoader.next_batch → Model.forward → wire → Aggregator.contribute`.
//!
//! Run with:
//! ```text
//! cargo run --example multi_target_network
//! ```

mod common;

use std::task::{Context, Waker};

use bytesandbrains::placeholders::{AggregatorSlot, DataLoaderSlot, ModelSlot};
use bytesandbrains::proto::onnx::ModelProto;
use bytesandbrains::{Address, Compiler, Graph, IngressEvent, Module, Node, PeerId};

use common::{Bus, StubAggregator, StubLoader, StubModel};

const PEER_SOURCE: u64 = 1;
const PEER_SINK: u64 = 2;

// ─── Leaf compute Modules ─────────────────────────────────────────
// Each leaf uses a framework role placeholder and never spells out
// the network. The composition (FedNetwork) wires them.

/// Pulls a batch via the bound `DataSource` impl.
struct LoaderLeaf;
impl Module for LoaderLeaf {
    fn name(&self) -> &str {
        "LoaderLeaf"
    }
    fn body(&self, g: &mut Graph) {
        let (batch, _labels) = DataLoaderSlot.next_batch(g);
        g.output("batch", batch);
    }
}

/// Runs a forward pass on the bound `Model` impl, producing a
/// prediction for the upstream batch.
struct TrainerLeaf;
impl Module for TrainerLeaf {
    fn name(&self) -> &str {
        "TrainerLeaf"
    }
    fn body(&self, g: &mut Graph) {
        let batch = g.input("batch");
        let prediction = ModelSlot.forward(g, batch);
        g.output("prediction", prediction);
    }
}

/// Folds the inbound contribution + its metadata into the bound
/// `Aggregator` impl. The metadata channel is what hierarchical
/// FedAvg rides on — a child aggregator's emitted `num_samples`
/// flows in alongside the tensor so the parent's reduction can
/// weight the contribution correctly.
struct SinkLeaf;
impl Module for SinkLeaf {
    fn name(&self) -> &str {
        "SinkLeaf"
    }
    fn body(&self, g: &mut Graph) {
        let contribution = g.input("contribution");
        let metadata = g.input("metadata");
        let _ = AggregatorSlot.contribute(g, contribution, metadata);
    }
}

// ─── Network composition ──────────────────────────────────────────

struct FedNetwork {
    loader: LoaderLeaf,
    trainer: TrainerLeaf,
    sink: SinkLeaf,
}

impl Module for FedNetwork {
    fn name(&self) -> &str {
        "FedNetwork"
    }
    fn body(&self, g: &mut Graph) {
        // The composition owns the destination peer-list input.
        // Its name (`sink_peers`) becomes the receiver partition's
        // `peer_class` so the compiler names + addresses the two
        // installables distinctly.
        let sink_peers = g.input("sink_peers");

        // LoaderLeaf (DataLoader.next_batch) → TrainerLeaf
        // (Model.forward) — both local on the source node.
        let loader_outs = self.loader.call().build(g);
        let batch = loader_outs.output("batch");
        let trainer_outs = self.trainer.call().input("batch", batch).build(g);
        let prediction = trainer_outs.output("prediction");

        // Network output: ship the prediction to every sink peer. The
        // compiler partitions Sender-side vs receiver-side; the synth
        // pass materializes the matching Recv on the sink partition.
        g.net_out("prediction_port", sink_peers, prediction);

        // SinkLeaf consumes the inbound payload + a metadata channel.
        // The composition lookup on the net_out port resolves to the
        // wire-Send's `data_out` which the synth pass routes to the
        // sink class.
        let received = g.lookup_output("prediction_port").expect("net_out port");
        // For demo simplicity, route the same payload as metadata —
        // real deployments wire a typed (params, metadata) tuple by
        // adding a second `g.net_out` call.
        let _ = self
            .sink
            .call()
            .input("contribution", received.clone())
            .input("metadata", received)
            .build(g);
    }
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bytesandbrains::ops::link_force();
    println!("─── STEP 1: Module composition → ModelProto");
    let app = FedNetwork {
        loader: LoaderLeaf,
        trainer: TrainerLeaf,
        sink: SinkLeaf,
    };
    let pre_compile: ModelProto = app.build()?;
    println!(
        "  FedNetwork.build() → {} function(s): {:?}",
        pre_compile.functions.len(),
        pre_compile
            .functions
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
    );

    println!("─── STEP 2: Compiler::new().bind_*().compile()");
    let compiled = Compiler::new()
        .bind_data_source::<StubLoader>("loader")
        .bind_model::<StubModel>("model")
        .bind_aggregator::<StubAggregator>("aggregator")
        .compile(pre_compile)?;
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
        println!(
            "  target `{}`: {sends} wire.Send, {recvs} wire.Recv",
            f.name
        );
    }
    println!(
        "  compiled: {} target function(s)",
        compiled.functions.len(),
    );

    println!("─── STEP 3: Install each target on its assigned Node");
    let assign = |name: &str| -> u64 {
        let base = name.split('#').next().unwrap_or("");
        if base.ends_with("sink_peers") {
            PEER_SINK
        } else {
            PEER_SOURCE
        }
    };
    let loader = StubLoader;
    let model = StubModel;
    let aggregator = StubAggregator;
    let mut nodes: Vec<(String, Node, PeerId)> = Vec::new();
    // Each target Module installs from the same compiled
    // ModelProto; the `target` argument picks which function the
    // engine installs as the root graph.
    let target_names: Vec<String> = compiled.functions.iter().map(|f| f.name.clone()).collect();
    for name in target_names {
        let peer_u64 = assign(&name);
        let peer = PeerId::from(peer_u64);
        let addr = Address::empty().p2p(peer);
        // Per the chosen-path install API, every concrete in this
        // example (LoaderLeaf, TrainerLeaf, SinkLeaf, etc.) uses
        // `type Config = ()` via the `#[derive(bb::Concrete)]`
        // default impl, so install supplies `&()` automatically.
        let _ = (&loader, &model, &aggregator);
        let mut node = bytesandbrains::install(
            peer,
            vec![addr],
            compiled.clone(),
            &[name.as_str()],
            bytesandbrains::Config::new(),
        )?;
        node.add_peer(
            PeerId::from(PEER_SOURCE),
            vec![Address::empty().p2p(PeerId::from(PEER_SOURCE))],
        )?;
        node.add_peer(
            PeerId::from(PEER_SINK),
            vec![Address::empty().p2p(PeerId::from(PEER_SINK))],
        )?;
        println!("    `{name}` → peer {peer_u64}");
        nodes.push((name, node, peer));
    }

    println!("─── STEP 4: Connect bus");
    let mut bus = Bus::new();
    for (_, node, peer) in &nodes {
        bus.connect(*peer, node.ingress_handle());
    }

    println!("─── STEP 5: Arm install-order bootstrap on every Node");
    // Host-driven bootstrap: every Node must call `run_bootstrap`
    // to drive its install-order bootstraps to completion before the
    // body phase observes its first poll. The leaf Modules in this
    // composition do not override `bootstrap()`, so the call returns
    // immediately, but it still transitions each partition into the
    // body-ready state the engine expects.
    for (_, node, _) in &mut nodes {
        node.run_bootstrap(&[]).expect("install-order kick");
    }

    println!("─── STEP 6: Drive the source partition's `sink_peers` input");
    let sink_peers_value =
        bytesandbrains::syscall::values::PeerIdVecValue(vec![PeerId::from(PEER_SINK)]);
    let sink_peers_bytes = bincode::serialize(&sink_peers_value)?;

    let source_idx = nodes
        .iter()
        .position(|(n, _, _)| assign(n) == PEER_SOURCE)
        .ok_or("no source partition")?;
    let source_name = nodes[source_idx].0.clone();
    let _ = nodes[source_idx]
        .1
        .ingress_handle()
        .push(IngressEvent::Invoke {
            module_name: source_name,
            inputs: vec![("sink_peers".into(), sink_peers_bytes)],
            exec_id: bytesandbrains::ids::ExecId::from(0u64),
        });

    println!("─── STEP 7: Poll cycle");
    let waker = Waker::noop();
    let mut envelopes_forwarded = 0;
    for cycle in 0..40 {
        let mut any_step = false;
        for (_, node, peer) in &mut nodes {
            let mut cx = Context::from_waker(waker);
            let steps = match node.poll(&mut cx) {
                std::task::Poll::Ready(s) => s,
                std::task::Poll::Pending => continue,
            };
            envelopes_forwarded += bus.forward(*peer, &steps);
            if !steps.is_empty() {
                any_step = true;
            }
        }
        if !any_step && cycle > 5 {
            break;
        }
    }
    println!("  {envelopes_forwarded} envelopes routed by the bus");
    println!(
        "  (Steps 1-6 cover the structural shape: Module composition,\n   \
         compiler partitioning, multi-target install on distinct peers,\n   \
         bus connection, and Node::run_bootstrap arming both partitions.\n   \
         The forwarded-envelope count depends on the engine driving the\n   \
         post-bootstrap body phase to completion; the wire envelope round-\n   \
         trip itself is covered exhaustively in tests/wire_envelope_routes.rs.)"
    );

    println!();
    println!("✓ Multi-target composition + bus routing demo complete");
    Ok(())
}
