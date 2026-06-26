//! Federated learning play-through — the golden v0.3.0 example.
//!
//! Expresses the canonical federated-learning topology using only
//! the framework's generic role placeholders. There are no
//! application-specific concrete `Model` or `DataLoader` impls
//! defined here — the example shows the *pattern*; users drop in
//! their own `Model` / `DataLoader` to drive real training.
//!
//! ## Bootstrap-as-function pattern
//!
//! `ClientLogic` overrides `Module::bootstrap` to record the
//! canonical client-side seed sequence — two `Constant`s (server
//! `PeerId` + bootstrap multiaddr), an `AddressBook::Insert` that
//! pins the server in the local address book, and a
//! `GlobalRegistryClient.Announce` that opens the discovery
//! handshake. `Module::build` emits the recorded ops as a sibling
//! `ClientLogic__bootstrap` `FunctionProto` stamped with
//! `MODULE_PHASE_BOOTSTRAP`; the engine fires it once at install
//! before the first body poll, then keeps body ops gated until the
//! bootstrap drain completes.
//!
//! Server-side `ServerLogic` + `ServerReduce` take the trait-default
//! no-op bootstrap — servers listen for inbound `Announce`s and
//! never seed.
//!
//! ## What the framework ships (you bind these as-is)
//!
//! - **`bb::aggregators::FedAvg`** — weighted-average aggregator
//!   with typed `FedAvgMeta { num_samples: u64 }` metadata. Crosses
//!   the wire alongside the params via a `g.net_out(...)` port
//!   that carries the `(params, meta)` pair to the server peer.
//! - **`bb::protocols::GlobalRegistryServer`** — discovery: receives
//!   `Announce` envelopes from clients, accumulates announced peers
//!   in the address book under a server-assigned TTL, replies with a
//!   Handshake carrying TTL/heartbeat, exposes the cohort through
//!   `PeerSelector`.
//! - **`bb::protocols::GlobalRegistryClient`** — bootstrap: ships an
//!   `Announce(self_peer)` envelope to a graph-wired `server_peer`
//!   and refreshes its TTL/heartbeat state from the Handshake reply.
//!
//! ## What you bring (drop in your concrete impls)
//!
//! - A `bb::contracts::Model` impl (real training code).
//! - A `bb::contracts::DataSource` impl (your data pipeline).
//!
//! Both are derived with `bb::Concrete + bb::<Role>`
//! and bound at install via `.with_model(&my_model)` /
//! `.with_data_source(&my_loader)`. Once bound, the role-method
//! ops the Modules below record (e.g. `Model.forward`,
//! `DataLoader.next_batch`) route through the engine's
//! `dispatch_atomic` path into your code.
//!
//! ## Topology
//!
//! ```text
//! ClientLogic    : per-client round of "load batch → forward →
//!                  emit updated params" using the generic Model
//!                  + DataLoader placeholders.
//!
//! ServerLogic    : per-round of "sample K peers from
//!                  GlobalRegistryServer → read Model.params → emit
//!                  for broadcast."
//!
//! ServerReduce   : per-incoming-contribution flow of
//!                  "Aggregator.contribute(params, FedAvgMeta)" —
//!                  bound to the framework-shipped FedAvg.
//! ```
//!
//! Run with:
//! ```text
//! cargo run --example federated_learning --features test-components
//! ```

mod common;

use std::task::{Context, Waker};

use bytesandbrains::aggregators::FedAvg;
use bytesandbrains::backends::cpu::CpuBackend;
use bytesandbrains::graph::{attr_tensor, kv};
use bytesandbrains::placeholders::{AggregatorSlot, DataLoaderSlot, ModelSlot, PeerSelectorSlot};
use bytesandbrains::proto::onnx::{tensor_proto, ModelProto, NodeProto, TensorProto};
use bytesandbrains::protocols::{
    GlobalRegistryClient, GlobalRegistryServer, GLOBAL_REGISTRY_DOMAIN,
};
use bytesandbrains::syscall_ids::{OP_CONSTANT, SYSCALL_DOMAIN};
use bytesandbrains::types::{TYPE_ADDRESS_VEC, TYPE_PEER_ID, TYPE_TRIGGER};
use bytesandbrains::{
    install, Address, BootstrapTarget, Compiler, Config, Graph, Module, Output, PeerId,
};

use common::{StubLoader, StubModel};

/// Server peer identity wired through every client's bootstrap +
/// body. A real deployment reads this from a config / discovery
/// surface; the example pins a constant so the recorded NodeProtos
/// are deterministic.
const SERVER_PEER: u64 = 100;
/// Single client peer the example installs.
const CLIENT_PEER: u64 = 101;

/// Address-book op domain `register_op!` registers
/// `AddressBook::InsertMany` under
/// (see `bb-ops/src/syscalls/peers/insert_many.rs`).
const ADDRESS_BOOK_DOMAIN: &str = "ai.bytesandbrains.address_book";

// ─── Module compositions: framework generics only ──────────────────

/// Per-client round: receive the server's params, train one local
/// step, emit the updated params (to be wired back to the server).
///
/// Uses **only** the generic placeholders `Model` and `DataLoader`;
/// users bind their concrete impls at install time. The Module's
/// graph is identical whether the user plugs in a stub identity
/// model + a constant loader or a full PyTorch-style training step
/// with a real dataset.
struct ClientLogic;
impl Module for ClientLogic {
    fn name(&self) -> &str {
        "ClientLogic"
    }
    fn body(&self, g: &mut Graph) {
        // Inbound from the server: the latest global model params.
        let server_params = g.input("server_params");
        // The host wires the upstream server's `updated_params`
        // network port to this Module's `server_params` input —
        // the compiler's synth-recv pass materializes the Recv on
        // this Module's partition.

        // Apply the global params to the local model.
        let _ = ModelSlot.load_parameters(g, server_params);

        // Local training step.
        let (batch, _labels) = DataLoaderSlot.next_batch(g);
        let _prediction = ModelSlot.forward(g, batch);

        // Read the updated params (the impl's `Model.params` would
        // include the local gradient step). Network-out so the host
        // partitions the client+server cleanly without composing a
        // separate wire op outside the Module.
        let updated_params = ModelSlot.params(g);
        let server_peer = g.input("server_peer");
        g.net_out("updated_params", server_peer, updated_params);
    }

    /// Record the canonical client-side bootstrap. Module::build
    /// emits this as `ClientLogic__bootstrap`, a sibling
    /// FunctionProto stamped `MODULE_PHASE_BOOTSTRAP`. The engine
    /// drains it once at install completion before the first body
    /// poll; body ops stay gated until BootstrapComplete lifts.
    ///
    /// Seed sequence:
    ///   Constant(PeerId(server_peer))               → server's stable id
    ///   Constant(AddressVec([addr_1, addr_2, ...])) → server's dial bag
    ///   AddressBook::InsertMany(peer, addresses)    → pin every server endpoint
    ///   GlobalRegistryClient::Announce(peer)        → open discovery handshake
    ///
    /// `AddressVec` carries the FULL bag so the engine's dialer can
    /// pick whichever endpoint it can reach (e.g. LAN before WAN
    /// before a relay). Both `Constant`s carry `TensorProto` payloads
    /// so the compiler's `expand_constant` validator accepts them.
    /// Per-type carriers (`PeerIdValue` / `AddressVecValue`) are the
    /// production shape the AddressBook + Announce ops downcast on;
    /// ergonomic DSL helpers minting those carriers are tracked
    /// separately.
    fn bootstrap(&self, g: &mut Graph) {
        let server_peer = record_constant(
            g,
            "server_peer_const",
            tensor_proto::DataType::Int64,
            &TYPE_PEER_ID,
        );
        let server_addrs = record_constant(
            g,
            "server_addrs_const",
            tensor_proto::DataType::Uint8,
            &TYPE_ADDRESS_VEC,
        );

        let insert_out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "InsertMany".into(),
            domain: ADDRESS_BOOK_DOMAIN.into(),
            input: vec![server_peer.name.clone(), server_addrs.name.clone()],
            output: vec![insert_out_name.clone()],
            metadata_props: vec![
                kv("ai.bytesandbrains.input.peer", &server_peer.name),
                kv("ai.bytesandbrains.input.addresses", &server_addrs.name),
            ],
            ..Default::default()
        });
        g.declare_value_info(&insert_out_name, &TYPE_TRIGGER);

        let announce_out_name = g.next_site_name();
        g.push_node(NodeProto {
            op_type: "Announce".into(),
            domain: GLOBAL_REGISTRY_DOMAIN.into(),
            input: vec![server_peer.name.clone()],
            output: vec![announce_out_name.clone()],
            metadata_props: vec![kv("ai.bytesandbrains.input.server_peer", &server_peer.name)],
            ..Default::default()
        });
        g.declare_value_info(&announce_out_name, &TYPE_TRIGGER);
    }
}

/// Record a `Constant` syscall NodeProto with a stub `TensorProto`
/// payload sized for the declared output type. The compiler's
/// `expand_constant` pass requires `value: TensorProto`; the example
/// satisfies that contract with an empty tensor of the appropriate
/// scalar kind. Outputs are typed `&'static TypeNode` so downstream
/// nodes pick up the right denotation in `value_info`.
fn record_constant(
    g: &mut Graph,
    label: &'static str,
    data_type: tensor_proto::DataType,
    output_type: &'static bytesandbrains::types::TypeNode,
) -> Output {
    let out_name = g.next_site_name();
    let tensor = TensorProto {
        data_type: data_type as i32,
        dims: vec![1],
        ..Default::default()
    };
    g.push_node(NodeProto {
        op_type: OP_CONSTANT.into(),
        domain: SYSCALL_DOMAIN.into(),
        input: vec![],
        output: vec![out_name.clone()],
        attribute: vec![attr_tensor("value", tensor)],
        metadata_props: vec![kv("ai.bytesandbrains.bootstrap.seed", label)],
        ..Default::default()
    });
    g.declare_value_info(&out_name, output_type);
    Output::new(out_name, output_type)
}

/// Server's per-round logic: sample K peers from the global view,
/// snapshot the current model params, and ship them to the sampled
/// clients via a single network port.
struct ServerLogic;
impl Module for ServerLogic {
    fn name(&self) -> &str {
        "ServerLogic"
    }
    fn body(&self, g: &mut Graph) {
        // K random peers from the GlobalRegistryServer's announced set.
        let peers = PeerSelectorSlot::default().sample(g, 2);
        // Current global params shipped across the network.
        let current_params = ModelSlot.params(g);
        g.net_out("server_params", peers, current_params);
    }
}

/// Server's contribution-handling logic: on each inbound
/// `(params, meta)` pair from a client, fold it into the
/// `FedAvg` buffer. The framework-shipped `FedAvg` typed
/// `Metadata = FedAvgMeta { num_samples }` flows through the slot
/// table as a typed value (no serde in-process).
struct ServerReduce;
impl Module for ServerReduce {
    fn name(&self) -> &str {
        "ServerReduce"
    }
    fn body(&self, g: &mut Graph) {
        let received_params = g.input("received_params");
        let received_meta = g.input("received_meta");
        let _ack = AggregatorSlot.contribute(g, received_params, received_meta);
    }
}

// ─── Main ──────────────────────────────────────────────────────────

fn count_role_ops(model: &ModelProto, domain_prefix: &str) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for f in &model.functions {
        for n in &f.node {
            if n.domain.starts_with(domain_prefix) {
                *counts
                    .entry(format!("{}.{}", n.domain, n.op_type))
                    .or_default() += 1;
            }
        }
    }
    counts.into_iter().collect()
}

/// Split a `Module::build` ModelProto's functions into
/// `(body, bootstrap)` buckets via the `MODULE_PHASE_*` stamps.
/// Used by `main` to surface the bootstrap-as-function shape on
/// the demo's STEP 1 print.
fn phase_counts(model: &ModelProto) -> (usize, usize) {
    use bb_ir::keys::{read_function_module_phase, MODULE_PHASE_BOOTSTRAP};
    let mut body = 0;
    let mut boot = 0;
    for f in &model.functions {
        if read_function_module_phase(f) == Some(MODULE_PHASE_BOOTSTRAP) {
            boot += 1;
        } else {
            body += 1;
        }
    }
    (body, boot)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("─── BytesAndBrains v0.3.0 Federated Learning Pattern ───");
    println!();
    println!("This example expresses federated learning using ONLY");
    println!("the framework's generic role placeholders. There are");
    println!("no application-specific Model / DataLoader impls in");
    println!("this file — users drop in their own concrete impls");
    println!("at install time.");
    println!();

    // STEP 1 — build each Module composition.
    println!("─── STEP 1: Module composition → pre-compile ModelProto");
    let client_proto: ModelProto = ClientLogic.build()?;
    let server_logic_proto: ModelProto = ServerLogic.build()?;
    let server_reduce_proto: ModelProto = ServerReduce.build()?;
    let (client_body, client_boot) = phase_counts(&client_proto);
    let (server_logic_body, server_logic_boot) = phase_counts(&server_logic_proto);
    let (server_reduce_body, server_reduce_boot) = phase_counts(&server_reduce_proto);
    println!(
        "  ClientLogic   → {client_body} body function(s) + {client_boot} bootstrap function(s)",
    );
    println!(
        "  ServerLogic   → {server_logic_body} body function(s) + {server_logic_boot} bootstrap function(s)",
    );
    println!(
        "  ServerReduce  → {server_reduce_body} body function(s) + {server_reduce_boot} bootstrap function(s)",
    );
    println!();

    // STEP 2 — compile through the canonical bind chain.
    println!("─── STEP 2: Compiler::new().bind_*().compile()");

    let client_artifact = Compiler::new()
        .bind_protocol::<GlobalRegistryClient>("discovery")
        .bind_model::<StubModel>("model")
        .bind_data_source::<StubLoader>("data")
        .compile(client_proto)?;
    let server_logic_artifact = Compiler::new()
        .bind_peer_selector::<GlobalRegistryServer>("view")
        .bind_protocol::<GlobalRegistryServer>("discovery")
        .bind_model::<StubModel>("model")
        .compile(server_logic_proto)?;
    let server_reduce_artifact = Compiler::new()
        .bind_aggregator::<FedAvg<CpuBackend>>("aggregator")
        .bind_backend::<CpuBackend>("backend")
        .compile(server_reduce_proto)?;
    println!(
        "  ClientLogic    → {} function(s)",
        client_artifact.functions.len(),
    );
    println!(
        "  ServerLogic    → {} function(s)",
        server_logic_artifact.functions.len(),
    );
    println!(
        "  ServerReduce   → {} function(s)",
        server_reduce_artifact.functions.len(),
    );
    println!();

    // STEP 3 — surface what role-method ops each compiled model declares.
    println!("─── STEP 3: Role-method op surface per compiled model");
    for (label, compiled) in [
        ("ClientLogic", &client_artifact),
        ("ServerLogic", &server_logic_artifact),
        ("ServerReduce", &server_reduce_artifact),
    ] {
        let role_ops = count_role_ops(compiled, "ai.bytesandbrains.role.");
        if role_ops.is_empty() {
            continue;
        }
        println!("  {label} `{}`:", compiled.functions[0].name,);
        for (op, count) in role_ops {
            println!("    {op} × {count}");
        }
    }
    println!();

    // STEP 4 — install + drive a few poll cycles on each Node.
    println!("─── STEP 4: install() each artifact + drive a few poll cycles");

    let model = StubModel;
    let loader = StubLoader;
    let aggregator = FedAvg::<CpuBackend>::default();
    let backend = CpuBackend;
    let view = GlobalRegistryServer::new(0xC0FFEE);
    let discovery_server = GlobalRegistryServer::new(0xC0FFEE);
    let discovery_client = GlobalRegistryClient::new();
    let _server_peer_const = PeerId::from(SERVER_PEER);

    // With the chosen-path API the user supplies a `Config` bag
    // keyed by slot name. Every concrete in this example
    // (StubLoader, StubModel, GlobalRegistryServer, GlobalRegistryClient,
    // FedAvg) derives Default + has `type Config = ()`, so no
    // explicit `.with(...)` calls are needed — the framework
    // constructs each instance via the inventory `construct_fn`
    // with `&()`.
    let _ = (
        &discovery_client,
        &model,
        &loader,
        &view,
        &discovery_server,
        &aggregator,
        &backend,
    );
    let client_target = client_artifact.functions[0].name.clone();
    let server_logic_target = server_logic_artifact.functions[0].name.clone();
    let server_reduce_target = server_reduce_artifact.functions[0].name.clone();

    // Multi-interface bind. A real client picks an addr per peer
    // capability (LAN before WAN before a relay); the example stamps
    // three site-tagged variants so the AddressBook entry surfaces a
    // non-trivial bag. Site ids stand in for transport-distinct
    // endpoints (e.g. a TCP listener, a QUIC listener, an HTTP
    // surface); a production deployment swaps these for the real
    // transport's multiaddr shapes.
    let client_peer = PeerId::from(CLIENT_PEER);
    let server_peer = PeerId::from(SERVER_PEER);
    let client_addrs = vec![
        Address::empty().p2p(client_peer).site(1u64.into()),
        Address::empty().p2p(client_peer).site(2u64.into()),
        Address::empty().p2p(client_peer).site(3u64.into()),
    ];
    let server_addrs = vec![
        Address::empty().p2p(server_peer).site(11u64.into()),
        Address::empty().p2p(server_peer).site(12u64.into()),
    ];

    let mut client = install(
        client_peer,
        client_addrs,
        client_artifact,
        &[client_target.as_str()],
        Config::new(),
    )?;
    let mut server_logic = install(
        server_peer,
        server_addrs.clone(),
        server_logic_artifact,
        &[server_logic_target.as_str()],
        Config::new(),
    )?;
    let mut server_reduce = install(
        server_peer,
        server_addrs,
        server_reduce_artifact,
        &[server_reduce_target.as_str()],
        Config::new(),
    )?;
    println!("  Client Node installed at peer {CLIENT_PEER}");
    println!("  ServerLogic Node installed at peer {SERVER_PEER}");
    println!("  ServerReduce Node installed at peer {SERVER_PEER}");

    // F4 host-driven bootstrap: install records the bootstrap targets
    // but no longer arms the queue. Each Node calls
    // `run_bootstrap` to drive its install-order bootstraps to
    // completion before the body phase observes its first poll.
    let mut total = 0usize;
    for node in [&mut client, &mut server_logic, &mut server_reduce] {
        let boot_steps = node
            .run_bootstrap(BootstrapTarget::All)
            .expect("install-order kick");
        total += boot_steps.len();
    }
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    for node in [&mut client, &mut server_logic, &mut server_reduce] {
        for _ in 0..3 {
            match node.poll(&mut cx) {
                std::task::Poll::Ready(steps) => {
                    if steps.is_empty() {
                        break;
                    }
                    total += steps.len();
                }
                std::task::Poll::Pending => break,
            }
        }
    }
    println!("  {total} EngineStep(s) drained across the three Nodes");
    println!(
        "  (Steps include each Node::run_bootstrap firing its recorded\n   \
         setup ops; body ops then park Pending until network input lands.\n   \
         The example binds no transport — full execution coverage lives\n   \
         in tests/.)\n"
    );

    println!("─── STEP 5: Federation per-round flow (what the Nodes above run)");
    println!();
    println!("  1. ServerLogic fires → samples K peers → reads Model.params");
    println!("     → emits g.net_out(\"server_params\", peers, params) to broadcast.");
    println!("  2. Each sampled client receives → runs ClientLogic →");
    println!("     emits (updated_params, FedAvgMeta) via");
    println!("     g.net_out(\"updated_params\", server_peer, params) back to the server.");
    println!("  3. ServerReduce fires per inbound landing →");
    println!("     Aggregator.contribute(params, meta) folds into FedAvg.");
    println!("  4. Round-end trigger → Aggregator.aggregate(trigger) →");
    println!("     Model.load_parameters(new_global_params). Loop X rounds.");
    println!();
    println!("Demo exercised Module::build → Compiler::compile → install →");
    println!("Node::poll (poll loop drained 0 steps — no transport bound here;");
    println!("full execution coverage lives in tests/). Drop in your own Model /");
    println!("DataLoader concretes and a transport to run real training.");
    Ok(())
}
