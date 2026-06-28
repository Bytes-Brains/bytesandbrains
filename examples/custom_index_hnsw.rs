//! Custom Index Contract backed by instant-distance + a worker thread.
//!
//! Demonstrates the canonical workflow plus async Contract dispatch:
//!
//!     Module  →  Compile  →  Install  →  Poll
//!
//! User-side authoring:
//!
//! 1. Define `HnswIndex` implementing `bb::contracts::Index`. The
//!    struct owns a channel to a worker thread that owns the actual
//!    HNSW data.
//! 2. Each Contract method sends work to the worker (passing the
//!    `CompletionHandle` along) and returns `ContractResponse::Later`.
//! 3. The worker processes each item off-thread and calls
//!    `completion.complete(result)` - pushing a Completion onto
//!    the engine's ingress queue from outside the engine's thread.
//! 4. `node.poll()` drains the ingress on its next cycle and
//!    unparks the suspended op with the result.
//!
//! Compile + install wires this all up; the Module body records
//! `index.search(...)` which fires once the host supplies the query
//! input via `IngressEvent::Invoke`.
//!
//! Run with:
//! ```text
//! cargo run --example custom_index_hnsw --features test-components
//! ```

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Waker};
use std::thread;

use serde::{Deserialize, Serialize};

use bytesandbrains::placeholders::IndexSlot;
use bytesandbrains::prelude::*;

use instant_distance::{Builder, HnswMap, Point, Search};

// ─── User-defined error type ─────────────────────────────────────

#[derive(Debug)]
enum HnswError {
    WorkerGone,
}

impl std::fmt::Display for HnswError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkerGone => write!(f, "HNSW worker thread has exited"),
        }
    }
}

impl std::error::Error for HnswError {}

// ─── HnswIndex: handle + lazily-spawned worker ───────────────────

const DIM: usize = 4;
const CAPACITY: u32 = 64;

/// User-facing struct. Holds a (lazily-initialized) sender to the
/// worker thread. The bincode round-trip in `install()` resets the
/// `tx` to `None`; `ensure_worker()` re-spawns on first use.
#[derive(Default, Clone, Serialize, Deserialize, Concrete, Index)]
struct HnswIndex {
    capacity: u32,
    #[serde(skip)]
    tx: Arc<Mutex<Option<mpsc::Sender<WorkItem>>>>,
}

enum WorkItem {
    Add {
        vec: Vec<f32>,
        completion: CompletionHandle<u64, HnswError>,
    },
    Search {
        query: Vec<f32>,
        k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, HnswError>,
    },
    Remove {
        completion: CompletionHandle<(), HnswError>,
    },
}

impl HnswIndex {
    fn new(capacity: u32) -> Self {
        let index = Self {
            capacity,
            tx: Arc::new(Mutex::new(None)),
        };
        index.ensure_worker();
        index
    }

    fn ensure_worker(&self) {
        let mut guard = self.tx.lock().expect("HnswIndex tx mutex poisoned");
        if guard.is_none() {
            let (tx, rx) = mpsc::channel::<WorkItem>();
            thread::spawn(move || hnsw_worker(rx));
            *guard = Some(tx);
        }
    }

    fn send(&self, item: WorkItem) {
        self.ensure_worker();
        let guard = self.tx.lock().expect("HnswIndex tx mutex poisoned");
        if let Some(tx) = guard.as_ref() {
            if let Err(mpsc::SendError(item)) = tx.send(item) {
                fail_item(item);
            }
        } else {
            fail_item(item);
        }
    }
}

fn fail_item(item: WorkItem) {
    match item {
        WorkItem::Add { completion, .. } => completion.complete(Err(HnswError::WorkerGone)),
        WorkItem::Search { completion, .. } => completion.complete(Err(HnswError::WorkerGone)),
        WorkItem::Remove { completion } => completion.complete(Err(HnswError::WorkerGone)),
    }
}

/// `DIM`-dimensional float vector. instant-distance's `Point`
/// trait describes the metric.
#[derive(Clone, Debug)]
struct FloatPoint(Vec<f32>);

impl Point for FloatPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

fn hnsw_worker(rx: mpsc::Receiver<WorkItem>) {
    let mut points: Vec<FloatPoint> = Vec::new();
    let mut ids: Vec<u64> = Vec::new();
    let mut hnsw: Option<HnswMap<FloatPoint, u64>> = None;
    let mut next_id: u64 = 1;

    while let Ok(item) = rx.recv() {
        match item {
            WorkItem::Add { vec, completion } => {
                if vec.len() != DIM {
                    completion.complete(Err(HnswError::WorkerGone));
                    continue;
                }
                let id = next_id;
                next_id += 1;
                points.push(FloatPoint(vec));
                ids.push(id);
                hnsw = None; // invalidate; rebuild on next search
                completion.complete(Ok(id));
            }
            WorkItem::Search {
                query,
                k,
                completion,
            } => {
                if query.len() != DIM {
                    completion.complete(Ok(Vec::new()));
                    continue;
                }
                if hnsw.is_none() {
                    hnsw = Some(Builder::default().build(points.clone(), ids.clone()));
                }
                let map = hnsw.as_ref().unwrap();
                let mut search = Search::default();
                let q = FloatPoint(query);
                let results: Vec<(u64, f32)> = map
                    .search(&q, &mut search)
                    .take(k as usize)
                    .map(|item| (*item.value, item.distance))
                    .collect();
                completion.complete(Ok(results));
            }
            WorkItem::Remove { completion } => {
                completion.complete(Ok(()));
            }
        }
    }
}

impl Index for HnswIndex {
    type Vector = [f32];
    type Error = HnswError;

    fn add(
        &mut self,
        _ctx: &mut bytesandbrains::runtime::RuntimeResourceRef<'_>,
        vec: &Self::Vector,
        completion: CompletionHandle<u64, Self::Error>,
    ) -> ContractResponse<u64, Self::Error> {
        self.send(WorkItem::Add {
            vec: vec.to_vec(),
            completion,
        });
        ContractResponse::Later
    }

    fn search(
        &self,
        _ctx: &mut bytesandbrains::runtime::RuntimeResourceRef<'_>,
        query: &Self::Vector,
        k: u32,
        completion: CompletionHandle<Vec<(u64, f32)>, Self::Error>,
    ) -> ContractResponse<Vec<(u64, f32)>, Self::Error> {
        self.send(WorkItem::Search {
            query: query.to_vec(),
            k,
            completion,
        });
        ContractResponse::Later
    }

    fn remove(
        &mut self,
        _ctx: &mut bytesandbrains::runtime::RuntimeResourceRef<'_>,
        _id: u64,
        completion: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        self.send(WorkItem::Remove { completion });
        ContractResponse::Later
    }
}

// ─── Module that exercises the Index ─────────────────────────────

struct HnswSearchApp {
    index: IndexSlot,
}

impl Module for HnswSearchApp {
    fn name(&self) -> &str {
        "HnswSearchApp"
    }
    fn body(&self, g: &mut Graph) {
        let query = g.input("query");
        let _ = self.index.search(g, query, 3);
    }
}

// ─── Main: canonical workflow ────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("─── Module → Compile");
    let app = HnswSearchApp { index: IndexSlot };
    let model = app.build()?;
    let compiled = Compiler::new()
        .bind_index::<HnswIndex>("hnsw")
        .compile(model)?;
    println!("  compiled: {} function(s)", compiled.functions.len(),);

    println!("\n─── Pre-seed the HNSW worker");
    let index = HnswIndex::new(CAPACITY);
    // The seed phase runs outside the engine, so it cannot synthesize
    // a `RuntimeResourceRef`. It speaks to the user-owned worker
    // directly via `send(WorkItem::Add { … })` — the Contract surface
    // is the engine-side API; the seed phase is host-side and owns the
    // same channel the Contract impl forwards onto.
    for i in 0..5 {
        let base = i as f32;
        let v = vec![base, base + 0.1, base + 0.2, base + 0.3];
        let (tx, rx) = mpsc::channel::<u64>();
        let sink = Arc::new(OneShotSink::<u64> {
            tx: Mutex::new(Some(tx)),
        });
        let handle = CompletionHandle::new(
            bytesandbrains::ids::CommandId::from(0u64),
            sink as Arc<dyn bytesandbrains::completion::CompletionSink>,
        );
        index.send(WorkItem::Add {
            vec: v,
            completion: handle,
        });
        let id = rx.recv()?;
        println!("  add(vec[{i}]) → id={id}");
    }

    println!("\n─── Install");
    let target = compiled.functions[0].name.clone();
    let mut node = install(
        PeerId::from(1u64),
        vec![Address::empty()],
        compiled,
        &[target.as_str()],
        // HnswIndex derives `bb::Concrete` with the default
        // `type Config = ()`, so install constructs a fresh
        // instance via `HnswIndex::default()`. The locally-built
        // `index` is only kept here for the dispatch demo below.
        Config::new(),
    )?;
    println!(
        "  Node installed. Slot hnsw = {:?}",
        node.slot("hnsw").is_some()
    );

    println!("\n─── Bootstrap: arm install-order queue");
    // Every Node must call `run_bootstrap` after `install` so the
    // engine arms its install-order bootstrap queue. With no Module
    // bootstrap overrides this drains immediately; the call is also
    // a no-op gate against accidental body-phase polling before the
    // engine has been kicked.
    let bootstrap_steps = node.run_bootstrap(&[])?;
    println!("  bootstrap drained {} step(s)", bootstrap_steps.len());

    println!("\n─── Drive: push query + poll");
    // Host pushes a query input via IngressEvent::Invoke; the
    // recorded Index.search op fires on the next poll cycle.
    let query: Vec<f32> = vec![2.0, 2.1, 2.2, 2.3];
    let query_bytes = bincode::serialize(&query)?;
    let _ = node.ingress_handle().push(IngressEvent::Invoke {
        module_name: artifact_module_name(&node),
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
    println!("  {total_steps} EngineStep(s) drained across the cycles");
    assert!(
        total_steps >= 1,
        "expected at least one EngineStep drained from the poll loop, got 0",
    );
    println!("\n✓ Async Index Contract installed + driven end-to-end.");
    Ok(())
}

fn artifact_module_name(node: &bytesandbrains::node::Node) -> String {
    // Read the only module the example installed.
    node.slots_iter()
        .next()
        .map(|(name, _)| name.to_string())
        .unwrap_or_else(|| "HnswSearchApp".to_string())
}

// ─── One-shot sink for the seed phase ────────────────────────────
// Used only to drive the pre-install training seed; once the Node
// is installed, completions route through the Node's ingress queue
// (the framework wires `ctx.open_completion()` to it inside the
// dispatch path - no scaffolding required).

struct OneShotSink<R> {
    tx: Mutex<Option<mpsc::Sender<R>>>,
}

impl<R> bytesandbrains::completion::CompletionSink for OneShotSink<R>
where
    R: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    fn complete(&self, _cmd_id: bytesandbrains::ids::CommandId, bytes: &[u8]) {
        if let Ok(value) = bincode::deserialize::<R>(bytes) {
            if let Some(tx) = self.tx.lock().unwrap().take() {
                let _ = tx.send(value);
            }
        }
    }
    fn fail(&self, _cmd_id: bytesandbrains::ids::CommandId, _detail: &str) {
        self.tx.lock().unwrap().take();
    }
}
