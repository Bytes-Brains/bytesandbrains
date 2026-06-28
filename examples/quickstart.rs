//! Smallest end-to-end shape: define two no-op Concretes, record a
//! one-line `Module::body`, compile, install, drive `Node::poll` to
//! quiescence. No bootstrap override, no wire boundary, no transport.
//!
//! Drop in your own `Model` + `DataSource` impls to graduate to a real
//! workload; the rest of the lifecycle stays the same.
//!
//! Run with:
//! ```text
//! cargo run --example quickstart --features test-components
//! ```

#![cfg(feature = "test-components")]

use std::task::{Context, Poll, Waker};

use bytesandbrains::bus::OpError;
use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::contracts::{DataSource as DataSourceContract, Model as ModelContract};
use bytesandbrains::placeholders::{DataLoaderSlot, ModelSlot};
use bytesandbrains::runtime::RuntimeResourceRef;
use bytesandbrains::{install, Address, Compiler, Config, Graph, Module, PeerId};

// ─── Concretes ────────────────────────────────────────────────────

type Ctx<'a> = RuntimeResourceRef<'a>;
type Done<T> = CompletionHandle<T, OpError>;
type Resp<T> = ContractResponse<T, OpError>;

#[derive(
    Clone,
    Default,
    serde::Serialize,
    serde::Deserialize,
    bytesandbrains::Concrete,
    bytesandbrains::Model,
)]
struct NoopModel;

impl ModelContract for NoopModel {
    type Tensor = [f32];
    type Error = OpError;
    fn forward(&mut self, _: &mut Ctx<'_>, x: &[f32], _: Done<Box<[f32]>>) -> Resp<Box<[f32]>> {
        ContractResponse::Now(Ok(x.to_vec().into_boxed_slice()))
    }
    fn load_parameters(&mut self, _: &mut Ctx<'_>, _: &[f32], _: Done<()>) -> Resp<()> {
        ContractResponse::Now(Ok(()))
    }
    fn backward(&mut self, _: &mut Ctx<'_>, _: &[f32], _: Done<()>) -> Resp<()> {
        ContractResponse::Now(Ok(()))
    }
    fn apply_delta(&mut self, _: &mut Ctx<'_>, _: &[f32], _: Done<()>) -> Resp<()> {
        ContractResponse::Now(Ok(()))
    }
    fn compute_loss(&mut self, _: &mut Ctx<'_>, _: &[f32], _: &[f32], _: Done<f32>) -> Resp<f32> {
        ContractResponse::Now(Ok(0.0))
    }
    fn params(&self, _: &mut Ctx<'_>, _: Done<Box<[f32]>>) -> Resp<Box<[f32]>> {
        ContractResponse::Now(Ok(Vec::<f32>::new().into_boxed_slice()))
    }
}

#[derive(
    Clone,
    Default,
    serde::Serialize,
    serde::Deserialize,
    bytesandbrains::Concrete,
    bytesandbrains::DataSource,
)]
struct NoopDataLoader;

impl DataSourceContract for NoopDataLoader {
    type Sample = [f32];
    type Error = OpError;
    fn next_batch(
        &mut self,
        _: &mut Ctx<'_>,
        _: Done<(Box<[f32]>, Box<[f32]>)>,
    ) -> Resp<(Box<[f32]>, Box<[f32]>)> {
        ContractResponse::Now(Ok((
            vec![1.0].into_boxed_slice(),
            vec![1.0].into_boxed_slice(),
        )))
    }
    fn reset(&mut self, _: &mut Ctx<'_>, _: Done<()>) -> Resp<()> {
        ContractResponse::Now(Ok(()))
    }
    fn on_data_loaded(&mut self, _: &mut Ctx<'_>, _: Done<()>) -> Resp<()> {
        ContractResponse::Now(Ok(()))
    }
}

// ─── Module ───────────────────────────────────────────────────────

struct Quickstart;
impl Module for Quickstart {
    fn name(&self) -> &str {
        "Quickstart"
    }
    fn body(&self, g: &mut Graph) {
        let (batch, _labels) = DataLoaderSlot.next_batch(g);
        let prediction = ModelSlot.forward(g, batch);
        g.output("prediction", prediction);
    }
}

// ─── Main ─────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    bytesandbrains::ops::link_force();

    let compiled = Compiler::new()
        .bind_data_source::<NoopDataLoader>("data_source")
        .bind_model::<NoopModel>("model")
        .compile(Quickstart.build()?)?;

    let peer = PeerId::from(0u64);
    let mut node = install(
        peer,
        vec![Address::empty().p2p(peer)],
        compiled,
        &["Quickstart"],
        Config::new(),
    )?;
    node.run_bootstrap(&[]).expect("install-order bootstrap");

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut total = 0usize;
    while let Poll::Ready(steps) = node.poll(&mut cx) {
        if steps.is_empty() {
            break;
        }
        total += steps.len();
    }
    println!("Quickstart drained {total} EngineStep(s).");
    Ok(())
}
