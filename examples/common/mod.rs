//! Shared example infrastructure.
//!
//! Minimal `ContractResponse::Now(Ok(...))` impls of the
//! framework's role traits. Each is a *demonstration stub*: just
//! enough to make `Compiler::new().bind_*().compile()` +
//! `install()` + `node.poll()` produce a clean run for a Module
//! body that records the corresponding role-method ops.
//!
//! Use these to focus on the workflow shape without rebuilding
//! real ML code in every example.

#![allow(dead_code)]

use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use serde::{Deserialize, Serialize};

use bytesandbrains::bus::OpError;
use bytesandbrains::completion::{CompletionHandle, ContractResponse};
use bytesandbrains::contracts::{
    Aggregator as AggregatorContract, DataSource as DataSourceContract, Model as ModelContract,
};
use bytesandbrains::engine::EngineStep;
use bytesandbrains::ingress::{IngressQueue, IngressQueueRef};
use bytesandbrains::runtime::RuntimeResourceRef;
use bytesandbrains::{
    Address, Aggregator, Concrete, DataSource, IngressEvent, Model, Node, PeerId,
};
use std::collections::HashMap;

// ─── Stub DataSource ───────────────────────────────────────────────

#[derive(Clone, Debug, Default, Serialize, Deserialize, Concrete, DataSource)]
pub struct StubLoader;

impl DataSourceContract for StubLoader {
    type Sample = [f32];
    type Error = OpError;
    fn next_batch(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<(Box<Self::Sample>, Box<Self::Sample>), Self::Error>,
    ) -> ContractResponse<(Box<Self::Sample>, Box<Self::Sample>), Self::Error> {
        ContractResponse::Now(Ok((
            vec![1.0_f32, 2.0, 3.0].into_boxed_slice(),
            Vec::<f32>::new().into_boxed_slice(),
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

// ─── Stub Model ────────────────────────────────────────────────────

#[derive(Clone, Debug, Default, Serialize, Deserialize, Concrete, Model)]
pub struct StubModel;

impl ModelContract for StubModel {
    type Tensor = [f32];
    type Error = OpError;

    fn forward(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        input: &Self::Tensor,
        _c: CompletionHandle<Box<Self::Tensor>, Self::Error>,
    ) -> ContractResponse<Box<Self::Tensor>, Self::Error> {
        ContractResponse::Now(Ok(input.to_vec().into_boxed_slice()))
    }

    fn load_parameters(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _p: &Self::Tensor,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }

    fn backward(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _g: &Self::Tensor,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }

    fn apply_delta(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _d: &Self::Tensor,
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }

    fn compute_loss(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _i: &Self::Tensor,
        _t: &Self::Tensor,
        _c: CompletionHandle<f32, Self::Error>,
    ) -> ContractResponse<f32, Self::Error> {
        ContractResponse::Now(Ok(0.0))
    }

    fn params(
        &self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<Box<Self::Tensor>, Self::Error>,
    ) -> ContractResponse<Box<Self::Tensor>, Self::Error> {
        ContractResponse::Now(Ok(Vec::<f32>::new().into_boxed_slice()))
    }
}

// ─── Stub Aggregator ───────────────────────────────────────────────

#[derive(Clone, Debug, Default, Serialize, Deserialize, Concrete, Aggregator)]
pub struct StubAggregator;

impl AggregatorContract for StubAggregator {
    type Element = [f32];
    type Error = OpError;
    type Metadata = ();
    fn contribute(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _src: PeerId,
        _tensor: &Self::Element,
        _metadata: (),
        _c: CompletionHandle<(), Self::Error>,
    ) -> ContractResponse<(), Self::Error> {
        ContractResponse::Now(Ok(()))
    }
    fn aggregate(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _c: CompletionHandle<(Box<Self::Element>, ()), Self::Error>,
    ) -> ContractResponse<(Box<Self::Element>, ()), Self::Error> {
        ContractResponse::Now(Ok((Vec::<f32>::new().into_boxed_slice(), ())))
    }
}

// ─── Tiny in-process bus + poll driver ─────────────────────────────

pub struct Bus {
    routes: HashMap<PeerId, Arc<IngressQueue>>,
}

impl Bus {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
        }
    }

    pub fn connect(&mut self, peer: PeerId, ingress: IngressQueueRef) {
        self.routes.insert(peer, ingress.arc().clone());
    }

    /// Forward every WireEnvelopeReady step's envelope to the
    /// matching destination peer's ingress queue. Returns how many
    /// envelopes were enqueued.
    pub fn forward(&self, src_peer: PeerId, steps: &[EngineStep]) -> usize {
        let mut forwarded = 0;
        for step in steps {
            if let EngineStep::SendEnvelope(env) = step {
                let peer = env
                    .dest_peer_addresses
                    .first()
                    .and_then(|bytes| Address::from_bytes(bytes).ok())
                    .and_then(|addr| addr.peer_id());
                if let Some(peer) = peer {
                    if let Some(ingress) = self.routes.get(&peer) {
                        let _ = ingress.push(IngressEvent::from_in_process(src_peer, env.clone()));
                        forwarded += 1;
                    }
                }
            }
        }
        forwarded
    }
}

/// Drive `node.poll()` up to `max_cycles` times, returning the
/// total step count drained across all cycles.
pub fn drive_poll(node: &mut Node, max_cycles: usize) -> usize {
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut total = 0;
    for _ in 0..max_cycles {
        match node.poll(&mut cx) {
            Poll::Ready(steps) => {
                if steps.is_empty() {
                    break;
                }
                total += steps.len();
            }
            Poll::Pending => break,
        }
    }
    total
}
