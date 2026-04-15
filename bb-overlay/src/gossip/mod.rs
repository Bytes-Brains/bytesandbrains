#[cfg(feature = "proto")]
pub(crate) mod proto;

pub mod config;
pub mod exchange;
pub mod peer;
pub mod selector;

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::time::Instant;

use rand::rngs::ThreadRng;

use bb_core::index::{NoopOpRef, OpId, OpRef};
use bb_core::pending_requests::{RequestId, RequestTracker};
use bb_core::{
    Address, OutMessage, OverlayProtocol, Peer, PeerId, PeerSampling, Step,
};

use config::{GossipConfig, GossipMode};
use exchange::ViewExchange;
use peer::GossipPeerType;
use selector::PeerSelector;

pub use peer::AgePeer;
pub use selector::{RandomizedSelector, RandomizedSelectorMode};
pub use exchange::RandomizedExchange;

/// Backwards-compatible alias for `AgePeer<A>`.
pub type GossipPeer<A> = AgePeer<A>;

/// The standard randomized gossip protocol.
pub type RandomizedGossip<A> = GossipSampling<A, AgePeer<A>, RandomizedSelector, RandomizedExchange>;

/// Handle wrapping an eagerly-computed peer selection result.
pub struct GossipSelectPeerRef<P: Clone> {
    result: Option<P>,
}

impl<P: Clone> OpRef for GossipSelectPeerRef<P> {
    type Info = ();
    type Stats = ();
    type Result = Option<P>;
    type Error = std::convert::Infallible;

    fn id(&self) -> &OpId {
        static ID: OpId = OpId(0);
        &ID
    }

    fn info(&self) -> Option<Self::Info> { Some(()) }
    fn stats(&self) -> Option<Self::Stats> { Some(()) }
    fn is_finished(&self) -> bool { true }

    fn finish(&mut self) -> Result<Self::Result, Self::Error> {
        Ok(self.result.take())
    }
}

/// Immutable reference to the gossip peer view.
pub struct GossipViewRef<'a, P> {
    peers: &'a [P],
}

impl<P> GossipViewRef<'_, P> {
    pub fn iter(&self) -> impl Iterator<Item = &P> {
        self.peers.iter()
    }

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn peers(&self) -> &[P] {
        self.peers
    }
}

/// Protocol messages exchanged between gossip peers.
#[derive(Clone, Debug)]
pub enum GossipMessage<P: Clone> {
    /// Outgoing gossip request. Mode determines behavior:
    /// - Push: view is populated, no response expected
    /// - Pull: view is empty, response expected
    /// - PushPull: view is populated, response expected
    Request {
        request_id: RequestId,
        mode: GossipMode,
        view: Vec<P>,
    },
    /// Response to a Pull or PushPull request.
    Response {
        request_id: RequestId,
        view: Vec<P>,
    },
}

/// Events emitted by the gossip protocol.
#[derive(Clone, Debug)]
pub enum GossipEvent<A: Address> {
    PeerAdded(Peer<A>),
    PeerRemoved(Peer<A>),
    RequestTimeout(PeerId),
}

/// The generic gossip overlay protocol.
///
/// Parameterized over:
/// - `A`: Address type
/// - `P`: Peer type (must implement `GossipPeerType<A>`)
/// - `S`: Peer selection strategy
/// - `X`: View exchange strategy
pub struct GossipSampling<A, P, S, X>
where
    A: Address,
    P: GossipPeerType<A>,
    S: PeerSelector<P, A>,
    X: ViewExchange<P, A>,
{
    local_peer: P,
    pub(crate) peers: Vec<P>,
    lut: HashMap<PeerId, usize>,
    config: GossipConfig,
    selector: S,
    exchange: X,
    pending: RequestTracker<Instant>,
    last_round_start: Option<Instant>,
    rng: ThreadRng,
    _phantom: PhantomData<A>,
}

impl<A, P, S, X> GossipSampling<A, P, S, X>
where
    A: Address,
    P: GossipPeerType<A>,
    S: PeerSelector<P, A>,
    X: ViewExchange<P, A>,
{
    pub fn new(address: A, config: GossipConfig, selector: S, exchange: X) -> Self {
        let local_peer = P::from_address(address);
        let pending = RequestTracker::new(config.request_timeout);
        let max_size = config.view_size;
        Self {
            local_peer,
            peers: Vec::with_capacity(max_size),
            lut: HashMap::with_capacity(max_size),
            config,
            selector,
            exchange,
            pending,
            last_round_start: None,
            rng: rand::thread_rng(),
            _phantom: PhantomData,
        }
    }

    /// Add an address to the local peer's underlying address book.
    pub fn add_local_address(&mut self, addr: A) {
        self.local_peer.peer_mut().addresses.seen(addr);
    }

    /// Number of peers in the view.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn contains(&self, peer_id: &PeerId) -> bool {
        self.lut.contains_key(peer_id)
    }

    fn select_peer_ref(&mut self, mode: &S::Mode) -> Option<&P> {
        let idx = self.selector.select(mode, &self.peers, &mut self.rng)?;
        Some(&self.peers[idx])
    }

    fn select_peer_excluding(&mut self, mode: &S::Mode, exclude: &HashSet<PeerId>) -> Option<&P> {
        let idx = self.selector.select_excluding(
            mode,
            &self.peers,
            exclude,
            &mut self.rng,
        )?;
        Some(&self.peers[idx])
    }

    fn build_tx(&mut self) -> Vec<P> {
        match self.config.mode {
            GossipMode::Pull => Vec::new(),
            GossipMode::Push | GossipMode::PushPull => {
                let max_size = self.config.view_size;
                self.exchange.prepare_tx(
                    &self.local_peer,
                    &mut self.peers,
                    max_size,
                    &mut self.rng,
                )
            }
        }
    }

    fn integrate_and_advance(&mut self, incoming: Vec<P>) -> (Vec<P>, Vec<P>) {
        let max_size = self.config.view_size;
        let local_id = self.local_peer.peer_id();
        let result = self.exchange.integrate_rx(
            &mut self.peers,
            &mut self.lut,
            incoming,
            &local_id,
            max_size,
            &mut self.rng,
        );
        self.exchange.on_round_complete(&mut self.peers);
        result
    }

    /// Push mode: fire-and-forget, no request tracking.
    fn poll_push(&mut self) -> Step<Self> {
        let now = Instant::now();
        let should_start_round = match self.last_round_start {
            None => true,
            Some(last) => now.duration_since(last) >= self.config.poll_interval,
        };
        if !should_start_round {
            return Step::new();
        }
        let default_mode = self.selector.default_mode();
        let target = match self.select_peer_ref(&default_mode) {
            Some(p) => p.peer().clone(),
            None => return Step::new(),
        };
        let tx = self.build_tx();
        self.exchange.on_round_complete(&mut self.peers);
        self.last_round_start = Some(now);
        let msg = GossipMessage::Request {
            request_id: RequestId::default(),
            mode: self.config.mode,
            view: tx,
        };
        Step::new().with_message(OutMessage {
            destination: target,
            message: msg,
        })
    }

    fn diff_events(added: Vec<P>, removed: Vec<P>) -> Vec<GossipEvent<A>> {
        let mut events = Vec::with_capacity(added.len() + removed.len());
        for p in added {
            events.push(GossipEvent::PeerAdded(p.peer().clone()));
        }
        for p in removed {
            events.push(GossipEvent::PeerRemoved(p.peer().clone()));
        }
        events
    }

    fn pending_peer_ids(&self) -> HashSet<PeerId> {
        self.pending.iter().map(|(key, _)| key.peer_id).collect()
    }

    fn send_request_excluding(
        &mut self,
        exclude: &HashSet<PeerId>,
        now: Instant,
    ) -> Option<OutMessage<Self>> {
        let default_mode = self.selector.default_mode();
        let target = self.select_peer_excluding(&default_mode, exclude)?.peer().clone();
        let tx = self.build_tx();
        let request_id = self.pending.insert(target.peer_id, now);
        let msg = GossipMessage::Request {
            request_id,
            mode: self.config.mode,
            view: tx,
        };
        Some(OutMessage {
            destination: target,
            message: msg,
        })
    }

    /// Insert a peer manually (for bootstrapping / testing).
    pub fn manual_insert(&mut self, peer: P) {
        let pid = peer.peer_id();
        let local_id = self.local_peer.peer_id();
        if pid == local_id {
            return;
        }
        if !self.lut.contains_key(&pid) {
            let idx = self.peers.len();
            self.lut.insert(pid, idx);
            self.peers.push(peer);
        }
    }

    /// Receive nodes directly into the view (for bootstrapping / testing).
    pub fn rx_nodes(&mut self, nodes: Vec<P>) -> (Vec<P>, Vec<P>) {
        let max_size = self.config.view_size;
        let local_id = self.local_peer.peer_id();
        self.exchange.integrate_rx(
            &mut self.peers,
            &mut self.lut,
            nodes,
            &local_id,
            max_size,
            &mut self.rng,
        )
    }

    fn remove_peer_by_id(&mut self, peer_id: &PeerId) -> Option<P> {
        let idx = *self.lut.get(peer_id)?;
        self.lut.remove(peer_id);
        let removed = self.peers.swap_remove(idx);
        if idx < self.peers.len() {
            self.lut.insert(self.peers[idx].peer_id(), idx);
        }
        Some(removed)
    }
}

impl<A, P, S, X> OverlayProtocol for GossipSampling<A, P, S, X>
where
    A: Address,
    P: GossipPeerType<A>,
    S: PeerSelector<P, A>,
    X: ViewExchange<P, A>,
{
    type Address = A;
    type Message = GossipMessage<P>;
    type Event = GossipEvent<A>;
    type Peer = P;
    type BootstrapConfig = ();
    type BootstrapRef<'a> = NoopOpRef where Self: 'a;

    const PROTOCOL_ID: &'static str = "/bb/gossip/1.0.0";

    fn poll(&mut self) -> Step<Self> {
        if self.config.mode == GossipMode::Push {
            return self.poll_push();
        }

        let now = Instant::now();
        let mut step = Step::new();

        let timed_out = self.pending.process_timeouts();
        for (key, _sent_at) in &timed_out {
            step.events.push(GossipEvent::RequestTimeout(key.peer_id));
        }
        if !timed_out.is_empty() {
            self.exchange.on_round_complete(&mut self.peers);
        }

        let retry_time = self.config.retry_time;
        let max_concurrent = self.config.max_concurrent_requests;
        let stale_count = self
            .pending
            .iter()
            .filter(|(_key, sent_at)| now.duration_since(**sent_at) >= retry_time)
            .count();

        for _ in 0..stale_count {
            if self.pending.len() >= max_concurrent {
                break;
            }
            let exclude = self.pending_peer_ids();
            if let Some(out_msg) = self.send_request_excluding(&exclude, now) {
                step.messages.push(out_msg);
            }
        }

        let should_start_round = match self.last_round_start {
            None => true,
            Some(last) => now.duration_since(last) >= self.config.poll_interval,
        };
        if should_start_round && self.pending.len() < max_concurrent {
            let exclude = self.pending_peer_ids();
            if let Some(out_msg) = self.send_request_excluding(&exclude, now) {
                step.messages.push(out_msg);
                self.last_round_start = Some(now);
            }
        }

        step
    }

    fn on_message(&mut self, from: Peer<A>, msg: GossipMessage<P>) -> Step<Self> {
        match msg {
            GossipMessage::Request {
                request_id: _,
                mode: GossipMode::Push,
                view,
            } => {
                let (added, removed) = self.integrate_and_advance(view);
                let mut step = Step::new();
                step.events = Self::diff_events(added, removed);
                step
            }
            GossipMessage::Request {
                request_id,
                mode: GossipMode::Pull,
                ..
            } => {
                let tx = self.build_tx();
                self.exchange.on_round_complete(&mut self.peers);
                Step::new().with_message(OutMessage {
                    destination: from,
                    message: GossipMessage::Response { request_id, view: tx },
                })
            }
            GossipMessage::Request {
                request_id,
                mode: GossipMode::PushPull,
                view,
            } => {
                let tx = self.build_tx();
                let (added, removed) = self.integrate_and_advance(view);
                let mut step = Step::new().with_message(OutMessage {
                    destination: from,
                    message: GossipMessage::Response { request_id, view: tx },
                });
                step.events = Self::diff_events(added, removed);
                step
            }
            GossipMessage::Response { request_id, view } => {
                let mut step = Step::new();
                if self.pending.remove(&from.peer_id, &request_id).is_some() {
                    let (added, removed) = self.integrate_and_advance(view);
                    step.events = Self::diff_events(added, removed);
                }
                step
            }
        }
    }

    fn local_peer_id(&self) -> &PeerId {
        &self.local_peer.peer().peer_id
    }

    fn bootstrap(&mut self, _config: Self::BootstrapConfig) -> Self::BootstrapRef<'_> {
        NoopOpRef::new(0)
    }

    fn on_connection_failed(&mut self, peer_id: &PeerId) -> Step<Self> {
        let mut step = Step::new();
        let removed = self.pending.remove_all_for_peer(peer_id);
        for (key, _) in removed {
            step.events.push(GossipEvent::RequestTimeout(key.peer_id));
        }
        step
    }

    fn add_peer(&mut self, peer: Self::Peer) -> Step<Self> {
        self.manual_insert(peer);
        Step::new()
    }

    fn remove_peer(&mut self, peer_id: &PeerId) -> Option<Self::Peer> {
        self.remove_peer_by_id(peer_id)
    }

    fn add_address(&mut self, peer_id: &PeerId, address: Self::Address) {
        if let Some(&idx) = self.lut.get(peer_id) {
            self.peers[idx].peer_mut().addresses.seen(address);
        }
    }

    fn remove_address(&mut self, peer_id: &PeerId, address: &Self::Address) -> Option<Self::Peer> {
        if let Some(&idx) = self.lut.get(peer_id) {
            self.peers[idx].peer_mut().addresses.remove(address);
            if self.peers[idx].peer().addresses.is_empty() {
                return self.remove_peer_by_id(peer_id);
            }
        }
        None
    }
}

impl<A, P, S, X> PeerSampling for GossipSampling<A, P, S, X>
where
    A: Address,
    P: GossipPeerType<A>,
    S: PeerSelector<P, A>,
    X: ViewExchange<P, A>,
{
    type Peer = P;

    type PeerView<'a> = GossipViewRef<'a, P>
    where Self: 'a;

    type SamplingMode = S::Mode;

    type SelectPeerRef<'a> = GossipSelectPeerRef<P>
    where Self: 'a;

    fn view(&self) -> Self::PeerView<'_> {
        GossipViewRef { peers: &self.peers }
    }

    fn view_len(&self) -> usize {
        self.peers.len()
    }

    fn select_peer(&mut self, mode: &Self::SamplingMode) -> Self::SelectPeerRef<'_> {
        GossipSelectPeerRef {
            result: self.select_peer_ref(mode).cloned(),
        }
    }

    fn broadcast(&self) -> Vec<Self::Peer> {
        self.peers.clone()
    }
}

/// Convenience constructor for `RandomizedGossip`.
impl<A: Address> RandomizedGossip<A> {
    /// Create a new randomized gossip protocol with default selector and exchange.
    pub fn new_randomized(address: A, config: GossipConfig) -> Self {
        Self::new(address, config, RandomizedSelector, RandomizedExchange::default())
    }

    /// Create with custom exchange parameters.
    pub fn new_with_exchange(address: A, config: GossipConfig, exchange: RandomizedExchange) -> Self {
        Self::new(address, config, RandomizedSelector, exchange)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::selector::RandomizedSelectorMode;
    use bb_core::address::AddressBook;

    fn make_protocol() -> RandomizedGossip<String> {
        RandomizedGossip::new_randomized("10.0.0.1:5000".to_string(), GossipConfig::default())
    }

    fn bootstrap_peers(proto: &mut RandomizedGossip<String>, n: usize) {
        let peers: Vec<AgePeer<String>> = (0..n)
            .map(|i| AgePeer::new(format!("10.0.0.{}:5000", i + 10), 0))
            .collect();
        proto.rx_nodes(peers);
    }

    #[test]
    fn poll_empty_view_returns_empty_step() {
        let mut proto = make_protocol();
        let step = proto.poll();
        assert!(step.is_empty());
    }

    #[test]
    fn poll_with_peers_produces_message() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 10);
        let step = proto.poll();
        assert_eq!(step.messages.len(), 1);
        match &step.messages[0].message {
            GossipMessage::Request { request_id: _, mode, view } => {
                assert_eq!(*mode, GossipMode::PushPull);
                assert!(!view.is_empty());
            }
            _ => panic!("Expected Request message"),
        }
    }

    #[test]
    fn on_message_push_no_response() {
        let mut proto = make_protocol();
        let from = Peer::new(
            PeerId::from_data("sender"),
            AddressBook::new("10.0.0.99:5000".to_string(), 5),
        );
        let incoming = vec![AgePeer::new("10.0.0.50:5000".to_string(), 0)];
        let step = proto.on_message(
            from,
            GossipMessage::Request {
                request_id: RequestId::default(),
                mode: GossipMode::Push,
                view: incoming,
            },
        );
        assert!(step.messages.is_empty());
        assert!(proto.len() > 0);
    }

    #[test]
    fn on_message_pull_sends_response() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 10);
        let from = Peer::new(
            PeerId::from_data("puller"),
            AddressBook::new("10.0.0.99:5000".to_string(), 5),
        );
        let req_id = RequestId::new(42);
        let step = proto.on_message(
            from.clone(),
            GossipMessage::Request {
                request_id: req_id,
                mode: GossipMode::Pull,
                view: vec![],
            },
        );
        assert_eq!(step.messages.len(), 1);
        assert_eq!(step.messages[0].destination, from);
        match &step.messages[0].message {
            GossipMessage::Response { request_id, view } => {
                assert_eq!(*request_id, req_id);
                assert!(!view.is_empty());
            }
            _ => panic!("Expected Response message"),
        }
    }

    #[test]
    fn on_message_pushpull_sends_response_and_ingests() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 5);
        let from = Peer::new(
            PeerId::from_data("partner"),
            AddressBook::new("10.0.0.99:5000".to_string(), 5),
        );
        let req_id = RequestId::new(99);
        let incoming = vec![AgePeer::new("10.0.0.200:5000".to_string(), 0)];
        let step = proto.on_message(
            from,
            GossipMessage::Request {
                request_id: req_id,
                mode: GossipMode::PushPull,
                view: incoming,
            },
        );
        assert_eq!(step.messages.len(), 1);
        match &step.messages[0].message {
            GossipMessage::Response { request_id, view } => {
                assert_eq!(*request_id, req_id);
                assert!(!view.is_empty());
            }
            _ => panic!("Expected Response message"),
        }
    }

    #[test]
    fn on_message_response_ingests_view() {
        let mut proto = make_protocol();
        let from = Peer::new(
            PeerId::from_data("responder"),
            AddressBook::new("10.0.0.99:5000".to_string(), 5),
        );
        let request_id = proto.pending.insert(from.peer_id, Instant::now());
        let incoming: Vec<AgePeer<String>> = (0..5)
            .map(|i| AgePeer::new(format!("10.0.0.{}:5000", i + 100), 0))
            .collect();
        let step = proto.on_message(from, GossipMessage::Response { request_id, view: incoming });
        assert!(step.messages.is_empty());
        assert!(proto.len() > 0);
    }

    #[test]
    fn protocol_id() {
        assert_eq!(RandomizedGossip::<String>::PROTOCOL_ID, "/bb/gossip/1.0.0");
    }

    #[test]
    fn local_peer_id_matches() {
        let proto = make_protocol();
        let expected_id = PeerId::from_data(&"10.0.0.1:5000".to_string());
        assert_eq!(*proto.local_peer_id(), expected_id);
    }

    // --- PeerSampling tests ---

    #[test]
    fn select_peer_on_empty_view() {
        let mut proto = make_protocol();
        assert!(proto.select_peer(&RandomizedSelectorMode::Tail).finish().unwrap().is_none());
    }

    #[test]
    fn select_peer_returns_valid_peer() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 10);
        assert!(proto.select_peer(&RandomizedSelectorMode::Tail).finish().unwrap().is_some());
    }

    #[test]
    fn view_iter_via_peer_sampling() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 5);
        assert_eq!(proto.view_len(), 5);
        let view_peers = proto.view();
        assert_eq!(view_peers.len(), 5);
    }

    // --- Request tracking tests ---

    fn make_protocol_with_config(config: GossipConfig) -> RandomizedGossip<String> {
        RandomizedGossip::new_randomized("10.0.0.1:5000".to_string(), config)
    }

    #[test]
    fn test_poll_push_mode_no_tracking() {
        let mut config = GossipConfig::default();
        config.mode = GossipMode::Push;
        let mut proto = make_protocol_with_config(config);
        bootstrap_peers(&mut proto, 10);
        let step = proto.poll();
        assert_eq!(step.messages.len(), 1);
        assert!(proto.pending.is_empty(), "Push mode should not track requests");
    }

    #[test]
    fn test_response_from_tracked_peer_accepted() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 10);

        let step = proto.poll();
        assert_eq!(step.messages.len(), 1);
        let target = step.messages[0].destination.clone();
        let request_id = match &step.messages[0].message {
            GossipMessage::Request { request_id, .. } => *request_id,
            _ => panic!("Expected Request"),
        };
        assert!(proto.pending.is_pending(&target.peer_id, &request_id));

        let view_before = proto.len();
        let response_view: Vec<AgePeer<String>> = (0..3)
            .map(|i| AgePeer::new(format!("10.0.1.{}:5000", i), 0))
            .collect();
        proto.on_message(target.clone(), GossipMessage::Response { request_id, view: response_view });

        assert!(!proto.pending.is_pending(&target.peer_id, &request_id));
        assert!(proto.len() >= view_before);
    }

    #[test]
    fn test_response_from_untracked_peer_discarded() {
        let mut proto = make_protocol();
        bootstrap_peers(&mut proto, 10);

        let untracked = Peer::new(
            PeerId::from_data("untracked-sender"),
            AddressBook::new("10.0.0.222:5000".to_string(), 5),
        );
        let view_before = proto.len();
        let response_view: Vec<AgePeer<String>> = (0..5)
            .map(|i| AgePeer::new(format!("10.0.2.{}:5000", i), 0))
            .collect();
        proto.on_message(untracked, GossipMessage::Response { request_id: RequestId::new(999), view: response_view });

        assert_eq!(proto.len(), view_before);
    }

    #[test]
    fn test_max_concurrent_requests_respected() {
        let mut config = GossipConfig::default();
        config.max_concurrent_requests = 2;
        config.poll_interval = std::time::Duration::ZERO;
        let mut proto = make_protocol_with_config(config);
        bootstrap_peers(&mut proto, 10);

        let step1 = proto.poll();
        assert_eq!(step1.messages.len(), 1);
        assert_eq!(proto.pending.len(), 1);

        let step2 = proto.poll();
        assert_eq!(step2.messages.len(), 1);
        assert_eq!(proto.pending.len(), 2);

        let step3 = proto.poll();
        assert_eq!(step3.messages.len(), 0);
        assert_eq!(proto.pending.len(), 2);
    }

    #[test]
    fn test_timeout_emits_event() {
        use std::time::Duration;

        let mut config = GossipConfig::default();
        config.request_timeout = Duration::from_millis(50);
        let mut proto = make_protocol_with_config(config);
        bootstrap_peers(&mut proto, 10);

        let step = proto.poll();
        assert_eq!(step.messages.len(), 1);
        let target_id = step.messages[0].destination.peer_id;

        std::thread::sleep(Duration::from_millis(60));

        let step = proto.poll();
        let timeout_events: Vec<_> = step
            .events
            .iter()
            .filter(|e| matches!(e, GossipEvent::RequestTimeout(_)))
            .collect();
        assert_eq!(timeout_events.len(), 1);
        match &timeout_events[0] {
            GossipEvent::RequestTimeout(pid) => assert_eq!(*pid, target_id),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_retry_sends_to_different_peer() {
        use std::time::Duration;

        let mut config = GossipConfig::default();
        config.retry_time = Duration::from_millis(50);
        config.request_timeout = Duration::from_secs(10);
        config.max_concurrent_requests = 3;
        let mut proto = make_protocol_with_config(config);
        bootstrap_peers(&mut proto, 10);

        let step = proto.poll();
        assert_eq!(step.messages.len(), 1);
        let original_target = step.messages[0].destination.peer_id;

        std::thread::sleep(Duration::from_millis(60));

        let step = proto.poll();
        assert!(!step.messages.is_empty());
        for msg in &step.messages {
            assert_ne!(
                msg.destination.peer_id, original_target,
                "Retry should go to a different peer"
            );
        }
    }

    #[test]
    fn test_late_response_after_timeout_discarded() {
        use std::time::Duration;

        let mut config = GossipConfig::default();
        config.request_timeout = Duration::from_millis(50);
        let mut proto = make_protocol_with_config(config);
        bootstrap_peers(&mut proto, 10);

        let step = proto.poll();
        let target = step.messages[0].destination.clone();
        let request_id = match &step.messages[0].message {
            GossipMessage::Request { request_id, .. } => *request_id,
            _ => panic!("Expected Request"),
        };

        std::thread::sleep(Duration::from_millis(60));
        let _step = proto.poll();

        let view_before = proto.len();
        let response_view: Vec<AgePeer<String>> = (0..5)
            .map(|i| AgePeer::new(format!("10.0.3.{}:5000", i), 0))
            .collect();
        proto.on_message(target, GossipMessage::Response { request_id, view: response_view });

        assert_eq!(proto.len(), view_before);
    }

    #[test]
    fn test_poll_interval_gates_new_rounds() {
        use std::time::Duration;

        let mut config = GossipConfig::default();
        config.poll_interval = Duration::from_millis(200);
        config.request_timeout = Duration::from_secs(10);
        config.max_concurrent_requests = 10;
        let mut proto = make_protocol_with_config(config);
        bootstrap_peers(&mut proto, 10);

        let step1 = proto.poll();
        assert_eq!(step1.messages.len(), 1);

        let step2 = proto.poll();
        assert_eq!(step2.messages.len(), 0);
    }

    #[test]
    fn test_select_peer_excluding() {
        let mut proto = make_protocol();
        let peers: Vec<AgePeer<String>> = (0..5)
            .map(|i| AgePeer::new(format!("10.0.0.{}:5000", i + 10), i as u32))
            .collect();
        let peer_ids: Vec<PeerId> = peers.iter().map(|p| p.peer.peer_id).collect();
        proto.rx_nodes(peers);

        // Exclude the oldest peer (age=4), tail selector should pick next oldest.
        let mut exclude = HashSet::new();
        exclude.insert(peer_ids[4]);
        let default_mode = RandomizedSelectorMode::Tail;
        let selected = proto.select_peer_excluding(&default_mode, &exclude).unwrap();
        assert_ne!(selected.peer_id(), peer_ids[4]);
    }

    #[test]
    fn test_all_peers_excluded_returns_none() {
        let mut proto = make_protocol();
        let peers: Vec<AgePeer<String>> = (0..3)
            .map(|i| AgePeer::new(format!("10.0.0.{}:5000", i + 10), 0))
            .collect();
        let all_ids: HashSet<PeerId> = peers.iter().map(|p| p.peer.peer_id).collect();
        proto.rx_nodes(peers);

        let default_mode = RandomizedSelectorMode::Tail;
        assert!(proto.select_peer_excluding(&default_mode, &all_ids).is_none());
    }

    #[test]
    fn test_full_cycle_with_tracking() {
        let config = GossipConfig::default();
        let mut node_a = RandomizedGossip::new_randomized("10.0.0.1:5000".to_string(), config.clone());
        let mut node_b = RandomizedGossip::new_randomized("10.0.0.2:5000".to_string(), config);

        let peers_a: Vec<AgePeer<String>> = (0..5)
            .map(|i| AgePeer::new(format!("10.0.1.{}:5000", i), 0))
            .collect();
        let peers_b: Vec<AgePeer<String>> = (0..5)
            .map(|i| AgePeer::new(format!("10.0.2.{}:5000", i), 0))
            .collect();
        node_a.rx_nodes(peers_a);
        node_a.manual_insert(AgePeer::new("10.0.0.2:5000".to_string(), 0));
        node_b.rx_nodes(peers_b);
        node_b.manual_insert(AgePeer::new("10.0.0.1:5000".to_string(), 0));

        let step_a = node_a.poll();
        assert!(!step_a.messages.is_empty());
        let msg = &step_a.messages[0];
        let target_peer = msg.destination.clone();
        let request_id = match &msg.message {
            GossipMessage::Request { request_id, .. } => *request_id,
            _ => panic!("Expected Request"),
        };

        assert!(node_a.pending.is_pending(&target_peer.peer_id, &request_id));

        let step_b = node_b.on_message(
            Peer::new(
                PeerId::from_data(&"10.0.0.1:5000".to_string()),
                AddressBook::new("10.0.0.1:5000".to_string(), 5),
            ),
            msg.message.clone(),
        );
        assert_eq!(step_b.messages.len(), 1);

        match &step_b.messages[0].message {
            GossipMessage::Response { request_id: resp_id, .. } => {
                assert_eq!(*resp_id, request_id);
            }
            _ => panic!("Expected Response"),
        }

        let step_a2 = node_a.on_message(target_peer.clone(), step_b.messages[0].message.clone());
        assert!(step_a2.messages.is_empty());
        assert!(node_a.pending.is_empty());
    }
}
