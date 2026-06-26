//! `ConstantView` — peer-selector concrete with a fixed peer list.
//!
//! Useful for tests + tiny fixed-size deployments where the peer
//! set is known at install time and never changes. Implements
//! [`bb_runtime::contracts::PeerSelector`] returning subsets of
//! the configured list per [`SelectParams`].

use bb_derive::{Concrete, PeerSelector};
use bb_runtime::completion::{CompletionHandle, ContractResponse};
use bb_runtime::contracts::peer_selector::SelectParams;
use bb_runtime::ids::PeerId;
use bb_runtime::runtime::RuntimeResourceRef;
use serde::{Deserialize, Serialize};

/// Fixed-list peer selector. Authors construct with the full
/// peer set at compile time; queries return slices.
#[derive(Clone, Debug, Default, Serialize, Deserialize, Concrete, PeerSelector)]
pub struct ConstantView {
    /// The fixed peer set, in insertion order.
    pub peers: Vec<PeerId>,
    /// Deterministic seed for Random selection.
    pub seed: u64,
}

impl ConstantView {
    /// Fresh view over `peers`. Random sampling uses a small
    /// xorshift seeded by `seed` so the same `seed` over the same
    /// peer set produces the same sequence — useful for tests.
    pub fn new(peers: Vec<PeerId>, seed: u64) -> Self {
        Self { peers, seed }
    }
}

/// Failure modes the `ConstantView` peer selector can surface to callers.
#[derive(Debug)]
pub enum ConstantViewError {
    /// The view has zero peers and `Random` / `NearKey` asked
    /// for at least one.
    Empty,
}

impl std::fmt::Display for ConstantViewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("ConstantView has no peers configured"),
        }
    }
}

impl std::error::Error for ConstantViewError {}

impl bb_runtime::contracts::PeerSelector for ConstantView {
    type Error = ConstantViewError;

    fn select(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        params: SelectParams,
        _completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error> {
        match params {
            SelectParams::All => ContractResponse::Now(Ok(self.peers.clone())),
            SelectParams::Random { n } => {
                if self.peers.is_empty() {
                    return ContractResponse::Now(Err(ConstantViewError::Empty));
                }
                let take = (n as usize).min(self.peers.len());
                let mut chosen: Vec<PeerId> = Vec::with_capacity(take);
                // Deterministic xorshift seeded by `self.seed`.
                let mut state = self.seed.wrapping_add(1);
                while chosen.len() < take {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let idx = (state as usize) % self.peers.len();
                    let pick = self.peers[idx];
                    if !chosen.contains(&pick) {
                        chosen.push(pick);
                    }
                }
                ContractResponse::Now(Ok(chosen))
            }
            SelectParams::NearKey { key: _, n } => {
                // ConstantView doesn't model a distance metric — it
                // just hands back the first `n` peers as a stable,
                // deterministic answer.
                if self.peers.is_empty() {
                    return ContractResponse::Now(Err(ConstantViewError::Empty));
                }
                let take = (n as usize).min(self.peers.len());
                ContractResponse::Now(Ok(self.peers[..take].to_vec()))
            }
        }
    }

    fn current_view(
        &mut self,
        _ctx: &mut RuntimeResourceRef<'_>,
        _completion: CompletionHandle<Vec<PeerId>, Self::Error>,
    ) -> ContractResponse<Vec<PeerId>, Self::Error> {
        ContractResponse::Now(Ok(self.peers.clone()))
    }
}

