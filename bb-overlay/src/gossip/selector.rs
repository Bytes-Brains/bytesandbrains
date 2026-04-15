use std::collections::HashSet;
use std::fmt;

use rand::rngs::ThreadRng;
use rand::Rng;

use bb_core::PeerId;
use bb_core::Address;

use super::peer::{AgePeer, GossipPeerType};

/// Trait for peer selection strategies in the gossip framework.
///
/// The `Mode` associated type is exposed as `PeerSampling::SamplingMode`,
/// allowing callers to choose a selection strategy at runtime.
pub trait PeerSelector<P, A>: Clone + fmt::Debug
where
    A: Address,
    P: GossipPeerType<A>,
{
    /// The mode type callers pass to `select_peer()` at runtime.
    type Mode: Clone + fmt::Debug;

    /// Select a peer index from the slice using the given mode.
    fn select(&self, mode: &Self::Mode, peers: &[P], rng: &mut ThreadRng) -> Option<usize>;

    /// Select a peer index, excluding peers whose ID is in `exclude`.
    fn select_excluding(
        &self,
        mode: &Self::Mode,
        peers: &[P],
        exclude: &HashSet<PeerId>,
        rng: &mut ThreadRng,
    ) -> Option<usize>;

    /// The default mode used by the gossip loop for internal selection.
    fn default_mode(&self) -> Self::Mode;
}

/// Selection modes for the randomized gossip protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomizedSelectorMode {
    /// Select the peer with the highest age (stalest entry).
    Tail,
    /// Select a peer uniformly at random.
    UniformRandom,
}

/// Peer selector for randomized gossip.
///
/// Supports `Tail` (select oldest by age) and `UniformRandom` selection.
#[derive(Clone, Debug)]
pub struct RandomizedSelector;

impl<A: Address> PeerSelector<AgePeer<A>, A> for RandomizedSelector {
    type Mode = RandomizedSelectorMode;

    fn select(
        &self,
        mode: &Self::Mode,
        peers: &[AgePeer<A>],
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        if peers.is_empty() {
            return None;
        }
        match mode {
            RandomizedSelectorMode::Tail => {
                peers
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, p)| p.age)
                    .map(|(i, _)| i)
            }
            RandomizedSelectorMode::UniformRandom => Some(rng.gen_range(0..peers.len())),
        }
    }

    fn select_excluding(
        &self,
        mode: &Self::Mode,
        peers: &[AgePeer<A>],
        exclude: &HashSet<PeerId>,
        rng: &mut ThreadRng,
    ) -> Option<usize> {
        if peers.is_empty() {
            return None;
        }
        match mode {
            RandomizedSelectorMode::Tail => {
                peers
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| !exclude.contains(&p.peer_id()))
                    .max_by_key(|(_, p)| p.age)
                    .map(|(i, _)| i)
            }
            RandomizedSelectorMode::UniformRandom => {
                let eligible: Vec<usize> = (0..peers.len())
                    .filter(|&i| !exclude.contains(&peers[i].peer_id()))
                    .collect();
                if eligible.is_empty() {
                    return None;
                }
                Some(eligible[rng.gen_range(0..eligible.len())])
            }
        }
    }

    fn default_mode(&self) -> Self::Mode {
        RandomizedSelectorMode::Tail
    }
}
