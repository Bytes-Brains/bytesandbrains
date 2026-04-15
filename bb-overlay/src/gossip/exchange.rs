use std::collections::HashMap;
use std::fmt;

use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;

use bb_core::{Address, PeerId};

use super::peer::{AgePeer, GossipPeerType};

/// Trait for gossip view exchange strategies.
///
/// Controls how the view is prepared for sending (`prepare_tx`) and how
/// received peers are integrated into the local view (`integrate_rx`).
pub trait ViewExchange<P, A>: Clone + fmt::Debug
where
    A: Address,
    P: GossipPeerType<A>,
{
    /// Prepare the outgoing buffer from the local peer and current view.
    fn prepare_tx(
        &self,
        local: &P,
        peers: &mut Vec<P>,
        max_size: usize,
        rng: &mut ThreadRng,
    ) -> Vec<P>;

    /// Integrate received peers into the view. Returns (added, removed).
    fn integrate_rx(
        &self,
        peers: &mut Vec<P>,
        lut: &mut HashMap<PeerId, usize>,
        incoming: Vec<P>,
        local_peer_id: &PeerId,
        max_size: usize,
        rng: &mut ThreadRng,
    ) -> (Vec<P>, Vec<P>);

    /// Called after each exchange round (e.g., increment ages).
    fn on_round_complete(&self, peers: &mut Vec<P>);
}

/// Configuration for the randomized exchange strategy.
#[derive(Clone, Debug)]
pub struct RandomizedExchange {
    /// Number of oldest peers to remove per exchange.
    pub healing: usize,
    /// Number of head (newest) peers to remove per exchange.
    pub swap: usize,
}

impl Default for RandomizedExchange {
    fn default() -> Self {
        Self {
            healing: 5,
            swap: 5,
        }
    }
}

impl<A: Address> ViewExchange<AgePeer<A>, A> for RandomizedExchange {
    fn prepare_tx(
        &self,
        local: &AgePeer<A>,
        peers: &mut Vec<AgePeer<A>>,
        max_size: usize,
        rng: &mut ThreadRng,
    ) -> Vec<AgePeer<A>> {
        let mut buf = Vec::with_capacity(max_size / 2);
        buf.push(local.clone());

        // Shuffle peers
        peers.shuffle(rng);

        // Move oldest to back
        move_old_to_back(peers, self.healing);

        let num_get = (max_size / 2).saturating_sub(1);
        let take = num_get.min(peers.len());
        for i in 0..take {
            buf.push(peers[i].clone());
        }
        buf
    }

    fn integrate_rx(
        &self,
        peers: &mut Vec<AgePeer<A>>,
        lut: &mut HashMap<PeerId, usize>,
        incoming: Vec<AgePeer<A>>,
        local_peer_id: &PeerId,
        max_size: usize,
        rng: &mut ThreadRng,
    ) -> (Vec<AgePeer<A>>, Vec<AgePeer<A>>) {
        // Append new peers
        let added = append_many(peers, lut, incoming, local_peer_id);

        let mut removed = Vec::new();

        let excess = peers.len().saturating_sub(max_size);
        if excess == 0 {
            return (added, removed);
        }

        // Remove oldest
        let remove_old_n = self.healing.min(excess);
        removed.extend(remove_old(peers, lut, remove_old_n));

        let excess = peers.len().saturating_sub(max_size);
        if excess == 0 {
            return (added, removed);
        }

        // Remove from head
        let remove_head_n = self.swap.min(excess);
        removed.extend(remove_head(peers, lut, remove_head_n));

        let excess = peers.len().saturating_sub(max_size);
        if excess > 0 {
            removed.extend(remove_random(peers, lut, excess, rng));
        }

        (added, removed)
    }

    fn on_round_complete(&self, peers: &mut Vec<AgePeer<A>>) {
        for peer in peers.iter_mut() {
            peer.age += 1;
        }
    }
}

// --- Helper functions extracted from the old View implementation ---

fn append_many<A: Address>(
    peers: &mut Vec<AgePeer<A>>,
    lut: &mut HashMap<PeerId, usize>,
    incoming: Vec<AgePeer<A>>,
    local_peer_id: &PeerId,
) -> Vec<AgePeer<A>> {
    let mut added = Vec::new();
    for peer in incoming {
        let pid = peer.peer.peer_id;
        if pid == *local_peer_id {
            continue;
        }
        if let Some(&idx) = lut.get(&pid) {
            if peers[idx].age > peer.age {
                peers[idx].age = peer.age;
            }
        } else {
            let new_idx = peers.len();
            lut.insert(pid, new_idx);
            added.push(peer.clone());
            peers.push(peer);
        }
    }
    added
}

fn move_old_to_back<A: Address>(peers: &mut Vec<AgePeer<A>>, n: usize) {
    if n == 0 || peers.is_empty() {
        return;
    }
    let n = n.min(peers.len());

    let mut aged: Vec<(u32, usize)> = peers
        .iter()
        .enumerate()
        .map(|(i, p)| (p.age, i))
        .collect();
    aged.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut back = peers.len() - 1;
    let mut pos: Vec<usize> = (0..peers.len()).collect();
    let mut inv: Vec<usize> = (0..peers.len()).collect();

    for i in 0..n {
        let orig_idx = aged[i].1;
        let cur = pos[orig_idx];
        if cur != back {
            peers.swap(cur, back);
            let other_orig = inv[back];
            pos[other_orig] = cur;
            inv[cur] = other_orig;
            pos[orig_idx] = back;
            inv[back] = orig_idx;
        }
        back = back.saturating_sub(1);
    }
}

fn remove_old<A: Address>(
    peers: &mut Vec<AgePeer<A>>,
    lut: &mut HashMap<PeerId, usize>,
    n: usize,
) -> Vec<AgePeer<A>> {
    if n == 0 || peers.is_empty() {
        return Vec::new();
    }
    let n = n.min(peers.len());
    move_old_to_back(peers, n);
    let mut removed = Vec::with_capacity(n);
    for _ in 0..n {
        if let Some(gp) = peers.pop() {
            lut.remove(&gp.peer.peer_id);
            removed.push(gp);
        }
    }
    rebuild_lut(peers, lut);
    removed
}

fn remove_head<A: Address>(
    peers: &mut Vec<AgePeer<A>>,
    lut: &mut HashMap<PeerId, usize>,
    n: usize,
) -> Vec<AgePeer<A>> {
    if n == 0 || peers.is_empty() {
        return Vec::new();
    }
    let n = n.min(peers.len());
    for i in 0..n {
        lut.remove(&peers[i].peer.peer_id);
    }
    let removed: Vec<AgePeer<A>> = peers.drain(..n).collect();
    rebuild_lut(peers, lut);
    removed
}

fn remove_random<A: Address>(
    peers: &mut Vec<AgePeer<A>>,
    lut: &mut HashMap<PeerId, usize>,
    n: usize,
    rng: &mut ThreadRng,
) -> Vec<AgePeer<A>> {
    if n == 0 || peers.is_empty() {
        return Vec::new();
    }
    let n = n.min(peers.len());
    let mut removed = Vec::with_capacity(n);
    for _ in 0..n {
        let idx = rng.gen_range(0..peers.len());
        let gp = peers.remove(idx);
        lut.remove(&gp.peer.peer_id);
        removed.push(gp);
    }
    rebuild_lut(peers, lut);
    removed
}

fn rebuild_lut<A: Address>(peers: &[AgePeer<A>], lut: &mut HashMap<PeerId, usize>) {
    lut.clear();
    for (i, peer) in peers.iter().enumerate() {
        lut.insert(peer.peer.peer_id, i);
    }
}
