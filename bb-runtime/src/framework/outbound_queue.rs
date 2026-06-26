//! `OutboundQueue` - FIFO of wire envelopes ready to ship.
//!
//! Used by Phase 8 of the engine's poll cycle: drained on each
//! cycle into `EngineStep::SendEnvelope` outputs. Carries an
//! optional FIFO-drop cap (`NodeConfig.max_outbound_queue`); when
//! a push would exceed the cap, the oldest envelope is evicted
//! and a counter increments. Phase 8 reads + resets the counter to
//! emit `EngineStep::OutboundDropped { count }`.

use crate::envelope::{WireEnvelope, ENVELOPE_SCHEMA_VERSION};
use std::collections::VecDeque;

/// FIFO of wire envelopes ready to ship.
#[derive(Default)]
pub struct OutboundQueue {
    queue: VecDeque<WireEnvelope>,
    cap: Option<usize>,
    dropped_since_last_drain: usize,
}

impl OutboundQueue {
    /// Construct a fresh outbound queue with no cap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a fresh outbound queue with the given FIFO-drop
    /// cap. `None` disables the cap.
    pub fn with_cap(cap: Option<usize>) -> Self {
        Self {
            queue: VecDeque::new(),
            cap,
            dropped_since_last_drain: 0,
        }
    }

    /// Set the FIFO-drop cap. `None` removes the cap.
    pub fn set_cap(&mut self, cap: Option<usize>) {
        self.cap = cap;
    }

    /// Push an envelope for shipment. If a cap is set and the
    /// queue is at the cap, FIFO-evict the oldest entry and bump
    /// the drop counter.
    ///
    /// Stamps `schema_version` here so the encode boundary can read
    /// `&WireEnvelope` and call `encode_to_vec` without cloning the
    /// payload to set the field defensively.
    pub fn push(&mut self, mut env: WireEnvelope) {
        if env.schema_version == 0 {
            env.schema_version = ENVELOPE_SCHEMA_VERSION;
        }
        if let Some(cap) = self.cap {
            while self.queue.len() >= cap {
                self.queue.pop_front();
                self.dropped_since_last_drain += 1;
            }
        }
        self.queue.push_back(env);
    }

    /// Drain all queued envelopes. Called by Phase 8 of the poll
    /// cycle.
    pub fn drain_all(&mut self) -> Vec<WireEnvelope> {
        self.queue.drain(..).collect()
    }

    /// Read + reset the count of FIFO-dropped envelopes since the
    /// last call. Returns 0 when no drops occurred.
    pub fn take_dropped_count(&mut self) -> usize {
        std::mem::take(&mut self.dropped_since_last_drain)
    }

    /// Number of queued envelopes.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Iterate queued envelopes for snapshot capture. The queue
    /// is not consumed - Phase 8 still drains on the next poll.
    /// .
    pub fn iter_for_snapshot(&self) -> impl Iterator<Item = &WireEnvelope> {
        self.queue.iter()
    }
}

