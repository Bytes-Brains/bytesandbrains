use std::time::Duration;

/// Push/Pull/PushPull gossip exchange mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GossipMode {
    Push,
    Pull,
    PushPull,
}

/// Protocol-level configuration for the gossip framework.
///
/// Strategy-specific config (healing, swap, etc.) lives in the exchange
/// and selector types, not here.
#[derive(Clone, Debug)]
pub struct GossipConfig {
    pub view_size: usize,
    pub mode: GossipMode,
    pub max_concurrent_requests: usize,
    pub retry_time: Duration,
    pub request_timeout: Duration,
    pub poll_interval: Duration,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            view_size: 10,
            mode: GossipMode::PushPull,
            max_concurrent_requests: 3,
            retry_time: Duration::from_secs(2),
            request_timeout: Duration::from_secs(5),
            poll_interval: Duration::from_secs(20),
        }
    }
}
