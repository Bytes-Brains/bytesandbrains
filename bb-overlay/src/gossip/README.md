# Gossip Peer Sampling

A gossip-based peer sampling service for maintaining self-healing overlay networks. Each node keeps a bounded, randomized view of peers and periodically exchanges subsets of that view with a selected partner, converging toward a uniform random sample of the network.

## Key Features

- **Exchange modes** — Push, Pull, or PushPull. PushPull (default) provides the fastest convergence and strongest self-healing.
- **Selection strategies** — Tail (oldest-first, default) and UniformRandom. Tail selection biases removal toward stale entries, improving view freshness. The strategy is passed to each selection call rather than stored as protocol state.
- **Cascade eviction** — Received peers are merged via a three-stage eviction cascade: remove `healing` oldest peers, remove `swap` head peers, then remove random peers until the view fits its capacity.
- **Request tracking** — `PendingRequestManager` prevents duplicate in-flight requests, retries to different peers after `retry_time`, and emits timeout events after `request_timeout`.
- **Event-driven** — Emits `PeerAdded`, `PeerRemoved`, and `RequestTimeout` events for higher layers to react to view changes.
- **Transport-agnostic** — Produces `OutMessage` values; never touches sockets.

## Configuration

`GossipConfig` exposes the following parameters (defaults in parentheses):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `view_size` | 10 | Maximum peers in the local view |
| `healing` | 5 | Oldest peers evicted per exchange |
| `swap` | 5 | Head peers evicted per exchange |
| `mode` | PushPull | Exchange mode |
| `selector` | Tail | Peer selection strategy |
| `max_concurrent_requests` | 3 | Bound on in-flight gossip requests |
| `retry_time` | 2 s | Wait before retrying to a different peer |
| `request_timeout` | 5 s | Time before a request is considered timed out |
| `poll_interval` | 20 s | Cycle time between gossip rounds |

## Module Structure

```
gossip/
├── mod.rs          # GossipSampling protocol (OverlayProtocol + PeerSampling impls)
├── config.rs       # GossipConfig, GossipMode
├── selector.rs     # Selection strategies (Tail, UniformRandom)
├── view.rs         # View<A> — bounded peer set with exchange and eviction logic
└── proto/          # Protocol Buffers schema and From/TryFrom conversions
    ├── gossip.proto
    ├── conversions.rs
    └── mod.rs
```

## Trait Implementations

- **`OverlayProtocol`** — `poll()` drives timeout processing, retries, and new gossip rounds. `on_message()` handles incoming Push/Pull/PushPull/Response messages.
- **`PeerSampling`** — `view()` returns the current peer set. `select_peer(mode)` / `select_peers(n, mode)` accept a `SelectorType` to choose the selection strategy.
