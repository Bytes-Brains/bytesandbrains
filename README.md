<p align="center">
  <img src="assets/logo.png" alt="Bytes and Brains" width="200" />
</p>

<h1 align="center">bytesandbrains</h1>

<p align="center">
  A Rust library for decentralized networking and edge AI.
</p>

<p align="center">
  <a href="https://bytesandbrains.ai">bytesandbrains.ai</a>
  &nbsp;·&nbsp;
  <a href="https://crates.io/crates/bytesandbrains">crates.io</a>
  &nbsp;·&nbsp;
  <a href="https://docs.rs/bytesandbrains">docs.rs</a>
</p>

<p align="center">
  <a href="https://crates.io/crates/bytesandbrains"><img src="https://img.shields.io/crates/v/bytesandbrains.svg" alt="crates.io" /></a>
  <a href="https://docs.rs/bytesandbrains"><img src="https://docs.rs/bytesandbrains/badge.svg" alt="docs.rs" /></a>
  <a href="https://github.com/Bytes-Brains/bytesandbrains/blob/main/LICENSE"><img src="https://img.shields.io/crates/l/bytesandbrains.svg" alt="license" /></a>
</p>

---

**bytesandbrains** provides composable, transport-agnostic building blocks for peer-to-peer overlay networks and distributed machine learning.

## Quick Start

```toml
[dependencies]
bytesandbrains = { version = "0.2", features = ["full"] }
```

Or pick only what you need:

```toml
[dependencies]
bytesandbrains = { version = "0.2", features = ["gossip", "proto"] }
```

## Workspace

| Crate | Description |
|-------|-------------|
| **[bb-core](bb-core/)** | Shared traits and types: `Address`, `PeerId`, `Peer`, `OverlayProtocol`, `PeerSampling`, `Embedding`/`EmbeddingSpace`, and distance metrics (optional SIMD via `simsimd`). |
| **[bb-overlay](bb-overlay/)** | Protocol implementations built on `bb-core`. |
| **[bb-codec](bb-codec/)** | Product Quantization: `ProductQuantizer`, `PQCode`, distance tables for approximate nearest neighbor search. |
| **[bb-index](bb-index/)** | Vector indexing: `FlatIndex` with add/search/remove/train operations. |
| **[bb-ml](bb-ml/)** | Machine learning utilities: K-Means clustering. |

## Protocols

### [Gossip Peer Sampling](bb-overlay/src/gossip/README.md)

A gossip-based peer sampling framework built on the T-Man protocol abstraction. The gossip loop (poll, message handling, request tracking) is written once, and peer selection and view exchange strategies are pluggable via traits:

- **`PeerSelector<P, A>`**: Controls which peer to gossip with. Ships with `RandomizedSelector` (Tail / UniformRandom modes).
- **`ViewExchange<P, A>`**: Controls how views are prepared and merged. Ships with `RandomizedExchange` (healing/swap eviction with age tracking).
- **`GossipPeerType<A>`**: Trait for peer types. Ships with `AgePeer<A>` for age-based ranking.

`RandomizedGossip<A>` is the standard type alias combining all three for the classic randomized gossip protocol.

## Feature Flags

| Flag | Description |
|------|-------------|
| `gossip` | Gossip peer sampling protocol |
| `codec` | Product quantization codec |
| `index` | Vector indexing structures |
| `ml` | ML utilities (k-means) |
| `proto` | Protocol Buffers serialization (via `prost`) |
| `simd` | SIMD-accelerated distance functions |
| `full` | All of the above |

## License

GPL-2.0
