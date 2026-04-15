//! # BytesAndBrains
//!
//! A Rust library for decentralized networking and edge AI.
//!
//! ## Feature Flags
//!
//! | Feature    | Description                          |
//! |------------|--------------------------------------|
//! | `overlay`  | Overlay protocol base                |
//! | `gossip`   | Gossip peer sampling protocol        |
//! | `codec`    | Product quantization codec           |
//! | `index`    | Vector indexing structures            |
//! | `ml`       | ML utilities (k-means)               |
//! | `proto`    | Protobuf serialization               |
//! | `simd`     | SIMD-accelerated distance functions  |
//! | `full`     | Enable everything                    |
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! bytesandbrains = { version = "0.1", features = ["gossip", "proto"] }
//! ```

/// Core types: embeddings, distances, peers, addresses, protocol traits.
pub use bb_core as core;

/// Overlay protocol implementations.
#[cfg(feature = "overlay")]
pub use bb_overlay as overlay;

/// Product quantization codec for vector compression.
#[cfg(feature = "codec")]
pub use bb_codec as codec;

/// Vector indexing structures.
#[cfg(feature = "index")]
pub use bb_index as index;

/// Machine learning utilities (k-means clustering).
#[cfg(feature = "ml")]
pub use bb_ml as ml;
