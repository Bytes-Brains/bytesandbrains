#![warn(missing_docs)]
// Doc links target facade-level items in `bytesandbrains`.
#![allow(rustdoc::broken_intra_doc_links)]

//! Authoring DSL for `bytesandbrains`. `Output` handles, Contract
//! traits, placeholders, and `ConcreteComponent`. Depends only on
//! `bb-ir`. `Module::build()` records the composition tree into
//! one `ModelProto` that `bb-compiler` consumes separately.

pub mod concrete;
pub mod contracts;
pub mod graph;
pub mod module;
pub mod output;
pub mod recorded;
pub mod syscalls;

pub use graph::Graph;
pub use module::{BuildError, Module};
pub use output::Output;
