//! Build script for the `bb-ir` crate (tender-noodling-sky
//! Phase 4).
//!
//! Compiles the proto definitions in `../proto/` via `prost-build`:
//!
//!   - `proto/onnx-ml.proto` - vendored canonical ONNX schema
//!     (Apache-2.0). Generated module: `onnx`.
//!   - `proto/bb_core.proto` - the framework's core schema covering
//!     the wire envelope, slot-fill batching, peer identity, and
//!     snapshots. Generated module: `bb.core` → `bb_core`.
//!
//! Generated bindings land in `$OUT_DIR` and are included from
//! `src/lib.rs` via `include!(concat!(env!("OUT_DIR"), "/..."))`.
//!
//! The proto sources live at the workspace root (one level up from
//! `bb-ir/`) so the `bytesandbrains` facade crate can still see the
//! same canonical proto files for IDE tooling.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../proto/onnx-ml.proto");
    println!("cargo:rerun-if-changed=../proto/bb_core.proto");
    println!("cargo:rerun-if-changed=build.rs");

    prost_build::Config::new().compile_protos(
        &["../proto/onnx-ml.proto", "../proto/bb_core.proto"],
        &["../proto/"],
    )?;

    Ok(())
}
