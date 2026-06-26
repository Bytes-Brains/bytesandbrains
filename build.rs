//! Build script for the `bytesandbrains` root crate.
//!
//! PLAN tender-noodling-sky Phase 4a — proto compilation has moved
//! to the `bb-ir` workspace crate. The root facade crate has no
//! build-time codegen of its own; this stub stays so cargo doesn't
//! warn on a vestigial `build = "build.rs"` declaration the
//! manifest may carry from earlier stages.

fn main() {
    // Nothing to do — bb-ir compiles the protos.
}
