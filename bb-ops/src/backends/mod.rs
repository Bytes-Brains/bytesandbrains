//! Concrete backend implementations. Each backend (cpu, cuda, …)
//! is one self-contained unit under `bb_ops::backends::<name>`.

#[cfg(feature = "cpu-backend")]
pub mod cpu;

/// Anchor the CPU backend's symbols + inventory submission against
/// linker DCE.
#[cfg(feature = "cpu-backend")]
pub fn link_force() {
    use std::hint::black_box;
    black_box(std::any::TypeId::of::<cpu::CpuBackend>());
    black_box(cpu::CpuBackend::new());
}
