//! Fallible `Vec::try_reserve_exact` wrapper used at engine
//! ingress boundaries per
//! `docs/internal/superpowers/specs/2026-06-24-engine-boundary-fallibility-and-backend-owned-tensors.md`
//! §1 (Principle 1) + §2.1 sites S4 / S5.
//!
//! Production code calls `Vec::try_reserve_exact` directly through
//! this single wrapper so the test harness can intercept the call
//! and force a `TryReserveError` without monkey-patching the global
//! allocator — keeps the `peak_alloc` / `dhat` test infrastructure
//! free of conflicts.
//!
//! At runtime the wrapper is a zero-cost forward to
//! `Vec::try_reserve_exact`; the `cfg(any(test, feature =
//! "test-components"))` arm threads the injection state through a
//! thread-local `Cell` so integration tests in the facade crate can
//! drive the seam through the same one-shot fault primitive the
//! crate-internal tests use.
//!
//! Hot path: one branch + one fn call (LLVM inlines the
//! non-test arm).

use std::collections::TryReserveError;

/// Fallibly reserve `additional` slots in `vec`. Wraps
/// `Vec::try_reserve_exact` so the boundary callers route through
/// a single seam.
///
/// Under `cfg(any(test, feature = "test-components"))` callers may
/// install a thread-local fault via [`testing::FailOnce`] which
/// forces the next invocation to return a synthetic
/// `TryReserveError`. Production builds (no test-components
/// feature) never pay for the test branch — the function is
/// `#[inline]` and the non-test arm is a direct delegate.
#[inline]
pub(crate) fn try_reserve_exact<T>(
    vec: &mut Vec<T>,
    additional: usize,
) -> Result<(), TryReserveError> {
    #[cfg(any(test, feature = "test-components"))]
    {
        if let Some(err) = testing::take_pending_fault() {
            return Err(err);
        }
    }
    vec.try_reserve_exact(additional)
}

#[cfg(any(test, feature = "test-components"))]
pub mod testing {
    //! Thread-local one-shot fault for the `try_reserve_exact` seam
    //! in the enclosing module. Reachable under
    //! `cfg(any(test, feature = "test-components"))` so integration
    //! tests in the facade crate can drive the same seam the
    //! crate-internal sibling tests use. `FailOnce::install` returns
    //! an RAII guard that clears the thread-local on drop so a
    //! panicking assertion does not poison the next test on the
    //! thread.
    //!
    //! `try_reserve_exact` against `Vec<()>` is the canonical way to
    //! mint a real `TryReserveError` without unsafe.
    use std::cell::Cell;
    use std::collections::TryReserveError;

    thread_local! {
        static PENDING_FAULT: Cell<Option<TryReserveError>> = const { Cell::new(None) };
    }

    /// Replace the thread-local fault with `Some(err)`. The next
    /// `try_reserve_exact` call consumes it and returns `Err`.
    pub fn arm_fault(err: TryReserveError) {
        PENDING_FAULT.with(|cell| cell.set(Some(err)));
    }

    /// Read-and-clear the pending fault, if any. Internal to
    /// `super::try_reserve_exact`.
    pub(super) fn take_pending_fault() -> Option<TryReserveError> {
        PENDING_FAULT.with(|cell| cell.take())
    }

    /// Synthesize a real `TryReserveError`. Asking for `usize::MAX`
    /// elements of a 32-byte type overflows the allocation-size
    /// computation in `Vec::try_reserve_exact` and returns
    /// `Err(CapacityOverflow)` without actually touching the
    /// allocator.
    pub fn synthetic_err() -> TryReserveError {
        // Sized payload so `additional * size_of::<T>()` overflows
        // and the CapacityOverflow branch fires before any alloc.
        let mut v: Vec<[u8; 32]> = Vec::new();
        v.try_reserve_exact(usize::MAX)
            .expect_err("usize::MAX reserve overflows TryReserveError")
    }

    /// RAII guard that arms a fault on construction and clears it
    /// on drop. Use when a test might panic between arming and
    /// consuming the fault.
    pub struct FailOnce {
        // No state needed; presence on stack guarantees drop runs.
        _private: (),
    }

    impl FailOnce {
        /// Install a one-shot fault. Returns the guard.
        pub fn install() -> Self {
            arm_fault(synthetic_err());
            Self { _private: () }
        }
    }

    impl Drop for FailOnce {
        fn drop(&mut self) {
            // Clear any uncon­sumed fault so leftover state does not
            // poison the next test on this thread.
            PENDING_FAULT.with(|cell| cell.set(None));
        }
    }
}
