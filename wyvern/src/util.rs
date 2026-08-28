//! Internal utilities shared across the interpolation implementation.

macro_rules! assert_unchecked_debug {
    ($cond:expr $(,)?) => {
        // In debug/test mode, perform a normal checking assertion
        debug_assert!($cond);

        // In release mode, tell the optimizer the condition is guaranteed true.
        // SAFETY: The caller must guarantee that `$cond` is always true,
        // as violating this results in undefined behavior.
        if !cfg!(debug_assertions) {
            unsafe {
                std::hint::assert_unchecked($cond);
            }
        }
    };
}

pub(crate) use assert_unchecked_debug;

#[cfg(test)]
mod tests {
    #[test]
    fn true_condition_does_not_panic() {
        // should be a no-op in both debug and release
        assert_unchecked_debug!(1 + 1 == 2);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn false_condition_panics_in_debug_mode() {
        // debug_assert! is active in test/debug builds, so this should panic
        assert_unchecked_debug!(1 + 1 == 3);
    }
}
