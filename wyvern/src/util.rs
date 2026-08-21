//! Interal utilities

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
