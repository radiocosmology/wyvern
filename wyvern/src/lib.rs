//! Fast linear interpolation routines for Python and Rust consumers.
//!
//! The crate exposes interpolation plans, kernel definitions, and numeric utility
//! traits that are used by the Python bindings and the core interpolation engine.

/// Core interpolation algorithms and plan objects.
pub mod interpolate;
/// Available interpolation kernels and kernel traits.
pub mod kernels;
/// Numeric traits used to abstract over real and complex inputs.
pub mod types;
pub(crate) mod util;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
