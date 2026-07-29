//! Fast linear interpolation for python.
use pyo3::prelude::*;

mod linear;
mod pylinear;
mod pyutils;
mod types;
mod utils;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Rust-based fast interpolation.
#[pymodule]
mod interprs {

    #[pymodule_export]
    use {super::pylinear::interpolate_linear, super::pylinear::interpolate_linear_weighted};
}
