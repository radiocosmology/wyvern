//! Fast linear interpolation for python.
use pyo3::prelude::*;

mod linear;
mod pylinear;
mod pyutils;
mod utils;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Rust-based fast interpolation.
#[pymodule]
fn interprs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pylinear::interpolate_linear, m)?)?;
    m.add_function(wrap_pyfunction!(pylinear::interpolate_linear_weighted, m)?)?;

    Ok(())
}
