//! Fast linear interpolation for python.
use pyo3::prelude::*;

mod interp;
mod kernels;
mod python;
mod types;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Rust-based fast interpolation.
#[pymodule]
fn interprs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::lanczos::interpolate_lanczos, m)?)?;
    m.add_function(wrap_pyfunction!(
        python::lanczos::interpolate_lanczos_weighted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(python::linear::interpolate_linear, m)?)?;
    m.add_function(wrap_pyfunction!(
        python::linear::interpolate_linear_weighted,
        m
    )?)?;

    Ok(())
}
