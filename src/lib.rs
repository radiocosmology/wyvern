//! Fast linear interpolation for python.
use pyo3::prelude::*;

mod linear;
mod linear_py;
mod types;
mod utils;

use linear_py::interpolate_linear;

/// Rust-based fast interpolation.
#[pymodule]
fn interprs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interpolate_linear, m)?)?;

    Ok(())
}
