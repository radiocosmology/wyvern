//! Fast linear interpolation for python.
use pyo3::prelude::*;

mod linear;
mod linear_py;

use linear_py::interp_last_axis_linear;

/// Rust-based fast interpolation.
#[pymodule]
fn interp_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(interp_last_axis_linear, m)?)?;

    Ok(())
}
