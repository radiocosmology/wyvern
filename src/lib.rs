//! Fast linear interpolation for python.
use pyo3::prelude::*;

mod interpolate;
mod kernels;
mod types;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Rust-based fast interpolation.
fn init_interpolate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        interpolate::python::lanczos::interpolate_lanczos,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        interpolate::python::lanczos::interpolate_lanczos_weighted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        interpolate::python::linear::interpolate_linear,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        interpolate::python::linear::interpolate_linear_weighted,
        m
    )?)?;

    Ok(())
}

#[pymodule]
fn wyvern(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // submodule container
    let interpolate_submodule = PyModule::new(py, "interpolate")?;
    init_interpolate(&interpolate_submodule)?;
    // add to parent module
    m.add_submodule(&interpolate_submodule)?;

    // add to sys.modules
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("wyvern.interpolate", &interpolate_submodule)?;

    Ok(())
}
