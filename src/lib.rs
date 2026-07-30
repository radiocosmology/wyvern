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
mod interprs {
    #[pymodule_export]
    use crate::python::lanczos::interpolate_lanczos;
    #[pymodule_export]
    use crate::python::lanczos::interpolate_lanczos_weighted;
    #[pymodule_export]
    use crate::python::linear::interpolate_linear;
    #[pymodule_export]
    use crate::python::linear::interpolate_linear_weighted;
}
