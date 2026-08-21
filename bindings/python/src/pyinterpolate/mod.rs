//! Python interface to the interpolators
use pyo3::prelude::*;

mod dispatch;
mod interface;

pub use dispatch::*;

#[pymodule(submodule)]
#[pyo3(name = "interpolate")]
pub mod _interpolate {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::interface::interpolate_kernel;
    #[pymodule_export]
    use super::interface::interpolate_kernel_weighted;
    #[pymodule_export]
    use super::interface::interpolate_linear;
    #[pymodule_export]
    use super::interface::interpolate_linear_weighted;

    #[pymodule_init]
    #[allow(
        clippy::missing_const_for_fn,
        clippy::unnecessary_wraps,
        unused_variables,
        reason = "generic init"
    )]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
