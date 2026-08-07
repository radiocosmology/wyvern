//! Python interface to the interpolators
use pyo3::prelude::*;

mod dispatch;
pub mod lanczos;
pub mod linear;
mod utils;

pub use dispatch::*;
pub use utils::*;

#[pymodule(submodule)]
pub mod interpolate {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::lanczos::interpolate_lanczos;
    #[pymodule_export]
    use super::lanczos::interpolate_lanczos_weighted;
    #[pymodule_export]
    use super::linear::interpolate_linear;
    #[pymodule_export]
    use super::linear::interpolate_linear_weighted;

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
