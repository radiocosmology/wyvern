//! Python interface to the interpolators
use pyo3::prelude::*;

mod dispatch;
pub mod lanczos;
pub mod linear;
mod utils;

pub use dispatch::*;
pub use utils::*;

pub const PLAN_CACHE_LIMIT: usize = 32;

#[inline]
pub fn samples_to_bits(samples: &[f64]) -> Box<[u64]> {
    samples
        .iter()
        .map(|sample| sample.to_bits())
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

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
