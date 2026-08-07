//! Python bindings for [`wyvern`] fast algorithms
mod importutil;
mod interpolate;

use pyo3::prelude::*;

#[pymodule]
pub mod wyvern {
    use crate::importutil::register_submodule;
    use pyo3::prelude::*;

    #[pymodule_export]
    use crate::interpolate::interpolate;

    #[allow(non_upper_case_globals, reason = "__version__ is a Python standard")]
    #[pymodule_export]
    pub const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        register_submodule!(m, "wyvern", interpolate);

        Ok(())
    }
}
