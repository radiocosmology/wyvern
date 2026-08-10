//! Python bindings for [`wyvern`] fast algorithms
mod importutil;
mod interpolate;

use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::define_stub_info_gatherer;

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

#[cfg(feature = "stub-gen")]
define_stub_info_gatherer!(stub_info);
