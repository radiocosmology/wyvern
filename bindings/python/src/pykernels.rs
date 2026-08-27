//! Python bindings for `wyvern::kernels`.
#![allow(
    clippy::missing_const_for_fn,
    clippy::let_unit_value,
    reason = "not necessary for wrappers"
)]
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::{
    PyStubType, TypeInfo,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
};

use wyvern::kernels;
use wyvern::kernels::traits::Kernel;

macro_rules! build_py_kernel {
    (
        $py_name:ident, $rust_type:ty, $py_class_name:literal, $repr:expr,
        methods { $( fn $method:ident(&self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty ;)* }
        methods_mut { $( fn $method_mut:ident(&mut self $(, $arg_mut:ident : $arg_mut_ty:ty)*) -> $ret_mut:ty ;)* }
    ) => {
        #[cfg_attr(
            feature = "stub-gen",
            gen_stub_pyclass(module = "wyvern.kernels")
        )]
        #[pyclass(name = $py_class_name, from_py_object)]
        /// Python wrapper around a concrete Rust kernel implementation.
        #[derive(Debug, Clone)]
        pub struct $py_name {
            inner: $rust_type,
        }

        #[cfg_attr(
            feature = "stub-gen",
            gen_stub_pymethods
        )]
        #[pymethods]
        impl $py_name {
            #[new]
            fn new(ntaps: usize) -> Self {
                Self {
                    inner: <$rust_type as Kernel>::build(ntaps),
                }
            }

            fn __repr__(&self) -> String {
                ($repr)(&self.inner)
            }

            $(
                fn $method(&self $(, $arg: $arg_ty)*) -> $ret {
                    self.inner.$method($($arg),*)
                }
            )*

            $(
                fn $method_mut(&mut self $(, $arg_mut: $arg_mut_ty)*) -> $ret_mut {
                    self.inner.$method_mut($($arg_mut),*)
                }
            )*
        }
    };
}

macro_rules! build_kernel_enum {
    (
        $($py_name:ident => $variant:ident),+ $(,)?
    ) => {
        /// A kernel variant accepted by the Python interpolation entry points.
        #[derive(FromPyObject)]
        pub enum AnyKernel {
            $($variant($py_name)),+
        }

        impl AnyKernel {
            /// Unwrap the concrete Python kernel into the underlying Rust kernel trait object.
            ///
            /// # Returns
            /// A boxed `dyn Kernel` containing the concrete kernel implementation.
            pub fn into_inner(self) -> Box<dyn Kernel> {
                match self {
                    $(AnyKernel::$variant(k) => Box::new(k.inner)),+
                }
            }
        }

        #[cfg(feature = "stub-gen")]
        impl PyStubType for AnyKernel {
            fn type_output() -> TypeInfo {
                build_kernel_enum!(@union_fold $($py_name),+)
            }
        }

        fn register_kernel_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_class::<$py_name>()?;)+
            Ok(())
        }
    };

    (@union_fold $first:ident $(, $rest:ident)*) => {
        {
            let mut combined = <$first as PyStubType>::type_output();
            $(
                combined = combined | <$rest as PyStubType>::type_output();
            )*
            combined
        }
    };
}

#[allow(clippy::wildcard_imports, reason = "clarity")]
#[pymodule(submodule)]
#[pyo3(name = "kernels")]
pub mod _kernels {
    use super::*;

    // construct kernels. Required once per kernel
    build_py_kernel!(
        PyLanczosKernel,
        kernels::LanczosKernel,
        "LanczosKernel",
        |k: &kernels::LanczosKernel| format!("LanczosKernel(a={})", k.half_width()),
        methods {
            fn evaluate(&self, x: f64) -> f64;
            fn half_width(&self) -> f64;
            fn ntaps(&self) -> usize;
        }
        methods_mut {
            fn set_ntaps(&mut self, ntaps: usize) -> ();
        }
    );

    build_py_kernel!(
        PyKaiserBesselKernel,
        kernels::KaiserBesselKernel,
        "KaiserBesselKernel",
        |k: &kernels::KaiserBesselKernel| format!(
            "KaiserBesselKernel(a={}, beta={})",
            k.half_width(),
            k.beta()
        ),
        methods {
            fn evaluate(&self, x: f64) -> f64;
            fn half_width(&self) -> f64;
            fn ntaps(&self) -> usize;
            fn beta(&self) -> f64;
        }
        methods_mut {
            fn set_ntaps(&mut self, ntaps: usize) -> ();
            fn set_beta(&mut self, beta: f64) -> ();
            fn set_beta_default(&mut self) -> ();
        }
    );

    build_py_kernel!(
        PyBoxcarKernel,
        kernels::BoxcarKernel,
        "BoxcarKernel",
        |k: &kernels::BoxcarKernel| format!("BoxcarKernel(a={})", k.half_width()),
        methods {
            fn evaluate(&self, x: f64) -> f64;
            fn half_width(&self) -> f64;
            fn ntaps(&self) -> usize;
        }
        methods_mut {
            fn set_ntaps(&mut self, ntaps: usize) -> ();
        }
    );

    build_kernel_enum!(
        PyLanczosKernel => Lanczos,
        PyKaiserBesselKernel => KaiserBessel,
        PyBoxcarKernel => Boxcar,
    );

    #[pymodule_init]
    #[allow(
        clippy::missing_const_for_fn,
        clippy::unnecessary_wraps,
        unused_variables,
        reason = "generic init"
    )]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // need to manually register the kernels due to macro expansion
        // and pymodule initialization ordering conflicts
        register_kernel_classes(m)
    }
}
