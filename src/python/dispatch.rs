//! Generic dispatch methods over interpolator types
/// Separate methods for interpolating real and complex, weighted
/// and unweighted
use num_complex::Complex;
use numpy::{
    Element, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray2, PyReadwriteArray2,
    PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use super::{ensure_array, require_ndim, split_complex_view, split_complex_view_mut};
use crate::core::{
    InterpolationPlan, Interpolator, IntoInterpolator, interp_last_ax_complex,
    interp_last_ax_complex_weighted, interp_last_ax_real, interp_last_ax_real_weighted,
};
use crate::types::ParFloatLike;

// ------ Dispatch to typed methods ------

/// Unweighted dispatch macro - expands to a sequence if dtype comparisons
macro_rules! try_dispatch_unweighted {
    (
        $py:expr, $plan:expr, $y_in:expr, $y_out:expr, $out_shape:expr, $y_dtype:expr,
        [ $( ($pytype:ty, $floatty:ty, $func:ident) ),+ $(,)? ]
    ) => {
        $(
            if $y_dtype.is_equiv_to(&dtype::<$pytype>($py)) {
                let y_in: PyReadonlyArray2<$pytype> = $y_in.extract()?;
                let y_out = ensure_array::<$pytype>($py, $y_out, $out_shape)?;
                let interpolator: &dyn Interpolator<$floatty> = $plan.as_interpolator();
                return Ok($func::<$floatty>($py, interpolator, &y_in, y_out));
            }
        )+
    };
}

/// Weighted dispatch macro - two-level structure which checks float weights
/// and interpolator types, before checking for real vs complex data
macro_rules! try_dispatch_weighted {
    (
        $py:expr, $plan:expr, $y_in:expr, $w_in:expr, $y_out:expr, $w_out:expr,
        $out_shape:expr, $y_dtype:expr, $w_dtype:expr,
        [ $( ($floatty:ty, [ $( ($pytype:ty, $func:ident) ),+ $(,)? ]) ),+ $(,)? ]
    ) => {
        $(
            if $w_dtype.is_equiv_to(&dtype::<$floatty>($py)) {
                let w_in: PyReadonlyArray2<$floatty> = $w_in.extract()?;
                let w_out = ensure_array::<$floatty>($py, $w_out, $out_shape)?;
                let interpolator: &dyn Interpolator<$floatty> = $plan.as_interpolator();

                $(
                    if $y_dtype.is_equiv_to(&dtype::<$pytype>($py)) {
                        let y_in: PyReadonlyArray2<$pytype> = $y_in.extract()?;
                        let y_out = ensure_array::<$pytype>($py, $y_out, $out_shape)?;
                        return Ok($func::<$floatty>(
                            $py, interpolator, &y_in, &w_in, y_out, w_out,
                        ));
                    }
                )+
            }
        )+
    };
}

/// Type dispatch for unweighted interpolator calls
pub fn dispatch_unweighted<'py, P: IntoInterpolator + InterpolationPlan>(
    py: Python<'py>,
    plan: &P,
    y_in: &Bound<'py, PyUntypedArray>,
    y_out: Option<&Bound<'py, PyUntypedArray>>,
) -> PyResult<Py<PyUntypedArray>> {
    // bounds checks, type checks, etc...
    require_ndim(Some(y_in), "y_in", 2)?;
    require_ndim(y_out, "y_in", 2)?;

    // sort out the expected output array shape
    #[allow(clippy::indexing_slicing, reason = "ndim already validated")]
    let out_shape = [y_in.shape()[0], plan.len()];

    // unfortunately, need to disdpatch based on types here. For
    // now, require that both data and weights have matching bit depth
    let y_dtype = y_in.dtype();

    try_dispatch_unweighted!(
        py, plan, y_in, y_out, out_shape, y_dtype,
        [
            (f32, f32, real_unweighted),
            (f64, f64, real_unweighted),
            (Complex<f32>, f32, complex_unweighted),
            (Complex<f64>, f64, complex_unweighted),
        ]
    );

    Err(PyTypeError::new_err(format!(
        "'y' has unsupported type '{y_dtype}'. supported types are: \
        float32, float64, complex64, complex128.",
    )))
}

/// Type dispatch for weighted interpolator calls
pub fn dispatch_weighted<'py, P: IntoInterpolator + InterpolationPlan>(
    py: Python<'py>,
    plan: &P,
    y_in: &Bound<'py, PyUntypedArray>,
    w_in: &Bound<'py, PyUntypedArray>,
    y_out: Option<&Bound<'py, PyUntypedArray>>,
    w_out: Option<&Bound<'py, PyUntypedArray>>,
) -> PyResult<(Py<PyUntypedArray>, Py<PyUntypedArray>)> {
    // bounds checks, type checks, etc...
    require_ndim(Some(y_in), "y_in", 2)?;
    require_ndim(Some(w_in), "w_in", 2)?;
    require_ndim(y_out, "y_in", 2)?;
    require_ndim(w_out, "w_in", 2)?;

    // sort out the expected output array shape
    #[allow(clippy::indexing_slicing, reason = "ndim already validated")]
    let out_shape = [y_in.shape()[0], plan.len()];

    // unfortunately, need to an annpying dispatch here
    let y_dtype = y_in.dtype();
    let w_dtype = w_in.dtype();

    try_dispatch_weighted!(
        py, plan, y_in, w_in, y_out, w_out, out_shape, y_dtype, w_dtype,
        [
            (f32, [ (f32, real_weighted), (Complex<f32>, complex_weighted) ]),
            (f64, [ (f64, real_weighted), (Complex<f64>, complex_weighted) ]),
        ]
    );

    Err(PyTypeError::new_err(format!(
        "Unsupported data types or combination! `y`: {y_dtype}, `w`: {w_dtype}. \
        Supported data types are: float32, float64, complex64, complex128. \
        Supported weight types are: float32, float64. \
        data and weight arrays must have matching bit depth."
    )))
}

// ------ Generic typed, multithreaded dispatch ------

fn real_unweighted<'py, T>(
    py: Python<'py>,
    interpolator: &dyn Interpolator<T>,
    y_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, T>,
) -> Py<PyUntypedArray>
where
    T: ParFloatLike + Element,
{
    // Need views of arrays before detaching
    let y_in_view = y_in.as_array();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| interp_last_ax_real(interpolator, &y_in_view, y_out_view));

    y_out.as_untyped().clone().unbind()
}

fn real_weighted<'py, T>(
    py: Python<'py>,
    interpolator: &dyn Interpolator<T>,
    y_in: &PyReadonlyArray2<'py, T>,
    w_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, T>,
    mut w_out: PyReadwriteArray2<'py, T>,
) -> (Py<PyUntypedArray>, Py<PyUntypedArray>)
where
    T: ParFloatLike + Element,
{
    // views before detach
    let y_in_view = y_in.as_array();
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| {
        interp_last_ax_real_weighted(interpolator, &y_in_view, &w_in_view, y_out_view, w_out_view);
    });

    (
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    )
}

fn complex_unweighted<'py, T>(
    py: Python<'py>,
    interpolator: &dyn Interpolator<T>,
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    mut y_out: PyReadwriteArray2<'py, Complex<T>>,
) -> Py<PyUntypedArray>
where
    T: ParFloatLike + Element,
    Complex<T>: Element,
{
    // views before detach. Provides strided re/im views
    let (yre_in, yim_in) = unsafe { split_complex_view(&y_in.as_array()) };
    let (yre_out, yim_out) = unsafe { split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| interp_last_ax_complex(interpolator, &yre_in, &yim_in, yre_out, yim_out));

    y_out.as_untyped().clone().unbind()
}

fn complex_weighted<'py, T>(
    py: Python<'py>,
    interpolator: &dyn Interpolator<T>,
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    w_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, Complex<T>>,
    mut w_out: PyReadwriteArray2<'py, T>,
) -> (Py<PyUntypedArray>, Py<PyUntypedArray>)
where
    T: ParFloatLike + Element,
    Complex<T>: Element,
{
    let (yre_in, yim_in) = unsafe { split_complex_view(&y_in.as_array()) };
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let (yre_out, yim_out) = unsafe { split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| {
        interp_last_ax_complex_weighted(
            interpolator,
            &yre_in,
            &yim_in,
            &w_in_view,
            yre_out,
            yim_out,
            w_out_view,
        );
    });

    (
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    )
}
