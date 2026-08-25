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

use wyvern::interpolate::{
    InterpolationPlan, Interpolator, IntoInterpolator, ParallelInterpolator,
};
use wyvern::types::MaybeComplex;

use crate::pyutils::{ensure_array, require_ndim};

// ------ Dispatch to typed methods ------

/// Unweighted dispatch macro - expands to a sequence of dtype comparisons
macro_rules! try_dispatch_unweighted {
    (
        $py:expr, $plan:expr, $y_in:expr, $y_out:expr, $out_shape:expr, $y_dtype:expr,
        [ $( $pytype:ty ),+ $(,)? ]
    ) => {
        $(
            if $y_dtype.is_equiv_to(&dtype::<$pytype>($py)) {
                let y_in: PyReadonlyArray2<$pytype> = $y_in.extract()?;
                let y_out = ensure_array::<$pytype>($py, $y_out, $out_shape)?;
                let interpolator: &dyn Interpolator<$pytype> = $plan.as_interpolator();
                let interpolator = ParallelInterpolator::<$pytype>::with_interpolator(interpolator);

                return Ok(unweighted::<$pytype>($py, &interpolator, &y_in, y_out));
            }
        )+
    };
}

/// Weighted dispatch macro
macro_rules! try_dispatch_weighted {
    (
        $py:expr, $plan:expr, $y_in:expr, $w_in:expr, $y_out:expr, $w_out:expr,
        $out_shape:expr, $y_dtype:expr, $w_dtype:expr,
        [ $( ($pytype:ty, $wtype:ty) ),+ $(,)? ]
    ) => {
        $(
            if $w_dtype.is_equiv_to(&dtype::<$wtype>($py)) {
                let w_in: PyReadonlyArray2<$wtype> = $w_in.extract()?;
                let w_out = ensure_array::<$wtype>($py, $w_out, $out_shape)?;
                // match on the possible-complex data type
                if $y_dtype.is_equiv_to(&dtype::<$pytype>($py)) {
                    let y_in: PyReadonlyArray2<$pytype> = $y_in.extract()?;
                    let y_out = ensure_array::<$pytype>($py, $y_out, $out_shape)?;
                    let interpolator: &dyn Interpolator<$pytype> = $plan.as_interpolator();
                    let interpolator = ParallelInterpolator::<$pytype>::with_interpolator(interpolator);

                    return Ok(weighted::<$pytype>($py, &interpolator, &y_in, &w_in, y_out, w_out));
                }
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

    // require that both data and weights have matching bit depth
    let y_dtype = y_in.dtype();

    try_dispatch_unweighted!(
        py, plan, y_in, y_out, out_shape, y_dtype,
        [ f32, f64, Complex<f32>, Complex<f64> ]
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
        [ (f32, f32), (f64, f64), (Complex<f32>, f32), (Complex<f64>, f64) ]
    );

    Err(PyTypeError::new_err(format!(
        "Unsupported data types or combination! `y`: {y_dtype}, `w`: {w_dtype}. \
        Supported data types are: float32, float64, complex64, complex128. \
        Supported weight types are: float32, float64. \
        data and weight arrays must have matching bit depth."
    )))
}

// ------ Generic typed, multithreaded dispatch ------

fn unweighted<'py, T>(
    py: Python<'py>,
    interpolator: &ParallelInterpolator<T>,
    y_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, T>,
) -> Py<PyUntypedArray>
where
    T: MaybeComplex + Element,
{
    // Need views of arrays before detaching
    let y_in_view = y_in.as_array();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| interpolator.interpolate(&y_in_view, y_out_view));

    y_out.as_untyped().clone().unbind()
}

fn weighted<'py, T>(
    py: Python<'py>,
    interpolator: &ParallelInterpolator<T>,
    y_in: &PyReadonlyArray2<'py, T>,
    w_in: &PyReadonlyArray2<'py, T::Real>,
    mut y_out: PyReadwriteArray2<'py, T>,
    mut w_out: PyReadwriteArray2<'py, T::Real>,
) -> (Py<PyUntypedArray>, Py<PyUntypedArray>)
where
    T: MaybeComplex + Element,
    T::Real: Element,
{
    // views before detach
    let y_in_view = y_in.as_array();
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| {
        interpolator.interpolate_weighted(&y_in_view, &w_in_view, y_out_view, w_out_view);
    });

    (
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    )
}
