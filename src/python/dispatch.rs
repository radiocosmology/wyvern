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
use crate::interp::{
    InterpolationPlan, interp_last_ax_complex, interp_last_ax_complex_weighted,
    interp_last_ax_real, interp_last_ax_real_weighted,
};
use crate::types::ParFloatLike;

/// Type dispatch for unweighted interpolator calls
pub fn dispatch_unweighted<'py, P: InterpolationPlan + Sync>(
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

    if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
        let y_out = ensure_array::<f32>(py, y_out, out_shape)?;
        return Ok(real_unweighted::<f32, P>(py, plan, &y_in, y_out));
    }
    if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
        let y_out = ensure_array::<f64>(py, y_out, out_shape)?;
        return Ok(real_unweighted::<f64, P>(py, plan, &y_in, y_out));
    }
    if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
        let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
        let y_out = ensure_array::<Complex<f32>>(py, y_out, out_shape)?;
        return Ok(complex_unweighted::<f32, P>(py, plan, &y_in, y_out));
    }
    if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
        let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
        let y_out = ensure_array::<Complex<f64>>(py, y_out, out_shape)?;
        return Ok(complex_unweighted::<f64, P>(py, plan, &y_in, y_out));
    }
    Err(PyTypeError::new_err(format!(
        "'y' has unsupported type '{y_dtype}'. supported types are: \
        float32, float64, complex64, complex128.",
    )))
}

/// Type dispatch for weighted interpolator calls
pub fn dispatch_weighted<'py, P: InterpolationPlan + Sync>(
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

    if w_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let w_in: PyReadonlyArray2<f32> = w_in.extract()?;
        let w_out = ensure_array::<f32>(py, w_out, out_shape)?;

        if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
            let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
            let y_out = ensure_array::<f32>(py, y_out, out_shape)?;
            return Ok(real_weighted::<f32, P>(
                py, plan, &y_in, &w_in, y_out, w_out,
            ));
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
            let y_out = ensure_array::<Complex<f32>>(py, y_out, out_shape)?;
            return Ok(complex_weighted::<f32, P>(
                py, plan, &y_in, &w_in, y_out, w_out,
            ));
        }
    }
    if w_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let w_in: PyReadonlyArray2<f64> = w_in.extract()?;
        let w_out = ensure_array::<f64>(py, w_out, out_shape)?;

        if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
            let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
            let y_out = ensure_array::<f64>(py, y_out, out_shape)?;
            return Ok(real_weighted::<f64, P>(
                py, plan, &y_in, &w_in, y_out, w_out,
            ));
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
            let y_out = ensure_array::<Complex<f64>>(py, y_out, out_shape)?;
            return Ok(complex_weighted::<f64, P>(
                py, plan, &y_in, &w_in, y_out, w_out,
            ));
        }
    }
    Err(PyTypeError::new_err(format!(
        "Unsupported data types or combination! `y`: {y_dtype}, `w`: {w_dtype}. \
        Supported data types are: float32, float64, complex64, complex128. \
        Supported weight types are: float32, float64. \
        data and weight arrays must have matching bit depth."
    )))
}

fn real_unweighted<'py, T, P>(
    py: Python<'py>,
    plan: &P,
    y_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, T>,
) -> Py<PyUntypedArray>
where
    T: ParFloatLike + Element,
    P: InterpolationPlan + Sync,
{
    // Need views of arrays before detaching
    let y_in_view = y_in.as_array();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| interp_last_ax_real(plan, &y_in_view, y_out_view));

    y_out.as_untyped().clone().unbind()
}

fn real_weighted<'py, T, P>(
    py: Python<'py>,
    plan: &P,
    y_in: &PyReadonlyArray2<'py, T>,
    w_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, T>,
    mut w_out: PyReadwriteArray2<'py, T>,
) -> (Py<PyUntypedArray>, Py<PyUntypedArray>)
where
    T: ParFloatLike + Element,
    P: InterpolationPlan + Sync,
{
    // views before detach
    let y_in_view = y_in.as_array();
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| {
        interp_last_ax_real_weighted(plan, &y_in_view, &w_in_view, y_out_view, w_out_view);
    });

    (
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    )
}

fn complex_unweighted<'py, T, P>(
    py: Python<'py>,
    plan: &P,
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    mut y_out: PyReadwriteArray2<'py, Complex<T>>,
) -> Py<PyUntypedArray>
where
    T: ParFloatLike + Element,
    Complex<T>: Element,
    P: InterpolationPlan + Sync,
{
    // views before detach. Provides strided re/im views
    let (yre_in, yim_in) = unsafe { split_complex_view(&y_in.as_array()) };
    let (yre_out, yim_out) = unsafe { split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| interp_last_ax_complex(plan, &yre_in, &yim_in, yre_out, yim_out));

    y_out.as_untyped().clone().unbind()
}

fn complex_weighted<'py, T, P>(
    py: Python<'py>,
    plan: &P,
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    w_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, Complex<T>>,
    mut w_out: PyReadwriteArray2<'py, T>,
) -> (Py<PyUntypedArray>, Py<PyUntypedArray>)
where
    T: ParFloatLike + Element,
    Complex<T>: Element,
    P: InterpolationPlan + Sync,
{
    let (yre_in, yim_in) = unsafe { split_complex_view(&y_in.as_array()) };
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let (yre_out, yim_out) = unsafe { split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| {
        interp_last_ax_complex_weighted(
            plan, &yre_in, &yim_in, &w_in_view, yre_out, yim_out, w_out_view,
        );
    });

    (
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    )
}
