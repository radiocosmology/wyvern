//! Python wrapper for linear interpolator.
use num_complex::Complex;
use num_traits::Float;
use numpy::{
    Element, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray2, PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::pyutils::{ensure_array, require_dtype, require_ndim};

/// Separate methods for interpolating real and complex, weighted
/// and unweighted
fn interpolate_real<'py, T>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, T>,
    mut y_out: PyReadwriteArray2<'py, T>,
) -> PyResult<Py<PyUntypedArray>>
where
    T: Float + Element + Sync + Send,
{
    // Need views of arrays before detaching
    let y_in_view = y_in.as_array();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_real(x_in, x_out, &y_in_view, y_out_view)
    })?;

    Ok(y_out.as_untyped().clone().unbind())
}

fn interpolate_real_weighted<'py, T, W>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, T>,
    w_in: &PyReadonlyArray2<'py, W>,
    mut y_out: PyReadwriteArray2<'py, T>,
    mut w_out: PyReadwriteArray2<'py, W>,
) -> PyResult<(Py<PyUntypedArray>, Py<PyUntypedArray>)>
where
    T: Float + Element + Sync + Send,
    W: Float + Element + Sync + Send,
{
    // views before detach
    let y_in_view = y_in.as_array();
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let y_out_view = y_out.as_array_mut();

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_real_weighted(
            x_in, x_out, &y_in_view, &w_in_view, y_out_view, w_out_view,
        )
    })?;

    Ok((
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    ))
}

fn interpolate_complex<'py, T>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    mut y_out: PyReadwriteArray2<'py, Complex<T>>,
) -> PyResult<Py<PyUntypedArray>>
where
    T: Float + Element + Sync + Send,
    Complex<T>: Element,
{
    // views before detach. Provides strided re/im views
    let (yre_in, yim_in) = unsafe { crate::utils::split_complex_view(&y_in.as_array()) };
    let (yre_out, yim_out) = unsafe { crate::utils::split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_complex(x_in, x_out, &yre_in, &yim_in, yre_out, yim_out)
    })?;

    Ok(y_out.as_untyped().clone().unbind())
}

fn interpolate_complex_weighted<'py, T, W>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    w_in: &PyReadonlyArray2<'py, W>,
    mut y_out: PyReadwriteArray2<'py, Complex<T>>,
    mut w_out: PyReadwriteArray2<'py, W>,
) -> PyResult<(Py<PyUntypedArray>, Py<PyUntypedArray>)>
where
    T: Float + Element + Sync + Send,
    W: Float + Element + Sync + Send,
    Complex<T>: Element,
{
    let (yre_in, yim_in) = unsafe { crate::utils::split_complex_view(&y_in.as_array()) };
    let w_in_view = w_in.as_array();

    let w_out_view = w_out.as_array_mut();
    let (yre_out, yim_out) = unsafe { crate::utils::split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_complex_weighted(
            x_in, x_out, &yre_in, &yim_in, &w_in_view, yre_out, yim_out, w_out_view,
        )
    })?;

    Ok((
        y_out.as_untyped().clone().unbind(),
        w_out.as_untyped().clone().unbind(),
    ))
}

/// Linearly interpolate a 2D array.
///
/// Parameters
/// ----------
/// ``x_in``
///     1D float64 sorted array with input sample indices
/// ``x_out``
///     1D float64 sorted array with output sample indices. Must
///     have uniform spacing.
/// ``y_in``
///     2D float or complex float array to be interpolated.
/// ``y_out``
///     Optional 2D float or complex float array to store output.
///     If this is None, a new array is allocated. Default is None.
///
/// Returns
/// -------
/// ``y_out``
///     2D float or complex float array, shape (-1, ``n_out``)
#[pyfunction]
#[pyo3(signature = (x_in, x_out, y_in, *, y_out = None))]
pub fn interpolate_linear<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    y_in: &Bound<'py, PyUntypedArray>,
    y_out: Option<&Bound<'py, PyUntypedArray>>,
) -> PyResult<Py<PyUntypedArray>> {
    // bounds checks, type checks, etc...
    require_ndim(Some(y_in), "y_in", 2)?;
    require_ndim(y_out, "y_in", 2)?;
    require_ndim(Some(x_in), "x_in", 1)?;
    require_ndim(Some(x_out), "x_out", 1)?;
    // Input samples are must be f64
    require_dtype(Some(x_in), "x_in", &dtype::<f64>(py))?;
    require_dtype(Some(x_out), "x_out", &dtype::<f64>(py))?;

    // extract
    let x_in: PyReadonlyArray1<f64> = x_in.extract()?;
    let x_in_sl = x_in.as_slice()?;
    let x_out: PyReadonlyArray1<f64> = x_out.extract()?;
    let x_out_sl = x_out.as_slice()?;

    // sort out the expected output array shape
    #[allow(clippy::indexing_slicing, reason = "ndim already validated")]
    let out_shape = [y_in.shape()[0], x_out.len()];

    // unfortunately, need to disdpatch based on types here. For
    // now, require that both data and weights have matching
    // bit depth
    let y_dtype = y_in.dtype();
    // TODO: break this dispatch tree into some sort of macro or something
    if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
        let y_out = ensure_array::<f32>(py, y_out, out_shape)?;
        return interpolate_real::<f32>(py, x_in_sl, x_out_sl, &y_in, y_out);
    }
    if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
        let y_out = ensure_array::<f64>(py, y_out, out_shape)?;
        return interpolate_real::<f64>(py, x_in_sl, x_out_sl, &y_in, y_out);
    }
    if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
        let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
        let y_out = ensure_array::<Complex<f32>>(py, y_out, out_shape)?;
        return interpolate_complex::<f32>(py, x_in_sl, x_out_sl, &y_in, y_out);
    }
    if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
        let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
        let y_out = ensure_array::<Complex<f64>>(py, y_out, out_shape)?;
        return interpolate_complex::<f64>(py, x_in_sl, x_out_sl, &y_in, y_out);
    }
    Err(PyTypeError::new_err(format!(
        "'y' has unsupported type '{y_dtype}'. supported types are: \
        float32, float64, complex32, complex64.",
    )))
}

/// Linearly interpolate a 2D array with corresponding weights.
///
/// Parameters
/// ----------
/// ``x_in``
///     1D float64 sorted array with input sample indices
/// ``x_out``
///     1D float64 sorted array with output sample indices. Must
///     have uniform spacing.
/// ``y_in``
///     2D float or complex float array to be interpolated.
/// ``w_in``
///     2D float array of inverse-variance sample weights. Weights are
///     propagated by propagating variances and inverting the result.
/// ``y_out``
///     Optional 2D float or complex float array to store output.
///     If this is None, a new array is allocated. Default is None.
/// ``w_out``
///     Optional 2D float array to store propagated weights.
///     If this is None, a new array is allocated. Default is None.
///
/// Returns
/// -------
/// ``y_out``
///     2D float or complex float array, shape (-1, ``n_out``)
/// ``w_out``
///     2D float array, shape (-1, ``n_out``)
#[pyfunction]
#[pyo3(signature = (x_in, x_out, y_in, w_in, *, y_out = None, w_out = None))]
pub fn interpolate_linear_weighted<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
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
    require_ndim(Some(x_in), "x_in", 1)?;
    require_ndim(Some(x_out), "x_out", 1)?;

    require_dtype(Some(x_in), "x_in", &dtype::<f64>(py))?;
    require_dtype(Some(x_out), "x_out", &dtype::<f64>(py))?;

    // extract and call
    let x_in: PyReadonlyArray1<f64> = x_in.extract()?;
    let x_in_sl = x_in.as_slice()?;
    let x_out: PyReadonlyArray1<f64> = x_out.extract()?;
    let x_out_sl = x_out.as_slice()?;

    // sort out the expected output array shape
    #[allow(clippy::indexing_slicing, reason = "ndim already validated")]
    let out_shape = [y_in.shape()[0], x_out.len()];

    // unfortunately, need to an annpying dispatch here
    let y_dtype = y_in.dtype();
    let w_dtype = w_in.dtype();

    if w_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let w_in: PyReadonlyArray2<f32> = w_in.extract()?;
        let w_out = ensure_array::<f32>(py, w_out, out_shape)?;

        if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
            let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
            let y_out = ensure_array::<f32>(py, y_out, out_shape)?;
            return interpolate_real_weighted::<f32, f32>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
        if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
            let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
            let y_out = ensure_array::<f64>(py, y_out, out_shape)?;
            return interpolate_real_weighted::<f64, f32>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
            let y_out = ensure_array::<Complex<f32>>(py, y_out, out_shape)?;
            return interpolate_complex_weighted::<f32, f32>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
            let y_out = ensure_array::<Complex<f64>>(py, y_out, out_shape)?;
            return interpolate_complex_weighted::<f64, f32>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
    }
    if w_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let w_in: PyReadonlyArray2<f64> = w_in.extract()?;
        let w_out = ensure_array::<f64>(py, w_out, out_shape)?;

        if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
            let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
            let y_out = ensure_array::<f32>(py, y_out, out_shape)?;
            return interpolate_real_weighted::<f32, f64>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
        if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
            let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
            let y_out = ensure_array::<f64>(py, y_out, out_shape)?;
            return interpolate_real_weighted::<f64, f64>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
            let y_out = ensure_array::<Complex<f32>>(py, y_out, out_shape)?;
            return interpolate_complex_weighted::<f32, f64>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
            let y_out = ensure_array::<Complex<f64>>(py, y_out, out_shape)?;
            return interpolate_complex_weighted::<f64, f64>(
                py, x_in_sl, x_out_sl, &y_in, &w_in, y_out, w_out,
            );
        }
    }
    Err(PyTypeError::new_err(format!(
        "Unsupported data types! `y`: {y_dtype}, `w`: {w_dtype}. \
        Supported data types are: float32, float64, complex64, complex128. \
        Supported weight types are: float32, float64."
    )))
}
