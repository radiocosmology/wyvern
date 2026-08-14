//! Python wrapper for linear interpolator.
use numpy::{PyReadonlyArray1, PyUntypedArray, dtype};
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::{dispatch_unweighted, dispatch_weighted, require_dtype, require_ndim};
use wyvern::interpolate::LinearInterpolator;

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
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(module = "wyvern.interpolate")
)]
#[pyfunction]
#[pyo3(signature = (x_in, x_out, y_in, *, y_out = None))]
pub fn interpolate_linear<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    y_in: &Bound<'py, PyUntypedArray>,
    y_out: Option<&Bound<'py, PyUntypedArray>>,
) -> PyResult<Py<PyUntypedArray>> {
    // validate input samples
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

    // construct the interpolation plan
    let plan = LinearInterpolator::build(x_in_sl, x_out_sl)?;

    dispatch_unweighted(py, &plan, y_in, y_out)
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
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(module = "wyvern.interpolate")
)]
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
    // validate input samples
    require_ndim(Some(x_in), "x_in", 1)?;
    require_ndim(Some(x_out), "x_out", 1)?;
    require_dtype(Some(x_in), "x_in", &dtype::<f64>(py))?;
    require_dtype(Some(x_out), "x_out", &dtype::<f64>(py))?;

    // extract and call
    let x_in: PyReadonlyArray1<f64> = x_in.extract()?;
    let x_in_sl = x_in.as_slice()?;
    let x_out: PyReadonlyArray1<f64> = x_out.extract()?;
    let x_out_sl = x_out.as_slice()?;

    // construct the interpolation plan
    let plan = LinearInterpolator::build(x_in_sl, x_out_sl)?;

    dispatch_weighted(py, &plan, y_in, w_in, y_out, w_out)
}
