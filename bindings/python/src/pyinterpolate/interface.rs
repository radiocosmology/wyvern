//! Interface for interpolation routines
#![allow(clippy::doc_markdown, reason = "require python-style docstrings")]
use numpy::{PyReadonlyArray1, PyUntypedArray, dtype};
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use wyvern::interpolate::{KernelInterpolator, LinearInterpolator};

use super::{dispatch_unweighted, dispatch_weighted};
use crate::pykernels::_kernels::AnyKernel;
use crate::pyutils::{require_dtype, require_ndim};

/// Linearly interpolate a 2D array.
///
/// Parameters
/// ----------
/// x_in
///     1D float64 sorted array with input sample indices.
/// x_out
///     1D float64 sorted array with output sample indices. Must have uniform spacing.
/// y_in
///     2D float or complex float array to be interpolated. Must be C-contiguous.
/// y_out
///     Optional 2D float or complex float array to store output. Must be C-contiguous.
///     If `None`, a new array is allocated.
///
/// Returns
/// -------
/// y_out
///     2D float or complex float array, shape `(-1, n_out)`.
///
/// Errors
/// ------
/// Raises `ValueError` when the input coordinates are not sorted, repeated, or have
/// incompatible dimensionality; raises `TypeError` for unsupported dtypes.
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
    // validate and extract input samples
    let (x_in, x_out) = validate_extract_samples(py, x_in, x_out)?;
    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;
    // construct the interpolation plan
    let plan = LinearInterpolator::build(x_in_sl, x_out_sl)?;

    dispatch_unweighted(py, &plan, y_in, y_out)
}

/// Linearly interpolate a 2D array with corresponding weights.
///
/// Parameters
/// ----------
/// x_in
///     1D float64 sorted array with input sample indices.
/// x_out
///     1D float64 sorted array with output sample indices. Must have uniform spacing.
/// y_in
///     2D float or complex float array to be interpolated. Must be C-contiguous.
/// w_in
///     2D float array of inverse-variance sample weights. Weights are propagated by
///     propagating variances and inverting the result. Must be C-contiguous.
/// y_out
///     Optional 2D float or complex float array to store output. Must be C-contiguous.
///     If `None`, a new array is allocated.
/// w_out
///     Optional 2D float array to store propagated weights. Must be C-contiguous. If
///     `None`, a new array is allocated.
///
/// Returns
/// -------
/// y_out
///     2D float or complex float array, shape `(-1, n_out)`.
/// w_out
///     2D float array, shape `(-1, n_out)`.
///
/// Errors
/// ------
/// Raises `ValueError` for invalid dimensions, unsorted coordinates, or malformed
/// weight arrays; raises `TypeError` when the dtypes are unsupported.
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
    // validate and extract input samples
    let (x_in, x_out) = validate_extract_samples(py, x_in, x_out)?;
    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;
    // construct the interpolation plan
    let plan = LinearInterpolator::build(x_in_sl, x_out_sl)?;

    dispatch_weighted(py, &plan, y_in, w_in, y_out, w_out)
}

/// Interpolate a 2D array using a provided kernel.
///
/// Parameters
/// ----------
/// x_in
///     1D float64 sorted array with input sample indices.
/// x_out
///     1D float64 sorted array with output sample indices. Must have uniform spacing.
/// kernel
///     [`AnyKernel`] instance describing the interpolation kernel.
/// y_in
///     2D float or complex float array to be interpolated. Must be C-contiguous.
/// scale
///     Optional kernel scaling factor. The inverse is multiplied with the sample spacing
///     before evaluating the kernel at each input sample. Default is `1.0`.
/// y_out
///     Optional 2D float or complex float array to store output. Must be C-contiguous.
///     If `None`, a new array is allocated.
///
/// Returns
/// -------
/// y_out
///     2D float or complex float array, shape `(-1, n_out)`.
///
/// Errors
/// ------
/// Raises `ValueError` when the kernel or coordinates are invalid; raises `TypeError`
/// for unsupported dtypes or invalid kernel objects.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(module = "wyvern.interpolate")
)]
#[pyfunction]
#[pyo3(signature = (x_in, x_out, kernel, y_in, *, scale = 1.0, y_out = None))]
pub fn interpolate_kernel<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    kernel: AnyKernel,
    y_in: &Bound<'py, PyUntypedArray>,
    scale: f64,
    y_out: Option<&Bound<'py, PyUntypedArray>>,
) -> PyResult<Py<PyUntypedArray>> {
    // validate and extract input samples
    let (x_in, x_out) = validate_extract_samples(py, x_in, x_out)?;
    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;
    // Build the interpolation plan based off of the inner kernel
    let mut boxed = kernel.into_inner();
    let plan = KernelInterpolator::build(x_in_sl, x_out_sl, &mut *boxed, Some(scale))?;

    dispatch_unweighted(py, &plan, y_in, y_out)
}

/// Interpolate a 2D array with corresponding weights using a kernel.
///
/// Parameters
/// ----------
/// x_in
///     1D float64 sorted array with input sample indices.
/// x_out
///     1D float64 sorted array with output sample indices. Must have uniform spacing.
/// kernel
///     [`AnyKernel`] instance describing the interpolation kernel.
/// y_in
///     2D float or complex float array to be interpolated. Must be C-contiguous.
/// w_in
///     2D float array of inverse-variance sample weights. Weights are propagated by
///     inverting the propagated variances. Must be C-contiguous.
/// scale
///     Optional kernel scaling factor. The inverse of this value is multiplied with the
///     sample spacing before evaluating the kernel at each input sample. Default is `1.0`.
/// y_out
///     Optional 2D float or complex float array to store output. Must be C-contiguous.
///     If `None`, a new array is allocated.
/// w_out
///     Optional 2D float array to store propagated weights. Must be C-contiguous. If
///     `None`, a new array is allocated.
///
/// Returns
/// -------
/// y_out
///     2D float or complex float array, shape `(-1, n_out)`.
/// w_out
///     2D float array, shape `(-1, n_out)`.
///
/// Errors
/// ------
/// Raises `ValueError` when the kernel, coordinates, or weighting arrays are invalid;
/// raises `TypeError` for unsupported dtypes.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(module = "wyvern.interpolate")
)]
#[pyfunction]
#[pyo3(signature = (x_in, x_out, kernel, y_in, w_in, *, scale = 1.0, y_out = None, w_out = None))]
pub fn interpolate_kernel_weighted<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    kernel: AnyKernel,
    y_in: &Bound<'py, PyUntypedArray>,
    w_in: &Bound<'py, PyUntypedArray>,
    scale: f64,
    y_out: Option<&Bound<'py, PyUntypedArray>>,
    w_out: Option<&Bound<'py, PyUntypedArray>>,
) -> PyResult<(Py<PyUntypedArray>, Py<PyUntypedArray>)> {
    // validate and extract input samples
    let (x_in, x_out) = validate_extract_samples(py, x_in, x_out)?;
    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;

    // Build the interpolation plan based off of the inner kernel
    let mut boxed = kernel.into_inner();
    let plan = KernelInterpolator::build(x_in_sl, x_out_sl, &mut *boxed, Some(scale))?;

    dispatch_weighted(py, &plan, y_in, w_in, y_out, w_out)
}

/// Validate and extract the input and output coordinate arrays.
///
/// # Parameters
/// * `py`: The active Python interpreter.
/// * `x_in`: Input sample coordinate array.
/// * `x_out`: Output sample coordinate array.
///
/// # Returns
/// A pair of readonly float64 arrays.
///
/// # Errors
/// Returns a `ValueError` when either coordinate array has unsupported dimensions or
/// dtypes.
fn validate_extract_samples<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
) -> PyResult<(PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>)> {
    // validate input samples
    require_ndim(Some(x_in), "x_in", 1)?;
    require_ndim(Some(x_out), "x_out", 1)?;
    require_dtype(Some(x_in), "x_in", &dtype::<f64>(py))?;
    require_dtype(Some(x_out), "x_out", &dtype::<f64>(py))?;

    // extract and call
    let x_in: PyReadonlyArray1<f64> = x_in.extract()?;
    let x_out: PyReadonlyArray1<f64> = x_out.extract()?;

    Ok((x_in, x_out))
}
