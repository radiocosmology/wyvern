//! Python wrapper for lanczos interpolation.
//! Currently supports the following kernel widths:
//! - 4
//! - 8
//! - 16
//! - 32
use numpy::{PyReadonlyArray1, PyUntypedArray, dtype};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{dispatch_unweighted, dispatch_weighted, require_dtype, require_ndim};
use crate::core::KernelPlan;
use crate::kernels::lanczos_kernel;

/// Interpolate a 2D array using a Lanczos kernel.
///
/// Parameters
/// ----------
/// ``x_in``
///     1D float64 sorted array with input sample indices
/// ``x_out``
///     1D float64 sorted array with output sample indices. Must
///     have uniform spacing.
/// ``n_taps``
///     Lanczos kernel taps. Must be one of {4, 8, 16, 32}.
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
#[pyo3(signature = (x_in, x_out, n_taps, y_in, *, y_out = None))]
pub fn interpolate_lanczos<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    n_taps: usize,
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

    // macro to simplify dispatch for various fixed widths
    macro_rules! build_and_dispatch_fixed_taps {
        ($n_taps:expr, widths = [ $( $n:literal ), + $(,)? ]) => {{
            pub const SUPPORTED_TAP_WIDTHS: &[usize] = &[ $( $n ),+ ];
            match $n_taps {
                $(
                    $n => {
                        let plan = KernelPlan::<$n>::build(x_in_sl, x_out_sl, lanczos_kernel)?;
                        dispatch_unweighted(py, &plan, y_in, y_out)
                    }
                )+
                other => Err(PyValueError::new_err(format!(
                    "Unsupported kernel width {other}; supported: {:?}", SUPPORTED_TAP_WIDTHS
                ))),
            }
        }};
    }
    // dispatch to fixed kernel widths
    build_and_dispatch_fixed_taps!(n_taps, widths = [4, 8, 16, 32])
}

/// Interpolate a 2D array with corresponding weights using a Lanczos kernel.
///
/// Parameters
/// ----------
/// ``x_in``
///     1D float64 sorted array with input sample indices
/// ``x_out``
///     1D float64 sorted array with output sample indices. Must
///     have uniform spacing.
/// ``n_taps``
///     Lanczos kernel taps. Must be one of {4, 8, 16, 32}.
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
#[pyo3(signature = (x_in, x_out, n_taps, y_in, w_in, *, y_out = None, w_out = None))]
pub fn interpolate_lanczos_weighted<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    n_taps: usize,
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

    // macro to simplify dispatch for various fixed widths
    macro_rules! build_and_dispatch_fixed_taps {
        ($n_taps:expr, widths = [ $( $n:literal ), + $(,)? ]) => {{
            pub const SUPPORTED_TAP_WIDTHS: &[usize] = &[ $( $n ),+ ];
            match $n_taps {
                $(
                    $n => {
                        let plan = KernelPlan::<$n>::build(x_in_sl, x_out_sl, lanczos_kernel)?;
                        dispatch_weighted(py, &plan, y_in, w_in, y_out, w_out)
                    }
                )+
                other => Err(PyValueError::new_err(format!(
                    "Unsupported kernel width {other}; supported: {:?}", SUPPORTED_TAP_WIDTHS
                ))),
            }
        }};
    }
    // dispatch to fixed kernel widths
    build_and_dispatch_fixed_taps!(n_taps, widths = [4, 8, 16, 32])
}
