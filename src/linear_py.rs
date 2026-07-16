//! Python wrapper for linear interpolator.
use numpy::ndarray::Axis;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::linear::interp_last_ax_lin;

/// Linearly interpolate a 2D array with corresponding weights.
///
/// Parameters
/// ----------
/// ``x_in``, ``x_out`` : 1D float64 arrays (sorted; ``x_out`` must have uniform spacing)
/// ``y_in``: 2D float array, shape (-1, ``n_in``)
/// ``w_in`` : 2D float array, same shape as ``y_in``
///
/// Returns
/// -------
/// ``y_out``,``w_out`` : 2D float arrays, shape (-1, ``n_out``)
#[allow(clippy::pedantic, reason = "required for numpy interop")]
#[allow(clippy::type_complexity, reason = "required for numpy interop")]
#[pyfunction]
pub fn interpolate_linear<'py>(
    py: Python<'py>,
    x_in: PyReadonlyArray1<f64>,
    x_out: PyReadonlyArray1<f64>,
    y_in: PyReadonlyArray2<f64>,
    w_in: PyReadonlyArray2<f64>,
) -> eyre::Result<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;

    let y_in_view = y_in.as_array();
    let w_in_view = w_in.as_array();

    let rows = y_in_view.len_of(Axis(0));
    let n_out = x_out_sl.len();

    // allocate output arrays directly on the numpy heap, so
    // no intermediate rust-owned buffer is copied afterwards.
    let y_out = PyArray2::<f64>::zeros(py, (rows, n_out), false);
    let w_out = PyArray2::<f64>::zeros(py, (rows, n_out), false);

    let y_out_view = unsafe { y_out.as_array_mut() };
    let w_out_view = unsafe { w_out.as_array_mut() };

    // release the GIL
    py.detach(|| -> eyre::Result<()> {
        interp_last_ax_lin(
            x_in_sl, x_out_sl, &y_in_view, &w_in_view, y_out_view, w_out_view,
        )
    })?;

    Ok((y_out, w_out))
}
