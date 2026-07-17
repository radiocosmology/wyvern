//! Python wrapper for linear interpolator.
use numpy::{
    Complex32, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::linear::interp_last_ax_lin;

/// Validate input dimension
fn require_2d(arr: &Bound<'_, PyUntypedArray>, name: &str) -> PyResult<()> {
    let ndim = arr.ndim();
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "'{name}' must be a 2D array; got {ndim}D with shape {:?}",
            arr.shape()
        )));
    }
    Ok(())
}

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
    x_in: PyReadonlyArray1<f32>,
    x_out: PyReadonlyArray1<f32>,
    y_in: &Bound<'py, PyUntypedArray>,
    w_in: &Bound<'py, PyUntypedArray>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    require_2d(y_in, "y_in")?;
    require_2d(w_in, "w_in")?;

    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;

    let n_out = x_out_sl.len();
    let element_type = y_in.dtype();

    if element_type.is_equiv_to(&dtype::<f32>(py)) {
        let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
        let w_in: PyReadonlyArray2<f32> = w_in.extract()?;

        let rows = y_in.as_array().nrows();

        // allocate output arrays directly on the numpy heap, so
        // no intermediate rust-owned buffer is copied afterwards.
        let y_out = PyArray2::<f32>::zeros(py, (rows, n_out), false);
        let w_out = PyArray2::<f32>::zeros(py, (rows, n_out), false);

        // need to work with views created before branching
        let y_in_view = y_in.as_array();
        let w_in_view = w_in.as_array();
        let y_out_view = unsafe { y_out.as_array_mut() };
        let w_out_view = unsafe { w_out.as_array_mut() };

        // release the GIL
        py.detach(|| -> eyre::Result<()> {
            interp_last_ax_lin(
                x_in_sl, x_out_sl, &y_in_view, &w_in_view, y_out_view, w_out_view,
            )
        })?;

        return Ok((y_out.into_any().unbind(), w_out.into_any().unbind()));
    }

    if element_type.is_equiv_to(&dtype::<Complex32>(py)) {
        println!("err");
    }

    Err(PyTypeError::new_err(format!(
        "'y_in' has unsupported type '{element_type}'. supported types are: float32, complex64."
    )))
}
