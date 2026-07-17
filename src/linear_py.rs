//! Python wrapper for linear interpolator.
use numpy::{
    Complex32, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Validate input dimension
fn require_ndim(arr: &Bound<'_, PyUntypedArray>, name: &str, expected_dim: usize) -> PyResult<()> {
    let ndim = arr.ndim();
    if ndim != expected_dim {
        return Err(PyValueError::new_err(format!(
            "'{name}' must be a {expected_dim}D array; got {ndim}D with shape {:?}",
            arr.shape()
        )));
    }
    Ok(())
}

/// Linearly interpolate a 2D array with corresponding weights.
///
/// Parameters
/// ----------
/// ``x_in``, ``x_out`` : 1D float64 arrays (sorted; ``x_out`` should have uniform spacing)
/// ``y_in``: 2D float array, shape (-1, ``n_in``)
/// ``w_in`` : 2D float array, same shape as ``y_in``
///
/// Returns
/// -------
/// ``y_out``,``w_out`` : 2D float arrays, shape (-1, ``n_out``)
#[allow(clippy::pedantic, reason = "required for numpy interop")]
#[pyfunction]
pub fn interpolate_linear<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    y_in: &Bound<'py, PyUntypedArray>,
    w_in: &Bound<'py, PyUntypedArray>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    // bounds checks, type checks, etc...
    require_ndim(y_in, "y_in", 2)?;
    require_ndim(w_in, "w_in", 2)?;
    require_ndim(x_in, "x_in", 1)?;
    require_ndim(x_out, "x_out", 1)?;

    // extract and call
    let x_in: PyReadonlyArray1<f32> = x_in.extract()?;
    let x_out: PyReadonlyArray1<f32> = x_out.extract()?;
    let w_in: PyReadonlyArray2<f32> = w_in.extract()?;

    interpolate_linear_unvalidated(py, &x_in, &x_out, y_in, &w_in)
}

/// Interpolate without clean python errors.
fn interpolate_linear_unvalidated<'py>(
    py: Python<'py>,
    x_in: &PyReadonlyArray1<f32>,
    x_out: &PyReadonlyArray1<f32>,
    y_in: &Bound<'py, PyUntypedArray>,
    w_in: &PyReadonlyArray2<f32>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    require_ndim(y_in, "y_in", 2)?;
    // extract input and output sample positions
    let x_in_sl = x_in.as_slice()?;
    let x_out_sl = x_out.as_slice()?;

    // extract weights, which are always real
    let w_in_view = w_in.as_array();
    // output dimensions
    let n_out = x_out_sl.len();
    let rows = w_in_view.nrows();

    // init output weights array
    let w_out = PyArray2::<f32>::zeros(py, (rows, n_out), false);
    // and a mutable view
    let w_out_view = unsafe { w_out.as_array_mut() };

    if y_in.dtype().is_equiv_to(&dtype::<f32>(py)) {
        // data is real
        let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
        let y_in_view = y_in.as_array();
        // allocate output arrays directly on the numpy heap, so
        // no intermediate rust-owned buffer is copied afterwards.
        let y_out = PyArray2::<f32>::zeros(py, (rows, n_out), false);
        let y_out_view = unsafe { y_out.as_array_mut() };

        // release the GIL
        py.detach(|| -> eyre::Result<()> {
            crate::linear::interp_last_ax_real(
                x_in_sl, x_out_sl, &y_in_view, &w_in_view, y_out_view, w_out_view,
            )
        })?;

        return Ok((y_out.into_any().unbind(), w_out.into_any().unbind()));
    }

    if y_in.dtype().is_equiv_to(&dtype::<Complex32>(py)) {
        // data is complex. interpolate real and imag independently
        let y_in: PyReadonlyArray2<Complex32> = y_in.extract()?;
        let (yre_i, yim_i) = unsafe { crate::types::split_complex_view(&y_in.as_array()) };
        // allocate outputs
        let y_out = PyArray2::<Complex32>::zeros(py, (rows, n_out), false);
        let (yre_o, yim_o) = unsafe { crate::types::split_complex_view_mut(&y_out.as_array_mut()) };

        py.detach(|| -> eyre::Result<()> {
            crate::linear::interp_last_ax_complex(
                x_in_sl, x_out_sl, &yre_i, &yim_i, &w_in_view, yre_o, yim_o, w_out_view,
            )
        })?;

        return Ok((y_out.into_any().unbind(), w_out.into_any().unbind()));
    }

    Err(PyTypeError::new_err(format!(
        "'y_in' has unsupported type '{:?}'. supported types are: float32, complex64.",
        y_in.dtype()
    )))
}
