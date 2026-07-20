//! Python wrapper for linear interpolator.
use num_complex::Complex;
use num_traits::Float;
use numpy::{
    Element, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Validate input dimension and return a python-compatible error
fn require_ndim(arr: &Bound<'_, PyUntypedArray>, name: &str, expected_dim: usize) -> PyResult<()> {
    // NB: without this check, an invalid dimension just returns
    // "`ndarray` does not have type ndarray" when called from
    // Python, which isn't very helpful
    if arr.ndim() != expected_dim {
        return Err(PyValueError::new_err(format!(
            "'{name}' must be a {expected_dim}D array; got {:?}D with shape {:?}",
            arr.ndim(),
            arr.shape()
        )));
    }
    Ok(())
}

/// Validate dtype and return a python-compatible error
fn require_dtype(
    arr: &Bound<'_, PyUntypedArray>,
    name: &str,
    expected_dtype: &Bound<'_, PyArrayDescr>,
) -> PyResult<()> {
    // NB: without this check, an invalid dtype just returns
    // "`ndarray` does not have type ndarray" when called from
    // Python, which isn't very helpful
    if !arr.dtype().is_equiv_to(expected_dtype) {
        return Err(PyValueError::new_err(format!(
            "'{name}' must have type {expected_dtype}; got {:?}",
            arr.dtype(),
        )));
    }
    Ok(())
}

/// Separate methods for interpolating real and complex, weighted
/// and unweighted
fn interpolate_real<'py, T>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, T>,
) -> PyResult<Py<PyAny>>
where
    T: Float + Element + Sync + Send,
{
    let y_in_view = y_in.as_array();

    let rows = y_in_view.nrows();
    let n_out = x_out.len();

    // allocate outputs and according views
    let y_out = PyArray2::<T>::zeros(py, (rows, n_out), false);
    let y_out_view = unsafe { y_out.as_array_mut() };

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_real(x_in, x_out, &y_in_view, y_out_view)
    })?;

    Ok(y_out.into_any().unbind())
}

fn interpolate_real_weighted<'py, T, W>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, T>,
    w_in: &PyReadonlyArray2<'py, W>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)>
where
    T: Float + Element + Sync + Send,
    W: Float + Element + Sync + Send,
{
    let y_in_view = y_in.as_array();
    let w_in_view = w_in.as_array();

    let rows = y_in_view.nrows();
    let n_out = x_out.len();

    // allocate outputs and according views
    let y_out = PyArray2::<T>::zeros(py, (rows, n_out), false);
    let w_out = PyArray2::<W>::zeros(py, (rows, n_out), false);

    let w_out_view = unsafe { w_out.as_array_mut() };
    let y_out_view = unsafe { y_out.as_array_mut() };

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_real_weighted(
            x_in, x_out, &y_in_view, &w_in_view, y_out_view, w_out_view,
        )
    })?;

    Ok((y_out.into_any().unbind(), w_out.into_any().unbind()))
}

fn interpolate_complex<'py, T>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
) -> PyResult<Py<PyAny>>
where
    T: Float + Element + Sync + Send,
    Complex<T>: Element,
{
    let (yre_in, yim_in) = unsafe { crate::types::split_complex_view(&y_in.as_array()) };

    let rows = yre_in.nrows();
    let n_out = x_out.len();

    // allocate outputs and according views
    let y_out = PyArray2::<Complex<T>>::zeros(py, (rows, n_out), false);
    let (yre_out, yim_out) = unsafe { crate::types::split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_complex(x_in, x_out, &yre_in, &yim_in, yre_out, yim_out)
    })?;

    Ok(y_out.into_any().unbind())
}

fn interpolate_complex_weighted<'py, T, W>(
    py: Python<'py>,
    x_in: &[f64],
    x_out: &[f64],
    y_in: &PyReadonlyArray2<'py, Complex<T>>,
    w_in: &PyReadonlyArray2<'py, W>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)>
where
    T: Float + Element + Sync + Send,
    W: Float + Element + Sync + Send,
    Complex<T>: Element,
{
    let (yre_in, yim_in) = unsafe { crate::types::split_complex_view(&y_in.as_array()) };
    let w_in_view = w_in.as_array();

    let rows = w_in_view.nrows();
    let n_out = x_out.len();

    // allocate outputs and according views
    let y_out = PyArray2::<Complex<T>>::zeros(py, (rows, n_out), false);
    let w_out = PyArray2::<W>::zeros(py, (rows, n_out), false);

    let w_out_view = unsafe { w_out.as_array_mut() };
    let (yre_out, yim_out) = unsafe { crate::types::split_complex_view_mut(&y_out.as_array_mut()) };

    py.detach(|| -> eyre::Result<()> {
        crate::linear::interp_last_ax_complex_weighted(
            x_in, x_out, &yre_in, &yim_in, &w_in_view, yre_out, yim_out, w_out_view,
        )
    })?;

    Ok((y_out.into_any().unbind(), w_out.into_any().unbind()))
}

/// Linearly interpolate a 2D array.
///
/// Parameters
/// ----------
/// ``x_in``, ``x_out`` : 1D float64 arrays (sorted; ``x_out`` should have uniform spacing)
/// ``y_in``: 2D float array, shape (-1, ``n_in``)
///
/// Returns
/// -------
/// ``y_out`` : 2D float array, shape (-1, ``n_out``)
#[allow(clippy::pedantic, reason = "required for numpy interop")]
#[pyfunction]
pub fn interpolate_linear<'py>(
    py: Python<'py>,
    x_in: &Bound<'py, PyUntypedArray>,
    x_out: &Bound<'py, PyUntypedArray>,
    y_in: &Bound<'py, PyUntypedArray>,
) -> PyResult<Py<PyAny>> {
    // bounds checks, type checks, etc...
    require_ndim(y_in, "y_in", 2)?;
    require_ndim(x_in, "x_in", 1)?;
    require_ndim(x_out, "x_out", 1)?;
    require_dtype(x_in, "x_in", &dtype::<f64>(py))?;
    require_dtype(x_out, "x_out", &dtype::<f64>(py))?;

    // extract and call
    let x_in: PyReadonlyArray1<f64> = x_in.extract()?;
    let x_in_sl = x_in.as_slice()?;
    let x_out: PyReadonlyArray1<f64> = x_out.extract()?;
    let x_out_sl = x_out.as_slice()?;

    // unfortunately, need to disdpatch based on types here. For
    // now, require that both data and weights have matching
    // bit depth
    let y_dtype = y_in.dtype();
    if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let y_in: PyReadonlyArray2<f32> = y_in.extract()?;

        return interpolate_real::<f32>(py, x_in_sl, x_out_sl, &y_in);
    }
    if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let y_in: PyReadonlyArray2<f64> = y_in.extract()?;

        return interpolate_real::<f64>(py, x_in_sl, x_out_sl, &y_in);
    }
    if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
        let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;

        return interpolate_complex::<f32>(py, x_in_sl, x_out_sl, &y_in);
    }
    if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
        let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;

        return interpolate_complex::<f64>(py, x_in_sl, x_out_sl, &y_in);
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
/// ``x_in``, ``x_out`` : 1D float64 arrays (sorted; ``x_out`` should have uniform spacing)
/// ``y_in``: 2D float array, shape (-1, ``n_in``)
/// ``w_in`` : 2D float array, same shape as ``y_in``
///
/// Returns
/// -------
/// ``y_out``,``w_out`` : 2D float arrays, shape (-1, ``n_out``)
#[allow(clippy::pedantic, reason = "required for numpy interop")]
#[pyfunction]
pub fn interpolate_linear_weighted<'py>(
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
    require_dtype(x_in, "x_in", &dtype::<f64>(py))?;
    require_dtype(x_out, "x_out", &dtype::<f64>(py))?;

    // extract and call
    let x_in: PyReadonlyArray1<f64> = x_in.extract()?;
    let x_in_sl = x_in.as_slice()?;
    let x_out: PyReadonlyArray1<f64> = x_out.extract()?;
    let x_out_sl = x_out.as_slice()?;

    // unfortunately, need to an annpying dispatch here
    let y_dtype = y_in.dtype();
    let w_dtype = w_in.dtype();

    if w_dtype.is_equiv_to(&dtype::<f32>(py)) {
        let w_in: PyReadonlyArray2<f32> = w_in.extract()?;

        if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
            let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
            return interpolate_real_weighted::<f32, f32>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
        if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
            let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
            return interpolate_real_weighted::<f64, f32>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
            return interpolate_complex_weighted::<f32, f32>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
            return interpolate_complex_weighted::<f64, f32>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
    }
    if w_dtype.is_equiv_to(&dtype::<f64>(py)) {
        let w_in: PyReadonlyArray2<f64> = w_in.extract()?;

        if y_dtype.is_equiv_to(&dtype::<f32>(py)) {
            let y_in: PyReadonlyArray2<f32> = y_in.extract()?;
            return interpolate_real_weighted::<f32, f64>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
        if y_dtype.is_equiv_to(&dtype::<f64>(py)) {
            let y_in: PyReadonlyArray2<f64> = y_in.extract()?;
            return interpolate_real_weighted::<f64, f64>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f32>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f32>> = y_in.extract()?;
            return interpolate_complex_weighted::<f32, f64>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
        if y_dtype.is_equiv_to(&dtype::<Complex<f64>>(py)) {
            let y_in: PyReadonlyArray2<Complex<f64>> = y_in.extract()?;
            return interpolate_complex_weighted::<f64, f64>(py, x_in_sl, x_out_sl, &y_in, &w_in);
        }
    }
    Err(PyTypeError::new_err(format!(
        "Unsupported data types! `y`: {y_dtype}, `w`: {w_dtype}. \
        Supported data types are: float32, float64, complex64, complex128. \
        Supported weight types are: float32, float64."
    )))
}
