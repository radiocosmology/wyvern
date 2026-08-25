//! Utilities for interacting with Python arrays
use num_traits::Zero;
use numpy::borrow::PyReadwriteArray2;
use numpy::{
    Element, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Ensure that an array exists, and create it if it does not
pub fn ensure_array<'py, T: Element + Zero>(
    py: Python<'py>,
    arr: Option<&Bound<'py, PyUntypedArray>>,
    shape: [usize; 2],
) -> PyResult<PyReadwriteArray2<'py, T>> {
    // extract an existing array or create a new one
    let out: PyReadwriteArray2<'py, T> = match arr {
        Some(untyped) => untyped.extract()?,
        None => PyArray2::<T>::zeros(py, shape, false).readwrite(),
    };

    if out.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "Invalid array shape! Expected {shape:?}, got {:?}",
            out.shape()
        )));
    }

    if !out.dtype().is_equiv_to(&dtype::<T>(py)) {
        return Err(PyTypeError::new_err(format!(
            "Invalid array type! Expected {:?}, got {:?}",
            out.dtype(),
            dtype::<T>(py)
        )));
    }

    Ok(out)
}

/// Validate input dimension and return a python-compatible error
pub fn require_ndim(
    arr: Option<&Bound<'_, PyUntypedArray>>,
    name: &str,
    expected_dim: usize,
) -> PyResult<()> {
    let Some(arr) = arr else {
        return Ok(());
    };
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
pub fn require_dtype(
    arr: Option<&Bound<'_, PyUntypedArray>>,
    name: &str,
    expected_dtype: &Bound<'_, PyArrayDescr>,
) -> PyResult<()> {
    let Some(arr) = arr else {
        return Ok(());
    };
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
