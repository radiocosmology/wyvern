//! Utilities for interacting with Python arrays
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Validate input dimension and return a python-compatible error
pub fn require_ndim(
    arr: &Bound<'_, PyUntypedArray>,
    name: &str,
    expected_dim: usize,
) -> PyResult<()> {
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
