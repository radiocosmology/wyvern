//! Utilities for interacting with Python arrays
use num_traits::Zero;
use numpy::borrow::PyReadwriteArray2;
use numpy::{
    Element, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods, dtype,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Ensure that a 2D array exists and matches the expected shape and dtype.
///
/// # Parameters
/// * `py`: The active Python interpreter.
/// * `arr`: An existing array to validate, or `None` to allocate a new one.
/// * `shape`: The expected two-dimensional shape of the array.
///
/// # Returns
/// A writable NumPy array with the requested shape and dtype.
///
/// # Errors
/// Returns a Python `ValueError` if the shape does not match, or a `TypeError`
/// if the array dtype is incompatible.
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

/// Validate that an array has the expected dimensionality.
///
/// # Parameters
/// * `arr`: The array to validate, or `None` if the parameter is optional.
/// * `name`: Human-readable name of the array for diagnostics.
/// * `expected_dim`: The number of dimensions the array must have.
///
/// # Returns
/// `Ok(())` when the array matches the expected dimensionality.
///
/// # Errors
/// Returns a Python `ValueError` when the provided array has an unexpected rank.
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

/// Validate that an array has the expected dtype.
///
/// # Parameters
/// * `arr`: The array to validate, or `None` if the parameter is optional.
/// * `name`: Human-readable name of the array for diagnostics.
/// * `expected_dtype`: The dtype that the array must match exactly.
///
/// # Returns
/// `Ok(())` when the array satisfies the expected dtype.
///
/// # Errors
/// Returns a Python `ValueError` when the array has an incompatible dtype.
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
