//! Python wrapper for lanczos interpolation.
//! Currently supports the following kernel widths:
//! - 4
//! - 8
//! - 16
//! - 32
use numpy::{PyReadonlyArray1, PyUntypedArray, dtype};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use std::collections::{HashMap, VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};

use super::{dispatch_unweighted, dispatch_weighted, require_dtype, require_ndim};
use wyvern::interpolate::DynamicKernelPlan;
use wyvern::kernels::lanczos_kernel;

const PLAN_CACHE_LIMIT: usize = 32;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
struct LanczosPlanKey {
    x_in_hash: u64,
    x_out_hash: u64,
    n_in: usize,
    n_out: usize,
    n_taps: usize,
}

#[derive(Default)]
struct LanczosPlanCache {
    order: VecDeque<LanczosPlanKey>,
    plans: HashMap<LanczosPlanKey, Arc<DynamicKernelPlan>>,
}

fn hash_samples(samples: &[f64]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for sample in samples {
        sample.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn get_cached_lanczos_plan(
    x_in: &[f64],
    x_out: &[f64],
    n_taps: usize,
) -> PyResult<Arc<DynamicKernelPlan>> {
    static LANCZOS_PLAN_CACHE: LazyLock<Mutex<LanczosPlanCache>> =
        LazyLock::new(|| Mutex::new(LanczosPlanCache::default()));

    let key = LanczosPlanKey {
        x_in_hash: hash_samples(x_in),
        x_out_hash: hash_samples(x_out),
        n_in: x_in.len(),
        n_out: x_out.len(),
        n_taps,
    };

    let mut cache = LANCZOS_PLAN_CACHE
        .lock()
        .map_err(|_| PyRuntimeError::new_err("lanczos plan cache lock poisoned"))?;
    if let Some(plan) = cache.plans.get(&key) {
        return Ok(Arc::clone(plan));
    }

    let plan = Arc::new(DynamicKernelPlan::build(
        x_in,
        x_out,
        n_taps,
        lanczos_kernel,
    )?);
    cache.order.push_back(key);
    cache.plans.insert(key, Arc::clone(&plan));
    if cache.order.len() > PLAN_CACHE_LIMIT
        && let Some(evict) = cache.order.pop_front()
    {
        cache.plans.remove(&evict);
    }

    Ok(plan)
}

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
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(module = "wyvern.interpolate")
)]
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

    let plan = get_cached_lanczos_plan(x_in_sl, x_out_sl, n_taps)?;

    dispatch_unweighted(py, plan.as_ref(), y_in, y_out)
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
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(module = "wyvern.interpolate")
)]
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

    let plan = get_cached_lanczos_plan(x_in_sl, x_out_sl, n_taps)?;

    dispatch_weighted(py, plan.as_ref(), y_in, w_in, y_out, w_out)
}
