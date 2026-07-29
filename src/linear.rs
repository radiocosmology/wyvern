//! Linear interpolation across the last axis.
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use rayon::prelude::*;
use std::cell::UnsafeCell;

use crate::types::{FloatLike, ParFloatLike};

/// Wrapper to allow &mut access into once Vec<f64> slot from
/// multiple threads without a mutex.
struct ScratchSlot(UnsafeCell<Vec<f64>>);
unsafe impl Sync for ScratchSlot {}

fn make_scratch_pool(n_in: usize) -> Vec<ScratchSlot> {
    let num_threads = rayon::current_num_threads();
    (0..num_threads)
        .map(|_| ScratchSlot(UnsafeCell::new(vec![0.0_f64; n_in])))
        .collect()
}

/// Precomputed interpolation plan for mapping input
/// and output samples.
struct InterpolationPlanLinear {
    // lower bracket index for input
    i0: Vec<usize>,
    // upper brackeet index for input
    i1: Vec<usize>,
    // interpolation coefficient for i1 sample (w0 = 1 - w1)
    c1: Vec<f64>,
    // mask for valid samples. 1.0 if valid, 0.0 otherwise
    valid: Vec<f64>,
}

impl InterpolationPlanLinear {
    /// ``x_in``: sorted, arbitrary spacing, len >= 2
    /// ``x_out``: sorted, uniform spacing, len >= 1
    pub fn build(x_in: &[f64], x_out: &[f64]) -> eyre::Result<Self> {
        let n_out = x_out.len();
        let n_in = x_in.len();

        debug_assert!(n_in >= 2, "minimum 2 input samples are required!");
        debug_assert!(n_out >= 1, "minimum 1 output sample is required!");

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut i1 = Vec::<usize>::with_capacity(n_out);
        let mut c1 = Vec::<f64>::with_capacity(n_out);
        let mut valid = Vec::<f64>::with_capacity(n_out);

        // both inputs are sorted, so step only advances forward. error
        // is eventually returned if this assumption fails
        let mut lo: usize = 0;
        let lo_max: usize = n_in - 2;

        let delta = crate::utils::median_abs_sample_spacing(x_in);

        for &xo in x_out {
            // advance the pointer while the next pair still brackets xo,
            // or we're at the last valid pair
            #[allow(
                clippy::indexing_slicing,
                reason = "max index is 2 less than `x_in.len()`"
            )]
            let (a, b) = {
                // move forward to the next target sample
                while lo < lo_max && x_in[lo + 1] <= xo {
                    lo += 1;
                }
                // fetch the input sample values
                (x_in[lo], x_in[lo + 1])
            };

            let span = b - a;
            if span <= 0.0 {
                eyre::bail!("inputs are unsorted or repeated!");
            }

            // check if this is more than one input spacing from
            // either input sample
            let distant = (b - xo).abs() > delta || (a - xo).abs() > delta;
            let v = f64::from(!distant);

            // interpolation indices
            i0.push(lo);
            i1.push(lo + 1);
            // weight coefficient
            c1.push((xo - a) / span);
            // mask
            valid.push(v);
        }

        Ok(Self { i0, i1, c1, valid })
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.i0.len()
    }
}

/// Apply a pre-computed plan to a single array row
/// with accompanied inverse variance weights.
#[inline]
fn interp_row_with_variance<T: FloatLike>(
    plan: &InterpolationPlanLinear,
    y_in: &ArrayView1<T>,
    weight_in: &ArrayView1<T>,
    var_scratch: &mut [f64],
    mut y_out: ArrayViewMut1<T>,
    mut weight_out: ArrayViewMut1<T>,
) {
    let n_in = y_in.len();
    let n_out = plan.len();

    debug_assert_eq!(n_in, weight_in.len());
    debug_assert_eq!(n_in, var_scratch.len());
    debug_assert_eq!(n_out, y_out.len());
    debug_assert_eq!(n_out, weight_out.len());

    unsafe {
        // invert weights once per pass, since input samples are often re-used
        // 1.0 / 0.0 == +inf under IEEE754, no panic, will revert to 0.0
        // when re-inverted to weights
        for k in 0..n_in {
            let w: f64 = (*weight_in.uget(k)).as_();
            *var_scratch.get_unchecked_mut(k) = 1.0 / w;
        }

        for j in 0..n_out {
            // extract the interpolation indices and weight
            let i0 = *plan.i0.get_unchecked(j);
            let i1 = *plan.i1.get_unchecked(j);
            // interpolation coefficients
            let s1 = *plan.c1.get_unchecked(j);

            // interpolate data onto the target sample
            let a: f64 = (*y_in.uget(i0)).as_();
            let b: f64 = (*y_in.uget(i1)).as_();
            *y_out.uget_mut(j) = T::from_f64((b - a).mul_add(s1, a));

            // propagate weights and masking
            let var_a = *var_scratch.get_unchecked(i0);
            let var_b = *var_scratch.get_unchecked(i1);
            let valid = *plan.valid.get_unchecked(j);

            let s0 = 1.0 - s1;
            // NaN guard: (0.0 * inf) -> NaN -> clamped to 0.0
            // for invalid items
            let c0 = (s0 * s0 * var_a).max(0.0);
            let c1 = (s1 * s1 * var_b).max(0.0);
            // keep is either 1.0 or 0.0
            *weight_out.uget_mut(j) = T::from_f64(valid / (c0 + c1));
        }
    }
}

/// Apply a pre-computed plan to a single array row.
#[inline]
fn interp_row<T: FloatLike>(
    plan: &InterpolationPlanLinear,
    y_in: &ArrayView1<T>,
    mut y_out: ArrayViewMut1<T>,
) {
    let n_out = plan.len();

    debug_assert_eq!(n_out, y_out.len());

    unsafe {
        for j in 0..n_out {
            // extract the interpolation indices and weight
            let i0 = *plan.i0.get_unchecked(j);
            let i1 = *plan.i1.get_unchecked(j);
            // interpolation coefficients
            let s1 = *plan.c1.get_unchecked(j);

            // interpolate data onto the target sample
            let a = (*y_in.uget(i0)).as_();
            let b = (*y_in.uget(i1)).as_();
            *y_out.uget_mut(j) = T::from_f64((b - a).mul_add(s1, a));
        }
    }
}

/// Interpolate over the last axis of a real array.
#[inline]
pub fn interp_last_ax_real<T: ParFloatLike>(
    x_in: &[f64],
    x_out: &[f64],
    y_in: &ArrayView2<T>,
    mut y_out: ArrayViewMut2<T>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_in.rows())
        .and(y_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, yo)| {
            interp_row(&plan, &yi, yo);
        });

    Ok(())
}

/// Interpolate over the last axis of a complex array
/// with accompanying weights
#[allow(clippy::too_many_arguments, reason = "inline helper function")]
#[inline]
pub fn interp_last_ax_complex<T: ParFloatLike>(
    x_in: &[f64],
    x_out: &[f64],
    y_re_in: &ArrayView2<T>,
    y_im_in: &ArrayView2<T>,
    mut y_re_out: ArrayViewMut2<T>,
    mut y_im_out: ArrayViewMut2<T>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_re_in.rows())
        .and(y_im_in.rows())
        .and(y_re_out.rows_mut())
        .and(y_im_out.rows_mut())
        .into_par_iter()
        .for_each(|(yre_i, yim_i, yre_o, yim_o)| {
            interp_row(&plan, &yre_i, yre_o);
            interp_row(&plan, &yim_i, yim_o);
        });

    Ok(())
}

/// Interpolate over the last axis of a real array
/// with accompanying weights.
#[inline]
pub fn interp_last_ax_real_weighted<T: ParFloatLike>(
    x_in: &[f64],
    x_out: &[f64],
    y_in: &ArrayView2<T>,
    weight_in: &ArrayView2<T>,
    mut y_out: ArrayViewMut2<T>,
    mut weight_out: ArrayViewMut2<T>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;
    // update the scratch buffer size
    let n_in = y_in.ncols();
    let scratch_pool = make_scratch_pool(n_in);
    let num_threads = scratch_pool.len();

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_in.rows())
        .and(weight_in.rows())
        .and(y_out.rows_mut())
        .and(weight_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, wi, yo, wo)| {
            // each thread owns one slot in the scratch buffer
            let sslot = rayon::current_thread_index().unwrap_or(0) % num_threads;
            #[allow(clippy::indexing_slicing, reason = "buffer size explicitly set")]
            let buf: &mut Vec<f64> = unsafe { &mut *scratch_pool[sslot].0.get() };

            interp_row_with_variance(&plan, &yi, &wi, buf, yo, wo);
        });

    Ok(())
}

/// Interpolate over the last axis of a complex array
/// with accompanying weights
#[allow(clippy::too_many_arguments, reason = "inline helper function")]
#[inline]
pub fn interp_last_ax_complex_weighted<T: ParFloatLike>(
    x_in: &[f64],
    x_out: &[f64],
    y_re_in: &ArrayView2<T>,
    y_im_in: &ArrayView2<T>,
    weight_in: &ArrayView2<T>,
    mut y_re_out: ArrayViewMut2<T>,
    mut y_im_out: ArrayViewMut2<T>,
    mut weight_out: ArrayViewMut2<T>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;
    // update the scratch buffer size
    let n_in = y_re_in.ncols();
    let scratch_pool = make_scratch_pool(n_in);
    let num_threads = scratch_pool.len();

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_re_in.rows())
        .and(y_im_in.rows())
        .and(weight_in.rows())
        .and(y_re_out.rows_mut())
        .and(y_im_out.rows_mut())
        .and(weight_out.rows_mut())
        .into_par_iter()
        .for_each(|(yre_i, yim_i, wi, yre_o, yim_o, wo)| {
            // each thread owns one slot in the scratch buffer
            let sslot = rayon::current_thread_index().unwrap_or(0) % num_threads;
            #[allow(clippy::indexing_slicing, reason = "buffer size explicitly set")]
            let buf: &mut Vec<f64> = unsafe { &mut *scratch_pool[sslot].0.get() };

            interp_row_with_variance(&plan, &yre_i, &wi, buf, yre_o, wo);
            interp_row(&plan, &yim_i, yim_o);
        });

    Ok(())
}
