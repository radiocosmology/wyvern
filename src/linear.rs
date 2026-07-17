//! Linear interpolation across the last axis.
use std::cell::UnsafeCell;

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use rayon::prelude::*;

/// Wrapper to allow &mut access into once Vec<f32> slot from
/// multiple threads without a mutex.
struct ScratchSlot(UnsafeCell<Vec<f32>>);
unsafe impl Sync for ScratchSlot {}

fn make_scratch_pool(n_in: usize) -> Vec<ScratchSlot> {
    let num_threads = rayon::current_num_threads();
    (0..num_threads)
        .map(|_| ScratchSlot(UnsafeCell::new(vec![0.0_f32; n_in])))
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
    c1: Vec<f32>,
    // mask for valid samples. 1.0 if valid, 0.0 otherwise
    valid: Vec<f32>,
}

impl InterpolationPlanLinear {
    /// ``x_in``: sorted, arbitrary spacing, len >= 2
    /// ``x_out``: sorted, uniform spacing, len >= 1
    pub fn build(x_in: &[f32], x_out: &[f32]) -> eyre::Result<Self> {
        let n_out = x_out.len();
        let n_in = x_in.len();

        debug_assert!(n_in >= 2, "minimum 2 input samples are required!");
        debug_assert!(n_out >= 1, "minimum 1 output sample is required!");

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut i1 = Vec::<usize>::with_capacity(n_out);
        let mut c1 = Vec::<f32>::with_capacity(n_out);
        let mut valid = Vec::<f32>::with_capacity(n_out);

        // both inputs are sorted, so step only advances forward. error
        // is eventually returned if this assumption fails
        let mut lo: usize = 0;
        let lo_max: usize = n_in - 2;

        let delta = crate::utils::median_abs_diff(x_in);

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
            let v = f32::from(!distant);

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
fn interp_row_with_variance(
    plan: &InterpolationPlanLinear,
    y_in: &ArrayView1<f32>,
    weight_in: &ArrayView1<f32>,
    var_scratch: &mut [f32],
    mut y_out: ArrayViewMut1<f32>,
    mut weight_out: ArrayViewMut1<f32>,
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
            *var_scratch.get_unchecked_mut(k) = 1.0 / *weight_in.uget(k);
        }

        for j in 0..n_out {
            // extract the interpolation indices and weight
            let i0 = *plan.i0.get_unchecked(j);
            let i1 = *plan.i1.get_unchecked(j);
            // interpolation coefficients
            let s1 = *plan.c1.get_unchecked(j);

            // interpolate data onto the target sample
            let a = *y_in.uget(i0);
            let b = *y_in.uget(i1);
            *y_out.uget_mut(j) = (b - a).mul_add(s1, a);

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
            *weight_out.uget_mut(j) = valid / (c0 + c1);
        }
    }
}

/// Apply a pre-computed plan to a single array row.
#[inline]
fn interp_row(
    plan: &InterpolationPlanLinear,
    y_in: &ArrayView1<f32>,
    mut y_out: ArrayViewMut1<f32>,
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
            let a = *y_in.uget(i0);
            let b = *y_in.uget(i1);
            *y_out.uget_mut(j) = (b - a).mul_add(s1, a);
        }
    }
}

/// Interpolate over the last axis of a real array.
#[inline]
pub fn interp_last_ax_real(
    x_in: &[f32],
    x_out: &[f32],
    y_in: &ArrayView2<f32>,
    weight_in: &ArrayView2<f32>,
    mut y_out: ArrayViewMut2<f32>,
    mut weight_out: ArrayViewMut2<f32>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;
    // update the scratch buffer size
    let n_in = y_in.ncols();
    let scratch_pool = make_scratch_pool(n_in);
    let num_threads = scratch_pool.len();

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    #[allow(clippy::indexing_slicing, reason = "buffer size explicitly set")]
    Zip::from(y_in.rows())
        .and(weight_in.rows())
        .and(y_out.rows_mut())
        .and(weight_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, wi, yo, wo)| {
            // each thread owns one slot in the scratch buffer
            let sslot = rayon::current_thread_index().unwrap_or(0) % num_threads;
            let buf: &mut Vec<f32> = unsafe { &mut *scratch_pool[sslot].0.get() };
            interp_row_with_variance(&plan, &yi, &wi, buf, yo, wo);
        });

    Ok(())
}

/// Interpolate over the last axis of a complex array
#[allow(clippy::too_many_arguments, reason = "inline helper function")]
#[inline]
pub fn interp_last_ax_complex(
    x_in: &[f32],
    x_out: &[f32],
    y_re_in: &ArrayView2<f32>,
    y_im_in: &ArrayView2<f32>,
    weight_in: &ArrayView2<f32>,
    mut y_re_out: ArrayViewMut2<f32>,
    mut y_im_out: ArrayViewMut2<f32>,
    mut weight_out: ArrayViewMut2<f32>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;
    // update the scratch buffer size
    let n_in = y_re_in.ncols();
    let scratch_pool = make_scratch_pool(n_in);
    let num_threads = scratch_pool.len();

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    #[allow(clippy::indexing_slicing, reason = "buffer size explicitly set")]
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
            let buf: &mut Vec<f32> = unsafe { &mut *scratch_pool[sslot].0.get() };
            interp_row_with_variance(&plan, &yre_i, &wi, buf, yre_o, wo);
            interp_row(&plan, &yim_i, yim_o);
        });

    Ok(())
}
