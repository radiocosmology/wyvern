//! Linear interpolation across the last axis.
use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;
use std::cell::UnsafeCell;

use super::plan::Interpolator;
use crate::types::ParFloatLike;

/// Wrapper to allow &mut access into one `Vec<f64>` slot from
/// multiple threads without a mutex.
struct ScratchSlot(UnsafeCell<Vec<f64>>);
unsafe impl Sync for ScratchSlot {}

fn make_scratch_pool(n_in: usize) -> Vec<ScratchSlot> {
    let num_threads = rayon::current_num_threads();
    (0..num_threads)
        .map(|_| ScratchSlot(UnsafeCell::new(vec![0.0_f64; n_in])))
        .collect()
}

/// Interpolate over the last axis of a real array.
#[inline]
pub fn interp_last_ax_real<T>(
    interpolator: &dyn Interpolator<T>,
    y_in: &ArrayView2<T>,
    mut y_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_in.rows())
        .and(y_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, yo)| {
            interpolator.interp_row(&yi, yo);
        });
}

/// Interpolate over the last axis of a complex array
/// with accompanying weights
#[allow(clippy::too_many_arguments, reason = "inline helper function")]
#[inline]
pub fn interp_last_ax_complex<T>(
    interpolator: &dyn Interpolator<T>,
    y_re_in: &ArrayView2<T>,
    y_im_in: &ArrayView2<T>,
    mut y_re_out: ArrayViewMut2<T>,
    mut y_im_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_re_in.rows())
        .and(y_im_in.rows())
        .and(y_re_out.rows_mut())
        .and(y_im_out.rows_mut())
        .into_par_iter()
        .for_each(|(yre_i, yim_i, yre_o, yim_o)| {
            interpolator.interp_row(&yre_i, yre_o);
            interpolator.interp_row(&yim_i, yim_o);
        });
}

/// Interpolate over the last axis of a real array
/// with accompanying weights.
#[inline]
pub fn interp_last_ax_real_weighted<T>(
    interpolator: &dyn Interpolator<T>,
    y_in: &ArrayView2<T>,
    weight_in: &ArrayView2<T>,
    mut y_out: ArrayViewMut2<T>,
    mut weight_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    // update the scratch buffer size
    let n_in = y_in.ncols();
    let scratch_pool = make_scratch_pool(n_in);
    let mask_pool = make_scratch_pool(n_in);
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
            let (vbuf, mbuf): (&mut Vec<f64>, &mut Vec<f64>) = unsafe {
                (
                    &mut *scratch_pool[sslot].0.get(),
                    &mut *mask_pool[sslot].0.get(),
                )
            };

            interpolator.interp_row_with_variance(&yi, &wi, vbuf, mbuf, yo, wo);
        });
}

/// Interpolate over the last axis of a complex array
/// with accompanying weights
#[allow(clippy::too_many_arguments, reason = "inline helper function")]
#[inline]
pub fn interp_last_ax_complex_weighted<T>(
    interpolator: &dyn Interpolator<T>,
    y_re_in: &ArrayView2<T>,
    y_im_in: &ArrayView2<T>,
    weight_in: &ArrayView2<T>,
    mut y_re_out: ArrayViewMut2<T>,
    mut y_im_out: ArrayViewMut2<T>,
    mut weight_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    // update the scratch buffer size
    let n_in = y_re_in.ncols();
    let scratch_pool = make_scratch_pool(n_in);
    let mask_pool = make_scratch_pool(n_in);
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
            let (vbuf, mbuf): (&mut Vec<f64>, &mut Vec<f64>) = unsafe {
                (
                    &mut *scratch_pool[sslot].0.get(),
                    &mut *mask_pool[sslot].0.get(),
                )
            };

            interpolator.interp_row_with_variance(&yre_i, &wi, vbuf, mbuf, yre_o, wo);
            interpolator.interp_row(&yim_i, yim_o);
        });
}
