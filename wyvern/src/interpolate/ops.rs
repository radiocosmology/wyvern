//! Linear interpolation across the last axis.
use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::OnceLock;

use super::plan::Interpolator;
use crate::types::ParFloatLike;

const SERIAL_ROWS_THRESHOLD: usize = 32;

/// Wrapper to allow &mut access into one scratch slot from
/// multiple threads without a mutex.
struct ScratchSlot(UnsafeCell<ScratchBuffers>);
unsafe impl Sync for ScratchSlot {}

struct ScratchBuffers {
    var: Vec<f64>,
    mask: Vec<f64>,
}

fn scratch_pool() -> &'static [ScratchSlot] {
    static SCRATCH_POOL: OnceLock<Vec<ScratchSlot>> = OnceLock::new();
    SCRATCH_POOL
        .get_or_init(|| {
            let available_threads =
                std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
            let num_threads = rayon::current_num_threads().max(available_threads);
            (0..num_threads)
                .map(|_| {
                    ScratchSlot(UnsafeCell::new(ScratchBuffers {
                        var: Vec::new(),
                        mask: Vec::new(),
                    }))
                })
                .collect()
        })
        .as_slice()
}

fn with_scratch<R>(
    n_in: usize,
    needs_mask: bool,
    f: impl FnOnce(&mut [f64], &mut [f64]) -> R,
) -> R {
    let pool = scratch_pool();
    let nslots = pool.len().max(1);
    let slot_idx = rayon::current_thread_index().unwrap_or(0) % nslots;

    #[allow(clippy::indexing_slicing, reason = "slot index is always in range")]
    unsafe {
        let buffers = &mut *pool[slot_idx].0.get();
        if buffers.var.len() < n_in {
            buffers.var.resize(n_in, 0.0);
        }
        let var = &mut buffers.var[..n_in];

        let mut empty = [];
        let mask: &mut [f64] = if needs_mask {
            if buffers.mask.len() < n_in {
                buffers.mask.resize(n_in, 0.0);
            }
            &mut buffers.mask[..n_in]
        } else {
            &mut empty
        };

        f(var, mask)
    }
}

/// Interpolate over the last axis of a real array.
#[inline]
pub fn interp_last_ax_real<T>(
    interpolator: &(impl Interpolator<T> + ?Sized),
    y_in: &ArrayView2<T>,
    mut y_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    if y_in.nrows() <= SERIAL_ROWS_THRESHOLD {
        Zip::from(y_in.rows())
            .and(y_out.rows_mut())
            .for_each(|yi, yo| interpolator.interp_row(&yi, yo));
        return;
    }

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
    interpolator: &(impl Interpolator<T> + ?Sized),
    y_re_in: &ArrayView2<T>,
    y_im_in: &ArrayView2<T>,
    mut y_re_out: ArrayViewMut2<T>,
    mut y_im_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    if y_re_in.nrows() <= SERIAL_ROWS_THRESHOLD {
        Zip::from(y_re_in.rows())
            .and(y_im_in.rows())
            .and(y_re_out.rows_mut())
            .and(y_im_out.rows_mut())
            .for_each(|yre_i, yim_i, yre_o, yim_o| {
                interpolator.interp_row(&yre_i, yre_o);
                interpolator.interp_row(&yim_i, yim_o);
            });
        return;
    }

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
    interpolator: &(impl Interpolator<T> + ?Sized),
    y_in: &ArrayView2<T>,
    weight_in: &ArrayView2<T>,
    mut y_out: ArrayViewMut2<T>,
    mut weight_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    let n_in = y_in.ncols();
    let needs_mask = interpolator.needs_mask_scratch();

    if y_in.nrows() <= SERIAL_ROWS_THRESHOLD {
        Zip::from(y_in.rows())
            .and(weight_in.rows())
            .and(y_out.rows_mut())
            .and(weight_out.rows_mut())
            .for_each(|yi, wi, yo, wo| {
                with_scratch(n_in, needs_mask, |vbuf, mbuf| {
                    interpolator.interp_row_with_variance(&yi, &wi, vbuf, mbuf, yo, wo);
                });
            });
        return;
    }

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_in.rows())
        .and(weight_in.rows())
        .and(y_out.rows_mut())
        .and(weight_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, wi, yo, wo)| {
            with_scratch(n_in, needs_mask, |vbuf, mbuf| {
                interpolator.interp_row_with_variance(&yi, &wi, vbuf, mbuf, yo, wo);
            });
        });
}

/// Interpolate over the last axis of a complex array
/// with accompanying weights
#[allow(clippy::too_many_arguments, reason = "inline helper function")]
#[inline]
pub fn interp_last_ax_complex_weighted<T>(
    interpolator: &(impl Interpolator<T> + ?Sized),
    y_re_in: &ArrayView2<T>,
    y_im_in: &ArrayView2<T>,
    weight_in: &ArrayView2<T>,
    mut y_re_out: ArrayViewMut2<T>,
    mut y_im_out: ArrayViewMut2<T>,
    mut weight_out: ArrayViewMut2<T>,
) where
    T: ParFloatLike,
{
    let n_in = y_re_in.ncols();
    let needs_mask = interpolator.needs_mask_scratch();

    if y_re_in.nrows() <= SERIAL_ROWS_THRESHOLD {
        Zip::from(y_re_in.rows())
            .and(y_im_in.rows())
            .and(weight_in.rows())
            .and(y_re_out.rows_mut())
            .and(y_im_out.rows_mut())
            .and(weight_out.rows_mut())
            .for_each(|yre_i, yim_i, wi, yre_o, yim_o, wo| {
                with_scratch(n_in, needs_mask, |vbuf, mbuf| {
                    interpolator.interp_row_with_variance(&yre_i, &wi, vbuf, mbuf, yre_o, wo);
                });
                interpolator.interp_row(&yim_i, yim_o);
            });
        return;
    }

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
            with_scratch(n_in, needs_mask, |vbuf, mbuf| {
                interpolator.interp_row_with_variance(&yre_i, &wi, vbuf, mbuf, yre_o, wo);
            });
            interpolator.interp_row(&yim_i, yim_o);
        });
}
