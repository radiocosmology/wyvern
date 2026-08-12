//! Linear interpolation across the last axis.
use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;
use std::cell::UnsafeCell;

use super::plan::Interpolator;
use crate::types::ParFloatLike;

/// Collection of scratch buffers
struct ScratchBuffers {
    var: Vec<f64>,
    mask: Vec<f64>,
}

/// Wrapper to allow &mut access into one `Vec<f64>` slot from
/// multiple threads without a mutex.
struct ScratchSlot(UnsafeCell<ScratchBuffers>);
unsafe impl Sync for ScratchSlot {}

/// Pool of [`ScratchSlot`]s
struct ScratchPool(Vec<ScratchSlot>);

impl ScratchPool {
    /// Make a new pool. Only allocates a mask buffer if required
    /// by the caller.
    fn build(n_in: usize, needs_mask: bool) -> Self {
        let num_threads = rayon::current_num_threads();
        Self(
            (0..num_threads)
                .map(|_| {
                    ScratchSlot(UnsafeCell::new(ScratchBuffers {
                        var: vec![0.0; n_in],
                        mask: if needs_mask { vec![0.0; n_in] } else { vec![] },
                    }))
                })
                .collect(),
        )
    }

    /// Call a closure with mutable references to this thread's scratch buffers
    #[inline]
    fn with<R>(&self, func: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
        let sslot = rayon::current_thread_index().unwrap_or(0) % self.0.len();
        #[allow(clippy::indexing_slicing, reason = "zeroth index guaranteed to exist")]
        let buffers = unsafe { &mut *self.0[sslot].0.get() };

        func(&mut buffers.var, &mut buffers.mask)
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
    // update the scratch buffer size
    let n_in = y_in.ncols();
    let pool = ScratchPool::build(n_in, interpolator.needs_mask_scratch());

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_in.rows())
        .and(weight_in.rows())
        .and(y_out.rows_mut())
        .and(weight_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, wi, yo, wo)| {
            pool.with(|vbuf, mbuf| {
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
    // update the scratch buffer size
    let n_in = y_re_in.ncols();
    let pool = ScratchPool::build(n_in, interpolator.needs_mask_scratch());

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
            pool.with(|vbuf, mbuf| {
                interpolator.interp_row_with_variance(&yre_i, &wi, vbuf, mbuf, yre_o, wo);
                interpolator.interp_row(&yim_i, yim_o);
            });
        });
}
