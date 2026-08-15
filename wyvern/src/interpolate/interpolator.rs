//! Implementation of a linear interpolator
use crate::types::{FloatLike, ParFloatLike};
use ndarray::{ArrayView1, ArrayViewMut1};
use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;

// ------ Traits ------

/// Implements methods required to construct an interpolation plan
pub trait InterpolationPlan {
    /// Number of output samples
    fn len(&self) -> usize;
    /// `true` if `len` is zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implements methods to convert this to a typed [`Interpolator`]
pub trait IntoInterpolator {
    /// extract the interpolator
    fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T>;
}

/// Implements interpolation methods for float-like values
pub trait Interpolator<T: FloatLike>: Send + Sync {
    /// Interpolate a single row's data onto output points
    fn interp_row(&self, y_in: &ArrayView1<T>, y_out: ArrayViewMut1<T>);

    /// Interpolate a single row's data onto output poi ts,
    /// accounting for an input mask
    fn interp_row_masked(&self, y_in: &ArrayView1<T>, mask_in: &mut [f64], y_out: ArrayViewMut1<T>);

    /// Interpolate a single row's data onto output points,
    /// and propagate corresponding inverse-variance weights
    fn interp_row_with_variance(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        y_out: ArrayViewMut1<T>,
        weight_out: ArrayViewMut1<T>,
    );
}

/// Implements a row-parallel interpolator, constructed
/// from an interpolator
pub struct ParallelInterpolator<'a, T>
where
    T: ParFloatLike,
{
    /// Internal interpolator
    interpolator: &'a dyn Interpolator<T>,
}

impl<'a, T> ParallelInterpolator<'a, T>
where
    T: ParFloatLike,
{
    /// Make a new parallel interpolator for a given
    /// number of rows.
    pub fn with_interpolator(interpolator: &'a dyn Interpolator<T>) -> Self {
        Self { interpolator }
    }

    /// Interpolate over the last axis of a real array.
    #[inline]
    pub fn interpolate_real(&self, y_in: &ArrayView2<T>, mut y_out: ArrayViewMut2<T>) {
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_in.rows())
            .and(y_out.rows_mut())
            .into_par_iter()
            .for_each(|(yi, yo)| {
                self.interpolator.interp_row(&yi, yo);
            });
    }

    /// Interpolate over the last axis of a complex array
    /// with accompanying weights
    #[allow(clippy::too_many_arguments, reason = "inline helper function")]
    #[inline]
    pub fn interpolate_complex(
        &self,
        y_re_in: &ArrayView2<T>,
        y_im_in: &ArrayView2<T>,
        mut y_re_out: ArrayViewMut2<T>,
        mut y_im_out: ArrayViewMut2<T>,
    ) {
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_re_in.rows())
            .and(y_im_in.rows())
            .and(y_re_out.rows_mut())
            .and(y_im_out.rows_mut())
            .into_par_iter()
            .for_each(|(yre_i, yim_i, yre_o, yim_o)| {
                self.interpolator.interp_row(&yre_i, yre_o);
                self.interpolator.interp_row(&yim_i, yim_o);
            });
    }

    /// Interpolate over the last axis of a real array
    /// with accompanying weights.
    #[inline]
    pub fn interpolate_real_weighted(
        &self,
        y_in: &ArrayView2<T>,
        weight_in: &ArrayView2<T>,
        mut y_out: ArrayViewMut2<T>,
        mut weight_out: ArrayViewMut2<T>,
    ) {
        // update the scratch buffer size
        let n_in = y_in.ncols();
        let scratch = (vec![0.0; n_in], vec![0.0; n_in]);

        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_in.rows())
            .and(weight_in.rows())
            .and(y_out.rows_mut())
            .and(weight_out.rows_mut())
            .into_par_iter()
            .for_each_with(scratch, |(vbuf, mbuf), (yi, wi, yo, wo)| {
                self.interpolator
                    .interp_row_with_variance(&yi, &wi, vbuf, mbuf, yo, wo);
            });
    }

    /// Interpolate over the last axis of a complex array
    /// with accompanying weights
    #[allow(clippy::too_many_arguments, reason = "inline helper function")]
    #[inline]
    pub fn interpolate_complex_weighted(
        &self,
        y_re_in: &ArrayView2<T>,
        y_im_in: &ArrayView2<T>,
        weight_in: &ArrayView2<T>,
        mut y_re_out: ArrayViewMut2<T>,
        mut y_im_out: ArrayViewMut2<T>,
        mut weight_out: ArrayViewMut2<T>,
    ) {
        let n_in = weight_in.ncols();
        let scratch = (vec![0.0; n_in], vec![0.0; n_in]);

        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_re_in.rows())
            .and(y_im_in.rows())
            .and(weight_in.rows())
            .and(y_re_out.rows_mut())
            .and(y_im_out.rows_mut())
            .and(weight_out.rows_mut())
            .into_par_iter()
            .for_each_with(
                scratch,
                |(vbuf, mbuf), (yre_i, yim_i, wi, yre_o, yim_o, wo)| {
                    self.interpolator
                        .interp_row_with_variance(&yre_i, &wi, vbuf, mbuf, yre_o, wo);
                    // mask scratch buffer already contains the mask for this row
                    self.interpolator.interp_row_masked(&yim_i, mbuf, yim_o);
                },
            );
    }
}
