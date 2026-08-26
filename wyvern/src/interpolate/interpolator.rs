//! Implementation of a linear interpolator
use crate::types::MaybeComplex;
use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;

// ------ Traits ------

/// Implements methods required to construct an interpolation plan
pub trait InterpolationPlan {
    /// Number of output samples
    fn len(&self) -> usize;
    /// Number of input samples
    fn n_in(&self) -> usize;
    /// `true` if `len` is zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implements methods to convert this to a typed [`Interpolator`]
pub trait IntoInterpolator {
    /// extract the interpolator
    fn as_interpolator<T: MaybeComplex>(&self) -> &dyn Interpolator<T>;
}

/// Implements interpolation methods for float-like values
pub trait Interpolator<T: MaybeComplex>: Send + Sync + InterpolationPlan {
    /// Interpolate a single row's data onto output points
    fn interp_row(&self, y_in: &[T], y_out: &mut [T]);

    /// Interpolate a single row's data onto output poi ts,
    /// accounting for an input mask
    fn interp_row_masked(&self, y_in: &[T], mask_in: &[f64], y_out: &mut [T]);

    /// Interpolate a single row's data onto output points,
    /// and propagate corresponding inverse-variance weights
    fn interp_row_with_variance(
        &self,
        y_in: &[T],
        weight_in: &[T::Real],
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        y_out: &mut [T],
        weight_out: &mut [T::Real],
    );
}

/// Implements a row-parallel interpolator, constructed
/// from an interpolator
pub struct ParallelInterpolator<'a, T>
where
    T: MaybeComplex,
{
    /// Internal interpolator
    interpolator: &'a dyn Interpolator<T>,
}

impl<'a, T> ParallelInterpolator<'a, T>
where
    T: MaybeComplex,
{
    /// Make a new parallel interpolator for a given
    /// number of rows.
    pub fn with_interpolator(interpolator: &'a dyn Interpolator<T>) -> Self {
        Self { interpolator }
    }

    /// Interpolate over the last axis of a real or [`num_complex::Complex`] array.
    ///
    /// # Panics
    /// Panics if the number of columns in `y_in` does not match the expected
    /// interpolator length.
    #[inline]
    pub fn interpolate(&self, y_in: &ArrayView2<T>, mut y_out: ArrayViewMut2<T>) {
        assert_eq!(y_out.ncols(), self.interpolator.len());
        assert_eq!(y_in.ncols(), self.interpolator.n_in());
        assert!(y_in.is_standard_layout(), "`y_in` is not c-contiguous!");
        assert!(y_out.is_standard_layout(), "`y_out` is not c-contiguous!");
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_in.rows())
            .and(y_out.rows_mut())
            .into_par_iter()
            .for_each(|(yi, mut yo)| {
                #[allow(clippy::unwrap_used, reason = "contiguity already asserted")]
                let (yin_sl, yo_sl) = { (yi.as_slice().unwrap(), yo.as_slice_mut().unwrap()) };
                self.interpolator.interp_row(yin_sl, yo_sl);
            });
    }

    /// Interpolate over the last axis of a real or [`Complex`] array with an
    /// accompanying mask.
    ///
    /// # Panics
    /// Panics if the number of columns in `y_in` does not match the expected
    /// interpolator length.
    #[inline]
    pub fn interpolate_masked(
        &self,
        y_in: &ArrayView2<T>,
        mask_in: &ArrayView2<f64>,
        mut y_out: ArrayViewMut2<T>,
    ) {
        assert_eq!(y_out.ncols(), self.interpolator.len());
        assert_eq!(y_in.ncols(), self.interpolator.n_in());
        assert_eq!(mask_in.ncols(), self.interpolator.n_in());
        assert!(y_in.is_standard_layout(), "`y_in` is not c-contiguous!");
        assert!(y_out.is_standard_layout(), "`y_out` is not c-contiguous!");
        assert!(
            mask_in.is_standard_layout(),
            "`mask_in` is not c-contiguous!"
        );
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_in.rows())
            .and(mask_in.rows())
            .and(y_out.rows_mut())
            .into_par_iter()
            .for_each(|(yi, mi, mut yo)| {
                #[allow(clippy::unwrap_used, reason = "contiguity already asserted")]
                let (yin_sl, yo_sl, min_sl) = {
                    (
                        yi.as_slice().unwrap(),
                        yo.as_slice_mut().unwrap(),
                        mi.as_slice().unwrap(),
                    )
                };
                self.interpolator.interp_row_masked(yin_sl, min_sl, yo_sl);
            });
    }

    /// Interpolate over the last axis of a real or [`Complex`] array
    /// with accompanying weights.
    ///
    /// # Panics
    /// Panics if the number of columns in `y_in` or `weight_in` do
    /// not match the expected interpolator length.
    #[inline]
    pub fn interpolate_weighted(
        &self,
        y_in: &ArrayView2<T>,
        weight_in: &ArrayView2<T::Real>,
        mut y_out: ArrayViewMut2<T>,
        mut weight_out: ArrayViewMut2<T::Real>,
    ) {
        assert_eq!(y_in.ncols(), self.interpolator.n_in());
        assert_eq!(weight_in.ncols(), self.interpolator.n_in());
        assert_eq!(y_out.ncols(), self.interpolator.len());
        assert_eq!(weight_out.ncols(), self.interpolator.len());
        assert!(y_in.is_standard_layout(), "`y_in` is not c-contiguous!");
        assert!(y_out.is_standard_layout(), "`y_out` is not c-contiguous!");
        assert!(
            weight_in.is_standard_layout(),
            "`weight_in` is not c-contiguous!"
        );
        assert!(
            weight_out.is_standard_layout(),
            "`weight_out` is not c-contiguous!"
        );

        // set the scratch buffer size
        let n_in = y_in.ncols();
        let init = || (vec![0.0; n_in], vec![0.0; n_in]);
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_in.rows())
            .and(weight_in.rows())
            .and(y_out.rows_mut())
            .and(weight_out.rows_mut())
            .into_par_iter()
            .for_each_init(init, |(vbuf, mbuf), (yi, wi, mut yo, mut wo)| {
                #[allow(clippy::unwrap_used, reason = "contiguity already asserted")]
                let (yin_sl, yo_sl, win_sl, wo_sl) = {
                    (
                        yi.as_slice().unwrap(),
                        yo.as_slice_mut().unwrap(),
                        wi.as_slice().unwrap(),
                        wo.as_slice_mut().unwrap(),
                    )
                };
                self.interpolator
                    .interp_row_with_variance(yin_sl, win_sl, vbuf, mbuf, yo_sl, wo_sl);
            });
    }
}
