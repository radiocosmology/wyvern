//! Implementation of a linear interpolator
use crate::types::MaybeComplex;
use ndarray::{ArrayView2, ArrayViewMut2, Zip};
use rayon::prelude::*;

// ------ Traits ------

/// A precomputed interpolation plan describing the relationship between input and
/// output samples.
pub trait InterpolationPlan {
    /// Returns the number of output samples produced by the plan.
    fn len(&self) -> usize;
    /// Returns the number of input samples expected by the plan.
    fn n_in(&self) -> usize;
    /// Returns `true` when the plan produces no output samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Converts an interpolation plan into a type-erased interpolator reference.
pub trait IntoInterpolator {
    /// Produces a typed interpolator view for the underlying plan.
    ///
    /// # Parameters
    /// * `T`: The numeric scalar type used by the interpolated data.
    ///
    /// # Returns
    /// A reference to a dynamically dispatched interpolator implementation.
    fn as_interpolator<T: MaybeComplex>(&self) -> &dyn Interpolator<T>;
}

/// A row-wise interpolation operation for real or complex floating-point data.
pub trait Interpolator<T: MaybeComplex>: Send + Sync + InterpolationPlan {
    /// Interpolate a single data row onto the output coordinates.
    ///
    /// # Parameters
    /// * `y_in`: The source data for a single row.
    /// * `y_out`: Mutable storage for the interpolated values.
    ///
    /// # Returns
    /// This method writes the interpolated row into `y_out` in place.
    fn interp_row(&self, y_in: &[T], y_out: &mut [T]);

    /// Interpolate a single row while honoring an input validity mask.
    ///
    /// # Parameters
    /// * `y_in`: The source data for a single row.
    /// * `mask_in`: Input validity mask used to zero invalid samples.
    /// * `y_out`: Mutable storage for the masked interpolation output.
    ///
    /// # Returns
    /// The masked interpolation result is written to `y_out` in place.
    fn interp_row_masked(&self, y_in: &[T], mask_in: &[f64], y_out: &mut [T]);

    /// Interpolate a row while propagating inverse-variance weights.
    ///
    /// # Parameters
    /// * `y_in`: The source data for a single row.
    /// * `weight_in`: Input inverse-variance weights for each sample.
    /// * `var_scratch`: Scratch space for inverted variances.
    /// * `mask_scratch`: Scratch space for validity masks.
    /// * `y_out`: Output data storage.
    /// * `weight_out`: Output inverse-variance weights.
    ///
    /// # Returns
    /// The interpolated values and propagated weights are written to `y_out` and
    /// `weight_out` in place.
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

/// A row-parallel wrapper around an interpolator implementation.
pub struct ParallelInterpolator<'a, T>
where
    T: MaybeComplex,
{
    /// The underlying interpolator used for each row operation.
    interpolator: &'a dyn Interpolator<T>,
}

impl<'a, T> ParallelInterpolator<'a, T>
where
    T: MaybeComplex,
{
    /// Construct a row-parallel interpolator from an underlying interpolator.
    ///
    /// # Parameters
    /// * `interpolator`: The interpolator implementation to execute in parallel.
    ///
    /// # Returns
    /// A row-parallel wrapper around `interpolator`.
    pub fn with_interpolator(interpolator: &'a dyn Interpolator<T>) -> Self {
        Self { interpolator }
    }

    /// Interpolate over the last axis of a real or [`num_complex::Complex`] array.
    ///
    /// # Parameters
    /// * `y_in`: Two-dimensional input array with one row per sample set.
    /// * `y_out`: Two-dimensional output array with the interpolated values.
    ///
    /// # Returns
    /// The method writes each interpolated row into `y_out` in place.
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

    /// Interpolate over the last axis of a real or [`num_complex::Complex`] array with an
    /// accompanying validity mask.
    ///
    /// # Parameters
    /// * `y_in`: Input samples arranged as rows of real or complex data.
    /// * `mask_in`: Per-sample validity mask aligned with `y_in`.
    /// * `y_out`: Mutable output storage for the masked interpolation result.
    ///
    /// # Returns
    /// The method writes the masked interpolation output into `y_out` in place.
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

    /// Interpolate over the last axis of a real or [`num_complex::Complex`] array with
    /// accompanying inverse-variance weights.
    ///
    /// # Parameters
    /// * `y_in`: Input samples arranged as rows of real or complex data.
    /// * `weight_in`: Input inverse-variance weights corresponding to each sample.
    /// * `y_out`: Mutable output data storage.
    /// * `weight_out`: Mutable output weight storage.
    ///
    /// # Returns
    /// The interpolated data and propagated weights are written into `y_out` and
    /// `weight_out` in place.
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
