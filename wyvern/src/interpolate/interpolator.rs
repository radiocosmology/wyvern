//! Implementation of a linear interpolator
use crate::types::{FloatLike, ParFloatLike};
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
    fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T>;
}

/// Implements interpolation methods for float-like values
pub trait Interpolator<T: FloatLike>: Send + Sync + InterpolationPlan {
    /// Interpolate a single row's data onto output points
    fn interp_row(&self, y_in: &[T], y_out: &mut [T]);

    /// Interpolate a single row's data onto output poi ts,
    /// accounting for an input mask
    fn interp_row_masked(&self, y_in: &[T], mask_in: &mut [f64], y_out: &mut [T]);

    /// Interpolate a single row's data onto output points,
    /// and propagate corresponding inverse-variance weights
    fn interp_row_with_variance(
        &self,
        y_in: &[T],
        weight_in: &[T],
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        y_out: &mut [T],
        weight_out: &mut [T],
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
    ///
    /// # Panics
    /// Panics if the number of columns in `y_in` does not match the expected
    /// interpolator length.
    #[inline]
    pub fn interpolate_real(&self, y_in: &ArrayView2<T>, mut y_out: ArrayViewMut2<T>) {
        assert_eq!(y_out.ncols(), self.interpolator.len());
        assert_eq!(y_in.ncols(), self.interpolator.n_in());
        #[allow(clippy::indexing_slicing, reason = "inputs are explicitly 2D")]
        {
            assert_eq!(y_in.strides()[1], 1);
            assert_eq!(y_out.strides()[1], 1);
        }
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_in.rows())
            .and(y_out.rows_mut())
            .into_par_iter()
            .for_each(|(yi, mut yo)| {
                #[allow(clippy::unwrap_used, reason = "stride = 1 already asserted")]
                let (yin_sl, yo_sl) = { (yi.as_slice().unwrap(), yo.as_slice_mut().unwrap()) };
                self.interpolator.interp_row(yin_sl, yo_sl);
            });
    }

    /// Interpolate over the last axis of a complex array
    /// with accompanying weights
    ///
    /// # Panics
    /// Panics if the number of columns in `y_re_in` or `y_re_in` do
    /// not match the expected interpolator length.
    #[allow(clippy::too_many_arguments, reason = "inline helper function")]
    #[inline]
    pub fn interpolate_complex(
        &self,
        y_re_in: &ArrayView2<T>,
        y_im_in: &ArrayView2<T>,
        mut y_re_out: ArrayViewMut2<T>,
        mut y_im_out: ArrayViewMut2<T>,
    ) {
        let n_in = y_re_in.ncols();
        let n_out = y_re_out.ncols();
        let scratch = (
            vec![T::from_f64(0.0); n_in],
            vec![T::from_f64(0.0); n_in],
            vec![T::from_f64(0.0); n_out],
            vec![T::from_f64(0.0); n_out],
        );

        #[allow(clippy::indexing_slicing, reason = "inputs are explicitly 2D")]
        {
            assert_eq!(y_re_in.shape()[1], self.interpolator.n_in());
            assert_eq!(y_im_in.shape()[1], self.interpolator.n_in());
            assert_eq!(y_re_out.shape()[1], self.interpolator.len());
            assert_eq!(y_im_out.shape()[1], self.interpolator.len());
            // assert_eq!(y_re_in.strides()[1], 2);
            // assert_eq!(y_im_in.strides()[1], 2);
        }
        // iterate over the 0th axis and interpolate the 1st
        // (contiguous) axis
        Zip::from(y_re_in.rows())
            .and(y_im_in.rows())
            .and(y_re_out.rows_mut())
            .and(y_im_out.rows_mut())
            .into_par_iter()
            .for_each_with(
                scratch,
                |(rebuf, imbuf, robuf, iobuf), (yre_i, yim_i, mut yre_o, mut yim_o)| {
                    // copy views into actual bufs
                    // TODO: make this into a function
                    rebuf
                        .iter_mut()
                        .zip(imbuf.iter_mut())
                        .zip(yre_i.iter())
                        .zip(yim_i.iter())
                        .for_each(|(((sre, sim), yre), yim)| {
                            *sre = *yre;
                            *sim = *yim;
                        });
                    self.interpolator.interp_row(rebuf, robuf);
                    self.interpolator.interp_row(imbuf, iobuf);

                    robuf
                        .iter()
                        .zip(iobuf.iter())
                        .zip(yre_o.iter_mut())
                        .zip(yim_o.iter_mut())
                        .for_each(|(((sre, sim), yre), yim)| {
                            *yre = *sre;
                            *yim = *sim;
                        });
                },
            );
    }

    /// Interpolate over the last axis of a real array
    /// with accompanying weights.
    ///
    /// # Panics
    /// Panics if the number of columns in `y_in` or `weight_in` do
    /// not match the expected interpolator length.
    #[inline]
    pub fn interpolate_real_weighted(
        &self,
        y_in: &ArrayView2<T>,
        weight_in: &ArrayView2<T>,
        mut y_out: ArrayViewMut2<T>,
        mut weight_out: ArrayViewMut2<T>,
    ) {
        assert_eq!(y_in.ncols(), self.interpolator.n_in());
        assert_eq!(weight_in.ncols(), self.interpolator.n_in());
        assert_eq!(y_out.ncols(), self.interpolator.len());
        assert_eq!(weight_out.ncols(), self.interpolator.len());
        #[allow(clippy::indexing_slicing, reason = "inputs are explicitly 2D")]
        {
            assert_eq!(y_in.strides()[1], 1);
            assert_eq!(y_out.strides()[1], 1);
            assert_eq!(weight_in.strides()[1], 1);
            assert_eq!(weight_out.strides()[1], 1);
        }

        // update the scratch buffer size
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
                #[allow(clippy::unwrap_used, reason = "stride = 1 already asserted")]
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

    /// Interpolate over the last axis of a complex array
    /// with accompanying weights.
    ///
    /// # Panics
    /// Panics if the number of columns in `y_re_in`, `y_im_in`, or
    /// `weight_in` do not match the expected interpolator length.
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
        let n_out = weight_out.ncols();
        let init = || {
            (
                vec![T::from_f64(0.0); n_in],
                vec![T::from_f64(0.0); n_in],
                vec![T::from_f64(0.0); n_out],
                vec![T::from_f64(0.0); n_out],
                vec![0.0; n_in],
                vec![0.0; n_in],
            )
        };

        #[allow(clippy::indexing_slicing, reason = "inputs are explicitly 2D")]
        {
            assert_eq!(y_re_in.shape()[1], self.interpolator.n_in());
            assert_eq!(y_im_in.shape()[1], self.interpolator.n_in());
            assert_eq!(weight_in.shape()[1], self.interpolator.n_in());
            assert_eq!(y_re_out.shape()[1], self.interpolator.len());
            assert_eq!(y_im_out.shape()[1], self.interpolator.len());
            assert_eq!(weight_out.shape()[1], self.interpolator.len());
            // stride checks
            // assert_eq!(y_re_in.strides()[1], 2);
            // assert_eq!(y_im_in.strides()[1], 2);
            assert_eq!(weight_in.strides()[1], 1);
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
            .for_each_init(
                init,
                |(rebuf, imbuf, robuf, iobuf, vbuf, mbuf),
                 (yre_i, yim_i, wi, mut yre_o, mut yim_o, mut wo)| {
                    // slice the contiguous weight arrays
                    #[allow(clippy::unwrap_used, reason = "stride = 1 already asserted")]
                    let (win_sl, wo_sl) = { (wi.as_slice().unwrap(), wo.as_slice_mut().unwrap()) };
                    // copy possibly non-contiguous arrays into bufs
                    rebuf
                        .iter_mut()
                        .zip(imbuf.iter_mut())
                        .zip(yre_i.iter())
                        .zip(yim_i.iter())
                        .for_each(|(((sre, sim), yre), yim)| {
                            *sre = *yre;
                            *sim = *yim;
                        });
                    self.interpolator
                        .interp_row_with_variance(rebuf, win_sl, vbuf, mbuf, robuf, wo_sl);
                    // mask scratch buffer already contains the mask for this row
                    self.interpolator.interp_row_masked(imbuf, mbuf, iobuf);
                    // copy back into array views. Weights are already written, since
                    // they were written directly into the slice instead of a temporary
                    // buffer
                    robuf
                        .iter()
                        .zip(iobuf.iter())
                        .zip(yre_o.iter_mut())
                        .zip(yim_o.iter_mut())
                        .for_each(|(((sre, sim), yre), yim)| {
                            *yre = *sre;
                            *yim = *sim;
                        });
                },
            );
    }
}
