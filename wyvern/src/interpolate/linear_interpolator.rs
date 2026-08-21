//! Linear implementation for a [`InterpolationPlan`].
use ndarray::{ArrayView1, ArrayViewMut1};

use super::helpers::{invert_no_zero, median_abs_sample_spacing};
use super::interpolator::{InterpolationPlan, Interpolator, IntoInterpolator};

use crate::types::FloatLike;
use crate::util::assert_unchecked_debug;

/// Precomputed interpolation plan for mapping input
/// and output samples.
pub struct LinearInterpolator {
    // lower bracket index for input
    i0: Vec<usize>,
    // interpolation coefficient for i1 sample (w0 = 1 - w1)
    c1: Vec<f64>,
    // mask for valid samples. 1.0 if valid, 0.0 otherwise
    valid: Vec<f64>,
    // number of input samples
    n_in: usize,
}

impl LinearInterpolator {
    /// Build an interpolation plan for a linear interpolator.
    ///
    /// # Parameters
    /// ``x_in``: sorted, arbitrary spacing, len >= 2
    /// ``x_out``: sorted, uniform spacing, len >= 1
    ///
    /// # Returns
    /// [`LinearInterpolator`]
    ///
    /// # Errors
    /// If input sample indices are unsorted or repeated
    pub fn build(x_in: &[f64], x_out: &[f64]) -> eyre::Result<Self> {
        let n_out = x_out.len();
        let n_in = x_in.len();

        if n_in < 2 {
            eyre::bail!("at least 2 input samples are required!");
        }
        if n_out < 1 {
            eyre::bail!("at least 1 output sample is required!");
        }

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut c1 = Vec::<f64>::with_capacity(n_out);
        let mut valid = Vec::<f64>::with_capacity(n_out);

        // both inputs are sorted, so step only advances forward. error
        // is eventually returned if this assumption fails
        let mut lo: usize = 0;
        let lo_max: usize = n_in - 2;

        let delta = median_abs_sample_spacing(x_in);

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
            // mask
            valid.push(f64::from(!distant));

            // interpolation indices
            i0.push(lo);
            // weight coefficient
            c1.push((xo - a) / span);
        }

        Ok(Self {
            i0,
            c1,
            valid,
            n_in,
        })
    }
}

impl InterpolationPlan for LinearInterpolator {
    #[inline]
    fn len(&self) -> usize {
        self.i0.len()
    }

    #[inline]
    fn n_in(&self) -> usize {
        self.n_in
    }
}

impl IntoInterpolator for LinearInterpolator {
    #[inline]
    fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T> {
        self
    }
}

impl<T: FloatLike> Interpolator<T> for LinearInterpolator {
    #[inline]
    fn interp_row(&self, y_in: &ArrayView1<T>, mut y_out: ArrayViewMut1<T>) {
        assert_eq!(self.len(), y_out.len());
        assert_eq!(self.n_in(), y_in.len());

        self.i0
            .iter()
            .zip(self.c1.iter())
            .zip(y_out.iter_mut())
            .for_each(|((i0, s1), yo)| {
                assert_unchecked_debug!(*i0 < self.n_in() - 1);

                let a = unsafe { y_in.uget(*i0) }.as_();
                let b = unsafe { y_in.uget(*i0 + 1) }.as_();

                *yo = T::from_f64((b - a).mul_add(*s1, a));
            });
    }

    #[inline]
    fn interp_row_masked(
        &self,
        y_in: &ArrayView1<T>,
        mask_in: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
    ) {
        assert_eq!(self.n_in(), y_in.len());
        assert_eq!(self.n_in(), mask_in.len());
        assert_eq!(self.len(), y_out.len());

        self.i0
            .iter()
            .zip(self.c1.iter())
            .zip(self.valid.iter())
            .zip(y_out.iter_mut())
            .for_each(|(((i0, s1), valid), yo)| {
                assert_unchecked_debug!(*i0 < self.n_in() - 1);

                let a = unsafe { y_in.uget(*i0) }.as_();
                let b = unsafe { y_in.uget(*i0 + 1) }.as_();
                let mask_a = unsafe { mask_in.get_unchecked(*i0) };
                let mask_b = unsafe { mask_in.get_unchecked(*i0 + 1) };

                let mask = valid * mask_a * mask_b;

                *yo = T::from_f64(mask * (b - a).mul_add(*s1, a));
            });
    }

    #[inline]
    fn interp_row_with_variance(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
        mut weight_out: ArrayViewMut1<T>,
    ) {
        assert_eq!(self.n_in(), y_in.len());
        assert_eq!(self.n_in(), weight_in.len());
        assert_eq!(self.n_in(), var_scratch.len());
        assert_eq!(self.n_in(), mask_scratch.len());
        assert_eq!(self.len(), y_out.len());
        assert_eq!(self.len(), weight_out.len());
        // confirm that _all_ indices are valid
        debug_assert!(self.i0.iter().all(|&i0| i0 < y_in.len() - 1));

        // invert weights once per pass, since input samples
        // are often reused
        weight_in
            .iter()
            .zip(var_scratch.iter_mut())
            .zip(mask_scratch.iter_mut())
            .for_each(|((w, vs), ms)| {
                let w: f64 = w.as_();
                *vs = invert_no_zero(w);
                *ms = f64::from(w > 0.0 && w.is_finite());
            });

        self.i0
            .iter()
            .zip(self.c1.iter())
            .zip(self.valid.iter())
            .zip(y_out.iter_mut())
            .zip(weight_out.iter_mut())
            .for_each(|((((i0, s1), valid), yo), wo)| {
                assert_unchecked_debug!(*i0 < self.n_in() - 1);

                let a = unsafe { y_in.uget(*i0) }.as_();
                let b = unsafe { y_in.uget(*i0 + 1) }.as_();
                let mask_a = unsafe { mask_scratch.get_unchecked(*i0) };
                let mask_b = unsafe { mask_scratch.get_unchecked(*i0 + 1) };
                let var_a = unsafe { var_scratch.get_unchecked(*i0) };
                let var_b = unsafe { var_scratch.get_unchecked(*i0 + 1) };

                // valid only if both plan mask and input mask agree
                let mask = valid * mask_a * mask_b;

                *yo = T::from_f64(mask * (b - a).mul_add(*s1, a));

                // propagate weights and masking
                let s0 = 1.0 - s1;
                let c0 = s0 * s0 * var_a;
                let c1 = s1 * s1 * var_b;
                let norm = invert_no_zero(c0 + c1);
                // valid is either 1.0 or 0.0
                *wo = T::from_f64(mask * norm);
            });
    }
}
