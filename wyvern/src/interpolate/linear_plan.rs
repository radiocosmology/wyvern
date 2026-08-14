//! Linear implementation for a [`InterpolationPlan`].
use super::helpers::{invert_no_zero, median_abs_sample_spacing};
use super::interpolator::{InterpolationPlan, Interpolator, IntoInterpolator};
use crate::types::FloatLike;
use ndarray::{ArrayView1, ArrayViewMut1};

/// Precomputed interpolation plan for mapping input
/// and output samples.
pub struct LinearPlan {
    // lower bracket index for input
    i0: Vec<usize>,
    // interpolation coefficient for i1 sample (w0 = 1 - w1)
    c1: Vec<f64>,
    // mask for valid samples. 1.0 if valid, 0.0 otherwise
    valid: Vec<f64>,
}

impl LinearPlan {
    /// Build an interpolation plan for a linear interpolator.
    ///
    /// # Parameters
    /// ``x_in``: sorted, arbitrary spacing, len >= 2
    /// ``x_out``: sorted, uniform spacing, len >= 1
    ///
    /// # Returns
    /// [`LinearPlan`]
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

        Ok(Self { i0, c1, valid })
    }
}

impl InterpolationPlan for LinearPlan {
    #[inline]
    fn len(&self) -> usize {
        self.i0.len()
    }
}

impl IntoInterpolator for LinearPlan {
    #[inline]
    fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T> {
        self
    }
}

impl<T: FloatLike> Interpolator<T> for LinearPlan {
    #[inline]
    fn interp_row(&self, y_in: &ArrayView1<T>, mut y_out: ArrayViewMut1<T>) {
        let n_out = self.len();

        debug_assert_eq!(n_out, y_out.len());

        unsafe {
            for j in 0..n_out {
                // extract the interpolation indices and weight
                let i0 = *self.i0.get_unchecked(j);
                let i1 = i0 + 1;
                // interpolation coefficients
                let s1 = *self.c1.get_unchecked(j);

                // interpolate data onto the target sample
                let a = (*y_in.uget(i0)).as_();
                let b = (*y_in.uget(i1)).as_();
                *y_out.uget_mut(j) = T::from_f64((b - a).mul_add(s1, a));
            }
        }
    }

    #[inline]
    fn interp_row_masked(
        &self,
        y_in: &ArrayView1<T>,
        mask_in: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
    ) {
        let n_in = y_in.len();
        let n_out = self.len();

        debug_assert_eq!(n_in, mask_in.len());
        debug_assert_eq!(n_out, y_out.len());

        unsafe {
            for j in 0..n_out {
                // extract the interpolation indices and weight
                let i0 = *self.i0.get_unchecked(j);
                let i1 = i0 + 1;
                // interpolation coefficients
                let s1 = *self.c1.get_unchecked(j);

                // check if sample is valid from the plan or masked
                // from the input mask
                let valid = *self.valid.get_unchecked(j) * *mask_in.get_unchecked(i0);

                // interpolate data onto the target sample
                let a = (*y_in.uget(i0)).as_();
                let b = (*y_in.uget(i1)).as_();
                *y_out.uget_mut(j) = T::from_f64(valid * (b - a).mul_add(s1, a));
            }
        }
    }

    #[inline]
    #[allow(clippy::indexing_slicing, clippy::unwrap_used, reason = "guaranteed")]
    fn interp_row_with_variance(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
        mut weight_out: ArrayViewMut1<T>,
    ) {
        let n_in = y_in.len();
        let n_out = self.len();

        debug_assert_eq!(n_in, weight_in.len());
        debug_assert_eq!(n_in, var_scratch.len());
        debug_assert_eq!(n_out, y_out.len());
        debug_assert_eq!(n_out, weight_out.len());

        unsafe {
            // invert weights once per pass, since input samples are often re-used
            for k in 0..n_in {
                let w: f64 = (*weight_in.uget(k)).as_();
                *var_scratch.get_unchecked_mut(k) = invert_no_zero(w);
                *mask_scratch.get_unchecked_mut(k) = f64::from(w > 0.0 && w < f64::INFINITY);
            }

            for j in 0..n_out {
                // extract the interpolation indices and weight
                let i0 = *self.i0.get_unchecked(j);
                let i1 = i0 + 1;
                // interpolation coefficients
                let s1 = *self.c1.get_unchecked(j);

                // check if sample is valid from the plan or masked
                // from the input mask
                let valid = *self.valid.get_unchecked(j)
                    * *mask_scratch.get_unchecked(i0)
                    * *mask_scratch.get_unchecked(i1);

                // interpolate data onto the target sample
                let a = (*y_in.uget(i0)).as_();
                let b = (*y_in.uget(i1)).as_();
                *y_out.uget_mut(j) = T::from_f64(valid * (b - a).mul_add(s1, a));

                // propagate weights and masking
                let var_a = *var_scratch.get_unchecked(i0);
                let var_b = *var_scratch.get_unchecked(i1);

                let s0 = 1.0 - s1;
                let c0 = s0 * s0 * var_a;
                let c1 = s1 * s1 * var_b;
                let norm = invert_no_zero(c0 + c1);
                // valid is either 1.0 or 0.0
                *weight_out.uget_mut(j) = T::from_f64(valid * norm);
            }
        }
    }
}
