//! Implementation of [`InterpolationPlan`] for a kernel-based interpolator
use super::plan::{InterpolationPlan, median_abs_sample_spacing};
use crate::types::FloatLike;
use ndarray::{ArrayView1, ArrayViewMut1};

/// Precomputed interpolation plan for a lanczos kernel
pub struct KernelPlan<const N: usize> {
    // index of the first window tap
    i0: Vec<usize>,
    // kernel coefficients
    coeffs: Vec<[f64; N]>,
    // mask for valid samples
    valid: Vec<f64>,
}

impl<const N: usize> KernelPlan<N> {
    // N is defined to be even, with N/2 taps on each side of the centre
    pub fn build(
        x_in: &[f64],
        x_out: &[f64],
        kernel: impl Fn(f64, f64) -> f64,
    ) -> eyre::Result<Self> {
        debug_assert!(
            N >= 2 && N.is_multiple_of(2),
            "kernel width N must be even and >= 2"
        );

        let n_in = x_in.len();
        let n_out = x_out.len();

        #[allow(
            clippy::cast_precision_loss,
            clippy::integer_division,
            reason = "values too small for precision loss"
        )]
        // kernel half-width as a float and integet
        let (a_half, a_half_isize) = {
            let ah = N / 2;
            (ah as f64, ah.cast_signed())
        };

        debug_assert!(n_in >= N, "need at least N={N} samples!");
        debug_assert!(n_out >= 1, "need at least 1 output sample!");

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut coeffs = Vec::with_capacity(n_out);
        let mut valid = Vec::<f64>::with_capacity(n_out);

        // inputs are assumed to be sorted
        let mut lo: usize = 0;
        let lo_max: usize = n_in - 2;
        let n_max = (n_in - N).cast_signed();

        let delta = median_abs_sample_spacing(x_in);

        for &xo in x_out {
            // move forward to the next target neighbourhood
            #[allow(
                clippy::indexing_slicing,
                reason = "max index is 2 less than `x_in.len()`"
            )]
            let (a, b) = {
                while lo < lo_max && x_in[lo + 1] <= xo {
                    lo += 1;
                }
                (x_in[lo], x_in[lo + 1])
            };

            let span = b - a;
            if span <= 0.0 {
                eyre::bail!("inputs are unsorted or repeated!");
            }

            // construct a window a N taps centred on the bracket, clamped
            // to [0, n_in - N]. Coefficients must be renormalized. Center
            // the window on the nearest input sample
            let center = if (xo - a) / span < 0.5 { lo } else { lo + 1 };
            // get the leftmost edge of the window
            let base = (center.cast_signed() - a_half_isize)
                .clamp(0, n_max)
                .cast_unsigned();

            // determine validity of this sample
            #[allow(clippy::indexing_slicing, reason = "indices are already clamped")]
            let outside_window = xo < x_in[base] || xo > x_in[base + N - 1];
            let distant = (b - xo).abs() > delta || (a - xo).abs() > delta;

            // interpolation indices
            let mut c = [0.0_f64; N];
            // normalization
            let mut sum = 0.0;

            // compute the kernel coefficients
            #[allow(clippy::indexing_slicing, reason = "indices are already clamped")]
            for k in 0..N {
                let xi = x_in[base + k];
                let dist = (xo - xi) / span;
                let w = kernel(dist, a_half);
                c[k] = w;
                sum += w;
            }
            let sum_degenerate = sum.abs() <= 1e-9;

            // Normalize as long as there are some samples. Otherwise, force
            // coefficients to be zero
            if sum_degenerate {
                c = [0.0_f64; N];
            } else {
                for ci in &mut c {
                    *ci /= sum;
                }
            }

            i0.push(base);
            coeffs.push(c);
            valid.push(f64::from(!(distant || outside_window || sum_degenerate)));
        }

        Ok(Self { i0, coeffs, valid })
    }
}

impl<const N: usize> InterpolationPlan for KernelPlan<N> {
    #[inline]
    fn len(&self) -> usize {
        self.i0.len()
    }

    #[inline]
    fn interp_row<T: FloatLike>(&self, y_in: &ArrayView1<T>, mut y_out: ArrayViewMut1<T>) {
        let n_out = self.len();

        debug_assert_eq!(n_out, y_out.len());

        unsafe {
            for j in 0..n_out {
                // indices and coefficients
                let i0 = *self.i0.get_unchecked(j);
                let coeffs = *self.coeffs.get_unchecked(j);

                // interpolate
                let mut value_acc: f64 = 0.0;

                for k in 0..N {
                    let c = *coeffs.get_unchecked(k);
                    let yk: f64 = (*y_in.uget(i0 + k)).as_();

                    value_acc = c.mul_add(yk, value_acc);
                }

                *y_out.uget_mut(j) = T::from_f64(value_acc);
            }
        }
    }

    #[inline]
    fn interp_row_with_variance<T: FloatLike>(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
        mut weight_out: ArrayViewMut1<T>,
    ) {
        let n_in = y_in.len();
        let n_out = self.len();
        // minimum number of window coefficients matching
        // valid weights in order to keep an interpolated
        // sample
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss,
            reason = "N is positive and is never large enough for truncation to occur"
        )]
        let min_valid_taps = (N as f64 * 0.75).ceil() as usize;

        debug_assert_eq!(n_in, weight_in.len());
        debug_assert_eq!(n_in, var_scratch.len());
        debug_assert_eq!(n_out, y_out.len());
        debug_assert_eq!(n_out, weight_out.len());

        unsafe {
            // invert weights once per pass, since input samples are often re-used
            // 1.0 / 0.0 == +inf under IEEE754, no panic, will revert to 0.0
            // when re-inverted to weights
            for k in 0..n_in {
                let w: f64 = (*weight_in.uget(k)).as_();
                *var_scratch.get_unchecked_mut(k) = 1.0 / w;
            }

            for j in 0..n_out {
                // indices and coefficients
                let i0 = *self.i0.get_unchecked(j);
                let coeffs = *self.coeffs.get_unchecked(j);
                let valid = *self.valid.get_unchecked(j);

                let mut good_coeff_sum: f64 = 0.0;
                let mut ngood: usize = 0;
                // sort out valid taps in this window
                for k in 0..N {
                    let var_k = *var_scratch.get_unchecked(i0 + k);
                    if var_k.is_finite() {
                        good_coeff_sum += *coeffs.get_unchecked(k);
                        ngood += 1;
                    }
                }

                let good = f64::from(u32::from(ngood >= min_valid_taps));

                let mut value_acc: f64 = 0.0;
                let mut var_acc: f64 = 0.0;
                // single loop over N taps, accumulating only good taps
                for k in 0..N {
                    let idx = i0 + k;
                    let var_k = *var_scratch.get_unchecked(idx);
                    if !var_k.is_finite() {
                        continue;
                    }
                    let c = *coeffs.get_unchecked(k) / good_coeff_sum;

                    let yk: f64 = (*y_in.uget(idx)).as_();
                    value_acc = c.mul_add(yk, value_acc);
                    var_acc += (c * c * var_k).max(0.0);
                }

                *y_out.uget_mut(j) = T::from_f64(value_acc);
                *weight_out.uget_mut(j) = T::from_f64(good * valid / var_acc);
            }
        }
    }
}
