//! Implementation of [`InterpolationPlan`] for a kernel-based interpolator
use super::plan::{InterpolationPlan, median_abs_sample_spacing};
use crate::types::FloatLike;
use ndarray::{ArrayView1, ArrayViewMut1, Axis};

/// Precomputed interpolation plan for a lanczos kernel
pub struct KernelPlan<const N: usize> {
    // index of the first window tap
    i0: Vec<usize>,
    // kernel coefficients
    coeffs: Vec<[f64; N]>,
    // mask for valid samples
    valid: Vec<f64>,
    // track the bracket indices for the kernel center
    center_a: Vec<usize>,
    center_b: Vec<usize>,
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
        let mut center_a = Vec::<usize>::with_capacity(n_out);
        let mut center_b = Vec::<usize>::with_capacity(n_out);
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
            let mut sum: f64 = 0.0;

            // compute the kernel coefficients
            #[allow(clippy::indexing_slicing, reason = "indices are already clamped")]
            for k in 0..N {
                let xi = x_in[base + k];
                let dist = (xo - xi) / span;
                let w = kernel(dist, a_half);
                c[k] = w;
                sum += w;
            }
            // force coefficients to 0.0 if the sum is extremely small
            if sum <= 1e-6 {
                // x / inf evaluates to zero
                sum = f64::INFINITY;
            }

            // Normalize. If `sum_degenerate` is true, `sum` is inf
            // and this evaluates ci to 0.0
            for ci in &mut c {
                *ci /= sum;
            }

            i0.push(base);
            center_a.push(lo);
            center_b.push(lo + 1);
            coeffs.push(c);
            valid.push(f64::from(!(distant || outside_window) && sum.is_finite()));
        }

        Ok(Self {
            i0,
            coeffs,
            valid,
            center_a,
            center_b,
        })
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

        // `y_in` might have stride 2 if this is a view into a complex array
        let ystride = y_in.stride_of(Axis(0));
        debug_assert!(ystride > 0, "y must have positive strides");
        let ystride = ystride.cast_unsigned();

        unsafe {
            for j in 0..n_out {
                // indices and coefficients
                let i0 = *self.i0.get_unchecked(j);
                let coeffs = (*self.coeffs.get_unchecked(j)).as_ptr();
                let yj = y_in.as_ptr().add(i0 + ystride);

                // interpolate
                let mut value_acc: f64 = 0.0;

                for k in 0..N {
                    let ck = *coeffs.add(k);
                    let yk = (*yj.add(k * ystride)).as_();

                    value_acc = ck.mul_add(yk, value_acc);
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
        mask_scratch: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
        mut weight_out: ArrayViewMut1<T>,
    ) {
        let n_in = y_in.len();
        let n_out = self.len();

        // `y_in` might have stride 2 if this is a view into a complex array
        let ystride = y_in.stride_of(Axis(0));
        debug_assert!(ystride > 0, "y must have positive strides");
        let ystride = ystride.cast_unsigned();

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
                *var_scratch.get_unchecked_mut(k) = invert_no_zero(w);
                *mask_scratch.get_unchecked_mut(k) = f64::from(w.is_finite() && w > 0.0);
            }

            for j in 0..n_out {
                // indices and coefficients
                let i0 = *self.i0.get_unchecked(j);
                // NB: using pointers here hopefully ensures that we get SIMD
                // optimisation through LLVM
                let coeffs = (*self.coeffs.get_unchecked(j)).as_ptr();
                let yj = y_in.as_ptr().add(i0 * ystride);
                let vj = var_scratch.as_ptr().add(i0);
                let mj = mask_scratch.as_ptr().add(i0);

                // record masked taps as well and accumulate
                // renormalisation factor
                let mut renorm: f64 = 0.0;
                let mut value_acc: f64 = 0.0;
                let mut var_acc: f64 = 0.0;

                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "mask values are only 0.0 or 1.0"
                )]
                for k in 0..N {
                    let ck = *coeffs.add(k);
                    let vk = *vj.add(k);
                    let mk = *mj.add(k);
                    let yk = (*yj.add(k * ystride)).as_();

                    // accumulate data and variance
                    let mck = mk * ck;
                    value_acc = mck.mul_add(yk, value_acc);
                    var_acc = (mck * ck).mul_add(vk, var_acc);
                    // accumulate updated coefficient norm
                    renorm += mck;
                }

                // Invert the norm, zeroing the sample if `renorm` is zero. The
                // corresponding weight will also be zeroed
                let inv_norm = invert_no_zero(renorm);
                *y_out.uget_mut(j) = T::from_f64(value_acc * inv_norm);
                // variance is normalized by the new coefficient sum squared, inverted,
                // and multiplied with the sample masks
                let valid = *self.valid.get_unchecked(j);
                // valid only if window center falls between two valid samples
                let a_idx = *self.center_a.get_unchecked(j);
                let b_idx = *self.center_b.get_unchecked(j);
                let center_mask =
                    *mask_scratch.get_unchecked(a_idx) * *mask_scratch.get_unchecked(b_idx);
                let inv_var = invert_no_zero(var_acc);
                *weight_out.uget_mut(j) =
                    T::from_f64(renorm * renorm * center_mask * valid * inv_var);
            }
        }
    }
}

// invert a value, or return zero if the value is zero
#[inline]
fn invert_no_zero(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

#[inline]
#[allow(dead_code, reason = "testing")]
fn invert_no_zero_branchless(x: f64) -> f64 {
    let inv = 1.0 / x;
    // bitmask - all zeros if x is zero, all ones otherwise
    let bitmask = u64::from(x != 0.0).wrapping_neg();

    f64::from_bits(inv.to_bits() & bitmask)
}
