//! Implementation of a linear interpolator
use ndarray::{ArrayView1, ArrayViewMut1};

use crate::types::FloatLike;

pub trait InterpolationPlan {
    // Number of output samples produced by this plan
    fn len(&self) -> usize;

    // Interpolate a single row's data onto output points
    fn interp_row<T: FloatLike>(&self, y_in: &ArrayView1<T>, y_out: ArrayViewMut1<T>);

    // Interpolate a single row's data onto output points,
    // and propagate corresponding inverse-variance weights
    fn interp_row_with_variance<T: FloatLike>(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        y_out: ArrayViewMut1<T>,
        weight_out: ArrayViewMut1<T>,
    );
}

/// Computes the median of |diff(x)| -- matches np.median(np.abs(np.diff(lsd))).
/// Requires a mutable scratch Vec to avoid an extra allocation if you
/// call this repeatedly; sorts in place.
fn median_abs_sample_spacing(x: &[f64]) -> f64 {
    debug_assert!(
        x.len() > 1,
        "need at least 2 input samples to compute a diff"
    );
    #[allow(
        clippy::indexing_slicing,
        reason = "windows(2) ensures that indexing won't panic"
    )]
    let mut diffs: Vec<f64> = x.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    let n = diffs.len();
    let mid = n >> 1;

    diffs.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *diffs.get(mid).unwrap_or(&0.0_f64);

    if n.is_multiple_of(2) {
        diffs.select_nth_unstable_by(mid - 1, f64::total_cmp);
        f64::midpoint(*diffs.get(mid - 1).unwrap_or(&0.0_f64), upper)
    } else {
        upper
    }
}

/// Precomputed interpolation plan for mapping input
/// and output samples.
pub struct LinearPlan {
    // lower bracket index for input
    i0: Vec<usize>,
    // upper brackeet index for input
    i1: Vec<usize>,
    // interpolation coefficient for i1 sample (w0 = 1 - w1)
    c1: Vec<f64>,
    // mask for valid samples. 1.0 if valid, 0.0 otherwise
    valid: Vec<f64>,
}

impl LinearPlan {
    /// ``x_in``: sorted, arbitrary spacing, len >= 2
    /// ``x_out``: sorted, uniform spacing, len >= 1
    pub fn build(x_in: &[f64], x_out: &[f64]) -> eyre::Result<Self> {
        let n_out = x_out.len();
        let n_in = x_in.len();

        debug_assert!(n_in >= 2, "minimum 2 input samples are required!");
        debug_assert!(n_out >= 1, "minimum 1 output sample is required!");

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut i1 = Vec::<usize>::with_capacity(n_out);
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
            i1.push(lo + 1);
            // weight coefficient
            c1.push((xo - a) / span);
        }

        Ok(Self { i0, i1, c1, valid })
    }
}

impl InterpolationPlan for LinearPlan {
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
                // extract the interpolation indices and weight
                let i0 = *self.i0.get_unchecked(j);
                let i1 = *self.i1.get_unchecked(j);
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
                // extract the interpolation indices and weight
                let i0 = *self.i0.get_unchecked(j);
                let i1 = *self.i1.get_unchecked(j);
                // interpolation coefficients
                let s1 = *self.c1.get_unchecked(j);

                // interpolate data onto the target sample
                let a: f64 = (*y_in.uget(i0)).as_();
                let b: f64 = (*y_in.uget(i1)).as_();
                *y_out.uget_mut(j) = T::from_f64((b - a).mul_add(s1, a));

                // propagate weights and masking
                let var_a = *var_scratch.get_unchecked(i0);
                let var_b = *var_scratch.get_unchecked(i1);
                let valid = *self.valid.get_unchecked(j);

                let s0 = 1.0 - s1;
                // NaN guard: (0.0 * inf) -> NaN -> clamped to 0.0
                // for invalid items
                let c0 = (s0 * s0 * var_a).max(0.0);
                let c1 = (s1 * s1 * var_b).max(0.0);
                // valid is either 1.0 or 0.0
                *weight_out.uget_mut(j) = T::from_f64(valid / (c0 + c1));
            }
        }
    }
}

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
        let (a_half, a_half_idx) = {
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

            let distant = (b - xo).abs() > delta || (a - xo).abs() > delta;
            valid.push(f64::from(!distant));

            // construct a window a N taps centred on the bracket, clamped
            // to [0, n_in - N]. Coefficients must be renormalized
            let center = lo + 1; // closest index biased towards `b`
            let base = (center.cast_signed() - a_half_idx)
                .clamp(0, n_max)
                .cast_unsigned();

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
            // Normalize as long as there are some samples
            if sum.abs() >= 1e-12 {
                for ci in &mut c {
                    *ci /= sum;
                }
            }

            i0.push(base);
            coeffs.push(c);
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

                let mut value_acc: f64 = 0.0;
                let mut var_acc: f64 = 0.0;

                // single loop over N taps
                for k in 0..N {
                    let idx = i0 + k;
                    let c = *coeffs.get_unchecked(k);

                    let yk: f64 = (*y_in.uget(idx)).as_();
                    value_acc = c.mul_add(yk, value_acc);

                    let var_k = *var_scratch.get_unchecked(idx);
                    var_acc += (c * c * var_k).max(0.0);
                }

                *y_out.uget_mut(j) = T::from_f64(value_acc);
                *weight_out.uget_mut(j) = T::from_f64(valid / var_acc);
            }
        }
    }
}
