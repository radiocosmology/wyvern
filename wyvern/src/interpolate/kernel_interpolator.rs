//! Implementation of [`InterpolationPlan`] for a kernel-based interpolator
use num_traits::AsPrimitive;

use super::helpers::{invert_no_zero, median_abs_sample_spacing};
use super::interpolator::{InterpolationPlan, Interpolator, IntoInterpolator};

use crate::kernels::traits::Kernel;
use crate::types::{FloatLike, MaybeComplex, as_real_slice, as_real_slice_mut};
use crate::util::assert_unchecked_debug;

/// Precomputed interpolation plan for a fixed-width kernel window.
pub struct FixedWidthKernelInterpolator<const N: usize> {
    /// Index of the first kernel tap used for each output sample.
    i0: Vec<usize>,
    /// Kernel coefficients computed for each output sample.
    coeffs: Vec<[f64; N]>,
    /// Validity mask for each output sample.
    valid: Vec<f64>,
    /// Bracket indices used to locate the kernel center in the input domain.
    center_a: Vec<usize>,
    /// Number of input samples managed by the plan.
    n_in: usize,
}

impl<const N: usize> FixedWidthKernelInterpolator<N> {
    /// Build an interpolation plan for a kernel-based interpolator.
    ///
    /// # Parameters
    /// * ``x_in``: sorted, arbitrary spacing, len >= N
    /// * ``x_out``: sorted, uniform spacing, len >= 1
    /// * ``kernel``: kernel function
    /// * ``filter_scale``: kernel point separation downscaling factor
    ///
    /// # Returns
    /// [`FixedWidthKernelInterpolator`]
    ///
    /// # Errors
    /// If input sample indices are unsorted or repeated, or too few
    /// samples are provided.
    pub fn build(
        x_in: &[f64],
        x_out: &[f64],
        kernel: &dyn Kernel,
        filter_scale: f64,
    ) -> eyre::Result<Self> {
        let n_in = x_in.len();
        let n_out = x_out.len();
        // validate inputs
        if n_in < N {
            eyre::bail!("need at least N={N} samples!");
        }
        if n_out < 1 {
            eyre::bail!("need at least 1 output sample!");
        }
        if kernel.ntaps() > N {
            eyre::bail!(
                "kernel num taps ({}) is greater than available window size {N}",
                kernel.ntaps()
            );
        }

        #[allow(
            clippy::cast_precision_loss,
            clippy::integer_division,
            reason = "values too small for precision loss and integer division is desired"
        )]
        // kernel half-width as a float and integer
        let a_half = kernel.half_width();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "possible kernel width values are far too small for truncation to occur"
        )]
        let a_half_isize = a_half as isize;

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut center_a = Vec::<usize>::with_capacity(n_out);
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
            let scaled_inv_span = 1.0 / (span * filter_scale);

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
                let dist = (xo - xi) * scaled_inv_span;
                // let w = kernel(dist, a_half);
                let w = kernel.evaluate(dist);
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
            coeffs.push(c);
            valid.push(f64::from(!(distant || outside_window) && sum.is_finite()));
        }

        Ok(Self {
            i0,
            coeffs,
            valid,
            center_a,
            n_in,
        })
    }
}

impl<const N: usize> InterpolationPlan for FixedWidthKernelInterpolator<N> {
    #[inline]
    fn len(&self) -> usize {
        self.i0.len()
    }

    #[inline]
    fn n_in(&self) -> usize {
        self.n_in
    }
}

impl<const N: usize> IntoInterpolator for FixedWidthKernelInterpolator<N> {
    #[inline]
    fn as_interpolator<T: MaybeComplex>(&self) -> &dyn Interpolator<T> {
        self
    }
}

impl<T: MaybeComplex, const N: usize> Interpolator<T> for FixedWidthKernelInterpolator<N> {
    #[inline]
    fn interp_row(&self, y_in: &[T], y_out: &mut [T]) {
        // reinterpret as real slice
        let y_in = as_real_slice(y_in);
        let y_out = as_real_slice_mut(y_out);
        let stride = if T::IS_COMPLEX { 2 } else { 1 };

        assert_eq!(self.len() * stride, y_out.len());
        assert_eq!(self.n_in() * stride, y_in.len());

        self.i0
            .iter()
            .zip(self.coeffs.iter())
            .zip(y_out.chunks_exact_mut(stride))
            .for_each(|((i0, c0), yo)| {
                assert_unchecked_debug!(*i0 + N <= self.n_in());

                for (k, yo_k) in yo.iter_mut().enumerate() {
                    // accumulate over the kernel coefficients
                    let mut value_acc: f64 = 0.0;

                    c0.iter().enumerate().for_each(|(j, cj)| {
                        let yj =
                            unsafe { *y_in.get_unchecked(*i0 * stride + k + j * stride) }.as_();
                        value_acc = cj.mul_add(yj, value_acc);
                    });

                    *yo_k = T::Real::from_f64(value_acc);
                }
            });
    }

    #[inline]
    fn interp_row_masked(&self, y_in: &[T], mask_in: &[f64], y_out: &mut [T]) {
        // interpret as real slices
        let y_in = as_real_slice(y_in);
        let y_out = as_real_slice_mut(y_out);
        let stride = if T::IS_COMPLEX { 2 } else { 1 };

        assert_eq!(self.n_in() * stride, y_in.len());
        assert_eq!(self.n_in(), mask_in.len());
        assert_eq!(self.len() * stride, y_out.len());

        let mut masked_coeffs: Vec<f64> = vec![0.0; N];

        self.i0
            .iter()
            .zip(self.coeffs.iter())
            .zip(self.valid.iter())
            .zip(self.center_a.iter())
            .zip(y_out.chunks_exact_mut(stride))
            .for_each(|((((i0, c0), valid), a_idx), yo)| {
                assert_unchecked_debug!(*i0 + N <= self.n_in());
                assert_unchecked_debug!(*a_idx + 1 < self.n_in());

                // A sample is valid only if the window centre falls
                // between two valid samples
                let mask_a = unsafe { mask_in.get_unchecked(*a_idx) };
                let mask_b = unsafe { mask_in.get_unchecked(*a_idx + 1) };
                let mask = valid * mask_a * mask_b;

                let msl = unsafe { mask_in.get_unchecked(*i0..*i0 + N) };

                // compute kernel renormalisation to account for masking
                let mut renorm: f64 = 0.0;
                // compute masked kernel coefficients
                c0.iter()
                    .zip(msl.iter())
                    .zip(masked_coeffs.iter_mut())
                    .for_each(|((cj, mj), mcj)| {
                        *mcj = cj * mj;
                        renorm += *mcj;
                    });

                // invert the norm, zeroing the sample if `renorm` is zero
                let inv_norm = invert_no_zero(renorm);

                for (k, yo_k) in yo.iter_mut().enumerate() {
                    let mut value_acc: f64 = 0.0;
                    // iterate over masked kernel coefficients
                    for (j, mcj) in masked_coeffs.iter().enumerate() {
                        let yj =
                            unsafe { *y_in.get_unchecked(*i0 * stride + k + j * stride) }.as_();
                        value_acc = mcj.mul_add(yj, value_acc);
                    }

                    *yo_k = T::Real::from_f64(mask * value_acc * inv_norm);
                }
            });
    }

    #[inline]
    fn interp_row_with_variance(
        &self,
        y_in: &[T],
        weight_in: &[T::Real],
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        y_out: &mut [T],
        weight_out: &mut [T::Real],
    ) {
        // interpret as real slices
        let y_in = as_real_slice(y_in);
        let y_out = as_real_slice_mut(y_out);
        let stride = if T::IS_COMPLEX { 2 } else { 1 };

        assert_eq!(self.n_in() * stride, y_in.len());
        assert_eq!(self.n_in(), weight_in.len());
        assert_eq!(self.n_in(), var_scratch.len());
        assert_eq!(self.n_in(), mask_scratch.len());
        assert_eq!(self.len() * stride, y_out.len());
        assert_eq!(self.len(), weight_out.len());

        // invert weights once per pass, since input samples
        // are often reused
        weight_in
            .iter()
            .zip(var_scratch.iter_mut())
            .zip(mask_scratch.iter_mut())
            .for_each(|((w, vs), ms)| {
                let w: f64 = w.as_();
                *vs = invert_no_zero(w);
                *ms = f64::from(w > 0.0 && w < f64::INFINITY);
            });

        let mut masked_coeffs: Vec<f64> = vec![0.0; N];

        self.i0
            .iter()
            .zip(self.coeffs.iter())
            .zip(self.valid.iter())
            .zip(self.center_a.iter())
            .zip(y_out.chunks_exact_mut(stride))
            .zip(weight_out.iter_mut())
            .for_each(|(((((i0, c0), valid), a_idx), yo), wo)| {
                assert_unchecked_debug!(*i0 + N <= self.n_in());
                assert_unchecked_debug!(*a_idx + 1 < self.n_in());

                let vsl = unsafe { var_scratch.get_unchecked(*i0..*i0 + N) };
                let msl = unsafe { mask_scratch.get_unchecked(*i0..*i0 + N) };

                // variance is normalized by the new coefficient sum squared, inverted,
                // and multiplied with the sample masks
                // valid only if window center falls between two valid samples
                let mask_a = unsafe { mask_scratch.get_unchecked(*a_idx) };
                let mask_b = unsafe { mask_scratch.get_unchecked(*a_idx + 1) };
                let mask = valid * mask_a * mask_b;

                // accumulate variance, only requires one pass
                let mut renorm: f64 = 0.0;
                let mut var_acc: f64 = 0.0;
                // the masked coefficients are re-used when accumulating
                // the data below, so avoid re-computing
                c0.iter()
                    .zip(msl.iter())
                    .zip(vsl.iter())
                    .zip(masked_coeffs.iter_mut())
                    .for_each(|(((cj, mj), vj), mcj)| {
                        *mcj = cj * mj;
                        renorm += *mcj;
                        var_acc = (*mcj * cj).mul_add(*vj, var_acc);
                    });

                // invert the norm, zeroing the sample if `renorm` is zero. The
                // corresponding weight will also be zeroed
                let inv_norm = invert_no_zero(renorm);
                let inv_var = invert_no_zero(var_acc);

                *wo = T::Real::from_f64(renorm * renorm * mask * inv_var);

                // now accumulate the data
                for (k, yo_k) in yo.iter_mut().enumerate() {
                    let mut value_acc: f64 = 0.0;
                    // iterate over masked kernel coefficients. normalisation
                    // has already been computed
                    for (j, mcj) in masked_coeffs.iter().enumerate() {
                        let yj =
                            unsafe { *y_in.get_unchecked(*i0 * stride + k + j * stride) }.as_();
                        // accumulate data and variance
                        value_acc = mcj.mul_add(yj, value_acc);
                    }

                    *yo_k = T::Real::from_f64(mask * value_acc * inv_norm);
                }
            });
    }
}

/// Construct a [`KernelInterpolator`] enum for any number of supported tap
/// widths.
macro_rules! define_dynamic_kernel_plan {
    ($($n:literal),+ $(,)?) => {
        paste::paste! {
            /// Runtime-dispatched kernel interpolator that selects the smallest
            /// fixed-width plan that satisfies the requested tap count.
            pub enum KernelInterpolator {
                $(
                    [<W $n>](FixedWidthKernelInterpolator<$n>),
                )+
            }

            impl KernelInterpolator {
                /// Build an interpolation plan for a kernel-based interpolator.
                ///
                /// Choose the smallest supported `N` that covers the required
                /// number of taps, after scaling to account for downsampling.
                ///
                /// # Parameters
                /// * ``x_in``: sorted, arbitrary spacing, len >= N
                /// * ``x_out``: sorted, uniform spacing, len >= 1
                /// * ``kernel``: kernel function
                /// * ``filter_scale``: kernel point separation downscaling factor
                ///
                /// # Returns
                /// [`KernelInterpolator`]
                ///
                /// # Errors
                /// If input sample indices are unsorted or repeated, or too few
                /// samples are provided.
                pub fn build(
                    x_in: &[f64],
                    x_out: &[f64],
                    kernel: &mut dyn Kernel,
                    filter_scale: Option<f64>,
                ) -> eyre::Result<Self> {
                    // let n_taps = kernel.ntaps();

                    let (required_taps, scale) = if let Some(scale) = filter_scale {
                        // check that the required scale is acceptable
                        if !scale.is_finite() || scale < 0.0 {
                            eyre::bail!("filter scale must be finite and positive; got {scale}!");
                        }
                        // compute the required filter scaling
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            clippy::cast_precision_loss,
                            reason = "kernel width will never be large enough to be truncated"
                        )]
                        let required_taps = (kernel.ntaps() as f64 * scale).ceil() as usize;
                        // rescale the kernel
                        kernel.set_ntaps(required_taps);

                        (required_taps, scale)
                    } else {
                        (kernel.ntaps(), 1.0)
                    };

                    $(
                        if required_taps <= $n {
                            return Ok(Self::[<W $n>](FixedWidthKernelInterpolator::<$n>::build(x_in, x_out, kernel, scale)?));
                        }
                    )+

                    let max_supported = [$($n),+].into_iter().max().unwrap();
                    eyre::bail!(
                        "required kernel width for requested taps `{:?}` and (down)scaling factor \
                        `{scale}` is `{required_taps}`, which is larger than the largest supported \
                        kernel size: `{max_supported}`. Note that when upsampling, the scaling factor is
                        fixed at `1.0`.", kernel.ntaps()
                    );
                }
            }

            impl InterpolationPlan for KernelInterpolator {
                #[inline]
                fn len(&self) -> usize {
                    match self {
                        $( Self::[<W $n>](p) => p.len(), )+
                    }
                }

                #[inline]
                fn n_in(&self) -> usize {
                    match self {
                        $( Self::[<W $n>](p) => p.n_in(), )+
                    }
                }
            }

            impl IntoInterpolator for KernelInterpolator {
                /// Returns a `%dyn Interpolator<T>` for callers to extract the
                /// underlying typed interpolator
                fn as_interpolator<T: MaybeComplex>(&self) -> &dyn Interpolator<T>
                where
                    $( FixedWidthKernelInterpolator<$n>: Interpolator<T>, )+
                {
                    match self {
                        $( Self::[<W $n>](p) => p, )+
                    }
                }
            }
        }
    }
}

// powers of 2 + 1 seems like reasonable choices for no
// valid reason
define_dynamic_kernel_plan!(4, 8, 16, 32, 64, 128, 256);

/// Compute the effective tap scaling needed for the requested output spacing.
///
/// # Parameters
/// * `x_in`: Sorted input sample coordinates.
/// * `x_out`: Sorted output sample coordinates.
/// * `requested_taps`: The minimum number of kernel taps requested by the caller.
///
/// # Returns
/// A tuple containing the adjusted tap count and the computed filter scale.
#[inline]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    dead_code,
    reason = "precision loss would only occur with unreasonable num samples"
)]
fn compute_scaled_taps(
    x_in: &[f64],
    x_out: &[f64],
    requested_taps: usize,
) -> eyre::Result<(usize, f64)> {
    let n_in = x_in.len();
    let n_out = x_out.len();

    if n_in < requested_taps {
        eyre::bail!("need at least N={requested_taps} samples!");
    }

    if n_out < 1 {
        eyre::bail!("need at least 1 output sample!");
    }

    #[allow(clippy::indexing_slicing, reason = "indices already checked")]
    let (in_domain, out_domain) = { (x_in[n_in - 1] - x_in[0], x_out[n_out - 1] - x_out[0]) };

    // ratio of physical spacings - >1 means downsampling
    let in_spacing = in_domain / n_in.saturating_sub(1).max(1) as f64;
    let out_spacing = out_domain / n_out.saturating_sub(1).max(1) as f64;

    let filter_scale = (out_spacing / in_spacing).max(1.0);

    Ok((
        (requested_taps as f64 * filter_scale).ceil() as usize,
        filter_scale,
    ))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::err_expect,
    reason = "unwrap/expect, exact comparisons, and direct indexing are acceptable in test code; \
    `err_expect` would require `Debug` on plan types that intentionally don't implement it"
)]
mod tests {
    use super::*;
    use crate::kernels::BoxcarKernel;

    #[test]
    fn build_rejects_too_few_input_samples() {
        let kernel = BoxcarKernel::build(4);
        let err = FixedWidthKernelInterpolator::<4>::build(&[0.0, 1.0], &[0.5], &kernel, 1.0)
            .err()
            .expect("build should fail with too few samples");
        assert!(err.to_string().contains("need at least N=4 samples"));
    }

    #[test]
    fn build_rejects_empty_output() {
        let x_in = [0.0, 1.0, 2.0, 3.0];
        let kernel = BoxcarKernel::build(4);
        let err = FixedWidthKernelInterpolator::<4>::build(&x_in, &[], &kernel, 1.0)
            .err()
            .expect("build should fail with empty output");
        assert!(err.to_string().contains("need at least 1 output sample"));
    }

    #[test]
    fn build_rejects_kernel_wider_than_window() {
        let x_in = [0.0, 1.0, 2.0, 3.0];
        let kernel = BoxcarKernel::build(8);
        let err = FixedWidthKernelInterpolator::<4>::build(&x_in, &[1.0], &kernel, 1.0)
            .err()
            .expect("build should fail when kernel exceeds window");
        assert!(
            err.to_string()
                .contains("greater than available window size")
        );
    }

    #[test]
    fn interp_row_reproduces_constant_signal() {
        let x_in = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let x_out = [2.0, 2.5, 3.0];
        let kernel = BoxcarKernel::build(4);
        let plan = FixedWidthKernelInterpolator::<4>::build(&x_in, &x_out, &kernel, 1.0).unwrap();

        assert_eq!(plan.n_in(), 6);
        assert_eq!(plan.len(), 3);

        let y_in = [3.0_f64; 6];
        let mut y_out = [0.0_f64; 3];
        Interpolator::<f64>::interp_row(&plan, &y_in, &mut y_out);

        // constant input should interpolate to the same constant, since
        // coefficients are renormalized to sum to 1
        for value in y_out {
            assert!((value - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn kernel_interpolator_dispatches_to_smallest_matching_width() {
        let x_in: Vec<f64> = (0..10).map(f64::from).collect();
        let x_out = vec![4.5];
        let mut kernel = BoxcarKernel::build(4);
        let plan = KernelInterpolator::build(&x_in, &x_out, &mut kernel, None).unwrap();

        assert_eq!(plan.n_in(), 10);
        assert_eq!(plan.len(), 1);
        assert!(matches!(plan, KernelInterpolator::W4(_)));
    }

    #[test]
    fn kernel_interpolator_errors_when_taps_exceed_max_supported() {
        let x_in: Vec<f64> = (0..300).map(f64::from).collect();
        let x_out = vec![150.5];
        let mut kernel = BoxcarKernel::build(300);
        let err = KernelInterpolator::build(&x_in, &x_out, &mut kernel, None)
            .err()
            .expect("build should fail when taps exceed max supported width");
        assert!(
            err.to_string()
                .contains("larger than the largest supported")
        );
    }

    #[test]
    fn compute_scaled_taps_upsampling_keeps_scale_at_one() {
        let x_in = [0.0, 1.0, 2.0, 3.0];
        let x_out = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let (taps, scale) = compute_scaled_taps(&x_in, &x_out, 4).unwrap();
        assert_eq!(scale, 1.0);
        assert_eq!(taps, 4);
    }

    #[test]
    fn compute_scaled_taps_downsampling_scales_up() {
        let x_in = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let x_out = [0.0, 7.0];
        let (taps, scale) = compute_scaled_taps(&x_in, &x_out, 4).unwrap();
        assert!(scale > 1.0);
        assert!(taps >= 4);
    }

    #[test]
    fn compute_scaled_taps_rejects_too_few_samples() {
        let x_in = [0.0, 1.0];
        let x_out = [0.5];
        let err = compute_scaled_taps(&x_in, &x_out, 4)
            .err()
            .expect("compute_scaled_taps should fail with too few samples");
        assert!(err.to_string().contains("need at least N=4 samples"));
    }
}
