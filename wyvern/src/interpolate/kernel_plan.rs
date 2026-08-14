//! Implementation of [`InterpolationPlan`] for a kernel-based interpolator
use super::helpers::{invert_no_zero, median_abs_sample_spacing};
use super::interpolator::{InterpolationPlan, Interpolator, IntoInterpolator};
use crate::kernels::Kernel;
use crate::types::FloatLike;
use ndarray::{ArrayView1, ArrayViewMut1, Axis};

/// Precomputed interpolation plan for a lanczos kernel
pub struct KernelInterpolator<const N: usize> {
    // index of the first window tap
    i0: Vec<usize>,
    // kernel coefficients
    coeffs: Vec<[f64; N]>,
    // mask for valid samples
    valid: Vec<f64>,
    // track the bracket indices for the kernel center
    center_a: Vec<usize>,
}

impl<const N: usize> KernelInterpolator<N> {
    /// Build an interpolation plan for a kernel-based interpolator.
    ///
    /// # Parameters
    /// ``x_in``: sorted, arbitrary spacing, len >= N
    /// ``x_out``: sorted, uniform spacing, len >= 1
    /// ``kernel``: kernel function
    /// ``filter_scale``: kernel point separation scaling factor
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
        kernel: &impl Kernel,
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

        #[allow(
            clippy::cast_precision_loss,
            clippy::integer_division,
            reason = "values too small for precision loss and integer division is desired"
        )]
        // kernel half-width as a float and integer
        let (a_half, a_half_isize) = {
            let ah = N / 2;
            (ah as f64, ah.cast_signed())
        };

        // require that the kernel matches the compiled half-width, rather
        // than deriving from the kernel itself
        // half-width should be an integer, so allow only a very
        // small tolerance
        // NB: this would probably be nice to change
        if (kernel.half_width() - a_half).abs() >= 1.0e-10 {
            eyre::bail!(
                "kernel half-width `{:?}` is not equal to expected half-width `{a_half}`",
                kernel.half_width()
            );
        }

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
        })
    }
}

impl<const N: usize> InterpolationPlan for KernelInterpolator<N> {
    #[inline]
    fn len(&self) -> usize {
        self.i0.len()
    }
}

impl<const N: usize> IntoInterpolator for KernelInterpolator<N> {
    #[inline]
    fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T> {
        self
    }
}

impl<T: FloatLike, const N: usize> Interpolator<T> for KernelInterpolator<N> {
    #[inline]
    fn interp_row(&self, y_in: &ArrayView1<T>, mut y_out: ArrayViewMut1<T>) {
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
                let yj = y_in.as_ptr().add(i0 * ystride);

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
    fn interp_row_masked(
        &self,
        y_in: &ArrayView1<T>,
        mask_in: &mut [f64],
        mut y_out: ArrayViewMut1<T>,
    ) {
        let n_in = y_in.len();
        let n_out = self.len();

        // `y_in` might have stride 2 if this is a view into a complex array
        let ystride = y_in.stride_of(Axis(0));
        debug_assert!(ystride > 0, "y must have positive strides");
        let ystride = ystride.cast_unsigned();

        debug_assert_eq!(n_in, mask_in.len());
        debug_assert_eq!(n_out, y_out.len());

        unsafe {
            for j in 0..n_out {
                // indices and coefficients
                let i0 = *self.i0.get_unchecked(j);

                let coeffs = (*self.coeffs.get_unchecked(j)).as_ptr();
                let yj = y_in.as_ptr().add(i0 * ystride);
                let mj = mask_in.as_ptr().add(i0);

                // record masked taps as well and accumulate
                // renormalisation factor
                let mut renorm: f64 = 0.0;
                let mut value_acc: f64 = 0.0;

                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "mask values are only 0.0 or 1.0"
                )]
                for k in 0..N {
                    let ck = *coeffs.add(k);
                    let mk = *mj.add(k);
                    let yk = (*yj.add(k * ystride)).as_();

                    // accumulate data and variance
                    let mck = mk * ck;
                    value_acc = mck.mul_add(yk, value_acc);
                    // accumulate updated coefficient norm
                    renorm += mck;
                }

                // variance is normalized by the new coefficient sum squared, inverted,
                // and multiplied with the sample masks
                let valid = *self.valid.get_unchecked(j);
                // valid only if window center falls between two valid samples
                let a_idx = *self.center_a.get_unchecked(j);
                let mask =
                    valid * *mask_in.get_unchecked(a_idx) * *mask_in.get_unchecked(a_idx + 1);
                // Invert the norm, zeroing the sample if `renorm` is zero. The
                // corresponding weight will also be zeroed
                let inv_norm = invert_no_zero(renorm);

                *y_out.uget_mut(j) = T::from_f64(mask * value_acc * inv_norm);
            }
        }
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
                *mask_scratch.get_unchecked_mut(k) = f64::from(w > 0.0 && w < f64::INFINITY);
            }

            for j in 0..n_out {
                // indices and coefficients
                let i0 = *self.i0.get_unchecked(j);

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

                // variance is normalized by the new coefficient sum squared, inverted,
                // and multiplied with the sample masks
                let valid = *self.valid.get_unchecked(j);
                // valid only if window center falls between two valid samples
                let a_idx = *self.center_a.get_unchecked(j);
                let mask = valid
                    * *mask_scratch.get_unchecked(a_idx)
                    * *mask_scratch.get_unchecked(a_idx + 1);
                // Invert the norm, zeroing the sample if `renorm` is zero. The
                // corresponding weight will also be zeroed
                let inv_norm = invert_no_zero(renorm);
                *y_out.uget_mut(j) = T::from_f64(mask * value_acc * inv_norm);
                let inv_var = invert_no_zero(var_acc);
                *weight_out.uget_mut(j) = T::from_f64(renorm * renorm * mask * inv_var);
            }
        }
    }
}

/// Construct a [`DynamicKernelInterpolator`] enum for any number of supported
/// tap widths
macro_rules! define_dynamic_kernel_plan {
    ($($n:literal),+ $(,)?) => {
        paste::paste! {
            pub enum DynamicKernelInterpolator {
                $(
                    [<W $n>](KernelInterpolator<$n>),
                )+
            }

            impl DynamicKernelInterpolator {
                /// Build an interpolation plan for a kernel-based interpolator.
                ///
                /// Choose the smallest supported `N` that covers the required
                /// number of taps, after scaling to account for downsampling.
                ///
                /// # Parameters
                /// ``x_in``: sorted, arbitrary spacing, len >= N
                /// ``x_out``: sorted, uniform spacing, len >= 1
                /// ``n_taps``: number of desired window taps. Window scaling is
                ///             applied when downsampling.
                /// ``kernel``: kernel function
                ///
                /// # Returns
                /// [`DynamicKernelInterpolator`]
                ///
                /// # Errors
                /// If input sample indices are unsorted or repeated, or too few
                /// samples are provided.
                pub fn build(
                    x_in: &[f64],
                    x_out: &[f64],
                    n_taps: usize,
                    mut kernel: impl Kernel,
                ) -> eyre::Result<Self> {
                    if n_taps < 2 {
                        eyre::bail!("require at least 2 taps!");
                    }
                    // compute the required filter scaling
                    let (required_taps, filter_scale) = compute_scaled_taps(x_in, x_out, n_taps)?;
                    // rescale the kernel
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "required_taps will not large enough for precision loss"
                    )]
                    kernel.update_half_width(required_taps as f64);

                    $(
                        if required_taps <= $n {
                            return Ok(Self::[<W $n>](KernelInterpolator::<$n>::build(x_in, x_out, &kernel, filter_scale)?));
                        }
                    )+

                    let max_supported = [$($n),+].into_iter().max().unwrap();
                    eyre::bail!(
                        "required kernel width for requested taps `{n_taps}` and (down)scaling factor \
                        `{filter_scale}` is `{required_taps}`, which is larger than the largest supported \
                        kernel size: `{max_supported}`. Note that when upsampling, the scaling factor is
                        fixed at `1.0`."
                    );
                }
            }

            impl InterpolationPlan for DynamicKernelInterpolator {
                #[inline]
                fn len(&self) -> usize {
                    match self {
                        $( Self::[<W $n>](p) => p.len(), )+
                    }
                }
            }

            impl IntoInterpolator for DynamicKernelInterpolator {
                /// Returns a `%dyn Interpolator<T>` for callers to extract the
                /// underlying typed interpolator
                fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T>
                where
                    $( KernelInterpolator<$n>: Interpolator<T>, )+
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

/// Kernel ratio scaling. Support is limited to be greater than 1.0,
/// meaning that support is unchanged when upsampling
#[inline]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
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
