//! Kaiser-bessel window with fixed a -> beta relationship
use super::traits::Kernel;
use puruspe::bessel::In;

/// A Kaiser-Bessel window kernel with a beta parameter derived from its width.
#[derive(Debug, Clone)]
pub struct KaiserBesselKernel {
    /// Number of taps in the kernel support.
    ntaps: usize,
    /// Half-width used to define the window support.
    a: f64,
    /// Shape parameter that controls the window taper.
    beta: f64,
    /// Cached value of the modified Bessel function at `beta`.
    i0_beta: f64,
}

impl KaiserBesselKernel {
    #[inline]
    fn beta_from_width_default(a: f64) -> f64 {
        // Empirical constant relating `beta` to `a`
        std::f64::consts::PI * a
    }

    /// Returns the current beta parameter used by the window.
    ///
    /// # Returns
    /// The shape parameter controlling the Kaiser-Bessel taper.
    #[inline]
    #[must_use]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Update the kernel beta parameter.
    ///
    /// # Parameters
    /// * `beta`: The new shape parameter for the window.
    ///
    /// This value will still be rescaled if `ntaps` is updated later.
    pub fn set_beta(&mut self, beta: f64) {
        self.beta = beta;
        self.i0_beta = In(0, beta);
    }

    /// Restore `beta` to the default value derived from the kernel width.
    pub fn set_beta_default(&mut self) {
        self.beta = Self::beta_from_width_default(self.a);
        self.i0_beta = In(0, self.beta);
    }
}

impl Kernel for KaiserBesselKernel {
    #[allow(
        clippy::integer_division,
        clippy::cast_precision_loss,
        reason = "truncated division is desired"
    )]
    fn build(ntaps: usize) -> Self {
        let a = (ntaps / 2) as f64;
        // use pi * a as a default
        let beta = Self::beta_from_width_default(a);
        let i0_beta = In(0, beta);

        Self {
            ntaps,
            a,
            beta,
            i0_beta,
        }
    }

    #[inline]
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() > self.a {
            0.0
        } else {
            let ratio = x / self.a;
            let arg = self.beta * ratio.mul_add(-ratio, 1.0).sqrt();
            In(0, arg) / self.i0_beta
        }
    }

    #[inline]
    fn half_width(&self) -> f64 {
        self.a
    }

    #[inline]
    fn ntaps(&self) -> usize {
        self.ntaps
    }

    #[allow(
        clippy::integer_division,
        clippy::cast_precision_loss,
        reason = "truncated division is desired"
    )]
    fn set_ntaps(&mut self, ntaps: usize) {
        self.ntaps = ntaps;
        let a_new = (ntaps / 2) as f64;
        // scale `beta` accordingly
        self.beta = if self.a == 0.0 {
            Self::beta_from_width_default(a_new)
        } else {
            self.beta * a_new / self.a
        };
        self.a = a_new;
        self.i0_beta = In(0, self.beta);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "exact comparisons are expected in tests")]
mod tests {
    use super::*;

    #[test]
    fn build_sets_ntaps_half_width_and_default_beta() {
        let k = KaiserBesselKernel::build(8);
        assert_eq!(k.ntaps(), 8);
        assert_eq!(k.half_width(), 4.0);
        assert_eq!(k.beta(), std::f64::consts::PI * 4.0);
    }

    #[test]
    fn evaluate_is_one_at_center_and_zero_outside_support() {
        let k = KaiserBesselKernel::build(8);
        assert!((k.evaluate(0.0) - 1.0).abs() < 1e-9);
        assert_eq!(k.evaluate(4.1), 0.0);
        assert_eq!(k.evaluate(-4.1), 0.0);
    }

    #[test]
    fn set_beta_updates_cached_bessel_value() {
        let mut k = KaiserBesselKernel::build(8);
        k.set_beta(2.0);
        assert_eq!(k.beta(), 2.0);
        // evaluating at the center should still normalize to ~1.0
        assert!((k.evaluate(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn set_beta_default_restores_width_derived_beta() {
        let mut k = KaiserBesselKernel::build(8);
        k.set_beta(2.0);
        k.set_beta_default();
        assert_eq!(k.beta(), std::f64::consts::PI * k.half_width());
    }

    #[test]
    fn set_ntaps_rescales_beta_proportionally() {
        let mut k = KaiserBesselKernel::build(8);
        let original_ratio = k.beta() / k.half_width();
        k.set_ntaps(16);
        assert_eq!(k.ntaps(), 16);
        assert_eq!(k.half_width(), 8.0);
        // beta scales linearly with half-width when ntaps changes
        assert!((k.beta() / k.half_width() - original_ratio).abs() < 1e-9);
    }
}
