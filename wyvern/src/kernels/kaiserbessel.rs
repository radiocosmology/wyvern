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
