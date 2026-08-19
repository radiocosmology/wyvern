//! Kaiser-bessel window with fixed a -> beta relationship
use super::traits::Kernel;

#[derive(Debug)]
pub struct KaiserBesselKernel {
    /// Number of taps
    ntaps: usize,
    /// Width parameter
    a: f64,
    /// Shape parameter, derived from `a`
    beta: f64,
    /// Cached beta to avoid recomputation
    i0_beta: f64,
}

impl KaiserBesselKernel {
    /// Empirical constant relating `beta` to `a`
    const BETA_SCALE: f64 = 2.34;

    #[inline]
    fn beta_from_width(a: f64) -> f64 {
        Self::BETA_SCALE * 2.0 * a
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
        let beta = Self::beta_from_width(a);
        let i0_beta = xsf::bessel_i0(beta);

        Self {
            ntaps,
            a,
            beta,
            i0_beta,
        }
    }

    #[inline]
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() >= self.a {
            0.0
        } else {
            let ratio = x / self.a;
            let arg = self.beta * ratio.mul_add(-ratio, 1.0).sqrt();
            xsf::bessel_i0(arg) / self.i0_beta
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
        self.a = (ntaps / 2) as f64;
        self.beta = Self::beta_from_width(self.a);
        self.i0_beta = xsf::bessel_i0(self.beta);
    }
}
