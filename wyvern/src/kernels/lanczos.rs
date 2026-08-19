//! Truncated sinc Lanczos kernel
use super::traits::Kernel;

#[derive(Debug)]
pub struct LanczosKernel {
    /// Number of taps
    ntaps: usize,
    /// Width parameter
    a: f64,
}

impl Kernel for LanczosKernel {
    #[allow(
        clippy::integer_division,
        clippy::cast_precision_loss,
        reason = "truncated division is desired"
    )]
    fn build(ntaps: usize) -> Self {
        let a = (ntaps / 2) as f64;
        Self { ntaps, a }
    }

    #[inline]
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() >= self.a {
            0.0
        } else {
            sinc(x) * sinc(x / self.a)
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
        self.a = (ntaps / 2) as f64;
        self.ntaps = ntaps;
    }
}

/// Simple `sinc` implementation
#[inline]
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}
