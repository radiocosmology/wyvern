//! Truncated sinc Lanczos kernel
use super::traits::Kernel;

/// A truncated sinc kernel based on the Lanczos window.
#[derive(Debug, Clone)]
pub struct LanczosKernel {
    /// Number of taps in the kernel support.
    ntaps: usize,
    /// Half-width used to define the sinc window.
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

/// Compute the normalized sinc function used by the Lanczos kernel.
///
/// # Parameters
/// * `x`: The coordinate at which to evaluate the sinc response.
///
/// # Returns
/// The sinc value at `x`, with the zero case handled as `1.0`.
#[inline]
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}
