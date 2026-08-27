//! Boxcar kernel
use super::traits::Kernel;

/// A top-hat kernel with a flat response inside its support window.
#[derive(Debug, Clone)]
pub struct BoxcarKernel {
    /// Number of taps in the kernel support.
    ntaps: usize,
    /// Half-width used to define the boxcar support.
    a: f64,
}

impl Kernel for BoxcarKernel {
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
        if x.abs() >= self.a { 0.0 } else { 1.0 }
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
    #[inline]
    fn set_ntaps(&mut self, ntaps: usize) {
        self.ntaps = ntaps;
        self.a = (ntaps / 2) as f64;
    }
}
