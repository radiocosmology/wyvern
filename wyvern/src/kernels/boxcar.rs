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

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "exact comparisons are expected in tests")]
mod tests {
    use super::*;

    #[test]
    fn build_sets_ntaps_and_half_width() {
        let k = BoxcarKernel::build(8);
        assert_eq!(k.ntaps(), 8);
        assert_eq!(k.half_width(), 4.0);
    }

    #[test]
    fn evaluate_is_flat_inside_support_and_zero_outside() {
        let k = BoxcarKernel::build(8);
        assert_eq!(k.evaluate(0.0), 1.0);
        assert_eq!(k.evaluate(3.9), 1.0);
        assert_eq!(k.evaluate(4.0), 0.0);
        assert_eq!(k.evaluate(-4.0), 0.0);
        assert_eq!(k.evaluate(10.0), 0.0);
    }

    #[test]
    fn set_ntaps_rescales_half_width() {
        let mut k = BoxcarKernel::build(8);
        k.set_ntaps(20);
        assert_eq!(k.ntaps(), 20);
        assert_eq!(k.half_width(), 10.0);
    }
}
