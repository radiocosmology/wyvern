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

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "exact comparisons are expected in tests")]
mod tests {
    use super::*;

    #[test]
    fn sinc_at_zero_is_one() {
        assert_eq!(sinc(0.0), 1.0);
    }

    #[test]
    fn sinc_at_integer_is_zero() {
        assert!(sinc(1.0).abs() < 1e-12);
        assert!(sinc(2.0).abs() < 1e-12);
    }

    #[test]
    fn build_sets_ntaps_and_half_width() {
        let k = LanczosKernel::build(6);
        assert_eq!(k.ntaps(), 6);
        assert_eq!(k.half_width(), 3.0);
    }

    #[test]
    fn evaluate_is_zero_at_and_beyond_half_width() {
        let k = LanczosKernel::build(6);
        assert_eq!(k.evaluate(3.0), 0.0);
        assert_eq!(k.evaluate(-3.0), 0.0);
        assert_eq!(k.evaluate(5.0), 0.0);
    }

    #[test]
    fn evaluate_is_one_at_center() {
        let k = LanczosKernel::build(6);
        assert_eq!(k.evaluate(0.0), 1.0);
    }

    #[test]
    fn set_ntaps_rescales_half_width() {
        let mut k = LanczosKernel::build(6);
        k.set_ntaps(10);
        assert_eq!(k.ntaps(), 10);
        assert_eq!(k.half_width(), 5.0);
    }
}
