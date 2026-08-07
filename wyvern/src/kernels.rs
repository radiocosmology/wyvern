//! Implementation of some interpolation kernels

/// Lanczos kernel
#[inline]
#[must_use]
pub fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() >= a {
        0.0
    } else {
        sinc(x) * sinc(x / a)
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
