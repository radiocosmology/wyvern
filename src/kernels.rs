//! Implementation of some interpolation kernels

/// Lanczos kernel
#[inline]
pub fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x == 0.0 || x.abs() >= a {
        0.0
    } else {
        let px = std::f64::consts::PI * x;
        a * px.sin() * (px / a).sin() / (px * px)
    }
}
