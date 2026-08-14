//! Implementation of some interpolation kernels

/// Implements a kernel
pub trait Kernel {
    /// Evaluate the kernel at a point
    fn evaluate(&self, x: f64) -> f64;
    /// Expected half-width
    fn half_width(&self) -> f64;
    /// Update the half-width.
    fn update_half_width(&mut self, a: f64);
}

#[derive(Debug)]
pub struct LanczosKernel {
    /// Width parameter
    pub a: f64,
}

impl Kernel for LanczosKernel {
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
    fn update_half_width(&mut self, a: f64) {
        self.a = a;
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
