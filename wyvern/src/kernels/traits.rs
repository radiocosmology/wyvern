//! Implementation of some interpolation kernels

/// Implements a basic kernel
pub trait Kernel {
    /// Build a kernel with a fixed number of taps
    fn build(ntaps: usize) -> Self;
    /// Evaluate the kernel at a point
    fn evaluate(&self, x: f64) -> f64;
    /// Half-width computed from `ntaps`
    fn half_width(&self) -> f64;
    /// Number of taps
    fn ntaps(&self) -> usize;
    /// Update the half-width.
    fn set_ntaps(&mut self, ntaps: usize);
}
