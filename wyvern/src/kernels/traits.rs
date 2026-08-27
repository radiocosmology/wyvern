//! Implementation of some interpolation kernels

/// A compact support kernel used to weight local samples during interpolation.
pub trait Kernel {
    /// Construct a kernel with a fixed number of taps.
    ///
    /// # Parameters
    /// * `ntaps`: The number of support taps used by the kernel.
    ///
    /// # Returns
    /// A kernel instance configured for the requested support width.
    fn build(ntaps: usize) -> Self
    where
        Self: Sized;
    /// Evaluate the kernel at a normalized coordinate.
    ///
    /// # Parameters
    /// * `x`: The coordinate at which the kernel should be evaluated.
    ///
    /// # Returns
    /// The kernel weight at `x`.
    fn evaluate(&self, x: f64) -> f64;
    /// Returns the kernel half-width implied by the current tap count.
    fn half_width(&self) -> f64;
    /// Returns the current number of support taps in the kernel.
    fn ntaps(&self) -> usize;
    /// Update the kernel to use a new support width.
    ///
    /// # Parameters
    /// * `ntaps`: The new number of taps for the kernel.
    fn set_ntaps(&mut self, ntaps: usize);
}
