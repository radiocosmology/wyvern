//! Fundamental interpolation primitives and public re-exports.
//!
//! This module contains the core interpolation planning APIs, including linear and
//! kernel-based interpolators used throughout the crate.
mod helpers;
mod interpolator;
mod kernel_interpolator;
mod linear_interpolator;

/// Re-exported interpolation traits and runtime plan types.
pub use interpolator::{InterpolationPlan, Interpolator, IntoInterpolator, ParallelInterpolator};
/// Re-exported fixed-width and dynamic kernel interpolation implementations.
pub use kernel_interpolator::{FixedWidthKernelInterpolator, KernelInterpolator};
/// Linear interpolation implementation used for simple value remapping.
pub use linear_interpolator::LinearInterpolator;
