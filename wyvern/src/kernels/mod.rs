//! Kernel definitions used by the interpolation algorithms.
//!
//! These kernels provide compact support windows for resampling and interpolation.
mod boxcar;
mod kaiserbessel;
mod lanczos;
/// Shared kernel trait definitions and extension points.
pub mod traits;

/// Boxcar window kernel used for top-hat style weighting.
pub use boxcar::BoxcarKernel;
/// Kaiser-Bessel window kernel with an adjustable beta parameter.
pub use kaiserbessel::KaiserBesselKernel;
/// Lanczos sinc-based window kernel.
pub use lanczos::LanczosKernel;
