//! Fundamental implementation of fast interpolators
mod kernel_plan;
mod linear_plan;
mod ops;
mod plan;

// re-export
pub use kernel_plan::{DynamicKernelPlan, KernelPlan};
pub use linear_plan::LinearPlan;
pub use ops::{
    interp_last_ax_complex, interp_last_ax_complex_weighted, interp_last_ax_real,
    interp_last_ax_real_weighted,
};
pub use plan::{InterpolationPlan, Interpolator, IntoInterpolator};
