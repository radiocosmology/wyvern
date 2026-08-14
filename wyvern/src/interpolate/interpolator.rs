//! Implementation of a linear interpolator
use crate::types::FloatLike;
use ndarray::{ArrayView1, ArrayViewMut1};

// ------ Traits ------

/// Implements methods required to construct an interpolation plan
pub trait InterpolationPlan {
    /// Number of output samples
    fn len(&self) -> usize;
    /// `true` if `len` is zero
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implements interpolation methods for float-like values
pub trait Interpolator<T: FloatLike>: Sync {
    /// Interpolate a single row's data onto output points
    fn interp_row(&self, y_in: &ArrayView1<T>, y_out: ArrayViewMut1<T>);

    /// Interpolate a single row's data onto output poi ts,
    /// accounting for an input mask
    fn interp_row_masked(&self, y_in: &ArrayView1<T>, mask_in: &mut [f64], y_out: ArrayViewMut1<T>);

    /// Interpolate a single row's data onto output points,
    /// and propagate corresponding inverse-variance weights
    fn interp_row_with_variance(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        mask_scratch: &mut [f64],
        y_out: ArrayViewMut1<T>,
        weight_out: ArrayViewMut1<T>,
    );
}

/// Implements methods to convert this to a typed [`Interpolator`]
pub trait IntoInterpolator {
    /// extract the interpolator
    fn as_interpolator<T: FloatLike>(&self) -> &dyn Interpolator<T>;
}
