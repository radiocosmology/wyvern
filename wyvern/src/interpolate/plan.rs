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
    /// Whether this interpolator requires a reusable scratch
    /// mask buffer
    fn needs_mask_scratch(&self) -> bool;

    /// Interpolate a single row's data onto output points
    fn interp_row(&self, y_in: &ArrayView1<T>, y_out: ArrayViewMut1<T>);

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

// ------ Utility functions ------

/// Computes the median of |diff(x)| -- matches np.median(np.abs(np.diff(lsd))).
/// Requires a mutable scratch Vec to avoid an extra allocation if you
/// call this repeatedly; sorts in place.
pub fn median_abs_sample_spacing(x: &[f64]) -> f64 {
    debug_assert!(
        x.len() > 1,
        "need at least 2 input samples to compute a diff"
    );
    #[allow(
        clippy::indexing_slicing,
        reason = "windows(2) ensures that indexing won't panic"
    )]
    let mut diffs: Vec<f64> = x.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    let n = diffs.len();
    let mid = n >> 1;

    diffs.select_nth_unstable_by(mid, f64::total_cmp);
    let upper = *diffs.get(mid).unwrap_or(&0.0_f64);

    if n.is_multiple_of(2) {
        diffs.select_nth_unstable_by(mid - 1, f64::total_cmp);
        f64::midpoint(*diffs.get(mid - 1).unwrap_or(&0.0_f64), upper)
    } else {
        upper
    }
}
