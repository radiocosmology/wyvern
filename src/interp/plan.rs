//! Implementation of a linear interpolator
use crate::types::FloatLike;
use ndarray::{ArrayView1, ArrayViewMut1};

pub trait InterpolationPlan {
    // Number of output samples produced by this plan
    fn len(&self) -> usize;

    // Interpolate a single row's data onto output points
    fn interp_row<T: FloatLike>(&self, y_in: &ArrayView1<T>, y_out: ArrayViewMut1<T>);

    // Interpolate a single row's data onto output points,
    // and propagate corresponding inverse-variance weights
    fn interp_row_with_variance<T: FloatLike>(
        &self,
        y_in: &ArrayView1<T>,
        weight_in: &ArrayView1<T>,
        var_scratch: &mut [f64],
        y_out: ArrayViewMut1<T>,
        weight_out: ArrayViewMut1<T>,
    );
}

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
