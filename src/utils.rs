//! Basic crate utilities

/// Computes the median of |diff(x)| -- matches np.median(np.abs(np.diff(lsd))).
/// Requires a mutable scratch Vec to avoid an extra allocation if you
/// call this repeatedly; sorts in place.
pub fn median_abs_diff(x: &[f32]) -> f32 {
    debug_assert!(
        x.len() > 1,
        "need at least 2 input samples to compute a diff"
    );
    #[allow(
        clippy::indexing_slicing,
        reason = "windows(2) ensures that indexing won't panic"
    )]
    let mut diffs: Vec<f32> = x.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    let n = diffs.len();
    let mid = n >> 1;

    diffs.select_nth_unstable_by(mid, f32::total_cmp);
    let upper = *diffs.get(mid).unwrap_or(&0.0_f32);

    if n.is_multiple_of(2) {
        diffs.select_nth_unstable_by(mid - 1, f32::total_cmp);
        f32::midpoint(*diffs.get(mid - 1).unwrap_or(&0.0_f32), upper)
    } else {
        upper
    }
}
