//! Internal helper functions

/// Computes the median absolute sample spacing for a sorted coordinate array.
///
/// The result matches `np.median(np.abs(np.diff(x)))` for the input coordinates.
///
/// # Parameters
/// * `x`: Sorted input coordinates from which to compute the sample spacing.
///
/// # Returns
/// The median absolute difference between consecutive values in `x`.
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

/// Invert a value or return zero when the input is zero.
///
/// # Parameters
/// * `x`: The value to invert.
///
/// # Returns
/// `1.0 / x` when `x` is nonzero, otherwise `0.0`.
#[inline]
pub fn invert_no_zero(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { 1.0 / x }
}

/// A branchless variant of `invert_no_zero` used for benchmark comparisons.
#[inline]
#[allow(dead_code, reason = "testing")]
pub fn invert_no_zero_branchless(x: f64) -> f64 {
    let inv = 1.0 / x;
    // bitmask - all zeros if x is zero, all ones otherwise
    let bitmask = u64::from(x != 0.0).wrapping_neg();

    f64::from_bits(inv.to_bits() & bitmask)
}
