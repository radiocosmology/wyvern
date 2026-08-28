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

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "exact comparisons are expected in tests")]
mod tests {
    use super::*;

    #[test]
    fn median_abs_sample_spacing_odd_length() {
        let x = [0.0, 1.0, 3.0, 6.0];
        // diffs: 1, 2, 3 -> median is 2
        assert_eq!(median_abs_sample_spacing(&x), 2.0);
    }

    #[test]
    fn median_abs_sample_spacing_even_length() {
        let x = [0.0, 1.0, 2.0, 4.0];
        // diffs: 1, 1, 2 -> odd number of diffs (3), median is 1
        assert_eq!(median_abs_sample_spacing(&x), 1.0);

        let x = [0.0, 1.0, 3.0, 4.0, 8.0];
        // diffs: 1, 2, 1, 4 -> even number (4), median is midpoint of 1 and 2
        assert_eq!(median_abs_sample_spacing(&x), 1.5);
    }

    #[test]
    fn median_abs_sample_spacing_handles_uniform_spacing() {
        let x = [0.0, 2.0, 4.0, 6.0, 8.0];
        assert_eq!(median_abs_sample_spacing(&x), 2.0);
    }

    #[test]
    fn invert_no_zero_returns_zero_for_zero_input() {
        assert_eq!(invert_no_zero(0.0), 0.0);
    }

    #[test]
    fn invert_no_zero_inverts_nonzero_input() {
        assert_eq!(invert_no_zero(2.0), 0.5);
        assert_eq!(invert_no_zero(-4.0), -0.25);
    }

    #[test]
    fn invert_no_zero_branchless_matches_invert_no_zero() {
        for x in [-3.0, 0.0, 0.5, 2.0, 10.0] {
            assert_eq!(invert_no_zero_branchless(x), invert_no_zero(x));
        }
    }
}
