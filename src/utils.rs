//! Basic crate utilities
use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder};
use num_complex::Complex;

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

/// zero-copy reinterpret of a Complex<T> buffer to two interleaved
/// Float<T> buffers.
pub unsafe fn split_complex_view<'a, T: Copy>(
    view: &ArrayView2<'a, Complex<T>>,
) -> (ArrayView2<'a, T>, ArrayView2<'a, T>) {
    let (rows, cols) = view.dim();
    #[allow(clippy::indexing_slicing, reason = "stride for index 0 must exist")]
    let row_stride = (view.strides()[0] * 2).cast_unsigned();
    let ptr = view.as_ptr().cast::<T>();
    let real = unsafe { ArrayView2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr) };
    let imag =
        unsafe { ArrayView2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr.add(1)) };

    (real, imag)
}

pub unsafe fn split_complex_view_mut<'a, T: Copy>(
    view: &ArrayViewMut2<'a, Complex<T>>,
) -> (ArrayViewMut2<'a, T>, ArrayViewMut2<'a, T>) {
    let (rows, cols) = view.dim();
    #[allow(clippy::indexing_slicing, reason = "stride for index 0 must exist")]
    let row_stride = (view.strides()[0] * 2).cast_unsigned();
    let ptr = view.as_ptr() as *mut T;
    let real = unsafe { ArrayViewMut2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr) };
    let imag =
        unsafe { ArrayViewMut2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr.add(1)) };

    (real, imag)
}
