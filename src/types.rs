//! Utils for handling complex/float types
use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder};
use numpy::Complex32;

/// zero-copy reinterpret of a Complex32 buffer to two interleaved
/// Float32 buffers.
pub unsafe fn split_complex_view<'a>(
    view: &ArrayView2<'a, Complex32>,
) -> (ArrayView2<'a, f32>, ArrayView2<'a, f32>) {
    let (rows, cols) = view.dim();
    #[allow(clippy::indexing_slicing, reason = "stride for index 0 must exist")]
    let row_stride = (view.strides()[0] * 2).cast_unsigned();
    let ptr = view.as_ptr().cast::<f32>();
    let real = unsafe { ArrayView2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr) };
    let imag =
        unsafe { ArrayView2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr.add(1)) };

    (real, imag)
}

pub unsafe fn split_complex_view_mut<'a>(
    view: &ArrayViewMut2<'a, Complex32>,
) -> (ArrayViewMut2<'a, f32>, ArrayViewMut2<'a, f32>) {
    let (rows, cols) = view.dim();
    #[allow(clippy::indexing_slicing, reason = "stride for index 0 must exist")]
    let row_stride = (view.strides()[0] * 2).cast_unsigned();
    let ptr = view.as_ptr() as *mut f32;
    let real = unsafe { ArrayViewMut2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr) };
    let imag =
        unsafe { ArrayViewMut2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr.add(1)) };

    (real, imag)
}
