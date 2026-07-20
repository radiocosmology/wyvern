//! Utils for handling complex/float types
use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder};
use num_complex::Complex;

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
