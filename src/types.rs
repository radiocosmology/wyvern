//! Utils for handling complex/float types
use ndarray::{ArrayView2, ArrayViewMut2, ShapeBuilder};
use numpy::Complex32;

/// zero-copy reinterpret of a Complex32 buffer to two interleaved
/// Float32 buffers.
unsafe fn split_complex_view(view: ArrayView2<Complex32>) -> (ArrayView2<f32>, ArrayView2<f32>) {
    let (rows, cols) = view.dim();
    let row_stride = (view.strides()[0] * 2) as usize;
    let ptr = view.as_ptr() as *const f32;
    let real = unsafe { ArrayView2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr) };
    let imag =
        unsafe { ArrayView2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr.add(1)) };

    (real, imag)
}

unsafe fn split_complex_view_mut(
    view: ArrayViewMut2<Complex32>,
) -> (ArrayViewMut2<f32>, ArrayViewMut2<f32>) {
    let (rows, cols) = view.dim();
    let row_stride = (view.strides()[0] * 2) as usize;
    let ptr = view.as_ptr() as *mut f32;
    let real = unsafe { ArrayViewMut2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr) };
    let imag =
        unsafe { ArrayViewMut2::from_shape_ptr((rows, cols).strides((row_stride, 2)), ptr.add(1)) };

    (real, imag)
}
