//! Type-related traits
use num_complex::Complex;
use num_traits::AsPrimitive;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// f64/f32 type which can be cast back and forth using ``as_()``
pub trait FloatLike:
    private::Sealed + AsPrimitive<f64> + Copy + std::fmt::Debug + Sync + Send + 'static
{
    fn from_f64(x: f64) -> Self;
}

impl FloatLike for f32 {
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "downcast truncation is the desired behaviour"
    )]
    fn from_f64(x: f64) -> Self {
        x as Self
    }
}
impl FloatLike for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

impl<T: FloatLike> private::Sealed for Complex<T> {}

/// `Complex` or real value field
pub trait MaybeComplex: private::Sealed + Copy + std::fmt::Debug + Sync + Send + 'static {
    type Real: FloatLike;
    const IS_COMPLEX: bool;
}

impl<T: FloatLike> MaybeComplex for T {
    type Real = T;
    const IS_COMPLEX: bool = false;
}

impl<T: FloatLike> MaybeComplex for Complex<T> {
    type Real = T;
    const IS_COMPLEX: bool = true;
}

/// Reinterpret a possibly-complex slice into a real slice
pub const fn as_real_slice<T>(x: &[T]) -> &[T::Real]
where
    T: MaybeComplex,
{
    let factor = if T::IS_COMPLEX { 2 } else { 1 };
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<T::Real>(), x.len() * factor) }
}

/// Reinterpret a possibly-complex slice into a real mutable slice
pub const fn as_real_slice_mut<T>(x: &mut [T]) -> &mut [T::Real]
where
    T: MaybeComplex,
{
    let factor = if T::IS_COMPLEX { 2 } else { 1 };
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<T::Real>(), x.len() * factor) }
}
