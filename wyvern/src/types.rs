//! Type-related traits
use num_complex::Complex;
use num_traits::AsPrimitive;

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Numeric scalar type that can be converted to and from an `f64`.
///
/// This is the common abstraction used by the interpolators for values that are
/// either real or complex-floating point types.
pub trait FloatLike:
    private::Sealed + AsPrimitive<f64> + Copy + std::fmt::Debug + Sync + Send + 'static
{
    /// Construct a value of this type from an `f64` representation.
    ///
    /// # Parameters
    /// * `x`: The source value to cast into the target floating-point type.
    ///
    /// # Returns
    /// A scalar value of the implementing type.
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

/// A numeric scalar type that may be a real value or a complex value.
///
/// The interpolation code uses this trait to operate uniformly across both real
/// and complex arrays while tracking the underlying real-valued storage layout.
pub trait MaybeComplex: private::Sealed + Copy + std::fmt::Debug + Sync + Send + 'static {
    /// The underlying real-valued type for this numeric scalar.
    type Real: FloatLike;
    /// Whether the underlying value is complex-valued.
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

/// Reinterpret a real or complex slice as the corresponding real-valued storage.
///
/// # Parameters
/// * `x`: A slice of `T`, where `T` may be a real scalar or a complex scalar.
///
/// # Returns
/// A slice whose memory layout matches the underlying real components used by the
/// interpolators.
pub const fn as_real_slice<T>(x: &[T]) -> &[T::Real]
where
    T: MaybeComplex,
{
    let factor = if T::IS_COMPLEX { 2 } else { 1 };
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast::<T::Real>(), x.len() * factor) }
}

/// Reinterpret a real or complex mutable slice as its underlying real storage.
///
/// # Parameters
/// * `x`: A mutable slice of `T`, where `T` may be a real scalar or a complex scalar.
///
/// # Returns
/// A mutable slice over the underlying real components used by the interpolators.
pub const fn as_real_slice_mut<T>(x: &mut [T]) -> &mut [T::Real]
where
    T: MaybeComplex,
{
    let factor = if T::IS_COMPLEX { 2 } else { 1 };
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<T::Real>(), x.len() * factor) }
}
