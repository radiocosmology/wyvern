//! Type-related traits
use num_traits::AsPrimitive;

/// f64/f32 type which can be cast back and forth using ``as_()``
pub trait FloatLike: AsPrimitive<f64> + Copy + 'static {
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

/// `[FloatLike]` type which can also be shared across threads
pub trait ParFloatLike: FloatLike + Send + Sync {}
impl<T> ParFloatLike for T where T: FloatLike + Send + Sync {}
