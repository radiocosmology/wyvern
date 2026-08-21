//! Fast linear interpolation for python.

pub mod interpolate;
pub mod kernels;
pub mod types;
pub(crate) mod util;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
