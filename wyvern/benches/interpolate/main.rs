//! Performance benchmarks for interpolation methods
mod common;
mod kernel;
mod linear;
mod parallel;

use criterion::criterion_main;

criterion_main!(
    linear::linear_benches,
    kernel::lanczos_benches,
    parallel::parallel_benches
);
