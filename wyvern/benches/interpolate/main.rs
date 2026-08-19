//! Performance benchmarks for interpolation methods
mod common;
mod kernel;
mod linear;

use criterion::criterion_main;

criterion_main!(linear::linear_benches, kernel::lanczos_benches);
