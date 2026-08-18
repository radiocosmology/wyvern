//! Performance benchmarks for interpolation methods
mod common;
mod lanczos;
mod linear;

use criterion::criterion_main;

criterion_main!(linear::linear_benches, lanczos::lanczos_benches);
