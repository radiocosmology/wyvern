//! Performance benchmarks for lanczos interpolator
#![allow(
    clippy::unwrap_used,
    reason = "unwrap allowed by benchmark construction"
)]

use criterion::{Criterion, criterion_group};
use num_complex::Complex;
use std::hint::black_box;
use wyvern::interpolate::{self, Interpolator, IntoInterpolator};
use wyvern::kernels::{BoxKernel, traits::Kernel};

use crate::common;

/// Create a lanczos interpolator with given input/output samples.
#[allow(
    clippy::cast_precision_loss,
    reason = "values too small for precision loss"
)]
fn make_lanczos_interpolator(
    n_in: usize,
    n_out: usize,
    n_taps: usize,
) -> eyre::Result<interpolate::DynamicKernelInterpolator> {
    // create the indices
    let x_in: Vec<f64> = (0..n_in).map(|i| i as f64 / n_in as f64).collect();
    let x_out: Vec<f64> = (0..n_out).map(|i| i as f64 / n_in as f64).collect();

    let mut kernel = BoxKernel::build(n_taps);
    interpolate::DynamicKernelInterpolator::build(&x_in, &x_out, &mut kernel)
}

fn bench_base_real(
    interpolator: &interpolate::DynamicKernelInterpolator,
    y_in: &[f64],
    y_out: &mut [f64],
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    interpolator.interp_row(y_in, y_out);
}

fn bench_base_complex(
    interpolator: &interpolate::DynamicKernelInterpolator,
    y_in: &[Complex<f64>],
    y_out: &mut [Complex<f64>],
) {
    let interpolator: &dyn Interpolator<Complex<f64>> = interpolator.as_interpolator();
    interpolator.interp_row(y_in, y_out);
}

fn bench_masked_real(
    interpolator: &interpolate::DynamicKernelInterpolator,
    y_in: &[f64],
    mask_in: &mut [f64],
    y_out: &mut [f64],
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    interpolator.interp_row_masked(y_in, mask_in, y_out);
}

fn bench_masked_complex(
    interpolator: &interpolate::DynamicKernelInterpolator,
    y_in: &[Complex<f64>],
    mask_in: &mut [f64],
    y_out: &mut [Complex<f64>],
) {
    let interpolator: &dyn Interpolator<Complex<f64>> = interpolator.as_interpolator();
    interpolator.interp_row_masked(y_in, mask_in, y_out);
}

fn bench_with_variance_real(
    interpolator: &interpolate::DynamicKernelInterpolator,
    y_in: &[f64],
    weight_in: &[f64],
    var_scratch: &mut [f64],
    mask_scratch: &mut [f64],
    y_out: &mut [f64],
    weight_out: &mut [f64],
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    interpolator.interp_row_with_variance(
        y_in,
        weight_in,
        var_scratch,
        mask_scratch,
        y_out,
        weight_out,
    );
}

fn bench_with_variance_complex(
    interpolator: &interpolate::DynamicKernelInterpolator,
    y_in: &[Complex<f64>],
    weight_in: &[f64],
    var_scratch: &mut [f64],
    mask_scratch: &mut [f64],
    y_out: &mut [Complex<f64>],
    weight_out: &mut [f64],
) {
    let interpolator: &dyn Interpolator<Complex<f64>> = interpolator.as_interpolator();
    interpolator.interp_row_with_variance(
        y_in,
        weight_in,
        var_scratch,
        mask_scratch,
        y_out,
        weight_out,
    );
}

fn run_benchmarks(c: &mut Criterion) {
    let n_in: usize = 9200;
    let n_out: usize = 16382;
    let n_taps: usize = 4;

    #[allow(clippy::unwrap_used, reason = "will succeed as defined for test")]
    let interpolator = make_lanczos_interpolator(n_in, n_out, n_taps).unwrap();

    let (y_in, mut y_out) = common::make_data(n_in, n_out);
    let (y_in_c, mut y_out_c) = common::make_data_complex(n_in, n_out);
    let (weight_in, mut weight_out) = common::make_weights(n_in, n_out);

    let mut mask = common::make_mask(n_in);
    let mut var = common::make_mask(n_in); // scratch buffer, contents don't matter

    let mut group = c.benchmark_group("kernel");

    group.bench_function("interpolate_base_real", |b| {
        b.iter(|| {
            bench_base_real(
                black_box(&interpolator),
                black_box(y_in.as_slice().unwrap()),
                black_box(y_out.as_slice_mut().unwrap()),
            );
        });
    });
    group.bench_function("interpolate_base_complex", |b| {
        b.iter(|| {
            bench_base_complex(
                black_box(&interpolator),
                black_box(y_in_c.as_slice().unwrap()),
                black_box(y_out_c.as_slice_mut().unwrap()),
            );
        });
    });
    group.bench_function("interpolate_masked_real", |b| {
        b.iter(|| {
            bench_masked_real(
                black_box(&interpolator),
                black_box(y_in.as_slice().unwrap()),
                black_box(mask.as_mut_slice()),
                black_box(y_out.as_slice_mut().unwrap()),
            );
        });
    });
    group.bench_function("interpolate_masked_complex", |b| {
        b.iter(|| {
            bench_masked_complex(
                black_box(&interpolator),
                black_box(y_in_c.as_slice().unwrap()),
                black_box(mask.as_mut_slice()),
                black_box(y_out_c.as_slice_mut().unwrap()),
            );
        });
    });
    group.bench_function("interpolate_with_variance_real", |b| {
        b.iter(|| {
            bench_with_variance_real(
                black_box(&interpolator),
                black_box(y_in.as_slice().unwrap()),
                black_box(weight_in.as_slice().unwrap()),
                black_box(var.as_mut_slice()),
                black_box(mask.as_mut_slice()),
                black_box(y_out.as_slice_mut().unwrap()),
                black_box(weight_out.as_slice_mut().unwrap()),
            );
        });
    });
    group.bench_function("interpolate_with_variance_complex", |b| {
        b.iter(|| {
            bench_with_variance_complex(
                black_box(&interpolator),
                black_box(y_in_c.as_slice().unwrap()),
                black_box(weight_in.as_slice().unwrap()),
                black_box(var.as_mut_slice()),
                black_box(mask.as_mut_slice()),
                black_box(y_out_c.as_slice_mut().unwrap()),
                black_box(weight_out.as_slice_mut().unwrap()),
            );
        });
    });
}

criterion_group!(lanczos_benches, run_benchmarks);
