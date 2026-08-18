//! Performance benchmarks for linear interpolator
use criterion::{Criterion, criterion_group};
use ndarray::{ArrayView1, ArrayViewMut1};
use std::hint::black_box;
use wyvern::interpolate::{self, Interpolator, IntoInterpolator};

use crate::common;

/// Create a linear interpolator with given input/output samples.
#[allow(
    clippy::cast_precision_loss,
    reason = "values too small for precision loss"
)]
fn make_linear_interpolator(
    n_in: usize,
    n_out: usize,
) -> eyre::Result<interpolate::LinearInterpolator> {
    // create the indices
    let x_in: Vec<f64> = (0..n_in).map(|i| i as f64 / n_in as f64).collect();
    let x_out: Vec<f64> = (0..n_out).map(|i| i as f64 / n_in as f64).collect();

    interpolate::LinearInterpolator::build(&x_in, &x_out)
}

fn bench_base(
    interpolator: &interpolate::LinearInterpolator,
    y_in: &ArrayView1<f64>,
    y_out: ArrayViewMut1<f64>,
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    interpolator.interp_row(y_in, y_out);
}

fn bench_masked(
    interpolator: &interpolate::LinearInterpolator,
    y_in: &ArrayView1<f64>,
    mask_in: &mut [f64],
    y_out: ArrayViewMut1<f64>,
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    interpolator.interp_row_masked(y_in, mask_in, y_out);
}

fn bench_with_variance(
    interpolator: &interpolate::LinearInterpolator,
    y_in: &ArrayView1<f64>,
    weight_in: &ArrayView1<f64>,
    var_scratch: &mut [f64],
    mask_scratch: &mut [f64],
    y_out: ArrayViewMut1<f64>,
    weight_out: ArrayViewMut1<f64>,
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

fn run_benchmarks(c: &mut Criterion) {
    let n_in: usize = 9200;
    let n_out: usize = 16382;

    #[allow(clippy::unwrap_used, reason = "will succeed as defined for test")]
    let interpolator = make_linear_interpolator(n_in, n_out).unwrap();
    let (y_in, mut y_out) = common::make_data(n_in, n_out);
    let (weight_in, mut weight_out) = common::make_weights(n_in, n_out);
    let mut mask = common::make_mask(n_in);
    let mut var = common::make_mask(n_in); // scratch buffer, contents don't matter

    let mut group = c.benchmark_group("linear");

    group.bench_function("interpolate_base", |b| {
        b.iter(|| {
            bench_base(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(y_out.view_mut()),
            );
        });
    });
    group.bench_function("interpolate_masked", |b| {
        b.iter(|| {
            bench_masked(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(mask.as_mut_slice()),
                black_box(y_out.view_mut()),
            );
        });
    });
    group.bench_function("interpolate_with_variance", |b| {
        b.iter(|| {
            bench_with_variance(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(&weight_in.view()),
                black_box(var.as_mut_slice()),
                black_box(mask.as_mut_slice()),
                black_box(y_out.view_mut()),
                black_box(weight_out.view_mut()),
            );
        });
    });
    group.finish();
}

criterion_group!(linear_benches, run_benchmarks);
