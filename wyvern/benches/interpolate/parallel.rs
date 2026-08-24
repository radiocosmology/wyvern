//! Performance benchmarks for parallel interpolators
#![allow(
    clippy::unwrap_used,
    reason = "unwrap allowed by benchmark construction"
)]

use criterion::{Criterion, criterion_group};
use ndarray::{ArrayView2, ArrayViewMut2};
use std::hint::black_box;
use wyvern::interpolate::{self, Interpolator, IntoInterpolator, ParallelInterpolator};

use crate::common;
use crate::linear::make_linear_interpolator;

fn bench_base_real(
    interpolator: &interpolate::LinearInterpolator,
    y_in: &ArrayView2<f64>,
    y_out: ArrayViewMut2<f64>,
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    let par_interpolator = ParallelInterpolator::<f64>::with_interpolator(interpolator);
    par_interpolator.interpolate_real(y_in, y_out);
}

fn bench_base_complex(
    interpolator: &interpolate::LinearInterpolator,
    y_re_in: &ArrayView2<f64>,
    y_im_in: &ArrayView2<f64>,
    y_re_out: ArrayViewMut2<f64>,
    y_im_out: ArrayViewMut2<f64>,
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    let par_interpolator = ParallelInterpolator::<f64>::with_interpolator(interpolator);
    par_interpolator.interpolate_complex(y_re_in, y_im_in, y_re_out, y_im_out);
}

fn bench_weighted_real(
    interpolator: &interpolate::LinearInterpolator,
    y_in: &ArrayView2<f64>,
    weight_in: &ArrayView2<f64>,
    y_out: ArrayViewMut2<f64>,
    weight_out: ArrayViewMut2<f64>,
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    let par_interpolator = ParallelInterpolator::<f64>::with_interpolator(interpolator);
    par_interpolator.interpolate_real_weighted(y_in, weight_in, y_out, weight_out);
}

fn bench_weighted_complex(
    interpolator: &interpolate::LinearInterpolator,
    y_re_in: &ArrayView2<f64>,
    y_im_in: &ArrayView2<f64>,
    weight_in: &ArrayView2<f64>,
    y_re_out: ArrayViewMut2<f64>,
    y_im_out: ArrayViewMut2<f64>,
    weight_out: ArrayViewMut2<f64>,
) {
    let interpolator: &dyn Interpolator<f64> = interpolator.as_interpolator();
    let par_interpolator = ParallelInterpolator::<f64>::with_interpolator(interpolator);
    par_interpolator
        .interpolate_complex_weighted(y_re_in, y_im_in, weight_in, y_re_out, y_im_out, weight_out);
}

fn run_benchmarks(c: &mut Criterion) {
    let n_in: usize = 9200;
    let n_out: usize = 16382;
    let n_rows = 2000;

    #[allow(clippy::unwrap_used, reason = "will succeed as defined for test")]
    let interpolator = make_linear_interpolator(n_in, n_out).unwrap();

    let (y_in, mut y_out) = common::make_2d_data(n_in, n_out, n_rows);
    let (y_im_in, mut y_im_out) = common::make_2d_data(n_in, n_out, n_rows);
    let (weight_in, mut weight_out) = common::make_2d_weights(n_in, n_out, n_rows);

    let mut group = c.benchmark_group("parallel");

    group.bench_function("interpolate_real", |b| {
        b.iter(|| {
            bench_base_real(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(y_out.view_mut()),
            );
        });
    });
    group.bench_function("interpolate_complex", |b| {
        b.iter(|| {
            bench_base_complex(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(&y_im_in.view()),
                black_box(y_out.view_mut()),
                black_box(y_im_out.view_mut()),
            );
        });
    });
    group.bench_function("interpolate_real_weighted", |b| {
        b.iter(|| {
            bench_weighted_real(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(&weight_in.view()),
                black_box(y_out.view_mut()),
                black_box(weight_out.view_mut()),
            );
        });
    });
    group.bench_function("interpolate_complex_weighted", |b| {
        b.iter(|| {
            bench_weighted_complex(
                black_box(&interpolator),
                black_box(&y_in.view()),
                black_box(&y_im_in.view()),
                black_box(&weight_in.view()),
                black_box(y_out.view_mut()),
                black_box(y_im_out.view_mut()),
                black_box(weight_out.view_mut()),
            );
        });
    });
    group.finish();
}
criterion_group!(parallel_benches, run_benchmarks);
