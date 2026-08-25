//! Common data generation for interpolation benchmarks
#![allow(
    clippy::unwrap_used,
    reason = "unwrap allowed by benchmark construction"
)]
use ndarray::{Array1, Array2, Zip};
use num_complex::Complex;

/// Make input and output arrays for a given number of samples.
pub fn make_data(n_in: usize, n_out: usize) -> (Array1<f64>, Array1<f64>) {
    let y_in = Array1::linspace(0., 1., n_in);
    let y_out = Array1::zeros(n_out);

    (y_in, y_out)
}

/// Make input and output arrays for a given number of samples.
pub fn make_data_complex(
    n_in: usize,
    n_out: usize,
) -> (Array1<Complex<f64>>, Array1<Complex<f64>>) {
    let y_in = Array1::linspace(0., 1., n_in);
    let y_in_im = y_in.clone();

    let y_in = Zip::from(&y_in)
        .and(&y_in_im)
        .map_collect(|&r, &i| Complex::new(r, i));

    let y_out: Array1<Complex<f64>> = Array1::zeros(n_out);

    (y_in, y_out)
}

/// Make 2D input and output arrays
pub fn make_2d_data(n_in: usize, n_out: usize, n_rows: usize) -> (Array2<f64>, Array2<f64>) {
    let y_in = Array1::linspace(0., 1., n_in * n_rows)
        .into_shape_with_order((n_rows, n_in))
        .unwrap();

    let y_out = Array2::zeros((n_rows, n_out));

    (y_in, y_out)
}

/// Make 2D input and output arrays
pub fn make_2d_data_complex(
    n_in: usize,
    n_out: usize,
    n_rows: usize,
) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {
    let y_in = Array1::linspace(0., 1., n_in * n_rows)
        .into_shape_with_order((n_rows, n_in))
        .unwrap();
    let y_in_im = y_in.clone();

    let y_in = Zip::from(&y_in)
        .and(&y_in_im)
        .map_collect(|&r, &i| Complex::new(r, i));

    let y_out: Array2<Complex<f64>> = Array2::zeros((n_rows, n_out));

    (y_in, y_out)
}

/// Generate a 1D `Vec<f64>` mask.
pub fn make_mask(n_in: usize) -> Vec<f64> {
    let mut mask: Vec<f64> = vec![1.0; n_in];

    mask.iter_mut().enumerate().for_each(|(i, m)| {
        if i.is_multiple_of(7) || i.is_multiple_of(9) {
            *m = 0.0;
        }
    });

    mask
}

/// Generate a 1D `Vec<f64>` mask.
pub fn make_2d_mask(n_in: usize, n_rows: usize) -> Array2<f64> {
    let mut mask = Array2::ones((n_rows, n_in));

    mask.iter_mut().enumerate().for_each(|(i, m)| {
        if i.is_multiple_of(7) || i.is_multiple_of(9) {
            *m = 0.0;
        }
    });

    mask
}

/// Make input and output weight arrays for a given number of samples.
pub fn make_weights(n_in: usize, n_out: usize) -> (Array1<f64>, Array1<f64>) {
    let y_in = make_mask(n_in);
    let y_in = Array1::from_vec(y_in);
    let y_out = Array1::zeros(n_out);

    (y_in, y_out)
}

/// Make input and output weight arrays for a given number of samples.
pub fn make_2d_weights(n_in: usize, n_out: usize, n_rows: usize) -> (Array2<f64>, Array2<f64>) {
    let y_in = make_2d_mask(n_in, n_rows);
    let y_out = Array2::zeros((n_rows, n_out));

    (y_in, y_out)
}
