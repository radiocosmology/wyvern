//! Linear interpolation across the last axis.
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use rayon::prelude::*;

/// Precomputed interpolation plan for mapping input
/// and output samples.
struct InterpolationPlanLinear {
    // lower bracket index for input
    i0: Vec<usize>,
    // upper brackeet index for input
    i1: Vec<usize>,
    // weight for i1 sample (w0 = 1 - w1)
    w1: Vec<f64>,
    // mask for invalid samples
    invalid: Vec<u8>,
}

impl InterpolationPlanLinear {
    /// ``x_in``: sorted, arbitrary spacing, len >= 2
    /// ``x_out``: sorted, uniform spacing, len >= 1
    pub fn build(x_in: &[f64], x_out: &[f64]) -> eyre::Result<Self> {
        let n_out = x_out.len();
        let n_in = x_in.len();

        assert!(n_in >= 2, "minimum 2 input samples are required!");
        assert!(n_out >= 1, "minimum 1 output sample is required!");

        let mut i0 = Vec::<usize>::with_capacity(n_out);
        let mut i1 = Vec::<usize>::with_capacity(n_out);
        let mut w1 = Vec::<f64>::with_capacity(n_out);
        let mut invalid = Vec::<u8>::with_capacity(n_out);

        // both inputs are sorted, so step only advances forward. error
        // is eventually returned if this assumption fails
        let mut lo: usize = 0;
        let lo_max: usize = n_in - 2;

        for &xo in x_out {
            // advance the pointer while the next pair still brackets xo,
            // or we're at the last valid pair
            #[allow(
                clippy::indexing_slicing,
                reason = "max index is 2 less than `x_in.len()`"
            )]
            let (a, b) = {
                // move forward to the next target sample
                while lo < lo_max && x_in[lo + 1] <= xo {
                    lo += 1;
                }
                // fetch the input sample values
                (x_in[lo], x_in[lo + 1])
            };

            let delta = b - a;
            let wb = if delta >= 0.0 {
                (xo - a) / delta
            } else {
                eyre::bail!("inputs are unsorted or repeated!");
            };

            // check if this is more than one input spacing from
            // either input sample
            let invld = b - xo > delta || xo - a > delta;

            i0.push(lo);
            i1.push(lo + 1);
            w1.push(wb);
            invalid.push(u8::from(invld));
        }

        Ok(Self {
            i0,
            i1,
            w1,
            invalid,
        })
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.i0.len()
    }
}

/// Apply a pre-computed plan to a single array row.
#[inline]
fn interp_row(
    plan: &InterpolationPlanLinear,
    y_in: &ArrayView1<f64>,
    mask_in: &ArrayView1<u8>,
    mut y_out: ArrayViewMut1<f64>,
    mut mask_out: ArrayViewMut1<u8>,
) {
    let n = plan.len();

    debug_assert_eq!(y_out.len(), n);

    unsafe {
        for j in 0..n {
            // extract the interpolation indices and weight
            let i0 = *plan.i0.get_unchecked(j);
            let i1 = *plan.i1.get_unchecked(j);
            let w1 = *plan.w1.get_unchecked(j);
            let invld = *plan.invalid.get_unchecked(j);

            // interpolate onto the target sample
            let a = *y_in.uget(i0);
            let b = *y_in.uget(i1);
            *y_out.uget_mut(j) = (b - a).mul_add(w1, a);

            // propagate masking
            let m0 = *mask_in.uget(i0);
            let m1 = *mask_in.uget(i1);
            *mask_out.uget_mut(j) = m0 | m1 | invld;
        }
    }
}

/// Interpolate over the last axis of an array.
#[inline]
pub fn interp_last_ax_lin(
    x_in: &[f64],
    x_out: &[f64],
    y_in: &ArrayView2<f64>,
    mask_in: &ArrayView2<u8>,
    mut y_out: ArrayViewMut2<f64>,
    mut mask_out: ArrayViewMut2<u8>,
) -> eyre::Result<()> {
    let plan = InterpolationPlanLinear::build(x_in, x_out)?;

    // iterate over the 0th axis and interpolate the 1st
    // (contiguous) axis
    Zip::from(y_in.rows())
        .and(mask_in.rows())
        .and(y_out.rows_mut())
        .and(mask_out.rows_mut())
        .into_par_iter()
        .for_each(|(yi, mi, yo, mo)| {
            interp_row(&plan, &yi, &mi, yo, mo);
        });

    Ok(())
}
