"""Tests for `wyvern.interpolate.interpolate_kernel[_weighted]`."""

import numpy as np
import pytest

from wyvern import interpolate, kernels


@pytest.mark.parametrize(
    "kernel_cls", [kernels.BoxcarKernel, kernels.LanczosKernel, kernels.KaiserBesselKernel]
)
def test_interpolate_kernel_reproduces_constant_signal(kernel_cls):
    # a constant input signal should interpolate to (approximately) the same
    # constant value regardless of which kernel is used, since kernel
    # coefficients are renormalized to sum to one
    x_in = np.arange(16, dtype=np.float64)
    x_out = np.array([6.0, 7.5, 9.0])
    y_in = np.full((1, 16), 3.0)

    kernel = kernel_cls(4)
    y_out = interpolate.interpolate_kernel(x_in, x_out, kernel, y_in)

    np.testing.assert_allclose(y_out, np.full((1, 3), 3.0), atol=1e-6)


def test_interpolate_kernel_weighted_propagates_positive_weights():
    x_in = np.arange(16, dtype=np.float64)
    x_out = np.array([6.0, 7.5, 9.0])
    y_in = np.full((1, 16), 2.0)
    w_in = np.ones((1, 16))

    kernel = kernels.BoxcarKernel(4)
    y_out, w_out = interpolate.interpolate_kernel_weighted(x_in, x_out, kernel, y_in, w_in)

    np.testing.assert_allclose(y_out, np.full((1, 3), 2.0), atol=1e-6)
    assert np.all(w_out > 0.0)


def test_interpolate_kernel_scale_parameter_widens_the_kernel():
    x_in = np.arange(64, dtype=np.float64)
    x_out = np.array([32.0])
    y_in = np.full((1, 64), 1.0)
    kernel = kernels.BoxcarKernel(4)

    # a larger `scale` should still reproduce a constant signal, and should
    # not raise despite requiring a wider effective kernel window
    y_out = interpolate.interpolate_kernel(x_in, x_out, kernel, y_in, scale=4.0)

    np.testing.assert_allclose(y_out, [[1.0]], atol=1e-6)


def test_interpolate_kernel_rejects_invalid_kernel_type():
    x_in = np.arange(16, dtype=np.float64)
    x_out = np.array([6.0])
    y_in = np.full((1, 16), 1.0)

    with pytest.raises(TypeError):
        interpolate.interpolate_kernel(x_in, x_out, "not-a-kernel", y_in)
