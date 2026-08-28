"""Tests for `wyvern.interpolate.interpolate_linear[_weighted]`."""

import numpy as np
import pytest

from wyvern import interpolate


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolate_linear_reproduces_linear_ramp(dtype):
    x_in = np.array([0.0, 1.0, 2.0, 3.0])
    x_out = np.array([0.5, 1.5, 2.5])
    y_in = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=dtype)

    y_out = interpolate.interpolate_linear(x_in, x_out, y_in)

    np.testing.assert_allclose(y_out, [[0.5, 1.5, 2.5]], atol=1e-5)
    assert y_out.dtype == dtype


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_interpolate_linear_supports_complex_dtypes(dtype):
    x_in = np.array([0.0, 1.0, 2.0])
    x_out = np.array([0.5, 1.5])
    y_in = np.array([[0 + 0j, 2 + 2j, 4 + 4j]], dtype=dtype)

    y_out = interpolate.interpolate_linear(x_in, x_out, y_in)

    np.testing.assert_allclose(y_out, [[1 + 1j, 3 + 3j]], atol=1e-5)


def test_interpolate_linear_can_reuse_output_array():
    x_in = np.array([0.0, 1.0, 2.0])
    x_out = np.array([0.5, 1.5])
    y_in = np.array([[0.0, 2.0, 4.0]])
    y_out = np.zeros((1, 2))

    result = interpolate.interpolate_linear(x_in, x_out, y_in, y_out=y_out)

    # the returned array should be the same buffer that was passed in
    assert result is y_out
    np.testing.assert_allclose(y_out, [[1.0, 3.0]])


def test_interpolate_linear_weighted_propagates_positive_weights():
    x_in = np.array([0.0, 1.0, 2.0])
    x_out = np.array([0.5, 1.5])
    y_in = np.array([[0.0, 2.0, 4.0]])
    w_in = np.array([[1.0, 1.0, 1.0]])

    y_out, w_out = interpolate.interpolate_linear_weighted(x_in, x_out, y_in, w_in)

    np.testing.assert_allclose(y_out, [[1.0, 3.0]])
    assert np.all(w_out > 0.0)


def test_interpolate_linear_rejects_too_few_input_samples():
    x_in = np.array([1.0])
    x_out = np.array([0.5])
    y_in = np.array([[1.0]])

    with pytest.raises(RuntimeError, match="at least 2 input samples"):
        interpolate.interpolate_linear(x_in, x_out, y_in)


def test_interpolate_linear_rejects_wrong_dimensionality():
    x_in = np.array([0.0, 1.0])
    x_out = np.array([0.5])
    # y_in must be 2D
    y_in = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="2D array"):
        interpolate.interpolate_linear(x_in, x_out, y_in)


def test_interpolate_linear_rejects_unsupported_dtype():
    x_in = np.array([0.0, 1.0])
    x_out = np.array([0.5])
    y_in = np.array([[1, 2]], dtype=np.int64)

    with pytest.raises(TypeError, match="unsupported type"):
        interpolate.interpolate_linear(x_in, x_out, y_in)
