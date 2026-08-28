"""Tests for the `wyvern.kernels` module."""

import pytest

from wyvern import kernels

KERNEL_TYPES = [kernels.BoxcarKernel, kernels.LanczosKernel, kernels.KaiserBesselKernel]


@pytest.mark.parametrize("kernel_cls", KERNEL_TYPES)
def test_construction_sets_ntaps_and_half_width(kernel_cls):
    k = kernel_cls(8)
    assert k.ntaps() == 8
    assert k.half_width() == pytest.approx(4.0)


@pytest.mark.parametrize("kernel_cls", KERNEL_TYPES)
def test_evaluate_at_center_is_finite(kernel_cls):
    k = kernel_cls(8)
    value = k.evaluate(0.0)
    assert value == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("kernel_cls", KERNEL_TYPES)
def test_evaluate_outside_support_is_zero(kernel_cls):
    k = kernel_cls(8)
    assert k.evaluate(1000.0) == 0.0


@pytest.mark.parametrize("kernel_cls", KERNEL_TYPES)
def test_set_ntaps_rescales_half_width(kernel_cls):
    k = kernel_cls(8)
    k.set_ntaps(16)
    assert k.ntaps() == 16
    assert k.half_width() == pytest.approx(8.0)


@pytest.mark.parametrize("kernel_cls", KERNEL_TYPES)
def test_repr_contains_class_name(kernel_cls):
    k = kernel_cls(4)
    assert kernel_cls.__name__ in repr(k)


def test_boxcar_kernel_is_flat_inside_support():
    k = kernels.BoxcarKernel(8)
    assert k.evaluate(0.0) == 1.0
    assert k.evaluate(3.9) == 1.0
    assert k.evaluate(4.0) == 0.0


def test_kaiser_bessel_beta_defaults_and_can_be_overridden():
    k = kernels.KaiserBesselKernel(8)
    default_beta = k.beta()
    assert default_beta > 0.0

    k.set_beta(2.0)
    assert k.beta() == pytest.approx(2.0)

    k.set_beta_default()
    assert k.beta() == pytest.approx(default_beta)
