"""Derivatives of the spherical Bessel functions of :mod:`xnn.gnn.featurizers.spherical_bessel`.

With a gradient to track, ``spherical_bessel_jn`` differentiates through the identity
``j_l' = (l j_{l-1} - (l+1) j_{l+1}) / (2l+1)``; the checks compare it with the
closed forms of ``j_0`` and ``j_1`` and with finite differences to second order.
"""
import pytest
import torch

pytest.importorskip("e3nn")  # the xnn.gnn package import chain needs e3nn

from xnn.gnn.featurizers import spherical_bessel_jn  # noqa: E402


def _x():
    """Arguments on both sides of the series/recurrence switch, the origin included."""
    return torch.tensor([0.0, 0.3, 1.7, 2.5, 4.0, 7.5, 12.0], dtype=torch.float64, requires_grad=True)


def test_first_derivative_matches_closed_forms():
    """d j_0 / dx = -j_1 and d j_1 / dx = j_0 - 2 j_1 / x (j_1'(0) = 1/3)."""
    x = _x()
    j = spherical_bessel_jn(3, x)
    d0, = torch.autograd.grad(j[:, 0].sum(), x, retain_graph=True)
    d1, = torch.autograd.grad(j[:, 1].sum(), x)
    jd = j.detach()
    xs = x.detach()
    assert torch.allclose(d0, -jd[:, 1], rtol=0, atol=1e-15)
    expect1 = torch.where(xs > 0, jd[:, 0] - 2.0 * jd[:, 1] / xs.clamp_min(1e-300), torch.full_like(xs, 1.0 / 3.0))
    assert torch.allclose(d1, expect1, rtol=0, atol=1e-14)


def test_second_derivative_matches_closed_form():
    """d^2 j_0 / dx^2 = -j_0 + 2 j_1 / x (-1/3 at the origin), through the same identity again."""
    x = _x()
    j0 = spherical_bessel_jn(2, x)[:, 0]
    d, = torch.autograd.grad(j0.sum(), x, create_graph=True)
    dd, = torch.autograd.grad(d.sum(), x)
    jd = spherical_bessel_jn(2, x.detach())
    xs = x.detach()
    expect = torch.where(xs > 0, -jd[:, 0] + 2.0 * jd[:, 1] / xs.clamp_min(1e-300),
                         torch.full_like(xs, -1.0 / 3.0))
    assert torch.allclose(dd, expect, rtol=0, atol=1e-14)


def test_gradients_match_finite_differences():
    """Every order's first and second derivatives against finite differences, float32 kept."""
    x = _x().detach()[1:].requires_grad_(True)       # away from the origin, where x >= 0 ends
    assert torch.autograd.gradcheck(lambda t: spherical_bessel_jn(4, t), (x,))
    assert torch.autograd.gradgradcheck(lambda t: spherical_bessel_jn(4, t), (x,))
    x32 = x.detach().float().requires_grad_(True)
    j32 = spherical_bessel_jn(4, x32)
    g32, = torch.autograd.grad(j32.sum(), x32)
    g64, = torch.autograd.grad(spherical_bessel_jn(4, x).sum(), x)
    assert j32.dtype == g32.dtype == torch.float32
    assert torch.allclose(g32.double(), g64, rtol=0, atol=1e-6)
