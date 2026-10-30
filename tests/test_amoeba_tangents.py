"""The generated tangents of the AMOEBA and HIPPO device functions (``amoeba_tangents``).

Every ladder ``D_n = (1/R d/dR)^n g`` satisfies ``dD_n/dR = R D_(n+1)``, so the tangent of a ladder along the
distance (``R_t = 1``, ``inv_t = -1/R^2``) must equal ``R`` times its next member: a check of the generated
derivatives that needs no other implementation. Runs on the GPU or under Triton's interpreter.
"""
import math

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

from xnn.ffnn.models import amoeba_kernels as km  # noqa: E402
from xnn.ffnn.models import amoeba_tangents as tg  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def _probe(R_ptr, zi_ptr, zk_ptr, c_ptr, sc_ptr, out_ptr, n, WHICH: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    R = tl.load(R_ptr + o, mask=m, other=1.0)
    zi = tl.load(zi_ptr + o, mask=m, other=1.0)
    zk = tl.load(zk_ptr + o, mask=m, other=2.0)
    # the Ewald coefficient and 1 / sqrt(pi) in float64 (a Python float argument would be float32)
    alpha = tl.load(sc_ptr)
    inv_sqrt_pi = tl.load(sc_ptr + 1)
    inv = 1.0 / R
    R_t = R * 0.0 + 1.0
    inv_t = -inv * inv
    if WHICH == 0:
        d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5 = tg._coulomb6_t_1(inv, inv_t)
    elif WHICH == 1:
        d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5 = tg._ewald6_t_1100(R, inv, R * 0.0 + alpha, inv_sqrt_pi,
                                                                            R_t, inv_t)
    elif WHICH == 2:
        d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5 = tg.overlap_deficit5_t_01100(c_ptr, R, inv, zi, zk, R_t,
                                                                                      inv_t)
    elif WHICH == 3:
        d0, d1, d2, d3, d4, d5, t0, t1, t2, t3, t4, t5 = tg.squared_overlap5_t_01100(c_ptr, R, inv, zi, zk, R_t,
                                                                                      inv_t)
    else:
        d0, d1, d2, d3, d4, t0, t1, t2, t3, t4 = tg.core_valence_deficit4_t_01100(c_ptr, R, inv, zi, zk, R_t, inv_t)
        d5 = d4 * 0.0
        t5 = t4 * 0.0
    tl.store(out_ptr + o, d0, mask=m)
    tl.store(out_ptr + n + o, d1, mask=m)
    tl.store(out_ptr + 2 * n + o, d2, mask=m)
    tl.store(out_ptr + 3 * n + o, d3, mask=m)
    tl.store(out_ptr + 4 * n + o, d4, mask=m)
    tl.store(out_ptr + 5 * n + o, d5, mask=m)
    tl.store(out_ptr + 6 * n + o, t0, mask=m)
    tl.store(out_ptr + 7 * n + o, t1, mask=m)
    tl.store(out_ptr + 8 * n + o, t2, mask=m)
    tl.store(out_ptr + 9 * n + o, t3, mask=m)
    tl.store(out_ptr + 10 * n + o, t4, mask=m)
    tl.store(out_ptr + 11 * n + o, t5, mask=m)


# Triton's interpreter has no libdevice (exp, erfc): those ladders are checked on a GPU
_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="libdevice runs on a GPU only")


@pytest.mark.parametrize("which, members", [(0, 6), pytest.param(1, 6, marks=_GPU), pytest.param(2, 6, marks=_GPU),
                                            pytest.param(3, 6, marks=_GPU), pytest.param(4, 5, marks=_GPU)],
                         ids=["coulomb", "ewald", "overlap_deficit", "squared_overlap", "core_valence_deficit"])
def test_the_ladder_tangents_are_the_next_members(which, members):
    g = torch.Generator().manual_seed(5)
    n = 96
    R = (1.2 + 5.5 * torch.rand(n, generator=g, dtype=torch.float64)).to(DEVICE)
    zi = (2.0 + 3.0 * torch.rand(n, generator=g, dtype=torch.float64)).to(DEVICE)
    # a third of the pairs with nearly equal exponents: the series form
    zk = torch.where(torch.arange(n) % 3 == 0, zi.cpu() + 1e-3,
                     (2.0 + 3.0 * torch.rand(n, generator=g, dtype=torch.float64))).to(DEVICE)
    out = torch.zeros(12 * n, dtype=torch.float64, device=DEVICE)
    sc = torch.tensor([0.5446, 1.0 / math.sqrt(math.pi)], dtype=torch.float64, device=DEVICE)
    _probe[(triton.cdiv(n, 32),)](R, zi, zk, km.constants(torch.device(DEVICE)), sc, out, n, WHICH=which, BLOCK=32)
    d, t = out.view(12, n)[:members], out.view(12, n)[6:6 + members]
    for k in range(members - 1):
        want = R * d[k + 1]
        scale = want.abs().max()
        assert float((t[k] - want).abs().max() / scale) < 1e-11, k
