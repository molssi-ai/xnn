"""Fused GPU kernels of periodic ByteFF-Pol's neighbor pairs (the fast path of molecular dynamics steps).

The real-space pairs of :class:`~xnn.hybnn.models.byteff.ByteFFPol` in a cell (repulsion, dispersion and
charge transfer of the pairs beyond 1,5, real-space Ewald electrostatics of every pair, the charges'
field and the mutual-induction ladders) as two Triton kernels with the numbers of the eager function
``_pol_real_space`` to round-off:

* the forward walks the compressed rows of a step plan's pairs
  (:func:`~xnn.ffnn.models.amoeba_kernels.pair_rows`), one program per atom, and sums the atom's half
  of every pair energy and its field in registers: no atomic adds; each pair's two ladders are
  written once, from the row of its first atom;
* the backward is one pass over the pairs: the gradient of the pair vectors from the gradients of the
  per-atom energies, the field and the ladders, through ``d D_n / dR = R D_(n+1)`` of the Ewald,
  Coulomb and Thole ladders. It is differentiable once: molecular dynamics, not training.

Imported only on CUDA devices, so that ``triton`` is needed only there.
"""
# no postponed annotations: Triton's interpreter recognizes the tl.constexpr parameters of a kernel only
# from evaluated annotations
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from xnn.ffnn.models.amoeba_kernels import _count

# the per-atom table: charge, damping volume, eps, lambda, r0, C6, charge-transfer eps and lambda
ATOM_COLUMNS = 8
# the float64 scalars the kernels read: squared cutoff, Ewald coefficient, 1/sqrt(pi), Thole factor,
# Coulomb constant, Born-Mayer prefactor, wall, dispersion damping, 1/6
N_SCALARS = 9
ROWS_CHUNK, ROWS_WARPS = 64, 2
TILE, WARPS = 128, 4

_SCALARS: dict = {}


def _scalars(values: tuple, device) -> Tensor:
    """The scalars of an evaluation as one float64 tensor on the device (cached: no copy from the host
    per call, which a CUDA-graph capture forbids)."""
    key = (values, str(device))
    if key not in _SCALARS:
        _SCALARS[key] = torch.tensor(values, dtype=torch.float64, device=device)
    return _SCALARS[key]


@triton.jit
def _atom(par_ptr, x, m, dt):
    """The eight parameters of atoms ``x`` (a block with mask ``m``, or one atom with ``m`` true)."""
    row = par_ptr + x * 8
    return (tl.load(row, mask=m, other=0.0).to(dt), tl.load(row + 1, mask=m, other=1.0).to(dt),
            tl.load(row + 2, mask=m, other=0.0).to(dt), tl.load(row + 3, mask=m, other=0.0).to(dt),
            tl.load(row + 4, mask=m, other=1.0).to(dt), tl.load(row + 5, mask=m, other=0.0).to(dt),
            tl.load(row + 6, mask=m, other=0.0).to(dt), tl.load(row + 7, mask=m, other=0.0).to(dt))


@triton.jit
def _pair_terms(R2, m, w_beyond, sc_ptr, q_a, d_a, eps_a, lam_a, r0_a, c6_a, cte_a, ctl_a, q_b, d_b, eps_b, lam_b,
                r0_b, c6_b, cte_b, ctl_b):
    """The terms of a block of pairs of atoms ``a`` and ``b`` (their parameters given): the energies
    (repulsion, dispersion, charge transfer, electrostatics) with their radial derivatives divided by R,
    the field factor ``f1`` and the ladders ``f2, f3`` (zero beyond the cutoff)."""
    dt = R2.dtype
    cut2 = tl.load(sc_ptr).to(dt)
    alpha = tl.load(sc_ptr + 1).to(dt)
    isp = tl.load(sc_ptr + 2).to(dt)
    thole = tl.load(sc_ptr + 3).to(dt)
    ke = tl.load(sc_ptr + 4).to(dt)
    pref = tl.load(sc_ptr + 5).to(dt)
    wall = tl.load(sc_ptr + 6).to(dt)
    damping = tl.load(sc_ptr + 7).to(dt)
    sixth = tl.load(sc_ptr + 8).to(dt)
    within = tl.where(m & (R2 < cut2), 1.0, 0.0).to(dt)
    w = within * w_beyond
    R2s = tl.where(m, R2, 1.0)
    R = libdevice.sqrt(R2s)
    inv = 1.0 / R
    inv2 = inv * inv
    eps = libdevice.sqrt(eps_a * eps_b)
    lam = libdevice.sqrt(lam_a * lam_b)
    r0 = 0.5 * (r0_a + r0_b)
    c6 = libdevice.sqrt(c6_a * c6_b)
    cte = libdevice.sqrt(cte_a * cte_b)
    ctl = libdevice.sqrt(ctl_a * ctl_b)
    # Born-Mayer repulsion with the (s12 / R)^12 wall
    born = pref * eps * libdevice.exp(lam * (1.0 - R / r0))
    s = wall * inv
    s2 = s * s
    s6 = s2 * s2 * s2
    wall12 = tl.where(wall > 0.0, s6 * s6, 0.0)
    rep = w * (born + wall12)
    drep = w * (-born * lam / r0 - 12.0 * wall12 * inv) * inv
    # damped C6 dispersion
    r06 = r0 * r0 * r0
    r06 = r06 * r06
    R6 = R2s * R2s * R2s
    den = damping * r06 + R6
    disp = -w * c6 / den
    ddisp = w * 6.0 * c6 * R2s * R2s / (den * den)
    # charge transfer
    k = ctl / r0
    k3 = k * k * k
    ect = libdevice.exp(-k3 * R2s * R)
    inv4 = inv2 * inv2
    ct = -w * cte * inv4 * ect
    dct = w * cte * ect * (4.0 * inv4 * inv2 + 3.0 * k3 * inv2 * inv)
    # Ewald ladder B_n of erfc(alpha R) / R
    ar = alpha * R
    expo = libdevice.exp(-ar * ar) * isp
    a2 = alpha * alpha
    b0 = libdevice.erfc(ar) * inv
    b1 = (b0 + 2.0 * alpha * expo) * inv2
    b2 = (3.0 * b1 + 4.0 * alpha * a2 * expo) * inv2
    b3 = (5.0 * b2 + 8.0 * alpha * a2 * a2 * expo) * inv2
    # Thole factors: 1 - lambda_n of d = a R^3 / sqrt(d_a d_b), capped at 50
    scale = libdevice.pow(d_a * d_b, sixth)
    x = R / scale
    d = tl.minimum(thole * x * x * x, 50.0)
    ex = libdevice.exp(-d)
    m3 = ex
    m5 = (1.0 + d) * ex
    m7 = (1.0 + d + 0.6 * d * d) * ex
    inv3 = inv2 * inv
    inv5 = inv3 * inv2
    inv7 = inv5 * inv2
    # the Ewald ladder less the bare one plus the damped one: D_n(Ewald) - (1 - lambda_n) D_n(Coulomb)
    f1 = within * (-b1 + m3 * inv3)
    f2 = within * (b2 - 3.0 * m5 * inv5)
    f3 = within * (-b3 + 15.0 * m7 * inv7)
    qq = ke * q_a * q_b
    elec = within * qq * b0
    delec = -within * qq * b1
    return rep, disp, ct, elec, drep, ddisp, dct, delec, f1, f2, f3


@triton.jit
def _pol_rows(ptr_ptr, pair_ptr, other_ptr, i_ptr, r_ptr, beyond_ptr, par_ptr, sc_ptr, n_ptr, e_ptr, f_ptr,
              du_ptr, CHUNK: tl.constexpr):
    """One program per atom ``a``: its half of the energies of the pairs of its row and the charges'
    field at it, summed in registers and stored once; the ladders ``f1, f2`` of each pair whose first
    atom is ``a``."""
    a = tl.program_id(0).to(tl.int64)
    start = tl.load(ptr_ptr + a)
    end = tl.load(ptr_ptr + a + 1)
    count = tl.load(n_ptr)
    acc = f_ptr.dtype.element_ty
    dt = r_ptr.dtype.element_ty
    q_a, d_a, eps_a, lam_a, r0_a, c6_a, cte_a, ctl_a = _atom(par_ptr, a, True, dt)
    s_rep = tl.zeros((CHUNK,), dtype=acc)
    s_disp = tl.zeros((CHUNK,), dtype=acc)
    s_ct = tl.zeros((CHUNK,), dtype=acc)
    s_el = tl.zeros((CHUNK,), dtype=acc)
    fx = tl.zeros((CHUNK,), dtype=acc)
    fy = tl.zeros((CHUNK,), dtype=acc)
    fz = tl.zeros((CHUNK,), dtype=acc)
    j = start
    while j < end:
        o = j + tl.arange(0, CHUNK)
        m = o < end
        e = tl.load(pair_ptr + o, mask=m, other=0)
        m = m & (e < count)
        e = e.to(tl.int64)
        b = tl.load(other_ptr + o, mask=m, other=0).to(tl.int64)
        first = tl.load(i_ptr + e, mask=m, other=0).to(tl.int64) == a
        sign = tl.where(first, 1.0, -1.0)
        rx = tl.load(r_ptr + e * 3, mask=m, other=1.0) * sign
        ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0) * sign
        rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0) * sign
        R2 = rx * rx + ry * ry + rz * rz
        wb = tl.load(beyond_ptr + e, mask=m, other=0.0).to(rx.dtype)
        q_b, d_b, eps_b, lam_b, r0_b, c6_b, cte_b, ctl_b = _atom(par_ptr, b, m, dt)
        rep, disp, ct, elec, _, _, _, _, f1, f2, _ = _pair_terms(R2, m, wb, sc_ptr, q_a, d_a, eps_a, lam_a, r0_a, c6_a,
                                                                 cte_a, ctl_a, q_b, d_b, eps_b, lam_b, r0_b, c6_b,
                                                                 cte_b, ctl_b)
        s_rep += tl.where(m, 0.5 * rep, 0.0).to(acc)
        s_disp += tl.where(m, 0.5 * disp, 0.0).to(acc)
        s_ct += tl.where(m, 0.5 * ct, 0.0).to(acc)
        s_el += tl.where(m, 0.5 * elec, 0.0).to(acc)
        g = tl.where(m, q_b * f1, 0.0)
        fx += (g * rx).to(acc)
        fy += (g * ry).to(acc)
        fz += (g * rz).to(acc)
        tl.store(du_ptr + e * 2, f1, mask=m & first)
        tl.store(du_ptr + e * 2 + 1, f2, mask=m & first)
        j += CHUNK
    tl.store(e_ptr + a * 4, tl.sum(s_rep, 0))
    tl.store(e_ptr + a * 4 + 1, tl.sum(s_disp, 0))
    tl.store(e_ptr + a * 4 + 2, tl.sum(s_ct, 0))
    tl.store(e_ptr + a * 4 + 3, tl.sum(s_el, 0))
    tl.store(f_ptr + a * 3, tl.sum(fx, 0))
    tl.store(f_ptr + a * 3 + 1, tl.sum(fy, 0))
    tl.store(f_ptr + a * 3 + 2, tl.sum(fz, 0))


@triton.jit
def _pol_pairs_backward(i_ptr, k_ptr, r_ptr, beyond_ptr, par_ptr, sc_ptr, n_ptr, ge_ptr, gf_ptr, gd_ptr, g_ptr,
                        n_pairs, BLOCK: tl.constexpr):
    """The gradient of the pair vectors ``(M, 3)`` from the gradients of the per-atom energies ``ge``
    ``(N, 4)``, of the field ``gf`` ``(N, 3)`` and of the ladders ``gd`` ``(M, 2)``: one pass over the
    pairs, each written once."""
    e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(n_ptr)
    m = (e < n_pairs) & (e < count)
    i = tl.load(i_ptr + e, mask=m, other=0).to(tl.int64)
    k = tl.load(k_ptr + e, mask=m, other=0).to(tl.int64)
    rx = tl.load(r_ptr + e * 3, mask=m, other=1.0)
    ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    R2 = rx * rx + ry * ry + rz * rz
    wb = tl.load(beyond_ptr + e, mask=m, other=0.0).to(rx.dtype)
    dt = rx.dtype
    q_i, d_i, eps_i, lam_i, r0_i, c6_i, cte_i, ctl_i = _atom(par_ptr, i, m, dt)
    q_k, d_k, eps_k, lam_k, r0_k, c6_k, cte_k, ctl_k = _atom(par_ptr, k, m, dt)
    _, _, _, _, drep, ddisp, dct, delec, f1, f2, f3 = _pair_terms(R2, m, wb, sc_ptr, q_i, d_i, eps_i, lam_i, r0_i,
                                                                  c6_i, cte_i, ctl_i, q_k, d_k, eps_k, lam_k, r0_k,
                                                                  c6_k, cte_k, ctl_k)
    g0 = 0.5 * (tl.load(ge_ptr + i * 4, mask=m, other=0.0) + tl.load(ge_ptr + k * 4, mask=m, other=0.0)).to(dt)
    g1 = 0.5 * (tl.load(ge_ptr + i * 4 + 1, mask=m, other=0.0) + tl.load(ge_ptr + k * 4 + 1, mask=m,
                                                                         other=0.0)).to(dt)
    g2 = 0.5 * (tl.load(ge_ptr + i * 4 + 2, mask=m, other=0.0) + tl.load(ge_ptr + k * 4 + 2, mask=m,
                                                                         other=0.0)).to(dt)
    g3 = 0.5 * (tl.load(ge_ptr + i * 4 + 3, mask=m, other=0.0) + tl.load(ge_ptr + k * 4 + 3, mask=m,
                                                                         other=0.0)).to(dt)
    # h = q_k gf[i] - q_i gf[k]: the field's pull on the pair
    hx = q_k * tl.load(gf_ptr + i * 3, mask=m, other=0.0).to(dt) - q_i * tl.load(gf_ptr + k * 3, mask=m,
                                                                                 other=0.0).to(dt)
    hy = q_k * tl.load(gf_ptr + i * 3 + 1, mask=m, other=0.0).to(dt) - q_i * tl.load(gf_ptr + k * 3 + 1, mask=m,
                                                                                     other=0.0).to(dt)
    hz = q_k * tl.load(gf_ptr + i * 3 + 2, mask=m, other=0.0).to(dt) - q_i * tl.load(gf_ptr + k * 3 + 2, mask=m,
                                                                                     other=0.0).to(dt)
    gd1 = tl.load(gd_ptr + e * 2, mask=m, other=0.0).to(dt)
    gd2 = tl.load(gd_ptr + e * 2 + 1, mask=m, other=0.0).to(dt)
    radial = (g0 * drep + g1 * ddisp + g2 * dct + g3 * delec + (rx * hx + ry * hy + rz * hz) * f2 + gd1 * f2
              + gd2 * f3)
    tl.store(g_ptr + e * 3, tl.where(m, radial * rx + f1 * hx, 0.0), mask=e < n_pairs)
    tl.store(g_ptr + e * 3 + 1, tl.where(m, radial * ry + f1 * hy, 0.0), mask=e < n_pairs)
    tl.store(g_ptr + e * 3 + 2, tl.where(m, radial * rz + f1 * hz, 0.0), mask=e < n_pairs)


class PolRealSpace(torch.autograd.Function):
    """Per-atom energies ``(N, 4)``, the charges' field ``(N, 3)`` and the mutual ladders ``f1, f2``
    ``(M,)`` of the pairs ``(i, k)`` with vectors ``r`` (``pos[k] - pos[i]``), as
    ``_pol_real_space`` of :mod:`~xnn.hybnn.models.byteff`; the gradient reaches ``r`` only."""

    @staticmethod
    def forward(ctx, r, i, k, beyond, atoms, rows, count, scalars):
        N, M = int(atoms.shape[0]), int(r.shape[0])
        ptr, pair, other = rows
        r, beyond, atoms = r.contiguous(), beyond.contiguous(), atoms.contiguous()
        energies = torch.empty(N, 4, dtype=r.dtype, device=r.device)
        field = torch.empty(N, 3, dtype=r.dtype, device=r.device)
        du = torch.zeros(M, 2, dtype=r.dtype, device=r.device)
        n = _count(count, M, r.device)
        if N:
            _pol_rows[(N,)](ptr, pair, other, i, r, beyond, atoms, scalars, n, energies, field, du,
                            CHUNK=ROWS_CHUNK, num_warps=ROWS_WARPS)
        ctx.save_for_backward(r, i, k, beyond, atoms, scalars, n)
        return energies, field, du[:, 0], du[:, 1]

    @staticmethod
    def backward(ctx, g_energies, g_field, g_d1, g_d2):
        r, i, k, beyond, atoms, scalars, n = ctx.saved_tensors
        N, M = int(atoms.shape[0]), int(r.shape[0])
        dtype, device = r.dtype, r.device

        def given(g, shape):
            return torch.zeros(shape, dtype=dtype, device=device) if g is None else g.contiguous()

        gd = torch.stack([given(g_d1, (M,)), given(g_d2, (M,))], 1)
        grad = torch.empty(M, 3, dtype=dtype, device=device)
        if M:
            _pol_pairs_backward[(triton.cdiv(M, TILE),)](
                i, k, r, beyond, atoms, scalars, n, given(g_energies, (N, 4)), given(g_field, (N, 3)), gd, grad, M,
                BLOCK=TILE, num_warps=WARPS)
        return grad, None, None, None, None, None, None, None


def pol_real_space(r: Tensor, i: Tensor, k: Tensor, beyond: Tensor, atoms: Tensor, rows: tuple,
                   count: Optional[Tensor], cutoff2: float, alpha: float, thole: float, ke: float, prefactor: float,
                   wall: float, damping: float) -> tuple:
    """The fused pairs of periodic ByteFF-Pol (see :class:`PolRealSpace`): per-atom energies ``(N, 4)``,
    the field ``(N, 3)`` and the ladders ``D1, D2`` ``(M,)``.

    Parameters
    ----------
    r : Tensor
        Pair vectors ``(M, 3)``.
    i, k : Tensor
        The pairs' atoms ``(M,)``, sorted by ``i``.
    beyond : Tensor
        Short-range weight per pair ``(M,)`` (zero within 1,5 and for the padding).
    atoms : Tensor
        Per-atom table ``(N, 8)`` (the columns of ``ATOM_COLUMNS``).
    rows : tuple
        The compressed rows of the pairs (:func:`~xnn.ffnn.models.amoeba_kernels.pair_rows`).
    count : Tensor, optional
        The number of valid pairs on the device (the rest is padding).
    cutoff2, alpha, thole, ke, prefactor, wall, damping : float
        Squared cutoff, Ewald coefficient, Thole factor, Coulomb constant, Born-Mayer prefactor, wall
        and dispersion damping.

    Returns
    -------
    tuple of Tensor
        ``energies``, ``field``, ``D1``, ``D2``.
    """
    scalars = _scalars((cutoff2, alpha, 1.0 / 1.7724538509055159, thole, ke, prefactor, wall, damping, 1.0 / 6.0),
                       r.device)
    return PolRealSpace.apply(r, i.to(torch.int64), k.to(torch.int64), beyond, atoms, rows, count, scalars)
