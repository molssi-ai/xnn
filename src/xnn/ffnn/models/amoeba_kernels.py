"""Fused Triton kernels of the AMOEBA and HIPPO nonbonded pair terms.

One program handles a block of pairs and keeps a pair's whole arithmetic in registers: the
distance, the derivative ladders of every radial kernel (:mod:`xnn.ffnn.models.amoeba_ladders`,
evaluated in float64 as the eager path does), the multipole contractions, the switches. It writes
the per-atom halves of the four pair energies, the direct- and polar-scaled permanent fields, the
mutual ladder members of the induction and the exchange-polarization tensors with atomic adds,
and the two mutual ladder members per pair. The eager functions of :mod:`xnn.ffnn.models.amoeba`
(``hippo_pair_block``, ``_real_space_fields``, ``exchange_polarization_tensor``) define what is
computed; these kernels compute the same numbers with the same scale factors and flags. The AMOEBA kernels
(``_amoeba_forward`` / ``_amoeba_backward``) follow ``amoeba_pair_block`` and the buffered 14-7 term. The
backward kernels are differentiable once more through the generated tangent and adjoint kernels of
:mod:`xnn.ffnn.models.amoeba_tangents` (second derivatives, parameter gradients).

Imported only on CUDA devices, so that ``triton`` is needed only there.
"""
# no postponed annotations: Triton's interpreter (TRITON_INTERPRET=1) recognizes the
# tl.constexpr parameters of a kernel only from evaluated annotations
import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import once_differentiable
from triton.language.extra import libdevice

from xnn.common.models.csr import compressed, narrow_keys

from . import amoeba_ladders as lad
from .geometry import TINY

# the per-atom parameter table: one row per atom, these columns
PARAMS = ("chgpen_core", "chgpen_alpha", "rep_k", "rep_alpha", "rep_valence", "disp_alpha", "disp_c6",
          "chgtrn_eps", "chgtrn_alpha", "expol_spring", "expol_size", "expol_damp")
NP = len(PARAMS) + 1          # plus the exchange-polarization use flag
# the float64 scalars of an evaluation, loaded by the kernel: the Coulomb constant, 1 / sqrt(pi),
# the three Ewald coefficients, the switch windows (taper, cutoff) of the repulsion, charge transfer
# and dispersion, the neighbor cutoff, and the real-space cutoffs of the Ewald sums of the
# electrostatics and the dispersion
SCALARS = ("coulomb", "inv_sqrt_pi", "alpha_e", "alpha_p", "alpha_d", "rep_taper", "rep_cutoff", "ct_taper",
           "ct_cutoff", "disp_taper", "disp_cutoff", "cutoff", "ewald_cutoff", "disp_ewald_cutoff")
TILE, WARPS = 64, 2
# the mutual field: one program per atom walks the pairs of its row in chunks of this many
ROWS_CHUNK, ROWS_WARPS = 64, 2
# in float64 the pair kernels run as one launch per term group (the permanent electrostatics, the
# polar fields, the repulsion with the dispersion, the charge transfer and the exchange polarization),
# each with the live state of its own terms: one launch of every term (GROUP 0) spills its working
# set in float64, while in float32 it fits and the groups' repeated gathers and ladders cost more
SPLIT_DTYPES = (torch.float64,)
SPLIT_BACKWARD = (torch.float64,)
# the register cap of the pair kernels: without it ptxas allots the float64 backward 32 registers
MAXNREG = 255
# the AMOEBA backward in float32 (180 registers a thread) in blocks of one warp: more blocks resident per
# multiprocessor hide its atomic adds better than two-warp blocks
AMOEBA_BACKWARD_LAUNCH = {torch.float32: (32, 1)}
_COUNTS: dict = {}


def _count(count, M: int, device) -> Tensor:
    """The number of pairs a kernel processes, read on the device: ``count`` (a plan's count of
    valid pairs, so a replayed graph skips the padding of the list) or a cached tensor holding
    ``M``."""
    if count is not None:
        return count
    key = (M, str(device))
    if key not in _COUNTS:
        _COUNTS[key] = torch.tensor([M], dtype=torch.int32, device=device)
    return _COUNTS[key]


_SCALARS: dict = {}


def _segments(N: int, widths: dict, device, flat: Optional[Tensor] = None) -> tuple:
    """Per-atom accumulators ``(N, width)`` (``(N,)`` for a width of one) as contiguous segments of one
    float64 buffer, zeroed in one fill (or views into ``flat``, e.g. that buffer cast once)."""
    if flat is None:
        flat = torch.zeros(N * sum(widths.values()), dtype=torch.float64, device=device)
    out, at = {}, 0
    for key, width in widths.items():
        view = flat[at * N:(at + width) * N]
        out[key] = view if width == 1 else view.view(N, width)
        at += width
    return out, flat


def _cast_segments(flat: Tensor, N: int, widths: dict, dtype) -> dict:
    """The accumulators of :func:`_segments` in ``dtype``: one cast for all of them."""
    return _segments(N, widths, flat.device, flat if flat.dtype == dtype else flat.to(dtype))[0]


def _scalar(value: float, device) -> Tensor:
    """A float64 scalar on the device for a kernel to read (cached: no copy from the host per call,
    which a CUDA-graph capture forbids)."""
    key = (value, str(device))
    if key not in _SCALARS:
        _SCALARS[key] = torch.tensor([value], dtype=torch.float64, device=device)
    return _SCALARS[key]


@triton.jit
def _add_sorted(ptr, first, last, idx, v, same, tail, rest, TAIL: tl.constexpr):
    """Atomic adds of ``v`` to ``ptr[idx]`` for a block of pairs sorted by ``idx``: the lanes of
    the block's first index (``same``) add their sum in one operation, and with ``TAIL`` those of
    its last index too (``tail``, when it differs; a uniform branch skips the reduction of a block
    inside one atom), the lanes of any other index (``rest``) on their own. Adds to one address
    serialize, so a sorted list fed to plain atomics costs a warp's width of operations per
    element where this costs one; the second reduction pays in the kernels whose block holds
    little else (the backward, the mutual field) and not in the forward."""
    tl.atomic_add(ptr + first, tl.sum(tl.where(same, v, 0.0)), sem="relaxed")
    if TAIL:
        if last != first:
            tl.atomic_add(ptr + last, tl.sum(tl.where(tail, v, 0.0)), sem="relaxed")
        tl.atomic_add(ptr + idx, v, mask=rest, sem="relaxed")
    else:
        tl.atomic_add(ptr + idx, v, mask=rest | tail, sem="relaxed")

_tables: dict = {}


def constants(device: torch.device) -> Tensor:
    """The float64 table of the ladder constants on ``device`` (made once)."""
    key = str(device)
    if key not in _tables:
        _tables[key] = torch.tensor(lad.CONSTANTS, dtype=torch.float64, device=device)
    return _tables[key]


@triton.jit
def _coulomb6(inv):
    """``1/R, -1/R^3, 3/R^5, -15/R^7, 105/R^9, -945/R^11``."""
    inv2 = inv * inv
    d1 = -inv * inv2
    d2 = -3.0 * d1 * inv2
    d3 = -5.0 * d2 * inv2
    d4 = -7.0 * d3 * inv2
    d5 = -9.0 * d4 * inv2
    return inv, d1, d2, d3, d4, d5


@triton.jit
def _ewald6(R, inv, a, inv_sqrt_pi):
    """``D_n = (-1)^n B_n`` of ``erfc(a R) / R`` (``a`` zero gives the Coulomb ladder)."""
    inv2 = inv * inv
    ar = a * R
    expo = libdevice.exp(-ar * ar) * inv_sqrt_pi
    a2 = a * a
    b0 = libdevice.erfc(ar) * inv
    b1 = (b0 + 2.0 * a * expo) * inv2
    b2 = (3.0 * b1 + 4.0 * a * a2 * expo) * inv2
    b3 = (5.0 * b2 + 8.0 * a * a2 * a2 * expo) * inv2
    b4 = (7.0 * b3 + 16.0 * a * a2 * a2 * a2 * expo) * inv2
    b5 = (9.0 * b4 + 32.0 * a * a2 * a2 * a2 * a2 * expo) * inv2
    return b0, -b1, b2, -b3, b4, -b5


@triton.jit
def _switch(R, taper, cutoff):
    """The quintic switch, one below the taper and zero beyond the cutoff."""
    x = (R - taper) / (cutoff - taper)
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    x3 = x * x * x
    return 1.0 - x3 * (10.0 - 15.0 * x + 6.0 * x * x)


@triton.jit
def _energy(A, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz,
            dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
            rx, ry, rz, Li1, Li2, Lk1, Lk2, L1, L2, L3, L4):
    """The interaction of two multipole sets through a ladder (``multipole_energy``) with the
    charges folded into the ladder members: ``A`` is the charge-charge energy, ``Li_n`` the ladder
    of the dipole and quadrupole of i against the charge of k (that charge included), ``Lk_n`` the
    other way round, ``L_n`` the ladder of the dipole and quadrupole blocks. A charge that splits
    into a core and a valence part with their own ladders is one call."""
    dir_ = dix * rx + diy * ry + diz * rz
    dkr = dkx * rx + dky * ry + dkz * rz
    dik = dix * dkx + diy * dky + diz * dkz
    uix = qixx * rx + qixy * ry + qixz * rz
    uiy = qixy * rx + qiyy * ry + qiyz * rz
    uiz = qixz * rx + qiyz * ry + qizz * rz
    ukx = qkxx * rx + qkxy * ry + qkxz * rz
    uky = qkxy * rx + qkyy * ry + qkyz * rz
    ukz = qkxz * rx + qkyz * ry + qkzz * rz
    qir = uix * rx + uiy * ry + uiz * rz
    qkr = ukx * rx + uky * ry + ukz * rz
    qik = uix * ukx + uiy * uky + uiz * ukz
    dkqi = dkx * uix + dky * uiy + dkz * uiz
    diqk = dix * ukx + diy * uky + diz * ukz
    qiqk = (qixx * qkxx + qiyy * qkyy + qizz * qkzz
            + 2.0 * (qixy * qkxy + qixz * qkxz + qiyz * qkyz))
    return (A + Lk1 * dkr - Li1 * dir_ - (L2 * dir_ * dkr + L1 * dik) + Li2 * qir + Lk2 * qkr
            + L3 * (dkr * qir - dir_ * qkr) + 2.0 * L2 * (dkqi - diqk) + L4 * qir * qkr
            + 4.0 * L3 * qik + 2.0 * L2 * qiqk)


@triton.jit
def _field(C1, dx, dy, dz, qxx, qxy, qxz, qyy, qyz, qzz, rx, ry, rz, D1, D2, D3, sign):
    """The field of a multipole set at the other end of ``r`` (``multipole_field``): ``C1`` is
    the charge times its own first ladder member (a core and a valence charge with their own
    ladders summed), ``sign`` is ``-1`` for the field at ``k`` of the set at ``i`` and ``+1`` the
    other way."""
    dr = dx * rx + dy * ry + dz * rz
    ux = qxx * rx + qxy * ry + qxz * rz
    uy = qxy * rx + qyy * ry + qyz * rz
    uz = qxz * rx + qyz * ry + qzz * rz
    qrr = ux * rx + uy * ry + uz * rz
    common = sign * C1 + D2 * dr + sign * D3 * qrr
    fx = common * rx + D1 * dx + sign * 2.0 * D2 * ux
    fy = common * ry + D1 * dy + sign * 2.0 * D2 * uy
    fz = common * rz + D1 * dz + sign * 2.0 * D2 * uz
    return fx, fy, fz


@triton.jit
def _pairs_forward(i_ptr, k_ptr, r_ptr, s_ptr, keep_ptr, keepd_ptr, per_ptr, q_ptr, d_ptr, Q_ptr, par_ptr,
                   c_ptr, sc_ptr, enode_ptr, epair_ptr, fd_ptr, fp_ptr, du_ptr, xp_ptr, n_ptr, sel_ptr,
                   SELECT: tl.constexpr, KEEP_R: tl.constexpr, POLAR: tl.constexpr, EXPOL: tl.constexpr,
                   WRITE_PAIRS: tl.constexpr,
                   SW_REP: tl.constexpr, SW_CT: tl.constexpr, SW_DISP: tl.constexpr, NP: tl.constexpr,
                   GROUP: tl.constexpr, MIXED: tl.constexpr, BLOCK: tl.constexpr):
    """The HIPPO pair terms of one block of pairs (see the module docstring). The per-atom
    accumulators (``enode``, the fields, the tensors) may be float64 buffers for a float32
    evaluation: every atomic add casts to their type. ``GROUP`` selects the terms of a launch: 1
    the permanent electrostatics with the fields and mutual ladders, 2 the repulsion, dispersion,
    charge transfer and exchange polarization, 0 all of them. ``MIXED`` evaluates the ladders in
    the working precision instead of float64. ``KEEP_R`` forms the Ewald factors ``keep`` and
    ``keep_d`` from the distance and the cutoffs of the scalars instead of loading them. With
    ``SELECT`` the lanes take the pairs listed in ``sel`` (increasing, so still sorted by i), the
    first ``n`` of them; the others are not visited."""
    acc = enode_ptr.dtype.element_ty
    start = tl.program_id(0) * BLOCK
    n = tl.load(n_ptr)
    if start >= n:
        # past the pairs to visit (a plan's padding, the pairs beyond the cutoff)
        return
    lane = (start + tl.arange(0, BLOCK)).to(tl.int64)
    m = lane < n
    last = tl.minimum(start + BLOCK - 1, n - 1)
    if SELECT:
        e = tl.load(sel_ptr + lane, mask=m, other=0)
        first = tl.load(sel_ptr + start)
        last = tl.load(sel_ptr + last)
    else:
        e = lane
        first = start
    i = tl.load(i_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0)
    # the pairs of a plan are sorted by i: the lanes of a block's first atom add their sums in one
    # operation each (_add_sorted); the k side spreads over the atoms on its own
    i0 = tl.load(i_ptr + first)
    i1 = tl.load(i_ptr + last)
    same = m & (i == i0)
    tail = m & (i == i1) & (i1 != i0)
    rest = m & (i != i0) & (i != i1)
    rx = tl.load(r_ptr + e * 3, mask=m, other=1.0)
    ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    s1 = tl.load(s_ptr + e * 8 + 1, mask=m, other=0.0)
    s2 = tl.load(s_ptr + e * 8 + 2, mask=m, other=0.0)
    s3 = tl.load(s_ptr + e * 8 + 3, mask=m, other=0.0)
    s4 = tl.load(s_ptr + e * 8 + 4, mask=m, other=0.0)
    s5 = tl.load(s_ptr + e * 8 + 5, mask=m, other=0.0)
    s6 = tl.load(s_ptr + e * 8 + 6, mask=m, other=0.0)
    s7 = tl.load(s_ptr + e * 8 + 7, mask=m, other=0.0)
    if not KEEP_R:
        keep = tl.load(keep_ptr + e, mask=m, other=0.0)
        keep_d = tl.load(keepd_ptr + e, mask=m, other=0.0)
    per = tl.load(per_ptr + e, mask=m, other=0.0)
    coulomb = tl.load(sc_ptr)
    inv_sqrt_pi = tl.load(sc_ptr + 1)
    alpha_e = tl.load(sc_ptr + 2)
    alpha_p = tl.load(sc_ptr + 3)
    alpha_d = tl.load(sc_ptr + 4)
    # the two atoms: multipoles and class parameters
    qi = tl.load(q_ptr + i, mask=m, other=0.0)
    qk = tl.load(q_ptr + k, mask=m, other=0.0)
    dix = tl.load(d_ptr + i * 3, mask=m, other=0.0)
    diy = tl.load(d_ptr + i * 3 + 1, mask=m, other=0.0)
    diz = tl.load(d_ptr + i * 3 + 2, mask=m, other=0.0)
    dkx = tl.load(d_ptr + k * 3, mask=m, other=0.0)
    dky = tl.load(d_ptr + k * 3 + 1, mask=m, other=0.0)
    dkz = tl.load(d_ptr + k * 3 + 2, mask=m, other=0.0)
    qixx = tl.load(Q_ptr + i * 9, mask=m, other=0.0)
    qixy = tl.load(Q_ptr + i * 9 + 1, mask=m, other=0.0)
    qixz = tl.load(Q_ptr + i * 9 + 2, mask=m, other=0.0)
    qiyy = tl.load(Q_ptr + i * 9 + 4, mask=m, other=0.0)
    qiyz = tl.load(Q_ptr + i * 9 + 5, mask=m, other=0.0)
    qizz = tl.load(Q_ptr + i * 9 + 8, mask=m, other=0.0)
    qkxx = tl.load(Q_ptr + k * 9, mask=m, other=0.0)
    qkxy = tl.load(Q_ptr + k * 9 + 1, mask=m, other=0.0)
    qkxz = tl.load(Q_ptr + k * 9 + 2, mask=m, other=0.0)
    qkyy = tl.load(Q_ptr + k * 9 + 4, mask=m, other=0.0)
    qkyz = tl.load(Q_ptr + k * 9 + 5, mask=m, other=0.0)
    qkzz = tl.load(Q_ptr + k * 9 + 8, mask=m, other=0.0)
    pi = par_ptr + i * NP
    pk = par_ptr + k * NP
    core_i = tl.load(pi, mask=m, other=0.0)
    core_k = tl.load(pk, mask=m, other=0.0)
    zi = tl.load(pi + 1, mask=m, other=1.0)
    zk = tl.load(pk + 1, mask=m, other=2.0)
    # the distance and the ladders, the latter in float64 as the eager path evaluates them
    # the float32 square root and division of the GPU are approximate and biased: both through float64
    R = tl.sqrt((rx * rx + ry * ry + rz * rz).to(tl.float64)).to(rx.dtype)
    inv = (1.0 / R.to(tl.float64)).to(rx.dtype)
    if KEEP_R:
        # the Ewald factors from the distance: within the real-space cutoffs in a cell, one otherwise
        keep = tl.where(per > 0.0, tl.where(R < tl.load(sc_ptr + 12).to(R.dtype), 1.0, 0.0), 1.0).to(R.dtype)
        keep_d = tl.where(per > 0.0, tl.where(R < tl.load(sc_ptr + 13).to(R.dtype), 1.0, 0.0), 1.0).to(R.dtype)
    if MIXED:
        # the ladders in the working precision (Tinker9's mixed precision)
        R64 = R
        inv64 = inv
        zi64 = zi
        zk64 = zk
    else:
        # the ladders in float64 whatever the working precision, cast afterwards
        R64 = R.to(tl.float64)
        inv64 = 1.0 / R64
        zi64 = zi.to(tl.float64)
        zk64 = zk.to(tl.float64)
    dtype = rx.dtype
    zero = rx * 0.0
    val_i = qi - core_i
    val_k = qk - core_k
    # the charges in the ladder precision, for the deficit sums
    ci = core_i.to(R64.dtype)
    ck = core_k.to(R64.dtype)
    vi64 = val_i.to(R64.dtype)
    vk64 = val_k.to(R64.dtype)
    if GROUP != 2:
        c0, c1, c2, c3, c4, c5 = _coulomb6(inv)
        ae = (alpha_e * per.to(tl.float64)).to(dtype)
        e0, e1, e2, e3, e4, e5 = _ewald6(R, inv, ae, inv_sqrt_pi.to(dtype))
        # the deficits of the damped kernels (the Coulomb ladder minus the damped one) in the ladder
        # precision: the valence of i seen by the core of k (exponent zi), of k by the core of i
        # (zk), and the valence-valence overlap
        ki0, ki1, ki2, ki3, ki4 = lad.core_valence_deficit4(c_ptr, R64, inv64, zk64, zi64)
        ik0, ik1, ik2, ik3, ik4 = lad.core_valence_deficit4(c_ptr, R64, inv64, zi64, zk64)
        ov0, ov1, ov2, ov3, ov4, ov5 = lad.overlap_deficit5(c_ptr, R64, inv64, zi64, zk64)
        # the deficit sums, cast once: of the charge-charge term, of the charge of k seen by the
        # multipoles of i (orders 1 and 2), of the charge of i seen by k, and of the overlap alone
        D0 = (vi64 * ck * ki0 + ci * vk64 * ik0 + vi64 * vk64 * ov0).to(dtype)
        Di1 = (ck * ki1 + vk64 * ov1).to(dtype)
        Di2 = (ck * ki2 + vk64 * ov2).to(dtype)
        Dk1 = (ci * ik1 + vi64 * ov1).to(dtype)
        Dk2 = (ci * ik2 + vi64 * ov2).to(dtype)
        ov1 = ov1.to(dtype)
        ov2 = ov2.to(dtype)
        ov3 = ov3.to(dtype)
        ov4 = ov4.to(dtype)
        # permanent electrostatics in one contraction: the total multipoles through the point
        # ladder (the Coulomb ladder scaled by s1, in a cell plus the Ewald interaction minus the
        # bare one) minus the scaled deficits, so that no core-sized term is formed in the working
        # precision; the whole on the kept pairs
        u0 = s1 * c0 + (e0 - c0) * per
        u1 = s1 * c1 + (e1 - c1) * per
        u2 = s1 * c2 + (e2 - c2) * per
        u3 = s1 * c3 + (e3 - c3) * per
        u4 = s1 * c4 + (e4 - c4) * per
        A = qi * qk * u0 - s1 * D0
        Li1 = qk * u1 - s1 * Di1
        Li2 = qk * u2 - s1 * Di2
        Lk1 = qi * u1 - s1 * Dk1
        Lk2 = qi * u2 - s1 * Dk2
        L1 = u1 - s1 * ov1
        L2 = u2 - s1 * ov2
        L3 = u3 - s1 * ov3
        L4 = u4 - s1 * ov4
        e_mp = _energy(A, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz,
                       dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rx, ry, rz,
                       Li1, Li2, Lk1, Lk2, L1, L2, L3, L4) * keep * coulomb.to(dtype)
    if GROUP != 1:
        # Pauli repulsion: K_i K_k [valence multipoles through the ladder of S^2] / R
        rep_k_i = tl.load(pi + 2, mask=m, other=0.0)
        rep_k_k = tl.load(pk + 2, mask=m, other=0.0)
        rep_a_i = tl.load(pi + 3, mask=m, other=1.0)
        rep_a_k = tl.load(pk + 3, mask=m, other=2.0)
        rep_v_i = tl.load(pi + 4, mask=m, other=0.0)
        rep_v_k = tl.load(pk + 4, mask=m, other=0.0)
        sq0, sq1, sq2, sq3, sq4, sq5 = lad.squared_overlap5(c_ptr, R64, inv64, rep_a_i.to(R64.dtype),
                                                             rep_a_k.to(R64.dtype))
        sq0 = sq0.to(dtype)
        sq1 = sq1.to(dtype)
        sq2 = sq2.to(dtype)
        sq3 = sq3.to(dtype)
        sq4 = sq4.to(dtype)
        e_rep = rep_k_i * rep_k_k * _energy(rep_v_i * rep_v_k * sq0, -dix, -diy, -diz, -qixx, -qixy, -qixz, -qiyy,
                                            -qiyz, -qizz, -dkx, -dky, -dkz, -qkxx, -qkxy, -qkxz, -qkyy, -qkyz, -qkzz,
                                            rx, ry, rz, rep_v_k * sq1, rep_v_k * sq2, rep_v_i * sq1, rep_v_i * sq2,
                                            sq1, sq2, sq3, sq4) * inv * s4
        if SW_REP > 0:
            sw = _switch(R, tl.load(sc_ptr + 5).to(dtype), tl.load(sc_ptr + 6).to(dtype))
            if SW_REP == 1:
                sw = tl.where(per > 0.0, sw, 1.0)
            e_rep = e_rep * sw
            sw_rep = sw
        else:
            sw_rep = zero + 1.0
        # damped dispersion
        da_i = tl.load(pi + 5, mask=m, other=1.0)
        da_k = tl.load(pk + 5, mask=m, other=2.0)
        c6 = tl.load(pi + 6, mask=m, other=0.0) * tl.load(pk + 6, mask=m, other=0.0)
        dd0, dd1, dd2, dd3 = lad.overlap_deficit3(c_ptr, R64, inv64, da_i.to(R64.dtype), da_k.to(R64.dtype))
        # the damping scale minus one, 0.5 (3 lam5 - lam3) - 1 with lam3 = 1 + D1 R^3, lam5 = 1 - D2 R^5 / 3
        R3 = R64 * R64 * R64
        fm1 = (-0.5 * (dd2 * R3 * R64 * R64 + dd1 * R3)).to(dtype)
        xd = (alpha_d * per.to(tl.float64)).to(dtype) * R
        xd2 = xd * xd
        g = libdevice.exp(-xd2) * (1.0 + xd2 + 0.5 * xd2 * xd2)
        inv2 = inv * inv
        e_disp = -c6 * inv2 * inv2 * inv2 * (s5 * fm1 * (fm1 + 2.0) + (s5 - 1.0) + g) * keep_d
        if SW_DISP > 0:
            sw = _switch(R, tl.load(sc_ptr + 9).to(dtype), tl.load(sc_ptr + 10).to(dtype))
            if SW_DISP == 1:
                sw = tl.where(per > 0.0, sw, 1.0)
            e_disp = e_disp * sw
        # charge transfer
        eps_i = tl.load(pi + 7, mask=m, other=0.0)
        eps_k = tl.load(pk + 7, mask=m, other=0.0)
        eta_i = tl.load(pi + 8, mask=m, other=0.0)
        eta_k = tl.load(pk + 8, mask=m, other=0.0)
        t1 = tl.where(eta_k > 0.0, eps_i * libdevice.exp(-eta_k * R), zero)
        t2 = tl.where(eta_i > 0.0, eps_k * libdevice.exp(-eta_i * R), zero)
        e_ct = -(t1 + t2) * s1
        if SW_CT > 0:
            sw = _switch(R, tl.load(sc_ptr + 7).to(dtype), tl.load(sc_ptr + 8).to(dtype))
            if SW_CT == 1:
                sw = tl.where(per > 0.0, sw, 1.0)
            e_ct = e_ct * sw
    # the energies: halves to both atoms (and the pairs themselves when asked)
    if GROUP != 2:
        if WRITE_PAIRS:
            tl.store(epair_ptr + e * 4, e_mp, mask=m)
        _add_sorted(enode_ptr, i0 * 4, i1 * 4, i * 4, (0.5 * e_mp).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(enode_ptr + k * 4, (0.5 * e_mp).to(acc), mask=m, sem="relaxed")
    if GROUP != 1:
        if WRITE_PAIRS:
            tl.store(epair_ptr + e * 4 + 1, e_rep, mask=m)
            tl.store(epair_ptr + e * 4 + 2, e_disp, mask=m)
            tl.store(epair_ptr + e * 4 + 3, e_ct, mask=m)
        _add_sorted(enode_ptr + 1, i0 * 4, i1 * 4, i * 4, (0.5 * e_rep).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(enode_ptr + k * 4 + 1, (0.5 * e_rep).to(acc), mask=m, sem="relaxed")
        _add_sorted(enode_ptr + 2, i0 * 4, i1 * 4, i * 4, (0.5 * e_disp).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(enode_ptr + k * 4 + 2, (0.5 * e_disp).to(acc), mask=m, sem="relaxed")
        _add_sorted(enode_ptr + 3, i0 * 4, i1 * 4, i * 4, (0.5 * e_ct).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(enode_ptr + k * 4 + 3, (0.5 * e_ct).to(acc), mask=m, sem="relaxed")
    if POLAR and GROUP != 2:
        # the permanent field at k from i: the total charge through the Coulomb ladder minus the
        # valence's deficit, the dipole and quadrupole through the damped ladders; at i from k likewise
        fkx, fky, fkz = _field(qi * c1 - (vi64 * ki1).to(dtype), dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz,
                               rx, ry, rz, c1 - ki1.to(dtype), c2 - ki2.to(dtype), c3 - ki3.to(dtype), -1.0)
        fix, fiy, fiz = _field(qk * c1 - (vk64 * ik1).to(dtype), dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
                               rx, ry, rz, c1 - ik1.to(dtype), c2 - ik2.to(dtype), c3 - ik3.to(dtype), 1.0)
        # the Ewald field of the total multipoles minus their bare field, on the kept pairs of a cell
        ap = (alpha_p * per.to(tl.float64)).to(dtype)
        p0, p1, p2, p3, p4, p5 = _ewald6(R, inv, ap, inv_sqrt_pi.to(dtype))
        v1 = (p1 - c1) * keep * per
        v2 = (p2 - c2) * keep * per
        v3 = (p3 - c3) * keep * per
        ewkx, ewky, ewkz = _field(qi * v1, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, rx, ry, rz,
                                  v1, v2, v3, -1.0)
        ewix, ewiy, ewiz = _field(qk * v1, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rx, ry, rz,
                                  v1, v2, v3, 1.0)
        sd = s6 * keep
        sp = s2 * keep
        tl.atomic_add(fd_ptr + k * 3, (ewkx + fkx * sd).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(fd_ptr + k * 3 + 1, (ewky + fky * sd).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(fd_ptr + k * 3 + 2, (ewkz + fkz * sd).to(acc), mask=m, sem="relaxed")
        _add_sorted(fd_ptr, i0 * 3, i1 * 3, i * 3, (ewix + fix * sd).to(acc), same, tail, rest, TAIL=0)
        _add_sorted(fd_ptr + 1, i0 * 3, i1 * 3, i * 3, (ewiy + fiy * sd).to(acc), same, tail, rest, TAIL=0)
        _add_sorted(fd_ptr + 2, i0 * 3, i1 * 3, i * 3, (ewiz + fiz * sd).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(fp_ptr + k * 3, (ewkx + fkx * sp).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(fp_ptr + k * 3 + 1, (ewky + fky * sp).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(fp_ptr + k * 3 + 2, (ewkz + fkz * sp).to(acc), mask=m, sem="relaxed")
        _add_sorted(fp_ptr, i0 * 3, i1 * 3, i * 3, (ewix + fix * sp).to(acc), same, tail, rest, TAIL=0)
        _add_sorted(fp_ptr + 1, i0 * 3, i1 * 3, i * 3, (ewiy + fiy * sp).to(acc), same, tail, rest, TAIL=0)
        _add_sorted(fp_ptr + 2, i0 * 3, i1 * 3, i * 3, (ewiz + fiz * sp).to(acc), same, tail, rest, TAIL=0)
        # the mutual ladder of the solve: the damped two-center overlap (the Coulomb ladder minus
        # the overlap deficit of the penetration exponents), scaled, plus the Ewald difference
        u_scale = s7 * s3
        tl.store(du_ptr + e * 2, (p1 * per + c1 * (u_scale - per) - ov1 * u_scale) * keep, mask=m)
        tl.store(du_ptr + e * 2 + 1, (p2 * per + c2 * (u_scale - per) - ov2 * u_scale) * keep, mask=m)
    if POLAR and (EXPOL and GROUP != 1):
        # the exchange-polarization tensors: the squared overlap of two orbitals along the pair,
        # weighted by the polar scale, the use flags, the neighbor cutoff and the repulsion switch
        spring_i = tl.load(pi + 9, mask=m, other=0.0)
        spring_k = tl.load(pk + 9, mask=m, other=0.0)
        size_i = tl.load(pi + 10, mask=m, other=0.0)
        size_k = tl.load(pk + 10, mask=m, other=0.0)
        damp_i = tl.load(pi + 11, mask=m, other=0.0)
        damp_k = tl.load(pk + 11, mask=m, other=0.0)
        use = tl.maximum(tl.load(pi + 12, mask=m, other=0.0), tl.load(pk + 12, mask=m, other=0.0))
        weight = s2 * use * tl.where(R < tl.load(sc_ptr + 11).to(dtype), 1.0, 0.0) * sw_rep
        x = 0.5 * tl.sqrt(damp_i * damp_k) * R
        poly = 1.0 + x + x * x / 3.0
        s2x = size_i * size_k * poly * poly * libdevice.exp(-2.0 * x) * weight
        ux = rx * inv
        uy = ry * inv
        uz = rz * inv
        wi = spring_i * s2x
        wk = spring_k * s2x
        # only the pairs where either atom uses the term add to the tensors
        mx = m & (use > 0.0)
        samex = mx & (i == i0)
        tailx = mx & (i == i1) & (i1 != i0)
        restx = mx & (i != i0) & (i != i1)
        _add_sorted(xp_ptr, i0 * 9, i1 * 9, i * 9, (wi * ux * ux).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 1, i0 * 9, i1 * 9, i * 9, (wi * ux * uy).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 2, i0 * 9, i1 * 9, i * 9, (wi * ux * uz).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 3, i0 * 9, i1 * 9, i * 9, (wi * uy * ux).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 4, i0 * 9, i1 * 9, i * 9, (wi * uy * uy).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 5, i0 * 9, i1 * 9, i * 9, (wi * uy * uz).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 6, i0 * 9, i1 * 9, i * 9, (wi * uz * ux).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 7, i0 * 9, i1 * 9, i * 9, (wi * uz * uy).to(acc), samex, tailx, restx, TAIL=0)
        _add_sorted(xp_ptr + 8, i0 * 9, i1 * 9, i * 9, (wi * uz * uz).to(acc), samex, tailx, restx, TAIL=0)
        tl.atomic_add(xp_ptr + k * 9, (wk * ux * ux).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 1, (wk * ux * uy).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 2, (wk * ux * uz).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 3, (wk * uy * ux).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 4, (wk * uy * uy).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 5, (wk * uy * uz).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 6, (wk * uz * ux).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 7, (wk * uz * uy).to(acc), mask=mx, sem="relaxed")
        tl.atomic_add(xp_ptr + k * 9 + 8, (wk * uz * uz).to(acc), mask=mx, sem="relaxed")


def pairs_forward(i_: Tensor, k_: Tensor, r_vec: Tensor, scales: Tensor, keep: Tensor, keep_d: Tensor, per: Tensor,
                  q: Tensor, d: Tensor, Q: Tensor, params: Tensor, scalars: Tensor, polar: bool, expol: bool,
                  switch_modes: tuple, write_pairs: bool = False, count=None, mixed: bool = False,
                  select: Optional[Tensor] = None, tangents: Optional[tuple] = None) -> dict:
    """Launch the pair kernel.

    Parameters
    ----------
    i_, k_ : Tensor
        The pairs ``(M,)`` (int64).
    r_vec : Tensor
        ``r_k - r_i`` ``(M, 3)`` in the working dtype.
    scales : Tensor
        The eight scale factors per pair ``(M, 8)``.
    keep, keep_d, per : Tensor
        Per pair: the Ewald real-space factor of the electrostatics and of the dispersion
        (one for pairs of molecules), and whether the pair belongs to a periodic cell. ``keep``
        and ``keep_d`` may be ``None``: the kernel then forms them from the distance and the
        real-space cutoffs of ``scalars``.
    q, d, Q : Tensor
        The global-frame multipoles ``(N,)``, ``(N, 3)``, ``(N, 3, 3)``.
    params : Tensor
        The per-atom parameter table ``(N, NP)`` (:data:`PARAMS` plus the use flag).
    scalars : Tensor
        The float64 scalars of the evaluation (:data:`SCALARS`).
    polar, expol : bool
        Whether the fields and mutual ladders, and the exchange-polarization tensors, are computed.
    switch_modes : tuple of int
        The switch of the repulsion, the charge transfer and the dispersion: 0 none, 1 on the
        pairs of cells only, 2 on every pair.
    write_pairs : bool, optional
        Also return the four energies per pair.
    count : Tensor, optional
        The number of valid pairs as an int32 tensor ``(1,)`` on the device (a plan's); the pairs
        after it (the padding of the list) are skipped. All ``M`` pairs by default.
    mixed : bool, optional
        Evaluate the ladders in the working precision (Tinker9's mixed precision) instead of
        float64; no effect on a float64 evaluation.
    select : Tensor, optional
        The pairs to visit, increasing (:func:`select_pairs`); ``count`` is then their number.
        The others contribute nothing (pairs beyond the cutoff) and get zeros per pair.
    tangents : tuple of Tensor, optional
        Directions of the pair vectors, charges, dipoles and quadrupoles: the generated tangent kernel
        (:mod:`.amoeba_tangents`) also returns the outputs' derivatives along them (``<key>_t``).

    Returns
    -------
    dict
        ``e_node`` ``(N, 4)``, ``field_d`` / ``field_p`` ``(N, 3)``, ``du`` ``(M, 2)``, ``expol``
        ``(N, 3, 3)`` (or ``None``) and ``e_pair`` ``(M, 4)`` (or ``None``).
    """
    M, N = int(i_.shape[0]), int(q.shape[0])
    dtype, device = r_vec.dtype, r_vec.device
    # the sums over pairs accumulate in float64 whatever the working precision
    widths = {"e_node": 4, "field_d": 3, "field_p": 3, **({"expol": 9} if (polar and expol) else {})}
    out, flat = _segments(N, widths, device)
    out.setdefault("expol", None)
    out.update({"du": torch.zeros(M, 2, dtype=dtype, device=device),
                "e_pair": ((torch.empty if select is None else torch.zeros)(M, 4, dtype=dtype, device=device)
                           if write_pairs else None)})

    keys = ("e_node", "field_d", "field_p", "du", "expol", "e_pair")
    if tangents is not None:
        out.update({k + "_t": None if out[k] is None else torch.zeros_like(out[k]) for k in keys})

    def finish(out):
        out.update(_cast_segments(flat, N, widths, dtype))
        for key in ("e_node_t", "field_d_t", "field_p_t", "expol_t"):
            if out.get(key) is not None:
                out[key] = out[key].to(dtype)
        for key in ("expol", "expol_t"):
            if out.get(key) is not None:
                out[key] = out[key].view(N, 3, 3)
        return out

    if M == 0:
        return finish(out)
    dummy = out["du"]
    kernel, extra = _pairs_forward, ()
    if tangents is not None:
        from . import amoeba_tangents as tg
        kernel = tg._pairs_forward_t
        r_t, q_t, d_t, Q_t = (t.contiguous() for t in tangents)
        extra = (r_t, q_t, d_t, Q_t.reshape(N, 9).contiguous(),
                 *[dummy if out[k + "_t"] is None else out[k + "_t"] for k in ("e_node", "e_pair", "field_d",
                                                                               "field_p", "du", "expol")])
    for group in ((1, 2) if dtype in SPLIT_DTYPES else (0,)):
        kernel[(triton.cdiv(M, TILE),)](
            i_, k_, r_vec.contiguous(), scales.contiguous(), dummy if keep is None else keep.contiguous(),
            dummy if keep_d is None else keep_d.contiguous(), per.contiguous(), q.contiguous(), d.contiguous(),
            Q.reshape(N, 9).contiguous(), params.contiguous(), constants(device), scalars, out["e_node"],
            out["e_pair"] if write_pairs else dummy, out["field_d"], out["field_p"], out["du"],
            out["expol"] if out["expol"] is not None else dummy, _count(count, M, device),
            dummy if select is None else select, *extra, SELECT=select is not None, KEEP_R=keep is None,
            POLAR=bool(polar), EXPOL=bool(polar and expol), WRITE_PAIRS=bool(write_pairs),
            SW_REP=int(switch_modes[0]), SW_CT=int(switch_modes[1]), SW_DISP=int(switch_modes[2]),
            NP=NP, GROUP=group, MIXED=bool(mixed), BLOCK=TILE, num_warps=WARPS, maxnreg=MAXNREG)
    return finish(out)


@triton.jit
def _energy_adjoint(gE, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz,
                    dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
                    rx, ry, rz, Li1, Li2, Lk1, Lk2, L1, L2, L3, L4):
    """The gradients of ``gE * _energy(...)``: with respect to the two dipoles, the two quadrupoles
    (nine entries each, as autograd of the eager contraction gives them), the pair vector at fixed
    ladders, and the eight ladder members. The charge gradients and the radial chain follow in
    the caller from how the ladders were composed."""
    dir_ = dix * rx + diy * ry + diz * rz
    dkr = dkx * rx + dky * ry + dkz * rz
    dik = dix * dkx + diy * dky + diz * dkz
    uix = qixx * rx + qixy * ry + qixz * rz
    uiy = qixy * rx + qiyy * ry + qiyz * rz
    uiz = qixz * rx + qiyz * ry + qizz * rz
    ukx = qkxx * rx + qkxy * ry + qkxz * rz
    uky = qkxy * rx + qkyy * ry + qkyz * rz
    ukz = qkxz * rx + qkyz * ry + qkzz * rz
    qir = uix * rx + uiy * ry + uiz * rz
    qkr = ukx * rx + uky * ry + ukz * rz
    qik = uix * ukx + uiy * uky + uiz * ukz
    dkqi = dkx * uix + dky * uiy + dkz * uiz
    diqk = dix * ukx + diy * uky + diz * ukz
    qiqk = (qixx * qkxx + qiyy * qkyy + qizz * qkzz
            + 2.0 * (qixy * qkxy + qixz * qkxz + qiyz * qkyz))
    g_dir = gE * (-Li1 - L2 * dkr - L3 * qkr)
    g_dkr = gE * (Lk1 - L2 * dir_ + L3 * qir)
    g_dik = -gE * L1
    g_qir = gE * (Li2 + L3 * dkr + L4 * qkr)
    g_qkr = gE * (Lk2 - L3 * dir_ + L4 * qir)
    g_qik = 4.0 * gE * L3
    g_dkqi = 2.0 * gE * L2
    g_diqk = -2.0 * gE * L2
    g_qiqk = 2.0 * gE * L2
    gLi1 = -gE * dir_
    gLi2 = gE * qir
    gLk1 = gE * dkr
    gLk2 = gE * qkr
    gL1 = -gE * dik
    gL2 = gE * (-dir_ * dkr + 2.0 * (dkqi - diqk) + 2.0 * qiqk)
    gL3 = gE * (dkr * qir - dir_ * qkr + 4.0 * qik)
    gL4 = gE * qir * qkr
    gdix = g_dir * rx + g_dik * dkx + g_diqk * ukx
    gdiy = g_dir * ry + g_dik * dky + g_diqk * uky
    gdiz = g_dir * rz + g_dik * dkz + g_diqk * ukz
    gdkx = g_dkr * rx + g_dik * dix + g_dkqi * uix
    gdky = g_dkr * ry + g_dik * diy + g_dkqi * uiy
    gdkz = g_dkr * rz + g_dik * diz + g_dkqi * uiz
    # Q_i u_k, Q_k u_i, Q_i d_k, Q_k d_i
    qiukx = qixx * ukx + qixy * uky + qixz * ukz
    qiuky = qixy * ukx + qiyy * uky + qiyz * ukz
    qiukz = qixz * ukx + qiyz * uky + qizz * ukz
    qkuix = qkxx * uix + qkxy * uiy + qkxz * uiz
    qkuiy = qkxy * uix + qkyy * uiy + qkyz * uiz
    qkuiz = qkxz * uix + qkyz * uiy + qkzz * uiz
    qidkx = qixx * dkx + qixy * dky + qixz * dkz
    qidky = qixy * dkx + qiyy * dky + qiyz * dkz
    qidkz = qixz * dkx + qiyz * dky + qizz * dkz
    qkdix = qkxx * dix + qkxy * diy + qkxz * diz
    qkdiy = qkxy * dix + qkyy * diy + qkyz * diz
    qkdiz = qkxz * dix + qkyz * diy + qkzz * diz
    grx = (g_dir * dix + g_dkr * dkx + 2.0 * g_qir * uix + 2.0 * g_qkr * ukx + g_qik * (qiukx + qkuix)
           + g_dkqi * qidkx + g_diqk * qkdix)
    gry = (g_dir * diy + g_dkr * dky + 2.0 * g_qir * uiy + 2.0 * g_qkr * uky + g_qik * (qiuky + qkuiy)
           + g_dkqi * qidky + g_diqk * qkdiy)
    grz = (g_dir * diz + g_dkr * dkz + 2.0 * g_qir * uiz + 2.0 * g_qkr * ukz + g_qik * (qiukz + qkuiz)
           + g_dkqi * qidkz + g_diqk * qkdiz)
    # the nine entries of the quadrupole gradients: (a, b) of r_a r_b, u_a r_b, d_a r_b and Q_ab
    gqixx = g_qir * rx * rx + g_qik * ukx * rx + g_dkqi * dkx * rx + g_qiqk * qkxx
    gqixy = g_qir * rx * ry + g_qik * ukx * ry + g_dkqi * dkx * ry + g_qiqk * qkxy
    gqixz = g_qir * rx * rz + g_qik * ukx * rz + g_dkqi * dkx * rz + g_qiqk * qkxz
    gqiyx = g_qir * ry * rx + g_qik * uky * rx + g_dkqi * dky * rx + g_qiqk * qkxy
    gqiyy = g_qir * ry * ry + g_qik * uky * ry + g_dkqi * dky * ry + g_qiqk * qkyy
    gqiyz = g_qir * ry * rz + g_qik * uky * rz + g_dkqi * dky * rz + g_qiqk * qkyz
    gqizx = g_qir * rz * rx + g_qik * ukz * rx + g_dkqi * dkz * rx + g_qiqk * qkxz
    gqizy = g_qir * rz * ry + g_qik * ukz * ry + g_dkqi * dkz * ry + g_qiqk * qkyz
    gqizz = g_qir * rz * rz + g_qik * ukz * rz + g_dkqi * dkz * rz + g_qiqk * qkzz
    gqkxx = g_qkr * rx * rx + g_qik * uix * rx + g_diqk * dix * rx + g_qiqk * qixx
    gqkxy = g_qkr * rx * ry + g_qik * uix * ry + g_diqk * dix * ry + g_qiqk * qixy
    gqkxz = g_qkr * rx * rz + g_qik * uix * rz + g_diqk * dix * rz + g_qiqk * qixz
    gqkyx = g_qkr * ry * rx + g_qik * uiy * rx + g_diqk * diy * rx + g_qiqk * qixy
    gqkyy = g_qkr * ry * ry + g_qik * uiy * ry + g_diqk * diy * ry + g_qiqk * qiyy
    gqkyz = g_qkr * ry * rz + g_qik * uiy * rz + g_diqk * diy * rz + g_qiqk * qiyz
    gqkzx = g_qkr * rz * rx + g_qik * uiz * rx + g_diqk * diz * rx + g_qiqk * qixz
    gqkzy = g_qkr * rz * ry + g_qik * uiz * ry + g_diqk * diz * ry + g_qiqk * qiyz
    gqkzz = g_qkr * rz * rz + g_qik * uiz * rz + g_diqk * diz * rz + g_qiqk * qizz
    return (gdix, gdiy, gdiz, gdkx, gdky, gdkz,
            gqixx, gqixy, gqixz, gqiyx, gqiyy, gqiyz, gqizx, gqizy, gqizz,
            gqkxx, gqkxy, gqkxz, gqkyx, gqkyy, gqkyz, gqkzx, gqkzy, gqkzz,
            grx, gry, grz, gLi1, gLi2, gLk1, gLk2, gL1, gL2, gL3, gL4)


@triton.jit
def _field_adjoint(gx, gy, gz, C1, dx, dy, dz, qxx, qxy, qxz, qyy, qyz, qzz, rx, ry, rz, D1, D2, D3, sign):
    """The gradients of ``g . _field(...)``: the folded charge ``C1``, the dipole, the quadrupole
    (nine entries), the pair vector at fixed ladder, and the three ladder members of the dipole
    and quadrupole parts."""
    dr = dx * rx + dy * ry + dz * rz
    ux = qxx * rx + qxy * ry + qxz * rz
    uy = qxy * rx + qyy * ry + qyz * rz
    uz = qxz * rx + qyz * ry + qzz * rz
    qrr = ux * rx + uy * ry + uz * rz
    gr_ = gx * rx + gy * ry + gz * rz
    gd_ = gx * dx + gy * dy + gz * dz
    gu = gx * ux + gy * uy + gz * uz
    gC1 = sign * gr_
    gdx = D2 * gr_ * rx + D1 * gx
    gdy = D2 * gr_ * ry + D1 * gy
    gdz = D2 * gr_ * rz + D1 * gz
    a = sign * D3 * gr_
    b = sign * 2.0 * D2
    gqxx = a * rx * rx + b * gx * rx
    gqxy = a * rx * ry + b * gx * ry
    gqxz = a * rx * rz + b * gx * rz
    gqyx = a * ry * rx + b * gy * rx
    gqyy = a * ry * ry + b * gy * ry
    gqyz = a * ry * rz + b * gy * rz
    gqzx = a * rz * rx + b * gz * rx
    gqzy = a * rz * ry + b * gz * ry
    gqzz = a * rz * rz + b * gz * rz
    qgx = qxx * gx + qxy * gy + qxz * gz
    qgy = qxy * gx + qyy * gy + qyz * gz
    qgz = qxz * gx + qyz * gy + qzz * gz
    common = sign * C1 + D2 * dr + sign * D3 * qrr
    grx = common * gx + D2 * gr_ * dx + sign * D3 * gr_ * 2.0 * ux + b * qgx
    gry = common * gy + D2 * gr_ * dy + sign * D3 * gr_ * 2.0 * uy + b * qgy
    grz = common * gz + D2 * gr_ * dz + sign * D3 * gr_ * 2.0 * uz + b * qgz
    gD1 = gd_
    gD2 = gr_ * dr + sign * 2.0 * gu
    gD3 = sign * gr_ * qrr
    return gC1, gdx, gdy, gdz, gqxx, gqxy, gqxz, gqyx, gqyy, gqyz, gqzx, gqzy, gqzz, grx, gry, grz, gD1, gD2, gD3


@triton.jit
def _switch_derivative(R, taper, cutoff):
    """``d S / d R`` of the quintic switch."""
    x = (R - taper) / (cutoff - taper)
    inside = (x > 0.0) & (x < 1.0)
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    y = 1.0 - x
    return tl.where(inside, -30.0 * x * x * y * y / (cutoff - taper), 0.0 * R)


@triton.jit
def _pairs_backward(i_ptr, k_ptr, r_ptr, s_ptr, keep_ptr, keepd_ptr, per_ptr, q_ptr, d_ptr, Q_ptr, par_ptr,
                    c_ptr, sc_ptr, ge_ptr, gfd_ptr, gfp_ptr, gdu_ptr, gxp_ptr, gr_ptr, gq_ptr, gd_ptr, gQ_ptr,
                    n_ptr, sel_ptr, SELECT: tl.constexpr, KEEP_R: tl.constexpr, POLAR: tl.constexpr,
                    EXPOL: tl.constexpr,
                    SW_REP: tl.constexpr, SW_CT: tl.constexpr,
                    SW_DISP: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr, MIXED: tl.constexpr,
                    BLOCK: tl.constexpr):
    """The gradients of the pair terms for incoming gradients of every output of the forward
    kernel: per pair with respect to the pair vector (stored), per atom with respect to the
    charges, dipoles and quadrupoles (atomic adds, cast to the accumulators' type). ``GROUP``
    selects the terms of a launch: 1 the permanent electrostatics, 2 the polar fields and mutual
    ladders, 3 the repulsion, dispersion, charge transfer and exchange polarization, 0 all; the
    groups after the first add to the stored pair-vector gradient. ``SELECT`` as in the forward
    kernel: the pairs not visited keep a zero gradient."""
    acc = gq_ptr.dtype.element_ty
    start = tl.program_id(0) * BLOCK
    n = tl.load(n_ptr)
    if start >= n:
        # past the pairs to visit (a plan's padding, the pairs beyond the cutoff)
        return
    lane = (start + tl.arange(0, BLOCK)).to(tl.int64)
    m = lane < n
    last = tl.minimum(start + BLOCK - 1, n - 1)
    if SELECT:
        e = tl.load(sel_ptr + lane, mask=m, other=0)
        first = tl.load(sel_ptr + start)
        last = tl.load(sel_ptr + last)
    else:
        e = lane
        first = start
    i = tl.load(i_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0)
    # the pairs of a plan are sorted by i: the lanes of a block's first atom add their sums in one
    # operation each (_add_sorted); the k side spreads over the atoms on its own
    i0 = tl.load(i_ptr + first)
    i1 = tl.load(i_ptr + last)
    same = m & (i == i0)
    tail = m & (i == i1) & (i1 != i0)
    rest = m & (i != i0) & (i != i1)
    rx = tl.load(r_ptr + e * 3, mask=m, other=1.0)
    ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    s1 = tl.load(s_ptr + e * 8 + 1, mask=m, other=0.0)
    s2 = tl.load(s_ptr + e * 8 + 2, mask=m, other=0.0)
    s3 = tl.load(s_ptr + e * 8 + 3, mask=m, other=0.0)
    s4 = tl.load(s_ptr + e * 8 + 4, mask=m, other=0.0)
    s5 = tl.load(s_ptr + e * 8 + 5, mask=m, other=0.0)
    s6 = tl.load(s_ptr + e * 8 + 6, mask=m, other=0.0)
    s7 = tl.load(s_ptr + e * 8 + 7, mask=m, other=0.0)
    if not KEEP_R:
        keep = tl.load(keep_ptr + e, mask=m, other=0.0)
        keep_d = tl.load(keepd_ptr + e, mask=m, other=0.0)
    per = tl.load(per_ptr + e, mask=m, other=0.0)
    coulomb = tl.load(sc_ptr)
    inv_sqrt_pi = tl.load(sc_ptr + 1)
    alpha_e = tl.load(sc_ptr + 2)
    alpha_p = tl.load(sc_ptr + 3)
    alpha_d = tl.load(sc_ptr + 4)
    qi = tl.load(q_ptr + i, mask=m, other=0.0)
    qk = tl.load(q_ptr + k, mask=m, other=0.0)
    dix = tl.load(d_ptr + i * 3, mask=m, other=0.0)
    diy = tl.load(d_ptr + i * 3 + 1, mask=m, other=0.0)
    diz = tl.load(d_ptr + i * 3 + 2, mask=m, other=0.0)
    dkx = tl.load(d_ptr + k * 3, mask=m, other=0.0)
    dky = tl.load(d_ptr + k * 3 + 1, mask=m, other=0.0)
    dkz = tl.load(d_ptr + k * 3 + 2, mask=m, other=0.0)
    qixx = tl.load(Q_ptr + i * 9, mask=m, other=0.0)
    qixy = tl.load(Q_ptr + i * 9 + 1, mask=m, other=0.0)
    qixz = tl.load(Q_ptr + i * 9 + 2, mask=m, other=0.0)
    qiyy = tl.load(Q_ptr + i * 9 + 4, mask=m, other=0.0)
    qiyz = tl.load(Q_ptr + i * 9 + 5, mask=m, other=0.0)
    qizz = tl.load(Q_ptr + i * 9 + 8, mask=m, other=0.0)
    qkxx = tl.load(Q_ptr + k * 9, mask=m, other=0.0)
    qkxy = tl.load(Q_ptr + k * 9 + 1, mask=m, other=0.0)
    qkxz = tl.load(Q_ptr + k * 9 + 2, mask=m, other=0.0)
    qkyy = tl.load(Q_ptr + k * 9 + 4, mask=m, other=0.0)
    qkyz = tl.load(Q_ptr + k * 9 + 5, mask=m, other=0.0)
    qkzz = tl.load(Q_ptr + k * 9 + 8, mask=m, other=0.0)
    pi = par_ptr + i * NP
    pk = par_ptr + k * NP
    core_i = tl.load(pi, mask=m, other=0.0)
    core_k = tl.load(pk, mask=m, other=0.0)
    zi = tl.load(pi + 1, mask=m, other=1.0)
    zk = tl.load(pk + 1, mask=m, other=2.0)
    # the incoming gradients: the energies' per-atom halves, the fields, the ladders, the tensors
    g_mp = 0.5 * (tl.load(ge_ptr + i * 4, mask=m, other=0.0) + tl.load(ge_ptr + k * 4, mask=m, other=0.0))
    g_rep = 0.5 * (tl.load(ge_ptr + i * 4 + 1, mask=m, other=0.0) + tl.load(ge_ptr + k * 4 + 1, mask=m, other=0.0))
    g_disp = 0.5 * (tl.load(ge_ptr + i * 4 + 2, mask=m, other=0.0) + tl.load(ge_ptr + k * 4 + 2, mask=m, other=0.0))
    g_ct = 0.5 * (tl.load(ge_ptr + i * 4 + 3, mask=m, other=0.0) + tl.load(ge_ptr + k * 4 + 3, mask=m, other=0.0))
    # the float32 square root and division of the GPU are approximate and biased: both through float64
    R = tl.sqrt((rx * rx + ry * ry + rz * rz).to(tl.float64)).to(rx.dtype)
    inv = (1.0 / R.to(tl.float64)).to(rx.dtype)
    if KEEP_R:
        # the Ewald factors from the distance: within the real-space cutoffs in a cell, one otherwise
        keep = tl.where(per > 0.0, tl.where(R < tl.load(sc_ptr + 12).to(R.dtype), 1.0, 0.0), 1.0).to(R.dtype)
        keep_d = tl.where(per > 0.0, tl.where(R < tl.load(sc_ptr + 13).to(R.dtype), 1.0, 0.0), 1.0).to(R.dtype)
    if MIXED:
        # the ladders in the working precision (Tinker9's mixed precision)
        R64 = R
        inv64 = inv
        zi64 = zi
        zk64 = zk
    else:
        # the ladders in float64 whatever the working precision, cast afterwards
        R64 = R.to(tl.float64)
        inv64 = 1.0 / R64
        zi64 = zi.to(tl.float64)
        zk64 = zk.to(tl.float64)
    dtype = rx.dtype
    zero = rx * 0.0
    if GROUP != 3:
        c0, c1, c2, c3, c4, c5 = _coulomb6(inv)
        # the deficits of the one-center ladders in the ladder precision (see the forward)
        ki0, ki1, ki2, ki3, ki4 = lad.core_valence_deficit4(c_ptr, R64, inv64, zk64, zi64)
        ik0, ik1, ik2, ik3, ik4 = lad.core_valence_deficit4(c_ptr, R64, inv64, zi64, zk64)
    if GROUP == 0 or GROUP == 1:
        ae = (alpha_e * per.to(tl.float64)).to(dtype)
        e0, e1, e2, e3, e4, e5 = _ewald6(R, inv, ae, inv_sqrt_pi.to(dtype))
        ov0, ov1, ov2, ov3, ov4, ov5 = lad.overlap_deficit5(c_ptr, R64, inv64, zi64, zk64)
    val_i = qi - core_i
    val_k = qk - core_k
    # the charges in the ladder precision, for the deficit sums
    ci = core_i.to(R64.dtype)
    ck = core_k.to(R64.dtype)
    vi64 = val_i.to(R64.dtype)
    vk64 = val_k.to(R64.dtype)
    # accumulators: the pair-vector gradient, the coefficient of r from the radial chains, and
    # the multipole gradients of both atoms
    grx = zero
    gry = zero
    grz = zero
    gsr = zero
    gqi = zero
    gqk = zero
    gdix = zero
    gdiy = zero
    gdiz = zero
    gdkx = zero
    gdky = zero
    gdkz = zero
    gQixx = zero
    gQixy = zero
    gQixz = zero
    gQiyx = zero
    gQiyy = zero
    gQiyz = zero
    gQizx = zero
    gQizy = zero
    gQizz = zero
    gQkxx = zero
    gQkxy = zero
    gQkxz = zero
    gQkyx = zero
    gQkyy = zero
    gQkyz = zero
    gQkzx = zero
    gQkzy = zero
    gQkzz = zero
    if GROUP == 0 or GROUP == 1:
        # permanent electrostatics: the forward's members (the total multipoles through the point
        # ladder minus the scaled deficit sums), its adjoint, and the chains to the charges
        # (through the deficit sums) and to R (every member to the next order)
        w_mp = g_mp * coulomb.to(dtype) * keep
        u0 = s1 * c0 + (e0 - c0) * per
        u1 = s1 * c1 + (e1 - c1) * per
        u2 = s1 * c2 + (e2 - c2) * per
        u3 = s1 * c3 + (e3 - c3) * per
        u4 = s1 * c4 + (e4 - c4) * per
        u5 = s1 * c5 + (e5 - c5) * per
        D1 = (vi64 * ck * ki1 + ci * vk64 * ik1 + vi64 * vk64 * ov1).to(dtype)
        Di0 = (ck * ki0 + vk64 * ov0).to(dtype)
        Dk0 = (ci * ik0 + vi64 * ov0).to(dtype)
        Di1 = (ck * ki1 + vk64 * ov1).to(dtype)
        Di2 = (ck * ki2 + vk64 * ov2).to(dtype)
        Di3 = (ck * ki3 + vk64 * ov3).to(dtype)
        Dk1 = (ci * ik1 + vi64 * ov1).to(dtype)
        Dk2 = (ci * ik2 + vi64 * ov2).to(dtype)
        Dk3 = (ci * ik3 + vi64 * ov3).to(dtype)
        A1 = qi * qk * u1 - s1 * D1
        Ai = qk * u0 - s1 * Di0
        Ak = qi * u0 - s1 * Dk0
        Li1 = qk * u1 - s1 * Di1
        Li2 = qk * u2 - s1 * Di2
        Li3 = qk * u3 - s1 * Di3
        Lk1 = qi * u1 - s1 * Dk1
        Lk2 = qi * u2 - s1 * Dk2
        Lk3 = qi * u3 - s1 * Dk3
        L1 = u1 - s1 * ov1.to(dtype)
        L2 = u2 - s1 * ov2.to(dtype)
        L3 = u3 - s1 * ov3.to(dtype)
        L4 = u4 - s1 * ov4.to(dtype)
        L5 = u5 - s1 * ov5.to(dtype)
        (a_dix, a_diy, a_diz, a_dkx, a_dky, a_dkz, a_qixx, a_qixy, a_qixz, a_qiyx, a_qiyy, a_qiyz, a_qizx, a_qizy,
         a_qizz, a_qkxx, a_qkxy, a_qkxz, a_qkyx, a_qkyy, a_qkyz, a_qkzx, a_qkzy, a_qkzz, a_rx, a_ry, a_rz,
         a_Li1, a_Li2, a_Lk1, a_Lk2, a_L1, a_L2, a_L3, a_L4) = _energy_adjoint(
            w_mp, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz,
            qkzz, rx, ry, rz, Li1, Li2, Lk1, Lk2, L1, L2, L3, L4)
        gqi += w_mp * Ai + a_Lk1 * L1 + a_Lk2 * L2
        gqk += w_mp * Ak + a_Li1 * L1 + a_Li2 * L2
        gdix += a_dix
        gdiy += a_diy
        gdiz += a_diz
        gdkx += a_dkx
        gdky += a_dky
        gdkz += a_dkz
        gQixx += a_qixx
        gQixy += a_qixy
        gQixz += a_qixz
        gQiyx += a_qiyx
        gQiyy += a_qiyy
        gQiyz += a_qiyz
        gQizx += a_qizx
        gQizy += a_qizy
        gQizz += a_qizz
        gQkxx += a_qkxx
        gQkxy += a_qkxy
        gQkxz += a_qkxz
        gQkyx += a_qkyx
        gQkyy += a_qkyy
        gQkyz += a_qkyz
        gQkzx += a_qkzx
        gQkzy += a_qkzy
        gQkzz += a_qkzz
        grx += a_rx
        gry += a_ry
        grz += a_rz
        gsr += (w_mp * A1 + a_Li1 * Li2 + a_Li2 * Li3 + a_Lk1 * Lk2 + a_Lk2 * Lk3
                + a_L1 * L2 + a_L2 * L3 + a_L3 * L4 + a_L4 * L5)
    if GROUP == 0 or GROUP == 3:
        # Pauli repulsion: K_i K_k E(valence multipoles; S^2 ladder) / R, switched
        rep_k_i = tl.load(pi + 2, mask=m, other=0.0)
        rep_k_k = tl.load(pk + 2, mask=m, other=0.0)
        rep_a_i = tl.load(pi + 3, mask=m, other=1.0)
        rep_a_k = tl.load(pk + 3, mask=m, other=2.0)
        rep_v_i = tl.load(pi + 4, mask=m, other=0.0)
        rep_v_k = tl.load(pk + 4, mask=m, other=0.0)
        sq0, sq1, sq2, sq3, sq4, sq5 = lad.squared_overlap5(c_ptr, R64, inv64, rep_a_i.to(R64.dtype),
                                                             rep_a_k.to(R64.dtype))
        sq0 = sq0.to(dtype)
        sq1 = sq1.to(dtype)
        sq2 = sq2.to(dtype)
        sq3 = sq3.to(dtype)
        sq4 = sq4.to(dtype)
        sq5 = sq5.to(dtype)
        if SW_REP > 0:
            sw_rep = _switch(R, tl.load(sc_ptr + 5).to(dtype), tl.load(sc_ptr + 6).to(dtype))
            dsw_rep = _switch_derivative(R, tl.load(sc_ptr + 5).to(dtype), tl.load(sc_ptr + 6).to(dtype))
            if SW_REP == 1:
                sw_rep = tl.where(per > 0.0, sw_rep, 1.0)
                dsw_rep = tl.where(per > 0.0, dsw_rep, 0.0)
        else:
            sw_rep = zero + 1.0
            dsw_rep = zero
        kk = rep_k_i * rep_k_k
        e_rep_raw = kk * _energy(rep_v_i * rep_v_k * sq0, -dix, -diy, -diz, -qixx, -qixy, -qixz, -qiyy, -qiyz, -qizz,
                                 -dkx, -dky, -dkz, -qkxx, -qkxy, -qkxz, -qkyy, -qkyz, -qkzz, rx, ry, rz,
                                 rep_v_k * sq1, rep_v_k * sq2, rep_v_i * sq1, rep_v_i * sq2, sq1, sq2, sq3, sq4) * inv
        w_rep = g_rep * s4 * sw_rep * kk * inv
        (a_dix, a_diy, a_diz, a_dkx, a_dky, a_dkz, a_qixx, a_qixy, a_qixz, a_qiyx, a_qiyy, a_qiyz, a_qizx, a_qizy,
         a_qizz, a_qkxx, a_qkxy, a_qkxz, a_qkyx, a_qkyy, a_qkyz, a_qkzx, a_qkzy, a_qkzz, a_rx, a_ry, a_rz,
         a_Li1, a_Li2, a_Lk1, a_Lk2, a_L1, a_L2, a_L3, a_L4) = _energy_adjoint(
            w_rep, -dix, -diy, -diz, -qixx, -qixy, -qixz, -qiyy, -qiyz, -qizz, -dkx, -dky, -dkz, -qkxx, -qkxy, -qkxz,
            -qkyy, -qkyz, -qkzz, rx, ry, rz, rep_v_k * sq1, rep_v_k * sq2, rep_v_i * sq1, rep_v_i * sq2,
            sq1, sq2, sq3, sq4)
        gdix -= a_dix
        gdiy -= a_diy
        gdiz -= a_diz
        gdkx -= a_dkx
        gdky -= a_dky
        gdkz -= a_dkz
        gQixx -= a_qixx
        gQixy -= a_qixy
        gQixz -= a_qixz
        gQiyx -= a_qiyx
        gQiyy -= a_qiyy
        gQiyz -= a_qiyz
        gQizx -= a_qizx
        gQizy -= a_qizy
        gQizz -= a_qizz
        gQkxx -= a_qkxx
        gQkxy -= a_qkxy
        gQkxz -= a_qkxz
        gQkyx -= a_qkyx
        gQkyy -= a_qkyy
        gQkyz -= a_qkyz
        gQkzx -= a_qkzx
        gQkzy -= a_qkzy
        gQkzz -= a_qkzz
        grx += a_rx
        gry += a_ry
        grz += a_rz
        gsr += (w_rep * rep_v_i * rep_v_k * sq1 + (a_Li1 * rep_v_k + a_Lk1 * rep_v_i) * sq2
                + (a_Li2 * rep_v_k + a_Lk2 * rep_v_i) * sq3 + a_L1 * sq2 + a_L2 * sq3 + a_L3 * sq4 + a_L4 * sq5)
        # the 1 / R factor and the switch: d(e_rep_raw sw)/dR times 1 / R
        gsr += g_rep * s4 * (-e_rep_raw * inv * sw_rep + e_rep_raw * dsw_rep) * inv
        # damped dispersion: a radial function only
        da_i = tl.load(pi + 5, mask=m, other=1.0)
        da_k = tl.load(pk + 5, mask=m, other=2.0)
        c6 = tl.load(pi + 6, mask=m, other=0.0) * tl.load(pk + 6, mask=m, other=0.0)
        dd0, dd1, dd2, dd3 = lad.overlap_deficit3(c_ptr, R64, inv64, da_i.to(R64.dtype), da_k.to(R64.dtype))
        # the damping scale minus one and its derivative from the deficits: lam3 = 1 + D1 R^3,
        # lam5 = 1 - D2 R^5 / 3 and D_n' = R D_(n+1)
        R2 = R64 * R64
        R3 = R2 * R64
        fm1 = (-0.5 * (dd2 * R3 * R2 + dd1 * R3)).to(dtype)
        dfdamp = (-0.5 * (dd3 * R3 * R3 + 6.0 * dd2 * R2 * R2 + 3.0 * dd1 * R2)).to(dtype)
        ad = (alpha_d * per.to(tl.float64)).to(dtype)
        xd = ad * R
        xd2 = xd * xd
        expd = libdevice.exp(-xd2)
        g = expd * (1.0 + xd2 + 0.5 * xd2 * xd2)
        dg = -xd2 * xd2 * xd * expd * ad
        inv2 = inv * inv
        inv6 = inv2 * inv2 * inv2
        bracket = s5 * fm1 * (fm1 + 2.0) + (s5 - 1.0) + g
        if SW_DISP > 0:
            sw_d = _switch(R, tl.load(sc_ptr + 9).to(dtype), tl.load(sc_ptr + 10).to(dtype))
            dsw_d = _switch_derivative(R, tl.load(sc_ptr + 9).to(dtype), tl.load(sc_ptr + 10).to(dtype))
            if SW_DISP == 1:
                sw_d = tl.where(per > 0.0, sw_d, 1.0)
                dsw_d = tl.where(per > 0.0, dsw_d, 0.0)
        else:
            sw_d = zero + 1.0
            dsw_d = zero
        de_disp = -c6 * keep_d * ((-6.0 * inv6 * inv) * bracket * sw_d
                                  + inv6 * (2.0 * s5 * (fm1 + 1.0) * dfdamp + dg) * sw_d
                                  + inv6 * bracket * dsw_d)
        gsr += g_disp * de_disp * inv
        # charge transfer: a radial function only
        eps_i = tl.load(pi + 7, mask=m, other=0.0)
        eps_k = tl.load(pk + 7, mask=m, other=0.0)
        eta_i = tl.load(pi + 8, mask=m, other=0.0)
        eta_k = tl.load(pk + 8, mask=m, other=0.0)
        t1 = tl.where(eta_k > 0.0, eps_i * libdevice.exp(-eta_k * R), zero)
        t2 = tl.where(eta_i > 0.0, eps_k * libdevice.exp(-eta_i * R), zero)
        if SW_CT > 0:
            sw_c = _switch(R, tl.load(sc_ptr + 7).to(dtype), tl.load(sc_ptr + 8).to(dtype))
            dsw_c = _switch_derivative(R, tl.load(sc_ptr + 7).to(dtype), tl.load(sc_ptr + 8).to(dtype))
            if SW_CT == 1:
                sw_c = tl.where(per > 0.0, sw_c, 1.0)
                dsw_c = tl.where(per > 0.0, dsw_c, 0.0)
        else:
            sw_c = zero + 1.0
            dsw_c = zero
        de_ct = (eta_k * t1 + eta_i * t2) * s1 * sw_c - (t1 + t2) * s1 * dsw_c
        gsr += g_ct * de_ct * inv
    if POLAR and (GROUP == 0 or GROUP == 2):
        gfdkx = tl.load(gfd_ptr + k * 3, mask=m, other=0.0)
        gfdky = tl.load(gfd_ptr + k * 3 + 1, mask=m, other=0.0)
        gfdkz = tl.load(gfd_ptr + k * 3 + 2, mask=m, other=0.0)
        gfdix = tl.load(gfd_ptr + i * 3, mask=m, other=0.0)
        gfdiy = tl.load(gfd_ptr + i * 3 + 1, mask=m, other=0.0)
        gfdiz = tl.load(gfd_ptr + i * 3 + 2, mask=m, other=0.0)
        gfpkx = tl.load(gfp_ptr + k * 3, mask=m, other=0.0)
        gfpky = tl.load(gfp_ptr + k * 3 + 1, mask=m, other=0.0)
        gfpkz = tl.load(gfp_ptr + k * 3 + 2, mask=m, other=0.0)
        gfpix = tl.load(gfp_ptr + i * 3, mask=m, other=0.0)
        gfpiy = tl.load(gfp_ptr + i * 3 + 1, mask=m, other=0.0)
        gfpiz = tl.load(gfp_ptr + i * 3 + 2, mask=m, other=0.0)
        sd = s6 * keep
        sp = s2 * keep
        # the damped fields enter the direct field with sd and the polar field with sp; the
        # Ewald difference of the total multipoles enters both unscaled
        gkx = gfdkx * sd + gfpkx * sp
        gky = gfdky * sd + gfpky * sp
        gkz = gfdkz * sd + gfpkz * sp
        gix = gfdix * sd + gfpix * sp
        giy = gfdiy * sd + gfpiy * sp
        giz = gfdiz * sd + gfpiz * sp
        gekx = gfdkx + gfpkx
        geky = gfdky + gfpky
        gekz = gfdkz + gfpkz
        geix = gfdix + gfpix
        geiy = gfdiy + gfpiy
        geiz = gfdiz + gfpiz
        # the damped one-center ladders of both sets and the deficits of their valence charges
        kd1 = c1 - ki1.to(dtype)
        kd2 = c2 - ki2.to(dtype)
        kd3 = c3 - ki3.to(dtype)
        kd4 = c4 - ki4.to(dtype)
        jd1 = c1 - ik1.to(dtype)
        jd2 = c2 - ik2.to(dtype)
        jd3 = c3 - ik3.to(dtype)
        jd4 = c4 - ik4.to(dtype)
        vki1 = (vi64 * ki1).to(dtype)
        vki2 = (vi64 * ki2).to(dtype)
        vik1 = (vk64 * ik1).to(dtype)
        vik2 = (vk64 * ik2).to(dtype)
        # the field at k of the set at i (the total charge through the Coulomb ladder minus the
        # valence's deficit, in one charge term) and the field at i of the set at k
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gkx, gky, gkz, qi * c1 - vki1, dix, diy, diz, qixx, qixy,
                                            qixz, qiyy, qiyz, qizz, rx, ry, rz, kd1, kd2, kd3, -1.0)
        gqi += b_C * kd1
        gdix += b_dx
        gdiy += b_dy
        gdiz += b_dz
        gQixx += b_qxx
        gQixy += b_qxy
        gQixz += b_qxz
        gQiyx += b_qyx
        gQiyy += b_qyy
        gQiyz += b_qyz
        gQizx += b_qzx
        gQizy += b_qzy
        gQizz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * (qi * c2 - vki2) + b_D1 * kd2 + b_D2 * kd3 + b_D3 * kd4
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gix, giy, giz, qk * c1 - vik1, dkx, dky, dkz, qkxx, qkxy,
                                            qkxz, qkyy, qkyz, qkzz, rx, ry, rz, jd1, jd2, jd3, 1.0)
        gqk += b_C * jd1
        gdkx += b_dx
        gdky += b_dy
        gdkz += b_dz
        gQkxx += b_qxx
        gQkxy += b_qxy
        gQkxz += b_qxz
        gQkyx += b_qyx
        gQkyy += b_qyy
        gQkyz += b_qyz
        gQkzx += b_qzx
        gQkzy += b_qzy
        gQkzz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * (qk * c2 - vik2) + b_D1 * jd2 + b_D2 * jd3 + b_D3 * jd4
        # the Ewald difference of the total multipoles, both ways
        ap = (alpha_p * per.to(tl.float64)).to(dtype)
        p0, p1, p2, p3, p4, p5 = _ewald6(R, inv, ap, inv_sqrt_pi.to(dtype))
        kp = keep * per
        v1 = (p1 - c1) * kp
        v2 = (p2 - c2) * kp
        v3 = (p3 - c3) * kp
        v4 = (p4 - c4) * kp
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gekx, geky, gekz, qi * v1, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz,
                                            qizz, rx, ry, rz, v1, v2, v3, -1.0)
        gqi += b_C * v1
        gdix += b_dx
        gdiy += b_dy
        gdiz += b_dz
        gQixx += b_qxx
        gQixy += b_qxy
        gQixz += b_qxz
        gQiyx += b_qyx
        gQiyy += b_qyy
        gQiyz += b_qyz
        gQizx += b_qzx
        gQizy += b_qzy
        gQizz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * qi * v2 + b_D1 * v2 + b_D2 * v3 + b_D3 * v4
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(geix, geiy, geiz, qk * v1, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz,
                                            qkzz, rx, ry, rz, v1, v2, v3, 1.0)
        gqk += b_C * v1
        gdkx += b_dx
        gdky += b_dy
        gdkz += b_dz
        gQkxx += b_qxx
        gQkxy += b_qxy
        gQkxz += b_qxz
        gQkyx += b_qyx
        gQkyy += b_qyy
        gQkyz += b_qyz
        gQkzx += b_qzx
        gQkzy += b_qzy
        gQkzz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * qk * v2 + b_D1 * v2 + b_D2 * v3 + b_D3 * v4
        # the mutual ladder members' derivatives: radial only, through the overlap deficits of the
        # penetration exponents (those of the electrostatics when the launch has them)
        if GROUP == 2:
            mu0, mu1, mu2, mu3 = lad.overlap_deficit3(c_ptr, R64, inv64, zi64, zk64)
            mu2 = mu2.to(dtype)
            mu3 = mu3.to(dtype)
        else:
            mu2 = ov2.to(dtype)
            mu3 = ov3.to(dtype)
        us = s7 * s3
        gdu1 = tl.load(gdu_ptr + e * 2, mask=m, other=0.0)
        gdu2 = tl.load(gdu_ptr + e * 2 + 1, mask=m, other=0.0)
        gsr += gdu1 * (p2 * per + c2 * (us - per) - mu2 * us) * keep
        gsr += gdu2 * (p3 * per + c3 * (us - per) - mu3 * us) * keep
    if POLAR and (EXPOL and (GROUP == 0 or GROUP == 3)):
        spring_i = tl.load(pi + 9, mask=m, other=0.0)
        spring_k = tl.load(pk + 9, mask=m, other=0.0)
        size_i = tl.load(pi + 10, mask=m, other=0.0)
        size_k = tl.load(pk + 10, mask=m, other=0.0)
        damp_i = tl.load(pi + 11, mask=m, other=0.0)
        damp_k = tl.load(pk + 11, mask=m, other=0.0)
        use = tl.maximum(tl.load(pi + 12, mask=m, other=0.0), tl.load(pk + 12, mask=m, other=0.0))
        inside = tl.where(R < tl.load(sc_ptr + 11).to(dtype), 1.0, 0.0)
        weight = s2 * use * inside * sw_rep
        dweight = s2 * use * inside * dsw_rep
        xr = 0.5 * tl.sqrt(damp_i * damp_k)
        x = xr * R
        poly = 1.0 + x + x * x / 3.0
        ex2 = libdevice.exp(-2.0 * x)
        base = size_i * size_k * poly * poly * ex2
        s2x = base * weight
        dbase = size_i * size_k * (-2.0 / 3.0) * x * (1.0 + x) * poly * ex2 * xr
        ds2x = dbase * weight + base * dweight
        ux = rx * inv
        uy = ry * inv
        uz = rz * inv
        # a = u^T G u for the incoming tensor gradients of both atoms, and (G + G^T) u
        gxx = tl.load(gxp_ptr + i * 9, mask=m, other=0.0)
        gxy = tl.load(gxp_ptr + i * 9 + 1, mask=m, other=0.0)
        gxz = tl.load(gxp_ptr + i * 9 + 2, mask=m, other=0.0)
        gyx = tl.load(gxp_ptr + i * 9 + 3, mask=m, other=0.0)
        gyy = tl.load(gxp_ptr + i * 9 + 4, mask=m, other=0.0)
        gyz = tl.load(gxp_ptr + i * 9 + 5, mask=m, other=0.0)
        gzx = tl.load(gxp_ptr + i * 9 + 6, mask=m, other=0.0)
        gzy = tl.load(gxp_ptr + i * 9 + 7, mask=m, other=0.0)
        gzz = tl.load(gxp_ptr + i * 9 + 8, mask=m, other=0.0)
        wi = spring_i * s2x
        ai = (ux * (gxx * ux + gxy * uy + gxz * uz) + uy * (gyx * ux + gyy * uy + gyz * uz)
              + uz * (gzx * ux + gzy * uy + gzz * uz))
        hx = wi * ((gxx + gxx) * ux + (gxy + gyx) * uy + (gxz + gzx) * uz)
        hy = wi * ((gyx + gxy) * ux + (gyy + gyy) * uy + (gyz + gzy) * uz)
        hz = wi * ((gzx + gxz) * ux + (gzy + gyz) * uy + (gzz + gzz) * uz)
        total = wi * ai
        dtotal = spring_i * ds2x * ai
        gxx = tl.load(gxp_ptr + k * 9, mask=m, other=0.0)
        gxy = tl.load(gxp_ptr + k * 9 + 1, mask=m, other=0.0)
        gxz = tl.load(gxp_ptr + k * 9 + 2, mask=m, other=0.0)
        gyx = tl.load(gxp_ptr + k * 9 + 3, mask=m, other=0.0)
        gyy = tl.load(gxp_ptr + k * 9 + 4, mask=m, other=0.0)
        gyz = tl.load(gxp_ptr + k * 9 + 5, mask=m, other=0.0)
        gzx = tl.load(gxp_ptr + k * 9 + 6, mask=m, other=0.0)
        gzy = tl.load(gxp_ptr + k * 9 + 7, mask=m, other=0.0)
        gzz = tl.load(gxp_ptr + k * 9 + 8, mask=m, other=0.0)
        wk = spring_k * s2x
        ak = (ux * (gxx * ux + gxy * uy + gxz * uz) + uy * (gyx * ux + gyy * uy + gyz * uz)
              + uz * (gzx * ux + gzy * uy + gzz * uz))
        hx += wk * ((gxx + gxx) * ux + (gxy + gyx) * uy + (gxz + gzx) * uz)
        hy += wk * ((gyx + gxy) * ux + (gyy + gyy) * uy + (gyz + gzy) * uz)
        hz += wk * ((gzx + gxz) * ux + (gzy + gyz) * uy + (gzz + gzz) * uz)
        total += wk * ak
        dtotal += spring_k * ds2x * ak
        # d(u^T G u)/dr = ((G + G^T) u - 2 (u^T G u) u) / R, and the radial weight's derivative
        grx += (hx - 2.0 * total * ux) * inv
        gry += (hy - 2.0 * total * uy) * inv
        grz += (hz - 2.0 * total * uz) * inv
        gsr += dtotal * inv
    if GROUP >= 2:
        # the groups launched after the first add to the stored pair-vector gradient
        grx += tl.load(gr_ptr + e * 3, mask=m, other=0.0)
        gry += tl.load(gr_ptr + e * 3 + 1, mask=m, other=0.0)
        grz += tl.load(gr_ptr + e * 3 + 2, mask=m, other=0.0)
    tl.store(gr_ptr + e * 3, grx + gsr * rx, mask=m)
    tl.store(gr_ptr + e * 3 + 1, gry + gsr * ry, mask=m)
    tl.store(gr_ptr + e * 3 + 2, grz + gsr * rz, mask=m)
    if GROUP != 3:
        _add_sorted(gq_ptr, i0, i1, i, (gqi).to(acc), same, tail, rest, TAIL=1)
        tl.atomic_add(gq_ptr + k, (gqk).to(acc), mask=m, sem="relaxed")
    _add_sorted(gd_ptr, i0 * 3, i1 * 3, i * 3, (gdix).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gd_ptr + 1, i0 * 3, i1 * 3, i * 3, (gdiy).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gd_ptr + 2, i0 * 3, i1 * 3, i * 3, (gdiz).to(acc), same, tail, rest, TAIL=1)
    tl.atomic_add(gd_ptr + k * 3, (gdkx).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gd_ptr + k * 3 + 1, (gdky).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gd_ptr + k * 3 + 2, (gdkz).to(acc), mask=m, sem="relaxed")
    _add_sorted(gQ_ptr, i0 * 9, i1 * 9, i * 9, (gQixx).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 1, i0 * 9, i1 * 9, i * 9, (gQixy).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 2, i0 * 9, i1 * 9, i * 9, (gQixz).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 3, i0 * 9, i1 * 9, i * 9, (gQiyx).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 4, i0 * 9, i1 * 9, i * 9, (gQiyy).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 5, i0 * 9, i1 * 9, i * 9, (gQiyz).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 6, i0 * 9, i1 * 9, i * 9, (gQizx).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 7, i0 * 9, i1 * 9, i * 9, (gQizy).to(acc), same, tail, rest, TAIL=1)
    _add_sorted(gQ_ptr + 8, i0 * 9, i1 * 9, i * 9, (gQizz).to(acc), same, tail, rest, TAIL=1)
    tl.atomic_add(gQ_ptr + k * 9, (gQkxx).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 1, (gQkxy).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 2, (gQkxz).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 3, (gQkyx).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 4, (gQkyy).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 5, (gQkyz).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 6, (gQkzx).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 7, (gQkzy).to(acc), mask=m, sem="relaxed")
    tl.atomic_add(gQ_ptr + k * 9 + 8, (gQkzz).to(acc), mask=m, sem="relaxed")


def pairs_backward(i_: Tensor, k_: Tensor, r_vec: Tensor, scales: Tensor, keep: Tensor, keep_d: Tensor, per: Tensor,
                   q: Tensor, d: Tensor, Q: Tensor, params: Tensor, scalars: Tensor, polar: bool, expol: bool,
                   switch_modes: tuple, g_node: Tensor, g_field_d=None, g_field_p=None, g_du=None,
                   g_expol=None, count=None, mixed: bool = False, select: Optional[Tensor] = None,
                   tangents: Optional[tuple] = None) -> dict:
    """Launch the backward kernel: the gradients of ``sum(g_node * e_node) + sum(g_field_d * field_d)
    + sum(g_field_p * field_p) + sum(g_du * du) + sum(g_expol * expol)`` with respect to the pair vectors ``(M, 3)``
    and the per-atom charges ``(N,)``, dipoles ``(N, 3)`` and quadrupoles ``(N, 3, 3)``.
    The inputs are those of :func:`pairs_forward` (``count``, ``mixed``, ``select`` and ``tangents``
    included: with tangents the gradients' derivatives along them come back as ``<key>_t``, the
    Hessian-vector products of a second derivative); a missing incoming gradient counts as zero."""
    M, N = int(i_.shape[0]), int(q.shape[0])
    dtype, device = r_vec.dtype, r_vec.device
    widths = {"grad_q": 1, "grad_d": 3, "grad_Q": 9}
    out, flat = _segments(N, widths, device)
    out["grad_r"] = torch.zeros(M, 3, dtype=dtype, device=device)

    keys = ("grad_r", "grad_q", "grad_d", "grad_Q")
    if tangents is not None:
        out.update({k + "_t": torch.zeros_like(out[k]) for k in keys})

    def finish(out):
        out.update(_cast_segments(flat, N, widths, dtype))
        for key in ("grad_q_t", "grad_d_t", "grad_Q_t"):
            if key in out:
                out[key] = out[key].to(dtype)
        for key in ("grad_Q", "grad_Q_t"):
            if key in out:
                out[key] = out[key].view(N, 3, 3)
        return out

    if M == 0:
        return finish(out)

    def given(t, shape):
        return torch.zeros(shape, dtype=dtype, device=device) if t is None else t.reshape(shape).contiguous()

    args = (i_, k_, r_vec.contiguous(), scales.contiguous(),
            out["grad_r"] if keep is None else keep.contiguous(),
            out["grad_r"] if keep_d is None else keep_d.contiguous(),
            per.contiguous(), q.contiguous(), d.contiguous(), Q.reshape(N, 9).contiguous(), params.contiguous(),
            constants(device), scalars, given(g_node, (N, 4)), given(g_field_d, (N, 3)), given(g_field_p, (N, 3)),
            given(g_du, (M, 2)), given(g_expol, (N, 9)), out["grad_r"], out["grad_q"], out["grad_d"], out["grad_Q"],
            _count(count, M, device), out["grad_r"] if select is None else select)
    kernel = _pairs_backward
    if tangents is not None:
        from . import amoeba_tangents as tg
        kernel = tg._pairs_backward_t
        r_t, q_t, d_t, Q_t = (t.contiguous() for t in tangents)
        args = args + (r_t, q_t, d_t, Q_t.reshape(N, 9).contiguous(), *[out[k + "_t"] for k in keys])
    groups = ((1, 2, 3) if polar else (1, 3)) if dtype in SPLIT_BACKWARD else (0,)
    for group in groups:
        kernel[(triton.cdiv(M, TILE),)](
            *args, SELECT=select is not None, KEEP_R=keep is None, POLAR=bool(polar), EXPOL=bool(polar and expol),
            SW_REP=int(switch_modes[0]),
            SW_CT=int(switch_modes[1]), SW_DISP=int(switch_modes[2]), NP=NP, GROUP=group, MIXED=bool(mixed),
            BLOCK=TILE, num_warps=WARPS, maxnreg=MAXNREG)
    return finish(out)


def pairs_param_grad(i_: Tensor, k_: Tensor, r_vec: Tensor, scales: Tensor, keep: Tensor, keep_d: Tensor, per: Tensor,
                     q: Tensor, d: Tensor, Q: Tensor, params: Tensor, scalars: Tensor, polar: bool, expol: bool,
                     switch_modes: tuple, g_node: Tensor, g_field_d=None, g_field_p=None, g_du=None, g_expol=None,
                     count=None, mixed: bool = False, select: Optional[Tensor] = None,
                     tangents: Optional[tuple] = None):
    """The gradient ``(N, NP)`` of ``sum(g_node * e_node) + sum(g_field_d * field_d) + ...`` (the outputs of
    :func:`pairs_forward`) with respect to the per-atom parameter table, from the generated adjoint of the
    forward kernel (:mod:`.amoeba_tangents`); the inputs are those of :func:`pairs_backward`. With
    ``tangents`` (of the pair vectors and multipoles) it returns the gradient and its derivative along them
    (the parameter part of a force loss's gradient)."""
    from . import amoeba_tangents as tg
    M, N = int(i_.shape[0]), int(q.shape[0])
    dtype, device = r_vec.dtype, r_vec.device
    grad = torch.zeros(N, NP, dtype=torch.float64, device=device)
    grad_t = None if tangents is None else torch.zeros_like(grad)
    if M:
        def given(t, shape):
            return torch.zeros(shape, dtype=dtype, device=device) if t is None else t.reshape(shape).contiguous()

        dummy = torch.zeros(1, dtype=dtype, device=device)
        kernel, extra = tg._pairs_forward_r, ()
        if tangents is not None:
            r_t, q_t, d_t, Q_t = (x.contiguous() for x in tangents)
            kernel, extra = tg._pairs_forward_r_t, (r_t, q_t, d_t, Q_t.reshape(N, 9).contiguous(), grad_t)
        for group in ((1, 2) if dtype in SPLIT_DTYPES else (0,)):
            kernel[(triton.cdiv(M, TILE),)](
                i_, k_, r_vec.contiguous(), scales.contiguous(), dummy if keep is None else keep.contiguous(),
                dummy if keep_d is None else keep_d.contiguous(), per.contiguous(), q.contiguous(), d.contiguous(),
                Q.reshape(N, 9).contiguous(), params.contiguous(), constants(device), scalars, dummy, dummy, dummy,
                dummy, dummy, dummy, _count(count, M, device), dummy if select is None else select,
                given(g_node, (N, 4)), given(g_field_d, (N, 3)), given(g_field_p, (N, 3)), given(g_du, (M, 2)),
                given(g_expol, (N, 9)), grad, *extra, SELECT=select is not None, KEEP_R=keep is None, POLAR=bool(polar),
                EXPOL=bool(polar and expol), WRITE_PAIRS=False, SW_REP=int(switch_modes[0]),
                SW_CT=int(switch_modes[1]), SW_DISP=int(switch_modes[2]), NP=NP, GROUP=group, MIXED=bool(mixed),
                BLOCK=TILE, num_warps=WARPS, maxnreg=MAXNREG)
    return grad.to(dtype) if tangents is None else (grad.to(dtype), grad_t.to(dtype))


def amoeba_param_grad(i_: Tensor, k_: Tensor, r_vec: Tensor, scales: Tensor, keep: Optional[Tensor], per: Tensor,
                      q: Tensor, d: Tensor, Q: Tensor, params: Tensor, offsets: Tensor, scalars: Tensor, polar: bool,
                      modes: tuple, g_node: Tensor, g_field_d=None, g_field_p=None, g_du=None, count=None,
                      select: Optional[Tensor] = None, tangents: Optional[tuple] = None):
    """The gradient ``(N, AMOEBA_NP)`` of the AMOEBA pair outputs' weighted sum with respect to the per-atom
    table (:func:`pairs_param_grad` for :func:`amoeba_pairs_forward`)."""
    from . import amoeba_tangents as tg
    M, N = int(i_.shape[0]), int(q.shape[0])
    dtype, device = r_vec.dtype, r_vec.device
    grad = torch.zeros(N, AMOEBA_NP, dtype=torch.float64, device=device)
    grad_t = None if tangents is None else torch.zeros_like(grad)
    if M:
        def given(t, shape):
            return torch.zeros(shape, dtype=dtype, device=device) if t is None else t.reshape(shape).contiguous()

        dummy = torch.zeros(1, dtype=dtype, device=device)
        kernel, extra = tg._amoeba_forward_r, ()
        if tangents is not None:
            r_t, q_t, d_t, Q_t, off_t = (x.contiguous() for x in tangents)
            kernel, extra = tg._amoeba_forward_r_t, (r_t, q_t, d_t, Q_t.reshape(N, 9).contiguous(), off_t, grad_t)
        for group in ((1, 2) if dtype in SPLIT_DTYPES else (0,)):
            kernel[(triton.cdiv(M, TILE),)](
                i_, k_, r_vec.contiguous(), scales.contiguous(), dummy if keep is None else keep.contiguous(),
                per.contiguous(), q.contiguous(), d.contiguous(), Q.reshape(N, 9).contiguous(), params.contiguous(),
                offsets.contiguous(), scalars, dummy, dummy, dummy, dummy, dummy, _count(count, M, device),
                dummy if select is None else select, given(g_node, (N, 2)), given(g_field_d, (N, 3)),
                given(g_field_p, (N, 3)), given(g_du, (M, 2)), grad, *extra, SELECT=select is not None,
                KEEP_R=keep is None, POLAR=bool(polar), WRITE_PAIRS=False, SW_VDW=int(modes[0]),
                RADIUS=int(modes[1]), EPSILON=int(modes[2]), NP=AMOEBA_NP, GROUP=group, BLOCK=TILE,
                num_warps=WARPS, maxnreg=MAXNREG)
    return grad.to(dtype) if tangents is None else (grad.to(dtype), grad_t.to(dtype))


@triton.jit
def _mutual_backward(i_ptr, k_ptr, r_ptr, du_ptr, mu_ptr, lam_ptr, gr_ptr, gdu_ptr, n_ptr, sel_ptr,
                     SELECT: tl.constexpr, BLOCK: tl.constexpr):
    """The gradients of ``lam . T mu`` summed over both directions of every pair: with respect to
    the pair vector, ``D2 [(r . mu) lam + (r . lam) mu]``, and to the ladder members,
    ``lam . mu`` and ``(r . mu)(r . lam)`` (the products of the implicit backward). ``SELECT`` as in
    the pair kernels."""
    start = tl.program_id(0) * BLOCK
    n = tl.load(n_ptr)
    if start >= n:
        return
    lane = (start + tl.arange(0, BLOCK)).to(tl.int64)
    m = lane < n
    if SELECT:
        e = tl.load(sel_ptr + lane, mask=m, other=0)
    else:
        e = lane
    i = tl.load(i_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0)
    rx = tl.load(r_ptr + e * 3, mask=m, other=0.0)
    ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    D2 = tl.load(du_ptr + e * 2 + 1, mask=m, other=0.0)
    mix = tl.load(mu_ptr + i * 3, mask=m, other=0.0)
    miy = tl.load(mu_ptr + i * 3 + 1, mask=m, other=0.0)
    miz = tl.load(mu_ptr + i * 3 + 2, mask=m, other=0.0)
    mkx = tl.load(mu_ptr + k * 3, mask=m, other=0.0)
    mky = tl.load(mu_ptr + k * 3 + 1, mask=m, other=0.0)
    mkz = tl.load(mu_ptr + k * 3 + 2, mask=m, other=0.0)
    lix = tl.load(lam_ptr + i * 3, mask=m, other=0.0)
    liy = tl.load(lam_ptr + i * 3 + 1, mask=m, other=0.0)
    liz = tl.load(lam_ptr + i * 3 + 2, mask=m, other=0.0)
    lkx = tl.load(lam_ptr + k * 3, mask=m, other=0.0)
    lky = tl.load(lam_ptr + k * 3 + 1, mask=m, other=0.0)
    lkz = tl.load(lam_ptr + k * 3 + 2, mask=m, other=0.0)
    # direction one: the dipole of i felt by k (adjoint at k); direction two the other way
    rmi = rx * mix + ry * miy + rz * miz
    rlk = rx * lkx + ry * lky + rz * lkz
    rmk = rx * mkx + ry * mky + rz * mkz
    rli = rx * lix + ry * liy + rz * liz
    tl.store(gr_ptr + e * 3, D2 * (rmi * lkx + rlk * mix + rmk * lix + rli * mkx), mask=m)
    tl.store(gr_ptr + e * 3 + 1, D2 * (rmi * lky + rlk * miy + rmk * liy + rli * mky), mask=m)
    tl.store(gr_ptr + e * 3 + 2, D2 * (rmi * lkz + rlk * miz + rmk * liz + rli * mkz), mask=m)
    tl.store(gdu_ptr + e * 2, lkx * mix + lky * miy + lkz * miz + lix * mkx + liy * mky + liz * mkz, mask=m)
    tl.store(gdu_ptr + e * 2 + 1, rmi * rlk + rmk * rli, mask=m)


@triton.jit
def _pair_vectors(pos_ptr, i_ptr, k_ptr, s_ptr, cell_ptr, out_ptr, n_pairs, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    """``out = pos[k] - pos[i] (+ shift @ cell)`` in the precision of ``pos`` (float64 geometry), stored
    in the precision of ``out`` (the model's): the gathers, the image shifts' product with the one cell
    and the cast in one pass, for every pair of the list (the padding repeats valid pairs)."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_pairs
    i = tl.load(i_ptr + e, mask=m, other=0).to(tl.int64)
    k = tl.load(k_ptr + e, mask=m, other=0).to(tl.int64)
    vx = tl.load(pos_ptr + k * 3, mask=m, other=0.0) - tl.load(pos_ptr + i * 3, mask=m, other=0.0)
    vy = tl.load(pos_ptr + k * 3 + 1, mask=m, other=0.0) - tl.load(pos_ptr + i * 3 + 1, mask=m, other=0.0)
    vz = tl.load(pos_ptr + k * 3 + 2, mask=m, other=0.0) - tl.load(pos_ptr + i * 3 + 2, mask=m, other=0.0)
    if SHIFT:
        s0 = tl.load(s_ptr + e * 3, mask=m, other=0.0).to(vx.dtype)
        s1 = tl.load(s_ptr + e * 3 + 1, mask=m, other=0.0).to(vx.dtype)
        s2 = tl.load(s_ptr + e * 3 + 2, mask=m, other=0.0).to(vx.dtype)
        vx += s0 * tl.load(cell_ptr) + s1 * tl.load(cell_ptr + 3) + s2 * tl.load(cell_ptr + 6)
        vy += s0 * tl.load(cell_ptr + 1) + s1 * tl.load(cell_ptr + 4) + s2 * tl.load(cell_ptr + 7)
        vz += s0 * tl.load(cell_ptr + 2) + s1 * tl.load(cell_ptr + 5) + s2 * tl.load(cell_ptr + 8)
    out = out_ptr.dtype.element_ty
    tl.store(out_ptr + e * 3, vx.to(out), mask=m)
    tl.store(out_ptr + e * 3 + 1, vy.to(out), mask=m)
    tl.store(out_ptr + e * 3 + 2, vz.to(out), mask=m)


@triton.jit
def _pair_scatter(g_ptr, i_ptr, k_ptr, out_ptr, n_ptr, BLOCK: tl.constexpr):
    """``out[k] += g`` and ``out[i] -= g`` over the pairs: the backward of ``pos[k] - pos[i]``, the
    i side (sorted) with one add per block and atom."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < tl.load(n_ptr)
    i = tl.load(i_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0)
    i0 = tl.load(i_ptr + tl.program_id(0) * BLOCK)
    i1 = tl.load(i_ptr + tl.minimum(tl.program_id(0) * BLOCK + BLOCK - 1, tl.maximum(tl.load(n_ptr) - 1, 0)))
    same = m & (i == i0)
    tail = m & (i == i1) & (i1 != i0)
    rest = m & (i != i0) & (i != i1)
    acc = out_ptr.dtype.element_ty
    gx = tl.load(g_ptr + e * 3, mask=m, other=0.0).to(acc)
    gy = tl.load(g_ptr + e * 3 + 1, mask=m, other=0.0).to(acc)
    gz = tl.load(g_ptr + e * 3 + 2, mask=m, other=0.0).to(acc)
    tl.atomic_add(out_ptr + k * 3, gx, mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + k * 3 + 1, gy, mask=m, sem="relaxed")
    tl.atomic_add(out_ptr + k * 3 + 2, gz, mask=m, sem="relaxed")
    _add_sorted(out_ptr, i0 * 3, i1 * 3, i * 3, -gx, same, tail, rest, TAIL=1)
    _add_sorted(out_ptr + 1, i0 * 3, i1 * 3, i * 3, -gy, same, tail, rest, TAIL=1)
    _add_sorted(out_ptr + 2, i0 * 3, i1 * 3, i * 3, -gz, same, tail, rest, TAIL=1)


@triton.jit
def _rows_scatter(ptr_ptr, pair_ptr, other_ptr, i_ptr, g_ptr, out_ptr, n_ptr, CHUNK: tl.constexpr):
    """:func:`_pair_scatter` on compressed rows (:func:`pair_rows`): one program per atom ``a`` sums
    ``g`` of the pairs of its row, ``+g`` where it is the second atom and ``-g`` where it is the first,
    and stores ``out[a]`` once (no atomic adds, no zeroed buffer). A pair of an atom with its own image
    lies twice in the row and adds nothing. Pairs at or beyond the count ``n`` are skipped."""
    a = tl.program_id(0).to(tl.int64)
    start = tl.load(ptr_ptr + a)
    end = tl.load(ptr_ptr + a + 1)
    count = tl.load(n_ptr)
    acc = out_ptr.dtype.element_ty
    sx = tl.zeros((CHUNK,), dtype=acc)
    sy = tl.zeros((CHUNK,), dtype=acc)
    sz = tl.zeros((CHUNK,), dtype=acc)
    j = start
    # a while loop: its bound comes from memory (Triton's interpreter takes no tensor range)
    while j < end:
        o = j + tl.arange(0, CHUNK)
        m = o < end
        e = tl.load(pair_ptr + o, mask=m, other=0)
        m = m & (e < count)
        e = e.to(tl.int64)
        b = tl.load(other_ptr + o, mask=m, other=0).to(tl.int64)
        first = tl.load(i_ptr + e, mask=m, other=0) == a
        sign = tl.where(b == a, 0.0, tl.where(first, -1.0, 1.0))
        sx += sign * tl.load(g_ptr + e * 3, mask=m, other=0.0).to(acc)
        sy += sign * tl.load(g_ptr + e * 3 + 1, mask=m, other=0.0).to(acc)
        sz += sign * tl.load(g_ptr + e * 3 + 2, mask=m, other=0.0).to(acc)
        j += CHUNK
    tl.store(out_ptr + a * 3, tl.sum(sx, 0))
    tl.store(out_ptr + a * 3 + 1, tl.sum(sy, 0))
    tl.store(out_ptr + a * 3 + 2, tl.sum(sz, 0))


class PairDifference(torch.autograd.Function):
    """``pos[k] - pos[i]`` ``(M, 3)`` of the pairs, plus ``shift @ cell`` of their image shifts
    ``(M, 3)`` with the one cell ``(3, 3)`` when given, computed in the precision of ``pos`` and
    stored in ``dtype`` (the model's) by one kernel, with the gradient scattered back to the atoms
    by one kernel (``index_add`` serializes the adds of a list sorted by atom), over the compressed
    ``rows`` of the pairs (:func:`pair_rows`) when given. Differentiable once, in the positions only;
    ``count`` as in :func:`pairs_forward`."""

    @staticmethod
    def forward(ctx, pos, i_, k_, count=None, shift=None, cell=None, dtype=None, rows=None):
        ctx.save_for_backward(i_, k_)
        ctx.count, ctx.n, ctx.dtype, ctx.rows = count, int(pos.shape[0]), pos.dtype, rows
        M = int(i_.shape[0])
        out = torch.empty(M, 3, dtype=dtype or pos.dtype, device=pos.device)
        if M:
            pos = pos.contiguous()
            _pair_vectors[(triton.cdiv(M, TILE),)](pos, i_, k_, pos if shift is None else shift.contiguous(),
                                                   pos if cell is None else cell.contiguous(), out, M,
                                                   SHIFT=shift is not None, BLOCK=TILE, num_warps=WARPS)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        i_, k_ = ctx.saved_tensors
        M = int(i_.shape[0])
        if ctx.rows is not None and M:
            ptr, pair, other = ctx.rows
            out = torch.empty(ctx.n, 3, dtype=ctx.dtype, device=g.device)
            _rows_scatter[(ctx.n,)](ptr, pair, other, i_, g.contiguous(), out, _count(ctx.count, M, g.device),
                                    CHUNK=ROWS_CHUNK, num_warps=ROWS_WARPS)
            return out, None, None, None, None, None, None, None
        out = torch.zeros(ctx.n, 3, dtype=ctx.dtype, device=g.device)
        if M:
            _pair_scatter[(triton.cdiv(M, TILE),)](g.contiguous(), i_, k_, out, _count(ctx.count, M, g.device),
                                                   BLOCK=TILE, num_warps=WARPS)
        return out, None, None, None, None, None, None, None


@triton.jit
def _mutual_rows(ptr_ptr, pair_ptr, other_ptr, r_ptr, du_ptr, mu_ptr, f_ptr, n_ptr, guard_ptr, n_atoms,
                 GUARD: tl.constexpr, PAIRED: tl.constexpr, CHUNK: tl.constexpr):
    """The real-space field of the dipoles ``mu`` through the mutual ladders on compressed rows: one
    program per atom ``a`` walks the pairs of its row (``pair`` the pair, ``other`` the atom at its
    other end) and sums ``D1 mu[b] + D2 (r . mu[b]) r`` in registers, then stores ``f[a]`` once: no
    atomic adds, no zeroed buffer. Pairs at or beyond the count ``n`` (a plan's padding) are skipped.
    With ``GUARD`` the kernel does nothing once the conjugate-gradient scalars ``guard`` (a row of
    :func:`pcg_scalars`) say the solve has converged. With ``PAIRED`` the dipoles are two sets
    stacked ``(2 n_atoms, 3)`` (the paired solve) and both fields are formed from one pass over the
    row's pairs."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    a = tl.program_id(0).to(tl.int64)
    start = tl.load(ptr_ptr + a)
    end = tl.load(ptr_ptr + a + 1)
    count = tl.load(n_ptr)
    acc = f_ptr.dtype.element_ty
    sx = tl.zeros((CHUNK,), dtype=acc)
    sy = tl.zeros((CHUNK,), dtype=acc)
    sz = tl.zeros((CHUNK,), dtype=acc)
    if PAIRED:
        px = tl.zeros((CHUNK,), dtype=acc)
        py = tl.zeros((CHUNK,), dtype=acc)
        pz = tl.zeros((CHUNK,), dtype=acc)
    j = start
    # a while loop: its bound comes from memory (Triton's interpreter takes no tensor range)
    while j < end:
        o = j + tl.arange(0, CHUNK)
        m = o < end
        e = tl.load(pair_ptr + o, mask=m, other=0)
        m = m & (e < count)
        e = e.to(tl.int64)
        b = tl.load(other_ptr + o, mask=m, other=0).to(tl.int64)
        rx = tl.load(r_ptr + e * 3, mask=m, other=0.0)
        ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
        rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
        D1 = tl.load(du_ptr + e * 2, mask=m, other=0.0)
        D2 = tl.load(du_ptr + e * 2 + 1, mask=m, other=0.0)
        mbx = tl.load(mu_ptr + b * 3, mask=m, other=0.0)
        mby = tl.load(mu_ptr + b * 3 + 1, mask=m, other=0.0)
        mbz = tl.load(mu_ptr + b * 3 + 2, mask=m, other=0.0)
        rm = D2 * (rx * mbx + ry * mby + rz * mbz)
        sx += (D1 * mbx + rm * rx).to(acc)
        sy += (D1 * mby + rm * ry).to(acc)
        sz += (D1 * mbz + rm * rz).to(acc)
        if PAIRED:
            nbx = tl.load(mu_ptr + (b + n_atoms) * 3, mask=m, other=0.0)
            nby = tl.load(mu_ptr + (b + n_atoms) * 3 + 1, mask=m, other=0.0)
            nbz = tl.load(mu_ptr + (b + n_atoms) * 3 + 2, mask=m, other=0.0)
            rn = D2 * (rx * nbx + ry * nby + rz * nbz)
            px += (D1 * nbx + rn * rx).to(acc)
            py += (D1 * nby + rn * ry).to(acc)
            pz += (D1 * nbz + rn * rz).to(acc)
        j += CHUNK
    tl.store(f_ptr + a * 3, tl.sum(sx, 0))
    tl.store(f_ptr + a * 3 + 1, tl.sum(sy, 0))
    tl.store(f_ptr + a * 3 + 2, tl.sum(sz, 0))
    if PAIRED:
        tl.store(f_ptr + (a + n_atoms) * 3, tl.sum(px, 0))
        tl.store(f_ptr + (a + n_atoms) * 3 + 1, tl.sum(py, 0))
        tl.store(f_ptr + (a + n_atoms) * 3 + 2, tl.sum(pz, 0))


def pair_rows(i_: Tensor, k_: Tensor, N: int, valid=None, sorted_i: bool = False) -> tuple:
    """The compressed rows of the unique pairs ``(i, k)`` over the ``N`` atoms, for
    :func:`mutual_field`: every pair appears in the row of each of its atoms. The pairs from
    ``valid`` on (a plan's padding) go to an extra row ``N`` that no program walks: in the rows
    of their atoms they would make one program walk the whole padding. Returns the row
    pointers ``(N + 2,)``, the pair of every entry and the atom at its other end ``(2 M,)``,
    all int32. ``valid`` may be a count on the device (no host read). With ``sorted_i`` (a plan's pairs,
    sorted by their first atom) the first atoms' entries are in place already and only the second atoms'
    are sorted: the same rows, from a sort of half the entries."""
    M = int(i_.shape[0])
    if sorted_i:
        return _sorted_pair_rows(i_, k_, N, valid)
    centers = torch.cat([i_, k_])
    if valid is not None and (torch.is_tensor(valid) or valid < M):
        pad = (torch.arange(M, device=i_.device) >= valid).repeat(2)
        centers = torch.where(pad, torch.full_like(centers, N), centers)
    order, ptr = compressed(centers, N + 1)
    pair = torch.arange(M, dtype=torch.int32, device=i_.device).repeat(2)[order]
    other = torch.cat([k_, i_])[order].to(torch.int32)
    return ptr.to(torch.int32), pair, other


def _sorted_pair_rows(i_: Tensor, k_: Tensor, N: int, valid=None) -> tuple:
    """:func:`pair_rows` for pairs sorted by ``i``: a row holds its atom's pairs as first atom (in pair
    order, as they lie) and then as second atom (in pair order, a stable sort of ``k``), the order the
    stable sort of both directions gives."""
    device = i_.device
    M = int(i_.shape[0])
    at = torch.arange(M, device=device)
    ci, ck = i_, k_
    if valid is not None:
        pad = at >= valid
        ci = torch.where(pad, torch.full_like(i_, N), i_)
        ck = torch.where(pad, torch.full_like(k_, N), k_)
    count_i = torch.bincount(ci, minlength=N + 1)
    count_k = torch.bincount(ck, minlength=N + 1)
    ptr = torch.zeros(N + 2, dtype=torch.int64, device=device)
    ptr[1:] = torch.cumsum(count_i + count_k, 0)
    first_i = torch.cumsum(count_i, 0) - count_i
    first_k = torch.cumsum(count_k, 0) - count_k
    by_k = torch.argsort(narrow_keys(ck, N + 1), stable=True)
    ck_sorted = ck.index_select(0, by_k)
    put_i = ptr.index_select(0, ci) + at - first_i.index_select(0, ci)
    put_k = (ptr.index_select(0, ck_sorted) + count_i.index_select(0, ck_sorted) + at
             - first_k.index_select(0, ck_sorted))
    pair = torch.empty(2 * M, dtype=torch.int32, device=device)
    other = torch.empty(2 * M, dtype=torch.int32, device=device)
    pair.index_copy_(0, put_i, at.to(torch.int32))
    other.index_copy_(0, put_i, k_.to(torch.int32))
    pair.index_copy_(0, put_k, by_k.to(torch.int32))
    other.index_copy_(0, put_k, i_.index_select(0, by_k).to(torch.int32))
    return ptr.to(torch.int32), pair, other


def mutual_field(mu: Tensor, r_vec: Tensor, du: Tensor, i_: Tensor, k_: Tensor, N: int, count=None,
                 guard: Optional[Tensor] = None, rows: Optional[tuple] = None) -> Tensor:
    """The real-space field ``(N, 3)`` of dipoles ``mu`` ``(N, 3)`` through the mutual ladders
    ``du`` ``(M, 2)`` of the pairs (``mutual_field_real`` of the eager path), summed per atom over
    the compressed rows ``rows`` of :func:`pair_rows` (built here when not given); ``count`` as in
    :func:`pairs_forward`. With the scalars ``guard`` of a conjugate-gradient iteration the kernel
    does nothing once the solve has converged, and the field is left unset (no update reads it).
    Two sets of dipoles stacked ``(2 N, 3)`` give their two fields stacked, from one launch."""
    ptr, pair, other = pair_rows(i_, k_, N) if rows is None else rows
    paired = int(mu.shape[0]) == 2 * N
    f = torch.empty(mu.shape[0], 3, dtype=mu.dtype, device=mu.device)
    if N:
        M = int(r_vec.shape[0])
        _mutual_rows[(N,)](ptr, pair, other, r_vec.contiguous(), du.contiguous(), mu.contiguous(), f,
                           _count(count, M, mu.device), f if guard is None else guard, N, GUARD=guard is not None,
                           PAIRED=paired, CHUNK=ROWS_CHUNK, num_warps=ROWS_WARPS)
    return f


def mutual_backward(mu: Tensor, lam: Tensor, r_vec: Tensor, du: Tensor, i_: Tensor, k_: Tensor,
                    count=None, select: Optional[Tensor] = None) -> tuple:
    """``grad_r`` ``(M, 3)`` and ``grad_du`` ``(M, 2)`` of ``sum(lam * (T mu))`` over the pairs (the
    products of the implicit backward of the induction); ``count`` and ``select`` as in
    :func:`pairs_forward` (the pairs not visited have zero ladders and get zero gradients)."""
    M = int(i_.shape[0])
    gr = torch.zeros(M, 3, dtype=mu.dtype, device=mu.device)
    gdu = torch.zeros(M, 2, dtype=mu.dtype, device=mu.device)
    if M:
        _mutual_backward[(triton.cdiv(M, TILE),)](i_, k_, r_vec.contiguous(), du.contiguous(), mu.contiguous(),
                                                  lam.contiguous(), gr, gdu, _count(count, M, mu.device),
                                                  gr if select is None else select, SELECT=select is not None,
                                                  BLOCK=TILE, num_warps=WARPS)
    return gr, gdu


def select_pairs(active: Tensor) -> tuple:
    """The pairs the kernels visit: the indices of the ``active`` ones ``(M,)`` in increasing
    order, followed by zeros up to ``M + 1`` entries, and their number as an int32 tensor
    ``(1,)``, both on the device without a host read (a CUDA graph records them)."""
    M = int(active.shape[0])
    rank = torch.cumsum(active, 0)
    sel = torch.zeros(M + 1, dtype=torch.int64, device=active.device)
    # the inactive pairs all go to the last entry, past the ones a kernel reads
    sel.scatter_(0, torch.where(active, rank - 1, M), torch.arange(M, device=active.device))
    return sel, rank[-1:].to(torch.int32)


@triton.jit
def _pair_inside(r_ptr, valid_ptr, cut_ptr, R_ptr, flag_ptr, n_pairs, tiny, VALID: tl.constexpr,
                 BLOCK: tl.constexpr):
    """The distances ``sqrt(r . r + tiny)`` of the pair vectors and the flags of the pairs within the
    cutoff (``r . r`` below the squared cutoff ``cut``, in the precision of ``r``) and valid."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_pairs
    x = tl.load(r_ptr + e * 3, mask=m, other=0.0)
    y = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    z = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    r2 = x * x + y * y + z * z
    tl.store(R_ptr + e, tl.sqrt(r2 + tiny), mask=m)
    inside = r2 < tl.load(cut_ptr)
    if VALID:
        inside = inside & (tl.load(valid_ptr + e, mask=m, other=0.0) > 0)
    tl.store(flag_ptr + e, inside.to(tl.int32), mask=m)


@triton.jit
def _pair_compact(flag_ptr, rank_ptr, sel_ptr, n_ptr, n_pairs, BLOCK: tl.constexpr):
    """``sel[rank - 1] = e`` for the flagged pairs ``e`` (``rank`` the inclusive count of the flags),
    and their number."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_pairs
    flag = tl.load(flag_ptr + e, mask=m, other=0)
    rank = tl.load(rank_ptr + e, mask=m, other=0).to(tl.int64)
    tl.store(sel_ptr + rank - 1, e, mask=m & (flag != 0))
    if tl.program_id(0) == 0:
        tl.store(n_ptr, tl.load(rank_ptr + n_pairs - 1))


def select_pairs_within(r: Tensor, cutoff_sq: Tensor, valid: Optional[Tensor] = None) -> tuple:
    """The distances ``(M,)`` of the pair vectors ``r`` ``(M, 3)`` (no gradient) and
    :func:`select_pairs` of the pairs within the cutoff (``r . r`` below ``cutoff_sq``, a tensor of
    one entry in the precision of ``r``) and valid (``valid > 0``): two kernels and one scan."""
    M = int(r.shape[0])
    R = torch.empty(M, dtype=r.dtype, device=r.device)
    flag = torch.empty(M, dtype=torch.int32, device=r.device)
    sel = torch.zeros(M + 1, dtype=torch.int64, device=r.device)
    count = torch.zeros(1, dtype=torch.int32, device=r.device)
    if M:
        grid = (triton.cdiv(M, 256),)
        r = r.detach().contiguous()
        _pair_inside[grid](r, r if valid is None else valid.contiguous(), cutoff_sq, R, flag, M, TINY,
                           VALID=valid is not None, BLOCK=256, num_warps=4)
        rank = torch.cumsum(flag, 0, dtype=torch.int32)
        _pair_compact[grid](flag, rank, sel, count, M, BLOCK=256, num_warps=4)
    return R, sel, count


@triton.jit
def _plan_checks(pos_ptr, ref_ptr, last_ptr, conv_ptr, need_ptr, out_ptr, n_atoms, n_need, half_bits, half_sq_bits,
                 steps1, BLOCK: tl.constexpr):
    """The rebuild tests of a step plan in one pass over the atoms (float64): ``out[0]`` whether an
    atom moved by more than half the skin since the pairs were found, ``out[1]`` whether it will
    after ``steps1 - 1`` more steps like the one from ``last``; ``last`` then takes the positions.
    The first program copies the solves' convergence and needed iterations to ``out[2:]``. The
    thresholds come as the bits of float64 numbers; ``out[:2]`` must be zero before."""
    e = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = e < n_atoms
    # the bits as int64 (a zero comes as int32)
    half = half_bits.to(tl.int64).to(tl.float64, bitcast=True)
    half_sq = half_sq_bits.to(tl.int64).to(tl.float64, bitcast=True)
    px = tl.load(pos_ptr + e * 3, mask=m, other=0.0)
    py = tl.load(pos_ptr + e * 3 + 1, mask=m, other=0.0)
    pz = tl.load(pos_ptr + e * 3 + 2, mask=m, other=0.0)
    lx = tl.load(last_ptr + e * 3, mask=m, other=0.0).to(tl.float64)
    ly = tl.load(last_ptr + e * 3 + 1, mask=m, other=0.0).to(tl.float64)
    lz = tl.load(last_ptr + e * 3 + 2, mask=m, other=0.0).to(tl.float64)
    dx = px.to(tl.float64) - tl.load(ref_ptr + e * 3, mask=m, other=0.0)
    dy = py.to(tl.float64) - tl.load(ref_ptr + e * 3 + 1, mask=m, other=0.0)
    dz = pz.to(tl.float64) - tl.load(ref_ptr + e * 3 + 2, mask=m, other=0.0)
    d2 = dx * dx + dy * dy + dz * dz
    sx = px.to(tl.float64) - lx
    sy = py.to(tl.float64) - ly
    sz = pz.to(tl.float64) - lz
    ahead = tl.sqrt(d2) + steps1.to(tl.float64) * tl.sqrt(sx * sx + sy * sy + sz * sz)
    outdated = tl.max(tl.where(m & (d2 > half_sq), 1, 0), 0)
    due = tl.max(tl.where(m & (ahead > half), 1, 0), 0)
    # every program that finds an atom stores the same 1
    if outdated > 0:
        tl.store(out_ptr, 1.0)
    if due > 0:
        tl.store(out_ptr + 1, 1.0)
    tl.store(last_ptr + e * 3, px.to(last_ptr.dtype.element_ty), mask=m)
    tl.store(last_ptr + e * 3 + 1, py.to(last_ptr.dtype.element_ty), mask=m)
    tl.store(last_ptr + e * 3 + 2, pz.to(last_ptr.dtype.element_ty), mask=m)
    if tl.program_id(0) == 0:
        tl.store(out_ptr + 2, tl.load(conv_ptr).to(tl.float64))
        for j in range(n_need):
            tl.store(out_ptr + 3 + j, tl.load(need_ptr + j).to(tl.float64))


def plan_checks(pos: Tensor, reference: Tensor, last: Tensor, converged: Tensor, needed: Tensor, skin: float,
                steps: int) -> Tensor:
    """``[outdated, due, converged, *needed]`` of a step plan as one float64 tensor from one kernel
    (see :meth:`~xnn.ffnn.models.amoeba.StepPlan.checks`), with ``last`` set to ``pos``."""
    import struct
    n = int(pos.shape[0])
    out = torch.zeros(3 + int(needed.numel()), dtype=torch.float64, device=pos.device)

    def bits(x: float) -> int:
        return struct.unpack("<q", struct.pack("<d", float(x)))[0]

    _plan_checks[(triton.cdiv(max(n, 1), 256),)](pos.detach().contiguous(), reference.contiguous(), last,
                                                  converged.reshape(-1), needed.reshape(-1).contiguous(), out, n,
                                                  int(needed.numel()), bits(0.5 * skin), bits((0.5 * skin) ** 2),
                                                  steps + 1, BLOCK=256, num_warps=4)
    return out


class HippoPairs(torch.autograd.Function):
    """The pair kernels as one autograd function: the forward kernel, and the backward kernel for
    the gradients with respect to the pair vectors and the per-atom multipoles (the other inputs get
    none: the class parameters are constants of an evaluation here, their gradients belong to the
    eager path of training). Differentiable once."""

    @staticmethod
    def forward(ctx, r_vec, q, d, Q, i_, k_, scales, keep, keep_d, per, params, scalars, polar, expol, modes,
                write_pairs, count=None, mixed=False, select=None):
        out = pairs_forward(i_, k_, r_vec, scales, keep, keep_d, per, q, d, Q, params, scalars, polar, expol, modes,
                            write_pairs, count, mixed, select)
        ctx.save_for_backward(r_vec, q, d, Q, i_, k_, scales, keep, keep_d, per, params, scalars)
        ctx.flags = (polar, expol, modes, count, mixed, select)
        empty = r_vec.new_zeros(0)
        return (out["e_node"], out["field_d"], out["field_p"], out["du"],
                empty if out["expol"] is None else out["expol"], empty if out["e_pair"] is None else out["e_pair"])

    @staticmethod
    def backward(ctx, g_enode, g_fd, g_fp, g_du, g_expol, g_epair):
        r_vec, q, d, Q, i_, k_, scales, keep, keep_d, per, params, scalars = ctx.saved_tensors
        polar, expol, modes, count, mixed, select = ctx.flags
        use_expol = polar and expol and g_expol is not None and g_expol.numel() > 0
        consts = (i_, k_, scales, keep, keep_d, per, scalars, polar, expol, modes, count, mixed, select)
        zero = r_vec.new_zeros(0)
        grads = _HippoPairsGrad.apply(r_vec, q, d, Q, params, g_enode, g_fd if polar else zero,
                                      g_fp if polar else zero, g_du if polar else zero,
                                      g_expol if use_expol else zero, consts)
        g_params = None
        if ctx.needs_input_grad[10]:
            # the per-atom parameters (training): the generated adjoint of the forward kernel
            g_params = pairs_param_grad(i_, k_, r_vec.detach(), scales, keep, keep_d, per, q.detach(), d.detach(),
                                        Q.detach(), params.detach(), scalars, polar, expol, modes, g_enode.detach(),
                                        _given(g_fd if polar else zero), _given(g_fp if polar else zero),
                                        _given(g_du if polar else zero), _given(g_expol if use_expol else zero),
                                        count, mixed, select)
        return (*grads, None, None, None, None, None, None, g_params, None, None, None, None, None, None, None, None)


def _given(t: Optional[Tensor]) -> Optional[Tensor]:
    """An incoming gradient, ``None`` when absent (an empty placeholder)."""
    return None if t is None or t.numel() == 0 else t


class _HippoPairsGrad(torch.autograd.Function):
    """The backward kernel of :class:`HippoPairs` as a function of the pair vectors, the multipoles and the
    incoming gradients, differentiable once more (force training, Hessians): its backward runs the
    generated tangent kernels along the incoming directions, the forward's for the gradients of the
    incoming gradients and the backward's for the second derivatives."""

    @staticmethod
    def forward(ctx, r_vec, q, d, Q, params, g_enode, g_fd, g_fp, g_du, g_expol, consts):
        i_, k_, scales, keep, keep_d, per, scalars, polar, expol, modes, count, mixed, select = consts
        grads = pairs_backward(i_, k_, r_vec, scales, keep, keep_d, per, q, d, Q, params, scalars, polar, expol, modes,
                               g_node=g_enode, g_field_d=_given(g_fd), g_field_p=_given(g_fp), g_du=_given(g_du),
                               g_expol=_given(g_expol), count=count, mixed=mixed, select=select)
        ctx.save_for_backward(r_vec, q, d, Q, params, g_enode, g_fd, g_fp, g_du, g_expol)
        ctx.consts = consts
        return grads["grad_r"], grads["grad_q"], grads["grad_d"], grads["grad_Q"]

    @staticmethod
    @once_differentiable
    def backward(ctx, u_r, u_q, u_d, u_Q):
        r_vec, q, d, Q, params, g_enode, g_fd, g_fp, g_du, g_expol = ctx.saved_tensors
        i_, k_, scales, keep, keep_d, per, scalars, polar, expol, modes, count, mixed, select = ctx.consts
        tangents = tuple(torch.zeros_like(x) if u is None else u
                         for u, x in ((u_r, r_vec), (u_q, q), (u_d, d), (u_Q, Q)))
        fwd = pairs_forward(i_, k_, r_vec, scales, keep, keep_d, per, q, d, Q, params, scalars, polar, expol, modes,
                            False, count, mixed, select, tangents=tangents)
        bwd = pairs_backward(i_, k_, r_vec, scales, keep, keep_d, per, q, d, Q, params, scalars, polar, expol, modes,
                             g_node=g_enode, g_field_d=_given(g_fd), g_field_p=_given(g_fp), g_du=_given(g_du),
                             g_expol=_given(g_expol), count=count, mixed=mixed, select=select, tangents=tangents)

        def matching(t, like):
            return None if like.numel() == 0 or t is None else t.reshape(like.shape)

        g_params = None
        if ctx.needs_input_grad[4]:
            # the parameter part: the derivative of the table's gradient along the same directions
            g_params = pairs_param_grad(i_, k_, r_vec, scales, keep, keep_d, per, q, d, Q, params, scalars, polar,
                                        expol, modes, g_enode, _given(g_fd), _given(g_fp), _given(g_du),
                                        _given(g_expol), count, mixed, select, tangents=tangents)[1]
        return (bwd["grad_r_t"], bwd["grad_q_t"], bwd["grad_d_t"], bwd["grad_Q_t"], g_params, fwd["e_node_t"],
                matching(fwd["field_d_t"], g_fd), matching(fwd["field_p_t"], g_fp), matching(fwd["du_t"], g_du),
                matching(fwd["expol_t"], g_expol), None)


# the AMOEBA pair kernels: the per-atom table (the van der Waals radius and well depth of the atom's
# class, the polarizability and Thole factor of its type) and the float64 scalars of an evaluation
AMOEBA_PARAMS = ("vdw_rmin", "vdw_eps", "polarizability", "thole")
AMOEBA_NP = len(AMOEBA_PARAMS)
AMOEBA_SCALARS = ("coulomb", "inv_sqrt_pi", "alpha_e", "alpha_p", "vdw_taper", "vdw_cutoff", "ewald_cutoff",
                  "delta", "gamma", "tiny", "sixth", "thole7", "thole9a", "thole9b")


@triton.jit
def _thole(R, ai, ak, ti, tk, sc_ptr):
    """The Thole factors ``lambda_3, lambda_5, lambda_7, lambda_9`` of a pair (``thole_ladder``; the
    ninth order is the next member of the ladder, ``(1/R d/dR) D_3 = D_4``), in the precision of
    ``R``: ``d = a (R / (alpha_i alpha_k)^(1/6))^3`` capped at 50, ``a`` the smaller Thole factor
    (the other when one is zero, undamped when both are)."""
    a = tl.where(ti > 0.0, tl.where(tk > 0.0, tl.minimum(ti, tk), ti), tk)
    scale = tl.maximum(libdevice.pow(ai * ak, tl.load(sc_ptr + 10).to(R.dtype)), 1e-12)
    x = R / scale
    d = tl.minimum(a * x * x * x, 50.0)
    ex = libdevice.exp(-d)
    undamped = a <= 0.0
    l3 = tl.where(undamped, 1.0, 1.0 - ex)
    l5 = tl.where(undamped, 1.0, 1.0 - (1.0 + d) * ex)
    l7 = tl.where(undamped, 1.0, 1.0 - (1.0 + d + tl.load(sc_ptr + 11).to(R.dtype) * d * d) * ex)
    l9 = tl.where(undamped, 1.0, 1.0 - (1.0 + d + (tl.load(sc_ptr + 12).to(R.dtype)
                                                   + tl.load(sc_ptr + 13).to(R.dtype) * d) * d * d) * ex)
    return l3, l5, l7, l9


@triton.jit
def _halgren(rv, rmin, eps, delta, gamma):
    """Halgren's buffered 14-7 energy (``buffered_14_7``) and its derivative with respect to ``rv``."""
    rho = rv / rmin
    rho3 = rho * rho * rho
    rho6 = rho3 * rho3
    rho7 = rho6 * rho
    a = (1.0 + delta) / (rho + delta)
    a3 = a * a * a
    a7 = a3 * a3 * a
    b = (1.0 + gamma) / (rho7 + gamma)
    e = eps * a7 * (b - 2.0)
    de = eps / rmin * a7 * (-7.0 * (b - 2.0) / (rho + delta) - 7.0 * rho6 * b / (rho7 + gamma))
    return e, de


@triton.jit
def _vdw_pair(r_ptr, off_ptr, par_ptr, sc_ptr, e, i, k, m, NP: tl.constexpr, RADIUS: tl.constexpr,
              EPSILON: tl.constexpr):
    """The van der Waals site vector of a pair (``r + off_k - off_i``), its length and the combined
    radius and well depth, in float64 (the combining rules: ``RADIUS`` 0 arithmetic, 1 geometric,
    2 cubic mean; ``EPSILON`` 0 geometric, 1 arithmetic, 2 HHG)."""
    vx = (tl.load(r_ptr + e * 3, mask=m, other=1.0) + tl.load(off_ptr + k * 3, mask=m, other=0.0)
          - tl.load(off_ptr + i * 3, mask=m, other=0.0)).to(tl.float64)
    vy = (tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0) + tl.load(off_ptr + k * 3 + 1, mask=m, other=0.0)
          - tl.load(off_ptr + i * 3 + 1, mask=m, other=0.0)).to(tl.float64)
    vz = (tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0) + tl.load(off_ptr + k * 3 + 2, mask=m, other=0.0)
          - tl.load(off_ptr + i * 3 + 2, mask=m, other=0.0)).to(tl.float64)
    tiny = tl.load(sc_ptr + 9)
    rv = tl.sqrt(vx * vx + vy * vy + vz * vz + tiny)
    ri = tl.load(par_ptr + i * NP, mask=m, other=1.0).to(tl.float64)
    rk = tl.load(par_ptr + k * NP, mask=m, other=1.0).to(tl.float64)
    ei = tl.load(par_ptr + i * NP + 1, mask=m, other=0.0).to(tl.float64)
    ek = tl.load(par_ptr + k * NP + 1, mask=m, other=0.0).to(tl.float64)
    if RADIUS == 2:
        rmin = (ri * ri * ri + rk * rk * rk) / (ri * ri + rk * rk + tiny)
    elif RADIUS == 1:
        rmin = tl.sqrt(ri * rk)
    else:
        rmin = 0.5 * (ri + rk)
    if EPSILON == 2:
        si = tl.sqrt(tl.maximum(ei, 0.0))
        sk = tl.sqrt(tl.maximum(ek, 0.0))
        eps = 4.0 * ei * ek / ((si + sk) * (si + sk) + tiny)
    elif EPSILON == 1:
        eps = 0.5 * (ei + ek)
    else:
        eps = tl.sqrt(tl.maximum(ei * ek, 0.0))
    return vx, vy, vz, rv, rmin, eps


@triton.jit
def _amoeba_forward(i_ptr, k_ptr, r_ptr, s_ptr, keep_ptr, per_ptr, q_ptr, d_ptr, Q_ptr, par_ptr, off_ptr, sc_ptr,
                    enode_ptr, epair_ptr, fd_ptr, fp_ptr, du_ptr, n_ptr, sel_ptr,
                    SELECT: tl.constexpr, KEEP_R: tl.constexpr, POLAR: tl.constexpr, WRITE_PAIRS: tl.constexpr,
                    SW_VDW: tl.constexpr, RADIUS: tl.constexpr, EPSILON: tl.constexpr, NP: tl.constexpr,
                    GROUP: tl.constexpr, BLOCK: tl.constexpr):
    """The AMOEBA pair terms of one block of pairs: the permanent multipole energy (``amoeba_pair_block``),
    the buffered 14-7 energy between the van der Waals sites, and with ``POLAR`` the direct- and
    polar-scaled real-space permanent fields and the mutual ladder members of the induction
    (``_real_space_fields``). ``GROUP`` selects the terms of a launch: 1 the electrostatics, 2 the van
    der Waals term, 0 both. ``SW_VDW`` switches the van der Waals energy: 0 not, 1 on the pairs of
    cells, 2 on every pair. ``KEEP_R`` and ``SELECT`` as in :func:`_pairs_forward`."""
    acc = enode_ptr.dtype.element_ty
    start = tl.program_id(0) * BLOCK
    n = tl.load(n_ptr)
    if start >= n:
        return
    lane = (start + tl.arange(0, BLOCK)).to(tl.int64)
    m = lane < n
    last = tl.minimum(start + BLOCK - 1, n - 1)
    if SELECT:
        e = tl.load(sel_ptr + lane, mask=m, other=0)
        first = tl.load(sel_ptr + start)
        last = tl.load(sel_ptr + last)
    else:
        e = lane
        first = start
    i = tl.load(i_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0)
    i0 = tl.load(i_ptr + first)
    i1 = tl.load(i_ptr + last)
    same = m & (i == i0)
    tail = m & (i == i1) & (i1 != i0)
    rest = m & (i != i0) & (i != i1)
    rx = tl.load(r_ptr + e * 3, mask=m, other=1.0)
    ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    per = tl.load(per_ptr + e, mask=m, other=0.0)
    dtype = rx.dtype
    if GROUP != 1:
        # van der Waals between the reduced sites, in float64 as a whole
        vx, vy, vz, rv, rmin, eps = _vdw_pair(r_ptr, off_ptr, par_ptr, sc_ptr, e, i, k, m, NP, RADIUS, EPSILON)
        e_vdw, de_vdw = _halgren(rv, rmin, eps, tl.load(sc_ptr + 7), tl.load(sc_ptr + 8))
        e_vdw = e_vdw * tl.load(s_ptr + e * 8, mask=m, other=0.0).to(tl.float64)
        if SW_VDW > 0:
            sw = _switch(rv, tl.load(sc_ptr + 4), tl.load(sc_ptr + 5))
            if SW_VDW == 1:
                sw = tl.where(per > 0.0, sw, 1.0)
            e_vdw = e_vdw * sw
        e_vdw = e_vdw.to(dtype)
        if WRITE_PAIRS:
            tl.store(epair_ptr + e * 2 + 1, e_vdw, mask=m)
        _add_sorted(enode_ptr + 1, i0 * 2, i1 * 2, i * 2, (0.5 * e_vdw).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(enode_ptr + k * 2 + 1, (0.5 * e_vdw).to(acc), mask=m, sem="relaxed")
    if GROUP != 2:
        s1 = tl.load(s_ptr + e * 8 + 1, mask=m, other=0.0)
        coulomb = tl.load(sc_ptr)
        inv_sqrt_pi = tl.load(sc_ptr + 1).to(dtype)
        qi = tl.load(q_ptr + i, mask=m, other=0.0)
        qk = tl.load(q_ptr + k, mask=m, other=0.0)
        dix = tl.load(d_ptr + i * 3, mask=m, other=0.0)
        diy = tl.load(d_ptr + i * 3 + 1, mask=m, other=0.0)
        diz = tl.load(d_ptr + i * 3 + 2, mask=m, other=0.0)
        dkx = tl.load(d_ptr + k * 3, mask=m, other=0.0)
        dky = tl.load(d_ptr + k * 3 + 1, mask=m, other=0.0)
        dkz = tl.load(d_ptr + k * 3 + 2, mask=m, other=0.0)
        qixx = tl.load(Q_ptr + i * 9, mask=m, other=0.0)
        qixy = tl.load(Q_ptr + i * 9 + 1, mask=m, other=0.0)
        qixz = tl.load(Q_ptr + i * 9 + 2, mask=m, other=0.0)
        qiyy = tl.load(Q_ptr + i * 9 + 4, mask=m, other=0.0)
        qiyz = tl.load(Q_ptr + i * 9 + 5, mask=m, other=0.0)
        qizz = tl.load(Q_ptr + i * 9 + 8, mask=m, other=0.0)
        qkxx = tl.load(Q_ptr + k * 9, mask=m, other=0.0)
        qkxy = tl.load(Q_ptr + k * 9 + 1, mask=m, other=0.0)
        qkxz = tl.load(Q_ptr + k * 9 + 2, mask=m, other=0.0)
        qkyy = tl.load(Q_ptr + k * 9 + 4, mask=m, other=0.0)
        qkyz = tl.load(Q_ptr + k * 9 + 5, mask=m, other=0.0)
        qkzz = tl.load(Q_ptr + k * 9 + 8, mask=m, other=0.0)
        # the float32 square root and division of the GPU are approximate and biased: both through float64
        R = tl.sqrt((rx * rx + ry * ry + rz * rz).to(tl.float64)).to(dtype)
        inv = (1.0 / R.to(tl.float64)).to(dtype)
        if KEEP_R:
            keep = tl.where(per > 0.0, tl.where(R < tl.load(sc_ptr + 6).to(dtype), 1.0, 0.0), 1.0).to(dtype)
        else:
            keep = tl.load(keep_ptr + e, mask=m, other=0.0)
        c0, c1, c2, c3, c4, c5 = _coulomb6(inv)
        ae = (tl.load(sc_ptr + 2) * per.to(tl.float64)).to(dtype)
        e0, e1, e2, e3, e4, e5 = _ewald6(R, inv, ae, inv_sqrt_pi)
        # the point ladder: the Coulomb ladder scaled by s1, in a cell the Ewald interaction minus the
        # excluded fraction of the bare one; the whole on the kept pairs
        u0 = s1 * c0 + (e0 - c0) * per
        u1 = s1 * c1 + (e1 - c1) * per
        u2 = s1 * c2 + (e2 - c2) * per
        u3 = s1 * c3 + (e3 - c3) * per
        u4 = s1 * c4 + (e4 - c4) * per
        e_mp = _energy(qi * qk * u0, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz,
                       dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rx, ry, rz,
                       qk * u1, qk * u2, qi * u1, qi * u2, u1, u2, u3, u4) * keep * coulomb.to(dtype)
        if WRITE_PAIRS:
            tl.store(epair_ptr + e * 2, e_mp, mask=m)
        _add_sorted(enode_ptr, i0 * 2, i1 * 2, i * 2, (0.5 * e_mp).to(acc), same, tail, rest, TAIL=0)
        tl.atomic_add(enode_ptr + k * 2, (0.5 * e_mp).to(acc), mask=m, sem="relaxed")
        if POLAR:
            # the Thole-damped ladder (in float64, cast) and the fields of each set at the other atom
            R64 = R.to(tl.float64)
            l3, l5, l7, l9 = _thole(R64, tl.load(par_ptr + i * NP + 2, mask=m, other=0.0).to(tl.float64),
                                    tl.load(par_ptr + k * NP + 2, mask=m, other=0.0).to(tl.float64),
                                    tl.load(par_ptr + i * NP + 3, mask=m, other=0.0).to(tl.float64),
                                    tl.load(par_ptr + k * NP + 3, mask=m, other=0.0).to(tl.float64), sc_ptr)
            t1 = l3.to(dtype) * c1
            t2 = l5.to(dtype) * c2
            t3 = l7.to(dtype) * c3
            fkx, fky, fkz = _field(qi * t1, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, rx, ry, rz,
                                   t1, t2, t3, -1.0)
            fix, fiy, fiz = _field(qk * t1, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rx, ry, rz,
                                   t1, t2, t3, 1.0)
            # the Ewald field of the multipoles minus their bare field, on the kept pairs of a cell
            ap = (tl.load(sc_ptr + 3) * per.to(tl.float64)).to(dtype)
            p0, p1, p2, p3, p4, p5 = _ewald6(R, inv, ap, inv_sqrt_pi)
            v1 = (p1 - c1) * keep * per
            v2 = (p2 - c2) * keep * per
            v3 = (p3 - c3) * keep * per
            ewkx, ewky, ewkz = _field(qi * v1, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, rx, ry, rz,
                                      v1, v2, v3, -1.0)
            ewix, ewiy, ewiz = _field(qk * v1, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rx, ry, rz,
                                      v1, v2, v3, 1.0)
            sd = tl.load(s_ptr + e * 8 + 6, mask=m, other=0.0) * keep
            sp = tl.load(s_ptr + e * 8 + 2, mask=m, other=0.0) * keep
            tl.atomic_add(fd_ptr + k * 3, (ewkx + fkx * sd).to(acc), mask=m, sem="relaxed")
            tl.atomic_add(fd_ptr + k * 3 + 1, (ewky + fky * sd).to(acc), mask=m, sem="relaxed")
            tl.atomic_add(fd_ptr + k * 3 + 2, (ewkz + fkz * sd).to(acc), mask=m, sem="relaxed")
            _add_sorted(fd_ptr, i0 * 3, i1 * 3, i * 3, (ewix + fix * sd).to(acc), same, tail, rest, TAIL=0)
            _add_sorted(fd_ptr + 1, i0 * 3, i1 * 3, i * 3, (ewiy + fiy * sd).to(acc), same, tail, rest, TAIL=0)
            _add_sorted(fd_ptr + 2, i0 * 3, i1 * 3, i * 3, (ewiz + fiz * sd).to(acc), same, tail, rest, TAIL=0)
            tl.atomic_add(fp_ptr + k * 3, (ewkx + fkx * sp).to(acc), mask=m, sem="relaxed")
            tl.atomic_add(fp_ptr + k * 3 + 1, (ewky + fky * sp).to(acc), mask=m, sem="relaxed")
            tl.atomic_add(fp_ptr + k * 3 + 2, (ewkz + fkz * sp).to(acc), mask=m, sem="relaxed")
            _add_sorted(fp_ptr, i0 * 3, i1 * 3, i * 3, (ewix + fix * sp).to(acc), same, tail, rest, TAIL=0)
            _add_sorted(fp_ptr + 1, i0 * 3, i1 * 3, i * 3, (ewiy + fiy * sp).to(acc), same, tail, rest, TAIL=0)
            _add_sorted(fp_ptr + 2, i0 * 3, i1 * 3, i * 3, (ewiz + fiz * sp).to(acc), same, tail, rest, TAIL=0)
            # the mutual ladder of the solve: the damped ladder scaled, plus the Ewald difference
            us = tl.load(s_ptr + e * 8 + 7, mask=m, other=0.0) * tl.load(s_ptr + e * 8 + 3, mask=m, other=0.0)
            tl.store(du_ptr + e * 2, ((p1 - c1) * per + t1 * us) * keep, mask=m)
            tl.store(du_ptr + e * 2 + 1, ((p2 - c2) * per + t2 * us) * keep, mask=m)


@triton.jit
def _amoeba_backward(i_ptr, k_ptr, r_ptr, s_ptr, keep_ptr, per_ptr, q_ptr, d_ptr, Q_ptr, par_ptr, off_ptr, sc_ptr,
                     ge_ptr, gfd_ptr, gfp_ptr, gdu_ptr, gr_ptr, gq_ptr, gd_ptr, gQ_ptr, goff_ptr, n_ptr, sel_ptr,
                     SELECT: tl.constexpr, KEEP_R: tl.constexpr, POLAR: tl.constexpr, SW_VDW: tl.constexpr,
                     RADIUS: tl.constexpr, EPSILON: tl.constexpr, NP: tl.constexpr, GROUP: tl.constexpr,
                     BLOCK: tl.constexpr):
    """The gradients of the AMOEBA pair terms for incoming gradients of every output of the forward
    kernel: per pair with respect to the pair vector (stored), per atom with respect to the charges,
    dipoles, quadrupoles and van der Waals site offsets (atomic adds). ``GROUP`` selects the terms
    of a launch: 1 the permanent multipole energy, 2 the fields and mutual ladders, 3 the van der
    Waals term, 0 all; the groups after the first add to the stored pair-vector gradient."""
    acc = gq_ptr.dtype.element_ty
    start = tl.program_id(0) * BLOCK
    n = tl.load(n_ptr)
    if start >= n:
        return
    lane = (start + tl.arange(0, BLOCK)).to(tl.int64)
    m = lane < n
    last = tl.minimum(start + BLOCK - 1, n - 1)
    if SELECT:
        e = tl.load(sel_ptr + lane, mask=m, other=0)
        first = tl.load(sel_ptr + start)
        last = tl.load(sel_ptr + last)
    else:
        e = lane
        first = start
    i = tl.load(i_ptr + e, mask=m, other=0)
    k = tl.load(k_ptr + e, mask=m, other=0)
    i0 = tl.load(i_ptr + first)
    i1 = tl.load(i_ptr + last)
    same = m & (i == i0)
    tail = m & (i == i1) & (i1 != i0)
    rest = m & (i != i0) & (i != i1)
    rx = tl.load(r_ptr + e * 3, mask=m, other=1.0)
    ry = tl.load(r_ptr + e * 3 + 1, mask=m, other=0.0)
    rz = tl.load(r_ptr + e * 3 + 2, mask=m, other=0.0)
    per = tl.load(per_ptr + e, mask=m, other=0.0)
    dtype = rx.dtype
    zero = rx * 0.0
    grx = zero
    gry = zero
    grz = zero
    gsr = zero
    if GROUP == 0 or GROUP == 3:
        # van der Waals: a radial function of the site vector, whose gradient goes to the pair vector
        # and to the offsets of both sites
        g_vdw = 0.5 * (tl.load(ge_ptr + i * 2 + 1, mask=m, other=0.0)
                       + tl.load(ge_ptr + k * 2 + 1, mask=m, other=0.0)).to(tl.float64)
        vx, vy, vz, rv, rmin, eps = _vdw_pair(r_ptr, off_ptr, par_ptr, sc_ptr, e, i, k, m, NP, RADIUS, EPSILON)
        e_vdw, de_vdw = _halgren(rv, rmin, eps, tl.load(sc_ptr + 7), tl.load(sc_ptr + 8))
        if SW_VDW > 0:
            sw = _switch(rv, tl.load(sc_ptr + 4), tl.load(sc_ptr + 5))
            dsw = _switch_derivative(rv, tl.load(sc_ptr + 4), tl.load(sc_ptr + 5))
            if SW_VDW == 1:
                sw = tl.where(per > 0.0, sw, 1.0)
                dsw = tl.where(per > 0.0, dsw, 0.0)
            de_vdw = de_vdw * sw + e_vdw * dsw
        gv = g_vdw * tl.load(s_ptr + e * 8, mask=m, other=0.0).to(tl.float64) * de_vdw / rv
        gvx = (gv * vx).to(dtype)
        gvy = (gv * vy).to(dtype)
        gvz = (gv * vz).to(dtype)
        grx += gvx
        gry += gvy
        grz += gvz
        tl.atomic_add(goff_ptr + k * 3, gvx.to(acc), mask=m, sem="relaxed")
        tl.atomic_add(goff_ptr + k * 3 + 1, gvy.to(acc), mask=m, sem="relaxed")
        tl.atomic_add(goff_ptr + k * 3 + 2, gvz.to(acc), mask=m, sem="relaxed")
        _add_sorted(goff_ptr, i0 * 3, i1 * 3, i * 3, (-gvx).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(goff_ptr + 1, i0 * 3, i1 * 3, i * 3, (-gvy).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(goff_ptr + 2, i0 * 3, i1 * 3, i * 3, (-gvz).to(acc), same, tail, rest, TAIL=1)
    if GROUP != 3:
        inv_sqrt_pi = tl.load(sc_ptr + 1).to(dtype)
        qi = tl.load(q_ptr + i, mask=m, other=0.0)
        qk = tl.load(q_ptr + k, mask=m, other=0.0)
        dix = tl.load(d_ptr + i * 3, mask=m, other=0.0)
        diy = tl.load(d_ptr + i * 3 + 1, mask=m, other=0.0)
        diz = tl.load(d_ptr + i * 3 + 2, mask=m, other=0.0)
        dkx = tl.load(d_ptr + k * 3, mask=m, other=0.0)
        dky = tl.load(d_ptr + k * 3 + 1, mask=m, other=0.0)
        dkz = tl.load(d_ptr + k * 3 + 2, mask=m, other=0.0)
        qixx = tl.load(Q_ptr + i * 9, mask=m, other=0.0)
        qixy = tl.load(Q_ptr + i * 9 + 1, mask=m, other=0.0)
        qixz = tl.load(Q_ptr + i * 9 + 2, mask=m, other=0.0)
        qiyy = tl.load(Q_ptr + i * 9 + 4, mask=m, other=0.0)
        qiyz = tl.load(Q_ptr + i * 9 + 5, mask=m, other=0.0)
        qizz = tl.load(Q_ptr + i * 9 + 8, mask=m, other=0.0)
        qkxx = tl.load(Q_ptr + k * 9, mask=m, other=0.0)
        qkxy = tl.load(Q_ptr + k * 9 + 1, mask=m, other=0.0)
        qkxz = tl.load(Q_ptr + k * 9 + 2, mask=m, other=0.0)
        qkyy = tl.load(Q_ptr + k * 9 + 4, mask=m, other=0.0)
        qkyz = tl.load(Q_ptr + k * 9 + 5, mask=m, other=0.0)
        qkzz = tl.load(Q_ptr + k * 9 + 8, mask=m, other=0.0)
        R = tl.sqrt((rx * rx + ry * ry + rz * rz).to(tl.float64)).to(dtype)
        inv = (1.0 / R.to(tl.float64)).to(dtype)
        if KEEP_R:
            keep = tl.where(per > 0.0, tl.where(R < tl.load(sc_ptr + 6).to(dtype), 1.0, 0.0), 1.0).to(dtype)
        else:
            keep = tl.load(keep_ptr + e, mask=m, other=0.0)
        c0, c1, c2, c3, c4, c5 = _coulomb6(inv)
        gqi = zero
        gqk = zero
        gdix = zero
        gdiy = zero
        gdiz = zero
        gdkx = zero
        gdky = zero
        gdkz = zero
        gQixx = zero
        gQixy = zero
        gQixz = zero
        gQiyx = zero
        gQiyy = zero
        gQiyz = zero
        gQizx = zero
        gQizy = zero
        gQizz = zero
        gQkxx = zero
        gQkxy = zero
        gQkxz = zero
        gQkyx = zero
        gQkyy = zero
        gQkyz = zero
        gQkzx = zero
        gQkzy = zero
        gQkzz = zero
    if GROUP == 0 or GROUP == 1:
        # permanent multipoles: the adjoint of the contraction and the chains to the charges and to
        # R (every member of the point ladder to the next order)
        g_mp = 0.5 * (tl.load(ge_ptr + i * 2, mask=m, other=0.0) + tl.load(ge_ptr + k * 2, mask=m, other=0.0))
        s1 = tl.load(s_ptr + e * 8 + 1, mask=m, other=0.0)
        ae = (tl.load(sc_ptr + 2) * per.to(tl.float64)).to(dtype)
        e0, e1, e2, e3, e4, e5 = _ewald6(R, inv, ae, inv_sqrt_pi)
        w_mp = g_mp * tl.load(sc_ptr).to(dtype) * keep
        u0 = s1 * c0 + (e0 - c0) * per
        u1 = s1 * c1 + (e1 - c1) * per
        u2 = s1 * c2 + (e2 - c2) * per
        u3 = s1 * c3 + (e3 - c3) * per
        u4 = s1 * c4 + (e4 - c4) * per
        u5 = s1 * c5 + (e5 - c5) * per
        (a_dix, a_diy, a_diz, a_dkx, a_dky, a_dkz, a_qixx, a_qixy, a_qixz, a_qiyx, a_qiyy, a_qiyz, a_qizx, a_qizy,
         a_qizz, a_qkxx, a_qkxy, a_qkxz, a_qkyx, a_qkyy, a_qkyz, a_qkzx, a_qkzy, a_qkzz, a_rx, a_ry, a_rz,
         a_Li1, a_Li2, a_Lk1, a_Lk2, a_L1, a_L2, a_L3, a_L4) = _energy_adjoint(
            w_mp, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz,
            qkzz, rx, ry, rz, qk * u1, qk * u2, qi * u1, qi * u2, u1, u2, u3, u4)
        gqi += w_mp * qk * u0 + a_Lk1 * u1 + a_Lk2 * u2
        gqk += w_mp * qi * u0 + a_Li1 * u1 + a_Li2 * u2
        gdix += a_dix
        gdiy += a_diy
        gdiz += a_diz
        gdkx += a_dkx
        gdky += a_dky
        gdkz += a_dkz
        gQixx += a_qixx
        gQixy += a_qixy
        gQixz += a_qixz
        gQiyx += a_qiyx
        gQiyy += a_qiyy
        gQiyz += a_qiyz
        gQizx += a_qizx
        gQizy += a_qizy
        gQizz += a_qizz
        gQkxx += a_qkxx
        gQkxy += a_qkxy
        gQkxz += a_qkxz
        gQkyx += a_qkyx
        gQkyy += a_qkyy
        gQkyz += a_qkyz
        gQkzx += a_qkzx
        gQkzy += a_qkzy
        gQkzz += a_qkzz
        grx += a_rx
        gry += a_ry
        grz += a_rz
        gsr += (w_mp * qi * qk * u1 + (a_Li1 * qk + a_Lk1 * qi) * u2 + (a_Li2 * qk + a_Lk2 * qi) * u3
                + a_L1 * u2 + a_L2 * u3 + a_L3 * u4 + a_L4 * u5)
    if POLAR and (GROUP == 0 or GROUP == 2):
        gfdkx = tl.load(gfd_ptr + k * 3, mask=m, other=0.0)
        gfdky = tl.load(gfd_ptr + k * 3 + 1, mask=m, other=0.0)
        gfdkz = tl.load(gfd_ptr + k * 3 + 2, mask=m, other=0.0)
        gfdix = tl.load(gfd_ptr + i * 3, mask=m, other=0.0)
        gfdiy = tl.load(gfd_ptr + i * 3 + 1, mask=m, other=0.0)
        gfdiz = tl.load(gfd_ptr + i * 3 + 2, mask=m, other=0.0)
        gfpkx = tl.load(gfp_ptr + k * 3, mask=m, other=0.0)
        gfpky = tl.load(gfp_ptr + k * 3 + 1, mask=m, other=0.0)
        gfpkz = tl.load(gfp_ptr + k * 3 + 2, mask=m, other=0.0)
        gfpix = tl.load(gfp_ptr + i * 3, mask=m, other=0.0)
        gfpiy = tl.load(gfp_ptr + i * 3 + 1, mask=m, other=0.0)
        gfpiz = tl.load(gfp_ptr + i * 3 + 2, mask=m, other=0.0)
        sd = tl.load(s_ptr + e * 8 + 6, mask=m, other=0.0) * keep
        sp = tl.load(s_ptr + e * 8 + 2, mask=m, other=0.0) * keep
        gkx = gfdkx * sd + gfpkx * sp
        gky = gfdky * sd + gfpky * sp
        gkz = gfdkz * sd + gfpkz * sp
        gix = gfdix * sd + gfpix * sp
        giy = gfdiy * sd + gfpiy * sp
        giz = gfdiz * sd + gfpiz * sp
        R64 = R.to(tl.float64)
        l3, l5, l7, l9 = _thole(R64, tl.load(par_ptr + i * NP + 2, mask=m, other=0.0).to(tl.float64),
                                tl.load(par_ptr + k * NP + 2, mask=m, other=0.0).to(tl.float64),
                                tl.load(par_ptr + i * NP + 3, mask=m, other=0.0).to(tl.float64),
                                tl.load(par_ptr + k * NP + 3, mask=m, other=0.0).to(tl.float64), sc_ptr)
        t1 = l3.to(dtype) * c1
        t2 = l5.to(dtype) * c2
        t3 = l7.to(dtype) * c3
        t4 = l9.to(dtype) * c4
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gkx, gky, gkz, qi * t1, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz,
                                            qizz, rx, ry, rz, t1, t2, t3, -1.0)
        gqi += b_C * t1
        gdix += b_dx
        gdiy += b_dy
        gdiz += b_dz
        gQixx += b_qxx
        gQixy += b_qxy
        gQixz += b_qxz
        gQiyx += b_qyx
        gQiyy += b_qyy
        gQiyz += b_qyz
        gQizx += b_qzx
        gQizy += b_qzy
        gQizz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * qi * t2 + b_D1 * t2 + b_D2 * t3 + b_D3 * t4
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gix, giy, giz, qk * t1, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz,
                                            qkzz, rx, ry, rz, t1, t2, t3, 1.0)
        gqk += b_C * t1
        gdkx += b_dx
        gdky += b_dy
        gdkz += b_dz
        gQkxx += b_qxx
        gQkxy += b_qxy
        gQkxz += b_qxz
        gQkyx += b_qyx
        gQkyy += b_qyy
        gQkyz += b_qyz
        gQkzx += b_qzx
        gQkzy += b_qzy
        gQkzz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * qk * t2 + b_D1 * t2 + b_D2 * t3 + b_D3 * t4
        # the Ewald difference of the multipoles, both ways, unscaled in both fields
        ap = (tl.load(sc_ptr + 3) * per.to(tl.float64)).to(dtype)
        p0, p1, p2, p3, p4, p5 = _ewald6(R, inv, ap, inv_sqrt_pi)
        kp = keep * per
        v1 = (p1 - c1) * kp
        v2 = (p2 - c2) * kp
        v3 = (p3 - c3) * kp
        v4 = (p4 - c4) * kp
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gfdkx + gfpkx, gfdky + gfpky, gfdkz + gfpkz, qi * v1, dix, diy, diz,
                                            qixx, qixy, qixz, qiyy, qiyz, qizz, rx, ry, rz, v1, v2, v3, -1.0)
        gqi += b_C * v1
        gdix += b_dx
        gdiy += b_dy
        gdiz += b_dz
        gQixx += b_qxx
        gQixy += b_qxy
        gQixz += b_qxz
        gQiyx += b_qyx
        gQiyy += b_qyy
        gQiyz += b_qyz
        gQizx += b_qzx
        gQizy += b_qzy
        gQizz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * qi * v2 + b_D1 * v2 + b_D2 * v3 + b_D3 * v4
        (b_C, b_dx, b_dy, b_dz, b_qxx, b_qxy, b_qxz, b_qyx, b_qyy, b_qyz, b_qzx, b_qzy, b_qzz, b_rx, b_ry, b_rz,
         b_D1, b_D2, b_D3) = _field_adjoint(gfdix + gfpix, gfdiy + gfpiy, gfdiz + gfpiz, qk * v1, dkx, dky, dkz,
                                            qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rx, ry, rz, v1, v2, v3, 1.0)
        gqk += b_C * v1
        gdkx += b_dx
        gdky += b_dy
        gdkz += b_dz
        gQkxx += b_qxx
        gQkxy += b_qxy
        gQkxz += b_qxz
        gQkyx += b_qyx
        gQkyy += b_qyy
        gQkyz += b_qyz
        gQkzx += b_qzx
        gQkzy += b_qzy
        gQkzz += b_qzz
        grx += b_rx
        gry += b_ry
        grz += b_rz
        gsr += b_C * qk * v2 + b_D1 * v2 + b_D2 * v3 + b_D3 * v4
        # the mutual ladder members: radial only
        us = tl.load(s_ptr + e * 8 + 7, mask=m, other=0.0) * tl.load(s_ptr + e * 8 + 3, mask=m, other=0.0)
        gdu1 = tl.load(gdu_ptr + e * 2, mask=m, other=0.0)
        gdu2 = tl.load(gdu_ptr + e * 2 + 1, mask=m, other=0.0)
        gsr += gdu1 * ((p2 - c2) * per + t2 * us) * keep
        gsr += gdu2 * ((p3 - c3) * per + t3 * us) * keep
    if GROUP >= 2:
        grx += tl.load(gr_ptr + e * 3, mask=m, other=0.0)
        gry += tl.load(gr_ptr + e * 3 + 1, mask=m, other=0.0)
        grz += tl.load(gr_ptr + e * 3 + 2, mask=m, other=0.0)
    tl.store(gr_ptr + e * 3, grx + gsr * rx, mask=m)
    tl.store(gr_ptr + e * 3 + 1, gry + gsr * ry, mask=m)
    tl.store(gr_ptr + e * 3 + 2, grz + gsr * rz, mask=m)
    if GROUP != 3:
        _add_sorted(gq_ptr, i0, i1, i, (gqi).to(acc), same, tail, rest, TAIL=1)
        tl.atomic_add(gq_ptr + k, (gqk).to(acc), mask=m, sem="relaxed")
        _add_sorted(gd_ptr, i0 * 3, i1 * 3, i * 3, (gdix).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gd_ptr + 1, i0 * 3, i1 * 3, i * 3, (gdiy).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gd_ptr + 2, i0 * 3, i1 * 3, i * 3, (gdiz).to(acc), same, tail, rest, TAIL=1)
        tl.atomic_add(gd_ptr + k * 3, (gdkx).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gd_ptr + k * 3 + 1, (gdky).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gd_ptr + k * 3 + 2, (gdkz).to(acc), mask=m, sem="relaxed")
        _add_sorted(gQ_ptr, i0 * 9, i1 * 9, i * 9, (gQixx).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 1, i0 * 9, i1 * 9, i * 9, (gQixy).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 2, i0 * 9, i1 * 9, i * 9, (gQixz).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 3, i0 * 9, i1 * 9, i * 9, (gQiyx).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 4, i0 * 9, i1 * 9, i * 9, (gQiyy).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 5, i0 * 9, i1 * 9, i * 9, (gQiyz).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 6, i0 * 9, i1 * 9, i * 9, (gQizx).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 7, i0 * 9, i1 * 9, i * 9, (gQizy).to(acc), same, tail, rest, TAIL=1)
        _add_sorted(gQ_ptr + 8, i0 * 9, i1 * 9, i * 9, (gQizz).to(acc), same, tail, rest, TAIL=1)
        tl.atomic_add(gQ_ptr + k * 9, (gQkxx).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 1, (gQkxy).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 2, (gQkxz).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 3, (gQkyx).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 4, (gQkyy).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 5, (gQkyz).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 6, (gQkzx).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 7, (gQkzy).to(acc), mask=m, sem="relaxed")
        tl.atomic_add(gQ_ptr + k * 9 + 8, (gQkzz).to(acc), mask=m, sem="relaxed")


def amoeba_pairs_forward(i_: Tensor, k_: Tensor, r_vec: Tensor, scales: Tensor, keep: Optional[Tensor], per: Tensor,
                         q: Tensor, d: Tensor, Q: Tensor, params: Tensor, offsets: Tensor, scalars: Tensor,
                         polar: bool, modes: tuple, write_pairs: bool = False, count=None,
                         select: Optional[Tensor] = None, tangents: Optional[tuple] = None) -> dict:
    """Launch the AMOEBA pair kernel.

    The inputs are those of :func:`pairs_forward` with the AMOEBA tables: ``params`` the per-atom
    table ``(N, AMOEBA_NP)`` (:data:`AMOEBA_PARAMS`), ``offsets`` the van der Waals site of every
    atom minus its position ``(N, 3)``, ``scalars`` the float64 scalars (:data:`AMOEBA_SCALARS`),
    ``modes`` the van der Waals switch (0 none, 1 cells, 2 every pair) and combining rules of the
    radius (0 arithmetic, 1 geometric, 2 cubic mean) and the well depth (0 geometric, 1
    arithmetic, 2 HHG). ``tangents`` (directions of the pair vectors, charges, dipoles, quadrupoles and
    site offsets) add the outputs' derivatives along them (``<key>_t``).

    Returns
    -------
    dict
        ``e_node`` ``(N, 2)`` (multipoles, van der Waals), ``field_d`` / ``field_p`` ``(N, 3)``,
        ``du`` ``(M, 2)`` and ``e_pair`` ``(M, 2)`` (or ``None``).
    """
    M, N = int(i_.shape[0]), int(q.shape[0])
    dtype, device = r_vec.dtype, r_vec.device
    widths = {"e_node": 2, "field_d": 3, "field_p": 3}
    out, flat = _segments(N, widths, device)
    out.update({"du": torch.zeros(M, 2, dtype=dtype, device=device),
                "e_pair": ((torch.empty if select is None else torch.zeros)(M, 2, dtype=dtype, device=device)
                           if write_pairs else None)})
    keys = ("e_node", "e_pair", "field_d", "field_p", "du")
    if tangents is not None:
        out.update({k + "_t": None if out[k] is None else torch.zeros_like(out[k]) for k in keys})
    if M > 0:
        dummy = out["du"]
        kernel, extra = _amoeba_forward, ()
        if tangents is not None:
            from . import amoeba_tangents as tg
            kernel = tg._amoeba_forward_t
            r_t, q_t, d_t, Q_t, off_t = (t.contiguous() for t in tangents)
            extra = (r_t, q_t, d_t, Q_t.reshape(N, 9).contiguous(), off_t,
                     *[dummy if out[k + "_t"] is None else out[k + "_t"] for k in keys])
        for group in ((1, 2) if dtype in SPLIT_DTYPES else (0,)):
            kernel[(triton.cdiv(M, TILE),)](
                i_, k_, r_vec.contiguous(), scales.contiguous(), dummy if keep is None else keep.contiguous(),
                per.contiguous(), q.contiguous(), d.contiguous(), Q.reshape(N, 9).contiguous(), params.contiguous(),
                offsets.contiguous(), scalars, out["e_node"], out["e_pair"] if write_pairs else dummy,
                out["field_d"], out["field_p"], out["du"], _count(count, M, device),
                dummy if select is None else select, *extra, SELECT=select is not None, KEEP_R=keep is None,
                POLAR=bool(polar), WRITE_PAIRS=bool(write_pairs), SW_VDW=int(modes[0]), RADIUS=int(modes[1]),
                EPSILON=int(modes[2]), NP=AMOEBA_NP, GROUP=group, BLOCK=TILE, num_warps=WARPS, maxnreg=MAXNREG)
    out.update(_cast_segments(flat, N, widths, dtype))
    for key in ("e_node_t", "field_d_t", "field_p_t"):
        if out.get(key) is not None:
            out[key] = out[key].to(dtype)
    return out


def amoeba_pairs_backward(i_: Tensor, k_: Tensor, r_vec: Tensor, scales: Tensor, keep: Optional[Tensor], per: Tensor,
                          q: Tensor, d: Tensor, Q: Tensor, params: Tensor, offsets: Tensor, scalars: Tensor,
                          polar: bool, modes: tuple, g_node: Tensor, g_field_d=None, g_field_p=None, g_du=None,
                          count=None, select: Optional[Tensor] = None, tangents: Optional[tuple] = None) -> dict:
    """Launch the AMOEBA backward kernel: the gradients of ``sum(g_node * e_node) + sum(g_field_d *
    field_d) + sum(g_field_p * field_p) + sum(g_du * du)`` with respect to the pair vectors ``(M, 3)``,
    the charges ``(N,)``, dipoles ``(N, 3)``, quadrupoles ``(N, 3, 3)`` and site offsets ``(N, 3)``.
    The inputs are those of :func:`amoeba_pairs_forward` (with ``tangents``, the gradients' derivatives
    along them as ``<key>_t``); a missing incoming gradient counts as zero."""
    M, N = int(i_.shape[0]), int(q.shape[0])
    dtype, device = r_vec.dtype, r_vec.device
    widths = {"grad_q": 1, "grad_d": 3, "grad_Q": 9, "grad_off": 3}
    out, flat = _segments(N, widths, device)
    out["grad_r"] = torch.zeros(M, 3, dtype=dtype, device=device)
    keys = ("grad_r", "grad_q", "grad_d", "grad_Q", "grad_off")
    if tangents is not None:
        out.update({k + "_t": torch.zeros_like(out[k]) for k in keys})
    if M > 0:
        def given(t, shape):
            return torch.zeros(shape, dtype=dtype, device=device) if t is None else t.reshape(shape).contiguous()

        args = (i_, k_, r_vec.contiguous(), scales.contiguous(), out["grad_r"] if keep is None else keep.contiguous(),
                per.contiguous(), q.contiguous(), d.contiguous(), Q.reshape(N, 9).contiguous(), params.contiguous(),
                offsets.contiguous(), scalars, given(g_node, (N, 2)), given(g_field_d, (N, 3)),
                given(g_field_p, (N, 3)), given(g_du, (M, 2)), out["grad_r"], out["grad_q"], out["grad_d"],
                out["grad_Q"], out["grad_off"], _count(count, M, device), out["grad_r"] if select is None else select)
        kernel = _amoeba_backward
        if tangents is not None:
            from . import amoeba_tangents as tg
            kernel = tg._amoeba_backward_t
            r_t, q_t, d_t, Q_t, off_t = (t.contiguous() for t in tangents)
            args = args + (r_t, q_t, d_t, Q_t.reshape(N, 9).contiguous(), off_t, *[out[k + "_t"] for k in keys])
        groups = ((1, 2, 3) if polar else (1, 3)) if dtype in SPLIT_BACKWARD else (0,)
        tile, warps = AMOEBA_BACKWARD_LAUNCH.get(dtype, (TILE, WARPS))
        for group in groups:
            kernel[(triton.cdiv(M, tile),)](
                *args, SELECT=select is not None, KEEP_R=keep is None, POLAR=bool(polar), SW_VDW=int(modes[0]),
                RADIUS=int(modes[1]), EPSILON=int(modes[2]), NP=AMOEBA_NP, GROUP=group, BLOCK=tile,
                num_warps=warps, maxnreg=MAXNREG)
    out.update(_cast_segments(flat, N, widths, dtype))
    for key in ("grad_q_t", "grad_d_t", "grad_Q_t", "grad_off_t"):
        if key in out:
            out[key] = out[key].to(dtype)
    for key in ("grad_Q", "grad_Q_t"):
        if key in out:
            out[key] = out[key].view(N, 3, 3)
    return out


class AmoebaPairs(torch.autograd.Function):
    """The AMOEBA pair kernels as one autograd function (see :class:`HippoPairs`): gradients with
    respect to the pair vectors, the per-atom multipoles and the van der Waals site offsets.
    Differentiable once."""

    @staticmethod
    def forward(ctx, r_vec, q, d, Q, offsets, i_, k_, scales, keep, per, params, scalars, polar, modes, write_pairs,
                count=None, select=None):
        out = amoeba_pairs_forward(i_, k_, r_vec, scales, keep, per, q, d, Q, params, offsets, scalars, polar, modes,
                                   write_pairs, count, select)
        ctx.save_for_backward(r_vec, q, d, Q, offsets, i_, k_, scales, keep, per, params, scalars)
        ctx.flags = (polar, modes, count, select)
        empty = r_vec.new_zeros(0)
        return (out["e_node"], out["field_d"], out["field_p"], out["du"],
                empty if out["e_pair"] is None else out["e_pair"])

    @staticmethod
    def backward(ctx, g_enode, g_fd, g_fp, g_du, g_epair):
        r_vec, q, d, Q, offsets, i_, k_, scales, keep, per, params, scalars = ctx.saved_tensors
        polar, modes, count, select = ctx.flags
        consts = (i_, k_, scales, keep, per, scalars, polar, modes, count, select)
        zero = r_vec.new_zeros(0)
        grads = _AmoebaPairsGrad.apply(r_vec, q, d, Q, offsets, params, g_enode, g_fd if polar else zero,
                                       g_fp if polar else zero, g_du if polar else zero, consts)
        g_params = None
        if ctx.needs_input_grad[10]:
            g_params = amoeba_param_grad(i_, k_, r_vec.detach(), scales, keep, per, q.detach(), d.detach(), Q.detach(),
                                         params.detach(), offsets.detach(), scalars, polar, modes, g_enode.detach(),
                                         _given(g_fd if polar else zero), _given(g_fp if polar else zero),
                                         _given(g_du if polar else zero), count, select)
        return (*grads, None, None, None, None, None, g_params, None, None, None, None, None, None)


class _AmoebaPairsGrad(torch.autograd.Function):
    """The backward kernel of :class:`AmoebaPairs`, differentiable once more (see :class:`_HippoPairsGrad`)."""

    @staticmethod
    def forward(ctx, r_vec, q, d, Q, offsets, params, g_enode, g_fd, g_fp, g_du, consts):
        i_, k_, scales, keep, per, scalars, polar, modes, count, select = consts
        grads = amoeba_pairs_backward(i_, k_, r_vec, scales, keep, per, q, d, Q, params, offsets, scalars, polar,
                                      modes, g_node=g_enode, g_field_d=_given(g_fd), g_field_p=_given(g_fp),
                                      g_du=_given(g_du), count=count, select=select)
        ctx.save_for_backward(r_vec, q, d, Q, offsets, params, g_enode, g_fd, g_fp, g_du)
        ctx.consts = consts
        return grads["grad_r"], grads["grad_q"], grads["grad_d"], grads["grad_Q"], grads["grad_off"]

    @staticmethod
    @once_differentiable
    def backward(ctx, u_r, u_q, u_d, u_Q, u_off):
        r_vec, q, d, Q, offsets, params, g_enode, g_fd, g_fp, g_du = ctx.saved_tensors
        i_, k_, scales, keep, per, scalars, polar, modes, count, select = ctx.consts
        tangents = tuple(torch.zeros_like(x) if u is None else u
                         for u, x in ((u_r, r_vec), (u_q, q), (u_d, d), (u_Q, Q), (u_off, offsets)))
        fwd = amoeba_pairs_forward(i_, k_, r_vec, scales, keep, per, q, d, Q, params, offsets, scalars, polar, modes,
                                   False, count, select, tangents=tangents)
        bwd = amoeba_pairs_backward(i_, k_, r_vec, scales, keep, per, q, d, Q, params, offsets, scalars, polar, modes,
                                    g_node=g_enode, g_field_d=_given(g_fd), g_field_p=_given(g_fp), g_du=_given(g_du),
                                    count=count, select=select, tangents=tangents)

        def matching(t, like):
            return None if like.numel() == 0 or t is None else t.reshape(like.shape)

        g_params = None
        if ctx.needs_input_grad[5]:
            g_params = amoeba_param_grad(i_, k_, r_vec, scales, keep, per, q, d, Q, params, offsets, scalars, polar,
                                         modes, g_enode, _given(g_fd), _given(g_fp), _given(g_du), count, select,
                                         tangents=tangents)[1]
        return (bwd["grad_r_t"], bwd["grad_q_t"], bwd["grad_d_t"], bwd["grad_Q_t"], bwd["grad_off_t"], g_params,
                fwd["e_node_t"], matching(fwd["field_d_t"], g_fd), matching(fwd["field_p_t"], g_fp),
                matching(fwd["du_t"], g_du), None)


def amoeba_scalar_table(coulomb: float, alpha_e: float, alpha_p: float, vdw_window: tuple, ewald_cutoff: float,
                        delta: float, gamma: float, tiny: float, device: torch.device) -> Tensor:
    """The float64 scalars of an AMOEBA evaluation (:data:`AMOEBA_SCALARS`); ``vdw_window`` is the
    ``(taper, cutoff)`` of the van der Waals switch (zeros without one)."""
    values = [coulomb, 1.0 / math.sqrt(math.pi), alpha_e, alpha_p, *vdw_window, ewald_cutoff, delta, gamma, tiny,
              1.0 / 6.0, 0.6, 18.0 / 35.0, 9.0 / 35.0]
    return torch.tensor(values, dtype=torch.float64, device=device)


def available() -> bool:
    """Whether the kernels can run here (a CUDA device and Triton)."""
    return torch.cuda.is_available()


# the particle mesh in fractional coordinates: a spline kernel tabulates the 1D weights and their
# first three derivatives of every atom once per plan, the spread and gather kernels loop over the
# p^3 stencil points forming the 3D products in registers, and the adjoint kernel returns the
# gradient with respect to the scaled coordinates directly, so no autograd runs through the splines.
# The FFTs stay in PyTorch. Sources and outputs are unordered multipole components in fractional
# coordinates: (c0), (c1_x, c1_y, c1_z), (c2_xx, c2_yy, c2_zz, c2_xy, c2_xz, c2_yz); the caller folds
# the factor two of a symmetric quadrupole's off-diagonal entries into c2.

MESH_BLOCK = 32
#: the Cartesian mesh kernels: atoms per program and warps, one lane per stencil point
SPREAD_BLOCK, SPREAD_WARPS = 4, 4


def _lanes(P: int) -> int:
    """The stencil lanes of the Cartesian mesh kernels: ``P^3`` padded to a power of two."""
    return triton.next_power_of_2(P * P * P)
PCG_BLOCK = 128
_SLOTS = 4                    # weight, first, second and third derivative per axis


@triton.jit
def _spline_axis(u, w0, w1, w2, w3, s_ptr, m, K: tl.constexpr, P: tl.constexpr):
    """One axis: the stencil start ``floor(u) mod K`` and, in the four slots of the atom's table,
    ``M_P(t + j)`` and its first three derivatives for ``j`` below ``P``, by the recursion
    ``M_n(x) = x / (n - 1) M_{n-1}(x) + (n - x) / (n - 1) M_{n-1}(x - 1)`` run in place in the
    weight slot, the orders ``P - 1``, ``P - 2`` and ``P - 3`` copied aside for the differences."""
    base = libdevice.floor(u)
    t = u - base
    s = base.to(tl.int32) % K
    s = tl.where(s < 0, s + K, s)
    tl.store(s_ptr, s, mask=m)
    one = t * 0.0 + 1.0
    tl.store(w0, one, mask=m)
    if P == 4:
        tl.store(w3, one, mask=m)
    for nn in tl.static_range(2, P + 1):
        for j in tl.static_range(nn - 1, -1, -1):
            val = t * 0.0
            if j < nn - 1:
                val += (t + j) / (nn - 1) * tl.load(w0 + j, mask=m, other=0.0)
            if j >= 1:
                val += (nn - t - j) / (nn - 1) * tl.load(w0 + j - 1, mask=m, other=0.0)
            tl.store(w0 + j, val, mask=m)
        if nn == P - 3:
            for j in tl.static_range(nn):
                tl.store(w3 + j, tl.load(w0 + j, mask=m, other=0.0), mask=m)
        if nn == P - 2:
            for j in tl.static_range(nn):
                tl.store(w2 + j, tl.load(w0 + j, mask=m, other=0.0), mask=m)
        if nn == P - 1:
            for j in tl.static_range(nn):
                tl.store(w1 + j, tl.load(w0 + j, mask=m, other=0.0), mask=m)
    # M_P' = M_{P-1}(x) - M_{P-1}(x - 1), and the binomial differences of the lower orders for
    # the second and third derivatives, each formed from the highest index down, in place
    for j in tl.static_range(P - 1, -1, -1):
        val = t * 0.0
        if j < P - 1:
            val += tl.load(w1 + j, mask=m, other=0.0)
        if j >= 1:
            val -= tl.load(w1 + j - 1, mask=m, other=0.0)
        tl.store(w1 + j, val, mask=m)
    for j in tl.static_range(P - 1, -1, -1):
        val = t * 0.0
        if j < P - 2:
            val += tl.load(w2 + j, mask=m, other=0.0)
        if j >= 1 and j - 1 < P - 2:
            val -= 2.0 * tl.load(w2 + j - 1, mask=m, other=0.0)
        if j >= 2:
            val += tl.load(w2 + j - 2, mask=m, other=0.0)
        tl.store(w2 + j, val, mask=m)
    for j in tl.static_range(P - 1, -1, -1):
        val = t * 0.0
        if j < P - 3:
            val += tl.load(w3 + j, mask=m, other=0.0)
        if j >= 1 and j - 1 < P - 3:
            val -= 3.0 * tl.load(w3 + j - 1, mask=m, other=0.0)
        if j >= 2 and j - 2 < P - 3:
            val += 3.0 * tl.load(w3 + j - 2, mask=m, other=0.0)
        if j >= 3:
            val -= tl.load(w3 + j - 3, mask=m, other=0.0)
        tl.store(w3 + j, val, mask=m)


@triton.jit
def _splines(u_ptr, w_ptr, start_ptr, n_atoms, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
             P: tl.constexpr, BLOCK: tl.constexpr):
    """The spline tables ``(N, 3, 4, P)`` and stencil starts ``(N, 3)`` of the scaled coordinates ``u``."""
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    row = w_ptr + n * (3 * 4 * P)
    ux = tl.load(u_ptr + n * 3, mask=m, other=0.0)
    uy = tl.load(u_ptr + n * 3 + 1, mask=m, other=0.0)
    uz = tl.load(u_ptr + n * 3 + 2, mask=m, other=0.0)
    _spline_axis(ux, row, row + P, row + 2 * P, row + 3 * P, start_ptr + n * 3, m, K1, P)
    _spline_axis(uy, row + 4 * P, row + 5 * P, row + 6 * P, row + 7 * P, start_ptr + n * 3 + 1, m, K2, P)
    _spline_axis(uz, row + 8 * P, row + 9 * P, row + 10 * P, row + 11 * P, start_ptr + n * 3 + 2, m, K3, P)


@triton.jit
def _coef(c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz, wx, dx, ddx, wy, dy, ddy, wz, dz, ddz,
          ORD: tl.constexpr):
    """The coefficient of one stencil point: ``c0 W + c1 . grad W + c2 : grad grad W``."""
    val = c0 * wx * wy * wz
    if ORD >= 1:
        val += c1x * dx * wy * wz + c1y * wx * dy * wz + c1z * wx * wy * dz
    if ORD >= 2:
        val += (c2xx * ddx * wy * wz + c2yy * wx * ddy * wz + c2zz * wx * wy * ddz
                + c2xy * dx * dy * wz + c2xz * dx * wy * dz + c2yz * wx * dy * dz)
    return val


@triton.jit
def _dcoef(c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz, wx, dx, ddx, dddx, wy, dy, ddy, dddy,
           wz, dz, ddz, dddz, ORD: tl.constexpr):
    """The three derivatives of :func:`_coef` with respect to the scaled coordinates of the atom."""
    gx = c0 * dx * wy * wz
    gy = c0 * wx * dy * wz
    gz = c0 * wx * wy * dz
    if ORD >= 1:
        gx += c1x * ddx * wy * wz + c1y * dx * dy * wz + c1z * dx * wy * dz
        gy += c1x * dx * dy * wz + c1y * wx * ddy * wz + c1z * wx * dy * dz
        gz += c1x * dx * wy * dz + c1y * wx * dy * dz + c1z * wx * wy * ddz
    if ORD >= 2:
        gx += (c2xx * dddx * wy * wz + c2yy * dx * ddy * wz + c2zz * dx * wy * ddz
               + c2xy * ddx * dy * wz + c2xz * ddx * wy * dz + c2yz * dx * dy * dz)
        gy += (c2xx * ddx * dy * wz + c2yy * wx * dddy * wz + c2zz * wx * dy * ddz
               + c2xy * dx * ddy * wz + c2xz * dx * dy * dz + c2yz * wx * ddy * dz)
        gz += (c2xx * ddx * wy * dz + c2yy * wx * ddy * dz + c2zz * wx * wy * dddz
               + c2xy * dx * dy * dz + c2xz * dx * wy * ddz + c2yz * wx * dy * ddz)
    return gx, gy, gz


@triton.jit
def _spread(c0_ptr, c1_ptr, c2_ptr, w_ptr, start_ptr, batch_ptr, grid_ptr, n_atoms,
            K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr, ORD: tl.constexpr,
            BLOCK: tl.constexpr):
    """``grid[batch, i - a, j - b, k - c] += coef(a, b, c)`` over the stencil of every atom."""
    acc = grid_ptr.dtype.element_ty
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    c0 = tl.load(c0_ptr + n, mask=m, other=0.0)
    zero = c0 * 0.0
    c1x = zero
    c1y = zero
    c1z = zero
    c2xx = zero
    c2yy = zero
    c2zz = zero
    c2xy = zero
    c2xz = zero
    c2yz = zero
    if ORD >= 1:
        c1x = tl.load(c1_ptr + n * 3, mask=m, other=0.0)
        c1y = tl.load(c1_ptr + n * 3 + 1, mask=m, other=0.0)
        c1z = tl.load(c1_ptr + n * 3 + 2, mask=m, other=0.0)
    if ORD >= 2:
        c2xx = tl.load(c2_ptr + n * 6, mask=m, other=0.0)
        c2yy = tl.load(c2_ptr + n * 6 + 1, mask=m, other=0.0)
        c2zz = tl.load(c2_ptr + n * 6 + 2, mask=m, other=0.0)
        c2xy = tl.load(c2_ptr + n * 6 + 3, mask=m, other=0.0)
        c2xz = tl.load(c2_ptr + n * 6 + 4, mask=m, other=0.0)
        c2yz = tl.load(c2_ptr + n * 6 + 5, mask=m, other=0.0)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    row = w_ptr + n * (3 * 4 * P)
    for a in tl.static_range(P):
        wx = tl.load(row + a, mask=m, other=0.0)
        dx = tl.load(row + P + a, mask=m, other=0.0)
        ddx = tl.load(row + 2 * P + a, mask=m, other=0.0)
        gx = ((ix - a + K1) % K1) * (K2 * K3)
        for b in tl.static_range(P):
            wy = tl.load(row + 4 * P + b, mask=m, other=0.0)
            dy = tl.load(row + 5 * P + b, mask=m, other=0.0)
            ddy = tl.load(row + 6 * P + b, mask=m, other=0.0)
            gy = ((iy - b + K2) % K2) * K3
            for c in tl.static_range(P):
                wz = tl.load(row + 8 * P + c, mask=m, other=0.0)
                dz = tl.load(row + 9 * P + c, mask=m, other=0.0)
                ddz = tl.load(row + 10 * P + c, mask=m, other=0.0)
                gz = (iz - c + K3) % K3
                val = _coef(c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz,
                            wx, dx, ddx, wy, dy, ddy, wz, dz, ddz, ORD)
                tl.atomic_add(grid_ptr + base + gx + gy + gz, val.to(acc), mask=m, sem="relaxed")


@triton.jit
def _gather(grid_ptr, w_ptr, start_ptr, batch_ptr, out_ptr, n_atoms,
            K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr, DERIV: tl.constexpr,
            BLOCK: tl.constexpr):
    """The potential at the atoms and its fractional derivatives up to ``DERIV``:
    ``out[n] = (phi, d_x, d_y, d_z, h_xx, h_yy, h_zz, h_xy, h_xz, h_yz)``."""
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    row = w_ptr + n * (3 * 4 * P)
    zero = ix.to(grid_ptr.dtype.element_ty) * 0.0
    phi = zero
    d0 = zero
    d1 = zero
    d2 = zero
    h00 = zero
    h11 = zero
    h22 = zero
    h01 = zero
    h02 = zero
    h12 = zero
    for a in tl.static_range(P):
        wx = tl.load(row + a, mask=m, other=0.0)
        dx = tl.load(row + P + a, mask=m, other=0.0)
        ddx = tl.load(row + 2 * P + a, mask=m, other=0.0)
        gx = ((ix - a + K1) % K1) * (K2 * K3)
        for b in tl.static_range(P):
            wy = tl.load(row + 4 * P + b, mask=m, other=0.0)
            dy = tl.load(row + 5 * P + b, mask=m, other=0.0)
            ddy = tl.load(row + 6 * P + b, mask=m, other=0.0)
            gy = ((iy - b + K2) % K2) * K3
            for c in tl.static_range(P):
                wz = tl.load(row + 8 * P + c, mask=m, other=0.0)
                dz = tl.load(row + 9 * P + c, mask=m, other=0.0)
                ddz = tl.load(row + 10 * P + c, mask=m, other=0.0)
                v = tl.load(grid_ptr + base + gx + gy + ((iz - c + K3) % K3), mask=m, other=0.0)
                phi += v * wx * wy * wz
                if DERIV >= 1:
                    d0 += v * dx * wy * wz
                    d1 += v * wx * dy * wz
                    d2 += v * wx * wy * dz
                if DERIV >= 2:
                    h00 += v * ddx * wy * wz
                    h11 += v * wx * ddy * wz
                    h22 += v * wx * wy * ddz
                    h01 += v * dx * dy * wz
                    h02 += v * dx * wy * dz
                    h12 += v * wx * dy * dz
    tl.store(out_ptr + n * 10, phi, mask=m)
    if DERIV >= 1:
        tl.store(out_ptr + n * 10 + 1, d0, mask=m)
        tl.store(out_ptr + n * 10 + 2, d1, mask=m)
        tl.store(out_ptr + n * 10 + 3, d2, mask=m)
    if DERIV >= 2:
        tl.store(out_ptr + n * 10 + 4, h00, mask=m)
        tl.store(out_ptr + n * 10 + 5, h11, mask=m)
        tl.store(out_ptr + n * 10 + 6, h22, mask=m)
        tl.store(out_ptr + n * 10 + 7, h01, mask=m)
        tl.store(out_ptr + n * 10 + 8, h02, mask=m)
        tl.store(out_ptr + n * 10 + 9, h12, mask=m)


@triton.jit
def _adjoint(ga_ptr, gb_ptr, a_ptr, b_ptr, w_ptr, start_ptr, batch_ptr, gu_ptr, n_atoms,
             K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr, ORDA: tl.constexpr,
             ORDB: tl.constexpr, THIRD: tl.constexpr, BLOCK: tl.constexpr):
    """``gu[n] = sum_points grid_a[point] d coef_a(point) / du + grid_b[point] d coef_b(point) / du``
    for two grids and two coefficient sets ``(N, 10)`` (components up to ``ORDA`` and ``ORDB``)
    sharing the stencil of every atom: the gradient of a convolution with respect to the scaled
    coordinates, the probe side on the potential and the source side on the adjoint potential."""
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    a0 = tl.load(a_ptr + n * 10, mask=m, other=0.0)
    zero = a0 * 0.0
    a1x = zero
    a1y = zero
    a1z = zero
    a2xx = zero
    a2yy = zero
    a2zz = zero
    a2xy = zero
    a2xz = zero
    a2yz = zero
    if ORDA >= 1:
        a1x = tl.load(a_ptr + n * 10 + 1, mask=m, other=0.0)
        a1y = tl.load(a_ptr + n * 10 + 2, mask=m, other=0.0)
        a1z = tl.load(a_ptr + n * 10 + 3, mask=m, other=0.0)
    if ORDA >= 2:
        a2xx = tl.load(a_ptr + n * 10 + 4, mask=m, other=0.0)
        a2yy = tl.load(a_ptr + n * 10 + 5, mask=m, other=0.0)
        a2zz = tl.load(a_ptr + n * 10 + 6, mask=m, other=0.0)
        a2xy = tl.load(a_ptr + n * 10 + 7, mask=m, other=0.0)
        a2xz = tl.load(a_ptr + n * 10 + 8, mask=m, other=0.0)
        a2yz = tl.load(a_ptr + n * 10 + 9, mask=m, other=0.0)
    b0 = tl.load(b_ptr + n * 10, mask=m, other=0.0)
    b1x = zero
    b1y = zero
    b1z = zero
    b2xx = zero
    b2yy = zero
    b2zz = zero
    b2xy = zero
    b2xz = zero
    b2yz = zero
    if ORDB >= 1:
        b1x = tl.load(b_ptr + n * 10 + 1, mask=m, other=0.0)
        b1y = tl.load(b_ptr + n * 10 + 2, mask=m, other=0.0)
        b1z = tl.load(b_ptr + n * 10 + 3, mask=m, other=0.0)
    if ORDB >= 2:
        b2xx = tl.load(b_ptr + n * 10 + 4, mask=m, other=0.0)
        b2yy = tl.load(b_ptr + n * 10 + 5, mask=m, other=0.0)
        b2zz = tl.load(b_ptr + n * 10 + 6, mask=m, other=0.0)
        b2xy = tl.load(b_ptr + n * 10 + 7, mask=m, other=0.0)
        b2xz = tl.load(b_ptr + n * 10 + 8, mask=m, other=0.0)
        b2yz = tl.load(b_ptr + n * 10 + 9, mask=m, other=0.0)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    row = w_ptr + n * (3 * 4 * P)
    gux = zero
    guy = zero
    guz = zero
    for a in tl.static_range(P):
        wx = tl.load(row + a, mask=m, other=0.0)
        dx = tl.load(row + P + a, mask=m, other=0.0)
        ddx = tl.load(row + 2 * P + a, mask=m, other=0.0)
        dddx = zero
        if THIRD:
            dddx = tl.load(row + 3 * P + a, mask=m, other=0.0)
        gx = ((ix - a + K1) % K1) * (K2 * K3)
        for b in tl.static_range(P):
            wy = tl.load(row + 4 * P + b, mask=m, other=0.0)
            dy = tl.load(row + 5 * P + b, mask=m, other=0.0)
            ddy = tl.load(row + 6 * P + b, mask=m, other=0.0)
            dddy = zero
            if THIRD:
                dddy = tl.load(row + 7 * P + b, mask=m, other=0.0)
            gy = ((iy - b + K2) % K2) * K3
            for c in tl.static_range(P):
                wz = tl.load(row + 8 * P + c, mask=m, other=0.0)
                dz = tl.load(row + 9 * P + c, mask=m, other=0.0)
                ddz = tl.load(row + 10 * P + c, mask=m, other=0.0)
                dddz = zero
                if THIRD:
                    dddz = tl.load(row + 11 * P + c, mask=m, other=0.0)
                at = base + gx + gy + ((iz - c + K3) % K3)
                va = tl.load(ga_ptr + at, mask=m, other=0.0)
                vb = tl.load(gb_ptr + at, mask=m, other=0.0)
                px, py, pz = _dcoef(a0, a1x, a1y, a1z, a2xx, a2yy, a2zz, a2xy, a2xz, a2yz,
                                    wx, dx, ddx, dddx, wy, dy, ddy, dddy, wz, dz, ddz, dddz, ORDA)
                sx, sy, sz = _dcoef(b0, b1x, b1y, b1z, b2xx, b2yy, b2zz, b2xy, b2xz, b2yz,
                                    wx, dx, ddx, dddx, wy, dy, ddy, dddy, wz, dz, ddz, dddz, ORDB)
                gux += va * px + vb * sx
                guy += va * py + vb * sy
                guz += va * pz + vb * sz
    tl.store(gu_ptr + n * 3, gux, mask=m)
    tl.store(gu_ptr + n * 3 + 1, guy, mask=m)
    tl.store(gu_ptr + n * 3 + 2, guz, mask=m)


def spline_tables(u: Tensor, mesh: list, order: int) -> tuple:
    """The spline tables ``(N, 3, 4, P)`` (weights and three derivatives per axis) and the
    stencil starts ``(N, 3)`` of scaled fractional coordinates ``u`` ``(N, 3)``."""
    N, P = int(u.shape[0]), int(order)
    w = torch.empty(N, 3, _SLOTS, P, dtype=u.dtype, device=u.device)
    start = torch.empty(N, 3, dtype=torch.int32, device=u.device)
    if N:
        _splines[(triton.cdiv(N, MESH_BLOCK),)](u.contiguous(), w, start, N, K1=int(mesh[0]), K2=int(mesh[1]),
                                                K3=int(mesh[2]), P=P, BLOCK=MESH_BLOCK, num_warps=2)
    return w, start


def _components(c0: Tensor, c1, c2) -> Tensor:
    """The components ``(N, 10)`` of a coefficient set, zero where absent."""
    out = c0.new_zeros(c0.shape[0], 10)
    out[:, 0] = c0
    if c1 is not None:
        out[:, 1:4] = c1
    if c2 is not None:
        out[:, 4:10] = c2
    return out


def spread(c0: Tensor, c1: Tensor, c2: Tensor, w: Tensor, start: Tensor, batch: Tensor, mesh: list, groups: int,
           order: int) -> Tensor:
    """The grid ``(groups, K1, K2, K3)`` of coefficients up to ``order`` (0, 1 or 2)."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    grid = torch.zeros(groups * K1 * K2 * K3, dtype=w.dtype, device=w.device)
    if N:
        _spread[(triton.cdiv(N, MESH_BLOCK),)](c0.contiguous(), c1.contiguous(), c2.contiguous(), w, start,
                                               batch.contiguous(), grid, N, K1=K1, K2=K2, K3=K3, P=P,
                                               ORD=int(order), BLOCK=MESH_BLOCK, num_warps=2)
    return grid.view(groups, K1, K2, K3)


def gather(grid: Tensor, w: Tensor, start: Tensor, batch: Tensor, mesh: list, deriv: int) -> Tensor:
    """``(N, 10)``: the potential and its fractional derivatives up to ``deriv`` (0, 1 or 2)."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    out = torch.zeros(N, 10, dtype=w.dtype, device=w.device)
    if N:
        _gather[(triton.cdiv(N, MESH_BLOCK),)](grid.contiguous(), w, start, batch.contiguous(), out, N, K1=K1, K2=K2,
                                               K3=K3, P=P, DERIV=int(deriv), BLOCK=MESH_BLOCK, num_warps=2)
    return out


def adjoint(grid_a: Tensor, coef_a: Tensor, order_a: int, grid_b: Tensor, coef_b: Tensor, order_b: int, w: Tensor,
            start: Tensor, batch: Tensor, mesh: list) -> Tensor:
    """The gradient ``(N, 3)`` with respect to the scaled coordinates of the two contractions
    ``sum_points grid[point] coef(point)``; the coefficient sets are ``(N, 10)``."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    gu = torch.empty(N, 3, dtype=w.dtype, device=w.device)
    if N:
        _adjoint[(triton.cdiv(N, MESH_BLOCK),)](grid_a.contiguous(), grid_b.contiguous(), coef_a.contiguous(),
                                                coef_b.contiguous(), w, start, batch.contiguous(), gu, N,
                                                K1=K1, K2=K2, K3=K3, P=P, ORDA=int(order_a), ORDB=int(order_b),
                                                THIRD=bool(max(order_a, order_b) >= 2), BLOCK=MESH_BLOCK,
                                                num_warps=2)
    return gu


class MeshConvolution(torch.autograd.Function):
    """The mesh potential of multipole sources at the atoms that carry them, differentiable
    once with respect to the scaled coordinates, the components and the influence function.

    ``out = gather(irfftn(rfftn(spread(c)) * influence), deriv)``. The backward spreads the
    incoming gradient as a source set, convolves it to the adjoint potential ``psi``, and reads
    the gradient of the sources off ``psi`` and the gradient of the coordinates off both grids
    with one adjoint kernel: a second convolution, no autograd through the splines.
    """

    @staticmethod
    def forward(ctx, u, c0, c1, c2, influence, w, start, batch, mesh, groups, order, deriv):
        k1, k2, k3 = (int(k) for k in mesh)
        grid = spread(c0, c1, c2, w, start, batch, mesh, groups, order)
        spectrum = torch.fft.rfftn(grid, dim=(1, 2, 3))
        pot = torch.fft.irfftn(spectrum * influence, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
        out = gather(pot, w, start, batch, mesh, deriv)
        ctx.save_for_backward(c0, c1, c2, influence, w, start, batch, spectrum, pot)
        ctx.mesh, ctx.groups, ctx.order, ctx.deriv = list(mesh), int(groups), int(order), int(deriv)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, g_out):
        c0, c1, c2, influence, w, start, batch, spectrum, pot = ctx.saved_tensors
        mesh, order, deriv = ctx.mesh, ctx.order, ctx.deriv
        k1, k2, k3 = mesh
        g_out = g_out.contiguous()
        g_grid = spread(g_out[:, 0].contiguous(), g_out[:, 1:4].contiguous(), g_out[:, 4:10].contiguous(), w, start,
                        batch, mesh, ctx.groups, deriv)
        g_spectrum = torch.fft.rfftn(g_grid, dim=(1, 2, 3))
        psi = torch.fft.irfftn(g_spectrum * influence, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
        gs = gather(psi, w, start, batch, mesh, order)
        gu = adjoint(pot, g_out, deriv, psi, _components(c0, c1, c2), order, w, start, batch, mesh)
        g_influence = None
        if ctx.needs_input_grad[4]:
            # the influence function is an eager tensor of the cells: its gradient through the
            # inverse transform, with the spread gradient as the potential's cotangent
            with torch.enable_grad():
                inf = influence.detach().requires_grad_(True)
                pot_ = torch.fft.irfftn(spectrum * inf, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
                g_influence = torch.autograd.grad(pot_, inf, g_grid)[0]
        gc1 = gs[:, 1:4] if order >= 1 else None
        gc2 = gs[:, 4:10] if order >= 2 else None
        return gu, gs[:, 0], gc1, gc2, g_influence, None, None, None, None, None, None, None


# the same mesh pass with the Jacobian of the scaled coordinates applied in registers: Cartesian
# multipoles in, the potential with its Cartesian gradient and Hessian out, and the gradients of the
# multipoles returned in Cartesian form. It serves whenever the cell carries no gradient (the
# Jacobian is then a constant of the evaluation); the fractional pass above keeps the stress exact.

@triton.jit
def _jac(j_ptr, batch, m):
    """The nine entries ``J[beta, a]`` of the structure of every atom."""
    base = j_ptr + batch * 9
    return (tl.load(base, mask=m, other=0.0), tl.load(base + 1, mask=m, other=0.0),
            tl.load(base + 2, mask=m, other=0.0), tl.load(base + 3, mask=m, other=0.0),
            tl.load(base + 4, mask=m, other=0.0), tl.load(base + 5, mask=m, other=0.0),
            tl.load(base + 6, mask=m, other=0.0), tl.load(base + 7, mask=m, other=0.0),
            tl.load(base + 8, mask=m, other=0.0))


@triton.jit
def _frac_vec(j00, j01, j02, j10, j11, j12, j20, j21, j22, vx, vy, vz):
    """``f_a = sum_beta v_beta J[beta, a]``: a Cartesian dipole as fractional coefficients."""
    return (vx * j00 + vy * j10 + vz * j20, vx * j01 + vy * j11 + vz * j21, vx * j02 + vy * j12 + vz * j22)


@triton.jit
def _frac_quad(j00, j01, j02, j10, j11, j12, j20, j21, j22, sxx, sxy, sxz, syy, syz, szz):
    """``M = J^T S J`` of a symmetric ``S`` as the six coefficients ``(M00, M11, M22, 2 M01, 2 M02, 2 M12)``."""
    # the rows of S J
    r0a = sxx * j00 + sxy * j10 + sxz * j20
    r0b = sxx * j01 + sxy * j11 + sxz * j21
    r0c = sxx * j02 + sxy * j12 + sxz * j22
    r1a = sxy * j00 + syy * j10 + syz * j20
    r1b = sxy * j01 + syy * j11 + syz * j21
    r1c = sxy * j02 + syy * j12 + syz * j22
    r2a = sxz * j00 + syz * j10 + szz * j20
    r2b = sxz * j01 + syz * j11 + szz * j21
    r2c = sxz * j02 + syz * j12 + szz * j22
    m00 = j00 * r0a + j10 * r1a + j20 * r2a
    m11 = j01 * r0b + j11 * r1b + j21 * r2b
    m22 = j02 * r0c + j12 * r1c + j22 * r2c
    m01 = j00 * r0b + j10 * r1b + j20 * r2b
    m02 = j00 * r0c + j10 * r1c + j20 * r2c
    m12 = j01 * r0c + j11 * r1c + j21 * r2c
    return m00, m11, m22, 2.0 * m01, 2.0 * m02, 2.0 * m12


@triton.jit
def _cart_vec(j00, j01, j02, j10, j11, j12, j20, j21, j22, fa, fb, fc):
    """``v_beta = sum_a J[beta, a] f_a``: fractional derivatives as a Cartesian gradient."""
    return (j00 * fa + j01 * fb + j02 * fc, j10 * fa + j11 * fb + j12 * fc, j20 * fa + j21 * fb + j22 * fc)


@triton.jit
def _cart_quad(j00, j01, j02, j10, j11, j12, j20, j21, j22, h00, h11, h22, h01, h02, h12):
    """``H = J h J^T`` of the symmetric fractional second derivatives: ``(Hxx, Hxy, Hxz, Hyy, Hyz, Hzz)``."""
    # the rows of J h
    r0a = j00 * h00 + j01 * h01 + j02 * h02
    r0b = j00 * h01 + j01 * h11 + j02 * h12
    r0c = j00 * h02 + j01 * h12 + j02 * h22
    r1a = j10 * h00 + j11 * h01 + j12 * h02
    r1b = j10 * h01 + j11 * h11 + j12 * h12
    r1c = j10 * h02 + j11 * h12 + j12 * h22
    r2a = j20 * h00 + j21 * h01 + j22 * h02
    r2b = j20 * h01 + j21 * h11 + j22 * h12
    r2c = j20 * h02 + j21 * h12 + j22 * h22
    hxx = r0a * j00 + r0b * j01 + r0c * j02
    hxy = r0a * j10 + r0b * j11 + r0c * j12
    hxz = r0a * j20 + r0b * j21 + r0c * j22
    hyy = r1a * j10 + r1b * j11 + r1c * j12
    hyz = r1a * j20 + r1b * j21 + r1c * j22
    hzz = r2a * j20 + r2b * j21 + r2c * j22
    return hxx, hxy, hxz, hyy, hyz, hzz


@triton.jit
def _load_multipoles(q_ptr, d_ptr, Q_ptr, j_ptr, batch_ptr, n, m, ORD: tl.constexpr):
    """A Cartesian multipole set of order ``ORD`` as the ten fractional coefficients (zeros beyond)."""
    c0 = tl.load(q_ptr + n, mask=m, other=0.0)
    zero = c0 * 0.0
    c1x = zero
    c1y = zero
    c1z = zero
    c2xx = zero
    c2yy = zero
    c2zz = zero
    c2xy = zero
    c2xz = zero
    c2yz = zero
    if ORD >= 1:
        batch = tl.load(batch_ptr + n, mask=m, other=0)
        j00, j01, j02, j10, j11, j12, j20, j21, j22 = _jac(j_ptr, batch, m)
        dx = tl.load(d_ptr + n * 3, mask=m, other=0.0)
        dy = tl.load(d_ptr + n * 3 + 1, mask=m, other=0.0)
        dz = tl.load(d_ptr + n * 3 + 2, mask=m, other=0.0)
        c1x, c1y, c1z = _frac_vec(j00, j01, j02, j10, j11, j12, j20, j21, j22, dx, dy, dz)
        if ORD >= 2:
            # the symmetric part of the nine entries
            qxx = tl.load(Q_ptr + n * 9, mask=m, other=0.0)
            qyy = tl.load(Q_ptr + n * 9 + 4, mask=m, other=0.0)
            qzz = tl.load(Q_ptr + n * 9 + 8, mask=m, other=0.0)
            qxy = 0.5 * (tl.load(Q_ptr + n * 9 + 1, mask=m, other=0.0) + tl.load(Q_ptr + n * 9 + 3, mask=m, other=0.0))
            qxz = 0.5 * (tl.load(Q_ptr + n * 9 + 2, mask=m, other=0.0) + tl.load(Q_ptr + n * 9 + 6, mask=m, other=0.0))
            qyz = 0.5 * (tl.load(Q_ptr + n * 9 + 5, mask=m, other=0.0) + tl.load(Q_ptr + n * 9 + 7, mask=m, other=0.0))
            c2xx, c2yy, c2zz, c2xy, c2xz, c2yz = _frac_quad(j00, j01, j02, j10, j11, j12, j20, j21, j22,
                                                            qxx, qxy, qxz, qyy, qyz, qzz)
    return c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz


@triton.jit
def _spread_c(q_ptr, d_ptr, Q_ptr, j_ptr, w_ptr, start_ptr, batch_ptr, grid_ptr, n_atoms, guard_ptr,
              K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr, ORD: tl.constexpr,
              BLOCK: tl.constexpr, SP: tl.constexpr, GUARD: tl.constexpr):
    """:func:`_spread` of Cartesian multipoles ``q``, ``d`` ``(N, 3)``, ``Q`` ``(N, 9)``: ``BLOCK`` atoms
    by ``SP`` lanes, one per stencil point (``P^3`` padded to a power of two), so that the adds of one
    atom's z-row go to consecutive addresses in one instruction instead of a loop of serialized adds
    per lane.
    With ``GUARD`` the kernel does nothing once the conjugate-gradient scalars ``guard`` (a row of
    :func:`pcg_scalars`) say the solve has converged: a solve of fixed length then pays little for
    the iterations it did not need."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    acc = grid_ptr.dtype.element_ty
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz = _load_multipoles(q_ptr, d_ptr, Q_ptr, j_ptr, batch_ptr,
                                                                             n, m, ORD)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    s = tl.arange(0, SP)
    a = s // (P * P)
    b = (s // P) % P
    c = s % P
    mm = m[:, None] & (s < P * P * P)[None, :]
    row = w_ptr + n[:, None] * (3 * 4 * P)
    wx = tl.load(row + a[None, :], mask=mm, other=0.0)
    wy = tl.load(row + 4 * P + b[None, :], mask=mm, other=0.0)
    wz = tl.load(row + 8 * P + c[None, :], mask=mm, other=0.0)
    if ORD >= 1:
        dx = tl.load(row + P + a[None, :], mask=mm, other=0.0)
        dy = tl.load(row + 5 * P + b[None, :], mask=mm, other=0.0)
        dz = tl.load(row + 9 * P + c[None, :], mask=mm, other=0.0)
    else:
        dx = wx
        dy = wy
        dz = wz
    if ORD >= 2:
        ddx = tl.load(row + 2 * P + a[None, :], mask=mm, other=0.0)
        ddy = tl.load(row + 6 * P + b[None, :], mask=mm, other=0.0)
        ddz = tl.load(row + 10 * P + c[None, :], mask=mm, other=0.0)
    else:
        ddx = wx
        ddy = wy
        ddz = wz
    gx = ((ix[:, None] - a[None, :] + K1) % K1) * (K2 * K3)
    gy = ((iy[:, None] - b[None, :] + K2) % K2) * K3
    gz = (iz[:, None] - c[None, :] + K3) % K3
    val = _coef(c0[:, None], c1x[:, None], c1y[:, None], c1z[:, None], c2xx[:, None], c2yy[:, None], c2zz[:, None],
                c2xy[:, None], c2xz[:, None], c2yz[:, None], wx, dx, ddx, wy, dy, ddy, wz, dz, ddz, ORD)
    tl.atomic_add(grid_ptr + base[:, None] + gx + gy + gz, val.to(acc), mask=mm, sem="relaxed")


@triton.jit
def _gather_c(grid_ptr, j_ptr, w_ptr, start_ptr, batch_ptr, phi_ptr, grad_ptr, hess_ptr, d_ptr, coef_ptr, n_atoms,
              guard_ptr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr, DERIV: tl.constexpr,
              FIELD: tl.constexpr, BLOCK: tl.constexpr, SP: tl.constexpr, GUARD: tl.constexpr):
    """:func:`_gather` with the potential, its Cartesian gradient ``(N, 3)`` and Hessian ``(N, 9)``
    written to their own arrays, one lane per stencil point as in :func:`_spread_c`. With ``FIELD``
    (``DERIV`` 1) the kernel writes the mesh field of the dipoles ``d`` instead,
    ``grad_ptr += coef d - gradient``, in place on the accumulator ``grad_ptr`` (the real-space field
    of the induction solve), and stores no potential; ``coef`` is read from a float64 scalar on the
    device (a float argument would be a float32 constant).
    With ``GUARD`` the kernel does nothing once the conjugate-gradient scalars ``guard`` (a row of
    :func:`pcg_scalars`) say the solve has converged: a solve of fixed length then pays little for
    the iterations it did not need."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    batch = tl.load(batch_ptr + n, mask=m, other=0)
    base = batch * (K1 * K2 * K3)
    s = tl.arange(0, SP)
    a = s // (P * P)
    b = (s // P) % P
    c = s % P
    mm = m[:, None] & (s < P * P * P)[None, :]
    row = w_ptr + n[:, None] * (3 * 4 * P)
    wx = tl.load(row + a[None, :], mask=mm, other=0.0)
    wy = tl.load(row + 4 * P + b[None, :], mask=mm, other=0.0)
    wz = tl.load(row + 8 * P + c[None, :], mask=mm, other=0.0)
    gx = ((ix[:, None] - a[None, :] + K1) % K1) * (K2 * K3)
    gy = ((iy[:, None] - b[None, :] + K2) % K2) * K3
    gz = (iz[:, None] - c[None, :] + K3) % K3
    v = tl.load(grid_ptr + base[:, None] + gx + gy + gz, mask=mm, other=0.0)
    if not FIELD:
        tl.store(phi_ptr + n, tl.sum(v * wx * wy * wz, axis=1), mask=m)
    if DERIV >= 1:
        dx = tl.load(row + P + a[None, :], mask=mm, other=0.0)
        dy = tl.load(row + 5 * P + b[None, :], mask=mm, other=0.0)
        dz = tl.load(row + 9 * P + c[None, :], mask=mm, other=0.0)
        d0 = tl.sum(v * dx * wy * wz, axis=1)
        d1 = tl.sum(v * wx * dy * wz, axis=1)
        d2 = tl.sum(v * wx * wy * dz, axis=1)
        j00, j01, j02, j10, j11, j12, j20, j21, j22 = _jac(j_ptr, batch, m)
        gx_, gy_, gz_ = _cart_vec(j00, j01, j02, j10, j11, j12, j20, j21, j22, d0, d1, d2)
        if FIELD:
            coef = tl.load(coef_ptr).to(gx_.dtype)
            gx_ = tl.load(grad_ptr + n * 3, mask=m, other=0.0) + coef * tl.load(d_ptr + n * 3, mask=m, other=0.0) - gx_
            gy_ = (tl.load(grad_ptr + n * 3 + 1, mask=m, other=0.0)
                   + coef * tl.load(d_ptr + n * 3 + 1, mask=m, other=0.0) - gy_)
            gz_ = (tl.load(grad_ptr + n * 3 + 2, mask=m, other=0.0)
                   + coef * tl.load(d_ptr + n * 3 + 2, mask=m, other=0.0) - gz_)
        tl.store(grad_ptr + n * 3, gx_, mask=m)
        tl.store(grad_ptr + n * 3 + 1, gy_, mask=m)
        tl.store(grad_ptr + n * 3 + 2, gz_, mask=m)
        if DERIV >= 2:
            ddx = tl.load(row + 2 * P + a[None, :], mask=mm, other=0.0)
            ddy = tl.load(row + 6 * P + b[None, :], mask=mm, other=0.0)
            ddz = tl.load(row + 10 * P + c[None, :], mask=mm, other=0.0)
            h00 = tl.sum(v * ddx * wy * wz, axis=1)
            h11 = tl.sum(v * wx * ddy * wz, axis=1)
            h22 = tl.sum(v * wx * wy * ddz, axis=1)
            h01 = tl.sum(v * dx * dy * wz, axis=1)
            h02 = tl.sum(v * dx * wy * dz, axis=1)
            h12 = tl.sum(v * wx * dy * dz, axis=1)
            hxx, hxy, hxz, hyy, hyz, hzz = _cart_quad(j00, j01, j02, j10, j11, j12, j20, j21, j22,
                                                      h00, h11, h22, h01, h02, h12)
            tl.store(hess_ptr + n * 9, hxx, mask=m)
            tl.store(hess_ptr + n * 9 + 1, hxy, mask=m)
            tl.store(hess_ptr + n * 9 + 2, hxz, mask=m)
            tl.store(hess_ptr + n * 9 + 3, hxy, mask=m)
            tl.store(hess_ptr + n * 9 + 4, hyy, mask=m)
            tl.store(hess_ptr + n * 9 + 5, hyz, mask=m)
            tl.store(hess_ptr + n * 9 + 6, hxz, mask=m)
            tl.store(hess_ptr + n * 9 + 7, hyz, mask=m)
            tl.store(hess_ptr + n * 9 + 8, hzz, mask=m)


@triton.jit
def _adjoint_c(ga_ptr, gb_ptr, aq_ptr, ad_ptr, aQ_ptr, bq_ptr, bd_ptr, bQ_ptr, j_ptr, w_ptr, start_ptr, batch_ptr,
               gu_ptr, n_atoms, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr,
               ORDA: tl.constexpr, ORDB: tl.constexpr, THIRD: tl.constexpr, BLOCK: tl.constexpr, SP: tl.constexpr):
    """:func:`_adjoint` with both coefficient sets given as Cartesian multipoles, one lane per stencil
    point as in :func:`_spread_c`."""
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    a0, a1x, a1y, a1z, a2xx, a2yy, a2zz, a2xy, a2xz, a2yz = _load_multipoles(aq_ptr, ad_ptr, aQ_ptr, j_ptr,
                                                                             batch_ptr, n, m, ORDA)
    b0, b1x, b1y, b1z, b2xx, b2yy, b2zz, b2xy, b2xz, b2yz = _load_multipoles(bq_ptr, bd_ptr, bQ_ptr, j_ptr,
                                                                             batch_ptr, n, m, ORDB)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    s = tl.arange(0, SP)
    a = s // (P * P)
    b = (s // P) % P
    c = s % P
    mm = m[:, None] & (s < P * P * P)[None, :]
    row = w_ptr + n[:, None] * (3 * 4 * P)
    wx = tl.load(row + a[None, :], mask=mm, other=0.0)
    dx = tl.load(row + P + a[None, :], mask=mm, other=0.0)
    ddx = tl.load(row + 2 * P + a[None, :], mask=mm, other=0.0)
    wy = tl.load(row + 4 * P + b[None, :], mask=mm, other=0.0)
    dy = tl.load(row + 5 * P + b[None, :], mask=mm, other=0.0)
    ddy = tl.load(row + 6 * P + b[None, :], mask=mm, other=0.0)
    wz = tl.load(row + 8 * P + c[None, :], mask=mm, other=0.0)
    dz = tl.load(row + 9 * P + c[None, :], mask=mm, other=0.0)
    ddz = tl.load(row + 10 * P + c[None, :], mask=mm, other=0.0)
    if THIRD:
        dddx = tl.load(row + 3 * P + a[None, :], mask=mm, other=0.0)
        dddy = tl.load(row + 7 * P + b[None, :], mask=mm, other=0.0)
        dddz = tl.load(row + 11 * P + c[None, :], mask=mm, other=0.0)
    else:
        dddx = wx * 0.0
        dddy = dddx
        dddz = dddx
    gx = ((ix[:, None] - a[None, :] + K1) % K1) * (K2 * K3)
    gy = ((iy[:, None] - b[None, :] + K2) % K2) * K3
    gz = (iz[:, None] - c[None, :] + K3) % K3
    at = base[:, None] + gx + gy + gz
    va = tl.load(ga_ptr + at, mask=mm, other=0.0)
    vb = tl.load(gb_ptr + at, mask=mm, other=0.0)
    px, py, pz = _dcoef(a0[:, None], a1x[:, None], a1y[:, None], a1z[:, None], a2xx[:, None], a2yy[:, None],
                        a2zz[:, None], a2xy[:, None], a2xz[:, None], a2yz[:, None],
                        wx, dx, ddx, dddx, wy, dy, ddy, dddy, wz, dz, ddz, dddz, ORDA)
    sx, sy, sz = _dcoef(b0[:, None], b1x[:, None], b1y[:, None], b1z[:, None], b2xx[:, None], b2yy[:, None],
                        b2zz[:, None], b2xy[:, None], b2xz[:, None], b2yz[:, None],
                        wx, dx, ddx, dddx, wy, dy, ddy, dddy, wz, dz, ddz, dddz, ORDB)
    tl.store(gu_ptr + n * 3, tl.sum(va * px + vb * sx, axis=1), mask=m)
    tl.store(gu_ptr + n * 3 + 1, tl.sum(va * py + vb * sy, axis=1), mask=m)
    tl.store(gu_ptr + n * 3 + 2, tl.sum(va * pz + vb * sz, axis=1), mask=m)


def _multipole_args(q: Tensor, d, Q, order: int) -> tuple:
    """The three arrays of a multipole set, the absent orders standing in as the charges."""
    N = int(q.shape[0])
    d_ = q if (order < 1 or d is None) else d.contiguous()
    Q_ = q if (order < 2 or Q is None) else Q.reshape(N, 9).contiguous()
    return q.contiguous(), d_, Q_


def spread_cartesian(q: Tensor, d, Q, jac: Tensor, w: Tensor, start: Tensor, batch: Tensor, mesh: list,
                     groups: int, order: int, guard: Optional[Tensor] = None) -> Tensor:
    """:func:`spread` of Cartesian multipoles with the structures' Jacobians ``jac`` ``(G, 3, 3)``; with
    the scalars ``guard`` of a conjugate-gradient iteration the grid stays zero once the solve has
    converged."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    grid = torch.zeros(groups * K1 * K2 * K3, dtype=w.dtype, device=w.device)
    if N:
        q_, d_, Q_ = _multipole_args(q, d, Q, order)
        _spread_c[(triton.cdiv(N, SPREAD_BLOCK),)](q_, d_, Q_, jac.contiguous(), w, start, batch.contiguous(), grid,
                                                   N, grid if guard is None else guard, K1=K1, K2=K2, K3=K3, P=P,
                                                   ORD=int(order), BLOCK=SPREAD_BLOCK, SP=_lanes(P),
                                                   GUARD=guard is not None, num_warps=SPREAD_WARPS)
    return grid.view(groups, K1, K2, K3)


def gather_cartesian(grid: Tensor, jac: Tensor, w: Tensor, start: Tensor, batch: Tensor, mesh: list,
                     deriv: int) -> tuple:
    """The potential ``(N,)``, its Cartesian gradient ``(N, 3)`` and Hessian ``(N, 3, 3)`` (zeros beyond ``deriv``)."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    # the kernel stores every entry up to its derivative order; the others are zeros
    phi = torch.empty(N, dtype=w.dtype, device=w.device)
    grad = (torch.empty if deriv >= 1 else torch.zeros)(N, 3, dtype=w.dtype, device=w.device)
    hess = (torch.empty if deriv >= 2 else torch.zeros)(N, 9, dtype=w.dtype, device=w.device)
    if N:
        _gather_c[(triton.cdiv(N, SPREAD_BLOCK),)](grid.contiguous(), jac.contiguous(), w, start, batch.contiguous(),
                                                 phi, grad, hess, phi, phi, N, phi, K1=K1, K2=K2, K3=K3, P=P,
                                                 DERIV=int(deriv), FIELD=False, BLOCK=SPREAD_BLOCK, SP=_lanes(P),
                                                 GUARD=False, num_warps=SPREAD_WARPS)
    return phi, grad, hess.view(N, 3, 3)


def gather_field(grid: Tensor, jac: Tensor, w: Tensor, start: Tensor, batch: Tensor, mesh: list, acc: Tensor,
                 d: Tensor, coef: float, guard: Optional[Tensor] = None) -> Tensor:
    """``acc += coef d - grad(potential)`` in place: the mesh part of the field of the dipoles ``d``
    (``acc`` ``(N, 3)`` holds their real-space field) with the self-field coefficient ``coef``; with
    the scalars ``guard`` of a conjugate-gradient iteration ``acc`` is left alone once the solve has
    converged."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    if N:
        _gather_c[(triton.cdiv(N, SPREAD_BLOCK),)](grid.contiguous(), jac.contiguous(), w, start, batch.contiguous(),
                                                 acc, acc, acc, d.contiguous(), _scalar(float(coef), acc.device), N,
                                                 acc if guard is None else guard, K1=K1, K2=K2, K3=K3, P=P, DERIV=1,
                                                 FIELD=True, BLOCK=SPREAD_BLOCK, SP=_lanes(P), GUARD=guard is not None,
                                                 num_warps=SPREAD_WARPS)
    return acc


@triton.jit
def _blocks_inverse(alpha_ptr, k_ptr, mask_ptr, out_ptr, n_atoms, BLOCK: tl.constexpr):
    """``out = alpha mask (I + K)^-1`` per atom by the adjugate (``I + K`` has a determinant of at
    least one): the diagonal blocks of the preconditioner with exchange polarization."""
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    s = tl.load(alpha_ptr + n, mask=m, other=0.0) * tl.load(mask_ptr + n, mask=m, other=0.0)
    a = 1.0 + tl.load(k_ptr + n * 9, mask=m, other=0.0)
    b = tl.load(k_ptr + n * 9 + 1, mask=m, other=0.0)
    c = tl.load(k_ptr + n * 9 + 2, mask=m, other=0.0)
    d = tl.load(k_ptr + n * 9 + 3, mask=m, other=0.0)
    e = 1.0 + tl.load(k_ptr + n * 9 + 4, mask=m, other=0.0)
    f = tl.load(k_ptr + n * 9 + 5, mask=m, other=0.0)
    g = tl.load(k_ptr + n * 9 + 6, mask=m, other=0.0)
    h = tl.load(k_ptr + n * 9 + 7, mask=m, other=0.0)
    i = 1.0 + tl.load(k_ptr + n * 9 + 8, mask=m, other=0.0)
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    s = s / det
    tl.store(out_ptr + n * 9, s * (e * i - f * h), mask=m)
    tl.store(out_ptr + n * 9 + 1, s * (c * h - b * i), mask=m)
    tl.store(out_ptr + n * 9 + 2, s * (b * f - c * e), mask=m)
    tl.store(out_ptr + n * 9 + 3, s * (f * g - d * i), mask=m)
    tl.store(out_ptr + n * 9 + 4, s * (a * i - c * g), mask=m)
    tl.store(out_ptr + n * 9 + 5, s * (c * d - a * f), mask=m)
    tl.store(out_ptr + n * 9 + 6, s * (d * h - e * g), mask=m)
    tl.store(out_ptr + n * 9 + 7, s * (b * g - a * h), mask=m)
    tl.store(out_ptr + n * 9 + 8, s * (a * e - b * d), mask=m)


def blocks_inverse(alpha: Tensor, expol: Tensor, mask: Tensor) -> Tensor:
    """``alpha mask (I + K)^-1`` ``(N, 3, 3)`` for the exchange-polarization tensors ``K`` ``(N, 3, 3)``
    (``_inverse3`` of the eager path, one launch)."""
    N = int(alpha.shape[0])
    out = torch.empty(N, 9, dtype=alpha.dtype, device=alpha.device)
    if N:
        _blocks_inverse[(triton.cdiv(N, PCG_BLOCK),)](alpha.contiguous(), expol.reshape(N, 9).contiguous(),
                                                      mask.contiguous(), out, N, BLOCK=PCG_BLOCK)
    return out.view(N, 3, 3)


def adjoint_cartesian(grid_a: Tensor, set_a: tuple, order_a: int, grid_b: Tensor, set_b: tuple, order_b: int,
                      jac: Tensor, w: Tensor, start: Tensor, batch: Tensor, mesh: list) -> Tensor:
    """:func:`adjoint` with Cartesian multipole sets ``(q, d, Q)``."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    gu = torch.empty(N, 3, dtype=w.dtype, device=w.device)
    if N:
        aq, ad, aQ = _multipole_args(set_a[0], set_a[1], set_a[2], order_a)
        bq, bd, bQ = _multipole_args(set_b[0], set_b[1], set_b[2], order_b)
        _adjoint_c[(triton.cdiv(N, SPREAD_BLOCK),)](grid_a.contiguous(), grid_b.contiguous(), aq, ad, aQ, bq, bd, bQ,
                                                  jac.contiguous(), w, start, batch.contiguous(), gu, N,
                                                  K1=K1, K2=K2, K3=K3, P=P, ORDA=int(order_a), ORDB=int(order_b),
                                                  THIRD=bool(max(order_a, order_b) >= 2), BLOCK=SPREAD_BLOCK,
                                                  SP=_lanes(P), num_warps=SPREAD_WARPS)
    return gu


class MeshConvolutionCartesian(torch.autograd.Function):
    """:class:`MeshConvolution` on Cartesian multipoles ``q``, ``d``, ``Q`` with the Jacobians of the
    structures applied inside the kernels; returns the potential, its gradient and its Hessian at the
    atoms. The Jacobians are constants (no cell gradient): the fractional function serves the stress."""

    @staticmethod
    def forward(ctx, u, q, d, Q, jac, batch, influence, w, start, mesh, groups, order, deriv):
        k1, k2, k3 = (int(k) for k in mesh)
        grid = spread_cartesian(q, d, Q, jac, w, start, batch, mesh, groups, order)
        spectrum = torch.fft.rfftn(grid, dim=(1, 2, 3))
        pot = torch.fft.irfftn(spectrum * influence, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
        phi, grad, hess = gather_cartesian(pot, jac, w, start, batch, mesh, deriv)
        ctx.save_for_backward(q, d if d is not None else q, Q if Q is not None else q, jac, batch, influence, w, start,
                              spectrum, pot)
        ctx.mesh, ctx.groups, ctx.order, ctx.deriv = list(mesh), int(groups), int(order), int(deriv)
        ctx.has = (d is not None, Q is not None)
        return phi, grad, hess

    @staticmethod
    @once_differentiable
    def backward(ctx, g_phi, g_grad, g_hess):
        q, d, Q, jac, batch, influence, w, start, spectrum, pot = ctx.saved_tensors
        mesh, order, deriv = ctx.mesh, ctx.order, ctx.deriv
        k1, k2, k3 = mesh
        has_d, has_Q = ctx.has
        # the spread kernel symmetrizes its quadrupole sources: the Hessian gradient goes in as it is
        g_set = (g_phi.contiguous(), g_grad if deriv >= 1 else None, g_hess if deriv >= 2 else None)
        g_grid = spread_cartesian(g_set[0], g_set[1], g_set[2], jac, w, start, batch, mesh, ctx.groups, deriv)
        g_spectrum = torch.fft.rfftn(g_grid, dim=(1, 2, 3))
        psi = torch.fft.irfftn(g_spectrum * influence, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
        gq, gd, gQ = gather_cartesian(psi, jac, w, start, batch, mesh, order)
        gu = adjoint_cartesian(pot, g_set, deriv, psi, (q, d if has_d else None, Q if has_Q else None), order,
                               jac, w, start, batch, mesh)
        g_influence = None
        if ctx.needs_input_grad[6]:
            with torch.enable_grad():
                inf = influence.detach().requires_grad_(True)
                pot_ = torch.fft.irfftn(spectrum * inf, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
                g_influence = torch.autograd.grad(pot_, inf, g_grid)[0]
        return (gu, gq, gd if (has_d and order >= 1) else None, gQ if (has_Q and order >= 2) else None, None, None,
                g_influence, None, None, None, None, None, None)


# the conjugate-gradient iteration of the induction solve on the device: the operator application
# with the first dot product, the update of the iterate with its two reductions, and the new
# direction with the convergence flag. The scalars of an iteration (the denominator, r.z, the new
# r.z, the squared step, the done flag) live in a float64 buffer that alternates between two copies,
# so no launch reads a value another program of the same launch writes, and the host reads nothing.

@triton.jit
def _pcg_apply(p_ptr, f_ptr, ap_ptr, inv_ptr, expol_ptr, mask_ptr, scal_ptr, n_atoms, EXPOL: tl.constexpr,
               BLOCK: tl.constexpr):
    """``ap = ((I + K) p / alpha - field) mask`` and ``scal[0] += p . ap``; nothing once the solve has
    converged (``ap`` then unset, read by no update)."""
    if tl.load(scal_ptr + 4) != 0.0:
        return
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < n_atoms
    px = tl.load(p_ptr + n * 3, mask=m, other=0.0)
    py = tl.load(p_ptr + n * 3 + 1, mask=m, other=0.0)
    pz = tl.load(p_ptr + n * 3 + 2, mask=m, other=0.0)
    fx = tl.load(f_ptr + n * 3, mask=m, other=0.0)
    fy = tl.load(f_ptr + n * 3 + 1, mask=m, other=0.0)
    fz = tl.load(f_ptr + n * 3 + 2, mask=m, other=0.0)
    inv = tl.load(inv_ptr + n, mask=m, other=0.0)
    mk = tl.load(mask_ptr + n, mask=m, other=0.0)
    ox = px
    oy = py
    oz = pz
    if EXPOL:
        k00 = tl.load(expol_ptr + n * 9, mask=m, other=0.0)
        k01 = tl.load(expol_ptr + n * 9 + 1, mask=m, other=0.0)
        k02 = tl.load(expol_ptr + n * 9 + 2, mask=m, other=0.0)
        k10 = tl.load(expol_ptr + n * 9 + 3, mask=m, other=0.0)
        k11 = tl.load(expol_ptr + n * 9 + 4, mask=m, other=0.0)
        k12 = tl.load(expol_ptr + n * 9 + 5, mask=m, other=0.0)
        k20 = tl.load(expol_ptr + n * 9 + 6, mask=m, other=0.0)
        k21 = tl.load(expol_ptr + n * 9 + 7, mask=m, other=0.0)
        k22 = tl.load(expol_ptr + n * 9 + 8, mask=m, other=0.0)
        ox = px + k00 * px + k01 * py + k02 * pz
        oy = py + k10 * px + k11 * py + k12 * pz
        oz = pz + k20 * px + k21 * py + k22 * pz
    ax = (ox * inv - fx) * mk
    ay = (oy * inv - fy) * mk
    az = (oz * inv - fz) * mk
    tl.store(ap_ptr + n * 3, ax, mask=m)
    tl.store(ap_ptr + n * 3 + 1, ay, mask=m)
    tl.store(ap_ptr + n * 3 + 2, az, mask=m)
    tl.atomic_add(scal_ptr, tl.sum((px * ax + py * ay + pz * az).to(tl.float64), 0), sem="relaxed")


@triton.jit
def _pcg_step(x_ptr, r_ptr, z_ptr, p_ptr, ap_ptr, diag_ptr, blocks_ptr, scal_ptr, n_atoms, BLOCKS: tl.constexpr,
              BLOCK: tl.constexpr):
    """``a = rz / (p . ap)``; ``x += a p`` and ``r -= a ap``; ``z = M r`` with the diagonal
    preconditioner (``alpha mask`` per atom, or its 3 x 3 blocks); ``scal[2] += r . z``,
    ``scal[3] += |a p|^2``. Nothing once the solve has converged: ``x``, ``r`` and ``z`` keep their
    values, and the direction keeps ``rz``."""
    if tl.load(scal_ptr + 4) != 0.0:
        return
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < n_atoms
    denom = tl.load(scal_ptr)
    rz = tl.load(scal_ptr + 1)
    a = tl.where(denom != 0.0, rz / tl.where(denom != 0.0, denom, 1.0), 0.0)
    px = tl.load(p_ptr + n * 3, mask=m, other=0.0)
    py = tl.load(p_ptr + n * 3 + 1, mask=m, other=0.0)
    pz = tl.load(p_ptr + n * 3 + 2, mask=m, other=0.0)
    a = a.to(px.dtype)
    apx = tl.load(ap_ptr + n * 3, mask=m, other=0.0)
    apy = tl.load(ap_ptr + n * 3 + 1, mask=m, other=0.0)
    apz = tl.load(ap_ptr + n * 3 + 2, mask=m, other=0.0)
    xx = tl.load(x_ptr + n * 3, mask=m, other=0.0)
    xy = tl.load(x_ptr + n * 3 + 1, mask=m, other=0.0)
    xz = tl.load(x_ptr + n * 3 + 2, mask=m, other=0.0)
    rx = tl.load(r_ptr + n * 3, mask=m, other=0.0)
    ry = tl.load(r_ptr + n * 3 + 1, mask=m, other=0.0)
    rz_ = tl.load(r_ptr + n * 3 + 2, mask=m, other=0.0)
    sx = a * px
    sy = a * py
    sz = a * pz
    xx = xx + sx
    xy = xy + sy
    xz = xz + sz
    rx = rx - a * apx
    ry = ry - a * apy
    rz_ = rz_ - a * apz
    if BLOCKS:
        b00 = tl.load(blocks_ptr + n * 9, mask=m, other=0.0)
        b01 = tl.load(blocks_ptr + n * 9 + 1, mask=m, other=0.0)
        b02 = tl.load(blocks_ptr + n * 9 + 2, mask=m, other=0.0)
        b10 = tl.load(blocks_ptr + n * 9 + 3, mask=m, other=0.0)
        b11 = tl.load(blocks_ptr + n * 9 + 4, mask=m, other=0.0)
        b12 = tl.load(blocks_ptr + n * 9 + 5, mask=m, other=0.0)
        b20 = tl.load(blocks_ptr + n * 9 + 6, mask=m, other=0.0)
        b21 = tl.load(blocks_ptr + n * 9 + 7, mask=m, other=0.0)
        b22 = tl.load(blocks_ptr + n * 9 + 8, mask=m, other=0.0)
        zx = b00 * rx + b01 * ry + b02 * rz_
        zy = b10 * rx + b11 * ry + b12 * rz_
        zz = b20 * rx + b21 * ry + b22 * rz_
    else:
        d = tl.load(diag_ptr + n, mask=m, other=0.0)
        zx = d * rx
        zy = d * ry
        zz = d * rz_
    tl.store(x_ptr + n * 3, xx, mask=m)
    tl.store(x_ptr + n * 3 + 1, xy, mask=m)
    tl.store(x_ptr + n * 3 + 2, xz, mask=m)
    tl.store(r_ptr + n * 3, rx, mask=m)
    tl.store(r_ptr + n * 3 + 1, ry, mask=m)
    tl.store(r_ptr + n * 3 + 2, rz_, mask=m)
    tl.store(z_ptr + n * 3, zx, mask=m)
    tl.store(z_ptr + n * 3 + 1, zy, mask=m)
    tl.store(z_ptr + n * 3 + 2, zz, mask=m)
    tl.atomic_add(scal_ptr + 2, tl.sum((rx * zx + ry * zy + rz_ * zz).to(tl.float64), 0), sem="relaxed")
    tl.atomic_add(scal_ptr + 3, tl.sum((sx * sx + sy * sy + sz * sz).to(tl.float64), 0), sem="relaxed")


@triton.jit
def _pcg_direction(p_ptr, z_ptr, in_ptr, out_ptr, n_atoms, bound, BLOCK: tl.constexpr):
    """Unless the solve is done (or converges now: squared step below ``bound``), ``p = z + (rz_new / rz) p``;
    the next iteration's scalars are written to ``out``: zero accumulators, the kept ``rz``, the flag and the
    count of iterations taken before convergence."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < n_atoms
    rz = tl.load(in_ptr + 1)
    rz_new = tl.load(in_ptr + 2)
    change = tl.load(in_ptr + 3)
    done = tl.load(in_ptr + 4) != 0.0
    taken = tl.load(in_ptr + 5)
    now = done | (change < bound)
    beta = rz_new / tl.where(rz != 0.0, rz, 1.0)
    # a direction that no longer changes is neither read nor written
    if change >= bound:
        if tl.load(in_ptr + 4) == 0.0:
            px = tl.load(p_ptr + n * 3, mask=m, other=0.0)
            py = tl.load(p_ptr + n * 3 + 1, mask=m, other=0.0)
            pz = tl.load(p_ptr + n * 3 + 2, mask=m, other=0.0)
            zx = tl.load(z_ptr + n * 3, mask=m, other=0.0)
            zy = tl.load(z_ptr + n * 3 + 1, mask=m, other=0.0)
            zz = tl.load(z_ptr + n * 3 + 2, mask=m, other=0.0)
            b = beta.to(px.dtype)
            tl.store(p_ptr + n * 3, zx + b * px, mask=m)
            tl.store(p_ptr + n * 3 + 1, zy + b * py, mask=m)
            tl.store(p_ptr + n * 3 + 2, zz + b * pz, mask=m)
    if tl.program_id(0) == 0:
        tl.store(out_ptr, rz * 0.0)
        tl.store(out_ptr + 1, tl.where(now, rz, rz_new))
        tl.store(out_ptr + 2, rz * 0.0)
        tl.store(out_ptr + 3, rz * 0.0)
        tl.store(out_ptr + 4, tl.where(now, rz * 0.0 + 1.0, rz * 0.0))
        tl.store(out_ptr + 5, tl.where(now, taken, taken + 1.0))


def pcg_scalars(rz: Tensor) -> Tensor:
    """The two scalar buffers ``(2, 6)`` of a solve, the first holding ``sum(r * z)`` of the start."""
    scal = torch.zeros(2, 6, dtype=torch.float64, device=rz.device)
    scal[0, 1] = rz
    return scal


def pcg_apply(p: Tensor, field: Tensor, inv_alpha: Tensor, expol, mask: Tensor, scal: Tensor) -> Tensor:
    """The operator on ``p`` ``(N, 3)`` given its field, accumulating ``sum(p * (A p))`` into ``scal[0]``."""
    N = int(p.shape[0])
    ap = torch.empty_like(p)
    if N:
        _pcg_apply[(triton.cdiv(N, PCG_BLOCK),)](p, field.contiguous(), ap, inv_alpha, ap if expol is None
                                                 else expol.reshape(N, 9).contiguous(), mask, scal, N,
                                                 EXPOL=expol is not None, BLOCK=PCG_BLOCK, num_warps=4)
    return ap


def pcg_step(x: Tensor, r: Tensor, z: Tensor, p: Tensor, ap: Tensor, diag, blocks, scal: Tensor) -> None:
    """The update of ``x``, ``r`` and ``z`` in place (see :func:`_pcg_step`)."""
    N = int(x.shape[0])
    if N:
        _pcg_step[(triton.cdiv(N, PCG_BLOCK),)](x, r, z, p, ap, x if diag is None else diag,
                                                x if blocks is None else blocks.reshape(N, 9).contiguous(), scal, N,
                                                BLOCKS=blocks is not None, BLOCK=PCG_BLOCK, num_warps=4)


def pcg_direction(p: Tensor, z: Tensor, scal_in: Tensor, scal_out: Tensor, bound: float) -> None:
    """The new direction in place and the next iteration's scalars (see :func:`_pcg_direction`)."""
    N = int(p.shape[0])
    if N:
        _pcg_direction[(triton.cdiv(N, PCG_BLOCK),)](p, z, scal_in, scal_out, N, float(bound), BLOCK=PCG_BLOCK,
                                                     num_warps=4)


def scalar_table(coulomb: float, alpha_e: float, alpha_p: float, alpha_d: float, windows: dict, cutoff: float,
                 device: torch.device, ewald_cutoffs: tuple = (0.0, 0.0)) -> Tensor:
    """The float64 scalars of an evaluation (:data:`SCALARS`); ``windows`` maps ``"rep"``, ``"ct"``,
    ``"disp"`` to ``(taper, cutoff)`` pairs (zeros when the term has no switch), ``ewald_cutoffs`` holds
    the real-space cutoffs of the electrostatics and the dispersion."""
    rep, ct, disp = windows.get("rep", (0.0, 0.0)), windows.get("ct", (0.0, 0.0)), windows.get("disp", (0.0, 0.0))
    values = [coulomb, 1.0 / math.sqrt(math.pi), alpha_e, alpha_p, alpha_d, rep[0], rep[1], ct[0], ct[1],
              disp[0], disp[1], cutoff, *ewald_cutoffs]
    return torch.tensor(values, dtype=torch.float64, device=device)


__all__ = ["PARAMS", "NP", "SCALARS", "HippoPairs", "PairDifference", "pairs_forward", "pairs_backward", "mutual_field",
           "mutual_backward", "MeshConvolution", "MeshConvolutionCartesian", "spline_tables", "spread", "gather",
           "adjoint", "spread_cartesian", "gather_cartesian", "gather_field", "blocks_inverse",
           "adjoint_cartesian", "pcg_scalars",
           "pcg_apply", "pcg_step", "pcg_direction", "scalar_table",
           "constants", "available"]
