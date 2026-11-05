"""The AMOEBA and HIPPO kernels on the CPU: :func:`xnn.common.cpu_kernels.launch` runs them as C++ loops, with
lane-wise twins of the helpers and kernels that reduce over a block."""
from typing import Optional

import torch
import triton
import triton.language as tl

from xnn.common import cpu_kernels

from . import amoeba_kernels as K

# each lane adds its own term: the C++ loops add into per-thread buffers, so a sorted list does not
# serialize on one address as GPU atomics do
cpu_kernels.override(K._add_sorted, '''
def _add_sorted(ptr, first, last, idx, v, same, tail, rest, TAIL):
    tl.atomic_add(ptr + idx, v, mask=same | tail | rest, sem="relaxed")
''')


@triton.jit
def _rows_scatter(ptr_ptr, pair_ptr, other_ptr, i_ptr, g_ptr, out_ptr, n_ptr, CHUNK: tl.constexpr,
                 BLOCK: tl.constexpr = 1):
    """:func:`.amoeba_kernels._rows_scatter` with one lane per atom walking its row (one lane a program)."""
    a = tl.program_id(0).to(tl.int64)
    count = tl.load(n_ptr)
    acc = out_ptr.dtype.element_ty
    sx = (a * 0).to(acc)
    sy = (a * 0).to(acc)
    sz = (a * 0).to(acc)
    for j in range(tl.load(ptr_ptr + a), tl.load(ptr_ptr + a + 1)):
        e = tl.load(pair_ptr + j).to(tl.int64)
        if e < count:
            b = tl.load(other_ptr + j).to(tl.int64)
            if b != a:
                sign = tl.where(tl.load(i_ptr + e) == a, -1.0, 1.0)
                sx += sign * tl.load(g_ptr + e * 3).to(acc)
                sy += sign * tl.load(g_ptr + e * 3 + 1).to(acc)
                sz += sign * tl.load(g_ptr + e * 3 + 2).to(acc)
    tl.store(out_ptr + a * 3, sx)
    tl.store(out_ptr + a * 3 + 1, sy)
    tl.store(out_ptr + a * 3 + 2, sz)


@triton.jit
def _mutual_rows(ptr_ptr, pair_ptr, other_ptr, r_ptr, du_ptr, mu_ptr, f_ptr, n_ptr, guard_ptr, n_atoms,
                 GUARD: tl.constexpr, PAIRED: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr = 1):
    """:func:`.amoeba_kernels._mutual_rows` with one lane per atom walking its row (one lane a program)."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    a = tl.program_id(0).to(tl.int64)
    count = tl.load(n_ptr)
    acc = f_ptr.dtype.element_ty
    sx = (a * 0).to(acc)
    sy = (a * 0).to(acc)
    sz = (a * 0).to(acc)
    px = (a * 0).to(acc)
    py = (a * 0).to(acc)
    pz = (a * 0).to(acc)
    for j in range(tl.load(ptr_ptr + a), tl.load(ptr_ptr + a + 1)):
        e = tl.load(pair_ptr + j).to(tl.int64)
        if e < count:
            b = tl.load(other_ptr + j).to(tl.int64)
            rx = tl.load(r_ptr + e * 3)
            ry = tl.load(r_ptr + e * 3 + 1)
            rz = tl.load(r_ptr + e * 3 + 2)
            D1 = tl.load(du_ptr + e * 2)
            D2 = tl.load(du_ptr + e * 2 + 1)
            mbx = tl.load(mu_ptr + b * 3)
            mby = tl.load(mu_ptr + b * 3 + 1)
            mbz = tl.load(mu_ptr + b * 3 + 2)
            rm = D2 * (rx * mbx + ry * mby + rz * mbz)
            sx += (D1 * mbx + rm * rx).to(acc)
            sy += (D1 * mby + rm * ry).to(acc)
            sz += (D1 * mbz + rm * rz).to(acc)
            if PAIRED:
                nbx = tl.load(mu_ptr + (b + n_atoms) * 3)
                nby = tl.load(mu_ptr + (b + n_atoms) * 3 + 1)
                nbz = tl.load(mu_ptr + (b + n_atoms) * 3 + 2)
                rn = D2 * (rx * nbx + ry * nby + rz * nbz)
                px += (D1 * nbx + rn * rx).to(acc)
                py += (D1 * nby + rn * ry).to(acc)
                pz += (D1 * nbz + rn * rz).to(acc)
    tl.store(f_ptr + a * 3, sx)
    tl.store(f_ptr + a * 3 + 1, sy)
    tl.store(f_ptr + a * 3 + 2, sz)
    if PAIRED:
        tl.store(f_ptr + (a + n_atoms) * 3, px)
        tl.store(f_ptr + (a + n_atoms) * 3 + 1, py)
        tl.store(f_ptr + (a + n_atoms) * 3 + 2, pz)


@triton.jit
def _spread_c(q_ptr, d_ptr, Q_ptr, j_ptr, w_ptr, start_ptr, batch_ptr, grid_ptr, n_atoms, guard_ptr, perm_ptr,
              K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr, ORD: tl.constexpr,
              BLOCK: tl.constexpr, SP: tl.constexpr, GUARD: tl.constexpr):
    """:func:`.amoeba_kernels._spread_c` with one lane per atom walking its stencil, the atoms taken in the
    order ``perm`` (by mesh cell: a thread's contiguous programs then add into a compact part of the mesh)."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    acc = grid_ptr.dtype.element_ty
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    n = tl.load(perm_ptr + n, mask=m, other=0)
    c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz = K._load_multipoles(q_ptr, d_ptr, Q_ptr, j_ptr,
                                                                               batch_ptr, n, m, ORD)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    row = w_ptr + n * (3 * 4 * P)
    for a in tl.static_range(P):
        wx = tl.load(row + a, mask=m, other=0.0)
        dx = wx
        ddx = wx
        if ORD >= 1:
            dx = tl.load(row + P + a, mask=m, other=0.0)
        if ORD >= 2:
            ddx = tl.load(row + 2 * P + a, mask=m, other=0.0)
        gx = base + ((ix - a + K1) % K1) * (K2 * K3)
        for b in tl.static_range(P):
            wy = tl.load(row + 4 * P + b, mask=m, other=0.0)
            dy = wy
            ddy = wy
            if ORD >= 1:
                dy = tl.load(row + 5 * P + b, mask=m, other=0.0)
            if ORD >= 2:
                ddy = tl.load(row + 6 * P + b, mask=m, other=0.0)
            gy = gx + ((iy - b + K2) % K2) * K3
            for c in tl.static_range(P):
                wz = tl.load(row + 8 * P + c, mask=m, other=0.0)
                dz = wz
                ddz = wz
                if ORD >= 1:
                    dz = tl.load(row + 9 * P + c, mask=m, other=0.0)
                if ORD >= 2:
                    ddz = tl.load(row + 10 * P + c, mask=m, other=0.0)
                val = K._coef(c0, c1x, c1y, c1z, c2xx, c2yy, c2zz, c2xy, c2xz, c2yz, wx, dx, ddx, wy, dy, ddy,
                              wz, dz, ddz, ORD)
                tl.atomic_add(grid_ptr + gy + (iz - c + K3) % K3, val.to(acc), mask=m, sem="relaxed")


@triton.jit
def _gather_c(grid_ptr, j_ptr, w_ptr, start_ptr, batch_ptr, phi_ptr, grad_ptr, hess_ptr, d_ptr, coef_ptr, n_atoms,
              guard_ptr, perm_ptr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr, P: tl.constexpr,
              DERIV: tl.constexpr, FIELD: tl.constexpr, BLOCK: tl.constexpr, SP: tl.constexpr, GUARD: tl.constexpr):
    """:func:`.amoeba_kernels._gather_c` with one lane per atom walking its stencil, the atoms taken in the
    order ``perm`` (by mesh cell: a thread's contiguous programs then read a compact part of the mesh)."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    n = tl.load(perm_ptr + n, mask=m, other=0)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    batch = tl.load(batch_ptr + n, mask=m, other=0)
    base = batch * (K1 * K2 * K3)
    row = w_ptr + n * (3 * 4 * P)
    zero = tl.load(grid_ptr + base, mask=m, other=0.0) * tl.load(row, mask=m, other=0.0) * 0.0
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
        gx = base + ((ix - a + K1) % K1) * (K2 * K3)
        if DERIV >= 1:
            dx = tl.load(row + P + a, mask=m, other=0.0)
        if DERIV >= 2:
            ddx = tl.load(row + 2 * P + a, mask=m, other=0.0)
        for b in tl.static_range(P):
            wy = tl.load(row + 4 * P + b, mask=m, other=0.0)
            gy = gx + ((iy - b + K2) % K2) * K3
            if DERIV >= 1:
                dy = tl.load(row + 5 * P + b, mask=m, other=0.0)
            if DERIV >= 2:
                ddy = tl.load(row + 6 * P + b, mask=m, other=0.0)
            for c in tl.static_range(P):
                wz = tl.load(row + 8 * P + c, mask=m, other=0.0)
                v = tl.load(grid_ptr + gy + (iz - c + K3) % K3, mask=m, other=0.0)
                if not FIELD:
                    phi += v * wx * wy * wz
                if DERIV >= 1:
                    dz = tl.load(row + 9 * P + c, mask=m, other=0.0)
                    d0 += v * dx * wy * wz
                    d1 += v * wx * dy * wz
                    d2 += v * wx * wy * dz
                if DERIV >= 2:
                    ddz = tl.load(row + 10 * P + c, mask=m, other=0.0)
                    h00 += v * ddx * wy * wz
                    h11 += v * wx * ddy * wz
                    h22 += v * wx * wy * ddz
                    h01 += v * dx * dy * wz
                    h02 += v * dx * wy * dz
                    h12 += v * wx * dy * dz
    if not FIELD:
        tl.store(phi_ptr + n, phi, mask=m)
    if DERIV >= 1:
        j00, j01, j02, j10, j11, j12, j20, j21, j22 = K._jac(j_ptr, batch, m)
        gx_, gy_, gz_ = K._cart_vec(j00, j01, j02, j10, j11, j12, j20, j21, j22, d0, d1, d2)
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
            hxx, hxy, hxz, hyy, hyz, hzz = K._cart_quad(j00, j01, j02, j10, j11, j12, j20, j21, j22,
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
    """:func:`.amoeba_kernels._adjoint_c` with one lane per atom walking its stencil."""
    n = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = n < n_atoms
    a0, a1x, a1y, a1z, a2xx, a2yy, a2zz, a2xy, a2xz, a2yz = K._load_multipoles(aq_ptr, ad_ptr, aQ_ptr, j_ptr,
                                                                               batch_ptr, n, m, ORDA)
    b0, b1x, b1y, b1z, b2xx, b2yy, b2zz, b2xy, b2xz, b2yz = K._load_multipoles(bq_ptr, bd_ptr, bQ_ptr, j_ptr,
                                                                               batch_ptr, n, m, ORDB)
    ix = tl.load(start_ptr + n * 3, mask=m, other=0)
    iy = tl.load(start_ptr + n * 3 + 1, mask=m, other=0)
    iz = tl.load(start_ptr + n * 3 + 2, mask=m, other=0)
    base = tl.load(batch_ptr + n, mask=m, other=0) * (K1 * K2 * K3)
    row = w_ptr + n * (3 * 4 * P)
    zero = tl.load(ga_ptr + base, mask=m, other=0.0) * tl.load(row, mask=m, other=0.0) * 0.0
    ux = zero
    uy = zero
    uz = zero
    for a in tl.static_range(P):
        wx = tl.load(row + a, mask=m, other=0.0)
        dx = tl.load(row + P + a, mask=m, other=0.0)
        ddx = tl.load(row + 2 * P + a, mask=m, other=0.0)
        dddx = wx * 0.0
        if THIRD:
            dddx = tl.load(row + 3 * P + a, mask=m, other=0.0)
        gx = base + ((ix - a + K1) % K1) * (K2 * K3)
        for b in tl.static_range(P):
            wy = tl.load(row + 4 * P + b, mask=m, other=0.0)
            dy = tl.load(row + 5 * P + b, mask=m, other=0.0)
            ddy = tl.load(row + 6 * P + b, mask=m, other=0.0)
            dddy = wy * 0.0
            if THIRD:
                dddy = tl.load(row + 7 * P + b, mask=m, other=0.0)
            gy = gx + ((iy - b + K2) % K2) * K3
            for c in tl.static_range(P):
                wz = tl.load(row + 8 * P + c, mask=m, other=0.0)
                dz = tl.load(row + 9 * P + c, mask=m, other=0.0)
                ddz = tl.load(row + 10 * P + c, mask=m, other=0.0)
                dddz = wz * 0.0
                if THIRD:
                    dddz = tl.load(row + 11 * P + c, mask=m, other=0.0)
                at = gy + (iz - c + K3) % K3
                va = tl.load(ga_ptr + at, mask=m, other=0.0)
                vb = tl.load(gb_ptr + at, mask=m, other=0.0)
                px, py, pz = K._dcoef(a0, a1x, a1y, a1z, a2xx, a2yy, a2zz, a2xy, a2xz, a2yz,
                                      wx, dx, ddx, dddx, wy, dy, ddy, dddy, wz, dz, ddz, dddz, ORDA)
                sx, sy, sz = K._dcoef(b0, b1x, b1y, b1z, b2xx, b2yy, b2zz, b2xy, b2xz, b2yz,
                                      wx, dx, ddx, dddx, wy, dy, ddy, dddy, wz, dz, ddz, dddz, ORDB)
                ux += va * px + vb * sx
                uy += va * py + vb * sy
                uz += va * pz + vb * sz
    tl.store(gu_ptr + n * 3, ux, mask=m)
    tl.store(gu_ptr + n * 3 + 1, uy, mask=m)
    tl.store(gu_ptr + n * 3 + 2, uz, mask=m)


@triton.jit
def _mutual_packed(ptr_ptr, other_ptr, pk_ptr, mu_ptr, f_ptr, guard_ptr, n_atoms, GUARD: tl.constexpr,
                   PAIRED: tl.constexpr, BLOCK: tl.constexpr = 1):
    """:func:`_mutual_rows` from row-ordered pair data ``pk`` ``(2 M, 5)``: the vector and the two mutual
    ladders of each entry's pair (zero past the count), so a row streams its data instead of gathering it."""
    if GUARD:
        if tl.load(guard_ptr + 4) != 0.0:
            return
    a = tl.program_id(0).to(tl.int64)
    acc = f_ptr.dtype.element_ty
    sx = (a * 0).to(acc)
    sy = (a * 0).to(acc)
    sz = (a * 0).to(acc)
    px = (a * 0).to(acc)
    py = (a * 0).to(acc)
    pz = (a * 0).to(acc)
    for j in range(tl.load(ptr_ptr + a), tl.load(ptr_ptr + a + 1)):
        b = tl.load(other_ptr + j).to(tl.int64)
        rx = tl.load(pk_ptr + j * 5)
        ry = tl.load(pk_ptr + j * 5 + 1)
        rz = tl.load(pk_ptr + j * 5 + 2)
        D1 = tl.load(pk_ptr + j * 5 + 3)
        D2 = tl.load(pk_ptr + j * 5 + 4)
        mbx = tl.load(mu_ptr + b * 3)
        mby = tl.load(mu_ptr + b * 3 + 1)
        mbz = tl.load(mu_ptr + b * 3 + 2)
        rm = D2 * (rx * mbx + ry * mby + rz * mbz)
        sx += (D1 * mbx + rm * rx).to(acc)
        sy += (D1 * mby + rm * ry).to(acc)
        sz += (D1 * mbz + rm * rz).to(acc)
        if PAIRED:
            nbx = tl.load(mu_ptr + (b + n_atoms) * 3)
            nby = tl.load(mu_ptr + (b + n_atoms) * 3 + 1)
            nbz = tl.load(mu_ptr + (b + n_atoms) * 3 + 2)
            rn = D2 * (rx * nbx + ry * nby + rz * nbz)
            px += (D1 * nbx + rn * rx).to(acc)
            py += (D1 * nby + rn * ry).to(acc)
            pz += (D1 * nbz + rn * rz).to(acc)
    tl.store(f_ptr + a * 3, sx)
    tl.store(f_ptr + a * 3 + 1, sy)
    tl.store(f_ptr + a * 3 + 2, sz)
    if PAIRED:
        tl.store(f_ptr + (a + n_atoms) * 3, px)
        tl.store(f_ptr + (a + n_atoms) * 3 + 1, py)
        tl.store(f_ptr + (a + n_atoms) * 3 + 2, pz)


@triton.jit
def _pack(pair_ptr, r_ptr, du_ptr, n_ptr, pk_ptr, n_entries, BLOCK: tl.constexpr):
    """Row entry ``j`` of :func:`pack_rows`: the vector and the two ladders of its pair, the ladders zero for a
    pair at or beyond the count ``n``."""
    j = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = j < n_entries
    e = tl.load(pair_ptr + j, mask=m, other=0).to(tl.int64)
    keep = m & (e < tl.load(n_ptr))
    for c in tl.static_range(3):
        tl.store(pk_ptr + j * 5 + c, tl.load(r_ptr + e * 3 + c, mask=m, other=0.0), mask=m)
    for c in tl.static_range(2):
        tl.store(pk_ptr + j * 5 + 3 + c, tl.load(du_ptr + e * 2 + c, mask=keep, other=0.0), mask=m)


def pack_rows(r_vec: torch.Tensor, du: torch.Tensor, rows: tuple, count=None) -> torch.Tensor:
    """The pairs' vectors ``(M, 3)`` and mutual ladders ``(M, 2)`` in the order of the compressed rows ``rows``
    (:func:`.amoeba_kernels.pair_rows`), ``(2 M, 5)``, with zero ladders for the pairs at or beyond ``count``:
    built once per solve for :func:`mutual_field`, by one kernel."""
    n = int(rows[1].shape[0])
    packed = torch.empty(n, 5, dtype=r_vec.dtype, device=r_vec.device)
    if n:
        M = int(r_vec.shape[0])
        cpu_kernels.launch(_pack, (triton.cdiv(n, 256),), rows[1], r_vec.contiguous(), du.contiguous(),
                           K._count(count, M, r_vec.device), packed, n, BLOCK=256)
    return packed


def mutual_field(mu: torch.Tensor, packed: torch.Tensor, rows: tuple, N: int,
                 guard: Optional[torch.Tensor] = None) -> torch.Tensor:
    """:func:`.amoeba_kernels.mutual_field` on the CPU from the row-ordered pair data of :func:`pack_rows`."""
    paired = int(mu.shape[0]) == 2 * N
    f = torch.empty(mu.shape[0], 3, dtype=mu.dtype, device=mu.device)
    if N:
        cpu_kernels.launch(_mutual_packed, (N,), rows[0], rows[2], packed, mu.contiguous(), f,
                           f if guard is None else guard, N, GUARD=guard is not None, PAIRED=paired)
    return f


_ORDERS: dict = {}
_SHARES: dict = {}


def _stencil_order(start: torch.Tensor, batch: torch.Tensor, mesh: list) -> torch.Tensor:
    """The atoms sorted by the mesh cell of their stencils (structure first), kept for the last stencil tables
    (any permutation gives the same sums; the order only keeps a thread's adds together)."""
    key = (start.data_ptr(), start._version, int(start.shape[0]), tuple(int(k) for k in mesh))
    order = _ORDERS.get(key)
    if order is None:
        K1, K2, K3 = (int(k) for k in mesh)
        s = start.long()
        cell = ((batch.long() * K1 + s[:, 0]) * K2 + s[:, 1]) * K3 + s[:, 2]
        order = torch.argsort(cell)
        if len(_ORDERS) > 8:
            _ORDERS.clear()
        _ORDERS[key] = order
    return order


def _even_shares(n: int) -> torch.Tensor:
    """The work prefix of ``n`` programs of equal work: each thread takes a contiguous share."""
    if n not in _SHARES:
        _SHARES[n] = torch.arange(n + 1, dtype=torch.int64)
    return _SHARES[n]


def spread_cartesian(q: torch.Tensor, d, Q, jac: torch.Tensor, w: torch.Tensor, start: torch.Tensor,
                     batch: torch.Tensor, mesh: list, groups: int, order: int,
                     guard: Optional[torch.Tensor] = None) -> torch.Tensor:
    """:func:`.amoeba_kernels.spread_cartesian` on the CPU: the atoms in mesh-cell order, a contiguous share
    of them per thread."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    grid = torch.zeros(groups * K1 * K2 * K3, dtype=w.dtype, device=w.device)
    if N:
        q_, d_, Q_ = K._multipole_args(q, d, Q, order)
        programs = triton.cdiv(N, K.SPREAD_BLOCK)
        cpu_kernels.prepare(_spread_c, (programs,), q_, d_, Q_, jac.contiguous(), w, start, batch.contiguous(),
                            grid, N, grid if guard is None else guard, _stencil_order(start, batch, mesh),
                            balance=_even_shares(programs), K1=K1, K2=K2, K3=K3, P=P, ORD=int(order),
                            BLOCK=K.SPREAD_BLOCK, SP=K._lanes(P), GUARD=guard is not None)()
    return grid.view(groups, K1, K2, K3)


def gather(grid: torch.Tensor, jac: torch.Tensor, w: torch.Tensor, start: torch.Tensor, batch: torch.Tensor,
           mesh: list, outputs: tuple, d: torch.Tensor, coef: torch.Tensor, guard: Optional[torch.Tensor],
           deriv: int, field: bool) -> None:
    """The launch of :func:`.amoeba_kernels.gather_cartesian` or :func:`.amoeba_kernels.gather_field` on the CPU
    (``outputs`` the potential, gradient and Hessian arrays it stores into): the atoms in mesh-cell order, a
    contiguous share of them per thread."""
    N, P = int(w.shape[0]), int(w.shape[-1])
    K1, K2, K3 = (int(k) for k in mesh)
    programs = triton.cdiv(N, K.SPREAD_BLOCK)
    phi, grad, hess = outputs
    cpu_kernels.prepare(_gather_c, (programs,), grid.contiguous(), jac.contiguous(), w, start, batch.contiguous(),
                        phi, grad, hess, d, coef, N, phi if guard is None else guard,
                        _stencil_order(start, batch, mesh), balance=_even_shares(programs), K1=K1, K2=K2, K3=K3, P=P,
                        DERIV=int(deriv), FIELD=field, BLOCK=K.SPREAD_BLOCK, SP=K._lanes(P),
                        GUARD=guard is not None)()


TWINS = {K._rows_scatter: _rows_scatter, K._mutual_rows: _mutual_rows, K._adjoint_c: _adjoint_c}
# compile options of the GPU launches, which the C++ loops do not take
_GPU_ONLY = ("maxnreg",)


def launch(kernel, grid, *args, **meta):
    """Run ``kernel[grid](*args, **meta)`` on CPU tensors, as its twin when it has one."""
    for key in _GPU_ONLY:
        meta.pop(key, None)
    cpu_kernels.launch(TWINS.get(kernel, kernel), grid, *args, **meta)
