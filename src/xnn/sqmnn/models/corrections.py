"""Molecular-mechanics and post-SCF corrections of the NDDO methods (kcal/mol).

* :func:`bond_topology` -- which atoms are bonded (covalent radii with pair
  safety factors), the connectivity the corrections are defined on;
* :func:`peptide_torsion` -- the ``k sin^2(phi)`` barrier added to every
  ``X-N-C=O`` torsion of an amide (all PM methods except PM6-ORG);
* :func:`yne_correction` -- 12 kcal/mol per acetylenic C-C bond, switched
  off smoothly between 1.21 and 1.33 Angstrom (PM6, PM7, PM6-ORG);
* :func:`nsp2_correction` -- stabilization of planar secondary and tertiary
  amine nitrogens, ``-0.5 exp(-10 (2 pi - sum of angles))`` (PM6, eq 12 of
  the PM6 paper);
* :func:`si_o_h_correction` -- the Si-O-H bending term of PM7.

Discrete choices (bonds, groups) are made on the detached geometry; the
energies are differentiable functions of the positions. The topology searches and
the terms are written in TorchScript-compatible form (the element data in
:func:`correction_tables`): an exported model compiles them against
:class:`~.tape.TorchOps`, while the eager model evaluates the terms on the host
tape (see :func:`~.tape.host_term`).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..common import elements as el
from .tape import TorchOps, host_term

# Jurecka-type dispersion of PM7: C6 (J nm^6 / mol), van der Waals radii (pm) and
# Slater-Kirkwood effective electron numbers
_DISP_C6 = {1: 0.16, 2: 0.084, 5: 5.79, 6: 1.65, 7: 1.11, 8: 0.70, 9: 0.57, 10: 0.45, 15: 3.25,
            16: 5.79, 17: 5.97, 18: 3.71, 27: 0.04, 35: 11.60, 36: 4.47, 53: 25.80, 54: 16.50}
_DISP_R = {1: 156.0, 2: 140.0, 5: 180.0, 6: 170.0, 7: 155.0, 8: 152.0, 9: 147.0, 10: 154.0,
           15: 180.0, 16: 180.0, 17: 175.0, 18: 188.0, 27: 140.0, 35: 185.0, 36: 202.0,
           53: 198.0, 54: 216.0}
_DISP_N = {1: 0.80, 2: 1.42, 5: 2.16, 6: 2.50, 7: 2.82, 8: 3.15, 9: 3.48, 10: 3.81, 15: 4.50,
           16: 4.80, 17: 5.10, 18: 5.40, 27: 2.90, 35: 6.00, 36: 6.30, 53: 6.95, 54: 7.25}


# covalent radii (Angstrom) of the hydrogen-bond topology, scaled by 4/3 in use
_HB_RADII = (0.0,
             0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04,
             1.10, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99,
             1.01, 1.09, 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24,
             1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76,
             1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
             1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31,
             1.38, 1.42)


# the same radii scaled by 4/3, by atomic number (to the end of the periodic table)
_NP_HB_RADIUS = np.asarray([4.0 / 3.0 * r for r in _HB_RADII] + [0.0] * (el.MAX_Z + 1 - len(_HB_RADII)))
_DEG = math.pi / 180.0
# the PM7 hydrogen-bond angles (shift, shift2, tshift) by kind of end: other, O with one bond, other O, N with two
# bonds, other N; and the scale of a bond end by atomic number
_NP_HB_SHIFTS = np.array([[0.0, 0.0, 0.0], [math.pi, 120.0 * _DEG, 0.0],
                          [109.48 * _DEG, 109.48 * _DEG, 54.74 * _DEG], [120.0 * _DEG, 120.0 * _DEG, 0.0],
                          [109.48 * _DEG, 109.48 * _DEG, 54.74 * _DEG]])
_NP_HB_SCALE = np.where(np.arange(el.MAX_Z + 1) == 7, -0.171271 * 0.529177210903 ** 2,
                        -0.098822 * 0.529177210903 ** 2)


def _host(pos) -> np.ndarray:
    """The positions as a float64 host array."""
    return pos if isinstance(pos, np.ndarray) else pos.detach().to("cpu", torch.float64).numpy()


def _distances(pos) -> np.ndarray:
    """The distance matrix of the detached positions (tensor or host array), each entry ``sqrt(dx^2 + dy^2 +
    dz^2)`` in float64 (the order of operations of a sum over the three components)."""
    xyz = _host(pos)
    d = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])


def _np_bond_topology(z: Sequence[int], pos: Tensor) -> List[List[int]]:
    """:func:`bond_topology` on the host (NumPy), the form of an eager model.

    Bonded neighbors of every atom of one molecule.

    Two atoms are bonded when closer than ``f (r_i + r_j)`` with the covalent
    radii of :data:`~xnn.sqmnn.common.elements.COVALENT_RADIUS` (alkali and
    alkaline-earth metals count with radius -1) and ``f = 1.1``, except 1.25
    for C-H, 1.0 for B-N, 1.2 for C-C, C-N and S-S; at most 15 bonds per
    atom (pairs taken in order of the first atom, then the second), and a
    hydrogen with several bonds keeps only those to heavy atoms.
    """
    n = len(z)
    bonds: List[List[int]] = [[] for _ in range(n)]
    if n < 2:
        return bonds
    zz = np.asarray(z)
    xyz = pos.detach().to("cpu", torch.float64).numpy()
    radius = np.array([(-1.0 if v in el.METALS else (el.COVALENT_RADIUS[v] or 999.0)) for v in z])
    lo, hi = np.minimum.outer(zz, zz), np.maximum.outer(zz, zz)
    f = np.full((n, n), 1.1)
    f[(lo == 1) & (hi == 6)] = 1.25
    f[(lo == 5) & (hi == 7)] = 1.0
    f[(lo == 6) & (hi < 8)] = 1.2
    f[(lo == 16) & (hi == 16)] = 1.2
    d = xyz[:, None, :] - xyz[None, :, :]
    d2 = d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2
    near = (d2 < (f * (radius[:, None] + radius[None, :])) ** 2) & np.tri(n, n, -1, dtype=bool)
    for i, j in zip(*np.nonzero(near)):
        if len(bonds[i]) < 15 and len(bonds[j]) < 15:
            bonds[i].append(int(j))
            bonds[j].append(int(i))
    for i in range(n):
        if z[i] == 1 and len(bonds[i]) >= 2:
            bonds[i] = [j for j in bonds[i] if z[j] != 1]
    return bonds




def _np_hbond_triples(z: Sequence[int], x: np.ndarray, D: np.ndarray, r_xh: float, cutoff: float,
                      heavy=(7, 8)) -> np.ndarray:
    """:func:`_hbond_candidates` of host arrays ``(T, 3)``."""
    none = np.zeros((0, 3), dtype=np.int64)
    zz = np.asarray(z)
    is_heavy = zz == heavy[0]
    for v in heavy[1:]:
        is_heavy |= zz == v
    acc = np.nonzero(is_heavy)[0]
    is_h = zz == 1
    if len(acc) == 0:
        return none
    near = (D[acc] < r_xh) & is_h[None, :]                                            # (n_heavy, n)
    claimed = near.any(0)
    first = np.where(claimed, near.argmax(0), len(acc))
    hyd = np.argsort(first, kind="stable")[:int(claimed.sum())]
    if len(hyd) == 0:
        return none
    Da = D[acc]
    pi, pj = np.nonzero(Da[:, hyd] < r_xh)
    close = Da[:, acc] < cutoff
    np.fill_diagonal(close, False)
    p, pk = np.nonzero(close[pi])
    i, j, k = acc[pi[p]], hyd[pj[p]], acc[pk]
    if len(i) == 0:
        return none
    v1, v2 = x[k] - x[j], x[i] - x[j]
    cosang = (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]) / (D[k, j] * D[i, j])
    wide = np.arccos(np.clip(cosang, -1.0, 1.0)) > 0.5 * math.pi
    i, j, k = i[wide], j[wide], k[wide]
    if len(i) == 0:
        return none
    n = len(zz)
    key, back = (i * n + j) * n + k, (k * n + j) * n + i
    order = np.argsort(key)
    where = np.minimum(np.searchsorted(key[order], back), len(key) - 1)
    earlier = (key[order][where] == back) & (order[where] < np.arange(len(key)))
    keep = ~earlier
    out = np.empty((int(keep.sum()), 3), dtype=np.int64)
    out[:, 0], out[:, 1], out[:, 2] = i[keep], j[keep], k[keep]
    return out


def _np_bonded_lists(D: np.ndarray, rz: np.ndarray, cap: int):
    """:func:`_bonded_lists` of host arrays."""
    n = D.shape[0]
    bonded = D < rz[:, None] + rz[None, :]
    np.fill_diagonal(bonded, False)
    c, j = np.nonzero(bonded)
    count = bonded.sum(1)
    start = np.cumsum(count) - count
    slot = np.arange(len(c)) - start[c]
    full = np.full((n, max(int(count.max()), 1)), -1, dtype=np.int64)
    full[c, slot] = j
    idx = np.lexsort((-j, D[c, j], c))
    rank = np.empty_like(slot)
    rank[idx] = np.arange(len(c)) - start[c[idx]]
    kept = rank < cap
    ck, jk = c[kept], j[kept]
    count_k = np.minimum(count, cap)
    start_k = np.cumsum(count_k) - count_k
    capped = np.full((n, cap), -1, dtype=np.int64)
    capped[ck, np.arange(len(ck)) - start_k[ck]] = jk
    return full, count, capped, count_k


def _np_farthest_of(dh: np.ndarray, cands: np.ndarray) -> np.ndarray:
    valid = cands >= 0
    pos = np.where(valid, dh, -1.0).argmax(-1)
    rows = np.arange(cands.shape[0])[:, None]
    return np.where(valid.any(-1), cands[rows, np.arange(cands.shape[1])[None, :], pos], -1)


def _np_pm7_hbond_tables(z: Sequence[int], x: np.ndarray):
    """:func:`_pm7_hbond_tables` of the host array ``x`` as host arrays, the form of an eager model."""
    D = _distances(x)
    t = _np_hbond_triples(z, x, D, 1.4, 7.0)
    if len(t) == 0:
        return None
    zz = np.asarray(z)
    rz = _NP_HB_RADIUS[zz]
    full, _, nbr, count = _np_bonded_lists(D, rz, 4)
    A, H, B = t[:, 0], t[:, 1], t[:, 2]
    na, nb = nbr[A], nbr[B]
    common = (na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0) & (na[:, :, None] != H[:, None, None])
    C = t[:, ::2]
    Hc = t[:, 1:2].repeat(2, 1)
    L = nbr[C]
    k = count[C]
    dh = D[Hc[..., None], np.maximum(L, 0)]
    order = np.argsort(np.where(L >= 0, -dh, np.inf), -1, kind="stable")
    far = L[np.arange(len(L))[:, None, None], np.arange(2)[None, :, None], order]
    dhc = D[Hc, C]
    side = np.where(dhc < rz[C] + rz[Hc], Hc, C)
    one = k == 1
    b1 = side
    if one.any():
        second = full[np.maximum(L[..., 0], 0)]
        b1 = _np_farthest_of(D[Hc[..., None], np.maximum(second, 0)], second)
    many = k >= 3
    r1 = np.where(many | (k == 2), far[..., 0], np.where(one, L[..., 0], side))
    r2 = np.where(many, far[..., 1], np.where(k == 2, np.where(L[..., 1] != far[..., 0], L[..., 1], L[..., 0]),
                                              np.where(one, b1, side)))
    r3 = np.where(many, far[..., 2], side)
    keep = ~((na == B[:, None]).any(-1) | common.any((1, 2))) & (np.minimum(np.minimum(r1, r2), r3) >= 0).all(-1)
    if not keep.any():
        return None
    zc = zz[C]
    oxygen, nitrogen = zc == 8, zc == 7
    tcs2 = oxygen & one
    kind = np.where(oxygen, np.where(one, 1, 2), np.where(nitrogen, np.where(k == 2, 3, 4), 0))
    atoms = np.stack([C, r1, r2, r3], -1)
    flags = np.stack([kind == 4, tcs2, (dhc > D[Hc, r1]) & tcs2, (r2 == r3) | (Hc == r3)], -1)
    A, B = A[keep], B[keep]
    scale = _NP_HB_SCALE[zz]
    return dict(ahb=t[keep], atoms=atoms[keep], flags=flags[keep], shifts=_NP_HB_SHIFTS[kind[keep]],
                s_c=0.5 * (scale[A] + scale[B]), oo=(zz[A] == 8) & (zz[B] == 8))


def correction_tables(device: torch.device) -> Dict[str, Tensor]:
    """The element data of the corrections by atomic number (``MAX_Z + 1`` entries): bond radii (metals -1,
    none 999), covalent radii (none 0), hydrogen-bond radii (scaled by 4/3) and the PM7 dispersion data
    (``disp_has``, ``disp_c6``, ``disp_r``, ``disp_n``)."""
    key = str(device)
    out = _TABLES.get(key)
    if out is None:
        n = el.MAX_Z + 1
        bond = [(-1.0 if v in el.METALS else (el.COVALENT_RADIUS[v] or 999.0)) for v in range(n)]
        cov = [el.COVALENT_RADIUS[v] or 0.0 for v in range(n)]
        hb = [4.0 / 3.0 * r for r in _HB_RADII] + [0.0] * (n - len(_HB_RADII))
        out = {
            "bond_radius": torch.tensor(bond, dtype=torch.float64, device=device),
            "covalent": torch.tensor(cov, dtype=torch.float64, device=device),
            "hb_radius": torch.tensor(hb, dtype=torch.float64, device=device),
            "hb_angles": torch.tensor(_NP_HB_SHIFTS, dtype=torch.float64, device=device),
            "hb_scale": torch.tensor(_NP_HB_SCALE, dtype=torch.float64, device=device),
            "disp_has": torch.tensor([v in _DISP_C6 for v in range(n)], device=device),
            "disp_c6": torch.tensor([_DISP_C6.get(v, 0.0) for v in range(n)], dtype=torch.float64, device=device),
            "disp_r": torch.tensor([_DISP_R.get(v, 0.0) for v in range(n)], dtype=torch.float64, device=device),
            "disp_n": torch.tensor([_DISP_N.get(v, 0.0) for v in range(n)], dtype=torch.float64, device=device),
        }
        _TABLES[key] = out
    return out


_TABLES: Dict[str, Dict[str, Tensor]] = {}


def _exact(values: List[float], device: torch.device) -> Tensor:
    """``values`` as a float64 tensor (in TorchScript ``torch.tensor`` of floats and the assignment of a float
    to an element round through float32; ``torch.full`` keeps the value)."""
    return torch.stack([torch.full((), v, dtype=torch.float64, device=device) for v in values])


def distances(x: Tensor) -> Tensor:
    """The distance matrix of detached positions, each entry ``sqrt(dx^2 + dy^2 + dz^2)`` in float64 (the
    order of operations of a sum over the three components)."""
    d = x[:, None, :] - x[None, :, :]
    return torch.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])


def _empty_lists(n: int) -> List[List[int]]:
    out: List[List[int]] = []
    for _ in range(n):
        out.append(torch.jit.annotate(List[int], []))
    return out


def bond_topology(z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> List[List[int]]:
    """Bonded neighbors of every atom of one molecule (:func:`_bond_topology`; on the host in an eager model)."""
    if torch.jit.is_scripting():
        return _bond_topology(z, pos, tables)
    return _host_bonds(z, pos)


@torch.jit.unused
def _host_bonds(z: List[int], pos: Tensor) -> List[List[int]]:
    return _np_bond_topology(z, pos)


def _bond_topology(z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> List[List[int]]:
    """Bonded neighbors of every atom of one molecule.

    Two atoms are bonded when closer than ``f (r_i + r_j)`` with the covalent
    radii of :data:`~xnn.sqmnn.common.elements.COVALENT_RADIUS` (alkali and
    alkaline-earth metals count with radius -1) and ``f = 1.1``, except 1.25
    for C-H, 1.0 for B-N, 1.2 for C-C, C-N and S-S; at most 15 bonds per
    atom (pairs taken in order of the first atom, then the second), and a
    hydrogen with several bonds keeps only those to heavy atoms.
    """
    n = len(z)
    bonds = _empty_lists(n)
    if n < 2:
        return bonds
    xyz = pos.detach().to(torch.float64)
    zz = torch.tensor(z, dtype=torch.long, device=xyz.device)
    radius = tables["bond_radius"].to(xyz.device)[zz]
    lo, hi = torch.minimum(zz[:, None], zz[None, :]), torch.maximum(zz[:, None], zz[None, :])
    f = torch.full((n, n), 1.1, dtype=torch.float64, device=xyz.device)
    f = torch.where((lo == 1) & (hi == 6), 1.25, f)
    f = torch.where((lo == 5) & (hi == 7), 1.0, f)
    f = torch.where((lo == 6) & (hi < 8), 1.2, f)
    f = torch.where((lo == 16) & (hi == 16), 1.2, f)
    d = xyz[:, None, :] - xyz[None, :, :]
    d2 = d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2
    below = torch.ones((n, n), dtype=torch.bool, device=xyz.device).tril(-1)
    near = (d2 < (f * (radius[:, None] + radius[None, :])) ** 2) & below
    pairs = torch.jit.annotate(List[List[int]], near.nonzero().tolist())
    for p in pairs:
        i, j = p[0], p[1]
        if len(bonds[i]) < 15 and len(bonds[j]) < 15:
            bonds[i].append(j)
            bonds[j].append(i)
    for i in range(n):
        if z[i] == 1 and len(bonds[i]) >= 2:
            heavy: List[int] = []
            for j in bonds[i]:
                if z[j] != 1:
                    heavy.append(j)
            bonds[i] = heavy
    return bonds


def _distance(pos: Tensor, i: int, j: int) -> Tensor:
    return (pos[i] - pos[j]).norm()


def _angle(pos: Tensor, i: int, j: int, k: int) -> Tensor:
    """Angle ``i-j-k`` (radians) from the three distances, as MOPAC's bangle."""
    d2ij = (pos[i] - pos[j]).pow(2).sum()
    d2jk = (pos[j] - pos[k]).pow(2).sum()
    d2ik = (pos[i] - pos[k]).pow(2).sum()
    c = 0.5 * (d2ij + d2jk - d2ik) / torch.sqrt(d2ij * d2jk)
    return torch.acos(c.clamp(-1.0, 1.0))


def _sin2_dihedral(pos: Tensor, i: int, j: int, k: int, l: int) -> Tensor:
    b1, b2, b3 = pos[j] - pos[i], pos[k] - pos[j], pos[l] - pos[k]
    n1, n2 = torch.cross(b1, b2, dim=0), torch.cross(b2, b3, dim=0)
    cos = (n1 * n2).sum() / (n1.norm() * n2.norm())
    return 1.0 - cos * cos


def _at(out: Tensor, i: int, value: Tensor) -> Tensor:
    return out.index_add(0, torch.tensor([i], device=out.device), value.reshape(1))


def peptide_torsion(z: List[int], pos: Tensor, barrier: float) -> Tensor:
    """Per-atom ``barrier * sin^2`` terms of the amide torsions (kcal/mol).

    Each H-N-C=O group found by distance (C=O <= 1.3, C-N <= 1.6, N-H <= 1.3,
    N-X <= 1.7 Angstrom) contributes the torsions O=C-N-X and O=C-N-H; the
    energy is assigned to the nitrogen.
    """
    out = pos.new_zeros(len(z))
    if barrier == 0.0:
        return out
    xyz = pos.detach()
    n = len(z)
    used_n: List[int] = []
    for j in range(n):
        if z[j] != 6:
            continue
        found = False
        for i in range(n):
            if found or z[i] != 8 or float((xyz[i] - xyz[j]).norm()) > 1.3:
                continue
            for kk in range(n):
                if found or z[kk] != 7 or float((xyz[kk] - xyz[j]).norm()) > 1.6:
                    continue
                for ll in range(n):
                    if found or z[ll] != 1 or float((xyz[kk] - xyz[ll]).norm()) > 1.3:
                        continue
                    for m in range(n):
                        if m == kk or m == ll or m == j or float((xyz[m] - xyz[kk]).norm()) > 1.7:
                            continue
                        if kk in used_n:
                            continue
                        used_n.append(kk)
                        out = _at(out, kk, barrier * (_sin2_dihedral(pos, i, j, kk, m)
                                                      + _sin2_dihedral(pos, i, j, kk, ll)))
                        found = True
                        break
    return out


def yne_correction(z: List[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """12 kcal/mol per C-C bond shorter than 1.21 Angstrom, smoothly to zero at 1.33."""
    out = pos.new_zeros(len(z))
    rmin, rmax = 1.21, 1.33
    for i in range(len(z)):
        if z[i] != 6:
            continue
        for j in bonds[i]:
            if j > i or z[j] != 6:
                continue
            r = _distance(pos, i, j)
            rv = float(r)
            if rv < rmin:
                out = _at(out, i, torch.full((1,), 12.0, dtype=pos.dtype, device=pos.device))
            elif rv < rmax:
                x = (r - rmin) / (rmax - rmin)
                f = (1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5
                     + (-5.0 + 25.0 * x) * (x ** 3 - 3.0 * x ** 4 + 3.0 * x ** 5 - x ** 6))
                out = _at(out, i, 12.0 * f)
    return out


def nsp2_correction(z: List[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """Planarity stabilization of nitrogens with three ligands, at most one of them H."""
    out = pos.new_zeros(len(z))
    for i in range(len(z)):
        if z[i] != 7 or len(bonds[i]) != 3:
            continue
        a, b, c = bonds[i][0], bonds[i][1], bonds[i][2]
        hydrogens = 0
        for x in (a, b, c):
            if z[x] == 1:
                hydrogens += 1
        if hydrogens >= 2:
            continue
        tot = 2.0 * math.pi - (_angle(pos, b, i, c) + _angle(pos, a, i, c) + _angle(pos, b, i, a))
        out = _at(out, i, -0.5 * torch.exp(-10.0 * tot))
    return out


def si_o_h_correction(z: List[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """PM7 bending term ``15 (theta - 125 deg)^2`` of Si-O-H, damped beyond Si-O 1.7 / O-H 1.0."""
    out = pos.new_zeros(len(z))
    ref = 125.0 * math.pi / 180.0
    for o in range(len(z)):
        if z[o] != 8:
            continue
        si, h = -1, -1
        for k in bonds[o]:
            if z[k] == 14:
                si = k
            if z[k] == 1:
                h = k
        if si < 0 or h < 0:
            continue
        r_si = (pos[o] - pos[si]).pow(2).sum() - 1.7 ** 2
        r_h = (pos[o] - pos[h]).pow(2).sum() - 1.0
        theta = _angle(pos, si, o, h)
        e = (15.0 * (theta - ref) ** 2 * torch.exp(-33.0 * torch.clamp(r_si, min=0.0))
             * torch.exp(-68.0 * torch.clamp(r_h, min=0.0)))
        out = _at(out, o, e)
    return out


def _dispersion_tables(z: List[int], bonds: List[List[int]], tables: Dict[str, Tensor],
                       device: torch.device) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """The atom pairs ``(ia, ib)`` of the elements with dispersion data, their C6 and R0 constants."""
    has = torch.jit.annotate(List[bool], tables["disp_has"].tolist())
    have: List[int] = []
    for i in range(len(z)):
        if has[z[i]]:
            have.append(i)
    if len(have) < 2:
        return None
    zh = torch.tensor([z[i] for i in have], dtype=torch.long, device=device)
    four = torch.tensor([len(bonds[i]) == 4 for i in have], dtype=torch.bool, device=device)
    carbon = _exact([0.95, 1.65], device)
    c6 = torch.where(zh == 6, torch.where(four, carbon[0], carbon[1]), tables["disp_c6"].to(device)[zh])
    nn_ = tables["disp_n"].to(device)[zh]
    rr = tables["disp_r"].to(device)[zh]
    ab = torch.triu_indices(len(have), len(have), 1, device=device)
    a, b = ab[0], ab[1]
    c6ab = (2.0 * (c6[a] ** 2 * c6[b] ** 2 * nn_[a] * nn_[b]) ** (1.0 / 3.0)
            / ((c6[a] * nn_[b] ** 2) ** (1.0 / 3.0) + (c6[b] * nn_[a] ** 2) ** (1.0 / 3.0)))
    r0 = (rr[a] ** 3 + rr[b] ** 3) / (rr[a] ** 2 + rr[b] ** 2) / 1000.0 * 2.0
    idx = torch.tensor(have, dtype=torch.long, device=device)
    return idx[a], idx[b], r0, c6ab


def dispersion_energy(ops: TorchOps, p: Tensor, ia: Tensor, ib: Tensor, r0: Tensor, c6ab: Tensor, n: int,
                      alpha: float, s: float, scale: float) -> Tensor:
    """The PM7 dispersion per atom of the pairs ``ia``, ``ib`` (a term of :func:`~.tape.host_term`)."""
    ja, jb = ops.array(ia, p), ops.array(ib, p)
    rij = ops.norm(ops.take(p, ja) - ops.take(p, jb))
    rnm = rij * 0.1
    damp = 1.0 / (1.0 + ops.exp(-alpha * (rnm / (s * ops.array(r0, p)) - 1.0)))
    r3 = rnm * rnm * rnm
    e = -scale * ops.array(c6ab, p) / (r3 * r3) * damp / (1000.0 * 4.184)
    e = 0.5 * ops.where(ops.value(rij) < 100.0, e, 0.0)
    return ops.index_add(n, ja, e) + ops.index_add(n, jb, e)


def pm7_dispersion(z: List[int], pos: Tensor, bonds: List[List[int]], tables: Dict[str, Tensor],
                   alpha: float = 15.450118, s: float = 1.226593, scale: float = 2.286419) -> Tensor:
    """PM7 dispersion (Jurecka-type damped C6/R^6, Stewart 2013), per atom (kcal/mol).

    ``E = -scale sum_ij C6_ij / R^6 / (1 + exp(-alpha (R / (s R0) - 1)))`` over
    the pairs of elements with data; carbon's C6 is 0.95 with four bonds and
    1.65 otherwise; Slater-Kirkwood combination of the C6 values.
    """
    t = _dispersion_tables(z, bonds, tables, pos.device) if torch.jit.is_scripting() else _host_dispersion(z, bonds,
                                                                                                         pos)
    if t is None:
        return pos.new_zeros(len(z))
    ia, ib, r0, c6ab = t
    n = len(z)
    if torch.jit.is_scripting():
        return dispersion_energy(TorchOps(), pos, ia, ib, r0, c6ab, n, alpha, s, scale)
    return _on_host_dispersion(pos, ia, ib, r0, c6ab, n, alpha, s, scale)


@torch.jit.unused
def _host_dispersion(z: List[int], bonds: List[List[int]], pos: Tensor
                     ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """:func:`_dispersion_tables` on the host (NumPy), the form of an eager model."""
    have = [i for i, v in enumerate(z) if v in _DISP_C6]
    if len(have) < 2:
        return None
    c6 = np.array([(0.95 if len(bonds[i]) == 4 else 1.65) if z[i] == 6 else _DISP_C6[z[i]] for i in have])
    nn_ = np.array([_DISP_N[z[i]] for i in have])
    rr = np.array([_DISP_R[z[i]] for i in have])
    a, b = np.triu_indices(len(have), 1)
    c6ab = (2.0 * (c6[a] ** 2 * c6[b] ** 2 * nn_[a] * nn_[b]) ** (1.0 / 3.0)
            / ((c6[a] * nn_[b] ** 2) ** (1.0 / 3.0) + (c6[b] * nn_[a] ** 2) ** (1.0 / 3.0)))
    r0 = (rr[a] ** 3 + rr[b] ** 3) / (rr[a] ** 2 + rr[b] ** 2) / 1000.0 * 2.0
    ia, ib = np.asarray(have)[a], np.asarray(have)[b]
    return tuple(torch.from_numpy(np.ascontiguousarray(v)).to(pos.device) for v in (ia, ib, r0, c6ab))


@torch.jit.unused
def _on_host_dispersion(pos: Tensor, ia: Tensor, ib: Tensor, r0: Tensor, c6ab: Tensor, n: int, alpha: float,
                        s: float, scale: float) -> Tensor:
    # the pair constants on the host, the distances on the tape (see :func:`~.tape.host_term`)
    ia, ib, r0, c6ab = ia.cpu(), ib.cpu(), r0.cpu(), c6ab.cpu()
    return host_term(lambda ops, p: dispersion_energy(ops, p, ia, ib, r0, c6ab, n, alpha, s, scale), pos)


def _hbond_candidates(z: List[int], x: Tensor, D: Tensor, r_xh: float, cutoff: float, heavy: List[int]) -> Tensor:
    """Candidate hydrogen bonds ``(donor, H, acceptor)`` ``(T, 3)`` found by distance and angle (``D`` the
    distance matrix of :func:`distances`), in the order of a loop over the heavy atoms, their hydrogens and the
    other heavy atoms; a bond found from both ends (a hydrogen within ``r_xh`` of both) in its first order."""
    device = x.device
    none = torch.zeros((0, 3), dtype=torch.long, device=device)
    zz = torch.tensor(z, dtype=torch.long, device=device)
    is_heavy = zz == heavy[0]
    for v in heavy[1:]:
        is_heavy = is_heavy | (zz == v)
    acc = is_heavy.nonzero().squeeze(-1)
    is_h = zz == 1
    if acc.shape[0] == 0:
        return none
    # the hydrogens in the order the heavy atoms claim them (each by the first heavy atom within r_xh)
    near = (D[acc] < r_xh) & is_h[None, :]                                            # (n_heavy, n)
    claimed = near.any(0)
    first = torch.where(claimed, near.to(torch.long).argmax(0), torch.full_like(zz, acc.shape[0]))
    hyd = torch.argsort(first, stable=True)[:int(claimed.sum())]
    if hyd.shape[0] == 0:
        return none
    # every (i, j, k) in loop order: i heavy, j a claimed hydrogen within r_xh of i, then k heavy (not i) within
    # the cutoff of i; row-major nonzero keeps the order
    pij = (D[acc][:, hyd] < r_xh).nonzero()
    pi, pj = pij[:, 0], pij[:, 1]
    close = D[acc][:, acc] < cutoff
    close.fill_diagonal_(False)
    ppk = close[pi].nonzero()
    p, pk = ppk[:, 0], ppk[:, 1]
    i, j, k = acc[pi[p]], hyd[pj[p]], acc[pk]
    if i.shape[0] == 0:
        return none
    v1, v2 = x[k] - x[j], x[i] - x[j]
    # the angle k-j-i at the hydrogen, the three products summed in order
    cosang = (v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1] + v1[:, 2] * v2[:, 2]) / (D[k, j] * D[i, j])
    wide = torch.acos(torch.clamp(cosang, -1.0, 1.0)) > 0.5 * 3.141592653589793
    i, j, k = i[wide], j[wide], k[wide]
    if i.shape[0] == 0:
        return none
    # the triples are distinct; one is dropped where its reverse comes earlier
    n = len(z)
    key, back = (i * n + j) * n + k, (k * n + j) * n + i
    order = torch.argsort(key)
    where = torch.searchsorted(key[order], back).clamp(max=key.shape[0] - 1)
    earlier = (key[order][where] == back) & (order[where] < torch.arange(key.shape[0], device=device))
    return torch.stack([i, j, k], 1)[~earlier]


def _hbond_triples(z: List[int], x: Tensor, D: Tensor, r_xh: float, cutoff: float,
                   heavy: List[int]) -> List[List[int]]:
    """:func:`_hbond_candidates` as a list."""
    return torch.jit.annotate(List[List[int]], _hbond_candidates(z, x, D, r_xh, cutoff, heavy).tolist())


def _row(D: Tensor, cache: Dict[int, List[float]], i: int) -> List[float]:
    """Row ``i`` of the distance matrix as numbers (read once; the matrix is symmetric)."""
    if i not in cache:
        cache[i] = torch.jit.annotate(List[float], D[i].tolist())
    return cache[i]


def _bonded_lists(D: Tensor, rz: Tensor, cap: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """The atoms bonded to every atom (within the sum of the radii ``rz``) in index order, ``(n, m)`` padded
    with -1, and their counts ``(n,)``; then the same capped at ``cap`` (MOPAC's list, which drops the longest
    bond whenever it meets one more: the ``cap`` shortest, the higher index first among equal lengths)."""
    n = D.shape[0]
    device = D.device
    bonded = D < rz[:, None] + rz[None, :]
    bonded.fill_diagonal_(False)
    pairs = bonded.nonzero()
    c, j = pairs[:, 0], pairs[:, 1]
    count = bonded.sum(1)
    start = torch.cumsum(count, 0) - count
    slot = torch.arange(pairs.shape[0], device=device) - start[c]
    full = torch.full((n, max(int(count.max()), 1)), -1, dtype=torch.long, device=device)
    full.index_put_((c, slot), j)
    # rank of every pair in its row by (length, then the higher index first)
    idx = torch.argsort(j, descending=True, stable=True)
    idx = idx[torch.argsort(D[c, j][idx], stable=True)]
    idx = idx[torch.argsort(c[idx], stable=True)]
    rank = torch.empty_like(slot)
    rank[idx] = torch.arange(pairs.shape[0], device=device) - start[c[idx]]
    kept = rank < cap
    ck, jk = c[kept], j[kept]
    count_k = torch.clamp(count, max=cap)
    start_k = torch.cumsum(count_k, 0) - count_k
    capped = torch.full((n, cap), -1, dtype=torch.long, device=device)
    capped.index_put_((ck, torch.arange(ck.shape[0], device=device) - start_k[ck]), jk)
    return full, count, capped, count_k


def _farthest_of(dh: Tensor, cands: Tensor) -> Tensor:
    """Per row, the first of the atoms ``cands`` (-1 padding) farthest by ``dh`` (their distances, same shape),
    -1 where there is none."""
    valid = cands >= 0
    pos = torch.where(valid, dh, torch.full_like(dh, -1.0)).argmax(-1, keepdim=True)
    return torch.where(valid.any(-1), cands.gather(-1, pos).squeeze(-1), torch.full_like(pos.squeeze(-1), -1))


def _pm7_hbond_tables(z: List[int], x: Tensor, tables: Dict[str, Tensor]) -> Optional[Dict[str, Tensor]]:
    """The candidate PM7 hydrogen bonds of the positions ``x`` as tables (``None`` if none): atoms ``A, H, B``
    ``(R, 3)``; per end (donor and acceptor side, second axis) the atoms ``center, r1, r2, r3`` ``(R, 2, 4)``
    (MOPAC's DH+ reference atoms), the flags ``tcs, tcs2, zero_t, one_t`` ``(R, 2, 4)`` and the shifts ``shift,
    shift2, tshift`` ``(R, 2, 3)``; the scale ``s_c`` and the O...O flag ``(R,)``."""
    device = x.device
    D = distances(x)
    t = _hbond_candidates(z, x, D, 1.4, 7.0, [7, 8])
    if t.shape[0] == 0:
        return None
    zz = torch.tensor(z, dtype=torch.long, device=device)
    rz = tables["hb_radius"].to(device)[zz]
    full, _, nbr, count = _bonded_lists(D, rz, 4)
    A, H, B = t[:, 0], t[:, 1], t[:, 2]
    # no shared neighbor other than the hydrogen
    na, nb = nbr[A], nbr[B]
    common = (na[:, :, None] == nb[:, None, :]) & (na[:, :, None] >= 0) & (na[:, :, None] != H[:, None, None])
    C = torch.stack([A, B], 1)                                                         # (R, 2)
    Hc = torch.stack([H, H], 1)
    L = nbr[C]                                                                         # (R, 2, 4)
    k = count[C]
    dh = D[Hc[..., None], L.clamp(min=0)]
    # the bonded atoms farthest from the hydrogen first (the first among equal distances)
    far = L.gather(-1, torch.argsort(torch.where(L >= 0, -dh, torch.full_like(dh, float("inf"))), stable=True))
    dhc = D[Hc, C]
    side = torch.where(dhc < rz[C] + rz[Hc], Hc, C)
    one = k == 1
    b1 = side
    if bool(one.any()):
        # one bond: the atom bonded to it farthest from the hydrogen
        second = full[L[..., 0].clamp(min=0)]
        b1 = _farthest_of(D[Hc[..., None], second.clamp(min=0)], second)
    many = k >= 3
    r1 = torch.where(many | (k == 2), far[..., 0], torch.where(one, L[..., 0], side))
    r2 = torch.where(many, far[..., 1], torch.where(
        k == 2, torch.where(L[..., 1] != far[..., 0], L[..., 1], L[..., 0]), torch.where(one, b1, side)))
    r3 = torch.where(many, far[..., 2], side)
    keep = ~((na == B[:, None]).any(-1) | common.any(-1).any(-1)) & \
        (torch.minimum(torch.minimum(r1, r2), r3) >= 0).all(-1)
    if not bool(keep.any()):
        return None
    zc = zz[C]
    oxygen, nitrogen = zc == 8, zc == 7
    tcs2 = oxygen & one
    zero = torch.zeros_like(k)
    # the kind of end, an index into the rows of the angles
    kind = torch.where(oxygen, torch.where(one, zero + 1, zero + 2),
                       torch.where(nitrogen, torch.where(k == 2, zero + 3, zero + 4), zero))
    atoms = torch.stack([C, r1, r2, r3], -1)
    flags = torch.stack([kind == 4, tcs2, (dhc > D[Hc, r1]) & tcs2, (r2 == r3) | (Hc == r3)], -1)
    A, B = A[keep], B[keep]
    scale = tables["hb_scale"].to(device)[zz]
    out: Dict[str, Tensor] = {
        "ahb": t[keep], "atoms": atoms[keep], "flags": flags[keep],
        "shifts": tables["hb_angles"].to(device)[kind[keep]],
        "s_c": 0.5 * (scale[A] + scale[B]), "oo": (zz[A] == 8) & (zz[B] == 8)}
    return out


def pm7_hbond_energy(ops: TorchOps, z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> Optional[Tensor]:
    """The PM7 hydrogen-bond energies per atom (a term of :func:`~.tape.host_term`; ``None`` without bonds)."""
    deg = math.pi / 180.0
    x = ops.host(pos)
    t = _pm7_hbond_tables(z, x, tables) if torch.jit.is_scripting() else _host_tables(0, z, x, ops.host_arrays)
    if t is None:
        return None
    ahb, atoms, flags = ops.array(t["ahb"], pos), ops.array(t["atoms"], pos), ops.array(t["flags"], pos)
    shifts, s_c, oo = ops.array(t["shifts"], pos), ops.array(t["s_c"], pos), ops.array(t["oo"], pos)
    last = ops.bools([False, True], pos)
    A, H, B = ahb[:, 0], ahb[:, 1], ahb[:, 2]
    cos_hb = -ops.cos(ops.angle(pos, A, H, B))
    keep = ops.value(cos_hb) > 0.0
    center, r1, r2, r3 = atoms[..., 0], atoms[..., 1], atoms[..., 2], atoms[..., 3]           # (R, 2)
    tcs, tcs2, zero_t, one_t = flags[..., 0], flags[..., 1], flags[..., 2], flags[..., 3]
    shift, shift2, tshift = shifts[..., 0], shifts[..., 1], shifts[..., 2]
    Hc = H[:, None] + 0 * center
    # the dihedrals r2-r1-center-r3 and r2-r1-center-H together
    two = 0 * r3[..., None] + ops.longs([0, 0], pos)
    last_atom = ops.vwhere(ops.bools([True, False], pos), r3[..., None] + 0 * two, Hc[..., None] + 0 * two)
    both = ops.dihedral(pos, r2[..., None] + two, r1[..., None] + two, center[..., None] + two, last_atom)
    tc = ops.reflect(ops.wrap(both[..., 0]))
    check = ops.vwhere(tcs, ops.value(tc), 0.0 * ops.value(tc))
    tcd = ops.abs(tc) / deg
    tshift = ops.where(tcs, tshift + ((54.74 - tcd) / 54.74 * 35.26) * deg, tshift)
    shift = ops.where(tcs, shift - ((54.74 - tcd) / 54.74 * 19.48) * deg, shift)
    shift2 = ops.where(tcs, shift, shift2)
    ang2 = ops.angle(pos, r1, center, Hc)
    c2 = ops.maximum(ops.cos(shift - ang2), ops.cos(shift2 - ang2))
    keep = keep & (ops.value(c2) > 0.0).all(-1)
    tcorr = ops.wrap(both[..., 1])
    flip = ~tcs2 | (ops.vabs(ops.value(tcorr)) / deg > 90.0)
    tcorr = ops.where(flip, ops.reflect(tcorr), tcorr)
    minus, plus = ops.cos(ops.wrap(tshift - tcorr)), ops.cos(ops.wrap(-tshift - tcorr))
    tcos = ops.where(check < 0, minus, ops.where(check > 0, plus, ops.maximum(minus, plus)))
    tcos = ops.where(zero_t, 0.0, tcos)
    tcos = ops.where(one_t, 1.0, tcos)
    # the acceptor side takes the magnitude, the donor side must not be negative
    keep = keep & (ops.value(tcos)[:, 0] >= 0.0)
    tcos = ops.where(last, ops.abs(tcos), tcos)
    hartree_kcal = 27.211386245988 * 23.060547830619029
    pa, ph, pb = ops.take(pos, A), ops.take(pos, H), ops.take(pos, B)
    ha, hb = ops.norm(ph - pa), ops.norm(ph - pb)
    xc = ops.minimum(ha, hb)
    damping = ops.where(ops.vabs(ops.value(ha) - ops.value(hb)) > 0.5,
                        1.0 - 1.0 / (1.0 + ops.exp(-60.0 * (xc / 1.2 - 1.0))), 1.0)
    rab = ops.norm(pa - pb)
    damping = damping / (1.0 + ops.exp(-100.0 * (rab / 2.4 - 1.0)))
    damping = damping * (1.0 - 1.0 / (1.0 + ops.exp(-10.0 * (rab / 7.0 - 1.0))))
    prod = c2[:, 0] * tcos[:, 0] * c2[:, 1] * tcos[:, 1]
    cos2 = cos_hb * cos_hb
    e = s_c / (rab * rab) * cos2 * (1.0 - (1.0 - prod) * (1.0 - prod)) * hartree_kcal * damping
    short = ops.clamp(rab - 2.67, 0.0)
    e = e - ops.where(oo, 2.5 * ops.exp(-80.0 * (short * short)) * (cos2 * cos2), 0.0)
    # only the bonds that take part enter the sum (and its derivatives)
    kept = ops.nonzero(keep)
    if len(kept) == 0:
        return None
    return ops.index_add(len(z), H[kept], e[kept])


def pm7_hbond(z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> Tensor:
    """PM7 hydrogen-bond correction, per hydrogen atom (kcal/mol).

    The third-generation correction of Korth (J. Chem. Theory Comput. 6, 3808,
    2010) with the PM7 scales and damping (Stewart 2013): for every
    X-H...Y (X, Y = N, O) found by distance (X-H < 1.4, X...Y < 7 Angstrom)
    and angle,

        E = s / R_XY^2 cos^2(theta) [1 - (1 - c_X t_X c_Y t_Y)^2] f_damp

    with ``theta`` the deviation of X-H...Y from linearity and ``c``, ``t``
    the cosines of the deviations of the lone-pair angle and torsion on each
    side from their hybridization targets, plus an extra short-range O...O
    term. The groups and reference atoms are chosen on the detached geometry;
    the energies of all hydrogen bonds are evaluated together (on the host, see
    :func:`~.tape.host_term`).
    """
    if torch.jit.is_scripting():
        e = pm7_hbond_energy(TorchOps(), z, pos, tables)
        return e if e is not None else pos.new_zeros(len(z))
    return _on_host(0, z, pos, tables)


@torch.jit.unused
def _on_host(kind: int, z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> Tensor:
    """A term on the host tape (see :func:`~.tape.host_term`): 0 PM7 hydrogen bonds, 1 H4, 2 H-H repulsion."""
    energy = (pm7_hbond_energy, h4_energy, lambda ops, z_, p, t_: hh_energy(ops, z_, p))[kind]
    return host_term(lambda ops, p: energy(ops, z, p, tables), pos)


def _horner(x: Tensor, coefficients: List[float]) -> Tensor:
    """``sum_k c_k x^k`` for the coefficients from the highest power down (Horner's scheme)."""
    out = coefficients[0] * x + coefficients[1]
    for c in coefficients[2:]:
        out = out * x + c
    return out


def _smoothstep7(x: Tensor) -> Tensor:
    x2 = x * x
    return x2 * x2 * _horner(x, [-20.0, 70.0, -84.0, 35.0])


def _h4_tables(z: List[int], x: Tensor, tables: Dict[str, Tensor]) -> Optional[Dict[str, Tensor]]:
    """The candidate H4 hydrogen bonds of the positions ``x`` as tables (``None`` if none): per bond the atoms
    ``A, B, H, donor, acceptor, C, O2`` (``tab``, the carboxylate carbon and its other oxygen -1 where none),
    the scale ``c`` and the group masks; and the atom pairs of the smooth bond counts (``pa``, ``pb``, their
    radius sums ``lo``, whether within them ``inner``, whether ``pb`` is a hydrogen ``b_h``)."""
    device = x.device
    D = distances(x)
    triples = _hbond_triples(z, x, D, 1.15, 5.5, [7, 8, 16])
    if len(triples) == 0:
        return None
    zz = torch.tensor(z, dtype=torch.long, device=device)
    cov = tables["covalent"].to(device)[zz]
    r0 = cov[:, None] + cov[None, :]
    some = (D > 0.0) & (D < 1.6 * r0)
    some.fill_diagonal_(False)
    ab = some.nonzero()
    a, b = ab[:, 0], ab[:, 1]
    cache: Dict[int, List[float]] = {}
    rows: List[List[int]] = []
    for t in triples:
        i0, h, k0 = t[0], t[1], t[2]
        a_, b_ = k0, i0           # the order of the reference implementation's arguments
        dh = _row(D, cache, h)
        d, acc = (a_, b_) if dh[a_] < dh[b_] else (b_, a_)
        cc = o2 = -1
        if z[acc] == 8:
            # the carbon nearest the acceptor oxygen among its neighbors, and that carbon's other nearest
            # oxygen (the lowest index among equal distances)
            best = 0.0
            da = _row(D, cache, acc)
            for m in torch.jit.annotate(List[int], some[acc].nonzero().squeeze(-1).tolist()):
                if z[m] == 6 and (cc < 0 or da[m] < best):
                    cc, best = m, da[m]
            if cc >= 0:
                dc = _row(D, cache, cc)
                for m in torch.jit.annotate(List[int], some[cc].nonzero().squeeze(-1).tolist()):
                    if m != acc and z[m] == 8 and (o2 < 0 or dc[m] < best):
                        o2, best = m, dc[m]
        rows.append([a_, b_, h, d, acc, cc, o2])
    tab = torch.tensor(rows, dtype=torch.long, device=device)
    carbox = (zz[tab[:, 4]] == 8) & (tab[:, 5] >= 0) & (tab[:, 6] >= 0)
    out: Dict[str, Tensor] = {
        "tab": tab,
        "c": _exact([0.0, 2.32, 3.10, 1.07, 2.01], device)[
            torch.tensor([_h4_code(z[r[3]], z[r[4]]) for r in rows], dtype=torch.long, device=device)],
        "water": (zz[tab[:, 3]] == 8) & (zz[tab[:, 4]] == 8), "ammonium": zz[tab[:, 3]] == 7, "carbox": carbox,
        "cc_safe": torch.where(carbox, tab[:, 5], tab[:, 4]), "o2_safe": torch.where(carbox, tab[:, 6], tab[:, 4]),
        "pa": a, "pb": b, "lo": r0[a, b], "inner": D[a, b] <= r0[a, b], "b_h": zz[b] == 1}
    return out


def _np_h4_tables(z: Sequence[int], x: np.ndarray):
    """:func:`_h4_tables` on the host (NumPy), the form of an eager model."""
    D = _distances(x)
    triples = _np_hbond_triples(z, x, D, 1.15, 5.5, heavy=(7, 8, 16)).tolist()
    if not triples:
        return None
    cov = np.array([el.COVALENT_RADIUS[v] or 0.0 for v in z])
    r0 = cov[:, None] + cov[None, :]
    some = (D > 0.0) & (D < 1.6 * r0)
    np.fill_diagonal(some, False)
    a, b = np.nonzero(some)
    rows = []
    for (i0, h, k0) in triples:
        a_, b_ = k0, i0           # the order of the reference implementation's arguments
        d, acc = (a_, b_) if D[a_, h] < D[b_, h] else (b_, a_)
        cc = o2 = -1
        if z[acc] == 8:
            cands = [m for m in np.nonzero(some[acc])[0] if z[m] == 6]
            if cands:
                cc = int(min(cands, key=lambda m: (D[acc, m], m)))
                oxy = [m for m in np.nonzero(some[cc])[0] if m != acc and z[m] == 8]
                if oxy:
                    o2 = int(min(oxy, key=lambda m: (D[cc, m], m)))
        rows.append((a_, b_, h, d, acc, cc, o2))
    tab = np.array(rows, dtype=np.int64)
    zz = np.asarray(z)
    carbox = (zz[tab[:, 4]] == 8) & (tab[:, 5] >= 0) & (tab[:, 6] >= 0)
    return dict(tab=tab, c=np.array([_h4_scale(z[r[3]], z[r[4]]) for r in rows], dtype=np.float64),
                water=(zz[tab[:, 3]] == 8) & (zz[tab[:, 4]] == 8), ammonium=zz[tab[:, 3]] == 7, carbox=carbox,
                cc_safe=np.where(carbox, tab[:, 5], tab[:, 4]), o2_safe=np.where(carbox, tab[:, 6], tab[:, 4]),
                pa=a, pb=b, lo=r0[a, b], inner=D[a, b] <= r0[a, b], b_h=zz[b] == 1)


@torch.jit.unused
def _host_tables(kind: int, z: List[int], x: Tensor, numpy_out: bool) -> Optional[Dict[str, Tensor]]:
    """The tables of a hydrogen-bond term (0 PM7, 1 H4) from the host (NumPy) topology searches, the form of an
    eager model: NumPy arrays for the tape (``numpy_out``), else tensors on the device of ``x``. The tape's
    search is handed, by atoms and exact positions, to the PyTorch form of the same evaluation's second
    derivatives (taken once)."""
    xn = x.detach().to("cpu", torch.float64).numpy()
    key = (kind, tuple(z), xn.tobytes())
    t = None if numpy_out else _TABLE_CACHE.pop(key, None)
    if t is None:
        t = (_np_pm7_hbond_tables, _np_h4_tables)[kind](list(z), xn)
    if numpy_out:
        _TABLE_CACHE[key] = t
        while len(_TABLE_CACHE) > 4:
            _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
    if t is None or numpy_out:
        return t
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(x.device) for k, v in t.items()}


_TABLE_CACHE: Dict[tuple, Optional[dict]] = {}


def _h4_code(zd: int, za: int) -> int:
    """The H4 scale of a donor and acceptor element as an index into (0, 2.32, 3.10, 1.07, 2.01)."""
    if zd == 8 and za == 8:
        return 1
    if zd == 8 and za == 7:
        return 2
    if zd == 7 and za == 8:
        return 3
    if zd == 7 and za == 7:
        return 4
    return 0


def _h4_scale(zd: int, za: int) -> float:
    if zd == 8 and za == 8:
        return 2.32
    if zd == 8 and za == 7:
        return 3.10
    if zd == 7 and za == 8:
        return 1.07
    if zd == 7 and za == 7:
        return 2.01
    return 0.0


def h4_energy(ops: TorchOps, z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> Optional[Tensor]:
    """The H4 hydrogen-bond energies per atom (a term of :func:`~.tape.host_term`; ``None`` without bonds)."""
    x = ops.host(pos)
    t = _h4_tables(z, x, tables) if torch.jit.is_scripting() else _host_tables(1, z, x, ops.host_arrays)
    if t is None:
        return None
    # smooth bond counts per atom: all neighbors, hydrogens and the others (a neighbor counts 1 within the sum
    # of the covalent radii r0, 0 beyond 1.6 r0, 1 - S7((r - r0) / (0.6 r0)) between)
    have_counts = t["pa"].shape[0] > 0
    cv = hyd = oth = t["lo"]                 # (typed placeholders, set below where there are pairs)
    if have_counts:
        ia, ib = ops.array(t["pa"], pos), ops.array(t["pb"], pos)
        r = ops.norm(ops.take(pos, ia) - ops.take(pos, ib))
        lo = ops.array(t["lo"], pos)
        v = ops.where(ops.array(t["inner"], pos), 1.0, 1.0 - _smoothstep7((r - lo) / (0.6 * lo)))
        cv = ops.index_add(len(z), ia, v)
        hyd = ops.index_add(len(z), ia, ops.where(ops.array(t["b_h"], pos), v, 0.0))
        oth = cv - hyd
    tab = t["tab"]
    A, B, H, Dn, Ac = (ops.array(tab[:, 0], pos), ops.array(tab[:, 1], pos), ops.array(tab[:, 2], pos),
                       ops.array(tab[:, 3], pos), ops.array(tab[:, 4], pos))
    c = ops.array(t["c"], pos)
    ph = ops.take(pos, H)
    rda = ops.norm(ops.take(pos, A) - ops.take(pos, B))
    rdh, rah = ops.norm(ops.take(pos, Dn) - ph), ops.norm(ops.take(pos, Ac) - ph)
    ang = math.pi - ops.angle(pos, A, H, B)
    e_rad = _horner(rda, [-0.00303407407407313510, 0.07357629629627092382, -0.70087111111082800452,
                          3.25309629629461749545, -7.20687407406838786983, 5.31754666665572184314,
                          3.40736000001102778967, -4.68512000000450434811])
    s_ang = _smoothstep7(ang / (math.pi / 2.0))
    e_ang = 1.0 - s_ang * s_ang
    far = ops.value(rdh) > 1.15
    sw = ops.where(far, (rdh - 1.15) / (0.5 * rdh + 0.5 * rah - 1.15), 0.0)
    e_sw = ops.where(far, 1.0 - _smoothstep7(sw), 1.0)
    e = c * e_rad * e_ang * e_sw
    if have_counts:
        # water donors and acceptors
        hd, od = ops.take(hyd, Dn), ops.take(oth, Dn)
        hv = ops.value(hd)
        fv = ops.where((hv > 1.0) & (hv <= 2.0), hd - 1.0, ops.where((hv > 2.0) & (hv < 3.0), 3.0 - hd, 0.0))
        water = ops.array(t["water"], pos) & (hv >= 1.0)
        e_w = ops.where(water, 1.0 + (0.42 - 1.0) * fv * ops.clamp(1.0 - od, 0.0), 1.0)
        # ammonium donors
        vd = ops.take(cv, Dn)
        e_chd = ops.where(ops.array(t["ammonium"], pos) & (ops.value(vd) > 3.0),
                          1.0 + (3.61 - 1.0) * ops.clamp(vd - 3.0, 0.0), 1.0)
        # carboxylate acceptors
        has = ops.array(t["carbox"], pos)
        cc_safe, o2_safe = ops.array(t["cc_safe"], pos), ops.array(t["o2_safe"], pos)
        f1 = ops.clamp(1.0 - ops.abs(1.0 - ops.take(cv, Ac)), 0.0)
        f2 = ops.clamp(1.0 - ops.abs(1.0 - ops.take(cv, o2_safe)), 0.0)
        f3 = ops.clamp(1.0 - ops.abs(3.0 - ops.take(cv, cc_safe)), 0.0)
        e_cha = ops.where(has, 1.0 + (1.41 - 1.0) * f1 * f2 * f3, 1.0)
        e = e * e_w * e_chd * e_cha
    return ops.index_add(len(z), H, e)


def h4_hbond(z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> Tensor:
    """Hydrogen-bond correction H4 (Rezac & Hobza, J. Chem. Theory Comput. 8,
    141, 2012) as used by PM6-ORG, per hydrogen atom (kcal/mol).

    For every X-H...Y (X, Y = N, O; X-H < 1.15, X...Y < 5.5 Angstrom) the
    energy ``c_XY E_rad(R_XY) E_ang(theta) f_switch f_water f_NR4+ f_COO-``
    with a seventh-order radial polynomial, the angular switch of the
    deviation from linearity, and the scalings of water, ammonium and
    carboxylate groups. The groups are chosen on the detached geometry; the
    energies of all hydrogen bonds are evaluated together (on the host, see
    :func:`~.tape.host_term`).
    """
    if torch.jit.is_scripting():
        e = h4_energy(TorchOps(), z, pos, tables)
        return e if e is not None else pos.new_zeros(len(z))
    return _on_host(1, z, pos, tables)


def hh_energy(ops: TorchOps, z: List[int], pos: Tensor) -> Optional[Tensor]:
    """The H-H repulsion per atom (a term of :func:`~.tape.host_term`; ``None`` below two hydrogens)."""
    hs: List[int] = []
    for i in range(len(z)):
        if z[i] == 1:
            hs.append(i)
    if len(hs) < 2:
        return None
    if torch.jit.is_scripting() or not ops.host_arrays:
        ab = torch.triu_indices(len(hs), len(hs), 1)
        hs_t = torch.tensor(hs, dtype=torch.long)
        ia, ib = ops.array(hs_t[ab[0]], pos), ops.array(hs_t[ab[1]], pos)
    else:
        ia, ib = _host_pairs(hs)
    r = ops.norm(ops.take(pos, ia) - ops.take(pos, ib))
    rc = ops.clamp(r, 1e-6)
    mid = _horner(rc, [-2714.952351603469651, 17103.650110591705015, -42511.857982217959943,
                       52063.196799138342612, -31430.658335972289933, 7516.084696095140316])
    far = 118.7326 * ops.exp(-1.53965 * rc ** 1.72905)
    vr = ops.value(r)
    e = 0.5 * ops.where(vr <= 1.0, 25.46293603147693, ops.where(vr < 1.5, mid, far))
    return ops.index_add(len(z), ia, e) + ops.index_add(len(z), ib, e)


@torch.jit.unused
def _host_pairs(hs: List[int]) -> Tuple[Tensor, Tensor]:
    """The pairs of the atoms ``hs`` as host index arrays (for the tape)."""
    a, b = np.triu_indices(len(hs), 1)
    return np.asarray(hs)[a], np.asarray(hs)[b]


def hh_repulsion(z: List[int], pos: Tensor, tables: Dict[str, Tensor]) -> Tensor:
    """Hydrogen-hydrogen repulsion of D3H4 (Rezac & Hobza 2012), per atom (kcal/mol)."""
    if torch.jit.is_scripting():
        e = hh_energy(TorchOps(), z, pos)
        return e if e is not None else pos.new_zeros(len(z))
    return _on_host(2, z, pos, tables)


def structure_corrections(kind: int, z: List[int], p: Tensor, tables: Dict[str, Tensor],
                          peptide_barrier: float) -> Tensor:
    """The corrections of one molecule per atom (kcal/mol) of a method (``kind``: 0 PM3, 1 PM6, 2 PM7,
    3 PM6-ORG without its D3 term)."""
    if kind == 0:
        return peptide_torsion(z, p, peptide_barrier)
    bonds = bond_topology(z, p, tables)
    barrier = 2.5 if kind == 1 else (3.1595 if kind == 2 else 0.0)
    out = peptide_torsion(z, p, barrier) + yne_correction(z, p, bonds)
    if kind == 1:
        out = out + nsp2_correction(z, p, bonds)
    if kind == 2:
        out = out + si_o_h_correction(z, p, bonds)
        out = out + pm7_dispersion(z, p, bonds, tables) + pm7_hbond(z, p, tables)
    if kind == 3:
        out = out + h4_hbond(z, p, tables) + hh_repulsion(z, p, tables)
    return out


def batch_corrections(kind: int, z_all: List[int], pos: Tensor, batch: Tensor, n_struct: int,
                      tables: Dict[str, Tensor], peptide_barrier: float) -> Tensor:
    """:func:`structure_corrections` of every structure of a batch, per atom."""
    out = pos.new_zeros(pos.shape[0])
    for b in range(n_struct):
        idx = (batch == b).nonzero().squeeze(-1)
        if idx.numel() == 0:
            continue
        ids = torch.jit.annotate(List[int], idx.tolist())
        zb: List[int] = []
        for i in ids:
            zb.append(z_all[i])
        out = out.index_add(0, idx, structure_corrections(kind, zb, pos[idx], tables, peptide_barrier))
    return out
