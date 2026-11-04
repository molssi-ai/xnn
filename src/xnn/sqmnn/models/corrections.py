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
energies are differentiable functions of the positions.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor

from ..common import elements as el


def _distances(pos: Tensor) -> np.ndarray:
    """The distance matrix of the detached positions, each entry ``sqrt(dx^2 + dy^2 + dz^2)`` in float64 (the
    order of operations of a sum over the three components)."""
    xyz = pos.detach().to("cpu", torch.float64).numpy()
    d = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])


def bond_topology(z: Sequence[int], pos: Tensor) -> List[List[int]]:
    """Bonded neighbors of every atom of one molecule.

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


def peptide_torsion(z: Sequence[int], pos: Tensor, barrier: float) -> Tensor:
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

    def d(a, b):
        return float((xyz[a] - xyz[b]).norm())

    used_n = set()
    for j in range(n):
        if z[j] != 6:
            continue
        found = False
        for i in range(n):
            if found or z[i] != 8 or d(i, j) > 1.3:
                continue
            for kk in range(n):
                if found or z[kk] != 7 or d(kk, j) > 1.6:
                    continue
                for ll in range(n):
                    if found or z[ll] != 1 or d(kk, ll) > 1.3:
                        continue
                    for m in range(n):
                        if m in (kk, ll, j) or d(m, kk) > 1.7:
                            continue
                        if kk in used_n:
                            continue
                        used_n.add(kk)
                        out = out.index_add(0, torch.tensor([kk], device=pos.device), barrier * (
                            _sin2_dihedral(pos, i, j, kk, m) + _sin2_dihedral(pos, i, j, kk, ll)).reshape(1))
                        found = True
                        break
    return out


def yne_correction(z: Sequence[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
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
                out = out.index_add(0, torch.tensor([i], device=pos.device),
                                    torch.full((1,), 12.0, dtype=pos.dtype, device=pos.device))
            elif rv < rmax:
                x = (r - rmin) / (rmax - rmin)
                f = (1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5
                     + (-5.0 + 25.0 * x) * (x ** 3 - 3.0 * x ** 4 + 3.0 * x ** 5 - x ** 6))
                out = out.index_add(0, torch.tensor([i], device=pos.device), (12.0 * f).reshape(1))
    return out


def nsp2_correction(z: Sequence[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """Planarity stabilization of nitrogens with three ligands, at most one of them H."""
    out = pos.new_zeros(len(z))
    for i in range(len(z)):
        if z[i] != 7 or len(bonds[i]) != 3:
            continue
        a, b, c = bonds[i]
        if sum(z[x] == 1 for x in (a, b, c)) >= 2:
            continue
        tot = 2.0 * math.pi - (_angle(pos, b, i, c) + _angle(pos, a, i, c) + _angle(pos, b, i, a))
        out = out.index_add(0, torch.tensor([i], device=pos.device),
                            (-0.5 * torch.exp(-10.0 * tot)).reshape(1))
    return out


def si_o_h_correction(z: Sequence[int], pos: Tensor, bonds: List[List[int]]) -> Tensor:
    """PM7 bending term ``15 (theta - 125 deg)^2`` of Si-O-H, damped beyond Si-O 1.7 / O-H 1.0."""
    out = pos.new_zeros(len(z))
    ref = 125.0 * math.pi / 180.0
    for o in range(len(z)):
        if z[o] != 8:
            continue
        si = h = None
        for k in bonds[o]:
            if z[k] == 14:
                si = k
            if z[k] == 1:
                h = k
        if si is None or h is None:
            continue
        r_si = (pos[o] - pos[si]).pow(2).sum() - 1.7 ** 2
        r_h = (pos[o] - pos[h]).pow(2).sum() - 1.0
        theta = _angle(pos, si, o, h)
        e = (15.0 * (theta - ref) ** 2 * torch.exp(-33.0 * torch.clamp(r_si, min=0.0))
             * torch.exp(-68.0 * torch.clamp(r_h, min=0.0)))
        out = out.index_add(0, torch.tensor([o], device=pos.device), e.reshape(1))
    return out


# Jurecka-type dispersion of PM7: C6 (J nm^6 / mol), van der Waals radii (pm) and
# Slater-Kirkwood effective electron numbers
_DISP_C6 = {1: 0.16, 2: 0.084, 5: 5.79, 6: 1.65, 7: 1.11, 8: 0.70, 9: 0.57, 10: 0.45, 15: 3.25,
            16: 5.79, 17: 5.97, 18: 3.71, 27: 0.04, 35: 11.60, 36: 4.47, 53: 25.80, 54: 16.50}
_DISP_R = {1: 156.0, 2: 140.0, 5: 180.0, 6: 170.0, 7: 155.0, 8: 152.0, 9: 147.0, 10: 154.0,
           15: 180.0, 16: 180.0, 17: 175.0, 18: 188.0, 27: 140.0, 35: 185.0, 36: 202.0,
           53: 198.0, 54: 216.0}
_DISP_N = {1: 0.80, 2: 1.42, 5: 2.16, 6: 2.50, 7: 2.82, 8: 3.15, 9: 3.48, 10: 3.81, 15: 4.50,
           16: 4.80, 17: 5.10, 18: 5.40, 27: 2.90, 35: 6.00, 36: 6.30, 53: 6.95, 54: 7.25}


def pm7_dispersion(z: Sequence[int], pos: Tensor, bonds: List[List[int]], alpha: float = 15.450118,
                   s: float = 1.226593, scale: float = 2.286419) -> Tensor:
    """PM7 dispersion (Jurecka-type damped C6/R^6, Stewart 2013), per atom (kcal/mol).

    ``E = -scale sum_ij C6_ij / R^6 / (1 + exp(-alpha (R / (s R0) - 1)))`` over
    the pairs of elements with data; carbon's C6 is 0.95 with four bonds and
    1.65 otherwise; Slater-Kirkwood combination of the C6 values.
    """
    out = pos.new_zeros(len(z))
    have = [i for i, v in enumerate(z) if v in _DISP_C6]
    if len(have) < 2:
        return out
    idx = torch.as_tensor(have, device=pos.device)
    c6 = [(0.95 if len(bonds[i]) == 4 else 1.65) if z[i] == 6 else _DISP_C6[z[i]] for i in have]
    c6 = torch.as_tensor(c6, dtype=pos.dtype, device=pos.device)
    nn_ = torch.as_tensor([_DISP_N[z[i]] for i in have], dtype=pos.dtype, device=pos.device)
    rr = torch.as_tensor([_DISP_R[z[i]] for i in have], dtype=pos.dtype, device=pos.device)
    a, b = torch.triu_indices(len(have), len(have), offset=1, device=pos.device)
    c6ab = (2.0 * (c6[a] ** 2 * c6[b] ** 2 * nn_[a] * nn_[b]) ** (1.0 / 3.0)
            / ((c6[a] * nn_[b] ** 2) ** (1.0 / 3.0) + (c6[b] * nn_[a] ** 2) ** (1.0 / 3.0)))
    r0 = (rr[a] ** 3 + rr[b] ** 3) / (rr[a] ** 2 + rr[b] ** 2) / 1000.0 * 2.0
    rij = (pos[idx[a]] - pos[idx[b]]).norm(dim=-1)
    keep = rij < 100.0
    rnm = rij * 0.1
    damp = 1.0 / (1.0 + torch.exp(-alpha * (rnm / (s * r0) - 1.0)))
    e = -scale * c6ab / rnm ** 6 * damp / (1000.0 * 4.184)
    e = torch.where(keep, e, torch.zeros_like(e))
    return out.index_add(0, idx[a], 0.5 * e).index_add(0, idx[b], 0.5 * e)


# covalent radii (Angstrom) of the hydrogen-bond topology, scaled by 4/3 in use
_HB_RADII = (0.0,
             0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04,
             1.10, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99,
             1.01, 1.09, 1.12, 1.09, 1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24,
             1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76,
             1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
             1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31,
             1.38, 1.42)


def _hbond_triples(z: Sequence[int], xyz, D: np.ndarray, r_xh: float, cutoff: float, heavy=(7, 8)):
    """Candidate hydrogen bonds ``(donor, H, acceptor)`` found by distance and angle (``D`` the distance
    matrix of :func:`_distances`)."""
    n = len(z)
    zz = np.asarray(z)
    acc = [i for i in range(n) if z[i] in heavy]
    used = np.zeros(n, dtype=bool)
    hyd = []
    for i in acc:
        for j in np.nonzero((zz == 1) & ~used & (D[i] < r_xh))[0]:
            hyd.append(int(j))
            used[j] = True
    triples, seen = [], set()
    for i in acc:
        for j in hyd:
            if D[i, j] >= r_xh:
                continue
            for k in acc:
                if k == i or D[k, i] >= cutoff:
                    continue
                # angle k-j-i (at the hydrogen)
                v1 = [xyz[k][c] - xyz[j][c] for c in range(3)]
                v2 = [xyz[i][c] - xyz[j][c] for c in range(3)]
                cosang = sum(a * b for a, b in zip(v1, v2)) / (D[k, j] * D[i, j])
                if math.acos(max(-1.0, min(1.0, cosang))) <= 0.5 * math.pi:
                    continue
                if (k, j, i) in seen or (i, j, k) in seen:
                    continue
                seen.add((i, j, k))
                triples.append((i, j, k))
    return triples


def _hb_neighbors(z, D: np.ndarray, center: int, radius):
    """Up to four atoms bonded to ``center`` (dropping the longest of a fifth)."""
    rz = np.asarray([radius[v] for v in z])
    out = []
    for j in np.nonzero(D[center] < rz + radius[z[center]])[0]:
        if j == center:
            continue
        out.append(int(j))
        if len(out) == 5:
            far, best = 0.0, 0
            for k in range(5):
                dk = D[center, out[k]]
                if far < dk:
                    far, best = dk, k
            del out[best]
    return out


def _hb_frame(z, D: np.ndarray, center: int, h: int, bonded: List[int], radius):
    """The three reference atoms of one side of a hydrogen bond (MOPAC's DH+ choice)."""
    def farthest(cands, exclude=()):
        best, far = None, -1.0
        for k in cands:
            dk = D[k, h]
            if dk > far and k not in exclude:
                far, best = dk, k
        return best

    close = D[center, h] < radius[z[center]] + radius[z[h]]
    nb = len(bonded)
    if nb in (3, 4):
        a = farthest(bonded)
        b = farthest(bonded, (a,))
        c = farthest(bonded, (a, b))
        return a, b, c
    if nb == 2:
        a = farthest(bonded)
        b = [k for k in bonded if k != a][-1] if any(k != a for k in bonded) else None
        return a, b, (h if close else center)
    if nb == 1:
        a = bonded[0]
        rz = np.asarray([radius[v] for v in z])
        second = [int(k) for k in np.nonzero(D[a] < rz + radius[z[a]])[0] if k != a]
        b = farthest(second)
        return a, b, (h if close else center)
    if nb == 0:
        x = h if close else center
        return x, x, x
    return None


def _angles(pos: Tensor, i: Tensor, j: Tensor, k: Tensor) -> Tensor:
    """Angles ``i-j-k`` (radians) from the three distances, as :func:`_angle`."""
    d2ij = (pos[i] - pos[j]).pow(2).sum(-1)
    d2jk = (pos[j] - pos[k]).pow(2).sum(-1)
    d2ik = (pos[i] - pos[k]).pow(2).sum(-1)
    c = 0.5 * (d2ij + d2jk - d2ik) / torch.sqrt(d2ij * d2jk)
    return torch.acos(c.clamp(-1.0, 1.0))


def _dihedrals(pos: Tensor, i: Tensor, j: Tensor, k: Tensor, l: Tensor) -> Tensor:
    """Dihedrals ``i-j-k-l`` in ``[0, 2 pi)`` (IUPAC sense); 0 for degenerate geometries (a bond's
    projection shorter than 1e-6 Angstrom) and for angles within 4e-5 of zero."""
    b0, b1, b2 = pos[i] - pos[j], pos[k] - pos[j], pos[l] - pos[k]
    axis = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * axis).sum(-1, keepdim=True) * axis
    w = b2 - (b2 * axis).sum(-1, keepdim=True) * axis
    flat = (v.detach().norm(dim=-1) < 1e-6) | (w.detach().norm(dim=-1) < 1e-6)
    x = (v * w).sum(-1)
    y = (torch.cross(axis, v, dim=-1) * w).sum(-1)
    # degenerate frames take a safe angle that the mask then zeroes
    phi = torch.atan2(torch.where(flat, torch.zeros_like(y), y), torch.where(flat, torch.ones_like(x), x))
    zero = flat | (phi.detach().abs() < 4e-5)
    phi = torch.where(phi < 0, phi + 2.0 * math.pi, phi)
    return torch.where(zero, torch.zeros_like(phi), phi)


def _wraps(a: Tensor) -> Tensor:
    """Angles into ``(-pi, pi]``."""
    return torch.where(a <= -math.pi, a + 2.0 * math.pi, torch.where(a > math.pi, a - 2.0 * math.pi, a))


def _reflect(a: Tensor) -> Tensor:
    """``-pi - a`` for negative ``a``, else ``pi - a``."""
    return torch.where(a < 0, -math.pi - a, math.pi - a)


def pm7_hbond(z: Sequence[int], pos: Tensor) -> Tensor:
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
    the energies of all hydrogen bonds are evaluated together.
    """
    out = pos.new_zeros(len(z))
    xyz = pos.detach().tolist()
    D = _distances(pos)
    radius = [4.0 / 3.0 * r for r in _HB_RADII]
    triples = _hbond_triples(z, xyz, D, 1.4, 7.0)
    deg = math.pi / 180.0
    rows = []
    for (A, H, B) in triples:
        na = _hb_neighbors(z, D, A, radius)
        nb = _hb_neighbors(z, D, B, radius)
        if B in na or any(x == y and y != H for x in na for y in nb):
            continue
        fa = _hb_frame(z, D, A, H, na, radius)
        fb = _hb_frame(z, D, B, H, nb, radius)
        if fa is None or fb is None or None in fa or None in fb:
            continue
        row = [A, H, B]
        for center, (r1, r2, r3), nbonds in ((A, fa, len(na)), (B, fb, len(nb))):
            tcs = tcs2 = False
            shift = shift2 = tshift = 0.0
            if z[center] == 8:
                if nbonds == 1:
                    shift, shift2, tshift, tcs2 = math.pi, 120.0 * deg, 0.0, True
                else:
                    shift = shift2 = 109.48 * deg
                    tshift = 54.74 * deg
            elif z[center] == 7:
                if nbonds == 2:
                    shift = shift2 = 120.0 * deg
                else:
                    shift = shift2 = 109.48 * deg
                    tshift = 54.74 * deg
                    tcs = True
            zero_t = bool(D[H, center] > D[H, r1]) and tcs2
            one_t = r2 == r3 or H == r3
            row += [center, r1, r2, r3, tcs, tcs2, shift, shift2, tshift, zero_t, one_t]
        rows.append(row)
    if not rows:
        return out
    dev = pos.device

    def col(j, dtype=torch.long):
        return torch.tensor([r[j] for r in rows], dtype=dtype, device=dev)

    def flag(j):
        return torch.tensor([bool(r[j]) for r in rows], dtype=torch.bool, device=dev)

    def energies(sel: Tensor, keep_only: bool):
        """The energies of the hydrogen bonds ``sel``, or (``keep_only``) which of them take part."""
        A, H, B = col(0)[sel], col(1)[sel], col(2)[sel]
        cos_hb = -torch.cos(_angles(pos, A, H, B))
        keep = cos_hb.detach() > 0.0
        terms = []
        for o, last in ((3, False), (14, True)):
            center, r1, r2, r3 = (col(o + q)[sel] for q in range(4))
            tcs, tcs2 = flag(o + 4)[sel], flag(o + 5)[sel]
            shift, shift2, tshift = (col(o + q, pos.dtype)[sel] for q in (6, 7, 8))
            zero_t, one_t = flag(o + 9)[sel], flag(o + 10)[sel]
            tc = _reflect(_wraps(_dihedrals(pos, r2, r1, center, r3)))
            check = torch.where(tcs, tc.detach(), torch.zeros_like(tc))
            tcd = tc.abs() / deg
            tshift = torch.where(tcs, tshift + ((54.74 - tcd) / 54.74 * 35.26) * deg, tshift)
            shift = torch.where(tcs, shift - ((54.74 - tcd) / 54.74 * 19.48) * deg, shift)
            shift2 = torch.where(tcs, shift, shift2)
            ang2 = _angles(pos, r1, center, H)
            c2 = torch.maximum(torch.cos(shift - ang2), torch.cos(shift2 - ang2))
            keep = keep & (c2.detach() > 0.0)
            tcorr = _wraps(_dihedrals(pos, r2, r1, center, H))
            flip = ~tcs2 | (tcorr.detach().abs() / deg > 90.0)
            tcorr = torch.where(flip, _reflect(tcorr), tcorr)
            minus, plus = torch.cos(_wraps(tshift - tcorr)), torch.cos(_wraps(-tshift - tcorr))
            tcos = torch.where(check < 0, minus, torch.where(check > 0, plus, torch.maximum(minus, plus)))
            tcos = torch.where(zero_t, torch.zeros_like(tcos), tcos)
            tcos = torch.where(one_t, torch.ones_like(tcos), tcos)
            if last:
                tcos = tcos.abs()
            else:
                keep = keep & (tcos.detach() >= 0.0)
            terms.append((c2, tcos))
        if keep_only:
            return keep
        hartree_kcal = 27.211386245988 * 23.060547830619029
        a0sq = 0.529177210903 ** 2
        zt = torch.as_tensor(z, device=dev)
        scale = torch.where(zt == 7, torch.full((), -0.171271 * a0sq, dtype=pos.dtype, device=dev),
                            torch.full((), -0.098822 * a0sq, dtype=pos.dtype, device=dev))
        s_c = 0.5 * (scale[A] + scale[B])
        ha, hb = (pos[H] - pos[A]).norm(dim=-1), (pos[H] - pos[B]).norm(dim=-1)
        xc = torch.minimum(ha, hb)
        damping = torch.where((torch.maximum(ha, hb) - xc).detach() > 0.5,
                              1.0 - 1.0 / (1.0 + torch.exp(-60.0 * (xc / 1.2 - 1.0))), torch.ones_like(xc))
        rab = (pos[A] - pos[B]).norm(dim=-1)
        damping = damping / (1.0 + torch.exp(-100.0 * (rab / 2.4 - 1.0)))
        damping = damping * (1.0 - 1.0 / (1.0 + torch.exp(-10.0 * (rab / 7.0 - 1.0))))
        prod = terms[0][0] * terms[0][1] * terms[1][0] * terms[1][1]
        e = s_c / rab ** 2 * cos_hb ** 2 * (1.0 - (1.0 - prod) ** 2) * hartree_kcal * damping
        oo = (zt[A] == 8) & (zt[B] == 8)
        e = e - torch.where(oo, 2.5 * torch.exp(-80.0 * torch.clamp(rab - 2.67, min=0.0) ** 2) * cos_hb ** 4,
                            torch.zeros_like(e))
        return out.index_add(0, H, e)

    everything = torch.arange(len(rows), device=dev)
    with torch.no_grad():
        kept = everything[energies(everything, True)]
    if kept.numel() == 0:
        return out
    # the energies of the hydrogen bonds that take part, so that no dropped one enters the derivatives
    return energies(kept, False)


def _smoothstep7(x: Tensor) -> Tensor:
    return -20.0 * x ** 7 + 70.0 * x ** 6 - 84.0 * x ** 5 + 35.0 * x ** 4


def _valences(z: Sequence[int], pos: Tensor, D: np.ndarray):
    """Smooth bond counts per atom ``(N,)``, all neighbors, hydrogens and the others: a neighbor counts 1 within
    the sum of the covalent radii ``r0``, 0 beyond ``1.6 r0``, ``1 - S7((r - r0) / (0.6 r0))`` between; and the
    matrix of the pairs counting more than zero (on the detached distances ``D``)."""
    n = len(z)
    cov = np.array([el.COVALENT_RADIUS[v] or 0.0 for v in z])
    r0 = cov[:, None] + cov[None, :]
    r1 = 1.6 * r0
    some = (D > 0.0) & (D < r1)
    np.fill_diagonal(some, False)
    a, b = np.nonzero(some)
    dev = pos.device
    zero = pos.new_zeros(n)
    if a.size == 0:
        return zero, zero, zero, some
    ia, ib = torch.as_tensor(a, device=dev), torch.as_tensor(b, device=dev)
    r = (pos[ia] - pos[ib]).norm(dim=-1)
    lo = torch.as_tensor(r0[a, b], dtype=pos.dtype, device=dev)
    full = torch.as_tensor(D[a, b] <= r0[a, b], device=dev)
    v = torch.where(full, torch.ones_like(r), 1.0 - _smoothstep7((r - lo) / (0.6 * lo)))
    is_h = torch.as_tensor(np.asarray(z)[b] == 1, device=dev)
    total = zero.index_add(0, ia, v)
    hyd = zero.index_add(0, ia, torch.where(is_h, v, torch.zeros_like(v)))
    return total, hyd, total - hyd, some


def h4_hbond(z: Sequence[int], pos: Tensor) -> Tensor:
    """Hydrogen-bond correction H4 (Rezac & Hobza, J. Chem. Theory Comput. 8,
    141, 2012) as used by PM6-ORG, per hydrogen atom (kcal/mol).

    For every X-H...Y (X, Y = N, O; X-H < 1.15, X...Y < 5.5 Angstrom) the
    energy ``c_XY E_rad(R_XY) E_ang(theta) f_switch f_water f_NR4+ f_COO-``
    with a seventh-order radial polynomial, the angular switch of the
    deviation from linearity, and the scalings of water, ammonium and
    carboxylate groups. The groups are chosen on the detached geometry; the
    energies of all hydrogen bonds are evaluated together.
    """
    out = pos.new_zeros(len(z))
    xyz = pos.detach().tolist()
    D = _distances(pos)
    triples = _hbond_triples(z, xyz, D, 1.15, 5.5, heavy=(7, 8, 16))
    if not triples:
        return out
    para = {(8, 8): 2.32, (8, 7): 3.10, (7, 8): 1.07, (7, 7): 2.01}
    cv, hyd, oth, some = _valences(z, pos, D)
    rows = []
    for (i0, h, k0) in triples:
        a_, b_ = k0, i0           # the order of the reference implementation's arguments
        d, acc = (a_, b_) if D[a_, h] < D[b_, h] else (b_, a_)
        cc = o2 = -1
        if z[acc] == 8:
            # the carbon nearest the acceptor oxygen among its neighbors, and that carbon's other nearest oxygen
            cands = [m for m in np.nonzero(some[acc])[0] if z[m] == 6]
            if cands:
                cc = int(min(cands, key=lambda m: (D[acc, m], m)))
                oxy = [m for m in np.nonzero(some[cc])[0] if m != acc and z[m] == 8]
                if oxy:
                    o2 = int(min(oxy, key=lambda m: (D[cc, m], m)))
        rows.append((a_, b_, h, d, acc, cc, o2, para.get((z[d], z[acc]), 0.0)))
    dev = pos.device

    def col(j, dtype=torch.long):
        return torch.tensor([r[j] for r in rows], dtype=dtype, device=dev)
    A, B, H, Dn, Ac, CC, O2 = (col(j) for j in range(7))
    c = col(7, pos.dtype)
    zt = torch.as_tensor(z, device=dev)
    rda = (pos[A] - pos[B]).norm(dim=-1)
    rdh, rah = (pos[Dn] - pos[H]).norm(dim=-1), (pos[Ac] - pos[H]).norm(dim=-1)
    ang = math.pi - _angles(pos, A, H, B)
    e_rad = (-0.00303407407407313510 * rda ** 7 + 0.07357629629627092382 * rda ** 6
             - 0.70087111111082800452 * rda ** 5 + 3.25309629629461749545 * rda ** 4
             - 7.20687407406838786983 * rda ** 3 + 5.31754666665572184314 * rda ** 2
             + 3.40736000001102778967 * rda - 4.68512000000450434811)
    e_ang = 1.0 - _smoothstep7(ang / (math.pi / 2.0)) ** 2
    far = rdh.detach() > 1.15
    x = torch.where(far, (rdh - 1.15) / (0.5 * rdh + 0.5 * rah - 1.15), torch.zeros_like(rdh))
    e_sw = torch.where(far, 1.0 - _smoothstep7(x), torch.ones_like(rdh))
    # water donors and acceptors
    hd, od = hyd[Dn], oth[Dn]
    hv = hd.detach()
    fv = torch.where((hv > 1.0) & (hv <= 2.0), hd - 1.0, torch.where((hv > 2.0) & (hv < 3.0), 3.0 - hd,
                                                                      torch.zeros_like(hd)))
    water = (zt[Dn] == 8) & (zt[Ac] == 8) & (hv >= 1.0)
    e_w = torch.where(water, 1.0 + (0.42 - 1.0) * fv * torch.clamp(1.0 - od, min=0.0), torch.ones_like(hd))
    # ammonium donors
    vd = cv[Dn]
    e_chd = torch.where((zt[Dn] == 7) & (vd.detach() > 3.0), 1.0 + (3.61 - 1.0) * torch.clamp(vd - 3.0, min=0.0),
                        torch.ones_like(vd))
    # carboxylate acceptors
    has = (zt[Ac] == 8) & (CC >= 0) & (O2 >= 0)
    cc_safe, o2_safe = torch.where(has, CC, Ac), torch.where(has, O2, Ac)
    f1 = torch.clamp(1.0 - (1.0 - cv[Ac]).abs(), min=0.0)
    f2 = torch.clamp(1.0 - (1.0 - cv[o2_safe]).abs(), min=0.0)
    f3 = torch.clamp(1.0 - (3.0 - cv[cc_safe]).abs(), min=0.0)
    e_cha = torch.where(has, 1.0 + (1.41 - 1.0) * f1 * f2 * f3, torch.ones_like(f1))
    e = c * e_rad * e_ang * e_sw * e_w * e_chd * e_cha
    return out.index_add(0, H, e)


def hh_repulsion(z: Sequence[int], pos: Tensor) -> Tensor:
    """Hydrogen-hydrogen repulsion of D3H4 (Rezac & Hobza 2012), per atom (kcal/mol)."""
    out = pos.new_zeros(len(z))
    hs = [i for i, v in enumerate(z) if v == 1]
    if len(hs) < 2:
        return out
    idx = torch.as_tensor(hs, device=pos.device)
    a, b = torch.triu_indices(len(hs), len(hs), offset=1, device=pos.device)
    r = (pos[idx[a]] - pos[idx[b]]).norm(dim=-1)
    rc = r.clamp(min=1e-6)
    mid = (-2714.952351603469651 * rc ** 5 + 17103.650110591705015 * rc ** 4
           - 42511.857982217959943 * rc ** 3 + 52063.196799138342612 * rc ** 2
           - 31430.658335972289933 * rc + 7516.084696095140316)
    far = 118.7326 * torch.exp(-1.53965 * rc ** 1.72905)
    e = torch.where(r <= 1.0, torch.full_like(r, 25.46293603147693), torch.where(r < 1.5, mid, far))
    return out.index_add(0, idx[a], 0.5 * e).index_add(0, idx[b], 0.5 * e)
