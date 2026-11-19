"""Topology and parameter rules of the GFN-FF force field.

GFN-FF (Spicher & Grimme, *Angew. Chem. Int. Ed.* 59, 15665, 2020) builds a
complete force field from the elements, the positions and the total charge of
a molecule. This module holds the geometry-independent part: the neighbour
lists, the topology charges, hybridization and pi flags, the pi systems with
their Hueckel bond orders (:class:`Topology`), and the parameter rows of
every energy term (``bond_rows``, ``angle_params``, ``torsion_rows``, ...).
:class:`~xnn.ffnn.models.gfnff.GFNFF` evaluates the energy from these rows.

The element parameters are the ones distributed with the xtb program; they
are downloaded on first use into the model cache (:func:`parameter_file`)
and are not part of xnn. The remaining tables ship in
``xnn/ffnn/data/gfnff_tables.json``.
"""
import functools
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Union

import networkx as nx
import numpy as np
import torch

BOHR = 0.52917721090449
RELEASE = "v6.8.0"
_URL = "https://raw.githubusercontent.com/grimme-lab/xtb/{release}/{file}"
_FILE = ".param_gfnff.xtb"
_MD5 = "674a04965dc9f9b0862293c031d44c95"
_NOTE = """GFN-FF element parameters of the xtb program, release {release}.

Source: https://github.com/grimme-lab/xtb (file {file})
License: LGPL-3.0-or-later; see COPYING and COPYING.LESSER in that repository.

The file was downloaded by xnn on first use of GFN-FF. It is not part of xnn
and is not covered by the license of xnn.
"""
HALOGEN, NOBLE = {9, 17, 35, 53, 85}, {2, 10, 18, 36, 54, 86}
#: Elements the rules cover.
SUPPORTED = frozenset(range(1, 87))
_PAR = {}


class _Lazy:
    """A table loaded on first use."""

    def __init__(self, load):
        self._load, self._value = load, None

    def _get(self):
        if self._value is None:
            self._value = self._load()
        return self._value

    def __getitem__(self, key):
        return self._get()[key]

    def get(self, key, default=None):
        return self._get().get(key, default)


def parameter_file(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """The GFN-FF parameter file of xtb, downloaded into the model cache if needed.

    A ``LICENSE.txt`` beside the cached file records its source and license,
    and the download itself is announced by a warning.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Model cache; defaults to the xnn model cache.

    Returns
    -------
    Path
        The cached file.

    Raises
    ------
    FileNotFoundError
        When the file is not cached and ``XNN_OFFLINE`` forbids downloading.
    """
    from ...common.data.hub._download import download_file, md5sum
    from ...common.models.hub.cache import offline, resolve_cache_dir

    dest = resolve_cache_dir(cache_dir) / "ffnn" / f"xtb-{RELEASE.lstrip('v')}" / _FILE
    if dest.exists() and md5sum(dest) == _MD5:
        return dest
    if offline():
        raise FileNotFoundError(f"{dest} is not cached and XNN_OFFLINE is set; use GFN-FF once with network access")
    warnings.warn(f"downloading the GFN-FF parameters of xtb {RELEASE} (LGPL-3.0-or-later) to {dest.parent}; "
                  "they are not part of xnn, see the LICENSE.txt written beside them", stacklevel=2)
    download_file(_URL.format(release=RELEASE, file=_FILE), dest, _MD5, quiet=True)
    (dest.parent / "LICENSE.txt").write_text(_NOTE.format(release=RELEASE, file=_FILE))
    return dest


def parameters(cache_dir: Optional[Union[str, Path]] = None) -> np.ndarray:
    """Element parameters ``(103, 11)`` of GFN-FF, read once from :func:`parameter_file`."""
    if "par" not in _PAR:
        _PAR["par"] = np.loadtxt(parameter_file(cache_dir), max_rows=103)[:, 1:]
    return _PAR["par"]


@functools.lru_cache(maxsize=None)
def _tables():
    raw = json.loads((Path(__file__).resolve().parents[1] / "data" / "gfnff_tables.json").read_text())
    pairs = lambda d: {tuple(int(v) for v in k.split("-")): f for k, f in d.items()}  # noqa: E731
    radii = {int(k): tuple(v) for k, v in raw["bond_radius"].items()}
    return pairs(raw["neighbour_radius"]), radii, pairs(raw["bond_pair_factor"])


@functools.lru_cache(maxsize=None)
def _reference():
    from ...common.models.d4 import _load_reference

    ref = _load_reference()
    return (np.asarray(ref["covalent_radius_aa"], dtype=float) / BOHR, np.asarray(ref["pauling_en"], dtype=float),
            np.asarray(ref["refcn_d3"], dtype=float))


PAR = _Lazy(parameters)
NB_R0 = _Lazy(lambda: _tables()[0])
BOND_RAD = _Lazy(lambda: _tables()[1])
F_PAIR = _Lazy(lambda: _tables()[2])
RCOV = _Lazy(lambda: _reference()[0])
PAULING = _Lazy(lambda: _reference()[1])
REFCN = _Lazy(lambda: _reference()[2])
def _erf(a):
    return torch.erf(torch.as_tensor(a, dtype=torch.float64)).numpy()


def coulomb(dist, alpha, hard):
    """Coulomb matrix of Gaussian charges of widths alpha with the hardness on the diagonal."""
    al, d = torch.as_tensor(alpha, dtype=torch.float64), torch.as_tensor(dist, dtype=torch.float64)
    d = d + torch.eye(len(al), dtype=torch.float64)
    a = torch.erf(torch.rsqrt(al[:, None] ** 2 + al[None, :] ** 2) * d) / d
    a.diagonal().copy_(torch.as_tensor(hard, dtype=torch.float64) + math.sqrt(2.0 / math.pi) / al)
    return a.numpy()


def _solve(mat, rhs):
    """Dense linear solve (torch's LAPACK, threaded)."""
    return torch.linalg.solve(torch.as_tensor(mat), torch.as_tensor(rhs)).numpy()


def fragment_charges(frags, charge):
    """Total charge of each covalent fragment: the whole charge sits on the fragment that holds the first atom."""
    return [float(charge) if 0 in f else 0.0 for f in frags]


def solve_fragments(a, x, frags, charge=0):
    """Charges with one total-charge constraint per covalent fragment (see :func:`fragment_charges`)."""
    n, m = len(x), len(frags)
    mat = np.zeros((n + m, n + m))
    mat[:n, :n] = a
    for k, f in enumerate(frags):
        mat[f, n + k] = mat[n + k, f] = 1.0
    return _solve(mat, np.append(x, fragment_charges(frags, charge)))[:n]


METALS = {3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56, 81, 82, 83, 84} | set(range(21, 31)) | \
    set(range(39, 49)) | set(range(57, 81))
S_NONMETAL = {1: 1.02, 5: 1.02, 8: 1.02, 9: 1.05, 10: 1.10, 18: 1.10, 15: 0.97, 34: 0.99, 51: 0.99, 52: 0.95, 53: 0.98}


S_METAL = {38: 1.02, 56: 1.02, 76: 1.02, 3: 1.025, 13: 1.025, 31: 1.025, 37: 1.025, 49: 1.025, 55: 1.025, 81: 1.025,
           84: 1.025, 11: 1.0353, 50: 1.0353, 12: 1.0455, 19: 1.0455, 20: 1.0455, 4: 1.0558, 82: 1.0865, 83: 0.97375}
TOPO_RAD = {
    1: 0.32, 2: 0.37, 3: 1.30, 4: 0.99, 5: 0.84, 6: 0.75, 7: 0.71, 8: 0.64, 9: 0.60, 10: 0.62, 11: 1.60, 12: 1.40, 13:
    1.24, 14: 1.14, 15: 1.09, 16: 1.04, 17: 1.00, 18: 1.01, 19: 2.00, 20: 1.74, 21: 1.59, 22: 1.48, 23: 1.44, 24:
    1.30, 25: 1.29, 26: 1.24, 27: 1.18, 28: 1.17, 29: 1.22, 30: 1.20, 31: 1.23, 32: 1.20, 33: 1.20, 34: 1.18, 35:
    1.17, 36: 1.16, 37: 2.15, 38: 1.90, 39: 1.76, 40: 1.64, 41: 1.56, 42: 1.46, 43: 1.38, 44: 1.36, 45: 1.34, 46:
    1.30, 47: 1.36, 48: 1.40, 49: 1.42, 50: 1.40, 51: 1.40, 52: 1.37, 53: 1.36, 54: 1.36, 55: 2.38, 56: 2.06, 57:
    1.94, 58: 1.84, 59: 1.90, 60: 1.88, 61: 1.86, 62: 1.85, 63: 1.83, 64: 1.82, 65: 1.81, 66: 1.80, 67: 1.79, 68:
    1.77, 69: 1.77, 70: 1.78, 71: 1.74, 72: 1.64, 73: 1.58, 74: 1.50, 75: 1.41, 76: 1.36, 77: 1.32, 78: 1.30, 79:
    1.30, 80: 1.32, 81: 1.44, 82: 1.45, 83: 1.50, 84: 1.42, 85: 1.48, 86: 1.46,
}


HALOGEN = {9, 17, 35, 53, 85}
D_METAL = {38, 56} | set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81))


TOPO_BOHR = 0.52917726


def delta(z):
    s = S_METAL.get(z, 1.0) * math.sqrt(2.0) if z in METALS else S_NONMETAL.get(z, 1.0)
    return 0.152138 * s * s


def en_shifts(z, nb, hyb, topo=False):
    """Environment shifts of the electronegativity; the amide H shift enters the
    final charge model only, not the topology charges."""
    out = []
    for i, zi in enumerate(z):
        d, nh = len(nb[i]), sum(z[j] == 1 for j in nb[i])
        s = 0.0
        if zi == 1 and d == 1:
            j = nb[i][0]
            if not topo and z[j] == 7 and hyb[j] == -3 and sum(z[k] == 1 for k in nb[j]) == 1 \
                    and not any(hyb[k] == 5 for k in nb[j]) and all(z[k] in (1, 6) for k in nb[j]):  # not hydrazide
                s = 0.02
        elif zi == 5:
            s = -0.015 * nh
        elif zi == 6 and hyb[i] == -2:
            s = 0.15
        elif zi == 8 and d == 1:
            j = nb[i][0]
            s = 0.005 if z[j] == 1 else (-0.15 if z[j] == 6 and len(nb[j]) == 1 else
                                         (-0.05 if z[j] == 7 and len(nb[j]) <= 3 else 0.0))
        elif zi == 8 and d == 2:
            s = 0.03 if nh == 2 else 0.005 * nh
        elif zi == 8 and d >= 3:
            s = 0.005 * (nh - d)
        elif zi == 16:
            s = 0.005 * nh - (0.005 * d if d >= 3 else 0.0)
        elif zi in (34, 52, 84) and d >= 3:
            s = -0.005 * d
        elif zi in HALOGEN and d >= 2:
            s = 0.0 if zi == 9 else (-0.05 * d if any(mclass(z[j]) for j in nb[i]) else 0.021 * d)
        elif zi in D_METAL and topo:
            s = -0.09
        out.append(s)
    return np.array(out)


def components(n, edges):
    g = nx.Graph(edges)
    g.add_nodes_from(range(n))
    return g, [sorted(c) for c in nx.connected_components(g)]


def topology_charges(z, edges, charge, frags=None, x=None, nbl=None):
    """Topology charges on a graph: shortest-path distances over bonds of 1.175 (r_A + r_B) and the square root of
    the neighbour count, capped at 4.4."""
    n = len(z)
    g, comps = components(n, edges)
    frags = comps if frags is None else frags
    nbl = [sorted(g[i]) for i in range(n)] if nbl is None else nbl
    nn = np.minimum(np.array([len(v) for v in ligand_view(z, nbl, eta_only=True)[0]], dtype=float), 4.4)
    # bond lengths r_A + r_B (Angstrom) rounded to single precision, shortest paths summed in single precision
    rad = np.array([TOPO_RAD[int(v)] for v in z])
    far = np.float32(1e10)
    path = np.full((n, n), far, dtype=np.float32)
    np.fill_diagonal(path, 0.0)
    for i, j in g.edges:
        path[i, j] = path[j, i] = np.float32(rad[i] + rad[j])
    for frag in comps:
        if len(frag) < 3:
            continue
        sub = path[np.ix_(frag, frag)]
        for k in range(len(frag)):
            np.minimum(sub, sub[:, k:k + 1] + sub[k:k + 1, :], out=sub)
        path[np.ix_(frag, frag)] = sub
    path = path.astype(np.float64)
    tdist = np.where(path > 12.0, 13.0, path) * 1.175 / TOPO_BOHR
    en, jj, alp = (PAR[np.asarray(z) - 1, c] for c in (0, 1, 3))
    kappa = PAR[np.asarray(z) - 1, 2]
    hyb, _ = hybridization(z, nbl, x)
    x = -(en + en_shifts(z, nbl, hyb, topo=True)) + kappa * np.sqrt(nn)
    a = coulomb(tdist, alp, jj)
    if len(frags) == 1:
        mat = np.zeros((n + 1, n + 1))
        mat[:n, :n] = a
        mat[:n, n] = mat[n, :n] = 1.0
        return _solve(mat, np.append(x, float(charge)))[:n]
    return solve_fragments(a, x, frags, charge)


#: an O between a metal and a non-metal bonded to that metal is sp3 when the pair is closer than its pass-2 threshold
#: divided by this factor (see hybridization)
HYB_METAL_SCALE = 1.025


@functools.lru_cache(maxsize=None)
def diatomic_metal_charge(zm, zx):
    """Topology charge of the metal in the neutral bonded pair of a metal and a non-metal."""
    return float(topology_charges([zm, zx], [(0, 1)], 0, x=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))[0])


def neighbours(z, pos, charge=0):
    """Two-pass neighbour rule: pass 1 R < R0_AB, pass 2 R < R0_AB - Delta_AB (q1_A + q1_B). For a metal M and a
    non-metal X the charges enter with the weights sqrt(2) Delta and Delta / sqrt(2) and the reference is
    R0_AB + Delta / sqrt(2) max(q0_M, 0), with q0_M the metal charge of the neutral pair. Also returns the
    metal-nonmetal bonds shorter than the pass-2 threshold divided by ``HYB_METAL_SCALE``."""
    z = [int(v) for v in z]
    x = np.asarray(pos, dtype=float)
    n = len(z)
    xt = torch.as_tensor(x, dtype=torch.float64)
    d = torch.sqrt(sum((xt[:, c, None] - xt[None, :, c]) ** 2 for c in range(3))).numpy()
    kinds = sorted(set(z))
    kind = np.array([kinds.index(a) for a in z])
    r0k = np.array([[NB_R0.get((min(a, b), max(a, b)), 0.0) for b in kinds] for a in kinds])
    dkk = np.array([delta(a) for a in kinds])
    mk = np.array([a in METALS for a in kinds])
    q0k = np.array([[diatomic_metal_charge(a, b) if mk[ia] and not mk[ib] else
                     (diatomic_metal_charge(b, a) if mk[ib] and not mk[ia] else 0.0)
                     for ib, b in enumerate(kinds)] for ia, a in enumerate(kinds)])
    # pairs within the largest pass-1 radius; the thresholds are evaluated on pairs only
    ii, jj = np.nonzero(np.triu(d < r0k.max(initial=0.0), 1))
    ki, kj = kind[ii], kind[jj]
    dp, r0 = d[ii, jj], r0k[ki, kj]
    e1 = [(int(i), int(j)) for i, j, keep in zip(ii, jj, dp < r0) if keep]
    q1 = topology_charges(z, e1, charge, x=x)
    # pass 2 can reach further: pairs within the largest threshold the charges allow
    root2 = math.sqrt(2.0)
    qmax = float(np.abs(q1).max(initial=0.0))
    dmax = float(dkk.max(initial=0.0))
    reach = r0k.max(initial=0.0) + dmax * (max(float(q0k.max(initial=0.0)), 0.0) / root2 + (root2 + 2.0) * qmax)
    ii, jj = np.nonzero(np.triu(d < reach, 1))
    ki, kj = kind[ii], kind[jj]
    dp, r0 = d[ii, jj], r0k[ki, kj]
    dl = np.sqrt(dkk[ki] * dkk[kj])
    mi, mj = mk[ki], mk[kj]
    mx = mi != mj  # metal and non-metal
    qa, qb = q1[ii] + 0.0 * q1[jj], q1[jj] + 0.0 * q1[ii]
    qm, qx = np.where(mi, qa, qb), np.where(mi, qb, qa)
    thr = np.where(mx, r0 + dl / root2 * (np.maximum(q0k[ki, kj], 0.0) - qx) - root2 * dl * qm, r0 - dl * (qa + qb))
    e2 = [(int(i), int(j)) for i, j, keep in zip(ii, jj, dp < thr) if keep]
    _, f12 = components(n, sorted(set(e1) & set(e2)))
    qt = topology_charges(z, e2, charge, frags=f12 if len(f12) > 1 else None, x=x)
    close = {(int(i), int(j)) for i, j, keep in zip(ii, jj, mx & (HYB_METAL_SCALE * dp < thr)) if keep}
    return e2, qt, f12, close


def fractional(pos, cell):
    """Fractional coordinates as the wrapping rule sees them: positions in Bohr times the cofactor inverse of the
    lattice (rows are the lattice vectors, the cofactors times the reciprocal determinant), summed component by
    component; atoms on a cell face fall on the side this arithmetic gives."""
    x = np.asarray(pos, dtype=float) / BOHR
    a = np.asarray(cell, dtype=float) / BOHR
    c = np.array([[a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1], a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2],
                   a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]],
                  [a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2], a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0],
                   a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]],
                  [a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0], a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1],
                   a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]]])
    inv = c * (1.0 / (a[0, 0] * c[0, 0] + a[0, 1] * c[1, 0] + a[0, 2] * c[2, 0]))
    return (x[:, 0, None] * inv[0] + x[:, 1, None] * inv[1]) + x[:, 2, None] * inv[2]


class Img(int):
    """A folded atom index of a periodic structure with the lattice shift of its image: it compares, hashes and
    indexes as the atom of the unit cell, and positions taken through :class:`Positions` include the shift."""

    def __new__(cls, i, s=(0, 0, 0)):
        obj = super().__new__(cls, int(i))
        obj.s = (int(s[0]), int(s[1]), int(s[2]))
        return obj

    def moved(self, s):
        """The same atom in the image shifted by the lattice vector ``s``."""
        a = self.s
        return Img(self, (a[0] + s[0], a[1] + s[1], a[2] + s[2]))


class Positions:
    """Positions (Angstrom) of atoms and their images: ``x[a]`` adds the lattice shift of an :class:`Img`."""

    def __init__(self, x, cell):
        self.x, self.cell = np.asarray(x, dtype=float), np.asarray(cell, dtype=float)
        self._shift = {}

    def __getitem__(self, a):
        s = getattr(a, "s", None)
        if s is None or not any(s):
            return self.x[int(a)]
        # the lattice vector of a shift, computed once
        v = self._shift.get(s)
        if v is None:
            v = self._shift[s] = np.asarray(s, dtype=float) @ self.cell
        return self.x[int(a)] + v

    def __len__(self):
        return len(self.x)

    def __array__(self, dtype=None, copy=None):
        return self.x if dtype is None else self.x.astype(dtype)


class Neighbours(list):
    """Neighbour lists of a periodic structure: the list of an :class:`Img` holds its neighbours moved to its image."""

    def __getitem__(self, a):
        base = list.__getitem__(self, a)
        s = getattr(a, "s", None)
        if isinstance(a, slice) or s is None or not any(s):
            return base
        # the moved lists are kept (a copy is handed out)
        memo = self.__dict__.setdefault("_moved", {})
        key = (int(a), s)
        if key not in memo:
            memo[key] = [j.moved(s) for j in base]
        return list(memo[key])


#: lattice shifts of the image pairs searched by the periodic neighbour rule
IMAGES = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]


def periodic_neighbours(z, pos, cell, charge=0):
    """The two-pass neighbour rule of :func:`neighbours` on the image pairs of a periodic structure. Returns the bonds
    (i, Img j) for every central atom i, the topology charges, the fragments of the charge constraint and the
    metal-nonmetal pairs of the hybridization rule; charges and paths are those of the folded graph."""
    z = [int(v) for v in z]
    x, cell = np.asarray(pos, dtype=float), np.asarray(cell, dtype=float)
    n = len(z)
    # the image pairs within 8 Angstrom: atoms, image and distance, image-major
    pi, pj, pk, pd = [], [], [], []
    for k, sh in enumerate(IMAGES):
        t = np.asarray(sh, dtype=float) @ cell
        d = np.linalg.norm(x[None, :, :] + t - x[:, None, :], axis=-1)
        i, j = np.nonzero(d < 8.0)
        keep = (i != j) | any(sh)
        pi.append(i[keep])
        pj.append(j[keep])
        pk.append(np.full(int(keep.sum()), k))
        pd.append(d[i, j][keep])
    pi, pj, pk, pd = (np.concatenate(v) for v in (pi, pj, pk, pd))
    za = np.asarray(z, dtype=np.int64)
    code = np.minimum(za[pi], za[pj]) * 256 + np.maximum(za[pi], za[pj])
    uc, inv = np.unique(code, return_inverse=True)
    r0 = np.array([NB_R0.get((int(c) // 256, int(c) % 256), 0.0) for c in uc], dtype=float)[inv]

    def edges(mask):
        return [(int(i), Img(j, IMAGES[k])) for i, j, k in zip(pi[mask].tolist(), pj[mask].tolist(),
                                                               pk[mask].tolist())]

    e1 = edges(pd < r0)
    q1 = topology_charges(z, folded_edges(e1), charge, x=Positions(x, cell), nbl=_lists(n, e1))
    q1a = np.array([float(q1[i]) for i in range(n)])
    root2 = math.sqrt(2.0)
    dz = np.array([delta(v) for v in z], dtype=float)
    dl = np.sqrt(dz[pi] * dz[pj])
    met = np.array([v in METALS for v in z], dtype=bool)
    ma, mb = met[pi], met[pj]
    mixed = ma != mb
    qm, qx = np.where(ma, q1a[pi], q1a[pj]), np.where(ma, q1a[pj], q1a[pi])
    zm, zx = np.where(ma, za[pi], za[pj]), np.where(ma, za[pj], za[pi])
    mc = zm * 256 + zx
    uq, iq = np.unique(mc[mixed], return_inverse=True)
    q0 = np.zeros(len(pd))
    q0[mixed] = np.array([diatomic_metal_charge(int(c) // 256, int(c) % 256) for c in uq], dtype=float)[iq]
    thr_m = r0 + dl / root2 * (np.maximum(q0, 0.0) - qx) - root2 * dl * qm
    cm = mixed & (HYB_METAL_SCALE * pd < thr_m)
    close = {(min(i, j), max(i, j)) for i, j in zip(pi[cm].tolist(), pj[cm].tolist())}
    thr = np.where(mixed, thr_m, r0 - dl * (q1a[pi] + q1a[pj]))
    e2 = edges(pd < thr)
    f2 = folded_edges(e2)
    _, f12 = components(n, sorted(set(folded_edges(e1)) & set(f2)))
    qt = topology_charges(z, f2, charge, frags=f12 if len(f12) > 1 else None, x=Positions(x, cell),
                          nbl=_lists(n, e2))
    return e2, qt, f12, close


#: rings kept per start atom: the search of each start keeps its first distinct rings up to this many
RING_CAP = 19
#: image order of the periodic neighbour lists: the cell, then the 26 neighbouring cells with the third index outermost
IMAGE_ORDER = {p: q for q, p in enumerate([(0, 0, 0)] + [(a, b, c) for c in (-1, 0, 1) for b in (-1, 0, 1)
                                                          for a in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])}


def _start_rings(nb, s0, cap, hops=None, swap=True):
    """Rings of the depth-first search from start ``s0`` in neighbour-list order, as atom tuples in walk order, the
    first ``cap`` distinct ones. A start whose first neighbour is the first atom of the structure visits its second
    neighbour before it (``swap``). With the bond counts ``hops`` to the start, walks that can no longer close are not
    followed (they find no ring)."""
    seen, rings = set(), []
    first = list(nb[s0])
    if swap and len(first) > 1 and first[0] == 0:
        first[0], first[1] = first[1], first[0]

    path = [s0]

    def walk(nbrs):
        for w in nbrs:
            if len(rings) >= cap:
                return
            if w == s0:
                if len(path) >= 3:
                    ring = frozenset(path)
                    if ring not in seen:
                        seen.add(ring)
                        rings.append(tuple(path))
            elif w not in path and len(path) < 6 and (hops is None or hops[w] <= 6 - len(path)):
                path.append(w)
                walk(nb[w])
                path.pop()
    walk(first)
    return rings


def _hops_to(nb, s0, limit=6):
    """Bond counts from ``s0`` over the folded graph, up to ``limit`` (beyond it: limit + 1)."""
    hops = [limit + 1] * len(nb)
    hops[s0], front = 0, [s0]
    for k in range(1, limit + 1):
        nxt = []
        for a in front:
            for w in nb[a]:
                if hops[w] > k:
                    hops[w] = k
                    nxt.append(w)
        front = nxt
    return hops


def _ring_lists(n, nb, cap=None):
    """Ring lists as xtb builds them: per start atom the first ``cap`` distinct rings of three to six atoms of a
    depth-first search over the neighbour lists ``nb`` (:func:`_start_rings`)."""
    cap = RING_CAP if cap is None else cap
    return [_start_rings(nb, s0, cap, _hops_to(nb, s0)) if nb[s0] else [] for s0 in range(n)]


def _listed_sizes(nb, lists):
    """Ring sizes as the list of each start shows them to the lookup of another atom: the sizes of its rings, except
    that a start whose first neighbour is the first atom of the structure shows for its first ring the size of the
    first ring through that neighbour."""
    sizes = [[len(c) for c in rings] for rings in lists]
    for s0, rings in enumerate(lists):
        if rings and len(nb[s0]) > 1 and nb[s0][0] == 0:
            direct = _start_rings(nb, s0, 1, _hops_to(nb, s0), swap=False)
            if direct and direct[0][1] == 0:
                sizes[s0][0] = len(direct[0])
    return sizes


def folded_edges(bonds):
    """Edges of the folded simple graph of image bonds (i, Img j)."""
    return sorted({(min(i, int(j)), max(i, int(j))) for i, j in bonds if i != int(j)})


def _lists(n, bonds):
    per = [[] for _ in range(n)]
    for i, j in bonds:
        per[i].append(j)
    return Neighbours(sorted(v) for v in per)


class Topology:
    def __init__(self, z, pos, charge=0, cell=None):
        self.z = [int(v) for v in z]
        self.charge = charge
        self.q0 = None
        n = len(self.z)
        self.cell = None if cell is None else np.asarray(cell, dtype=float)
        if cell is None:
            self.x = np.asarray(pos, dtype=float)
            self.edges, self.qt, self.frags, close = neighbours(self.z, self.x, charge)
            self.g = nx.Graph(self.edges)
            self.g.add_nodes_from(range(n))
            self.nb = [sorted(self.g[i]) for i in range(n)]
        else:
            # periodic: the atoms are wrapped into the cell (fractional coordinates in [0, 1)), neighbours are images
            # (Img) of the wrapped atoms, the graph rules see the folded graph of the unit cell
            frac = fractional(pos, self.cell)
            #: lattice shifts that wrap the given positions into the cell
            self.wrap = np.floor(frac).astype(int)
            pos = np.asarray(pos, dtype=float) - self.wrap @ self.cell
            self.x = Positions(pos, cell)
            bonds, self.qt, self.frags, close = periodic_neighbours(self.z, pos, cell, charge)
            self.bonds = [(Img(i), j) for i, j in bonds if i < int(j) or (i == int(j) and j.s > (0, 0, 0))]
            self.edges = folded_edges(bonds)
            self.g = nx.Graph(self.edges)
            self.g.add_nodes_from(range(n))
            self.nb = _lists(n, bonds)
        self.hyb, self.pi = hybridization(self.z, self.nb, self.x, close)
        # neighbours without eta-bonded metals; the rings are the cycles of up to six atoms that the search of each
        # start atom keeps, none through a metal, an H, a two-neighbour halogen, an S with three or more neighbours or
        # any atom with five or more, nor through atoms sharing an eta metal
        self.lig = lig = ligand_view(self.z, self.nb, eta_only=True)[0]
        if self.cell is not None:
            self.lig = lig = Neighbours(lig)
        self.eta = {j for i in range(n) for j in self.nb[i] if j not in lig[i]}
        gr = nx.Graph([(i, j) for i, j in self.edges if j in lig[i] and i in lig[j]])
        # an H or a halogen with three or more neighbours closes no ring of a molecule; the folded rings of a periodic
        # structure keep them
        self._out = {a for a in gr if (self.z[a] == 16 and len(self.nb[a]) >= 3) or mclass(self.z[a])
                     or (self.z[a] == 1 and self.cell is None) or (self.z[a] in HALOGEN and self.hyb[a] in (1, 5))
                     or (self.z[a] in HALOGEN and len(self.nb[a]) >= 3 and self.cell is None)
                     or len(self.nb[a]) >= 5}
        # no walk passes through an atom with five or more neighbours
        walk = {a for a in gr if len(self.nb[a]) < 5}
        if self.cell is None:
            walk_nb = [sorted(j for j in gr[i] if j in walk) if i in walk else [] for i in range(n)]
        else:
            # image neighbours in image order, then atom order
            walk_nb = [[int(j) for j in sorted((j for j in list.__getitem__(lig, i) if int(j) in walk
                                               and gr.has_edge(i, int(j))),
                                              key=lambda w: (IMAGE_ORDER.get(tuple(w.s), 99), int(w)))]
                       if i in walk else [] for i in range(n)]
        #: per start atom, its rings in the order of the search (bonds and angles see these lists)
        self._ring_lists = _ring_lists(n, walk_nb)
        self._ring_sets = [[frozenset(c) for c in rings] for rings in self._ring_lists]
        self._ring_shown = _listed_sizes(walk_nb, self._ring_lists)
        # the bond of a torsion sees every ring that some start atom kept, its four atoms their own four lists
        self._cycles = [list(c) for c in sorted({tuple(sorted(c)) for rings in self._ring_lists for c in rings},
                                                key=lambda c: (len(c), c))]
        self._cyc_at = {}  # per atom, the cycles holding it in the order of _cycles
        for c in self._cycles:
            for a in c:
                self._cyc_at.setdefault(a, []).append(c)
        self._cyc_tors = self._cyc_at
        self.rings = [c for c in self._cycles if not self._out & set(c)]
        #: atoms of three-membered rings
        self.ring3 = {a for c in self.rings if len(c) == 3 for a in c}
        #: N and O atoms bonded to a metal: their rings do not count for angles and torsion ends
        self.mo = {i for i in range(n) if self.z[i] in (7, 8) and any(mclass(self.z[j]) for j in self.nb[i])}
        full = nx.Graph(self.edges)
        #: atoms on a cycle through a metal: the ends of metal bonds that are no bridge of the bond graph
        bridges = {frozenset(e) for e in nx.bridges(full)}
        self.mring = set()
        for a, m in self.edges:
            if (mclass(self.z[a]) or mclass(self.z[m])) and frozenset((a, m)) not in bridges:
                self.mring |= {a, m}
        self.pi_systems()

    def bond_pairs(self):
        """Every bond once: the graph edges, or the image bonds of a periodic structure."""
        return self.g.edges if self.cell is None else self.bonds

    def ring(self, *atoms, tors=False):
        """Size of the smallest ring of up to six atoms that the ring lists show for the atoms: a bond sees the lists
        of its two atoms, an angle (centre first) those of its centre and its ends, a torsion every kept ring; 0 when
        such a smallest ring passes through a metal, an H (in a molecule), a two-neighbour or hypervalent halogen or
        an S with three or more neighbours, or when two of the atoms share an eta metal."""
        if len(atoms) == 3 and not tors:
            # the end listed later in the neighbour list of the centre comes first
            rank = self._end_ranks(atoms[0])
            atoms = (atoms[0],) + tuple(sorted(atoms[1:], key=lambda a: rank(atoms[0], a), reverse=True))
        # the answer depends on the atoms of the cell only (not their images): kept per atom tuple
        memo = self.__dict__.setdefault("_ring_memo", {})
        key = (tuple(int(a) for a in atoms), bool(tors))
        if key not in memo:
            memo[key] = self._ring(atoms, tors)
        return memo[key]

    def _end_ranks(self, c):
        """Rank of an angle end in the neighbour list of its centre ``c``: atom order in a molecule. A periodic
        structure lists the neighbours per image; an end ranks by its place in the list of its image, then by the
        image, earlier images higher."""
        if self.cell is None:
            return lambda c, a: (int(a), 0)
        memo = self.__dict__.setdefault("_end_memo", {})
        if int(c) not in memo:
            per = {}
            for j in list.__getitem__(self.nb, int(c)):
                per.setdefault(j.s, []).append(int(j))
            memo[int(c)] = {(j, sh): (sorted(v).index(j), -IMAGE_ORDER.get(sh, 99)) for sh, v in per.items() for j in v}
        table = memo[int(c)]

        def rank(c, a):
            cs, s = getattr(c, "s", (0, 0, 0)), getattr(a, "s", (0, 0, 0))
            return table.get((int(a), (s[0] - cs[0], s[1] - cs[1], s[2] - cs[2])), (-1, 0))
        return rank

    def _ring(self, atoms, tors):
        if self.eta and any(m in self.nb[b] for a in atoms for m in self.nb[a] if m in self.eta
                            for b in atoms if b != a):
            return 0
        held = {int(a) for a in atoms}
        if tors:
            hold = [(len(c), bool(self._out & members)) for c, members in self._cycle_sets(tors, atoms[0])
                    if held <= members]
        else:
            sets = self._ring_sets
            starts = sorted(held) if len(atoms) != 3 else sorted({int(atoms[0]), int(atoms[1])})
            hold = [(len(c), bool(self._out & c)) for s in starts for c in sets[s] if held <= c]
            if len(atoms) == 3 and int(atoms[2]) not in starts:
                # the list of the other end is read with the ring sizes of the list of the first end
                first = self._ring_shown[int(atoms[1])]
                hold += [(first[k], bool(self._out & c)) for k, c in enumerate(sets[int(atoms[2])][:len(first)])
                         if len(c) <= first[k] and held <= c]
        if not hold:
            return 0
        size = min(hold)[0]
        return 0 if any(out for n_, out in hold if n_ == size) else size

    def _cycle_sets(self, tors, a):
        """The cycles of atom ``a`` with their atom sets (built per atom when first asked, then kept)."""
        tors = bool(tors and self.cell is not None)
        cache = self.__dict__.setdefault("_cycle_set_cache", {})
        key = (tors, a)
        if key not in cache:
            cyc = self._cyc_tors if tors else self._cyc_at
            cache[key] = [(c, set(c)) for c in cyc.get(a, ())]
        return cache[key]

    def ring_sizes(self, *atoms):
        """Sizes of the rings holding all the atoms of a torsion, as the ring lists of its four atoms show them."""
        memo = self.__dict__.setdefault("_sizes_memo", {})
        key = tuple(int(a) for a in atoms)
        if key not in memo:
            held = set(key)
            memo[key] = sorted({len(c) for a in held for c in self._ring_sets[a] if held <= c and not self._out & c})
        return list(memo[key])

    def pi_degree(self, i):
        """Neighbours of a pi atom that count in its electron contribution and Hueckel diagonal: a metal does not count
        for an sp or sp2 N, O or S, except for an N with a one-neighbour O, nor for an sp3 O or S with two other
        neighbours; a hypervalent halogen does not count for a two-neighbour sp2 O, nor a four-neighbour B for an sp or
        sp2 N."""
        sp = self.z[i] in (7, 8, 16) and self.hyb[i] in (1, 2)
        if self.z[i] in (8, 16) and self.hyb[i] == 3 and sum(not mclass(self.z[j]) for j in self.nb[i]) == 2:
            sp = True
        if self.z[i] == 7 and any(self.z[k] == 8 and len(self.nb[k]) == 1 for k in self.nb[i]):
            sp = False
        bridge = self.z[i] == 8 and self.hyb[i] == 2 and len(self.nb[i]) == 2  # sp2 O on a hypervalent halogen
        return sum(not (sp and (atom_class(self, j) or (bridge and self.z[j] in HALOGEN and self.hyb[j] == 5)
                                or (self.z[i] == 7 and self.z[j] == 5 and len(self.nb[j]) == 4)))
                   for j in self.nb[i])

    def contribution(self, i):
        z, h, d = self.z[i], self.hyb[i], self.pi_degree(i)
        if z == 5:
            return 1 if d == 2 else 0
        if z == 7 and d == 3:
            return 2
        if z == 17 and h == 5:
            return 1
        if z in HALOGEN:
            return 2 + (h == 1)
        if h in (1, 2) and z in (6, 7, 8, 16):
            return 1
        if z == 6 and h == -2:
            return 0
        return 2

    def system_charge(self, mem):
        """Charge of a pi system: the total charge moved onto its atoms and their H atoms, scaled by 1.1 and rounded."""
        if self.charge == 0:
            return 0
        if self.q0 is None:
            frags = self.frags if len(self.frags) > 1 else None
            self.q0 = topology_charges(self.z, self.edges, 0, frags=frags, x=self.x,
                                       nbl=None if self.cell is None else self.nb)
        dq = self.qt - self.q0
        atoms = set(mem) | {b for a in mem for b in self.nb[a] if self.z[b] == 1}
        return int(np.rint(1.1 * sum(dq[a] for a in atoms)))

    def pi_systems(self):
        n = len(self.z)
        self.pisys = [-1] * n
        self.systems, self.P = [], {}
        for s in range(n):
            if not self.pi[s] or self.pisys[s] >= 0:
                continue
            mem, st = [s], [s]
            self.pisys[s] = len(self.systems)
            while st:
                u = st.pop()
                for v in self.nb[u]:
                    if self.pi[v] and self.pisys[v] < 0:
                        self.pisys[v] = self.pisys[s]
                        st.append(v)
                        mem.append(v)
            self.systems.append(sorted(mem))
        for mem in self.systems:
            if len(mem) < 2:
                for a in mem:
                    self.pisys[a] = -1
                continue
            loc = {a: i for i, a in enumerate(mem)}
            nel = sum(self.contribution(a) for a in mem) - self.system_charge(mem)
            if nel % 2 and any(self.z[a] == 17 and self.hyb[a] == 5 for a in mem):
                nel -= 1  # a hypervalent Cl pairs the electrons of its pi system
            if nel >= 2 * len(mem):  # a filled system has no pi bonds
                continue
            sd = dict(z=[self.z[a] for a in mem], nb=[self.pi_degree(a) for a in mem], hyb=[self.hyb[a] for a in mem],
                      q=[self.qt[a] for a in mem], nel=nel,
                      edges=[(loc[i], loc[j]) for i in mem for j in self.nb[i] if j in loc and i < j])
            # the length of each bond, through its image in a periodic structure
            sd["r"] = [float(np.linalg.norm(self.x[i] - self.x[j])) / BOHR for i in mem for j in self.nb[i]
                       if j in loc and i < j]
            p = hueckel(sd)
            for (i, j), pp in zip(sd["edges"], p):
                self.P[(mem[i], mem[j]) if mem[i] < mem[j] else (mem[j], mem[i])] = float(pp)

    def bo(self, i, j):
        return self.P.get((min(i, j), max(i, j)), 0.0)


TETREL = {14, 32, 50, 82}
PI_ELEM = {5, 6, 7, 8, 16}
#: Elements that are sp with one neighbour when that neighbour has no other.
SP_PAIR = {8, 14, 15, 16, 32, 33, 34, 50, 51, 52, 82, 83, 84}


TM_GROUP = {z0 + g - 3: g for z0 in (21, 39) for g in range(3, 12)}
TM_GROUP.update({**{z0: 3 for z0 in range(57, 72)}, **{72 + g - 4: g for g in range(4, 12)}})


METAL = {3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56, 81, 82, 83, 84} | set(range(21, 31)) \
    | set(range(39, 49)) | set(range(57, 81))


def _angle_deg(x, c, a, b):
    u, v = x[a] - x[c], x[b] - x[c]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))))


def _metal_hyb(zi, d):
    """Hybridization code of a metal atom with d neighbours."""
    if zi in (3, 11, 19, 37, 55, 4, 12, 20, 38, 56):
        return {2: 1, 3: 3, 4: 3}.get(d, 0)
    if zi in (13, 31, 49, 81):
        return {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}.get(d, 5)
    if zi in (30, 48, 80):
        return 3 if d == 4 else (0 if d >= 5 else 2)
    if d <= 2:
        return 1 if TM_GROUP[zi] <= 5 else 2
    return {3: 2, 4: 3}.get(d, 3 if d == 5 and TM_GROUP[zi] == 3 else 0)


def _d_hydrides(z, nb, full, i):
    """H neighbours a d metal does not count: all of them unless it has only H, then those bridging to another atom."""
    nh = sum(z[j] == 1 for j in nb[i])
    return nh if nh < full[i] else sum(z[j] == 1 and full[j] == 2 for j in nb[i])


def _same(a, b):
    """The same atom, in the same image for the images of a periodic structure."""
    return int(a) == int(b) and getattr(a, "s", (0, 0, 0)) == getattr(b, "s", (0, 0, 0))


def _holds(atoms, a):
    for v in atoms:
        if _same(v, a):
            return True
    return False


def ligand_view(z, nb, eta_only=False):
    """Neighbour lists with the metal hybridizations: carbon atoms ignore an eta-bonded metal and, unless eta_only,
    ligand atoms ignore a metal with five or more neighbours."""
    n = len(z)
    full = [len(v) for v in nb]
    # a d metal does not count its H neighbours (see _d_hydrides); the ligand test uses the count with the H that
    # bridge to another atom left out
    mh = {i: _metal_hyb(z[i], full[i] - (_d_hydrides(z, nb, full, i) if mclass(z[i]) == 3 else 0))
          for i in range(n) if z[i] in METAL and z[i] not in (50, 82, 83, 84)}

    mh0 = {i: _metal_hyb(z[i], full[i] - (sum(z[j] == 1 and full[j] == 2 for j in nb[i]) if mclass(z[i]) == 3 else 0))
           for i in mh}

    def counted(i, j):
        if j not in mh or z[i] in METAL or z[i] == 1:
            return True
        return not ((mh0[j] == 0 and full[j] >= 2 and not eta_only
                     and not (z[i] == 6 and sum(k not in mh for k in nb[i]) >= 2))
                    or (z[i] == 6 and sum(k not in mh for k in nb[i]) <= 3  # an sp3 carbon keeps the metal
                        and any(_holds(nb[j], k) for k in nb[i] if not _same(k, j))))

    return [[j for j in nb[i] if counted(i, j)] for i in range(n)], mh


def hybridization(z, nb, x, close=frozenset()):
    """Per atom: (hyb code, pi flag) (-3 amide N, 5 hypervalent)."""
    n = len(z)
    full = [len(v) for v in nb]
    nb, mh = ligand_view(z, nb)
    deg = [len(v) for v in nb]
    hyb = [0] * n

    def o_terminal(i):
        return [j for j in nb[i] if z[j] == 8 and deg[j] == 1]

    for i in range(n):
        zi, d = z[i], deg[i]
        if i in mh:
            h = mh[i]
        elif zi == 1:
            h = 1 if d == 2 else 0
        elif zi in HALOGEN:  # a halogen counts every neighbour
            h = 1 if full[i] == 2 else (5 if full[i] >= 3 and zi != 9 else 0)
        elif zi in NOBLE:
            h = 5 if d and zi != 2 else 0  # helium stays unhybridized
        elif zi == 6:
            h = {4: 3, 3: 2, 1: 1}.get(d, 3)
            if d == 2:
                h = 1 if _angle_deg(x, i, *nb[i]) > 150.0 else -2
        elif zi == 7:
            h = {4: 3, 3: 3, 1: 1}.get(d, 3)
            if d == 3 and (o_terminal(i) or any(z[j] in METAL or (z[j] == 5 and deg[j] == 4) for j in nb[i])
                           or all(z[j] in TETREL and deg[j] >= 5 for j in nb[i])):  # between hypervalent tetrels
                h = 2
            elif d == 2:
                h = 1 if any(deg[j] == 1 and z[j] in (6, 7) or z[j] in METAL for j in nb[i]) \
                    or all(z[j] == 7 and deg[j] == 2 for j in nb[i]) \
                    or _angle_deg(x, i, *nb[i]) > 160.0 else 2
        elif zi == 8:
            h = 3 if d >= 2 else 2
            if d == 1:
                j = nb[i][0]
                if (z[j] == 1 and full[i] == 1) or (z[j] == 6 and deg[j] == 1):
                    h = 1
                elif z[j] == 1:  # a hydroxide on metals the ligand view drops stays sp3
                    h = 3
        elif zi == 5:
            h = {2: 1, 3: 2}.get(d, 3)
        elif zi in TETREL:
            h = 5 if d >= 5 else 3
            if zi in (50, 82) and d == 2:
                h = 1 if _angle_deg(x, i, *nb[i]) > 150.0 else 2
        elif zi in (15, 33, 51, 83):
            h = 5 if d >= 5 else 3
            if d == 2:
                h = 1 if _angle_deg(x, i, *nb[i]) > 160.0 else 2
        elif zi in (16, 34, 52):
            # bonded only to metals the ligand view ignores: the count of all neighbours
            h = {1: 2, 2: 3, 3: 3}.get(d or full[i], 5)
        elif zi == 84:
            h = 5 if d >= 4 else 3
        else:
            h = 3
        hyb[i] = h
    # two bent two-neighbour carbons bonded to each other form a bent double bond: both sp2
    bent = [i for i in range(n) if hyb[i] == -2 and z[i] == 6]
    for i in [i for i in bent if any(hyb[j] == -2 and z[j] == 6 for j in nb[i])]:
        hyb[i] = 2
    side = {}  # O and S between a metal and an atom with three neighbours are sp2, like a terminal atom on that atom
    for i in range(n):
        if z[i] in (8, 16) and deg[i] == 2 and sum(j in mh for j in nb[i]) == 1:
            j = next(j for j in nb[i] if j not in mh)
            m = next(k for k in nb[i] if k in mh)
            if sum(k not in mh for k in nb[j]) == 3 and (min(j, m), max(j, m)) not in close:
                hyb[i], side[i] = 2, j
        elif z[i] == 8 and deg[i] == 2 and sum(z[j] in HALOGEN and deg[j] >= 3 for j in nb[i]) == 1:
            j = next(j for j in nb[i] if not (z[j] in HALOGEN and deg[j] >= 3))  # O on a hypervalent halogen
            if deg[j] == 3:
                hyb[i], side[i] = 2, j
    for i in range(n):  # terminal atoms of an isolated pair: boron as a terminal atom, heavier p-block atoms sp
        if deg[i] == 1 and deg[nb[i][0]] == 1 and i not in mh and full[i] == 1:
            if z[i] == 5:
                hyb[i] = 0
            elif z[i] in SP_PAIR:
                hyb[i] = 1
    for i in range(n):  # Po between two sp2 atoms
        if z[i] == 84 and deg[i] == 2 and all(hyb[j] == 2 for j in nb[i]):
            hyb[i] = 2

    def bridge_pi(i):  # a two-neighbour halogen is pi only next to a non-metal with lone pairs, sp/sp2 or hypervalent
        return any(z[j] != 1 and z[j] not in METAL and (hyb[j] in (1, 2, 5) or z[j] in (7, 8, 16)) for j in nb[i])

    # a bent two-neighbour C (carbene, sigma radical) is pi with an empty p orbital
    pi = [int((z[i] in PI_ELEM and hyb[i] in (1, 2)) or (z[i] in HALOGEN and hyb[i] == 1 and bridge_pi(i))
              or (z[i] == 6 and hyb[i] == -2))
          for i in range(n)]
    base = list(pi)
    for i in range(n):
        if z[i] == 7 and deg[i] == 3 and hyb[i] == 2 and not o_terminal(i) \
                and not any(z[j] in (7, 8, 16) or base[j] for j in nb[i]):
            pi[i] = 0  # N on a metal without pi neighbours
        if z[i] == 7 and len(nb[i]) == 2 and hyb[i] == 2 and all(z[j] in (1, 6) and hyb[j] in (0, 3) for j in nb[i]):
            pi[i] = 0  # sp2 N between saturated atoms
    for i in range(n):  # O on four-neighbour P or S: double bond without a pi system
        j = nb[i][0] if deg[i] == 1 else side.get(i, -1)
        if z[i] == 8 and j >= 0 and not ((z[j] in PI_ELEM and (hyb[j] in (1, 2) or i in side))
                                         or (z[j] == 16 and deg[j] <= 3) or z[j] == 8):
            pi[i] = 0
        if z[i] == 5 and pi[i] and not any(z[j] in (7, 8, 9, 17) or (z[j] == 6 and hyb[j] == 2) for j in nb[i]):
            pi[i] = 0
    for i in range(n):  # a two-neighbour halogen stays pi next to a hypervalent atom or between N, O and S only
        nm = [j for j in nb[i] if z[j] not in METAL]
        if z[i] in HALOGEN and hyb[i] == 1 and pi[i] and not (any(hyb[j] == 5 for j in nb[i])
                                                              or (nm and all(z[j] in (7, 8, 16) for j in nm))):
            pi[i] = 0
    core = list(pi)
    for i in range(n):
        nm = sum(j not in mh for j in nb[i])  # an O or S on a metal with two other neighbours keeps its lone pair
        lone = (z[i] == 7 and deg[i] == 3 and hyb[i] == 3) or (z[i] in (8, 16) and (deg[i] == 2 or nm == 2)) \
            or (z[i] == 16 and deg[i] == 3) or \
            (z[i] in (9, 17) and deg[i] != 2)
        if lone and any(core[j] for j in nb[i]):
            pi[i] = 1
    for i in range(n):  # two bonded O atoms on the same metal are pi
        if z[i] == 8 and any(z[j] == 8 and set(nb[i]) & set(nb[j]) & set(mh) for j in nb[i]):
            pi[i] = 1
    for i in range(n):  # a hypervalent Cl and its terminal O form a pi system
        if z[i] == 17 and hyb[i] == 5 and any(z[j] == 8 and deg[j] == 1 for j in nb[i]):
            pi[i] = 1
            for j in nb[i]:
                if z[j] == 8 and deg[j] == 1:
                    pi[j] = 1
    for i in range(n):  # amide N: three neighbours, a carbonyl C its only pi neighbour (imides, anilides stay 3);
        # in a cell the carbonyl O must lie in the image of its C
        if z[i] == 7 and deg[i] == 3 and hyb[i] == 3 and sum(
                z[j] == 6 and any(z[k] == 8 and deg[k] == 1 and not any(getattr(k, "s", ())) for k in nb[j])
                for j in nb[i]) == 1 \
                and sum(bool(core[j]) for j in nb[i]) == 1:
            hyb[i] = -3
    return hyb, pi


#: Hueckel model in Hartree: charge coefficient of the diagonal, stop tolerance of the band energy, iteration limit,
#: level above which one electron is removed, coupling at P = 2/3, its slope in P with the factor per sp atom, the
#: decrease of the element scale of the coupling per Bohr of bond length, the smearing energy and the frontier gap
#: below which the first iteration of an even electron count fills its levels with whole electrons.
HUECKEL_Q, HUECKEL_TOL, HUECKEL_MAXIT, HUECKEL_EHIGH = -0.024, 1e-4 / 27.2114, 5, 0.4 / 27.2114
HUECKEL_B0, HUECKEL_SLOPE, HUECKEL_SP, HUECKEL_R = 0.1, 0.07, 1.45, 1e-9
HUECKEL_KT = 0.012667229356482
HUECKEL_DEGEN = 2e-4 * HUECKEL_KT
#: Scales of the distance damping of the angle and the torsion terms.
ANGLE_DAMP, TORS_DAMP = (225.0 / 238.0) ** 2, (225.0 / 202.0) ** 2
HUECKEL_D = {(5, 2): -0.05, (5, 3): 0.003, (7, 1): 0.014, (7, 2): 0.014, (7, 3): -0.039, (8, 1): -0.038, (8, 2): -0.091,
             (9, 1): -0.082, (16, 1): -0.030, (16, 2): -0.083, (16, 3): -0.083, (17, 1): -0.083,
             (17, 3): 0.023, (17, 4): 0.023, (17, 5): 0.023, (17, 6): 0.023}
#: Hueckel diagonal of a bent two-neighbour C (hyb -2)
HUECKEL_CARBENE = 0.053
HUECKEL_H = {5: 0.5, 6: 1.0, 7: 0.66, 8: 1.1, 9: 0.23, 16: 0.6, 17: 1.0}


def _fermi(eps, ne):
    """Fermi occupations of ``ne`` electrons: bracketed Newton iteration for the level."""
    if ne <= 0:
        return np.zeros_like(eps)
    occ = lambda mu: 1.0 / (1.0 + np.exp(np.clip((eps - mu) / HUECKEL_KT, -700, 700)))  # noqa: E731
    lo, hi = eps.min() - 50.0, eps.max() + 50.0
    mu = 0.5 * (lo + hi)
    for _ in range(200):
        f = occ(mu)
        miss = f.sum() - ne
        if miss < 0:
            lo = mu
        else:
            hi = mu
        slope = float((f * (1.0 - f)).sum()) / HUECKEL_KT
        new = mu - miss / slope if slope > 0 else lo
        if not lo < new < hi:
            new = 0.5 * (lo + hi)
        if new == mu or abs(new - mu) <= 4e-16 * max(1.0, abs(mu)):
            mu = new
            break
        mu = new
    return occ(mu)


def smear(eps, nel):
    """Occupations 0..2: alpha ((nel + 1) // 2) and beta (nel // 2) electrons smeared separately at 4000 K."""
    return _fermi(eps, (nel + 1) // 2) + _fermi(eps, nel // 2)


def _coupling(s, i, j):
    """(element scale, slope in P) of a pi coupling; the slope grows by ``HUECKEL_SP`` per sp atom, and atoms without
    a hybrid (halogens) couple like sp2."""
    g = math.sqrt(HUECKEL_H[s["z"][i]] * HUECKEL_H[s["z"][j]])
    return g, HUECKEL_SLOPE * HUECKEL_SP ** sum(s["hyb"][a] == 1 for a in (i, j))


def hueckel(s):
    """Pi bond orders of one pi system (dict with z, nb, hyb, q, nel, edges, optional bond lengths r in Bohr) from
    the iterated Hueckel model.

    The diagonal is ``HUECKEL_Q q + d(Z, neighbours)``, the coupling of a bond ``-(g - HUECKEL_R r) (B0 + b (P - 2/3))``
    with the bond order P started at 2/3; at most five iterations, stopped once the band energy changes by less than the
    tolerance. An even electron count whose frontier levels lie closer than ``HUECKEL_DEGEN`` in an iteration fills
    them with whole electrons there, which breaks the symmetry of a degenerate pair. When the highest occupied level
    lies above ``HUECKEL_EHIGH`` the bond orders are those of the same orbitals with one electron less, with whole
    electrons when the count is odd and its alpha frontier levels are degenerate.
    """
    p = np.full(len(s["edges"]), 2.0 / 3.0)
    diag = [HUECKEL_Q * q + (0.0 if zz == 6 and (zz, nb) not in HUECKEL_D else HUECKEL_D[(zz, nb)])
            + (HUECKEL_CARBENE if zz == 6 and hh == -2 else 0.0)
            for q, zz, nb, hh in zip(s["q"], s["z"], s["nb"], s["hyb"])]
    eb_old = None
    rr = s.get("r", [0.0] * len(s["edges"]))
    nel, half = s["nel"], s["nel"] // 2
    for _ in range(HUECKEL_MAXIT):
        h = np.diag(diag)
        for (i, j), pp, r in zip(s["edges"], p, rr):
            g, slope = _coupling(s, i, j)
            h[i, j] = h[j, i] = -(g - HUECKEL_R * r) * (HUECKEL_B0 + slope * (pp - 2.0 / 3.0))
        eps, vec = np.linalg.eigh(h)
        top = (nel + 1) // 2  # levels holding the alpha electrons
        degen = 0 < top < len(eps) and eps[top] - eps[top - 1] < HUECKEL_DEGEN
        o = np.where(np.arange(len(eps)) < half, 2.0, 0.0) if degen and nel % 2 == 0 else smear(eps, nel)
        pm = (vec * o) @ vec.T
        p = np.array([pm[i, j] for i, j in s["edges"]])
        eb = float((o * eps).sum())
        if eb_old is not None and abs(eb - eb_old) < HUECKEL_TOL:
            break
        eb_old = eb
    nh = int(np.flatnonzero(o > 0.5 + 1e-6).max()) if (o > 0.5 + 1e-6).any() else 0
    if eps[nh] > HUECKEL_EHIGH:
        whole = np.where(np.arange(len(eps)) < half, 2.0, 0.0)  # an odd count with a degenerate alpha pair
        low = 0 < half < len(eps) and eps[half] - eps[half - 1] < HUECKEL_DEGEN  # the frontier of nel - 1
        pm = (vec * (whole if (degen or low) and nel % 2 else smear(eps, nel - 1))) @ vec.T
        p = np.array([pm[i, j] for i, j in s["edges"]])
    return p


ROW = {z: (1 if z <= 2 else 2 if z <= 10 else 3 if z <= 18 else 4 if z <= 36 else 5 if z <= 54 else 6)
       for z in range(1, 87)}
F_HYB = {(3, 3): 1.0, (3, 2): 1.0792, (3, 1): 1.3234, (2, 2): 1.24, (2, 1): 1.4842, (1, 1): 1.98}


#: Bond exponent in 1/Bohr^2: its base value, the coefficient of the squared electronegativity difference and the
#: coefficient of f - 1.
BOND_A0, BOND_AEN, BOND_AHYB = 0.46779278, 0.11831001, 0.09469278


def hclass(t, i):
    h = t.hyb[i]
    return 3 if h in (0, -3, 3, 5) else (2 if h in (2, -2) else 1)


def bond_f(t, i, j):
    """Hybridization-pair factor f of the bond: k is multiplied by f, alpha = a0 + b dEN^2 + c (f - 1)."""
    hi, hj = t.hyb[i], t.hyb[j]
    for a in (i, j):  # a two-neighbour halogen: 0.3 for F, 0.5 for the heavier ones
        if t.z[a] in HALOGEN and t.hyb[a] == 1:
            return 0.3 if t.z[a] == 9 else 0.5
    if 5 in (hi, hj):
        return 1.22
    for a, b in ((i, j), (j, i)):
        if t.z[a] == 6 and t.hyb[a] == 1 and t.z[b] == 8:
            return 1.782
    if 7 in (t.z[i], t.z[j]) and {hi, hj} & {2, -2} and {hi, hj} & {3, -3}:
        return 1.2896
    return F_HYB[tuple(sorted((hclass(t, i), hclass(t, j)), reverse=True))]


ROW_SHIFT = {3: -0.11, 4: -0.22, 5: -0.28, 6: -0.28}
METAL_SHIFT = {1: 0.2, 2: 0.05, 3: 0.15}


METAL_ALPHA = {1: 0.28, 2: 0.28, 3: -0.22}
METAL_C = {1: 0.4, 2: 0.85, 3: 0.75}


METAL_NN = {1: 0.1, 2: 0.03, 3: 0.036}
METAL_H = {1: 1.2 / 1.1, 2: 1.2 / 1.1}


TM_H = {4: 0.8 / 1.1, 5: 0.8 / 1.1, 6: 1.0 / 1.1}
#: Electronegativities of the bond exponent that differ from the Pauling table.
BOND_EN = {74: 1.70, 76: 2.10, 78: 2.20, 79: 2.20}


CHALCOGEN = {8, 16, 34, 52, 84}
H_METAL = METAL


def mclass(z):
    """Metal class of the bond term: 1 groups 1 and 2, 2 group 13, 3 transition metals with Sr and Ba, 0 otherwise."""
    return 3 if z in D_METAL else (2 if z in (13, 31, 49, 81) else (1 if z in (3, 4, 11, 12, 19, 20, 37, 55) else 0))


def metal_metal_bond(t, i, j):
    """(shift, alpha, k) of a bond between two metals: the class shifts and coordination denominators of both atoms
    and a charge coefficient summed over both classes."""
    zi, zj = t.z[i], t.z[j]
    ci, cj = mclass(zi), mclass(zj)
    rows = (ROW[zi], ROW[zj])
    base = -0.11 if 2 in rows else ROW_SHIFT[rows[0]] + ROW_SHIFT[rows[1]]
    en = [BOND_EN.get(z, PAULING[z]) for z in (zi, zj)]
    alpha = BOND_A0 + METAL_ALPHA[max(ci, cj)] * BOND_AEN * (en[0] - en[1]) ** 2
    sig = 1.0 / (1.0 + math.exp(375.0 * t.qt[i] * t.qt[j]))
    k = (1.0 + (METAL_C[ci] + METAL_C[cj] - 1.0) * sig) / (
        (1.0 + METAL_NN[ci] * len(t.nb[i]) ** 2) * (1.0 + METAL_NN[cj] * len(t.nb[j]) ** 2))
    # a transition metal bonded to a main-group metal below the second row
    if max(ci, cj) == 3 and ROW[zj if ci == 3 else zi] > 2:
        k *= 0.65
    if ci == cj == 3:  # two transition metals: alpha and k scale with the coordination gradient of both atoms
        f = tm_tm_factor(t, i, j)
        alpha, k = alpha + BOND_AHYB * (f - 1.0), k * f
    return base + METAL_SHIFT[ci] + METAL_SHIFT[cj], alpha, -PAR[zi - 1, 4] * PAR[zj - 1, 4] * k


def tm_tm_factor(t, i, j):
    """Bond factor of two transition metals, 1.7 (2 - min(T, 1)). T is the mean over both atoms a of
    16 exp(-0.005 EN_a^8) times half the summed norms of the gradients of ln(1 + logCN_a) on all atoms, with the erf
    coordination number of the charge model (logCN damped at 4.4)."""
    x = np.asarray(t.x, dtype=float) / BOHR
    z = np.asarray(t.z)
    if t.cell is not None:
        # all atoms of the images within the largest count radius
        cut = 2.0 * (4.0 / 3.0) * 2.0 * float(RCOV[z].max()) * BOHR
        tr = np.array(shifts_within(t.cell, cut), dtype=float) @ t.cell / BOHR
        x = (x[None, :, :] + tr[:, None, :]).reshape(-1, 3)
        folded = np.tile(np.arange(len(z)), len(tr))
        z = np.tile(z, len(tr))
    else:
        folded = np.full(len(z), -1)
    rcov = RCOV[z]
    tot = 0.0
    for a in (i, j):
        xa = np.asarray(t.x[a], dtype=float) / BOHR
        d = np.linalg.norm(x - xa, axis=1)
        rc = 4.0 / 3.0 * (rcov[a] + rcov)
        near = (d > 0.0) & (d < 2.0 * rc)
        y = 7.5 * (d[near] / rc[near] - 1.0)
        # the images of the atom count twice in its coordination number and move with it (no gradient)
        own = folded[near] == int(a)
        cn = 0.5 * float(sum(math.erfc(v) * (2.0 if o else 1.0) for v, o in zip(y, own)))
        g = np.where(own, 0.0, 7.5 / (math.sqrt(math.pi) * rc[near]) * np.exp(-y ** 2))
        grad = ((g / d[near])[:, None] * (xa - x[near])).sum(axis=0)
        logcn = math.log(1.0 + math.exp(4.4)) - math.log(1.0 + math.exp(4.4 - cn))
        dlog = 1.0 / ((1.0 + math.exp(cn - 4.4)) * (1.0 + logcn))
        en = BOND_EN.get(t.z[a], PAULING[t.z[a]])
        tot += 0.5 * 16.0 * math.exp(-0.005 * en ** 8) * dlog * 0.5 * (float(np.linalg.norm(grad)) + float(g.sum()))
    return 1.7 * (2.0 - min(tot, 1.0))


def bridging_h(t, h):
    """An H between a metal and a non-metal."""
    return len(t.nb[h]) == 2 and sorted(bool(mclass(t.z[a])) for a in t.nb[h]) == [False, True]


def atom_class(t, a):
    """Metal class of atom a: Sn, Pb, Bi and Po with more than four neighbours count as group 13 metals."""
    c = mclass(t.z[a])
    return 2 if not c and t.z[a] in (50, 82, 83, 84) and len(t.nb[a]) > 4 else c


def metal_bond(t, i, j):
    """(shift, alpha, k) of a bond to a metal (the atom of the higher class is the metal)."""
    if atom_class(t, i) and atom_class(t, j):
        return metal_metal_bond(t, i, j)
    m, l = (i, j) if atom_class(t, i) >= atom_class(t, j) else (j, i)
    zm, zl = t.z[m], t.z[l]
    c = atom_class(t, m)
    rows = (ROW[zm], ROW[zl])
    base = -0.11 if (zl == 1 or min(rows) <= 2) else ROW_SHIFT[rows[0]] + ROW_SHIFT[rows[1]]
    alpha = BOND_A0 + METAL_ALPHA[c] * BOND_AEN * (BOND_EN.get(zm, PAULING[zm]) - PAULING[zl]) ** 2
    sig = 1.0 / (1.0 + math.exp(375.0 * t.qt[i] * t.qt[j]))
    k = (1.0 + (METAL_C[c] - 0.6 * (zl == 1)) * sig) / (1.0 + METAL_NN[c] * len(t.nb[m]) ** 2)
    if zl == 1:
        k *= TM_H[ROW[zm]] if c == 3 else METAL_H[c]
        if bridging_h(t, l):
            k *= 11.0 / 14.0
    elif c == 3:
        k *= 1.3 if zl in HALOGEN else (0.85 if zl in CHALCOGEN else (1.6 if zl == 15 else (
            0.65 if ROW[zl] > 2 else 1.0)))
    if c == 3 and zl == 6 and t.hyb[l] == 1:
        base, k = base - 0.45, 1.5 * k
    elif c == 3 and zl == 7 and t.hyb[l] == 1 and len(t.nb[l]) >= 2:  # a coordinated sp N, not an N-M pair
        base, k = base + 0.47, 0.4 * k
    elif c == 3 and zl == 6 and m not in t.lig[l]:
        # eta-bonded carbon
        base, alpha, k = base + 0.04 * len(t.nb[m]), alpha + METAL_ALPHA[c] * BOND_AHYB, 0.78 * k
    return base + METAL_SHIFT[c], alpha, -PAR[zm - 1, 4] * PAR[zl - 1, 4] * k


def bond_shift(t, i, j):
    zi, zj = sorted((t.z[i], t.z[j]))
    pi = (min(i, j), max(i, j)) in t.P
    if zi == 1:
        other = j if t.z[i] == 1 else i
        if zj in NOBLE:
            return -0.16
        s = -0.16 if (zj in HALOGEN or t.hyb[other] in (0, 5) or hclass(t, other) == 2) else (
            -0.02 if hclass(t, other) == 1 else -0.182)
    elif zi == zj == 9:
        s = 0.11
    elif ROW[zi] <= 2 or ROW[zj] <= 2:  # helium like the second row
        s = -0.11
    else:
        s = ROW_SHIFT[ROW[zi]] + ROW_SHIFT[ROW[zj]]
    # terminal halogens (an F with three neighbours included), terminal boron and helium
    term = [(t.z[a] in HALOGEN and t.hyb[a] == 0) or (t.z[a] == 5 and t.hyb[a] == 0) or t.z[a] == 2 for a in (i, j)]
    if zi != 1 and term[0] != term[1] and not pi:
        other = j if term[0] else i
        if t.hyb[other] == 3:
            s -= 0.022
        elif t.hyb[other] == 1:
            s += 0.14
    if pi:
        s += 0.1258 - 0.34 * t.bo(i, j)
    if 5 in (t.hyb[i], t.hyb[j]) and zi != 1 and not pi:
        s += 0.03
    return s


def bond_rows(t, hbb):
    """Geometry-independent part per bond: (i, j, shift, pair factor, alpha without the hydrogen-bond term, k, H atom
    or -1, acceptors of that H)."""
    z = np.asarray(t.z)
    nn = [len(v) for v in t.nb]
    out = []
    for i, j in t.bond_pairs():
        zi, zj = sorted((int(z[i]), int(z[j])))
        sh = bond_shift(t, i, j)
        fp = 1.0 if zi == zj else F_PAIR[(zi, zj)]
        fh = bond_f(t, i, j)
        alpha_b = BOND_A0 + BOND_AEN * (PAULING[zi] - PAULING[zj]) ** 2 + BOND_AHYB * (fh - 1.0)
        h = i if z[i] == 1 else (j if z[j] == 1 else -1)
        if h >= 0 and not mclass(zj) and len(t.nb[h]) == 2:  # the non-metal bond of an H with two neighbours
            # an H with two neighbours is sp: a terminal (hyb 0) halogen partner adds its sp-partner shift
            sh, fh = -0.16 + (0.14 if z[i + j - h] in HALOGEN and t.hyb[i + j - h] == 0 else 0.0), 0.3
            alpha_b = BOND_A0 + BOND_AEN * (PAULING[zi] - PAULING[zj]) ** 2 + BOND_AHYB * (fh - 1.0)
        acc = ()
        if h >= 0 and zj in (7, 8):
            acc = tuple(sorted({b_ for a0, b_, h_ in hbb if h_ == h and z[b_] in (7, 8)}))
        ring = t.ring(i, j)
        f = {3: 1.18, 4: 1.08, 5: 1.02}.get(ring, 1.0)
        if zi == 1 and zj == 6 and (i in t.ring3 or j in t.ring3):
            f = 1.05
        if zi == 1 and zj == 6:
            c_ = i if z[i] == 6 else j
            if hclass(t, c_) == 2 and sum(z[a] == 8 for a in t.nb[c_]) == 1:
                f *= 0.95
        f *= {(1, 8): 0.93, (1, 7): 1.06, (1, 5): 1.1}.get((zi, zj), 1.0)
        if ROW[zi] >= 3 and ROW[zj] >= 3:
            f *= 1.0 / (1.0 + 0.007 * nn[i] ** 2) / (1.0 + 0.007 * nn[j] ** 2)
        qq = min(1050.0 * t.qt[i] * t.qt[j], 700.0)  # bounded: an ion bonded to its own images
        k = -PAR[zi - 1, 4] * PAR[zj - 1, 4] * (1.0 + 0.047 / (1.0 + math.exp(qq))) * f
        k *= fh
        if (min(i, j), max(i, j)) in t.P:
            k *= 0.685 + t.bo(i, j)
        if atom_class(t, i) or atom_class(t, j):
            sh, alpha_b, k = metal_bond(t, i, j)
        # the image of a periodic bond stays on its second atom
        out.append(((min(i, j), max(i, j)) if t.cell is None else (i, j)) + (sh, fp, alpha_b, k, h, acc))
    return out


CJ = {1: -0.08, 5: -0.05, 9: 0.1, 35: -0.11, 53: -0.07, **{z: 0.0 for z in NOBLE}}
CA = {1: 0.0, 6: 0.09, 7: -0.21, 8: -0.03, 9: 0.5, 16: -0.03, 17: 0.5, 34: -0.03, 35: 0.5, 52: -0.03, 53: 0.5,
      84: -0.03, 85: 0.5}
for _z in range(1, 87):
    _c = mclass(_z)
    CJ.setdefault(_z, -0.9 if _c == 3 else (-0.08 if _c else -0.02))
    CA.setdefault(_z, -0.1 if _c == 3 else (0.3 if _c else 0.0))


def hardness_shift(t):
    """Per-atom c_J of the charge-dependent hardness J' = J + c_J qt.

    The pi test of N is positional, as in the reference program: an atom passes when its index is below the number
    of pi atoms (sp2 N and O outside pi systems, except those on a hypervalent atom, and halogens on an sp or sp2
    metal included). An amide N (sp3 or the N of an anilide) whose carbonyl C and O pass takes -0.16.
    """
    z, n = t.z, len(t.z)
    npi = sum(bool(p) or (z[i] in (7, 8) and t.hyb[i] == 2 and not any(t.hyb[j] == 5 for j in t.nb[i]))
              for i, p in enumerate(t.pi)) + sum(
        z[i] in HALOGEN and len(t.nb[i]) == 1 and z[t.nb[i][0]] in METAL and t.hyb[t.nb[i][0]] in (1, 2)
        for i in range(n))
    ispi = [i < npi for i in range(n)]
    out = []
    for i, zi in enumerate(z):
        if zi == 6:
            out.append({1: -0.34, 2: -0.45, -2: -0.45}.get(t.hyb[i], -0.27))
        elif zi == 7:
            cpi = [c for c in t.nb[i] if z[c] == 6 and ispi[c]]
            amide = len(cpi) == 1 and any(z[o] == 8 and len(t.nb[o]) == 1 and ispi[o]  # in a cell, the O in the
                                          and getattr(o, "s", None) == getattr(cpi[0], "s", None)  # image of its C
                                          for o in t.nb[cpi[0]])
            out.append(-0.16 if ispi[i] and amide and t.hyb[i] in (-3, 3) else (-0.14 if ispi[i] else -0.13))
        elif zi == 8:
            out.append(-0.08 if t.pi_degree(i) == 1 or len(t.nb[i]) == 1 else -0.15)  # a terminal oxo on a metal too
        else:
            out.append(CJ[zi] if mclass(zi) or not atom_class(t, i) else CJ[13])
    return np.array(out)


ANGLE_RAD = dict(enumerate((
    0.32, 0.46, 1.2, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.4, 1.25, 1.13, 1.04, 1.1, 1.02, 0.99, 0.96,
    1.76, 1.54, 1.33, 1.22, 1.21, 1.1, 1.07, 1.04, 1.0, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.1, 1.14, 1.17,
    1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31,
    2.09, 1.76, 1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.5, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37,
    1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.3, 1.3, 1.36, 1.31, 1.38, 1.42), start=1))
N_OUTER = {1: 0.39, 6: 0.21, 8: 0.45}


HEAVY_T0 = {14: 90.0, 32: 90.0, 50: 99.5, 15: 79.0, 33: 79.0, 51: 79.0, 16: 79.0, 34: 79.0, 52: 79.0, 82: 99.5,
            83: 79.0, 84: 79.0}


def is_pi(t, a):
    return t.pisys[a] >= 0


HYPER_LINEAR = 160.0
LINEAR_COS = math.cos(math.radians(170.0))


def hyper_angle(t, c, a, b):
    # the input geometry is fixed: kept per atom triple (with the images)
    memo = t.__dict__.setdefault("_angle_memo", {})
    key = tuple((int(v), getattr(v, "s", None)) for v in (c, a, b))
    if key not in memo:
        # bond vectors and lengths of the centre kept by angle_params (periodic), else computed here
        pos, uu, ln = t.__dict__.get("_angle_vecs", {}).get(key[0][0] if key[0][1] is None else None, (None,) * 3)
        if pos is not None and key[1] in pos and key[2] in pos:
            ia, ib = pos[key[1]], pos[key[2]]
            v = float(uu[ia] @ uu[ib] / ln[ia] / ln[ib])
        else:
            u, w = t.x[a] - t.x[c], t.x[b] - t.x[c]
            v = float(u @ w / np.linalg.norm(u) / np.linalg.norm(w))
        memo[key] = math.degrees(math.acos(max(-1.0, min(1.0, v))))
    return memo[key]


def angle_t0(t, c, a, b):
    """(t0 in degrees, factor) of the angle a-c-b; t0 None when no rule applies."""
    z, h = t.z, t.hyb
    zc, hc = z[c], h[c]
    def ring_():  # evaluated only by the rules that use it
        return 0 if t.mo & {c, a, b} else t.ring(c, a, b)
    zab = tuple(sorted((z[a], z[b])))
    sp = any(h[v] == 1 for v in (a, b))
    if sp and any((z[v] in (5, 6) or z[v] in H_METAL or z[v] in HALOGEN or (z[v] == 1 and mclass(zc) == 3))
                  and h[v] == 1 for v in (a, b)) and 7 not in zab:
        f = 0.6
    else:
        f = 1.0
    if mclass(zc) and hc != 5:
        lin = hyper_angle(t, c, a, b) > HYPER_LINEAR
        if hc == 0:
            return (180.0 if lin else 90.0), 1.35
        if hc != 1 and lin:
            return 180.0, f
        return {1: 180.0, 2: 120.0}.get(hc, 109.5), ((1.0 if 7 in zab else 0.6) if hc == 1 else f)
    if zc == 9 and len(t.nb[c]) >= 3:  # an F with three or more neighbours (the ring value in a periodic structure)
        return ({3: 82.0, 4: 96.0}.get(ring_(), 100.0) if t.cell is not None else 100.0), f
    if zc in HALOGEN and len(t.nb[c]) == 2:
        # a two-neighbour halogen: 0.6 Z^-0.15 whatever its ends (an end at a two-neighbour H brings that 0.6 itself);
        # an F stays bent, an At is linear at any geometry
        hend = any(z[v] == 1 and len(t.nb[v]) == 2 for v in (a, b))
        lin = zc == 85 or (zc != 9 and hyper_angle(t, c, a, b) > HYPER_LINEAR)
        return (180.0 if lin else 90.0), zc ** -0.15 * (1.0 if hend else 0.6)
    if zc == 6 and hc == 3 and any(mclass(z[v]) for v in (a, b)) and hyper_angle(t, c, a, b) > HYPER_LINEAR:
        return 180.0, f
    if zc == 6 and hc == -2:  # a bent two-neighbour C
        return 145.0, f
    if hc == 1:
        if zc == 6 and any(mclass(z[v]) == 3 for v in (a, b)):
            return 180.0, 14.0
        if zc == 7 and any(mclass(z[v]) == 3 for v in (a, b)):  # N2 bends; a linear nitrile is stiffer
            return (135.0, 1.0) if 7 in zab else (180.0, 10.0 if hyper_angle(t, c, a, b) > HYPER_LINEAR else 0.6)
        # an sp atom in a three- or four-membered cycle of a periodic structure takes the small-ring t0
        return {3: 82.0, 4: 96.0}.get(ring_(), 180.0), \
            (2.0 if zab == (8, 8) and zc == 6 else (1.0 if 7 in zab else 0.6))
    if zc == 6 and hc == 2:
        if sp:
            return 120.0, f
        no = zab.count(8)
        r3 = c in t.ring3 and not ring_()
        rc = 0 if any(mclass(z[v]) for v in t.lig[c]) else ring_()  # like N and O, a carbon on a metal loses its ring
        # small rings set the angle even between two O ends (cyclic carbonates); 6-rings keep the carbonyl value
        t0 = {3: 82.0, 4: 96.0, 5: 109.0}.get(rc) or (122.0 if no == 2 else {6: 120.0}.get(rc, 124.0 if r3 else 120.0))
        return t0, f * (0.7 if no == 1 else 1.0)
    if zc == 5:
        return {3: 82.0, 4: 96.0}.get(ring_(), 115.0), f
    if zc == 7 and hc == -3 and any(is_pi(t, v) for v in (a, b)):
        return {3: 82.0, 4: 96.0}.get(ring_(), 115.0), f * 1.2
    if zc == 7 and hc == 2:
        if len(t.nb[c]) == 3:
            return 120.0, f
        t0 = 103.0 if any(z[v] == 8 for v in t.nb[c]) else (105.0 if ring_() else 115.0)
        return t0, f * (1.0 if sp else 1.4)
    if hc == 5:
        f = f if f != 1.0 or any(z[v] == 1 and len(t.nb[v]) == 2 for v in (a, b)) else 0.11
        if hyper_angle(t, c, a, b) > HYPER_LINEAR:
            return 180.0, f
        return {3: 82.0, 4: 96.0}.get(ring_()) or (115.0 if zc == 16 and 8 in zab else 90.0), f
    if zc in (15, 33, 51, 83, 84) and hc == 2:
        return 120.0, f
    if zc in (15, 33, 51, 83) and len(t.nb[c]) == 4:
        return {3: 82.0, 4: 96.0}.get(ring_()) or 99.5 + 5.0 * sum(v != 1 for v in zab), f
    if zc == 16 and is_pi(t, c):
        return (120.0 if hc == 2 and any(mclass(v) for v in zab) else HEAVY_T0[16] + 5.0 * sum(v != 1 for v in zab)), f
    if zc == 8 and len(t.nb[c]) == 2 and any(mclass(v) for v in zab) and hyper_angle(t, c, a, b) > HYPER_LINEAR:
        return 180.0, f * (0.3 if hc == 2 and any(mclass(v) == 3 for v in zab) else 1.0)
    if zc == 8 and 14 in zab:
        return (125.5 if any(mclass(v) for v in zab) else (118.5 if zab == (14, 14) else 111.5)), f
    if zc == 8 and len(t.nb[c]) == 2 and any(v in H_METAL for v in zab):
        return 118.5, f
    if zc in (50, 82) and hc == 2:
        return 90.0, f
    if zc == 8 and is_pi(t, c) and len(t.nb[c]) == 2:
        return {3: 82.0, 4: 96.0}.get(ring_()) or (109.0 if is_pi(t, a) and is_pi(t, b) else 104.5), f
    if zc == 7 and hc == 3 and zab != (1, 1) and any(is_pi(t, v) for v in (a, b)):
        return {3: 82.0, 4: 96.0}.get(ring_(), 113.0), (f if sp else f * (1.0 - 0.70 * (t.bo(c, a) + t.bo(c, b))))
    if zc == 7 and not sp:
        f *= N_OUTER.get(z[a], 0.2) + N_OUTER.get(z[b], 0.2)
    if zc == 8 and zab == (1, 1) and len(t.nb[c]) == 2:
        f *= 1.2
    heavy = sum(v != 1 for v in zab)
    if zc == 6:
        t0 = 108.6 if zab == (1, 1) else (108.5 if zab.count(8) == 1 else 109.5)
    elif zc == 7:
        t0 = 104.0
    elif zc == 8:
        t0 = 109.5 if len(t.nb[c]) >= 3 else (100.0 if zab == (1, 1) else 104.5)
    elif zc in HEAVY_T0:
        t0 = (79.0 if zc in (14, 32, 50) and len(t.nb[c]) == 3 else HEAVY_T0[zc]) + 5.0 * heavy
    else:
        return None, f
    if ring_() in (3, 4) or (ring_() == 5 and zc in (6, 7, 8) and not (zc == 8 and len(t.nb[c]) >= 3)):
        t0 = {3: 82.0, 4: 96.0, 5: {6: 109.0, 7: 104.0, 8: 104.5}.get(zc)}[ring_()]
    # at a 3-ring atom, an angle outside rings opens by 4 degrees when exactly one end is a 3-ring atom and the other
    # shares no ring with the centre
    in3 = [v in t.ring3 for v in (a, b)]
    if not ring_() and c in t.ring3 and sum(in3) == 1 and not t.ring(b if in3[0] else a):
        t0 += 4.0
    return t0, f


def angle_params(t):
    """Per angle (c, a, b, t0 in rad, k); t0 None where no rule is known."""
    z = t.z
    cls = [atom_class(t, i) for i in range(len(z))]  # per atom of the cell (images alike)
    par = PAR._get()
    # the parameters and charges as plain floats (the same doubles)
    p7 = [float(par[v - 1, 7]) for v in z]
    p8 = [float(par[v - 1, 8]) for v in z]
    qt = [float(v) for v in t.qt]
    periodic = t.cell is not None
    vecs = t.__dict__.setdefault("_angle_vecs", {}) if periodic else None
    out = []
    for c in range(len(z)):
        # metal centres: none beyond six neighbours, none between two bonded neighbours of which one is a metal and none
        # below 60 degrees at the input geometry; other centres: none through an eta-bonded metal
        metal = c in t.eta or bool(cls[c])
        nb = ([] if len(t.nb[c]) > 6 else t.nb[c]) if metal else t.lig[c]
        n = len(nb)
        if metal or periodic:
            # bond vectors and lengths of the neighbours, once per centre; a membership test sees the atom, not its
            # image
            uu = [t.x[a] - t.x[c] for a in nb]
            ln = [np.linalg.norm(u) for u in uu]
        if periodic:
            # the bond vectors of this centre for hyper_angle (the same arithmetic, evaluated when asked)
            vecs[c] = ({(int(a), getattr(a, "s", None)): ia for ia, a in enumerate(nb)}, uu, ln)
        for ia, a in enumerate(nb):
            for ib, b in enumerate(nb[ia + 1:], ia + 1):
                if metal:
                    u, v = uu[ia], uu[ib]
                    bonded = b in list.__getitem__(t.nb, int(a)) and (cls[int(a)] or cls[int(b)])
                    if bonded or u @ v > 0.5 * ln[ia] * ln[ib]:
                        continue
                t0, f = angle_t0(t, c, a, b)
                if mclass(z[c]) == 3:
                    f *= 0.3 ** sum(c not in t.lig[v] for v in (a, b))
                if not (mclass(z[c]) and t.hyb[c] == 0) and mclass(z[c]) != 3:
                    nh = sum(z[v] == 1 and len(t.nb[v]) == 2 for v in (a, b))  # ends at an H with two neighbours
                    f *= 0.6 ** (min(nh, 1) if z[c] in HALOGEN else nh)  # once at a halogen centre
                if t0 is None:
                    out.append((c, a, b, None, None))
                    continue
                t0 = math.radians(t0)
                cq = 1.35 if cls[c] or cls[int(a)] or cls[int(b)] else 0.54
                a_, b_ = int(a), int(b)
                k = p7[c] * p8[a_] * p8[b_] * (1.0 + cq * qt[c] * (qt[a_] + qt[b_])) \
                    * (1.0 - 2.36 / n ** 2) * (1.0 - 0.5 * math.exp(-0.64 * (t0 - math.pi) ** 2))
                if k * f != 0.0:
                    out.append((c, a, b, t0, k * f))
    return out


def _pair_tables():
    """Twofold or threefold forms (TORS_NP) and pair factors (PAIR_P) of sp3-sp3 bonds between pnictogens (N, P, As,
    Sb) and chalcogens (O, S, Se, Te): pnictogen pairs (3, 60) with 3; mixed pairs (2, 90) with 1, or 20 when both
    atoms are below the second row; chalcogen pairs (2, 90) with 5, or 25 when both are below the second row."""
    pn, ch = (7, 15, 33, 51), (8, 16, 34, 52)
    pair, form = {}, {}
    for a in pn + ch:
        for b in pn + ch:
            if a > b:
                continue
            heavy = a not in (7, 8) and b not in (7, 8)
            if a in pn and b in pn:
                form[(a, b)], pair[(a, b)] = (3, 60.0), 3.0
            elif a in pn or b in pn:
                form[(a, b)], pair[(a, b)] = (2, 90.0), 20.0 if heavy else 1.0
            else:
                form[(a, b)], pair[(a, b)] = (2, 90.0), 25.0 if heavy else 5.0
    return {k: v for k, v in pair.items() if v != 1.0}, form


PAIR_P, TORS_NP = _pair_tables()
#: central elements of the extra one-fold torsion term
EXTRA_N1_ELEM = frozenset({6, 7, 8, 14, 15, 16, 32, 33, 34, 50, 51, 52, 82})


def extra_n1(cz, boron=False):
    """Factor of the extra one-fold torsion about a bond of the element pair cz (None without one): -2 with an O,
    0.7 with an N, -0.9 otherwise, divided by the pair factor; boron counts when both bond atoms are sp3 (boron)."""
    if not set(cz) <= (EXTRA_N1_ELEM | {5} if boron else EXTRA_N1_ELEM):
        return None
    return (-2.0 if 8 in cz else 0.7 if 7 in cz else -0.9) / PAIR_P.get(cz, 1.0)


RING_TORS = {4: {0: (6, 30.0, 0.3), 1: (1, 0.0, 1.0), 2: (6, 30.0, 1.0)},
             5: {0: (6, 30.0, 0.3), 1: (1, 0.0, 1.0), 2: (6, 30.0, 1.5)},
             6: {0: (6, 30.0, 0.3), 1: (1, 0.0, 1.0), 2: (3, 60.0, 5.7)}}
GROUP = {1: 1, 6: 14, 7: 15, 8: 16, 9: 17, 14: 14, 15: 15, 16: 16, 17: 17}


CONGENER = {83: 15, 84: 16}


def g_of_p(p, slope=1.0, const=0.55):
    """Pi-torsion factor of the bond order: 0.55 + 11.8 P s, switched on by s = exp(-2.5 (1.24 - P)^14)."""
    return const + 11.8 * slope * p * math.exp(-2.5 * (1.24 - p) ** 14)


def pi_end_slope(t, a):
    """Slope factor of g(P) for an end atom: 1.3 for non-pi atoms beyond neon."""
    return 1.0 if t.z[a] <= 10 or is_pi(t, a) else 1.3


def pi_end_scale(t, a):
    """Overall factor of g(P) for an end atom: 1.5 for a hypervalent non-pi atom beyond neon."""
    return 1.5 if t.z[a] > 10 and not is_pi(t, a) and t.hyb[a] == 5 else 1.0


def _tors_factors(t):
    """Per atom: the end factor (neighbour count and non-pi N), whether it is a hypervalent halogen, Si or Se, and the
    centre factor of its H neighbours, as :func:`tors_base` multiplies them."""
    z = t.z
    # an atom without neighbours is never a torsion end
    end = [len(list.__getitem__(t.lig, a)) ** -0.14 * (0.5 if z[a] == 7 and not is_pi(t, a) else 1.0)
           if list.__getitem__(t.lig, a) else 0.0 for a in range(len(z))]
    hv = [z[a] in (9, 14, 17, 34, 35, 53) and t.hyb[a] == 5 for a in range(len(z))]  # hypervalent halogen (not At)
    ctr = [((min(3, sum(z[b] == 1 for b in list.__getitem__(t.nb, a))) + 1) / 4) ** 0.07 for a in range(len(z))]
    # the element parameters, charges and congener elements of every atom as plain floats and ints (the same doubles)
    par = PAR._get()
    p9 = [float(par[v - 1, 9]) for v in z]
    p10 = [float(par[v - 1, 10]) for v in z]
    q = [float(v) for v in t.qt]
    cg = [CONGENER.get(v, v) for v in z]
    return end, hv, ctr, p9, p10, q, cg


def tors_base(t, i, j, k, l, pair=True):
    # neighbour counts do not depend on the image; the per-atom factors are kept for one torsion_rows call
    fac = t.__dict__.get("_tors_fac") or _tors_factors(t)
    end, hv, ctr, p9, p10, q, cg = fac
    i_, j_, k_, l_ = int(i), int(j), int(k), int(l)
    v = p9[j_] * p9[k_] * p10[i_] * p10[l_] * (1 + 12 * abs(q[j_] * q[k_]))
    v *= end[i_]
    v *= end[l_]
    if hv[i_] or hv[l_]:
        v *= 1.5
    v *= ctr[j_]
    v *= ctr[k_]
    if pair:
        a, b = cg[j_], cg[k_]
        v *= 1.2141948844 * PAIR_P.get((a, b) if a <= b else (b, a), 1.0)
    return v


def tpi(t, a):
    """Pi flag as the torsion rules see it: an S with three or more neighbours and two bonded O atoms on the same
    metal count as saturated."""
    flags = t.__dict__.get("_tpi")  # per atom, kept for one torsion_rows call
    if flags is not None:
        return flags[int(a)]
    return _tpi(t, a)


def _tpi(t, a):
    return is_pi(t, a) and not (t.z[a] == 16 and len(t.nb[a]) >= 3) and not (
        t.z[a] == 8 and any(t.z[j] == 8 and any(mclass(t.z[m]) for m in set(t.nb[a]) & set(t.nb[j])) for j in t.nb[a]))


#: torsions with a smaller force constant are dropped
TORS_KMIN = 1e-3


def lig_cycle(t, j, k):
    """True when the bond j-k lies on a cycle of the graph of mutual ligand neighbours."""
    if getattr(t, "_lig_graph", None) is None:
        t._lig_graph = nx.Graph([(a, b) for a in range(len(t.z)) for b in t.lig[a] if a < b and a in t.lig[b]])
    g = t._lig_graph
    if not g.has_edge(j, k):
        return False
    h = g.copy()
    h.remove_edge(j, k)
    return nx.has_path(h, j, k)


def mutual_triangle(t, j, k):
    """True when the bond j-k closes a three-membered ring of mutual ligand neighbours."""
    return any(v in t.lig[j] and j in t.lig[v] and v in t.lig[k] and k in t.lig[v] for v in t.nb[j] if v != k)


def extra_ok(t, i, j, k, l, cz, s5):
    """Whether a torsion takes the extra one-fold term (extra_n1): ends with neighbours of their own, no H, no d metal
    other than sp3 and no sp metal end, none at a hypervalent S; sp or sp2 ends take none (a pi S only when it is a
    torsion pi atom), sp3 ends with a pi flag (ethers, amines) keep it."""
    z = t.z
    if len(t.nb[i]) < 2 or len(t.nb[l]) < 2 or extra_n1(cz, t.hyb[j] == t.hyb[k] == 3) is None or s5:
        return False
    if any(z[v] == 1 or (mclass(z[v]) == 3 and t.hyb[v] != 3) or (mclass(z[v]) and t.hyb[v] == 1) for v in (i, l)):
        return False
    if any(z[c] == 7 and t.hyb[c] == 2 and is_pi(t, c) for c in (j, k)):  # about a bond to an sp2 pi N
        return False
    return not any(t.hyb[v] in (1, 2, -2) and (tpi(t, v) or z[v] != 16 or not is_pi(t, v)) for v in (i, l))


def torsion_rows(t):
    """Proper torsions (i, j, k, l, n, phi0 in degrees, k) in chain order i-j-k-l."""
    z = t.z
    extra = []
    cl = [atom_class(t, a) for a in range(len(z))]  # metal classes, hypervalent Sn, Pb, Bi and Po included
    out = []
    t._tors_fac = _tors_factors(t)
    t._tpi = [_tpi(t, a) for a in range(len(z))]
    for j, k in t.bond_pairs():
        mbond = bool(cl[j] or cl[k])  # an sp atom bonded to a metal does not block that bond
        # a bent N between a non-metal and a main-group metal does not block the bond to the non-metal
        # a two-neighbour halogen blocks its bonds except those to a metal or a hypervalent atom
        linear = any(t.hyb[c] == 1 and (t.hyb[j + k - c] != 5 and not mbond if z[c] in HALOGEN else
                                        ((not cl[c] and not mbond) or z[c] in (4, 12, 20, 38)))
                     and not (z[c] == 7 and len(t.nb[c]) == 2 and z[j + k - c] not in METAL
                              and any(z[a] in METAL and not cl[a] for a in t.nb[c])
                              and _angle_deg(t.x, c, *t.nb[c]) <= 170.0) for c in (j, k))
        if len(t.nb[j]) < 2 or len(t.nb[k]) < 2 or linear:
            continue
        if cl[j] or cl[k]:
            # none about metal-metal bonds, eta bonds of d metals or at a d metal with five or more neighbours; ends
            # skip eta metals and metal-metal bonds; an eta bond of an sp3 metal is an sp3-sp2 torsion
            eta = j not in t.lig[k] or k not in t.lig[j]
            sblock = cl[j] == cl[k] == 1  # bonds between group 1 and 2 metals keep their torsions
            if (cl[j] and cl[k] and not sblock) \
                    or any(cl[c] == 3 and (_metal_hyb(z[c], len(t.nb[c])) == 0 or eta) for c in (j, k)):
                continue
            nper = 1 if any(t.hyb[c] in (1, 2, 5) or (cl[c] and t.hyb[c] == 0) for c in (j, k)) else 3
            f = 1.0
            if eta and 3 in (t.hyb[j], t.hyb[k]):
                nper, f = 3, 0.5
            elif ({j, k} & t.eta and not mutual_triangle(t, j, k) and ({t.hyb[j], t.hyb[k]} == {2, 3}
                                      or (t.hyb[j] == t.hyb[k] == 3 and lig_cycle(t, j, k)))) \
                    or all(t.hyb[c] == 3 if cl[c] else (t.hyb[c] == 2 and is_pi(t, c)) for c in (j, k)):
                # at a metal with an eta bond: sp3-sp2 form, and sp3-sp3 on a cycle of the ligand graph; an sp3 metal
                # and an sp2 pi atom take the same form
                nper, f = 3, 0.2 if z[j if t.hyb[j] == 2 else k] == 7 else 0.5
            elif any(cl[c] == 3 and t.hyb[c] == 3 for c in (j, k)) \
                    and any(z[c] in (8, 16) and t.hyb[c] == 3 and is_pi(t, c) for c in (j, k)):
                nper, f = 3, 0.5  # a lone-pair pi O or S on an sp3 transition metal
            # the force constants of tors_base(pair=False), by the same products in the same order
            end, hv, ctr, p9, p10, q, _ = t._tors_fac
            j_, k_ = int(j), int(k)
            pa = p9[j_] * p9[k_]
            qq = 1 + 12 * abs(q[j_] * q[k_])
            cj, ck, pref = ctr[j_], ctr[k_], 1.2141948844 * f
            lk = t.lig[k]
            for i in t.lig[j]:
                if _same(i, k) or (cl[i] and cl[j] and not cl[i] == cl[j] == 1):
                    continue
                i_ = int(i)
                pi_, ei, hi = pa * p10[i_], end[i_], hv[i_]
                for l in lk:
                    if _same(l, j) or _same(i, l) or (cl[l] and cl[k] and not cl[l] == cl[k] == 1):
                        continue
                    l_ = int(l)
                    v = pi_ * p10[l_] * qq
                    v *= ei
                    v *= end[l_]
                    if hi or hv[l_]:
                        v *= 1.5
                    v *= cj
                    v *= ck
                    out.append((i, j, k, l, nper, 180.0, pref * v))
            continue
        cz = tuple(sorted(CONGENER.get(z[c], z[c]) for c in (j, k)))
        ring = inring = t.ring(j, k, tors=True)
        # larger rings, and bonds between two saturated equal heteroatoms (O-O, N-N, S-S, P-P), take the open-chain
        # forms (without the extra one-fold term inside a ring)
        if ring > 6 or (cz in TORS_NP and (cz[0] == cz[1] or t.hyb[j] == t.hyb[k] == 3)
                        and not (is_pi(t, j) or is_pi(t, k))):
            ring = 0
        pipi = is_pi(t, j) and is_pi(t, k) and t.pisys[j] == t.pisys[k] and (min(j, k), max(j, k)) in t.P
        pi3 = pipi and t.hyb[j] == t.hyb[k] == 3  # a pi bond between sp3 atoms: open-chain form scaled by g(P)
        pipi = pipi and not pi3
        for i in t.lig[j]:
            for l in t.lig[k]:
                if _holds((j, k), i) or _holds((j, k), l) or _same(i, l):
                    continue
                if pipi:
                    ends = pi_end_scale(t, i) * pi_end_scale(t, l)
                    amide5 = ring == 5 and any(z[c] == 7 and t.hyb[c] == -3 for c in (j, k))  # a 5-ring lactam bond
                    kk = tors_base(t, i, j, k, l, pair=False) * 1.2141948844 \
                        * g_of_p(t.bo(j, k), pi_end_slope(t, i) * pi_end_slope(t, l), 2.75 if amide5 else 0.55) * ends
                    out.append((i, j, k, l, 2, 0.0 if ring else 180.0, kk))
                    continue
                base = tors_base(t, i, j, k, l)
                if z[i] == 7 and z[l] == 7 and ring and not (is_pi(t, i) or is_pi(t, l)):
                    base *= 2.0
                s5 = any(t.hyb[v] == 5 and z[v] == 16 for v in (i, l))  # an end at a hypervalent S
                if s5:
                    base *= 1.5
                if ring:
                    # a torsion inside rings of one size takes that size with both ends; ends in rings of other sizes
                    # or in different rings count one end
                    sizes, size = t.ring_sizes(i, j, k, l), ring
                    # both ends in a ring of one size; in a periodic cell not when the torsion runs over two atoms only
                    # (a bond to its own image with both ends on one atom)
                    if ring > 3 and len(sizes) == 1 and not (tpi(t, i) or tpi(t, l)) \
                            and len({int(i), int(j), int(k), int(l)}) > 2:
                        ends, size = 2, sizes[0]
                    else:
                        ends = 1 if any(t.ring(v, j, k, tors=True) for v in (i, l)) else 0
                    if not ends and any(len(t.nb[v]) > 1 for v in (i, l)):  # an exocyclic end with its own neighbours
                        ends = 1
                    if t.mo & {i, j, k, l} or t.mring & {j, k}:
                        ends = 1
                    n_, p0, f_ = RING_TORS[size][ends] if 3 < size <= 6 else \
                        (1, 0.0, 1.0 if tpi(t, i) or tpi(t, l) else 0.3)
                else:
                    f_ = 1.0
                if 5 in (t.hyb[j], t.hyb[k]) or any(z[c] == 9 and len(t.nb[c]) >= 3 for c in (j, k)):
                    if z[i] == 7 and z[l] == 7 and not ring and not (is_pi(t, i) or is_pi(t, l)):
                        base *= 2.0  # N...N ends without pi
                    if any(t.hyb[c] == 5 and is_pi(t, c) for c in (j, k)) and 3 in (t.hyb[j], t.hyb[k]):
                        out.append((i, j, k, l, 3, 180.0, 0.1 * base / PAIR_P.get(cz, 1.0)))  # pi Cl to an sp3 atom
                    elif 5 not in (t.hyb[j], t.hyb[k]):  # between two F with three neighbours
                        # in a ring the minimum is at 0, and in a three-ring two H ends take 0.3
                        f3 = 0.3 if inring == 3 and z[i] == 1 and z[l] == 1 else 1.0
                        out.append((i, j, k, l, 1, 0.0 if 3 <= inring <= 6 else 180.0,
                                    f3 * base / PAIR_P.get(cz, 1.0)))
                    else:
                        p0 = 0.0 if 3 <= inring <= 6 else 180.0  # in a ring the one-fold term has its minimum at 0
                        out.append((i, j, k, l, 1, p0, 0.2 * base / PAIR_P.get(cz, 1.0)))
                    continue
                if tpi(t, j) or tpi(t, k):
                    a, b = (j, k) if tpi(t, j) else (k, j)
                    za, zb = z[a], z[b]
                    if za == 6 and zb == 6:
                        oxo = sum(z[v] == 8 and len(t.nb[v]) == 1 for v in t.nb[a]) == 1
                        f_ *= (2.0 if ring else 1.0) * (0.65 if oxo else 0.5)
                    elif zb == 6 and za == 7:
                        f_ *= (0.26 if t.hyb[a] == -3 else 0.2) * (5.0 if ring else 1.0)
                    elif zb in (6, 14) and za in (8, 16):
                        f_ *= 1.0 if ring else 0.5
                    elif ring and za == 6 and zb == 7 and len(t.nb[b]) == 4 and any(mclass(z[v]) for v in t.nb[b]):
                        n_, p0 = 2, 0.0
                    elif za == 6 and zb in (7, 14, 15, 32, 33, 34, 50, 51, 52, 83, 84) and t.hyb[b] == 3:
                        f_ *= 1.0 if ring and zb in (34, 52) else 0.5
                    elif za == 5 and zb == 6 and t.hyb[b] == 3:
                        f_ *= 0.5
                    elif za == 7 and zb == 7 and t.hyb[b] == 3:  # a hydrazide keeps the N-N form, in a ring too
                        out.append((i, j, k, l, *TORS_NP[cz], base))
                        continue
                    elif za == 7 and zb in (34, 52, 84) and t.hyb[b] == 3:
                        f_ *= 0.2
                    elif za in (7, 8) and zb == 5 and t.hyb[b] == 3 and ring:  # a pi N or O on a four-neighbour ring B
                        out.append((i, j, k, l, 1, 0.0, base))
                        continue
                    elif za in (7, 8) and zb == 8 and t.hyb[b] == 3 and not ring:  # N-O and O-O with a pi end
                        out.append((i, j, k, l, *TORS_NP[cz], base))
                        if extra_ok(t, i, j, k, l, cz, s5):
                            extra.append(len(out))
                            out.append((i, j, k, l, 1, 180.0, extra_n1(cz, t.hyb[j] == t.hyb[k] == 3) * base))
                        continue
                    elif za == 8 and zb in (15, 33) and t.hyb[b] == 3 and not ring:  # aryl and enol phosphate esters
                        out.append((i, j, k, l, *TORS_NP[cz], base))
                        continue
                    elif not (zb == 84 and t.hyb[b] == 2):  # a flat Po conjugates like a pi atom
                        f_ = float("nan")
                    if ring:
                        out.append((i, j, k, l, n_, p0, base * f_))
                    else:
                        out.append((i, j, k, l, 2 if t.hyb[b] == 2 else 3, 180.0, base * f_))
                        # an N conjugated to a pi system bonded to an sp3 C keeps the extra one-fold term of its
                        # saturated ends, on the force constant of the N (f / 0.2)
                        if za == 7 and zb == 6 and t.hyb[b] == 3 and extra_ok(t, i, j, k, l, cz, s5):
                            extra.append(len(out))
                            out.append((i, j, k, l, 1, 180.0, extra_n1(cz) * base * f_ / 0.2))
                    continue
                if ring:
                    if any(t.hyb[c] == 2 and z[c] == 7 and not is_pi(t, c) for c in (j, k)):
                        f_ /= PAIR_P.get(cz, 1.0)  # a flat N carries no pair factor
                    out.append((i, j, k, l, n_, p0, f_ * base))
                    continue
                nper, p0 = TORS_NP.get(cz, (3, 180.0))
                flats = [(z[c] in (5, 13, 15, 33, 51, 50, 82, 83, 84) or (z[c] in (6, 7, 8) and not is_pi(t, c)))
                         and t.hyb[c] == 2 for c in (j, k)]
                flat = any(flats)
                if 5 in (t.hyb[j], t.hyb[k]) or flat:
                    two = cz == (5, 5) or (all(flats) and {z[j], z[k]} & {50, 82, 83, 84}) \
                        or (cz == (7, 8) and any(z[c] == 7 and t.hyb[c] == 3 for c in (j, k)))  # flat O on an sp3 N
                    nper, p0 = (2, 180.0) if two else (1, 180.0)
                    base /= PAIR_P.get(cz, 1.0)  # flat torsions carry no pair factor
                if z[i] == 7 and z[l] == 7 and not (is_pi(t, i) and is_pi(t, l)):
                    base *= 2.0
                sox = 0.5 if nper == 3 and any(z[c] == 16 and is_pi(t, c) for c in (j, k)) else 1.0
                if pi3:  # the pi factor with its switch divided by the pair factor
                    sox *= g_of_p(t.bo(j, k), pi_end_slope(t, i) * pi_end_slope(t, l) / PAIR_P.get(cz, 1.0)) \
                        * pi_end_scale(t, i) * pi_end_scale(t, l)
                out.append((i, j, k, l, nper, p0, base * sox))
                if not flat and not 3 <= inring <= 6 and extra_ok(t, i, j, k, l, cz, s5):
                    extra.append(len(out))
                    out.append((i, j, k, l, 1, 180.0, extra_n1(cz, t.hyb[j] == t.hyb[k] == 3) * base))
    # none through an angle beyond 170 degrees at the input geometry; the force constant cut leaves the extra terms
    keep = set(extra)
    t._tors_fac = t._tpi = None
    return [r for m, r in enumerate(out) if (m in keep or not abs(r[6]) < TORS_KMIN) and _in_reach(r[:4])
            and not _near_linear(t, *r[:4])]


def _in_reach(atoms):
    """Whether a periodic torsion is within the searched images: the shifts of its atoms relative to the central atom
    of lower index stay within one cell in every direction (always true without a cell)."""
    shifts = [getattr(a, "s", None) for a in atoms]
    if shifts[0] is None and shifts[1] is None and shifts[2] is None and shifts[3] is None:
        return True
    shifts = [s or (0, 0, 0) for s in shifts]
    r0, r1, r2 = shifts[1] if int(atoms[1]) < int(atoms[2]) else shifts[2]
    for s0, s1, s2 in shifts:
        if abs(s0 - r0) > 1 or abs(s1 - r1) > 1 or abs(s2 - r2) > 1:
            return False
    return True


def _near_linear(t, i, j, k, l):
    """True when the angle i-j-k or j-k-l of a torsion reaches 170 degrees (within round-off)."""
    memo = t.__dict__.setdefault("_cos_memo", {})

    def cos(a, c, b):
        key = (int(a), getattr(a, "s", None), int(c), getattr(c, "s", None), int(b), getattr(b, "s", None))
        if key not in memo:
            u, w = t.x[a] - t.x[c], t.x[b] - t.x[c]
            memo[key] = float(u @ w) / float(np.linalg.norm(u) * np.linalg.norm(w))
        return memo[key]
    return min(cos(i, j, k), cos(j, k, l)) < LINEAR_COS + 1e-9


def inversion_rows(t):
    """Inversions (centre, j, k, l, n, w0 in degrees, k); neighbours ordered by distance from the centre, so l is the
    farthest one (the out-of-plane bond of the energy); ties go in reverse list order."""
    z, qt = t.z, t.qt
    out = []
    for c in range(len(z)):
        nbs = t.lig[c]
        if len(nbs) != 3:
            continue
        # farthest first in list order, then reversed: equal distances come in reverse list order
        j, k, l = sorted(nbs, key=lambda v: float(np.linalg.norm(t.x[v] - t.x[c])), reverse=True)[::-1]
        if is_pi(t, c) and (t.hyb[c] == 2 or z[c] in (7, 8, 16) or z[c] in HALOGEN):
            fq = 1 + 5 * qt[c]
            if z[c] == 7 and any(z[a] == 8 or (z[a] == 16 and t.hyb[a] == 5) for a in nbs):  # N-O, sulfonamide N
                kk = 10.5 * fq
            else:
                a = 1.05 if z[c] in (8, 16) or z[c] in HALOGEN else (39.9 if any(z[v] in (8, 16) for v in nbs) else
                                             (10.5 if z[c] == 6 and any(z[v] in HALOGEN for v in nbs) else 1.05))
                kk = a * (1 - 0.5 * sum(t.bo(c, v) for v in nbs)) * fq
            out.append((c, j, k, l, 0, 0.0, kk))
        elif z[c] == 7:
            kk = sum(0.6 * math.sqrt(VALENCE[z[v]]) for v in nbs)
            out.append((c, j, k, l, -1, 80.0, kk))
    return out


def _valence(z):
    """Valence electrons of the repulsion prefactor: s and p electrons of main-group atoms, the group number of d metals
    (lanthanides 3)."""
    start = max(v for v in (1, 3, 11, 19, 37, 55) if v <= z)
    k = z - start + 1
    if start in (19, 37):
        return k if k <= 12 else k - 10
    if start == 55:
        return k if k <= 2 else (3 if z <= 71 else (z - 68 if z <= 80 else z - 78))
    return k


VALENCE = {z: _valence(z) for z in range(1, 87)}


def near_hops(t, eta=True):
    """Per atom, the atoms within three bonds with their bond counts (cached on the topology); when eta is False the
    paths avoid eta bonds, but bonded atoms stay one bond apart. In a periodic structure the keys are the images
    ``(atom, shift)`` and the paths run through the images."""
    key = "_near" if eta else "_near_lig"
    if t.cell is not None:
        return _near_hops_images(t, eta, key)
    if getattr(t, key, None) is None:
        nb = t.nb if eta else [[j for j in t.lig[i] if i in t.lig[j]] for i in range(len(t.z))]
        out = []
        for i in range(len(t.z)):
            seen, shell = {i: 0}, [i]
            for depth in (1, 2, 3):
                nxt = []
                for a in shell:
                    for b in nb[a]:
                        if b not in seen:
                            seen[b] = depth
                            nxt.append(b)
                shell = nxt
            if not eta:
                seen.update((j, 1) for j in t.nb[i])
            out.append(seen)
        setattr(t, key, out)
    return getattr(t, key)


def _near_hops_images(t, eta, key):
    if getattr(t, key, None) is None:
        n = len(t.z)
        if eta:
            base = [list.__getitem__(t.nb, i) for i in range(n)]
        else:
            # mutual ligand neighbours: the test does not depend on the image of the atom
            base = [[j for j in list.__getitem__(t.lig, i) if _holds(t.lig[j], Img(i))] for i in range(n)]
        nbr = [[(int(j), j.s) for j in v] for v in base]
        out = []
        for i in range(n):
            seen, shell = {(i, (0, 0, 0)): 0}, [(i, (0, 0, 0))]
            for depth in (1, 2, 3):
                nxt = []
                for a, sa in shell:
                    for b, sb in nbr[a]:
                        k = (b, (sb[0] + sa[0], sb[1] + sa[1], sb[2] + sa[2]))
                        if k not in seen:
                            seen[k] = depth
                            nxt.append(k)
                shell = nxt
            if not eta:
                seen.update(((int(j), j.s), 1) for j in t.nb[i])
            out.append(seen)
        setattr(t, key, out)
    return getattr(t, key)


def shifts_within(cell, cut):
    """Lattice shifts of the images that can hold an atom within ``cut`` (Angstrom) of an atom of the cell."""
    cell = np.asarray(cell, dtype=float)
    h = np.abs(np.linalg.det(cell)) / np.array([np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3]))
                                                for k in range(3)])
    m = [int(math.ceil(cut / v)) + 1 for v in h]
    return [(a, b, c) for a in range(-m[0], m[0] + 1) for b in range(-m[1], m[1] + 1) for c in range(-m[2], m[2] + 1)]


def repulsion_reach(t):
    """Largest distance (Angstrom) at which a repulsion pair can still enter: its exponent stays below 60 for the
    smallest exponent the pair parameters allow."""
    z = np.asarray(t.z)
    nn = np.array([len(v) for v in t.lig])
    expo = PAR[z - 1, 6] * (1.0 - 0.127 / (1.0 + nn ** 2)) * (1.0 + 0.348 * t.qt)
    bonded = np.sqrt(PAR[z - 1, 5])
    m = min(float(expo.min()) * 0.445332, float(bonded.min()) ** 2)
    return (60.0 / m) ** (2.0 / 3.0) * BOHR


def repulsion_rows(t):
    """Pairs ``(i, j)`` (i < j, all of them), prefactors c and exponents m of c / r exp(-m r^1.5), as arrays."""
    z = np.asarray(t.z)
    n = len(z)
    nn = np.array([len(v) for v in t.lig])
    expo = PAR[z - 1, 6] * (1.0 - 0.127 / (1.0 + nn ** 2)) * (1.0 + 0.348 * t.qt)
    val = np.array([VALENCE[int(v)] for v in z], dtype=float)
    iu, ju = np.triu_indices(n, 1)
    zz = val[iu] * val[ju]
    base = np.sqrt(expo[iu] * expo[ju])
    zi, zj = z[iu], z[ju]
    both = (zi == 1) & (zj == 1)
    one = (zi == 1) != (zj == 1)
    other = np.where(zi == 1, zj, zi)
    fac = np.where(both, 0.629, 1.0)
    fac = np.where(one & np.isin(other, sorted(H_METAL)), 0.85, fac)
    fac = np.where(one & (other == 6), 0.91, fac)
    fac = np.where(one & (other == 8), 1.04, fac)
    c, m = 0.427 * zz, base * fac
    for i, near in enumerate(near_hops(t, eta=False)):
        for j, d in near.items():
            if j <= i:
                continue
            k = i * n - i * (i + 1) // 2 + j - i - 1
            if d == 1:
                c[k], m[k] = 1.7583 * zz[k], math.sqrt(PAR[z[i] - 1, 5] * PAR[z[j] - 1, 5])
            elif both[k]:
                m[k] = base[k] * (0.917082 if d == 2 else 0.445332)
    return np.stack([iu, ju], axis=1), c, m


#: squared radii (Bohr^2) of the two translation lists of the bonded three-body term (the second also lists the
#: repulsion images): shifts of third atoms are looked
#: up in the first and the lattice vector at the same position of the second is the one applied
TRANS_LOOKUP2, TRANS_ENERGY2 = 450.0, 500.0


def translation_list(cell, r2, cap=False):
    """Lattice shifts of a translation list with radius ``sqrt(r2)`` (Bohr) for a cell (rows, Angstrom): shifts in the
    box ``|n_i| <= ceil(R / h_i)`` (``h_i`` the cell heights), ordered by shell ``s = max |n_i|`` and within a shell
    with the third index outermost and the first innermost; shells of two and more keep ``|T| (s - 1) / s < R``.
    With ``cap`` the shells stop at ``ceil(R / a)``, ``a`` the shortest lattice vector."""
    key = (np.asarray(cell, dtype=float).tobytes(), float(r2), bool(cap))
    hit = _TRANSLATIONS.get(key)
    if hit is None:
        if len(_TRANSLATIONS) > 64:
            _TRANSLATIONS.clear()
        hit = _TRANSLATIONS[key] = _translation_list(cell, r2, cap)
    return list(hit)


#: translation lists already built, per cell, radius and cap
_TRANSLATIONS = {}


def _translation_list(cell, r2, cap):
    cell = np.asarray(cell, dtype=float) / BOHR
    r = math.sqrt(r2)
    h = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    k = [max(1, math.ceil(r / x)) for x in h]
    box = [(a, b, c) for c in range(-k[2], k[2] + 1) for b in range(-k[1], k[1] + 1) for a in range(-k[0], k[0] + 1)]
    top = max(k) if not cap else min(max(k), max(1, math.ceil(r / float(np.linalg.norm(cell, axis=1).min()))))
    out = []
    for sh in range(top + 1):
        for p in box:
            if max(abs(v) for v in p) != sh:
                continue
            if sh < 2 or float(np.linalg.norm(np.asarray(p) @ cell)) * (sh - 1) / sh < r:
                out.append(p)
    return out


def _atm_shift(t, s):
    """The lattice shift applied to a third atom whose composed shift is ``s``: the shift at the position of ``s`` in
    the lookup list taken from the energy list (the same shift whenever ``s`` is next to the cell)."""
    if max(abs(v) for v in s) <= 1:
        return s
    lists = getattr(t, "_atm_lists", None)
    if lists is None:
        look = translation_list(t.cell, TRANS_LOOKUP2)
        lists = t._atm_lists = ({p: q for q, p in enumerate(look)}, translation_list(t.cell, TRANS_ENERGY2))
    pos, energy = lists
    return energy[pos[s]] if s in pos and pos[s] < len(energy) else None


def batm_rows(t):
    """Triples (i, j, k, coefficient) of the bonded three-body term: 1-4 pairs i, j counted without eta bonds and the
    ligand neighbours k of i and of j (a neighbour of both counts twice)."""
    z = np.asarray(t.z)
    hops = near_hops(t, eta=False)
    aa = np.where(z == 1, 0.25, z.astype(float)) * (1.0 - 3.0 * t.qt)
    if t.cell is not None:
        out = []
        # the ligand lists in index order (a stable sort: an image list sorts as the list of its atom)
        lig = [sorted(list.__getitem__(t.lig, i)) for i in range(len(z))]
        third = {}
        av = aa.tolist()  # the same doubles, without numpy scalars
        imgs = {}  # one image object per atom and shift (the rows only read them)

        def img(a, sh):
            key = (a, sh)
            v = imgs.get(key)
            if v is None:
                v = imgs[key] = Img(a, sh)
            return v
        for i in range(len(z)):
            ii = Img(i)
            for (j, s), d in sorted(hops[i].items()):
                # 1-4 pairs of the bond counts xtb keeps: images next to the cell, an atom and its own images at 0; the
                # atom in the cell is the one with the larger index, which decides the shifts the third atoms take
                if d == 3 and j < i and -1 <= s[0] <= 1 and -1 <= s[1] <= 1 and -1 <= s[2] <= 1:
                    jj = img(j, s)
                    ks = list(lig[i])
                    for k in lig[j]:
                        ks_ = (k.s[0] + s[0], k.s[1] + s[1], k.s[2] + s[2])
                        if ks_ not in third:
                            third[ks_] = _atm_shift(t, ks_)
                        if third[ks_] is not None:
                            ks.append(img(int(k), third[ks_]))
                    cij = -0.3 * av[i] * av[j]
                    out += [(ii, jj, k, cij * av[k]) for k in ks]
        return out
    return [(i, j, k, -0.3 * aa[i] * aa[j] * aa[k]) for i in range(len(z)) for j in sorted(hops[i])
            if j > i and hops[i][j] == 3 for k in sorted(t.lig[i]) + sorted(t.lig[j])]


XB_X = {15, 16, 17, 33, 34, 35, 51, 52, 53}
XB_C = {15: 1.0, 16: 1.0, 17: 0.5, 33: 1.2, 34: 1.2, 35: 0.9, 51: 1.2, 52: 1.2, 53: 1.2}


XB_CARBON_CUT = 0.05


def hb_neighbours(t):
    """Neighbour sets of the hydrogen- and halogen-bond terms: ligand atoms do not count an eta-bonded metal."""
    return [set(v) for v in t.lig]


#: Squared distances (Bohr^2) below which a hydrogen bond (donor to acceptor), a loose hydrogen bond (sum over the
#: three sides) and a halogen bond (donor to acceptor) enter the energy.
HB_CUT, HB_CUT3, XB_CUT = 250.0, 450.0, 450.0


def _sq_dist(x, rows, cols):
    """Squared distances ``(len(rows), len(cols))`` summed per coordinate."""
    return sum((x[rows, c, None] - x[None, cols, c]) ** 2 for c in range(3))


def hbb_list(t, x=None, cut=HB_CUT):
    """Predicted hydrogen-bond triples (donor A bonded to H, acceptor B, H) at the positions ``x`` (Bohr, the
    topology's by default) with ``r_AB^2`` below ``cut``."""
    z, nb, q = t.z, hb_neighbours(t), t.qt
    x = np.asarray(t.x / BOHR if x is None else x)
    act = np.array([hb_active(z, nb, q, i) for i in range(len(z))])
    on = np.flatnonzero(act)
    out = []
    for h in range(len(z)):
        if z[h] != 1 or len(nb[h]) != 1 or q[h] <= HB_H_CUT:
            continue
        a = next(iter(nb[h]))
        if not act[a]:
            continue
        ok = (act[on] + act[a] >= 3) & (_sq_dist(x, [a], on)[0] < cut)
        out.extend((a, int(b), h) for b in on[ok] if b != a and b not in nb[a])
    return out


def xb_rows(t, x=None, cut=XB_CUT):
    """Predicted halogen-bond rows (D, Y, X) at the positions ``x`` (Bohr, the topology's by default) with
    ``r_DY^2`` below ``cut``."""
    z, nb, q = t.z, hb_neighbours(t), t.qt
    x = t.x / BOHR if x is None else x
    hops = near_hops(t, eta=False)  # bond counts over mutual ligand neighbours, as for the bonded ATM
    ys = [i for i in range(len(z)) if (z[i] == 6 and len(nb[i]) < 4 and q[i] < XB_CARBON_CUT)
          or (z[i] not in (6, 14) and z[i] in HB_STRONG | HB_WEAK)]  # heteroatom acceptors at any charge
    out = []
    for xa in range(len(z)):
        if z[xa] not in XB_X or (z[xa] == 16 and len(nb[xa]) > 2):
            continue
        for d in nb[xa]:
            for y in ys:
                if y in (xa, d) or hops[xa].get(y, 99) < 4 or float(np.sum((x[d] - x[y]) ** 2)) >= cut:
                    continue
                out.append((d, y, xa))
    return out


HB_RAD = {1: 0.32, 6: 0.75, 7: 0.71, 8: 0.64, 9: 0.60, 14: 1.14, 15: 1.09, 16: 1.04, 17: 1.00, 33: 1.20, 34: 1.18,
       35: 1.17, 51: 1.40, 52: 1.37, 53: 1.36}
HB_CA = {6: 0.15, 7: 0.32, 8: 0.30, 9: 0.30, 14: 0.0, 15: 0.30, 16: 0.30, 17: 0.50, 33: 0.0, 34: 0.0, 35: 0.50, 51: 0.0,
      52: 0.0, 53: 0.50}


HB_CB = {6: 0.80, 7: 1.68, 8: 0.67, 9: 0.52, 14: 4.0, 15: 3.5, 16: 2.0, 17: 1.5, 33: 3.5, 34: 2.0, 35: 1.5, 51: 3.5,
      52: 2.0, 53: 1.9}
HB_CB_O1 = {6: 0.68, 7: 0.47}           # O with one neighbour: carbonyl-type (C) and nitro-type (N) basicity


HB_STRONG, HB_WEAK = {7, 8, 9, 15, 16, 17, 35, 53}, {14, 33, 34, 51, 52}
#: H atoms on a first-row atom at or below this charge form no hydrogen bonds.
HB_H_CUT = 0.01
#: Hydrogen- and halogen-bond rows whose angular switch argument reaches this value contribute nothing.
HB_SWITCH_CUT = 15.0


def hb_active(z, nb, q, i):
    """0 inactive, 1 weak, 2 strong hydrogen-bond atom."""
    if z[i] == 6:
        return 2 if len(nb[i]) < 4 and q[i] < 0.1 else 0
    if z[i] in HB_STRONG:
        return 2 if q[i] < (0.1 if z[i] in (7, 8, 9) else 0.3) else 0
    return 1 if z[i] in HB_WEAK and q[i] < 0.3 else 0


def o1_class(z, nb, i):
    if z[i] == 8 and len(nb[i]) == 1:
        return z[next(iter(nb[i]))] if z[next(iter(nb[i]))] in HB_CB_O1 else 0
    return 0


def basicity(z, nb, i):
    c = o1_class(z, nb, i)
    return HB_CB_O1[c] if c else HB_CB[z[i]]


def acidity(z, nb, a):
    """Donor strength of the hydrogen-bond terms; the N-H of a secondary amide on a saturated carbon is scaled by
    0.8."""
    co = [j for j in nb[a] if z[j] == 6 and any(z[k] == 8 and len(nb[k]) == 1 for k in nb[j])]
    amide = z[a] == 7 and len(nb[a]) == 3 and sum(z[j] == 1 for j in nb[a]) == 1 and len(co) == 1 \
        and all(z[j] == 1 or (z[j] == 6 and (j in co or len(nb[j]) == 4)) for j in nb[a])
    return HB_CA[z[a]] * (0.8 if amide else 1.0)


def hb_rows(z, nb, q, x, cut=HB_CUT, cut3=HB_CUT3):
    """Membership of the two hydrogen-bond lists at the positions ``x`` (Bohr): bonded-H triples (donor, acceptor,
    H) with ``r_AB^2`` below ``cut`` and loose triples (A, B, H) that also have ``r_AB^2 + r_AH^2 + r_BH^2`` below
    ``cut3``, as two ``(R, 3)`` integer arrays."""
    n = len(z)
    act = [hb_active(z, nb, q, i) for i in range(n)]
    hs = [h for h in range(n) if z[h] == 1 and len(nb[h]) == 1 and (q[h] > HB_H_CUT or z[next(iter(nb[h]))] > 10)]
    bb, bl = [], []
    on = np.array([i for i in range(n) if act[i]], dtype=int)
    level = np.array([act[i] for i in on], dtype=int)
    close = _sq_dist(np.asarray(x), on, on) < cut
    close &= (level[:, None] + level[None, :] >= 3) & ~np.eye(len(on), dtype=bool)
    loose = [h for h in hs if not (z[next(iter(nb[h]))] == 6 and len(nb[next(iter(nb[h]))]) >= 4 and q[h] <= 0.06)]
    xx = np.asarray(x)
    lh = np.array(loose, dtype=np.int64)
    # the donor of every H with one neighbour
    dh = np.array([next(iter(nb[h])) for h in lh], dtype=np.int64)
    donors = {}
    for h in hs:
        donors.setdefault(next(iter(nb[h])), []).append(h)
    for a in range(len(on)):
        i, js = int(on[a]), on[np.flatnonzero(close[a])]
        bonded = np.array([int(j) in nb[i] for j in js], dtype=bool)
        if i in donors and len(js):
            # every free acceptor with every H of the donor, acceptor-major
            hi = np.asarray(donors[i], dtype=np.int64)
            jj = js[~bonded]
            bb.append(np.stack([np.full(len(jj) * len(hi), i), np.repeat(jj, len(hi)), np.tile(hi, len(jj))], 1))
        low, lbond = js[js < i], bonded[js < i]
        if len(low) and len(lh):
            # a loose triple needs r_AB^2 + r_AH^2 + r_BH^2 below cut3; its H is bonded to both partners or to neither
            d2 = _sq_dist(xx, low, [i]) + _sq_dist(xx, [i], lh) + _sq_dist(xx, low, lh)
            ok = (d2 < cut3) & (lbond[:, None] | ((dh[None, :] != i) & (dh[None, :] != low[:, None])))
            r, c = np.nonzero(ok)
            bl.append(np.stack([np.full(len(r), i), low[r], lh[c]], 1))
    cat = lambda parts: np.concatenate(parts).astype(np.int64) if parts else np.zeros((0, 3), np.int64)  # noqa: E731
    return cat(bb), cat(bl)
