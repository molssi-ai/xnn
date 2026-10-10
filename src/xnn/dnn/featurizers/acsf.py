"""Atom-centered symmetry functions of the HDNNP (Behler, J. Chem. Phys. 134, 074106, 2011).

Each element carries its own list of symmetry functions; every function has a
central element, one neighbour element (radial) or an unordered pair of them
(angular), its parameters and one of the declared cutoff functions. The
descriptor of an atom is the vector of its element's functions, optionally
scaled with statistics of a training set. Usable on its own::

    acsf = AtomCenteredSymmetryFunctions(
        species=[1, 8],
        cutoffs=[{"kind": "cosine", "r_cut": 6.0}],
        functions=[{"element": 1, "type": 2, "neighbors": [8], "eta": 0.5, "rs": 0.0},
                   {"element": 8, "type": 3, "neighbors": [1, 1], "eta": 0.01,
                    "lambda": 1.0, "zeta": 4.0}])
    g = acsf(graph)          # (N, acsf.output_dim), zero-padded per element

Function types (``r_ij``, ``r_ik``, ``r_jk`` the distances of a neighbour pair
``j < k`` of the central atom ``i``, ``theta`` the angle at ``i``, ``fc`` the
function's cutoff):

* ``1``: ``sum_j fc(r_ij)^(2^(m-1))``, ``m = 1 .. n_features`` (a coordination count).
* ``2``: ``sum_j exp(-eta (r_ij - r_s)^2) fc(r_ij)``.
* ``3``: ``2^(1-zeta) sum_{j<k} (1 + lambda cos theta)^zeta exp(-eta (r_ij^2 +
  r_ik^2 + r_jk^2)) fc(r_ij) fc(r_ik) fc(r_jk)``, with ``1 + lambda cos theta``
  floored at zero.
* ``8``: ``sum_{j<k} g(theta) fc(r_ij) fc(r_ik) fc(r_jk)`` with ``g`` the sum of
  four Gaussians ``exp(-eta (theta - t)^2)`` at ``t = theta_s, 360 - theta_s,
  -theta_s, 360 + theta_s`` (angles in degrees).
* ``9``: type 3 without the ``r_jk`` terms (``exp(-eta (r_ij^2 + r_ik^2))
  fc(r_ij) fc(r_ik)``).

The angular sums run over unordered neighbour pairs, so ``2^(1-zeta)`` counts
each pair twice as in the ``j != k`` double sum of the paper. These are the
conventions of the RuNNer code; :mod:`xnn.dnn.common.runner` reads its
``input.nn`` and ``scaling.data`` files into this featurizer.

Cutoff functions (``r_c`` the cutoff, ``r_i`` an inner radius, ``x = (r -
r_i) / (r_c - r_i)``; every function is zero for ``r >= r_c``):

* ``cosine``: ``1`` below ``r_i``, else ``(cos(pi x) + 1) / 2``.
* ``tanh``: ``tanh^3(1 - x)`` (``tanh^3(1)`` below ``r_i``).
* ``tanh_approx``: the same with ``tanh`` replaced by ``y (27 + y^2) / (27 + 9 y^2)``.
* ``polynomial``: ``(1 - r^2 / r_c^2)^n`` with integer ``n`` (``exponent``).
* ``hard``: ``1``.

Scaling (``mode``), with ``min``, ``max`` and ``avg`` per element and function
over the atoms of a training set (:meth:`AtomCenteredSymmetryFunctions.fit_scaling`):
``"none"`` ``G``; ``"scale"`` ``(G - min) / (max - min)``; ``"center"`` ``G -
avg``; ``"center_scale"`` ``(G - avg) / (max - min)``; ``"range"`` ``s_min +
(s_max - s_min) (G - min) / (max - min)``. A function with ``max == min`` is
not divided.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Union

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.data.elements import atomic_number
from xnn.common.featurizers import Featurizer
from xnn.common.models.ops import build_triplets

#: cutoff function kinds and their RuNNer keywords
CUTOFF_KINDS = ("cosine", "tanh", "tanh_approx", "polynomial", "hard")
#: scaling modes of the descriptor
SCALING_MODES = ("none", "scale", "center", "center_scale", "range")

_TANH1_CUBED = math.tanh(1.0) ** 3


def _as_z(value) -> int:
    return int(value) if not isinstance(value, str) else atomic_number(value)


def cutoff_function(r: Tensor, kind: str, r_cut: float, r_inner: float = 0.0,
                    exponent: int = 2) -> Tensor:
    """One cutoff function of :mod:`~xnn.dnn.featurizers.acsf` at distances ``r``.

    Parameters
    ----------
    r : Tensor
        Distances, any shape.
    kind : str
        One of :data:`CUTOFF_KINDS`.
    r_cut : float
        Outer cutoff; the function is zero at and beyond it.
    r_inner : float, optional
        Inner radius of the ``cosine``, ``tanh`` and ``tanh_approx`` kinds,
        by default 0.
    exponent : int, optional
        Power ``n`` of the ``polynomial`` kind, by default 2.

    Returns
    -------
    Tensor
        The cutoff values, the shape of ``r``.
    """
    inside = r < r_cut
    if kind == "hard":
        return inside.to(r.dtype)
    if kind == "polynomial":
        base = 1.0 - (r / r_cut) ** 2
        val = torch.ones_like(r)
        for _ in range(int(exponent)):
            val = val * base
        return torch.where(inside, val, torch.zeros_like(r))
    x = ((r - r_inner) / (r_cut - r_inner)).clamp(min=0.0, max=1.0)
    if kind == "cosine":
        val = 0.5 * (torch.cos(math.pi * x) + 1.0)
    elif kind == "tanh":
        val = torch.tanh(1.0 - x) ** 3
    elif kind == "tanh_approx":
        y = 1.0 - x
        val = (y * (27.0 + y * y) / (27.0 + 9.0 * y * y)) ** 3
    else:
        raise ValueError(f"unknown cutoff kind {kind!r}; choose one of {CUTOFF_KINDS}")
    return torch.where(inside, val, torch.zeros_like(r))


def _normalize_cutoff(spec) -> dict:
    spec = dict(spec)
    kind = str(spec.get("kind", "cosine")).lower()
    if kind not in CUTOFF_KINDS:
        raise ValueError(f"unknown cutoff kind {kind!r}; choose one of {CUTOFF_KINDS}")
    out = {"kind": kind, "r_cut": float(spec["r_cut"]),
           "r_inner": float(spec.get("r_inner", 0.0) or 0.0),
           "exponent": int(spec.get("exponent", 2))}
    if out["r_inner"] >= out["r_cut"]:
        raise ValueError(f"cutoff {spec}: the inner radius must be below r_cut")
    if kind in ("polynomial", "hard") and out["r_inner"] != 0.0:
        raise ValueError(f"cutoff {spec}: the {kind} kind has no inner radius")
    return out


def _normalize_function(spec, n_cutoffs: int) -> list[dict]:
    """A function spec as one or more single-feature rows (type 1 may give several)."""
    spec = dict(spec)
    t = int(spec["type"])
    if t not in (1, 2, 3, 8, 9):
        raise ValueError(f"unsupported symmetry function type {t} (1, 2, 3, 8 or 9)")
    neighbors = [_as_z(z) for z in spec["neighbors"]]
    if len(neighbors) != (1 if t in (1, 2) else 2):
        raise ValueError(f"type {t} takes {1 if t in (1, 2) else 2} neighbour element(s): {spec}")
    cut = int(spec.get("cutoff", 0))
    if not 0 <= cut < n_cutoffs:
        raise ValueError(f"function {spec} names cutoff {cut}; {n_cutoffs} are declared")
    row = {"element": _as_z(spec["element"]), "type": t, "neighbors": tuple(sorted(neighbors)),
           "cutoff": cut, "eta": float(spec.get("eta", 0.0)), "rs": float(spec.get("rs", 0.0)),
           "lambda": float(spec.get("lambda", 1.0)), "zeta": float(spec.get("zeta", 1.0)),
           "theta_s": float(spec.get("theta_s", 0.0)), "power": 1}
    if t == 1:
        return [dict(row, power=2 ** m) for m in range(int(spec.get("n_features", 1)))]
    if t in (3, 9) and row["lambda"] not in (-1.0, 1.0):
        raise ValueError(f"lambda must be +1 or -1: {spec}")
    return [row]


class AtomCenteredSymmetryFunctions(Featurizer):
    """Behler atom-centered symmetry functions, a separate list per element.

    Parameters
    ----------
    species : sequence of int or str
        The elements; the descriptor rows of atoms of other elements are zero.
    cutoffs : sequence of dict
        The cutoff functions, each ``{"kind", "r_cut"}`` plus ``r_inner``
        (``cosine``, ``tanh``, ``tanh_approx``) or ``exponent``
        (``polynomial``); functions refer to them by position.
    functions : sequence of dict
        The symmetry functions, each with ``element``, ``type`` (1, 2, 3, 8 or
        9), ``neighbors`` (one element for types 1 and 2, two for the angular
        types, in any order), ``cutoff`` (index into ``cutoffs``, default 0)
        and the parameters of its type: ``eta`` and ``rs`` (2), ``eta``,
        ``lambda`` and ``zeta`` (3 and 9), ``theta_s`` and ``eta`` in degrees
        (8), ``n_features`` (1). An element's features keep the order of the
        list.
    mode : str, optional
        Scaling mode, one of :data:`SCALING_MODES`, by default ``"none"``.
    scale_range : tuple of float, optional
        ``(s_min, s_max)`` of the ``"range"`` mode, by default ``(0, 1)``.

    Attributes
    ----------
    species : list[int]
        The elements.
    n_features : dict[int, int]
        Number of features of every element.
    cutoff : float
        Neighbour-list radius, the largest cutoff.
    stat_min, stat_max, stat_avg : Tensor
        Scaling statistics ``(n_species, output_dim)``.
    """

    def __init__(self, species: Sequence[Union[int, str]], cutoffs: Sequence[dict],
                 functions: Sequence[dict], mode: str = "none",
                 scale_range: Sequence[float] = (0.0, 1.0)):
        super().__init__()
        self.species = [_as_z(z) for z in species]
        if len(set(self.species)) != len(self.species):
            raise ValueError(f"repeated species in {self.species}")
        self.cutoffs = [_normalize_cutoff(c) for c in cutoffs]
        if not self.cutoffs:
            raise ValueError("at least one cutoff function is needed")
        rows = [r for f in functions for r in _normalize_function(f, len(self.cutoffs))]
        for r in rows:
            if r["element"] not in self.species or any(z not in self.species for z in r["neighbors"]):
                raise ValueError(f"symmetry function {r} names an element outside {self.species}")
        self.functions = rows
        sp_index = {z: i for i, z in enumerate(self.species)}
        n_sp = len(self.species)
        cols = {z: 0 for z in self.species}
        for r in rows:
            r["column"] = cols[r["element"]]
            cols[r["element"]] += 1
        self.n_features = dict(cols)
        self._dim = max(cols.values()) if rows else 0
        self.cutoff = max(c["r_cut"] for c in self.cutoffs)
        self.mode = mode
        if mode not in SCALING_MODES:
            raise ValueError(f"unknown scaling mode {mode!r}; choose one of {SCALING_MODES}")
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))

        z2i = torch.full((119,), -1, dtype=torch.long)
        for z, i in sp_index.items():
            z2i[z] = i
        self.register_buffer("_z2i", z2i, persistent=False)
        mask = torch.zeros(n_sp, max(self._dim, 1), dtype=torch.bool)
        for z, n in cols.items():
            mask[sp_index[z], :n] = True
        self.register_buffer("_valid", mask[:, :self._dim], persistent=False)

        # per-function parameter table and the (centre, neighbour[, neighbour]) lookups
        names = ("type", "cutoff", "column", "eta", "rs", "lambda", "zeta", "theta_s", "power")
        table = torch.tensor([[float(r[k]) for k in names] for r in rows] or [[0.0] * len(names)],
                             dtype=torch.float64)
        self.register_buffer("_params", table, persistent=False)
        rad = [i for i, r in enumerate(rows) if r["type"] in (1, 2)]
        ang = [i for i, r in enumerate(rows) if r["type"] in (3, 8, 9)]
        self.register_buffer("_rad_lookup", self._lookup(rad, sp_index, 1), persistent=False)
        self.register_buffer("_ang_lookup", self._lookup(ang, sp_index, 2), persistent=False)
        self._ang_cutoff = max((self.cutoffs[rows[i]["cutoff"]]["r_cut"] for i in ang), default=0.0)

        self.register_buffer("stat_min", torch.zeros(n_sp, self._dim))
        self.register_buffer("stat_max", torch.ones(n_sp, self._dim))
        self.register_buffer("stat_avg", torch.zeros(n_sp, self._dim))

    def _lookup(self, idx: list[int], sp_index: dict, order: int) -> Tensor:
        """Function rows per (centre, neighbour species...), padded with -1."""
        n_sp = len(self.species)
        shape = (n_sp,) * (order + 1)
        buckets: dict[tuple, list[int]] = {}
        for i in idx:
            r = self.functions[i]
            nb = [sp_index[z] for z in r["neighbors"]]
            keys = {(sp_index[r["element"]], *nb)}
            if order == 2:
                keys.add((sp_index[r["element"]], nb[1], nb[0]))
            for key in keys:
                buckets.setdefault(key, []).append(i)
        width = max((len(v) for v in buckets.values()), default=0)
        out = torch.full(shape + (max(width, 1),), -1, dtype=torch.long)
        for key, rows in buckets.items():
            out[key + (slice(0, len(rows)),)] = torch.tensor(rows)
        return out[..., :width]

    @property
    def output_dim(self) -> int:
        """int: Length of the (zero-padded) descriptor, the largest feature count of an element."""
        return self._dim

    def _cutoff_values(self, r: Tensor) -> Tensor:
        """Every declared cutoff function at ``r``, stacked on a new last axis."""
        return torch.stack([cutoff_function(r, c["kind"], c["r_cut"], c["r_inner"], c["exponent"])
                            for c in self.cutoffs], dim=-1)

    def raw(self, data: AtomicGraph) -> Tensor:
        """The unscaled descriptor ``(N, output_dim)`` (zero beyond each element's features)."""
        vec = data.edge_vectors()
        dtype = vec.dtype
        n = data.num_nodes
        out = torch.zeros(n * max(self._dim, 1), dtype=dtype, device=vec.device)
        if self._dim == 0:
            return out.reshape(n, 0)
        params = self._params.to(dtype)
        sp = self._z2i[data.atomic_numbers]
        src, dst = data.edge_index[0], data.edge_index[1]
        r = torch.linalg.norm(vec, dim=-1)

        if self._rad_lookup.shape[-1] > 0:
            rows = self._rad_lookup[sp[dst], sp[src]]                      # (E, K)
            ok = rows >= 0
            rows = rows.clamp(min=0)
            p = params[rows]                                               # (E, K, 9)
            fc = self._cutoff_values(r)                                    # (E, C)
            f = torch.gather(fc, 1, p[..., 1].long())                      # (E, K)
            gauss = torch.exp(-p[..., 3] * (r[:, None] - p[..., 4]) ** 2) * f
            power = torch.pow(f, p[..., 8])
            val = torch.where(p[..., 0] == 1.0, power, gauss)
            val = torch.where(ok, val, torch.zeros_like(val))
            index = dst[:, None] * self._dim + p[..., 2].long()
            out = out.index_add(0, index[ok], val[ok])

        if self._ang_lookup.shape[-1] > 0:
            near = r < self._ang_cutoff
            eidx = near.nonzero().squeeze(-1)
            e1, e2, center = build_triplets(data.edge_index[:, eidx], n)
            if center.numel() > 0:
                e1, e2 = eidx[e1], eidx[e2]
                rows = self._ang_lookup[sp[center], sp[src[e1]], sp[src[e2]]]  # (T, K)
                ok = rows >= 0
                rows = rows.clamp(min=0)
                p = params[rows]
                v1, v2 = vec[e1], vec[e2]
                r1, r2 = r[e1], r[e2]
                r3 = torch.linalg.norm(v1 - v2, dim=-1)
                cos = (v1 * v2).sum(-1) / (r1 * r2)
                cut = p[..., 1].long()
                f1 = torch.gather(self._cutoff_values(r1), 1, cut)
                f2 = torch.gather(self._cutoff_values(r2), 1, cut)
                f3 = torch.gather(self._cutoff_values(r3), 1, cut)
                eta, lam, zeta = p[..., 3], p[..., 5], p[..., 6]
                base = (1.0 + lam * cos[:, None]).clamp(min=0.0)
                ang = torch.pow(2.0, 1.0 - zeta) * torch.pow(base, zeta)
                rsq = (r1 * r1 + r2 * r2)[:, None]
                g3 = ang * torch.exp(-eta * (rsq + (r3 * r3)[:, None])) * f1 * f2 * f3
                g9 = ang * torch.exp(-eta * rsq) * f1 * f2
                theta = torch.rad2deg(torch.acos(cos.clamp(-1.0 + 1e-15, 1.0 - 1e-15)))[:, None]
                ts = p[..., 7]
                gt = (torch.exp(-eta * (theta - ts) ** 2) + torch.exp(-eta * (theta - (360.0 - ts)) ** 2)
                      + torch.exp(-eta * (theta + ts) ** 2) + torch.exp(-eta * (theta - (360.0 + ts)) ** 2))
                g8 = gt * f1 * f2 * f3
                t = p[..., 0]
                val = torch.where(t == 3.0, g3, torch.where(t == 9.0, g9, g8))
                val = torch.where(ok, val, torch.zeros_like(val))
                index = center[:, None] * self._dim + p[..., 2].long()
                out = out.index_add(0, index[ok], val[ok])
        return out.reshape(n, self._dim)

    def scaling(self) -> tuple[Tensor, Tensor]:
        """The ``(shift, factor)`` of ``G' = (G - shift) * factor``, each ``(n_species, output_dim)``."""
        span = self.stat_max - self.stat_min
        flat = span == 0
        inv = torch.where(flat, torch.ones_like(span), 1.0 / torch.where(flat, torch.ones_like(span), span))
        zero = torch.zeros_like(span)
        if self.mode == "none":
            return zero, torch.ones_like(span)
        if self.mode == "scale":
            return self.stat_min, inv
        if self.mode == "center":
            return self.stat_avg, torch.ones_like(span)
        if self.mode == "center_scale":
            return self.stat_avg, inv
        smin, smax = self.scale_range
        factor = (smax - smin) * inv
        return self.stat_min - smin / factor, factor

    def forward(self, data: AtomicGraph) -> Tensor:
        """The scaled descriptor ``(N, output_dim)``; columns beyond an element's features are zero.

        Parameters
        ----------
        data : AtomicGraph
            Atomic graph providing atomic numbers, edge index and edge vectors.

        Returns
        -------
        Tensor
            Per-atom descriptors.
        """
        g = self.raw(data)
        if self.mode == "none":
            return g
        sp = self._z2i[data.atomic_numbers].clamp(min=0)
        shift, factor = self.scaling()
        scaled = (g - shift.to(g.dtype)[sp]) * factor.to(g.dtype)[sp]
        return torch.where(self._valid[sp], scaled, torch.zeros_like(scaled))

    @torch.no_grad()
    def fit_scaling(self, graphs: Iterable[AtomicGraph], mode: Optional[str] = None) -> None:
        """Set ``stat_min``, ``stat_max`` and ``stat_avg`` from the atoms of ``graphs``.

        Parameters
        ----------
        graphs : iterable of AtomicGraph
            Training structures (single graphs or batches).
        mode : str, optional
            Also switch the scaling mode.
        """
        if mode is not None:
            if mode not in SCALING_MODES:
                raise ValueError(f"unknown scaling mode {mode!r}")
            self.mode = mode
        n_sp = len(self.species)
        lo = torch.full((n_sp, self._dim), math.inf, dtype=torch.float64)
        hi = torch.full((n_sp, self._dim), -math.inf, dtype=torch.float64)
        total = torch.zeros(n_sp, self._dim, dtype=torch.float64)
        count = torch.zeros(n_sp, dtype=torch.float64)
        for data in graphs:
            g = self.raw(data).detach().double().cpu()
            sp = self._z2i[data.atomic_numbers].cpu()
            for i in range(n_sp):
                rows = g[sp == i]
                if rows.shape[0] == 0:
                    continue
                lo[i] = torch.minimum(lo[i], rows.min(0).values)
                hi[i] = torch.maximum(hi[i], rows.max(0).values)
                total[i] += rows.sum(0)
                count[i] += rows.shape[0]
        missing = [self.species[i] for i in range(n_sp) if count[i] == 0]
        if missing:
            raise ValueError(f"no atoms of elements {missing} to compute the scaling from")
        valid = self._valid.cpu()
        dtype = self.stat_min.dtype
        self.stat_min.copy_(torch.where(valid, lo, torch.zeros_like(lo)).to(dtype))
        self.stat_max.copy_(torch.where(valid, hi, torch.ones_like(hi)).to(dtype))
        self.stat_avg.copy_(torch.where(valid, total / count[:, None], torch.zeros_like(total)).to(dtype))
