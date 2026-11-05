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
import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.data.elements import atomic_number
from xnn.common.featurizers import Featurizer
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import build_triplets, gather_rows

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
        raise ValueError("unknown cutoff kind " + kind + "; choose one of cosine, tanh, tanh_approx, "
                         "polynomial, hard")
    return torch.where(inside, val, torch.zeros_like(r))


def _acos_margin(dtype: torch.dtype) -> float:
    """The distance from +-1 at which ``acos`` arguments are clamped: the machine epsilon of
    ``dtype``, at least 1e-15."""
    if dtype == torch.float64:
        return 1e-15
    if dtype == torch.float32:
        return 1.1920928955078125e-07
    if dtype == torch.bfloat16:
        return 0.0078125
    return 0.0009765625


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


class AtomCenteredSymmetryFunctions(FastPathModule, Featurizer):
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

    # the specs as Python dicts (eager only; the typed copies serve TorchScript)
    __jit_ignored_attributes__ = ["cutoffs", "functions"]

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
        # the cutoff specs as typed lists (TorchScript)
        self._cut_kind: List[str] = [c["kind"] for c in self.cutoffs]
        self._cut_r: List[float] = [c["r_cut"] for c in self.cutoffs]
        self._cut_inner: List[float] = [c["r_inner"] for c in self.cutoffs]
        self._cut_exponent: List[int] = [c["exponent"] for c in self.cutoffs]
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
        # the parameter columns as separate tables (one gather per used column), 2^(1 - zeta)
        # precomputed, and the function types present (the others are not evaluated)
        for k, name in enumerate(names):
            column = table[:, k]
            if name in ("type", "cutoff", "column"):
                column = column.long()
            self.register_buffer(f"_p_{name}", column.clone(), persistent=False)
        self.register_buffer("_p_pref", torch.pow(2.0, 1.0 - table[:, 6]), persistent=False)
        types = {r["type"] for r in rows}
        self._has_type1, self._has_type2 = 1 in types, 2 in types
        self._has_type3, self._has_type8, self._has_type9 = 3 in types, 8 in types, 9 in types
        self._single_cutoff = len(self.cutoffs) == 1
        # the Triton kernels of xnn.dnn.fast.acsf_kernels: types 1, 2, 3 and 9, up to two cosine / tanh cutoffs
        self._kernel_ok = (types <= {1, 2, 3, 9} and len(self.cutoffs) <= 2
                           and all(c["kind"] in ("cosine", "tanh") for c in self.cutoffs))
        self.fast_active = False
        self._factor_tables(rows, n_sp)

    def _factor_tables(self, rows: list, n_sp: int) -> None:
        """Tables of the factorized angular functions (types 3 and 9).

        Each is a product of an angular factor ``2^(1-zeta) (1 + lambda cos)^zeta`` and a radial factor
        (``exp(-eta (r_ij^2 + r_ik^2 [+ r_jk^2]))`` times the cutoffs), so both are evaluated once per
        distinct parameter set and every function reads its two factors by index. Per bucket
        ``(centre, neighbour, neighbour)`` and lookup slot: the angular factor, the radial factor and the
        output column.
        """
        lookup = self._ang_lookup.reshape(n_sp ** 3, self._ang_lookup.shape[-1])
        fac = [i for i, r in enumerate(rows) if r["type"] in (3, 9)]
        a_keys = sorted({(float(rows[i]["lambda"]), float(rows[i]["zeta"])) for i in fac})
        r_keys = sorted({(int(rows[i]["type"]), float(rows[i]["eta"]), int(rows[i]["cutoff"])) for i in fac})
        a_index = {k: j for j, k in enumerate(a_keys)}
        r_index = {k: j for j, k in enumerate(r_keys)}
        # an empty slot reads the angular factor's extra zero column and adds to column 0
        ia = torch.full_like(lookup, len(a_keys))
        ir = torch.zeros_like(lookup)
        col = torch.zeros_like(lookup)
        for b in range(lookup.shape[0]):
            for k in range(lookup.shape[1]):
                i = int(lookup[b, k])
                if i >= 0 and rows[i]["type"] in (3, 9):
                    r = rows[i]
                    ia[b, k] = a_index[(float(r["lambda"]), float(r["zeta"]))]
                    ir[b, k] = r_index[(int(r["type"]), float(r["eta"]), int(r["cutoff"]))]
                    col[b, k] = r["column"]
        self._n_factor_a, self._n_factor_r = len(a_keys), len(r_keys)
        f64 = torch.float64
        self.register_buffer("_a_lambda", torch.tensor([k[0] for k in a_keys], dtype=f64), persistent=False)
        self.register_buffer("_a_zeta", torch.tensor([k[1] for k in a_keys], dtype=f64), persistent=False)
        self.register_buffer("_a_pref", torch.pow(2.0, 1.0 - torch.tensor([k[1] for k in a_keys], dtype=f64)),
                             persistent=False)
        self.register_buffer("_r_type3", torch.tensor([k[0] == 3 for k in r_keys], dtype=torch.bool),
                             persistent=False)
        self.register_buffer("_r_eta", torch.tensor([k[1] for k in r_keys], dtype=f64), persistent=False)
        self.register_buffer("_r_cutoff", torch.tensor([k[2] for k in r_keys], dtype=torch.long),
                             persistent=False)
        self.register_buffer("_ang_ia", ia, persistent=False)
        self.register_buffer("_ang_ir", ir, persistent=False)
        self.register_buffer("_ang_col", col, persistent=False)
        self._n_sp = n_sp

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
        values: List[Tensor] = []
        for i in range(len(self._cut_kind)):
            values.append(cutoff_function(r, self._cut_kind[i], self._cut_r[i], self._cut_inner[i],
                                          self._cut_exponent[i]))
        return torch.stack(values, dim=-1)

    @torch.jit.ignore
    def raw(self, data: AtomicGraph) -> Tensor:
        """The unscaled descriptor ``(N, output_dim)`` (zero beyond each element's features)."""
        return self.compute_raw(data.atomic_numbers, data.edge_index, data.edge_vectors(), data.num_nodes)

    def compute_raw(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                    n: int) -> Tensor:
        """:meth:`raw` on tensors: atomic numbers ``(N,)``, edges ``(2, E)``, edge vectors
        ``(E, 3)`` and the atom count (TorchScript-compatible)."""
        vec = edge_vec
        dtype = vec.dtype
        out = torch.zeros(n * max(self._dim, 1), dtype=dtype, device=vec.device)
        if self._dim == 0:
            return out.reshape(n, 0)
        sp = self._z2i[atomic_numbers]
        src, dst = edge_index[0], edge_index[1]
        r = torch.linalg.norm(vec, dim=-1)
        fast = False
        if not torch.jit.is_scripting():
            if self.fast_active and vec.is_cuda and self._kernel_ok:
                return self._raw_kernels(vec, r, sp, edge_index, n)
            # the compiled blocks and the CPU kernels are differentiable once: a training step keeps
            # the reference
            fast = self.fast_active and not (self.training and torch.is_grad_enabled())
            if fast and not vec.is_cuda and self._cpu_kernels_ok():
                return self._raw_cpu(vec, sp, edge_index, n)
        if self._rad_lookup.shape[-1] > 0:
            if fast:
                out = out + self._radial_fast(r, sp, src, dst, n)
            else:
                out = out + self._radial(r, sp, src, dst, n)
        if self._ang_lookup.shape[-1] > 0:
            near = r < self._ang_cutoff
            eidx = near.nonzero().squeeze(-1)
            e1, e2, center = build_triplets(edge_index[:, eidx], n)
            if center.numel() > 0:
                e1, e2 = eidx[e1], eidx[e2]
                if fast:
                    out = out + self._angular_fast(vec, r, sp, src, e1, e2, center, n)
                else:
                    out = out + self._angular(vec, r, sp, src, e1, e2, center, n)
        return out.reshape(n, self._dim)

    def _radial(self, r: Tensor, sp: Tensor, src: Tensor, dst: Tensor, n: int) -> Tensor:
        """The radial functions (types 1 and 2) of every edge summed onto the centers ``(n D,)``."""
        dtype = r.dtype
        rows = self._rad_lookup[sp[dst], sp[src]]                      # (E, K)
        ok = rows >= 0
        rows = rows.clamp(min=0)
        fc = self._cutoff_values(r)                                    # (E, C)
        f = fc if self._single_cutoff else torch.gather(fc, 1, self._p_cutoff[rows])
        if self._has_type2:
            val = (torch.exp(-self._p_eta.to(dtype)[rows] * (r.unsqueeze(-1) - self._p_rs.to(dtype)[rows]) ** 2)
                   * f)
            if self._has_type1:
                val = torch.where(self._p_type[rows] == 1, torch.pow(f, self._p_power.to(dtype)[rows]), val)
        else:
            val = torch.pow(f, self._p_power.to(dtype)[rows])
        # a padded lookup entry adds an exact zero to column 0 of its atom
        val = torch.where(ok, val, torch.zeros_like(val))
        index = dst.unsqueeze(-1) * self._dim + self._p_column[rows]
        return torch.zeros(n * self._dim, dtype=dtype, device=r.device).index_add(0, index.reshape(-1),
                                                                                 val.reshape(-1))

    def _angular(self, vec: Tensor, r: Tensor, sp: Tensor, src: Tensor, e1: Tensor, e2: Tensor,
                 center: Tensor, n: int) -> Tensor:
        """The angular functions (types 3, 8 and 9) of the triplets ``(e1, e2)`` of edges sharing
        ``center``, summed onto the centers ``(n D,)``."""
        dtype = vec.dtype
        out = torch.zeros(n * self._dim, dtype=dtype, device=vec.device)
        v1, v2 = gather_rows(vec, e1), gather_rows(vec, e2)
        r1, r2 = gather_rows(r, e1), gather_rows(r, e2)
        r3 = torch.linalg.norm(v1 - v2, dim=-1)
        cos = (v1 * v2).sum(-1) / (r1 * r2)
        # the cutoff products of every declared cutoff
        f12_all = self._cutoff_values(r1) * self._cutoff_values(r2)          # (T, C)
        f123_all = f12_all * self._cutoff_values(r3)
        rsq = r1 * r1 + r2 * r2
        if self._n_factor_a > 0:
            # types 3 and 9: each distinct factor once per triplet, then one gather per function
            s_c, s_1, s_2 = sp[center].clamp(min=0), sp[src[e1]].clamp(min=0), sp[src[e2]].clamp(min=0)
            bucket = (s_c * self._n_sp + s_1) * self._n_sp + s_2
            base = (1.0 + self._a_lambda.to(dtype) * cos.unsqueeze(-1)).clamp(min=0.0)
            a_fac = self._a_pref.to(dtype) * torch.pow(base, self._a_zeta.to(dtype))  # (T, nA)
            a_fac = torch.cat([a_fac, torch.zeros_like(cos).unsqueeze(-1)], dim=-1)
            third = (r3 * r3).unsqueeze(-1) * self._r_type3.to(dtype)
            if self._single_cutoff:
                # one cutoff: a broadcast, no column selection (whose backward, an indirect store, keeps
                # the compiled block from vectorizing)
                f_sel = torch.where(self._r_type3, f123_all, f12_all)
            else:
                cut = self._r_cutoff
                f_sel = torch.where(self._r_type3, f123_all.index_select(1, cut), f12_all.index_select(1, cut))
            r_fac = torch.exp(-self._r_eta.to(dtype) * (rsq.unsqueeze(-1) + third)) * f_sel  # (T, nR)
            val = (torch.gather(a_fac, 1, self._ang_ia.index_select(0, bucket))
                   * torch.gather(r_fac, 1, self._ang_ir.index_select(0, bucket)))
            index = center.unsqueeze(-1) * self._dim + self._ang_col.index_select(0, bucket)
            out = out.index_add(0, index.reshape(-1), val.reshape(-1))
        if self._has_type8:
            rows = self._ang_lookup[sp[center], sp[src[e1]], sp[src[e2]]]  # (T, K)
            ok = rows >= 0
            rows = rows.clamp(min=0)
            angle = (self._p_type[rows] == 8) & ok
            f123 = f123_all if self._single_cutoff else torch.gather(f123_all, 1, self._p_cutoff[rows])
            eta = self._p_eta.to(dtype)[rows]
            ts = self._p_theta_s.to(dtype)[rows]
            # acos with its argument kept off the endpoints everywhere: its derivative stays
            # finite, so the other functions' entries add exact zeros, also to the gradient
            margin = _acos_margin(dtype)
            theta = torch.rad2deg(torch.acos(cos.clamp(-1.0 + margin, 1.0 - margin))).unsqueeze(-1)
            gt = (torch.exp(-eta * (theta - ts) ** 2)
                  + torch.exp(-eta * (theta - (360.0 - ts)) ** 2)
                  + torch.exp(-eta * (theta + ts) ** 2)
                  + torch.exp(-eta * (theta - (360.0 + ts)) ** 2))
            g8 = torch.where(angle, gt * f123, torch.zeros_like(gt))
            index = center.unsqueeze(-1) * self._dim + self._p_column[rows]
            out = out.index_add(0, index.reshape(-1), g8.reshape(-1))
        return out

    @torch.jit.unused
    def _cpu_kernels_ok(self) -> bool:
        """Whether the C++ kernels of :mod:`xnn.dnn.fast.acsf_cpu` build and cover the function set."""
        from xnn.dnn.fast import acsf_cpu
        return self._kernel_ok and acsf_cpu.available()

    @torch.jit.unused
    def _raw_cpu(self, vec: Tensor, sp: Tensor, edge_index: Tensor, n: int) -> Tensor:
        """:meth:`compute_raw` on the C++ / OpenMP kernels of :mod:`xnn.dnn.fast.acsf_cpu` (evaluation)."""
        from xnn.dnn.fast import acsf_cpu
        from xnn.dnn.fast.acsf_tables import tables
        key = (vec.dtype, "cpu")
        cached = self.__dict__.get("_cpu_tables")
        if cached is None or cached[0] != key:
            rad, ang = tables(self.functions, self.species, self.cutoffs, self._dim, vec.dtype, "cpu")
            self.__dict__["_cpu_tables"] = (key, acsf_cpu._Tables(
                rad if self._rad_lookup.shape[-1] > 0 else None,
                ang if self._ang_lookup.shape[-1] > 0 else None, len(self.species)))
        return acsf_cpu.symmetry_functions(vec, edge_index, sp, n, self.__dict__["_cpu_tables"][1], self._dim,
                                           self._ang_cutoff)

    @torch.jit.unused
    def _raw_kernels(self, vec: Tensor, r: Tensor, sp: Tensor, edge_index: Tensor, n: int) -> Tensor:
        """:meth:`compute_raw` on the Triton kernels of :mod:`xnn.dnn.fast.acsf_kernels` (differentiable
        twice: evaluation and training): the edges and triplets as in the reference, sorted into segments
        of one center and neighbor species (species pair), the functions and their sums fused."""
        from xnn.dnn.fast import acsf_kernels as k
        key = (vec.dtype, vec.device)
        cached = self.__dict__.get("_kernel_tables")
        if cached is None or cached[0] != key:
            self.__dict__["_kernel_tables"] = (key,) + k.tables(self.functions, self.species, self.cutoffs,
                                                                self._dim, vec.dtype, vec.device)
        _, rad, ang = self.__dict__["_kernel_tables"]
        n_sp = len(self.species)
        src, dst = edge_index[0], edge_index[1]
        sp32 = sp.to(torch.int32)
        out = vec.new_zeros(n, self._dim)
        if self._rad_lookup.shape[-1] > 0:
            s_nb = sp[src]
            order, ptr = k.segments(dst, s_nb, n, n_sp)
            row = (sp[dst] * n_sp + s_nb).index_select(0, order).to(torch.int32)
            out = out + k.RadialFunction.apply(r.index_select(0, order), row, dst.index_select(0, order), ptr,
                                               sp32, n, rad)
        if self._ang_lookup.shape[-1] > 0:
            eidx = (r < self._ang_cutoff).nonzero().squeeze(-1)
            e1, e2, center = build_triplets(edge_index[:, eidx], n)
            if center.numel() > 0:
                e1, e2 = eidx[e1], eidx[e2]
                pair = k.pair_index(sp[src[e1]], sp[src[e2]], n_sp)
                order, ptr = k.segments(center, pair, n, ang.q)
                e1, e2, center, pair = (t.index_select(0, order) for t in (e1, e2, center, pair))
                v1, v2 = gather_rows(vec, e1), gather_rows(vec, e2)
                r1, r2 = gather_rows(r, e1), gather_rows(r, e2)
                r3 = torch.linalg.norm(v1 - v2, dim=-1)
                cos = (v1 * v2).sum(-1) / (r1 * r2)
                row = (sp[center] * ang.q + pair).to(torch.int32)
                out = out + k.AngularFunction.apply(r1.contiguous(), r2.contiguous(), r3.contiguous(),
                                                    cos.contiguous(), row, center, ptr, sp32, n, ang)
        return out

    @torch.jit.unused
    def _compiled(self, name: str, scalar: bool):
        """:meth:`_radial` or :meth:`_angular` compiled for CPUs (``torch.compile``, dynamic sizes)."""
        cache = self.__dict__.setdefault("_compiled_blocks", {})
        key = (name, scalar)
        if key not in cache:
            cache[key] = torch.compile(getattr(self, "_" + name), dynamic=True)
        return cache[key]

    @torch.jit.unused
    def _run_compiled(self, name: str, *args):
        """The block compiled with vectorized loops, else with scalar loops (a construct inductor
        cannot vectorize), else the reference block (the choice is made once per block). With several
        cutoffs the angular block starts with scalar loops: the backward of its cutoff-column selection
        is an indirect store, which inductor does not vectorize; with one cutoff there is no selection."""
        import contextlib

        from torch._inductor import config as inductor_config
        modes = self.__dict__.setdefault("_compiled_modes", {})
        known = modes.get(name)
        inputs = [a for a in args if isinstance(a, Tensor) and a.requires_grad]
        order = (["scalar", "vector", "eager"] if name == "angular" and not self._single_cutoff
                 else ["vector", "scalar", "eager"])
        for mode in [known] if known is not None else order:
            if mode == "eager":
                modes[name] = mode
                return getattr(self, "_" + name)(*args)
            setting = (inductor_config.patch({"cpp.simdlen": 1}) if mode == "scalar"
                       else contextlib.nullcontext())
            try:
                with setting:
                    block = self._compiled(name, mode == "scalar")
                    if known is None and inputs:
                        # the backward compiles at its first use: probed here on copies, so that
                        # its failure also moves on to the next mode
                        probe = [a.detach().requires_grad_(True) if isinstance(a, Tensor) and a.requires_grad
                                 else a for a in args]
                        leaves = [a for a in probe if isinstance(a, Tensor) and a.requires_grad]
                        torch.autograd.grad(block(*probe).sum(), leaves, allow_unused=True)
                    out = block(*args)
            except Exception:  # noqa: BLE001  (a compiler failure: the next mode)
                if known is not None:
                    raise
                continue
            if inputs or known is not None:
                modes[name] = mode
            return out
        raise RuntimeError("unreachable")

    @torch.jit.unused
    def _radial_fast(self, r: Tensor, sp: Tensor, src: Tensor, dst: Tensor, n: int) -> Tensor:
        """:meth:`_radial`, compiled."""
        return self._run_compiled("radial", r, sp, src, dst, n)

    @torch.jit.unused
    def _angular_fast(self, vec: Tensor, r: Tensor, sp: Tensor, src: Tensor, e1: Tensor, e2: Tensor,
                      center: Tensor, n: int) -> Tensor:
        """:meth:`_angular`, compiled."""
        return self._run_compiled("angular", vec, r, sp, src, e1, e2, center, n)

    @torch.jit.unused
    def fast_policies(self):
        """The ``"auto"`` policies (evaluation, training) of :mod:`xnn.dnn.fast.acsf`; on a CPU, the
        C++ kernels' where they build and cover the function set."""
        from xnn.dnn.fast.acsf import AUTO_POLICY, AUTO_POLICY_CPU, AUTO_POLICY_TRAINING
        if self._params.device.type == "cpu" and self._cpu_kernels_ok():
            return AUTO_POLICY_CPU, AUTO_POLICY_TRAINING
        return AUTO_POLICY, AUTO_POLICY_TRAINING

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """CPUs where the C++ kernels build or ``torch.compile`` produces code, and CUDA devices with
        Triton (:mod:`xnn.dnn.fast.acsf`)."""
        if dtype not in (torch.float32, torch.float64):
            return False
        if device.type == "cuda":
            from xnn.common.models.dispersion_fast._triton import supported
            return supported(device, dtype)
        if device.type != "cpu":
            return False
        if self._cpu_kernels_ok():
            return True
        from xnn.dnn.fast.acsf import available
        return available()

    def scaling(self) -> Tuple[Tensor, Tensor]:
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

    @torch.jit.ignore
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
        return self.compute(data.atomic_numbers, data.edge_index, data.edge_vectors(), data.num_nodes)

    def compute(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                num_nodes: int) -> Tensor:
        """:meth:`forward` on tensors (TorchScript-compatible)."""
        return self.scale(self.compute_raw(atomic_numbers, edge_index, edge_vec, num_nodes), atomic_numbers)

    def scale(self, g: Tensor, atomic_numbers: Tensor) -> Tensor:
        """The scaled descriptor from the unscaled one ``g`` ``(N, output_dim)``."""
        if self.mode == "none":
            return g
        sp = self._z2i[atomic_numbers].clamp(min=0)
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
