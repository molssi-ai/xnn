"""Atom-centered symmetry functions (Behler-Parrinello / ANI style).

Standalone, independently usable featurizers::

    sf = RadialSymmetryFunctions(species=[1, 6, 8], cutoff=6.0)
    desc = sf(graph)            # (N, sf.output_dim) invariant per-atom descriptor

They are element-resolved: contributions from neighbours are bucketed by the
neighbour's chemical species (and, for the angular term, by the unordered pair
of neighbour species), which is what gives the descriptor chemical awareness.

Two knobs make the same math cover both the original Behler-Parrinello (BP,
2007) convention and the ANI / NeuroChem convention used by ``torchani`` (see
Smith et al. 2017, and https://github.com/aiqm/torchani):

* ``prefactor`` on the radial term -- ``1.0`` for BP (eqn 3 of the ANI paper as
  written), ``0.25`` for ANI/NeuroChem (torchani multiplies the radial term by
  ``0.25``; it is a constant absorbed by the network's first layer and does not
  change expressiveness).
* ``cos_factor`` applied to ``cos(theta)`` before ``acos`` in the angular term
  -- ``1.0`` for BP, ``0.95`` for ANI/NeuroChem (torchani scales the cosine by
  ``0.95`` so ``acos`` never sees exactly +-1, where its gradient is infinite).

With ``prefactor=0.25`` and ``cos_factor=0.95`` these featurizers reproduce
``torchani.AEVComputer`` element-for-element for the same parameter grid.
"""
from __future__ import annotations

import itertools
from typing import Optional

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, Featurizer
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import build_triplets, gather_rows

#: Size of the atomic-number lookup tables (Z = 0..118).
_N_Z = 119


class RadialSymmetryFunctions(FastPathModule, Featurizer):
    """Behler G2 radial symmetry functions, resolved by neighbour species.

    Computes ``G2_i = pref * sum_j exp(-eta (r_ij - Rs)^2) fc(r_ij)`` for each
    central atom ``i``, over a grid of ``(eta, Rs)`` parameters and bucketed by
    the chemical species of the neighbour ``j``, giving an invariant per-atom
    descriptor with chemical awareness.

    Parameters
    ----------
    species : list[int]
        Atomic numbers the descriptor resolves neighbours into (one bucket per
        species).
    cutoff : float, optional
        Cutoff radius for the cosine cutoff, by default 6.0.
    etas : sequence of float, optional
        Width parameters of the radial Gaussians, by default
        ``(0.05, 0.5, 2.0, 8.0)``.
    rs : sequence of float, optional
        Radial shifts ``Rs`` of the Gaussians, by default ``(0.0,)``.
    prefactor : float, optional
        Constant multiplying every term. ``1.0`` (Behler-Parrinello, default) or
        ``0.25`` (ANI / NeuroChem / torchani convention).

    Attributes
    ----------
    species : list[int]
        The resolved neighbour species.
    cutoff : float
        Cutoff radius.
    z_to_idx : dict[int, int]
        Mapping from atomic number to its species bucket index.
    cutoff_fn : CosineCutoff
        Smooth cutoff function applied to each edge.
    """

    def __init__(self, species: list[int], cutoff: float = 6.0,
                 etas=(0.05, 0.5, 2.0, 8.0), rs=(0.0,), prefactor: float = 1.0):
        super().__init__()
        self.species = list(species)
        self.cutoff = cutoff
        self.prefactor = float(prefactor)
        self.z_to_idx = {z: i for i, z in enumerate(self.species)}
        # species bucket by atomic number (-1: not a species of the model)
        bucket = torch.full((_N_Z,), -1, dtype=torch.long)
        for z, i in self.z_to_idx.items():
            bucket[z] = i
        self.register_buffer("_bucket", bucket, persistent=False)
        self.register_buffer("etas", torch.tensor(list(etas)))
        self.register_buffer("rs", torch.tensor(list(rs)))
        self.cutoff_fn = CosineCutoff(cutoff)
        self._n_params = len(etas) * len(rs)
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Triton on ``device`` (the fused kernels of :mod:`xnn.dnn.fast.aev`)."""
        from xnn.dnn.fast.aev import supported
        return supported(device, dtype)

    @torch.jit.unused
    def _compute_fast(self, atomic_numbers: Tensor, edge_index: Tensor, r: Tensor, num_nodes: int) -> Tensor:
        """:meth:`compute` on the fused kernels of :mod:`xnn.dnn.fast.aev`."""
        from xnn.common.models.csr import compressed
        from xnn.dnn.fast.aev_kernels import RadialFunction, RadialSpec
        key = (r.dtype, r.device, self.etas._version, self.rs._version, self.prefactor, self.cutoff)
        cached = self.__dict__.get("_fast_spec")
        if cached is None or cached[0] != key:
            n_rs = self.rs.numel()
            spec = RadialSpec(self.etas.to(r.dtype).repeat_interleave(n_rs),
                              self.rs.to(r.dtype).repeat(self.etas.numel()), len(self.species), self.prefactor,
                              self.cutoff)
            self.__dict__["_fast_spec"] = (key, spec)
        spec = self.__dict__["_fast_spec"][1]
        species = self._bucket[atomic_numbers[edge_index[0]]]
        rows = compressed(edge_index[1], num_nodes)
        return RadialFunction.apply(r.contiguous(), species, edge_index[1], rows, num_nodes, spec)

    @property
    def output_dim(self) -> int:
        """int: Descriptor length, ``n_params * n_species`` (``n_eta * n_rs``
        parameters per species bucket)."""
        return self._n_params * len(self.species)

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Tensor:
        """Compute the radial symmetry-function descriptor.

        Parameters
        ----------
        data : AtomicGraph
            Atomic graph providing atomic numbers, edge index and edge vectors.

        Returns
        -------
        Tensor
            Per-atom radial descriptor of shape ``(N, output_dim)``.
        """
        return self.compute(data.atomic_numbers, data.edge_index, data.edge_vectors(), data.num_nodes)

    def compute(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                num_nodes: int, r: Optional[Tensor] = None) -> Tensor:
        """:meth:`forward` on tensors: atomic numbers ``(N,)``, edges ``(2, E)``, edge vectors
        ``(E, 3)`` and optionally their lengths ``(E,)`` (TorchScript-compatible)."""
        if r is None:
            r = torch.linalg.norm(edge_vec, dim=-1)
        if self.fast_active and not torch.jit.is_scripting():
            return self._compute_fast(atomic_numbers, edge_index, r, num_nodes)
        src, dst = edge_index[0], edge_index[1]
        # every edge into the block of its neighbor's species, in one pass; a neighbor of
        # another element adds an exact zero (its cutoff factor is zeroed)
        bucket = self._bucket[atomic_numbers[src]]
        known = bucket >= 0
        fc = torch.where(known, self.prefactor * self.cutoff_fn(r), torch.zeros_like(r))
        # (E, n_eta, n_rs); ordering (eta, rs) matches torchani's (EtaR, ShfR).
        # views by unsqueeze / view: indexing with ``[:, None]`` records a slice per call
        g = torch.exp(-self.etas.view(-1, 1) * (r.view(-1, 1, 1) - self.rs) ** 2)
        g = (g * fc.view(-1, 1, 1)).reshape(r.shape[0], self._n_params)
        n_species = len(self.species)
        row = dst * n_species + bucket.clamp(min=0)
        out = torch.zeros(num_nodes * n_species, self._n_params, device=r.device, dtype=r.dtype)
        return out.index_add_(0, row, g).reshape(num_nodes, n_species * self._n_params)


class AngularSymmetryFunctions(FastPathModule, Featurizer):
    """ANI-style angular AEV term, resolved by unordered neighbour-species pair.

    ``G^A_i = 2^(1-zeta) * sum_{j,k} (1 + cos(theta_ijk - theta_s))^zeta
    * exp(-eta ((r_ij + r_ik)/2 - Rs)^2) * fc(r_ij) fc(r_ik)``

    summed over distinct neighbour pairs ``(j, k)`` of centre ``i``, with a grid
    over ``(eta, zeta, Rs, theta_s)``. Contributions are bucketed by the
    unordered pair of neighbour chemical species, giving an invariant per-atom
    descriptor. Equivalent to ``torchani``'s ``2 * ((1+cos)/2)^zeta * ...``
    form.

    Parameters
    ----------
    species : list[int]
        Atomic numbers the descriptor resolves; the angular term is bucketed by
        the unordered pairs of these species.
    cutoff : float, optional
        Cutoff radius for the cosine cutoff, by default 4.0.
    etas : sequence of float, optional
        Radial width parameters, by default ``(0.5,)``.
    zetas : sequence of float, optional
        Angular resolution exponents, by default ``(8.0,)``.
    rs : sequence of float, optional
        Radial shifts ``Rs``, by default ``(0.0, 1.5, 3.0)``.
    theta_s : sequence of float, optional
        Angular shifts (in radians), by default
        ``(0.0, 1.5708, 3.1416, 4.7124)``.
    cos_factor : float, optional
        Value multiplying ``cos(theta)`` before ``acos``. ``1.0``
        (Behler-Parrinello, default) or ``0.95`` (ANI / NeuroChem / torchani,
        which keeps ``acos`` away from its infinite-gradient endpoints).

    Attributes
    ----------
    species : list[int]
        The resolved species.
    cutoff : float
        Cutoff radius.
    cutoff_fn : CosineCutoff
        Smooth cutoff function applied to each edge.
    pairs : list[tuple[int, int]]
        Unordered neighbour-species pairs (buckets).
    pair_to_idx : dict[tuple[int, int], int]
        Mapping from a species pair to its bucket index.
    """

    # a Python-side mapping with tuple keys, which TorchScript cannot type (``_bucket`` is the table)
    __jit_ignored_attributes__ = ["pair_to_idx"]

    def __init__(self, species: list[int], cutoff: float = 4.0,
                 etas=(0.5,), zetas=(8.0,), rs=(0.0, 1.5, 3.0),
                 theta_s=(0.0, 1.5708, 3.1416, 4.7124), cos_factor: float = 1.0):
        super().__init__()
        self.species = list(species)
        self.cutoff = cutoff
        self.cos_factor = float(cos_factor)
        self.cutoff_fn = CosineCutoff(cutoff)
        for name, vals in [("etas", etas), ("zetas", zetas),
                           ("rs", rs), ("theta_s", theta_s)]:
            self.register_buffer(name, torch.tensor(list(vals)))
        # flat parameter grid; ordering (eta, zeta, rs, theta_s) matches
        # torchani's (EtaA, Zeta, ShfA, ShfZ).
        grid = list(itertools.product(etas, zetas, rs, theta_s))
        self.register_buffer("grid", torch.tensor(grid))   # (P, 4)
        self._n_params = len(grid)
        # unordered species pairs, upper-triangular order == torchani triu_index.
        self.pairs = list(itertools.combinations_with_replacement(self.species, 2))
        self.pair_to_idx = {p: i for i, p in enumerate(self.pairs)}
        # bucket of an unordered species pair by atomic numbers (-1: not a pair of the model)
        bucket = torch.full((_N_Z, _N_Z), -1, dtype=torch.long)
        for (za, zb), i in self.pair_to_idx.items():
            bucket[za, zb] = bucket[zb, za] = i
        self.register_buffer("_bucket", bucket, persistent=False)
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Triton on ``device`` (the fused kernels of :mod:`xnn.dnn.fast.aev`)."""
        from xnn.dnn.fast.aev import supported
        return supported(device, dtype)

    @torch.jit.unused
    def _compute_fast(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor, r_all: Tensor,
                      num_nodes: int) -> Tensor:
        """:meth:`compute` on the fused kernels of :mod:`xnn.dnn.fast.aev`: the triplets and
        their ``(r1, r2, theta)`` as in the reference, the functions and their sums fused."""
        from xnn.dnn.fast.aev_kernels import AngularFunction, AngularSpec
        inside = torch.nonzero(r_all <= self.cutoff).squeeze(-1)
        e1, e2, center = build_triplets(edge_index[:, inside], num_nodes)
        n_pairs = len(self.pairs)
        if center.numel() == 0:
            return edge_vec.new_zeros(num_nodes, n_pairs * self._n_params)
        i1, i2 = inside[e1], inside[e2]
        va, vb = gather_rows(edge_vec, i1), gather_rows(edge_vec, i2)
        r1, r2 = torch.linalg.norm(va, dim=-1), torch.linalg.norm(vb, dim=-1)
        cos_theta = (va * vb).sum(-1) / (r1 * r2).clamp(min=1e-10)
        theta = torch.acos((self.cos_factor * cos_theta).clamp(-1.0, 1.0))
        src = edge_index[0]
        bucket = self._bucket[atomic_numbers[src[i1]], atomic_numbers[src[i2]]]
        # triplets per center (in center order) as row pointers, without a host copy
        counts = torch.zeros(num_nodes, dtype=torch.long, device=center.device).index_add_(
            0, center, torch.ones_like(center))
        ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
        axes = (self.etas, self.zetas, self.rs, self.theta_s)
        key = (edge_vec.dtype, edge_vec.device, self.cutoff) + tuple(t._version for t in axes)
        cached = self.__dict__.get("_fast_spec")
        if cached is None or cached[0] != key:
            spec = AngularSpec(*(t.to(edge_vec.dtype) for t in axes), n_pairs, self.cutoff)
            self.__dict__["_fast_spec"] = (key, spec)
        spec = self.__dict__["_fast_spec"][1]
        return AngularFunction.apply(r1.contiguous(), r2.contiguous(), theta.contiguous(), bucket, center, ptr,
                                     num_nodes, spec)

    @property
    def output_dim(self) -> int:
        """int: Descriptor length, ``n_params * n_pairs`` (grid size over
        ``(eta, zeta, Rs, theta_s)`` per unordered species pair)."""
        return self._n_params * len(self.pairs)

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Tensor:
        """Compute the angular symmetry-function descriptor.

        Parameters
        ----------
        data : AtomicGraph
            Atomic graph providing atomic numbers, edge index and edge vectors.

        Returns
        -------
        Tensor
            Per-atom angular descriptor of shape ``(N, output_dim)``. Returns
            all zeros when the graph contains no neighbour triplets.
        """
        return self.compute(data.atomic_numbers, data.edge_index, data.edge_vectors(), data.num_nodes)

    def compute(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                num_nodes: int, r: Optional[Tensor] = None) -> Tensor:
        """:meth:`forward` on tensors: atomic numbers ``(N,)``, edges ``(2, E)``, edge vectors
        ``(E, 3)`` and optionally their lengths ``(E,)`` (TorchScript-compatible)."""
        r_all = torch.linalg.norm(edge_vec, dim=-1) if r is None else r
        if self.fast_active and not torch.jit.is_scripting():
            return self._compute_fast(atomic_numbers, edge_index, edge_vec, r_all, num_nodes)
        # the triplets of the edges within this cutoff only: the others carry fc = 0
        inside = torch.nonzero(r_all <= self.cutoff).squeeze(-1)
        sub = edge_index[:, inside]
        e1, e2, center = build_triplets(sub, num_nodes)

        n_pairs = len(self.pairs)
        out = torch.zeros(num_nodes * n_pairs, self._n_params, device=edge_vec.device, dtype=edge_vec.dtype)
        if center.numel() == 0:
            return out.reshape(num_nodes, n_pairs * self._n_params)

        # vector, length and cutoff of every edge in one row: one gather per triplet side
        # (split / unbind below: their backward is one concatenation, not a zero fill per slice)
        rows = torch.cat([edge_vec, r_all.unsqueeze(-1), self.cutoff_fn(r_all).unsqueeze(-1)], dim=-1)
        i1, i2 = inside[e1], inside[e2]
        va, la = gather_rows(rows, i1).split([3, 2], dim=-1)
        vb, lb = gather_rows(rows, i2).split([3, 2], dim=-1)
        r12, fc12 = (la * lb).unbind(-1)                               # r1 r2, fc1 fc2
        r_sum = (la + lb).unbind(-1)[0]                                # r1 + r2
        cos_theta = (va * vb).sum(-1) / r12.clamp(min=1e-10)
        # cos_factor (0.95 for ANI) keeps acos away from +-1; clamp is a no-op
        # there but guards the Behler-Parrinello cos_factor=1.0 path.
        theta = torch.acos((self.cos_factor * cos_theta).clamp(-1.0, 1.0))  # (T,)

        # the grid is the product (eta, zeta, Rs, theta_s): the angular factor depends on
        # (zeta, theta_s) and the radial one on (eta, Rs), so each is evaluated on its own
        # axes and the (T, P) term is their outer product; 2^(1-zeta) (1 + cos)^zeta is
        # written 2 ((1 + cos) / 2)^zeta, the 2 carried by the radial factor
        ang = ((1.0 + torch.cos(theta.view(-1, 1, 1) - self.theta_s)) * 0.5) ** self.zetas.view(-1, 1)
        # every triplet into the block of its neighbors' species pair, in one pass; a pair of
        # other elements adds an exact zero (its cutoff factor is zeroed)
        src = edge_index[0]
        bucket = self._bucket[atomic_numbers[src[i1]], atomic_numbers[src[i2]]]
        weight = torch.where(bucket >= 0, 2.0 * fc12, torch.zeros_like(fc12))
        rad = torch.exp(-self.etas.view(-1, 1) * ((0.5 * r_sum).view(-1, 1, 1) - self.rs) ** 2)
        rad = rad * weight.view(-1, 1, 1)                              # (T, n_eta, n_rs)
        n_eta, n_zeta, n_rs, n_th = rad.shape[1], ang.shape[1], rad.shape[2], ang.shape[2]
        g = (rad.view(-1, n_eta, 1, n_rs, 1) * ang.view(-1, 1, n_zeta, 1, n_th)).reshape(-1, self._n_params)
        row = center * n_pairs + bucket.clamp(min=0)
        return out.index_add_(0, row, g).reshape(num_nodes, n_pairs * self._n_params)
