"""Atom-centered symmetry functions (Behler-Parrinello / ANI style).

Standalone, independently usable featurizers:

    sf = RadialSymmetryFunctions(species=[1, 6, 8], cutoff=6.0)
    desc = sf(graph)            # (N, sf.output_dim) invariant per-atom descriptor

They are element-resolved: contributions from neighbours are bucketed by the
neighbour's chemical species (and, for the angular term, by the unordered pair
of neighbour species), which is what gives the descriptor chemical awareness.
"""
from __future__ import annotations

import itertools

import torch
from torch import Tensor

from xnns.common.data import AtomicGraph
from xnns.common.featurizers import Featurizer, CosineCutoff


def build_triplets(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Enumerate neighbour pairs (j, k) sharing a centre i.

    Pairs are unordered (``j < k`` by edge index). Uses a clear per-centre
    construction; swap for a fully vectorised scheme for very large systems.

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``(2, E)``; row 0 is the source (neighbour) and
        row 1 the destination (centre) node of each edge.
    num_nodes : int
        Number of atoms (nodes) in the graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(edge_jk_first, edge_jk_second, center)``, each of shape ``(T,)``
        where ``T`` is the number of triplets. The first two index into the
        edge dimension (the two edges forming a triplet) and ``center`` is the
        shared centre node. All three are empty long tensors when no triplet
        exists.
    """
    dst = edge_index[1]
    first, second, center = [], [], []
    for i in range(num_nodes):
        edges_i = torch.nonzero(dst == i, as_tuple=False).flatten()
        if edges_i.numel() < 2:
            continue
        pair = torch.combinations(edges_i, r=2)  # (P, 2)
        first.append(pair[:, 0])
        second.append(pair[:, 1])
        center.append(torch.full((pair.shape[0],), i, device=edge_index.device))
    if not first:
        empty = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty, empty, empty
    return torch.cat(first), torch.cat(second), torch.cat(center)


class RadialSymmetryFunctions(Featurizer):
    """Behler G2 radial symmetry functions, resolved by neighbour species.

    Computes ``G2_i = sum_j exp(-eta (r_ij - Rs)^2) fc(r_ij)`` for each central
    atom ``i``, over a grid of ``(eta, Rs)`` parameters and bucketed by the
    chemical species of the neighbour ``j``, giving an invariant per-atom
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
                 etas=(0.05, 0.5, 2.0, 8.0), rs=(0.0,)):
        super().__init__()
        self.species = list(species)
        self.cutoff = cutoff
        self.z_to_idx = {z: i for i, z in enumerate(self.species)}
        self.register_buffer("etas", torch.tensor(list(etas)))
        self.register_buffer("rs", torch.tensor(list(rs)))
        self.cutoff_fn = CosineCutoff(cutoff)
        self._n_params = len(etas) * len(rs)

    @property
    def output_dim(self) -> int:
        """int: Descriptor length, ``n_params * n_species`` (``n_eta * n_rs``
        parameters per species bucket)."""
        return self._n_params * len(self.species)

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
        vec = data.edge_vectors()
        r = torch.linalg.norm(vec, dim=-1)
        src, dst = data.edge_index[0], data.edge_index[1]
        fc = self.cutoff_fn(r)
        # (E, n_eta, n_rs)
        g = torch.exp(-self.etas[None, :, None]
                      * (r[:, None, None] - self.rs[None, None, :]) ** 2)
        g = (g * fc[:, None, None]).reshape(r.shape[0], self._n_params)

        out = torch.zeros(data.num_nodes, self.output_dim,
                          device=r.device, dtype=r.dtype)
        z_src = data.atomic_numbers[src]
        for z, k in self.z_to_idx.items():
            mask = z_src == z
            if mask.any():
                contrib = torch.zeros(data.num_nodes, self._n_params,
                                      device=r.device, dtype=r.dtype)
                contrib.index_add_(0, dst[mask], g[mask])
                out[:, k * self._n_params:(k + 1) * self._n_params] += contrib
        return out


class AngularSymmetryFunctions(Featurizer):
    """ANI-style angular AEV term, resolved by unordered neighbour-species pair.

    G^A_i = 2^(1-zeta) * sum_{j,k} (1 + cos(theta_ijk - theta_s))^zeta
            * exp(-eta ((r_ij + r_ik)/2 - Rs)^2) * fc(r_ij) fc(r_ik)

    summed over distinct neighbour pairs (j, k) of centre i, with a grid over
    (eta, zeta, Rs, theta_s). Contributions are bucketed by the unordered pair
    of neighbour chemical species, giving an invariant per-atom descriptor.

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

    def __init__(self, species: list[int], cutoff: float = 4.0,
                 etas=(0.5,), zetas=(8.0,), rs=(0.0, 1.5, 3.0),
                 theta_s=(0.0, 1.5708, 3.1416, 4.7124)):
        super().__init__()
        self.species = list(species)
        self.cutoff = cutoff
        self.cutoff_fn = CosineCutoff(cutoff)
        for name, vals in [("etas", etas), ("zetas", zetas),
                           ("rs", rs), ("theta_s", theta_s)]:
            self.register_buffer(name, torch.tensor(list(vals)))
        # flat parameter grid
        grid = list(itertools.product(etas, zetas, rs, theta_s))
        self.register_buffer("grid", torch.tensor(grid))   # (P, 4)
        self._n_params = len(grid)
        # unordered species pairs
        self.pairs = list(itertools.combinations_with_replacement(self.species, 2))
        self.pair_to_idx = {p: i for i, p in enumerate(self.pairs)}

    @property
    def output_dim(self) -> int:
        """int: Descriptor length, ``n_params * n_pairs`` (grid size over
        ``(eta, zeta, Rs, theta_s)`` per unordered species pair)."""
        return self._n_params * len(self.pairs)

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
        vec = data.edge_vectors()
        r = torch.linalg.norm(vec, dim=-1)
        fc = self.cutoff_fn(r)
        e1, e2, center = build_triplets(data.edge_index, data.num_nodes)

        out = torch.zeros(data.num_nodes, self.output_dim,
                          device=r.device, dtype=r.dtype)
        if center.numel() == 0:
            return out

        v1, v2 = vec[e1], vec[e2]
        r1, r2 = r[e1], r[e2]
        cos_theta = (v1 * v2).sum(-1) / (r1 * r2).clamp(min=1e-8)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)                                  # (T,)

        eta, zeta, rs, ths = (self.grid[:, 0], self.grid[:, 1],
                              self.grid[:, 2], self.grid[:, 3])         # (P,)
        ang = (2.0 ** (1.0 - zeta)[None]
               * (1.0 + torch.cos(theta[:, None] - ths[None])) ** zeta[None])
        rad = torch.exp(-eta[None] * (((r1 + r2) / 2)[:, None] - rs[None]) ** 2)
        fcfc = (fc[e1] * fc[e2])[:, None]
        g = ang * rad * fcfc                                           # (T, P)

        # species of the two neighbours (src node of each edge)
        src = data.edge_index[0]
        z1 = data.atomic_numbers[src[e1]]
        z2 = data.atomic_numbers[src[e2]]
        for (za, zb), pidx in self.pair_to_idx.items():
            mask = ((z1 == za) & (z2 == zb)) | ((z1 == zb) & (z2 == za))
            if mask.any():
                contrib = torch.zeros(data.num_nodes, self._n_params,
                                      device=r.device, dtype=r.dtype)
                contrib.index_add_(0, center[mask], g[mask])
                out[:, pidx * self._n_params:(pidx + 1) * self._n_params] += contrib
        return out
