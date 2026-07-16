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

import torch
from torch import Tensor

from xnns.common.data import AtomicGraph
from xnns.common.featurizers import Featurizer, CosineCutoff


def build_triplets(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
    """Enumerate neighbour pairs ``(j, k)`` sharing a centre ``i``.

    For every centre atom, all unordered pairs of its incoming edges are
    returned. Fully vectorised (no Python loop over atoms): edges are grouped by
    their destination (centre) node and, within each group of ``c`` edges, the
    ``c*(c-1)/2`` pairs are enumerated with a shared lower-triangular index
    template -- the same scheme ``torchani`` uses.

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
        edge dimension (the two edges forming a triplet, ``first < second`` in
        the per-centre ordering) and ``center`` is the shared centre node. All
        three are empty long tensors when no triplet exists.
    """
    device = edge_index.device
    dst = edge_index[1]
    # Group edges by their centre (destination) node.
    order = torch.argsort(dst, stable=True)          # (E,) edge ids, centre-grouped
    counts = torch.bincount(dst, minlength=num_nodes)  # (num_nodes,) edges per centre
    n_pairs = counts * (counts - 1) // 2               # pairs per centre
    total = int(n_pairs.sum())
    if total == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    # Start offset of each centre's block within `order`.
    offsets = torch.cumsum(counts, 0) - counts         # (num_nodes,)
    # Centre node of every emitted pair, and that centre's block offset/count.
    center = torch.repeat_interleave(torch.arange(num_nodes, device=device), n_pairs)
    base = torch.repeat_interleave(offsets, n_pairs)   # (T,) offset into `order`
    cnt = torch.repeat_interleave(counts, n_pairs)     # (T,) edges at this centre

    # Local pair (a, b) with 0 <= a < b < cnt, laid out from a shared template
    # sized to the largest centre, then masked down to each centre's count.
    m = int(counts.max())
    tri = torch.tril_indices(m, m, -1, device=device)  # (2, m*(m-1)/2): b > a
    # position of each emitted pair within its centre's pair-list
    pair_pos = torch.arange(total, device=device) - (
        torch.cumsum(n_pairs, 0) - n_pairs).repeat_interleave(n_pairs)
    a_local = tri[1][pair_pos]                          # smaller local edge index
    b_local = tri[0][pair_pos]                          # larger local edge index
    first = order[base + a_local]
    second = order[base + b_local]
    return first, second, center


class RadialSymmetryFunctions(Featurizer):
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
        # (E, n_eta, n_rs); ordering (eta, rs) matches torchani's (EtaR, ShfR).
        g = torch.exp(-self.etas[None, :, None]
                      * (r[:, None, None] - self.rs[None, None, :]) ** 2)
        g = (self.prefactor * g * fc[:, None, None]).reshape(r.shape[0], self._n_params)

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
        cos_theta = (v1 * v2).sum(-1) / (r1 * r2).clamp(min=1e-10)
        # cos_factor (0.95 for ANI) keeps acos away from +-1; clamp is a no-op
        # there but guards the Behler-Parrinello cos_factor=1.0 path.
        theta = torch.acos((self.cos_factor * cos_theta).clamp(-1.0, 1.0))  # (T,)

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
