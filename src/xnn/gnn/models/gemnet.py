"""GemNet (Gasteiger, Becker, Guennemann, NeurIPS 2021, arXiv:2106.08903).

Geometric message passing over directed edge embeddings with one-hop
(triplet) and two-hop (quadruplet) interactions, built on the xnn bases
shared with DimeNet; one class covers GemNet-Q, -T, -dQ and -dT.
"""
# NOTE: no `from __future__ import annotations` here -- TorchScript needs the
# real annotations of the scriptable core.
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import scatter_sum
from xnn.common.models.registry import register_model

from ..featurizers import BesselRBF, PolynomialCutoff, SphericalBesselBasis, real_spherical_harmonics, zonal_harmonics

_MAX_Z = 100


def he_orthogonal_(weight: Tensor) -> Tensor:
    """He-scaled orthogonal initialization with exact standardization (paper section 6).

    A random (semi-)orthogonal matrix is standardized per output unit to zero
    mean and unit variance and scaled to variance ``1 / fan_in``.

    Parameters
    ----------
    weight : Tensor
        A ``(fan_out, fan_in)`` weight matrix, modified in place.

    Returns
    -------
    Tensor
        ``weight``.
    """
    with torch.no_grad():
        nn.init.orthogonal_(weight)
        var, mean = torch.var_mean(weight, dim=1, unbiased=True, keepdim=True)
        weight.sub_(mean).div_(torch.sqrt(var + 1e-6)).mul_(weight.shape[1] ** -0.5)
    return weight


class _Dense(nn.Linear):
    """Bias-free dense layer, optionally followed by the scaled SiLU ``silu(x) / 0.6``."""

    def __init__(self, n_in: int, n_out: int, activation: bool = False,
                 init: str = "he_orthogonal"):
        super().__init__(n_in, n_out, bias=False)
        self.activation = activation
        if init == "zeros":
            nn.init.zeros_(self.weight)
        elif init == "he_orthogonal":
            he_orthogonal_(self.weight)
        else:
            raise ValueError(f"init must be 'he_orthogonal' or 'zeros', got {init!r}")

    def forward(self, x: Tensor) -> Tensor:
        y = nn.functional.linear(x, self.weight)
        if self.activation:
            y = nn.functional.silu(y) * (1.0 / 0.6)
        return y


class _Residual(nn.Module):
    """``(x + sigma(W_2 sigma(W_1 x))) / sqrt(2)``."""

    def __init__(self, n_features: int):
        super().__init__()
        self.lin1 = _Dense(n_features, n_features, activation=True)
        self.lin2 = _Dense(n_features, n_features, activation=True)
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, x: Tensor) -> Tensor:
        return (x + self.lin2(self.lin1(x))) * self.inv_sqrt2


class ScaleFactor(nn.Module):
    """A constant factor that keeps the activation variance of a layer (paper section 6).

    It multiplies the output ``y`` of a sum aggregation or a Hadamard /
    bilinear product; :meth:`GemNet.fit_scaling_factors` sets it so that the
    variance of ``y`` matches that of the layer input ``ref`` on data.

    Attributes
    ----------
    scale_factor : Tensor
        The scalar factor (a buffer, 1 until fitted).
    fitting : bool
        Whether :meth:`forward` accumulates the variances of its inputs.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("scale_factor", torch.tensor(1.0))
        self.register_buffer("stats", torch.zeros(3, dtype=torch.float64), persistent=False)
        self.fitting = False

    def forward(self, ref: Tensor, y: Tensor) -> Tensor:
        y = y * self.scale_factor
        if self.fitting and ref.shape[0] > 1 and y.shape[0] > 1:
            # mean over features of the variance over rows, weighted by the rows of y
            n = float(y.shape[0])
            var_ref = torch.var(ref.detach(), dim=0).mean().to(torch.float64)
            var_y = torch.var(y.detach(), dim=0).mean().to(torch.float64)
            self.stats.add_(torch.stack([var_ref * n, var_y * n,
                                         torch.ones_like(var_y) * n]))
        return y


class _EdgeEmbedding(nn.Module):
    """``sigma(W [h_c || h_a || x_ca])`` for every edge ``c -> a`` (embedding and concat layer)."""

    def __init__(self, n_atom: int, n_in: int, n_out: int):
        super().__init__()
        self.dense = _Dense(2 * n_atom + n_in, n_out, activation=True)

    def forward(self, h: Tensor, x: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        return self.dense(torch.cat([h[src], h[dst], x], dim=-1))


class _AtomUpdate(nn.Module):
    """Atom embedding block: ``h_a = MLP(sum_c m_ca o W e_RBF(x_ca))`` (Fig. 4, "Atom emb.")."""

    def __init__(self, n_atom: int, n_edge: int, n_rbf_emb: int, n_residual: int):
        super().__init__()
        self.lin_rbf = _Dense(n_rbf_emb, n_edge)
        self.scale_sum = ScaleFactor()
        self.dense = _Dense(n_edge, n_atom, activation=True)
        self.residuals = nn.ModuleList([_Residual(n_atom) for _ in range(n_residual)])

    def forward(self, m: Tensor, rbf: Tensor, dst: Tensor,
                num_nodes: int) -> Tuple[Tensor, Tensor]:
        gated = m * self.lin_rbf(rbf)
        h = self.scale_sum(m, scatter_sum(gated, dst, num_nodes))
        h = self.dense(h)
        for res in self.residuals:
            h = res(h)
        return h, gated


class _ForceHead(nn.Module):
    """Per-edge force magnitude of the direct-force variants (paper eq 5)."""

    def __init__(self, n_edge: int, n_residual: int, output_init: str):
        super().__init__()
        self.scale_rbf = ScaleFactor()
        self.dense = _Dense(n_edge, n_edge, activation=True)
        self.residuals = nn.ModuleList([_Residual(n_edge) for _ in range(n_residual)])
        self.final = _Dense(n_edge, 1, init=output_init)

    def forward(self, m: Tensor, gated: Tensor) -> Tensor:
        x = self.dense(self.scale_rbf(m, gated))
        for res in self.residuals:
            x = res(x)
        return self.final(x).squeeze(-1)


class _Output(nn.Module):
    """Output block: an atom embedding block and a final linear layer to ``t_a``."""

    def __init__(self, n_atom: int, n_edge: int, n_rbf_emb: int, n_residual: int,
                 output_init: str, direct_forces: bool):
        super().__init__()
        self.atom = _AtomUpdate(n_atom, n_edge, n_rbf_emb, n_residual)
        self.final = _Dense(n_atom, 1, init=output_init)
        self.forces: Optional[_ForceHead] = (
            _ForceHead(n_edge, n_residual, output_init) if direct_forces else None)

    def forward(self, m: Tensor, rbf: Tensor, dst: Tensor,
                num_nodes: int) -> Tuple[Tensor, Tensor, Tensor]:
        h, gated = self.atom(m, rbf, dst, num_nodes)
        t = self.final(h).squeeze(-1)
        f = m.new_zeros(0)
        if self.forces is not None:
            f = self.forces(m, gated)
        return h, t, f


def _padded_bilinear(x: Tensor, out_idx: Tensor, k_idx: Tensor, num_out: int, kmax: int,
                     sph: Tensor, rad: Tensor, weight: Tensor) -> Tensor:
    """The efficient bilinear layer of paper App. D on zero-padded neighbor matrices.

    ``y_e = sum_k (W1 e_basis(e, k))^T W x_k``, with the basis split into the
    radial part of the output edge (already contracted with ``W1``) and the
    angular part of each neighbor, summed over neighbors before the weights
    are applied.

    Parameters
    ----------
    x : Tensor
        Neighbor features, shape ``(K_total, F)``.
    out_idx, k_idx : Tensor
        Output edge and slot of every neighbor, shape ``(K_total,)``.
    num_out, kmax : int
        Number of output edges and slots per edge.
    sph : Tensor
        Padded angular parts, shape ``(num_out, A, kmax)``.
    rad : Tensor
        ``W1``-contracted radial parts, shape ``(num_out, J, A)``.
    weight : Tensor
        Bilinear weight, shape ``(J, F, F_out)``.

    Returns
    -------
    Tensor
        Shape ``(num_out, F_out)``.
    """
    n_feat = x.shape[1]
    padded = x.new_zeros(num_out, kmax, n_feat)
    padded[out_idx, k_idx] = x
    summed = torch.bmm(sph, padded)                         # (E, A, F): sum over neighbors
    mixed = torch.bmm(rad, summed)                          # (E, J, F): sum over the basis
    j = weight.shape[0]
    return mixed.reshape(num_out, j * n_feat) @ weight.reshape(j * n_feat, weight.shape[2])


class _TripletInteraction(nn.Module):
    """One-hop geometric message passing (T-MP of Fig. 4) with symmetric output."""

    def __init__(self, n_edge: int, n_trip: int, n_rbf_emb: int, n_cbf_emb: int,
                 n_bilinear: int):
        super().__init__()
        self.dense_ba = _Dense(n_edge, n_edge, activation=True)
        self.lin_rbf = _Dense(n_rbf_emb, n_edge)
        self.scale_rbf = ScaleFactor()
        self.down = _Dense(n_edge, n_trip, activation=True)
        self.bilinear = nn.Parameter(torch.empty(n_cbf_emb, n_trip, n_bilinear))
        _bilinear_init_(self.bilinear)
        self.scale_cbf_sum = ScaleFactor()
        self.up_ca = _Dense(n_bilinear, n_edge, activation=True)
        self.up_ac = _Dense(n_bilinear, n_edge, activation=True)
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, m: Tensor, rbf: Tensor, cbf_rad: Tensor, cbf_sph: Tensor,
                t_out: Tensor, t_in: Tensor, t_k: Tensor, rev: Tensor) -> Tensor:
        x = self.dense_ba(m)
        x = self.scale_rbf(x, x * self.lin_rbf(rbf))                # e_RBF(x_ba)
        x = self.down(x)[t_in]
        y = _padded_bilinear(x, t_out, t_k, m.shape[0], cbf_sph.shape[2],
                             cbf_sph, cbf_rad, self.bilinear)       # e_CBF(x_ca, phi_cab)
        y = self.scale_cbf_sum(x, y)
        return (self.up_ca(y) + self.up_ac(y)[rev]) * self.inv_sqrt2


class _QuadrupletInteraction(nn.Module):
    """Two-hop geometric message passing (Q-MP of Fig. 4, eq 11) with symmetric output."""

    def __init__(self, n_edge: int, n_quad: int, n_rbf_emb: int, n_cbf_emb: int,
                 n_sbf_emb: int, n_bilinear: int):
        super().__init__()
        self.dense_db = _Dense(n_edge, n_edge, activation=True)
        self.lin_rbf = _Dense(n_rbf_emb, n_edge)
        self.scale_rbf = ScaleFactor()
        self.down = _Dense(n_edge, n_quad, activation=True)
        self.lin_cbf = _Dense(n_cbf_emb, n_quad)
        self.scale_cbf = ScaleFactor()
        self.bilinear = nn.Parameter(torch.empty(n_sbf_emb, n_quad, n_bilinear))
        _bilinear_init_(self.bilinear)
        self.scale_sbf_sum = ScaleFactor()
        self.up_ca = _Dense(n_bilinear, n_edge, activation=True)
        self.up_ac = _Dense(n_bilinear, n_edge, activation=True)
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, m: Tensor, rbf: Tensor, cbf: Tensor, sbf_rad: Tensor, sbf_sph: Tensor,
                intm_db: Tensor, q_out: Tensor, q_intm: Tensor, q_k: Tensor,
                rev: Tensor) -> Tensor:
        x = self.dense_db(m)
        x = self.scale_rbf(x, x * self.lin_rbf(rbf))                # e_RBF(x_db)
        x = self.down(x)[intm_db]
        x = self.scale_cbf(x, x * self.lin_cbf(cbf))                # e_CBF(x_ba, phi_abd)
        x = x[q_intm]
        y = _padded_bilinear(x, q_out, q_k, m.shape[0], sbf_sph.shape[2],
                             sbf_sph, sbf_rad, self.bilinear)       # e_SBF(x_ca, phi_cab, theta_cabd)
        y = self.scale_sbf_sum(x, y)
        return (self.up_ca(y) + self.up_ac(y)[rev]) * self.inv_sqrt2


def _bilinear_init_(weight: Tensor) -> None:
    """He-orthogonal initialization of a ``(J, F_in, F_out)`` bilinear weight (fan-in ``J F_in``)."""
    j, f_in, f_out = weight.shape
    flat = torch.empty(f_out, j * f_in, dtype=weight.dtype)
    he_orthogonal_(flat)
    with torch.no_grad():
        weight.copy_(flat.t().reshape(j, f_in, f_out))


class _Interaction(nn.Module):
    """Interaction block (Fig. 4): message passing, atom self-interaction, edge update."""

    def __init__(self, n_atom: int, n_edge: int, n_trip: int, n_quad: int,
                 n_rbf_emb: int, n_cbf_emb: int, n_sbf_emb: int, n_bil_trip: int,
                 n_bil_quad: int, n_before_skip: int, n_after_skip: int, n_concat: int,
                 n_atom_residual: int, quadruplets: bool):
        super().__init__()
        self.dense_ca = _Dense(n_edge, n_edge, activation=True)
        self.quad: Optional[_QuadrupletInteraction] = (
            _QuadrupletInteraction(n_edge, n_quad, n_rbf_emb, n_cbf_emb, n_sbf_emb,
                                   n_bil_quad) if quadruplets else None)
        self.trip = _TripletInteraction(n_edge, n_trip, n_rbf_emb, n_cbf_emb, n_bil_trip)
        self.before_skip = nn.ModuleList([_Residual(n_edge) for _ in range(n_before_skip)])
        self.after_skip = nn.ModuleList([_Residual(n_edge) for _ in range(n_after_skip)])
        self.atom_update = _AtomUpdate(n_atom, n_edge, n_rbf_emb, n_atom_residual)
        self.concat = _EdgeEmbedding(n_atom, n_edge, n_edge)
        self.residual_m = nn.ModuleList([_Residual(n_edge) for _ in range(n_concat)])
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.inv_sqrt3 = 1.0 / math.sqrt(3.0)

    def forward(self, h: Tensor, m: Tensor, src: Tensor, dst: Tensor, rev: Tensor,
                rbf3: Tensor, cbf3_rad: Tensor, cbf3_sph: Tensor, t_out: Tensor,
                t_in: Tensor, t_k: Tensor, rbf4: Tensor, cbf4: Tensor, sbf_rad: Tensor,
                sbf_sph: Tensor, intm_db: Tensor, q_out: Tensor, q_intm: Tensor,
                q_k: Tensor, rbf_h: Tensor) -> Tuple[Tensor, Tensor]:
        x_skip = self.dense_ca(m)
        x3 = self.trip(m, rbf3, cbf3_rad, cbf3_sph, t_out, t_in, t_k, rev)
        if self.quad is not None:
            x4 = self.quad(m, rbf4, cbf4, sbf_rad, sbf_sph, intm_db, q_out, q_intm, q_k, rev)
            x = (x_skip + x3 + x4) * self.inv_sqrt3
        else:
            x = (x_skip + x3) * self.inv_sqrt2
        for res in self.before_skip:
            x = res(x)
        m = (m + x) * self.inv_sqrt2
        for res in self.after_skip:
            m = res(m)
        h_new, _ = self.atom_update(m, rbf_h, dst, h.shape[0])
        h = (h + h_new) * self.inv_sqrt2
        m_new = self.concat(h, m, src, dst)
        for res in self.residual_m:
            m_new = res(m_new)
        m = (m + m_new) * self.inv_sqrt2
        return h, m


def _lexsort(cols: List[Tensor]) -> Tensor:
    """Permutation sorting rows by the key columns, highest priority first."""
    idx = torch.arange(cols[0].shape[0], device=cols[0].device)
    for i in range(len(cols) - 1, -1, -1):
        idx = idx[torch.sort(cols[i][idx], stable=True)[1]]
    return idx


def reverse_edges(edge_index: Tensor, edge_vec: Tensor) -> Tensor:
    """The index of the reverse ``a -> c`` of every directed edge ``c -> a``.

    The reverse of an edge connects the same atoms the other way with the
    exactly negated edge vector, so periodic images are told apart.

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``(2, E)`` (row 0 source, row 1 destination).
    edge_vec : Tensor
        Edge vectors ``x_dst - x_src`` (cell shifts included), shape ``(E, 3)``.

    Returns
    -------
    Tensor
        ``rev`` of shape ``(E,)``.

    Raises
    ------
    RuntimeError
        If the edge list is not closed under reversal.
    """
    src, dst = edge_index[0], edge_index[1]
    fwd = _lexsort([src, dst, edge_vec[:, 0], edge_vec[:, 1], edge_vec[:, 2]])
    bwd = _lexsort([dst, src, -edge_vec[:, 0], -edge_vec[:, 1], -edge_vec[:, 2]])
    rev = torch.empty_like(fwd)
    rev[bwd] = fwd
    ok = (src[rev] == dst) & (dst[rev] == src) & (edge_vec[rev] == -edge_vec).all(dim=-1)
    if not bool(ok.all()):
        raise RuntimeError("GemNet needs a neighbor list that contains the reverse "
                           "of every edge with the exactly negated edge vector")
    return rev


def _group_slots(group: Tensor, num_groups: int) -> Tensor:
    """Position of every item within its group, for items sorted by group."""
    counts = torch.bincount(group, minlength=num_groups)
    starts = torch.cumsum(counts, 0) - counts
    return torch.arange(group.shape[0], device=group.device) - starts[group]


def triplets(edge_index: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor, Tensor]:
    """The triplets ``c -> a <- b`` of one-hop message passing (``b != c``).

    Every edge ``c -> a`` is paired with every other edge ``b -> a`` into
    the same atom; the output is grouped by the edge ``c -> a``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``(2, E)``.
    num_nodes : int
        Number of atoms.

    Returns
    -------
    tuple of Tensor
        ``(ca, ba, slot)`` of shape ``(T,)``: the edge being updated, the
        incoming edge, and the position of the triplet within the group of
        its edge ``c -> a``.
    """
    dst = edge_index[1]
    num_edges = dst.shape[0]
    order = torch.sort(dst, stable=True)[1]
    counts = torch.bincount(dst, minlength=num_nodes)
    starts = torch.cumsum(counts, 0) - counts
    n_in = counts[dst]
    ca = torch.repeat_interleave(torch.arange(num_edges, device=dst.device), n_in)
    local = torch.arange(ca.shape[0], device=dst.device) - torch.repeat_interleave(
        torch.cumsum(n_in, 0) - n_in, n_in)
    ba = order[starts[dst[ca]] + local]
    keep = ba != ca
    ca, ba = ca[keep], ba[keep]
    return ca, ba, _group_slots(ca, num_edges)


@register_model("gemnet")
class GemNet(InteratomicPotential):
    """GemNet: geometric message passing (Gasteiger, Becker, Guennemann, NeurIPS 2021).

    Directed edge embeddings ``m_ca`` within the embedding cutoff are updated
    by one-hop messages from ``m_ba`` (distance, angle), two-hop messages
    from ``m_db`` via an interaction edge ``a - b`` (distances, two angles,
    dihedral; ``quadruplets=True``) and atom self-interactions; variance
    scaling factors keep activations normalized. The defaults are the paper's
    GemNet-Q (App. F, Table 6).

    Parameters
    ----------
    n_features : int, optional
        Atom embedding size, by default 128.
    n_edge_features : int, optional
        Edge embedding size, by default 128.
    n_interactions : int, optional
        Interaction blocks, by default 4.
    n_rbf : int, optional
        Radial basis size ``N_RBF`` (and zeros per degree ``N_SRBF`` of the
        circular and spherical bases), by default 6.
    n_spherical : int, optional
        Degrees ``N_SHBF`` of the circular and spherical bases, by default 7.
    cutoff : float, optional
        Embedding cutoff ``c_emb`` in Angstrom, by default 5.0.
    int_cutoff : float, optional
        Interaction cutoff ``c_int`` of the quadruplets' middle edge, by
        default 10.0 (unused without quadruplets).
    p : int, optional
        Exponent of the polynomial envelope, by default 5.
    n_triplet_features, n_quadruplet_features : int, optional
        Down-projected sizes in the triplet / quadruplet interaction, by
        default 64 and 32.
    n_rbf_features, n_cbf_features, n_sbf_features : int, optional
        Sizes of the shared down-projections of the radial, circular and
        spherical bases, by default 16, 16 and 32.
    n_bilinear_triplet, n_bilinear_quadruplet : int, optional
        Output sizes of the bilinear layers, by default 64 and 32.
    n_before_skip, n_after_skip, n_concat : int, optional
        Residual blocks before / after the first skip connection and after the
        atom-edge concatenation, by default 1, 1, 1.
    n_atom_residual : int, optional
        Residual blocks of the atom embedding and output blocks, by default 2.
    quadruplets : bool, optional
        Two-hop message passing (GemNet-Q, default) or triplets only (GemNet-T).
    direct_forces : bool, optional
        Predict forces directly from the edge embeddings (GemNet-dQ / -dT)
        instead of as the energy gradient; by default ``False``.
    forces_coupled : bool, optional
        With ``direct_forces``, average the force magnitudes of ``c -> a``
        and ``a -> c``; by default ``False``.
    output_init : str, optional
        Initialization of the final layers, ``"he_orthogonal"`` (default) or
        ``"zeros"``.
    trainable_rbf : bool, optional
        Learn the Bessel frequencies, by default ``True``.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.

    Attributes
    ----------
    cutoff : float
        Neighbor-list radius, ``max(c_emb, c_int)`` with quadruplets.
    emb_cutoff, int_cutoff : float
        The embedding and interaction cutoffs.
    scaling_fitted : Tensor
        Boolean buffer, whether :meth:`fit_scaling_factors` has run.
    node_feature_dim : int
        Width of the ``"node_features"`` output (the summed output-block atom
        features before their final layer).
    """

    head_modules = ("outputs", "atom_ref")

    def __init__(self, n_features: int = 128, n_edge_features: int = 128,
                 n_interactions: int = 4, n_rbf: int = 6, n_spherical: int = 7,
                 cutoff: float = 5.0, int_cutoff: float = 10.0, p: int = 5,
                 n_triplet_features: int = 64, n_quadruplet_features: int = 32,
                 n_rbf_features: int = 16, n_cbf_features: int = 16,
                 n_sbf_features: int = 32, n_bilinear_triplet: int = 64,
                 n_bilinear_quadruplet: int = 32, n_before_skip: int = 1,
                 n_after_skip: int = 1, n_concat: int = 1, n_atom_residual: int = 2,
                 quadruplets: bool = True, direct_forces: bool = False,
                 forces_coupled: bool = False, output_init: str = "he_orthogonal",
                 trainable_rbf: bool = True, species=None, atomic_energies=None):
        super().__init__()
        if output_init not in ("he_orthogonal", "zeros"):
            raise ValueError(f"output_init must be 'he_orthogonal' or 'zeros', got {output_init!r}")
        self.quadruplets = bool(quadruplets)
        self.direct_forces = bool(direct_forces)
        self.forces_coupled = bool(forces_coupled)
        self.emb_cutoff = float(cutoff)
        self.int_cutoff = float(int_cutoff)
        self.cutoff = max(self.emb_cutoff, self.int_cutoff) if self.quadruplets else self.emb_cutoff
        self.n_spherical = n_spherical
        self.n_rbf = n_rbf
        self.node_feature_dim = n_features
        if species is not None:
            self.species = [int(z) for z in species]

        # bases (eq 8-10) and their envelopes
        self.rbf = BesselRBF(n_rbf, cutoff, trainable=trainable_rbf)
        self.envelope = PolynomialCutoff(cutoff, p)
        self.cbf = SphericalBesselBasis(n_spherical, n_rbf, cutoff)
        self.int_envelope = PolynomialCutoff(int_cutoff, p)
        self.cbf_int = SphericalBesselBasis(n_spherical, n_rbf, int_cutoff)
        # down-projections of the bases, shared by all interaction blocks
        n_cbf = n_spherical * n_rbf
        self.lin_rbf3 = _Dense(n_rbf, n_rbf_features)
        self.lin_cbf3 = _Dense(n_cbf, n_cbf_features)
        self.lin_rbf_h = _Dense(n_rbf, n_rbf_features)
        self.lin_rbf_out = _Dense(n_rbf, n_rbf_features)
        self.lin_rbf4: Optional[_Dense] = None
        self.lin_cbf4: Optional[_Dense] = None
        self.lin_sbf4: Optional[_Dense] = None
        if self.quadruplets:
            self.lin_rbf4 = _Dense(n_rbf, n_rbf_features)
            self.lin_cbf4 = _Dense(n_cbf, n_cbf_features)
            self.lin_sbf4 = _Dense(n_spherical ** 2 * n_rbf, n_sbf_features)
        # degree of every (l, m) column of the spherical basis
        self.register_buffer("sbf_degree", torch.repeat_interleave(
            torch.arange(n_spherical), 2 * torch.arange(n_spherical) + 1), persistent=False)

        self.atom_embedding = nn.Embedding(_MAX_Z, n_features)
        nn.init.uniform_(self.atom_embedding.weight, -math.sqrt(3.0), math.sqrt(3.0))
        self.edge_embedding = _EdgeEmbedding(n_features, n_rbf, n_edge_features)
        self.interactions = nn.ModuleList([
            _Interaction(n_features, n_edge_features, n_triplet_features,
                         n_quadruplet_features, n_rbf_features, n_cbf_features,
                         n_sbf_features, n_bilinear_triplet, n_bilinear_quadruplet,
                         n_before_skip, n_after_skip, n_concat, n_atom_residual,
                         self.quadruplets)
            for _ in range(n_interactions)])
        self.outputs = nn.ModuleList([
            _Output(n_features, n_edge_features, n_rbf_features, n_atom_residual,
                    output_init, self.direct_forces)
            for _ in range(n_interactions + 1)])
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.register_buffer("scaling_fitted", torch.tensor(False))
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)

    @torch.jit.ignore
    def set_atomic_energies(self, species, values) -> None:
        """Initialize the per-element reference energies ``atom_ref``.

        Parameters
        ----------
        species : list of int
            Atomic numbers the values refer to, in order.
        values : array-like
            One reference energy per entry of ``species``.

        Raises
        ------
        ValueError
            If the number of values does not match the number of species.
        """
        ae = torch.as_tensor(values, dtype=self.atom_ref.weight.dtype)
        if ae.numel() != len(list(species)):
            raise ValueError(
                f"got {ae.numel()} atomic energies for {len(list(species))} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(list(species)), 0] = ae

    def _quadruplet_geometry(self, src: Tensor, dst: Tensor, vec: Tensor, r: Tensor,
                             emb_ids: Tensor, edge_index: Tensor, edge_vec: Tensor,
                             r_all: Tensor, rev_all: Tensor, num_nodes: int
                             ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                                        Tensor]:
        """Quadruplets ``c -> a - b <- d`` and their bases (paper section 5).

        Returns the intermediate triplets ``a - b <- d`` (their edge ``d -> b``
        and interaction edge), the interaction-edge distances, the angles
        ``phi_abd`` and, per quadruplet (grouped by ``c -> a``), the edge
        ``c -> a``, the intermediate triplet, the slot, and the spherical
        harmonics of ``(phi_cab, theta_cabd)``.
        """
        num_emb = src.shape[0]
        int_ids = torch.nonzero(r_all <= self.int_cutoff).squeeze(-1)
        b, a = edge_index[0][int_ids], edge_index[1][int_ids]
        v_ba, r_ba = edge_vec[int_ids], r_all[int_ids]
        order = torch.sort(dst, stable=True)[1]                    # embedding edges by atom
        counts = torch.bincount(dst, minlength=num_nodes)
        starts = torch.cumsum(counts, 0) - counts
        n_a, n_b = counts[a], counts[b]
        arange_int = torch.arange(int_ids.shape[0], device=src.device)
        # intermediate triplets a - b <- d: every interaction edge with every d -> b
        intm_q = torch.repeat_interleave(arange_int, n_b)
        intm_first = torch.cumsum(n_b, 0) - n_b
        intm_local = torch.arange(intm_q.shape[0], device=src.device) - intm_first[intm_q]
        intm_db = order[starts[b][intm_q] + intm_local]
        cos_abd = -(v_ba[intm_q] * vec[intm_db]).sum(-1) / (r_ba[intm_q] * r[intm_db])
        # quadruplets: every interaction edge with every pair (c -> a, d -> b)
        n_q = n_a * n_b
        quad_q = torch.repeat_interleave(arange_int, n_q)
        local = torch.arange(quad_q.shape[0], device=src.device) - (torch.cumsum(n_q, 0) - n_q)[quad_q]
        nb_q = n_b[quad_q]
        ia, ib = torch.div(local, nb_q, rounding_mode="floor"), torch.remainder(local, nb_q)
        ca = order[starts[a][quad_q] + ia]
        q_intm = intm_first[quad_q] + ib
        db = intm_db[q_intm]
        # four distinct atoms (periodic images count as distinct): c != b and
        # d != a are the same edges; c != d compares positions
        cd = v_ba[quad_q] + vec[db] - vec[ca]                       # x_c - x_d
        same_cd = (src[ca] == src[db]) & ((cd * cd).sum(-1) < 1e-8)
        keep = ((emb_ids[ca] != int_ids[quad_q]) & (emb_ids[db] != rev_all[int_ids][quad_q])
                & ~same_cd)
        ca, db, quad_q, q_intm = ca[keep], db[keep], quad_q[keep], q_intm[keep]
        perm = torch.sort(ca, stable=True)[1]
        ca, db, quad_q, q_intm = ca[perm], db[perm], quad_q[perm], q_intm[perm]
        q_k = _group_slots(ca, num_emb)
        # angles: polar phi_cab about the axis a -> b, azimuth theta_cabd
        # from the plane of d (unsigned)
        axis = -v_ba[quad_q] / r_ba[quad_q][:, None]                # a -> b
        u_c = -vec[ca] / r[ca][:, None]                             # a -> c
        to_d = -vec[db]                                             # b -> d
        p_d = to_d - (to_d * axis).sum(-1, keepdim=True) * axis
        norm_d = torch.sqrt(torch.clamp((p_d * p_d).sum(-1), min=1e-18))
        e_x = p_d / norm_d[:, None]
        e_y = torch.cross(axis, e_x, dim=-1)
        z = (u_c * axis).sum(-1).clamp(-1.0, 1.0)
        x = (u_c * e_x).sum(-1)
        y = (u_c * e_y).sum(-1).abs()
        # an undefined azimuth (d on the axis a - b) counts as 90 degrees
        flat = norm_d <= 1e-9
        x = torch.where(flat, torch.zeros_like(x), x)
        y = torch.where(flat, torch.sqrt(torch.clamp(1.0 - z * z, min=1e-18)), y)
        sph = real_spherical_harmonics(self.n_spherical - 1, x, y, z)
        return intm_db, intm_q, r_ba, cos_abd, ca, q_intm, q_k, sph

    def _evaluate(self, atomic_numbers: Tensor, edge_index: Tensor,
                  edge_vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Features, per-atom energies and (direct variants) per-atom forces."""
        num_nodes = atomic_numbers.shape[0]
        r_all = torch.linalg.norm(edge_vec, dim=-1)
        rev_all = reverse_edges(edge_index, edge_vec)
        emb_ids = torch.nonzero(r_all <= self.emb_cutoff).squeeze(-1)
        in_emb = torch.full_like(r_all, -1, dtype=torch.long)
        in_emb[emb_ids] = torch.arange(emb_ids.shape[0], device=emb_ids.device)
        src, dst = edge_index[0][emb_ids], edge_index[1][emb_ids]
        vec, r = edge_vec[emb_ids], r_all[emb_ids]
        rev = in_emb[rev_all[emb_ids]]
        num_emb = src.shape[0]
        L, N = self.n_spherical, self.n_rbf

        # radial bases of the embedding edges with the envelope (eq 8, 9, 10)
        env = self.envelope(r)
        rbf = self.rbf(r) * env[:, None]
        rad = self.cbf.radial(r) * env[:, None, None]                # (E, L, N)

        # triplets c -> a <- b: angle phi_cab, circular basis e_CBF(x_ca, phi_cab)
        t_out, t_in, t_k = triplets(torch.stack([src, dst]), num_nodes)
        cos_cab = ((vec[t_out] * vec[t_in]).sum(-1) / (r[t_out] * r[t_in])).clamp(-1.0, 1.0)
        kmax3 = int(t_k.max()) + 1 if t_k.shape[0] > 0 else 0
        sph3 = vec.new_zeros(num_emb, kmax3, L)
        sph3[t_out, t_k] = zonal_harmonics(L - 1, cos_cab)
        cbf3_sph = sph3.transpose(1, 2)
        # the down-projection acts on the radial part only (App. D); its effective
        # weight comes from the layer itself, so adapters such as LoRA apply
        eye = torch.eye(L * N, dtype=rad.dtype, device=rad.device)
        cbf3_rad = torch.einsum("eln,jln->ejl", rad, self.lin_cbf3(eye).t().reshape(-1, L, N))
        rbf3 = self.lin_rbf3(rbf)
        rbf_h = self.lin_rbf_h(rbf)
        rbf_out = self.lin_rbf_out(rbf)

        empty_long = t_out.new_zeros(0)
        rbf4, cbf4 = vec.new_zeros(0), vec.new_zeros(0)
        sbf_rad, sbf_sph = vec.new_zeros(0), vec.new_zeros(0)
        intm_db, q_out, q_intm, q_k = empty_long, empty_long, empty_long, empty_long
        if (self.lin_rbf4 is not None and self.lin_cbf4 is not None
                and self.lin_sbf4 is not None):
            intm_db, intm_q, r_ba, cos_abd, q_out, q_intm, q_k, sph4 = self._quadruplet_geometry(
                src, dst, vec, r, emb_ids, edge_index, edge_vec, r_all, rev_all, num_nodes)
            # e_CBF(x_ba, phi_abd) with the interaction cutoff, per intermediate triplet
            rad_ba = self.cbf_int.radial(r_ba) * self.int_envelope(r_ba)[:, None, None]
            y_abd = zonal_harmonics(L - 1, cos_abd.clamp(-1.0, 1.0))
            cbf4 = self.lin_cbf4((rad_ba[intm_q] * y_abd[:, :, None]).reshape(-1, L * N))
            # e_SBF(x_ca, phi_cab, theta_cabd): radial part per edge, angular per quadruplet
            kmax4 = int(q_k.max()) + 1 if q_k.shape[0] > 0 else 0
            pad4 = vec.new_zeros(num_emb, kmax4, L * L)
            pad4[q_out, q_k] = sph4
            sbf_sph = pad4.transpose(1, 2)
            eye4 = torch.eye(L * L * N, dtype=rad.dtype, device=rad.device)
            sbf_rad = torch.einsum("ekn,jkn->ejk", rad[:, self.sbf_degree],
                                   self.lin_sbf4(eye4).t().reshape(-1, L * L, N))
            rbf4 = self.lin_rbf4(rbf)

        h = self.atom_embedding(atomic_numbers)
        m = self.edge_embedding(h, rbf, src, dst)
        messages: List[Tensor] = [m]
        for block in self.interactions:
            h, m = block(h, m, src, dst, rev, rbf3, cbf3_rad, cbf3_sph, t_out, t_in, t_k,
                         rbf4, cbf4, sbf_rad, sbf_sph, intm_db, q_out, q_intm, q_k, rbf_h)
            messages.append(m)
        features = h.new_zeros(num_nodes, self.node_feature_dim)
        node_energy = self.atom_ref(atomic_numbers).squeeze(-1)
        edge_force = vec.new_zeros(num_emb)
        for i, out in enumerate(self.outputs):
            h_out, t, f = out(messages[i], rbf_out, dst, num_nodes)
            features = features + h_out
            node_energy = node_energy + t
            if self.direct_forces:
                edge_force = edge_force + f
        forces = vec.new_zeros(0, 3)
        if self.direct_forces:
            if self.forces_coupled:                      # |F_ca| = |F_ac|
                edge_force = 0.5 * (edge_force + edge_force[rev])
            forces = scatter_sum(edge_force[:, None] * vec / r[:, None], dst, num_nodes)
        return features, node_energy, forces

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: tensors in, features + per-atom energy out.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)`` (row 0 source ``c``, row 1
            destination ``a``), within :attr:`cutoff`.
        edge_vec : Tensor
            Edge vectors ``x_a - x_c``, shape ``(E, 3)`` (cell shifts included).

        Returns
        -------
        tuple of Tensor
            Per-atom features ``(N, node_feature_dim)`` and energies ``(N,)``.

        Raises
        ------
        RuntimeError
            For the direct-force variants, whose forces are not the energy
            gradient.
        """
        if self.direct_forces:
            raise RuntimeError("a GemNet with direct_forces has no energy-only core "
                               "(its forces are not the gradient of the energy)")
        out = self._evaluate(atomic_numbers, edge_index, edge_vec)
        return out[0], out[1]

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy, shape ``(N,)`` (the deploy entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Compute per-atom and total energies (and direct forces) for a batch.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph built with :attr:`cutoff`.

        Returns
        -------
        dict[str, Tensor]
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``,
            ``"node_features"`` ``(N, node_feature_dim)`` and, with
            ``direct_forces``, ``"forces"`` ``(N, 3)``.
        """
        features, node_energy, forces = self._evaluate(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        out = {"node_energy": node_energy,
               "energy": self.aggregate_energy(node_energy, data),
               "node_features": features}
        if self.direct_forces:
            out["forces"] = forces.to(data.pos.dtype)
        return out

    @torch.jit.ignore
    def scale_factors(self) -> Dict[str, ScaleFactor]:
        """The scaling factors in forward order, keyed by the reference code's names.

        Returns
        -------
        dict[str, ScaleFactor]
            ``QuadInteraction_i_had_rbf``, ``TripInteraction_i_sum_cbf``,
            ``AtomUpdate_i_sum``, ``OutBlock_i_sum``, ... (blocks counted from 1,
            output blocks from 0).
        """
        names: Dict[str, ScaleFactor] = {}

        def output(i):
            names[f"OutBlock_{i}_sum"] = self.outputs[i].atom.scale_sum
            if self.outputs[i].forces is not None:
                names[f"OutBlock_{i}_had"] = self.outputs[i].forces.scale_rbf

        output(0)
        for i, block in enumerate(self.interactions, start=1):
            if block.quad is not None:
                names[f"QuadInteraction_{i}_had_rbf"] = block.quad.scale_rbf
                names[f"QuadInteraction_{i}_had_cbf"] = block.quad.scale_cbf
                names[f"QuadInteraction_{i}_sum_sbf"] = block.quad.scale_sbf_sum
            names[f"TripInteraction_{i}_had_rbf"] = block.trip.scale_rbf
            names[f"TripInteraction_{i}_sum_cbf"] = block.trip.scale_cbf_sum
            names[f"AtomUpdate_{i}_sum"] = block.atom_update.scale_sum
            output(i)
        return names

    @torch.jit.ignore
    def fit_scaling_factors(self, batches) -> Dict[str, float]:
        """Fit the scaling factors on data, one after the other (paper section 6).

        Each factor is set so that the mean feature variance of its output
        matches that of its input over all ``batches``, with the factors
        before it already fitted. Labels are not used.

        Parameters
        ----------
        batches : sequence of AtomicGraph
            Batches of structures (iterated once per factor), on the model's
            device.

        Returns
        -------
        dict[str, float]
            The fitted factors.
        """
        batches = list(batches)
        fitted: Dict[str, float] = {}
        was_training = self.training
        self.eval()
        for name, factor in self.scale_factors().items():
            factor.stats.zero_()
            factor.fitting = True
            try:
                with torch.no_grad():
                    for graph in batches:
                        self(graph)
            finally:
                factor.fitting = False
            var_ref, var_y, n = factor.stats.tolist()
            if n == 0 or var_y == 0:
                raise ValueError(f"no data reached the scaling factor {name}")
            with torch.no_grad():
                factor.scale_factor.mul_(math.sqrt(var_ref / var_y))
            fitted[name] = float(factor.scale_factor)
        self.scaling_fitted.fill_(True)
        self.train(was_training)
        return fitted

    @torch.jit.ignore
    def load_scaling_factors(self, factors) -> None:
        """Set the scaling factors from a dict or a JSON file of the reference code.

        Parameters
        ----------
        factors : dict or str or pathlib.Path
            ``{name: value}`` with the names of :meth:`scale_factors`, or the
            path of a ``scaling_factors.json``; entries of absent blocks (e.g.
            the quadruplet factors for GemNet-T) are ignored.

        Raises
        ------
        KeyError
            If a factor of the model has no value.
        """
        if not isinstance(factors, dict):
            import json
            with open(factors) as f:
                factors = json.load(f)
        with torch.no_grad():
            for name, factor in self.scale_factors().items():
                if name not in factors:
                    raise KeyError(f"no value for the scaling factor {name}")
                factor.scale_factor.fill_(float(factors[name]))
        self.scaling_fitted.fill_(True)

    @classmethod
    def from_config(cls, cfg) -> "GemNet":
        """Build a :class:`GemNet` from a configuration object.

        Core fields: ``cfg.n_features`` (atom embedding size),
        ``cfg.n_interactions``, ``cfg.n_rbf`` and ``cfg.cutoff`` (the
        embedding cutoff). ``cfg.extra`` may hold every other constructor
        option, ``scaling_factors`` (a dict or the path of a reference-code
        JSON file) and the reference code's ``triplets_only`` and
        ``envelope_exponent``; its other key spellings (``emb_size_edge``,
        ``num_atom``, ...) are translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        GemNet
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra["species"])
                   if extra.get("species") is not None else None)
        quadruplets = extra.get("quadruplets")
        if quadruplets is None:
            quadruplets = not bool(extra.get("triplets_only", False))
        p = extra.get("p", extra.get("envelope_exponent", 5))
        output_init = str(extra.get("output_init", "he_orthogonal")).lower()
        output_init = "he_orthogonal" if output_init == "heorthogonal" else output_init
        ints = {key: int(extra[key]) for key in (
            "n_edge_features", "n_spherical", "n_triplet_features", "n_quadruplet_features",
            "n_rbf_features", "n_cbf_features", "n_sbf_features", "n_bilinear_triplet",
            "n_bilinear_quadruplet", "n_before_skip", "n_after_skip", "n_concat",
            "n_atom_residual") if extra.get(key) is not None}
        model = cls(
            n_features=cfg.n_features, n_interactions=cfg.n_interactions, n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff, int_cutoff=float(extra.get("int_cutoff", 10.0)), p=int(p),
            quadruplets=bool(quadruplets),
            direct_forces=bool(extra.get("direct_forces", False)),
            forces_coupled=bool(extra.get("forces_coupled", False)),
            output_init=output_init,
            trainable_rbf=bool(extra.get("trainable_rbf", True)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
            **ints)
        if extra.get("scaling_factors") is not None:
            model.load_scaling_factors(extra["scaling_factors"])
        return model
