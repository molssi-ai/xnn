"""PaiNN (Schuett, Unke, Gastegger, ICML 2021) -- equivariant message passing.

A faithful implementation of the polarizable atom interaction neural network
of

* K. T. Schuett, O. T. Unke, M. Gastegger, "Equivariant message passing for
  the prediction of tensorial properties and molecular spectra", ICML 2021
  (arXiv:2102.03150),

built from the paper's equations on the xnn abstractions
(:class:`~xnn.gnn.featurizers.BesselRBF`,
:class:`~xnn.common.featurizers.CosineCutoff`,
:func:`~xnn.common.models.ops.scatter_sum`, ...) and consistent with the
conventions of the authors' reference code (the ``schnetpack`` package:
feature layout, initialization, numerical stabilization), without copying
it.

Representation (paper section IV.A). Every atom carries ``F`` rotationally
invariant scalar features ``s_i`` and ``F`` equivariant vector features
``v_i`` in ``R^3``. The scalars start as a learned embedding of the nuclear
charge, ``s_i^0 = a_{Z_i}``, the vectors as zero. ``T`` pairs of a *message*
and an *update* block (Fig. 2) then refine both, each adding a residual.

Message block (eq 7 and 8, Fig. 2b)::

    ds_i = sum_j phi_s(s_j) o W_s(r_ij)
    dv_i = sum_j v_j o phi_vv(s_j) o W_vv(r_ij)
         + sum_j phi_vs(s_j) o W_vs(r_ij) r_ij / |r_ij|

where ``r_ij = r_j - r_i``, the shared network ``phi = W silu(W s + b) + b``
maps ``F`` to ``3F`` features that are split three ways, and the filters
``W(r) = W rbf(r) f_cut(r)`` are linear combinations of the radial basis
``sin(n pi r / r_cut) / r``, ``1 <= n <= 20``, times the cosine cutoff.

Update block (eq 9 and 10, Fig. 2c)::

    ds_i = a_ss(s_i, |V v_i|) + a_sv(s_i, |V v_i|) <U v_i, V v_i>
    dv_i = a_vv(s_i, |V v_i|) U v_i

with the linear (bias-free) maps ``U`` and ``V`` of the vector features and
the shared network ``a = W silu(W [s_i, |V v_i|] + b) + b`` from ``2F`` to
``3F`` features, split into ``a_vv``, ``a_sv``, ``a_ss``.

Readout (Fig. 2a): ``E = sum_i W silu(W s_i + b) + b`` with widths ``F ->
F/2 -> 1``, plus the per-atom standardization and per-element reference of
every xnn model (``energy_scale``, ``energy_shift``, ``atom_ref``).

Tensorial properties (section IV.B). With ``dipole=True`` the dipole moment
is built from latent charges and atomic dipoles (eq 13)::

    mu = sum_i mu_i(v_i) + q_i(s_i) r_i

and with ``polarizability=True`` the polarizability tensor from the rank-1
decomposition of eq 14::

    alpha = sum_i alpha_i(s_i) I + nu_i(v_i) (x) r_i + r_i (x) nu_i(v_i),

where the scalars and vectors come from a stack of two gated equivariant
blocks (Fig. 3) and ``r_i`` are the positions relative to the center of mass
of the structure (the paper assumes the center of mass at the origin). The
latent charges are shifted so that they sum to the structure's net charge
(``total_charge``, zero by default), as in the reference code.

Conventions worth knowing:

* xnn edge vectors are ``pos[dst] - pos[src] = r_i - r_j``; the paper's
  direction ``r_ij / |r_ij|`` with ``r_ij = r_j - r_i`` is their negative,
  so the model flips the sign. The sign is observable only through the
  vector features (the dipoles and polarizabilities), not the energy.
* The norm in the update block is the stabilized ``sqrt(|V v|^2 + epsilon)``
  of the reference code (``epsilon = 1e-8``), so the gradient of an atom
  without neighbors (``v = 0``) is finite.
* The radial basis is the paper's Bessel form by default; the
  ``schnetpack`` default, Gaussians every ``r_cut / (n_rbf - 1)`` with the
  spacing as width, is available as ``radial_basis="gaussian"``.
* The ablations of the paper's Table IV are options: ``scalar_product``
  (the ``<U v, V v>`` term of eq 9) and ``vector_propagation`` (the
  ``v_j o phi_vv`` term of eq 8).

Works for molecules and periodic systems through ``data.edge_vectors()``
(the energy, forces and stress); the dipole and polarizability heads use
absolute positions and are meant for molecules.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve (the scriptable
# `node_energy` core is what LAMMPS deployment uses).
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, GaussianRBF
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import scatter_sum
from xnn.common.models.registry import register_model
from xnn.gnn.featurizers import BesselRBF

_MAX_Z = 100

#: Standard atomic weights (u) by atomic number, Z = 0 to 99 (the ASE table),
#: for the center of mass of the tensorial heads.
ATOMIC_MASSES = [
    1.0000, 1.0080, 4.0026, 6.9400, 9.0122, 10.8100, 12.0110, 14.0070, 15.9990, 18.9984,
    20.1797, 22.9898, 24.3050, 26.9815, 28.0850, 30.9738, 32.0600, 35.4500, 39.9480, 39.0983,
    40.0780, 44.9559, 47.8670, 50.9415, 51.9961, 54.9380, 55.8450, 58.9332, 58.6934, 63.5460,
    65.3800, 69.7230, 72.6300, 74.9216, 78.9710, 79.9040, 83.7980, 85.4678, 87.6200, 88.9058,
    91.2240, 92.9064, 95.9500, 97.9072, 101.0700, 102.9055, 106.4200, 107.8682, 112.4140, 114.8180,
    118.7100, 121.7600, 127.6000, 126.9045, 131.2930, 132.9055, 137.3270, 138.9055, 140.1160, 140.9077,
    144.2420, 144.9128, 150.3600, 151.9640, 157.2500, 158.9254, 162.5000, 164.9303, 167.2590, 168.9342,
    173.0540, 174.9668, 178.4900, 180.9479, 183.8400, 186.2070, 190.2300, 192.2170, 195.0840, 196.9666,
    200.5920, 204.3800, 207.2000, 208.9804, 208.9824, 209.9872, 222.0176, 223.0197, 226.0254, 227.0277,
    232.0377, 231.0359, 238.0289, 237.0482, 244.0642, 243.0614, 247.0703, 247.0703, 251.0796, 252.0830,
]


def center_of_mass(pos: Tensor, atomic_numbers: Tensor, batch: Tensor,
                   num_graphs: int, masses: Tensor) -> Tensor:
    """Center of mass of every structure of a batch.

    Parameters
    ----------
    pos : Tensor
        Positions ``(N, 3)``.
    atomic_numbers : Tensor
        Atomic numbers ``(N,)``.
    batch : Tensor
        Structure index of every atom ``(N,)``.
    num_graphs : int
        Number of structures ``B``.
    masses : Tensor
        Mass table indexed by atomic number.

    Returns
    -------
    Tensor
        The centers of mass ``(B, 3)``.
    """
    m = masses[atomic_numbers].to(pos.dtype).unsqueeze(-1)
    total = scatter_sum(m, batch, num_graphs)
    return scatter_sum(m * pos, batch, num_graphs) / total


class GatedEquivariantBlock(nn.Module):
    """Gated equivariant block (paper Fig. 3).

    Two linear maps of the vector features give ``W_1 v`` and ``W_2 v``; the
    norm of the first is stacked with the scalars and fed through a two-layer
    network whose output is split into new scalars and a gate that scales
    ``W_2 v``. Scalars pass through a nonlinearity, vectors are only scaled,
    so the block is equivariant.

    Parameters
    ----------
    n_in_s, n_in_v : int
        Input scalar and vector widths.
    n_out_s, n_out_v : int
        Output scalar and vector widths.
    n_hidden : int or None, optional
        Width of the hidden layer of the scalar network, by default
        ``n_in_s``.
    scalar_activation : bool, optional
        Whether a SiLU is applied to the output scalars (the hidden blocks of
        a stack); off for the last block. Default ``True``.
    """

    def __init__(self, n_in_s: int, n_in_v: int, n_out_s: int, n_out_v: int,
                 n_hidden: Optional[int] = None, scalar_activation: bool = True):
        super().__init__()
        n_hidden = n_in_s if n_hidden is None else int(n_hidden)
        self.n_out_s, self.n_out_v = n_out_s, n_out_v
        self.lin_v = nn.Linear(n_in_v, 2 * n_out_v, bias=False)
        self.net = nn.Sequential(
            nn.Linear(n_in_s + n_out_v, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_out_s + n_out_v))
        self.scalar_activation = scalar_activation

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform scalar and vector features.

        Parameters
        ----------
        s : Tensor
            Scalar features ``(N, n_in_s)``.
        v : Tensor
            Vector features ``(N, 3, n_in_v)``.

        Returns
        -------
        tuple of Tensor
            New scalars ``(N, n_out_s)`` and vectors ``(N, 3, n_out_v)``.
        """
        parts = torch.split(self.lin_v(v), self.n_out_v, dim=-1)
        v1, v2 = parts[0], parts[1]
        x = self.net(torch.cat([s, torch.linalg.norm(v1, dim=1)], dim=-1))
        s_out, gate = x[:, :self.n_out_s], x[:, self.n_out_s:]
        v_out = gate.unsqueeze(1) * v2
        if self.scalar_activation:
            s_out = torch.nn.functional.silu(s_out)
        return s_out, v_out


class _Message(nn.Module):
    """The message block of eq 7 and 8 (Fig. 2b).

    The shared network ``phi`` and the filter layer produce one ``F``-wide
    part per term in the order ``(s, vv, vs)``; an ablated term has no part
    (and no parameters).

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions the filter is built from.
    vector_propagation : bool
        Whether the ``v_j o phi_vv o W_vv`` term of eq 8 is included.
    vectors : bool
        Whether the model carries vector features at all (``False``: only
        the scalar message of eq 7).
    filter_layer : torch.nn.Linear or None
        A filter-generating layer shared with the other blocks; ``None``
        builds this block's own.
    """

    def __init__(self, n_features: int, n_rbf: int, vector_propagation: bool = True,
                 vectors: bool = True, filter_layer: Optional[nn.Linear] = None):
        super().__init__()
        self.n_features = n_features
        self.vectors = vectors
        self.vector_propagation = vectors and vector_propagation
        self.n_parts = 1 + int(vectors) + int(self.vector_propagation)
        self.phi = nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(),
            nn.Linear(n_features, self.n_parts * n_features))
        self.filter = (nn.Linear(n_rbf, self.n_parts * n_features)
                       if filter_layer is None else filter_layer)

    def forward(self, s: Tensor, v: Tensor, rbf: Tensor, fcut: Tensor,
                direction: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the residuals ``ds_i`` and ``dv_i`` of every atom.

        Parameters
        ----------
        s : Tensor
            Scalar features ``(N, F)``.
        v : Tensor
            Vector features ``(N, 3, F)``.
        rbf : Tensor
            Radial basis expansion of the edge lengths ``(E, n_rbf)``.
        fcut : Tensor
            Cosine cutoff of the edge lengths ``(E,)``.
        direction : Tensor
            Unit vectors ``r_ij / |r_ij|`` with ``r_ij = r_j - r_i``, ``(E, 3)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the center ``i``.

        Returns
        -------
        tuple of Tensor
            ``ds`` of shape ``(N, F)`` and ``dv`` of shape ``(N, 3, F)``.
        """
        src, dst = edge_index[0], edge_index[1]
        n = s.shape[0]
        W = self.filter(rbf) * fcut.unsqueeze(-1)                  # (E, parts F)
        parts = torch.split(self.phi(s)[src] * W, self.n_features, dim=-1)
        ds = scatter_sum(parts[0], dst, n)
        if not self.vectors:
            return ds, torch.zeros_like(v)
        dv = parts[self.n_parts - 1].unsqueeze(1) * direction.unsqueeze(-1)   # (E, 3, F)
        if self.vector_propagation:
            dv = dv + v[src] * parts[1].unsqueeze(1)
        return ds, scatter_sum(dv, dst, n)


class _Update(nn.Module):
    """The update block of eq 9 and 10 (Fig. 2c).

    The network ``a`` produces one ``F``-wide part per term in the order
    ``(vv, sv, ss)``; an ablated term has no part.

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    epsilon : float
        Stabilizer of the norm, ``sqrt(|V v|^2 + epsilon)``.
    scalar_product : bool
        Whether the ``a_sv <U v, V v>`` term of eq 9 is included.
    vectors : bool
        Whether the model carries vector features at all (``False``: the
        update is ``ds = a_ss(s)`` alone).
    """

    def __init__(self, n_features: int, epsilon: float = 1e-8,
                 scalar_product: bool = True, vectors: bool = True):
        super().__init__()
        self.n_features = n_features
        self.vectors = vectors
        self.scalar_product = vectors and scalar_product
        self.n_parts = 1 + int(vectors) + int(self.scalar_product)
        # [U, V]; nothing to mix without vector features
        self.lin_v = (nn.Linear(n_features, 2 * n_features, bias=False) if vectors
                      else nn.Identity())
        self.net = nn.Sequential(
            nn.Linear((1 + int(vectors)) * n_features, n_features), nn.SiLU(),
            nn.Linear(n_features, self.n_parts * n_features))
        self.epsilon = epsilon

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the residuals ``ds_i`` and ``dv_i`` of every atom.

        Parameters
        ----------
        s : Tensor
            Scalar features ``(N, F)``.
        v : Tensor
            Vector features ``(N, 3, F)``.

        Returns
        -------
        tuple of Tensor
            ``ds`` of shape ``(N, F)`` and ``dv`` of shape ``(N, 3, F)``.
        """
        if not self.vectors:
            return self.net(s), torch.zeros_like(v)
        mixed = torch.split(self.lin_v(v), self.n_features, dim=-1)
        uv, vv = mixed[0], mixed[1]
        norm = torch.sqrt((vv * vv).sum(dim=1) + self.epsilon)      # (N, F)
        parts = torch.split(self.net(torch.cat([s, norm], dim=-1)), self.n_features, dim=-1)
        dv = parts[0].unsqueeze(1) * uv
        ds = parts[self.n_parts - 1]
        if self.scalar_product:
            ds = ds + parts[1] * (uv * vv).sum(dim=1)
        return ds, dv


class _Interaction(nn.Module):
    """One message block followed by one update block, both residual.

    Parameters
    ----------
    message : _Message
        The message block.
    update : _Update
        The update block.
    """

    def __init__(self, message: _Message, update: _Update):
        super().__init__()
        self.message = message
        self.update = update

    def forward(self, s: Tensor, v: Tensor, rbf: Tensor, fcut: Tensor,
                direction: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply the two residual blocks and return the new ``(s, v)``."""
        ds, dv = self.message(s, v, rbf, fcut, direction, edge_index)
        s, v = s + ds, v + dv
        ds, dv = self.update(s, v)
        return s + ds, v + dv


class _TensorHead(nn.Module):
    """A stack of gated equivariant blocks ending in one scalar and one vector per atom.

    The widths halve from block to block (``F -> F/2 -> ... -> 1``), as in
    the reference code's output networks; the last block applies no scalar
    nonlinearity.

    Parameters
    ----------
    n_features : int
        Input width ``F`` of both scalars and vectors.
    n_blocks : int
        Number of gated equivariant blocks (two in the paper).
    """

    def __init__(self, n_features: int, n_blocks: int = 2):
        super().__init__()
        widths = [n_features]
        for _ in range(n_blocks - 1):
            widths.append(max(1, widths[-1] // 2))
        widths.append(1)
        self.blocks = nn.ModuleList([
            GatedEquivariantBlock(widths[i], widths[i], widths[i + 1], widths[i + 1],
                                  n_hidden=widths[i],
                                  scalar_activation=i < n_blocks - 1)
            for i in range(n_blocks)])

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Map the representation to one scalar ``(N,)`` and one vector ``(N, 3)`` per atom."""
        for block in self.blocks:
            s, v = block(s, v)
        return s.squeeze(-1), v.squeeze(-1)


@register_model("painn")
class PaiNN(InteratomicPotential):
    """PaiNN interatomic potential with optional dipole and polarizability heads.

    Faithful to the ICML 2021 manuscript (see the module docstring for the
    equation-by-equation walk-through): embeds atoms by nuclear charge,
    refines scalar and vector features through ``T`` message/update block
    pairs and reads out per-atom energies through a two-layer network with a
    SiLU nonlinearity. Per-atom energies are standardized (``energy_scale``,
    ``energy_shift``) and shifted by the per-element reference ``atom_ref``,
    then sum-pooled into the total energy.

    The defaults reproduce the paper's architecture: ``F = 128`` features,
    ``T = 3`` blocks, 20 Bessel radial functions with a cosine cutoff at
    5 Angstrom.

    Parameters
    ----------
    n_features : int, optional
        Width ``F`` of the scalar and vector features, by default 128.
    n_interactions : int, optional
        Number of message/update block pairs ``T``, by default 3.
    n_rbf : int, optional
        Number of radial basis functions, by default 20.
    cutoff : float, optional
        Neighbor-list radius and cosine cutoff (Angstrom), by default 5.0.
    radial_basis : str, optional
        ``"bessel"`` (default; the paper's ``sin(n pi r / r_cut) / r``) or
        ``"gaussian"`` (the reference code's default).
    shared_filters : bool, optional
        Use one filter-generating layer for all blocks instead of one per
        block. Default ``False``.
    epsilon : float, optional
        Stabilizer of the vector norm in the update block, by default 1e-8.
    dipole : bool, optional
        Add the dipole head of eq 13 (outputs ``"dipole"`` ``(B, 3)`` and the
        latent ``"charges"`` ``(N,)``). Default ``False``.
    polarizability : bool, optional
        Add the polarizability head of eq 14 (output ``"polarizability"``
        ``(B, 3, 3)``). Default ``False``.
    n_output_blocks : int, optional
        Gated equivariant blocks per tensorial head, by default 2.
    correct_charges : bool, optional
        Shift the latent charges so they sum to the structure's net charge
        (``total_charge``, zero when absent). Default ``True``.
    atomic_dipoles : bool, optional
        Include the atomic dipoles ``mu_i(v_i)`` of eq 13 in the dipole
        moment. ``False`` keeps the latent charges alone (eq 12), the
        comparison of the paper's spectra section. Default ``True``.
    scalar_product : bool, optional
        Keep the ``<U v, V v>`` term of eq 9 (Table IV ablation when off).
    vector_propagation : bool, optional
        Keep the ``v_j o phi_vv`` term of eq 8 (Table IV ablation when off).
    vector_features : bool, optional
        Keep the vector features at all; ``False`` is the invariant model of
        the last Table IV row (scalar messages of eq 7 and ``ds = a_ss(s)``
        updates; the tensorial heads then see zero vectors).
        Ablated terms carry no parameters.
    energy_shift : float, optional
        Additive per-atom energy standardization, by default 0.0.
    energy_scale : float, optional
        Multiplicative per-atom energy standardization, by default 1.0.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``; the model handles all
        elements up to Z = 99.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.

    Attributes
    ----------
    cutoff : float
        Neighbor-list cutoff radius.
    embedding : torch.nn.Embedding
        Nuclear-charge-to-feature embedding ``a_Z``.
    rbf : torch.nn.Module
        The radial basis (:class:`~xnn.gnn.featurizers.BesselRBF` or
        :class:`~xnn.common.featurizers.GaussianRBF`).
    cutoff_fn : CosineCutoff
        The cosine cutoff multiplying every filter.
    interactions : torch.nn.ModuleList
        The ``T`` message/update block pairs (``message`` and ``update``
        submodules).
    readout : torch.nn.Sequential
        Atom-wise energy network ``F -> F/2 -> 1``.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference, initialized to zero.
    dipole_head, polarizability_head : torch.nn.Module or None
        The tensorial heads, when requested.
    """

    head_modules = ("readout", "energy_scale", "energy_shift", "atom_ref")

    def __init__(self, n_features: int = 128, n_interactions: int = 3,
                 n_rbf: int = 20, cutoff: float = 5.0,
                 radial_basis: str = "bessel", shared_filters: bool = False,
                 epsilon: float = 1e-8, dipole: bool = False,
                 polarizability: bool = False, n_output_blocks: int = 2,
                 correct_charges: bool = True, atomic_dipoles: bool = True,
                 scalar_product: bool = True,
                 vector_propagation: bool = True, vector_features: bool = True,
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        if radial_basis not in ("bessel", "gaussian"):
            raise ValueError(
                f"radial_basis must be 'bessel' or 'gaussian', got {radial_basis!r}")
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.n_interactions = n_interactions
        self.shared_filters = shared_filters
        self.correct_charges = correct_charges
        self.atomic_dipoles = atomic_dipoles
        self.embedding = nn.Embedding(_MAX_Z, n_features)
        if radial_basis == "bessel":
            # the paper's sin(n pi r / r_cut) / r without a normalization
            self.rbf = BesselRBF(n_rbf, cutoff, trainable=False, prefactor=1.0)
        else:
            self.rbf = GaussianRBF(n_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        n_parts = 1 + int(vector_features) + int(vector_features and vector_propagation)
        shared = nn.Linear(n_rbf, n_parts * n_features) if shared_filters else None
        self.interactions = nn.ModuleList([
            _Interaction(_Message(n_features, n_rbf, vector_propagation, vector_features,
                                  shared),
                         _Update(n_features, epsilon, scalar_product, vector_features))
            for _ in range(n_interactions)])
        self.readout = nn.Sequential(
            nn.Linear(n_features, n_features // 2), nn.SiLU(),
            nn.Linear(n_features // 2, 1))
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.dipole_head = _TensorHead(n_features, n_output_blocks) if dipole else None
        self.polarizability_head = (_TensorHead(n_features, n_output_blocks)
                                    if polarizability else None)
        self.register_buffer("masses", torch.tensor(ATOMIC_MASSES, dtype=torch.float64))
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)

    @torch.jit.ignore
    def set_energy_scale_shift(self, scale: float, shift: float) -> None:
        """Set the per-atom energy standardization from training statistics.

        Parameters
        ----------
        scale : float
            Standard deviation of the training-set energy per atom.
        shift : float
            Mean training-set energy per atom.
        """
        with torch.no_grad():
            self.energy_scale.fill_(float(scale))
            self.energy_shift.fill_(float(shift))

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

    @torch.jit.export
    def representation(self, atomic_numbers: Tensor, edge_index: Tensor,
                       edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible trunk: the scalar and vector features of every atom.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the
            center ``i``.
        edge_vec : Tensor
            Edge vectors ``pos[i] - pos[j]`` of shape ``(E, 3)`` (with any
            periodic image shift applied).

        Returns
        -------
        tuple of Tensor
            Scalars ``s`` of shape ``(N, F)`` and vectors ``v`` of shape
            ``(N, 3, F)`` after the last update block.
        """
        r = torch.linalg.norm(edge_vec, dim=-1)
        # the paper's r_ij = r_j - r_i is the negative of the xnn edge vector
        direction = -edge_vec / r.clamp(min=1e-12).unsqueeze(-1)
        rbf = self.rbf(r)
        fcut = self.cutoff_fn(r)
        s = self.embedding(atomic_numbers)
        v = torch.zeros((s.shape[0], 3, s.shape[1]), dtype=s.dtype, device=s.device)
        for block in self.interactions:
            s, v = block(s, v, rbf, fcut, direction, edge_index)
        return s, v

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: scalar features and per-atom energies.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.

        Returns
        -------
        tuple of Tensor
            The scalar features ``(N, F)`` and the per-atom energy ``(N,)``
            (standardized and including the per-element reference).
        """
        s, _ = self.representation(atomic_numbers, edge_index, edge_vec)
        return s, self._energy(s, atomic_numbers)

    def _energy(self, s: Tensor, atomic_numbers: Tensor) -> Tensor:
        """Per-atom energies from the scalar features."""
        return (self.energy_scale * self.readout(s).squeeze(-1)
                + self.energy_shift + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy ``(N,)`` (the LAMMPS / TorchScript entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies and, when the heads are present, dipoles and polarizabilities.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``,
            ``"node_features"`` (the scalars, ``(N, F)``), ``"node_vectors"``
            (``(N, 3, F)``) and, with the heads, ``"dipole"`` ``(B, 3)``
            with ``"charges"`` ``(N,)`` and ``"polarizability"`` ``(B, 3, 3)``.
        """
        s, v = self.representation(data.atomic_numbers, data.edge_index,
                                   data.edge_vectors())
        node_energy = self._energy(s, data.atomic_numbers)
        out = {"node_energy": node_energy,
               "energy": self.aggregate_energy(node_energy, data),
               "node_features": s, "node_vectors": v}
        if self.dipole_head is None and self.polarizability_head is None:
            return out
        n_graphs = data.num_graphs
        pos = data.pos.to(s.dtype)
        com = center_of_mass(pos, data.atomic_numbers, data.batch, n_graphs, self.masses)
        r = pos - com[data.batch]
        if self.dipole_head is not None:
            q, mu_atom = self.dipole_head(s, v)
            if self.correct_charges:
                total = (data.total_charge.to(q.dtype) if data.total_charge is not None
                         else torch.zeros(n_graphs, dtype=q.dtype, device=q.device))
                surplus = total - scatter_sum(q, data.batch, n_graphs)
                q = q + (surplus / data.n_atoms.to(q.dtype))[data.batch]
            out["charges"] = q
            mu = q.unsqueeze(-1) * r
            if self.atomic_dipoles:
                mu = mu + mu_atom
            out["dipole"] = scatter_sum(mu, data.batch, n_graphs)
        if self.polarizability_head is not None:
            a0, nu = self.polarizability_head(s, v)
            eye = torch.eye(3, dtype=s.dtype, device=s.device)
            outer = nu.unsqueeze(-1) * r.unsqueeze(-2)                  # nu (x) r
            alpha = a0[:, None, None] * eye + outer + outer.transpose(-1, -2)
            out["polarizability"] = scatter_sum(alpha, data.batch, n_graphs)
        return out

    @classmethod
    def from_config(cls, cfg) -> "PaiNN":
        """Build a :class:`PaiNN` from a configuration object.

        Core fields: ``cfg.n_features``, ``cfg.n_interactions``,
        ``cfg.n_rbf``, ``cfg.cutoff``. Everything else is read from
        ``cfg.extra`` (``radial_basis``, ``shared_filters``, ``epsilon``,
        ``dipole``, ``polarizability``, ``n_output_blocks``,
        ``correct_charges``, ``atomic_dipoles``, ``scalar_product``,
        ``vector_propagation``, ``vector_features``,
        ``energy_shift``, ``energy_scale``, ``species``,
        ``atomic_energies``); the reference code's key spellings are
        translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        PaiNN
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            radial_basis=str(extra.get("radial_basis", "bessel")),
            shared_filters=bool(extra.get("shared_filters", False)),
            epsilon=float(extra.get("epsilon", 1e-8)),
            dipole=bool(extra.get("dipole", False)),
            polarizability=bool(extra.get("polarizability", False)),
            n_output_blocks=int(extra.get("n_output_blocks", 2)),
            correct_charges=bool(extra.get("correct_charges", True)),
            atomic_dipoles=bool(extra.get("atomic_dipoles", True)),
            scalar_product=bool(extra.get("scalar_product", True)),
            vector_propagation=bool(extra.get("vector_propagation", True)),
            vector_features=bool(extra.get("vector_features", True)),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
        )
