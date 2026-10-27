"""PaiNN (Schuett, Unke, Gastegger, ICML 2021) -- equivariant message passing.

A faithful implementation of the polarizable atom interaction neural network
of

* K. T. Schuett, O. T. Unke, M. Gastegger, "Equivariant message passing for
  the prediction of tensorial properties and molecular spectra", ICML 2021
  (arXiv:2102.03150),

built from the paper's equations on the xnn abstractions
(:func:`~xnn.common.featurizers.make_radial_basis`,
:class:`~xnn.common.featurizers.CosineCutoff`,
:func:`~xnn.common.models.ops.scatter_sum`, ...).

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
``W(r) = W rbf(r) f_cut(r)`` are linear combinations of a radial basis times
the cosine cutoff. The paper's basis is ``sin(n pi r / r_cut) / r``,
``1 <= n <= 20`` (``radial_basis="bessel"``); the default is 20 Gaussians on
``[0, r_cut]``, and any basis of
:func:`~xnn.common.featurizers.make_radial_basis` can be used.

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
is built from latent charges (eq 12, the default), computed from the scalars
by an atom-wise network ``F -> F/2 -> 1``::

    mu = sum_i q_i(s_i) r_i

or, with ``atomic_dipoles=True``, from latent charges and atomic dipoles
(eq 13, the paper's dipole model)::

    mu = sum_i mu_i(v_i) + q_i(s_i, v_i) r_i

and with ``polarizability=True`` the polarizability tensor from the rank-1
decomposition of eq 14::

    alpha = sum_i alpha_i(s_i) I + nu_i(v_i) (x) r_i + r_i (x) nu_i(v_i),

where the scalars and vectors of eq 13 and 14 come from a stack of two gated
equivariant blocks (Fig. 3) and ``r_i`` are the positions relative to the center of mass
of the structure (the paper assumes the center of mass at the origin). The
latent charges are shifted so that they sum to the structure's net charge
(``total_charge``, zero by default); ``correct_charges=False`` leaves them
unconstrained.

Conventions worth knowing:

* xnn edge vectors are ``pos[dst] - pos[src] = r_i - r_j``; the paper's
  direction ``r_ij / |r_ij|`` with ``r_ij = r_j - r_i`` is their negative,
  so the model flips the sign. The sign is observable only through the
  vector features (the dipoles and polarizabilities), not the energy.
* The norm in the update block is stabilized, ``sqrt(|V v|^2 + epsilon)``
  with ``epsilon = 1e-8``, so that the gradient of an atom without
  neighbors (``v = 0``) is finite.
* The ablations of the paper's Table IV are options: ``scalar_product``
  (the ``<U v, V v>`` term of eq 9) and ``vector_propagation`` (the
  ``v_j o phi_vv`` term of eq 8).
* Weight sharing across the blocks is optional: ``shared_filters`` (the
  same filters for every block) and ``shared_interactions`` (one set of
  message and update networks, with filters per block unless they are
  shared too).

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
from xnn.common.featurizers import CosineCutoff, make_radial_basis
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.blocks import GatedEquivariantBlock
from xnn.common.models.constants import ExactFloat64Constants
from xnn.common.models.ops import ATOMIC_MASSES, center_of_mass, init_linears_, scatter_sum
from xnn.common.models.registry import register_model

_MAX_Z = 100


class _Message(nn.Module):
    """The message block of eq 7 and 8 (Fig. 2b).

    The shared network ``phi`` and the filters (made for all blocks at once,
    :attr:`PaiNN.filters`) have one ``F``-wide part per term in the order
    ``(s, vv, vs)``; an ablated term has no part (and no parameters).

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    vector_propagation : bool
        Whether the ``v_j o phi_vv o W_vv`` term of eq 8 is included.
    vectors : bool
        Whether the model carries vector features at all (``False``: only
        the scalar message of eq 7).
    phi : torch.nn.Sequential or None
        The network ``phi`` of another block to share; ``None`` builds this
        block's own.
    """

    def __init__(self, n_features: int, vector_propagation: bool = True,
                 vectors: bool = True, phi: Optional[nn.Sequential] = None):
        super().__init__()
        self.n_features = n_features
        self.vectors = vectors
        self.vector_propagation = vectors and vector_propagation
        self.n_parts = 1 + int(vectors) + int(self.vector_propagation)
        self.phi = phi if phi is not None else nn.Sequential(
            nn.Linear(n_features, n_features), nn.SiLU(),
            nn.Linear(n_features, self.n_parts * n_features))

    def forward(self, s: Tensor, v: Tensor, W: Tensor, direction: Tensor,
                edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the residuals ``ds_i`` and ``dv_i`` of every atom.

        Parameters
        ----------
        s : Tensor
            Scalar features ``(N, F)``.
        v : Tensor
            Vector features ``(N, 3, F)``.
        W : Tensor
            This block's filters ``W(r_ij)``, the cutoff included, ``(E, parts F)``.
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

    def forward(self, s: Tensor, v: Tensor, W: Tensor, direction: Tensor,
                edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply the two residual blocks and return the new ``(s, v)``."""
        ds, dv = self.message(s, v, W, direction, edge_index)
        s, v = s + ds, v + dv
        ds, dv = self.update(s, v)
        return s + ds, v + dv


def _stack_block_filters(module, state_dict, prefix, *args) -> None:
    """Load a state dict with one filter layer per block (xnn 0.22 and earlier).

    Their weights and biases are stacked into :attr:`PaiNN.filters` (the first
    block's alone when the filters are shared).
    """
    old = [f"{prefix}interactions.{k}.message.filter." for k in range(len(module.interactions))]
    if old[0] + "weight" not in state_dict or prefix + "filters.weight" in state_dict:
        return
    for name in ("weight", "bias"):
        parts = [state_dict.pop(key + name) for key in old]
        state_dict[prefix + "filters." + name] = parts[0] if module.shared_filters else torch.cat(parts)


def _mlp(n_features: int, n_layers: int) -> nn.Sequential:
    """An atom-wise network ``F -> F/2 -> ... -> 1`` with SiLU between the layers."""
    widths = [n_features]
    for _ in range(n_layers - 1):
        widths.append(max(1, widths[-1] // 2))
    widths.append(1)
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < n_layers - 1:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class _TensorHead(nn.Module):
    """A stack of gated equivariant blocks ending in one scalar and one vector per atom.

    The widths halve from block to block (``F -> F/2 -> ... -> 1``); the
    last block applies no scalar nonlinearity.

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
class PaiNN(ExactFloat64Constants, InteratomicPotential):
    """PaiNN interatomic potential with optional dipole and polarizability heads.

    Faithful to the ICML 2021 manuscript (see the module docstring for the
    equation-by-equation walk-through): embeds atoms by nuclear charge,
    refines scalar and vector features through ``T`` message/update block
    pairs and reads out per-atom energies through a two-layer network with a
    SiLU nonlinearity. Per-atom energies are standardized (``energy_scale``,
    ``energy_shift``) and shifted by the per-element reference ``atom_ref``,
    then sum-pooled into the total energy.

    The defaults are ``F = 128`` features, ``T = 3`` blocks and 20 Gaussian
    radial functions with a cosine cutoff at 5 Angstrom; the paper's model
    uses 20 Bessel functions (``radial_basis="bessel"``) and, for dipoles,
    atomic dipoles (``atomic_dipoles=True``).

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
        Any basis of :func:`~xnn.common.featurizers.make_radial_basis`:
        ``"gaussian"`` (default; Gaussians on ``[0, cutoff]`` with the center
        spacing as width), ``"bessel"`` (the paper's ``sin(n pi r / r_cut) / r``,
        without a normalization) or ``"expnorm"``.
    trainable_rbf : bool, optional
        Learn the parameters of the radial basis, by default ``False``.
    shared_filters : bool, optional
        The same filters for every block instead of a set per block. Default
        ``False``.
    shared_interactions : bool, optional
        One set of message and update networks for all blocks (each block
        keeps its filters unless ``shared_filters``). Default ``False``.
    epsilon : float, optional
        Stabilizer of the vector norm in the update block, by default 1e-8.
    dipole : bool, optional
        Add a dipole head (outputs ``"dipole"`` ``(B, 3)`` and the latent
        ``"charges"`` ``(N,)``): eq 12, or eq 13 with ``atomic_dipoles``.
        Default ``False``.
    polarizability : bool, optional
        Add the polarizability head of eq 14 (output ``"polarizability"``
        ``(B, 3, 3)``). Default ``False``.
    n_output_blocks : int, optional
        Layers per output head (gated equivariant blocks of the tensorial
        heads, dense layers of the charge network), widths halving from
        ``F`` to 1; by default 2.
    correct_charges : bool, optional
        Shift the latent charges so they sum to the structure's net charge
        (``total_charge``, zero when absent). Default ``True``.
    atomic_dipoles : bool, optional
        The dipole model of eq 13: latent charges and atomic dipoles
        ``mu_i(v_i)`` from gated equivariant blocks (the paper's dipole
        model). ``False`` (default) keeps latent charges alone, computed
        from the scalars by an atom-wise network (eq 12).
    weight_init : str, optional
        Initialization of every dense layer (a name of
        :data:`~xnn.common.models.ops.WEIGHT_INITS`; biases start at zero),
        by default ``"xavier_uniform"``; ``"torch"`` keeps PyTorch's.
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
        The radial basis (see ``radial_basis``).
    cutoff_fn : CosineCutoff
        The cosine cutoff multiplying every filter.
    filters : torch.nn.Linear
        The filter layer of all blocks at once: block ``k``'s filters are the
        ``k``-th ``filter_width`` outputs (one block's when ``shared_filters``).
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
                 radial_basis: str = "gaussian", trainable_rbf: bool = False,
                 shared_filters: bool = False, shared_interactions: bool = False,
                 epsilon: float = 1e-8, dipole: bool = False,
                 polarizability: bool = False, n_output_blocks: int = 2,
                 correct_charges: bool = True, atomic_dipoles: bool = False,
                 weight_init: str = "xavier_uniform",
                 scalar_product: bool = True,
                 vector_propagation: bool = True, vector_features: bool = True,
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.n_interactions = n_interactions
        self.shared_filters = shared_filters
        self.shared_interactions = shared_interactions
        self.correct_charges = correct_charges
        self.atomic_dipoles = atomic_dipoles
        self.embedding = nn.Embedding(_MAX_Z, n_features)
        # the Bessel form of the paper carries no normalization
        self.rbf = make_radial_basis(radial_basis, n_rbf, cutoff, trainable_rbf,
                                     prefactor=1.0 if radial_basis == "bessel" else None)
        self.cutoff_fn = CosineCutoff(cutoff)
        # the filters of every block from one dense layer (one block's width when shared)
        self.filter_width = (1 + int(vector_features)
                             + int(vector_features and vector_propagation)) * n_features
        self.filters = nn.Linear(
            n_rbf, (1 if shared_filters else n_interactions) * self.filter_width)
        blocks = []
        for _ in range(n_interactions):
            if shared_interactions and blocks:
                first = blocks[0]
                message = _Message(n_features, vector_propagation, vector_features,
                                   phi=first.message.phi)
                blocks.append(_Interaction(message, first.update))
            else:
                blocks.append(_Interaction(
                    _Message(n_features, vector_propagation, vector_features),
                    _Update(n_features, epsilon, scalar_product, vector_features)))
        self.interactions = nn.ModuleList(blocks)
        self._register_load_state_dict_pre_hook(_stack_block_filters, with_module=True)
        self.readout = _mlp(n_features, 2)
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.dipole_head: Optional[nn.Module] = None
        if dipole:
            self.dipole_head = (_TensorHead(n_features, n_output_blocks) if atomic_dipoles
                                else _mlp(n_features, n_output_blocks))
        self.polarizability_head = (_TensorHead(n_features, n_output_blocks)
                                    if polarizability else None)
        self.register_buffer("masses", torch.tensor(ATOMIC_MASSES, dtype=torch.float64))
        init_linears_(self, weight_init)
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
        # the filters of all blocks in one product, times the cutoff
        W = self.filters(self.rbf(r)) * self.cutoff_fn(r).unsqueeze(-1)
        filters = torch.split(W, self.filter_width, dim=-1)
        s = self.embedding(atomic_numbers)
        v = torch.zeros((s.shape[0], 3, s.shape[1]), dtype=s.dtype, device=s.device)
        for k, block in enumerate(self.interactions):
            s, v = block(s, v, filters[k % len(filters)], direction, edge_index)
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
            if self.atomic_dipoles:
                q, mu_atom = self.dipole_head(s, v)
            else:
                q, mu_atom = self.dipole_head(s).squeeze(-1), None
            if self.correct_charges:
                total = (data.total_charge.to(q.dtype) if data.total_charge is not None
                         else torch.zeros(n_graphs, dtype=q.dtype, device=q.device))
                surplus = total - scatter_sum(q, data.batch, n_graphs)
                q = q + (surplus / data.n_atoms.to(q.dtype))[data.batch]
            out["charges"] = q
            mu = q.unsqueeze(-1) * r
            if mu_atom is not None:
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
        ``cfg.extra`` (``radial_basis``, ``trainable_rbf``, ``shared_filters``,
        ``shared_interactions``, ``epsilon``,
        ``dipole``, ``polarizability``, ``n_output_blocks``,
        ``correct_charges``, ``atomic_dipoles``, ``weight_init``, ``scalar_product``,
        ``vector_propagation``, ``vector_features``,
        ``energy_shift``, ``energy_scale``, ``species``,
        ``atomic_energies``); other codes' key spellings are translated by
        :mod:`xnn.common.config.translate`.

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
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            radial_basis=str(extra.get("radial_basis", "gaussian")),
            trainable_rbf=bool(extra.get("trainable_rbf", False)),
            shared_filters=bool(extra.get("shared_filters", False)),
            shared_interactions=bool(extra.get("shared_interactions", False)),
            epsilon=float(extra.get("epsilon", 1e-8)),
            dipole=bool(extra.get("dipole", False)),
            polarizability=bool(extra.get("polarizability", False)),
            n_output_blocks=int(extra.get("n_output_blocks", 2)),
            correct_charges=bool(extra.get("correct_charges", True)),
            atomic_dipoles=bool(extra.get("atomic_dipoles", False)),
            weight_init=str(extra.get("weight_init", "xavier_uniform")),
            scalar_product=bool(extra.get("scalar_product", True)),
            vector_propagation=bool(extra.get("vector_propagation", True)),
            vector_features=bool(extra.get("vector_features", True)),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
        )
