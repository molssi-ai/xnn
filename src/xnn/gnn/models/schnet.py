"""SchNet (Schuett et al., NIPS 2017) -- continuous-filter convolutional network.

A faithful implementation of the architecture described in the manuscripts

* K. T. Schuett, P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko,
  K.-R. Mueller, "SchNet: A continuous-filter convolutional neural network for
  modeling quantum interactions", NIPS 30 (2017) -- the architecture; and
* K. T. Schuett, F. Arbabzadah, S. Chmiela, K. R. Mueller, A. Tkatchenko,
  "Quantum-chemical insights from deep tensor neural networks", Nat. Commun.
  8, 13890 (2017) -- the DTNN predecessor, for the conventions SchNet
  inherits (per-atom energy standardization, sum pooling).

It is built directly from the papers' equations on the xnn abstractions
(:func:`~xnn.common.featurizers.make_radial_basis`,
:func:`~xnn.common.models.ops.scatter_sum`,
:func:`~xnn.common.models.ops.shifted_softplus`, ...).

Architecture (NIPS paper section 4, Fig. 2):

* atoms are embedded by nuclear charge, ``x^0_i = a_{Z_i}`` (eq 3);
* interatomic distances are expanded in Gaussian radial basis functions
  ``e_k(r) = exp(-gamma (r - mu_k)^2)`` with centers every 0.1 Angstrom and
  ``gamma = 10`` per Angstrom^2 (section "Filter-generating networks");
* ``T`` interaction blocks (no weight sharing across blocks) refine the atom
  features through the ResNet-style residual ``x^{l+1}_i = x^l_i + v^l_i``,
  where the residual is *atom-wise -> cfconv -> atom-wise -> shifted softplus
  -> atom-wise* (Fig. 2, middle);
* the continuous-filter convolution (cfconv, eq 2) gates each neighbor's
  features element-wise with a filter generated from the distance,
  ``x_i = sum_j x_j o W(r_ij)``, where the filter-generating network is two
  dense layers with shifted-softplus activations over the RBF expansion
  (Fig. 2, right);
* the readout maps the final features through *atom-wise (F -> F/2) ->
  shifted softplus -> atom-wise (F/2 -> 1)* and sum-pools the per-atom
  energies over each structure (Fig. 2, left), after the DTNN per-atom
  standardization ``E_i = E_sigma * E^hat_i + E_mu`` (DTNN Methods, step 4;
  :meth:`SchNet.set_energy_scale_shift`).

The shifted softplus ``ssp(x) = ln(0.5 e^x + 0.5)`` is used throughout, which
keeps the potential-energy surface smooth (infinitely differentiable), so the
autograd forces added by
:class:`~xnn.common.models.outputs.ForceStressOutput` are smooth and
energy-conserving by construction (paper eqs 1 and 4).

Defaults and the paper's options. The defaults are ``F = 128``, ``T = 6``,
20 Gaussians on ``[0, 5]`` Angstrom with the center spacing as width, a
cosine envelope of the filters, a linear second filter layer, a bias-free
first atom-wise layer in every block and a SiLU readout. The paper's model
sets ``n_features=64``, ``n_interactions=3``, ``n_rbf=301``,
``cutoff=30.0``, ``gamma=10.0``, ``cutoff_fn=None`` (no envelope: the grid
ends at 30 Angstrom, beyond any distance of its molecular datasets),
``filter_output_activation=True``, ``interaction_input_bias=True`` and
``readout_activation="ssp"``. Further options: any basis of
:func:`~xnn.common.featurizers.make_radial_basis` (``radial_basis``), a
filter width other than ``F`` (``n_filters``) and one interaction block
shared by all ``T`` (``shared_interactions``), the weight initialization
(``weight_init``, Xavier-uniform by default) and a zero-initialized readout
(``zero_init_readout``, the DTNN convention). ``atom_ref`` is a learnable
per-element reference energy (the xnn convention shared by every model
here), initialized to zero so it is inert unless set or trained.

Works for molecules and periodic solids unchanged: periodicity enters only
through ``data.edge_vectors()``, which already accounts for cell shifts.
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
from xnn.common.models.constants import ExactFloat64Constants
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import ShiftedSoftplus, init_linears_, make_activation, scatter_sum, shifted_softplus
from xnn.common.models.registry import register_model

_MAX_Z = 100


class _CFConv(nn.Module, FastPathModule):
    """Continuous-filter convolution (NIPS paper eq 2 and Fig. 2, right).

    Each neighbor's features are gated element-wise by a filter generated
    from the interatomic distance, then summed onto the central atom:
    ``x_i = sum_j x_j o W(r_ij)``. The filter-generating network is two dense
    layers over the radial expansion of the distance, a shifted softplus after
    the first and, optionally, after the second (the paper's "two dense layers
    with softplus activations").

    Parameters
    ----------
    n_filters : int
        Width of the filters and of the convolved features.
    n_rbf : int
        Number of radial basis functions expanding the distances.
    output_activation : bool
        Apply the shifted softplus to the second layer's output too.

    Notes
    -----
    A fused GPU implementation (:mod:`xnn.gnn.fast.schnet`) evaluates the same
    block when :attr:`fast_active` is set (see :meth:`SchNet.set_use_fast`).
    """

    def __init__(self, n_filters: int, n_rbf: int, output_activation: bool = False):
        super().__init__()
        layers = [nn.Linear(n_rbf, n_filters), ShiftedSoftplus(), nn.Linear(n_filters, n_filters)]
        if output_activation:
            layers.append(ShiftedSoftplus())
        self.filter_net = nn.Sequential(*layers)
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Triton on ``device``, a power-of-two feature width and plain dense layers."""
        from xnn.gnn.fast.schnet import supported
        n = self.filter_net[2].out_features
        plain = all(type(self.filter_net[k]) is nn.Linear for k in (0, 2))
        return plain and 16 <= n <= 512 and n & (n - 1) == 0 and supported(device, dtype)

    def forward(self, x: Tensor, edge_index: Tensor, rbf: Tensor,
                envelope: Optional[Tensor]) -> Tensor:
        """Apply one continuous-filter convolution.

        Parameters
        ----------
        x : Tensor
            Per-atom features, shape ``(N, F)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbor
            ``j``) and row 1 the destination (center ``i``) of each edge.
        rbf : Tensor
            Radial basis expansion of the distances, shape ``(E, n_rbf)``.
        envelope : Tensor or None
            Optional smooth filter envelope per edge, shape ``(E,)``.

        Returns
        -------
        Tensor
            Convolved per-atom features ``sum_j x_j o W(r_ij)``, shape
            ``(N, F)``.
        """
        if self.fast_active and not self.training and not torch.jit.is_scripting():
            return self._forward_fast(x, edge_index, rbf, envelope)
        src, dst = edge_index[0], edge_index[1]
        W = self.filter_net(rbf)                                # (E, n_filters)
        if envelope is not None:
            W = W * envelope.unsqueeze(-1)
        return scatter_sum(x[src] * W, dst, x.shape[0])

    @torch.jit.unused
    def _forward_fast(self, x: Tensor, edge_index: Tensor, rbf: Tensor,
                      envelope: Optional[Tensor]) -> Tensor:
        from xnn.gnn.fast.schnet_kernels import cfconv
        first, second = self.filter_net[0], self.filter_net[2]
        return cfconv(x, rbf, envelope, first.weight, first.bias, second.weight, second.bias,
                      edge_index, activated=len(self.filter_net) == 4)


class _Interaction(nn.Module):
    """One SchNet interaction block (NIPS paper Fig. 2, middle).

    Computes the residual ``v = W3 ssp(W2 cfconv(W1 x)) `` -- an atom-wise
    layer, the continuous-filter convolution, an atom-wise layer, a shifted
    softplus, and a final atom-wise layer. The caller adds it to the input
    (``x^{l+1} = x^l + v^l``, the ResNet-style connection). All atom-wise
    layers are dense layers applied per atom with weights shared across
    atoms.

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions feeding the filter network.
    n_filters : int
        Width of the filters (``W1`` maps ``F`` to it, ``W2`` back to ``F``).
    filter_output_activation : bool
        Apply the shifted softplus to the filter network's output.
    input_bias : bool
        Give the first atom-wise layer ``W1`` a bias.
    """

    def __init__(self, n_features: int, n_rbf: int, n_filters: int,
                 filter_output_activation: bool = False, input_bias: bool = False):
        super().__init__()
        self.lin_in = nn.Linear(n_features, n_filters, bias=input_bias)
        self.cfconv = _CFConv(n_filters, n_rbf, filter_output_activation)
        self.lin_mid = nn.Linear(n_filters, n_features)
        self.lin_out = nn.Linear(n_features, n_features)

    def forward(self, x: Tensor, edge_index: Tensor, rbf: Tensor,
                envelope: Optional[Tensor]) -> Tensor:
        """Compute the interaction residual ``v`` for every atom.

        Parameters
        ----------
        x : Tensor
            Per-atom features, shape ``(N, F)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``.
        rbf : Tensor
            Radial basis expansion of the distances, shape ``(E, n_rbf)``.
        envelope : Tensor or None
            Optional smooth filter envelope per edge, shape ``(E,)``.

        Returns
        -------
        Tensor
            The residual ``v``, shape ``(N, F)`` (add it to ``x`` outside).
        """
        v = self.cfconv(self.lin_in(x), edge_index, rbf, envelope)
        return self.lin_out(shifted_softplus(self.lin_mid(v)))


@register_model("schnet")
class SchNet(ExactFloat64Constants, InteratomicPotential):
    """SchNet continuous-filter convolutional interatomic potential.

    Faithful to the NIPS 2017 manuscript (see the module docstring for the
    equation-by-equation walk-through): embeds atoms by nuclear charge,
    refines their features through ``T`` residual interaction blocks built
    around the continuous-filter convolution, and reads out a per-atom energy
    through a two-layer atom-wise network. Per-atom energies are standardized with the training-set
    statistics (``E_i = energy_scale * E^hat_i + energy_shift``, the DTNN
    convention) plus a per-element reference ``atom_ref``, then sum-pooled
    into the total energy. Works unchanged for molecules and periodic solids,
    since periodicity enters only through the edge vectors.

    The defaults and the paper's options are listed in the module docstring;
    the paper's grid is 301 Gaussians every 0.1 Angstrom from 0 to 30
    Angstrom with ``gamma = 10`` ("centers 0 <= mu_k <= 30 every 0.1
    Angstrom").

    Parameters
    ----------
    n_features : int, optional
        Dimension ``F`` of the per-atom feature vectors, by default 128 (the
        paper: 64).
    n_interactions : int, optional
        Number of stacked interaction blocks ``T``, by default 6 (the
        paper: 3).
    n_rbf : int, optional
        Number of radial basis functions expanding the distances, by default
        20 (the paper: 301, a 0.1-Angstrom spacing on ``[0, 30]``).
    cutoff : float, optional
        Neighbor-list radius and upper end of the radial basis (Angstrom), by
        default 5.0. The paper uses no explicit cutoff; 30 Angstrom covers
        all pairs of its molecular datasets.
    radial_basis : str, optional
        Any basis of :func:`~xnn.common.featurizers.make_radial_basis`, by
        default ``"gaussian"`` (the paper's).
    gamma : float or None, optional
        Exponent of the Gaussians, ``exp(-gamma (r - mu)^2)``; ``None``
        (default) ties the width to the center spacing (the paper: 10.0 per
        Angstrom^2).
    cutoff_fn : str or None, optional
        ``"cosine"`` (default) multiplies the generated filters by a smooth
        :class:`~xnn.common.featurizers.CosineCutoff` envelope, which keeps
        the energy smooth at a finite cutoff; ``None`` is the paper's
        unmodulated filter.
    n_filters : int or None, optional
        Width of the filters, by default ``None`` (equal to ``F``, the
        paper's).
    filter_output_activation : bool, optional
        A shifted softplus after the second filter layer too (the paper's two
        activated layers), by default ``False``.
    interaction_input_bias : bool, optional
        A bias in the first atom-wise layer of every block (the paper's), by
        default ``False``.
    readout_activation : str, optional
        Activation of the readout network (any name of
        :func:`~xnn.common.models.ops.make_activation`), by default
        ``"silu"`` (the paper: ``"ssp"``).
    shared_interactions : bool, optional
        One interaction block shared by all ``T``, by default ``False``.
    weight_init : str, optional
        Initialization of every dense layer (a name of
        :data:`~xnn.common.models.ops.WEIGHT_INITS`; biases start at zero),
        by default ``"xavier_uniform"``; ``"torch"`` keeps PyTorch's.
    zero_init_readout : bool, optional
        Start the last readout layer at zero, so that a fresh model predicts
        ``energy_shift + atom_ref`` (the DTNN convention), by default
        ``False``.
    energy_shift : float, optional
        Additive per-atom energy standardization ``E_mu`` (the training-set
        mean energy per atom, DTNN Methods step 4), by default 0.0. Stored as
        a non-trainable buffer; see :meth:`set_energy_scale_shift`.
    energy_scale : float, optional
        Multiplicative per-atom energy standardization ``E_sigma`` (the
        training-set standard deviation of the energy per atom), by default
        1.0.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``; the model itself handles
        all elements up to Z = 99.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref`` (aligned
        with ``species``).

    Attributes
    ----------
    cutoff : float
        Neighbor-list cutoff radius.
    embedding : torch.nn.Embedding
        Nuclear-charge-to-feature embedding ``a_Z`` (paper eq 3).
    rbf : torch.nn.Module
        Radial basis expansion of interatomic distances.
    interactions : torch.nn.ModuleList
        Stack of :class:`_Interaction` blocks.
    cutoff_fn : CosineCutoff or None
        The filter envelope, or ``None`` for the paper's unmodulated filter.
    readout : torch.nn.Sequential
        Atom-wise MLP (``F -> F/2 -> 1``) mapping final features to the
        unstandardized per-atom energy ``E^hat_i`` (its last layer starts at
        zero with ``zero_init_readout``).
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference (shift), initialized to zero.
    """

    # one readout head = the atom-wise readout, the DTNN scale/shift and the
    # reference energies (see MultiHead)
    head_modules = ("readout", "energy_scale", "energy_shift", "atom_ref")

    def __init__(self, n_features: int = 128, n_interactions: int = 6,
                 n_rbf: int = 20, cutoff: float = 5.0,
                 radial_basis: str = "gaussian", gamma: Optional[float] = None,
                 cutoff_fn: Optional[str] = "cosine", n_filters: Optional[int] = None,
                 filter_output_activation: bool = False, interaction_input_bias: bool = False,
                 readout_activation: str = "silu", shared_interactions: bool = False,
                 weight_init: str = "xavier_uniform", zero_init_readout: bool = False,
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        if cutoff_fn not in (None, "cosine"):
            raise ValueError(
                f"cutoff_fn must be None or 'cosine', got {cutoff_fn!r}")
        self.cutoff = cutoff
        if species is not None:          # the elements the references are set for
            self.species = [int(z) for z in species]
        self.node_feature_dim = n_features  # invariant features (for e.g. LES)
        self.embedding = nn.Embedding(_MAX_Z, n_features)
        self.rbf = make_radial_basis(radial_basis, n_rbf, cutoff, gamma=gamma)
        n_filters = n_features if n_filters is None else n_filters
        self.shared_interactions = shared_interactions

        def block():
            return _Interaction(n_features, n_rbf, n_filters, filter_output_activation,
                                interaction_input_bias)

        shared = block() if shared_interactions else None
        self.interactions = nn.ModuleList([
            shared if shared is not None else block() for _ in range(n_interactions)])
        self.cutoff_fn = CosineCutoff(cutoff) if cutoff_fn == "cosine" else None
        self.readout = nn.Sequential(
            nn.Linear(n_features, n_features // 2),
            make_activation(readout_activation),
            nn.Linear(n_features // 2, 1),
        )
        init_linears_(self, weight_init)
        if zero_init_readout:
            # the DTNN convention: the initial prediction is exactly the
            # standardization shift, a good starting point
            nn.init.zeros_(self.readout[-1].weight)
            nn.init.zeros_(self.readout[-1].bias)
        # DTNN per-atom standardization E_i = E_sigma * E^hat_i + E_mu,
        # fixed from training-set statistics (buffers, not parameters)
        self.register_buffer("energy_scale",
                             torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift",
                             torch.tensor(float(energy_shift)))
        # per-element energy reference (learnable shift), key for
        # transferability across compositions
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        if atomic_energies is not None:
            if species is None:
                raise ValueError(
                    "species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)
        # the fused convolution (see set_use_fast); "auto" takes it where measured faster
        self.use_fast = "auto"

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "SchNet":
        """Choose the implementation of the continuous-filter convolutions.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the fused GPU kernels of
            :mod:`xnn.gnn.fast.schnet` where they are available and faster for
            the input (a CUDA device, Triton, a graph above the measured size
            threshold); ``True``: wherever they are available; ``False``:
            always the reference implementation. Evaluations in training mode
            always use the reference, which is twice differentiable and
            differentiates the weights. In float32 the kernels agree with the
            reference to about 1e-6 relative, in float64 to about 1e-14.

        Returns
        -------
        SchNet
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> None:
        """Switch the fused convolutions on or off for one evaluation."""
        from xnn.common.models.fast import fast_modules, select
        from xnn.gnn.fast.schnet import AUTO_POLICY
        blocks = self.__dict__.get("_fast_blocks")
        if blocks is None:
            blocks = fast_modules(self)
            self.__dict__["_fast_blocks"] = blocks
        if self.training:
            for block in blocks:
                block.fast_active = False
            return
        like = self.embedding.weight
        select(blocks, self.use_fast, AUTO_POLICY, like.device, like.dtype, num_edges)

    @torch.jit.ignore
    def set_energy_scale_shift(self, scale: float, shift: float) -> None:
        """Set the DTNN per-atom energy standardization from training stats.

        Parameters
        ----------
        scale : float
            ``E_sigma``, the standard deviation of the training-set energy
            per atom.
        shift : float
            ``E_mu``, the mean training-set energy per atom.
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
                f"got {ae.numel()} atomic energies for "
                f"{len(list(species))} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(list(species)), 0] = ae

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: tensors in, features + energy out.

        The single implementation reused by :meth:`node_energy` (the deploy
        entry point) and :meth:`forward`, so it must avoid the AtomicGraph
        dataclass and any Python-only constructs.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbor)
            and row 1 the destination (center) node of each edge.
        edge_vec : Tensor
            Edge displacement vectors, shape ``(E, 3)`` (already accounting
            for any periodic cell shifts).

        Returns
        -------
        tuple of Tensor
            The invariant node features after the last interaction block
            ``(N, node_feature_dim)`` and the per-atom energy ``(N,)``
            (standardized and including the per-element reference shift).
        """
        x = self.embedding(atomic_numbers)
        r = torch.linalg.norm(edge_vec, dim=-1)
        rbf = self.rbf(r)
        envelope: Optional[Tensor] = None
        if self.cutoff_fn is not None:
            envelope = self.cutoff_fn(r)
        for block in self.interactions:
            x = x + block(x, edge_index, rbf, envelope)    # x^{l+1} = x^l + v^l
        node_energy = (self.energy_scale * self.readout(x).squeeze(-1)
                       + self.energy_shift
                       + self.atom_ref(atomic_numbers).squeeze(-1))
        return x, node_energy

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy, shape ``(N,)`` (thin wrapper over
        :meth:`node_features_energy`; the deploy wrappers call this)."""
        out = self.node_features_energy(atomic_numbers, edge_index, edge_vec)
        return out[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Compute per-atom and total energies for a batch of structures.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph providing atomic numbers, edge index and
            edge vectors.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with ``"node_energy"`` (per-atom energy, shape
            ``(N,)``), ``"energy"`` (per-structure total energy, the sum
            pooling of Fig. 2) and ``"node_features"`` (invariant per-atom
            features, shape ``(N, node_feature_dim)``).
        """
        self._select_fast(int(data.edge_index.shape[1]))
        features, node_energy = self.node_features_energy(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy,
                "node_features": features}

    @classmethod
    def from_config(cls, cfg) -> "SchNet":
        """Build a :class:`SchNet` from a configuration object.

        Core fields: ``cfg.n_features`` -> ``F``, ``cfg.n_interactions`` ->
        ``T``, ``cfg.n_rbf`` and ``cfg.cutoff`` -> the RBF grid /
        neighbor-list radius. Everything else is read from ``cfg.extra``
        (``radial_basis``, ``gamma``, ``cutoff_fn`` (``null`` for none),
        ``n_filters``, ``filter_output_activation``,
        ``interaction_input_bias``, ``readout_activation``,
        ``shared_interactions``, ``weight_init``, ``zero_init_readout``,
        ``energy_shift``, ``energy_scale``,
        ``species``, ``atomic_energies``); other codes' key spellings are
        translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        SchNet
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        gamma = extra.get("gamma")
        n_filters = extra.get("n_filters")
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            radial_basis=str(extra.get("radial_basis", "gaussian")),
            gamma=None if gamma is None else float(gamma),
            cutoff_fn=extra.get("cutoff_fn", "cosine"),
            n_filters=None if n_filters is None else int(n_filters),
            filter_output_activation=bool(extra.get("filter_output_activation", False)),
            interaction_input_bias=bool(extra.get("interaction_input_bias", False)),
            readout_activation=str(extra.get("readout_activation", "silu")),
            shared_interactions=bool(extra.get("shared_interactions", False)),
            weight_init=str(extra.get("weight_init", "xavier_uniform")),
            zero_init_readout=bool(extra.get("zero_init_readout", False)),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [],
                "atomic_energies"),
        )
