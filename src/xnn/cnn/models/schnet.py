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
(:class:`~xnn.common.featurizers.GaussianRBF`,
:func:`~xnn.common.models.ops.scatter_sum`,
:func:`~xnn.common.models.ops.shifted_softplus`, ...); nothing is taken from
the schnetpack code base.

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

Deviations from the papers, all optional and off by default:

* ``cutoff_fn="cosine"`` multiplies the generated filter by a smooth
  :class:`~xnn.common.featurizers.CosineCutoff` envelope so the PES stays
  smooth when a *finite* neighbor-list cutoff truncates the graph. The paper
  itself trains without a cutoff -- its RBF grid simply ends at 30 Angstrom,
  beyond any distance in its molecular datasets -- which is what the default
  (``None``) reproduces.
* ``atom_ref``, a learnable per-element reference energy (the xnn
  convention shared by every model here), initialized to zero so it is inert
  unless set/trained. It plays the role of a per-element ``E_mu``.

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
from xnn.common.featurizers import GaussianRBF, CosineCutoff
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import scatter_sum, shifted_softplus
from xnn.common.models.registry import register_model

_MAX_Z = 100


class _ShiftedSoftplus(nn.Module):
    """Module wrapper of the shared exact
    :func:`~xnn.common.models.ops.shifted_softplus` (``ln(0.5 e^x + 0.5)``),
    for use inside ``nn.Sequential``."""

    def forward(self, x: Tensor) -> Tensor:
        return shifted_softplus(x)


class _CFConv(nn.Module):
    """Continuous-filter convolution (NIPS paper eq 2 and Fig. 2, right).

    Each neighbor's features are gated element-wise by a filter generated
    from the interatomic distance, then summed onto the central atom:
    ``x_i = sum_j x_j o W(r_ij)``. The filter-generating network is two dense
    layers with shifted-softplus activations over the Gaussian RBF expansion
    of the distance ("we feed the expanded distances into two dense layers
    with softplus activations to compute the filter weight").

    Parameters
    ----------
    n_features : int
        Dimension ``F`` of the per-atom feature vectors (the paper keeps the
        filter dimension equal to ``F``).
    n_rbf : int
        Number of radial basis functions expanding the distances.
    cutoff_fn : CosineCutoff or None
        Optional smooth envelope multiplying the filter (off in the paper;
        see the module docstring).
    """

    def __init__(self, n_features: int, n_rbf: int,
                 cutoff_fn: Optional[CosineCutoff] = None):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features), _ShiftedSoftplus(),
            nn.Linear(n_features, n_features), _ShiftedSoftplus(),
        )
        self.cutoff_fn = cutoff_fn

    def forward(self, x: Tensor, edge_index: Tensor, r: Tensor,
                rbf: Tensor) -> Tensor:
        """Apply one continuous-filter convolution.

        Parameters
        ----------
        x : Tensor
            Per-atom features, shape ``(N, F)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbor
            ``j``) and row 1 the destination (center ``i``) of each edge.
        r : Tensor
            Interatomic distances per edge, shape ``(E,)``.
        rbf : Tensor
            Radial basis expansion of ``r``, shape ``(E, n_rbf)``.

        Returns
        -------
        Tensor
            Convolved per-atom features ``sum_j x_j o W(r_ij)``, shape
            ``(N, F)``.
        """
        src, dst = edge_index[0], edge_index[1]
        W = self.filter_net(rbf)                                # (E, F)
        if self.cutoff_fn is not None:
            W = W * self.cutoff_fn(r).unsqueeze(-1)
        return scatter_sum(x[src] * W, dst, x.shape[0])


class _Interaction(nn.Module):
    """One SchNet interaction block (NIPS paper Fig. 2, middle).

    Computes the residual ``v = W3 ssp(W2 cfconv(W1 x)) `` -- an atom-wise
    layer, the continuous-filter convolution, an atom-wise layer, a shifted
    softplus, and a final atom-wise layer. The caller adds it to the input
    (``x^{l+1} = x^l + v^l``, the ResNet-style connection). All atom-wise
    layers are dense layers applied per atom with weights shared across
    atoms; the feature width ``F`` is constant throughout, as in the paper.

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions feeding the filter network.
    cutoff_fn : CosineCutoff or None
        Optional smooth filter envelope, passed to :class:`_CFConv`.
    """

    def __init__(self, n_features: int, n_rbf: int,
                 cutoff_fn: Optional[CosineCutoff] = None):
        super().__init__()
        self.lin_in = nn.Linear(n_features, n_features)
        self.cfconv = _CFConv(n_features, n_rbf, cutoff_fn)
        self.lin_mid = nn.Linear(n_features, n_features)
        self.lin_out = nn.Linear(n_features, n_features)

    def forward(self, x: Tensor, edge_index: Tensor, r: Tensor,
                rbf: Tensor) -> Tensor:
        """Compute the interaction residual ``v`` for every atom.

        Parameters
        ----------
        x : Tensor
            Per-atom features, shape ``(N, F)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``.
        r : Tensor
            Interatomic distances per edge, shape ``(E,)``.
        rbf : Tensor
            Radial basis expansion of ``r``, shape ``(E, n_rbf)``.

        Returns
        -------
        Tensor
            The residual ``v``, shape ``(N, F)`` (add it to ``x`` outside).
        """
        v = self.cfconv(self.lin_in(x), edge_index, r, rbf)
        return self.lin_out(shifted_softplus(self.lin_mid(v)))


@register_model("schnet")
class SchNet(InteratomicPotential):
    """SchNet continuous-filter convolutional interatomic potential.

    Faithful to the NIPS 2017 manuscript (see the module docstring for the
    equation-by-equation walk-through): embeds atoms by nuclear charge,
    refines their features through ``T`` residual interaction blocks built
    around the continuous-filter convolution, and reads out a per-atom energy
    through a two-layer atom-wise network with shifted-softplus
    nonlinearities. Per-atom energies are standardized with the training-set
    statistics (``E_i = energy_scale * E^hat_i + energy_shift``, the DTNN
    convention) plus a per-element reference ``atom_ref``, then sum-pooled
    into the total energy. Works unchanged for molecules and periodic solids,
    since periodicity enters only through the edge vectors.

    The defaults reproduce the paper's architecture: ``F = 64`` feature maps,
    ``T = 3`` interaction blocks, and Gaussian RBFs on a 0.1-Angstrom grid
    from 0 to 30 Angstrom with ``gamma = 10`` (301 centers -- the grid the
    paper states as "centers 0 <= mu_k <= 30 every 0.1 Angstrom").

    Parameters
    ----------
    n_features : int, optional
        Dimension ``F`` of the per-atom feature vectors, by default 64 (the
        paper's value; kept constant through the interaction blocks).
    n_interactions : int, optional
        Number of stacked interaction blocks ``T`` (no weight sharing), by
        default 3.
    n_rbf : int, optional
        Number of Gaussian radial basis functions expanding the distances,
        by default 301 (0.1-Angstrom spacing on ``[0, 30]``).
    cutoff : float, optional
        Neighbor-list radius and upper end of the RBF center grid (Angstrom),
        by default 30.0. The paper uses no explicit cutoff; 30 Angstrom
        covers all pairs of its molecular datasets. For condensed phases use
        a finite cutoff (e.g. 5.0) with ``n_rbf ~ cutoff / 0.1`` and
        ``cutoff_fn="cosine"``.
    gamma : float or None, optional
        Width parameter of the Gaussian RBFs, by default 10.0 (the paper's
        value, per Angstrom^2). ``None`` ties the width to the center
        spacing instead (see :class:`~xnn.common.featurizers.GaussianRBF`).
    cutoff_fn : str or None, optional
        ``"cosine"`` multiplies the generated filters by a smooth
        :class:`~xnn.common.featurizers.CosineCutoff` envelope (recommended
        with finite cutoffs); ``None`` (default) is the paper's unmodulated
        filter.
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
    rbf : GaussianRBF
        Gaussian radial basis expansion of interatomic distances.
    interactions : torch.nn.ModuleList
        Stack of :class:`_Interaction` blocks.
    readout : torch.nn.Sequential
        Atom-wise MLP (``F -> F/2 -> 1``) mapping final features to the
        unstandardized per-atom energy ``E^hat_i``; its last layer is
        zero-initialized (the DTNN convention) so initial predictions equal
        ``energy_shift + atom_ref``.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference (shift), initialized to zero.
    """

    def __init__(self, n_features: int = 64, n_interactions: int = 3,
                 n_rbf: int = 301, cutoff: float = 30.0,
                 gamma: Optional[float] = 10.0,
                 cutoff_fn: Optional[str] = None,
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        if cutoff_fn not in (None, "cosine"):
            raise ValueError(
                f"cutoff_fn must be None or 'cosine', got {cutoff_fn!r}")
        self.cutoff = cutoff
        self.node_feature_dim = n_features  # invariant features (for e.g. LES)
        self.embedding = nn.Embedding(_MAX_Z, n_features)
        self.rbf = GaussianRBF(n_rbf, cutoff, gamma=gamma)
        self.interactions = nn.ModuleList([
            _Interaction(
                n_features, n_rbf,
                CosineCutoff(cutoff) if cutoff_fn == "cosine" else None)
            for _ in range(n_interactions)
        ])
        self.readout = nn.Sequential(
            nn.Linear(n_features, n_features // 2), _ShiftedSoftplus(),
            nn.Linear(n_features // 2, 1),
        )
        # zero-init the output head (DTNN convention): the initial prediction
        # is exactly the standardization shift, a good starting point
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
        for block in self.interactions:
            x = x + block(x, edge_index, r, rbf)     # x^{l+1} = x^l + v^l
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
        (``gamma``, ``cutoff_fn``, ``energy_shift``, ``energy_scale``,
        ``species``, ``atomic_energies``); schnetpack key spellings are
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

        extra = dict(cfg.extra or {})
        gamma = extra.get("gamma", 10.0)
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            gamma=None if gamma is None else float(gamma),
            cutoff_fn=extra.get("cutoff_fn"),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [],
                "atomic_energies"),
        )
