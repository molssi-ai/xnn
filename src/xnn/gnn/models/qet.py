"""QET (Ko et al., arXiv:2511.07249), the charge-equilibrated TensorNet.

An implementation of

* T. W. Ko, R. Liu, A. R. Mishra, Z. Yu, J. Qi, S. P. Ong, "A Linear-Scaling,
  Charge-Aware Foundation Potential for Atomistic Simulations",
  arXiv:2511.07249,

written from the paper on :class:`~xnn.gnn.models.tensornet.TensorNet` and
consistent with the conventions of the authors' code (MatGL, BSD-3-Clause)
without copying it. Parameter names follow MatGL where xnn has the same layer,
so that its published weights transplant.

The TensorNet trunk gives every atom invariant features ``h_i``. Two networks
read them: an electronegativity ``chi_i`` and a positive hardness ``eta_i``
(one value per element, or a softplus network of ``h_i``). An external
potential ``phi_i`` on an atom shifts its electronegativity. Minimizing
``sum_i (chi_i q_i + eta_i q_i^2 / 2)`` at a fixed total charge ``Q`` of the
structure gives the charges in closed form (eq 2 of the paper)::

    q_i = (lambda - chi_i) / eta_i,   lambda = (Q + sum_j chi_j / eta_j) / sum_j 1 / eta_j

which costs one sum per structure, so the model stays linear in the number of
atoms. The charges set a damped Coulomb potential at every atom (eq 3)::

    V_i = k_e sum_j q_j erf(r_ij / (sqrt(2) gamma_ij)) f_cut(r_ij) / r_ij,
    gamma_ij = sqrt(sigma_i^2 + sigma_j^2)

over the neighbors within the cutoff, with Gaussian charge widths ``sigma``
(the covalent radii of the elements, or learned per element), the polynomial
envelope of degree 3 as ``f_cut`` and ``k_e`` = 14.399645478425668 eV
Angstrom / e^2. The atomic energy is a gated network of the layer-normed
``[h_i, q_i, V_i]``.

Conventions worth knowing:

* the defaults are those of the authors' code: 32 Gaussians on
  ``[0, cutoff + 1]`` with exponent 0.5, no self pairs, two interaction
  layers of 64 features, a cutoff of 5 Angstrom and one hardness per element;
  the paper's foundation potentials use the smooth Bessel basis
  (``radial_basis="smooth_bessel"``) and the environment-dependent hardness
  (``hardness="environment"``);
* the trunk takes no charge: the total charge of a structure enters through
  the charge equilibration alone;
* the charge constraint couples every atom of a structure, so the model has
  no bounded receptive field (:meth:`QET.receptive_field`).
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import PolynomialCutoff
from xnn.common.models.blocks import MLP
from xnn.common.models.charge_solve import closed_form_charges
from xnn.common.models.constants import register_constant
from xnn.common.models.ops import gather_rows, scatter_sum
from xnn.common.models.registry import register_model

from .tensornet import TensorNet

#: Coulomb's constant ``e^2 / (4 pi epsilon_0)`` in eV Angstrom / e^2 (CODATA 2018).
COULOMB_CONSTANT = 14.399645478425668

#: Covalent radii (Angstrom) indexed by atomic number, from Cordero et al., Dalton Trans.
#: 2008, 2832 (sp3 carbon, low-spin manganese, iron and cobalt); 0.2 for the dummy
#: element 0 and every element past curium.
COVALENT_RADII = (
    0.2,
    0.31, 0.28,
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44,
    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
    2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89,
    1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
    2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69,
)

_HARDNESS = ("element", "environment")


class _ElementHardness(nn.Module):
    """One hardness per element, indexed by atomic number (initialized to one)."""

    def __init__(self, n_elements: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_elements))

    def forward(self, x: Tensor, atomic_numbers: Tensor) -> Tensor:
        return self.weight[atomic_numbers]


class _EnvironmentHardness(nn.Module):
    """A hardness from the atom's features: three dense layers, each through a softplus."""

    def __init__(self, n_features: int):
        super().__init__()
        self.mlp = MLP([n_features, n_features, n_features, 1], "softplus", activate_last=True)

    def forward(self, x: Tensor, atomic_numbers: Tensor) -> Tensor:
        return self.mlp(x).squeeze(-1)


@register_model("qet")
class QET(TensorNet):
    """QET interatomic potential: TensorNet with analytic charge equilibration.

    See the module docstring for the equations. The options of
    :class:`~xnn.gnn.models.tensornet.TensorNet` keep their meaning; the
    defaults follow the authors' code.

    Parameters
    ----------
    n_features : int, optional
        Number of tensor features ``F``, by default 64.
    n_interactions : int, optional
        Number of interaction layers, by default 2.
    n_rbf : int, optional
        Number of radial basis functions, by default 32.
    cutoff : float, optional
        Neighbor-list radius, cosine cutoff of the trunk and range of the
        Coulomb potential (Angstrom), by default 5.0.
    radial_basis : str, optional
        Any basis of :func:`~xnn.common.featurizers.make_radial_basis`, by
        default ``"gaussian"``.
    trainable_rbf : bool, optional
        Learn the parameters of the basis, by default ``False``.
    activation : str, optional
        Nonlinearity of the trunk, the electronegativity network and the
        energy network, by default ``"silu"``.
    equivariance_group : str, optional
        ``"O(3)"`` (default) or ``"SO(3)"``.
    n_elements : int, optional
        Size of the per-element tables, indexed by ``Z``, by default 128.
    energy_shift, energy_scale : float, optional
        Per-atom energy standardization, by default 0 and 1.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.
    readout : str, optional
        The atom-wise energy network, ``"gated"`` (default) or ``"mlp"``.
    readout_hidden : sequence of int or None, optional
        Hidden widths of the energy network; by default ``(F + 2, F, F)``
        for ``"gated"``.
    self_pairs : bool, optional
        Whether every atom is its own neighbor in the trunk, by default
        ``False``.
    rbf_max : float or None, optional
        Position of the last Gaussian of the ``"gaussian"`` basis; ``None``
        (default) is ``cutoff + 1``.
    gamma : float or None, optional
        Exponent of the Gaussians, ``exp(-gamma (r - mu)^2)``; ``None``
        (default) is 0.5.
    hardness : str, optional
        ``"element"`` (default; one learned hardness per element) or
        ``"environment"`` (a softplus network of the atom's features, as in
        the paper).
    trainable_widths : bool, optional
        Learn the Gaussian charge widths per element (initialized to one)
        instead of using the covalent radii, by default ``False``.

    Attributes
    ----------
    chi_readout : xnn.common.models.blocks.MLP
        The electronegativity network.
    hardness_readout : torch.nn.Module
        The hardness, per element or per environment.
    sigma : torch.Tensor
        Gaussian charge widths per element (Angstrom), indexed by ``Z``.
    coulomb_cutoff : xnn.common.featurizers.PolynomialCutoff
        The envelope of the Coulomb potential.
    readout_norm : torch.nn.LayerNorm
        Layer norm of ``[h_i, q_i, V_i]``.
    """

    refreshes_ghosts = False
    readout_extra_inputs = 2

    def __init__(self, n_features: int = 64, n_interactions: int = 2, n_rbf: int = 32,
                 cutoff: float = 5.0, radial_basis: str = "gaussian",
                 trainable_rbf: bool = False, activation: str = "silu",
                 equivariance_group: str = "O(3)", n_elements: int = 128,
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None, readout: str = "gated",
                 readout_hidden=None, self_pairs: bool = False,
                 rbf_max: Optional[float] = None, gamma: Optional[float] = None,
                 hardness: str = "element", trainable_widths: bool = False):
        if hardness not in _HARDNESS:
            raise ValueError(f"hardness must be one of {_HARDNESS}, got {hardness!r}")
        if radial_basis == "gaussian":
            rbf_max = float(cutoff) + 1.0 if rbf_max is None else rbf_max
            gamma = 0.5 if gamma is None else gamma
        super().__init__(n_features=n_features, n_interactions=n_interactions, n_rbf=n_rbf,
                         cutoff=cutoff, radial_basis=radial_basis, trainable_rbf=trainable_rbf,
                         activation=activation, equivariance_group=equivariance_group,
                         n_elements=n_elements, energy_shift=energy_shift,
                         energy_scale=energy_scale, species=species,
                         atomic_energies=atomic_energies, readout=readout,
                         readout_hidden=readout_hidden, self_pairs=self_pairs,
                         rbf_max=rbf_max, gamma=gamma)
        self.hardness_mode = hardness
        self.chi_readout = MLP([n_features, n_features, n_features, 1], activation, activate_last=True)
        self.hardness_readout = (_ElementHardness(n_elements) if hardness == "element"
                                 else _EnvironmentHardness(n_features))
        if trainable_widths:
            self.sigma = nn.Parameter(torch.ones(n_elements))
        else:
            register_constant(self, "sigma", [COVALENT_RADII[z] if z < len(COVALENT_RADII) else 0.2
                                              for z in range(n_elements)])
        self.coulomb_cutoff = PolynomialCutoff(cutoff, p=3)
        self.readout_norm = nn.LayerNorm(n_features + self.readout_extra_inputs)
        self.erf_scale = 1.0 / math.sqrt(2.0)
        self.coulomb_constant = COULOMB_CONSTANT

    def receptive_field(self) -> Optional[float]:
        """``None``: the charge constraint couples every atom of a structure."""
        return None

    def _charge_head(self, x: Tensor, atomic_numbers: Tensor, edge_index: Tensor,
                     edge_vec: Tensor, batch: Tensor, num_graphs: int, total_charge: Tensor,
                     external_potential: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Per-atom energies, charges and Coulomb potentials from the trunk's features."""
        chi = self.chi_readout(x).squeeze(-1)
        if external_potential is not None:
            chi = chi + external_potential.to(chi.dtype)
        inverse = 1.0 / self.hardness_readout(x, atomic_numbers)
        # the sum of inverse hardnesses is bounded below by 1e-12, as in the authors' code
        q = closed_form_charges(chi, inverse, total_charge, batch, num_graphs)

        src, dst = edge_index[0], edge_index[1]
        r = torch.linalg.norm(edge_vec, dim=-1)
        width2 = self.sigma.to(chi.dtype)[atomic_numbers] ** 2
        gamma = torch.sqrt(gather_rows(width2, src) + gather_rows(width2, dst))
        pair = gather_rows(q, src) * torch.erf(self.erf_scale * r / gamma) * self.coulomb_cutoff(r) / r
        v = self.coulomb_constant * scatter_sum(pair, dst, x.shape[0])

        h = self.readout_norm(torch.cat([x, q.unsqueeze(-1), v.unsqueeze(-1)], dim=-1))
        return self._energy(h, atomic_numbers), q, v

    @torch.jit.export
    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor,
                                     edge_vec: Tensor, pos: Tensor, cell: Tensor,
                                     pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float
                                     ) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript core of one structure: features, atomic energies and charges.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor, row 1 the center.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.
        pos, cell, pbc : Tensor
            Unused (part of the shared ABI).
        total_charge : float
            Net charge of the structure.
        spin_multiplicity : float
            Unused.

        Returns
        -------
        tuple of Tensor
            The trunk's invariant features ``(N, F)``, the per-atom energy
            ``(N,)`` and the charges ``(N,)``.
        """
        n = atomic_numbers.shape[0]
        zeros = torch.zeros(n, dtype=edge_vec.dtype, device=edge_vec.device)
        x = self._features(self.tensors(atomic_numbers, edge_index, edge_vec, zeros))
        batch = torch.zeros(n, dtype=torch.long, device=edge_vec.device)
        total = torch.full((1,), total_charge, dtype=edge_vec.dtype, device=edge_vec.device)
        energy, q, _ = self._charge_head(x, atomic_numbers, edge_index, edge_vec, batch, 1, total, None)
        return x, energy, q

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript core of one neutral structure: features and per-atom energies."""
        empty = torch.zeros(0, dtype=edge_vec.dtype, device=edge_vec.device)
        out = self.node_features_energy_charges(atomic_numbers, edge_index, edge_vec, empty, empty,
                                                empty, 0.0, 1.0)
        return out[0], out[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies and charges of a batched graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph; its ``total_charge`` (per structure, zero
            when absent) and ``external_potential`` (per atom, none when
            absent) enter the charge equilibration.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``, ``"charges"``
            ``(N,)``, ``"electrostatic_potential"`` ``(N,)`` (the Coulomb
            potential ``V_i`` in V), ``"node_features"`` (the trunk's invariant
            features, ``(N, F)``) and ``"node_tensors"`` (``(N, 3, 3, F)``).
        """
        self._select_fast(int(data.edge_index.shape[1]))
        edge_vec = data.edge_vectors()
        zeros = torch.zeros(data.num_nodes, dtype=edge_vec.dtype, device=edge_vec.device)
        X = self.tensors(data.atomic_numbers, data.edge_index, edge_vec, zeros)
        x = self._features(X)
        total = (data.total_charge if data.total_charge is not None
                 else torch.zeros(data.num_graphs, dtype=edge_vec.dtype, device=edge_vec.device))
        node_energy, q, v = self._charge_head(x, data.atomic_numbers, data.edge_index, edge_vec,
                                              data.batch, data.num_graphs, total,
                                              data.external_potential)
        return {"node_energy": node_energy, "energy": self.aggregate_energy(node_energy, data),
                "charges": q, "electrostatic_potential": v, "node_features": x, "node_tensors": X}

    @classmethod
    def from_foundation(cls, name: str = "qet-matpes-pbe-2025.2", dtype=None,
                        cache_dir=None) -> "QET":
        """Load a MatGL QET foundation potential (MatPES PBE or r2SCAN).

        See :mod:`xnn.gnn.models.qet_foundation`.

        Parameters
        ----------
        name : str, optional
            ``"qet-matpes-pbe-2025.2"`` (default) or
            ``"qet-matpes-r2scan-2025.2"``.
        dtype : torch.dtype or str, optional
            Final dtype; ``None`` keeps the checkpoint's float32.
        cache_dir : str or pathlib.Path, optional
            Model cache directory.

        Returns
        -------
        QET
            The converted potential, in evaluation mode.
        """
        from .qet_foundation import from_foundation
        return from_foundation(name, dtype=dtype, cache_dir=cache_dir)

    @classmethod
    def from_config(cls, cfg) -> "QET":
        """Build a :class:`QET` from a configuration object.

        Core fields: ``cfg.n_features``, ``cfg.n_interactions``, ``cfg.n_rbf``,
        ``cfg.cutoff``. Everything else is read from ``cfg.extra``: the
        options of :meth:`TensorNet.from_config
        <xnn.gnn.models.tensornet.TensorNet.from_config>` and ``hardness``,
        ``trainable_widths``; MatGL's key spellings are translated by
        :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        QET
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        gamma = extra.get("gamma", None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            radial_basis=str(extra.get("radial_basis", "gaussian")),
            trainable_rbf=bool(extra.get("trainable_rbf", False)),
            activation=str(extra.get("activation", "silu")),
            equivariance_group=str(extra.get("equivariance_group", "O(3)")),
            n_elements=int(extra.get("n_elements", 128)),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
            readout=str(extra.get("readout", "gated")),
            readout_hidden=extra.get("readout_hidden", None),
            self_pairs=bool(extra.get("self_pairs", False)),
            rbf_max=None if extra.get("rbf_max") is None else float(extra["rbf_max"]),
            gamma=None if gamma is None else float(gamma),
            hardness=str(extra.get("hardness", "element")),
            trainable_widths=bool(extra.get("trainable_widths", False)),
        )
