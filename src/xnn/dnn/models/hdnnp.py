"""High-dimensional neural network potentials of the first to the fourth generation.

The four generations of Behler's classification (*Chem. Rev.* 121, 10037, 2021):

* **1G** (:class:`NNP1G`): one feed-forward network maps a fixed set of
  coordinates of a fixed system to its energy (Blank *et al.* 1995, Lorenz *et
  al.* 2004; Fig. 1 and eq. 1 of Behler and Parrinello 2007).
* **2G** (:class:`HDNNP`, ``generation=2``): the energy is a sum of atomic
  energies, each the output of the network of the atom's element on the
  atom-centered symmetry functions of its environment (Behler and Parrinello,
  *Phys. Rev. Lett.* 98, 146401, 2007; eq. 2).
* **3G** (``generation=3``): a second set of element networks gives
  environment-dependent atomic charges, whose electrostatic energy is added to
  the short-range one (Artrith, Morawietz and Behler, *Phys. Rev. B* 83,
  153101, 2011).
* **4G** (``generation=4``): the networks give atomic electronegativities;
  the charges follow from a charge equilibration over the whole system with
  Gaussian charges and element hardnesses, so they respond to distant changes
  and to the total charge, and they enter the short-range networks as an
  extra input (Ko, Finkler, Goedecker and Behler, *Nat. Commun.* 12, 398,
  2021; eqs. 1-9).

The descriptor is :class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`.
The conventions follow the RuNNer code, so that a RuNNer model (``input.nn``,
the weights and scaling files) loads into this class unchanged
(:func:`xnn.dnn.common.runner.load_runner_model`):

* Charges are Gaussians of width ``sigma`` per element (zero: point charges),
  by default ``0.45`` times the covalent radius clamped to ``[0.25, 0.6]``
  Angstrom (:func:`default_gaussian_widths`). The electrostatic energy is that
  of the Gaussian charges, ``sum_{i<j} q_i q_j erf(r_ij / (sqrt(2) gamma_ij)) /
  r_ij + sum_i q_i^2 / (2 sigma_i sqrt(pi))``, Ewald-summed in a periodic cell.
* An optional screening cutoff ``f_s`` removes the electrostatics inside the
  short-range sphere: the pair terms within its radius are multiplied by ``1 -
  f_s(r)`` and the Gaussian self energies are dropped.
* 3G charges are shifted uniformly to the total charge, ``q_i - (sum_j q_j -
  Q) / N``; the output ``charges`` are the shifted ones, ``charges_raw`` the
  network outputs.
* 4G hardnesses are element constants (the paper; ``hardness="element"``,
  ``J = activation(p_Z)``) or the outputs of element networks
  (``hardness="network"``). The electrostatic energy reported and added to the
  total is that of the Gaussian charges alone (the electronegativity and
  hardness terms only select the charges). The charge enters the short-range
  networks last, as ``(q - c_Z) s_Z``.

Units are those of the data (xnn: Angstrom, eV, e). ``energy_scale``,
``chi_scale`` and ``hardness_scale`` multiply the raw network outputs (one for
models trained in xnn; the hartree for models read from RuNNer files), and
``coulomb_constant`` is ``e^2 / (4 pi epsilon_0)`` in the energy and length
units.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.data.elements import atomic_number
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.charge_solve import gaussian_coulomb_matrix, solve_charges
from xnn.common.models.electrostatics import COULOMB_CONSTANT
from xnn.common.models.ops import make_activation, scatter_sum
from xnn.common.models.registry import register_model
from xnn.dnn.featurizers import AtomCenteredSymmetryFunctions
from xnn.dnn.featurizers.acsf import cutoff_function
from .base import DescriptorPotential, _ElementNetworks

#: covalent radii in Angstrom (WebElements), indexed by atomic number; ``None``
#: where no value is tabulated. The default Gaussian charge widths derive from them.
COVALENT_RADII = (
    None, 0.37, 0.32, 1.34, 0.90, 0.82, 0.77, 0.75, 0.73, 0.71, 0.69,              # H-Ne
    1.54, 1.30, 1.18, 1.11, 1.06, 1.02, 0.99, 0.97,                                # Na-Ar
    1.96, 1.74, 1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31,        # K-Zn
    1.26, 1.22, 1.19, 1.16, 1.14, 1.10,                                            # Ga-Kr
    2.11, 1.92, 1.62, 1.48, 1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48,        # Rb-Cd
    1.44, 1.41, 1.38, 1.35, 1.33, 1.30,                                            # In-Xe
    2.25, 1.98, 1.69, None, None, None, None, None, None, None, None, None,        # Cs-Dy
    None, None, None, None, 1.60, 1.50, 1.38, 1.46, 1.59, 1.28, 1.37, 1.28,        # Ho-Pt
    1.44, 1.49, 1.48, 1.47, 1.46, None, None, 1.45,                                # Au-Rn
)


def default_gaussian_widths(species: Sequence[int]) -> list[float]:
    """Gaussian charge widths in Angstrom: ``0.45 r_cov`` clamped to ``[0.25, 0.6]``.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers.

    Returns
    -------
    list of float
        One width per element.
    """
    out = []
    for z in species:
        r = COVALENT_RADII[z] if z < len(COVALENT_RADII) else None
        if r is None:
            raise ValueError(f"no covalent radius for Z={z}; give gaussian_widths explicitly")
        out.append(min(0.6, max(0.25, 0.45 * r)))
    return out


def default_symmetry_functions(species: Sequence[int], cutoff: float, n_radial: int = 6,
                               angular: bool = True) -> tuple[list[dict], list[dict]]:
    """A generic symmetry-function set: radial and angular functions for every element.

    Radial type 2 functions centred at zero with widths from the cutoff down to
    ``cutoff / n_radial`` (``eta = (m / cutoff)^2``, ``m = 1 .. n_radial``) for
    every (central, neighbour) element pair, and angular type 3 functions with
    ``eta = 1 / cutoff^2``, ``lambda = +-1`` and ``zeta = 1, 4`` for every
    central element and unordered neighbour pair, all under one cosine cutoff.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers.
    cutoff : float
        The cutoff radius.
    n_radial : int, optional
        Radial functions per element pair, by default 6.
    angular : bool, optional
        Include the angular functions, by default ``True``.

    Returns
    -------
    (list of dict, list of dict)
        The ``cutoffs`` and ``functions`` of
        :class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`.
    """
    funcs = []
    for zc in species:
        for zn in species:
            for m in range(1, n_radial + 1):
                funcs.append({"element": zc, "type": 2, "neighbors": [zn],
                              "eta": (m / cutoff) ** 2, "rs": 0.0})
        if angular:
            for za, zb in itertools.combinations_with_replacement(species, 2):
                for zeta in (1.0, 4.0):
                    for lam in (-1.0, 1.0):
                        funcs.append({"element": zc, "type": 3, "neighbors": [za, zb],
                                      "eta": 1.0 / cutoff ** 2, "lambda": lam, "zeta": zeta})
    return [{"kind": "cosine", "r_cut": float(cutoff)}], funcs


def _per_species(value, species: Sequence[int], name: str) -> list[float]:
    """A per-element value given as a sequence aligned with ``species`` or a ``{Z or symbol: v}`` dict."""
    if isinstance(value, dict):
        table = {(atomic_number(k) if isinstance(k, str) and not k.isdigit() else int(k)): float(v)
                 for k, v in value.items()}
        missing = [z for z in species if z not in table]
        if missing:
            raise ValueError(f"{name}: no value for elements {missing}")
        return [table[z] for z in species]
    values = [float(v) for v in value]
    if len(values) != len(species):
        raise ValueError(f"{name}: {len(values)} values for {len(species)} species")
    return values


@register_model("hdnnp")
class HDNNP(DescriptorPotential):
    """High-dimensional neural network potential, generations 2, 3 and 4.

    Parameters
    ----------
    species : sequence of int
        The elements (atomic numbers).
    featurizer : AtomCenteredSymmetryFunctions
        Descriptor of the short-range networks.
    generation : int, optional
        2 (default), 3 or 4.
    hidden : sequence of int or dict, optional
        Hidden-layer widths of the short-range networks, by default ``(15, 15)``.
    activation : str, sequence or dict, optional
        Activations of the short-range networks (see
        :class:`~xnn.dnn.models.base._ElementNetworks`), by default ``"tanh"``
        with a linear output.
    atomic_energies : sequence of float, optional
        Per-element energy added to every atom (the free-atom energies),
        aligned with ``species``.
    charge_featurizer : AtomCenteredSymmetryFunctions, optional
        Descriptor of the charge (3G) or electronegativity (4G) networks; the
        short-range descriptor by default.
    charge_hidden, charge_activation : optional
        Architecture of the charge or electronegativity networks; those of the
        short-range networks by default.
    constrain_charges : bool, optional
        3G: shift the network charges uniformly to the total charge (default,
        RuNNer; the "scaled" charges of Ko *et al.* 2021); ``False`` keeps the
        raw network outputs ("unscaled").
    gaussian_widths : sequence of float or dict, optional
        Width ``sigma`` of the Gaussian charge of every element (zero for
        point charges); :func:`default_gaussian_widths` by default.
    screening : dict, optional
        Cutoff function (``kind``, ``r_cut``, ``r_inner``, ``exponent`` as in
        :func:`~xnn.dnn.featurizers.acsf.cutoff_function`) that screens the
        electrostatics inside its radius; ``None`` (default) for none.
    hardness : str, optional
        4G: ``"element"`` (default, one trainable constant per element) or
        ``"network"`` (element networks on the charge descriptor).
    hardness_init : float or sequence, optional
        Initial element hardness parameters ``p_Z``, by default 10.0.
    hardness_activation : str, optional
        ``J = hardness_scale * activation(p_Z)`` of the element hardness, by
        default ``"linear"``.
    hardness_hidden, hardness_activation_nn : optional
        Architecture of the hardness networks (``hardness="network"``).
    charge_neuron : bool, optional
        4G: feed the charge to the short-range networks, by default ``True``.
    ewald_accuracy : float, optional
        Truncation of the Ewald sums of periodic cells, by default ``1e-10``.
    coulomb_constant : float, optional
        ``e^2 / (4 pi epsilon_0)`` in the model's units, by default eV Angstrom.
    energy_scale, chi_scale, hardness_scale : float, optional
        Factors on the short-range, electronegativity and hardness outputs, by
        default 1.

    Attributes
    ----------
    generation : int
        The generation.
    element_nets : _ElementNetworks
        Short-range networks.
    charge_nets : _ElementNetworks or None
        Charge (3G) or electronegativity (4G) networks.
    hardness_nets : _ElementNetworks or None
        Hardness networks (4G, ``hardness="network"``).
    hardness_raw : torch.nn.Parameter or None
        Element hardness parameters ``p_Z`` (4G, ``hardness="element"``).
    """

    head_modules = ("element_nets", "_self_energies_by_z")

    def __init__(self, species: Sequence[int], featurizer: AtomCenteredSymmetryFunctions,
                 generation: int = 2, hidden: Union[Sequence[int], dict] = (15, 15),
                 activation: Union[str, Sequence, dict] = "tanh",
                 atomic_energies: Optional[Sequence[float]] = None,
                 charge_featurizer: Optional[AtomCenteredSymmetryFunctions] = None,
                 charge_hidden=None, charge_activation=None, constrain_charges: bool = True,
                 gaussian_widths=None, screening: Optional[dict] = None,
                 hardness: str = "element", hardness_init=10.0,
                 hardness_activation: str = "linear", hardness_hidden=None,
                 hardness_activation_nn=None, charge_neuron: bool = True,
                 ewald_accuracy: float = 1e-10, coulomb_constant: float = COULOMB_CONSTANT,
                 energy_scale: float = 1.0, chi_scale: float = 1.0, hardness_scale: float = 1.0):
        if generation not in (2, 3, 4):
            raise ValueError(f"HDNNP generation must be 2, 3 or 4 (1: NNP1G), got {generation}")
        species = [int(z) for z in species]
        extra = 1 if generation == 4 and charge_neuron else 0
        super().__init__(featurizer, species, hidden, activation, True, atomic_energies, extra)
        self.generation = int(generation)
        self.charge_neuron = bool(extra)
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.charge_featurizer = charge_featurizer
        self.charge_nets = None
        self.hardness_nets = None
        self.hardness_raw = None
        self.hardness = hardness
        self.constrain_charges = bool(constrain_charges)
        self.screening = None
        if generation >= 3:
            feat = charge_featurizer if charge_featurizer is not None else featurizer
            dims = getattr(feat, "n_features", None) or feat.output_dim
            self.charge_nets = _ElementNetworks(
                species, dims, hidden if charge_hidden is None else charge_hidden,
                activation if charge_activation is None else charge_activation)
            widths = (default_gaussian_widths(species) if gaussian_widths is None
                      else _per_species(gaussian_widths, species, "gaussian_widths"))
            sigma = torch.zeros(119)
            for z, w in zip(species, widths):
                sigma[z] = w
            self.register_buffer("gaussian_widths", sigma)
            self.screening = None if screening is None else dict(screening)
            self.ewald_accuracy = float(ewald_accuracy)
            self.coulomb_constant = float(coulomb_constant)
            self.cutoff = max(self.cutoff, feat.cutoff,
                              float(self.screening["r_cut"]) if self.screening else 0.0)
        if generation == 4:
            self.register_buffer("chi_scale", torch.tensor(float(chi_scale)))
            self.register_buffer("hardness_scale", torch.tensor(float(hardness_scale)))
            if hardness == "element":
                init = ([float(hardness_init)] * len(species) if isinstance(hardness_init, (int, float))
                        else _per_species(hardness_init, species, "hardness_init"))
                self.hardness_raw = nn.Parameter(torch.tensor(init))
                self.hardness_act = make_activation(hardness_activation)
                z2i = torch.full((119,), -1, dtype=torch.long)
                for i, z in enumerate(species):
                    z2i[z] = i
                self.register_buffer("_hardness_index", z2i, persistent=False)
            elif hardness == "network":
                feat = charge_featurizer if charge_featurizer is not None else featurizer
                dims = getattr(feat, "n_features", None) or feat.output_dim
                self.hardness_nets = _ElementNetworks(
                    species, dims, hidden if hardness_hidden is None else hardness_hidden,
                    activation if hardness_activation_nn is None else hardness_activation_nn)
            else:
                raise ValueError(f"hardness must be 'element' or 'network', got {hardness!r}")
            # scaling of the charge input of the short-range networks, (q - shift) * factor
            self.register_buffer("charge_input_shift", torch.zeros(119))
            self.register_buffer("charge_input_factor", torch.ones(119))

    # the pieces
    def coulomb_matrices(self, data: AtomicGraph, dtype: torch.dtype) -> list[tuple[Tensor, Tensor]]:
        """``(atom indices, A)`` of every structure, ``A`` the Gaussian-charge Coulomb matrix in
        the model's energy unit per ``e^2``."""
        z = data.atomic_numbers
        pos = data.pos.to(dtype)
        out = []
        for b in range(data.num_graphs):
            idx = (data.batch == b).nonzero().squeeze(-1)
            out.append((idx, self.coulomb_constant * self._coulomb_matrix(data, b, pos[idx], z[idx])))
        return out

    def charges_and_parameters(self, data: AtomicGraph, desc: Tensor, desc_q: Tensor,
                               matrices: Optional[list] = None) -> dict[str, Tensor]:
        """The charges of a 3G or 4G model, with the network outputs that set them.

        ``matrices`` are the :meth:`coulomb_matrices` of ``data`` (computed when not given).
        """
        z = data.atomic_numbers
        total = (data.total_charge.to(desc.dtype) if data.total_charge is not None
                 else desc.new_zeros(data.num_graphs))
        out: dict[str, Tensor] = {}
        if self.generation == 3:
            raw = self.charge_nets(desc_q, z)
            n = data.n_atoms.to(raw.dtype)
            excess = (scatter_sum(raw, data.batch, data.num_graphs) - total) / n
            out["charges_raw"] = raw
            out["charges"] = raw - excess[data.batch] if self.constrain_charges else raw
            return out
        chi = self.chi_scale.to(desc.dtype) * self.charge_nets(desc_q, z)
        if self.hardness_nets is not None:
            hard = self.hardness_nets(desc_q, z)
        else:
            hard = self.hardness_act(self.hardness_raw.to(desc.dtype))[self._hardness_index[z]]
        hard = self.hardness_scale.to(desc.dtype) * hard
        q = torch.zeros_like(chi)
        mu = chi.new_zeros(data.num_graphs)
        if matrices is None:
            matrices = self.coulomb_matrices(data, desc.dtype)
        for b, (idx, amat) in enumerate(matrices):
            qb, mub = solve_charges(chi[idx], hard[idx], amat, total[b])
            q = q.index_put((idx,), qb)
            mu = mu.index_put((torch.tensor([b], device=mu.device),), mub)
        out.update(charges=q, electronegativities=chi, hardness=hard, chemical_potential=mu)
        return out

    def _coulomb_matrix(self, data: AtomicGraph, b: int, pos: Tensor, z: Tensor) -> Tensor:
        """``gaussian_coulomb_matrix`` of structure ``b`` (Angstrom units, no Coulomb constant)."""
        cell = None
        if data.cell is not None and data.pbc is not None and bool(data.pbc[b].any()):
            if not bool(data.pbc[b].all()):
                raise NotImplementedError("HDNNP electrostatics need full or no periodicity")
            cell = data.cell[b].to(pos.dtype)
        return gaussian_coulomb_matrix(pos, self.gaussian_widths.to(pos.dtype)[z], cell,
                                       accuracy=self.ewald_accuracy, background=True)

    def electrostatic_energy(self, data: AtomicGraph, charges: Tensor,
                             matrices: Optional[list] = None) -> Tensor:
        """Per-atom shares of the (screened) electrostatic energy of ``charges``, ``(N,)``.

        ``matrices`` are the :meth:`coulomb_matrices` of ``data`` (computed when not given).
        """
        if matrices is None:
            matrices = self.coulomb_matrices(data, charges.dtype)
        node = torch.zeros_like(charges)
        for idx, amat in matrices:
            qb = charges[idx]
            node = node.index_put((idx,), 0.5 * qb * (amat @ qb))
        if self.screening is not None:
            node = node - self.coulomb_constant * self._screening_energy(data, charges)
        return node

    def _screening_energy(self, data: AtomicGraph, q: Tensor) -> Tensor:
        """Per-atom electrostatics removed by the screening function (no Coulomb constant)."""
        s = self.screening
        vec = data.edge_vectors().to(q.dtype)
        r = torch.linalg.norm(vec, dim=-1)
        src, dst = data.edge_index[0], data.edge_index[1]
        fc = cutoff_function(r, s.get("kind", "cosine"), float(s["r_cut"]),
                             float(s.get("r_inner", 0.0) or 0.0), int(s.get("exponent", 2)))
        sigma = self.gaussian_widths.to(q.dtype)[data.atomic_numbers]
        gamma2 = sigma[src] ** 2 + sigma[dst] ** 2
        point = gamma2 <= 0
        safe = torch.where(point, torch.ones_like(gamma2), gamma2)
        kern = torch.where(point, torch.ones_like(r), torch.special.erf(r / torch.sqrt(2.0 * safe))) / r
        pair = 0.5 * q[dst] * q[src] * kern * fc
        node = scatter_sum(pair, dst, q.shape[0])
        self_term = torch.where(sigma > 0, q * q / (2.0 * math.sqrt(math.pi) * torch.where(
            sigma > 0, sigma, torch.ones_like(sigma))), torch.zeros_like(q))
        return node + self_term

    def charge_input(self, data: AtomicGraph, charges: Tensor) -> Tensor:
        """The scaled charge input ``(q - shift_Z) factor_Z`` of the 4G short-range networks."""
        z = data.atomic_numbers
        return ((charges - self.charge_input_shift.to(charges.dtype)[z])
                * self.charge_input_factor.to(charges.dtype)[z])

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Energies (and for 3G/4G charges and electrostatics) of a batch.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict[str, Tensor]
            ``node_energy``, ``energy``, ``energy_short`` (per structure),
            ``node_energy_short`` (the atomic energies of the short-range
            networks) and ``node_features`` (the short-range descriptor); 3G and 4G add
            ``charges``, ``energy_elec`` and ``dipole`` (for molecules; the
            charges times the positions), 3G ``charges_raw``, 4G
            ``electronegativities``, ``hardness`` and ``chemical_potential``.
        """
        desc = self.featurizer(data)
        z = data.atomic_numbers
        out: dict[str, Tensor] = {}
        extra = None
        matrices = None
        if self.generation >= 3:
            desc_q = desc if self.charge_featurizer is None else self.charge_featurizer(data)
            matrices = self.coulomb_matrices(data, desc.dtype)
            out.update(self.charges_and_parameters(data, desc, desc_q, matrices))
            if self.charge_neuron:
                extra = self.charge_input(data, out["charges"])[:, None]
        short = self.energy_scale.to(desc.dtype) * self.element_nets(desc, z, extra)
        short = short + self._self_energies_by_z.to(desc.dtype)[z]
        node_energy = short
        out["node_energy_short"] = short
        out["energy_short"] = self.aggregate_energy(short, data)
        if self.generation >= 3:
            elec = self.electrostatic_energy(data, out["charges"], matrices)
            node_energy = node_energy + elec
            out["energy_elec"] = self.aggregate_energy(elec, data)
            if data.cell is None:
                out["dipole"] = scatter_sum(out["charges"][:, None] * data.pos.to(desc.dtype),
                                            data.batch, data.num_graphs)
        out["node_energy"] = node_energy
        out["energy"] = self.aggregate_energy(node_energy, data)
        out["node_features"] = desc
        return out

    @torch.no_grad()
    def fit_scaling(self, graphs: Iterable[AtomicGraph], mode: Optional[str] = None,
                    charge_mode: Optional[str] = None) -> None:
        """Fit the symmetry-function scaling of every featurizer to training structures.

        Parameters
        ----------
        graphs : iterable of AtomicGraph
            Training structures; iterated once per featurizer.
        mode, charge_mode : str, optional
            Scaling modes of the short-range and the charge descriptors.
        """
        graphs = list(graphs)
        self.featurizer.fit_scaling(graphs, mode)
        if self.charge_featurizer is not None:
            self.charge_featurizer.fit_scaling(graphs, charge_mode or mode)

    @classmethod
    def from_config(cls, cfg) -> Union["HDNNP", "NNP1G"]:
        """Build an HDNNP (or, for ``generation: 1``, an :class:`NNP1G`) from a config.

        ``cfg.extra`` keys: ``species``, ``generation`` (2), ``cutoffs`` and
        ``symmetry_functions`` (the specs of
        :class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`;
        a generic set from :func:`default_symmetry_functions` with
        ``cfg.cutoff`` when absent, or radial-only functions from the older
        ``etas`` / ``rs`` keys), ``scaling``, ``scale_range``,
        ``charge_cutoffs`` / ``charge_symmetry_functions`` / ``charge_scaling``,
        ``hidden``, ``activation``, ``charge_hidden``, ``charge_activation``,
        ``atomic_energies``, ``gaussian_widths``, ``screening``, ``hardness``,
        ``hardness_init``, ``hardness_activation``, ``hardness_hidden``,
        ``charge_neuron``, ``constrain_charges``, ``ewald_accuracy``; ``runner`` (a RuNNer model
        directory) loads that model instead. 1G: see :meth:`NNP1G.from_config`.
        """
        extra = dict(cfg.extra or {})
        if extra.get("runner"):
            from xnn.dnn.common.runner import load_runner_model
            return _check_cutoff(load_runner_model(extra["runner"]), cfg)
        generation = int(extra.get("generation", 2))
        if generation == 1:
            return NNP1G.from_config(cfg)
        species = [atomic_number(z) if isinstance(z, str) else int(z)
                   for z in extra.get("species", [1, 6, 8])]
        cutoffs, functions = extra.get("cutoffs"), extra.get("symmetry_functions")
        if functions is None:
            if "etas" in extra or "rs" in extra:
                cutoffs = [{"kind": "cosine", "r_cut": float(cfg.cutoff)}]
                functions = [{"element": zc, "type": 2, "neighbors": [zn], "eta": float(eta), "rs": float(rs)}
                             for zc in species for zn in species
                             for eta in extra.get("etas", (0.05, 0.5, 2.0, 8.0))
                             for rs in extra.get("rs", (0.0,))]
            else:
                cutoffs, functions = default_symmetry_functions(species, float(cfg.cutoff))
        featurizer = AtomCenteredSymmetryFunctions(
            species, cutoffs, functions, mode=extra.get("scaling", "none"),
            scale_range=extra.get("scale_range", (0.0, 1.0)))
        charge_featurizer = None
        if extra.get("charge_symmetry_functions") is not None:
            charge_featurizer = AtomCenteredSymmetryFunctions(
                species, extra.get("charge_cutoffs", cutoffs), extra["charge_symmetry_functions"],
                mode=extra.get("charge_scaling", extra.get("scaling", "none")),
                scale_range=extra.get("scale_range", (0.0, 1.0)))
        energies = extra.get("atomic_energies")
        if energies is not None:
            energies = _per_species(energies, species, "atomic_energies")
        return _check_cutoff(cls(species, featurizer, generation=generation,
                   hidden=extra.get("hidden", (15, 15)), activation=extra.get("activation", "tanh"),
                   atomic_energies=energies, charge_featurizer=charge_featurizer,
                   charge_hidden=extra.get("charge_hidden"), charge_activation=extra.get("charge_activation"),
                   gaussian_widths=extra.get("gaussian_widths"), screening=extra.get("screening"),
                   hardness=extra.get("hardness", "element"), hardness_init=extra.get("hardness_init", 10.0),
                   hardness_activation=extra.get("hardness_activation", "linear"),
                   hardness_hidden=extra.get("hardness_hidden"),
                   charge_neuron=bool(extra.get("charge_neuron", True)),
                   constrain_charges=bool(extra.get("constrain_charges", True)),
                   ewald_accuracy=float(extra.get("ewald_accuracy", 1e-10))), cfg)


def _check_cutoff(model: "HDNNP", cfg) -> "HDNNP":
    """Refuse a model whose functions reach beyond the graph the config builds."""
    if model.cutoff > float(cfg.cutoff) + 1e-9:
        raise ValueError(f"the symmetry functions (or screening) reach {model.cutoff:g}, beyond "
                         f"the config cutoff {float(cfg.cutoff):g}; raise model.cutoff")
    return model


def _like_atom_permutations(species: Sequence[int], limit: int) -> list[list[int]]:
    """Every permutation of the atom indices that only exchanges atoms of one element."""
    groups: dict[int, list[int]] = {}
    for i, z in enumerate(species):
        groups.setdefault(int(z), []).append(i)
    size = 1
    for members in groups.values():
        size *= math.factorial(len(members))
    if size > limit:
        raise ValueError(f"symmetrizing over {size} permutations exceeds max_permutations={limit}")
    perms = [list(range(len(species)))]
    for members in groups.values():
        grown = []
        for base in perms:
            for order in itertools.permutations(members):
                p = list(base)
                for src, dst in zip(members, order):
                    p[src] = dst
                grown.append(p)
        perms = grown
    return perms


@register_model("nnp1g")
class NNP1G(InteratomicPotential):
    """First-generation neural network potential: one network for the whole system.

    The low-dimensional potentials that preceded the HDNNP map a fixed set of
    coordinates of a fixed system directly to its energy with a single
    feed-forward network (Blank *et al.* 1995, Lorenz *et al.* 2004; Fig. 1 and
    eq. 1 of Behler and Parrinello 2007). Here the coordinates are the ``n (n -
    1) / 2`` interatomic distances of a system of ``n`` atoms in a fixed order
    (or a function of them), so the energy is invariant to translations and
    rotations. As in those potentials the network is tied to its number of atoms
    and their order; ``symmetrize=True`` makes it invariant under permutations
    of like atoms by averaging the network over all of them (exact, at one
    evaluation per permutation).

    Parameters
    ----------
    species : sequence of int
        Atomic number of every atom, in the order the structures list them.
    hidden : sequence of int, optional
        Hidden-layer widths, by default ``(30, 30)``.
    activation : str, optional
        Hidden-layer activation, by default ``"tanh"`` (the output is linear).
    coordinates : str, optional
        Network inputs: ``"distances"`` (default) ``r_ij``,
        ``"inverse_distances"`` ``1 / r_ij`` or ``"exponential"`` ``exp(-r_ij /
        length)``.
    length : float, optional
        Length scale of the ``"exponential"`` coordinates, by default 1.0.
    symmetrize : bool, optional
        Average over every permutation of like atoms, by default ``False``.
    max_permutations : int, optional
        Largest permutation group accepted with ``symmetrize``, by default 5040.
    cutoff : float, optional
        Neighbour-list radius of the data pipeline; the model reads positions
        only. By default 1.0.

    Attributes
    ----------
    n_atoms : int
        Number of atoms of the system.
    net : torch.nn.Sequential
        The network.
    input_mean, input_std : Tensor
        Standardization of the inputs (:meth:`fit_scaling`).
    energy_shift, energy_scale : Tensor
        ``E = energy_shift + energy_scale * net(x)``.
    """

    head_modules = ("net", "energy_shift", "energy_scale")

    def __init__(self, species: Sequence[int], hidden: Sequence[int] = (30, 30),
                 activation: str = "tanh", coordinates: str = "distances",
                 length: float = 1.0, symmetrize: bool = False,
                 max_permutations: int = 5040, cutoff: float = 1.0):
        super().__init__()
        if coordinates not in ("distances", "inverse_distances", "exponential"):
            raise ValueError(f"unknown coordinates {coordinates!r}")
        self.species = [atomic_number(z) if isinstance(z, str) else int(z) for z in species]
        self.n_atoms = len(self.species)
        if self.n_atoms < 2:
            raise ValueError("a first-generation potential needs at least two atoms")
        self.cutoff = float(cutoff)
        self.coordinates = coordinates
        self.length = float(length)
        self.register_buffer("_species", torch.tensor(self.species, dtype=torch.long), persistent=False)
        pairs = torch.triu_indices(self.n_atoms, self.n_atoms, 1)
        self.register_buffer("_pairs", pairs, persistent=False)
        n_in = pairs.shape[1]
        perms = (_like_atom_permutations(self.species, max_permutations) if symmetrize
                 else [list(range(self.n_atoms))])
        index = {(a, b): k for k, (a, b) in enumerate(pairs.t().tolist())}
        table = [[index[(min(p[a], p[b]), max(p[a], p[b]))] for a, b in pairs.t().tolist()]
                 for p in perms]
        self.register_buffer("_perm_pairs", torch.tensor(table, dtype=torch.long), persistent=False)
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), make_activation(activation)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(n_in))
        self.register_buffer("input_std", torch.ones(n_in))
        self.register_buffer("energy_shift", torch.zeros(()))
        self.register_buffer("energy_scale", torch.ones(()))

    def inputs(self, data: AtomicGraph) -> Tensor:
        """The coordinates of every structure, ``(B, n_atoms (n_atoms - 1) / 2)``."""
        b = data.num_graphs
        if not bool((data.n_atoms == self.n_atoms).all()):
            raise ValueError(f"NNP1G is built for {self.n_atoms} atoms; got structures of "
                             f"{data.n_atoms.tolist()} atoms")
        if not bool((data.atomic_numbers.reshape(b, self.n_atoms) == self._species).all()):
            raise ValueError("the atoms are not the elements of the model in its order")
        pos = data.pos.reshape(b, self.n_atoms, 3).to(data.model_dtype)
        i, j = self._pairs
        r = torch.linalg.norm(pos[:, j] - pos[:, i], dim=-1)
        if self.coordinates == "inverse_distances":
            return 1.0 / r
        if self.coordinates == "exponential":
            return torch.exp(-r / self.length)
        return r

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Energy of each structure; the per-atom energies are its equal shares.

        Parameters
        ----------
        data : AtomicGraph
            Batched structures of the model's atoms.

        Returns
        -------
        dict[str, Tensor]
            ``node_energy`` and ``energy``.
        """
        x = self.inputs(data)[:, self._perm_pairs]                    # (B, P, n_in)
        x = (x - self.input_mean.to(x.dtype)) / self.input_std.to(x.dtype)
        e = self.net(x).squeeze(-1).mean(-1)
        energy = self.energy_shift.to(e.dtype) + self.energy_scale.to(e.dtype) * e
        node_energy = (energy / self.n_atoms)[data.batch]
        return {"node_energy": node_energy, "energy": self.aggregate_energy(node_energy, data)}

    @torch.no_grad()
    def fit_scaling(self, graphs: Iterable[AtomicGraph]) -> None:
        """Standardize the inputs and the energy on training structures.

        The input statistics are taken over every permuted copy, so they are
        the same for like-atom pairs and keep a symmetrized model invariant.

        Parameters
        ----------
        graphs : iterable of AtomicGraph
            Training structures with energies.
        """
        xs, es = [], []
        for data in graphs:
            xs.append(self.inputs(data)[:, self._perm_pairs].reshape(-1, self._pairs.shape[1]).double())
            if data.energy is not None:
                es.append(data.energy.double())
        x = torch.cat(xs)
        dtype = self.input_mean.dtype
        self.input_mean.copy_(x.mean(0).to(dtype))
        self.input_std.copy_(x.std(0).clamp(min=1e-12).to(dtype))
        if es:
            e = torch.cat(es)
            self.energy_shift.copy_(e.mean().to(dtype))
            self.energy_scale.copy_((e.std() if e.numel() > 1 else e.new_ones(())).clamp(min=1e-12).to(dtype))

    @classmethod
    def from_config(cls, cfg) -> "NNP1G":
        """Build from ``cfg.extra``: ``species`` (one entry per atom, in order),
        ``hidden``, ``activation``, ``coordinates``, ``length``, ``symmetrize``,
        ``max_permutations``; ``cfg.cutoff`` sizes the data pipeline's graph."""
        extra = dict(cfg.extra or {})
        return cls(extra["species"], hidden=extra.get("hidden", (30, 30)),
                   activation=extra.get("activation", "tanh"),
                   coordinates=extra.get("coordinates", "distances"),
                   length=float(extra.get("length", 1.0)),
                   symmetrize=bool(extra.get("symmetrize", False)),
                   max_permutations=int(extra.get("max_permutations", 5040)),
                   cutoff=float(cfg.cutoff))
