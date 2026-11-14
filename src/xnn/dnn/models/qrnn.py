"""QRNN (Jacobson et al., J. Chem. Theory Comput. 18, 2354, 2022), the charge recursive neural network.

An implementation of

* L. D. Jacobson, J. M. Stevenson, F. Ramezanghorbani, D. Ghoreishi,
  K. Leswing, E. D. Harder, R. Abel, "Transferable Neural Network Potential
  Energy Surfaces for Closed-Shell Organic Molecules: Extension to Ions",
  J. Chem. Theory Comput. 18, 2354 (2022),

written from the paper. QRNN is a charge-aware HDNNP: atomic charges that add
up to the net charge of the molecule enter the atomic energies as descriptors.

Every atom carries the ANI atomic environment vector ``G^AEV`` (radial and
angular symmetry functions, :class:`~xnn.dnn.featurizers.AEV`) and a
charge-weighted radial vector over all neighbors (eq 7 of the paper)::

    G^qR_i = sum_j q_j exp(-eta (r_ij - Rs)^2) f_c(r_ij)

on the radial grid of the AEV. The charges come from effective
electronegativities ``chi~``, predicted recursively by per-element networks
(eq 13) and turned into charges in closed form at the fixed net charge ``Q``
(eq 12)::

    chi~_i^(I) = NN^QR_Z(q_i^(I-1), G^qR_i, G^AEV_i)
    q_i = -(chi~_i - lambda) / J_ii,   lambda = (Q + sum_i chi~_i / J_ii) / sum_i 1 / J_ii

with fixed diagonal hardnesses ``J_ii`` (the EEQ model of Caldeweyher et al.,
J. Chem. Phys. 150, 154122, 2019, used by D4). The energy is (eq 8)::

    E = sum_i NN^E_Z(q_i, G^qR_i, G^AEV_i) + E_coul(q) + E_disp

with the Coulomb energy of the charges attenuated at short range (eq 9)::

    E_coul = 1/2 sum_{i != j} k_e q_i q_j / r_ij / (1 + exp(b (a - r_ij)))

and the dispersion correction of wB97X-D (Chai and Head-Gordon, Phys. Chem.
Chem. Phys. 10, 6615, 2008) on the C6 coefficients and radii of Grimme
(J. Comput. Chem. 27, 1787, 2006)::

    E_disp = -s6 sum_{i<j} C6_ij / r_ij^6 / (1 + a_d (r_ij / R0_ij)^-12)

``charge_model="qeq"`` gives the paper's QeqNN instead: one network per element
predicts ``X_i, S_i`` from ``G^AEV_i``, and the charges minimize the charge
equilibration energy with Gaussian charges of width ``sigma_i = sigma_0 +
S_i^2`` and electronegativity ``chi_i = X_i^2`` (eq 4-6).

Conventions worth knowing:

* the paper leaves the starting charges of the recursion open: the default
  spreads the net charge evenly over the atoms (``initial_charges="uniform"``),
  ``"zero"`` starts from zero;
* one set of charge networks serves every cycle of the recursion;
* energies are in eV, lengths in Angstrom and charges in ``e``;
* the Coulomb and dispersion sums cover every pair of a molecule; periodic
  structures need ``coulomb=False`` and ``dispersion=False`` (delta learning on
  a semiempirical method omits both terms as well);
* the charge constraint couples every atom of a structure, so the model has no
  bounded receptive field.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.models.charge_solve import closed_form_charges, gaussian_coulomb_matrix, solve_charges
from xnn.common.models.constants import ExactFloat64Constants, register_constant
from xnn.common.models.dispersion import HARTREE
from xnn.common.models.electrostatics import COULOMB_CONSTANT, all_pairs, coulomb_direct, dense_pairs
from xnn.common.models.ops import build_triplets, gather_rows, scatter_sum
from xnn.common.models.registry import register_model
from xnn.dnn.featurizers import AEV

from .base import DescriptorPotential, _ElementNetworks

#: The elements of the paper's data set: H, C, N, O, F, P, S, Cl.
QRNN_SPECIES = [1, 6, 7, 8, 9, 15, 16, 17]

#: C6 coefficients (J nm^6 / mol) and van der Waals radii R0 (Angstrom) of
#: Grimme, J. Comput. Chem. 27, 1787 (2006), Table 1, for H to Xe.
D2_C6 = (
    0.14, 0.08,
    1.61, 1.61, 3.13, 1.75, 1.23, 0.70, 0.75, 0.63,
    5.71, 5.71, 10.79, 9.23, 7.84, 5.57, 5.07, 4.61,
    10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80,
    16.99, 17.10, 16.37, 12.64, 12.47, 12.01,
    24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67,
    37.32, 38.71, 38.44, 31.74, 31.50, 29.99,
)
D2_R0 = (
    1.001, 1.012,
    0.825, 1.408, 1.485, 1.452, 1.397, 1.342, 1.287, 1.243,
    1.144, 1.364, 1.639, 1.716, 1.705, 1.683, 1.639, 1.595,
    1.485, 1.474, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.562,
    1.650, 1.727, 1.760, 1.771, 1.749, 1.727,
    1.628, 1.606, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639,
    1.672, 1.804, 1.881, 1.892, 1.892, 1.881,
)
#: One J nm^6 / mol in eV Angstrom^6: 10^6 / (N_A e) with the exact SI constants.
C6_UNIT = 1.0e6 / (6.02214076e23 * 1.602176634e-19)

_CHARGE_MODELS = ("recursive", "qeq")
_INITIAL_CHARGES = ("uniform", "zero")
_N_Z = 119


def eeq_hardness(species: Sequence[int]) -> List[float]:
    """Diagonal EEQ hardnesses of D4 (Caldeweyher et al. 2019) in eV per ``e^2``.

    The diagonal of the EEQ matrix, ``J_A + sqrt(2 / pi) / a_A``: the chemical
    hardness plus the self energy of a Gaussian charge of radius ``a_A``.

    Parameters
    ----------
    species : sequence of int
        Atomic numbers.

    Returns
    -------
    list of float
        The hardnesses, aligned with ``species``.
    """
    from xnn.common.models.d4 import _load_reference
    ref = _load_reference()
    eta, rad = ref["eeq_eta"], ref["eeq_rad"]
    return [float((eta[z] + math.sqrt(2.0 / math.pi) / rad[z]) * HARTREE) for z in species]


def chai_head_gordon_dispersion(atomic_numbers: Tensor, pos: Tensor, batch: Tensor, c6: Tensor,
                                r0: Tensor, s6: float, damping: float, num_graphs: int = 0,
                                dense: bool = False) -> Tensor:
    """Per-atom share of the wB97X-D dispersion energy of molecular structures.

    ``E = -s6 sum_{i<j} C6_ij / r^6 / (1 + a (r / R0_ij)^-12)`` over the atom
    pairs of every structure, with ``C6_ij = sqrt(C6_i C6_j)`` and ``R0_ij =
    R0_i + R0_j``; half of every pair term goes to each of its atoms.

    Parameters
    ----------
    atomic_numbers : Tensor
        Atomic numbers ``(N,)``.
    pos : Tensor
        Positions ``(N, 3)`` in Angstrom.
    batch : Tensor
        Structure index per atom ``(N,)``.
    c6 : Tensor
        C6 coefficients in eV Angstrom^6 by atomic number.
    r0 : Tensor
        Radii in Angstrom by atomic number.
    s6 : float
        Global scale.
    damping : float
        Damping parameter ``a``.
    num_graphs : int, optional
        Number of structures, when the caller knows it: one structure sums its
        pair matrix directly (no host synchronization for the pair list), by
        default 0 (unknown).
    dense : bool, optional
        Sum several structures over a masked pair matrix as well (a captured
        evaluation), by default ``False``.

    Returns
    -------
    Tensor
        Per-atom energies ``(N,)`` in eV, float64.
    """
    if num_graphs == 1 or dense:
        # the diagonal at unit distance: a finite term, masked out with the pairs across structures
        r2, keep = dense_pairs(pos, batch)
        cz, rz = c6[atomic_numbers], r0[atomic_numbers]
        r0ij = rz.unsqueeze(1) + rz.unsqueeze(0)
        damp = 1.0 / (1.0 + damping * (r0ij * r0ij / r2) ** 6)
        pair = (-0.5 * s6) * torch.sqrt(cz.unsqueeze(1) * cz.unsqueeze(0)) * damp / (r2 * r2 * r2)
        return torch.where(keep, pair, torch.zeros_like(pair)).to(torch.float64).sum(1)
    row, col = all_pairs(batch)
    zr, zc = atomic_numbers[row], atomic_numbers[col]
    d = gather_rows(pos, row) - gather_rows(pos, col)
    r2 = (d * d).sum(-1)
    c6ij = torch.sqrt(c6[zr] * c6[zc])
    r0ij = r0[zr] + r0[zc]
    damp = 1.0 / (1.0 + damping * (r0ij * r0ij / r2) ** 6)
    pair = (-0.5 * s6) * c6ij * damp / (r2 * r2 * r2)
    return scatter_sum(pair.to(torch.float64), row, atomic_numbers.shape[0])


@register_model("qrnn")
class QRNN(ExactFloat64Constants, DescriptorPotential):
    """QRNN interatomic potential: an HDNNP with recursively predicted charges.

    See the module docstring for the equations. The element networks read the
    AEV followed by the charge and the charge-weighted radial vector.

    Parameters
    ----------
    species : sequence of int, optional
        Atomic numbers, by default H, C, N, O, F, P, S, Cl.
    radial_cutoff : float, optional
        Radial cutoff of the AEV, by default 5.2.
    angular_cutoff : float, optional
        Angular cutoff of the AEV, by default 3.5.
    hidden : sequence of int or dict[int, sequence of int], optional
        Hidden widths of the energy networks (shared or per element), by
        default ``(128, 128, 64)``.
    charge_hidden : sequence of int or dict[int, sequence of int], optional
        Hidden widths of the charge networks; ``None`` (default) takes
        ``hidden``.
    activation : str, optional
        Hidden activation of every network, by default ``"gelu"``.
    atomic_energies : sequence of float or None, optional
        Per-species energy added to every atom (aligned with ``species``).
    aev_kwargs : dict, optional
        Extra arguments of :class:`~xnn.dnn.featurizers.AEV` (grids,
        ``radial_prefactor``, ``angular_cos_factor``); the defaults are the
        ANI-1x grids.
    charge_model : str, optional
        ``"recursive"`` (default, QRNN) or ``"qeq"`` (QeqNN).
    n_iterations : int, optional
        Cycles of the recursion, by default 2.
    initial_charges : str, optional
        Charges entering the first cycle: ``"uniform"`` (default, the net
        charge spread evenly) or ``"zero"``.
    hardness : sequence of float or None, optional
        Fixed hardnesses ``J_ii`` in eV per ``e^2`` aligned with ``species``;
        ``None`` (default) takes :func:`eeq_hardness`.
    min_width : float, optional
        QeqNN: the smallest Gaussian width ``sigma_0`` in Angstrom, by default
        0.05.
    coulomb : bool, optional
        Add the attenuated Coulomb energy of the charges, by default True.
    coulomb_midpoint : float, optional
        Distance ``a`` of the attenuation in Angstrom, by default 2.2.
    coulomb_steepness : float, optional
        Steepness ``b`` of the attenuation in inverse Angstrom, by default 8.5.
    dispersion : bool, optional
        Add the wB97X-D dispersion energy, by default True.
    dispersion_s6 : float, optional
        Scale of the dispersion energy, by default 1.0.
    dispersion_damping : float, optional
        Damping parameter ``a_d``, by default 6.0.

    Attributes
    ----------
    charge_nets : _ElementNetworks
        Per-element networks of the effective electronegativities (QRNN) or of
        ``X`` and ``S`` (QeqNN).
    """

    def __init__(self, species: Sequence[int] = QRNN_SPECIES, radial_cutoff: float = 5.2,
                 angular_cutoff: float = 3.5, hidden: Union[Sequence[int], dict] = (128, 128, 64),
                 charge_hidden: Optional[Union[Sequence[int], dict]] = None, activation: str = "gelu",
                 atomic_energies: Optional[Sequence[float]] = None, aev_kwargs: Optional[dict] = None,
                 charge_model: str = "recursive", n_iterations: int = 2, initial_charges: str = "uniform",
                 hardness: Optional[Sequence[float]] = None, min_width: float = 0.05,
                 coulomb: bool = True, coulomb_midpoint: float = 2.2, coulomb_steepness: float = 8.5,
                 dispersion: bool = True, dispersion_s6: float = 1.0, dispersion_damping: float = 6.0):
        if charge_model not in _CHARGE_MODELS:
            raise ValueError(f"charge_model must be one of {_CHARGE_MODELS}, got {charge_model!r}")
        if initial_charges not in _INITIAL_CHARGES:
            raise ValueError(f"initial_charges must be one of {_INITIAL_CHARGES}, got {initial_charges!r}")
        if int(n_iterations) < 1:
            raise ValueError(f"n_iterations must be at least 1, got {n_iterations}")
        species = [int(z) for z in species]
        if dispersion and max(species) > len(D2_C6):
            raise ValueError(f"the dispersion parameters cover H to Xe (Z <= {len(D2_C6)}); "
                             f"got species {species}; set dispersion=False")
        aev = AEV(species, radial_cutoff=radial_cutoff, angular_cutoff=angular_cutoff, **(aev_kwargs or {}))
        n_charge = int(aev.radial.etas.numel() * aev.radial.rs.numel())
        super().__init__(aev, species, hidden, activation=activation, atomic_energies=atomic_energies,
                         extra_inputs=1 + n_charge)
        self.n_charge_features = n_charge
        self.charge_model = charge_model
        self.n_iterations = int(n_iterations)
        self.initial_charges = initial_charges
        widths = hidden if charge_hidden is None else charge_hidden
        if charge_model == "recursive":
            self.charge_nets = _ElementNetworks(species, aev.output_dim, widths, activation,
                                                extra_inputs=1 + n_charge)
        else:
            self.charge_nets = _ElementNetworks(species, aev.output_dim, widths, activation, n_outputs=2)

        values = eeq_hardness(species) if hardness is None else [float(h) for h in hardness]
        if len(values) != len(species):
            raise ValueError(f"hardness: got {len(values)} values for {len(species)} species")
        if min(values) <= 0.0:
            raise ValueError(f"hardness must be positive, got {values}")
        inverse = [0.0] * _N_Z
        for z, h in zip(species, values):
            inverse[z] = 1.0 / h
        register_constant(self, "_inverse_hardness_by_z", inverse, persistent=False)
        self.min_width = float(min_width)

        self.coulomb = bool(coulomb)
        self.coulomb_midpoint = float(coulomb_midpoint)
        self.coulomb_steepness = float(coulomb_steepness)
        self.coulomb_constant = COULOMB_CONSTANT
        self.dispersion = bool(dispersion)
        self.dispersion_s6 = float(dispersion_s6)
        self.dispersion_damping = float(dispersion_damping)
        c6 = [0.0] * _N_Z
        r0 = [1.0] * _N_Z
        for z in range(1, len(D2_C6) + 1):
            c6[z] = D2_C6[z - 1] * C6_UNIT
            r0[z] = D2_R0[z - 1]
        register_constant(self, "_c6_by_z", c6, persistent=False)
        register_constant(self, "_r0_by_z", r0, persistent=False)
        # the accuracy of the lattice sums of QeqNN's Gaussian charges in a periodic cell
        self.ewald_accuracy = 1e-10
        # the fast path evaluates the networks of all elements together when they share one shape
        for nets in (self.element_nets, self.charge_nets):
            nets.grouping = nets.groupable()

    @torch.jit.unused
    def fast_policies(self):
        """The ``"auto"`` policies (evaluation, training) of :mod:`xnn.dnn.fast.qrnn`."""
        from xnn.dnn.fast.qrnn import AUTO_POLICY, AUTO_POLICY_TRAINING
        return AUTO_POLICY, AUTO_POLICY_TRAINING

    def receptive_field(self) -> Optional[float]:
        """``None``: the charge constraint couples every atom of a structure."""
        return None

    def _charge_features(self, basis: Tensor, q: Tensor, edge_index: Tensor, n: int) -> Tensor:
        """The charge-weighted radial vectors ``(N, P)`` of the charges ``q`` (eq 7)."""
        weighted = basis * gather_rows(q, edge_index[0]).unsqueeze(-1)
        return scatter_sum(weighted, edge_index[1], n)

    def _qeq_charges(self, chi: Tensor, sigma: Tensor, pos: Tensor, batch: Tensor, num_graphs: int,
                     total: Tensor, cell: Optional[Tensor], periodic: Tensor) -> Tensor:
        """QeqNN charges: the charge equilibration of every structure (eq 4-5)."""
        q = torch.zeros_like(chi)
        for b in range(num_graphs):
            idx = torch.nonzero(batch == b).squeeze(-1)
            box: Optional[Tensor] = None
            if cell is not None and bool(periodic[b]):
                box = cell[b].to(chi.dtype)
            amat = self.coulomb_constant * gaussian_coulomb_matrix(
                pos[idx].to(chi.dtype), sigma[idx], box, self.ewald_accuracy, True)
            qb = solve_charges(chi[idx], torch.zeros_like(idx, dtype=chi.dtype), amat, total[b].reshape(1))[0]
            q = q.index_copy(0, idx, qb)
        return q

    def compute(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor, pos: Tensor,
                batch: Tensor, num_graphs: int, total_charge: Tensor, cell: Optional[Tensor],
                pbc: Optional[Tensor], triplets: Optional[Tuple[Tensor, Tensor]] = None,
                width: int = 0) -> Dict[str, Tensor]:
        """Energies and charges of a batch from tensors (TorchScript-compatible).

        Parameters
        ----------
        atomic_numbers : Tensor
            Atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor, row 1 the center.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.
        pos : Tensor
            Positions ``(N, 3)`` (the all-pair terms and QeqNN).
        batch : Tensor
            Structure index per atom ``(N,)``.
        num_graphs : int
            Number of structures ``B``.
        total_charge : Tensor
            Net charge of every structure ``(B,)``.
        cell : Tensor or None
            Cells ``(B, 3, 3)``.
        pbc : Tensor or None
            Periodicity flags ``(B, 3)``.
        triplets : tuple of Tensor, optional
            The enumerated triplets of the AEV (:class:`QRNNPlanner`).
        width : int, optional
            A positive value evaluates without reading a value back from the
            device (a captured step): the networks of every element together
            with blocks of ``width`` rows, the all-pair terms as masked
            matrices. By default 0.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"charges"`` ``(N,)``,
            ``"electronegativities"`` ``(N,)`` (the effective ones of the last
            cycle, or QeqNN's ``chi``), ``"node_energy_short"``,
            ``"node_energy_elec"`` and ``"node_energy_dispersion"`` ``(N,)``
            and ``"node_features"`` (the AEV).
        """
        z = atomic_numbers
        n = z.shape[0]
        desc = self.featurizer.compute(z, edge_index, edge_vec, n, triplets)
        dtype = desc.dtype
        r = torch.linalg.norm(edge_vec, dim=-1)
        radial = self.featurizer.radial
        basis = radial.edge_terms(r, radial.prefactor * radial.cutoff_fn(r))
        total = total_charge.to(dtype)
        periodic = torch.zeros(num_graphs, dtype=torch.bool, device=z.device)
        if pbc is not None:
            periodic = pbc.any(dim=1)
        # the atoms sorted by element once, for every network evaluation (a captured step sorts on the
        # device in each evaluation instead)
        captured = width > 0
        plan = self.element_nets.plan(z) if not captured else (z, z, [0])
        if self.charge_model == "recursive":
            inverse = self._inverse_hardness_by_z.to(dtype)[z]
            if self.initial_charges == "uniform":
                counts = scatter_sum(torch.ones_like(desc[:, 0]), batch, num_graphs)
                q = gather_rows(total / counts, batch)
            else:
                q = torch.zeros_like(desc[:, 0])
            chi = torch.zeros_like(q)
            for _ in range(self.n_iterations):
                inputs = torch.cat([q.unsqueeze(-1), self._charge_features(basis, q, edge_index, n)], dim=-1)
                if captured and not torch.jit.is_scripting():
                    chi = self.charge_nets.forward_padded(desc, z, inputs, width)
                else:
                    chi = self.charge_nets(desc, z, inputs, plan)
                q = closed_form_charges(chi, inverse, total, batch, num_graphs)
        else:
            if captured:
                raise ValueError("QeqNN (charge_model='qeq') does not run in a captured step")
            xs = self.charge_nets(desc, z, None, plan)
            chi = xs[:, 0] * xs[:, 0]
            sigma = self.min_width + xs[:, 1] * xs[:, 1]
            q = self._qeq_charges(chi, sigma, pos, batch, num_graphs, total, cell, periodic)
        inputs = torch.cat([q.unsqueeze(-1), self._charge_features(basis, q, edge_index, n)], dim=-1)
        if captured and not torch.jit.is_scripting():
            short = self.element_nets.forward_padded(desc, z, inputs, width)
        else:
            short = self.element_nets(desc, z, inputs, plan)
        short = short + self._self_energies_by_z.to(dtype)[z]
        elec = torch.zeros(n, dtype=torch.float64, device=z.device)
        disp = torch.zeros(n, dtype=torch.float64, device=z.device)
        if self.coulomb or self.dispersion:
            # a captured step's planner refuses periodic structures instead (no host read here)
            if not captured and bool(periodic.any()):
                raise ValueError("the QRNN Coulomb and dispersion terms sum every pair of a molecule; "
                                 "periodic structures need coulomb=False and dispersion=False")
            p = pos.to(dtype)
            if self.coulomb:
                elec = coulomb_direct(q, p, batch, self.coulomb_constant, num_graphs,
                                      self.coulomb_midpoint, self.coulomb_steepness, captured)
            if self.dispersion:
                disp = chai_head_gordon_dispersion(z, p, batch, self._c6_by_z.to(dtype),
                                                   self._r0_by_z.to(dtype), self.dispersion_s6,
                                                   self.dispersion_damping, num_graphs, captured)
        node_energy = short + elec + disp if (self.coulomb or self.dispersion) else short
        return {"node_energy": node_energy, "charges": q, "electronegativities": chi,
                "node_energy_short": short, "node_energy_elec": elec, "node_energy_dispersion": disp,
                "node_features": desc}

    @torch.jit.export
    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor,
                                     edge_vec: Tensor, pos: Tensor, cell: Tensor,
                                     pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript core of one structure: the AEV, atomic energies and charges.

        Parameters
        ----------
        atomic_numbers : Tensor
            Atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor, row 1 the center.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.
        pos : Tensor
            Positions ``(N, 3)``.
        cell : Tensor
            Cell ``(3, 3)`` (used by QeqNN where ``pbc`` is set).
        pbc : Tensor
            Periodicity flags ``(3,)``.
        total_charge : float
            Net charge of the structure.
        spin_multiplicity : float
            Unused.

        Returns
        -------
        tuple of Tensor
            The AEV ``(N, D)``, the atomic energies ``(N,)`` and the charges
            ``(N,)``.
        """
        n = atomic_numbers.shape[0]
        batch = torch.zeros(n, dtype=torch.long, device=edge_vec.device)
        total = torch.full((1,), total_charge, dtype=edge_vec.dtype, device=edge_vec.device)
        flags: Optional[Tensor] = None
        box: Optional[Tensor] = None
        if pbc.numel() == 3:
            flags = pbc.reshape(1, 3).to(torch.bool)
            box = cell.reshape(1, 3, 3)
        out = self.compute(atomic_numbers, edge_index, edge_vec, pos, batch, 1, total, box, flags)
        return out["node_features"], out["node_energy"], out["charges"]

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript core of one neutral structure from its edges alone: the AEV and the atomic
        energies. Only for the recursive charges without the Coulomb and dispersion terms, which
        need the positions (:meth:`node_features_energy_charges` serves every setting)."""
        if self.coulomb or self.dispersion or self.charge_model != "recursive":
            raise RuntimeError("QRNN with the Coulomb or dispersion terms or the QeqNN charges needs the "
                               "positions: export through node_features_energy_charges "
                               "(export_torchscript_potential)")
        empty = torch.zeros(0, dtype=edge_vec.dtype, device=edge_vec.device)
        out = self.node_features_energy_charges(atomic_numbers, edge_index, edge_vec, empty, empty,
                                                empty.to(torch.bool), 0.0, 1.0)
        return out[0], out[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies and charges of a batched graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph; its ``total_charge`` (per structure, zero
            when absent) fixes the sum of the charges.

        Returns
        -------
        dict of str to Tensor
            The outputs of :meth:`compute`, with ``"energy"`` ``(B,)``,
            ``"energy_short"``, ``"energy_elec"`` and ``"energy_dispersion"``
            per structure and, for molecules, ``"dipole"`` ``(B, 3)`` (the
            charges times the positions).
        """
        index_plan = getattr(data, "index_plan", None)
        triplets: Optional[Tuple[Tensor, Tensor]] = None
        width = 0
        if index_plan is not None:
            # a padded batch of fixed counts (a captured step): the enumerated triplets, the grouped
            # networks with blocks as long as the batch, no host reads
            from xnn.common.models.fast import deactivate
            deactivate(self)
            triplets = (index_plan["aev_e1"].long(), index_plan["aev_e2"].long())
            width = data.num_nodes
        else:
            self._select_fast(int(data.edge_index.shape[1]))
        edge_vec = data.edge_vectors()
        total = (data.total_charge if data.total_charge is not None
                 else torch.zeros(data.num_graphs, dtype=edge_vec.dtype, device=edge_vec.device))
        out = self.compute(data.atomic_numbers, data.edge_index, edge_vec, data.pos, data.batch,
                           data.num_graphs, total, data.cell, data.pbc, triplets, width)
        out["energy"] = self.aggregate_energy(out["node_energy"], data)
        out["energy_short"] = self.aggregate_energy(out["node_energy_short"], data)
        out["energy_elec"] = self.aggregate_energy(out["node_energy_elec"], data)
        out["energy_dispersion"] = self.aggregate_energy(out["node_energy_dispersion"], data)
        if index_plan is not None or data.pbc is None or not bool(data.pbc.any()):
            q = out["charges"]
            out["dipole"] = scatter_sum(q.unsqueeze(-1) * data.pos.to(q.dtype), data.batch, data.num_graphs)
        return out

    @torch.jit.unused
    def index_planner(self) -> "QRNNPlanner":
        """The planner of a captured evaluation or training step (:class:`QRNNPlanner`).

        Raises
        ------
        ValueError
            For QeqNN, whose charge equilibration does not run in a captured step.
        """
        if self.charge_model != "recursive":
            raise ValueError("a captured step needs charge_model='recursive' (QeqNN solves a linear system "
                             "per structure)")
        return QRNNPlanner(self.coulomb or self.dispersion)

    @classmethod
    def from_config(cls, cfg) -> "QRNN":
        """Build a :class:`QRNN` from a configuration object.

        Every option is read from ``cfg.extra`` under its constructor name.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The model config.

        Returns
        -------
        QRNN
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        species = coerce_species(extra.get("species"), default=QRNN_SPECIES)
        atomic_energies = coerce_per_species(extra.get("atomic_energies"), species, "atomic_energies")
        hardness = coerce_per_species(extra.get("hardness"), species, "hardness")
        return cls(
            species=species,
            radial_cutoff=float(extra.get("radial_cutoff", 5.2)),
            angular_cutoff=float(extra.get("angular_cutoff", 3.5)),
            hidden=extra.get("hidden", [128, 128, 64]),
            charge_hidden=extra.get("charge_hidden", None),
            activation=str(extra.get("activation", "gelu")),
            atomic_energies=None if atomic_energies is None else atomic_energies.tolist(),
            aev_kwargs=extra.get("aev_kwargs", None),
            charge_model=str(extra.get("charge_model", "recursive")),
            n_iterations=int(extra.get("n_iterations", 2)),
            initial_charges=str(extra.get("initial_charges", "uniform")),
            hardness=None if hardness is None else hardness.tolist(),
            min_width=float(extra.get("min_width", 0.05)),
            coulomb=bool(extra.get("coulomb", True)),
            coulomb_midpoint=float(extra.get("coulomb_midpoint", 2.2)),
            coulomb_steepness=float(extra.get("coulomb_steepness", 8.5)),
            dispersion=bool(extra.get("dispersion", True)),
            dispersion_s6=float(extra.get("dispersion_s6", 1.0)),
            dispersion_damping=float(extra.get("dispersion_damping", 6.0)),
        )


class QRNNPlanner:
    """The triplets of the AEV, enumerated outside the model step for a captured step.

    A captured evaluation or training step (:mod:`xnn.common.train.graphed`) needs fixed shapes and no
    host synchronization, while the triplet count of a structure is known only after enumerating them.
    The planner enumerates every pair of edges at each center on the CPU (the pairs beyond the angular
    cutoff add exact zeros), and the padded batch carries them as ``index_plan``. Holds no tensors, so
    data-loader workers receive it cheaply.

    Parameters
    ----------
    all_pairs : bool
        Whether the model sums every atom pair (Coulomb or dispersion term), which needs molecules.
    """

    #: The count each plan entry indexes (for batching the plans of single structures).
    offsets = {"aev_e1": "edges", "aev_e2": "edges"}
    #: Batches are padded by an inert structure to fixed counts.
    pads = True

    def __init__(self, all_pairs: bool):
        self.all_pairs = bool(all_pairs)

    def __call__(self, data: AtomicGraph) -> Dict[str, Tensor]:
        if self.all_pairs and data.pbc is not None and bool(data.pbc.any()):
            raise ValueError("the QRNN Coulomb and dispersion terms sum every pair of a molecule; "
                             "periodic structures need coulomb=False and dispersion=False")
        e1, e2, _ = build_triplets(data.edge_index, data.num_nodes)
        return {"aev_e1": e1, "aev_e2": e2}

    @staticmethod
    def sizes(plan: Dict[str, Tensor]) -> Dict[str, int]:
        """The sizes a padded batch fixes: the triplet count."""
        return {"triplets": int(plan["aev_e1"].shape[0])}

    @staticmethod
    def pad(plan: Dict[str, Tensor], sizes: Dict[str, int], dummy_edge: int) -> Dict[str, Tensor]:
        """``plan`` filled up to ``sizes["triplets"]`` with triplets of the padding edge ``dummy_edge``."""
        fill = plan["aev_e1"].new_full((sizes["triplets"] - plan["aev_e1"].shape[0],), dummy_edge)
        return {"aev_e1": torch.cat([plan["aev_e1"], fill]), "aev_e2": torch.cat([plan["aev_e2"], fill])}
