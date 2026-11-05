"""The scriptable core of an exported NDDO model (see :func:`~xnn.common.deploy.export_torchscript_potential`).

:class:`NDDOExport` holds the tables of every element of the parameter set, made once from the model (derived
quantities, pair tables, constants, core-core parameters, reference data, correction data), and evaluates one
molecule with the functions the eager model calls: the plan, the integral build, the core-core terms, the
electron counts, the SCF (with the model's starts and the warm start of a molecular dynamics run), the
corrections and the energies. All atom pairs enter, so the exported model needs the whole molecule on one rank;
periodic cells are refused.
"""
import copy
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..common import elements as el
from .angular import L_OF, MOLECULAR, dipole_factors, pair_type
from .core import (
    build_integrals,
    charges_and_dipole,
    electron_counts,
    energy_outputs,
    initial_guess,
    make_plan,
    pm3_core_core,
    pm6_core_core,
    table_key,
)
from .corrections import batch_corrections, correction_tables
from .integrals import constants
from .scf import pack_one_center, pack_pairs, scf_solve, scf_with_starts
from .twocenter import pair_tables


class NDDOExport(nn.Module):
    """The TorchScript core of an NDDO model: ``node_features_energy_charges`` of one molecule (energies in eV,
    the Mulliken charges), the interface of the exported potentials' charge-predicting cores."""

    feather: bool
    balance: bool
    steric: bool
    average_missing: bool
    pm3: bool
    use_corrections: bool
    correction_kind: int
    peptide_barrier: float
    unrestricted: bool
    max_iter: int
    tol_density: float
    tol_energy: float
    restart_gap: float
    reuse_density: bool
    warm_start_distance: float

    def __init__(self, model):
        super().__init__()
        dtype = torch.float64
        device = model.element_table.device
        # the working precision the exported potential reads from the parameters (the tables are attributes)
        self.precision = nn.Parameter(torch.zeros(0, dtype=dtype, device=device), requires_grad=False)
        uz = sorted(int(v) for v in model.supported)
        with torch.no_grad():
            q = model._element_quantities(uz)
            self.qt: Dict[str, Tensor] = {k: v.detach().to(device, dtype) if v.is_floating_point() else v.to(device)
                                          for k, v in q.items() if isinstance(v, Tensor)}
            self.params: Dict[str, Tensor] = {k: v.detach().to(device, dtype)
                                              for k, v in model.core_parameters().items()}
        lookup = torch.full((el.MAX_Z + 1,), -1, dtype=torch.long, device=device)
        lookup[torch.tensor(uz, device=device)] = torch.arange(len(uz), device=device)
        self.lookup = lookup
        self.n_orb_el = torch.tensor(q["n_orb"], device=device)
        self.tables: Dict[str, Dict[str, Tensor]] = {}
        for ka in (1, 4, 9):
            for kb in (1, 4, 9):
                for general in (False, True):
                    if general and max(ka, kb) < 9:
                        continue
                    self.tables[table_key(ka, kb, general)] = pair_tables(q, ka, kb, general, device, dtype)
        self.c: Dict[str, Tensor] = constants(device, dtype)
        self.correction_data: Dict[str, Tensor] = correction_tables(device)
        self.occ_d = torch.tensor(el.OCC_D, device=device)
        self.shells = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], device=device)
        self.occ_ref = torch.tensor([[el.OCC_S[v], el.OCC_P[v] / 3.0, el.OCC_D[v] / 5.0] for v in uz], dtype=dtype,
                                    device=device)
        self.atom_heat = torch.tensor([el.ATOM_HEAT[v] if v < len(el.ATOM_HEAT) else 0.0
                                       for v in range(el.MAX_Z + 1)], dtype=dtype, device=device)
        self.mass = torch.tensor([el.ATOMIC_MASS[v] if v < len(el.ATOMIC_MASS) else 0.0
                                  for v in range(el.MAX_Z + 1)], dtype=dtype, device=device)
        self.dfac = torch.as_tensor(dipole_factors(), dtype=dtype, device=device)
        types = [[pair_type(L_OF[a], L_OF[b]) for b in MOLECULAR] for a in MOLECULAR]
        self.sp_mask = torch.tensor([[t == "sp" for t in row] for row in types], dtype=dtype, device=device)
        self.pd_mask = torch.tensor([[t == "pd" for t in row] for row in types], dtype=dtype, device=device)
        self.feather = bool(model.feather)
        self.balance = bool(model.balance_integrals)
        self.steric = bool(getattr(model, "steric_pairs", False))
        self.average_missing = bool(getattr(model, "average_missing_pairs", False))
        self.pm3 = model.core_form == 1
        self.use_corrections = bool(model.use_corrections)
        self.correction_kind = int(model.correction_kind)
        self.peptide_barrier = float(model.peptide_barrier)
        self.unrestricted = bool(model.unrestricted)
        self.max_iter, self.tol_density, self.tol_energy = int(model.max_iter), float(model.tol_density), \
            float(model.tol_energy)
        self.starts: List[Tuple[float, int]] = [(float(kt), int(n)) for kt, n in model.scf_starts]
        self.restart_gap = float(model.restart_gap)
        self.reuse_density = bool(model.reuse_density)
        self.warm_start_distance = float(model.warm_start_distance)
        d3 = getattr(model, "d3", None)
        if d3 is not None:
            # the scriptable paths of the dispersion term, as the exported dispersion heads pin them
            d3 = copy.deepcopy(d3).to(dtype)
            for name, value in (("regime", "dense"), ("checkpoint_triplets", False), ("recompute_pairs", False)):
                if hasattr(d3, name):
                    setattr(d3, name, value)
        self.d3 = d3
        # the previous evaluation, the start of the next one's SCF (molecular dynamics)
        self.last_z = torch.zeros(0, dtype=torch.long, device=device)
        self.last_pos = torch.zeros((0, 3), dtype=dtype, device=device)
        self.last_P = torch.zeros((0, 0, 0, 0), dtype=dtype, device=device)
        self.last_C = torch.zeros((0, 0, 0, 0), dtype=dtype, device=device)
        self.last_occ = torch.zeros((0, 0, 0), dtype=dtype, device=device)

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """No features ``(N, 0)``, the energies per atom (heat of formation, eV) and the Mulliken charges of one
        molecule; the neighbor list is not used (every atom pair enters)."""
        if bool(pbc.any()) and bool((cell.abs() > 0).any()):
            raise ValueError("an NDDO model evaluates molecules: periodic cells are not supported")
        dtype, device = pos.dtype, pos.device
        z = atomic_numbers.long()
        n = z.shape[0]
        batch = torch.zeros(n, dtype=torch.long, device=device)
        if bool((self.lookup[z] < 0).any()):
            raise ValueError("the parameter set has no parameters for an element of the structure")
        ea, n_orb, k, pa, pb, layout, has, valid = make_plan(z, batch, 1, self.lookup, self.n_orb_el, -1)
        shells = self.shells[:k]
        ints, H, W1, _, _ = build_integrals(self.qt, self.tables, self.c, layout, pos, ea, z, n_orb, k, pa, pb,
                                            has, valid, self.feather, self.balance, self.occ_d, shells)
        W2, r_pair, gab = ints["W"], ints["r"], ints["gab"]
        tore = self.qt["tore"][ea]
        za, zb = z[pa], z[pb]
        if self.pm3:
            e_core_pair = pm3_core_core(self.params, za, zb, r_pair, gab, tore[pa], tore[pb])
        else:
            e_core_pair = pm6_core_core(self.params, za, zb, r_pair, gab, tore[pa], tore[pb], self.average_missing,
                                        self.steric)
        charge = torch.full((1,), total_charge, dtype=dtype, device=device)
        mult = torch.full((1,), spin_multiplicity, dtype=dtype, device=device)
        n_e, n_alpha, n_beta, uhf = electron_counts(tore, batch, 1, charge, mult, self.unrestricted)
        guess = initial_guess(self.occ_ref, ea, shells, has, batch, 1, n_e, layout, pa.shape[0], k)
        min_basis = 48 if pos.is_cuda else 110                         # PSEUDO_MIN_BASIS
        rotate = 0.5 if pos.is_cuda else 0.05                          # PSEUDO_THRESHOLD
        warm_orbitals = pos.is_cuda                                    # WARM_ORBITALS
        with torch.no_grad():
            Hd, W1p, W2p = H.detach(), pack_one_center(W1.detach()), pack_pairs(layout, W2.detach())
            warm = (self.reuse_density and self.last_z.shape[0] == n and bool((self.last_z == z).all())
                    and self.last_P.shape[0] == (2 if uhf else 1) and self.last_P.shape[-1] == H.shape[-1]
                    and float((pos.detach() - self.last_pos).norm(p=2, dim=-1).max()) <= self.warm_start_distance)
            if warm:
                last = self.last_P
                start = last.sum(0) if uhf else last[0]
                spin: Optional[Tensor] = last if uhf else None
                C: Optional[Tensor] = self.last_C if warm_orbitals else None
                occ: Optional[Tensor] = self.last_occ if warm_orbitals else None
                result = scf_solve(layout, Hd, W1p, W2p, n_alpha, n_beta, uhf, start, self.max_iter,
                                   self.tol_density, self.tol_energy, 8, 2.0, 0, spin, rotate, min_basis, C, occ)
            else:
                result = scf_with_starts(layout, Hd, W1p, W2p, n_alpha, n_beta, uhf, guess, self.max_iter,
                                         self.tol_density, self.tol_energy, self.starts, self.restart_gap,
                                         rotate, min_basis)
            self.last_z, self.last_pos, self.last_P = z.clone(), pos.detach().clone(), result.densities.clone()
            self.last_C, self.last_occ = result.orbitals.clone(), result.occupations.clone()
        P = result.densities
        corr = torch.zeros(n, dtype=dtype, device=device)
        if self.use_corrections:
            corr = batch_corrections(self.correction_kind, torch.jit.annotate(List[int], z.tolist()), pos, batch,
                                     1, self.correction_data, self.peptide_barrier)
            d3 = self.d3
            if d3 is not None:
                same = batch.unsqueeze(0) == batch.unsqueeze(1)
                same.fill_diagonal_(False)
                pairs = same.nonzero()
                src, dst = pairs[:, 0], pairs[:, 1]
                vec = pos[dst] - pos[src]
                e_d3 = d3.evaluate(z, pos, torch.stack([src, dst]), vec, batch, 1,
                                   torch.zeros((1, 3, 3), dtype=dtype, device=device),
                                   torch.zeros((1, 3), dtype=torch.bool, device=device),
                                   torch.zeros(1, dtype=dtype, device=device))["node_energy"]
                corr = e_d3 * 23.060547830619029 + corr
        out = energy_outputs(layout, H, W1, W2, P, uhf, e_core_pair, pa, pb, batch, 1, self.qt["eisol"][ea],
                             self.atom_heat[z], corr)
        charges, _ = charges_and_dipole(layout, P, uhf, tore, k, self.dfac, self.sp_mask, self.pd_mask,
                                        self.qt["dd"][ea], self.qt["ddp"][ea, 5], n_orb, self.mass[z], batch, 1,
                                        charge, pos)
        return torch.zeros((n, 0), dtype=dtype, device=device), out["node_energy"], charges
