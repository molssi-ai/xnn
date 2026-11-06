"""The scriptable core of an exported ByteFF-Pol model (see :func:`~xnn.common.deploy.export_torchscript_potential`).

A ByteFF-Pol model predicts the parameters of the system it is bound to once; the export keeps those parameter
rows and evaluates the energy block of the reference (:class:`~xnn.hybnn.models.byteff._PolEnergy`) with the
induced-dipole solve, for molecules and clusters (every pair, no cell).
"""
# no postponed annotations: TorchScript reads the class-level attribute annotations below
from typing import List, Tuple

import torch
from torch import Tensor, nn

from xnn.ffnn.models.oplslib import KCAL_TO_EV
from xnn.ffnn.models.terms import solve_dipoles

#: the parameter rows the energy block takes, in its argument order
_ROWS = ("bond_k", "bond_x0", "angle_k", "angle_x0", "torsion_k", "torsion_sign")
_PAIR_ROWS = ("vdw_scale", "lam", "eps", "r0", "c6", "qq", "ct_eps", "ct_lam", "charge", "alpha", "damp")
_INDEX = ("src", "dst", "b0", "b1", "a1", "t_owner", "p0", "p1")


class ByteFFPolExport(nn.Module):
    """``node_features_energy_charges`` of a bound ByteFF-Pol model: the per-atom energies (eV) of one structure
    of the bound system, its charges, and no features.

    Parameters
    ----------
    model : ByteFFPol
        A bound model that takes every pair (``all_pairs``).

    Raises
    ------
    ValueError
        If the model is not bound, or evaluates periodic cells (a cutoff).
    """

    cutoff: float
    n_atoms: int
    coulomb: float
    kcal_to_ev: float
    sizes: List[int]

    def __init__(self, model: nn.Module):
        super().__init__()
        if getattr(model, "atomic_numbers", None) is None:
            raise ValueError("bind the model to its system before exporting it")
        if not model.all_pairs:
            raise ValueError("the export evaluates molecules and clusters (all_pairs: true); periodic cells "
                             "are not exported yet")
        dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
        device = model.atomic_numbers.device
        with torch.no_grad():
            R = model._rows(model._parameters_cached(), 1, dtype)
            plan = model._plan(1, device)
        self.cutoff = float(model.cutoff)
        self.n_atoms = int(model.n_atoms)
        self.coulomb = float(model.coulomb_constant)
        self.kcal_to_ev = float(KCAL_TO_EV)
        self.sizes = [int(k) for k in plan["sizes"]]
        self.energy = model._energy
        for name in _ROWS + _PAIR_ROWS:
            self.register_buffer(name, R[name].detach().clone())
        b, a, p = plan["bonds"], plan["angles"], plan["pairs"]
        index = {"src": plan["src"], "dst": plan["dst"], "b0": b[0], "b1": b[1], "a1": a[1],
                 "t_owner": plan["t_owner"], "p0": p[0], "p1": p[1]}
        for name in _INDEX:
            self.register_buffer(name, index[name].clone())
        self.register_buffer("improper", plan["improper"].to(dtype))
        self.register_buffer("field_scale", model._field_scale.to(dtype).clone())
        self.register_buffer("atomic_numbers", model.atomic_numbers.clone())

    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                                     pos: Tensor, cell: Tensor, pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """The features ``(N, 0)``, the per-atom energies ``(N,)`` in eV and the charges ``(N,)`` of the bound
        system at ``pos``; the neighbor list is not used (every pair interacts).

        Raises
        ------
        RuntimeError
            If the atoms are not those of the bound system in its order, or the structure is periodic.
        """
        if atomic_numbers.shape[0] != self.n_atoms or not bool(torch.equal(atomic_numbers, self.atomic_numbers)):
            raise RuntimeError("the atoms differ from those of the system the model was bound to")
        if bool(pbc.any()):
            raise RuntimeError("the exported ByteFF-Pol model evaluates molecules and clusters, not cells")
        e, A, field = self.energy._reference(
            pos, self.bond_k, self.bond_x0, self.angle_k, self.angle_x0, self.torsion_k, self.torsion_sign,
            self.improper, self.vdw_scale, self.lam, self.eps, self.r0, self.c6, self.qq, self.ct_eps, self.ct_lam,
            self.charge, self.alpha, self.damp, self.field_scale, self.src, self.dst, self.b0, self.b1, self.a1,
            self.t_owner, self.p0, self.p1, self.sizes, 1, self.n_atoms)
        mu = solve_dipoles(A, field)
        polarization = -0.5 * self.coulomb * (mu * field).sum(-1).reshape(-1)
        node = (e.sum(1) + polarization) * self.kcal_to_ev
        return torch.zeros((self.n_atoms, 0), dtype=pos.dtype, device=pos.device), node, self.charge
