"""xnn potentials in OpenMM: a TorchForce for openmm-torch and an ``"xnn"`` potential for openmm-ml.

A model runs inside an OpenMM simulation as an openmm-torch ``TorchForce``,
alone or on a subset of the atoms of a classical system (ML/MM)::

    from openmmml import MLPotential

    potential = MLPotential("xnn", model="mace-off23-small")
    system = potential.createSystem(topology)                                # all atoms
    mixed = potential.createMixedSystem(topology, mm_system, ligand_atoms)   # ML/MM

The ``"xnn"`` name is registered with openmm-ml through the
``openmmml.potentials`` entry point. Without openmm-ml,
:func:`create_torch_force` builds the force directly. The force evaluates the
TorchScript export of the model (:class:`~xnn.common.deploy.TorchScriptPotential`,
the reference implementation, with its own neighbor list), converts OpenMM's
nanometers and kJ/mol to and from xnn's Angstrom and eV, and returns the
energy and the forces of every particle (zero outside the model's atoms).
"""
# no `from __future__ import annotations`: TorchScript scripts OpenMMPotential and cannot read string annotations
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..models.fast import deactivate as deactivate_fast_paths
from .torchscript import TorchScriptPotential

#: kJ/mol per eV (CODATA 2018: the elementary charge times Avogadro's number, over 1000).
KJ_PER_MOL_PER_EV = 96.48533212331002
#: Angstrom per nanometer.
ANGSTROM_PER_NM = 10.0


class OpenMMPotential(nn.Module):
    """A scriptable module with openmm-torch's calling convention.

    ``forward(positions, boxvectors)`` takes the positions of every particle of
    the OpenMM system (nanometers) and, for a periodic system, the box vectors
    as rows, and returns the energy in kJ/mol and the forces in kJ/mol/nm of
    every particle. The model sees only ``atoms``.

    Parameters
    ----------
    potential : TorchScriptPotential
        The exported model.
    atomic_numbers : sequence of int
        Atomic numbers of the model's atoms, in the order of ``atoms``.
    atoms : sequence of int or None, optional
        Indices of the model's atoms among the system's particles; ``None``
        (default) for every particle.
    periodic : bool, optional
        Whether the system is periodic, by default ``False``.
    """

    periodic: bool
    subset: bool
    length: float
    energy: float

    def __init__(self, potential: TorchScriptPotential, atomic_numbers: Sequence[int],
                 atoms: Optional[Sequence[int]] = None, periodic: bool = False):
        super().__init__()
        self.potential = potential
        self.register_buffer("atomic_numbers", torch.tensor([int(z) for z in atomic_numbers],
                                                            dtype=torch.long))
        self.subset = atoms is not None
        index = torch.tensor([int(i) for i in atoms] if atoms is not None else [], dtype=torch.long)
        if self.subset and index.numel() != self.atomic_numbers.numel():
            raise ValueError(f"{index.numel()} atoms for {self.atomic_numbers.numel()} atomic numbers")
        self.register_buffer("atoms", index)
        self.periodic = bool(periodic)
        # the unit factors as attributes: scripted code cannot read module-level floats
        self.length = ANGSTROM_PER_NM
        self.energy = KJ_PER_MOL_PER_EV

    def forward(self, positions: Tensor, boxvectors: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Energy (kJ/mol, a scalar) and forces (kJ/mol/nm, every particle) of the positions (nm)."""
        pos = positions.index_select(0, self.atoms) if self.subset else positions
        pos = pos * self.length
        cell: Optional[Tensor] = None
        pbc: Optional[Tensor] = None
        if self.periodic and boxvectors is not None:
            cell = boxvectors.to(pos.dtype) * self.length
            pbc = torch.ones(3, dtype=torch.bool, device=pos.device)
        out = self.potential(pos, self.atomic_numbers, cell, pbc)
        energy = out["energy"].sum() * self.energy
        forces = out["forces"] * (self.energy * self.length)
        if self.subset:
            forces = torch.zeros_like(positions).index_copy(0, self.atoms, forces.to(positions.dtype))
        return energy.to(positions.dtype), forces.to(positions.dtype)


def _load(model, cutoff: Optional[float], dtype) -> Tuple[nn.Module, float]:
    """The bare potential and its cutoff, from a module or any source ``from_pretrained`` takes."""
    if isinstance(model, nn.Module):
        from ..models.outputs import ForceStressOutput
        bare = model.model if isinstance(model, ForceStressOutput) else model
        radius = cutoff if cutoff is not None else getattr(bare, "cutoff", None)
        if radius is None:
            raise ValueError("pass cutoff= for a model without a cutoff attribute")
    else:
        from ..models.hub import load_pretrained
        loaded = load_pretrained(model, dtype=dtype)
        bare, radius = loaded.model.model, (cutoff if cutoff is not None else loaded.cutoff)
    if dtype is not None:
        bare = bare.to(getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    return bare, float(radius)


def openmm_potential(model, atomic_numbers: Sequence[int], atoms: Optional[Sequence[int]] = None,
                     periodic: bool = False, total_charge: float = 0.0, spin_multiplicity: float = 1.0,
                     cutoff: Optional[float] = None, dtype=None) -> torch.jit.ScriptModule:
    """The scripted :class:`OpenMMPotential` of a model.

    Parameters
    ----------
    model : torch.nn.Module, str or tuple
        An xnn potential (bare or inside
        :class:`~xnn.common.models.outputs.ForceStressOutput`), or any source of
        :func:`~xnn.common.models.from_pretrained` (a hub name, a checkpoint or
        model directory); a fine-tuned checkpoint comes with its LoRA updates
        folded in and its recorded dispersion term added back; or a
        ``(potential, cutoff)`` pair already loaded.
    atomic_numbers : sequence of int
        Atomic numbers of the model's atoms.
    atoms : sequence of int or None, optional
        Their indices among the system's particles; ``None`` for all.
    periodic : bool, optional
        Whether the system is periodic, by default ``False``.
    total_charge, spin_multiplicity : float, optional
        Charge state of the model's atoms, by default neutral singlet.
    cutoff : float, optional
        Neighbor-list cutoff in Angstrom, by default the model's.
    dtype : torch.dtype or str, optional
        The model's dtype, by default its own.

    Returns
    -------
    torch.jit.ScriptModule
        The module, ready for ``openmmtorch.TorchForce``.
    """
    bare, radius = model if isinstance(model, tuple) else _load(model, cutoff, dtype)
    deactivate_fast_paths(bare)
    potential = TorchScriptPotential(bare, radius, total_charge, spin_multiplicity).eval()
    return torch.jit.script(OpenMMPotential(potential, atomic_numbers, atoms, periodic).eval())


def create_torch_force(model, topology, atoms: Optional[Sequence[int]] = None, force_group: int = 0,
                       total_charge: float = 0.0, spin_multiplicity: float = 1.0,
                       cutoff: Optional[float] = None, dtype=None):
    """An openmm-torch ``TorchForce`` evaluating ``model`` on the atoms of an OpenMM topology.

    Parameters
    ----------
    model : torch.nn.Module, str or tuple
        As for :func:`openmm_potential`, or a ``(potential, cutoff)`` pair already loaded.
    topology : openmm.app.Topology
        The system's topology; its periodic box vectors make the force periodic.
    atoms : sequence of int or None, optional
        The model's atoms; ``None`` (default) for all.
    force_group : int, optional
        The force group, by default 0.
    total_charge, spin_multiplicity, cutoff, dtype
        As for :func:`openmm_potential`.

    Returns
    -------
    openmmtorch.TorchForce
        The force, to add to an ``openmm.System``.
    """
    import openmmtorch
    elements = [a.element for a in topology.atoms()]
    index = list(range(len(elements))) if atoms is None else [int(i) for i in atoms]
    if any(elements[i] is None for i in index):
        raise ValueError("every atom of the model needs an element (virtual sites cannot be among them)")
    periodic = topology.getPeriodicBoxVectors() is not None
    module = openmm_potential(model, [elements[i].atomic_number for i in index],
                              None if atoms is None else index, periodic, total_charge,
                              spin_multiplicity, cutoff, dtype)
    force = openmmtorch.TorchForce(module)
    force.setForceGroup(force_group)
    force.setUsesPeriodicBoundaryConditions(periodic)
    force.setOutputsForces(True)
    return force


# openmm-ml imports its registered potentials (this module among them) when it is imported
# itself, so the two classes below implement its interface without importing it


class XNNPotentialImpl:
    """openmm-ml's implementation of the ``"xnn"`` potential: one TorchForce per system.

    Parameters
    ----------
    model : torch.nn.Module or str
        As for :func:`openmm_potential`.
    total_charge, spin_multiplicity, cutoff, dtype
        As for :func:`openmm_potential`; ``createSystem`` and
        ``createMixedSystem`` keyword arguments override them.
    """

    def __init__(self, model, total_charge: float = 0.0, spin_multiplicity: float = 1.0,
                 cutoff: Optional[float] = None, dtype=None):
        self.model = model
        self.options = {"total_charge": total_charge, "spin_multiplicity": spin_multiplicity,
                        "cutoff": cutoff, "dtype": dtype}
        self._loaded = None

    def _potential(self) -> Tuple[nn.Module, float]:
        """The model and its cutoff, loaded once."""
        if self._loaded is None:
            self._loaded = _load(self.model, self.options["cutoff"], self.options["dtype"])
        return self._loaded

    def addForces(self, topology, system, atoms, forceGroup, **args):
        options = {**self.options, **{k: v for k, v in args.items() if k in self.options}}
        loaded = self._potential() if (options["cutoff"], options["dtype"]) == (
            self.options["cutoff"], self.options["dtype"]) else self.model
        system.addForce(create_torch_force(loaded, topology, atoms, forceGroup, **options))

    def getSupportedEmbeddings(self):
        """No embedding of its own: openmm-ml's generic embeddings make the mixed systems."""
        return []

    def createMixedSystem(self, *args, **kwargs):
        raise NotImplementedError("xnn has no embedding of its own; use one of openmm-ml's")

    def getMLLongRange(self):
        """``True`` for a model with a long-range term, ``False`` for one whose energy is
        local (a bounded receptive field), ``None`` when the model does not say."""
        from ..models.hub.checkpoint import core_model
        bare, radius = self._potential()
        if TorchScriptPotential(bare, radius).has_long_range:
            return True
        field = getattr(core_model(bare), "receptive_field", None)
        return False if callable(field) and field() is not None else None


class XNNPotentialImplFactory:
    """Creates :class:`XNNPotentialImpl` for ``MLPotential("xnn", model=...)`` (openmm-ml)."""

    def createImpl(self, name: str, model=None, **args) -> XNNPotentialImpl:
        if model is None:
            raise ValueError('MLPotential("xnn") needs model=: an xnn model, a hub name or a checkpoint')
        return XNNPotentialImpl(model, **args)


__all__ = ["OpenMMPotential", "openmm_potential", "create_torch_force", "XNNPotentialImpl",
           "XNNPotentialImplFactory", "KJ_PER_MOL_PER_EV"]
