"""ASE Calculator wrapping a trained model.

    from ase.build import molecule
    from xnn.common.deploy import XNNCalculator
    atoms = molecule("H2O")
    atoms.calc = XNNCalculator(model, cutoff=5.0)
    atoms.get_potential_energy(); atoms.get_forces()

Requires `ase` (optional extra: pip install "xnn[ase]").
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import structure_to_graph

try:
    from ase.calculators.calculator import Calculator, all_changes
    _HAS_ASE = True
except ModuleNotFoundError:  # keep import-safe without ase
    Calculator, all_changes, _HAS_ASE = object, None, False


class XNNCalculator(Calculator):
    """ASE Calculator wrapping a trained xnn model.

    This is an `ase.calculators.calculator.Calculator` subclass that lets a
    trained model drive standard ASE workflows (energy, force and stress
    evaluation, dynamics, optimization, etc.). It converts an ``Atoms`` object
    into the model's graph representation, runs the model and stores the
    results in the ASE-expected units and layout.

    Requires ``ase`` to be installed (optional extra:
    ``pip install "xnn[ase]"``); constructing an instance raises
    ``ImportError`` if ASE is not available.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model. It is moved to ``device`` and put in ``eval`` mode.
        Its forward is expected to accept an ``AtomicGraph`` and return a dict
        with an ``"energy"`` key and optionally ``"forces"`` and ``"stress"``.
    cutoff : float
        Neighbor-list cutoff radius (in ASE length units) used when building
        the graph from an ``Atoms`` object.
    device : str, optional
        Torch device the model runs on. Defaults to ``"cpu"``.
    **kwargs
        Additional keyword arguments forwarded to the base ``Calculator``.

    Attributes
    ----------
    implemented_properties : list of str
        Properties this calculator can produce: ``"energy"``, ``"forces"``
        and ``"stress"``.
    model : torch.nn.Module
        The wrapped model (on ``device``, in eval mode).
    cutoff : float
        The neighbor-list cutoff radius.
    device : str
        The torch device string.

    Raises
    ------
    ImportError
        If ASE is not installed.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, cutoff: float, device: str = "cpu", **kwargs):
        if not _HAS_ASE:
            raise ImportError("ASE is required: pip install \"xnn[ase]\"")
        super().__init__(**kwargs)
        self.model = model.to(device).eval()
        self.cutoff = cutoff
        self.device = device

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """Compute requested properties for an ``Atoms`` object.

        Builds an :class:`AtomicGraph` from the atoms' positions, atomic
        numbers, cell and periodic-boundary flags (the cell is passed only
        when any PBC direction is active), runs the model, and populates
        ``self.results``. The total energy is the sum of the model's per-atom
        (node) energies. Forces, when produced, are stored as an
        ``(n_atoms, 3)`` NumPy array. Stress, when produced and the system is
        periodic, is converted from the model's ``3 x 3`` tensor to ASE's
        Voigt 6-vector ordering ``[xx, yy, zz, yz, xz, xy]``.

        Parameters
        ----------
        atoms : ase.Atoms or None, optional
            The atomic structure to evaluate. Passed through to the base
            ``Calculator``.
        properties : sequence of str, optional
            Names of the properties to compute. Defaults to ``("energy",)``.
        system_changes : list of str, optional
            Which aspects of the system have changed since the last call.
            Defaults to ASE's ``all_changes``.

        Returns
        -------
        None
            Results are written into ``self.results``.
        """
        super().calculate(atoms, properties, system_changes)
        struct = {
            "pos": np.asarray(atoms.get_positions()),
            "atomic_numbers": np.asarray(atoms.get_atomic_numbers()),
            "cell": np.asarray(atoms.get_cell()) if atoms.pbc.any() else None,
            "pbc": np.asarray(atoms.pbc),
        }
        graph = structure_to_graph(struct, self.cutoff, device=self.device)
        out = self.model(graph)

        self.results["energy"] = float(out["energy"].sum().detach())
        if "forces" in out:
            self.results["forces"] = out["forces"].detach().cpu().numpy()
        if "stress" in out and atoms.pbc.any():
            s = out["stress"][0].detach().cpu().numpy()
            # ASE wants Voigt 6-vector
            self.results["stress"] = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
