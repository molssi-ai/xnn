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
from ..models.hub.checkpoint import float_dtype

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
    dtype : torch.dtype
        Floating-point dtype of the model; graph tensors are built in it.
    eeq_reuse : bool, optional
        Carry the large-regime EEQ solve of a D4 term from one step to the
        next (:meth:`~xnn.common.models.d4.DFTD4.enable_eeq_reuse`), for
        molecular dynamics and geometry optimization; results agree with the
        fresh solve to the solver tolerance. Default ``False``.

    Raises
    ------
    ImportError
        If ASE is not installed.
    """

    implemented_properties = ["energy", "forces", "stress", "charges", "dipole"]

    def __init__(self, model, cutoff: float, device: str = "cpu", eeq_reuse: bool = False,
                 **kwargs):
        if not _HAS_ASE:
            raise ImportError("ASE is required: pip install \"xnn[ase]\"")
        super().__init__(**kwargs)
        self.model = model.to(device).eval()
        self.dtype = float_dtype(self.model)
        self.cutoff = cutoff
        self.device = device
        if eeq_reuse:
            # molecular dynamics / optimization: carry the D4 EEQ solve from one
            # step to the next (no effect on a model without a D4 term)
            from ..models.d4 import enable_eeq_reuse
            enable_eeq_reuse(self.model)

    @classmethod
    def from_pretrained(cls, source, device: str = "cpu", eeq_reuse: bool = False,
                        calculator_kwargs: dict | None = None, **kwargs) -> "XNNCalculator":
        """Build a calculator for a pre-trained model.

        Loads ``source`` with :func:`~xnn.common.models.hub.load_pretrained`
        (a registered name such as ``"mace-off23-small"``, a local checkpoint
        or model directory, a URL or a Zenodo DOI) with the stress head on,
        and takes the neighbor-list cutoff from the loaded model::

            atoms.calc = XNNCalculator.from_pretrained("mace-mp-0-medium")

        Parameters
        ----------
        source : str or pathlib.Path
            The model to load.
        device : str, optional
            Torch device. Defaults to ``"cpu"``.
        eeq_reuse : bool, optional
            As for the constructor.
        calculator_kwargs : dict, optional
            Keyword arguments for the base ASE ``Calculator``.
        **kwargs
            Options of :func:`~xnn.common.models.hub.load_pretrained`
            (``cache_dir``, ``head``, ``filename``, ``dtype``,
            ``dispersion``, ``local_files_only``, ``use_fast``, ...).

        Returns
        -------
        XNNCalculator
            The calculator.
        """
        from ..models.hub import load_pretrained
        kwargs.setdefault("compute_stress", True)
        loaded = load_pretrained(source, **kwargs)
        return cls(loaded.model, cutoff=loaded.cutoff, device=device,
                   eeq_reuse=eeq_reuse, **(calculator_kwargs or {}))

    @staticmethod
    def _charge_state(atoms):
        """The net charge and spin multiplicity an ``Atoms`` object declares in ``info``."""
        info = atoms.info
        return (info.get("total_charge", info.get("charge")),
                info.get("spin_multiplicity", info.get("multiplicity")))

    def check_state(self, atoms, tol=1e-15):
        """ASE's change detection, plus the net charge and spin multiplicity.

        ASE compares positions, numbers, cell and periodicity only, so the same
        geometry with another ``atoms.info["charge"]`` would otherwise be served
        the cached result.
        """
        changes = list(super().check_state(atoms, tol))
        if (self.atoms is not None and not changes
                and self._charge_state(self.atoms) != self._charge_state(atoms)):
            changes.append("charge")
        return changes

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
        Voigt 6-vector ordering ``[xx, yy, zz, yz, xz, xy]``. A model that
        predicts partial charges (PhysNet, BAMBOO, AIMNet2) also fills
        ``"charges"`` and ``"dipole"`` (in the model's units). The net charge and spin
        multiplicity of the structure are read from ``atoms.info`` (keys
        ``"charge"`` / ``"total_charge"`` and ``"spin_multiplicity"`` /
        ``"multiplicity"``), as :func:`~xnn.common.data.atoms_to_structure`
        does.

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
        # positions and cell in float64 whatever the model's dtype (edge vectors
        # from float32 absolute coordinates lose accuracy with the box size);
        # the model computes in its own dtype through graph.compute_dtype
        struct = {
            "pos": torch.as_tensor(np.asarray(atoms.get_positions()), dtype=torch.float64),
            "atomic_numbers": np.asarray(atoms.get_atomic_numbers()),
            "cell": (torch.as_tensor(np.asarray(atoms.get_cell()), dtype=torch.float64)
                     if atoms.pbc.any() else None),
            "pbc": np.asarray(atoms.pbc),
        }
        for key in ("total_charge", "charge"):
            if key in atoms.info:
                struct["total_charge"] = float(atoms.info[key])
                break
        for key in ("spin_multiplicity", "multiplicity"):
            if key in atoms.info:
                struct["spin_multiplicity"] = float(atoms.info[key])
                break
        if "fragment_charges" in atoms.arrays:
            struct["fragment_charges"] = np.asarray(atoms.arrays["fragment_charges"], dtype=float)
        graph = structure_to_graph(struct, self.cutoff, device=self.device)
        graph.compute_dtype = self.dtype
        out = self.model(graph)

        self.results["energy"] = float(out["energy"].sum().detach())
        if "forces" in out:
            self.results["forces"] = out["forces"].detach().cpu().numpy()
        if "stress" in out and atoms.pbc.any():
            s = out["stress"][0].detach().cpu().numpy()
            # ASE wants Voigt 6-vector
            self.results["stress"] = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        if "charges" in out:
            self.results["charges"] = out["charges"].detach().cpu().numpy()
        if "dipole" in out:
            self.results["dipole"] = out["dipole"][0].detach().cpu().numpy()
