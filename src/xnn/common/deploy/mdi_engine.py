"""MDI engine serving any trained xnn model to an external driver.

The `MolSSI Driver Interface <https://github.com/MolSSI-MDI/MDI_Library>`_
(MDI) lets simulation codes (LAMMPS, QCEngine, SEAMM, ...) drive an external
"engine" through a small command protocol. :class:`MDIEngine` implements the
engine side for xnn: the driver sends the system (``>NATOMS``, ``>ELEMENTS``,
``>CELL``, ``>COORDS``, optionally ``>TOTCHARGE``) and requests results
(``<ENERGY``, ``<FORCES``, ``<STRESS``), and the engine evaluates the wrapped
model each time the geometry changes.

The engine is model agnostic: it speaks to the model exclusively through the
:class:`~xnn.common.data.AtomicGraph` contract shared by every model in the
library (the same contract :class:`XNNCalculator` uses), so any registered
family (MACE, NequIP, Allegro, CACE, SchNet, ANI, PhysNet, HDNNP, BAMBOO, ...)
works unchanged. Graphs are built with the library's own
:func:`~xnn.common.data.structure_to_graph`; forces and stress come from the
:class:`~xnn.common.models.ForceStressOutput` wrapper.

Typical use, from a trainer checkpoint::

    from xnn.common.deploy import MDIEngine
    engine = MDIEngine.from_checkpoint("runs/exp/best.pt", device="cuda")
    engine.run("-role ENGINE -name xnn -method TCP -port 8021 -hostname localhost")

or from the command line (see :func:`main`)::

    xnn mdi --ckpt runs/exp/best.pt -mdi "-role ENGINE -name xnn -method TCP ..."

A checkpoint trained without dispersion can be served with a D3 / D4
correction added on top (``--dispersion d4`` or a YAML mapping such as
``"{name: d4, cutoff_pair: 12.0, switch_width_pair: 2.0}"``), and the
system's net charge (used by D4's EEQ charges and by charge-aware models)
is set with ``--total-charge`` or by the driver through ``>TOTCHARGE``.

Requires the ``pymdi`` package (``pip install pymdi``); MPI communication
additionally requires ``mpi4py``.

Units: MDI communicates in atomic units (Bohr / Hartree) while xnn models
follow the library's ASE-style convention of angstrom / eV (the units of the
training data). The engine converts at the boundary in both directions.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from ..data import structure_to_graph

try:
    import mdi
    _HAS_MDI = True
except ModuleNotFoundError:  # keep import-safe without pymdi
    mdi, _HAS_MDI = None, False

logger = logging.getLogger(__name__)

# Wire conversion constants (CODATA 2018, the values used in
# xnn.common.data.hub). Public on purpose: a driver that converts with the
# same constants gets bit-clean round trips. Codes with a different CODATA
# vintage (e.g. ase.units, CODATA 2014) differ at the 1e-8 relative level.
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_EV = 27.211386245988

# MDI commands the engine understands, registered on the @DEFAULT node.
_COMMANDS = (
    ">NATOMS", ">COORDS", ">CELL", ">ELEMENTS", ">TOTCHARGE",
    "<ENERGY", "<FORCES", "<STRESS",
    "SCF", "EXIT",
)


def _float_dtype(source) -> torch.dtype:
    """Floating-point dtype of a module's parameters or of a state dict."""
    tensors = (source.values() if isinstance(source, dict)
               else source.parameters())
    for t in tensors:
        if torch.is_tensor(t) and t.is_floating_point():
            return t.dtype
    return torch.get_default_dtype()


def _has_dispersion(model: torch.nn.Module) -> bool:
    """Whether a model already includes a D3 / D4 wrapper (at any nesting)."""
    from ..models.dispersion import DispersionCorrection
    from ..models.les import LatentEwald
    while True:
        if isinstance(model, DispersionCorrection):
            return True
        if isinstance(model, LatentEwald):
            model = model.model
        else:
            return False


class MDIEngine:
    """MDI engine exposing a trained xnn model to an external driver.

    Holds the driver-supplied system state (atom count, elements, cell,
    coordinates) and lazily re-evaluates the model whenever a result is
    requested after the geometry changed. Each evaluation converts the MDI
    atomic-unit inputs to angstrom, builds an
    :class:`~xnn.common.data.AtomicGraph` via
    :func:`~xnn.common.data.structure_to_graph` (which handles molecular and
    periodic systems alike), runs the model, and converts the eV / angstrom
    outputs back to Hartree / Bohr.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model. It is moved to ``device``, put in ``eval`` mode and
        its parameter gradients are disabled. Its forward must accept an
        ``AtomicGraph`` and return a dict with an ``"energy"`` key and, for
        the ``<FORCES`` / ``<STRESS`` commands, ``"forces"`` / ``"stress"``
        keys; wrap a bare model in
        :class:`~xnn.common.models.ForceStressOutput` to provide them (as
        :meth:`from_checkpoint` does).
    cutoff : float
        Neighbor-list cutoff radius in angstrom used when building the graph.
        Use the model's own ``cutoff`` (a dispersion wrapper widens it beyond
        the core model's radius), as :meth:`from_checkpoint` does.
    device : str, optional
        Torch device the model runs on. Defaults to ``"cpu"``.
    total_charge : float, optional
        Net charge of the system in units of e, passed to the model as
        :attr:`~xnn.common.data.AtomicGraph.total_charge` (D4's EEQ charges
        and charge-aware models such as PhysNet use it). Defaults to 0; a
        driver can change it at run time with ``>TOTCHARGE``.

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped model (on ``device``, in eval mode).
    cutoff : float
        The neighbor-list cutoff radius in angstrom.
    device : torch.device
        The torch device.
    total_charge : float
        The current net charge (e).
    dtype : torch.dtype
        Floating-point dtype of the model parameters; graph tensors are built
        in this dtype.
    energy, forces, stress
        Results of the latest evaluation, in MDI atomic units (Hartree,
        Hartree/Bohr, Hartree/Bohr^3). ``None`` before the first evaluation
        (``stress`` also for non-periodic systems).
    """

    def __init__(self, model: torch.nn.Module, cutoff: float,
                 device: str = "cpu", total_charge: float = 0.0):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cutoff = float(cutoff)
        self.total_charge = float(total_charge)
        self.dtype = _float_dtype(model)

        # system state, set by driver commands (MDI atomic units)
        self.natoms: int | None = None
        self.atomic_numbers: np.ndarray | None = None
        self.coords_bohr: np.ndarray | None = None   # (N, 3)
        self.cell_bohr: np.ndarray | None = None     # (3, 3); None => molecular

        # results of the latest evaluation (MDI atomic units)
        self.energy: float | None = None
        self.forces: np.ndarray | None = None
        self.stress: np.ndarray | None = None
        self._needs_calculation = True

        # timing
        self._n_calc = 0
        self._t_total = 0.0
        self._t_graph = 0.0     # neighbour list + tensor assembly
        self._t_model = 0.0     # the forward pass
        self._t_extract = 0.0   # pulling results back to the host
        self._n_edges = 0       # from the most recent graph

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu",
                        dtype: torch.dtype | None = None,
                        dispersion: Any = None,
                        total_charge: float = 0.0) -> "MDIEngine":
        """Build an engine from a trainer checkpoint (``best.pt``).

        The checkpoint is the dictionary written by
        :meth:`~xnn.common.train.Trainer.save`: ``{"model": state_dict,
        "cfg": Config}``. The model is rebuilt with
        :func:`~xnn.common.models.build_model` from the stored config,
        wrapped in :class:`~xnn.common.models.ForceStressOutput` (with the
        stress head enabled, matching the state-dict layout the trainer
        saves), and the weights are loaded.

        The model is built in float64, so the constant tables of the
        physics terms (D3 / D4 reference data, LES kernels) hold their exact
        values, and cast once afterwards, to ``dtype`` or to the
        checkpoint's own floating-point dtype. Building in float32 and
        upcasting would keep float32-rounded tables, which costs about
        5e-8 hartree in a D4 energy even when serving in float64.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        device : str, optional
            Torch device the model runs on. Defaults to ``"cpu"``.
        dtype : torch.dtype, optional
            Floating-point dtype to serve in (``torch.float64`` for NVE energy
            conservation with a float32-trained model, ``torch.float32`` for
            speed). Defaults to the dtype of the checkpoint's weights.
        dispersion : dict, str or None, optional
            Add a D3 / D4 dispersion correction to a checkpoint that was
            trained without one: ``"d4"`` / ``"d3"`` for the defaults or a
            mapping as in the config's ``extra["dispersion"]`` (see
            :func:`~xnn.common.models.add_dispersion`). Refused when the
            checkpoint already carries dispersion, which would count it
            twice. Defaults to ``None`` (serve the checkpoint as is).
        total_charge : float, optional
            Net charge of the system in units of e, by default 0. A driver can
            change it at run time with ``>TOTCHARGE``.

        Returns
        -------
        MDIEngine
            An engine wrapping the restored model, with the neighbor-list
            cutoff taken from the built model (a dispersion wrapper widens it
            beyond the config's core-model radius).

        Raises
        ------
        ValueError
            If ``dispersion`` is given for a checkpoint whose model already
            includes a dispersion correction.
        """
        from ..models import add_dispersion, build_model, ForceStressOutput
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            base = build_model(cfg.model)
            # ForceStressOutput has no parameters of its own; loading through
            # it matches the trainer's state-dict layout
            ForceStressOutput(base).load_state_dict(ckpt["model"])
            if dispersion is not None:
                if _has_dispersion(base):
                    raise ValueError(
                        f"{path} already includes a dispersion correction "
                        f"({cfg.model.name} with extra['dispersion']); adding "
                        f"another one would count dispersion twice")
                base = add_dispersion(base, dispersion)
                logger.info("added %s dispersion on top of the checkpoint",
                            type(base).__name__)
        finally:
            torch.set_default_dtype(prev_dtype)
        if dtype is None:
            dtype = _float_dtype(ckpt["model"])
        model = ForceStressOutput(base, compute_stress=True).to(dtype)
        cutoff = float(getattr(base, "cutoff", cfg.model.cutoff))
        logger.info("Loaded %s checkpoint %s (cutoff=%.3f A, %s, total charge %g)",
                    cfg.model.name, path, cutoff, str(dtype).replace("torch.", ""),
                    total_charge)
        return cls(model, cutoff=cutoff, device=device, total_charge=total_charge)

    # evaluation

    def calculate(self) -> None:
        """Evaluate the model on the current system state.

        Converts positions and cell from Bohr to angstrom, builds the graph
        with :func:`~xnn.common.data.structure_to_graph` (tensors are created
        in the model's dtype so float32 and float64 models both work; the
        current ``total_charge`` rides along), runs the model and stores
        ``energy`` (Hartree), ``forces`` (Hartree/Bohr) and, for periodic
        systems, ``stress`` (Hartree/Bohr^3).

        Raises
        ------
        RuntimeError
            If no coordinates or elements have been received yet.
        """
        if self.coords_bohr is None or self.atomic_numbers is None:
            raise RuntimeError(
                "cannot calculate: driver has not sent >ELEMENTS/>COORDS yet")
        t0 = time.perf_counter()

        # Build the graph on the device the model runs on, rather than on the
        # CPU and moving it afterwards. The neighbour list is the expensive
        # part and it is rebuilt every step, so where it runs decides the cost
        # of the step: on 5001 atoms this was 1.8 s of CPU per step against
        # 36 ms on the GPU, and 0.6 ms once vesin's cell list is available.
        struct = {
            "pos": torch.as_tensor(self.coords_bohr * BOHR_TO_ANGSTROM,
                                   dtype=self.dtype, device=self.device),
            "atomic_numbers": torch.as_tensor(self.atomic_numbers,
                                              dtype=torch.long,
                                              device=self.device),
            "cell": (torch.as_tensor(self.cell_bohr * BOHR_TO_ANGSTROM,
                                     dtype=self.dtype, device=self.device)
                     if self.cell_bohr is not None else None),
            "pbc": (torch.ones(3, dtype=torch.bool, device=self.device)
                    if self.cell_bohr is not None else None),
            "total_charge": self.total_charge,
        }
        graph = structure_to_graph(struct, self.cutoff, device=self.device)
        self._sync()
        t_graph = time.perf_counter()

        out = self.model(graph)
        self._sync()
        t_model = time.perf_counter()

        self.energy = float(out["energy"].sum().detach()) / HARTREE_TO_EV
        if "forces" in out:
            self.forces = (out["forces"].detach().cpu().double().numpy()
                           / (HARTREE_TO_EV / BOHR_TO_ANGSTROM))
        if "stress" in out and self.cell_bohr is not None:
            # MDI expects the pressure-sign convention (negated vs the
            # dE/d(strain)/V tensor the model returns), as in reference MDI
            # engines.
            self.stress = (-out["stress"][0].detach().cpu().double().numpy()
                           / (HARTREE_TO_EV / BOHR_TO_ANGSTROM**3))
        else:
            self.stress = None

        t_end = time.perf_counter()
        self._n_calc += 1
        self._t_graph += t_graph - t0
        self._t_model += t_model - t_graph
        self._t_extract += t_end - t_model
        self._t_total += t_end - t0
        self._n_edges = int(graph.edge_index.shape[1])

        if self._n_calc % 100 == 0:
            n = self._n_calc
            ms = 1000.0 / n
            # katom-step/s makes runs of different size comparable, which
            # ms/step on its own does not.
            rate = self.natoms * n / self._t_total / 1000.0
            logger.info(
                "step %d: graph=%.1f model=%.1f extract=%.1f total=%.1f ms/step"
                "  %.1f katom-step/s  %d edges",
                n, self._t_graph * ms, self._t_model * ms,
                self._t_extract * ms, self._t_total * ms, rate, self._n_edges,
            )

    def _sync(self) -> None:
        """Wait for queued device work to finish.

        GPU work is asynchronous, so a timestamp taken without this records
        when a kernel was *launched*, not when it finished -- which would put
        the model's cost into whatever ran next and make the breakdown
        meaningless. Costs a little per step and is worth it only because the
        numbers are then attributable.
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _ensure_results(self) -> None:
        """Re-evaluate the model if the geometry changed since the last call."""
        if self._needs_calculation:
            self.calculate()
            self._needs_calculation = False

    # MDI communication loop

    def run(self, mdi_options: str, mpi_comm=None) -> None:
        """Run the MDI engine loop until the driver sends ``EXIT``.

        Initializes the MDI library, registers the supported commands on the
        ``@DEFAULT`` node, accepts the driver connection and then services
        commands: system updates (``>NATOMS``, ``>ELEMENTS``, ``>CELL``,
        ``>COORDS``, ``>TOTCHARGE``) mark the results stale, result requests (``<ENERGY``,
        ``<FORCES``, ``<STRESS``) trigger a model evaluation when needed and
        send the values in MDI atomic units. ``<STRESS`` sends zeros for
        non-periodic systems. ``SCF`` forces an immediate evaluation.

        Parameters
        ----------
        mdi_options : str
            The MDI option string, e.g.
            ``"-role ENGINE -name xnn -method TCP -port 8021 -hostname
            localhost"``.
        mpi_comm : mpi4py.MPI.Comm, optional
            MPI communicator to hand to ``MDI_Init`` (required for
            ``-method MPI``). Defaults to ``None`` (TCP method).

        Raises
        ------
        ImportError
            If the ``pymdi`` package is not installed.
        RuntimeError
            If the driver sends a command this engine does not support, or
            requests forces/stress from a model that does not produce them.
        """
        if not _HAS_MDI:
            raise ImportError("the MDI library is required: pip install pymdi")

        if mpi_comm is not None:
            mdi.MDI_Init(mdi_options, mpi_comm)
        else:
            mdi.MDI_Init(mdi_options)

        mdi.MDI_Register_Node("@DEFAULT")
        for cmd in _COMMANDS:
            mdi.MDI_Register_Command("@DEFAULT", cmd)

        comm = mdi.MDI_Accept_Communicator()
        logger.info("MDI connection established")

        while True:
            command = mdi.MDI_Recv_Command(comm)
            logger.debug("MDI command: %s", command)

            if command == "EXIT":
                break

            elif command == ">NATOMS":
                self.natoms = mdi.MDI_Recv(1, mdi.MDI_INT, comm)

            elif command == ">ELEMENTS":
                elements = mdi.MDI_Recv(self.natoms, mdi.MDI_INT, comm)
                self.atomic_numbers = np.array(elements, dtype=np.int64)
                self._needs_calculation = True
                logger.info("received %d atoms, elements %s", self.natoms,
                            sorted(set(self.atomic_numbers.tolist())))

            elif command == ">CELL":
                cell = mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)
                self.cell_bohr = np.array(cell, dtype=np.float64).reshape(3, 3)
                self._needs_calculation = True

            elif command == ">COORDS":
                coords = mdi.MDI_Recv(3 * self.natoms, mdi.MDI_DOUBLE, comm)
                self.coords_bohr = np.array(coords, dtype=np.float64).reshape(
                    self.natoms, 3)
                self._needs_calculation = True

            elif command == ">TOTCHARGE":
                self.total_charge = float(mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm))
                self._needs_calculation = True
                logger.info("total charge set to %g e", self.total_charge)

            elif command == "<ENERGY":
                self._ensure_results()
                mdi.MDI_Send(self.energy, 1, mdi.MDI_DOUBLE, comm)

            elif command == "<FORCES":
                self._ensure_results()
                if self.forces is None:
                    raise RuntimeError(
                        "model produced no forces; wrap it in "
                        "ForceStressOutput before serving over MDI")
                mdi.MDI_Send(self.forces.flatten(), 3 * self.natoms,
                             mdi.MDI_DOUBLE, comm)

            elif command == "<STRESS":
                self._ensure_results()
                stress = (self.stress if self.stress is not None
                          else np.zeros(9))
                mdi.MDI_Send(stress.flatten(), 9, mdi.MDI_DOUBLE, comm)

            elif command == "SCF":
                self.calculate()
                self._needs_calculation = False

            else:
                raise RuntimeError(f"unhandled MDI command: {command}")

        n = max(self._n_calc, 1)
        ms = 1000.0 / n
        logger.info(
            "engine finished: %d calculations, avg %.1f ms/step "
            "(graph %.1f, model %.1f, extract %.1f)",
            self._n_calc, self._t_total * ms, self._t_graph * ms,
            self._t_model * ms, self._t_extract * ms,
        )


def main(argv=None) -> None:
    """Command-line entry point serving a checkpoint as an MDI engine.

    Invoked as ``xnn mdi ...`` or
    ``python -m xnn.common.deploy.mdi_engine ...``::

        xnn mdi --ckpt runs/exp/best.pt \\
            -mdi "-role ENGINE -name xnn -method TCP -port 8021 -hostname localhost"

    Serve a plain checkpoint with D4 added and a net charge of -1::

        xnn mdi --ckpt best.pt --dispersion "{name: d4, cutoff_pair: 12.0}" \\
            --total-charge -1 -mdi "..."

    For the MPI communication method, launch under ``mpirun`` alongside the
    driver (``mpi4py`` required)::

        mpirun -np 1 xnn mdi --ckpt best.pt -mdi "-role ENGINE -name xnn -method MPI" \\
            : -np 1 lmp -mdi "-role DRIVER -name LAMMPS -method MPI" -in input.dat

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector excluding the program name. When ``None`` (the
        default), ``sys.argv[1:]`` is used.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="xnn mdi",
        description="Serve a trained xnn checkpoint as an MDI engine.")
    p.add_argument("--ckpt", required=True,
                   help="trainer checkpoint (best.pt) holding model + config")
    p.add_argument("-mdi", "--mdi", dest="mdi_options", required=True,
                   help='MDI option string, e.g. "-role ENGINE -name xnn '
                        '-method TCP -port 8021 -hostname localhost"')
    p.add_argument("--device", default="cpu",
                   help='torch device, e.g. "cpu" or "cuda:0" (default: cpu)')
    p.add_argument("--dtype", choices=["float32", "float64"], default=None,
                   help="dtype to serve in (default: the checkpoint's own)")
    p.add_argument("--dispersion", default=None,
                   help="add D3/D4 dispersion to a checkpoint trained without "
                        "it: 'd4', 'd3', or a YAML mapping as in the config's "
                        "extra.dispersion, e.g. \"{name: d4, cutoff_pair: 12.0, "
                        "switch_width_pair: 2.0}\" (refused if the checkpoint "
                        "already carries dispersion)")
    p.add_argument("--total-charge", type=float, default=0.0,
                   help="net charge of the system in e (default 0); the driver "
                        "can change it with >TOTCHARGE")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    mpi_comm = None
    if "MPI" in args.mdi_options.split():
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD

    dtype = getattr(torch, args.dtype) if args.dtype else None
    dispersion = None
    if args.dispersion:
        import yaml
        dispersion = yaml.safe_load(args.dispersion)
    engine = MDIEngine.from_checkpoint(args.ckpt, device=args.device,
                                       dtype=dtype, dispersion=dispersion,
                                       total_charge=args.total_charge)
    engine.run(args.mdi_options, mpi_comm=mpi_comm)


if __name__ == "__main__":
    main()
