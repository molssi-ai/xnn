"""NDDO semiempirical Hamiltonians (MNDO family) as xnn models.

:class:`NDDO` is the shared base of PM3, PM6, PM6-ORG and PM7: a restricted
(closed shell) or unrestricted Hartree-Fock calculation with the neglect of
diatomic differential overlap,

    E = E_el + sum_{A<B} E_core(A, B),     E_el = 1/2 sum_mu,nu P_mu,nu (H_mu,nu + F_mu,nu),

over a minimal valence basis of Slater orbitals (``s``, ``p`` and, for some
elements, ``d``), with

* one-center core energies ``U`` and two-electron integrals (parameters for
  the main group, Slater-Condon integrals of internal exponents for ``d``
  shells and transition metals);
* two-center resonance integrals ``H_mu,la = (beta_mu + beta_la) / 2 S_mu,la``
  from the exact overlaps of the Slater orbitals;
* two-center two-electron and electron-core integrals from the multipole
  expansion of the charge distributions (see :mod:`.twocenter`);
* a method-specific core-core repulsion and, for some methods, post-SCF
  corrections (dispersion, hydrogen bonds, molecular-mechanics terms).

The model's ``energy`` is the heat of formation in eV (``Delta H_f / 23.06``),
the quantity semiempirical methods are parameterized for: it is the total
energy minus the energies of the isolated atoms in their reference
configurations, plus the experimental heats of formation of the gaseous atoms.
Forces are its negative gradient. The heat of formation in kcal/mol, the
total and electronic energies, the core-core energy, the atomic charges, the
dipole moment, the energy of the highest occupied orbital (``homo_energy``, eV)
and Koopmans' ionization potential (``ionization_potential``, its negative)
are returned as well.

Original and trainable
----------------------
Built with the published parameters (``PM6()`` reads the ``pm6`` values of
MOPAC 23.2.5, see :mod:`xnn.sqmnn.common.published`) the models reproduce the
reference program MOPAC; ``trainable=`` turns named parameter groups
(``"BETAS"``, ``"USS"``, ``"ZS"``, ``"ALPB"``, ``"FN11"``, ...) into
``torch.nn.Parameter`` tables refit by gradient descent like any xnn model.
:meth:`NDDO.parameter_set` exports the current values,
which :func:`~xnn.sqmnn.common.parameters.write_parameters` writes as a MOPAC
``EXTERNAL`` file.

Derivatives
-----------
The SCF energy is stationary in the density, so forces and parameter
gradients of the energy follow from the converged density held fixed. When
parameters are trained (training mode, parameters requiring gradients) the
converged density is in addition returned by :class:`~.scf.SCFDensity`, whose
backward pass solves the coupled-perturbed equations: the parameter gradients
of the forces, charges and dipoles are then exact as well.

With ``position_response`` the density is in addition a function of the
positions, differentiable to any order. Second derivatives of the energy taken
by autograd (Hessians, Hessian-vector products, see
:func:`xnn.common.models.hessian_vector_product`) are then exact, and so are
their parameter gradients.

Scope: molecules (no periodic images); closed-shell RHF, and UHF for open
shells (``spin_multiplicity`` > 1, or an odd electron count).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential

from ..common import elements as el
from ..common.parameters import ELEMENT_PARAMETERS, GAUSSIAN_PARAMETERS, PAIR_PARAMETERS, ParameterSet, read_parameters
from . import scf as scf_module
from .angular import L_OF, MOLECULAR, dipole_factors, pair_type
from .bond_order import BondOrder, with_values
from .bond_order import from_tensors as bond_order_from_tensors
from .bond_order import index as bond_order_index
from .bond_order import pair_values as bond_order_pair_values
from .core import (
    build_integrals,
    charges_and_dipole,
    electron_counts,
    energy_outputs,
    initial_guess,
    make_plan,
    table_key,
)
from .corrections import DISPERSION, HBOND, HH_SHORT, YNE
from .derived import element_quantities
from .graphed import GraphedFunction, replay, synchronizes
from .integrals import constants, device_constant, take0
from .scf import (
    CAPTURE_AFTER,
    Layout,
    SCFDensity,
    SCFGraphCache,
    _per_device,
    homo_orbitals,
    lowest,
    needs_restart,
    run_scf,
    search_guess,
    search_schedule,
)
from .twocenter import pair_tables

_ELEMENT_NAMES = ELEMENT_PARAMETERS + GAUSSIAN_PARAMETERS
# the energies of energy_outputs, in the order of the graphed outputs
_ENERGY_KEYS = ("node_energy", "energy", "heat_of_formation", "total_energy", "electronic_energy",
                "core_core_energy")
# the parameters of the core-core terms (see NDDO.core_parameters)
_CORE_PARAMETERS = ("XFAC", "ALPB", "ALP") + tuple(f"FN{n}{j}" for n in (1, 2, 3) for j in (1, 2, 3, 4)) + ("PAR",)
#: Whether the fast path pads a batch to the shapes of a bucket (ghost atoms without orbitals or core charge, see
#: :meth:`NDDO._bucket_plan`), so that its CUDA graphs replay for every batch of those shapes, per device type.
BUCKETS = {"cpu": False, "cuda": True}
#: Buckets (and their graphs) kept, the most recently used.
BUCKET_CAPACITY = 64
#: Paddings of batch compositions kept, the most recently used (see :meth:`NDDO._bucket_plan`).
PADDED_CAPACITY = 512
#: Bucket sizes per doubling of a size above 4 (a power of two): 4 means sizes 4, 5, 6, 7, 8 times a power of two
#: (at most a quarter of padding), 2 means 2, 3, 4 times a power of two (fewer buckets, at most half of padding).
BUCKET_SPACING = 2
_CORRECTION_CONSTANTS = {"YNE": YNE, "DISP": DISPERSION, "HBOND": HBOND, "HH_SHORT": HH_SHORT}
# the trainable values of the bond-order and resonance-overlap terms
_TERM_GROUPS = ("BOND_ORDER", "BOND_ORDER_WINDOW", "BOND_ORDER_CUTOFF", "RESONANCE_OVERLAP")


def _bond_order_args(bo: Optional[BondOrder]) -> list:
    """The trailing arguments of :class:`~.scf.SCFDensity` for a bond-order term (or none)."""
    return [bo, bo.coef, bo.lower, bo.width] if bo is not None else [None, None, None, None]


def _element_pair(key) -> Tuple[int, int]:
    """The atomic numbers of an element pair written ``"C-N"``."""
    parts = str(key).split("-")
    if len(parts) != 2:
        raise ValueError(f"element pair {key!r}: write it as two symbols joined by '-', such as 'C-N'")
    return el.atomic_number(parts[0]), el.atomic_number(parts[1])
_N_GLOBALS = 61


@dataclass
class _Plan:
    q: dict
    ea: Tensor
    n_orb: Tensor
    k: int
    pa: Tensor
    pb: Tensor
    layout: Layout
    has: Tensor
    valid: Tensor
    signature: tuple = ()
    W1: Optional[Tensor] = None
    # CUDA graphs of the build by kind and inputs with gradients: "eval" (frozen parameters baked in, of q_ref),
    # "train" (positions and the derived quantities as inputs), "response" (the same at detached positions);
    # False where rejected
    graphs: Dict[tuple, object] = field(default_factory=dict)
    q_ref: Optional[dict] = None
    uses: int = 0


@dataclass
class _Bucket:
    """The graphs of the batches of one bucket (sizes, orbital count, elements): the integral build by kind (see
    :class:`_Plan`), the derived quantities baked into the ``"eval"`` graph, and the evaluations seen."""
    key: tuple
    graphs: Dict[tuple, object] = field(default_factory=dict)
    q_ref: Optional[dict] = None
    uses: int = 0


@dataclass
class _Padded:
    """A batch padded to its bucket: the atoms and pairs that are real, the positions of the ghost atoms and the
    plan's index tensors (the inputs of the bucket's graphs)."""
    n_real: int
    real: Tensor
    pair_real: Tensor
    ghost_pos: Tensor
    z: Tensor
    batch: Tensor
    tensors: List[Tensor]
    # the tensors in one byte buffer (the input of the bucket's graphs) and its layout
    index: Optional[Tensor] = None
    spec: list = field(default_factory=list)


def _round_up(x: int) -> int:
    """The smallest bucket size that is at least ``x``: ``m 2^e`` with ``m`` from :data:`BUCKET_SPACING` to twice
    that, sizes up to 4 as they are."""
    if x <= 4:
        return max(x, 1)
    e = max(0, x.bit_length() - BUCKET_SPACING.bit_length())
    return -(-x // (1 << e)) << e


class _PlanCache(dict):
    """Plans per composition; copies and pickles of a model start with an empty cache (graphs belong to one
    process and device)."""

    def __getstate__(self):
        return {}

    def __reduce__(self):
        return (_PlanCache, ())

    def __deepcopy__(self, memo):
        return _PlanCache()


def _tensor_leaves(q: dict, prefix: tuple = ()):
    """The tensors of the derived quantities ``q`` (nested dictionaries included) with their key paths."""
    out = []
    for key, v in q.items():
        if isinstance(v, Tensor):
            out.append((prefix + (key,), v))
        elif isinstance(v, dict):
            out.extend(_tensor_leaves(v, prefix + (key,)))
    return out


def _composition(z: Tensor, batch: Tensor) -> tuple:
    """The atomic numbers and structure indices of a batch as tuples (one read from the device)."""
    both = torch.cat([z.long(), batch.long()]).tolist()
    n = int(z.shape[0])
    return tuple(both[:n]), tuple(both[n:])


def _hybrid_mask(kind: str) -> List[List[bool]]:
    """Whether a product of two orbitals in molecular order is an ``sp`` (or ``pd``) hybrid."""
    return [[pair_type(L_OF[a], L_OF[b]) == kind for b in MOLECULAR] for a in MOLECULAR]


def _host_quantities(uz: Sequence[int], tables: Dict[str, Tensor]) -> dict:
    """:func:`~.derived.element_quantities` of the elements ``uz`` from per-element parameter tables on a GPU,
    computed on the host and moved to the device in one copy each way (with their derivatives): the many
    operations on tensors of a few elements cost a kernel launch each on a GPU."""
    names = list(tables)
    device = tables[names[0]].device
    host = torch.stack([tables[name] for name in names]).cpu()[:, list(uz)]
    q = element_quantities(uz, dict(zip(names, host.unbind(0))))
    leaves = _tensor_leaves(q)
    packed = [(path, v) for path, v in leaves if v.dtype == host.dtype]
    flat = torch.cat([v.reshape(-1) for _, v in packed]).to(device)
    moved = [x.view(v.shape) for x, (_, v) in zip(flat.split([v.numel() for _, v in packed]), packed)]
    rest = [(path, v) for path, v in leaves if v.dtype != host.dtype]
    return _with_leaves(q, [path for path, _ in packed] + [path for path, _ in rest],
                        moved + [v.to(device) for _, v in rest])


def _with_leaves(q: dict, paths, values) -> dict:
    """``q`` with the tensors at ``paths`` replaced by ``values``."""
    out = dict(q)
    for path, v in zip(paths, values):
        d = out
        for key in path[:-1]:
            d[key] = dict(d[key])
            d = d[key]
        d[path[-1]] = v
    return out


def _q_signature(q: dict) -> tuple:
    """The structural (non-tensor) part of the derived quantities: equal signatures share a plan."""
    def walk(d):
        return tuple((key, walk(v) if isinstance(v, dict) else repr(v)) for key, v in d.items()
                     if not isinstance(v, Tensor))
    return walk(q)


class NDDO(InteratomicPotential):
    """Base class of the NDDO methods; see the module docstring.

    Parameters
    ----------
    parameters : str, Path or ParameterSet, optional
        The parameter set: a published name, a MOPAC ``EXTERNAL`` file or a
        :class:`~xnn.sqmnn.common.parameters.ParameterSet`; defaults to the
        method's published set.
    trainable : sequence of str, optional
        Parameter groups to refit: per-element names (``"USS"``, ``"BETAS"``,
        ``"ZS"``, ``"GSS"``, ``"FN11"``, ...), ``"ALPB"``/``"XFAC"`` (the
        diatomic core-core tables), ``"PAR"`` (the global parameters),
        the constants of the corrections (not part of a MOPAC parameter file):
        ``"YNE"`` (the acetylenic C-C correction of PM6, PM7 and PM6-ORG: its
        energy and the distances where its switch starts and ends), ``"DISP"``
        and ``"HBOND"`` (the scale and damping of the PM7 dispersion, the
        scales and O...O energy of the PM7 hydrogen bonds), ``"HH_SHORT"``
        (the height, width factor and centre of the short-range H-H term,
        which training switches on), the values of the
        terms below: ``"BOND_ORDER"`` (the energies), ``"BOND_ORDER_WINDOW"``
        (the windows, as lower edge and logarithm of the width),
        ``"BOND_ORDER_CUTOFF"`` (the cutoffs, as start and logarithm of the
        width) and ``"RESONANCE_OVERLAP"``, or ``"all"``.
        Empty (default) freezes the published method.
    yne_switch : str, optional
        The switch of the acetylenic C-C term: ``"polynomial"`` (default,
        the published method) has continuous first and second derivatives;
        ``"smooth"`` has continuous derivatives of every order, so that
        Hessians change smoothly with the C-C distance (for fits to
        curvatures); ``"off"`` removes the term.
    hbond_switch : str, optional
        The form of the PM7 hydrogen-bond term: ``"sharp"`` (default, the
        published method) has jumps where the search or a choice of
        reference atoms changes; ``"smooth"`` replaces each of them by a
        switch with continuous second derivatives and keeps every value
        outside the switches (see :func:`~.corrections.pm7_hbond`).
    hbond_torsion : str, optional
        The lone-pair and torsion factors of the PM7 hydrogen-bond term:
        ``"sharp"`` (default, the published method) has kinks where a
        hydrogen bond is planar or linear; ``"smooth"`` rounds them, so that
        forces and Hessians are continuous there (for fits to forces or
        curvatures; meant with ``hbond_switch="smooth"``).
    bond_order_terms : dict, optional
        An energy switched on by the bond order of atom pairs (see
        :mod:`~.bond_order`): per element pair (``"C-C"``, ``"N-N"``, ...),
        the energy in kcal/mol and the window of the Wiberg bond index where
        it switches on, ``[c, W_lo, W_hi]``, or with a cutoff of its own,
        ``[c, W_lo, W_hi, R_on, R_off]``. Default: none.
    bond_order_cutoff : sequence of float, optional
        The distances (Angstrom) where the bond-order terms start and finish
        switching off, for the pairs without a cutoff of their own; default
        ``(1.40, 1.65)``.
    resonance_overlap : dict, optional
        Factors ``kappa`` of the overlap in the resonance integrals of the
        p orbitals, ``beta (1 + kappa S)``, per element pair: one value for
        the pi overlaps, or ``[kappa_pi, kappa_sigma]``. Default: none.
    correction_constants : dict, optional
        Values of the constants of corrections by group (``"YNE"``,
        ``"DISP"``, ``"HBOND"``, as many values as the published ones), also
        the start of a trainable group; ``{"YNE": [12.0, 1.21, 1.45]}``, for
        example, widens the switch of the acetylenic term. Default: the
        published values. ``"HH_SHORT"`` adds the short-range H-H term
        ``A exp(-b (R - R_c)^2) S(R)`` (``[A, b, R_c]`` in kcal/mol,
        1/Angstrom^2 and Angstrom; ``S`` falls from 1 at 1.1 to 0 at 1.4
        Angstrom), which no published method has; its defaults
        (:data:`~.corrections.HH_SHORT`) give H2 its experimental bond length,
        heat of formation and frequency with the PM7 parameters.
    unrestricted : bool, optional
        Force spin-unrestricted SCF even for closed shells (default: only
        open shells are unrestricted).
    response : str, optional
        ``"auto"`` (default): exact density response in training mode when
        parameters require gradients; ``"always"`` / ``"never"``.
    position_response : bool, optional
        Make the SCF density a function of the positions as well, so that
        second derivatives of the energy (Hessians) and their parameter
        gradients are exact, as are the position derivatives of the HOMO
        energy, ionization potential, charges and dipole; default ``False``
        (forces need only the density itself). :func:`xnn.common.models.hessian` and
        :func:`~xnn.common.models.hessian_vector_product` switch it on for
        their own evaluations.
    max_iter, tol_density, tol_energy : optional
        SCF iteration limit and convergence thresholds (density, eV).
    corrections : bool, optional
        Include the method's post-SCF and molecular-mechanics corrections
        (default ``True``; switching them off reproduces MOPAC's ``NOMM`` and
        the bare Hamiltonian).
    reuse_density : bool, optional
        Start the SCF from the previous call's converged density when the
        batch has the same atoms, charges and spins and no atom moved by more
        than :attr:`warm_start_distance` (a step of a geometry optimization or
        of molecular dynamics); default ``True``. Other batches start cold.
    scf_starts : sequence of (float, int), optional
        SCF runs of a cold start, as (initial Fermi temperature in eV,
        annealing iterations); the lowest-energy solution of each structure
        is kept. The default (plain aufbau and two annealing schedules)
        finds the ground state of open shells and of closed shells where a
        single run can settle in an excited aufbau state (cumulenes, ketenes,
        strained rings).
    restart_gap : float, optional
        The runs after the first of ``scf_starts`` take place when the batch
        is unrestricted, a structure's first run did not converge, or its
        first solution has a HOMO-LUMO gap below this (eV); default 6.5
        (excited closed-shell solutions have gaps of 3-6 eV).
        ``float("inf")`` runs them always.
    search_starts : int, optional
        Further SCF runs of a cold start from perturbed initial densities,
        alternately random orbital occupations (the two spins apart,
        annealed) and random density noise (plain); deterministic. The
        lowest solution is kept. They find lower solutions where every
        other start settles in the same excited one (open shells with
        degenerate pi orbitals, some transition states); default 0. About
        6 suffice; a cold start then takes 4-6 times as long.
    use_fast : bool or str, optional
        The fast path (see :meth:`set_use_fast`): on a GPU, a batch composition evaluated before replays CUDA
        graphs. ``"auto"`` (default) and ``True`` take it, ``False`` never.
    """

    #: Name of the published parameter set of the method.
    default_parameters = ""
    #: Method label (the MOPAC keyword).
    method = ""
    #: Blend the two-center Coulomb integrals into point charges below 7 Angstrom
    #: (PM7, PM6-ORG; see :func:`~.core.feather`).
    feather = False
    #: PM7's balance of the two-center integrals of transition metals (see :func:`~.core.pm7_balance`).
    balance_integrals = False
    #: The corrections of the method (see :func:`~.corrections.structure_corrections`) and the peptide
    #: torsion barrier of PM3 (kcal/mol).
    correction_kind = 0
    peptide_barrier = 0.0
    #: The constants of the corrections that ``trainable`` may name (see :meth:`correction_constants`).
    correction_groups = ()
    #: The core-core form (see :mod:`~.core`): 0 that of PM6, PM7 and PM6-ORG, 1 that of PM3.
    core_form = 0
    #: Largest atomic displacement (Angstrom) since the previous call for which
    #: the SCF restarts from that call's density. A density carried to an
    #: unrelated geometry (another sample of a training set) can converge to an
    #: excited SCF solution.
    warm_start_distance = 0.3

    def __init__(self, parameters: Union[str, ParameterSet, None] = None,
                 trainable: Sequence[str] = (), unrestricted: bool = False,
                 response: str = "auto", max_iter: int = 300, tol_density: float = 1e-9,
                 tol_energy: float = 1e-10, corrections: bool = True,
                 reuse_density: bool = True,
                 scf_starts: Sequence = ((0.0, 0), (1.0, 80), (2.0, 25)), restart_gap: float = 6.5,
                 search_starts: int = 0, use_fast="auto", position_response: bool = False,
                 yne_switch: str = "polynomial", hbond_switch: str = "sharp", hbond_torsion: str = "sharp",
                 correction_constants: Optional[Dict[str, Sequence[float]]] = None,
                 bond_order_terms: Optional[Dict[str, Sequence[float]]] = None,
                 bond_order_cutoff: Sequence[float] = (1.40, 1.65),
                 resonance_overlap: Optional[Dict[str, Union[float, Sequence[float]]]] = None):
        super().__init__()
        pset = (parameters if isinstance(parameters, ParameterSet)
                else read_parameters(parameters or self.default_parameters))
        self.parameter_source = pset.name
        # the model forms all atom pairs itself; the data pipeline only needs a
        # (positive) neighbor-list radius
        self.cutoff = 0.1
        self.unrestricted = bool(unrestricted)
        if response not in ("auto", "always", "never"):
            raise ValueError("response must be 'auto', 'always' or 'never'")
        self.response = response
        self.position_response = bool(position_response)
        if yne_switch not in ("polynomial", "smooth", "off"):
            raise ValueError("yne_switch must be 'polynomial', 'smooth' or 'off'")
        self.yne_switch = yne_switch
        if hbond_switch not in ("sharp", "smooth"):
            raise ValueError("hbond_switch must be 'sharp' or 'smooth'")
        self.hbond_switch = hbond_switch
        if hbond_torsion not in ("sharp", "smooth"):
            raise ValueError("hbond_torsion must be 'sharp' or 'smooth'")
        self.hbond_torsion = hbond_torsion
        self.max_iter, self.tol_density, self.tol_energy = int(max_iter), float(tol_density), float(tol_energy)
        self.use_corrections = bool(corrections)
        self.last_scf = None
        self.reuse_density = bool(reuse_density)
        self.scf_starts = tuple((float(kt), int(n)) for kt, n in scf_starts)
        self.restart_gap = float(restart_gap)
        self.search_starts = int(search_starts)
        self.set_use_fast(use_fast)
        # the SCF and response graphs of the buckets (two entries each)
        self._scf_graphs = SCFGraphCache(capacity=2 * BUCKET_CAPACITY)
        self._plans = _PlanCache()
        self._padded = _PlanCache()
        self._bucket_elements: tuple = ()
        self._scf_key = None
        self._scf_pos = None
        self._q_cache = {}
        # the pairs of the bond-order term per batch composition (see _bond_order_pairs)
        self._bo_pairs = _PlanCache()

        nz = el.MAX_Z + 1
        table = torch.zeros(len(_ELEMENT_NAMES), nz, dtype=torch.float64)
        for z, values in pset.elements.items():
            if z < nz:
                for name, v in values.items():
                    table[_ELEMENT_NAMES.index(name), z] = v
        pairs = torch.zeros(len(PAIR_PARAMETERS), nz, nz, dtype=torch.float64)
        for (z1, z2), values in pset.pairs.items():
            if z1 < nz and z2 < nz:
                for name, v in values.items():
                    i = PAIR_PARAMETERS.index(name)
                    pairs[i, z1, z2] = pairs[i, z2, z1] = v
        glob = torch.zeros(_N_GLOBALS, dtype=torch.float64)
        for n_, v in pset.globals.items():
            if n_ < _N_GLOBALS:
                glob[n_] = v
        self.supported = tuple(z for z in pset.atomic_numbers if z < nz)

        term_groups = self._set_terms(bond_order_terms, bond_order_cutoff, resonance_overlap)
        groups = set(trainable or ())
        # the short-range H-H term is no part of a published method: on when its constants are given or trained
        self.hh_short = "HH_SHORT" in (correction_constants or {}) or "HH_SHORT" in groups
        if "all" in groups:
            groups = (set(_ELEMENT_NAMES) | set(PAIR_PARAMETERS) | {"PAR"} | set(self.correction_groups)
                      | set(term_groups))
            if yne_switch == "off":
                groups.discard("YNE")
            if not self.hh_short:
                groups.discard("HH_SHORT")
        unknown = (groups - set(_ELEMENT_NAMES) - set(PAIR_PARAMETERS) - {"PAR"} - set(self.correction_groups)
                   - set(term_groups))
        if unknown:
            raise ValueError(f"unknown trainable groups {sorted(unknown)} (the groups of the bond-order and "
                             "resonance terms need the terms)")
        if yne_switch == "off" and "YNE" in groups:
            raise ValueError("the YNE group cannot be trained with yne_switch='off'")
        self.trainable_groups = tuple(sorted(groups))
        self.register_buffer("element_table", table)
        self.register_buffer("pair_table", pairs)
        self.register_buffer("global_table", glob)
        # fixed in MOPAC, so not part of a parameter set or of the state of an untrained model
        given = dict(correction_constants or {})
        unknown = set(given) - set(self.correction_groups)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no correction constants {sorted(unknown)}")
        for name, values in _CORRECTION_CONSTANTS.items():
            values = [float(v) for v in given.get(name, values)]
            if len(values) != len(_CORRECTION_CONSTANTS[name]):
                raise ValueError(f"correction constants {name} take {len(_CORRECTION_CONSTANTS[name])} values")
            if name == "YNE" and yne_switch == "off":
                values[0] = 0.0
            self.register_buffer(f"constants_{name}", torch.tensor(values, dtype=torch.float64), persistent=False)
        self.params = nn.ParameterDict()
        for name in self.trainable_groups:
            if name in _CORRECTION_CONSTANTS:
                self.params[name] = nn.Parameter(getattr(self, f"constants_{name}").clone())
            elif name in _TERM_GROUPS:
                self.params[name] = nn.Parameter(getattr(self, f"terms_{name}").clone())
            elif name in _ELEMENT_NAMES:
                self.params[name] = nn.Parameter(table[_ELEMENT_NAMES.index(name)].clone())
            elif name in PAIR_PARAMETERS:
                self.params[name] = nn.Parameter(pairs[PAIR_PARAMETERS.index(name)].clone())
            else:
                self.params["PAR"] = nn.Parameter(glob.clone())

    # parameter access

    def element_parameter(self, name: str, given: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Per-element table ``(MAX_Z + 1,)`` of a parameter (trainable copy if any, or the one ``given``)."""
        if given and name in given:
            return given[name]
        if name in self.params:
            return self.params[name]
        return self.element_table[_ELEMENT_NAMES.index(name)]

    def pair_parameter(self, name: str, given: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Symmetric diatomic table ``(MAX_Z + 1, MAX_Z + 1)`` of ``ALPB`` or ``XFAC`` (from the trainable copy if
        any, or the one ``given``)."""
        if (given and name in given) or name in self.params:
            m = given[name] if given and name in given else self.params[name]
            return 0.5 * (m + m.transpose(0, 1))
        return self.pair_table[PAIR_PARAMETERS.index(name)]

    def global_parameters(self, given: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """The ``PARn`` values ``(61,)`` (index ``n``; the trainable copy if any, or the one ``given``)."""
        if given and "PAR" in given:
            return given["PAR"]
        return self.params["PAR"] if "PAR" in self.params else self.global_table

    def correction_constants(self, name: str) -> Tensor:
        """The constants of a correction, the trainable copy if any: ``"YNE"`` (see
        :data:`~.corrections.YNE`), ``"DISP"`` (:data:`~.corrections.DISPERSION`), ``"HBOND"``
        (:data:`~.corrections.HBOND`) or ``"HH_SHORT"`` (:data:`~.corrections.HH_SHORT`)."""
        return self.params[name] if name in self.params else getattr(self, f"constants_{name}")

    def _set_terms(self, bond_order_terms, bond_order_cutoff, resonance_overlap) -> Tuple[str, ...]:
        """Tables of the bond-order and resonance-overlap terms (non-persistent buffers: the values are options);
        the groups they offer for training."""
        nz = el.MAX_Z + 1
        self.bond_order_terms = {str(k): [float(x) for x in v] for k, v in (bond_order_terms or {}).items()}
        self.bond_order_cutoff = [float(x) for x in bond_order_cutoff]
        self.resonance_overlap = {str(k): ([float(v)] if isinstance(v, (int, float)) else [float(x) for x in v])
                                  for k, v in (resonance_overlap or {}).items()}
        if len(self.bond_order_cutoff) != 2 or not self.bond_order_cutoff[1] > self.bond_order_cutoff[0]:
            raise ValueError("bond_order_cutoff takes two distances, the second larger")
        slots = torch.full((nz, nz), -1, dtype=torch.long)
        energy, window, cut = [], [], []
        sizes = {len(v) for v in self.bond_order_terms.values()}
        if sizes - {3, 5} or len(sizes) > 1:
            raise ValueError("bond_order_terms take [c, W_lo, W_hi] or [c, W_lo, W_hi, R_on, R_off] per element "
                             "pair, the same form for every pair")
        for i, (key, v) in enumerate(self.bond_order_terms.items()):
            za, zb = _element_pair(key)
            if slots[za, zb] >= 0:
                raise ValueError(f"bond_order_terms: element pair {key} given twice")
            slots[za, zb] = slots[zb, za] = i
            if not v[2] > v[1]:
                raise ValueError(f"bond_order_terms {key}: W_hi must be larger than W_lo")
            energy.append(v[0])
            window.append([v[1], math.log(v[2] - v[1])])
            if len(v) == 5:
                if not v[4] > v[3]:
                    raise ValueError(f"bond_order_terms {key}: R_off must be larger than R_on")
                cut.append([v[3], math.log(v[4] - v[3])])
        if not cut:
            r_on, r_off = self.bond_order_cutoff
            cut = [[r_on, math.log(r_off - r_on)]]
        self.register_buffer("bond_order_slots", slots, persistent=False)
        self.register_buffer("terms_BOND_ORDER", torch.tensor(energy, dtype=torch.float64), persistent=False)
        self.register_buffer("terms_BOND_ORDER_WINDOW", torch.tensor(window, dtype=torch.float64).reshape(-1, 2),
                             persistent=False)
        self.register_buffer("terms_BOND_ORDER_CUTOFF", torch.tensor(cut, dtype=torch.float64), persistent=False)
        slots = torch.full((nz, nz), -1, dtype=torch.long)
        width = max([len(v) for v in self.resonance_overlap.values()], default=1)
        if width > 2:
            raise ValueError("resonance_overlap takes kappa_pi or [kappa_pi, kappa_sigma] per element pair")
        kappa = []
        for i, (key, v) in enumerate(self.resonance_overlap.items()):
            za, zb = _element_pair(key)
            if slots[za, zb] >= 0:
                raise ValueError(f"resonance_overlap: element pair {key} given twice")
            slots[za, zb] = slots[zb, za] = i
            kappa.append(v + [0.0] * (width - len(v)))
        self.register_buffer("resonance_slots", slots, persistent=False)
        self.register_buffer("terms_RESONANCE_OVERLAP", torch.tensor(kappa, dtype=torch.float64).reshape(-1, width),
                             persistent=False)
        groups = ()
        if self.bond_order_terms:
            groups += _TERM_GROUPS[:3]
        if self.resonance_overlap:
            groups += _TERM_GROUPS[3:]
        return groups

    def _bond_order_pairs(self, z: Tensor, pa: Tensor, pb: Tensor, layout: Layout, n_struct: int, pos: Tensor,
                          padded: Optional[_Padded], key: tuple):
        """The bond-order term of a batch at the detached positions (for the SCF; ``None`` without the term) and
        the function giving it at other positions. Its pairs are those with an entry, or on a bucket's graphs
        every pair of the bucket (the shapes are fixed; the others get zero energy); their index is kept per
        batch composition ``key``."""
        if not self.bond_order_terms:
            return None, lambda positions: None
        key = key + (str(pos.device),)
        entry = self._bo_pairs.pop(key, None)
        if entry is None:
            if padded is None:
                keep = (self.bond_order_slots[take0(z, pa), take0(z, pb)] >= 0).nonzero().squeeze(-1)
                pa, pb = pa[keep], pb[keep]
            entry = None
            if pa.shape[0] > 0:
                zero = pos.new_zeros(pa.shape[0], dtype=torch.float64)
                entry = (pa, pb, take0(z, pa), take0(z, pb),
                         bond_order_index(layout.atom_struct, layout.ao, pa, pb, n_struct, layout.n_basis, zero,
                                          zero, zero))
        self._bo_pairs[key] = entry                                # most recent last
        while len(self._bo_pairs) > PADDED_CAPACITY:
            self._bo_pairs.pop(next(iter(self._bo_pairs)))
        if entry is None:
            return None, lambda positions: None
        pa, pb, za, zb, skeleton = entry

        def values(positions: Tensor):
            r = (take0(positions, pb) - take0(positions, pa)).norm(dim=-1)
            coef, lower, width = self.bond_order_values(za, zb, r)
            if padded is not None:
                coef = coef * padded.pair_real
            return coef, lower, width

        here = with_values(skeleton, *values(pos))
        scf_term = with_values(skeleton, here.coef.detach(), here.lower.detach(), here.width.detach())
        return scf_term, lambda positions: here if positions is pos else with_values(skeleton, *values(positions))

    def term_values(self, name: str) -> Tensor:
        """The values of a group of the bond-order or resonance-overlap terms (the trainable copy if any):
        ``"BOND_ORDER"`` ``(n,)`` the energies (kcal/mol) of the element pairs in the order given,
        ``"BOND_ORDER_WINDOW"`` ``(n, 2)`` the window's lower edge and the logarithm of its width,
        ``"BOND_ORDER_CUTOFF"`` ``(n, 2)`` or ``(1, 2)`` the start of the cutoff and the logarithm of its width
        (Angstrom), ``"RESONANCE_OVERLAP"`` ``(n, 1)`` or ``(n, 2)`` the factors ``kappa``."""
        return self.params[name] if name in self.params else getattr(self, f"terms_{name}")

    def bond_order_values(self, za: Tensor, zb: Tensor, r: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """The values of :class:`~.bond_order.BondOrder` (eV, window edge and width) of pairs of atomic numbers
        ``za``, ``zb`` at distances ``r`` (Angstrom); zero energy for pairs without an entry."""
        return bond_order_pair_values(self.bond_order_slots, self.term_values("BOND_ORDER"),
                                      self.term_values("BOND_ORDER_WINDOW"), self.term_values("BOND_ORDER_CUTOFF"),
                                      za, zb, r)

    def resonance_factors(self, za: Tensor, zb: Tensor) -> Tensor:
        """The factors ``kappa`` ``(P, 1)`` or ``(P, 2)`` of :func:`~.core.resonance_overlap` of pairs of atomic
        numbers ``za``, ``zb`` (zero for pairs without an entry)."""
        slot = self.resonance_slots[za, zb]
        values = self.term_values("RESONANCE_OVERLAP")
        rows = values.index_select(0, slot.clamp(min=0))
        return torch.where((slot >= 0)[:, None], rows, torch.zeros_like(rows))

    def parameter_set(self) -> ParameterSet:
        """The current parameters as a :class:`ParameterSet` (for MOPAC export)."""
        out = ParameterSet(name=f"{self.parameter_source}")
        with torch.no_grad():
            for z in self.supported:
                values = {}
                for name in _ELEMENT_NAMES:
                    v = float(self.element_parameter(name)[z])
                    if v != 0.0:
                        values[name] = v
                out.elements[z] = values
            for name in PAIR_PARAMETERS:
                m = self.pair_parameter(name)
                for z1 in self.supported:
                    for z2 in self.supported:
                        if z2 <= z1 and float(m[z1, z2]) != 0.0:
                            out.pairs.setdefault((z1, z2), {})[name] = float(m[z1, z2])
            for n_, v in enumerate(self.global_parameters().tolist()):
                if v != 0.0:
                    out.globals[n_] = v
        return out

    # method hooks

    def core_core(self, za: Tensor, zb: Tensor, r: Tensor, gab: Tensor, tore_a: Tensor, tore_b: Tensor,
                  params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        """Core-core repulsion of each pair (eV); ``r`` in Angstrom; ``params`` the tables of
        :meth:`core_parameters` (the model's where not given)."""
        raise NotImplementedError

    def export_core(self) -> nn.Module:
        """The scriptable core of :func:`~xnn.common.deploy.export_torchscript_potential` (see
        :class:`~.export.NDDOExport`): this model's tables on its device, one molecule per call."""
        from .export import NDDOExport
        return NDDOExport(self)

    def core_parameters(self, given: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """The parameter tables of the core-core terms (trainable copies where trained); ``given``: trained
        parameters (by name, as in :attr:`params`) in place of the model's, such as the inputs of a graph."""
        out = {name: self.pair_parameter(name, given) for name in ("XFAC", "ALPB")}
        out["ALP"] = self.element_parameter("ALP", given)
        for n in (1, 2, 3):
            for j in (1, 2, 3, 4):
                out[f"FN{n}{j}"] = self.element_parameter(f"FN{n}{j}", given)
        out["PAR"] = self.global_parameters(given)
        return out

    def corrections(self, data: AtomicGraph, pos: Tensor) -> Tensor:
        """Per-atom post-SCF and molecular-mechanics energy corrections (kcal/mol)."""
        return pos.new_zeros(pos.shape[0])

    # helpers

    def _gaussian(self, z: Tensor, j: int):
        return (self.element_parameter(f"FN1{j}")[z], self.element_parameter(f"FN2{j}")[z],
                self.element_parameter(f"FN3{j}")[z])

    def _element_quantities(self, uz: Sequence[int]):
        # only the NDDO parameters and tables enter (not those of submodules such as a dispersion term)
        own = list(self.params.values()) + list(self.buffers(recurse=False))
        frozen = not (torch.is_grad_enabled() and any(p.requires_grad for p in self.params.values()))
        key = (tuple(uz), str(self.element_table.device), tuple(t._version for t in own))
        if frozen and key in self._q_cache:
            return self._q_cache[key]
        if self.element_table.is_cuda and self._graph_cache() is not None:
            q = _host_quantities(uz, {name: self.element_parameter(name) for name in _ELEMENT_NAMES})
        else:
            idx = torch.as_tensor(list(uz), device=self.element_table.device)
            q = element_quantities(uz, {name: self.element_parameter(name)[idx] for name in _ELEMENT_NAMES})
        if frozen:
            self._q_cache = {key: q}
        return q

    def set_use_fast(self, use_fast) -> "NDDO":
        """Choose the fast path: ``True``, ``False`` or ``"auto"``.

        On a GPU a batch is padded to a bucket of sizes (see :meth:`_bucket_plan` and :data:`BUCKETS`); a bucket
        evaluated more than :data:`~.scf.CAPTURE_AFTER` times replays CUDA graphs of its SCF iterations, of its
        density response and of its integral build and core-core terms with their derivatives to second order (from
        the parameters when they are trained), for every batch of it. The graphs compute what the launched kernels
        compute (see :mod:`~xnn.sqmnn.models.graphed`). ``"auto"`` takes the fast path wherever it runs.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    def _graph_cache(self):
        return self._scf_graphs if self.use_fast in (True, "auto") else None

    def _plan(self, z: Tensor, batch: Tensor, n_struct: int, uz: Sequence[int], q,
              composition: Optional[tuple] = None):
        """Element indices, pairs and the basis layout of a batch; on the fast path kept per composition (the
        graphs replay on its tensors)."""
        device = z.device
        key = None
        if self._graph_cache() is not None and z.is_cuda:
            key = (composition or _composition(z, batch)) + (n_struct,)
            plan = self._plans.get(key)
            if plan is not None and plan.signature == _q_signature(q):
                self._plans[key] = self._plans.pop(key)           # most recent last
                return plan
        lookup = torch.full((el.MAX_Z + 1,), -1, dtype=torch.long, device=device)
        lookup[torch.as_tensor(uz, device=device)] = torch.arange(len(uz), device=device)
        ea, n_orb, k, pa, pb, layout, has, valid = make_plan(
            z, batch, n_struct, lookup, torch.as_tensor(q["n_orb"], device=device),
            _per_device(scf_module.GROUP_MIN_PAIRS, device))
        plan = _Plan(q=q, ea=ea, n_orb=n_orb, k=k, pa=pa, pb=pb, layout=layout, has=has, valid=valid,
                     signature=_q_signature(q))
        if key is not None:
            self._plans[key] = plan
            while len(self._plans) > 8:
                self._plans.pop(next(iter(self._plans)))
        return plan

    def _bucket_plan(self, z: Tensor, batch: Tensor, n_struct: int, uz: Sequence[int], q, pos: Tensor,
                     composition: Optional[tuple] = None):
        """A batch padded to the shapes of a bucket, and the bucket (its graphs replay for every batch of them).

        The atoms (at least two more than the batch has), their pairs and the basis functions of a structure are
        those of a kept bucket that holds the batch, else rounded up (:meth:`_bucket_sizes`). The extra atoms
        are ghosts of the structure last in the batch: the first element of ``uz`` without orbitals or core
        charge, far from the molecules and from each other, paired only with each other; every pair of a
        structure stays in place, and the masks of the plan clear the ghosts' terms. The layout keeps one padded
        group of all pairs (:attr:`~.scf.Layout.fixed`), so that its index tensors (:func:`~.scf.layout_tensors`)
        have the bucket's shapes whatever the batch. The padding of a
        composition is kept (:meth:`_pad`); the ghosts' positions follow the batch's. ``composition``: the atomic
        numbers and structure indices as tuples, where already read."""
        key = (composition or _composition(z, batch)) + (n_struct, tuple(uz), tuple(q["n_orb"]))
        entry = self._padded.pop(key, None)
        if entry is None:
            entry = self._pad(z, batch, n_struct, uz, q)
        self._padded[key] = entry                                  # most recent last
        while len(self._padded) > PADDED_CAPACITY:
            self._padded.pop(next(iter(self._padded)))
        plan, padded, bucket_key = entry
        # the ghosts 1000 A beyond the batch's atoms in each coordinate, 50 A apart (their offsets kept in the
        # padding), placed without a host synchronization
        far = pos.detach().abs().amax() + 1.0e3 if padded.n_real > 0 else pos.new_full((), 1.0e3)
        padded = replace(padded, ghost_pos=padded.ghost_pos + far)
        bucket = self._plans.get(bucket_key)
        if bucket is None:
            bucket = _Bucket(bucket_key)
        self._plans[bucket_key] = self._plans.pop(bucket_key, bucket)          # most recent last
        while len(self._plans) > BUCKET_CAPACITY:
            self._plans.pop(next(iter(self._plans)))
        return plan, padded, bucket

    def _bucket_sizes(self, n_struct: int, k: int, uz: Sequence[int], need: Tuple[int, int, int]):
        """The atoms, pairs and basis functions of the bucket of a batch that needs ``need`` of them: those of
        the smallest kept bucket (of its structures, orbitals per atom and elements) that holds it with at most
        twice each size, else ``need`` rounded up by :func:`_round_up` (so that a few buckets serve batches of
        many sizes)."""
        best = None
        for key in self._plans:
            if key[0] != "bucket" or key[1] != n_struct or key[5] != k or key[6] != tuple(uz):
                continue
            sizes = key[2:5]
            if all(m <= size <= 2 * m for m, size in zip(need, sizes)):
                if best is None or (sizes[1], sizes[2], sizes[0]) < (best[1], best[2], best[0]):
                    best = sizes
        return best if best is not None else tuple(_round_up(m) for m in need)

    def _pad(self, z: Tensor, batch: Tensor, n_struct: int, uz: Sequence[int], q):
        """The padding of a composition for :meth:`_bucket_plan`: its plan, the :class:`_Padded` tensors with the
        ghosts' offsets from the first ghost as ``ghost_pos``, and the key of its bucket."""
        device = z.device
        n = int(z.shape[0])
        lookup = torch.full((el.MAX_Z + 1,), -1, dtype=torch.long, device=device)
        lookup[torch.as_tensor(list(uz), device=device)] = torch.arange(len(uz), device=device)
        n_orb_el = torch.as_tensor(q["n_orb"], device=device)
        k = int(n_orb_el.max())
        counts = torch.zeros(n_struct, dtype=torch.long, device=device).index_add(0, batch, n_orb_el[lookup[z]])
        same = (batch.unsqueeze(0) == batch.unsqueeze(1)).triu(diagonal=1).nonzero()
        n_pairs = int(same.shape[0])
        n_atoms, n_pairs_b, n_basis = self._bucket_sizes(n_struct, k, uz, (n + 2, n_pairs + 1, int(counts.max())))
        n_ghost = n_atoms - n
        z_w = torch.cat([z, torch.full((n_ghost,), int(uz[0]), dtype=z.dtype, device=device)])
        batch_w = torch.cat([batch, torch.full((n_ghost,), n_struct - 1, dtype=batch.dtype, device=device)])
        ghost_pairs = torch.triu_indices(n_ghost, n_ghost, 1, device=device) + n
        reps = -(-(n_pairs_b - n_pairs) // int(ghost_pairs.shape[1]))
        filler = ghost_pairs.repeat(1, reps)[:, :n_pairs_b - n_pairs]
        pa = torch.cat([same[:, 0], filler[0]])
        pb = torch.cat([same[:, 1], filler[1]])
        real = torch.arange(n_atoms, device=device) < n
        ea = lookup[z_w]
        n_orb = n_orb_el[ea] * real
        cum = torch.cumsum(n_orb, 0) - n_orb
        start = cum - (torch.cumsum(counts, 0) - counts)[batch_w]
        orb = torch.arange(k, device=device)
        has = orb.unsqueeze(0) < n_orb.unsqueeze(1)
        ao = torch.where(has, start.unsqueeze(1) + orb.unsqueeze(0), torch.full_like(has, n_basis, dtype=torch.long))
        valid = torch.arange(n_basis, device=device).unsqueeze(0) < counts.unsqueeze(1)
        layout = Layout(n_struct, n_basis, k, batch_w, ao, pa, pb, valid)
        layout.fixed = True
        offsets = torch.zeros(n_ghost, 3, dtype=torch.float64, device=device)
        offsets[:, 0] = 50.0 * torch.arange(n_ghost, dtype=torch.float64, device=device)
        pair_real = (torch.arange(n_pairs_b, device=device) < n_pairs).to(torch.float64)
        index, spec = scf_module.pack_index([ea, z_w, n_orb, has, valid] + scf_module.layout_tensors(layout))
        plan = _Plan(q=q, ea=ea, n_orb=n_orb, k=k, pa=pa, pb=pb, layout=layout, has=has, valid=valid)
        padded = _Padded(n_real=n, real=real, pair_real=pair_real, ghost_pos=offsets, z=z_w, batch=batch_w,
                         tensors=scf_module.unpack_index(index, spec), index=index, spec=spec)
        return plan, padded, ("bucket", n_struct, n_atoms, n_pairs_b, n_basis, k, tuple(uz))

    def _pair_tables(self, q: dict, layout: Layout, k: int, like: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """The tables of :func:`~.twocenter.pair_kernel` for the pair groups of a batch (cached)."""
        general = k == 9
        keys = {(ka, kb, general and max(ka, kb) == 9) for ka, kb, _, _, _ in layout.pair_groups()}
        keys.add((k, k, general))
        return {table_key(ka, kb, g): pair_tables(q, ka, kb, g, like.device, like.dtype) for ka, kb, g in keys}

    def _build_graph(self, holder, kind: str, fn, q: dict, inputs) -> Optional[GraphedFunction]:
        """The CUDA graphs of the integral build ``fn(*inputs)`` of a composition (``holder`` a :class:`_Plan`) or a
        bucket (a :class:`_Bucket`, whose inputs end with the plan's index tensors) evaluated before: ``"eval"``
        with frozen parameters (the derived quantities ``q`` baked in), ``"train"`` and ``"response"`` with the
        derived quantities as inputs, whose derivatives are replayed to second order; captured after
        :data:`~.scf.CAPTURE_AFTER` launched evaluations."""
        # one graph per kind and per set of inputs that carry gradients
        kind = (kind, tuple(x.requires_grad for x in inputs))
        graph = holder.graphs.get(kind)
        if kind[0] == "eval" and graph and holder.q_ref is not q:
            del holder.graphs[kind]                 # baked derived quantities of other parameters
            graph = None
        if graph is not None:
            return graph or None
        if not inputs[0].is_cuda or self._graph_cache() is None or holder.uses <= CAPTURE_AFTER:
            return None

        def work():
            with torch.enable_grad():
                xs = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
                outs = [o for o in fn(*xs) if o.requires_grad]
                need = [x for x in xs if x.requires_grad]
                if outs and need:
                    torch.autograd.grad(outs, need, [torch.ones_like(o) for o in outs], allow_unused=True)
        if synchronizes(work):
            holder.graphs[kind] = False
            return None
        graph = GraphedFunction(fn, inputs)
        holder.graphs[kind] = graph
        if kind[0] == "eval":
            holder.q_ref = q
        return graph

    def _use_response(self) -> bool:
        if self.response == "never":
            return False
        needs = torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())
        if self.response == "always":
            return torch.is_grad_enabled()
        return self.training and needs

    @classmethod
    def from_config(cls, cfg) -> "NDDO":
        """Build from a :class:`~xnn.common.config.schema.ModelConfig`.

        Options in ``cfg.extra``: ``parameters`` (set name or file),
        ``trainable`` (list of groups), ``unrestricted``, ``response``,
        ``max_iter``, ``tol_density``, ``tol_energy``, ``corrections``,
        ``reuse_density``, ``scf_starts``, ``restart_gap``, ``search_starts``, ``use_fast``,
        ``position_response``, ``yne_switch``, ``hbond_switch``, ``hbond_torsion``, ``correction_constants``,
        ``bond_order_terms``,
        ``bond_order_cutoff``, ``resonance_overlap``.
        """

        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        keys = ("parameters", "trainable", "unrestricted", "response", "max_iter",
                "tol_density", "tol_energy", "corrections", "reuse_density", "scf_starts",
                "restart_gap", "search_starts", "use_fast", "position_response", "yne_switch", "hbond_switch",
                "hbond_torsion", "correction_constants", "bond_order_terms", "bond_order_cutoff", "resonance_overlap")
        return cls(**{k: extra[k] for k in keys if k in extra})

    # the calculation

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Heats of formation, energies, charges and dipoles of a batch of molecules."""
        dtype = torch.float64
        pos = data.pos.to(dtype)
        z = data.atomic_numbers.long()
        batch = data.batch
        n_struct = data.num_graphs
        device = pos.device
        # the atomic numbers and structure indices on the host, in one read
        composition = _composition(z, batch)
        missing = sorted(set(composition[0]) - set(self.supported))
        if missing:
            raise ValueError(f"{type(self).__name__} has no parameters for Z = {missing}")

        uz = sorted(set(composition[0]))
        # on the fast path of a GPU the batch is padded to a bucket, whose graphs replay for any of its batches
        bucketed = self._graph_cache() is not None and bool(_per_device(BUCKETS, device))
        padded = None
        if bucketed:
            if not set(uz) <= set(self._bucket_elements):
                self._bucket_elements = tuple(sorted(set(self._bucket_elements) | set(uz)))
            uz = list(self._bucket_elements)
            q = self._element_quantities(uz)
            plan, padded, holder = self._bucket_plan(z, batch, n_struct, uz, q, pos, composition)
            pos_w, z_w, batch_w = torch.cat([pos, padded.ghost_pos]), padded.z, padded.batch
            ptens = [padded.index]
        else:
            q = self._element_quantities(uz)
            plan = holder = self._plan(z, batch, n_struct, uz, q, composition)
            pos_w, z_w, batch_w, ptens = pos, z, batch, []
        ea, n_orb, k, pa, pb, layout, has, valid = (plan.ea, plan.n_orb, plan.k, plan.pa, plan.pb, plan.layout,
                                                     plan.has, plan.valid)
        tore = take0(q["tore"], ea)
        if padded is not None:
            tore = tore * padded.real

        occ_d = device_constant(("occ_d",), lambda: el.OCC_D, device)
        shells = device_constant(("shells", k), lambda: [0, 1, 1, 1, 2, 2, 2, 2, 2][:k], device)

        # the factors of the resonance-overlap term of every pair (an input of the integral build and its graphs)
        kappa = (self.resonance_factors(take0(z_w, pa), take0(z_w, pb)) if self.resonance_overlap and k >= 4
                 else None)
        kap_in = [kappa] if kappa is not None else []
        n_kap = len(kap_in)

        def build(positions: Tensor, q: dict, index=None, kap=None):
            # the plan's tensors, or those of the given buffer (the input of a bucket's graphs); the integrals,
            # and the layout, atomic numbers and element indices they used
            if index is None:
                lay, ea_, z_, n_orb_, has_, valid_ = layout, ea, z_w, n_orb, has, valid
            else:
                tensors = scf_module.unpack_index(index, padded.spec)
                ea_, z_, n_orb_, has_, valid_ = tensors[:5]
                lay = scf_module.layout_from_tensors(n_struct, layout.n_basis, k, list(tensors[5:]))
            qt = {key: v for key, v in q.items() if isinstance(v, Tensor)}
            out = build_integrals(qt, self._pair_tables(q, lay, k, positions), constants(device, dtype), lay,
                                  positions, ea_, z_, n_orb_, k, lay.pair_a, lay.pair_b, has_, valid_, self.feather,
                                  self.balance_integrals, occ_d, shells, kap)
            return out, lay, z_, ea_

        def integrals(positions: Tensor, q: dict, index=None, given=None, core: bool = True, kap=None):
            # the two-center integrals, core Hamiltonian, one-center integrals and, with ``core``, the pairs'
            # core-core energies (from the trained core parameters ``given``; ghost pairs are cleared later)
            (ints, H, W1, _, _), lay, z_, ea_ = build(positions, q, index, kap)
            outs = (ints["W"], ints["r"], ints["gab"], H, W1)
            if not core:
                return outs
            pa_, pb_ = lay.pair_a, lay.pair_b
            tore_ = take0(q["tore"], ea_)
            return outs + (self.core_core(take0(z_, pa_), take0(z_, pb_), ints["r"], ints["gab"], take0(tore_, pa_),
                                          take0(tore_, pb_), self.core_parameters(given)),)

        q_values = [v for _, v in _tensor_leaves(q)]
        paths = [path for path, _ in _tensor_leaves(q)]
        shapes = [v.shape for v in q_values]
        # a bucket's graphs take the derived quantities in one tensor (one copy per replay, one gradient)
        packed = padded is not None and all(v.dtype == dtype for v in q_values)
        q_in = [torch.cat([v.reshape(-1) for v in q_values])] if packed else q_values
        n_q = len(q_in)
        # the trained parameters of the core-core terms enter the graphs as inputs (their gradients)
        core_names = [name for name in _CORE_PARAMETERS if name in self.params]
        core_in = [self.params[name] for name in core_names]
        n_core = len(core_in)

        def unpacked(values):
            if not packed:
                return _with_leaves(q, paths, values)
            return _with_leaves(q, paths, [x.view(shape) for x, shape in
                                           zip(values[0].split([v.numel() for v in q_values]), shapes)])

        def eval_fn(positions, *rest):
            kap, index = rest[:n_kap], rest[n_kap:]
            return integrals(positions, q, index[0] if padded is not None else None, kap=kap[0] if n_kap else None)

        def train_fn(positions, *rest):
            values, given = rest[:n_q], rest[n_q:n_q + n_core]
            kap, index = rest[n_q + n_core:n_q + n_core + n_kap], rest[n_q + n_core + n_kap:]
            return integrals(positions, unpacked(values), index[0] if padded is not None else None,
                             dict(zip(core_names, given)), kap=kap[0] if n_kap else None)

        def response_fn(positions, *rest):
            values, kap, index = rest[:n_q], rest[n_q:n_q + n_kap], rest[n_q + n_kap:]
            return integrals(positions, unpacked(values), index[0] if padded is not None else None, core=False,
                             kap=kap[0] if n_kap else None)

        trained = any(v.requires_grad for v in q_values + core_in + kap_in)
        # the density as a function of the positions: the integrals of the energy enter the SCF density (their
        # derivatives to any order, so built eagerly)
        exact = self.position_response and torch.is_grad_enabled()
        if pos.is_cuda and self._graph_cache() is not None:
            holder.uses += 1
        if exact:
            graph = None
        else:
            graph = (self._build_graph(holder, "train", train_fn, q, [pos_w, *q_in, *core_in, *kap_in, *ptens])
                     if trained else self._build_graph(holder, "eval", eval_fn, q, [pos_w, *kap_in, *ptens]))
        if graph is not None and trained:
            W2, r_pair, gab, H, W1, e_core_pair = replay(graph, [pos_w, *q_in, *core_in, *kap_in, *ptens])
        elif graph is not None:
            W2, r_pair, gab, H, W1, e_core_pair = replay(graph, [pos_w, *kap_in, *ptens])
        else:
            W2, r_pair, gab, H, W1, e_core_pair = integrals(pos_w, q, kap=kappa)
        if graph is not None and graph.wants_second:
            graph.capture_second()
        if padded is not None:
            e_core_pair = e_core_pair * padded.pair_real

        # electrons and spin
        charge = (data.total_charge.to(dtype) if data.total_charge is not None
                  else torch.zeros(n_struct, dtype=dtype, device=device))
        n_e, n_alpha, n_beta, uhf = electron_counts(tore, batch_w, n_struct, charge, data.spin_multiplicity,
                                                    self.unrestricted)
        # initial guess: reference occupations spread over each shell
        occ_ref = device_constant(("occ_ref", tuple(uz)),
                                  lambda: [[el.OCC_S[v], el.OCC_P[v] / 3.0, el.OCC_D[v] / 5.0] for v in uz],
                                  device, dtype)
        guess = initial_guess(occ_ref, ea, shells, has, batch_w, n_struct, n_e, layout, pa.shape[0], k)
        bo_scf, bo_at = self._bond_order_pairs(z_w, pa, pb, layout, n_struct, pos_w, padded,
                                               composition + (n_struct, holder.key if padded is not None else None))

        spins = torch.stack([n_alpha, n_beta]).tolist()
        key = composition + (tuple(spins[0]), tuple(spins[1]), uhf, holder.key if padded is not None else None)
        # the SCF and response graphs of a bucket replay for every batch of it
        graph_key = holder.key if padded is not None else key
        warm = (self.reuse_density and self.last_scf is not None and self._scf_key == key
                and float((pos.detach() - self._scf_pos).norm(dim=-1).max()) <= self.warm_start_distance)
        if warm:
            last = self.last_scf.densities
            guess = last.sum(0) if uhf else last[0]
        if warm:
            result = run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, self.max_iter,
                             self.tol_density, self.tol_energy, smear_steps=0,
                             spin_guess=self.last_scf.densities if uhf else None,
                             graph_cache=self._graph_cache(), graph_key=graph_key, orbitals=self.last_scf.orbitals,
                             occupations=self.last_scf.occupations, bond_order=bo_scf)
        else:
            def start(kt, steps):
                return run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, self.max_iter, self.tol_density,
                               self.tol_energy, smearing=kt, smear_steps=steps, graph_cache=self._graph_cache(),
                               graph_key=graph_key, bond_order=bo_scf)
            results = [start(*self.scf_starts[0])]
            # the other runs where the first may have settled in an excited aufbau state
            if len(self.scf_starts) > 1 and needs_restart(results[0], uhf, self.restart_gap):
                results += [start(kt, steps) for kt, steps in self.scf_starts[1:]]
            for i in range(self.search_starts):
                kicked, spin = search_guess(guess, layout, n_alpha, n_beta, uhf, i)
                kt, steps = search_schedule(i)
                results.append(run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, kicked, self.max_iter,
                                       self.tol_density, self.tol_energy, smearing=kt, smear_steps=steps,
                                       spin_guess=spin, graph_cache=self._graph_cache(), graph_key=graph_key,
                                       bond_order=bo_scf))
            result = lowest(results)
        self.last_scf, self._scf_key, self._scf_pos = result, key, pos.detach().clone()
        # the bond-order term at the positions (its energy, and the density's response where exact)
        bo = bo_at(pos_w)
        if exact:
            P = SCFDensity.apply(H, W1, W2, result, layout, uhf, 1e-10, None, (), *_bond_order_args(bo))
        elif self._use_response():
            pos_fixed = pos_w.detach()
            response = (self._build_graph(holder, "response", response_fn, q, [pos_fixed, *q_in, *kap_in, *ptens])
                        if trained else None)
            if response is not None:
                W_t, _, _, H_t, W1_t = replay(response, [pos_fixed, *q_in, *kap_in, *ptens])
            else:
                W_t, _, _, H_t, W1_t = integrals(pos_fixed, q, core=False, kap=kappa)
            P = SCFDensity.apply(H_t, W1_t, W_t, result, layout, uhf, 1e-10, self._graph_cache(), graph_key,
                                 *_bond_order_args(bo_at(pos_fixed)))
        else:
            P = result.densities
        n_pad = int(z_w.shape[0]) - int(z.shape[0])
        # per-atom values from tables of the elements (on the device once), none for the ghosts
        atom_heat = take0(device_constant(("atom_heat",), lambda: el.ATOM_HEAT, device, dtype), z_w)
        mass = take0(device_constant(("atomic_mass",), lambda: el.ATOMIC_MASS, device, dtype), z_w)
        if padded is not None:
            atom_heat, mass = atom_heat * padded.real, mass * padded.real
        corr = (self.corrections(data, pos) if self.use_corrections
                else pos.new_zeros(pos.shape[0]))
        eisol = take0(q["eisol"], ea)
        if padded is not None:
            corr = torch.cat([corr, corr.new_zeros(n_pad)])
            eisol = eisol * padded.real
        # energies, charges and dipole moments (e Angstrom) from the densities: graphed with their derivatives
        # where the integral build is (per composition or bucket, and spin form)
        sp_mask = device_constant(("sp_mask",), lambda: _hybrid_mask("sp"), device, dtype)
        pd_mask = device_constant(("pd_mask",), lambda: _hybrid_mask("pd"), device, dtype)
        dfac = device_constant(("dipole_factors",), dipole_factors, device, dtype)

        bo_in = bo.tensors() if bo is not None else []
        n_bo = len(bo_in)

        def outputs_fn(H_, W1_, W2_, P_, e_core_, eisol_, heat_, corr_, tore_, d_sp, d_pd, mass_, charge_, pos_,
                       homo_v, homo_s, *rest):
            t, index = rest[:n_bo], rest[n_bo:]
            bo_ = bond_order_from_tensors(list(t), n_struct, layout.n_basis, k, bo.masked) if n_bo else None
            if index:
                tensors = scf_module.unpack_index(index[0], padded.spec)
                lay, n_orb_ = scf_module.layout_from_tensors(n_struct, layout.n_basis, k, list(tensors[5:])), tensors[2]
                pa_, pb_, batch_ = lay.pair_a, lay.pair_b, lay.atom_struct
            else:
                lay, n_orb_, pa_, pb_, batch_ = layout, n_orb, pa, pb, batch_w
            res = energy_outputs(lay, H_, W1_, W2_, P_, uhf, e_core_, pa_, pb_, batch_, n_struct, eisol_, heat_,
                                 corr_, bo_, homo_v, homo_s)
            charges, dipole = charges_and_dipole(lay, P_, uhf, tore_, k, dfac, sp_mask, pd_mask, d_sp, d_pd, n_orb_,
                                                 mass_, batch_, n_struct, charge_, pos_)
            return tuple(res[key] for key in _ENERGY_KEYS) + (charges, dipole, res["homo_energy"])

        # the highest occupied orbital of each structure (its energy, differentiable through the density)
        homo_vec, homo_spin = homo_orbitals(result)
        inputs = [H, W1, W2, P, e_core_pair, eisol, atom_heat, corr, tore, take0(q["dd"], ea),
                  take0(q["ddp"][:, 5], ea), mass, charge, pos_w, homo_vec, homo_spin, *bo_in, *ptens]
        post = (self._build_graph(holder, "outputs_uhf" if uhf else "outputs", outputs_fn, q, inputs)
                if graph is not None else None)
        values = replay(post, inputs) if post is not None else outputs_fn(*inputs)
        if post is not None and post.wants_second:
            post.capture_second()
        out = dict(zip(_ENERGY_KEYS + ("charges", "dipole", "homo_energy"), values))
        # Koopmans' ionization potential
        out["ionization_potential"] = -out["homo_energy"]
        if padded is not None:
            n_real = padded.n_real
            out["node_energy"], out["charges"] = out["node_energy"][:n_real], out["charges"][:n_real]
        out["scf_converged"] = result.converged
        return out
