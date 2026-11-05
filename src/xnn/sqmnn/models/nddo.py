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
total and electronic energies, the core-core energy, the atomic charges and the
dipole moment are returned as well.

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

Scope: molecules (no periodic images); closed-shell RHF, and UHF for open
shells (``spin_multiplicity`` > 1, or an odd electron count).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential

from ..common import elements as el
from ..common.parameters import ELEMENT_PARAMETERS, GAUSSIAN_PARAMETERS, PAIR_PARAMETERS, ParameterSet, read_parameters
from . import scf as scf_module
from .angular import L_OF, MOLECULAR, dipole_factors, pair_type
from .core import (
    build_integrals,
    charges_and_dipole,
    electron_counts,
    energy_outputs,
    initial_guess,
    make_plan,
    table_key,
)
from .derived import element_quantities
from .graphed import GraphedFunction, replay, synchronizes
from .integrals import constants, device_constant, take0
from .scf import (
    CAPTURE_AFTER,
    Layout,
    SCFDensity,
    SCFGraphCache,
    _per_device,
    frontier_gap,
    lowest,
    run_scf,
)
from .twocenter import pair_tables

_ELEMENT_NAMES = ELEMENT_PARAMETERS + GAUSSIAN_PARAMETERS
_N_GLOBALS = 61
#: Atom pairs per block of the two-center integral evaluation.
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
        diatomic core-core tables), ``"PAR"`` (the global parameters), or
        ``"all"``. Empty (default) freezes the published method.
    unrestricted : bool, optional
        Force spin-unrestricted SCF even for closed shells (default: only
        open shells are unrestricted).
    response : str, optional
        ``"auto"`` (default): exact density response in training mode when
        parameters require gradients; ``"always"`` / ``"never"``.
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
        finds the ground state of open shells and small-gap systems where a
        single run can settle in an excited aufbau state.
    restart_gap : float, optional
        The runs after the first of ``scf_starts`` take place when the batch
        is unrestricted or a structure's first solution has a HOMO-LUMO gap
        below this (eV); default 3.0. ``float("inf")`` runs them always.
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
                 scf_starts: Sequence = ((0.0, 0), (1.0, 80), (2.0, 25)), restart_gap: float = 3.0,
                 use_fast="auto"):
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
        self.max_iter, self.tol_density, self.tol_energy = int(max_iter), float(tol_density), float(tol_energy)
        self.use_corrections = bool(corrections)
        self.last_scf = None
        self.reuse_density = bool(reuse_density)
        self.scf_starts = tuple((float(kt), int(n)) for kt, n in scf_starts)
        self.restart_gap = float(restart_gap)
        self.set_use_fast(use_fast)
        self._scf_graphs = SCFGraphCache()
        self._plans = _PlanCache()
        self._scf_key = None
        self._scf_pos = None
        self._q_cache = {}

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

        groups = set(trainable or ())
        if "all" in groups:
            groups = set(_ELEMENT_NAMES) | set(PAIR_PARAMETERS) | {"PAR"}
        unknown = groups - set(_ELEMENT_NAMES) - set(PAIR_PARAMETERS) - {"PAR"}
        if unknown:
            raise ValueError(f"unknown trainable groups {sorted(unknown)}")
        self.trainable_groups = tuple(sorted(groups))
        self.register_buffer("element_table", table)
        self.register_buffer("pair_table", pairs)
        self.register_buffer("global_table", glob)
        self.params = nn.ParameterDict()
        for name in self.trainable_groups:
            if name in _ELEMENT_NAMES:
                self.params[name] = nn.Parameter(table[_ELEMENT_NAMES.index(name)].clone())
            elif name in PAIR_PARAMETERS:
                self.params[name] = nn.Parameter(pairs[PAIR_PARAMETERS.index(name)].clone())
            else:
                self.params["PAR"] = nn.Parameter(glob.clone())

    # parameter access

    def element_parameter(self, name: str) -> Tensor:
        """Per-element table ``(MAX_Z + 1,)`` of a parameter (trainable copy if any)."""
        if name in self.params:
            return self.params[name]
        return self.element_table[_ELEMENT_NAMES.index(name)]

    def pair_parameter(self, name: str) -> Tensor:
        """Symmetric diatomic table ``(MAX_Z + 1, MAX_Z + 1)`` of ``ALPB`` or ``XFAC``."""
        if name in self.params:
            m = self.params[name]
            return 0.5 * (m + m.transpose(0, 1))
        return self.pair_table[PAIR_PARAMETERS.index(name)]

    def global_parameters(self) -> Tensor:
        """The ``PARn`` values ``(61,)`` (index ``n``)."""
        return self.params["PAR"] if "PAR" in self.params else self.global_table

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

    def core_core(self, za: Tensor, zb: Tensor, r: Tensor, gab: Tensor, tore_a: Tensor,
                  tore_b: Tensor) -> Tensor:
        """Core-core repulsion of each pair (eV); ``r`` in Angstrom."""
        raise NotImplementedError

    def export_core(self) -> nn.Module:
        """The scriptable core of :func:`~xnn.common.deploy.export_torchscript_potential` (see
        :class:`~.export.NDDOExport`): this model's tables on its device, one molecule per call."""
        from .export import NDDOExport
        return NDDOExport(self)

    def core_parameters(self) -> Dict[str, Tensor]:
        """The parameter tables of the core-core terms (trainable copies where trained)."""
        out = {name: self.pair_parameter(name) for name in ("XFAC", "ALPB")}
        out["ALP"] = self.element_parameter("ALP")
        for n in (1, 2, 3):
            for j in (1, 2, 3, 4):
                out[f"FN{n}{j}"] = self.element_parameter(f"FN{n}{j}")
        out["PAR"] = self.global_parameters()
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
        idx = torch.as_tensor(list(uz), device=self.element_table.device)
        p = {name: self.element_parameter(name)[idx] for name in _ELEMENT_NAMES}
        q = element_quantities(uz, p)
        if frozen:
            self._q_cache = {key: q}
        return q

    def set_use_fast(self, use_fast) -> "NDDO":
        """Choose the fast path: ``True``, ``False`` or ``"auto"``.

        On a GPU, a batch composition (atoms, charges, spins) evaluated more than :data:`~.scf.CAPTURE_AFTER`
        times replays CUDA graphs of its SCF iterations and of its integral build with its derivatives to second
        order (from the parameters when they are trained), which compute what the launched kernels compute (see
        :mod:`~xnn.sqmnn.models.graphed`). ``"auto"`` takes the fast path wherever it runs.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    def _graph_cache(self):
        return self._scf_graphs if self.use_fast in (True, "auto") else None

    def _plan(self, z: Tensor, batch: Tensor, n_struct: int, uz: Sequence[int], q):
        """Element indices, pairs and the basis layout of a batch; on the fast path kept per composition (the
        graphs replay on its tensors)."""
        device = z.device
        key = None
        if self._graph_cache() is not None and z.is_cuda:
            key = (tuple(z.tolist()), tuple(batch.tolist()), n_struct)
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

    def _pair_tables(self, q: dict, layout: Layout, k: int, like: Tensor) -> Dict[str, Dict[str, Tensor]]:
        """The tables of :func:`~.twocenter.pair_kernel` for the pair groups of a batch (cached)."""
        general = k == 9
        keys = {(ka, kb, general and max(ka, kb) == 9) for ka, kb, _, _, _ in layout.pair_groups()}
        keys.add((k, k, general))
        return {table_key(ka, kb, g): pair_tables(q, ka, kb, g, like.device, like.dtype) for ka, kb, g in keys}

    def _build_graph(self, plan: "_Plan", kind: str, build, q: dict, inputs) -> Optional[GraphedFunction]:
        """The CUDA graphs of the integral build of a composition evaluated before: ``"eval"`` with frozen
        parameters (positions in; the derived quantities ``q`` baked in), ``"train"`` and ``"response"`` with the
        derived quantities as inputs (``inputs`` = positions and the tensors of ``q``), whose derivatives are
        replayed to second order; captured after :data:`~.scf.CAPTURE_AFTER` launched evaluations."""
        # one graph per kind and per set of inputs that carry gradients
        kind = (kind, tuple(x.requires_grad for x in inputs))
        graph = plan.graphs.get(kind)
        if kind[0] == "eval" and graph and plan.q_ref is not q:
            del plan.graphs[kind]                   # baked derived quantities of other parameters
            graph = None
        if graph is not None:
            return graph or None
        if not inputs[0].is_cuda or self._graph_cache() is None or plan.uses <= CAPTURE_AFTER:
            return None
        paths = [path for path, _ in _tensor_leaves(q)]
        if kind[0] == "eval":
            def fn(positions):
                ints, H, _, _, _ = build(positions, q)
                return ints["W"], ints["r"], ints["gab"], H
        else:
            def fn(positions, *values):
                ints, H, W1, _, _ = build(positions, _with_leaves(q, paths, values))
                return ints["W"], ints["r"], ints["gab"], H, W1

        def work():
            with torch.enable_grad():
                xs = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
                outs = [o for o in fn(*xs) if o.requires_grad]
                need = [x for x in xs if x.requires_grad]
                if outs and need:
                    torch.autograd.grad(outs, need, [torch.ones_like(o) for o in outs], allow_unused=True)
        if synchronizes(work):
            plan.graphs[kind] = False
            return None
        graph = GraphedFunction(fn, inputs)
        plan.graphs[kind] = graph
        if kind[0] == "eval":
            plan.q_ref = q
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
        ``reuse_density``, ``scf_starts``, ``restart_gap``, ``use_fast``.
        """

        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        keys = ("parameters", "trainable", "unrestricted", "response", "max_iter",
                "tol_density", "tol_energy", "corrections", "reuse_density", "scf_starts",
                "restart_gap", "use_fast")
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
        missing = sorted(set(z.unique().tolist()) - set(self.supported))
        if missing:
            raise ValueError(f"{type(self).__name__} has no parameters for Z = {missing}")

        uz = z.unique().tolist()
        q = self._element_quantities(uz)
        plan = self._plan(z, batch, n_struct, uz, q)
        ea, n_orb, k, pa, pb, layout, has, valid = (plan.ea, plan.n_orb, plan.k, plan.pa, plan.pb, plan.layout,
                                                     plan.has, plan.valid)
        tore = take0(q["tore"], ea)

        occ_d = device_constant(("occ_d",), lambda: el.OCC_D, device)
        shells = device_constant(("shells", k), lambda: [0, 1, 1, 1, 2, 2, 2, 2, 2][:k], device)

        def build(positions: Tensor, q: dict):
            qt = {key: v for key, v in q.items() if isinstance(v, Tensor)}
            return build_integrals(qt, self._pair_tables(q, layout, k, positions), constants(device, dtype), layout,
                                   positions, ea, z, n_orb, k, pa, pb, has, valid, self.feather,
                                   self.balance_integrals, occ_d, shells)

        q_values = [v for _, v in _tensor_leaves(q)]
        trained = any(v.requires_grad for v in q_values)
        if pos.is_cuda and self._graph_cache() is not None:
            plan.uses += 1
        graph = (self._build_graph(plan, "train", build, q, [pos, *q_values]) if trained
                 else self._build_graph(plan, "eval", build, q, [pos]))
        if graph is not None and trained:
            W2, r_pair, gab, H, W1 = replay(graph, [pos, *q_values])
        elif graph is not None:
            W2, r_pair, gab, H = replay(graph, [pos])
            W1 = plan.W1
        else:
            ints, H, W1, _, _ = build(pos, q)
            W2, r_pair, gab = ints["W"], ints["r"], ints["gab"]
            plan.W1 = W1.detach()
        if graph is not None and graph.wants_second:
            graph.capture_second()
        e_core_pair = self.core_core(z[pa], z[pb], r_pair, gab, take0(tore, pa), take0(tore, pb))

        # electrons and spin
        charge = (data.total_charge.to(dtype) if data.total_charge is not None
                  else torch.zeros(n_struct, dtype=dtype, device=device))
        n_e, n_alpha, n_beta, uhf = electron_counts(tore, batch, n_struct, charge, data.spin_multiplicity,
                                                    self.unrestricted)
        # initial guess: reference occupations spread over each shell
        occ_ref = torch.as_tensor([[el.OCC_S[v], el.OCC_P[v] / 3.0, el.OCC_D[v] / 5.0] for v in uz],
                                  dtype=dtype, device=device)
        guess = initial_guess(occ_ref, ea, shells, has, batch, n_struct, n_e, layout, pa.shape[0], k)

        key = (tuple(z.tolist()), tuple(batch.tolist()), tuple(n_alpha.tolist()),
               tuple(n_beta.tolist()), uhf)
        warm = (self.reuse_density and self.last_scf is not None and self._scf_key == key
                and float((pos.detach() - self._scf_pos).norm(dim=-1).max()) <= self.warm_start_distance)
        if warm:
            last = self.last_scf.densities
            guess = last.sum(0) if uhf else last[0]
        if warm:
            result = run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, self.max_iter,
                             self.tol_density, self.tol_energy, smear_steps=0,
                             spin_guess=self.last_scf.densities if uhf else None,
                             graph_cache=self._graph_cache(), graph_key=key, orbitals=self.last_scf.orbitals,
                             occupations=self.last_scf.occupations)
        else:
            def start(kt, steps):
                return run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, self.max_iter, self.tol_density,
                               self.tol_energy, smearing=kt, smear_steps=steps, graph_cache=self._graph_cache(),
                               graph_key=key)
            results = [start(*self.scf_starts[0])]
            # the other runs where the first may have settled in an excited aufbau state
            if len(self.scf_starts) > 1 and (uhf or bool((frontier_gap(results[0]) < self.restart_gap).any())):
                results += [start(kt, steps) for kt, steps in self.scf_starts[1:]]
            result = lowest(results)
        self.last_scf, self._scf_key, self._scf_pos = result, key, pos.detach().clone()
        if self._use_response():
            pos_fixed = pos.detach()
            response = self._build_graph(plan, "response", build, q, [pos_fixed, *q_values]) if trained else None
            if response is not None:
                W_t, _, _, H_t, W1_t = replay(response, [pos_fixed, *q_values])
            else:
                ints_t, H_t, W1_t, _, _ = build(pos_fixed, q)
                W_t = ints_t["W"]
            P = SCFDensity.apply(H_t, W1_t, W_t, result, layout, uhf, 1e-10, self._graph_cache(), key)
        else:
            P = result.densities
        atom_heat = torch.as_tensor([el.ATOM_HEAT[v] for v in z.tolist()], dtype=dtype, device=device)
        corr = (self.corrections(data, pos) if self.use_corrections
                else pos.new_zeros(pos.shape[0]))
        out = energy_outputs(layout, H, W1, W2, P, uhf, e_core_pair, pa, pb, batch, n_struct, take0(q["eisol"], ea),
                             atom_heat, corr)

        # charges and dipole moment (e Angstrom)
        types = [[pair_type(L_OF[a], L_OF[b]) for b in MOLECULAR] for a in MOLECULAR]
        sp_mask = torch.as_tensor([[t == "sp" for t in row] for row in types], dtype=dtype, device=device)
        pd_mask = torch.as_tensor([[t == "pd" for t in row] for row in types], dtype=dtype, device=device)
        mass = torch.as_tensor([el.ATOMIC_MASS[v] for v in z.tolist()], dtype=dtype, device=device)
        out["charges"], out["dipole"] = charges_and_dipole(
            layout, P, uhf, tore, k, torch.as_tensor(dipole_factors(), dtype=dtype, device=device), sp_mask,
            pd_mask, q["dd"][ea], q["ddp"][ea, 5], n_orb, mass, batch, n_struct, charge, pos)
        out["scf_converged"] = result.converged
        return out
