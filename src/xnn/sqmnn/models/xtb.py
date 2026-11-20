"""GFN1-xTB and GFN2-xTB extended tight-binding methods as xnn models.

The self-consistent-charge tight-binding energy of Grimme et al. (GFN1-xTB:
J. Chem. Theory Comput. 13, 1989 (2017); GFN2-xTB: Bannwarth et al., J. Chem.
Theory Comput. 15, 1652 (2019); review: Bannwarth et al., WIREs Comput. Mol.
Sci. 11, e1493 (2021)) in a minimal valence basis of contracted Gaussians,

    E = E_rep + E_disp + sum_mu,nu P_mu,nu H0_mu,nu + E_gamma + E_Gamma (+ E_AES + E_AXC) - T S_el,

* ``H0`` is the extended Hueckel matrix with environment-dependent levels,
  ``H0_mu,nu = K_ll' (H_mu + H_nu) / 2 S_mu,nu X(EN) Y(zeta) Pi(R)``;
* ``E_gamma`` is the isotropic second-order energy of the Mulliken shell
  charges with a short-range damped Coulomb kernel, ``E_Gamma`` its on-site
  third-order correction;
* GFN2-xTB adds the damped interactions of cumulative atomic dipole and
  quadrupole moments (``E_AES``), their on-site exchange-correlation terms
  (``E_AXC``) and a D4 dispersion energy evaluated with the Mulliken charges
  inside the self-consistent cycle; GFN1-xTB uses D3 dispersion and a halogen
  bond correction;
* the occupations follow a Fermi distribution at the electronic temperature
  (300 K by default) and ``-T S_el`` is the electronic entropy term.

The model's ``energy`` is this total energy in eV; forces are its negative
gradient. The parameters are those of the xtb program's parameter files (see
:mod:`xnn.sqmnn.common.xtb_published`).

Scope: molecules (no periodic images), spin-restricted occupations with
separate Fermi levels for the alpha and beta electrons.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.d4 import charge_scaling_terms

from ..common.xtb_published import XTBParameters, parse_parameters, published_parameters, write_parameters
from ..common.xtb_tables import GFN0_SRB, GFN0_SRB_FACTOR, GFN2_AES_RADII, GFN2_RADII, STO_NG, n_gaussians
from .gaussian import N_SPHERICAL, primitive_norm, self_overlap, shell_pair_integrals

#: Largest atomic number with parameters.
MAX_Z = 86
#: Element parameters that can be trained (names of the parameter file).
TRAINABLE = ("LEV", "KCN", "POLY", "GAM", "LPAR", "GAM3", "REPA", "REPB", "DPOL", "QPOL")
#: Bohr radius (Angstrom) and Hartree energy (eV) of the parameterization.
BOHR = 0.5291772109044924
AUTOEV = 27.21138505
#: Boltzmann constant, hartree per kelvin.
KB_AU = 3.166808578545117e-06


def _reference_tables():
    from xnn.common.models.d4 import _load_reference
    ref = _load_reference()
    return ref["pauling_en"], ref["covalent_radius_aa"]


class _Basis:
    """Shells, atomic orbitals and primitive tables of a batch of molecules."""

    def __init__(self, model: "GFN", z: List[int], batch: List[int], n_struct: int, dtype, device):
        shell_atom, shell_l, shell_key, alphas, coefs = [], [], [], [], []
        ao_shell, ao_atom, ao_index, ao_struct = [], [], [], []
        counts = [0] * n_struct
        primitives = {za: model.shell_primitives(za) for za in set(z)}
        for a, (za, b) in enumerate(zip(z, batch)):
            for k, (alpha, coef, l) in enumerate(primitives[za]):
                s = len(shell_atom)
                shell_atom.append(a)
                shell_l.append(l)
                shell_key.append((za, k))
                alphas.append(alpha)
                coefs.append(coef)
                for _ in range(N_SPHERICAL[l]):
                    ao_shell.append(s)
                    ao_atom.append(a)
                    ao_struct.append(b)
                    ao_index.append(counts[b])
                    counts[b] += 1
        kmax = max(len(x) for x in alphas)
        self.kmax = kmax
        self.alpha = torch.tensor([x + [1.0] * (kmax - len(x)) for x in alphas], dtype=dtype, device=device)
        self.coef = torch.tensor([x + [0.0] * (kmax - len(x)) for x in coefs], dtype=dtype, device=device)
        self.shell_atom = torch.tensor(shell_atom, dtype=torch.long, device=device)
        self.shell_l = shell_l
        self.shell_key = shell_key
        self.ao_shell = torch.tensor(ao_shell, dtype=torch.long, device=device)
        self.ao_atom = torch.tensor(ao_atom, dtype=torch.long, device=device)
        self.ao_struct = torch.tensor(ao_struct, dtype=torch.long, device=device)
        self.ao_index = torch.tensor(ao_index, dtype=torch.long, device=device)
        self.n_ao = counts
        self.nb = max(counts)
        self.shell_struct = [batch[a] for a in shell_atom]
        first = [0] * len(shell_atom)
        for i in range(len(ao_shell) - 1, -1, -1):
            first[ao_shell[i]] = i
        self.shell_first = first
        self.shell_l_t = torch.tensor(shell_l, dtype=torch.long, device=device)
        self.shell_z = torch.tensor([k[0] for k in shell_key], dtype=torch.long, device=device)
        self.shell_index = torch.tensor([k[1] for k in shell_key], dtype=torch.long, device=device)
        if "EXP" in model.trainable and torch.is_grad_enabled():
            self.alpha, self.coef = model._trained_primitives(self)
        # contraction coefficients times the norms of their primitives
        norms = torch.stack([primitive_norm(self.alpha, l) for l in range(3)])
        self.weight = self.coef * norms[self.shell_l_t, torch.arange(len(shell_l), device=device)]
        self.shell_k = torch.tensor([len(x) for x in alphas], dtype=torch.long, device=device)
        self.shell_struct_t = torch.tensor(self.shell_struct, dtype=torch.long, device=device)
        self.shell_first_t = torch.tensor(first, dtype=torch.long, device=device)
        # what the integral kernels read per shell: atom, first function, structure, exponents, weights
        self.kernel_tables = (self.shell_atom, self.ao_index[self.shell_first_t], self.shell_struct_t,
                              self.alpha.contiguous(), (self.weight * math.pi ** 0.75).contiguous())


class GFN(InteratomicPotential):
    """Base class of the GFN tight-binding methods; see the module docstring.

    Parameters
    ----------
    parameters : str or XTBParameters, optional
        ``"gfn0"``, ``"gfn1"`` or ``"gfn2"`` (the published sets), a parameter file (path or
        text) in the xtb format, or parsed parameters; defaults to the method's
        published set.
    electronic_temperature : float
        Fermi smearing temperature in kelvin (default 300).
    max_iter : int
        Iteration limit of the self-consistent cycle.
    accuracy : float
        Scale of the convergence threshold of the cycle, as in the xtb program (default 1.0, its
        tightest 0.0001): the cycle ends when the root-mean-square change of the shell charges and
        atomic moments of every structure is below ``2e-5 * accuracy``.
    mixing : float
        Damping of the Anderson mixing.
    history : int
        Iterations kept by the Anderson mixing.
    guess : str
        Starting charges of the cycle: ``"eeq"`` (default, charges of the
        electronegativity-equilibration model) or ``"neutral"`` (neutral atoms).
    trainable : bool or sequence of str
        Element parameters to train, by their names in the parameter file: ``LEV`` (shell
        levels), ``KCN`` (their coordination-number shifts, GFN2-xTB), ``POLY`` (distance
        polynomial), ``GAM``, ``LPAR`` and ``GAM3`` (hardness, its shell scaling and derivative),
        ``REPA`` and ``REPB`` (repulsion), ``DPOL`` and ``QPOL`` (anisotropic exchange-correlation,
        GFN2-xTB); ``True`` trains all of them, ``False`` (default) none. By name only, or with
        ``"all"``: ``EXP`` (Slater exponents), ``GLOBAL`` (the global parameters; those of the
        dispersion term stay fixed) and ``KPAIR`` (element-pair factors, GFN1-xTB). With trained
        exponents the integrals take the reference's functions.
    use_fast : bool or str, optional
        The fast path (see :meth:`set_use_fast`). ``"auto"`` (default) and ``True`` take it
        where it applies, ``False`` never.
    """

    method = ""
    #: Second derivatives of the energy by plain autograd lack the response of the electronic density;
    #: :meth:`energy_gradient` gives a gradient whose derivatives are exact (:func:`xnn.common.models.hessian`
    #: takes it).
    exact_second_derivatives = False
    #: Shell-resolved third-order term (GFN2) instead of the atomic one (GFN1).
    shell_third_order = False
    #: Anisotropic electrostatics and exchange-correlation (GFN2).
    multipoles = False
    #: Shell pairs with ``a b / (a + b) R^2`` above this for their most diffuse primitives are skipped.
    integral_screening = 25.0
    #: Elements of the largest temporary of one block of the integral evaluation.
    integral_budget = 1 << 23
    #: Primitive pairs with ``a b / (a + b) R^2`` above this are neglected, as in the reference program.
    integral_neglect = 25.0
    #: Element tables the ``trainable`` option accepts.
    trainable_names = TRAINABLE
    #: Trainable on request only (by name, or all with ``trainable="all"``): the Slater exponents
    #: and the global parameters (those of the dispersion term stay fixed).
    extra_trainable: Tuple[str, ...] = ("EXP", "GLOBAL")
    #: Tables written to a parameter file: by angular momentum, and one value per element.
    shell_tables = ("KCN", "POLY", "LPAR")
    atom_tables = ("GAM", "GAM3", "REPA", "REPB", "DPOL", "QPOL")
    #: With the fast path, a batch on a GPU is evaluated on the host when the root of the summed
    #: squares of its structures' atomic-orbital counts is at most this (one structure: its orbital
    #: count). Below it the evaluation is many short operations, which the host runs faster.
    host_basis = 200
    #: A cycle longer than this is run again with half the damping (see :meth:`_safeguarded`).
    long_cycle = 100
    #: Electronic temperature (K) of the warm start of a cycle that did not converge.
    warm_temperature = 3000.0
    #: Root-mean-square change of the charges and moments that ends the cycle at ``accuracy = 1``.
    charge_convergence = 2e-5
    #: Relative residual that ends the linear solve of the density response in training.
    response_tolerance = 1e-10
    #: Orbitals with occupations below this are left out of the density matrix.
    occupation_neglect = 1e-20

    def __init__(self, parameters: Union[str, XTBParameters, None] = None, electronic_temperature: float = 300.0,
                 max_iter: int = 500, accuracy: float = 1.0, mixing: float = 0.4, history: int = 8, use_fast="auto",
                 guess: str = "eeq", trainable: Union[bool, str, Sequence[str]] = False):
        super().__init__()
        if guess not in ("eeq", "neutral"):
            raise ValueError(f"guess must be 'eeq' or 'neutral', got {guess!r}")
        self.guess = guess
        from xnn.common.models.d4 import DFTD4
        self._eeq = DFTD4()
        self.set_use_fast(use_fast)
        if parameters is None or str(parameters).lower() in ("gfn0", "gfn1", "gfn2"):
            par = published_parameters(str(parameters or self.method).lower())
        elif isinstance(parameters, XTBParameters):
            par = parameters
        elif "\n" in str(parameters):
            par = parse_parameters(str(parameters), self.method)
        else:
            with open(parameters) as fh:
                par = parse_parameters(fh.read(), self.method)
        self.par = par
        self.cutoff = 0.1
        self.electronic_temperature = float(electronic_temperature)
        if not float(accuracy) > 0.0:
            raise ValueError(f"accuracy must be positive, got {accuracy!r}")
        self.max_iter, self.accuracy = int(max_iter), float(accuracy)
        self.mixing, self.history = float(mixing), int(history)
        en, rcov = _reference_tables()
        self.en = [float(x) for x in en]
        self.rcov = [float(x) / BOHR for x in rcov]
        self.supported = tuple(sorted(par.shells))
        self.last_scf: Optional[Dict[str, Tensor]] = None
        self._primitives: Dict[int, List[Tuple[List[float], List[float], int]]] = {}
        every = self.trainable_names + self.extra_trainable
        if trainable is True:
            names = self.trainable_names
        elif isinstance(trainable, str):
            names = every if trainable.lower() == "all" else (trainable,)
        else:
            names = tuple(trainable or ())
        unknown = sorted(set(names) - set(every))
        if unknown:
            raise ValueError(f"not trainable: {unknown}; choose from {list(every)}")
        self.trainable = tuple(n for n in every if n in names)
        self.global_names = tuple(sorted(par.globals))
        for name, table in self._element_tables().items():
            if name in self.trainable:
                self.register_parameter(f"table_{name}", torch.nn.Parameter(table))
            else:
                self.register_buffer(f"table_{name}", table, persistent=False)

    def _element_tables(self) -> Dict[str, Tensor]:
        """The element parameters as float64 tables indexed by the atomic number: ``LEV`` by the shell
        of the element, ``KCN``, ``POLY`` and ``LPAR`` by the angular momentum, the others one value."""
        par = self.par
        out = {name: torch.zeros(MAX_Z + 1, 3, dtype=torch.float64) for name in ("LEV", "EXP", "KCN", "POLY", "LPAR")}
        out["GLOBAL"] = torch.tensor([par.globals[name] for name in self.global_names], dtype=torch.float64)
        out.update({name: torch.zeros(MAX_Z + 1, dtype=torch.float64) for name in ("GAM", "GAM3", "REPA", "REPB",
                                                                                   "DPOL", "QPOL")})
        for z, shells in par.shells.items():
            for k in range(len(shells)):
                out["LEV"][z, k] = par.value(z, "LEV", k)
                out["EXP"][z, k] = par.value(z, "EXP", k)
            for l, letter in enumerate("SPD"):
                for name in ("KCN", "POLY", "LPAR"):
                    out[name][z, l] = par.value(z, name + letter)
            for name in ("GAM", "GAM3", "REPA", "REPB", "DPOL", "QPOL"):
                out[name][z] = par.value(z, name)
        return out

    def export_core(self) -> torch.nn.Module:
        """The scriptable core of :func:`~xnn.common.deploy.export_torchscript_potential` (see
        :mod:`~xnn.sqmnn.models.xtb_export`): this model's tables on its device, one molecule per call."""
        from .xtb_export import GFN0Export, GFN1Export, GFN2Export
        source = self
        if set(self.trainable) & set(self.extra_trainable):
            # trained exponents, global or pair parameters: a model read from their current values
            source = type(self)(parameters=self.parameter_text(digits=17), accuracy=self.accuracy,
                                electronic_temperature=self.electronic_temperature, max_iter=self.max_iter,
                                mixing=self.mixing, history=self.history, guess=self.guess)
            source = source.to(self.table("LEV").device)
        return {"gfn0": GFN0Export, "gfn1": GFN1Export, "gfn2": GFN2Export}[self.method](source)

    def parameter_text(self, digits: int = 10) -> str:
        """The parameter file of the model with the current values of its tables, in the xtb
        format: it loads again as ``parameters=`` and runs in the xtb program (``--vparam``)."""
        table = self.table("GLOBAL")
        return write_parameters(self.par, self._file_values(), digits=digits, pairs=self._file_pairs(),
                                globals={name: float(table[k]) for k, name in enumerate(self.global_names)})

    def _file_pairs(self) -> Optional[Dict[Tuple[int, int], float]]:
        """The element-pair factors of the parameter file (methods that have them)."""
        return None

    def _g(self):
        """The global parameters by name: values of the trained table, or the numbers of the file."""
        if "GLOBAL" in self.trainable:
            return _Globals(self.global_names, self.table("GLOBAL"))
        return self.par.globals

    def _file_values(self) -> Dict[Tuple[int, str], List[float]]:
        """The element tables as entries of the parameter file, ``(atomic number, key) -> values``."""
        values: Dict[Tuple[int, str], List[float]] = {}
        for z, shells in self.par.shells.items():
            values[(z, "LEV")] = [float(v) for v in self.table("LEV")[z, :len(shells)]]
            values[(z, "EXP")] = [float(v) for v in self.table("EXP")[z, :len(shells)]]
            for name in self.shell_tables:
                for l, letter in enumerate("SPD"):
                    values[(z, name + letter)] = [float(self.table(name)[z, l])]
            for name in self.atom_tables:
                values[(z, name)] = [float(self.table(name)[z])]
        return values

    def table(self, name: str) -> Tensor:
        """One table of :meth:`_element_tables` (a parameter when it is trained)."""
        return getattr(self, f"table_{name}").double()

    def set_use_fast(self, use_fast) -> "GFN":
        """Choose the fast path: ``True``, ``False`` or ``"auto"``.

        The integrals, their first derivatives and the derivatives of a force loss in the trained
        parameters run on the kernels of :mod:`xnn.sqmnn.models.xtb_kernels` (C++ loops on a CPU,
        Triton programs on a GPU). On a CPU, the three-body dispersion term of GFN2-xTB runs on the
        compiled tiles of :mod:`xnn.common.models.dispersion_fast.atm_cpu`. Second derivatives in
        the positions take the reference's functions. On a GPU, small batches (``host_basis``)
        are evaluated on the host.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    def _fused_integrals(self, pos: Tensor) -> bool:
        """Whether the shell-pair integrals take the kernels (the fast path where they run)."""
        if self.use_fast is False:
            return False
        from .xtb_kernels import available
        return available(pos.device)

    def _on_host(self, data: AtomicGraph) -> bool:
        """Whether a batch on a GPU is small enough for the host (fast path, see ``host_basis``)."""
        if self.use_fast is False or data.pos.device.type != "cuda" or self.trainable:
            return False
        per_element = torch.tensor([sum(2 * l + 1 for _, l in self.par.shells.get(z, ())) for z in range(MAX_Z + 1)],
                                   dtype=torch.float64)
        counts = torch.zeros(data.num_graphs, dtype=torch.float64).index_add(
            0, data.batch.cpu(), per_element[data.atomic_numbers.long().cpu()])
        return float((counts * counts).sum()) <= float(self.host_basis) ** 2

    def _host_model(self) -> "GFN":
        """This model on the host with the same options (kept apart from the registered modules, so
        that it stays there)."""
        options = (self.accuracy, self.max_iter, self.mixing, self.history, self.electronic_temperature, self.guess,
                   self.use_fast)
        twin = self.__dict__.get("_twin")
        if twin is None or self.__dict__.get("_twin_options") != options:
            self.__dict__["_twin"] = None
            twin = copy.deepcopy(self).to("cpu")
            self.__dict__["_twin"], self.__dict__["_twin_options"] = twin, options
        return twin.train(self.training)

    @classmethod
    def from_config(cls, cfg) -> "GFN":
        """Build from a :class:`~xnn.common.config.schema.ModelConfig`.

        Options in ``cfg.extra``: ``parameters`` (set name or file), ``electronic_temperature``,
        ``max_iter``, ``accuracy``, ``mixing``, ``history``, ``use_fast``, ``guess``, ``trainable``.
        """
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        keys = ("parameters", "electronic_temperature", "max_iter", "accuracy", "mixing", "history", "use_fast",
                "guess", "trainable")
        return cls(**{k: extra[k] for k in keys if k in extra})

    # element data

    def shell_primitives(self, z: int) -> List[Tuple[List[float], List[float], int]]:
        """``(exponents, coefficients, l)`` of every shell of an element (bohr^-2, normalized primitives)."""
        if z in self._primitives:
            return self._primitives[z]
        with torch.no_grad():
            out = [(a.tolist(), c.tolist(), l) for a, c, l in self._shell_tensors(z)]
        if "EXP" not in self.trainable:
            self._primitives[z] = out
        return out

    def _second_s(self, z: int, k: int) -> bool:
        """Whether shell ``k`` of an element is a second ``s`` shell (the diffuse hydrogen 2s)."""
        shells = self.par.shells[z]
        return shells[k][1] == 0 and any(l == 0 for _, l in shells[:k])

    def _shell_tensors(self, z: int) -> List[Tuple[Tensor, Tensor, int]]:
        """:meth:`shell_primitives` as float64 tensors that follow the table of Slater exponents."""
        out: List[Tuple[Tensor, Tensor, int]] = []
        table = self.table("EXP")
        for k, (n, l) in enumerate(self.par.shells[z]):
            second_s = self._second_s(z, k)
            a, c = STO_NG[(n, l, n_gaussians(self.method, n, l, second_s))]
            alpha = torch.tensor(a, dtype=torch.float64, device=table.device) * table[z, k] ** 2
            coef = torch.tensor(c, dtype=torch.float64, device=table.device)
            if second_s:
                # a second s shell is orthogonalized to the first
                alpha, coef = self._orthogonalized(out[0][0], out[0][1], alpha, coef)
            out.append((alpha, coef, l))
        return out

    @staticmethod
    def _orthogonalized(a1: Tensor, c1: Tensor, alpha: Tensor, coef: Tensor) -> Tuple[Tensor, Tensor]:
        """A second ``s`` shell Schmidt-orthogonalized to the first one on the same atom."""
        al = torch.cat([a1, alpha])
        c1t = c1 / self_overlap(a1, c1, 0).sqrt()
        c2t = coef / self_overlap(alpha, coef, 0).sqrt()
        cross = torch.cat([torch.zeros_like(c1t), c2t])
        base = torch.cat([c1t, torch.zeros_like(c2t)])
        s12 = (self_overlap(al, base + cross, 0) - 2.0) / 2.0
        new = cross - s12 * base
        return al, new / self_overlap(al, new, 0).sqrt()

    def _trained_primitives(self, basis: "_Basis") -> Tuple[Tensor, Tensor]:
        """Exponents and coefficients of the primitives of every shell, ``(n_shell, kmax)``, as
        functions of the trained Slater exponents."""
        zeta = self.table("EXP")[basis.shell_z, basis.shell_index]
        alpha = basis.alpha * ((zeta / zeta.detach()) ** 2)[:, None]
        coef = basis.coef
        rows = [s for s, (z, k) in enumerate(basis.shell_key) if self._second_s(z, k)]
        if rows:
            # a second s shell mixes the primitives of the first: both exponents enter its coefficients
            shells = {z: self._shell_tensors(z) for z in {basis.shell_key[s][0] for s in rows}}
            pad = torch.nn.functional.pad
            new = [shells[basis.shell_key[s][0]][basis.shell_key[s][1]] for s in rows]
            at = torch.tensor(rows, dtype=torch.long, device=alpha.device)
            alpha = alpha.index_copy(0, at, torch.stack([pad(a, (0, basis.kmax - a.shape[0]), value=1.0)
                                                         for a, _, _ in new]))
            coef = coef.index_copy(0, at, torch.stack([pad(c, (0, basis.kmax - c.shape[0])) for _, c, _ in new]))
        return alpha, coef

    def reference_occupations(self, z: int) -> List[float]:
        """Electrons per shell of the neutral reference atom."""
        raise NotImplementedError

    # method-specific pieces (implemented by GFN1 and GFN2)

    def coordination_numbers(self, z: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        """Coordination numbers of the atoms (pairs within each structure)."""
        raise NotImplementedError

    def shell_levels(self, z: List[int], basis: _Basis, cn: Tensor) -> Tensor:
        raise NotImplementedError

    def shell_pair_factor(self, basis: _Basis, pos: Tensor) -> Tensor:
        """``K X Y Pi`` of the Hamiltonian for every pair of shells, shape ``(n_shell, n_shell)``."""
        raise NotImplementedError

    def classical_energy(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int) -> Dict[str, Tensor]:
        """Repulsion and the charge-independent terms, per structure (hartree)."""
        raise NotImplementedError

    def _terms(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict) -> Dict[str, Tensor]:
        """Charge- and moment-dependent energies per structure (hartree), by name: those of
        ``scc_terms`` add up to the energy, those of ``reported_terms`` are returned by the model."""
        raise NotImplementedError

    def second_order_energy(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict,
                            per_structure: bool = False) -> Tensor:
        """Charge- and moment-dependent energy (hartree), summed over the batch or per structure."""
        terms = self._terms(q_shell, mu, theta, ctx)
        e = sum(terms[k] for k in self.scc_terms)
        return e if per_structure else e.sum()

    # the calculation

    def energy_gradient(self, data: AtomicGraph, create_graph: bool = False) -> Tuple[Tensor, Tensor]:
        """The positions and the gradient of the energy with its graph, whose derivatives are exact.

        The free energy is stationary in the self-consistent density, so its gradient is the derivative
        at a fixed density (with the overlap term of the energy-weighted density). The energy is formed
        with two aliases of the positions, one for the explicit dependences and one for the density
        (a function of the positions through the self-consistent equations, see
        :meth:`_responsive_density`); the gradient is taken in the first. Differentiated once more
        through both, it gives the exact Hessian.

        Parameters
        ----------
        data : AtomicGraph
            The structures.
        create_graph : bool, optional
            The Hessian will be differentiated once more (parameter gradients of curvatures): the
            evaluation takes the reference's functions, whose derivatives go to any order (the fast
            path's kernels stop at the second); default ``False``.

        Returns
        -------
        tuple of Tensor
            The positions (a tensor requiring gradients; ``data.pos`` itself when it does) and the
            gradient ``(N, 3)`` in eV / Angstrom, differentiable in them.
        """
        pos = data.pos if data.pos.requires_grad else data.pos.detach().requires_grad_(True)
        explicit, density = pos.view_as(pos), pos.view_as(pos)
        self.__dict__["_density_positions"] = density
        use_fast = self.use_fast
        if create_graph:
            self.use_fast = False
            self.__dict__["_second_order_density"] = True
        try:
            with torch.enable_grad():
                energy = self(dataclasses.replace(data, pos=explicit))["energy"]
                (grad,) = torch.autograd.grad(energy.sum(), explicit, create_graph=True)
        finally:
            self.__dict__.pop("_density_positions", None)
            self.__dict__.pop("_second_order_density", None)
            self.use_fast = use_fast
        return pos, grad

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Total energies (eV), charges and moments of a batch of molecules."""
        # the density at positions of its own (an alias of the positions, see energy_gradient)
        density_pos = self.__dict__.get("_density_positions") if torch.is_grad_enabled() else None
        if density_pos is None and self._on_host(data):
            twin = self._host_model()
            out = twin(data.to("cpu"))
            self.last_scf = twin.last_scf
            return {k: (v.to(data.pos.device) if isinstance(v, Tensor) else v) for k, v in out.items()}
        dtype = torch.float64
        device = data.pos.device
        pos = data.pos.to(dtype) / BOHR
        z = data.atomic_numbers.long()
        zl = z.tolist()
        missing = sorted(set(zl) - set(self.supported))
        if missing:
            raise ValueError(f"{type(self).__name__} has no parameters for Z = {missing}")
        batch = data.batch
        n_struct = data.num_graphs
        bl = batch.tolist()
        basis = _Basis(self, zl, bl, n_struct, dtype, device)

        charge = (data.total_charge.to(dtype) if data.total_charge is not None
                  else torch.zeros(n_struct, dtype=dtype, device=device))
        self._begin(z, pos, batch, n_struct, charge)

        # integrals and Hamiltonian (differentiable in the positions)
        S, DA, QA = self._integrals(basis, pos, n_struct)
        cn = self.coordination_numbers(z, pos, batch)
        H0 = self._hamiltonian(zl, basis, pos, S, cn, n_struct)
        ctx = self.scc_context(z, pos, basis, cn, batch, n_struct)

        n0 = torch.tensor([o for za in zl for o in self.reference_occupations(za)], dtype=dtype, device=device)
        shell_struct = torch.tensor(basis.shell_struct, dtype=torch.long, device=device)
        n_el = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, shell_struct, n0) - charge
        n_el_i = [int(round(float(x))) for x in n_el]
        mult = getattr(data, "spin_multiplicity", None)
        uhf = []
        for b in range(n_struct):
            m = int(mult[b]) if mult is not None else 0
            uhf.append(m - 1 if m > 0 else n_el_i[b] % 2)
        n_alpha = torch.tensor([(n + u) / 2 for n, u in zip(n_el_i, uhf)], dtype=dtype, device=device)
        n_beta = torch.tensor([(n - u) / 2 for n, u in zip(n_el_i, uhf)], dtype=dtype, device=device)

        with torch.no_grad():
            start = self._initial_charges(z, pos.detach(), batch, n_struct, charge, basis, n0)
            inputs = [t.detach() for t in (S, DA, QA, H0)]
            scc_ctx = {k: (v.detach() if isinstance(v, Tensor) else v) for k, v in ctx.items()}
            res = self._scc(basis, *inputs, scc_ctx, n0, n_alpha, n_beta, n_struct, start)
            if start is not None and not bool(res["converged"]):
                # the cycle did not converge from the model charges: start from neutral atoms
                retry = self._scc(basis, *inputs, scc_ctx, n0, n_alpha, n_beta, n_struct, None)
                if bool(retry["converged"]):
                    res = retry
            res = self._safeguarded(res, basis, inputs, scc_ctx, n0, n_alpha, n_beta, n_struct, start)
        self.last_scf = res
        P, W, ts = res["P"], res["W"], res["ts"]
        fixed = S.detach()
        if density_pos is not None:
            # the density as a function of the positions (and of trained parameters)
            pos_s = density_pos.to(dtype) / BOHR
            still = self._integrals(basis, pos_s, n_struct)
            fixed = still[0]
            P, W, ts = self._responsive_density(z, zl, pos_s, batch, basis, *still, n0, n_alpha, n_beta, n_struct,
                                                res["x"])
        elif self.trainable and self.training and torch.is_grad_enabled():
            # training: the density follows the parameters through the self-consistent solution
            still = (fixed, DA.detach(), QA.detach())
            if basis.alpha.requires_grad:
                # trained exponents: the integrals at fixed positions, as functions of the exponents
                still = self._integrals(basis, pos.detach(), n_struct)
                fixed = still[0]
            P, W, ts = self._responsive_density(z, zl, pos.detach(), batch, basis, *still, n0, n_alpha, n_beta,
                                                n_struct, res["x"])
        q_shell, mu, theta = self._moments(basis, P, S, DA, QA, n0)
        e_h0 = (P * H0).sum((-1, -2))
        terms = self._terms(q_shell, mu, theta, ctx)
        e_scc = sum(terms[k] for k in self.scc_terms)
        cls = self.classical_energy(z, pos, batch, n_struct)
        # energy-weighted density: makes the gradient include the overlap derivative (a zero that
        # depends on the positions only through the overlap)
        e_total = e_h0 + e_scc + cls["total"] - ts - (W * (S - fixed)).sum((-1, -2))
        energy = e_total * AUTOEV
        n_atoms = z.shape[0]
        counts = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, torch.ones_like(pos[:, 0]))
        node_energy = (energy / counts)[batch]
        q_atom = torch.zeros(n_atoms, dtype=dtype, device=device).index_add(0, basis.shell_atom, q_shell)
        out = {
            "energy": energy,
            "node_energy": node_energy,
            "charges": q_atom.detach(),
            "atomic_dipoles": mu.detach(),
            "atomic_quadrupoles": theta.detach(),
            "electronic_energy": ((e_h0 + e_scc - ts) * AUTOEV).detach(),
            "repulsion_energy": (cls["repulsion"] * AUTOEV).detach(),
            "scf_converged": res["converged"],
        }
        for k, v in cls.items():
            if k not in ("total", "repulsion"):
                out[f"{k}_energy"] = (v * AUTOEV).detach()
        for k in self.reported_terms:
            out[f"{k}_energy"] = (terms[k] * AUTOEV).detach()
        return out

    def scc_context(self, z, pos, basis, cn, batch, n_struct) -> dict:
        return {}

    def _begin(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int, charge: Tensor) -> None:
        """What a method prepares from the structures and their total charges before the Hamiltonian."""

    def _integrals(self, basis: _Basis, pos: Tensor, n_struct: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Dense overlap ``S``, dipole ``DA`` and second-moment ``QA`` matrices per structure.

        ``DA[b, c, mu, nu]`` and ``QA[b, c, mu, nu]`` are the integrals of ``r - R`` and
        ``(r - R)_i (r - R)_j`` with ``R`` the centre of ``nu``."""
        ti, tj, kind = self._shell_pairs(basis, pos)
        if self._fused_integrals(pos) and not (basis.alpha.requires_grad or basis.weight.requires_grad):
            classes = self._pair_classes(ti, tj, kind)
            if pos.requires_grad and torch.is_grad_enabled():
                M = _FusedIntegrals.apply(pos, self, basis, n_struct, ti, tj, kind, classes)
            else:
                from .xtb_kernels import integral_matrices
                M = integral_matrices(pos, basis.kernel_tables, classes, n_struct, 10 if self.multipoles else 1,
                                      basis.nb, self.integral_neglect)
        else:
            M = self._integral_stack(basis, pos, n_struct, ti, tj, kind)
        # padding: unit overlap keeps the generalized eigenproblem regular
        pad = torch.zeros(n_struct, basis.nb, dtype=pos.dtype, device=pos.device)
        for b in range(n_struct):
            pad[b, basis.n_ao[b]:] = 1.0
        S = M[:, 0] + torch.diag_embed(pad)
        return S, M[:, 1:4], M[:, 4:10]

    def _shell_pairs(self, basis: _Basis, pos: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """The shell pairs ``i <= j`` of each structure with integrals above round-off, and the class of
        each (angular momenta and primitive counts, packed)."""
        ns = len(basis.shell_l)
        ti, tj = torch.triu_indices(ns, ns, 0, device=pos.device)
        keep = basis.shell_struct_t[ti] == basis.shell_struct_t[tj]
        ti, tj = ti[keep], tj[keep]
        # pairs whose most diffuse primitives decay below exp(-integral_screening) vanish to round-off
        with torch.no_grad():
            amin = basis.alpha.masked_fill(basis.coef == 0, float("inf")).min(-1).values
            sa = basis.shell_atom
            r2 = ((pos[sa[ti]] - pos[sa[tj]]) ** 2).sum(-1)
            near = amin[ti] * amin[tj] / (amin[ti] + amin[tj]) * r2 <= self.integral_screening
        ti, tj = ti[near], tj[near]
        sl, sk = basis.shell_l_t, basis.shell_k
        return ti, tj, ((sl[ti] * 3 + sl[tj]) * 64 + sk[ti]) * 64 + sk[tj]

    @staticmethod
    def _pair_classes(ti: Tensor, tj: Tensor, kind: Tensor) -> list:
        """The shell pairs by class: ``(la, lb, ka, kb, bra shells, ket shells)`` (what the kernels take)."""
        classes = []
        for code in torch.unique(kind).tolist():
            sel = torch.nonzero(kind == code).squeeze(1)
            classes.append((code // (3 * 4096), code // 4096 % 3, code // 64 % 64, code % 64,
                            ti.index_select(0, sel), tj.index_select(0, sel)))
        return classes

    def _integral_stack(self, basis: _Basis, pos: Tensor, n_struct: int, ti: Tensor, tj: Tensor,
                        kind: Tensor) -> Tensor:
        """The matrices ``(n_struct, n_ops, nb, nb)`` of the shell pairs (overlap, then with multipoles
        the three dipole and six second-moment ones)."""
        nb = basis.nb
        n_ops = 10 if self.multipoles else 1
        index: List[Tensor] = []
        values: List[Tensor] = []
        # one class per pair of angular momenta and primitive counts: no padded primitives
        for code in torch.unique(kind).tolist():
            la, lb, ka, kb = code // (3 * 4096), code // 4096 % 3, code // 64 % 64, code % 64
            sel = torch.nonzero(kind == code).squeeze(1)
            per_pair = ka * kb * 10 * (2 * la + 1) * (2 * lb + 1) * (6 if la == 2 or lb == 2 else 1)
            chunk = max(64, self.integral_budget // per_pair)
            for start in range(0, sel.shape[0], chunk):
                part = sel[start:start + chunk]
                self._pair_block(basis, pos, ti[part], tj[part], la, lb, ka, kb, index, values)
        # one scatter of all blocks into the stacked matrices
        M = torch.zeros(n_struct * n_ops * nb * nb, dtype=pos.dtype, device=pos.device)
        if index:
            M = M.index_add(0, torch.cat(index), torch.cat(values))
        return M.view(n_struct, n_ops, nb, nb)

    def _pair_block(self, basis: _Basis, pos: Tensor, ii: Tensor, jj: Tensor, la: int, lb: int, ka: int, kb: int,
                    index: List[Tensor], values: List[Tensor]) -> None:
        """Append the integrals of the shell pairs ``(ii, jj)`` (``ii <= jj``) and their positions in the
        flattened stack of dense matrices ``M[b, op]`` (overlap, then with multipoles the three dipole
        and six second-moment ones)."""
        device = pos.device
        sa = basis.shell_atom
        ra, rb = pos.index_select(0, sa[ii]), pos.index_select(0, sa[jj])
        args = (ra, rb, basis.alpha[ii, :ka], basis.weight[ii, :ka], la, basis.alpha[jj, :kb],
                basis.weight[jj, :kb], lb, self.multipoles, self.integral_neglect, True)
        s, d, q = shell_pair_integrals(*args)
        if self.multipoles:
            # origin on the bra centre A, and on the ket centre B: <a|r - B|b> = <a|r - A|b> + (A - B) S_ab
            # (the second moments likewise): a linear map of the ten operators of each pair
            dvec = ra - rb
            first, second = _origin_shift(dvec.dtype, device)
            pairs = dvec[:, [0, 1, 2, 0, 0, 1]] * dvec[:, [0, 1, 2, 1, 2, 2]]
            shift = torch.eye(10, dtype=dvec.dtype, device=device).view(100) + dvec @ first + pairs @ second
            on_a = torch.cat([s[:, None], d, q], 1)                      # (P, 10, na, nb)
            on_b = torch.bmm(shift.view(-1, 10, 10), on_a.flatten(2)).view(on_a.shape)
        else:
            on_a = on_b = s[:, None]
        n_ops, na, nbf, nb = on_a.shape[1], N_SPHERICAL[la], N_SPHERICAL[lb], basis.nb
        first_a, first_b = basis.shell_first_t[ii], basis.shell_first_t[jj]
        ia = (basis.ao_index[first_a][:, None] + torch.arange(na, device=device))[:, None, :, None]
        ib = (basis.ao_index[first_b][:, None] + torch.arange(nbf, device=device))[:, None, None, :]
        matrix = (basis.ao_struct[first_a][:, None] * n_ops + torch.arange(n_ops, device=device))[:, :, None, None]
        # element [mu, nu] carries the origin on the centre of nu
        index.append(((matrix * nb + ia) * nb + ib).reshape(-1))
        values.append(on_b.reshape(-1))
        off = ii != jj
        index.append(((matrix * nb + ib) * nb + ia).expand(on_a.shape)[off].reshape(-1))
        values.append(on_a[off].reshape(-1))

    def _hamiltonian(self, z: List[int], basis: _Basis, pos: Tensor, S: Tensor, cn: Tensor, n_struct: int) -> Tensor:
        """The extended Hueckel matrix ``H0`` per structure (hartree)."""
        dtype, device = pos.dtype, pos.device
        nb = basis.nb
        hs = self.shell_levels(z, basis, cn)                         # (n_shell,) hartree
        factor = self.shell_pair_factor(basis, pos)                  # (n_shell, n_shell)
        H0 = torch.zeros(n_struct, nb, nb, dtype=dtype, device=device)
        for b in range(n_struct):
            sel = torch.nonzero(basis.ao_struct == b).squeeze(1)
            n = sel.shape[0]
            idx = basis.ao_index[sel]
            sh = basis.ao_shell[sel]
            at = basis.ao_atom[sel]
            # index_select: its backward is an index_add, that of advanced indexing a sorted scatter
            hb = hs.index_select(0, sh)
            Sb = S[b].index_select(0, idx).index_select(1, idx)
            off = 0.5 * (hb[:, None] + hb[None, :]) * Sb * factor.index_select(0, sh).index_select(1, sh)
            # on-site blocks: the levels times the (not exactly unit) self-overlaps of the expansions
            Hb = torch.where(at[:, None] == at[None, :], torch.zeros_like(off), off) + torch.diag(hb * Sb.diagonal())
            H0[b, :n, :n] = Hb
            if n < nb:
                pad = torch.arange(n, nb, device=device)
                H0[b, pad, pad] = 1000.0
        return H0

    def _shell_constants(self, basis: _Basis, dt, dev) -> Tuple[Tensor, Tensor]:
        """Slater exponents and distance-polynomial coefficients of every shell."""
        zeta = self.table("EXP")[basis.shell_z, basis.shell_index]
        kpoly = 0.01 * self.table("POLY")[basis.shell_z, basis.shell_l_t]
        return zeta, kpoly

    def _electronegativity_factor(self, basis: _Basis, dt, dev) -> Tensor:
        """``1 + k_EN (EN_A - EN_B)^2`` for every pair of shells."""
        en = torch.tensor(self.en, dtype=dt, device=dev)[basis.shell_z]
        return 1.0 + 0.01 * self._g()["enscale"] * (en[:, None] - en[None, :]) ** 2

    def _distance_polynomial(self, basis: _Basis, pos: Tensor, kpoly: Tensor) -> Tensor:
        """``(1 + k_A sqrt(R / R_AB)) (1 + k_B sqrt(R / R_AB))`` for every pair of shells (unit on one atom)."""
        sa = basis.shell_atom
        p = pos.index_select(0, sa)
        diff = p[:, None, :] - p[None, :, :]
        same = sa[:, None] == sa[None, :]
        r = torch.sqrt((diff * diff).sum(-1) + same.to(pos.dtype))
        rc = torch.tensor(self.radii, dtype=pos.dtype, device=pos.device)[basis.shell_z]
        ratio = torch.sqrt(r / (rc[:, None] + rc[None, :]))
        return (1.0 + kpoly[:, None] * ratio) * (1.0 + kpoly[None, :] * ratio)

    def _moments(self, basis: _Basis, P: Tensor, S: Tensor, DA: Tensor, QA: Tensor, n0: Tensor,
                 stack: Optional[Tensor] = None):
        """Mulliken shell charges, atomic dipoles and second moments (atomic units); with the integral
        matrices as ``stack[b, nu, c, mu]``, one product per column gives all of them."""
        n_shell = len(basis.shell_l)
        n_atoms = int(basis.ao_atom.max()) + 1 if basis.ao_atom.numel() else 0
        b, i = basis.ao_struct, basis.ao_index
        nb = P.shape[-1]
        flat = b * nb + i
        if stack is not None:
            sums = torch.matmul(stack, P.transpose(1, 2).unsqueeze(-1)).squeeze(-1)      # (B, nb, n_ops)
            sums = sums.reshape(-1, sums.shape[-1]).index_select(0, flat)
            q_shell = n0 - torch.zeros(n_shell, dtype=P.dtype, device=P.device).index_add(0, basis.ao_shell, sums[:, 0])
            mu = torch.zeros(n_atoms, 3, dtype=P.dtype, device=P.device)
            theta = torch.zeros(n_atoms, 6, dtype=P.dtype, device=P.device)
            if self.multipoles:
                mu = mu.index_add(0, basis.ao_atom, -sums[:, 1:4])
                theta = theta.index_add(0, basis.ao_atom, -sums[:, 4:10])
            return q_shell, mu, theta
        PS = (P * S).sum(-1)                                         # (B, nb): sum_nu P_mu,nu S_nu,mu
        pop_ao = PS.reshape(-1).index_select(0, flat)
        pop_shell = torch.zeros(n_shell, dtype=P.dtype, device=P.device).index_add(0, basis.ao_shell, pop_ao)
        q_shell = n0 - pop_shell
        if not self.multipoles:
            return (q_shell, torch.zeros(n_atoms, 3, dtype=P.dtype, device=P.device),
                    torch.zeros(n_atoms, 6, dtype=P.dtype, device=P.device))
        # dipole: mu_A = - sum_{k in A} sum_l P_lk DA[l, k]
        pd = (P[:, None] * DA).sum(-2)                               # (B, 3, nb): column sums over l
        pq = (P[:, None] * QA).sum(-2)                               # (B, 6, nb)
        pd = pd.transpose(1, 2).reshape(-1, 3).index_select(0, flat)
        pq = pq.transpose(1, 2).reshape(-1, 6).index_select(0, flat)
        mu = -torch.zeros(n_atoms, 3, dtype=P.dtype, device=P.device).index_add(0, basis.ao_atom, pd)
        theta = -torch.zeros(n_atoms, 6, dtype=P.dtype, device=P.device).index_add(0, basis.ao_atom, pq)
        return q_shell, mu, theta

    def _ao_potentials(self, basis: _Basis, v_shell: Tensor, v_mu: Tensor, v_theta: Tensor, n_struct: int) -> Tensor:
        """The potentials of the shell charges and, with multipoles, of the atomic dipoles and second
        moments on the atomic orbitals, ``(B, n_ops, nb)``. The Fock matrix is
        ``H0 - (Z + Z^T) / 2`` with ``Z[mu, nu] = sum_c M_c[mu, nu] V[c, nu]`` over the overlap, dipole
        and second-moment matrices ``M_c``."""
        vals = v_shell[basis.ao_shell][:, None]
        if self.multipoles:
            vals = torch.cat([vals, v_mu[basis.ao_atom], v_theta[basis.ao_atom]], 1)
        V = vals.new_zeros(n_struct, vals.shape[1], basis.nb)
        V[basis.ao_struct, :, basis.ao_index] = vals
        return V

    def potentials(self, q_shell: Tensor, mu: Tensor, theta: Tensor, ctx: dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Derivatives of the charge-dependent energy with respect to the shell charges, atomic
        dipoles and second moments (by automatic differentiation; the methods give closed forms)."""
        inputs = [t.detach().clone().requires_grad_(True) for t in (q_shell, mu, theta)]
        with torch.enable_grad():
            e = self.second_order_energy(*inputs, ctx)
            grads = torch.autograd.grad(e, inputs, allow_unused=True)
        return tuple(g if g is not None else torch.zeros_like(t) for g, t in zip(grads, inputs))

    def _fock(self, q_shell: Tensor, mu: Tensor, theta: Tensor, basis: _Basis, ctx: dict, ortho: "_Orthogonalizer",
              left: Tensor, h0: Tensor) -> Tensor:
        """The Fock matrix in the orthogonal basis from the charges and moments: one product (or
        triangular solve) with the transformed integral matrices ``left`` (in the layout of
        :func:`_stacked`) and ``h0``."""
        V = self._ao_potentials(basis, *self.potentials(q_shell, mu, theta, ctx), left.shape[0])
        # Z^T[nu, mu] = sum_c left_c[mu, nu] V[c, nu]: one product per column over its operators
        Zt = torch.matmul(left, V.transpose(1, 2).unsqueeze(-1)).squeeze(-1)
        K = ortho.left(Zt)
        return h0 - 0.5 * (K + K.transpose(-1, -2))

    def _scc_step(self, x, basis, S, DA, QA, ctx, n0, n_alpha, n_beta, ortho, left, h0, kt, n_shell, n_atoms,
                  stack: Optional[Tensor] = None):
        """One map of the cycle: charges and moments -> Fock matrices -> occupations -> new charges and moments."""
        dtype, device = S.dtype, S.device
        q_shell = x[:n_shell]
        if self.multipoles:
            mu = x[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3)
            theta = x[n_shell + 3 * n_atoms:].view(n_atoms, 6)
        else:
            mu = torch.zeros(n_atoms, 3, dtype=dtype, device=device)
            theta = torch.zeros(n_atoms, 6, dtype=dtype, device=device)
        eps, Co = torch.linalg.eigh(self._fock(q_shell, mu, theta, basis, ctx, ortho, left, h0))
        occ, ts = _fermi(eps, n_alpha, n_beta, kt, basis.n_ao)
        # the levels are ascending: orbitals beyond the last occupied one add nothing to the density
        n_occ = max(1, int((occ > self.occupation_neglect).sum(-1).max()))
        eps, occ = eps[:, :n_occ], occ[:, :n_occ]
        C = ortho.back(Co[:, :, :n_occ])
        P = (C * occ[:, None, :]) @ C.transpose(-1, -2)
        qn, mun, thn = self._moments(basis, P, S, DA, QA, n0, stack)
        xn = torch.cat([qn, mun.reshape(-1), thn.reshape(-1)]) if self.multipoles else qn
        return {"P": P, "C": C, "eps": eps, "occ": occ, "ts": ts}, xn

    def _responsive_density(self, z: Tensor, zl: List[int], pos: Tensor, batch: Tensor, basis: _Basis, S: Tensor,
                            DA: Tensor, QA: Tensor, n0: Tensor, n_alpha: Tensor, n_beta: Tensor, n_struct: int,
                            x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """The converged density, energy-weighted density and ``T S_el`` as differentiable functions
        of the trained parameters (at fixed positions).

        With the map ``g`` of one iteration, the converged charges and moments obey ``x = g(x, p)``, so
        a gradient ``v`` in ``x`` reaches the parameters ``p`` as ``y dg/dp`` with
        ``(1 - (dg/dx)^T) y = v``; this linear system is solved by Anderson-mixed fixed-point steps,
        each one backward pass through an iteration (whose density uses first-order perturbation
        theory at the Fermi occupations, :class:`_SpectralDensity`).
        """
        cn = self.coordination_numbers(z, pos, batch)
        H0 = self._hamiltonian(zl, basis, pos, S, cn, n_struct)
        ctx = self.scc_context(z, pos, basis, cn, batch, n_struct)
        ortho = _Orthogonalizer(S)
        if ortho.L is None:
            raise RuntimeError("training needs a positive definite overlap matrix")
        left = _stacked(ortho.left(torch.cat([S[:, None], DA, QA], 1) if self.multipoles else S[:, None]))
        h0 = ortho.right(ortho.left(H0))
        n_shell, n_atoms = n0.shape[0], int(basis.ao_atom.max()) + 1
        kt = KB_AU * self.electronic_temperature

        def fock(xx: Tensor, c: dict) -> Tensor:
            if self.multipoles:
                mu = xx[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3)
                theta = xx[n_shell + 3 * n_atoms:].view(n_atoms, 6)
            else:
                mu, theta = xx.new_zeros(n_atoms, 3), xx.new_zeros(n_atoms, 6)
            return self._fock(xx[:n_shell], mu, theta, basis, c, ortho, left, h0)

        with torch.no_grad():
            # on a copy of the context: matrices it caches here would be constants for the density below
            eps, U = torch.linalg.eigh(fock(x, dict(ctx)))
            f, level = _fermi_channels(eps, n_alpha, n_beta, kt, basis.n_ao)

        def density(xx: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            Po, Wo, ts = _SpectralDensity.apply(fock(xx, ctx), eps, U, f, level, kt)
            # back to the atomic orbitals: L^-T M L^-1 of the symmetric matrices
            P = ortho.back(ortho.back(Po).transpose(-1, -2)).transpose(-1, -2)
            W = ortho.back(ortho.back(Wo).transpose(-1, -2)).transpose(-1, -2)
            q, m, th = self._moments(basis, P, S, DA, QA, n0)
            return P, W, ts, (torch.cat([q, m.reshape(-1), th.reshape(-1)]) if self.multipoles else q)

        x_in = x.detach().clone().requires_grad_(True)
        x_out = density(x_in)[3]

        if self.__dict__.get("_second_order_density", False):
            # derivatives to second order (parameter gradients of Hessians): with M = (1 - dg/dx)^-1 at the
            # solution, x1 = x with the adjoint M^T (exact first derivatives, linear in the incoming gradient)
            # and the step x2 = x1 + M (g(x1) - x1), whose second derivatives are those of the solution (the
            # residual and its first derivative vanish there, so M enters as a constant)
            eye = torch.eye(x_out.numel(), dtype=x_out.dtype, device=x_out.device)
            rows = [torch.autograd.grad(x_out, x_in, e, retain_graph=True, allow_unused=True)[0] for e in eye]
            J = torch.stack([torch.zeros_like(x_out) if r is None else r for r in rows])
            M = torch.linalg.inv(eye - J)
            x1 = x_out.clone()
            x1.register_hook(lambda v: M.transpose(0, 1) @ v)
            x2 = x1 + M @ (density(x1)[3] - x1)
            return density(x2)[:3]

        def adjoint(v: Tensor) -> Tensor:
            scale = float(v.detach().abs().max())
            if scale == 0.0:
                return v
            y, ys, rs = v, [], []
            for _ in range(self.max_iter):
                back = torch.autograd.grad(x_out, x_in, y, retain_graph=True, allow_unused=True)[0]
                r = v - y if back is None else v + back - y
                if float(r.abs().max()) <= self.response_tolerance * scale:
                    break
                ys, rs = (ys + [y])[-self.history:], (rs + [r])[-self.history:]
                y = _anderson(ys, rs, 1.0)
            return y

        x_sc = x_out.clone()
        x_sc.register_hook(adjoint)
        return density(x_sc)[:3]

    def _initial_charges(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int, charge: Tensor,
                         basis: _Basis, n0: Tensor) -> Optional[Tensor]:
        """Shell charges that start the cycle: with ``guess="eeq"`` the charges of the
        electronegativity-equilibration model, shared among the shells of an atom in
        proportion to their reference occupations; ``None`` (neutral atoms) otherwise."""
        if self.guess != "eeq":
            return None
        eeq = self._eeq.to(pos.dtype)
        i, j = _all_pairs(batch)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        dij = pos.index_select(0, j) - pos.index_select(0, i)
        rr = torch.cat([dij, -dij]).norm(dim=-1)
        _, cn = eeq.coordination_numbers(z, ei, rr, z.shape[0])
        q = torch.zeros(z.shape[0], dtype=pos.dtype, device=pos.device)
        cell = torch.zeros(3, 3, dtype=pos.dtype, device=pos.device)
        for b in range(n_struct):
            members = torch.nonzero(batch == b).squeeze(1)
            q[members] = eeq.eeq_charges(z[members], pos[members], cn[members], charge[b], cell, False)
        sa = basis.shell_atom
        total = torch.zeros_like(q).index_add(0, sa, n0)
        return q[sa] * n0 / total[sa]

    def _safeguarded(self, res: dict, basis: _Basis, inputs: list, ctx: dict, n0: Tensor, n_alpha: Tensor,
                     n_beta: Tensor, n_struct: int, start: Optional[Tensor]) -> dict:
        """The result of the cycle after two safeguards. A cycle that did not converge is run again
        from the charges that converge at a higher electronic temperature (``warm_temperature``). A
        cycle longer than ``long_cycle`` iterations may have several solutions: it is run again with
        half the damping, and each structure keeps the solution of lower electronic free energy."""
        def run(**options) -> dict:
            return self._scc(basis, *inputs, ctx, n0, n_alpha, n_beta, n_struct, **options)

        n_shell = n0.shape[0]
        if not bool(res["converged"]):
            hot = max(self.warm_temperature, 10.0 * self.electronic_temperature)
            warm = run(start=start, temperature=hot)
            alt = run(start=warm["x"][:n_shell])
            alt["iterations"] += res["iterations"] + warm["iterations"]
            return alt if bool(alt["converged"]) else res
        if res["iterations"] <= self.long_cycle:
            return res
        alt = run(start=start, mixing=0.5 * self.mixing)
        if not bool(alt["converged"]):
            return res
        n_atoms = int(basis.ao_atom.max()) + 1
        atom_struct = torch.zeros(n_atoms, dtype=torch.long, device=n0.device)
        atom_struct[basis.shell_atom] = basis.shell_struct_t
        owner = basis.shell_struct_t
        if self.multipoles:
            owner = torch.cat([owner, atom_struct.repeat_interleave(3), atom_struct.repeat_interleave(6)])

        def free_energy(r: dict) -> Tensor:
            x = r["x"]
            mu = x[n_shell:n_shell + 3 * n_atoms].view(n_atoms, 3) if self.multipoles else x.new_zeros(n_atoms, 3)
            theta = x[n_shell + 3 * n_atoms:].view(n_atoms, 6) if self.multipoles else x.new_zeros(n_atoms, 6)
            terms = self._terms(x[:n_shell], mu, theta, ctx)
            return (r["P"] * inputs[3]).sum((-1, -2)) + sum(terms[k] for k in self.scc_terms) - r["ts"]

        take = free_energy(alt) < free_energy(res) - 1e-9
        out = dict(res)
        out["iterations"] = res["iterations"] + alt["iterations"]
        if not bool(take.any()):
            return out
        width = max(res["occ"].shape[-1], alt["occ"].shape[-1])
        for key in ("P", "W", "ts", "C", "eps", "occ"):
            a, b = res[key], alt[key]
            if key in ("C", "eps", "occ"):
                a = torch.nn.functional.pad(a, (0, width - a.shape[-1]))
                b = torch.nn.functional.pad(b, (0, width - b.shape[-1]))
            out[key] = torch.where(take.view(-1, *([1] * (a.dim() - 1))), b, a)
        out["x"] = torch.where(take[owner], alt["x"], res["x"])
        return out

    def _scc(self, basis: _Basis, S, DA, QA, H0, ctx, n0, n_alpha, n_beta, n_struct,
             start: Optional[Tensor] = None, mixing: Optional[float] = None,
             temperature: Optional[float] = None) -> Dict[str, Tensor]:
        """Self-consistent charges by Anderson mixing of the shell charges and atomic moments; the mixing
        restarts from the best iterate with half the damping when the residual grows or stalls.
        ``mixing`` and ``temperature`` replace the model's damping and electronic temperature."""
        dtype, device = S.dtype, S.device
        n_shell = n0.shape[0]
        n_atoms = int(basis.ao_atom.max()) + 1
        # the integral matrices and H0 with the orthogonalizing transformation applied
        ortho = _Orthogonalizer(S)
        M = torch.cat([S[:, None], DA, QA], 1) if self.multipoles else S[:, None]
        stack = M.permute(0, 3, 1, 2).contiguous()
        left = _stacked(ortho.left(M))
        del M
        h0 = ortho.right(ortho.left(H0))
        kt = KB_AU * (self.electronic_temperature if temperature is None else temperature)
        x = torch.zeros(n_shell + 9 * n_atoms * int(self.multipoles), dtype=dtype, device=device)
        if start is not None:
            x[:n_shell] = start
        # the structure of every mixed quantity, for the convergence test per structure
        atom_struct = torch.zeros(n_atoms, dtype=torch.long, device=device)
        atom_struct[basis.shell_atom] = basis.shell_struct_t
        owner = basis.shell_struct_t
        if self.multipoles:
            owner = torch.cat([owner, atom_struct.repeat_interleave(3), atom_struct.repeat_interleave(6)])
        count = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, owner, torch.ones_like(x))
        threshold = self.charge_convergence * self.accuracy
        xs, rs = [], []
        converged = False
        result = None
        iterations = 0
        beta = self.mixing if mixing is None else mixing
        best_err, best_x, worse = float("inf"), x.clone(), 0
        checkpoint = float("inf")
        for _ in range(self.max_iter):
            iterations += 1
            result, xn = self._scc_step(x, basis, S, DA, QA, ctx, n0, n_alpha, n_beta, ortho, left, h0, kt, n_shell,
                                        n_atoms, stack)
            r = xn - x
            err = float(r.abs().max())
            rms = torch.sqrt(torch.zeros_like(count).index_add(0, owner, r * r) / count)
            if float(rms.max()) < threshold:
                converged = True
                break
            if err < best_err:
                best_err, best_x, worse = err, x.clone(), 0
            else:
                worse += 1
            stalled = iterations % 40 == 0 and best_err > 1e-2 * checkpoint
            if iterations % 40 == 0:
                checkpoint = best_err
            if (worse >= 8 and err > 2.0 * best_err) or stalled:
                # restart from the best iterate with a smaller damping
                beta, worse = 0.5 * beta, 0
                xs, rs = [], []
                x = best_x.clone()
                continue
            xs.append(x.clone())
            rs.append(r.clone())
            if len(xs) > self.history:
                xs.pop(0)
                rs.pop(0)
            x = _anderson(xs, rs, beta)
        W = (result["C"] * (result["occ"] * result["eps"])[:, None, :]) @ result["C"].transpose(-1, -2)
        result["W"] = W
        result["x"] = x
        result["converged"] = torch.tensor(converged, device=device)
        result["iterations"] = iterations
        return result


class _FusedIntegrals(torch.autograd.Function):
    """The dense integral matrices from the kernels, with their gradient in the positions. A
    gradient that is differentiated again is the sum of two nodes: :class:`_IntegralGradient`,
    which carries its dependence on the incoming weights, and :class:`_IntegralCurvature`, which
    carries the one on the positions and is evaluated only when a position gradient is asked for."""

    @staticmethod
    def forward(ctx, pos, model, basis, n_struct, ti, tj, kind, classes):
        from .xtb_kernels import integral_matrices
        ctx.save_for_backward(pos)
        ctx.constants = (model, basis, n_struct, ti, tj, kind, classes)
        return integral_matrices(pos, basis.kernel_tables, classes, n_struct, 10 if model.multipoles else 1,
                                 basis.nb, model.integral_neglect)

    @staticmethod
    def backward(ctx, grad):
        (pos,) = ctx.saved_tensors
        if torch.is_grad_enabled():
            g = _IntegralGradient.apply(grad, pos.detach(), *ctx.constants)
            g = g + _IntegralCurvature.apply(pos, grad.detach(), *ctx.constants)
        else:
            g = _IntegralGradient.forward(None, grad, pos, *ctx.constants)
        return g, None, None, None, None, None, None, None


class _IntegralGradient(torch.autograd.Function):
    """The gradient in the positions of the integral matrices contracted with a weight, as a
    function of the weight (linear: its gradient is the matrices differentiated along the
    incoming displacement)."""

    @staticmethod
    def forward(ctx, weight, pos, model, basis, n_struct, ti, tj, kind, classes):
        from .xtb_kernels import integral_gradient
        if ctx is not None:
            ctx.save_for_backward(pos)
            ctx.constants = (model, basis, n_struct, classes)
        return integral_gradient(pos, basis.kernel_tables, classes, 10 if model.multipoles else 1, basis.nb,
                                 model.integral_neglect, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, direction):
        from .xtb_kernels import integral_tangent
        (pos,) = ctx.saved_tensors
        model, basis, n_struct, classes = ctx.constants
        tangent = integral_tangent(pos, basis.kernel_tables, classes, n_struct, 10 if model.multipoles else 1,
                                   basis.nb, model.integral_neglect, direction)
        return tangent, None, None, None, None, None, None, None, None


class _IntegralCurvature(torch.autograd.Function):
    """A zero that carries the dependence on the positions of the gradient of
    :class:`_IntegralGradient`: its backward is the second derivative of the contracted matrices
    along the incoming displacement, from the reference's functions."""

    @staticmethod
    def forward(ctx, pos, weight, model, basis, n_struct, ti, tj, kind, classes):
        ctx.save_for_backward(pos, weight)
        ctx.constants = (model, basis, n_struct, ti, tj, kind)
        return torch.zeros_like(pos)

    @staticmethod
    @once_differentiable
    def backward(ctx, direction):
        pos, weight = ctx.saved_tensors
        model, basis, n_struct, ti, tj, kind = ctx.constants
        with torch.enable_grad():
            p = pos.detach().requires_grad_(True)
            M = model._integral_stack(basis, p, n_struct, ti, tj, kind)
            (g,) = torch.autograd.grad(M, p, weight, create_graph=True)
            (h,) = torch.autograd.grad(g, p, direction)
        return h, None, None, None, None, None, None, None, None


class _Orthogonalizer:
    """The transformation ``T`` that turns ``F C = S C eps`` into a standard eigenproblem of
    ``T F T^T``: the inverse of the Cholesky factor of the overlap, ``S = L L^T``, or, where the
    factorization fails (a numerically singular overlap), ``T^T = U s^-1/2`` from the eigenvectors
    and eigenvalues of ``S``."""

    def __init__(self, S: Tensor):
        L, info = torch.linalg.cholesky_ex(S)
        self.L: Optional[Tensor] = None if bool(info.any()) else L
        self.X: Optional[Tensor] = None
        if self.L is None:
            es, us = torch.linalg.eigh(S)
            self.X = us * es.clamp_min(1e-14).rsqrt()[:, None, :]

    def left(self, M: Tensor) -> Tensor:
        """``T M`` for ``M`` of shape ``(B, nb, nb)`` or ``(B, k, nb, nb)``."""
        if self.L is None:
            Xt = self.X.transpose(-1, -2)
            return (Xt if M.dim() == 3 else Xt[:, None]) @ M
        return torch.linalg.solve_triangular(self.L if M.dim() == 3 else self.L[:, None], M, upper=False)

    def right(self, M: Tensor) -> Tensor:
        """``M T^T``."""
        if self.L is None:
            return M @ self.X
        return torch.linalg.solve_triangular(self.L.transpose(-1, -2), M, upper=True, left=False)

    def back(self, C: Tensor) -> Tensor:
        """Eigenvectors of the standard problem to those of the generalized one, ``T^T C``."""
        if self.L is None:
            return self.X @ C
        return torch.linalg.solve_triangular(self.L.transpose(-1, -2), C, upper=True)


class _Globals:
    """The global parameters by name, as elements of their trained table."""

    def __init__(self, names: Tuple[str, ...], table: Tensor):
        self.index = {name: k for k, name in enumerate(names)}
        self.table = table

    def __getitem__(self, name: str) -> Tensor:
        return self.table[self.index[name]]

    def get(self, name: str, default: float = 0.0):
        return self.table[self.index[name]] if name in self.index else default


def _vector(values, dtype, device) -> Tensor:
    """A tensor of numbers or of zero-dimensional tensors (which keep their gradients)."""
    if any(isinstance(v, Tensor) for v in values):
        return torch.stack([torch.as_tensor(v, dtype=dtype, device=device) for v in values])
    return torch.tensor(values, dtype=dtype, device=device)


def _stacked(M: Tensor) -> Tensor:
    """Integral matrices ``(B, n_ops, mu, nu)`` as ``(B, nu, mu, n_ops)``: the operators of an element
    side by side, the elements of a column in a row."""
    return M.permute(0, 3, 2, 1).contiguous()


@functools.lru_cache(maxsize=None)
def _origin_shift(dtype: torch.dtype, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Coefficients ``(3, 100)`` and ``(6, 100)`` of the components of ``A - B`` and of their products
    (``xx, yy, zz, xy, xz, yz``) in the ``10 x 10`` map that moves the origin of the overlap, dipole
    and second-moment integrals from ``A`` to ``B``: ``<r_x - B_x> = <r_x - A_x> + (A - B)_x <1>`` and
    ``<(r - B)_x (r - B)_y> = <(r - A)_x (r - A)_y> + (A - B)_x <r_y - A_y> + (A - B)_y <r_x - A_x>
    + (A - B)_x (A - B)_y <1>``."""
    first = torch.zeros(3, 10, 10, dtype=dtype, device=device)
    second = torch.zeros(6, 10, 10, dtype=dtype, device=device)
    for k in range(3):
        first[k, 1 + k, 0] = 1.0
    for c, (x, y) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        first[x, 4 + c, 1 + y] += 1.0
        first[y, 4 + c, 1 + x] += 1.0
        second[c, 4 + c, 0] = 1.0
    return first.view(3, 100), second.view(6, 100)


def _anderson(xs: List[Tensor], rs: List[Tensor], beta: float) -> Tensor:
    """Anderson (Pulay) mixing of the iterates ``xs`` with residuals ``rs``."""
    m = len(xs)
    if m == 1:
        return xs[0] + beta * rs[0]
    R = torch.stack(rs)
    A = R @ R.T
    A = A + torch.eye(m, dtype=A.dtype, device=A.device) * 1e-14 * A.diagonal().max()
    ones = torch.ones(m, dtype=A.dtype, device=A.device)
    try:
        w = torch.linalg.solve(A, ones)
    except RuntimeError:
        return xs[-1] + beta * rs[-1]
    c = w / w.sum()
    X = torch.stack(xs)
    return (c[:, None] * (X + beta * R)).sum(0)


def _fermi(eps: Tensor, n_alpha: Tensor, n_beta: Tensor, kt: float, n_ao: List[int]) -> Tuple[Tensor, Tensor]:
    """Spin-summed Fermi occupations and ``T S_el`` (hartree) per structure."""
    f, _ = _fermi_channels(eps, n_alpha, n_beta, kt, n_ao)
    return f.sum(1), _electronic_entropy(f, kt)


def _electronic_entropy(f: Tensor, kt: float) -> Tensor:
    """``T S_el`` (hartree) per structure of the channel occupations ``(B, 2, nb)``."""
    return -kt * (torch.special.xlogy(f, f) + torch.special.xlogy(1.0 - f, 1.0 - f)).sum((-1, -2))


def _fermi_channels(eps: Tensor, n_alpha: Tensor, n_beta: Tensor, kt: float, n_ao: List[int]) -> Tuple[Tensor, Tensor]:
    """Fermi occupations ``(B, 2, nb)`` and levels ``(B, 2)`` of the two spin channels of all
    structures at once; the Fermi levels by Newton steps on the host, bisecting where a step
    leaves the bracket."""
    nb = eps.shape[-1]
    e_host = eps.detach().cpu().numpy()
    n_valid = np.asarray(n_ao)
    valid = np.arange(nb)[None, :] < n_valid[:, None]
    nel = np.stack([n_alpha.detach().cpu().numpy(), n_beta.detach().cpu().numpy()], 1)       # (B, 2)
    if kt <= 0:
        f = ((np.arange(nb)[None, None, :] < nel[:, :, None]) & valid[:, None, :]).astype(np.float64)
        return (torch.as_tensor(f, dtype=eps.dtype, device=eps.device),
                torch.zeros(eps.shape[0], 2, dtype=eps.dtype, device=eps.device))
    e = np.where(valid, e_host, np.inf)[:, None, :]
    lo = np.repeat((np.where(valid, e_host, np.inf).min(-1) - 1.0)[:, None], 2, 1)
    hi = np.repeat((np.where(valid, e_host, -np.inf).max(-1) + 1.0)[:, None], 2, 1)
    # start between the highest occupied and the lowest empty level (the levels are ascending)
    top = np.clip(np.ceil(nel).astype(np.int64), 1, n_valid[:, None])
    rows = np.arange(e_host.shape[0])[:, None]
    below = e_host[rows, top - 1]
    above = e_host[rows, np.minimum(top, n_valid[:, None] - 1)]
    mid = 0.5 * (below + above)
    target = 4.0 * np.finfo(np.float64).eps * np.maximum(nel, 1.0)
    for _ in range(200):
        f = 0.5 * (1.0 - np.tanh(0.5 * (e - mid[:, :, None]) / kt))
        res = f.sum(-1) - nel
        done = (np.abs(res) <= target) | (nel <= 0)
        if done.all():
            break
        over = res > 0
        hi = np.where(over, mid, hi)
        lo = np.where(over, lo, mid)
        slope = (f * (1.0 - f)).sum(-1) / kt
        step = mid - res / np.maximum(slope, 1e-300)
        inside = (step > lo) & (step < hi)
        new = np.where(inside, step, 0.5 * (lo + hi))
        if (done | (new == mid)).all():
            break
        mid = np.where(done, mid, new)
    mu = torch.as_tensor(mid, dtype=eps.dtype, device=eps.device)
    has = torch.as_tensor(nel > 0, device=eps.device).to(eps.dtype)
    ee = torch.where(torch.as_tensor(valid, device=eps.device), eps, torch.full_like(eps, float("inf")))[:, None, :]
    return torch.sigmoid(-(ee - mu[:, :, None]) / kt) * has[:, :, None], mu


#: Levels closer than this (hartree) take the derivative limits of the divided differences.
_NEAR = 1e-8


def _divided1(lam: Tensor, v: Tensor, dv: Tensor) -> Tensor:
    """First divided differences ``[v_i, v_j]`` ``(B, n, n)`` of a function with values ``v`` and derivatives
    ``dv`` at the levels ``lam`` ``(B, n)`` (the mean derivative for close levels)."""
    d = lam[:, :, None] - lam[:, None, :]
    near = d.abs() < _NEAR
    return torch.where(near, 0.5 * (dv[:, :, None] + dv[:, None, :]),
                       (v[:, :, None] - v[:, None, :]) / torch.where(near, torch.ones_like(d), d))


def _divided2(lam: Tensor, v: Tensor, dv: Tensor, d2v: Tensor) -> Tensor:
    """Second divided differences ``[v_i, v_k, v_j]`` ``(B, n, n, n)`` (indices ``i, k, j``), with the limits of
    two and three close levels."""
    d1 = _divided1(lam, v, dv)
    dij = lam[:, :, None, None] - lam[:, None, None, :]
    dik = lam[:, :, None, None] - lam[:, None, :, None]
    far_ij, far_ik = dij.abs() >= _NEAR, dik.abs() >= _NEAR
    general = (d1[:, :, :, None] - d1[:, None, :, :]) / torch.where(far_ij, dij, torch.ones_like(dij))
    pair = (dv[:, :, None, None] - d1[:, :, :, None]) / torch.where(far_ik, dik, torch.ones_like(dik))
    triple = (d2v[:, :, None, None] + d2v[:, None, :, None] + d2v[:, None, None, :]) / 6.0
    return torch.where(far_ij, general, torch.where(far_ik, pair, triple))


def _fermi_derivatives(fs: Tensor, kt: float) -> Tuple[Tensor, Tensor]:
    """The first and second derivatives of Fermi occupations ``fs`` in their level, at a fixed Fermi level."""
    if kt <= 0:
        return torch.zeros_like(fs), torch.zeros_like(fs)
    d1 = -fs * (1.0 - fs) / kt
    return d1, -d1 * (1.0 - 2.0 * fs) / kt


def _sym(M: Tensor) -> Tensor:
    return 0.5 * (M + M.transpose(-1, -2))


def _spectral_gradient(eps: Tensor, U: Tensor, f: Tensor, level: Tensor, kt: float, gP: Optional[Tensor],
                       gW: Optional[Tensor], gts: Optional[Tensor]) -> Tensor:
    """The gradient in ``F`` of ``<gP, P> + <gW, W> + gts T S_el`` (see :class:`_SpectralDensity`)."""
    Ut = U.transpose(-1, -2)
    A = None if gP is None else Ut @ _sym(gP) @ U
    B = None if gW is None else Ut @ _sym(gW) @ U
    gap = eps[:, :, None] - eps[:, None, :]
    near = gap.abs() < _NEAR
    gap = torch.where(near, torch.ones_like(gap), gap)

    def quotient(a: Tensor, slope: Tensor) -> Tensor:
        equal = 0.5 * (slope[:, :, None] + slope[:, None, :])
        return torch.where(near, equal, (a[:, :, None] - a[:, None, :]) / gap)

    M = torch.zeros_like(gap)
    diagonal = torch.zeros_like(eps)
    for s in range(2):
        fs = f[:, s]
        slope = -fs * (1.0 - fs) / kt if kt > 0 else torch.zeros_like(fs)      # df / d eps at a fixed level
        total = slope.sum(-1, keepdim=True)
        share = torch.where(total != 0, slope / torch.where(total != 0, total, torch.ones_like(total)),
                            torch.zeros_like(slope))
        moved = torch.zeros_like(total[:, 0])                                   # gradient in the Fermi level
        if A is not None:
            M = M + quotient(fs, slope) * A
            moved = moved - (A.diagonal(dim1=-2, dim2=-1) * slope).sum(-1)
        if B is not None:
            M = M + quotient(fs * eps, fs + eps * slope) * B
            moved = moved - (B.diagonal(dim1=-2, dim2=-1) * eps * slope).sum(-1)
        if gts is not None:
            entropy = (eps - level[:, s, None]) * slope
            diagonal = diagonal + gts[:, None] * entropy
            moved = moved - gts * entropy.sum(-1)
        diagonal = diagonal + moved[:, None] * share
    return _sym(U @ (M + torch.diag_embed(diagonal)) @ Ut)


def _spectral_second(eps: Tensor, U: Tensor, f: Tensor, level: Tensor, kt: float, gP: Optional[Tensor],
                     gW: Optional[Tensor], gts: Optional[Tensor], gG: Tensor):
    """The derivatives of ``<gG, G>``, ``G`` the gradient of :func:`_spectral_gradient`: in ``F`` (the second
    derivative of ``L = <gP, P> + <gW, W> + gts T S_el`` along ``gG``) and in ``gP``, ``gW``, ``gts`` (the
    first-order changes of ``P``, ``W`` and ``T S_el`` along ``gG``).

    Each spin channel contributes ``L_s(F, mu_s(F))`` with the matrix functions ``a = f(e - mu)``,
    ``b = e f(e - mu)``, the trace of ``t = -kT H(f(e - mu))`` and the Fermi level fixed by
    ``tr f(F - mu) = N``. In the eigenbasis the second derivatives take the second divided differences of
    ``a`` and ``b``, the first of ``t'``, ``df / dmu`` and ``f'``, and the first and second derivatives of
    ``mu``.
    """
    Ut = U.transpose(-1, -2)
    zero = torch.zeros_like(U)
    A = zero if gP is None else Ut @ _sym(gP) @ U
    B = zero if gW is None else Ut @ _sym(gW) @ U
    g = torch.zeros_like(eps[:, 0]) if gts is None else gts.to(eps.dtype)
    E = Ut @ _sym(gG) @ U
    Ad, Bd, Ed = (M.diagonal(dim1=-2, dim2=-1) for M in (A, B, E))
    G = torch.zeros_like(U)
    dP, dW = torch.zeros_like(U), torch.zeros_like(U)
    dT = torch.zeros_like(eps[:, 0])
    for s in range(2):
        fs = f[:, s]
        f1, f2 = _fermi_derivatives(fs, kt)
        x = eps - level[:, s, None]
        tp, tpp = x * f1, f1 + x * f2                                           # t' and t''
        S1 = f1.sum(-1)
        ok = S1 != 0
        inv = torch.where(ok, 1.0 / torch.where(ok, S1, torch.ones_like(S1)), torch.zeros_like(S1))
        share = f1 * inv[:, None]
        dmu = (f1 * Ed).sum(-1) * inv                                           # d mu along gG
        # the first-order changes: the gradients in gP, gW and gts
        a1 = _divided1(eps, fs, f1)
        b1 = _divided1(eps, eps * fs, fs + eps * f1)
        dP = dP + a1 * E - torch.diag_embed(f1 * dmu[:, None])
        dW = dW + b1 * E - torch.diag_embed(eps * f1 * dmu[:, None])
        dT = dT + (tp * (Ed - dmu[:, None])).sum(-1)
        # the second derivative at the fixed Fermi level
        a2 = _divided2(eps, fs, f1, f2)
        b2 = _divided2(eps, eps * fs, fs + eps * f1, 2.0 * f1 + eps * f2)
        X = torch.einsum("bikj,bik,bij->bkj", a2, E, A) + torch.einsum("bikj,bik,bij->bkj", b2, E, B)
        G = G + X + X.transpose(-1, -2) + g[:, None, None] * _divided1(eps, tp, tpp) * E
        # the terms of the moving Fermi level
        K = (_divided1(eps, -f1, -f2) * A + _divided1(eps, -eps * f1, -(f1 + eps * f2)) * B
             - torch.diag_embed(g[:, None] * tpp))
        kE = (K * E).sum((-1, -2))
        L_mu = -(f1 * Ad + eps * f1 * Bd + g[:, None] * tp).sum(-1)
        L_mumu = (f2 * Ad + eps * f2 * Bd + g[:, None] * tpp).sum(-1)
        c_fmu = -(f2 * Ed).sum(-1)
        c_mumu = f2.sum(-1)
        d2mu = (_divided1(eps, f1, f2) * E + torch.diag_embed((c_fmu + c_mumu * dmu)[:, None] * share
                                                               - dmu[:, None] * f2)) * inv[:, None, None]
        moving = (dmu[:, None, None] * K + torch.diag_embed((kE + L_mumu * dmu)[:, None] * share)
                  + L_mu[:, None, None] * d2mu)
        G = G + torch.where(ok[:, None, None], moving, torch.zeros_like(moving))
    rot = lambda M: U @ M @ Ut                                                  # noqa: E731
    return _sym(rot(G)), rot(_sym(dP)), rot(_sym(dW)), dT


class _SpectralDensity(torch.autograd.Function):
    """Density ``U f U^T``, energy-weighted density ``U (f eps) U^T`` and ``T S_el`` of a symmetric
    matrix ``F`` with eigenvalues ``eps``, eigenvectors ``U`` and Fermi occupations ``f`` (two spin
    channels, each with its own level).

    The backward pass is first-order perturbation theory: a spectral function ``a(eps)`` responds to
    ``dF`` with ``U [D * (U^T dF U)] U^T``, ``D_ij = (a_i - a_j) / (eps_i - eps_j)`` (its derivative
    for equal levels), and the Fermi level moves with the levels so that the electron number stays,
    ``d mu = sum_k f'_k d eps_k / sum_k f'_k``. Under ``create_graph`` it is :class:`_SpectralResponse`,
    differentiable once more.
    """

    @staticmethod
    def forward(ctx, F, eps, U, f, level, kt):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(F, eps, U, f, level)
        ctx.kt = kt
        occ = f.sum(1)
        Ut = U.transpose(-1, -2)
        return (U * occ[:, None, :]) @ Ut, (U * (occ * eps)[:, None, :]) @ Ut, _electronic_entropy(f, kt)

    @staticmethod
    def backward(ctx, gP, gW, gts):
        F, eps, U, f, level = ctx.saved_tensors
        if torch.is_grad_enabled():
            g = _SpectralResponse.apply(F, gP, gW, gts, eps, U, f, level, ctx.kt)
        else:
            g = _spectral_gradient(eps, U, f, level, ctx.kt, gP, gW, gts)
        return g, None, None, None, None, None


class _SpectralResponse(torch.autograd.Function):
    """The gradient of :class:`_SpectralDensity` as a function of ``F`` and the incoming gradients, whose
    backward pass is second-order perturbation theory (:func:`_spectral_second`)."""

    @staticmethod
    def forward(ctx, F, gP, gW, gts, eps, U, f, level, kt):
        ctx.save_for_backward(eps, U, f, level, gP, gW, gts)
        ctx.kt = kt
        return _spectral_gradient(eps, U, f, level, kt, gP, gW, gts)

    @staticmethod
    @once_differentiable
    def backward(ctx, gG):
        eps, U, f, level, gP, gW, gts = ctx.saved_tensors
        dF, dP, dW, dT = _spectral_second(eps, U, f, level, ctx.kt, gP, gW, gts, gG)
        need = ctx.needs_input_grad
        return (dF if need[0] else None, dP if need[1] else None, dW if need[2] else None,
                dT if need[3] else None, None, None, None, None, None)


def _all_pairs(batch: Tensor) -> Tuple[Tensor, Tensor]:
    """Atom pairs ``i < j`` within each structure."""
    n = batch.shape[0]
    i, j = torch.triu_indices(n, n, 1, device=batch.device)
    keep = batch[i] == batch[j]
    return i[keep], j[keep]


@register_model("gfn2_xtb")
class GFN2(GFN):
    """GFN2-xTB (Bannwarth, Ehlert and Grimme, J. Chem. Theory Comput. 15, 1652 (2019)); see :class:`GFN`."""

    method = "gfn2"
    scc_terms = ("gamma", "third_order", "anisotropic_es", "anisotropic_xc", "dispersion_scc")
    reported_terms = ("isotropic_es", "anisotropic_es", "anisotropic_xc", "dispersion_scc")
    shell_third_order = True
    multipoles = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from xnn.common.models.d4 import DFTD4
        g = self.par.globals
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            # as in the reference program: pairs within 50 bohr, the three-body term within 25 bohr with
            # each edge switched off over the last 0.05 bohr
            self.d4 = DFTD4(s6=1.0, s8=g["s8"], a1=g["a1"], a2=g["a2"], s9=g["s9"], alp=16.0,
                            reference_charges="gfn2", cutoff_pair=50.0 * BOHR, cutoff_triple=25.0 * BOHR,
                            switch_width_triple=0.05 * BOHR)
        finally:
            torch.set_default_dtype(previous)
        self.radii = [0.0] + [GFN2_RADII[z] / BOHR for z in range(1, 87)]

    def reference_occupations(self, z: int) -> List[float]:
        return gfn2_reference_occupations(self.par, z)

    def coordination_numbers(self, z: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        """The GFN2 coordination number ``CN'`` (double-exponential counting function).

        ``f = 1 / (1 + exp(-10 (r0 / R - 1))) / (1 + exp(-20 ((r0 + 2) / R - 1)))`` with
        ``r0 = 4/3 (R_A + R_B)`` and the 2 bohr added after the scaling, as the xtb program
        evaluates it (the paper's eq 18 writes the shift inside the 4/3 factor).
        """
        n = z.shape[0]
        rc = torch.tensor(self.rcov, dtype=pos.dtype, device=pos.device)[z]
        i, j = _all_pairs(batch)
        r = (pos.index_select(0, i) - pos.index_select(0, j)).norm(dim=-1)
        r0 = 4.0 / 3.0 * (rc[i] + rc[j])
        f = torch.sigmoid(10.0 * (r0 / r - 1.0)) * torch.sigmoid(20.0 * ((r0 + 2.0) / r - 1.0))
        return torch.zeros(n, dtype=pos.dtype, device=pos.device).index_add(0, i, f).index_add(0, j, f)

    def shell_levels(self, z: List[int], basis: _Basis, cn: Tensor) -> Tensor:
        lev = self.table("LEV")[basis.shell_z, basis.shell_index]
        kcn = 0.1 * self.table("KCN")[basis.shell_z, basis.shell_l_t]
        return (lev - kcn * cn.index_select(0, basis.shell_atom)) / AUTOEV

    def shell_pair_factor(self, basis: _Basis, pos: Tensor) -> Tensor:
        g = self._g()
        dt, dev = pos.dtype, pos.device
        ks, kp, kd = g["ks"], g["kp"], g["kd"]
        K = _vector([ks, 0.5 * (ks + kp), g["ksd"], 0.5 * (ks + kp), kp, g["kpd"], g["ksd"], g["kpd"], kd],
                    dt, dev).view(3, 3)
        zeta, kpoly = self._shell_constants(basis, dt, dev)
        ls = basis.shell_l_t
        X = self._electronegativity_factor(basis, dt, dev)
        Y = torch.sqrt(2.0 * torch.sqrt(zeta[:, None] * zeta[None, :]) / (zeta[:, None] + zeta[None, :]))
        return K[ls[:, None], ls[None, :]] * X * Y * self._distance_polynomial(basis, pos, kpoly)

    def scc_context(self, z, pos, basis, cn, batch, n_struct) -> dict:
        g = self._g()
        dtype, device = pos.dtype, pos.device
        zl = z.tolist()
        sz, sl = basis.shell_z, basis.shell_l_t
        k3 = _vector([g["gam3s"], g["gam3p"], g["gam3d1"]], dtype, device)
        kappa = torch.where(sl > 0, 0.1 * self.table("LPAR")[sz, sl], torch.zeros_like(sl, dtype=dtype))
        eta = self.table("GAM")[sz] * (1.0 + kappa)
        gam3 = 0.1 * self.table("GAM3")[sz] * k3[sl]
        sa = basis.shell_atom
        ps = pos.index_select(0, sa)
        d = ps[:, None, :] - ps[None, :, :]
        r2 = (d * d).sum(-1)
        eta_ab = 0.5 * (eta[:, None] + eta[None, :])
        gamma = 1.0 / torch.sqrt(r2 + eta_ab ** -2)
        ss = torch.tensor(basis.shell_struct, device=device)
        gamma = torch.where(ss[:, None] == ss[None, :], gamma, torch.zeros_like(gamma))
        # AES damping radii from the coordination numbers
        r0 = torch.tensor([GFN2_AES_RADII.get(za, (0.0, 0.0))[0] for za in zl], dtype=dtype, device=device)
        listed = torch.tensor([za in GFN2_AES_RADII for za in zl], device=device)
        r0 = torch.where(listed, r0, g["aesrmax"] * torch.ones_like(r0))
        nval = torch.tensor([GFN2_AES_RADII.get(za, (0.0, 0.0))[1] for za in zl], dtype=dtype, device=device)
        rad = r0 + (g["aesrmax"] - r0) * torch.sigmoid(g["aesexp"] * (cn - nval - g["aesshift"]))
        i, j = _all_pairs(batch)
        rij = pos.index_select(0, j) - pos.index_select(0, i)
        r = rij.norm(dim=-1)
        rab = 0.5 * (rad.index_select(0, i) + rad.index_select(0, j))
        f3 = 1.0 / (r ** 3 * (1.0 + 6.0 * (rab / r) ** g["aesdmp3"]))
        f5 = 1.0 / (r ** 5 * (1.0 + 6.0 * (rab / r) ** g["aesdmp5"]))
        dpol, qpol = 0.01 * self.table("DPOL")[z], 0.01 * self.table("QPOL")[z]
        # D4: covalent coordination numbers and pairs (bohr)
        d4 = self.d4.to(dtype)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        rr = torch.cat([r, r])
        cn_d4, _ = d4.coordination_numbers(z, ei, rr, z.shape[0])
        n = z.shape[0]
        damping = torch.zeros(n, n, dtype=dtype, device=device)
        inside = rr < d4.cutoff_pair / d4.bohr
        damping = damping.index_put((ei[1], ei[0]), torch.where(inside, d4.pair_damping(z, ei, rr),
                                                                torch.zeros_like(rr)))
        d4_weights = d4.gaussian_weights(z, cn_d4)                           # (N, 7)
        d4_refalpha = d4.refalpha[:, :, z].permute(2, 1, 0)                 # (N, 7, 23)
        d4_zeff = d4.zeff[z][:, None]
        return {"d4_damping": damping, "d4_weights": d4_weights, "d4_refalpha": d4_refalpha, "gamma": gamma,
                "d4_zeff": d4_zeff, "d4_hardness": (d4.hardness[z] * d4.gc)[:, None],
                "d4_qref": d4.refq[:, z].t() + d4_zeff, "d4_cp": d4.cp_weights,
                "gam3": gam3,
                "i": i, "j": j, "rij": rij, "r": r, "f3": f3, "f5": f5, "dpol": dpol, "qpol": qpol, "z": z,
                "cn_d4": cn_d4, "ei": ei, "rr": rr, "shell_atom": sa, "shell_struct": ss, "batch": batch,
                "n_struct": n_struct}

    def _terms(self, q_shell, mu, theta, ctx) -> Dict[str, Tensor]:
        """Per-structure charge-dependent energies (hartree)."""
        n_struct, batch, ss = ctx["n_struct"], ctx["batch"], ctx["shell_struct"]
        dtype = q_shell.dtype

        def per(values, index):
            return torch.zeros(n_struct, dtype=dtype, device=q_shell.device).index_add(0, index, values)

        e_gamma = per(0.5 * q_shell * (ctx["gamma"] @ q_shell), ss)
        e_third = per(q_shell ** 3 * ctx["gam3"] / 3.0, ss)
        n_atoms = batch.shape[0]
        q = torch.zeros(n_atoms, dtype=dtype, device=q_shell.device).index_add(0, ctx["shell_atom"], q_shell)
        i, j, R = ctx["i"], ctx["j"], ctx["rij"]
        Th = _traceless(theta)                                       # (N, 3, 3)
        r2 = (R * R).sum(-1)
        mi_r, mj_r = (mu.index_select(0, i) * R).sum(-1), (mu.index_select(0, j) * R).sum(-1)
        Rc = R[:, :, None]
        rtr_i = ((Th.index_select(0, i) @ Rc).squeeze(-1) * R).sum(-1)
        rtr_j = ((Th.index_select(0, j) @ Rc).squeeze(-1) * R).sum(-1)
        qi, qj = q.index_select(0, i), q.index_select(0, j)
        mm = (mu.index_select(0, i) * mu.index_select(0, j)).sum(-1)
        e_pair = (ctx["f3"] * (qj * mi_r - qi * mj_r)
                  + ctx["f5"] * (qi * rtr_j + qj * rtr_i + mm * r2 - 3.0 * mi_r * mj_r))
        e_aes = per(e_pair, batch[i])
        e_axc = per(ctx["dpol"] * (mu * mu).sum(-1) + ctx["qpol"] * (Th * Th).sum((-1, -2)), batch)
        e_disp = per(self._d4_atoms(q, ctx), batch)
        return {"isotropic_es": e_gamma + e_third, "gamma": e_gamma, "third_order": e_third,
                "anisotropic_es": e_aes, "anisotropic_xc": e_axc, "dispersion_scc": e_disp}

    def potentials(self, q_shell, mu, theta, ctx):
        """Closed-form potentials of the isotropic, third-order, AES, AXC and D4 terms.

        The anisotropic electrostatics is bilinear in the atomic charges, dipoles and traceless
        quadrupoles ``Th``,

            ``E = sum_a mu_a T1_a q + sum_ab q T2_ab Th_ab + 1/2 sum_a mu_a T0 mu_a - 3/2 sum_ab mu_a T2_ab mu_b``,

        with the atom-pair matrices ``T1_a = f3 R_a``, ``T2_ab = f5 R_a R_b`` and ``T0 = f5 R^2``
        (``R = R_j - R_i``), so that its potentials are matrix products.
        """
        n_atoms, sa = ctx["batch"].shape[0], ctx["shell_atom"]
        q = torch.zeros(n_atoms, dtype=q_shell.dtype, device=q_shell.device).index_add(0, sa, q_shell)
        v_shell = ctx["gamma"] @ q_shell + q_shell * q_shell * ctx["gam3"]
        if "aes" not in ctx:
            ctx["aes"] = self._aes_matrices(ctx, n_atoms)
        tr = theta[:, :3].sum(-1, keepdim=True)
        th = torch.cat([1.5 * theta[:, :3] - 0.5 * tr, 1.5 * theta[:, 3:]], 1)       # xx, yy, zz, xy, xz, yz
        # products of the ten matrices (T0, T1_a, T2_c) with q, mu and Th (off-diagonal Th counted twice)
        out = torch.matmul(ctx["aes"], torch.cat([q[:, None], mu, th[:, :3], 2.0 * th[:, 3:]], 1))
        v_q = out[4:, :, 4:].diagonal(dim1=0, dim2=2).sum(-1) - out[1:4, :, 1:4].diagonal(dim1=0, dim2=2).sum(-1)
        t2mu = torch.stack([out[4 + c, :, 1 + b] for c, b in ((0, 0), (3, 1), (4, 2), (3, 0), (1, 1), (5, 2),
                                                               (4, 0), (5, 1), (2, 2))], -1).view(n_atoms, 3, 3)
        v_mu = out[1:4, :, 0].t() + out[0, :, 1:4] - 3.0 * t2mu.sum(-1) + 2.0 * ctx["dpol"][:, None] * mu
        # derivative with respect to the full traceless tensor, then to the six second moments
        G = out[4:, :, 0].t() + 2.0 * ctx["qpol"][:, None] * th
        v_theta = torch.cat([1.5 * G[:, :3] - 0.5 * G[:, :3].sum(-1, keepdim=True), 3.0 * G[:, 3:]], 1)
        v_shell = v_shell + (v_q + self._d4_potential(q, ctx))[sa]
        return v_shell, v_mu, v_theta

    @staticmethod
    def _aes_matrices(ctx: dict, n_atoms: int) -> Tensor:
        """The matrices ``T0``, ``T1_x..z`` and ``T2_xx, yy, zz, xy, xz, yz`` of the anisotropic
        electrostatics, ``(10, N, N)``."""
        i, j, R, f3, f5 = ctx["i"], ctx["j"], ctx["rij"], ctx["f3"], ctx["f5"]
        x, y, z = R.unbind(-1)
        rows = torch.stack([f5 * (R * R).sum(-1), f3 * x, f3 * y, f3 * z, f5 * x * x, f5 * y * y, f5 * z * z,
                            f5 * x * y, f5 * x * z, f5 * y * z])
        sign = rows.new_ones(10, 1)
        sign[1:4] = -1.0                                             # R changes sign with the order of the pair
        T = rows.new_zeros(10, n_atoms, n_atoms)
        T[:, i, j] = rows
        T[:, j, i] = rows * sign
        return T

    def _compiled_three_body(self, pos: Tensor) -> bool:
        """Whether the three-body term takes the compiled CPU tiles: the fast path is on, the
        tensors are on a CPU and the compiler works here."""
        if self.use_fast is False or pos.is_cuda:
            return False
        from xnn.common.models.dispersion_fast import atm_cpu
        return atm_cpu.available()

    def _three_body_reference(self, z: Tensor, ei: Tensor, vec: Tensor, cn: Tensor, pos: Tensor, batch: Tensor,
                              n_struct: int) -> Tensor:
        """Three-body energies per atom (hartree) from the edges within the cutoff and the D4
        coordination numbers, by the reference's functions (differentiable at any order)."""
        d4 = self.d4
        a0 = d4.dynamic_polarizabilities(z, d4.reference_weights(z, cn, torch.zeros_like(cn)))
        cell = torch.zeros(n_struct, 3, 3, dtype=vec.dtype, device=vec.device)
        return d4._three_body_chunked(z, pos, ei, vec, vec.norm(dim=-1), a0, cell, batch, z.shape[0])

    def _three_body_energies(self, z: Tensor, ei: Tensor, vec: Tensor, cn: Tensor) -> Tensor:
        """Three-body energies per atom (hartree) on the compiled tiles, from the edges within the cutoff
        and the D4 coordination numbers."""
        d4 = self.d4
        a0 = d4.dynamic_polarizabilities(z, d4.reference_weights(z, cn, torch.zeros_like(cn)))
        c6 = (3.0 / math.pi) * (a0 * d4.cp_weights) @ a0.t()
        return d4._three_body_cpu(z, ei, vec, vec.norm(dim=-1), c6, z.shape[0])

    def _three_body_direct(self, z: Tensor, ei: Tensor, vec: Tensor, cn: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """:meth:`_three_body_energies` together with the gradient of their sum in the edge vectors and
        the coordination numbers, formed in the same pass over the triangles."""
        from xnn.common.models.dispersion_fast.atm_cpu import Plan, three_body_direct
        from xnn.common.models.dispersion_fast.d4_direct import DIRECT_TILE, _alpha
        d4 = self.d4
        _, _, _, a0, da0 = _alpha(d4, z, cn, torch.zeros_like(cn), True)
        c6 = (3.0 / math.pi) * (a0 * d4.cp_weights) @ a0.t()
        dc6 = (3.0 / math.pi) * (da0 * d4.cp_weights) @ a0.t()
        plan = Plan(ei, z.shape[0], tile=DIRECT_TILE)
        v = vec.index_select(0, plan.order).contiguous()
        e3, g, g_cn = three_body_direct(z, plan, v, v.norm(dim=-1), c6, dc6, dc6.t().contiguous(),
                                        d4.pair_radius_table(), float(d4.s9), d4.alp / 3.0,
                                        d4.cutoff_triple / d4.bohr, d4.switch_width_triple / d4.bohr, z.shape[0])
        return e3, torch.zeros_like(vec).index_copy(0, plan.order, g), g_cn

    def _d4_atoms(self, q: Tensor, ctx: dict) -> Tensor:
        """Charge-dependent D4 two-body energy per atom (hartree): ``-1/2 C6_AB`` times the damping of every pair."""
        d4 = self.d4
        alpha = ((ctx["d4_weights"] * d4.charge_scaling(ctx["z"], q))[:, :, None] * ctx["d4_refalpha"]).sum(1)
        weighted = alpha * d4.cp_weights
        return -0.5 * (3.0 / math.pi) * (weighted * (ctx["d4_damping"] @ alpha)).sum(-1)

    def _d4_potential(self, q: Tensor, ctx: dict) -> Tensor:
        """Derivative of the charge-dependent D4 energy with respect to the atomic charges (from the
        tables of the context alone)."""
        w = ctx["d4_weights"]
        zeta, slope = charge_scaling_terms(q, ctx["d4_zeff"], ctx["d4_hardness"], ctx["d4_qref"], self.d4.ga)
        both = torch.bmm(torch.stack([w * zeta, w * slope], 1), ctx["d4_refalpha"])       # (N, 2, 23)
        alpha, dalpha = both[:, 0], both[:, 1]
        return -(3.0 / math.pi) * (ctx["d4_cp"] * dalpha * (ctx["d4_damping"] @ alpha)).sum(-1)

    def classical_energy(self, z, pos, batch, n_struct) -> Dict[str, Tensor]:
        g = self._g()
        i, j = _all_pairs(batch)
        alpha, zeff = self.table("REPA")[z], self.table("REPB")[z]
        light = z <= 2
        one = torch.ones_like(alpha[i])
        kexp = torch.where(light[i] & light[j], g["kexplight"] * one, g["kexp"] * one)
        r = (pos.index_select(0, j) - pos.index_select(0, i)).norm(dim=-1)
        e_rep = zeff[i] * zeff[j] / r * torch.exp(-torch.sqrt(alpha[i] * alpha[j]) * r ** kexp)
        rep = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device).index_add(0, batch[i], e_rep)
        # D4 three-body term with neutral charges
        d4 = self.d4.to(pos.dtype)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        dij = pos.index_select(0, j) - pos.index_select(0, i)
        vec = torch.cat([dij, -dij])
        rr = vec.norm(dim=-1)
        cn_d4, _ = d4.coordination_numbers(z, ei, rr, z.shape[0])
        keep = torch.nonzero(rr <= d4.cutoff_triple / d4.bohr).squeeze(1)
        ei, vec = ei.index_select(1, keep), vec.index_select(0, keep)
        if not self._compiled_three_body(pos):
            e3 = self._three_body_reference(z, ei, vec, cn_d4, pos, batch, n_struct)
        elif vec.requires_grad and torch.is_grad_enabled():
            e3 = _ThreeBody.apply(vec, cn_d4, self, z, ei, pos.detach(), batch, n_struct)
        else:
            e3 = self._three_body_energies(z, ei, vec, cn_d4)
        disp3 = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device).index_add(0, batch, e3)
        return {"total": rep + disp3, "repulsion": rep, "dispersion_3body": disp3}


class _ThreeBody(torch.autograd.Function):
    """The three-body energies per atom of :meth:`GFN2._three_body_energies` with the gradient of
    their sum formed along with them. A gradient that is differentiated again is linear in the
    incoming weight; its dependence on the geometry is carried by :class:`_ThreeBodyCurvature`."""

    @staticmethod
    def forward(ctx, vec, cn, model, z, ei, pos, batch, n_struct):
        e3, g_vec, g_cn = model._three_body_direct(z, ei, vec, cn)
        ctx.save_for_backward(vec, cn, g_vec, g_cn)
        ctx.constants = (model, z, ei, pos, batch, n_struct)
        return e3

    @staticmethod
    def backward(ctx, grad):
        vec, cn, g_vec, g_cn = ctx.saved_tensors
        model, z, ei, pos, batch, n_struct = ctx.constants
        nothing = (None,) * 6
        if bool((grad == grad[0]).all()):
            if not torch.is_grad_enabled():
                return (g_vec * grad[0], g_cn * grad[0]) + nothing
            c_vec, c_cn = _ThreeBodyCurvature.apply(vec, cn, grad[0].detach(), *ctx.constants)
            return (g_vec * grad[0] + c_vec, g_cn * grad[0] + c_cn) + nothing
        # weights that differ between the atoms: differentiate the weighted energies
        if torch.is_grad_enabled():
            e3 = model._three_body_reference(z, ei, vec, cn, pos, batch, n_struct)
            return torch.autograd.grad((e3 * grad).sum(), (vec, cn), create_graph=True) + nothing
        with torch.enable_grad():
            v, c = vec.detach().requires_grad_(True), cn.detach().requires_grad_(True)
            g = torch.autograd.grad((model._three_body_energies(z, ei, v, c) * grad).sum(), (v, c))
        return g + nothing


class _ThreeBodyCurvature(torch.autograd.Function):
    """Zeros that carry the dependence on the geometry of the gradient of :class:`_ThreeBody`
    (times a uniform weight): the backward is the second derivative of the summed energies along
    the incoming displacements, from the reference's functions."""

    @staticmethod
    def forward(ctx, vec, cn, weight, model, z, ei, pos, batch, n_struct):
        ctx.save_for_backward(vec, cn, weight)
        ctx.constants = (model, z, ei, pos, batch, n_struct)
        return torch.zeros_like(vec), torch.zeros_like(cn)

    @staticmethod
    @once_differentiable
    def backward(ctx, d_vec, d_cn):
        vec, cn, weight = ctx.saved_tensors
        model, z, ei, pos, batch, n_struct = ctx.constants
        with torch.enable_grad():
            v, c = vec.detach().requires_grad_(True), cn.detach().requires_grad_(True)
            e3 = model._three_body_reference(z, ei, v, c, pos, batch, n_struct).sum() * weight
            g = torch.autograd.grad(e3, (v, c), create_graph=True)
            h = torch.autograd.grad((g[0] * d_vec).sum() + (g[1] * d_cn).sum(), (v, c))
        return h + (None,) * 7


def _traceless(theta: Tensor) -> Tensor:
    """``3/2 theta - 1/2 tr(theta)`` as full 3x3 matrices from ``xx, yy, zz, xy, xz, yz``."""
    xx, yy, zz, xy, xz, yz = theta.unbind(-1)
    m = torch.stack([torch.stack([xx, xy, xz], -1), torch.stack([xy, yy, yz], -1),
                     torch.stack([xz, yz, zz], -1)], -2)
    tr = xx + yy + zz
    eye = torch.eye(3, dtype=theta.dtype, device=theta.device)
    return 1.5 * m - 0.5 * tr[..., None, None] * eye


def _metal_pair_term(z: int) -> float:
    """GFN1-xTB enhancement of the Hamiltonian factor between two transition metals, per metal:
    ``K = 1.1`` for 3d-3d and 1.2 for 4d/5d/4f pairs, their mean for mixed pairs (Table 2 of
    the GFN1 paper)."""
    return 0.1 if z <= 30 else 0.2


def gfn1_reference_occupations(par: XTBParameters, z: int) -> List[float]:
    """Neutral-atom electrons per shell of GFN1-xTB, the canonical aufbau occupations of the
    paper: ``s^2 p^(n-2)`` for the main group, ``d^(n-2) s^2`` for transition metals (Cr
    ``3d^4 4s^2``) and lanthanides (f in core)."""
    shells = par.shells[z]
    nval = float(valence_electrons(z))
    ls = [l for _, l in shells]
    occ = [0.0] * len(shells)
    if z == 1:
        return [1.0, 0.0]
    if len(ls) == 1:
        return [nval]
    s_ = min(2.0, nval)
    occ[ls.index(0)] = s_
    if ls[0] == 2:
        occ[ls.index(2)] = nval - s_
    else:
        occ[ls.index(1)] = nval - s_
    return occ


#: Elements whose GFN1-xTB levels carry no coordination-number shift: the paper's eq 12 shifts the
#: levels of nonmetals only, and this is the set of metals of the reference program.
GFN1_NO_CN_SHIFT = frozenset([3, 4, 5, 11, 12, 13, 19, 20, 25, 26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43,
                              44, 45, 46, 47, 48, 49, 55, 56, *range(57, 81), 81])


def valence_electrons(z: int) -> int:
    """Valence electrons of the GFN methods: the group number for the main group and the
    transition metals, two for Zn, Cd and Hg, three for the lanthanides (f in core)."""
    if z <= 2:
        return z
    for start, end in ((3, 10), (11, 18)):
        if start <= z <= end:
            return z - start + 1
    starts = {19: 36, 37: 54, 55: 86}
    for start, end in starts.items():
        if start <= z <= end:
            col = z - start + 1
            if start == 55:
                if 57 <= z <= 71:
                    return 3
                if z > 71:
                    col = z - 71 + 3
            if col <= 11:
                return col
            if col == 12:
                return 2
            return col - 10
    raise ValueError(f"no valence count for Z = {z}")


#: Promoted reference configurations (``s``, ``p``) of light group 14 and 15 elements.
_PROMOTED = {6: (1.0, 3.0), 7: (1.5, 3.5), 14: (1.5, 2.5), 15: (1.5, 3.5), 32: (1.5, 2.5), 33: (1.5, 3.5)}


def gfn2_reference_occupations(par: XTBParameters, z: int) -> List[float]:
    """Neutral-atom electrons per shell of GFN2-xTB.

    Section 2.2.4 of the GFN2 paper: aufbau occupations for the main group, ``2s^1 2p^3``
    for C and ``ns^1.5 np^(x-1.5)`` for N, Si, P, Ge and As, ``d^(x-2) s^1 p^1`` for the
    transition metals (group 11 ``d^10 s^1``, group 3 and the lanthanides
    ``d^1 s^1 p^1``); polarization ``d`` shells empty."""
    shells = par.shells[z]
    nval = float(valence_electrons(z))
    ls = [l for _, l in shells]
    occ = [0.0] * len(shells)
    if len(ls) == 1:
        return [nval]
    if ls[0] == 2:                                                   # transition metal: (n-1)d ns np
        if nval >= 11:
            d, s_, p_ = 10.0, nval - 10.0, 0.0
        else:
            d, s_, p_ = nval - 2.0, 1.0, 1.0
        occ[ls.index(2)], occ[ls.index(0)], occ[ls.index(1)] = d, s_, p_
        return occ
    if z in _PROMOTED:
        s_, p_ = _PROMOTED[z]
    else:
        s_ = min(2.0, nval)
        p_ = nval - s_
    occ[ls.index(0)] = s_
    occ[ls.index(1)] = p_
    return occ


@register_model("gfn1_xtb")
class GFN1(GFN):
    """GFN1-xTB (Grimme, Bannwarth and Shushkov, J. Chem. Theory Comput. 13, 1989 (2017)); see :class:`GFN`."""

    method = "gfn1"
    scc_terms = ("gamma", "third_order")
    reported_terms = ("electrostatic",)
    #: Scale of the per-shell hardness factors ``LPARP``/``LPARD`` and of ``GAM3``.
    kappa_scale = 0.1
    gam3_scale = 0.1
    #: D3 reference set of the dispersion term.
    d3_references = "2010"
    #: Elements without coordination-number shifts of their levels.
    no_cn_shift = GFN1_NO_CN_SHIFT
    #: Acceptor elements of the halogen-bond correction.
    xb_acceptors = frozenset([7, 8, 15, 16])
    #: Pairs beyond this distance (bohr) do not count in the coordination number.
    cn_cutoff = 25.0
    host_basis = 170
    extra_trainable = ("EXP", "GLOBAL", "KPAIR")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from xnn.common.models.d3 import DFTD3
        g = self.par.globals
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            # pairs within 50 bohr and coordination numbers within 25 bohr, as in the reference program
            self.d3 = DFTD3(damping="bj", s6=1.0, s8=g["s8"], a1=g["a1"], a2=g["a2"], s9=g["s9"],
                            references=self.d3_references, cutoff_pair=50.0 * BOHR, cutoff_cn=25.0 * BOHR)
        finally:
            torch.set_default_dtype(previous)
        self.radii = [0.0] + [GFN2_RADII[z] / BOHR for z in range(1, 87)]

    def reference_occupations(self, z: int) -> List[float]:
        if z == 1:
            return [1.0, 0.0]
        return gfn1_reference_occupations(self.par, z)

    def coordination_numbers(self, z: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        """The D3 coordination number (exponential counting function)."""
        n = z.shape[0]
        rc = torch.tensor(self.rcov, dtype=pos.dtype, device=pos.device)[z]
        i, j = _all_pairs(batch)
        r = (pos.index_select(0, i) - pos.index_select(0, j)).norm(dim=-1)
        f = torch.sigmoid(16.0 * (4.0 / 3.0 * (rc[i] + rc[j]) / r - 1.0))
        f = torch.where(r <= self.cn_cutoff, f, torch.zeros_like(f))
        return torch.zeros(n, dtype=pos.dtype, device=pos.device).index_add(0, i, f).index_add(0, j, f)

    def _kcn(self, z: int) -> List[float]:
        """Relative CN slopes of the levels: ``cns``/``cnp`` for ``s``/``p``, ``cnd2`` for the valence
        ``d`` of transition metals, ``cnd1`` for polarization ``d`` (percent); zero for some metals."""
        g = self._g()
        if z in self.no_cn_shift:
            return [0.0] * len(self.par.shells[z])
        valence_d = self.par.shells[z][0][1] == 2
        out = []
        for _, l in self.par.shells[z]:
            key = ("cns", "cnp", "cnd2" if valence_d else "cnd1")[l]
            out.append(0.01 * g[key])
        return out

    def shell_levels(self, z: List[int], basis: _Basis, cn: Tensor) -> Tensor:
        lev = self.table("LEV")[basis.shell_z, basis.shell_index]
        kcn = _vector([k for za in z for k in self._kcn(int(za))], cn.dtype, cn.device)
        return lev * (1.0 + kcn * cn.index_select(0, basis.shell_atom)) / AUTOEV

    def shell_pair_factor(self, basis: _Basis, pos: Tensor) -> Tensor:
        g = self._g()
        dt, dev = pos.dtype, pos.device
        ls = basis.shell_l_t
        zs = basis.shell_z
        ks, kp, kd = g["ks"], g["kp"], g["kd"]
        kl = _vector([ks, kp, kd], dt, dev)
        K = _vector([ks, g["ksp"], 0.5 * (ks + kd), g["ksp"], kp, 0.5 * (kp + kd), 0.5 * (ks + kd), 0.5 * (kp + kd),
                     kd], dt, dev).view(3, 3)
        K = K[ls[:, None], ls[None, :]] * self._pair_table(dt, dev)[zs[:, None], zs[None, :]]
        # pairs of transition metals: 1 + (a_A + a_B) / 2, a = 0.1 (3d), 0.2 (4d, 5d)
        metal = torch.tensor([_metal_pair_term(z) if self.par.shells[z][0][1] == 2 else 0.0
                              for z in zs.tolist()], dtype=dt, device=dev)
        both = (metal[:, None] > 0) & (metal[None, :] > 0)
        K = torch.where(both, K * (1.0 + 0.5 * (metal[:, None] + metal[None, :])), K)
        # the second hydrogen s shell: kdiff in the average, no pair or electronegativity factor
        diffuse = torch.tensor([z == 1 and k == 1 for z, k in basis.shell_key], device=dev)
        kd = torch.where(diffuse, g["kdiff"] * torch.ones_like(kl[ls]), kl[ls])
        either = diffuse[:, None] | diffuse[None, :]
        K = torch.where(either, 0.5 * (kd[:, None] + kd[None, :]), K)
        X = self._electronegativity_factor(basis, dt, dev)
        X = torch.where(either, torch.ones_like(X), X)
        _, kpoly = self._shell_constants(basis, dt, dev)
        return K * X * self._distance_polynomial(basis, pos, kpoly)

    def _element_tables(self) -> Dict[str, Tensor]:
        out = super()._element_tables()
        out["KPAIR"] = torch.ones(MAX_Z + 1, MAX_Z + 1, dtype=torch.float64)
        for (a, b), v in self.par.pairs.items():
            out["KPAIR"][a, b] = out["KPAIR"][b, a] = v
        return out

    def _pair_table(self, dt, dev) -> Tensor:
        """Element-pair scaling factors of the Hamiltonian (one where none is given), symmetric."""
        table = self.table("KPAIR").to(device=dev, dtype=dt)
        return 0.5 * (table + table.t())

    def _file_pairs(self) -> Dict[Tuple[int, int], float]:
        table = self._pair_table(torch.float64, None)
        pairs = {key: float(table[key]) for key in self.par.pairs}
        for a, b in torch.nonzero((table - 1.0).abs() > 1e-12).tolist():
            if a >= b:
                pairs[(a, b)] = float(table[a, b])
        return pairs

    def scc_context(self, z, pos, basis, cn, batch, n_struct) -> dict:
        dtype, device = pos.dtype, pos.device
        sz, sl = basis.shell_z, basis.shell_l_t
        kappa = torch.where(sl > 0, self.kappa_scale * self.table("LPAR")[sz, sl], torch.zeros_like(sl, dtype=dtype))
        eta = self.table("GAM")[sz] * (1.0 + kappa)
        sa = basis.shell_atom
        ps = pos.index_select(0, sa)
        d = ps[:, None, :] - ps[None, :, :]
        r2 = (d * d).sum(-1)
        eta_ab = 2.0 / (1.0 / eta[:, None] + 1.0 / eta[None, :])
        gamma = 1.0 / torch.sqrt(r2 + eta_ab ** -2)
        ss = torch.tensor(basis.shell_struct, device=device)
        gamma = torch.where(ss[:, None] == ss[None, :], gamma, torch.zeros_like(gamma))
        gam3 = self.gam3_scale * self.table("GAM3")[z]
        return {"gamma": gamma, "gam3": gam3, "shell_atom": sa, "shell_struct": ss, "batch": batch,
                "n_struct": n_struct}

    def _terms(self, q_shell, mu, theta, ctx) -> Dict[str, Tensor]:
        n_struct, batch, ss = ctx["n_struct"], ctx["batch"], ctx["shell_struct"]
        dtype = q_shell.dtype

        def per(values, index):
            return torch.zeros(n_struct, dtype=dtype, device=q_shell.device).index_add(0, index, values)

        e_gamma = per(0.5 * q_shell * (ctx["gamma"] @ q_shell), ss)
        q = torch.zeros(batch.shape[0], dtype=dtype, device=q_shell.device).index_add(0, ctx["shell_atom"], q_shell)
        e_third = per(q ** 3 * ctx["gam3"] / 3.0, batch)
        return {"electrostatic": e_gamma + e_third, "gamma": e_gamma, "third_order": e_third}

    def potentials(self, q_shell, mu, theta, ctx):
        """Closed-form potentials of the isotropic and atomic third-order terms."""
        sa = ctx["shell_atom"]
        q = torch.zeros(ctx["batch"].shape[0], dtype=q_shell.dtype, device=q_shell.device).index_add(0, sa, q_shell)
        v = ctx["gamma"] @ q_shell + (q * q * ctx["gam3"])[sa]
        return v, torch.zeros_like(mu), torch.zeros_like(theta)

    def classical_energy(self, z, pos, batch, n_struct) -> Dict[str, Tensor]:
        g = self._g()
        i, j = _all_pairs(batch)
        alpha, zeff = self.table("REPA")[z], self.table("REPB")[z]
        light = z <= 2
        one = torch.ones_like(alpha[i])
        kexp = torch.where(light[i] & light[j], g["kexplight"] * one, g["kexp"] * one)
        r = (pos.index_select(0, j) - pos.index_select(0, i)).norm(dim=-1)
        e_rep = zeff[i] * zeff[j] / r * torch.exp(-torch.sqrt(alpha[i] * alpha[j]) * r ** kexp)
        rep = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device).index_add(0, batch[i], e_rep)
        d3 = self.d3.to(pos.dtype)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        dij = pos.index_select(0, j) - pos.index_select(0, i)
        vec = torch.cat([dij, -dij]) * BOHR
        b = n_struct
        e3 = d3.evaluate(z, pos * BOHR, ei, vec, batch, b, torch.zeros(b, 3, 3, dtype=pos.dtype, device=pos.device),
                         torch.zeros(b, 3, dtype=torch.bool, device=pos.device),
                         torch.zeros(b, dtype=pos.dtype, device=pos.device))["node_energy"] / d3.hartree
        disp = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device).index_add(0, batch, e3)
        xb = self.halogen_bond_energy(z, pos, batch, n_struct)
        return {"total": rep + disp + xb, "repulsion": rep, "dispersion": disp, "halogen_bond": xb}

    def halogen_bond_energy(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int) -> Tensor:
        """Halogen-bond correction (GFN1 paper eqs 14 and 15) of the halogens X with a ``CXB``
        parameter (Br, I, At) and the acceptors A in ``xb_acceptors``:
        ``0.1 CXB f(theta) (x^12 - k x^6) / (x^12 + 1)``, ``x = s (R_A + R_X) / R_AX`` with the
        atomic radii of the distance polynomial, ``k`` = ``xbdamp``, ``s`` = ``xbrad`` and
        ``f = ((1 - cos theta) / 2)^6`` of the angle A-X-B, B the atom nearest to X. As in the
        reference program, P and S act as acceptors besides the paper's N and O."""
        out = torch.zeros(n_struct, dtype=pos.dtype, device=pos.device)
        zl = z.tolist()
        g = self._g()
        halogens = [i for i, za in enumerate(zl) if self.par.value(za, "CXB") != 0.0]
        if not halogens:
            return out
        bl = batch.tolist()
        for x in halogens:
            same = [j for j in range(len(zl)) if bl[j] == bl[x] and j != x]
            acceptors = [j for j in same if zl[j] in self.xb_acceptors]
            if not same or not acceptors:
                continue
            d = (pos[same] - pos[x]).norm(dim=-1)
            b = same[int(torch.argmin(d.detach()))]
            idx = torch.tensor(acceptors, device=pos.device)
            rxa = pos[idx] - pos[x]
            rxb = pos[b] - pos[x]
            dxa = rxa.norm(dim=-1)
            cos = (rxa * rxb).sum(-1) / (dxa * rxb.norm())
            damp = (0.5 * (1.0 - cos)) ** 6
            rad = torch.tensor([self.radii[zl[j]] for j in acceptors], dtype=pos.dtype, device=pos.device)
            xx = g["xbrad"] * (rad + self.radii[zl[x]]) / dxa
            x6 = xx ** 6
            e = 0.1 * self.par.value(zl[x], "CXB") * damp * (x6 * x6 - g["xbdamp"] * x6) / (x6 * x6 + 1.0)
            out = out.index_add(0, batch[idx], torch.where(idx == b, torch.zeros_like(e), e))
        return out


@register_model("gfn0_xtb")
class GFN0(GFN):
    """GFN0-xTB (Pracht, Caldeweyher, Ehlert and Grimme, ChemRxiv (2019), doi 10.26434/chemrxiv.8326202.v1);
    see :class:`GFN`.

    The method has no self-consistent cycle. The charges of an electronegativity-equilibration model
    (with the parameters of the method) give the electrostatic energy and shift the shell levels,
    ``h = h0 - k_CN CN - k_q q - Gamma q^2``; the extended Hueckel matrix is diagonalized once. The
    classical terms are the repulsion (its exponent scaled with the electronegativity difference), the
    two-body D4 dispersion and a short-range bond correction for pairs of unlike atoms from B to F.
    ``accuracy``, ``max_iter``, ``mixing``, ``history`` and ``guess`` have no effect. ``trainable``
    takes ``LEV``, ``KCN``, ``KQ`` and ``KQAT2`` (shell levels and their shifts with the coordination
    number and the charge), ``POLY``, ``REPA`` and ``REPB``; on request also the Slater exponents, the
    global parameters and the parameters of the charge model (``XI``, ``KAPPA``, ``GAM``, ``ALPG``).
    """

    method = "gfn0"
    multipoles = False
    scc_terms = ()
    reported_terms = ()
    trainable_names = ("LEV", "KCN", "KQ", "KQAT2", "POLY", "REPA", "REPB")
    shell_tables = ("KCN", "KQ", "POLY")
    #: Parameters of the charge model: electronegativity, its slope with the coordination number,
    #: hardness and charge width.
    charge_tables = ("XI", "KAPPA", "GAM", "ALPG")
    atom_tables = ("KQAT2", "REPA", "REPB") + charge_tables
    extra_trainable = ("EXP", "GLOBAL") + charge_tables
    #: Factors of the Hamiltonian for pairs of transition metals and of coinage metals.
    metal_pair = 1.10
    coinage_pair = 0.90

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from xnn.common.models.d4 import DFTD4
        par, g = self.par, self.par.globals
        self.radii = [0.0] + [GFN2_RADII[z] / BOHR for z in range(1, 87)]
        # electronegativities: the method's where it gives them, the standard ones otherwise
        self.en_method = [par.value(z, "EN") if "EN" in par.elements.get(z, {}) else self.en[z]
                          for z in range(MAX_Z + 1)]
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            # the charge model with the method's parameters, and the two-body D4 term (its own charges)
            self.charge_model = DFTD4()
            for buffer, key in (("eeq_chi", "XI"), ("eeq_kcnchi", "KAPPA"), ("eeq_eta", "GAM"), ("eeq_rad", "ALPG")):
                for z in par.shells:
                    getattr(self.charge_model, buffer)[z] = par.value(z, key)
            self.d4 = DFTD4(s6=1.0, s8=g["s8"], a1=g["a1"], a2=g["a2"], s9=0.0)
        finally:
            torch.set_default_dtype(previous)

    def _element_tables(self) -> Dict[str, Tensor]:
        out = super()._element_tables()
        out["KQ"] = torch.zeros(MAX_Z + 1, 3, dtype=torch.float64)
        for name in ("KQAT2",) + self.charge_tables:
            out[name] = torch.zeros(MAX_Z + 1, dtype=torch.float64)
        for z in self.par.shells:
            for l, letter in enumerate("SPD"):
                out["KQ"][z, l] = self.par.value(z, "KQ" + letter)
            for name in ("KQAT2",) + self.charge_tables:
                out[name][z] = self.par.value(z, name)
        return out

    def reference_occupations(self, z: int) -> List[float]:
        if z == 1:
            return [1.0, 0.0]
        return gfn1_reference_occupations(self.par, z)

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Total energies (eV), the charges of the charge model and the energy terms of a batch of molecules."""
        out = super().forward(data)
        state = self.__dict__.pop("_state", None)
        if state is not None:
            out["charges"] = state["q"].detach()
        return out

    def _begin(self, z: Tensor, pos: Tensor, batch: Tensor, n_struct: int, charge: Tensor) -> None:
        """Charges, coordination numbers and electrostatic energy of the charge model."""
        model = self.charge_model
        if set(self.trainable) & set(self.charge_tables):
            # trained parameters of the charge model: it reads the tables
            for buffer, key in zip(("eeq_chi", "eeq_kcnchi", "eeq_eta", "eeq_rad"), self.charge_tables):
                setattr(model, buffer, self.table(key).to(pos.dtype) * 1.0)
        n = z.shape[0]
        i, j = _all_pairs(batch)
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        r = (pos.index_select(0, j) - pos.index_select(0, i)).norm(dim=-1)
        cn = model.coordination_numbers(z, ei, torch.cat([r, r]), n)[1]
        x = -model.eeq_chi[z] + model.eeq_kcnchi[z] * cn / torch.sqrt(cn + model.eeq_cn_reg)
        q = torch.zeros(n, dtype=pos.dtype, device=pos.device)
        energy = []
        cell = torch.zeros(3, 3, dtype=pos.dtype, device=pos.device)
        for b in range(n_struct):
            members = torch.nonzero(batch == b).squeeze(1)
            qb = model.eeq_charges(z[members], pos[members], cn[members], charge[b], cell, False)
            A = model._coulomb_matrix_molecular(z[members], pos[members])
            energy.append(qb @ (0.5 * A @ qb - x[members]))
            q = q.index_put((members,), qb)
        self.__dict__["_state"] = {"q": q, "cn": cn, "electrostatic": torch.stack(energy), "charge": charge,
                                   "pairs": (i, j, r)}

    def coordination_numbers(self, z: Tensor, pos: Tensor, batch: Tensor) -> Tensor:
        return self._state["cn"]

    def _responsive_density(self, *args):
        """As in :class:`GFN`, with the charges and coordination numbers of the charge model at the
        density's positions: functions of them when they carry gradients (see :meth:`energy_gradient`),
        else constants, or functions of the trained parameters of the charge model."""
        state = self._state
        if args[2].requires_grad or set(self.trainable) & set(self.charge_tables):
            self._begin(args[0], args[2], args[3], args[11], state["charge"])
        else:
            self.__dict__["_state"] = {k: (v.detach() if isinstance(v, Tensor) else v) for k, v in state.items()}
        try:
            return super()._responsive_density(*args)
        finally:
            self.__dict__["_state"] = state

    def shell_levels(self, z: List[int], basis: _Basis, cn: Tensor) -> Tensor:
        sz, sl, sa = basis.shell_z, basis.shell_l_t, basis.shell_atom
        q = self._state["q"].index_select(0, sa)
        return (self.table("LEV")[sz, basis.shell_index] - 0.1 * self.table("KCN")[sz, sl] * cn.index_select(0, sa)
                - self.table("KQ")[sz, sl] * q - self.table("KQAT2")[sz] * q * q) / AUTOEV

    def shell_pair_factor(self, basis: _Basis, pos: Tensor) -> Tensor:
        g = self._g()
        dt, dev = pos.dtype, pos.device
        ls, zs = basis.shell_l_t, basis.shell_z
        kl = _vector([g["ks"], g["kp"], g["kd"]], dt, dev)[ls]
        K = 0.5 * (kl[:, None] + kl[None, :])
        # the second hydrogen s shell: scaled by kdiff, no interaction between two of them
        diffuse = torch.tensor([z == 1 and k == 1 for z, k in basis.shell_key], device=dev)
        K = torch.where(diffuse[:, None] | diffuse[None, :], K * g["kdiff"], K)
        K = torch.where(diffuse[:, None] & diffuse[None, :], torch.zeros_like(K), K)
        metal = torch.tensor([self.par.shells[z][0][1] == 2 for z in zs.tolist()], device=dev)
        coinage = (zs == 29) | (zs == 47) | (zs == 79)
        K = torch.where(metal[:, None] & metal[None, :], K * self.metal_pair, K)
        K = torch.where(coinage[:, None] & coinage[None, :], K * (self.coinage_pair / self.metal_pair), K)
        ken = 0.01 * _vector([g["ens"], g["enp"], g["end"]], dt, dev)[ls]
        ken = 0.5 * (ken[:, None] + ken[None, :])
        en = torch.tensor(self.en_method, dtype=dt, device=dev)[zs]
        d2 = (en[:, None] - en[None, :]) ** 2
        X = 1.0 + ken * d2 + ken * g["enscale4"] * d2 * d2
        zeta, kpoly = self._shell_constants(basis, dt, dev)
        Y = 2.0 * torch.sqrt(zeta[:, None] * zeta[None, :]) / (zeta[:, None] + zeta[None, :])
        return K * X * Y * self._distance_polynomial(basis, pos, kpoly)

    def scc_context(self, z, pos, basis, cn, batch, n_struct) -> dict:
        return {}

    def _terms(self, q_shell, mu, theta, ctx) -> Dict[str, Tensor]:
        return {}

    def potentials(self, q_shell, mu, theta, ctx):
        return torch.zeros_like(q_shell), torch.zeros_like(mu), torch.zeros_like(theta)

    def _initial_charges(self, z, pos, batch, n_struct, charge, basis, n0) -> Optional[Tensor]:
        return None

    def _scc(self, basis: _Basis, S, DA, QA, H0, ctx, n0, n_alpha, n_beta, n_struct,
             start: Optional[Tensor] = None, mixing: Optional[float] = None,
             temperature: Optional[float] = None) -> Dict[str, Tensor]:
        """One diagonalization: the Hamiltonian does not depend on the density."""
        n_shell, n_atoms = n0.shape[0], int(basis.ao_atom.max()) + 1
        ortho = _Orthogonalizer(S)
        left = _stacked(ortho.left(S[:, None]))
        h0 = ortho.right(ortho.left(H0))
        x = torch.zeros(n_shell, dtype=S.dtype, device=S.device)
        result, _ = self._scc_step(x, basis, S, DA, QA, ctx, n0, n_alpha, n_beta, ortho, left, h0,
                                   KB_AU * self.electronic_temperature, n_shell, n_atoms)
        result["W"] = (result["C"] * (result["occ"] * result["eps"])[:, None, :]) @ result["C"].transpose(-1, -2)
        result["x"] = x
        result["converged"] = torch.tensor(True, device=S.device)
        result["iterations"] = 1
        return result

    def classical_energy(self, z, pos, batch, n_struct) -> Dict[str, Tensor]:
        g = self._g()
        dt, dev = pos.dtype, pos.device
        state = self._state
        i, j, r = state["pairs"]
        cn = state["cn"]

        def per(values, index):
            return torch.zeros(n_struct, dtype=dt, device=dev).index_add(0, index, values)

        # repulsion: the exponent grows with the electronegativity difference
        en = torch.tensor(self.en_method, dtype=dt, device=dev)[z]
        d2 = (en[i] - en[j]) ** 2
        alpha, zeff = self.table("REPA")[z], self.table("REPB")[z]
        exponent = torch.sqrt(alpha[i] * alpha[j]) * (1.0 + 0.01 * g["renscale"] * (d2 + d2 * d2))
        rep = per(zeff[i] * zeff[j] / r * torch.exp(-exponent * r ** g["kexp"]), batch[i])
        # two-body D4 dispersion
        ei = torch.cat([torch.stack([i, j]), torch.stack([j, i])], 1)
        dij = pos.index_select(0, j) - pos.index_select(0, i)
        d4 = self.d4
        out = d4.evaluate(z, pos * BOHR, ei, torch.cat([dij, -dij]) * BOHR, batch, n_struct,
                          torch.zeros(n_struct, 3, 3, dtype=dt, device=dev),
                          torch.zeros(n_struct, 3, dtype=torch.bool, device=dev), state["charge"])
        disp = out["energy_2body"] / d4.hartree
        # short-range bond correction of unlike pairs from B to F
        table = torch.zeros(MAX_Z + 1, 3, dtype=dt, device=dev)
        for element, values in GFN0_SRB.items():
            table[element] = torch.tensor(values, dtype=dt, device=dev)
        inside = (z >= 5) & (z <= 9)
        sel = torch.nonzero(inside[i] & inside[j] & (z[i] != z[j])).squeeze(1)
        a, b, rs = i[sel], j[sel], r[sel]
        ta, tb = table[z[a]], table[z[b]]
        dd = (ta[:, 2] - tb[:, 2]).abs()
        r0 = ((1.0 - GFN0_SRB_FACTOR[0] * dd - GFN0_SRB_FACTOR[1] * dd * dd)
              * (ta[:, 0] + tb[:, 0] + 2.0 * g["srbshift"] + ta[:, 1] * cn[a] + tb[:, 1] * cn[b]))
        standard = torch.tensor(self.en, dtype=dt, device=dev)[z]
        width = g["srbexp"] * (1.0 + g["srbken"] * (standard[a] - standard[b]) ** 2)
        srb = per(g["srbpre"] * torch.exp(-width * (rs - r0) ** 2), batch[a])
        es = state["electrostatic"]
        return {"total": rep + es + disp + srb, "repulsion": rep, "electrostatic": es, "dispersion": disp,
                "short_range": srb}

