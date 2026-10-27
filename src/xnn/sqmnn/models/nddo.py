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

from typing import Dict, Sequence, Union

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential

from ..common import elements as el
from ..common.constants import BOHR, COULOMB, EV_TO_KCAL
from ..common.parameters import ELEMENT_PARAMETERS, GAUSSIAN_PARAMETERS, PAIR_PARAMETERS, ParameterSet, read_parameters
from .angular import L_OF, MOLECULAR, dipole_factors, pair_type
from .derived import element_quantities
from .scf import PAD_ENERGY, Layout, SCFDensity, electronic_energy, fock_matrices, lowest, run_scf
from .twocenter import pair_integrals

_ELEMENT_NAMES = ELEMENT_PARAMETERS + GAUSSIAN_PARAMETERS
_N_GLOBALS = 61
#: Atom pairs per block of the two-center integral evaluation.
PAIR_CHUNK = 20000


def _shell_of(n_orb: int):
    return [L_OF[lab] for lab in MOLECULAR[:n_orb]]


def _feather(ints: Dict[str, Tensor], mask_a: Tensor, mask_b: Tensor, tore_a: Tensor,
             tore_b: Tensor, r_point: float = 7.0, width: float = 0.22) -> None:
    """Blend the two-center Coulomb integrals into point-charge values (in place).

    With ``c = 1 - exp(-width (R - r_point)^2)`` below ``r_point`` (Angstrom) and
    0 beyond, every integral becomes ``c`` times its NDDO value, and the
    Coulomb ones (``(mu mu|la la)``, the core attractions of the diagonal
    distributions and the core-core ``gamma``) gain ``(1 - c)`` times the
    point-charge value ``e^2 / R``: the integrals reach the exact
    point-charge limit at ``r_point``. Used by PM7 and PM6-ORG.
    """
    r = ints["r"]
    c = torch.where(r < r_point, 1.0 - torch.exp(-width * (r - r_point) ** 2), torch.zeros_like(r))
    point = COULOMB / r
    one = 1.0 - c
    eye_a = torch.diag_embed(mask_a)
    eye_b = torch.diag_embed(mask_b)
    ints["W"] = (c[:, None, None, None, None] * ints["W"]
                 + (one * point)[:, None, None, None, None] * eye_a[:, :, :, None, None]
                 * eye_b[:, None, None, :, :])
    ints["VA"] = c[:, None, None] * ints["VA"] - (one * point * tore_b)[:, None, None] * eye_a
    ints["VB"] = c[:, None, None] * ints["VB"] - (one * point * tore_a)[:, None, None] * eye_b
    ints["gab"] = c * ints["gab"] + one * point


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
    """

    #: Name of the published parameter set of the method.
    default_parameters = ""
    #: Method label (the MOPAC keyword).
    method = ""
    #: Blend the two-center Coulomb integrals into point charges below 7 Angstrom
    #: (PM7, PM6-ORG; see :func:`_feather`).
    feather = False
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
                 scf_starts: Sequence = ((0.0, 0), (1.0, 80), (2.0, 25))):
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

    def adjust_integrals(self, ints: Dict[str, Tensor], za: Tensor, zb: Tensor, n_orb_a: Tensor,
                         n_orb_b: Tensor) -> None:
        """Method-specific changes of the molecular-frame pair integrals (in place)."""

    def corrections(self, data: AtomicGraph, pos: Tensor) -> Tensor:
        """Per-atom post-SCF and molecular-mechanics energy corrections (kcal/mol)."""
        return pos.new_zeros(pos.shape[0])

    # helpers

    def _gaussian(self, z: Tensor, j: int):
        return (self.element_parameter(f"FN1{j}")[z], self.element_parameter(f"FN2{j}")[z],
                self.element_parameter(f"FN3{j}")[z])

    def _element_quantities(self, uz: Sequence[int]):
        frozen = not (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()))
        key = (tuple(uz), str(self.element_table.device),
               tuple(t._version for t in list(self.parameters()) + list(self.buffers())))
        if frozen and key in self._q_cache:
            return self._q_cache[key]
        idx = torch.as_tensor(list(uz), device=self.element_table.device)
        p = {name: self.element_parameter(name)[idx] for name in _ELEMENT_NAMES}
        q = element_quantities(uz, p)
        if frozen:
            self._q_cache = {key: q}
        return q

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
        ``reuse_density``, ``scf_starts``.
        """

        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        keys = ("parameters", "trainable", "unrestricted", "response", "max_iter",
                "tol_density", "tol_energy", "corrections", "reuse_density", "scf_starts")
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
        lookup = torch.full((el.MAX_Z + 1,), -1, dtype=torch.long, device=device)
        lookup[torch.as_tensor(uz, device=device)] = torch.arange(len(uz), device=device)
        ea = lookup[z]
        n_orb_e = torch.as_tensor(q["n_orb"], device=device)
        n_orb = n_orb_e[ea]
        k = int(n_orb.max())

        # pairs i < j of the same structure
        same = (batch.unsqueeze(0) == batch.unsqueeze(1)).triu(diagonal=1)
        pa, pb = same.nonzero(as_tuple=True)

        # basis layout
        counts = torch.zeros(n_struct, dtype=torch.long, device=device).index_add(0, batch, n_orb)
        n_basis = int(counts.max())
        # atoms of a structure are contiguous (the collate convention)
        cum = torch.cumsum(n_orb, 0) - n_orb
        start = cum - (torch.cumsum(counts, 0) - counts)[batch]
        orb = torch.arange(k, device=device)
        has = orb.unsqueeze(0) < n_orb.unsqueeze(1)                          # (N, k)
        ao = torch.where(has, start.unsqueeze(1) + orb.unsqueeze(0), torch.full_like(has, n_basis, dtype=torch.long))
        valid = torch.arange(n_basis, device=device).unsqueeze(0) < counts.unsqueeze(1)
        layout = Layout(n_struct, n_basis, k, batch, ao, pa, pb, valid)

        tore = q["tore"][ea]

        def build(positions: Tensor):
            vec = positions[pb] - positions[pa]
            # pairs in chunks: the multipole intermediates scale as pairs x components
            chunk = PAIR_CHUNK if k < 9 else PAIR_CHUNK // 16
            parts = [pair_integrals(q, ea[pa[s:s + chunk]], ea[pb[s:s + chunk]], vec[s:s + chunk], k,
                                    need_general=(k == 9))
                     for s in range(0, max(int(pa.shape[0]), 1), chunk)]
            ints = {key: torch.cat([p[key] for p in parts]) for key in parts[0]}
            mask = has.to(dtype)
            mA, mB = mask[pa], mask[pb]
            ints["S"] = ints["S"] * mA[:, :, None] * mB[:, None, :]
            ints["VA"] = ints["VA"] * mA[:, :, None] * mA[:, None, :]
            ints["VB"] = ints["VB"] * mB[:, :, None] * mB[:, None, :]
            ints["W"] = (ints["W"] * mA[:, :, None, None, None] * mA[:, None, :, None, None]
                         * mB[:, None, None, :, None] * mB[:, None, None, None, :])
            if self.feather:
                _feather(ints, mask[pa], mask[pb], tore[pa], tore[pb])
            self.adjust_integrals(ints, z[pa], z[pb], n_orb[pa], n_orb[pb])
            # core Hamiltonian
            shells = torch.as_tensor([[0, 1, 1, 1, 2, 2, 2, 2, 2][i] for i in range(k)], device=device)
            u = torch.stack([q["uss"], q["upp"], q["udd"]], -1)[ea][:, shells] * mask
            beta = torch.stack([q["betas"], q["betap"], q["betad"]], -1)[ea][:, shells] * mask
            atom_blocks = torch.diag_embed(u)
            atom_blocks = atom_blocks.index_add(0, pa, ints["VA"]).index_add(0, pb, ints["VB"])
            pair_blocks = ints["S"] * 0.5 * (beta[pa][:, :, None] + beta[pb][:, None, :])
            H = layout.scatter(atom_blocks, pair_blocks)
            diag_pad = torch.where(valid, torch.zeros((), dtype=dtype, device=device),
                                   torch.full((), PAD_ENERGY, dtype=dtype, device=device))
            H = H + torch.diag_embed(diag_pad)
            W1 = q["one_center"][ea][:, :k, :k, :k, :k]
            W1 = (W1 * mask[:, :, None, None, None] * mask[:, None, :, None, None]
                  * mask[:, None, None, :, None] * mask[:, None, None, None, :])
            return ints, H, W1, atom_blocks, pair_blocks

        ints, H, W1, _, _ = build(pos)
        W2 = ints["W"]
        e_core_pair = self.core_core(z[pa], z[pb], ints["r"], ints["gab"], tore[pa], tore[pb])

        # electrons and spin
        charge = (data.total_charge.to(dtype) if data.total_charge is not None
                  else torch.zeros(n_struct, dtype=dtype, device=device))
        n_elec = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, tore.detach()) - charge
        n_e = torch.round(n_elec).long()
        if data.spin_multiplicity is not None:
            mult = torch.round(data.spin_multiplicity.to(dtype)).long()
        else:
            mult = 1 + (n_e % 2)
        if bool(((n_e + mult - 1) % 2 != 0).any()):
            raise ValueError("spin multiplicity inconsistent with the electron count")
        n_alpha = (n_e + mult - 1) // 2
        n_beta = (n_e - mult + 1) // 2
        uhf = self.unrestricted or bool((mult != 1).any())

        # initial guess: reference occupations spread over each shell
        occ_ref = torch.as_tensor([[el.OCC_S[v], el.OCC_P[v] / 3.0, el.OCC_D[v] / 5.0] for v in uz],
                                  dtype=dtype, device=device)
        shells = torch.as_tensor([[0, 1, 1, 1, 2, 2, 2, 2, 2][i] for i in range(k)], device=device)
        diag = occ_ref[ea][:, shells] * has.to(dtype)
        tot = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, diag.sum(-1))
        scale = torch.where(tot > 0, n_e.to(dtype) / tot.clamp(min=1e-12), torch.zeros_like(tot))
        diag = diag * scale[batch].unsqueeze(-1)
        guess = layout.scatter(torch.diag_embed(diag), H.new_zeros((pa.shape[0], k, k)))

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
                             spin_guess=self.last_scf.densities if uhf else None)
        else:
            result = lowest([run_scf(layout, H, W1, W2, n_alpha, n_beta, uhf, guess, self.max_iter,
                                     self.tol_density, self.tol_energy, smearing=kt, smear_steps=steps)
                             for kt, steps in self.scf_starts])
        self.last_scf, self._scf_key, self._scf_pos = result, key, pos.detach().clone()
        if self._use_response():
            pos_fixed = pos.detach()
            ints_t, H_t, W1_t, _, _ = build(pos_fixed)
            P = SCFDensity.apply(H_t, W1_t, ints_t["W"], result, layout, uhf, 1e-10)
        else:
            P = result.densities
        F = fock_matrices(layout, H, W1, W2, P, uhf)
        e_el = electronic_energy(H, F, P, uhf)

        # per-atom partition of the electronic energy
        if uhf:
            e_mat = 0.5 * ((P[0] + P[1]) * H + (P * F).sum(0))
        else:
            e_mat = 0.5 * P[0] * (H + F[0])
        e_atom = layout.gather_atom(e_mat).sum((-1, -2))
        if pa.shape[0]:
            e_pair = layout.gather_pair(e_mat).sum((-1, -2))
            e_atom = e_atom.index_add(0, pa, e_pair).index_add(0, pb, e_pair)
            e_atom = e_atom.index_add(0, pa, 0.5 * e_core_pair).index_add(0, pb, 0.5 * e_core_pair)
        eisol = q["eisol"][ea]
        atom_heat = torch.as_tensor([el.ATOM_HEAT[v] for v in z.tolist()], dtype=dtype, device=device)
        corr = (self.corrections(data, pos) if self.use_corrections
                else pos.new_zeros(pos.shape[0]))
        node_kcal = (e_atom - eisol) * EV_TO_KCAL + atom_heat + corr
        node_energy = node_kcal / EV_TO_KCAL
        hof = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, node_kcal)
        e_core = torch.zeros(n_struct, dtype=dtype, device=device)
        if pa.shape[0]:
            e_core = e_core.index_add(0, batch[pa], e_core_pair)

        # charges and dipole moment (e Angstrom)
        P_tot = P.sum(0) if uhf else P[0]
        P_atom = layout.gather_atom(P_tot)
        charges = tore - P_atom.diagonal(dim1=-2, dim2=-1).sum(-1)
        dfac = torch.as_tensor(dipole_factors()[:, :k, :k], dtype=dtype, device=device)
        types = [[pair_type(L_OF[a], L_OF[b]) for b in MOLECULAR[:k]] for a in MOLECULAR[:k]]
        sp_mask = torch.as_tensor([[t == "sp" for t in row] for row in types], dtype=dtype, device=device)
        pd_mask = torch.as_tensor([[t == "pd" for t in row] for row in types], dtype=dtype, device=device)
        d_sp, d_pd = q["dd"][ea], q["ddp"][ea, 5]
        fac = dfac.unsqueeze(0) * (sp_mask * d_sp[:, None, None] + pd_mask * d_pd[:, None, None]).unsqueeze(1)
        has_p = (n_orb > 1).to(dtype)
        hybrid = -BOHR * torch.einsum("aij,acij->ac", P_atom, fac) * has_p[:, None]
        mass = torch.as_tensor([el.ATOMIC_MASS[v] for v in z.tolist()], dtype=dtype, device=device)
        msum = torch.zeros(n_struct, dtype=dtype, device=device).index_add(0, batch, mass)
        com = torch.zeros(n_struct, 3, dtype=dtype, device=device).index_add(
            0, batch, mass[:, None] * pos) / msum[:, None]
        charged = (charge.abs() > 0.5).to(dtype)
        ref_point = com * charged[:, None]
        dip_atom = charges[:, None] * (pos - ref_point[batch]) + hybrid
        dipole = torch.zeros(n_struct, 3, dtype=dtype, device=device).index_add(0, batch, dip_atom)

        return {
            "node_energy": node_energy,
            "energy": hof / EV_TO_KCAL,
            "heat_of_formation": hof,
            "total_energy": e_el + e_core,
            "electronic_energy": e_el,
            "core_core_energy": e_core,
            "charges": charges,
            "dipole": dipole,
            "scf_converged": result.converged,
        }
