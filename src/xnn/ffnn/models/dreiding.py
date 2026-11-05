"""DREIDING: the rule-generated generic force field.

DREIDING (Mayo, Olafson & Goddard, *J. Phys. Chem.* 94, 8897, 1990) writes
the total energy as valence terms generated from per-atom parameters by
hybridization rules plus nonbonded terms:

    E = E_bond + E_angle + E_torsion + E_inversion
        + E_vdw + E_coulomb + E_hbond

* **Bonds** (eqs 4-9): harmonic ``1/2 k_n (R - R0)^2`` (or the Morse form,
  the DREIDING/M variant) with ``R0 = R0_I + R0_J - 0.01`` A from per-type
  bond radii and ``k_n = n * 700 (kcal/mol)/A^2`` for bond order ``n``.
* **Angles** (eqs 10-12): the harmonic-cosine form
  ``1/2 K/sin^2(t0) (cos t - cos t0)^2`` with ``t0`` set by the central atom
  and ``K = 100 (kcal/mol)/rad^2``; linear centers use ``K (1 + cos t)``.
  The plain harmonic-theta form (eq 11) is available as an option.
* **Torsions** (eqs 13-23): ``1/2 V {1 - cos[n (phi - phi0)]}`` with
  ``(V, n, phi0)`` set by the rule engine
  (:func:`~xnn.ffnn.models.dreidinglib.torsion_rule`) from the central
  atoms' hybridizations and bond order; ``V`` is the total barrier of the
  central bond, split evenly over its dihedrals.
* **Inversions** (eq 28): at every 3-coordinate center of a listed type, the
  spectroscopic umbrella term ``K (1 - cos psi)`` (planar centers) or
  ``1/2 K/sin^2(psi0) (cos psi - cos psi0)^2``, averaged over the three
  axis choices with weight 1/3.
* **van der Waals**: Lennard-Jones ``D0 [rho^-12 - 2 rho^-6]`` with
  ``rho = R/R0`` (eq 31'), or the exponential-6 form (eq 32', the
  DREIDING/X6 variant); well depths combine geometrically, LJ radii
  arithmetically by default (eq 36c). 1,2 and 1,3 pairs are excluded; 1,4
  pairs count in full (the DREIDING default).
* **Electrostatics** (eq 37): ``322.0637 Q_i Q_j / R`` kcal/mol over the
  same pairs, when partial charges are supplied (DREIDING itself
  prescribes none; the paper uses Gasteiger charges where needed).
* **Hydrogen bonds** (eq 38): ``D_hb [5 (R_hb/R)^12 - 6 (R_hb/R)^10]
  cos^4(t_DHA)`` over donor-hydrogen...acceptor triplets involving the
  explicit ``H__HB`` hydrogen type, with the donor-acceptor distance ``R``
  and the angle restricted beyond 90 degrees.

Like OPLS, DREIDING is a fixed-topology force field: an instance binds the
parameter library to one molecular topology (with bond orders), and every
structure it evaluates must be a conformation of that system. Unlike OPLS,
all bonded parameters are *generated*, so what is trainable are the
generators themselves -- per-type radii, angles and vdW parameters, and the
global force constants and rule barriers -- which stays true to the
DREIDING philosophy while letting the force field be refit to new systems
by gradient descent (``trainable=...``), exactly like ReaxFF's and OPLS's
parameters. Several :class:`Dreiding` models can share one
:class:`DreidingForceField` to fit transferable parameters jointly.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import Parameter, ParameterDict

from xnn.common.data import AtomicGraph
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.fast import resolve_use_fast
from xnn.common.models.ops import scatter_sum

from .classical_fast import (
    ClassicalFastPath,
    evaluate_spec,
    fast_bonded,
    fast_nonbonded,
    image_shifts,
    lj_table,
    needs_twice,
    used_types,
    wants_fast,
    x6_table,
)
from .dreidinglib import TORSION_RULES, DreidingLibrary, read_dreiding, torsion_rule
from .geometry import TINY, dihedral_cos, geometric_mean, half_edges, pair_image_shifts, pair_vectors
from .oplslib import KCAL_TO_EV
from .topology import MolecularTopology, read_topology

# DREIDING's Coulomb constant (eq 37), 332.0637 kcal/mol * A / e^2, in
# eV * A. It differs slightly from the CODATA value OPLS uses; kept as
# published for fidelity with DREIDING codes.
KE = 332.0637 * KCAL_TO_EV
DEG = math.pi / 180.0
# Torsion rule ids in tensor order.
RULE_IDS = tuple(TORSION_RULES)

# Trainable parameter groups of the force field ("charge" additionally
# unlocks the per-atom charges held by each model).
_GROUPS = ("radius", "theta0", "bond_k", "bond_d", "angle_k", "torsion_v",
           "oop_k", "oop_psi0", "vdw_r0", "vdw_d0", "x6_zeta", "hbond_d0",
           "hbond_r0")


def _as_library(source: Union[DreidingLibrary, str, Path]) -> DreidingLibrary:
    """Coerce a library argument to a :class:`DreidingLibrary`.

    Parameters
    ----------
    source : DreidingLibrary, str or Path
        A parsed library, or anything
        :func:`~xnn.ffnn.models.dreidinglib.read_dreiding` accepts:
        ``"dreiding"``, ``"dreiding/X6"``, a ``.frc`` path or a native JSON
        path.

    Returns
    -------
    DreidingLibrary
        The library.
    """
    if isinstance(source, DreidingLibrary):
        return source
    return read_dreiding(source)


class DreidingForceField(nn.Module):
    """The trainable generator tensors of a DREIDING parameter set.

    Assembles the library's per-type generators and global rule constants
    into tensors (eV / Angstrom / radians) exposed as a
    :class:`torch.nn.ParameterDict`, so any group can be refit by gradient
    descent. One instance can be shared by several :class:`Dreiding` models
    (different molecules) to train transferable parameters jointly.

    Parameters
    ----------
    library : DreidingLibrary, str or Path
        The parameter library (or a spec for
        :func:`~xnn.ffnn.models.dreidinglib.read_dreiding`; default specs
        are ``"dreiding"`` and ``"dreiding/X6"``).
    trainable : sequence of str or "all", optional
        Parameter groups to expose to the optimizer: any of ``"radius"``,
        ``"theta0"``, ``"bond_k"``, ``"bond_d"``, ``"angle_k"``,
        ``"torsion_v"``, ``"oop_k"``, ``"oop_psi0"``, ``"vdw_r0"``,
        ``"vdw_d0"``, ``"x6_zeta"``, ``"hbond_d0"``, ``"hbond_r0"``;
        ``"all"`` unfreezes every group. Default: none (the fixed classical
        force field).

    Attributes
    ----------
    params : torch.nn.ParameterDict
        ``radius`` ``(T,)`` A, ``theta0`` ``(T,)`` rad, ``bond_k`` eV/A^2,
        ``bond_d`` eV, ``angle_k`` eV/rad^2, ``torsion_v`` ``(9,)`` eV (one
        total barrier per rule, in :data:`~xnn.ffnn.models.dreidinglib.TORSION_RULES`
        order), ``oop_k`` ``(T,)`` eV/rad^2, ``oop_psi0`` ``(T,)`` rad,
        ``vdw_r0`` ``(T,)`` A, ``vdw_d0`` ``(T,)`` eV, ``x6_zeta`` ``(T,)``,
        ``hbond_d0`` eV and ``hbond_r0`` A.
    type_names : list of str
        Row labels of the per-type tensors.
    form : str
        ``"lj"`` or ``"x6"``.
    combination : str
        The LJ ``R0`` combination rule (``"arithmetic"`` or
        ``"geometric"``).
    """

    def __init__(self, library: Union[DreidingLibrary, str, Path] = "dreiding",
                 trainable: Union[Sequence[str], str] = ()):
        super().__init__()
        lib = _as_library(library)
        dtype = torch.get_default_dtype()
        self.lib = lib
        self.name = lib.name
        self.form = lib.form
        self.combination = str(lib.combination)
        self.delta = float(lib.delta)
        self.templates = dict(lib.templates)
        self.fragments = dict(lib.fragments)

        if isinstance(trainable, str):
            trainable_set = set(_GROUPS) if trainable.lower() == "all" \
                else {trainable}
        else:
            trainable_set = set(trainable)
        unknown = trainable_set - set(_GROUPS)
        if unknown:
            raise ValueError(f"unknown trainable group(s) {sorted(unknown)}; "
                             f"available: {list(_GROUPS)}")

        self.type_names = list(lib.atom_types)
        self._type_index = {n: i for i, n in enumerate(self.type_names)}
        self.hybrids = [lib.hybrid(n) for n in self.type_names]
        from xnn.common.data.elements import SYMBOL_TO_Z
        elements = [lib.element(n) for n in self.type_names]
        self.register_buffer("type_z", torch.tensor(
            [SYMBOL_TO_Z.get(e, 0) for e in elements], dtype=torch.long))
        self.register_buffer("type_mass", torch.tensor(
            [float(lib.atom_types[n].get("mass", 0.0))
             for n in self.type_names], dtype=dtype))
        self.register_buffer("type_acceptor", torch.tensor(
            [lib.is_acceptor(n) for n in self.type_names]))
        self.donor_types = {n for n in self.type_names
                            if n in lib.hb_donor_types}
        # which generators each type actually has (missing ones raise when a
        # topology uses the type)
        self._has_radius = {n for n in self.type_names if n in lib.radius}
        self._has_vdw = {n for n in self.type_names if n in lib.vdw_r0}
        self.oop_types = set(lib.oop)

        def per_type(table, default, scale=1.0):
            """Dense per-type value list from a sparse library table."""
            return [float(table.get(n, default)) * scale
                    for n in self.type_names]

        self.params = ParameterDict()

        def make(name, values):
            """Register value list as the (possibly trainable) group ``name``."""
            self.params[name] = Parameter(
                torch.tensor(values, dtype=dtype),
                requires_grad=name in trainable_set)

        make("radius", per_type(lib.radius, 0.0))
        make("theta0", per_type(lib.theta0, 180.0, DEG))
        make("bond_k", float(lib.bond_k1) * KCAL_TO_EV)
        make("bond_d", float(lib.bond_d1) * KCAL_TO_EV)
        make("angle_k", float(lib.angle_k) * KCAL_TO_EV)
        make("torsion_v", [float(lib.torsion_v.get(r, TORSION_RULES[r][0]))
                           * KCAL_TO_EV for r in RULE_IDS])
        make("oop_k", [float(lib.oop.get(n, (0.0, 0.0))[0]) * KCAL_TO_EV
                       for n in self.type_names])
        make("oop_psi0", [float(lib.oop.get(n, (0.0, 0.0))[1]) * DEG
                          for n in self.type_names])
        make("vdw_r0", per_type(lib.vdw_r0, 0.0))
        make("vdw_d0", per_type(lib.vdw_d0, 0.0, KCAL_TO_EV))
        make("x6_zeta", per_type(lib.x6_zeta, 12.0))
        make("hbond_d0", float(lib.hbond_d0) * KCAL_TO_EV)
        make("hbond_r0", float(lib.hbond_r0))

        # fixed periodicity and phase sign per torsion rule: the energy is
        # 1/2 V {1 - cos[n (phi - phi0)]} = 1/2 V {1 - s cos(n phi)} with
        # s = cos(n phi0) = +-1 for the published phi0 values
        n_rule, s_rule = [], []
        for r in RULE_IDS:
            _, n, phi0 = TORSION_RULES[r]
            s = math.cos(n * phi0 * DEG)
            if abs(abs(s) - 1.0) > 1e-9:
                raise ValueError(f"torsion rule {r}: n*phi0 must be a "
                                 "multiple of 180 degrees")
            n_rule.append(n)
            s_rule.append(round(s))
        self.register_buffer("rule_n", torch.tensor(n_rule, dtype=torch.long))
        self.register_buffer("rule_sign", torch.tensor(s_rule, dtype=dtype))

    # resolution
    def type_index(self, name: str) -> int:
        """Row index of atom type ``name``.

        Parameters
        ----------
        name : str
            The DREIDING atom type (e.g. ``"C_3"``).

        Returns
        -------
        int
            Index into the per-type parameter tensors.

        Raises
        ------
        KeyError
            If the library has no such atom type.
        """
        if name not in self._type_index:
            raise KeyError(f"atom type {name!r} is not in library "
                           f"{self.name!r}")
        return self._type_index[name]

    def check_generators(self, name: str) -> None:
        """Raise if a type lacks the per-type generators terms need."""
        if name not in self._has_radius:
            raise KeyError(f"atom type {name!r} has no bond radius / angle "
                           f"in library {self.name!r}")
        if name not in self._has_vdw:
            raise KeyError(f"atom type {name!r} has no van der Waals "
                           f"parameters in library {self.name!r}")

    # export
    def export_library(self) -> DreidingLibrary:
        """Write the current (possibly trained) parameters back to a library.

        Returns
        -------
        DreidingLibrary
            A new library in DREIDING units (kcal/mol, Angstrom, degrees),
            round-trippable through :meth:`DreidingLibrary.save` /
            :func:`~xnn.ffnn.models.dreidinglib.read_dreiding`.
        """
        P = {k: v.detach().cpu() for k, v in self.params.items()}
        lib = self.lib
        names = self.type_names
        return DreidingLibrary(
            atom_types={n: dict(lib.atom_types[n]) for n in names},
            radius={n: float(P["radius"][i]) for i, n in enumerate(names)
                    if n in lib.radius},
            theta0={n: float(P["theta0"][i]) / DEG for i, n in enumerate(names)
                    if n in lib.theta0},
            vdw_r0={n: float(P["vdw_r0"][i]) for i, n in enumerate(names)
                    if n in lib.vdw_r0},
            vdw_d0={n: float(P["vdw_d0"][i]) / KCAL_TO_EV
                    for i, n in enumerate(names) if n in lib.vdw_d0},
            x6_zeta={n: float(P["x6_zeta"][i]) for i, n in enumerate(names)
                     if n in lib.x6_zeta},
            oop={n: (float(P["oop_k"][self._type_index[n]]) / KCAL_TO_EV,
                     float(P["oop_psi0"][self._type_index[n]]) / DEG)
                 for n in lib.oop},
            form=self.form, combination=self.combination, delta=self.delta,
            bond_k1=float(P["bond_k"]) / KCAL_TO_EV,
            bond_d1=float(P["bond_d"]) / KCAL_TO_EV,
            angle_k=float(P["angle_k"]) / KCAL_TO_EV,
            torsion_v={r: float(P["torsion_v"][i]) / KCAL_TO_EV
                       for i, r in enumerate(RULE_IDS)},
            hbond_d0=float(P["hbond_d0"]) / KCAL_TO_EV,
            hbond_r0=float(P["hbond_r0"]),
            hb_donor_types=set(lib.hb_donor_types),
            templates=dict(self.templates), fragments=dict(self.fragments),
            name=self.name)


@register_model("dreiding")
class Dreiding(ClassicalFastPath, InteratomicPotential):
    """The DREIDING rule-generated force field, bound to one topology.

    An instance binds the parameter library to one molecular topology: all
    valence terms are generated once at construction by the DREIDING rules
    (from the atom types, connectivity and bond orders), and ``forward``
    evaluates the full energy on any conformation (or batch of
    conformations) of that system. Forces and stress come from autograd via
    :class:`~xnn.common.models.outputs.ForceStressOutput`.

    Parameters
    ----------
    ffield : DreidingLibrary, DreidingForceField, str or Path, optional
        The parameter library -- ``"dreiding"`` (Lennard-Jones nonbonds, the
        default), ``"dreiding/X6"`` (exponential-6), a ``.frc`` or native
        JSON path, a parsed :class:`DreidingLibrary`, or an existing
        :class:`DreidingForceField` to *share* parameters with other models.
    topology : MolecularTopology, str or Path
        The system's topology (or a path to a topology JSON file). Its
        ``bond_orders`` drive the bond force constants and torsion rules;
        an empty list means all single bonds.
    cutoff : float, optional
        Nonbonded (van der Waals / Coulomb) cutoff in Angstrom, and the
        model's neighbor-list ``cutoff``; by default 10.0.
    switch_width : float, optional
        Width in Angstrom of a quintic switching function that takes the
        nonbonded interactions smoothly to zero at the cutoff; by default
        0.0 (plain truncation).
    charges : sequence of float, optional
        Per-atom partial charges in e (DREIDING prescribes none itself; the
        paper recommends Gasteiger charges where electrostatics matter, see
        :meth:`from_atoms`). Default: no electrostatics.
    bond_style : str, optional
        ``"harmonic"`` (default) or ``"morse"`` (the DREIDING/M variant).
    angle_style : str, optional
        ``"cosine"`` (the harmonic-cosine default, eq 10) or ``"harmonic"``
        (the theta form, eq 11). Linear centers always use eq 10'.
    hbond : bool, optional
        Include the explicit hydrogen-bond term (eq 38) on donor
        triplets; default ``True`` (it vanishes without ``H__HB`` atoms).
    hbond_cutoff : float, optional
        Donor-acceptor distance cutoff of the hydrogen-bond term, at most
        the nonbonded ``cutoff`` (the donor-acceptor pairs come from the
        neighbor list); by default the nonbonded ``cutoff``.
    hbond_angle : float, optional
        Donor-hydrogen-acceptor angle cutoff in degrees (the term counts
        only angles beyond it); by default 90.0, the paper's restriction.
    use_fast : bool or "auto", optional
        Evaluate the terms with the fused GPU kernels of
        :mod:`~xnn.ffnn.models.classical_kernels` (the same numbers to
        round-off). ``"auto"`` (default) and ``True`` take them on a CUDA
        device with Triton, in evaluation and in training; ``False``
        evaluates every term eagerly. The kernels are differentiable twice
        (force training, Hessians), and once with a cell gradient (stress);
        training with a stress loss evaluates every term eagerly.
    trainable : sequence of str or "all", optional
        Trainable parameter groups, forwarded to
        :class:`DreidingForceField` (ignored when ``ffield`` is already
        one); the extra group ``"charge"`` unfreezes this model's per-atom
        charges.
    keep_intermediates : bool, optional
        If ``True``, stash the intermediate tensors of the last evaluation
        in ``self.intermediates``. Default ``False``.

    Notes
    -----
    ``forward`` returns, besides the standard ``node_energy`` / ``energy`` /
    ``node_features`` keys, the fixed partial ``charges`` ``(N,)`` and one
    per-structure tensor per term: ``e_bond``, ``e_angle``, ``e_torsion``,
    ``e_inversion``, ``e_vdw``, ``e_coulomb``, ``e_hbond``. Every structure
    in a batch must have this topology's atom count and element sequence.
    """

    def __init__(self, ffield: Union[DreidingLibrary, DreidingForceField,
                                     str, Path] = "dreiding",
                 topology: Union[MolecularTopology, str, Path] = None, *,
                 cutoff: float = 10.0,
                 switch_width: float = 0.0,
                 charges: Optional[Sequence[float]] = None,
                 bond_style: str = "harmonic",
                 angle_style: str = "cosine",
                 hbond: bool = True,
                 hbond_cutoff: Optional[float] = None,
                 hbond_angle: float = 90.0,
                 trainable: Union[Sequence[str], str] = (),
                 keep_intermediates: bool = False,
                 use_fast: Union[bool, str] = "auto"):
        super().__init__()
        if topology is None:
            raise ValueError("Dreiding needs a topology (or use "
                             "Dreiding.from_atoms)")
        if isinstance(trainable, str) and trainable.lower() == "all":
            ff_trainable: Union[Sequence[str], str] = "all"
            self._train_charge = True
        else:
            groups = {trainable} if isinstance(trainable, str) \
                else set(trainable)
            self._train_charge = "charge" in groups
            ff_trainable = sorted(groups - {"charge"})
        self.ff = ffield if isinstance(ffield, DreidingForceField) \
            else DreidingForceField(ffield, trainable=ff_trainable)
        if bond_style not in ("harmonic", "morse"):
            raise ValueError("bond_style must be 'harmonic' or 'morse'")
        if angle_style not in ("cosine", "harmonic"):
            raise ValueError("angle_style must be 'cosine' or 'harmonic'")
        self.bond_style = bond_style
        self.angle_style = angle_style
        self.cutoff = float(cutoff)
        self.switch_width = float(switch_width)
        if not 0.0 <= self.switch_width < self.cutoff:
            raise ValueError("switch_width must satisfy "
                             "0 <= switch_width < cutoff")
        self.use_hbond = bool(hbond)
        self.hbond_cutoff = self.cutoff if hbond_cutoff is None \
            else float(hbond_cutoff)
        if self.hbond_cutoff > self.cutoff + 1e-9:
            raise ValueError(f"hbond_cutoff {self.hbond_cutoff} exceeds the nonbonded cutoff {self.cutoff}: "
                             "the hydrogen bonds come from the neighbor list")
        self.hbond_cos = math.cos(float(hbond_angle) * DEG)
        self.keep_intermediates = bool(keep_intermediates)
        self.intermediates: dict = {}
        self.node_feature_dim = 1
        self.use_fast = resolve_use_fast(use_fast)
        self._bind_topology(
            topology if isinstance(topology, MolecularTopology)
            else read_topology(topology), charges)

    @classmethod
    def from_atoms(cls, structure,
                   ffield: Union[DreidingLibrary, DreidingForceField,
                                 str, Path] = "dreiding", *,
                   charge: int = 0,
                   bonds: Optional[Sequence[Sequence[int]]] = None,
                   charges: Union[str, Sequence[float], None] = None,
                   **kwargs) -> "Dreiding":
        """Build a DREIDING model for a structure, typing it with the
        library's SMARTS templates.

        Bonding and bond orders are perceived from the coordinates (or the
        orders alone when ``bonds`` is given), atom types are assigned by
        :func:`~xnn.ffnn.common.typing.assign_atom_types`, and the topology
        follows from the perceived bonds.

        Parameters
        ----------
        structure : object
            An ``ase.Atoms``, a ``(positions, atomic_numbers)`` pair, a dict
            with ``"pos"`` / ``"atomic_numbers"``, an RDKit molecule or a
            SMILES string.
        ffield : DreidingLibrary, DreidingForceField, str or Path, optional
            The parameter library (must carry templates); default
            ``"dreiding"``.
        charge : int, optional
            Total charge of the structure (for bond-order perception).
        bonds : sequence of (int, int), optional
            Known connectivity; bond orders are then perceived, not bonds.
        charges : "gasteiger" or sequence of float, optional
            Per-atom partial charges, or ``"gasteiger"`` to compute
            Gasteiger charges with RDKit (the paper's recommendation when
            electrostatics matter). Default: none.
        **kwargs
            Forwarded to the constructor (``cutoff``, ``trainable``, ...).

        Returns
        -------
        Dreiding
            The model, bound to the derived topology.
        """
        from ..common.typing import assign_atom_types, perceive_bond_orders, perceive_bonds
        source = ffield.lib if isinstance(ffield, DreidingForceField) \
            else _as_library(ffield)
        types, mol = assign_atom_types(structure, source, charge=charge,
                                       bonds=bonds, return_mol=True)
        top = MolecularTopology.from_bonds(
            types, perceive_bonds(mol), bond_orders=perceive_bond_orders(mol))
        if isinstance(charges, str):
            if charges.lower() != "gasteiger":
                raise ValueError(f"unknown charge scheme {charges!r}; use "
                                 "'gasteiger' or explicit values")
            from rdkit.Chem import AllChem
            AllChem.ComputeGasteigerCharges(mol)
            charges = [float(a.GetDoubleProp("_GasteigerCharge"))
                       for a in mol.GetAtoms()]
        ff = ffield if isinstance(ffield, DreidingForceField) else source
        return cls(ff, top, charges=charges, **kwargs)

    # topology binding
    def _bind_topology(self, top: MolecularTopology,
                       charges: Optional[Sequence[float]]) -> None:
        """Generate every valence term from the DREIDING rules.

        Parameters
        ----------
        top : MolecularTopology
            The topology to bind (``bond_orders`` optional).
        charges : sequence of float or None
            Per-atom charges in e.

        Raises
        ------
        KeyError
            Listing every atom type the library lacks (or that lacks the
            generators its terms need).
        """
        self.topology = top
        ff = self.ff
        lib = ff.lib
        n = top.n_atoms
        dtype = torch.get_default_dtype()
        missing: list[str] = []
        for name in dict.fromkeys(top.types):
            try:
                ff.type_index(name)
                ff.check_generators(name)
            except KeyError as err:
                missing.append(str(err.args[0]))
        if missing:
            raise KeyError("the topology needs parameters the library does "
                           "not provide:\n  " + "\n  ".join(missing))

        t_idx = [ff.type_index(name) for name in top.types]
        orders = list(top.bond_orders) or [1.0] * len(top.bonds)

        def buf(name, data, shape, dt=torch.long):
            """Register an index/value buffer (possibly empty)."""
            t = torch.tensor(data, dtype=dt)
            self.register_buffer(name, t.reshape(*shape))

        buf("top_type", t_idx, (-1,))
        self.register_buffer("top_z", ff.type_z[self.top_type].clone())

        # bonds: R0 from radii, K from the bond order
        buf("bond_index", [list(b) for b in top.bonds], (-1, 2))
        self.bond_index = self.bond_index.t().contiguous()
        buf("bond_order", orders, (-1,), dtype)

        # angles: theta0 of the center; linear centers use eq 10'
        buf("angle_index", [list(a) for a in top.angles], (-1, 3))
        self.angle_index = self.angle_index.t().contiguous()
        linear = [abs(lib.theta0.get(top.types[j], 180.0) - 180.0) < 1e-8
                  for _, j, _ in top.angles]
        buf("angle_linear", linear, (-1,), torch.bool)

        # torsions: the rule per dihedral, the total barrier per central bond
        n_shared: dict[tuple[int, int], int] = {}
        for _i, j, k, _l in top.dihedrals:
            key = (min(j, k), max(j, k))
            n_shared[key] = n_shared.get(key, 0) + 1
        order_of = {b: o for b, o in zip(top.bonds, orders)}
        rule_index = {r: m for m, r in enumerate(RULE_IDS)}
        d_idx, d_rule, d_w = [], [], []
        for i, j, k, l in top.dihedrals:
            key = (min(j, k), max(j, k))
            rule = torsion_rule(lib, top.types[i], top.types[j],
                                top.types[k], top.types[l],
                                order_of.get(key, 1.0))
            if rule is None:
                continue
            d_idx.append([i, j, k, l])
            d_rule.append(rule_index[rule])
            d_w.append(1.0 / n_shared[key])
        buf("dihedral_index", d_idx, (-1, 4))
        self.dihedral_index = self.dihedral_index.t().contiguous()
        buf("dihedral_rule", d_rule, (-1,))
        buf("dihedral_weight", d_w, (-1,), dtype)

        # inversions: three axis choices per listed 3-coordinate center
        neighbors: dict[int, list[int]] = {}
        for i, j in top.bonds:
            neighbors.setdefault(i, []).append(j)
            neighbors.setdefault(j, []).append(i)
        inv = []
        for m, nb in sorted(neighbors.items()):
            if len(nb) != 3 or top.types[m] not in ff.oop_types:
                continue
            a, b, c = sorted(nb)
            inv += [[m, a, b, c], [m, b, c, a], [m, c, a, b]]
        buf("inversion_index", inv, (-1, 4))
        self.inversion_index = self.inversion_index.t().contiguous()

        # nonbonded exclusions: 1,2 and 1,3 only (1,4 in full)
        buf("excl_index", [list(p) for p in top.exclusions], (-1, 2))
        self.excl_index = self.excl_index.t().contiguous()

        # hydrogen bonds: the donor hydrogens of every heavy atom ``(n, K)``, padded with -1, and the
        # acceptors; the (donor, H, acceptor) triplets come from the neighbor list's donor-acceptor pairs
        # within the hydrogen-bond cutoff at evaluation (rebuilt from the topology: not saved)
        donors = [i for i, name in enumerate(top.types)
                  if name in ff.donor_types]
        acceptor = [bool(ff.type_acceptor[ti]) for ti in t_idx]
        hydrogens: dict[int, list[int]] = {}
        for h in donors:
            nb = neighbors.get(h, [])
            if len(nb) != 1:
                raise ValueError(
                    f"hydrogen-bond donor atom {h} ({top.types[h]}) must "
                    f"have exactly one bond, found {len(nb)}")
            hydrogens.setdefault(nb[0], []).append(h)
        width = max((len(v) for v in hydrogens.values()), default=0)
        table = [[-1] * width for _ in range(n)]
        for d, hs in hydrogens.items():
            table[d][:len(hs)] = hs
        self.register_buffer("hbond_hydrogens", torch.tensor(table, dtype=torch.long).reshape(n, width),
                             persistent=False)
        self.register_buffer("hbond_acceptor", torch.tensor(acceptor, dtype=torch.bool).reshape(n),
                             persistent=False)

        # charges
        if charges is None:
            q = torch.zeros(n, dtype=dtype)
        else:
            q = torch.as_tensor([float(c) for c in charges], dtype=dtype)
            if q.shape != (n,):
                raise ValueError(f"got {q.numel()} charges for {n} atoms")
        self.has_coulomb = bool(q.abs().sum() > 0.0) or self._train_charge
        if self._train_charge:
            self.charge = Parameter(q)
        else:
            self.register_buffer("charge", q)

    def _fast_spec(self, B: int, n: int, q: Tensor) -> dict:
        """The terms of a batch of ``B`` copies for the kernels of :mod:`.classical_kernels` (see
        :func:`~xnn.ffnn.models.classical_fast.evaluate_spec`): the harmonic or Morse bonds with their
        generated parameters, the angle forms per center, the cosine series of the torsion rules, the inversions and the
        neighbor list's pairs (the hydrogen bonds are evaluated eagerly)."""
        P, ff = self.ff.params, self.ff
        t = self.top_type.repeat(B)
        starts = torch.arange(B, device=q.device) * n

        def expand(idx: Tensor) -> Tensor:
            return (idx.unsqueeze(1) + starts.view(1, B, 1)).reshape(idx.shape[0], -1)

        bi, ai, di = expand(self.bond_index), expand(self.angle_index), expand(self.dihedral_index)
        r0 = P["radius"][t[bi[0]]] + P["radius"][t[bi[1]]] - ff.delta
        th0 = P["theta0"][t[ai[1]]]
        ka = P["angle_k"] * torch.ones_like(th0)
        lin = self.angle_linear.repeat(B)
        if self.angle_style == "harmonic":
            style, k, x0 = torch.zeros_like(th0, dtype=torch.int32), 0.5 * ka, th0
        else:
            sin2 = torch.clamp(torch.sin(th0) ** 2, min=TINY)
            style, k, x0 = torch.ones_like(th0, dtype=torch.int32), 0.5 * ka / sin2, torch.cos(th0)
        style = torch.where(lin, torch.full_like(style, 2), style)
        dr = self.dihedral_rule.repeat(B)
        kd = (0.5 * self.dihedral_weight.repeat(B) * P["torsion_v"][dr])[:, None]
        # inversions: K (1 - cos psi) / 3 at a planar center, K / (2 sin^2 psi0) (cos psi - cos psi0)^2 / 3 else
        ii = expand(self.inversion_index)
        k_i = P["oop_k"][t[ii[0]]]
        psi0 = P["oop_psi0"][t[ii[0]]]
        planar = psi0.abs() < 1e-8
        sin2_psi0 = torch.clamp(torch.sin(psi0) ** 2, min=TINY)
        k_inv = torch.where(planar, k_i, 0.5 * k_i / sin2_psi0) / 3.0
        used, compact = used_types(self, self.top_type)
        bo = self.bond_order.repeat(B)
        if self.bond_style == "morse":
            k_n, d_n = bo * P["bond_k"], bo * P["bond_d"]
            bonds = ("e_bond_node", "morse", bi, (d_n, torch.sqrt(k_n / (2.0 * d_n + TINY)), r0), None)
        else:
            bonds = ("e_bond_node", "bond", bi, (0.5 * bo * P["bond_k"], r0), None)
        return {
            "bonded": [
                bonds,
                ("e_angle_node", "angle", ai, (style, torch.where(lin, ka, k), x0), None),
                ("e_tors_node", "torsion", di, (di[1], kd, ff.rule_n[dr].to(torch.int32)[:, None],
                                                -ff.rule_sign[dr][:, None], torch.zeros_like(kd),
                                                torch.zeros_like(kd[:, 0])), dict(FORM=0, S=1, NMAX=6)),
                ("e_inv_node", "inversion", ii, (k_inv, planar.to(torch.int32), torch.cos(psi0)), None),
            ],
            "pairs14": None,
            "nonbonded": (("e_vdw_node", "e_coul_node"), compact, q if self.has_coulomb else torch.zeros_like(q),
                          self._pair_table(used), KE, self.excl_index, n, self.cutoff, self.switch_width,
                          1 if self.ff.form == "x6" else 0),
        }

    def _edges(self, data: AtomicGraph, N: int, B: int, expand) -> tuple:
        """The neighbor list's pairs once each (:func:`~.geometry.half_edges`) with their distances."""
        src, dst, vec, r2, half, included, shift = half_edges(data, self.excl_index, N // B, self)
        return src, dst, vec, torch.sqrt(r2 + TINY), half, included, shift

    def _eager_nonbonded(self, data: AtomicGraph, N: int, B: int, t: Tensor, q: Tensor, P, expand,
                         inter: dict) -> tuple:
        """The per-atom van der Waals and Coulomb energies of the neighbor list's pairs (each pair once, half
        to each atom), without the excluded pairs, switched, and the edges (see :meth:`_edges`); the
        reference."""
        src, dst, vec, r, half, included, shift = self._edges(data, N, B, expand)
        r2 = (vec * vec).sum(1)
        r0_t, d0_t = P["vdw_r0"][t], P["vdw_d0"][t]
        d0 = geometric_mean(d0_t[src], d0_t[dst])
        if self.ff.form == "x6":
            zeta_t = torch.clamp(P["x6_zeta"][t], min=6.0 + 1e-6)
            a_t = d0_t * (6.0 / (zeta_t - 6.0)) * torch.exp(zeta_t)
            b_t = d0_t * (zeta_t / (zeta_t - 6.0)) \
                * torch.clamp(r0_t, min=TINY) ** 6
            c_t = zeta_t / torch.clamp(r0_t, min=TINY)
            a_ij = geometric_mean(a_t[src], a_t[dst])
            b_ij = geometric_mean(b_t[src], b_t[dst])
            c_ij = 0.5 * (c_t[src] + c_t[dst])
            vdw_e = a_ij * torch.exp(-c_ij * r) - b_ij / (r2 ** 3 + TINY)
        else:
            if self.ff.combination == "geometric":
                r0_ij = geometric_mean(r0_t[src], r0_t[dst])
            else:
                r0_ij = 0.5 * (r0_t[src] + r0_t[dst])
            s2 = (r0_ij * r0_ij) / (r2 + TINY)
            s6 = s2 * s2 * s2
            vdw_e = d0 * (s6 * s6 - 2.0 * s6)
        coul_e = KE * q[src] * q[dst] / r if self.has_coulomb \
            else torch.zeros_like(r)

        w = included.to(r.dtype) * half
        if self.switch_width > 0.0:
            x = torch.clamp((r - (self.cutoff - self.switch_width))
                            / self.switch_width, 0.0, 1.0)
            w = w * (1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x))
        h_vdw, h_coul = 0.5 * w * vdw_e, 0.5 * w * coul_e
        e_vdw_node = scatter_sum(h_vdw, src, N) + scatter_sum(h_vdw, dst, N)
        e_coul_node = scatter_sum(h_coul, src, N) + scatter_sum(h_coul, dst, N)
        if self.keep_intermediates:
            inter.update(nb_r=r, nb_weight=w, nb_vdw=vdw_e, nb_coulomb=coul_e)
        return e_vdw_node, e_coul_node, (src, dst, vec, r, half, included, shift)

    def _pair_table(self, used: Tensor) -> Tensor:
        """The van der Waals table ``(T, T, 3)`` of the types ``used`` (see :mod:`.classical_fast`): the
        12-6 form with the library's combination rule for ``R0``, or the exponential-6 form."""
        P = self.ff.params
        r0, d0 = P["vdw_r0"][used], P["vdw_d0"][used]
        T = len(used)

        def pairs(x):
            return x[:, None].expand(-1, T), x[None, :].expand(T, -1)
        d0_ij = geometric_mean(*pairs(d0))
        if self.ff.form == "x6":
            zeta = torch.clamp(P["x6_zeta"][used], min=6.0 + 1e-6)
            a = d0 * (6.0 / (zeta - 6.0)) * torch.exp(zeta)
            b = d0 * (zeta / (zeta - 6.0)) * torch.clamp(r0, min=TINY) ** 6
            c = zeta / torch.clamp(r0, min=TINY)
            ci, ck = pairs(c)
            return x6_table(geometric_mean(*pairs(a)), geometric_mean(*pairs(b)), 0.5 * (ci + ck))
        r0_ij = geometric_mean(*pairs(r0)) if self.ff.combination == "geometric" else 0.5 * sum(pairs(r0))
        return lj_table(d0_ij, 2.0, r0_ij)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # the hydrogen-bond triplets of earlier versions were saved; they now come from the neighbor list
        state_dict.pop(prefix + "hbond_index", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # forward
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Evaluate the DREIDING energy on a (batched) atomic graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph; every structure must be a conformation of
            the bound topology (same atom count and element sequence), and
            the neighbor list must have been built with this model's
            ``cutoff``.

        Returns
        -------
        dict of str to Tensor
            ``node_energy`` ``(N,)``, ``energy`` ``(B,)``, ``charges``
            ``(N,)``, ``node_features`` ``(N, 1)`` and the per-structure
            energy decomposition (see the class docstring).

        Raises
        ------
        ValueError
            If a structure in the batch does not match the bound topology.
        """
        N = data.num_nodes
        B = data.num_graphs
        n = int(self.top_type.shape[0])
        device = data.pos.device
        P = self.ff.params
        inter: dict = {}

        if not bool((data.n_atoms == n).all()):
            raise ValueError(
                f"every structure must have the bound topology's {n} atoms "
                f"(got {data.n_atoms.tolist()})")
        if not bool((data.atomic_numbers == self.top_z.repeat(B)).all()):
            raise ValueError("atomic numbers do not match the bound "
                             "topology's element sequence")

        t = self.top_type.repeat(B)
        starts = torch.arange(B, device=device) * n

        def expand(idx: Tensor) -> Tensor:
            """Tile per-structure index rows ``(k, m)`` to ``(k, B*m)``."""
            return (idx.unsqueeze(1) + starts.view(1, B, 1)).reshape(
                idx.shape[0], -1)

        def tile(v: Tensor) -> Tensor:
            """Tile per-interaction rows ``(m,)`` to ``(B*m,)``."""
            return v.repeat(B)

        q = self.charge.repeat(B)
        fast = wants_fast(self, self.use_fast, data, q)
        if fast:
            spec = self._fast_spec(B, n, q)
            ev = evaluate_spec(data, {"bonded": spec["bonded"]}, self)
        if fast:
            e_bond_node, e_angle_node, e_tors_node = ev["e_bond_node"], ev["e_angle_node"], ev["e_tors_node"]
            e_inv_node = ev["e_inv_node"]
        else:
            bi, ai = expand(self.bond_index), expand(self.angle_index)
            di, ii = expand(self.dihedral_index), expand(self.inversion_index)
            # the image shifts of every bonded vector from one search
            sb, sa, sd, si = (([None] * 3) if s is None else s for s in image_shifts(
                data, [(bi, "bond"), (ai, "angle"), (di, "torsion"), (ii, "inversion")]))

            # bonds (eqs 4-9)
            bo = tile(self.bond_order)
            bt0, bt1 = t[bi[0]], t[bi[1]]
            bvec = pair_vectors(data, bi[0], bi[1], sb[0])
            br = torch.sqrt((bvec * bvec).sum(1) + TINY)
            r0 = P["radius"][bt0] + P["radius"][bt1] - self.ff.delta
            k_n = bo * P["bond_k"]
            if self.bond_style == "morse":
                d_n = bo * P["bond_d"]
                alpha = torch.sqrt(k_n / (2.0 * d_n + TINY))
                e_bond = d_n * (torch.exp(-alpha * (br - r0)) - 1.0) ** 2
            else:
                e_bond = 0.5 * k_n * (br - r0) ** 2
            e_bond_node = scatter_sum(0.5 * e_bond, bi[0], N) \
                + scatter_sum(0.5 * e_bond, bi[1], N)

            # angles (eqs 10-12)
            lin = tile(self.angle_linear)
            u = pair_vectors(data, ai[1], ai[0], sa[0])
            v = pair_vectors(data, ai[1], ai[2], sa[1])
            cos_th = (u * v).sum(1) / torch.sqrt(
                (u * u).sum(1) * (v * v).sum(1) + TINY)
            th0 = P["theta0"][t[ai[1]]]
            ka = P["angle_k"]
            if self.angle_style == "harmonic":
                bound = min(0.9999999999,
                            1.0 - 4.0 * torch.finfo(cos_th.dtype).eps)
                theta = torch.arccos(torch.clamp(cos_th, -bound, bound))
                e_nonlin = 0.5 * ka * (theta - th0) ** 2
            else:
                sin2 = torch.clamp(torch.sin(th0) ** 2, min=TINY)
                e_nonlin = 0.5 * (ka / sin2) * (cos_th - torch.cos(th0)) ** 2
            e_lin = ka * (1.0 + cos_th)
            e_angle = torch.where(lin, e_lin, e_nonlin)
            e_angle_node = scatter_sum(e_angle, ai[1], N)

            # torsions (eqs 13-23)
            dr = tile(self.dihedral_rule)
            dw = tile(self.dihedral_weight)
            c1 = dihedral_cos(data, di, sd)
            t2 = 2.0 * c1 * c1 - 1.0
            t3 = c1 * (4.0 * c1 * c1 - 3.0)
            t6 = 2.0 * t3 * t3 - 1.0
            n_d = self.ff.rule_n[dr]
            tn = torch.where(n_d == 2, t2, torch.where(n_d == 3, t3, t6))
            e_tors = dw * 0.5 * P["torsion_v"][dr] \
                * (1.0 - self.ff.rule_sign[dr] * tn)
            e_tors_node = scatter_sum(e_tors, di[1], N)

            # inversions (eq 28)
            it = t[ii[0]]
            rij = pair_vectors(data, ii[0], ii[1], si[0])
            rik = pair_vectors(data, ii[0], ii[2], si[1])
            ril = pair_vectors(data, ii[0], ii[3], si[2])
            nrm = torch.cross(rij, rik, dim=1)
            nrm = nrm / torch.sqrt((nrm * nrm).sum(1, keepdim=True) + TINY)
            ril = ril / torch.sqrt((ril * ril).sum(1, keepdim=True) + TINY)
            sin_psi = (nrm * ril).sum(1)
            cos_psi = torch.sqrt(torch.clamp(1.0 - sin_psi * sin_psi, min=TINY))
            k_i = P["oop_k"][it]
            psi0 = P["oop_psi0"][it]
            planar = psi0.abs() < 1e-8
            sin2_psi0 = torch.clamp(torch.sin(psi0) ** 2, min=TINY)
            e_pl = k_i * (1.0 - cos_psi)
            e_np = 0.5 * (k_i / sin2_psi0) * (cos_psi - torch.cos(psi0)) ** 2
            e_inv = torch.where(planar, e_pl, e_np) / 3.0
            e_inv_node = scatter_sum(e_inv, ii[0], N)

        # nonbonded: neighbor list minus 1,2 / 1,3 exclusions
        edges = None
        if fast:
            names, compact, qq, table, ke, excl, n_local, cutoff, width, form = spec["nonbonded"]
            e_vdw_node, e_coul_node = fast_nonbonded(data, n_local, B, compact, qq, table, ke, excl, cutoff, width,
                                                     form, self, needs_twice(self, data))
        else:
            e_vdw_node, e_coul_node, edges = self._eager_nonbonded(data, N, B, t, q, P, expand, inter)

        # hydrogen bonds (eq 38)
        if self.use_hbond and self.hbond_hydrogens.shape[1] > 0:
            src, dst, vec, r, _, included, shift = edges if edges is not None else self._edges(data, N, B, expand)
            # the donor-acceptor pairs of the neighbor list in either direction (the donor heavy atom first;
            # an own image's two directions are both in the list), not excluded, within the cutoff, once per
            # donor hydrogen of the heavy atom
            K = int(self.hbond_hydrogens.shape[1])
            hyd = self.hbond_hydrogens.repeat(B, 1)
            hyd = torch.where(hyd >= 0, hyd + starts.repeat_interleave(n)[:, None], hyd)
            acceptor = self.hbond_acceptor.repeat(B)
            near = included & (r.detach() < self.hbond_cutoff)
            fwd = ((hyd[src, 0] >= 0) & acceptor[dst] & near).nonzero()[:, 0]
            rev = ((hyd[dst, 0] >= 0) & acceptor[src] & near & (src != dst)).nonzero()[:, 0]
            donor = torch.cat([src[fwd], dst[rev]])
            accept = torch.cat([dst[fwd], src[rev]])
            h_all = hyd[donor].reshape(-1)
            keep = h_all >= 0
            e_da = torch.arange(donor.shape[0], device=device).repeat_interleave(K)[keep]
            h_i = h_all[keep]
            hb = torch.stack([donor[e_da], h_i, accept[e_da]])
            # the donor-acceptor vectors take the list's image (a small cell has several within the cutoff)
            s_da = None if shift is None else torch.cat([shift[fwd], -shift[rev]])[e_da]
        if fast and self.use_hbond and self.hbond_hydrogens.shape[1] > 0:
            # the triplets' energies from the kernel (the vectors by the minimum image)
            M = int(hb.shape[1])
            s_hd = pair_image_shifts(data, hb[1], hb[0])
            shifts = None if s_hd is None else torch.stack([s_da.to(s_hd.dtype), s_hd])
            e_hb_node = fast_bonded(data, hb, "hbond", (P["hbond_d0"].expand(M), P["hbond_r0"].expand(M)),
                                    shifts, None, (self.hbond_cutoff ** 2, self.hbond_cos),
                                    twice=needs_twice(self, data))
        elif self.use_hbond and self.hbond_hydrogens.shape[1] > 0:
            vec_da = torch.cat([vec[fwd], -vec[rev]])[e_da]
            r_da = torch.cat([r[fwd], r[rev]])[e_da]
            v_hd = pair_vectors(data, hb[1], hb[0])
            v_ha = v_hd + vec_da
            cos_dha = (v_hd * v_ha).sum(1) / torch.sqrt(
                (v_hd * v_hd).sum(1) * (v_ha * v_ha).sum(1) + TINY)
            rho2 = (P["hbond_r0"] / r_da) ** 2
            rho10 = rho2 ** 5
            e_hb = P["hbond_d0"] * (5.0 * rho10 * rho2 - 6.0 * rho10) \
                * cos_dha ** 4
            mask = ((r_da < self.hbond_cutoff)
                    & (cos_dha < self.hbond_cos)).to(e_hb.dtype)
            e_hb = e_hb * mask
            e_hb_node = scatter_sum(e_hb, hb[1], N)
        else:
            e_hb_node = torch.zeros_like(e_vdw_node)

        node_energy = (e_bond_node + e_angle_node + e_tors_node
                       + e_inv_node + e_vdw_node + e_coul_node + e_hb_node)
        energy = self.aggregate_energy(node_energy, data)

        if self.keep_intermediates:
            inter.update(bond_r=br, bond_r0=r0, e_bond=e_bond,
                         angle_cos=cos_th, e_angle=e_angle,
                         dihedral_cos=c1, e_torsion=e_tors,
                         inversion_sin=sin_psi, e_inversion=e_inv,
                         type_index=t)
            if self.use_hbond and self.hbond_hydrogens.shape[1] > 0:
                inter.update(hb_index=hb, hb_r=r_da, hb_cos=cos_dha, e_hbond=e_hb)
            self.intermediates = inter

        return {
            "node_energy": node_energy,
            "energy": energy,
            "charges": q,
            "node_features": q.unsqueeze(1),
            "e_bond": self.aggregate_energy(e_bond_node, data),
            "e_angle": self.aggregate_energy(e_angle_node, data),
            "e_torsion": self.aggregate_energy(e_tors_node, data),
            "e_inversion": self.aggregate_energy(e_inv_node, data),
            "e_vdw": self.aggregate_energy(e_vdw_node, data),
            "e_coulomb": self.aggregate_energy(e_coul_node, data),
            "e_hbond": self.aggregate_energy(e_hb_node, data),
        }

    # conveniences
    def export_library(self) -> DreidingLibrary:
        """Export the current parameters as a :class:`DreidingLibrary`.

        Returns
        -------
        DreidingLibrary
            See :meth:`DreidingForceField.export_library`.
        """
        return self.ff.export_library()

    @property
    def masses(self) -> Tensor:
        """Tensor : Per-atom masses (u) of the bound topology."""
        return self.ff.type_mass[self.top_type]

    @classmethod
    def from_config(cls, cfg) -> "Dreiding":
        """Construct a :class:`Dreiding` model from a core model config.

        Core field: ``cfg.cutoff`` is the nonbonded cutoff. Everything else
        is read from ``cfg.extra``: ``ffield`` (``"dreiding"`` when absent;
        ``"dreiding/X6"``, a ``.frc`` path or a native JSON path) and either
        ``topology`` (path to a topology JSON file) or ``types`` + ``bonds``
        (+ optional ``bond_orders``) inline. Optional: ``charges`` (a list),
        ``switch_width``, ``bond_style``, ``angle_style``, ``hbond``,
        ``hbond_cutoff``, ``hbond_angle``, ``trainable``. Alternative
        spellings used by other MD packages are translated by
        :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        Dreiding
            The model.

        Raises
        ------
        ValueError
            If the topology specification is missing.
        """

        from xnn.common.config.options import model_options

        extra = model_options(cfg)

        def listy(value):
            """Coerce a possibly stringified list to a Python value."""
            if isinstance(value, str) and value.lstrip().startswith(("[", "(")):
                return ast.literal_eval(value)
            return value

        ffield = extra.get("ffield") or "dreiding"
        topo_path = extra.get("topology")
        if topo_path is not None:
            topology: MolecularTopology = read_topology(topo_path)
        else:
            types = listy(extra.get("types"))
            bonds = listy(extra.get("bonds"))
            if types is None or bonds is None:
                raise ValueError("Dreiding needs a topology: set "
                                 "model.topology to a topology JSON path, or "
                                 "give model.types and model.bonds inline")
            topology = MolecularTopology.from_bonds(
                types, bonds, bond_orders=listy(extra.get("bond_orders")))

        kwargs = {}
        for key in ("bond_style", "angle_style"):
            if extra.get(key) is not None:
                kwargs[key] = str(extra[key])
        for key in ("hbond_cutoff", "hbond_angle"):
            if extra.get(key) is not None:
                kwargs[key] = float(extra[key])
        if extra.get("hbond") is not None:
            kwargs["hbond"] = bool(extra["hbond"])
        return cls(ffield, topology, cutoff=float(cfg.cutoff),
                   switch_width=float(extra.get("switch_width") or 0.0),
                   charges=listy(extra.get("charges")),
                   trainable=listy(extra.get("trainable")) or (), **kwargs)
