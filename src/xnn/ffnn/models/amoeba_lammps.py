"""Inputs for the AMOEBA package of LAMMPS (``pair_style amoeba`` and ``hippo``) from an AMOEBA or HIPPO model.

LAMMPS reads Tinker's parameter and coordinate formats through its ``tools/tinker/tinker2lmp.py`` converter,
which writes the LAMMPS data file. :func:`write_lammps_amoeba_input` writes what the converter and LAMMPS read:
the model's current parameters as a Tinker parameter file in the form LAMMPS takes, the coordinates, a key file
with the model's cutoffs, Ewald coefficients and meshes, and the LAMMPS input script, which names the converter
command in its header.

The LAMMPS package has no charge flux and no exchange polarization: a library with either is refused unless the
call names it in ``drop``, which leaves it out with a warning (LAMMPS then evaluates another model). Its HIPPO
repulsion takes the size parameter ``K / 8`` of the one of Tinker 8 and the 2021 paper (the Slater normalisation
folded into ``K``). It also rejects the argument-less ``d-equals-p`` keyword, which is not written. A library with
the terms of ions (``ionct``, ``mbct``, ``ionpair``) has no LAMMPS counterpart and is refused. A library whose
type and class names are integers keeps them as the numbers of the parameter file.
"""
import copy
import warnings
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .amoeba_ewald import ewald_coefficient, pme_grid_size
from .amoebalib import write_tinker_prm, write_tinker_xyz

# the cutoff of every term for a cluster: the model sums all its pairs, LAMMPS needs finite cutoffs
CLUSTER_CUTOFF = 50.0
# the terms LAMMPS lacks that an export may leave out on request
DROPPABLE = ("cflux", "exchpol")


def _key(core, hippo: bool, cell) -> list:
    """The keywords of the model's settings: Ewald coefficients, meshes, orders and the terms' cutoffs and
    tapers in a cell, the cluster cutoff of every term otherwise, and the induction's tolerance."""
    lines = []
    if cell is None:
        cut = CLUSTER_CUTOFF
        lines += [f"cutoff {cut}", f"vdw-cutoff {cut}", f"mpole-cutoff {cut}"]
        if hippo:
            lines += [f"repulsion-cutoff {cut}", f"dispersion-cutoff {cut}", f"ctrn-cutoff {cut}"]
    else:
        lengths = [float(np.linalg.norm(v)) for v in cell]
        rc = core.ewald_cutoff if core.ewald_cutoff is not None else min(7.0, core.cutoff)
        alpha_e = core.ewald_alpha if core.ewald_alpha is not None else ewald_coefficient(rc)
        alpha_p = core.pewald_alpha if core.pewald_alpha is not None else alpha_e
        grid = list(core.pme_grid) if core.pme_grid is not None else pme_grid_size(lengths)
        lines += ["ewald", f"ewald-cutoff {rc!r}", f"ewald-alpha {alpha_e!r}", f"pewald-alpha {alpha_p!r}",
                  "pme-grid " + " ".join(str(int(k)) for k in grid), f"pme-order {core.pme_order}",
                  f"ppme-order {core.ppme_order or core.pme_order}"]
        if hippo:
            rc_d = core.dispersion_cutoff if core.dispersion_cutoff is not None else rc
            alpha_d = core.dewald_alpha if core.dewald_alpha is not None else ewald_coefficient(rc_d)
            dgrid = list(core.dpme_grid) if core.dpme_grid is not None else pme_grid_size(lengths, density=0.8)
            lines += ["dewald", f"dewald-cutoff {rc_d!r}", f"dewald-alpha {alpha_d!r}",
                      "dpme-grid " + " ".join(str(int(k)) for k in dgrid), f"dpme-order {core.dpme_order}"]
            for key, value in (("repulsion-cutoff", core.repulsion_cutoff), ("ctrn-cutoff", core.chgtrn_cutoff)):
                cut = value if value is not None else core.cutoff
                lines.append(f"{key} {cut!r}")
            if core.taper is not None:
                lines.append(f"taper {core.taper!r}")
        else:
            vdw = core.vdw_cutoff if core.vdw_cutoff is not None else core.cutoff
            lines.append(f"vdw-cutoff {vdw!r}")
            if core.taper is not None:
                lines.append(f"vdw-taper {core.taper!r}")
    lines.append(f"polar-eps {core.polar_eps!r}")
    if core.polarization == "direct":
        lines.append("polarization direct")
    return lines


def _script(name: str, hippo: bool, periodic: bool, pitors: bool, tortor: bool) -> list:
    """The LAMMPS commands of the AMOEBA package for the files of :func:`write_lammps_amoeba_input`."""
    style = "hippo" if hippo else "amoeba"
    lines = ["units real", f"boundary {'p p p' if periodic else 's s s'}", "atom_modify sort 0 0.0",
             "atom_style amoeba", "bond_style class2", "angle_style amoeba", "dihedral_style fourier",
             "improper_style amoeba",
             "fix amtype all property/atom i_amtype ghost yes",
             "fix extra all property/atom i_amgroup d_redID d_pval ghost yes",
             "fix extra2 all property/atom i_polaxe d2_xyzaxis 3"]
    read = [f'read_data {name}.data fix amtype NULL "Tinker Types"']
    if pitors:
        lines += ["fix pit all amoeba/pitorsion", "fix_modify pit energy yes"]
        read += ['fix pit "pitorsion types" "PiTorsion Coeffs"', "fix pit pitorsions PiTorsions"]
    if tortor:
        lines += [f"fix bit all amoeba/bitorsion {name}.bitorsion", "fix_modify bit energy yes"]
        read += ["fix bit bitorsions BiTorsions"]
    lines.append(" &\n    ".join(read))
    lines += [f"pair_style {style}", f"pair_coeff * * {name}.prm {name}.key",
              "special_bonds lj/coul 0.5 0.5 0.5 one/five yes",
              f"compute terms all pair {style}",
              "thermo_style custom step pe ebond eangle edihed eimp epair c_terms[*]",
              "thermo_modify format float %20.12f", "run 0"]
    return lines


def write_lammps_amoeba_input(model, pos, cell=None, directory: Union[str, Path] = ".",
                              name: str = "system", drop: Sequence[str] = ()) -> Path:
    """Write the inputs of an AMOEBA or HIPPO model for LAMMPS's AMOEBA package into ``directory``.

    The files are ``<name>.prm`` (the model's current parameters, see the module notes), ``<name>.xyz``
    (Tinker coordinates), ``<name>.key`` (the model's settings) and ``in.<name>`` (the LAMMPS commands, ending
    in ``run 0`` with the energy terms in the thermo output). Make the data file with LAMMPS's converter first;
    the script's header names the command.

    Parameters
    ----------
    model : nn.Module
        An :class:`~xnn.ffnn.models.Amoeba` or :class:`~xnn.ffnn.models.Hippo` model (or a wrapper of one).
    pos : array_like
        Positions ``(n, 3)`` in Angstrom, in the order of the bound topology.
    cell : array_like, optional
        The cell ``(3, 3)`` in Angstrom, lattice vectors as rows; ``None`` for a cluster.
    directory : str or Path, optional
        Where the files go (created if needed).
    name : str, optional
        The stem of the file names.
    drop : sequence of str, optional
        Terms of the library that LAMMPS lacks and that may be left out, with
        a warning: ``"cflux"`` (the charge flux), ``"exchpol"`` (the exchange
        polarization). A library with one of them is refused otherwise.

    Returns
    -------
    Path
        The path of the input script.

    Raises
    ------
    ValueError
        For a library with ion terms, or with charge flux or exchange
        polarization not in ``drop``.
    """
    from .amoeba import Amoeba
    core = getattr(model, "model", model)
    if not isinstance(core, Amoeba):
        raise TypeError(f"no LAMMPS AMOEBA inputs for {type(core).__name__}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(pos.detach().cpu() if hasattr(pos, "detach") else pos, float).reshape(-1, 3)
    cell = None if cell is None else np.asarray(cell.detach().cpu() if hasattr(cell, "detach") else cell,
                                                 float).reshape(3, 3)
    lib = copy.deepcopy(core.export_library())
    if lib.ionct or lib.mbct or lib.ionpair:
        raise ValueError("LAMMPS's AMOEBA package has none of the ion terms (ionct, mbct, ionpair) of this library")
    hippo = lib.model == "hippo"
    drop = (drop,) if isinstance(drop, str) else tuple(drop)
    unknown = set(drop) - set(DROPPABLE)
    if unknown:
        raise ValueError(f"drop takes {DROPPABLE} (got {sorted(unknown)})")
    for term, tables, what in (("cflux", ("bndcflux", "angcflux"), "charge flux"),
                               ("exchpol", ("exchpol",), "exchange polarization")):
        if not any(getattr(lib, t) for t in tables):
            continue
        if term not in drop:
            raise ValueError(f"LAMMPS's AMOEBA package has no {what} ({term}) of this library: pass "
                             f"drop=({term!r},) to write the model without it")
        warnings.warn(f"the library's {what} is not written: LAMMPS evaluates the model without it", stacklevel=2)
        for t in tables:
            setattr(lib, t, {})
    if hippo:
        for row in lib.repulsion.values():
            row["k"] = row["k"] / 8.0
    prm = directory / f"{name}.prm"
    # a library numbered by its type and class names keeps those numbers
    try:
        maps = write_tinker_prm(lib, prm, keep_numbers=True)
    except ValueError:
        maps = write_tinker_prm(lib, prm)
    prm.write_text("\n".join(ln for ln in prm.read_text().splitlines() if not ln.startswith("d-equals-p")) + "\n")
    top = core.topology
    # the converter reads no lattice line: the cell goes to its -pbc option
    write_tinker_xyz(directory / f"{name}.xyz", pos, top.types, top.bonds, maps["types"], title=name)
    (directory / f"{name}.key").write_text("\n".join(_key(core, hippo, cell)) + "\n")
    convert = (f"python tinker2lmp.py -xyz {name}.xyz -{'hippo' if hippo else 'amoeba'} {name}.prm -data {name}.data"
               + ("" if cell is None else " -pbc " + " ".join(f"{float(x)!r}" for x in np.linalg.norm(cell, axis=1)))
               + (f" -bitorsion {name}.bitorsion" if lib.tortor_types else ""))
    header = ["# LAMMPS AMOEBA package input written by xnn. Make the data file first with LAMMPS's converter",
              f"# (tools/tinker/tinker2lmp.py): {convert}"]
    if cell is not None and not np.allclose(cell, np.diag(np.diag(cell))):
        header.append("# The cell is not orthorhombic: the converter's -pbc option takes the three edge lengths only.")
    lines = header + _script(name, hippo, cell is not None, bool(lib.pitors_types), bool(lib.tortor_types))
    script = directory / f"in.{name}"
    script.write_text("\n".join(lines) + "\n")
    return script


__all__ = ["write_lammps_amoeba_input", "CLUSTER_CUTOFF"]
