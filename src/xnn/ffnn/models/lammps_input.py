"""LAMMPS inputs of the classical force fields (OPLS, AMBER, CHARMM, DREIDING, ReaxFF): a data file and the
commands that evaluate the model's energy function with LAMMPS's own styles, from the model's resolved
parameters.

The inputs carry the bound topology and the per-term parameter rows of the model (trained parameters
included). The charges are scaled by ``sqrt(C_model / 332.06371)``, so LAMMPS's Coulomb constant reproduces
the model's energies. The styles:

- OPLS: ``lj/cut/coul/cut`` with geometric mixing, harmonic bonds and angles, ``dihedral opls`` (the series
  constant ``V0`` has no LAMMPS term and is returned as an energy offset), ``improper cvff`` with ``d = -1``,
  ``n = 2``.
- AMBER: ``lj/cut/coul/cut`` with arithmetic mixing, ``dihedral fourier``, ``improper cvff``.
- CHARMM: ``lj/charmm/coul/charmm`` switched 1e-6 A below the cutoff (plain truncation), ``angle charmm``
  with the Urey-Bradley terms, ``dihedral charmm`` carrying the 1,4 pairs, ``improper harmonic``.
- DREIDING: ``lj/cut`` or ``buck`` with ``coul/cut``, ``hbond/dreiding/lj`` overlaid when the model has
  hydrogen bonds, ``angle cosine/squared`` (``cosine`` at linear centers), ``dihedral harmonic``, ``improper
  umbrella``; harmonic bonds (``morse`` for Morse bonds).
- ReaxFF (classical fields with ``conventions="lammps"``): ``pair_style reaxff`` with ``fix qeq/reaxff`` and
  the model's field written as an ``ffield`` file (:func:`~xnn.ffnn.models.ffield.write_ffield`); LAMMPS
  solves the charges for a neutral system to its tolerance.

A model with a switched cutoff (``switch_width > 0``) has no LAMMPS twin: its quintic switch is not one of
LAMMPS's. Cells must be orthorhombic; a structure without a cell becomes a shrink-wrapped box (at least its
extent plus the cutoff).
"""
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

LAMMPS_COULOMB = 332.06371          # LAMMPS's Coulomb constant in real units, kcal/mol A / e^2
_DEG = math.pi / 180.0
_SECTIONS = (("bond", "Bonds"), ("angle", "Angles"), ("dihedral", "Dihedrals"), ("improper", "Impropers"))


def _r(x, nd: int = 14) -> float:
    return round(float(x), nd)


def _box(pos: np.ndarray, cell: Optional[np.ndarray], pad: float) -> tuple:
    """The positions in the box, the box bounds ``(3, 2)`` and the boundary flags: a cluster's box is
    shrink-wrapped with a minimum of its extent plus ``pad`` (the cutoff) on every side, so LAMMPS's
    neighbor bins stay larger than the cutoff."""
    if cell is None:
        lo, hi = pos.min(0) - pad, pos.max(0) + pad
        return pos, np.stack([lo, hi], 1), "m m m"
    cell = np.asarray(cell, float).reshape(3, 3)
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-12:
        raise ValueError("the LAMMPS inputs take an orthorhombic cell (a diagonal cell matrix)")
    lengths = np.diag(cell)
    return pos % lengths, np.stack([np.zeros(3), lengths], 1), "p p p"


def _data(title: str, pos: np.ndarray, bounds: np.ndarray, masses: dict, types: list, charges: np.ndarray,
          sections: dict, counts: dict) -> str:
    """The data file (``atom_style full``, one molecule): ``masses`` {type id: mass}, ``types`` the type id
    of each atom, ``sections`` {name: rows of ints}, ``counts`` {name: number of types}."""
    lines = [title, "", f"{len(types)} atoms"] + [f"{len(sections.get(s, []))} {s.lower()}" for _, s in _SECTIONS]
    lines += ["", f"{len(masses)} atom types"] + [f"{counts.get(s, 0)} {k} types" for k, s in _SECTIONS] + [""]
    lines += [f"{float(bounds[a, 0])!r} {float(bounds[a, 1])!r} {x}lo {x}hi" for a, x in enumerate("xyz")]
    lines += ["", "Masses", ""] + [f"{t} {m:.6f}" for t, m in masses.items()]
    lines += ["", "Atoms # full", ""]
    lines += [f"{i + 1} 1 {t} {q:.15g} {x:.14f} {y:.14f} {z:.14f}"
              for i, (t, q, (x, y, z)) in enumerate(zip(types, charges, pos))]
    for _, s in _SECTIONS:
        rows = sections.get(s, [])
        if rows:
            lines += ["", s, ""] + [" ".join(str(v) for v in (m + 1, *row)) for m, row in enumerate(rows)]
    return "\n".join(lines) + "\n"


def _styles(sections: dict, styles: dict) -> list:
    """The style commands of the term classes present (LAMMPS rejects a style without terms)."""
    return [f"{kind}_style {styles[kind]}" for kind, s in _SECTIONS if sections.get(s)]


def _check_switch(model) -> None:
    if float(getattr(model, "switch_width", 0.0)) > 0.0:
        raise ValueError("a switched cutoff (switch_width > 0) has no LAMMPS style; use switch_width=0")


def _opls(model, pos, bounds, boundary):
    from .opls import KE
    from .oplslib import KCAL_TO_EV
    ev = 1.0 / KCAL_TO_EV
    ff = model.ff
    P = {k: v.detach().double().cpu() for k, v in ff.params.items()}
    t_atom = model.top_type.tolist()
    used = list(dict.fromkeys(t_atom))
    at_id = {t: i + 1 for i, t in enumerate(used)}
    q = ff.charges.detach().double().cpu()[model.top_type.cpu()].numpy() * math.sqrt(KE * ev / LAMMPS_COULOMB)

    def rows(index, types, key):
        out, table = [], {}
        for m in range(index.shape[1]):
            k = key(int(types[m]))
            table.setdefault(k, len(table) + 1)
            out.append((table[k], *[int(a) + 1 for a in index[:, m]]))
        return out, table

    sections, tables = {}, {}
    sections["Bonds"], tables["bond"] = rows(model.bond_index, model.bond_type,
                                             lambda t: (_r(P["bond_k"][t] * ev), _r(P["bond_r0"][t])))
    sections["Angles"], tables["angle"] = rows(model.angle_index, model.angle_type,
                                               lambda t: (_r(P["angle_k"][t] * ev), _r(P["angle_theta0"][t] / _DEG)))
    sections["Dihedrals"], tables["dihedral"] = rows(
        model.dihedral_index, model.dihedral_type,
        lambda t: tuple(_r(P["dihedral_v"][t, s] * ev) for s in (1, 2, 3, 4)))
    sections["Impropers"], tables["improper"] = rows(model.improper_index, model.improper_type,
                                                     lambda t: (_r(0.5 * P["improper_v2"][t] * ev),))
    v0 = float(P["dihedral_v"][model.dihedral_type.cpu(), 0].sum()) * ev
    data = _data("xnn OPLS", pos, bounds, {at_id[t]: float(ff.type_mass[t]) for t in used},
                 [at_id[t] for t in t_atom], q, sections, {s: len(tables[k]) for k, s in _SECTIONS})
    cmds = ["units real", "atom_style full", f"boundary {boundary}",
            f"pair_style lj/cut/coul/cut {float(model.cutoff)!r}", "pair_modify mix geometric",
            f"special_bonds lj 0.0 0.0 {model.fudge_lj!r} coul 0.0 0.0 {model.fudge_qq!r}"]
    cmds += _styles(sections, {"bond": "harmonic", "angle": "harmonic", "dihedral": "opls", "improper": "cvff"})
    cmds.append("read_data DATAFILE")
    for t in used:
        cmds.append(f"pair_coeff {at_id[t]} {at_id[t]} {float(P['epsilon'][t]) * ev:.15g} {float(P['sigma'][t]):.15g}")
    cmds += [f"bond_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tables["bond"].items()]
    cmds += [f"angle_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tables["angle"].items()]
    cmds += [f"dihedral_coeff {i} " + " ".join(f"{v:.15g}" for v in k) for k, i in tables["dihedral"].items()]
    cmds += [f"improper_coeff {i} {k[0]:.15g} -1 2" for k, i in tables["improper"].items()]
    return data, cmds, v0


def _amber(model, pos, bounds, boundary):
    from .amber import KE_KCAL
    from .amberlib import KCAL_TO_EV
    ev = 1.0 / KCAL_TO_EV
    ff, top = model.ff, model.topology
    P = {k: v.detach().double().cpu() for k, v in ff.params.items()}
    used = list(dict.fromkeys(top.types))
    at_id = {t: i + 1 for i, t in enumerate(used)}
    ti = {t: ff.type_index(t) for t in used}
    q = model.charges.detach().double().cpu().numpy() * math.sqrt(KE_KCAL / LAMMPS_COULOMB)
    sections = {s: [] for _, s in _SECTIONS}
    tb, ta, td, tm = {}, {}, {}, {}
    for m, (i, j) in enumerate(top.bonds):
        k = int(model.bond_type[m])
        key = (_r(P["bond_k"][k] * ev), _r(P["bond_r0"][k]))
        tb.setdefault(key, len(tb) + 1)
        sections["Bonds"].append((tb[key], i + 1, j + 1))
    for m, (i, j, k) in enumerate(top.angles):
        t = int(model.angle_type[m])
        key = (_r(P["angle_k"][t] * ev), _r(P["angle_theta0"][t] / _DEG))
        ta.setdefault(key, len(ta) + 1)
        sections["Angles"].append((ta[key], i + 1, j + 1, k + 1))
    for m, (i, j, k, l) in enumerate(top.dihedrals):
        t = int(model.dihedral_type[m])
        terms = tuple((_r(P["dihedral_k"][t, s] * ev), int(ff.dihedral_n[t, s]),
                       _r(math.degrees(math.atan2(float(ff.dihedral_sin[t, s]), float(ff.dihedral_cos[t, s])))))
                      for s in range(P["dihedral_k"].shape[1]) if float(P["dihedral_k"][t, s]) != 0.0)
        if not terms:
            continue
        td.setdefault(terms, len(td) + 1)
        sections["Dihedrals"].append((td[terms], i + 1, j + 1, k + 1, l + 1))
    for m, (i, j, k, l) in enumerate(model.impropers):
        t = int(model.improper_type[m])
        # cvff: K [1 + d cos(n phi)] with d = cos(gamma)
        key = (_r(P["improper_k"][t, 0] * ev), int(round(float(ff.improper_cos[t, 0]))), int(ff.improper_n[t, 0]))
        tm.setdefault(key, len(tm) + 1)
        sections["Impropers"].append((tm[key], i + 1, j + 1, k + 1, l + 1))
    counts = {"Bonds": len(tb), "Angles": len(ta), "Dihedrals": len(td), "Impropers": len(tm)}
    data = _data("xnn AMBER", pos, bounds, {at_id[t]: float(ff.type_mass[ti[t]]) for t in used},
                 [at_id[t] for t in top.types], q, sections, counts)
    cmds = ["units real", "atom_style full", f"boundary {boundary}",
            f"pair_style lj/cut/coul/cut {float(model.cutoff)!r}", "pair_modify mix arithmetic",
            f"special_bonds lj 0.0 0.0 {model.fudge_lj!r} coul 0.0 0.0 {model.fudge_qq!r}"]
    cmds += _styles(sections, {"bond": "harmonic", "angle": "harmonic", "dihedral": "fourier", "improper": "cvff"})
    cmds.append("read_data DATAFILE")
    sig = 2.0 / 2.0 ** (1.0 / 6.0)
    for t in used:
        cmds.append(f"pair_coeff {at_id[t]} {at_id[t]} {float(P['epsilon'][ti[t]]) * ev:.15g} "
                    f"{float(P['rstar'][ti[t]]) * sig:.15g}")
    cmds += [f"bond_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tb.items()]
    cmds += [f"angle_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in ta.items()]
    cmds += [f"dihedral_coeff {i} {len(k)} " + " ".join(f"{a:.15g} {n} {d:.15g}" for a, n, d in k)
             for k, i in td.items()]
    cmds += [f"improper_coeff {i} {k[0]:.15g} {k[1]} {k[2]}" for k, i in tm.items()]
    return data, cmds, 0.0


def _charmm(model, pos, bounds, boundary):
    from .charmmlib import COULOMB_CONSTANT, KCAL_TO_EV
    ev = 1.0 / KCAL_TO_EV
    ff, top = model.ff, model.topology
    P = {k: v.detach().double().cpu() for k, v in ff.params.items()}
    used = list(dict.fromkeys(top.types))
    at_id = {t: i + 1 for i, t in enumerate(used)}
    ti = {t: ff.type_index(t) for t in used}
    coulomb = float(getattr(model, "coulomb_constant", COULOMB_CONSTANT))
    q = model.charges.detach().double().cpu().numpy() * math.sqrt(coulomb / LAMMPS_COULOMB)
    rmin_tab, eps_tab, rmin14_tab, eps14_tab = [x.detach().double().cpu() for x in ff.pair_tables()]
    sections = {s: [] for _, s in _SECTIONS}
    tb, ta, td, tm = {}, {}, {}, {}
    for m, (i, j) in enumerate(top.bonds):
        k = int(model.bond_type[m])
        key = (_r(P["bond_k"][k] * ev), _r(P["bond_r0"][k]))
        tb.setdefault(key, len(tb) + 1)
        sections["Bonds"].append((tb[key], i + 1, j + 1))
    ub = {}
    for m in range(model.ub_index.shape[1]):
        i, k = int(model.ub_index[0, m]), int(model.ub_index[1, m])
        t = int(model.ub_type[m])
        ub[(i, k)] = (_r(P["ub_k"][t] * ev), _r(P["ub_s0"][t]))
    for m, (i, j, k) in enumerate(top.angles):
        t = int(model.angle_type[m])
        key = (_r(P["angle_k"][t] * ev), _r(P["angle_theta0"][t] / _DEG)) + ub.get((i, k), (0.0, 0.0))
        ta.setdefault(key, len(ta) + 1)
        sections["Angles"].append((ta[key], i + 1, j + 1, k + 1))
    # dihedral charmm carries the 1,4 pairs: each pair once over the dihedrals that share it
    pairs14 = set(top.pairs14)
    count = {}
    for i, _j, _k, l in top.dihedrals:
        p = (min(i, l), max(i, l))
        count[p] = count.get(p, 0) + 1
    weights = [1.0 / count[(min(i, l), max(i, l))] if (min(i, l), max(i, l)) in pairs14 else 0.0
               for i, j, k, l in top.dihedrals]
    for m, (i, j, k, l) in enumerate(top.dihedrals):
        t = int(model.dihedral_type[m])
        for s in range(P["dihedral_k"].shape[1]):
            if float(ff.dihedral_mask[t, s]) == 0.0:
                continue
            d = int(round(math.degrees(math.atan2(float(ff.dihedral_sin[t, s]), float(ff.dihedral_cos[t, s])))))
            key = (_r(P["dihedral_k"][t, s] * ev), int(ff.dihedral_n[t, s]), d % 360, weights[m] if s == 0 else 0.0)
            td.setdefault(key, len(td) + 1)
            sections["Dihedrals"].append((td[key], i + 1, j + 1, k + 1, l + 1))
    for m, (i, j, k, l) in enumerate(model.impropers):
        t = int(model.improper_type[m])
        key = (_r(P["improper_k"][t] * ev),
               _r(math.degrees(math.atan2(float(ff.improper_sin[t]), float(ff.improper_cos[t])))))
        tm.setdefault(key, len(tm) + 1)
        sections["Impropers"].append((tm[key], i + 1, j + 1, k + 1, l + 1))
    counts = {"Bonds": len(tb), "Angles": len(ta), "Dihedrals": len(td), "Impropers": len(tm)}
    data = _data("xnn CHARMM", pos, bounds, {at_id[t]: float(ff.type_mass[ti[t]]) for t in used},
                 [at_id[t] for t in top.types], q, sections, counts)
    rc = float(model.cutoff)
    cmds = ["units real", "atom_style full", f"boundary {boundary}",
            f"pair_style lj/charmm/coul/charmm {rc - 1e-6!r} {rc!r}", "special_bonds charmm"]
    cmds += _styles(sections, {"bond": "harmonic", "angle": "charmm", "dihedral": "charmm", "improper": "harmonic"})
    cmds.append("read_data DATAFILE")
    root = 2.0 ** (1.0 / 6.0)
    for a in used:
        for b in used:
            if at_id[a] > at_id[b]:
                continue
            ia, ib = ti[a], ti[b]
            cmds.append(f"pair_coeff {at_id[a]} {at_id[b]} {float(eps_tab[ia, ib]) * ev:.15g} "
                        f"{float(rmin_tab[ia, ib]) / root:.15g} {float(eps14_tab[ia, ib]) * ev:.15g} "
                        f"{float(rmin14_tab[ia, ib]) / root:.15g}")
    cmds += [f"bond_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tb.items()]
    cmds += [f"angle_coeff {i} {k[0]:.15g} {k[1]:.15g} {k[2]:.15g} {k[3]:.15g}" for k, i in ta.items()]
    cmds += [f"dihedral_coeff {i} {k[0]:.15g} {k[1]} {k[2]} {k[3]:.15g}" for k, i in td.items()]
    cmds += [f"improper_coeff {i} {k[0]:.15g} {k[1]:.15g}" for k, i in tm.items()]
    return data, cmds, 0.0


def _dreiding(model, pos, bounds, boundary):
    from .dreiding import KE
    from .oplslib import KCAL_TO_EV
    ev = 1.0 / KCAL_TO_EV
    ff, top = model.ff, model.topology
    P = {k: v.detach().double().cpu() for k, v in ff.params.items()}
    n, cutoff = top.n_atoms, float(model.cutoff)
    used = list(dict.fromkeys(top.types))
    at_id = {t: i + 1 for i, t in enumerate(used)}
    ti = {t: ff.type_index(t) for t in used}
    orders = list(top.bond_orders) or [1.0] * len(top.bonds)
    sections = {s: [] for _, s in _SECTIONS}
    b_types, a_types, d_types, i_types = {}, {}, {}, {}
    morse = model.bond_style == "morse"
    for (i, j), o in zip(top.bonds, orders):
        r0 = float(P["radius"][ti[top.types[i]]] + P["radius"][ti[top.types[j]]]) - ff.delta
        k_n = o * float(P["bond_k"])
        if morse:
            d_n = o * float(P["bond_d"])
            key = (_r(d_n * ev), _r(math.sqrt(k_n / (2.0 * d_n))), _r(r0))
        else:
            key = (_r(0.5 * k_n * ev), _r(r0))
        b_types.setdefault(key, len(b_types) + 1)
        sections["Bonds"].append((b_types[key], i + 1, j + 1))
    ka = float(P["angle_k"]) * ev
    for m, (i, j, k) in enumerate(top.angles):
        th0 = float(P["theta0"][ti[top.types[j]]])
        key = ("cosine", _r(ka), 0.0) if bool(model.angle_linear[m]) else \
            ("cosine/squared", _r(0.5 * ka / math.sin(th0) ** 2), _r(th0 / _DEG))
        a_types.setdefault(key, len(a_types) + 1)
        sections["Angles"].append((a_types[key], i + 1, j + 1, k + 1))
    for m in range(model.dihedral_index.shape[1]):
        i, j, k, l = [int(x) for x in model.dihedral_index[:, m]]
        rule = int(model.dihedral_rule[m])
        v = float(model.dihedral_weight[m]) * float(P["torsion_v"][rule]) * ev
        key = (_r(0.5 * v), -int(ff.rule_sign[rule]), int(ff.rule_n[rule]))
        d_types.setdefault(key, len(d_types) + 1)
        sections["Dihedrals"].append((d_types[key], i + 1, j + 1, k + 1, l + 1))
    for m in range(model.inversion_index.shape[1]):
        c, a, b, d = [int(x) for x in model.inversion_index[:, m]]
        idx = ti[top.types[c]]
        key = (_r(float(P["oop_k"][idx]) * ev / 3.0), _r(float(P["oop_psi0"][idx]) / _DEG))
        i_types.setdefault(key, len(i_types) + 1)
        sections["Impropers"].append((i_types[key], c + 1, a + 1, b + 1, d + 1))
    q = (model.charge.detach().double().cpu().numpy() * math.sqrt(KE * ev / LAMMPS_COULOMB) if model.has_coulomb
         else np.zeros(n))
    counts = {"Bonds": len(b_types), "Angles": len(a_types), "Dihedrals": len(d_types), "Impropers": len(i_types)}
    data = _data("xnn DREIDING", pos, bounds, {at_id[t]: float(ff.type_mass[ti[t]]) or 1.0 for t in used},
                 [at_id[t] for t in top.types], q, sections, counts)
    hyd = model.hbond_hydrogens.cpu()
    hb = model.use_hbond and bool((hyd >= 0).any())
    kernel = "buck" if ff.form == "x6" else "lj/cut"
    base = f"{kernel}{'/coul/cut' if model.has_coulomb else ''} {cutoff!r}"
    hb_ang = math.acos(model.hbond_cos) / _DEG
    cmds = ["units real", "atom_style full", f"boundary {boundary}", "special_bonds lj/coul 0.0 0.0 1.0",
            (f"pair_style hybrid/overlay {base} hbond/dreiding/lj 4 {model.hbond_cutoff!r} {model.hbond_cutoff!r} "
             f"{hb_ang!r}") if hb else f"pair_style {base}"]
    if b_types:
        cmds.append(f"bond_style {'morse' if morse else 'harmonic'}")
    styles = sorted({k[0] for k in a_types})
    if a_types:
        cmds.append(f"angle_style hybrid {' '.join(styles)}" if len(styles) > 1 else f"angle_style {styles[0]}")
    if d_types:
        cmds.append("dihedral_style harmonic")
    if i_types:
        cmds.append("improper_style umbrella")
    cmds.append("read_data DATAFILE")
    sub = f"{base.split()[0]} " if hb else ""
    for a in used:
        for b in used:
            if at_id[a] > at_id[b]:
                continue
            ia, ib = ti[a], ti[b]
            if ff.form == "x6":
                zz = [max(float(P["x6_zeta"][x]), 6.0 + 1e-6) for x in (ia, ib)]
                r0 = [float(P["vdw_r0"][x]) for x in (ia, ib)]
                dd = [float(P["vdw_d0"][x]) * ev for x in (ia, ib)]
                A = [dd[m] * 6.0 / (zz[m] - 6.0) * math.exp(zz[m]) for m in (0, 1)]
                C = [dd[m] * zz[m] / (zz[m] - 6.0) * r0[m] ** 6 for m in (0, 1)]
                c_ij = 0.5 * (zz[0] / r0[0] + zz[1] / r0[1])
                cmds.append(f"pair_coeff {at_id[a]} {at_id[b]} {sub}{math.sqrt(A[0] * A[1]):.14g} {1.0 / c_ij:.14g} "
                            f"{math.sqrt(C[0] * C[1]):.14g}")
            else:
                d0 = math.sqrt(float(P["vdw_d0"][ia] * P["vdw_d0"][ib])) * ev
                r0a, r0b = float(P["vdw_r0"][ia]), float(P["vdw_r0"][ib])
                r0 = math.sqrt(r0a * r0b) if ff.combination == "geometric" else 0.5 * (r0a + r0b)
                cmds.append(f"pair_coeff {at_id[a]} {at_id[b]} {sub}{d0:.14g} {r0 / 2.0 ** (1.0 / 6.0):.14g}")
    if hb:
        dt = {top.types[d] for d in range(n) if int(hyd[d, 0]) >= 0}
        ht = {top.types[int(h)] for h in hyd[hyd >= 0]}
        ac = {top.types[a] for a in range(n) if bool(model.hbond_acceptor[a])}
        for d in sorted(dt):
            for a in sorted(ac):
                for h in sorted(ht):
                    cmds.append(f"pair_coeff {min(at_id[d], at_id[a])} {max(at_id[d], at_id[a])} hbond/dreiding/lj "
                                f"{at_id[h]} i {float(P['hbond_d0']) * ev:.14g} {float(P['hbond_r0']):.14g} 4 "
                                f"{model.hbond_cutoff!r} {model.hbond_cutoff!r} {hb_ang!r}")
    cmds += [f"bond_coeff {t} " + " ".join(str(v) for v in k) for k, t in b_types.items()]
    multi = len(styles) > 1
    for k, t in a_types.items():
        args = f"{k[1]} {k[2]}" if k[0] == "cosine/squared" else f"{k[1]}"
        cmds.append(f"angle_coeff {t} {k[0] + ' ' if multi else ''}{args}")
    cmds += [f"dihedral_coeff {t} {k[0]} {k[1]} {k[2]}" for k, t in d_types.items()]
    cmds += [f"improper_coeff {t} {k[0]} {k[1]}" for k, t in i_types.items()]
    return data, cmds, 0.0


def _reaxff(model, pos, bounds, boundary, atomic_numbers):
    from .ffield import write_ffield
    if model.nn or model.conventions != "lammps":
        raise ValueError("LAMMPS evaluates classical ReaxFF fields with conventions='lammps'")
    if atomic_numbers is None:
        raise ValueError("the ReaxFF inputs need the atomic numbers")
    z = np.asarray(atomic_numbers.detach().cpu() if hasattr(atomic_numbers, "detach") else atomic_numbers, int)
    from ase.data import atomic_numbers as z_of
    species = list(model.species)
    present = [e for e in species if z_of[e] in set(z.tolist())]
    at_id = {z_of[e]: i + 1 for i, e in enumerate(present)}
    masses = {at_id[z_of[e]]: float(model.mass[species.index(e)]) for e in present}
    lines = ["xnn ReaxFF", "", f"{len(z)} atoms", f"{len(present)} atom types", ""]
    lines += [f"{float(bounds[a, 0])!r} {float(bounds[a, 1])!r} {x}lo {x}hi" for a, x in enumerate("xyz")]
    lines += ["", "Masses", ""] + [f"{t} {m:.6f}" for t, m in masses.items()] + ["", "Atoms # charge", ""]
    lines += [f"{i + 1} {at_id[int(zi)]} 0.0 {x:.14f} {y:.14f} {zz:.14f}"
              for i, (zi, (x, y, zz)) in enumerate(zip(z, pos))]
    data = "\n".join(lines) + "\n"
    cmds = ["units real", "atom_style charge", f"boundary {boundary}", "read_data DATAFILE",
            "pair_style reaxff NULL checkqeq yes", f"pair_coeff * * FFIELD {' '.join(present)}",
            "neighbor 2.0 bin", f"fix qeq all qeq/reaxff 1 0.0 {float(model.cutoff)!r} 1.0e-10 reaxff maxiter 1000"]
    return data, cmds, 0.0, {"ffield": write_ffield(model.export_library())}


def lammps_input(model, pos, cell=None, atomic_numbers=None) -> tuple:
    """The LAMMPS data file, the commands, the energy offset and the further files of a classical force field.

    Parameters
    ----------
    model : nn.Module
        An :class:`~xnn.ffnn.models.OPLS`, :class:`~xnn.ffnn.models.Amber`,
        :class:`~xnn.ffnn.models.Charmm`, :class:`~xnn.ffnn.models.Dreiding` or classical
        :class:`~xnn.ffnn.models.ReaxFF` model.
    pos : array_like
        Positions ``(n, 3)`` in Angstrom, in the order of the bound topology.
    cell : array_like, optional
        An orthorhombic cell ``(3, 3)`` in Angstrom (lattice vectors as rows); ``None`` for a cluster.
    atomic_numbers : array_like, optional
        The atomic numbers ``(n,)`` (ReaxFF, which has no bound topology).

    Returns
    -------
    tuple
        The data file text, the commands (``read_data DATAFILE`` names the data file, ``FFIELD`` the ReaxFF
        field file), the energy in kcal/mol that LAMMPS leaves out (the OPLS series constants ``V0``; 0
        otherwise) and the further files ``{"ffield": text}`` (ReaxFF; empty otherwise).
    """
    from .amber import Amber
    from .charmm import Charmm
    from .dreiding import Dreiding
    from .opls import OPLS
    from .reaxff import ReaxFF
    model = getattr(model, "model", model)
    pos = np.asarray(pos.detach().cpu() if hasattr(pos, "detach") else pos, float).reshape(-1, 3)
    cell = None if cell is None else np.asarray(cell.detach().cpu() if hasattr(cell, "detach") else cell, float)
    pos, bounds, boundary = _box(pos, cell, float(model.cutoff))
    if isinstance(model, ReaxFF):
        return _reaxff(model, pos, bounds, boundary, atomic_numbers)
    _check_switch(model)
    for cls, fn in ((OPLS, _opls), (Amber, _amber), (Charmm, _charmm), (Dreiding, _dreiding)):
        if isinstance(model, cls):
            return fn(model, pos, bounds, boundary) + ({},)
    raise TypeError(f"no LAMMPS inputs for {type(model).__name__}")


def write_lammps_input(model, pos, cell=None, directory: Union[str, Path] = ".", name: str = "system",
                       atomic_numbers=None) -> Path:
    """Write ``<name>.data``, ``in.<name>`` (the commands, ``run 0`` ready) and, for ReaxFF, ``<name>.ffield``
    for LAMMPS into ``directory`` (see :func:`lammps_input`) and return the path of the input script."""
    data, cmds, offset, files = lammps_input(model, pos, cell, atomic_numbers)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.data").write_text(data)
    if "ffield" in files:
        (directory / f"{name}.ffield").write_text(files["ffield"])
    lines = [c.replace("DATAFILE", f"{name}.data").replace("FFIELD", f"{name}.ffield") for c in cmds]
    if offset:
        lines.insert(0, f"# energy offset not in LAMMPS (the OPLS series constants V0): {offset:.15g} kcal/mol")
    script = directory / f"in.{name}"
    script.write_text("\n".join(lines) + "\n")
    return script


__all__ = ["lammps_input", "write_lammps_input", "LAMMPS_COULOMB"]
