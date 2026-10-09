"""Read and write the RuNNer ``input.data`` structure format.

``input.data`` is the data set format of the RuNNer code (Behler group) for
high-dimensional neural network potentials: a sequence of structures, each
between a ``begin`` and an ``end`` line, with ``lattice`` lines for a periodic
cell, one ``atom`` line per atom, and the ``energy`` and total ``charge`` of the
structure. RuNNer works in atomic units (bohr, hartree, elementary charges);
the reader converts to the xnn units (Angstrom, eV) and returns the structure
dicts of :func:`~xnn.common.data.dataset.structure_to_graph`.

The columns of the ``atom`` lines follow the ``begin`` line. A bare ``begin``
means the default layout::

    atom  x y z  element  charge  energy  fx fy fz

and a ``begin`` line may name its own, with the width of a vector property in
parentheses, e.g. ``begin position(3) element charges hirshfeld_volume
forces(3)``. The reader understands ``position``, ``element``, ``charge``
(``charges``), ``forces`` (``force``) and skips every other property.

Example::

    from xnn.common.data.runner_io import read_runner_data

    structures = read_runner_data("input.data")      # Angstrom, eV, e
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Union

import numpy as np

from .elements import CHEMICAL_SYMBOLS, atomic_number
from .hub.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV

#: the ``atom`` columns of a bare ``begin`` line
DEFAULT_COLUMNS = (("position", 3), ("element", 1), ("charge", 1), ("energy", 1),
                   ("forces", 3))

_ALIASES = {"positions": "position", "pos": "position", "charges": "charge",
            "force": "forces", "elements": "element", "atomic_energy": "energy",
            "energies": "energy"}


def _columns(tokens: list[str]) -> tuple[tuple[str, int], ...]:
    """The ``(name, width)`` layout of the ``atom`` lines from a ``begin`` line."""
    if not tokens:
        return DEFAULT_COLUMNS
    out = []
    for tok in tokens:
        m = re.fullmatch(r"([A-Za-z_][\w-]*)(?:\((\d+)\))?", tok)
        if m is None:
            raise ValueError(f"cannot parse the property {tok!r} of a begin line")
        name = m.group(1).lower()
        out.append((_ALIASES.get(name, name), int(m.group(2) or 1)))
    return tuple(out)


def _structure(lattice: list, atoms: list, layout, energy, charge, comment,
               length: float, energy_unit: float, read_charges: bool) -> dict[str, Any]:
    """One structure dict in xnn units from the parsed lines of a block."""
    pos, z, q, f = [], [], [], []
    has_f = any(name == "forces" for name, _ in layout)
    has_q = any(name == "charge" for name, _ in layout)
    for values in atoms:
        col = 0
        for name, width in layout:
            chunk = values[col:col + width]
            if len(chunk) < width:
                raise ValueError(f"atom line too short for its begin line: {' '.join(values)}")
            if name == "position":
                pos.append([float(v) for v in chunk])
            elif name == "element":
                z.append(atomic_number(chunk[0]))
            elif name == "charge":
                q.append(float(chunk[0]))
            elif name == "forces":
                f.append([float(v) for v in chunk])
            col += width
    s: dict[str, Any] = {
        "pos": np.asarray(pos, dtype=np.float64) * length,
        "atomic_numbers": np.asarray(z, dtype=np.int64),
        "cell": None,
        "pbc": np.zeros(3, dtype=bool),
    }
    if lattice:
        if len(lattice) != 3:
            raise ValueError(f"a periodic structure needs three lattice lines, got {len(lattice)}")
        s["cell"] = np.asarray(lattice, dtype=np.float64) * length
        s["pbc"] = np.ones(3, dtype=bool)
    if energy is not None:
        s["energy"] = float(energy) * energy_unit
    if has_f:
        s["forces"] = np.asarray(f, dtype=np.float64) * (energy_unit / length)
    if has_q and read_charges:
        s["charges"] = np.asarray(q, dtype=np.float64)
    if charge is not None:
        s["total_charge"] = float(charge)
    if comment:
        s["comment"] = " ".join(comment)
    return s


def iter_runner_data(path: Union[str, Path], units: str = "xnn",
                     charges: bool = True) -> Iterable[dict[str, Any]]:
    """Iterate over the structures of a RuNNer ``input.data`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.
    units : str, optional
        ``"xnn"`` (default): Angstrom, eV and eV/Angstrom; ``"atomic"``: the
        file's bohr, hartree and hartree/bohr unchanged.
    charges : bool, optional
        Return the per-atom charge column as the ``charges`` label (default).
        Data sets without reference charges carry zeros in that column; pass
        ``False`` so they are not taken for labels.

    Yields
    ------
    dict
        Structure dicts with ``pos``, ``atomic_numbers``, ``cell``, ``pbc`` and,
        when present, ``energy``, ``forces``, ``charges``, ``total_charge`` and
        ``comment``.
    """
    if units not in ("xnn", "atomic"):
        raise ValueError(f"units must be 'xnn' or 'atomic', got {units!r}")
    length = BOHR_TO_ANGSTROM if units == "xnn" else 1.0
    energy_unit = HARTREE_TO_EV if units == "xnn" else 1.0
    inside = False
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            tokens = line.split()
            if not tokens or tokens[0].startswith("#"):
                continue
            key = tokens[0].lower()
            if key == "begin":
                if inside:
                    raise ValueError(f"{path}:{lineno}: begin inside an open structure")
                inside = True
                layout = _columns(tokens[1:])
                lattice, atoms, comment = [], [], []
                energy = charge = None
            elif not inside:
                raise ValueError(f"{path}:{lineno}: {tokens[0]!r} outside begin/end")
            elif key == "lattice":
                lattice.append([float(v) for v in tokens[1:4]])
            elif key == "atom":
                atoms.append(tokens[1:])
            elif key == "energy":
                energy = float(tokens[1])
            elif key == "charge":
                charge = float(tokens[1])
            elif key == "comment":
                comment = tokens[1:]
            elif key == "end":
                inside = False
                yield _structure(lattice, atoms, layout, energy, charge, comment,
                                 length, energy_unit, charges)
            else:
                raise ValueError(f"{path}:{lineno}: unknown keyword {tokens[0]!r}")
    if inside:
        raise ValueError(f"{path}: the last structure has no end line")


def read_runner_data(path: Union[str, Path], units: str = "xnn",
                     charges: bool = True) -> list[dict[str, Any]]:
    """Read every structure of a RuNNer ``input.data`` file.

    See :func:`iter_runner_data` for the parameters and the structure dicts.

    Returns
    -------
    list of dict
        The structures, in file order.
    """
    return list(iter_runner_data(path, units=units, charges=charges))


def write_runner_data(structures: Iterable[dict[str, Any]], path: Union[str, Path],
                      units: str = "xnn") -> None:
    """Write structure dicts to a RuNNer ``input.data`` file (default layout).

    Each structure gives ``pos`` and ``atomic_numbers`` and optionally
    ``cell`` (written when ``pbc`` is set or absent), ``energy``, ``forces``,
    ``charges``, ``total_charge`` and ``comment``; missing per-atom values are
    written as zeros and a missing energy as zero.

    Parameters
    ----------
    structures : iterable of dict
        The structures, in xnn units (``units="xnn"``) or atomic units.
    path : str or pathlib.Path
        The file to write.
    units : str, optional
        The units of ``structures``: ``"xnn"`` (default, converted to bohr and
        hartree) or ``"atomic"`` (written unchanged).
    """
    length = 1.0 / BOHR_TO_ANGSTROM if units == "xnn" else 1.0
    energy_unit = 1.0 / HARTREE_TO_EV if units == "xnn" else 1.0
    lines = []
    for s in structures:
        pos = np.asarray(s["pos"], dtype=np.float64) * length
        z = np.asarray(s["atomic_numbers"]).reshape(-1)
        n = len(z)
        forces = (np.asarray(s["forces"], dtype=np.float64) * (energy_unit / length)
                  if s.get("forces") is not None else np.zeros((n, 3)))
        q = (np.asarray(s["charges"], dtype=np.float64).reshape(n)
             if s.get("charges") is not None else np.zeros(n))
        lines.append("begin")
        if s.get("comment"):
            lines.append(f"comment {s['comment']}")
        cell = s.get("cell")
        pbc = s.get("pbc")
        if cell is not None and (pbc is None or np.asarray(pbc).any()):
            for row in np.asarray(cell, dtype=np.float64).reshape(3, 3) * length:
                lines.append("lattice " + " ".join(f"{v: .16e}" for v in row))
        for i in range(n):
            lines.append("atom " + " ".join(f"{v: .16e}" for v in pos[i])
                         + f" {CHEMICAL_SYMBOLS[int(z[i])]:>2s} {q[i]: .16e} {0.0: .16e} "
                         + " ".join(f"{v: .16e}" for v in forces[i]))
        energy = s.get("energy")
        lines.append(f"energy {(0.0 if energy is None else float(energy) * energy_unit): .16e}")
        lines.append(f"charge {float(s.get('total_charge', 0.0) or 0.0): .16e}")
        lines.append("end")
    Path(path).write_text("\n".join(lines) + "\n")

