"""The LAMMPS inputs of the classical force fields: the data file holds the bound topology (atoms, bonds,
angles, dihedrals, impropers and their type counts), the commands name the LAMMPS styles of each force
field, and the inputs refuse what LAMMPS cannot evaluate (a switched cutoff, a triclinic cell). The energies
against LAMMPS itself are checked outside the test suite (LAMMPS is not a dependency)."""
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from xnn.ffnn.models import lammps_input, write_lammps_input

_spec = importlib.util.spec_from_file_location("_tcf", Path(__file__).with_name("test_classical_fast.py"))
tcf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcf)
SYSTEMS = tcf.SYSTEMS

STYLES = {"opls": "dihedral_style opls", "amber": "dihedral_style fourier", "charmm": "dihedral_style charmm",
          "dreiding": "improper_style umbrella"}


def _box_lines(data: str) -> list:
    """The box bounds of a data file, which LAMMPS reads as two plain numbers each."""
    rows = [line.split() for line in data.splitlines() if line.endswith(("xlo xhi", "ylo yhi", "zlo zhi"))]
    return [(float(r[0]), float(r[1])) for r in rows]


def _count(data: str, key: str) -> int:
    match = re.search(rf"^(\d+) {key}$", data, re.M)
    return int(match.group(1)) if match else 0


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_the_data_file_holds_the_topology(case, periodic):
    name, cls, lib, top, z, pos, kw = case
    model = cls(lib, top, cutoff=8.0, **kw)
    cell = np.eye(3) * 20.0 if periodic else None
    data, cmds, offset, files = lammps_input(model, np.asarray(pos), cell)
    assert files == {}
    assert len(_box_lines(data)) == 3
    assert _count(data, "atoms") == len(z)
    assert _count(data, "bonds") == len(top.bonds)
    family = name.split("-")[0].split("/")[0]
    if family != "dreiding":
        assert _count(data, "angles") == len(top.angles)
    assert ("boundary p p p" if periodic else "boundary m m m") in cmds
    assert any(c.startswith("read_data DATAFILE") for c in cmds)
    if family in STYLES and (family != "dreiding" or _count(data, "impropers")):
        assert STYLES[family] in cmds
    if "morse" in name:
        assert "bond_style morse" in cmds
    assert offset == 0.0 or family == "opls"


def test_the_input_files_are_written(tmp_path):
    name, cls, lib, top, z, pos, kw = SYSTEMS[0]
    script = write_lammps_input(cls(lib, top, cutoff=8.0, **kw), np.asarray(pos), np.eye(3) * 20.0, tmp_path, "butane")
    text = script.read_text()
    assert script.name == "in.butane" and (tmp_path / "butane.data").exists()
    assert "read_data butane.data" in text


def test_what_lammps_cannot_evaluate_is_refused():
    name, cls, lib, top, z, pos, kw = SYSTEMS[0]
    with pytest.raises(ValueError, match="switch"):
        lammps_input(cls(lib, top, cutoff=8.0, switch_width=1.0, **kw), np.asarray(pos), None)
    cell = np.eye(3) * 20.0
    cell[1, 0] = 2.0
    with pytest.raises(ValueError, match="orthorhombic"):
        lammps_input(cls(lib, top, cutoff=8.0, **kw), np.asarray(pos), cell)


def test_reaxff_inputs_carry_the_field():
    from xnn.ffnn.models import ReaxFF, template_library
    pos = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.60, 0.90, 0.0], [-0.60, -0.50, 0.80],
                    [-0.60, -0.50, -0.80], [1.80, 0.85, 0.0]])
    z = np.array([6, 8, 1, 1, 1, 1])
    model = ReaxFF(template_library(["C", "H", "O"], nn=False), nn=False, conventions="lammps")
    data, cmds, offset, files = lammps_input(model, pos, np.eye(3) * 20.0, atomic_numbers=z)
    assert _count(data, "atoms") == 6 and "Atoms # charge" in data and len(_box_lines(data)) == 3
    assert "pair_style reaxff NULL checkqeq yes" in cmds and any(c.startswith("fix qeq") for c in cmds)
    assert "ffield" in files and files["ffield"].strip()
    with pytest.raises(ValueError, match="classical"):
        lammps_input(ReaxFF(template_library(["C", "H", "O"], nn=True), nn=True), pos, None, atomic_numbers=z)
