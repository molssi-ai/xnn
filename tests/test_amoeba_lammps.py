"""The inputs of an AMOEBA or HIPPO model for LAMMPS's AMOEBA package (write_lammps_input)."""
import numpy as np
import pytest
import torch

from xnn.ffnn.models import Amoeba, Hippo, MolecularTopology, builtin_amoeba, read_tinker_prm, write_lammps_input
from xnn.ffnn.models.amoeba_lammps import CLUSTER_CUTOFF

WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239988, 0.926627, 0.0],
                  [2.9, 0.1, 0.2], [3.6, 0.6, 0.2], [3.1, -0.8, 0.5]])


def _topology(n):
    return MolecularTopology.from_bonds(["OW", "HW", "HW"] * n,
                                        [(3 * m, 3 * m + 1) for m in range(n)] + [(3 * m, 3 * m + 2) for m in range(n)])


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def test_amoeba_cluster_inputs(tmp_path):
    """A cluster: the four files, every term with the cluster cutoff, the converter's command in the header."""
    model = Amoeba("amoeba-water-2003", _topology(2), polar_eps=1e-8)
    script = write_lammps_input(model, WATER, directory=tmp_path, name="dimer")
    assert script == tmp_path / "in.dimer"
    assert {p.name for p in tmp_path.iterdir()} == {"dimer.prm", "dimer.xyz", "dimer.key", "in.dimer"}
    text = script.read_text()
    assert "tinker2lmp.py -xyz dimer.xyz -amoeba dimer.prm -data dimer.data" in text
    assert "pair_style amoeba" in text and "boundary s s s" in text and "pair_coeff * * dimer.prm dimer.key" in text
    key = (tmp_path / "dimer.key").read_text().split("\n")
    assert f"vdw-cutoff {CLUSTER_CUTOFF}" in key and "polar-eps 1e-08" in key
    assert (tmp_path / "dimer.xyz").read_text().split("\n")[0].split()[0] == "6"


def test_hippo_box_inputs(tmp_path):
    """A cell: the model's Ewald settings in the key, the repulsion size K / 8, no charge flux (with a warning),
    no d-equals-p keyword, the plain and in-plane angles under one section banner."""
    lib = builtin_amoeba("hippo-water-2021")
    cell = np.diag([12.0, 12.0, 12.0])
    model = Hippo(lib, _topology(2), cutoff=6.3, ewald_cutoff=6.0, pme_grid=[24, 24, 24], dpme_grid=[16, 16, 16])
    with pytest.warns(UserWarning, match="charge flux") if (lib.bndcflux or lib.angcflux) else _nothing():
        write_lammps_input(model, WATER, cell=cell, directory=tmp_path, name="box")
    prm_text = (tmp_path / "box.prm").read_text()
    assert "d-equals-p" not in prm_text and "bndcflux" not in prm_text and "angcflux" not in prm_text
    assert prm_text.count("Angle Bending Parameters") <= 1
    written = read_tinker_prm(tmp_path / "box.prm")
    k_lib = sorted(row["k"] for row in lib.repulsion.values())
    k_out = sorted(row["k"] for row in written.repulsion.values())
    assert np.allclose(np.array(k_out) * 8.0, k_lib)
    key = (tmp_path / "box.key").read_text().split("\n")
    assert "ewald" in key and "dewald" in key and "pme-grid 24 24 24" in key and "dpme-grid 16 16 16" in key
    assert "ewald-cutoff 6.0" in key
    text = (tmp_path / "in.box").read_text()
    assert "pair_style hippo" in text and "boundary p p p" in text and "-pbc 12.0 12.0 12.0" in text


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

