"""Tests for the D4 long-range tail correction (``DFTD4(tail_correction=True)``).

The correction restores, for periodic structures, the two-body dispersion the
pair cutoff and its switching window remove, assuming a uniform distribution
of atoms beyond the window. Checked: convergence of the corrected energy with
the cutoff, stress against finite strains, forces against finite
differences, and that molecules and the default (off) are unchanged.
"""
import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.common.models import D4Dispersion, ForceStressOutput

WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                  [0.0, -0.763239, -0.477047]])
# pair term only, short CN cutoffs: keeps the long pair cutoffs affordable
PAIR_ONLY = dict(s9=0.0, cutoff_cn=8.0, cutoff_eeq_cn=8.0, cutoff_triple=5.0)


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _water_box(n_side=3, a=3.104, seed=5):
    """n_side^3 randomly oriented waters at ~1 g/cm^3 (Angstrom)."""
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q *= np.sign(np.linalg.det(q))
                pos.extend((WATER - WATER[0]) @ q.T + np.array([i, j, k]) * a
                           + rng.normal(scale=0.1, size=3))
    return np.array(pos), [8, 1, 1] * n_side ** 3, np.eye(3) * a * n_side


def _run(model, pos, z, cell=None, stress=False):
    s = {"pos": pos, "atomic_numbers": z}
    if cell is not None:
        s["cell"], s["pbc"] = np.asarray(cell, dtype=float), [True] * 3
    out = ForceStressOutput(model, compute_stress=stress)(
        structure_to_graph(s, model.cutoff))
    return {k: v.detach() for k, v in out.items()}


def test_tail_converges_the_pair_energy():
    pos, z, cell = _water_box()
    energies = {}
    for rc in (10.0, 14.0, 25.0):
        for tail in (False, True):
            d4 = D4Dispersion(cutoff_pair=rc, switch_width_pair=2.0,
                              tail_correction=tail, **PAIR_ONLY)
            energies[rc, tail] = float(_run(d4, pos, z, cell)["energy"])
    raw_spread = abs(energies[10.0, False] - energies[25.0, False])
    tail_spread = abs(energies[10.0, True] - energies[25.0, True])
    # the correction removes most of the cutoff dependence ...
    assert tail_spread < 0.1 * raw_spread
    # ... is attractive, and shrinks as the cutoff grows (~ rc^-3)
    t10 = energies[10.0, True] - energies[10.0, False]
    t25 = energies[25.0, True] - energies[25.0, False]
    assert t10 < t25 < 0.0
    assert t25 / t10 == pytest.approx((10.0 / 25.0) ** 3, rel=0.35)


def test_tail_stress_matches_finite_strain():
    pos, z, cell = _water_box(2)
    cell = cell + np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, -0.2, 0.0]])
    d4 = D4Dispersion(cutoff_pair=7.0, switch_width_pair=2.0, tail_correction=True,
                      **PAIR_ONLY)
    out = _run(d4, pos, z, cell, stress=True)
    vol = abs(np.linalg.det(cell))
    h = 1e-5
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2)]:
        e = []
        for sign in (1, -1):
            eps = np.eye(3)
            eps[i, j] += sign * h / (1 if i == j else 2)
            if i != j:
                eps[j, i] += sign * h / 2
            e.append(float(_run(d4, pos @ eps.T, z, cell @ eps.T)["energy"]))
        de = (e[0] - e[1]) / (2 * h)
        assert float(out["stress"][0, i, j]) == pytest.approx(de / vol, rel=1e-5, abs=1e-10)
    # the tail's own stress is -E_tail / V on the diagonal when the C6 are
    # held fixed; the C6 also change with strain (CN, EEQ charges), which
    # autograd includes, here a ~7 % effect
    plain = _run(D4Dispersion(cutoff_pair=7.0, switch_width_pair=2.0, **PAIR_ONLY),
                 pos, z, cell, stress=True)
    e_tail = float(out["energy_tail"])
    ds = (out["stress"][0] - plain["stress"][0]).numpy()
    assert np.trace(ds) / 3 == pytest.approx(-e_tail / vol, rel=0.15)


def test_tail_forces_match_finite_differences():
    pos, z, cell = _water_box(2)
    d4 = D4Dispersion(cutoff_pair=7.0, switch_width_pair=2.0, tail_correction=True,
                      **PAIR_ONLY)
    forces = _run(d4, pos, z, cell)["forces"].numpy()
    h = 1e-5
    for atom, xyz in [(0, 0), (4, 1), (7, 2)]:
        e = []
        for sign in (1, -1):
            p = pos.copy()
            p[atom, xyz] += sign * h
            e.append(float(_run(d4, p, z, cell)["energy"]))
        assert forces[atom, xyz] == pytest.approx(-(e[0] - e[1]) / (2 * h), abs=1e-8)


def test_tail_leaves_molecules_and_the_default_unchanged():
    pos, z, cell = _water_box(2)
    on = D4Dispersion(cutoff_pair=7.0, switch_width_pair=2.0, tail_correction=True,
                      **PAIR_ONLY)
    off = D4Dispersion(cutoff_pair=7.0, switch_width_pair=2.0, **PAIR_ONLY)
    assert not D4Dispersion().d4.tail_correction
    mol_on, mol_off = _run(on, pos, z), _run(off, pos, z)
    assert float(mol_on["energy"]) == float(mol_off["energy"])
    assert float(mol_on["energy_tail"]) == 0.0
    per_on, per_off = _run(on, pos, z, cell), _run(off, pos, z, cell)
    assert float(per_on["energy_tail"]) < 0.0
    assert float(per_off["energy_tail"]) == 0.0
    assert float(per_on["energy"]) == pytest.approx(
        float(per_off["energy"]) + float(per_on["energy_tail"]), abs=1e-12)


def test_tail_from_the_config_hook():
    from xnn.common.config import from_dict
    from xnn.common.models import build_model
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.5, "n_interactions": 1,
                               "n_rbf": 6, "n_features": 8,
                               "extra": {"dispersion": {"name": "d4", "cutoff_pair": 12.0,
                                                        "tail_correction": True}}}})
    assert build_model(cfg.model).d4.tail_correction
