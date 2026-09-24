"""Tests for the faithful PhysNet (registered as ``physnet``).

Covers rotation/translation/permutation invariance (energies, forces,
charges, dipoles), charge conservation, the switched/shielded electrostatics,
the D3(BJ) dispersion port (against reference values generated with the
original TensorFlow implementation), flexible depth, periodic stress,
upstream ``train.py`` key translation, and -- when TensorFlow *and* a clone
of MMunibas/PhysNet are available -- machine-precision weight-transplant
parity against the original graph.
"""
import math
import os

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.common.models import d3
from xnn.dnn.models.physnet import KEHALF, shifted_softplus


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=8, cutoff=6.0, periodic=False, R=None, shift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 4, (n, 3)) + shift
    if R is not None:
        pos = pos @ R.T
    s = {"pos": pos, "atomic_numbers": ([8, 1, 1, 6] * n)[:n]}
    if periodic:
        s["cell"] = np.eye(3) * 7.0
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _build(num_blocks=3, use_ele=True, use_disp=True, seed=0, **extra):
    cfg = from_dict({"model": {
        "name": "physnet", "cutoff": 6.0, "n_features": 24, "n_rbf": 16,
        "n_interactions": num_blocks,
        "extra": {"use_electrostatics": use_ele, "use_dispersion": use_disp,
                  **extra},
    }})
    model = build_model(cfg.model)
    # the k2f and output heads are zero-initialized (upstream convention);
    # randomize them so tests exercise the full network
    torch.manual_seed(seed)
    with torch.no_grad():
        for ob in model.output_blocks:
            ob.dense.weight.normal_(0, 0.1)
        for ib in model.interaction_blocks:
            ib.interaction.k2f.weight.normal_(0, 0.1)
    return model


def _proper_rotation(seed=3):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def test_registered():
    assert "physnet" in available_models()


def test_shifted_softplus_exact():
    """The activation keeps its log1p tail where F.softplus goes linear."""
    x = torch.tensor([-40.0, -1.0, 0.0, 25.0, 40.0], dtype=torch.float64)
    ref = torch.log1p(torch.exp(-x.abs())) + x.clamp(min=0) - math.log(2.0)
    assert torch.allclose(shifted_softplus(x), ref, atol=0)
    assert shifted_softplus(torch.tensor(0.0)).abs() < 1e-15
    # F.softplus would return exactly 25.0 - log(2); the exact form keeps
    # the ~1.4e-11 tail
    assert (shifted_softplus(torch.tensor(25.0, dtype=torch.float64))
            - (25.0 - math.log(2.0))) > 1e-12


@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_model_equivariance(num_blocks):
    model = ForceStressOutput(_build(num_blocks=num_blocks))
    R = _proper_rotation()
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())
    o0 = model(_graph())
    o1 = model(_graph(R=R))
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-10
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T,
                          atol=1e-10)
    assert torch.allclose(o1["charges"].detach(), o0["charges"].detach(),
                          atol=1e-10)
    assert torch.allclose(o1["dipole"].detach(), o0["dipole"].detach() @ Rt.T,
                          atol=1e-10)
    o2 = model(_graph(shift=11.0))
    assert abs(float(o0["energy"].detach()) - float(o2["energy"].detach())) < 1e-10


def test_permutation_invariance():
    model = ForceStressOutput(_build())
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 4, (8, 3))
    Z = [8, 1, 1, 6, 1, 1, 7, 16]
    perm = rng.permutation(8)
    o0 = model(structure_to_graph({"pos": pos, "atomic_numbers": Z}, 6.0))
    o1 = model(structure_to_graph(
        {"pos": pos[perm], "atomic_numbers": [Z[i] for i in perm]}, 6.0))
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-10
    assert torch.allclose(o1["charges"].detach(),
                          o0["charges"].detach()[perm], atol=1e-10)


def test_charge_conservation():
    model = ForceStressOutput(_build())
    g = _graph()
    assert abs(float(model(g)["charges"].sum())) < 1e-12
    g.total_charge = torch.tensor([-2.0])
    assert abs(float(model(g)["charges"].sum()) + 2.0) < 1e-12


def test_flexible_num_blocks():
    for t in (1, 2, 4):
        model = _build(num_blocks=t)
        assert len(model.interaction_blocks) == t
        out = ForceStressOutput(model)(_graph())
        assert out["forces"].shape == (8, 3)


def test_periodic_stress():
    model = ForceStressOutput(_build(num_blocks=1), compute_stress=True)
    out = model(_graph(n=6, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


def test_atomic_energies_shift():
    """With zero-init heads and no long-range terms, E is exactly sum(E0)."""
    cfg = from_dict({"model": {
        "name": "physnet", "cutoff": 6.0, "n_features": 16, "n_rbf": 8,
        "n_interactions": 1,
        "extra": {"use_electrostatics": False, "use_dispersion": False,
                  "species": [1, 6, 8], "atomic_energies": [-13.6, -1000.0, -2000.0]}}})
    model = build_model(cfg.model)
    g = _graph()
    e = float(ForceStressOutput(model)(g)["energy"])
    e0 = {1: -13.6, 6: -1000.0, 8: -2000.0}
    assert abs(e - sum(e0[int(z)] for z in g.atomic_numbers)) < 1e-9


def test_electrostatics_is_coulomb_beyond_switch():
    """Two unit charges beyond sr_cut/2 feel exactly ke q1 q2 / r."""
    model = _build(num_blocks=1, use_disp=False)
    r = 5.0  # > sr_cut/2 = 3 -> pure 1/r
    Dij = torch.tensor([r, r], dtype=torch.float64)
    Qa = torch.tensor([1.0, -1.0], dtype=torch.float64)
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])
    e = model.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j).sum()
    assert abs(float(e) - 2 * KEHALF * (-1.0) / r) < 1e-12


def test_electrostatics_vanishes_at_lr_cutoff():
    model = _build(num_blocks=1, use_disp=False, lr_cutoff=8.0)
    Dij = torch.tensor([8.0, 8.0, 9.0], dtype=torch.float64)
    Qa = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    e = model.electrostatic_energy_per_atom(
        Dij, Qa, torch.tensor([0, 1, 0]), torch.tensor([1, 0, 2]))
    assert e.abs().max() < 1e-14


def test_d3_matches_original_tf_values():
    """The D3(BJ) port reproduces values generated with the upstream TF code."""
    rng = np.random.default_rng(5)
    N = 6
    pos = rng.uniform(0, 4.0, (N, 3))
    Z = torch.tensor([8, 1, 1, 6, 7, 16])
    idx_i = np.repeat(np.arange(N), N - 1)
    idx_j = np.concatenate([[j for j in range(N) if j != i] for i in range(N)])
    D = torch.tensor(np.linalg.norm(pos[idx_i] - pos[idx_j], axis=-1))
    r_bohr = D / d3.d3_autoang
    ii, jj = torch.tensor(idx_i), torch.tensor(idx_j)
    # reference values from MMunibas/PhysNet neural_network/grimme_d3 (TF)
    ref_nocut = [-0.00624853170553548, -0.00264062509158721,
                 -0.00229069034147807, -0.00701731586754319,
                 -0.00216131912272694, -0.01061926229239554]
    ref_cut9 = [-0.00580151918621369, -0.00237798164855353,
                -0.00204755221145079, -0.00632581685918044,
                -0.0017560829343922, -0.0094957268097902]
    e = d3.edisp(Z, r_bohr, ii, jj)
    assert np.abs(e.numpy() - np.array(ref_nocut)).max() < 1e-14
    e9 = d3.edisp(Z, r_bohr, ii, jj, cutoff=9.0)
    assert np.abs(e9.numpy() - np.array(ref_cut9)).max() < 1e-14


def test_upstream_physnet_key_translation():
    """Keys spelled as in the upstream train.py map onto the xnn names."""
    cfg = from_dict({"model": {
        "name": "physnet",
        "num_features": 64, "num_basis": 32, "num_blocks": 4,
        "cutoff": 8.0, "lr_cut": 12.0,
        "num_residual_atomic": 1, "num_residual_interaction": 2,
        "num_residual_output": 1, "use_electrostatic": True,
        "grimme_s6": 1.0, "grimme_s8": 2.0,
    }})
    assert cfg.model.n_features == 64
    assert cfg.model.n_rbf == 32
    assert cfg.model.n_interactions == 4
    assert cfg.model.cutoff == 8.0
    m = build_model(cfg.model)
    assert len(m.interaction_blocks) == 4
    assert m.sr_cut == 8.0 and m.lr_cut == 12.0
    assert m.cutoff == 12.0  # neighbor-list radius follows lr_cut
    assert not m._s6_learnable and float(m.s6) == 1.0
    assert m._a1_learnable  # unset -> learnable, softplus-positive
    assert float(m.a1) > 0


def test_parity_vs_original_physnet():
    """Weight transplant from the original TF PhysNet gives identical E/F/q.

    Needs TensorFlow and a clone of MMunibas/PhysNet (path in the
    ``PHYSNET_UPSTREAM_PATH`` environment variable).
    """
    pytest.importorskip("tensorflow")
    path = os.environ.get("PHYSNET_UPSTREAM_PATH")
    if not path or not os.path.isdir(path):
        pytest.skip("set PHYSNET_UPSTREAM_PATH to a clone of MMunibas/PhysNet")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "physnet_tf_parity.py")
    res = subprocess.run([sys.executable, script, path],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    worst = float(res.stdout.strip().splitlines()[-1])
    assert worst < 1e-12


def test_d3_reference_set_switch():
    """PhysNet keeps Grimme's 2010 tables by default; the 2024 set is opt-in."""
    torch.manual_seed(0)
    default = _build(num_blocks=1)
    assert default._d3_c6ab.shape == (95, 95, 5, 5, 3)
    assert torch.equal(default._d3_c6ab, d3.d3_c6ab.to(default._d3_c6ab.dtype))
    torch.manual_seed(0)
    newer = _build(num_blocks=1, d3_references=2024)
    assert newer._d3_c6ab.shape == (95, 95, 7, 7, 3)
    assert torch.equal(newer._d3_c6ab[:87, :87, :5, :5], default._d3_c6ab[:87, :87])
    assert float(newer._d3_c6ab[92, 8, 0, 0, 0]) != float(default._d3_c6ab[92, 8, 0, 0, 0])
    g = _graph()
    assert torch.allclose(default(g)["energy"], newer(g)["energy"])   # no actinides
