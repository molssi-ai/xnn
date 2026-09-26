"""D4 for molecular dynamics: the ATM triple cache and the EEQ reuse between steps.

* ``triplet_cache`` keeps each three-body block's triples from the forward to the
  backward pass, so they are enumerated once per step; the results are unchanged.
* :meth:`DFTD4.enable_eeq_reuse` / :class:`~xnn.common.models.eeq.EEQReuse` carries
  the large-regime EEQ solve from one structure to the next; the results agree
  with the fresh solve to the solver tolerance, and it resets itself or falls
  back to LU rather than ever returning a worse answer.
"""
import numpy as np
import pytest
import torch

import xnn.common.models.dispersion as dispersion
from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, MDIEngine, main
from xnn.common.models import ForceStressOutput, build_model
from xnn.common.models.d4 import D4Dispersion, DFTD4
from xnn.common.models.eeq import EEQReuse

pytestmark = pytest.mark.usefixtures("_f64")


@pytest.fixture
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                  [0.0, -0.763239, -0.477047]])
# periodic large regime: the EEQ range must reach 20 bohr (10.6 A)
PER = dict(cutoff_pair=11.0, cutoff_triple=6.0, cutoff_cn=8.0, cutoff_eeq_cn=8.0)


def _water_box(n_side=3, a=3.104, seed=0):
    """``n_side**3`` randomly oriented waters on a jittered lattice (~1 g/cm^3)."""
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


def _run(model, pos, z, cell, grad=True):
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(z),
         "cell": torch.tensor(cell), "pbc": torch.tensor([True, True, True])}
    g = structure_to_graph(s, model.cutoff)
    if not grad:
        with torch.no_grad():
            return {"energy": model(g)["energy"]}
    return ForceStressOutput(model, compute_stress=True)(g)


@pytest.fixture
def count_enumerations(monkeypatch):
    """Count the calls of the block enumeration."""
    calls = {"n": 0}
    orig = dispersion._block_triplets

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(dispersion, "_block_triplets", counted)
    return calls


# ----------------------------------------------------------------------------
# the triple cache
# ----------------------------------------------------------------------------



def test_triplet_cache_is_exact_and_enumerates_once(count_enumerations):
    pos, z, cell = _water_box()
    ref = _run(D4Dispersion(triplet_chunk=2000, triplet_cache=0, **PER), pos, z, cell)
    n_uncached = count_enumerations["n"]
    count_enumerations["n"] = 0
    out = _run(D4Dispersion(triplet_chunk=2000, **PER), pos, z, cell)
    n_cached = count_enumerations["n"]
    assert n_uncached > 2                         # several blocks, each enumerated twice
    assert n_uncached == 2 * n_cached             # the backward pass reused every block
    assert abs(float(out["energy"] - ref["energy"])) < 1e-12
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-12, rtol=0)
    assert torch.allclose(out["stress"], ref["stress"], atol=1e-14, rtol=0)


def test_triplet_cache_budget_is_respected(count_enumerations):
    pos, z, cell = _water_box()
    ref = _run(D4Dispersion(triplet_chunk=2000, triplet_cache=0, **PER), pos, z, cell)
    n_uncached = count_enumerations["n"]
    count_enumerations["n"] = 0
    # room for roughly one block of ~1000 triples (14 bytes each)
    out = _run(D4Dispersion(triplet_chunk=2000, triplet_cache=2e-5, **PER), pos, z, cell)
    n_partial = count_enumerations["n"]
    assert n_uncached // 2 < n_partial < n_uncached
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-12, rtol=0)


def test_triplet_cache_idle_without_a_backward_pass(count_enumerations):
    pos, z, cell = _water_box()
    e_ref = _run(D4Dispersion(triplet_chunk=2000, triplet_cache=0, **PER), pos, z, cell,
                 grad=False)["energy"]
    count_enumerations["n"] = 0
    e = _run(D4Dispersion(triplet_chunk=2000, **PER), pos, z, cell, grad=False)["energy"]
    assert count_enumerations["n"] > 0
    assert abs(float(e - e_ref)) < 1e-12


def test_triplet_cache_option_from_config():
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.0, "n_interactions": 1,
                               "n_rbf": 6, "n_features": 8,
                               "extra": {"dispersion": {"name": "d4", "triplet_cache": 0.5}}}})
    model = build_model(cfg.model)
    assert model.d4.triplet_cache == 0.5


# ----------------------------------------------------------------------------
# the EEQ reuse
# ----------------------------------------------------------------------------

def _trajectory(pos, n=6, step=0.01, seed=3):
    """A random walk of small displacements, like consecutive MD steps."""
    rng = np.random.default_rng(seed)
    frames = [pos]
    for _ in range(n - 1):
        frames.append(frames[-1] + rng.normal(scale=step, size=pos.shape))
    return frames


def test_eeq_reuse_matches_fresh_solves_along_a_trajectory():
    pos, z, cell = _water_box()
    fresh = D4Dispersion(regime="large", **PER)
    reused = D4Dispersion(regime="large", **PER)
    reused.d4.enable_eeq_reuse()
    frames = _trajectory(pos)
    for x in frames:
        a, b = _run(fresh, x, z, cell), _run(reused, x, z, cell)
        assert abs(float(a["energy"] - b["energy"])) < 1e-8
        assert torch.allclose(a["forces"], b["forces"], atol=1e-7, rtol=0)
        assert torch.allclose(a["stress"], b["stress"], atol=1e-10, rtol=0)
        assert torch.allclose(a["eeq_charges"], b["eeq_charges"], atol=1e-8, rtol=0)
    stats = reused.d4.__dict__["_eeq_reuse"].stats
    assert stats["preconditioners"] == 1            # one inverse for the whole run
    assert stats["solves"] == 3 * len(frames)       # charges, residual, adjoint
    assert stats["fallbacks"] == 0
    assert stats["iterations"] / len(frames) < 30


def test_eeq_reuse_resets_for_another_system_and_can_be_disabled():
    pos, z, cell = _water_box()
    small, zs, cs = _water_box(n_side=2, a=5.4)
    fresh = D4Dispersion(regime="large", **PER)
    reused = D4Dispersion(regime="large", **PER)
    reused.d4.enable_eeq_reuse()
    _run(reused, pos, z, cell)
    b = _run(reused, small, zs, cs)                 # other atom count: starts over
    a = _run(fresh, small, zs, cs)
    assert torch.allclose(a["forces"], b["forces"], atol=1e-7, rtol=0)
    assert reused.d4.__dict__["_eeq_reuse"].stats["preconditioners"] == 2
    reused.d4.enable_eeq_reuse(False)
    c = _run(reused, small, zs, cs)
    assert torch.allclose(a["forces"], c["forces"], atol=1e-12, rtol=0)


def test_eeq_reuse_falls_back_to_lu():
    pos, z, cell = _water_box()
    fresh = D4Dispersion(regime="large", eeq_solver="lu", **PER)
    reused = D4Dispersion(regime="large", eeq_solver="lu", **PER)
    # no iterations allowed: both the attempt and the retry with a re-formed
    # preconditioner fail, and the step falls back to LU
    reused.d4.enable_eeq_reuse(maxiter=0)
    a, b = _run(fresh, pos, z, cell), _run(reused, pos, z, cell)
    assert reused.d4.__dict__["_eeq_reuse"].stats["fallbacks"] > 0
    assert torch.allclose(a["forces"], b["forces"], atol=1e-12, rtol=0)
    assert abs(float(a["energy"] - b["energy"])) < 1e-12


def test_eeq_reuse_leaves_the_dense_regime_alone():
    pos, z, cell = _water_box(n_side=2, a=5.4)
    fresh = D4Dispersion(regime="dense", **PER)
    reused = D4Dispersion(regime="dense", **PER)
    reused.d4.enable_eeq_reuse()
    a, b = _run(fresh, pos, z, cell), _run(reused, pos, z, cell)
    assert torch.allclose(a["forces"], b["forces"], atol=1e-14, rtol=0)
    assert reused.d4.__dict__["_eeq_reuse"].stats["solves"] == 0


def test_eeq_reuse_state_is_not_a_scripted_attribute():
    """Enabling the reuse must not break TorchScript export of the D4 head."""
    term = DFTD4(**PER)
    term.enable_eeq_reuse()
    torch.jit.script(term)


# ----------------------------------------------------------------------------
# the MDI engine
# ----------------------------------------------------------------------------

BASE = {"name": "schnet", "cutoff": 4.0, "n_interactions": 1, "n_rbf": 6, "n_features": 8}
D4_LARGE = {"name": "d4", "regime": "large", **PER}


def _checkpoint(tmp_path):
    cfg = from_dict({"model": BASE})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    return str(path)


def _serve(engine, pos, z, cell):
    engine.natoms = len(z)
    engine.atomic_numbers = np.array(z)
    engine.cell_bohr = cell / BOHR_TO_ANGSTROM
    engine.coords_bohr = pos / BOHR_TO_ANGSTROM
    engine._needs_calculation = True
    engine.calculate()
    return engine.energy, engine.forces.copy(), engine.stress.copy()


def test_engine_eeq_reuse_matches_fresh_engine(tmp_path):
    path = _checkpoint(tmp_path)
    fresh = MDIEngine.from_checkpoint(path, dispersion=D4_LARGE, dtype=torch.float64)
    reused = MDIEngine.from_checkpoint(path, dispersion=D4_LARGE, dtype=torch.float64,
                                       eeq_reuse=True)
    terms = [m for m in reused.model.modules() if isinstance(m, DFTD4)]
    assert terms and isinstance(terms[0].__dict__["_eeq_reuse"], EEQReuse)
    pos, z, cell = _water_box()
    for x in _trajectory(pos, n=4):
        e0, f0, s0 = _serve(fresh, x, z, cell)
        e1, f1, s1 = _serve(reused, x, z, cell)
        assert abs(e0 - e1) < 1e-9                 # hartree
        assert np.abs(f0 - f1).max() < 1e-8
        assert np.abs(s0 - s1).max() < 1e-11
    assert terms[0].__dict__["_eeq_reuse"].stats["preconditioners"] == 1


def test_cli_eeq_reuse_flag(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    seen = {}

    def fake_run(self, mdi_options, mpi_comm=None):
        terms = [m for m in self.model.modules() if isinstance(m, DFTD4)]
        seen["reuse"] = [t.__dict__.get("_eeq_reuse") is not None for t in terms]

    monkeypatch.setattr(MDIEngine, "run", fake_run)
    args = ["--ckpt", path, "-mdi", "-role ENGINE -name xnn -method TCP -port 1 -hostname h",
            "--dispersion", "{name: d4, regime: large, cutoff_pair: 11.0}"]
    main(args + ["--eeq-reuse"])
    assert seen["reuse"] == [True]
    main(args)
    assert seen["reuse"] == [False]
