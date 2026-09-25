"""MDI engine: checkpoint loading (cutoff, dtype, route-B dispersion) and the
total-charge plumbing. The MDI wire protocol itself needs a driver process and
is exercised by the examples/deploy notebooks; here the engine is driven
through its Python state directly."""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.deploy.mdi_engine import (
    BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine, main,
)
from xnn.common.models import ForceStressOutput, build_model
from xnn.common.models.d4 import DFTD4
from xnn.common.models.dispersion import DispersionCorrection

BASE = {"name": "schnet", "cutoff": 4.0, "n_interactions": 1, "n_rbf": 6, "n_features": 8}
FAST_D4 = {"name": "d4", "cutoff_pair": 9.0, "cutoff_triple": 6.0,
           "cutoff_cn": 7.0, "cutoff_eeq_cn": 7.0}


def _checkpoint(tmp_path, model_cfg, dtype=torch.float32, name="best.pt"):
    cfg = from_dict({"model": model_cfg})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True).to(dtype)
    path = tmp_path / name
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    return str(path)


def _water(engine, charge=None):
    pos = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                    [0.0, -0.763239, -0.477047]])
    engine.natoms = 3
    engine.atomic_numbers = np.array([8, 1, 1])
    engine.coords_bohr = pos / BOHR_TO_ANGSTROM
    if charge is not None:
        engine.total_charge = charge
    engine.calculate()
    return pos


def test_cutoff_follows_the_built_model_not_the_config(tmp_path):
    """Route A: a checkpoint trained with D4 must build graphs at the wrapper's
    cutoff (D4's 9 A here), not at the core model's 4 A."""
    path = _checkpoint(tmp_path, {**BASE, "extra": {"dispersion": FAST_D4}})
    engine = MDIEngine.from_checkpoint(path)
    assert engine.cutoff == pytest.approx(9.0)
    assert engine.cutoff > 4.0
    plain = MDIEngine.from_checkpoint(_checkpoint(tmp_path, BASE, name="plain.pt"))
    assert plain.cutoff == pytest.approx(4.0)


def test_route_b_adds_dispersion_after_loading(tmp_path):
    path = _checkpoint(tmp_path, BASE)
    plain = MDIEngine.from_checkpoint(path)
    with_d4 = MDIEngine.from_checkpoint(path, dispersion=FAST_D4)
    assert isinstance(with_d4.model.model, DispersionCorrection)
    assert with_d4.cutoff == pytest.approx(9.0)
    # same core weights: energies differ by exactly the dispersion energy
    pos = _water(plain)
    _water(with_d4)
    graph = structure_to_graph({"pos": torch.tensor(pos, dtype=with_d4.dtype),
                                "atomic_numbers": torch.tensor([8, 1, 1])}, 9.0)
    e_disp = float(with_d4.model.model.dispersion(graph)["energy"]) / HARTREE_TO_EV
    assert with_d4.energy - plain.energy == pytest.approx(e_disp, abs=1e-6)
    # a bare name selects the defaults; a mapping is passed through
    named = MDIEngine.from_checkpoint(path, dispersion="d3")
    assert type(named.model.model).__name__ == "D3Dispersion"
    with pytest.raises(KeyError, match="unknown dispersion"):
        MDIEngine.from_checkpoint(path, dispersion={"name": "d5"})


def test_route_b_refused_when_checkpoint_has_dispersion(tmp_path):
    path = _checkpoint(tmp_path, {**BASE, "extra": {"dispersion": FAST_D4}})
    with pytest.raises(ValueError, match="already includes a dispersion"):
        MDIEngine.from_checkpoint(path, dispersion="d4")
    MDIEngine.from_checkpoint(path)          # serving it as is still works


def test_total_charge_reaches_the_model(tmp_path):
    """D4's EEQ charges depend on the net charge, so the energy must change."""
    path = _checkpoint(tmp_path, {**FAST_D4, "cutoff": 9.0}, dtype=torch.float64)
    engine = MDIEngine.from_checkpoint(path, total_charge=0.0)
    assert engine.total_charge == 0.0
    _water(engine)
    e0 = engine.energy
    _water(engine, charge=1.0)              # what >TOTCHARGE does at run time
    assert engine.total_charge == 1.0
    assert abs(engine.energy - e0) > 1e-8
    launched = MDIEngine.from_checkpoint(path, total_charge=1.0)
    _water(launched)
    assert launched.energy == pytest.approx(engine.energy, abs=1e-12)


def test_dtype_default_and_exact_tables(tmp_path):
    """Serve in the checkpoint's dtype unless told otherwise, and build the
    constant tables in float64 so a float64 run sees their exact values."""
    # (a model with weights: a parameter-free standalone D4 checkpoint carries
    # no dtype of its own and is served in the default dtype)
    wrapped = {**BASE, "extra": {"dispersion": FAST_D4}}
    path32 = _checkpoint(tmp_path, wrapped, dtype=torch.float32)
    path64 = _checkpoint(tmp_path, wrapped, dtype=torch.float64, name="f64.pt")
    assert MDIEngine.from_checkpoint(path32).dtype == torch.float32
    assert MDIEngine.from_checkpoint(path64).dtype == torch.float64
    up = MDIEngine.from_checkpoint(path32, dtype=torch.float64)
    assert up.dtype == torch.float64
    # tables equal a natively float64-built D4 term bit for bit
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        native = DFTD4(**{k: v for k, v in FAST_D4.items() if k != "name"})
    finally:
        torch.set_default_dtype(prev)
    served = up.model.model.term
    persistent = set(native.state_dict())            # s6, s8, ...: come from the checkpoint
    checked = 0
    for name, buf in native.named_buffers():
        if name not in persistent:                   # the constant tables
            assert torch.equal(buf, getattr(served, name)), name
            checked += 1
    assert checked > 10
    assert torch.get_default_dtype() == prev         # no leak from from_checkpoint
    _water(up)
    assert np.isfinite(up.energy)


def test_cli_parses_dispersion_and_charge(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path, BASE)
    seen = {}

    def fake_run(self, mdi_options, mpi_comm=None):
        seen.update(cutoff=self.cutoff, charge=self.total_charge, dtype=self.dtype,
                    disp=isinstance(self.model.model, DispersionCorrection))

    monkeypatch.setattr(MDIEngine, "run", fake_run)
    main(["--ckpt", path, "-mdi", "-role ENGINE -name xnn -method TCP -port 1 -hostname h",
          "--dispersion", "{name: d4, cutoff_pair: 12.0, switch_width_pair: 2.0, "
          "cutoff_triple: 6.0, cutoff_cn: 7.0, cutoff_eeq_cn: 7.0}",
          "--total-charge", "-1", "--dtype", "float64"])
    assert seen == {"cutoff": pytest.approx(12.0), "charge": -1.0,
                    "dtype": torch.float64, "disp": True}


@pytest.mark.parametrize("extra", [{"long_range": True},
                                   {"dispersion": {**FAST_D4, "regime": "large", "cutoff_pair": 11.0}},
                                   {"dispersion": {**FAST_D4, "regime": "dense"}}])
def test_float32_model_under_float64_default(tmp_path, extra):
    """A float32 checkpoint served in a process whose default dtype is float64
    must still give forces and stress on a periodic cell (torch.det's backward
    would otherwise mix in the default dtype; cell volumes use cell_volume)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        path = _checkpoint(tmp_path, {**BASE, "extra": extra})
        torch.set_default_dtype(torch.float64)
        engine = MDIEngine.from_checkpoint(path, dtype=torch.float32)
        rng = np.random.default_rng(0)
        engine.natoms = 30
        engine.atomic_numbers = np.array([8, 1, 1] * 10)
        engine.coords_bohr = rng.uniform(0, 8.0, (30, 3)) / BOHR_TO_ANGSTROM
        engine.cell_bohr = np.eye(3) * 8.0 / BOHR_TO_ANGSTROM
        engine.calculate()
        assert engine.dtype == torch.float32
        assert np.isfinite(engine.energy) and engine.forces.shape == (30, 3) and engine.stress.shape == (3, 3)
    finally:
        torch.set_default_dtype(prev)
