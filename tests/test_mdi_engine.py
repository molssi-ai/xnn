"""MDI engine: checkpoint loading (cutoff, dtype, route-B dispersion) and the
total-charge plumbing. The MDI wire protocol itself needs a driver process and
is exercised by the examples/deploy notebooks; here the engine is driven
through its Python state directly."""
import logging
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.models.registry import recorded_dispersion, resolve_dispersion
from xnn.common.data import structure_to_graph
from xnn.common.deploy.mdi_engine import (
    BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine, main,
)
from xnn.common.models import ForceStressOutput, build_model
from xnn.common.models.d4 import DFTD4
from xnn.common.models.dispersion import DispersionCorrection

BASE = {"name": "schnet", "cutoff": 4.0, "n_interactions": 1, "n_rbf": 6, "n_features": 8}
FAST_D4 = {"name": "d4", "cutoff_pair": 9.0, "cutoff_triple": 6.0,
           "cutoff_cn": 7.0, "cutoff_eeq_cn": 7.0, "cutoff_eeq": 9.0}


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
          "cutoff_triple: 6.0, cutoff_cn: 7.0, cutoff_eeq_cn: 7.0, cutoff_eeq: 12.0}",
          "--total-charge", "-1", "--dtype", "float64"])
    assert seen == {"cutoff": pytest.approx(12.0), "charge": -1.0,
                    "dtype": torch.float64, "disp": True}


@pytest.mark.parametrize("extra", [{"long_range": True},
                                   {"dispersion": {**FAST_D4, "regime": "large", "cutoff_pair": 11.0,
                                                   "cutoff_eeq": 11.0}},
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


# route B: the checkpoint records what its labels had subtracted

RECORD = {"name": "d4", "s8": 1.1, "cutoff_pair": 9.0, "cutoff_triple": 6.0, "cutoff_cn": 7.0,
          "cutoff_eeq_cn": 7.0, "cutoff_eeq": 9.0, "dataset": "water_minusD4"}
TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "record_subtracted_dispersion.py"


def _recorded_checkpoint(tmp_path, record=RECORD, name="recorded.pt"):
    cfg = from_dict({"model": BASE, "subtracted_dispersion": record})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model), compute_stress=True).to(torch.float32)
    path = tmp_path / name
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    return str(path)


def test_config_validates_the_recorded_subtraction():
    cfg = from_dict({"model": BASE, "subtracted_dispersion": "d4"})
    assert cfg.subtracted_dispersion == {"name": "d4"}
    assert from_dict({"model": BASE}).subtracted_dispersion is None
    assert from_dict({"model": BASE, "subtracted_dispersion": RECORD}).subtracted_dispersion["s8"] == 1.1
    with pytest.raises(ValueError, match="unknown d4 option"):
        from_dict({"model": BASE, "subtracted_dispersion": {"cutoff_par": 9.0}})
    with pytest.raises(ValueError, match="unknown term"):
        from_dict({"model": BASE, "subtracted_dispersion": "d5"})
    with pytest.raises(ValueError, match="use one or the other"):
        from_dict({"model": {**BASE, "extra": {"dispersion": FAST_D4}}, "subtracted_dispersion": "d4"})


def test_resolve_dispersion_rules():
    assert resolve_dispersion(None, None) is None
    assert resolve_dispersion("d4", None) == "d4"
    assert resolve_dispersion(False, RECORD) is None
    assert resolve_dispersion(None, RECORD) == RECORD
    merged = resolve_dispersion({"cutoff_pair": 12.0}, RECORD)
    assert merged["cutoff_pair"] == 12.0 and merged["s8"] == 1.1 and merged["name"] == "d4"
    with pytest.raises(ValueError, match="cannot be added instead"):
        resolve_dispersion({"name": "d3"}, RECORD)


def test_recorded_subtraction_is_added_back_by_default(tmp_path, caplog):
    """A route-B checkpoint serves with its recorded term and no option, the
    same as the explicit --dispersion route, and says so in the log."""
    path = _recorded_checkpoint(tmp_path)
    with caplog.at_level(logging.INFO):
        engine = MDIEngine.from_checkpoint(path)
    assert "recorded as subtracted" in caplog.text
    assert isinstance(engine.model.model, DispersionCorrection)
    assert engine.cutoff == pytest.approx(9.0)
    assert float(engine.model.model.term.s8) == pytest.approx(1.1)
    explicit = MDIEngine.from_checkpoint(_checkpoint(tmp_path, BASE, name="plain.pt"),
                                         dispersion={k: v for k, v in RECORD.items() if k != "dataset"})
    _water(engine)
    _water(explicit)
    assert engine.energy == pytest.approx(explicit.energy, abs=1e-12)


def test_recorded_subtraction_overrides_and_opt_out(tmp_path, caplog):
    path = _recorded_checkpoint(tmp_path)
    merged = MDIEngine.from_checkpoint(path, dispersion={"cutoff_pair": 12.0, "cutoff_eeq": 12.0})
    assert merged.cutoff == pytest.approx(12.0)
    assert float(merged.model.model.term.s8) == pytest.approx(1.1)      # recorded, kept
    with pytest.raises(ValueError, match="cannot be added instead"):
        MDIEngine.from_checkpoint(path, dispersion="d3")
    with caplog.at_level(logging.WARNING):
        plain = MDIEngine.from_checkpoint(path, dispersion=False)
    assert not isinstance(plain.model.model, DispersionCorrection)
    assert "WITHOUT the dispersion recorded" in caplog.text


def test_cli_serves_the_record_unless_told_not_to(tmp_path, monkeypatch):
    path = _recorded_checkpoint(tmp_path)
    seen = {}

    def fake_run(self, mdi_options, mpi_comm=None):
        seen["disp"] = isinstance(self.model.model, DispersionCorrection)
        seen["cutoff"] = self.cutoff

    monkeypatch.setattr(MDIEngine, "run", fake_run)
    mdi = ["-mdi", "-role ENGINE -name xnn -method TCP -port 1 -hostname h"]
    main(["--ckpt", path, *mdi])
    assert seen == {"disp": True, "cutoff": pytest.approx(9.0)}
    main(["--ckpt", path, *mdi, "--dispersion", "{cutoff_pair: 12.0, cutoff_eeq: 12.0}"])
    assert seen == {"disp": True, "cutoff": pytest.approx(12.0)}
    main(["--ckpt", path, *mdi, "--no-dispersion"])
    assert seen == {"disp": False, "cutoff": pytest.approx(4.0)}
    with pytest.raises(SystemExit):
        main(["--ckpt", path, *mdi, "--no-dispersion", "--dispersion", "d4"])


def test_checkpoint_from_before_the_record_field_still_loads(tmp_path):
    """A pickled Config written before ``subtracted_dispersion`` existed
    unpickles without the attribute; it must read as 'no record'."""
    cfg = from_dict({"model": BASE})
    del cfg.__dict__["subtracted_dispersion"]
    path = tmp_path / "old.pt"
    torch.save({"model": ForceStressOutput(build_model(cfg.model), compute_stress=True).state_dict(),
                "cfg": cfg}, path)
    assert recorded_dispersion(torch.load(path, weights_only=False)["cfg"]) is None
    engine = MDIEngine.from_checkpoint(str(path))
    assert not isinstance(engine.model.model, DispersionCorrection)
    assert isinstance(MDIEngine.from_checkpoint(str(path), dispersion=FAST_D4).model.model,
                      DispersionCorrection)


MACE_SMALL = {"name": "mace", "cutoff": 4.0, "n_features": 8, "n_interactions": 1, "n_rbf": 4,
              "extra": {"species": [1, 8], "max_ell": 1, "correlation": 2, "max_L": 0,
                        "MLP_irreps": "4x0e", "radial_MLP": [8]}}


def test_mace_checkpoint_without_scale_shift_loads_as_built(tmp_path, caplog):
    """Checkpoints from before MACE had ``scale_shift`` carry no
    ``scale``/``shift``; they load with the values the model was built with
    (the identity by default), instead of failing the strict load."""
    path = _checkpoint(tmp_path, MACE_SMALL, dtype=torch.float64)
    ckpt = torch.load(path, weights_only=False)
    ckpt["model"] = {k: v for k, v in ckpt["model"].items() if "scale_shift" not in k}
    legacy = tmp_path / "legacy.pt"
    torch.save(ckpt, legacy)
    with caplog.at_level(logging.INFO):
        engine = MDIEngine.from_checkpoint(str(legacy))
    assert "keeping scale=1, shift=0" in caplog.text
    full = MDIEngine.from_checkpoint(path)
    _water(engine)
    _water(full)
    assert engine.energy == pytest.approx(full.energy, abs=1e-12)
    # a model built with other constants keeps them when the keys are missing
    scaled = {**MACE_SMALL, "extra": {**MACE_SMALL["extra"], "scale": 2.0, "shift": -0.5}}
    ckpt["cfg"] = from_dict({"model": scaled})
    torch.save(ckpt, legacy)
    engine = MDIEngine.from_checkpoint(str(legacy))
    assert float(engine.model.model.scale_shift.scale) == 2.0
    assert float(engine.model.model.scale_shift.shift) == -0.5


def test_record_tool_writes_the_field(tmp_path):
    path = _checkpoint(tmp_path, BASE)
    spec = "{name: d4, s8: 1.1, cutoff_pair: 9.0, cutoff_triple: 6.0, cutoff_cn: 7.0, " \
           "cutoff_eeq_cn: 7.0, cutoff_eeq: 9.0, dataset: water_minusD4}"
    run = subprocess.run([sys.executable, str(TOOL), path, "--spec", spec], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "best.pt.bak").exists()
    assert torch.load(path, weights_only=False)["cfg"].subtracted_dispersion["s8"] == 1.1
    engine = MDIEngine.from_checkpoint(path)
    assert isinstance(engine.model.model, DispersionCorrection) and engine.cutoff == pytest.approx(9.0)
    again = subprocess.run([sys.executable, str(TOOL), path, "--spec", "d4"], capture_output=True, text=True)
    assert again.returncode != 0 and "--force" in again.stderr
    bad = subprocess.run([sys.executable, str(TOOL), path, "--spec", "{cutoff_par: 1}", "--force"],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "unknown d4 option" in bad.stderr


def test_export_adds_the_recorded_dispersion(tmp_path):
    path = _recorded_checkpoint(tmp_path)
    for flag, expect in (([], "True"), (["--no-dispersion"], "False")):
        out = str(tmp_path / f"deployed{len(flag)}.pt")
        run = subprocess.run([sys.executable, "-m", "xnn.common.cli.main", "export", "--ckpt", path,
                              "--out", out, *flag], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        extra = {"dispersion": "", "cutoff": ""}
        torch.jit.load(out, _extra_files=extra)
        assert extra["dispersion"].decode() == expect
        assert float(extra["cutoff"].decode()) == pytest.approx(9.0 if expect == "True" else 4.0)
