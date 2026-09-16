"""Tests for the self-contained TorchScript export.

Covers the scriptable neighbor list (parity with the reference
:func:`~xnn.common.data.build_neighbor_list` for molecular, orthorhombic,
triclinic and unwrapped inputs), and the exported artifact itself: that it
scripts, that it reproduces the eager model's energy/forces/stress, that its
two entry points agree, and -- the point of the whole exercise -- that it loads
and runs with ``xnn`` and ``e3nn`` blocked from import.

Both a plain model and a :class:`~xnn.common.models.les.LatentEwald`-wrapped
one are exercised; the latter is what a long-range checkpoint deploys as.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import build_neighbor_list, structure_to_graph
from xnn.common.deploy import (
    TorchScriptPotential,
    build_neighbor_list_ts,
    export_torchscript_potential,
)
from xnn.common.models import ForceStressOutput, build_model

CUTOFF = 4.0


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _water(n=2, seed=0, spacing=3.0):
    """A small cluster of ``n`` water molecules on a loose grid."""
    rng = np.random.default_rng(seed)
    base = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    pos, z = [], []
    for i in range(n):
        shift = np.array([i * spacing, 0.0, 0.0]) + rng.normal(0, 0.05, 3)
        pos.append(base + shift)
        z += [8, 1, 1]
    return np.concatenate(pos), np.array(z)


def _config(long_range):
    m = {"name": "mace", "cutoff": CUTOFF, "n_features": 8,
         "n_interactions": 2, "n_rbf": 6,
         "species": [1, 8], "max_ell": 2, "max_L": 1, "correlation": 2,
         "hidden_irreps": "8x0e+8x1o", "MLP_irreps": "8x0e",
         "radial_MLP": [16, 16], "num_polynomial_cutoff": 5,
         "avg_num_neighbors": 4.0, "atomic_energies": [0.0, 0.0]}
    if long_range:
        m["long_range"] = {"n_channels": 2, "sigma": 1.0, "dl": 3.0}
    return from_dict({"model": m, "data": {"cutoff": CUTOFF}})


def _model(long_range):
    torch.manual_seed(0)
    model = build_model(_config(long_range).model)
    return model.eval()


# --------------------------------------------------------------------------
# scriptable neighbor list
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["molecular", "cubic", "triclinic",
                                  "unwrapped", "partial_pbc"])
def test_neighbor_list_matches_reference(case):
    """The TorchScript neighbor list reproduces the reference one exactly."""
    pos, _ = _water(3, seed=1)
    pos = torch.tensor(pos)
    if case == "molecular":
        cell, pbc = None, None
    elif case == "cubic":
        cell, pbc = torch.eye(3) * 9.0, torch.ones(3, dtype=torch.bool)
    elif case == "triclinic":
        cell = torch.tensor([[9.0, 0.0, 0.0], [1.5, 8.5, 0.0],
                             [0.7, 1.1, 9.3]])
        pbc = torch.ones(3, dtype=torch.bool)
    elif case == "partial_pbc":
        cell = torch.eye(3) * 9.0
        pbc = torch.tensor([True, False, True])
    else:  # positions pushed outside the cell
        cell, pbc = torch.eye(3) * 9.0, torch.ones(3, dtype=torch.bool)
        pos = pos + torch.tensor([9.0, -18.0, 27.0])

    ref_ei, ref_cs = build_neighbor_list(pos, CUTOFF, cell, pbc)
    cell_t = torch.zeros(3, 3) if cell is None else cell
    pbc_t = torch.zeros(3, dtype=torch.bool) if pbc is None else pbc
    ts_ei, ts_cs = build_neighbor_list_ts(pos, CUTOFF, cell_t, pbc_t)

    # compare as sets of (src, dst, shift) so edge ordering is not load-bearing
    def key(ei, cs):
        return sorted(map(tuple, np.concatenate(
            [ei.T.numpy(), cs.numpy()], axis=1).tolist()))

    assert key(ts_ei, ts_cs) == key(ref_ei, ref_cs)
    assert ts_ei.shape[1] > 0


def test_neighbor_list_is_scriptable():
    """``build_neighbor_list_ts`` compiles under TorchScript."""
    fn = torch.jit.script(build_neighbor_list_ts)
    pos, _ = _water(2)
    ei, cs = fn(torch.tensor(pos), CUTOFF, torch.zeros(3, 3),
                torch.zeros(3, dtype=torch.bool))
    assert ei.shape[0] == 2 and cs.shape[1] == 3


# --------------------------------------------------------------------------
# exported potential
# --------------------------------------------------------------------------

@pytest.mark.parametrize("long_range", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_scripted_matches_eager(long_range, periodic):
    """The scripted artifact reproduces the eager model's energy and forces."""
    model = _model(long_range)
    pos, z = _water(3, seed=2)
    cell = np.eye(3) * 9.0 if periodic else None
    pbc = np.array([periodic] * 3)

    eager = ForceStressOutput(model, compute_stress=True)
    graph = structure_to_graph(
        {"pos": pos, "atomic_numbers": z, "cell": cell, "pbc": pbc}, CUTOFF)
    ref = eager(graph)

    scripted = torch.jit.script(TorchScriptPotential(model, CUTOFF).eval())
    out = scripted(torch.tensor(pos), torch.tensor(z),
                   torch.tensor(cell) if cell is not None else None,
                   torch.tensor(pbc))

    assert float(out["energy"]) == pytest.approx(
        float(ref["energy"].sum()), abs=1e-9)
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-9)
    # per-atom energies must still sum to the total (LES spreads E_lr)
    assert float(out["node_energy"].sum()) == pytest.approx(
        float(out["energy"]), abs=1e-9)
    if periodic:
        assert torch.allclose(out["stress"], ref["stress"][0], atol=1e-9)
    if long_range:
        assert abs(float(out["energy_lr"])) > 0.0
        assert float(out["energy_sr"]) + float(out["energy_lr"]) == \
            pytest.approx(float(out["energy"]), abs=1e-9)


@pytest.mark.parametrize("long_range", [False, True])
def test_two_entry_points_agree(long_range):
    """``forward`` and ``forward_lammps`` give the same answer."""
    model = _model(long_range)
    pos, z = _water(3, seed=3)
    cell = torch.eye(3) * 9.0
    pbc = torch.ones(3, dtype=torch.bool)
    scripted = torch.jit.script(TorchScriptPotential(model, CUTOFF).eval())

    whole = scripted(torch.tensor(pos), torch.tensor(z), cell, pbc)
    ei, cs = build_neighbor_list(torch.tensor(pos), CUTOFF, cell, pbc)
    pair = scripted.forward_lammps(torch.tensor(pos), ei, cs,
                                   torch.tensor(z), cell)

    assert float(pair["energy"]) == pytest.approx(float(whole["energy"]),
                                                  abs=1e-9)
    assert torch.allclose(pair["forces"], whole["forces"], atol=1e-9)


def test_forces_are_translation_invariant_and_conservative():
    """Forces sum to zero and match a finite-difference energy derivative."""
    model = _model(long_range=True)
    pos, z = _water(2, seed=4)
    scripted = torch.jit.script(TorchScriptPotential(model, CUTOFF).eval())
    out = scripted(torch.tensor(pos), torch.tensor(z))
    assert torch.allclose(out["forces"].sum(0), torch.zeros(3), atol=1e-8)

    h = 1e-5
    for atom, comp in [(0, 0), (3, 2)]:
        plus, minus = pos.copy(), pos.copy()
        plus[atom, comp] += h
        minus[atom, comp] -= h
        e_p = float(scripted(torch.tensor(plus), torch.tensor(z))["energy"])
        e_m = float(scripted(torch.tensor(minus), torch.tensor(z))["energy"])
        assert float(out["forces"][atom, comp]) == pytest.approx(
            -(e_p - e_m) / (2 * h), abs=1e-6)


def test_export_embeds_metadata(tmp_path):
    """The saved archive carries cutoff, long-range flag and caller metadata."""
    path = str(tmp_path / "deployed.pt")
    export_torchscript_potential(_model(long_range=True), CUTOFF, path,
                                 metadata={"species": [1, 8]})
    extra = {"cutoff": "", "long_range": "", "species": ""}
    torch.jit.load(path, _extra_files=extra)
    assert extra["cutoff"].decode() == str(CUTOFF)
    assert extra["long_range"].decode() == "True"
    assert extra["species"].decode() == "[1, 8]"


def test_rejects_model_without_tensor_core():
    """A model with no scriptable core is refused with a clear message."""
    with pytest.raises(TypeError, match="node_features_energy"):
        TorchScriptPotential(torch.nn.Linear(3, 3), CUTOFF)


def test_artifact_runs_without_xnn(tmp_path):
    """The whole point: load and run the artifact with xnn/e3nn unimportable.

    Runs in a subprocess with an import hook that raises on ``xnn`` and
    ``e3nn``, so any residual dependency of the serialized module shows up as
    a failure rather than being silently satisfied by the test environment.
    """
    model = _model(long_range=True)
    pos, z = _water(3, seed=5)
    ref = float(ForceStressOutput(model)(structure_to_graph(
        {"pos": pos, "atomic_numbers": z, "cell": None,
         "pbc": np.array([False] * 3)}, CUTOFF))["energy"].sum())

    path = str(tmp_path / "deployed.pt")
    export_torchscript_potential(model, CUTOFF, path)
    np.save(str(tmp_path / "pos.npy"), pos)
    np.save(str(tmp_path / "z.npy"), z)

    script = textwrap.dedent(f"""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in ('xnn', 'e3nn'):
                    raise ImportError('BLOCKED ' + name)
                return None
        sys.meta_path.insert(0, Block())
        import numpy as np, torch
        torch.set_default_dtype(torch.float64)
        m = torch.jit.load({path!r})
        pos = torch.tensor(np.load({str(tmp_path / 'pos.npy')!r}))
        z = torch.tensor(np.load({str(tmp_path / 'z.npy')!r}))
        print(repr(float(m(pos, z)['energy'])))
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert float(proc.stdout.strip().splitlines()[-1]) == pytest.approx(
        ref, abs=1e-9)


def test_runs_under_no_grad():
    """A caller wrapping the call in ``torch.no_grad()`` still gets forces.

    Ordinary inference loops and MD drivers do this; the artifact has to
    re-enable grad internally because its forces come from autograd.
    """
    model = _model(long_range=True)
    pos, z = _water(2, seed=6)
    scripted = torch.jit.script(TorchScriptPotential(model, CUTOFF).eval())
    ref = scripted(torch.tensor(pos), torch.tensor(z))
    with torch.no_grad():
        out = scripted(torch.tensor(pos), torch.tensor(z))
    assert float(out["energy"]) == pytest.approx(float(ref["energy"]), abs=1e-9)
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-9)
    assert torch.abs(out["forces"]).max() > 0
    # the caller's grad mode must be left as it was found
    assert torch.is_grad_enabled()


def test_caller_dtype_is_preserved():
    """Inputs in either float dtype work; outputs come back in that dtype."""
    model = _model(long_range=True).float()
    pos, z = _water(2, seed=7)
    scripted = torch.jit.script(TorchScriptPotential(model, CUTOFF).eval())

    out32 = scripted(torch.tensor(pos, dtype=torch.float32), torch.tensor(z))
    out64 = scripted(torch.tensor(pos, dtype=torch.float64), torch.tensor(z))
    assert out32["forces"].dtype == torch.float32
    assert out64["forces"].dtype == torch.float64
    # float64 in, float64 out, but the model still computes in float32
    assert float(out64["energy"]) == pytest.approx(float(out32["energy"]),
                                                   abs=1e-5)


def test_double_precision_module():
    """``.double()`` on the module promotes the working dtype."""
    model = _model(long_range=True)
    pos, z = _water(2, seed=8)
    scripted = torch.jit.script(
        TorchScriptPotential(model, CUTOFF).eval()).double()
    out = scripted(torch.tensor(pos, dtype=torch.float64), torch.tensor(z))
    assert out["forces"].dtype == torch.float64
    ref = ForceStressOutput(model)(structure_to_graph(
        {"pos": pos, "atomic_numbers": z, "cell": None,
         "pbc": np.array([False] * 3)}, CUTOFF))
    assert float(out["energy"]) == pytest.approx(
        float(ref["energy"].sum()), abs=1e-9)
