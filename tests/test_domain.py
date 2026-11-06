"""Domain decomposition (DomainParallel): partition, ghost shell, agreement with one process.

Several processes are emulated in one by summing every process's local terms;
the cross-process charge constraint of AIMNet2 runs in a 3-process Gloo launch.
"""
import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.deploy import DomainParallel, choose_grid  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402
from xnn.gnn.models.aimnet2 import AIMNet2  # noqa: E402
from xnn.gnn.models.tensornet import TensorNet  # noqa: E402

CELL = np.array([[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [0.7, -1.0, 9.5]])
GRIDS = [(2, 1, 1), (1, 3, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2), (3, 1, 2)]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(periodic=True, n=40, seed=0, species=(1, 6, 8)):
    rng = np.random.default_rng(seed)
    if periodic:
        # positions partly outside the cell, as in an unwrapped trajectory
        pos = rng.uniform(-0.3, 1.3, (n, 3)) @ CELL
        s = {"pos": pos, "cell": CELL, "pbc": [True, True, True]}
    else:
        s = {"pos": rng.uniform(0.0, 11.0, (n, 3))}
    s["atomic_numbers"] = [species[i % len(species)] for i in range(n)]
    return s


def _mace():
    torch.manual_seed(0)
    cfg = from_dict({"model": {
        "name": "mace", "cutoff": 3.5, "n_features": 8, "n_interactions": 2,
        "extra": {"species": [1, 6, 8], "max_ell": 2, "correlation": 2, "num_channels": 8, "max_L": 1}}})
    return build_model(cfg.model).eval()


def _tensornet():
    torch.manual_seed(1)
    model = TensorNet(n_features=8, n_interactions=2, n_rbf=6, cutoff=3.0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.add_(0.1 * torch.randn_like(p))
    return model.eval()


def _aimnet2(coulomb=None):
    torch.manual_seed(2)
    return AIMNet2([1, 6, 7, 8], cutoff=3.5, n_features=4, n_rbf=6,
                   hidden=[[16, 12], [16, 12], [16, 12, 12]], aim_size=10, readout_hidden=[8],
                   n_vector_combinations=3, coulomb=coulomb).eval()


def _reference(model, s):
    graph = structure_to_graph(s, model.cutoff)
    out = ForceStressOutput(model, compute_stress="cell" in s)(graph)
    return out, graph


def _emulated(dp, graph, grid):
    """Every process's local terms, summed: what the all-reduce of a launch returns."""
    world = int(np.prod(grid))
    plans = [dp.plan(graph, rank=r, world_size=world) for r in range(world)]
    owned = torch.cat([p.index[:p.n_owned] for p in plans])
    assert sorted(owned.tolist()) == list(range(graph.num_nodes))
    terms = [dp.local_terms(graph, p) for p in plans]
    energy = sum(t[0] for t in terms)
    forces = -sum(t[1] for t in terms)
    strain = sum(t[2] for t in terms)
    return energy, forces, strain, plans


def test_choose_grid():
    """The grid evaluates the fewest ghosts: long axes are cut, uncut axes need none."""
    assert choose_grid(4, [400.0, 40.0, 40.0], 10.0) == (4, 1, 1)
    assert choose_grid(1, [5.0, 5.0, 5.0], 10.0) == (1, 1, 1)
    # a cube: two cut axes shed more ghosts than three
    assert sorted(choose_grid(8, [100.0, 100.0, 100.0], 10.0)) == [1, 2, 4]
    assert sorted(choose_grid(6, [80.0, 80.0, 80.0], 10.0)) == [1, 1, 6]
    assert sorted(choose_grid(6, [80.0, 80.0, 80.0], 1.0)) == [1, 2, 3]


def test_receptive_fields():
    """The receptive field is one cutoff per message-passing step."""
    assert _mace().receptive_field() == pytest.approx(7.0)
    assert _tensornet().receptive_field() == pytest.approx(9.0)
    assert _aimnet2().receptive_field() == pytest.approx(10.5)
    assert _aimnet2(coulomb="simple").receptive_field() is None


@pytest.mark.parametrize("grid", [(2, 2, 2), (3, 1, 2)])
def test_shell_holds_every_atom_within_the_receptive_field(grid):
    """Each process holds every periodic image within the receptive field of its owned atoms
    (along an uncut axis, the images are the box's periodicity rather than copies)."""
    model = _mace()
    s = _structure()
    graph = structure_to_graph(s, model.cutoff)
    dp = DomainParallel(model, grid=grid)
    rf = model.receptive_field()
    cell = torch.as_tensor(CELL)
    images = torch.stack(torch.meshgrid(*[torch.arange(-2, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
    every = (graph.pos[:, None, :] + (images.double() @ cell)[None]).reshape(-1, 3)
    world = int(np.prod(grid))
    for r in range(world):
        plan = dp.plan(graph, rank=r, world_size=world)
        local = graph.pos[plan.index] + plan.shifts.double() @ cell
        owned = local[:plan.n_owned]
        if not len(owned):
            continue
        exact = "donot_use_mm_for_euclid_dist"
        near = every[torch.cdist(every, owned, compute_mode=exact).min(1).values < rf - 1e-6]
        inverse = torch.linalg.inv(cell)
        diff = (near @ inverse)[:, None, :] - (local @ inverse)[None]
        uncut = torch.tensor([g == 1 for g in grid])
        diff[..., uncut] -= torch.round(diff[..., uncut])
        assert (diff @ cell).norm(dim=-1).min(1).values.max() < 1e-9


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("build", [_mace, _tensornet])
def test_matches_one_process(build, periodic):
    """Energy, forces and stress summed over the sub-domains equal one evaluation of the box."""
    model = build()
    s = _structure(periodic=periodic)
    ref, graph = _reference(model, s)
    scale = float(ref["forces"].abs().max())
    for grid in GRIDS:
        dp = DomainParallel(model, grid=grid)
        energy, forces, strain, plans = _emulated(dp, graph, grid)
        assert float(energy) == pytest.approx(float(ref["energy"]), rel=1e-11, abs=1e-11)
        assert float((forces - ref["forces"]).abs().max()) < 1e-10 * max(scale, 1.0)
        if periodic:
            volume = abs(float(torch.linalg.det(graph.cell[0])))
            assert torch.allclose(strain / volume, ref["stress"][0], atol=1e-11)
            assert all(p.pbc is None or bool(p.pbc.any()) for p in plans)


def test_single_process_forward():
    """Without a launch the module is one sub-domain and returns the whole-box values."""
    for model, s in ((_mace(), _structure()), (_aimnet2(), _structure(species=(1, 6, 7, 8))),
                     (_aimnet2(), {**_structure(periodic=False, species=(1, 6, 7, 8)), "total_charge": 1.0})):
        ref, graph = _reference(model, s)
        out = DomainParallel(ForceStressOutput(model))(graph)
        assert float(out["energy"]) == pytest.approx(float(ref["energy"]), rel=1e-12)
        assert torch.allclose(out["forces"], ref["forces"], atol=1e-11)
        if "stress" in ref:
            assert torch.allclose(out["stress"], ref["stress"], atol=1e-11)


def test_refusals():
    """Long-range models, batches, partial periodicity and mismatched grids are refused."""
    with pytest.raises(ValueError, match="receptive field"):
        DomainParallel(_aimnet2(coulomb="simple"))
    model = _mace()
    s = _structure()
    graph = structure_to_graph(s, model.cutoff)
    with pytest.raises(ValueError, match="does not hold"):
        DomainParallel(model, grid=(2, 2, 1)).plan(graph, rank=0, world_size=2)
    partial = structure_to_graph({**s, "pbc": [True, True, False]}, model.cutoff)
    with pytest.raises(ValueError, match="periodic"):
        DomainParallel(model).plan(partial)
    from xnn.common.data import collate
    with pytest.raises(ValueError, match="one structure"):
        DomainParallel(model).plan(collate([graph, graph]))
    plan = DomainParallel(model).plan(graph)
    other = structure_to_graph(_structure(n=12), model.cutoff)
    with pytest.raises(ValueError, match="another structure"):
        DomainParallel(model)(other, plan)


_SCRIPT = """
import numpy as np
import torch
import torch.distributed as dist
from xnn.common.data import structure_to_graph
from xnn.common.deploy import DomainParallel
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.aimnet2 import AIMNet2

torch.set_default_dtype(torch.float64)
dist.init_process_group("gloo")
torch.manual_seed(2)
model = AIMNet2([1, 6, 7, 8], cutoff=3.5, n_features=4, n_rbf=6,
                hidden=[[16, 12], [16, 12], [16, 12, 12]], aim_size=10, readout_hidden=[8],
                n_vector_combinations=3, coulomb=None).eval()
cell = np.array([[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [0.7, -1.0, 9.5]])
rng = np.random.default_rng(0)
z = [(1, 6, 7, 8)[i % 4] for i in range(40)]
cases = [{"pos": rng.uniform(0, 1, (40, 3)) @ cell, "cell": cell, "pbc": [True] * 3, "total_charge": 1.0},
         {"pos": rng.uniform(0, 11, (40, 3)), "total_charge": -1.0}]
for s in cases:
    s["atomic_numbers"] = z
    graph = structure_to_graph(s, model.cutoff)
    ref = ForceStressOutput(model, compute_stress="cell" in s)(graph)
    out = DomainParallel(model)(graph)
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-10 * abs(float(ref["energy"])), (out, ref)
    assert float((out["forces"] - ref["forces"]).abs().max()) < 1e-10
    if "stress" in ref:
        assert float((out["stress"] - ref["stress"]).abs().max()) < 1e-11
dist.destroy_process_group()
print("ok")
"""


def test_charge_constraint_across_processes(tmp_path):
    """AIMNet2's charge equilibration sums the owned atoms of every process (3-process Gloo launch)."""
    script = tmp_path / "domain.py"
    script.write_text(_SCRIPT)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "3",
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0", str(script)],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ok") == 3
