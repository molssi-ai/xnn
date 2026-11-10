"""Domain decomposition (DomainParallel): partition, ghost shells, exchanges, plan reuse and
agreement with one process.

The processes of a launch are emulated by threads whose collectives meet on a shared board;
a real 3-process Gloo launch runs the same paths through torch.distributed.
"""
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.deploy import DomainParallel, choose_grid, domain  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model, ops  # noqa: E402
from xnn.gnn.models.aimnet2 import AIMNet2  # noqa: E402
from xnn.gnn.models.tensornet import TensorNet  # noqa: E402

CELL = np.array([[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [0.7, -1.0, 9.5]])


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


class _Board:
    """The collectives of ``world`` threads: every thread posts, all read, then all move on."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=120)
        self.slots = [None] * world
        self.local = threading.local()

    @property
    def rank(self):
        return self.local.rank

    def share(self, value):
        self.slots[self.rank] = value
        self.barrier.wait()
        everyone = list(self.slots)
        self.barrier.wait()
        return everyone


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_counts, in_counts, board):
        ctx.counts, ctx.board = (list(out_counts), list(in_counts)), board
        everyone = board.share(list(x.detach().split(list(in_counts))))
        return torch.cat([everyone[s][board.rank] for s in range(board.world)]).clone()

    @staticmethod
    def backward(ctx, grad):
        out_counts, in_counts = ctx.counts
        return _AllToAll.apply(grad.contiguous(), in_counts, out_counts, ctx.board), None, None, None


class _AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op, board):
        ctx.board = board
        everyone = torch.stack(board.share(x.detach().clone()))
        return everyone.amax(0) if op == "max" else everyone.sum(0)

    @staticmethod
    def backward(ctx, grad):
        return _AllReduce.apply(grad.contiguous(), "sum", ctx.board), None, None


def _threads(world, body):
    """``body(rank)`` on ``world`` threads joined by emulated collectives; the results by rank."""
    board = _Board(world)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(domain, "_process", lambda group: (board.rank, world))
        mp.setattr(domain, "_all_to_all", lambda x, o, i, group: _AllToAll.apply(x, o, i, board))
        mp.setattr(domain, "_all_reduce", lambda x, group, op="sum": _AllReduce.apply(x, op, board))
        mp.setattr(domain, "_all_gather", lambda x, group: board.share(x.clone()))
        return _run_threads(board, world, body)


def _run_threads(board, world, body):
    results, errors = [None] * world, []

    def run(rank):
        board.local.rank = rank
        try:
            results[rank] = body(rank)
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
            board.barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    if errors:
        primary = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        raise (primary or errors)[0]
    return results


def _decomposed(world, graph, model, **options):
    """``DomainParallel(model, **options)(graph)`` on ``world`` emulated processes, each with its own
    module (as in a launch); the outputs and the plans by rank."""
    def body(rank):
        dp = DomainParallel(model, **options)
        return dp(graph), dp._plan
    return _threads(world, body)


def _local_positions(plan, graph):
    """Where a plan puts its local atoms: owned atoms wrapped into the cell, ghosts as their images."""
    if plan.cell is None:
        return graph.pos[plan.index]
    cell = graph.cell[0]
    frac = graph.pos @ torch.linalg.inv(cell)
    wrapped = (frac - torch.floor(frac)) @ cell
    return torch.cat([wrapped[plan.owned], wrapped[plan.ghosts] + plan.ghost_shifts.double() @ cell])


def _assert_matches(out, ref, forces=None):
    forces = out["forces"] if forces is None else forces
    scale = max(float(ref["forces"].abs().max()), 1.0)
    assert float(out["energy"]) == pytest.approx(float(ref["energy"]), rel=1e-11, abs=1e-11)
    assert float((forces - ref["forces"]).abs().max()) < 1e-10 * scale
    if "stress" in ref:
        assert torch.allclose(out["stress"], ref["stress"], atol=1e-11)


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


def test_halo_options():
    """``"auto"`` refreshes the ghosts of the models that support it; the shell is one cutoff then,
    the receptive field otherwise; unsupported choices are refused."""
    for build in (_mace, _tensornet, _aimnet2):
        model = build()
        assert model.refreshes_ghosts
        dp = DomainParallel(model)
        assert dp.halo == "layer" and dp.need == pytest.approx(model.cutoff)
        dp = DomainParallel(model, halo="field")
        assert dp.halo == "field" and dp.need == pytest.approx(model.receptive_field())
    plain = _mace()
    plain.refreshes_ghosts = False
    assert DomainParallel(plain).halo == "field"
    with pytest.raises(ValueError, match="does not refresh"):
        DomainParallel(plain, halo="layer")
    with pytest.raises(ValueError, match="halo="):
        DomainParallel(_mace(), halo="shell")
    with pytest.raises(ValueError, match="skin"):
        DomainParallel(_mace(), skin=-0.1)
    with pytest.raises(ValueError, match="receptive field"):
        DomainParallel(_aimnet2(coulomb="simple"))


@pytest.mark.parametrize("halo", ["layer", "field"])
@pytest.mark.parametrize("grid", [(2, 2, 2), (3, 1, 2)])
def test_shell_holds_every_atom_the_owned_atoms_need(grid, halo):
    """Each process holds every periodic image within one cutoff (``"layer"``) or the receptive
    field (``"field"``) of its owned atoms; along an uncut axis, the images are the box's periodicity."""
    model = _mace()
    graph = structure_to_graph(_structure(), model.cutoff)
    dp = DomainParallel(model, grid=grid, halo=halo)
    plans = _threads(int(np.prod(grid)), lambda rank: dp.plan(graph))
    assert sorted(torch.cat([p.owned for p in plans]).tolist()) == list(range(graph.num_nodes))
    assert all(torch.equal(p.order, plans[0].order) and p.counts == plans[0].counts for p in plans)
    cell = graph.cell[0]
    images = torch.stack(torch.meshgrid(*[torch.arange(-2, 3)] * 3, indexing="ij"), -1).reshape(-1, 3)
    every = (graph.pos[:, None, :] + (images.double() @ cell)[None]).reshape(-1, 3)
    inverse = torch.linalg.inv(cell)
    uncut = torch.tensor([g == 1 for g in grid])
    for plan in plans:
        assert plan.n_local == plan.n_owned + plan.n_ghost
        # the one-cutoff shell evaluates the edges into owned atoms only
        into_ghosts = bool((plan.edge_index[1] >= plan.n_owned).any())
        assert into_ghosts == (halo == "field" and plan.n_ghost > 0)
        assert sum(plan.recv_counts) == plan.n_ghost and plan.send_rows.numel() == sum(plan.send_counts)
        local = _local_positions(plan, graph)
        owned = local[:plan.n_owned]
        if not len(owned):
            continue
        exact = "donot_use_mm_for_euclid_dist"
        near = every[torch.cdist(every, owned, compute_mode=exact).min(1).values < dp.need - 1e-6]
        diff = (near @ inverse)[:, None, :] - (local @ inverse)[None]
        diff[..., uncut] -= torch.round(diff[..., uncut])
        assert (diff @ cell).norm(dim=-1).min(1).values.max() < 1e-9


@pytest.mark.parametrize("halo", ["layer", "field"])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("build", [_mace, _tensornet])
def test_matches_one_process(build, periodic, halo):
    """Energy, forces and stress of the decomposed box equal one evaluation of the box, on every process."""
    model = build()
    ref, graph = _reference(model, _structure(periodic=periodic))
    for grid in [(2, 1, 1), (1, 3, 1), (2, 2, 1), (2, 2, 2), (3, 1, 2)]:
        for out, _ in _decomposed(int(np.prod(grid)), graph, model, grid=grid, halo=halo):
            _assert_matches(out, ref)


@pytest.mark.parametrize("halo", ["layer", "field"])
def test_charge_constraint_across_processes(halo):
    """AIMNet2's charge equilibration sums the owned atoms of every process; its charges and
    features cross between processes with the ghosts."""
    model = _aimnet2()
    for s in ({**_structure(species=(1, 6, 7, 8)), "total_charge": 1.0},
              {**_structure(periodic=False, species=(1, 6, 7, 8)), "total_charge": -1.0}):
        ref, graph = _reference(model, s)
        for out, _ in _decomposed(4, graph, model, grid=(1, 2, 2), halo=halo):
            _assert_matches(out, ref)


def test_single_process_forward():
    """Without a launch the module is one sub-domain and returns the whole-box values."""
    for model, s in ((_mace(), _structure()), (_aimnet2(), _structure(species=(1, 6, 7, 8))),
                     (_aimnet2(), {**_structure(periodic=False, species=(1, 6, 7, 8)), "total_charge": 1.0})):
        ref, graph = _reference(model, s)
        dp = DomainParallel(ForceStressOutput(model))
        _assert_matches(dp(graph), ref)
        plan = dp._plan
        assert plan.world == 1 and plan.n_ghost == 0 and plan.n_owned == graph.num_nodes


def _moved(s, seed, size, jumps=False, strain=0.0):
    """``s`` with every atom displaced by up to ``size``, some by whole lattice vectors, and the cell strained."""
    rng = np.random.default_rng(seed)
    step = rng.normal(size=s["pos"].shape)
    step *= size * rng.uniform(0.2, 1.0, (len(step), 1)) / np.linalg.norm(step, axis=1, keepdims=True)
    cell = CELL @ (np.eye(3) + strain * np.array([[1.0, 0.3, 0.0], [0.3, -0.5, 0.2], [0.0, 0.2, 0.8]]))
    frac = s["pos"] @ np.linalg.inv(CELL)
    pos = frac @ cell + step
    if jumps:
        pos += rng.integers(-1, 2, (len(pos), 3)) @ cell
    return {**s, "pos": pos, "cell": cell}


def test_forward_reuses_its_plan_while_it_holds():
    """With a skin, small displacements, lattice jumps and a slight strain keep the plan and the
    results exact; a large displacement, other elements or another charge build a new one."""
    model = _aimnet2()
    s = {**_structure(species=(1, 6, 7, 8)), "total_charge": 1.0}
    near = _moved(s, 1, 0.12, jumps=True, strain=0.004)
    far = _moved(s, 2, 1.0)
    other = {**near, "atomic_numbers": list(reversed(near["atomic_numbers"]))}
    charged = {**other, "total_charge": -1.0}
    cases = [s, near, far, other, charged]
    graphs = [_reference(model, c) for c in cases]

    def body(rank):
        dp = DomainParallel(model, grid=(2, 1, 1), skin=0.6)
        seen = []
        for _, graph in graphs:
            seen.append((dp(graph), dp._plan))
        return seen

    for seen in _threads(2, body):
        for (out, _), (ref, _) in zip(seen, graphs):
            _assert_matches(out, ref)
        plans = [p for _, p in seen]
        assert plans[1] is plans[0] and plans[2] is not plans[1]
        assert plans[3] is not plans[2] and plans[4] is plans[3]


def test_sharded_trajectory():
    """Each process holding only its own atoms: evaluate them while the plan holds, then reshard
    with the velocities following their atoms; forces of the owned atoms and the stress are exact."""
    model = _mace()
    s = _structure()
    graph = structure_to_graph(s, model.cutoff)
    velocities = torch.randn(graph.num_nodes, 3)
    near = _moved(s, 3, 0.2, jumps=True, strain=0.003)
    far = _moved(s, 4, 0.9)
    (ref_near, g_near), (ref_far, g_far) = _reference(model, near), _reference(model, far)
    dp = DomainParallel(model, grid=(2, 2, 1), skin=1.0)

    def body(rank):
        plan = dp.plan(graph)
        vel = velocities[plan.owned]
        cell = g_near.cell[0]
        assert dp.holds(plan, g_near.pos[plan.owned], cell)
        out_near = dp.evaluate(plan, g_near.pos[plan.owned], cell)
        assert not dp.holds(plan, g_far.pos[plan.owned], g_far.cell)
        new, pos, vel = dp.reshard(plan, g_far.pos[plan.owned], vel, cell=g_far.cell)
        assert torch.equal(pos, g_far.pos[new.owned]) and torch.equal(vel, velocities[new.owned])
        assert dp.holds(new, pos)
        return plan, out_near, new, dp.evaluate(new, pos)

    results = _threads(4, body)
    for ref, k, l in ((ref_near, 0, 1), (ref_far, 2, 3)):
        forces = torch.zeros_like(ref["forces"])
        for r in results:
            forces[r[k].owned] = r[l]["forces"]
            assert r[l]["forces"].shape == (r[k].n_owned, 3)
        for r in results:
            _assert_matches(r[l], ref, forces)


def test_molecule_plan_holds_until_an_atom_moves_too_far():
    """A molecule's plan measures plain displacements against the skin."""
    model = _mace()
    graph = structure_to_graph(_structure(periodic=False), model.cutoff)
    dp = DomainParallel(model, skin=0.4)
    plan = dp.plan(graph)
    assert plan.cell is None and plan.ghost_shifts is None
    assert dp.holds(plan, graph.pos + 0.19 / 3 ** 0.5)
    assert not dp.holds(plan, graph.pos + 0.21 / 3 ** 0.5)
    ref, _ = _reference(model, {**_structure(periodic=False), "pos": (graph.pos + 0.1).numpy()})
    _assert_matches(dp.evaluate(plan, graph.pos + 0.1), ref)
    with pytest.raises(ValueError, match="positions for the"):
        dp.evaluate(plan, graph.pos[:3])


def test_an_empty_subdomain_takes_part_in_every_exchange():
    """A process that owns no atom, or holds none at all, joins the exchanges of the others, also
    with frozen parameters (its exchanged rows then depend on nothing)."""
    rng = np.random.default_rng(5)
    blob = rng.uniform(0, 2.5, (12, 3))
    for gap, empty in ((17.5, lambda p: p.n_owned == 0 and p.n_ghost > 0), (27.5, lambda p: p.n_local == 0)):
        pos = np.concatenate([blob, blob[::-1] + [gap, 0.0, 0.0]])
        for model, species in ((_mace(), (1, 6, 8)), (_aimnet2(), (1, 6, 7, 8))):
            model.requires_grad_(False)
            s = {"pos": pos, "atomic_numbers": [species[i % len(species)] for i in range(24)]}
            if isinstance(model, AIMNet2):
                s["total_charge"] = 1.0
            ref, graph = _reference(model, s)
            results = _decomposed(4, graph, model, grid=(4, 1, 1))
            assert any(empty(p) for _, p in results)
            for out, _ in results:
                _assert_matches(out, ref)


def test_refusals():
    """Batches, partial periodicity, mismatched grids, disagreeing processes, wrong refreshes and
    neighbors across a cut axis through the periodicity are refused."""
    model = _mace()
    s = _structure()
    graph = structure_to_graph(s, model.cutoff)
    with pytest.raises(ValueError, match="does not hold"):
        DomainParallel(model, grid=(2, 2, 1)).plan(graph)
    partial = structure_to_graph({**s, "pbc": [True, True, False]}, model.cutoff)
    with pytest.raises(ValueError, match="periodic"):
        DomainParallel(model).plan(partial)
    with pytest.raises(ValueError, match="one structure"):
        DomainParallel(model).plan(collate([graph, graph]))

    # the processes see different structures, so they disagree on the owners
    other = structure_to_graph(_structure(seed=9), model.cutoff)
    dp = DomainParallel(model, grid=(2, 1, 1))
    with pytest.raises(RuntimeError, match="disagree"):
        _threads(2, lambda rank: dp.plan(graph if rank == 0 else other))

    class Bad(type(model)):
        def forward(self, data):
            ops.refresh_ghosts(data.pos[:1])
            return super().forward(data)

    bad = model
    bad.__class__ = Bad
    with pytest.raises(ValueError, match="refresh_ghosts got 1 rows"):
        DomainParallel(bad)(graph)


def test_a_neighbor_across_a_cut_axis_is_refused(monkeypatch):
    """A neighbor list that reaches across a cut axis through the box's periodicity is an error."""
    original = domain.build_neighbor_list

    def across(*args, **kwargs):
        edge_index, cell_shifts = original(*args, **kwargs)
        cell_shifts = cell_shifts.clone()
        cell_shifts[0, 0] = 1
        return edge_index, cell_shifts

    monkeypatch.setattr(domain, "build_neighbor_list", across)
    model = _mace()
    graph = structure_to_graph(_structure(), model.cutoff)
    dp = DomainParallel(model, grid=(2, 1, 1))
    with pytest.raises(RuntimeError, match="cut axis"):
        _threads(2, lambda rank: dp.plan(graph))


def test_model_dtype_from_buffers():
    """A model without floating parameters takes its dtype from its buffers, and has none without either."""
    buffered = torch.nn.Module()
    buffered.register_buffer("index", torch.arange(3))
    assert domain._model_dtype(buffered) is None
    buffered.register_buffer("scale", torch.ones(2, dtype=torch.float32))
    assert domain._model_dtype(buffered) == torch.float32


def test_collectives_call_torch_distributed(monkeypatch):
    """The collectives hand their arguments to torch.distributed: the autograd all-to-all and all-reduce
    on the process group (the whole launch by default), and the all-gather into one tensor per process."""
    import torch.distributed as dist
    import torch.distributed.nn.functional as functional
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    assert domain._process(None) == (0, 1)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    assert domain._process(None) == (1, 2)
    calls = []

    def all_to_all_single(out, x, out_counts, in_counts, group=None):
        calls.append(("a2a", tuple(out.shape), out_counts, in_counts, group))
        return out.copy_(x.flip(0))

    def all_reduce(x, op=None, group=None):
        calls.append(("reduce", op, group))
        return x * 2

    def all_gather(out, x, group=None):
        calls.append(("gather", len(out), group))
        for o in out:
            o.copy_(x)

    monkeypatch.setattr(functional, "all_to_all_single", all_to_all_single)
    monkeypatch.setattr(functional, "all_reduce", all_reduce)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    x = torch.arange(6.0).view(3, 2)
    assert torch.equal(domain._all_to_all(x, [1, 2], [2, 1], None), x.flip(0))
    assert torch.equal(domain._all_reduce(x, "g", op="max"), 2 * x)
    domain._all_reduce(x, None)
    assert [torch.equal(p, x) for p in domain._all_gather(x, "g")] == [True, True]
    assert calls == [("a2a", (3, 2), [1, 2], [2, 1], dist.group.WORLD),
                     ("reduce", dist.ReduceOp.MAX, "g"), ("reduce", dist.ReduceOp.SUM, dist.group.WORLD),
                     ("gather", 2, "g")]


def test_refresh_is_the_identity_outside_a_decomposition():
    """Outside a decomposition the models' ghost refresh returns its input."""
    x = torch.randn(4, 3)
    assert ops.refresh_ghosts(x) is x


_SCRIPT = """
import numpy as np
import torch
import torch.distributed as dist

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.deploy import DomainParallel
from xnn.common.models import ForceStressOutput, build_model
from xnn.gnn.models.aimnet2 import AIMNet2

torch.set_default_dtype(torch.float64)
dist.init_process_group("gloo")
torch.manual_seed(2)
aimnet2 = AIMNet2([1, 6, 7, 8], cutoff=3.5, n_features=4, n_rbf=6,
                  hidden=[[16, 12], [16, 12], [16, 12, 12]], aim_size=10, readout_hidden=[8],
                  n_vector_combinations=3, coulomb=None).eval()
mace = build_model(from_dict({"model": {
    "name": "mace", "cutoff": 3.5, "n_features": 8, "n_interactions": 2,
    "extra": {"species": [1, 6, 7, 8], "max_ell": 2, "correlation": 2, "num_channels": 8,
              "max_L": 1}}}).model).eval()
cell = np.array([[9.0, 0.0, 0.0], [1.5, 8.5, 0.0], [0.7, -1.0, 9.5]])
rng = np.random.default_rng(0)
z = [(1, 6, 7, 8)[i % 4] for i in range(40)]
cases = [{"pos": rng.uniform(0, 1, (40, 3)) @ cell, "cell": cell, "pbc": [True] * 3, "total_charge": 1.0},
         {"pos": rng.uniform(0, 11, (40, 3)), "total_charge": -1.0}]


def check(out, ref, forces=None):
    forces = out["forces"] if forces is None else forces
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-10 * abs(float(ref["energy"])), (out, ref)
    assert float((forces - ref["forces"]).abs().max()) < 1e-10
    if "stress" in ref:
        assert float((out["stress"] - ref["stress"]).abs().max()) < 1e-11


for model in (aimnet2, mace):
    for halo in ("layer", "field"):
        for s in cases:
            s["atomic_numbers"] = z
            graph = structure_to_graph(s, model.cutoff)
            ref = ForceStressOutput(model, compute_stress="cell" in s)(graph)
            check(DomainParallel(model, halo=halo)(graph), ref)

# each process with its own atoms only, through a reshard
dp = DomainParallel(mace, skin=0.5)
graph = structure_to_graph(cases[0], mace.cutoff)
plan = dp.plan(graph)
moved = {**cases[0], "pos": cases[0]["pos"] + rng.normal(0, 0.4, (40, 3))}
target = structure_to_graph(moved, mace.cutoff)
ref = ForceStressOutput(mace)(target)
pos = target.pos[plan.owned]
assert not dp.holds(plan, pos)
plan, pos = dp.reshard(plan, pos)
out = dp.evaluate(plan, pos)
forces = torch.zeros(40, 3)
forces[plan.owned] = out["forces"]
dist.all_reduce(forces)
check(out, ref, forces)
dist.destroy_process_group()
print("ok")
"""


def test_gloo_launch(tmp_path):
    """The exchanges, sums and gathers of a 3-process Gloo launch give the single-process results."""
    script = tmp_path / "domain.py"
    script.write_text(_SCRIPT)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "3",
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0", str(script)],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ok") == 3
