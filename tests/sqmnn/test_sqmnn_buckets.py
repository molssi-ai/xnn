"""NDDO semiempirical methods (sqmnn): batches padded to buckets.

A batch padded to the shapes of a bucket (ghost atoms without orbitals or core charge, masked pairs) gives the
energies, forces, charges, dipoles, parameter gradients and Hessians of the batch itself; batches of different
compositions with the same rounded sizes share a bucket; the index tensors of a fixed-shape layout rebuild it
exactly, also on the buffers of a static copy; a new element starts new buckets. On a GPU the bucket's graphs
replay for shuffled batches with the reference's numbers.
"""
import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, hessian
from xnn.sqmnn.models import PM3, PM6, PM6ORG, PM7, nddo, scf

WATER = (np.array([[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.05, -0.75, -0.49]]), [8, 1, 1])
FORMAMIDE = (np.array([[0, 0.42, 0], [1.2, 0.5, 0.02], [-0.72, -0.73, 0], [-0.55, 1.36, 0.01],
                       [-1.73, -0.69, 0.05], [-0.27, -1.63, -0.03]]), [6, 8, 7, 1, 1, 1])
METHYL = (np.array([[0, 0, 0.02], [1.08, 0, 0], [-0.54, 0.94, 0.01], [-0.55, -0.93, 0]]), [6, 1, 1, 1])
AMMONIUM = (np.array([[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6], [0.6, -0.6, -0.6]]),
            [7, 1, 1, 1, 1])
CARBON = (np.zeros((1, 3)), [6])
# the water geometry with nitrogen: another composition of the same rounded sizes
AMINO = (WATER[0], [7, 1, 1])


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture
def buckets(monkeypatch):
    """Buckets on the CPU too (without graphs there)."""
    monkeypatch.setitem(nddo.BUCKETS, "cpu", True)


def _graph(system, charge=0.0, mult=None, device="cpu"):
    pos, z = system
    s = {"pos": torch.as_tensor(np.asarray(pos, dtype=np.float64)), "atomic_numbers": np.asarray(z),
         "total_charge": charge}
    if mult is not None:
        s["spin_multiplicity"] = mult
    return structure_to_graph(s, 0.1, device=device)


def _batch(device="cpu"):
    return collate([_graph(FORMAMIDE, device=device), _graph(CARBON, mult=3.0, device=device),
                    _graph(AMMONIUM, charge=1.0, device=device), _graph(WATER, device=device)])


def _outputs(model, data, train=False):
    model.train(train)
    out = ForceStressOutput(model).train(train)(data)
    values = [out[k].detach() for k in ("energy", "node_energy", "forces", "charges", "dipole")]
    if train:
        w = torch.linspace(0.3, 1.1, out["forces"].numel(), device=data.pos.device).reshape(out["forces"].shape)
        model.zero_grad()
        (1e-3 * out["energy"].pow(2).sum() + 1e-2 * (out["forces"] * w).pow(2).sum()).backward()
        values += [p.grad.detach().clone() for p in model.parameters()]
    return values


def _close(a, b, tol):
    for x, y in zip(a, b):
        assert x.shape == y.shape
        assert torch.allclose(x.cpu(), y.cpu(), rtol=0.0, atol=tol * max(1.0, float(y.abs().max())))


@pytest.mark.parametrize("cls", [PM3, PM6, PM7, PM6ORG])
def test_padded_batches_match_the_batch(cls, buckets):
    """A batch with a triplet atom, a cation and closed shells, padded to its bucket, gives the energies, per-atom
    energies, forces, charges and dipoles of the batch itself."""
    settings = dict(tol_density=1e-12, tol_energy=1e-13)
    padded = cls(**settings)
    _close(_outputs(padded, _batch()), _outputs(cls(use_fast=False, **settings), _batch()), 1e-10)
    (key,) = list(padded._plans)
    assert key[0] == "bucket" and key[2] >= 14 + 2


def test_padded_training_and_hessians_match(buckets):
    """The parameter gradients of an energy and force loss and the Hessians of a padded batch are those of the
    batch."""
    groups = ["USS", "BETAS", "BETAP", "ZP", "GSS"]
    settings = dict(tol_density=1e-12, tol_energy=1e-13)
    padded, ref = PM7(trainable=groups, **settings), PM7(trainable=groups, use_fast=False, **settings)
    _close(_outputs(padded, _batch(), True), _outputs(ref, _batch(), True), 1e-9)
    data = collate([_graph(WATER), _graph(METHYL, mult=2.0)])
    for H, R in zip(hessian(PM7(**settings), data), hessian(PM7(use_fast=False, **settings), data)):
        assert torch.allclose(H, R, rtol=0.0, atol=1e-8)


def test_bucket_sizes_and_shared_buckets(buckets, monkeypatch):
    """Sizes round up to 2 to 4 times a power of two (4 to 8 with a finer spacing); two batches of other
    compositions with the same rounded sizes share a bucket and give index tensors of the same shapes; a new
    element starts new buckets and no warm start from the old bucket's density."""
    sizes = (1, 4, 5, 9, 17, 100, 243)
    assert [nddo._round_up(x) for x in sizes] == [1, 4, 6, 12, 24, 128, 256]
    monkeypatch.setattr(nddo, "BUCKET_SPACING", 4)
    assert [nddo._round_up(x) for x in sizes] == [1, 4, 5, 10, 20, 112, 256]
    monkeypatch.undo()
    monkeypatch.setitem(nddo.BUCKETS, "cpu", True)
    model = PM6()
    shapes = []
    for system in (WATER, AMINO):
        data = _graph(system)
        q = model._element_quantities([1, 7, 8])
        _, padded, bucket = model._bucket_plan(data.atomic_numbers, data.batch, 1, [1, 7, 8], q, data.pos)
        shapes.append(([tuple(t.shape) for t in padded.tensors], bucket.key))
        assert int(padded.real.sum()) == len(system[1]) and float(padded.pair_real.sum()) == len(system[1]) * (
            len(system[1]) - 1) // 2
    assert shapes[0] == shapes[1]
    model = PM6(reuse_density=True)
    model(_graph(WATER))
    first = set(model._plans)
    moved = (WATER[0] + 0.01, WATER[1])
    model(collate([_graph(moved)]))
    assert set(model._plans) == first
    model(_graph(METHYL, mult=2.0))
    model(_graph(moved))
    assert set(model._plans) - first and model._bucket_elements == (1, 6, 8)


def test_smaller_batches_fill_kept_buckets(buckets):
    """A batch that a kept bucket holds with at most twice each size (atoms, pairs, basis functions) fills that
    bucket; on its own it has a bucket of its rounded sizes."""
    uz = [1, 6, 7, 8]

    def key(model, system, charge=0.0):
        data = _graph(system, charge)
        q = model._element_quantities(uz)
        return model._bucket_plan(data.atomic_numbers, data.batch, 1, uz, q, data.pos)[2].key

    model = PM6()
    large = key(model, FORMAMIDE)
    assert large[2:5] == (8, 16, 16)
    assert key(model, AMMONIUM, 1.0) == large
    assert key(PM6(), AMMONIUM, 1.0)[2:5] == (8, 12, 8)


def test_fixed_layout_rebuilds_from_its_tensors(buckets):
    """A fixed-shape layout rebuilt from its index tensors, and a static copy loaded with another batch of the
    bucket, gather and scatter as the batch's own layout does."""
    model = PM6()
    layouts = []
    for system in (WATER, AMINO):
        data = _graph(system)
        q = model._element_quantities([1, 7, 8])
        plan, _, _ = model._bucket_plan(data.atomic_numbers, data.batch, 1, [1, 7, 8], q, data.pos)
        layouts.append(plan.layout)
    first, second = layouts
    static = scf.static_layout(first)
    scf.load_layout(static, second)
    rebuilt = scf.layout_from_tensors(second.n_struct, second.n_basis, second.k, scf.layout_tensors(second))
    n, k = second.n_basis, second.k
    M = torch.arange(n * n, dtype=torch.float64).reshape(1, n, n)
    blocks = torch.arange(second.atom_struct.shape[0] * k * k, dtype=torch.float64).reshape(-1, k, k)
    pairs = torch.arange(second.pair_a.shape[0] * k * k, dtype=torch.float64).reshape(-1, k, k)
    for lay in (static, rebuilt):
        assert lay.fixed and lay.padded_pairs
        assert torch.equal(lay.gather_atom(M), second.gather_atom(M))
        assert torch.equal(lay.gather_pair(M), second.gather_pair(M))
        assert torch.equal(lay.scatter(blocks, pairs), second.scatter(blocks, pairs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cls", [PM7, PM6ORG])
def test_bucket_graphs_replay_shuffled_batches_on_gpu(cls):
    """On a GPU, shuffled batches of one size replay their buckets' graphs (build with the core-core terms, SCF,
    density response) with the reference's energies, forces and force-loss parameter gradients, those of trained
    core-core parameters included, also after graphs are evicted."""
    systems = [(FORMAMIDE, {}), (WATER, {}), (METHYL, {"mult": 2.0}), (AMMONIUM, {"charge": 1.0}),
               (CARBON, {"mult": 3.0})]
    groups = ["USS", "BETAS", "BETAP", "XFAC", "ALPB"]
    settings = dict(tol_density=1e-12, tol_energy=1e-13)
    fast = cls(trainable=groups, **settings).cuda()
    ref = cls(trainable=groups, use_fast=False, **settings).cuda()
    fast._scf_graphs.capacity = 2
    rng = np.random.default_rng(0)
    for _ in range(14):
        idx = rng.permutation(len(systems))[:3].tolist()

        def data(idx=idx):
            return collate([_graph(systems[i][0], device="cuda", **systems[i][1]) for i in idx])
        _close(_outputs(fast, data(), True), _outputs(ref, data(), True), 1e-8)
    assert any(g for b in fast._plans.values() for g in b.graphs.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_bucket_graphs_take_each_batchs_electrons():
    """Water and NH2+ fill one bucket with other electron counts: alternating after the capture, each replay
    occupies the orbitals of its own batch."""
    settings = dict(tol_density=1e-12, tol_energy=1e-13)
    fast = PM7(**settings).cuda()
    ref = PM7(use_fast=False, **settings).cuda()
    for system, charge in [(WATER, 0.0), (AMINO, 1.0)] * 4:
        _close(_outputs(fast, _graph(system, charge, device="cuda")),
               _outputs(ref, _graph(system, charge, device="cuda")), 1e-8)
    # the first water batch has a bucket of its own (nitrogen comes in with NH2+); the other seven share one
    assert sorted(b.uses for b in fast._plans.values()) == [1, 7] and fast._scf_graphs.entries
