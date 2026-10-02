"""Float32 models on large boxes: the geometry and the energy totals stay accurate.

A float32 model used to form its edge vectors from float32 absolute coordinates,
so every vector carried an error of about 6e-8 times the coordinates' size, and
to sum its per-atom energies in float32, which loses tenths of an eV at 1e4 to
1e5 atoms. The deploy paths now keep the positions and cell in float64 (the
model computes in its own dtype through ``AtomicGraph.compute_dtype``) and every
per-structure energy is accumulated in float64.
"""
import copy
import math

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402
from xnn.common.models.ops import structure_sum  # noqa: E402

WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])
SPACING = 3.104


def _water(n, shift=0.0, seed=2):
    rng = np.random.default_rng(seed)
    pos = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q *= np.sign(np.linalg.det(q))
                pos.extend((WATER - WATER[0]) @ q.T + np.array([i, j, k]) * SPACING
                           + rng.normal(scale=0.1, size=3))
    return {"pos": np.array(pos) + shift, "atomic_numbers": [8, 1, 1] * n ** 3,
            "cell": np.eye(3) * SPACING * n, "pbc": [True] * 3}


def _nequip(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        core = build_model(from_dict({"model": {
            "name": "nequip", "cutoff": 4.0, "n_features": 8, "n_interactions": 2, "l_max": 1,
            "species": [1, 8], "avg_num_neighbors": 30.0}}).model)
    finally:
        torch.set_default_dtype(old)
    return ForceStressOutput(core.to(dtype), compute_stress=True).eval()


def _forces(model, s, geometry):
    g = structure_to_graph({**s, "pos": torch.tensor(s["pos"], dtype=geometry),
                            "cell": torch.tensor(s["cell"], dtype=geometry)}, 4.0)
    out = model(copy.copy(g))
    return out["forces"].detach().double(), out


def test_structure_sum_is_float64_and_exact():
    values = (-2041.37 + 1e-3 * torch.randn(50_000, generator=torch.Generator().manual_seed(0))).float()
    total = structure_sum(values, torch.zeros(50_000, dtype=torch.long), 1)
    exact = math.fsum(float(v) for v in values.double())
    assert total.dtype == torch.float64
    assert abs(float(total) - exact) < 1e-6
    assert abs(float(values.sum()) - exact) > 1e-3            # what a float32 sum loses


def test_energy_totals_are_float64_for_a_float32_model():
    model = _nequip(torch.float32)
    _, out = _forces(model, _water(2), torch.float32)
    assert out["node_energy"].dtype == torch.float32
    assert out["energy"].dtype == torch.float64
    assert abs(float(out["energy"]) - float(out["node_energy"].double().sum())) < 1e-9


def test_float64_geometry_removes_the_coordinate_size_error():
    exact = _nequip(torch.float64)
    model = _nequip(torch.float32)
    base = _water(3)
    reference, _ = _forces(exact, base, torch.float64)
    far = _water(3, shift=40 * SPACING * 3)                  # ~370 A from the origin
    err32 = float((_forces(model, far, torch.float32)[0] - reference).abs().max())
    err64 = float((_forces(model, far, torch.float64)[0] - reference).abs().max())
    near = float((_forces(model, base, torch.float64)[0] - reference).abs().max())
    assert err64 < err32 / 5                                 # the float32 coordinates' error is gone
    assert err64 < 2 * near + 1e-7                           # and independent of the shift


def test_ase_forces_do_not_depend_on_where_the_box_sits():
    ase = pytest.importorskip("ase")
    from xnn.common.deploy import XNNCalculator

    s = _water(3)
    atoms = ase.Atoms(numbers=s["atomic_numbers"], positions=s["pos"], cell=s["cell"], pbc=True)
    atoms.calc = XNNCalculator(_nequip(torch.float32), cutoff=4.0)
    f0 = atoms.get_forces().copy()
    atoms.positions += 1000.0
    f1 = atoms.get_forces()
    assert np.abs(f1 - f0).max() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_vesin_retries_after_releasing_cached_memory(monkeypatch):
    from xnn.common.data import neighborlist

    if not neighborlist._HAS_VESIN:
        pytest.skip("needs vesin")
    real = neighborlist.VesinNeighborList
    calls = {"compute": 0, "empty_cache": 0}

    class Flaky:
        def __init__(self, **kwargs):
            self.inner = real(**kwargs)

        def compute(self, **kwargs):
            calls["compute"] += 1
            if calls["compute"] == 1:
                raise RuntimeError("cudaMalloc failed: out of memory")
            return self.inner.compute(**kwargs)

    empty = torch.cuda.empty_cache
    monkeypatch.setattr(neighborlist, "VesinNeighborList", Flaky)
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: (calls.__setitem__("empty_cache", calls["empty_cache"] + 1), empty()))
    s = _water(3)
    g = structure_to_graph({**s, "pos": torch.tensor(s["pos"]), "cell": torch.tensor(s["cell"])},
                           4.0, device="cuda")
    assert calls == {"compute": 2, "empty_cache": 1}
    assert g.edge_index.shape[1] > 0


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("block", [None, 7])
def test_edge_vectors_match_the_plain_product(periodic, block, monkeypatch):
    # edge_vectors forms the vectors in blocks of edges and keeps only the
    # integer shifts for its backward; values, first and second derivatives are
    # those of pos[dst] - pos[src] + shifts @ cell
    from dataclasses import replace

    from xnn.common.data import atomic_data

    if block is not None:
        monkeypatch.setattr(atomic_data, "EDGE_BLOCK", block)
    s = _water(2)
    if periodic:
        g = structure_to_graph({**s, "pos": torch.tensor(s["pos"]),
                                "cell": torch.tensor(s["cell"])}, 4.0)
        assert bool(g.cell_shifts.abs().sum() > 0)
    else:
        g = structure_to_graph({"pos": torch.tensor(s["pos"]),
                                "atomic_numbers": s["atomic_numbers"]}, 4.0)
    pos = g.pos.to(torch.float64).requires_grad_(True)
    cell = g.cell.to(torch.float64).requires_grad_(True) if periodic else None
    args = (pos, cell) if periodic else (pos,)

    def ours(p, c=None):
        return replace(g, pos=p, cell=c).edge_vectors()

    def plain(p, c=None):
        src, dst = g.edge_index
        vec = p[dst] - p[src]
        return vec + g.cell_shifts.to(p.dtype) @ c[0] if c is not None else vec

    assert torch.equal(ours(*args), plain(*args))
    probe = torch.randn_like(ours(*args))
    grads = [torch.autograd.grad((f(*args) * probe).sum(), args) for f in (ours, plain)]
    for a, b in zip(*grads):
        assert torch.allclose(a, b, rtol=0, atol=1e-12)
    assert torch.autograd.gradcheck(ours, args)
    assert torch.autograd.gradgradcheck(ours, args)
    # a float32 model's vectors from the float64 geometry
    out = replace(g, pos=pos, cell=cell, compute_dtype=torch.float32).edge_vectors()
    assert out.dtype == torch.float32 and torch.equal(out, plain(*args).float())
