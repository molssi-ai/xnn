"""The classical kernels on the CPU (xnn.common.cpu_kernels: the Triton kernels as C++ loops) against the eager
terms: the model's kernel evaluation (energies of every term, forces, stress) and the stepper's steps, on the
systems of test_classical_fast, float64 and float32."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common import cpu_kernels
from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput

pytestmark = pytest.mark.skipif(not cpu_kernels.available(), reason="no C++ compiler or Triton")


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fx = _load("test_classical_fast")
SYSTEMS = fx.SYSTEMS


@pytest.mark.parametrize("layout", ["box", "cluster", "small"])
@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_the_cpu_kernels_match_the_eager_terms(case, layout, monkeypatch):
    import xnn.ffnn.models.classical_fast as cf
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    name, cls, lib, top1, z1, pos1, kw = case
    n1, cutoff = len(z1), 8.0
    n_side, spacing = (1, 7.5) if layout == "small" else (2, 9.0)
    periodic = layout != "cluster"
    pos, L = fx._box(pos1, n_side, spacing)
    n_mol = n_side ** 3
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, fx._replicate(top1, n_mol, n1), cutoff=cutoff, switch_width=1.5, **kw)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1) * n_mol)}
    if periodic:
        s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
    else:
        s["pos"] = torch.tensor(fx._box(pos1, n_side, spacing, unwrap=True)[0])
    for dtype, rtol, atol in ((torch.float64, 1e-11, 1e-10), (torch.float32, 2e-5, 2e-4)):
        results = []
        for fast in (False, True):
            m = model.to(dtype=dtype).eval()
            m.use_fast = fast
            g = structure_to_graph(s, cutoff)
            g.compute_dtype = dtype
            results.append(ForceStressOutput(m, compute_forces=True, compute_stress=periodic)(g))
        ref, got = results
        for key in ref:
            if key.startswith("e_") or key in ("energy", "forces", "stress"):
                a, b = ref[key].double(), got[key].double()
                scale = float(a.abs().max()) or 1.0
                assert torch.allclose(a, b, rtol=rtol, atol=atol * scale), \
                    (name, str(dtype), key, float((a - b).abs().max()), scale)


@pytest.mark.parametrize("L,half", [(21.0, False), (46.0, False), (46.0, True)], ids=["all_pairs", "grid", "half"])
def test_the_cpu_searches_find_every_pair_once(L, half, monkeypatch):
    """The CPU searches of a stepper's list (every pair of a small cell, or a grid of half-cutoff cells, its whole
    stencil or half of it) find the pairs of the reference search with their shifts; positions spread over 1.5
    edges are left to it."""
    import xnn.common.data.neighborlist as nl
    from xnn.common.data.neighborlist import unique_pairs
    from xnn.ffnn.models.classical_cpu import cpu_pairs, decode_images
    monkeypatch.setattr(nl, "GRID_CPU_ATOMS", 0)
    rng = np.random.default_rng(3)
    n, cutoff = int(0.03 * L ** 3), 9.5
    # uniform in the cell, some moved by an edge: positions within 1.3 edges
    frac = rng.uniform(0.0, 1.0, size=(n, 3))
    flip = rng.uniform(size=(n, 3)) < 0.5
    frac = frac - (flip & (frac > 0.85)) + (flip & (frac < 0.15))
    pos = torch.tensor(frac * L)
    cell, pbc = torch.eye(3, dtype=torch.float64) * L, torch.ones(3, dtype=torch.bool)
    i, k, img, ptr = cpu_pairs(pos, cutoff, cell, pbc, half=half)
    assert torch.equal(ptr[1:] - ptr[:-1], torch.bincount(i.long(), minlength=n))
    s = decode_images(img)
    # a pair in the row of its higher atom: the reverse pair, its shift negated
    swap = (k < i)[:, None]
    lo, hi = torch.minimum(i, k), torch.maximum(i, k)
    got = set(zip(lo.tolist(), hi.tolist(), *torch.where(swap, -s, s).T.tolist()))
    i0, k0, s0 = unique_pairs(pos, cutoff, cell, pbc)
    want = set(zip(i0.tolist(), k0.tolist(), *s0.long().T.tolist()))
    assert len(got) == int(i.shape[0]) and got == want
    assert half == bool((k < i).any())
    pos[0, 0] += 2.0 * L
    assert cpu_pairs(pos, cutoff, cell, pbc, half=half) is None


@pytest.mark.parametrize("search", ["all_pairs", "grid", "unwrapped"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_the_cpu_stepper_follows_the_eager_model(case, dtype, search, monkeypatch):
    """Steps on a neighbor list with a skin (a rebuild included) give the eager model's energies and forces; the
    list from testing every pair of the cell, from the grid search, or (molecules moved by whole edges, the
    positions spread over two edges) from the search with general shifts."""
    import xnn.common.data.neighborlist as nl
    from xnn.ffnn.models import ClassicalStepper
    monkeypatch.setattr(nl, "GRID_CPU_ATOMS", 10 ** 9 if search == "all_pairs" else 0)
    name, cls, lib, top1, z1, pos1, kw = case
    n1, n_side, cutoff = len(z1), 3, 8.0
    pos, L = fx._box(pos1, n_side, 9.0)
    n_mol = n_side ** 3
    if search == "unwrapped":
        pos = pos + np.repeat(np.arange(n_mol) % 3, n1)[:, None] * np.array([L, 0.0, 0.0])
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, fx._replicate(top1, n_mol, n1), cutoff=cutoff, **kw).to(dtype=dtype).eval()
    stepper = ClassicalStepper(model, skin=1.0)
    rng = np.random.default_rng(1)
    reference = ForceStressOutput(model, compute_forces=True)
    rtol, atol = (1e-10, 1e-9) if dtype == torch.float64 else (2e-5, 2e-4)
    for step in range(5):
        moved = pos + rng.normal(scale=0.02 if step != 3 else 0.4, size=pos.shape)
        s = {"pos": torch.tensor(moved), "atomic_numbers": torch.tensor(list(z1) * n_mol),
             "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
        got = stepper(s)
        g = structure_to_graph(s, cutoff)
        g.compute_dtype = dtype
        want = reference(g)
        e, f = got["energy"].double(), got["forces"].double()
        e0, f0 = want["energy"].detach().double(), want["forces"].detach().double()
        assert torch.isfinite(e).all() and torch.isfinite(f).all(), (name, step)
        scale = float(f0.abs().max()) or 1.0
        assert torch.allclose(e, e0, rtol=rtol, atol=atol), (name, step, float((e - e0).abs().max()))
        assert torch.allclose(f, f0, rtol=rtol, atol=atol * scale), (name, step, float((f - f0).abs().max()))
        # the forces alone (no energy summed) are the same forces
        only = stepper(s, energy=False)
        assert "energy" not in only and torch.allclose(only["forces"].double(), f, rtol=1e-12, atol=1e-12 * scale)
    assert stepper.rebuilds >= 2
    # the one-pass CPU search with image indices, except where a pair's shift may exceed one edge
    assert stepper.images == (search != "unwrapped")


def _losses(model, s, cutoff, dtype, fast, w):
    """Energy, forces, energy- and force-loss parameter gradients and the Hessian-vector product along ``w``."""
    model.use_fast = fast
    g = structure_to_graph(s, cutoff)
    g.compute_dtype = dtype
    pos = g.pos.detach().clone().requires_grad_(True)
    g.pos = pos
    energy = model(g)["energy"].sum()
    params = [p for p in model.parameters() if p.requires_grad]
    (forces,) = torch.autograd.grad(-energy, pos, create_graph=True)
    e_grads = torch.autograd.grad(energy, params, retain_graph=True, allow_unused=True)
    f_grads = torch.autograd.grad((forces * w).sum(), params + [pos], allow_unused=True)
    return energy.detach(), forces.detach(), e_grads, f_grads[:-1], f_grads[-1]


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_the_cpu_training_kernels_match_the_eager_derivatives(case, periodic, monkeypatch):
    """The kernels differentiable twice: energy- and force-loss gradients of every trained parameter and the
    Hessian-vector product in the positions."""
    import xnn.ffnn.models.classical_fast as cf
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    name, cls, lib, top1, z1, pos1, kw = case
    n1, n_side, cutoff = len(z1), 2, 8.0
    pos, L = fx._box(pos1, n_side, 9.0)
    n_mol = n_side ** 3
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, fx._replicate(top1, n_mol, n1), cutoff=cutoff, trainable="all", **kw)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1) * n_mol)}
    if periodic:
        s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
    else:
        s["pos"] = torch.tensor(fx._box(pos1, n_side, 9.0, unwrap=True)[0])
    w = torch.tensor(np.random.default_rng(2).normal(size=(n1 * n_mol, 3)))
    for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 2e-4)):
        m = model.to(dtype=dtype).train()
        ref = _losses(m, s, cutoff, dtype, False, w.to(dtype))
        got = _losses(m, s, cutoff, dtype, True, w.to(dtype))
        pairs = [("energy", ref[0], got[0]), ("forces", ref[1], got[1]), ("hvp", ref[4], got[4])]
        pairs += [(f"energy grad {j}", a, b) for j, (a, b) in enumerate(zip(ref[2], got[2]))]
        pairs += [(f"force grad {j}", a, b) for j, (a, b) in enumerate(zip(ref[3], got[3]))]
        for label, a, b in pairs:
            if a is None or b is None:
                assert a is None or float(a.abs().max()) == 0.0, (name, label)
                assert b is None or float(b.abs().max()) == 0.0, (name, label)
                continue
            a, b = a.double(), b.double()
            scale = float(a.abs().max()) or 1.0
            assert float((a - b).abs().max()) <= tol * scale, (name, str(dtype), label,
                                                                 float((a - b).abs().max()), scale)


def _water_box(n_side, spacing, single):
    """DREIDING water dimers (or single waters, ``single``) on a lattice, unwrapped: a cluster or a cell."""
    dreiding = _load("test_dreiding")
    pos1, z1, top1 = dreiding._water_dimer(2.9)
    charges = [-0.8, 0.4, 0.4, -0.8, 0.4, 0.4]
    if single:
        from xnn.ffnn.models import MolecularTopology
        pos1, z1, charges = pos1[:3], z1[:3], charges[:3]
        top1 = MolecularTopology.from_bonds(top1.types[:3], [(0, 1), (0, 2)], bond_orders=[1.0, 1.0])
    pos, L = fx._box(pos1, n_side, spacing, unwrap=True)
    n_mol = n_side ** 3
    return pos, list(z1) * n_mol, fx._replicate(top1, n_mol, len(z1)), charges * n_mol, L


@pytest.mark.parametrize("layout", ["cluster", "far", "opls"])
def test_the_cpu_stepper_on_general_lists_follows_the_eager_model(layout):
    """Steps without the one-pass grid search (a cluster of DREIDING water dimers with their hydrogen bonds, in
    the harmonic angle form; single waters in a cell searched with general shifts, no hydrogen bond in reach; a
    cluster of OPLS butanes with their 1-4 pairs): the eager model's energies and forces; the autograd
    evaluation of a step (the one a GPU records) gives the same."""
    from xnn.ffnn.models import ClassicalStepper, Dreiding
    periodic = layout == "far"
    if layout == "opls":
        name, cls, lib, top1, z1, pos1, kw = SYSTEMS[0]
        pos, L = fx._box(pos1, 2, 9.0, unwrap=True)
        z = list(z1) * 8
        model = cls(lib, fx._replicate(top1, 8, len(z1)), cutoff=8.0, **kw).double().eval()
    else:
        pos, z, top, charges, L = _water_box(2, 9.0, single=periodic)
        model = Dreiding("dreiding", top, cutoff=8.0, charges=charges, angle_style="harmonic",
                         hbond_cutoff=3.0 if periodic else None).double().eval()
    stepper = ClassicalStepper(model, skin=1.0)
    reference = ForceStressOutput(model, compute_forces=True)
    rng = np.random.default_rng(4)
    for step in range(3):
        moved = pos + rng.normal(scale=0.02 if step != 1 else 0.6, size=pos.shape)
        s = {"pos": torch.tensor(moved), "atomic_numbers": torch.tensor(z)}
        if periodic:
            s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
        got = stepper(s)
        want = reference(structure_to_graph(s, 8.0))
        e0, f0 = want["energy"].detach(), want["forces"].detach()
        scale = float(f0.abs().max())
        assert torch.allclose(got["energy"], e0, rtol=1e-10, atol=1e-9), (layout, step)
        assert torch.allclose(got["forces"], f0, rtol=1e-10, atol=1e-9 * scale), (layout, step)
        assert not stepper._outdated(torch.tensor(moved))
        e, f = stepper._run()
        assert torch.allclose(e.reshape(1), e0, rtol=1e-10, atol=1e-9), (layout, step)
        assert torch.allclose(f, f0, rtol=1e-10, atol=1e-9 * scale), (layout, step)
    assert stepper.rebuilds >= 2 and not stepper.images
    assert stepper._outdated(torch.tensor(pos) + 0.6)
    assert (stepper.p14 is not None) == (layout == "opls")
    if layout != "opls":
        assert (float(stepper.hb[1].abs().max()) == 0.0) == periodic


def test_the_cpu_stepper_refuses_other_models_and_small_cells():
    """A model without kernel terms and a cell no wider than cutoff plus skin are refused."""
    from xnn.ffnn.models import ClassicalStepper, ReaxFF
    with pytest.raises(ValueError, match="kernel terms"):
        ClassicalStepper(ReaxFF("CHO_cho_2008", conventions="lammps"))
    name, cls, lib, top1, z1, pos1, kw = SYSTEMS[0]
    pos, L = fx._box(pos1, 1, 7.5)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1)),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    with pytest.raises(ValueError, match="width"):
        ClassicalStepper(cls(lib, top1, cutoff=8.0, **kw).eval())(s)


def _water_graph(scale_cell):
    pos, z, top, charges, L = _water_box(2, 9.0, single=False)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    g = structure_to_graph(s, 8.0)
    g.pos = g.pos.detach().clone().requires_grad_(True)
    if scale_cell:
        g.cell = g.cell.detach().clone().requires_grad_(True)
    return g, top, charges


def test_the_cpu_training_kernels_differentiate_the_incoming_gradients(monkeypatch):
    """A force loss of an energy scaled by a trained factor: its gradient in the factor (through the incoming
    gradients of the kernels' energies, pair and bonded) is the force loss at factor one, as in the eager terms."""
    import xnn.ffnn.models.classical_fast as cf
    from xnn.ffnn.models import Dreiding
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    g, top, charges = _water_graph(False)
    model = Dreiding("dreiding", top, cutoff=8.0, charges=charges).double().train()
    w = torch.tensor(np.random.default_rng(5).normal(size=tuple(g.pos.shape)))
    results = []
    for fast in (False, True):
        model.use_fast = fast
        factor = torch.ones((), dtype=torch.float64, requires_grad=True)
        energy = factor * model(g)["energy"].sum()
        (forces,) = torch.autograd.grad(-energy, g.pos, create_graph=True)
        loss = (forces * w).sum()
        (grad,) = torch.autograd.grad(loss, factor)
        assert torch.allclose(grad, loss.detach(), rtol=1e-12, atol=1e-12)
        results.append(grad)
    assert torch.allclose(results[0], results[1], rtol=1e-10, atol=1e-10)


def test_the_kernels_with_a_cell_gradient_refuse_a_second_derivative(monkeypatch):
    """With the cell differentiated (a stress), the kernels are differentiable once: the forces are those of
    the eager terms, and a second derivative raises instead of coming out zero."""
    import xnn.ffnn.models.classical_fast as cf
    from xnn.ffnn.models import Dreiding
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    g, top, charges = _water_graph(True)
    model = Dreiding("dreiding", top, cutoff=8.0, charges=charges).double().eval()
    forces = []
    for fast in (False, True):
        model.use_fast = fast
        energy = model(g)["energy"].sum()
        (f,) = torch.autograd.grad(-energy, g.pos, create_graph=True)
        forces.append(f)
    assert torch.allclose(forces[0], forces[1], rtol=1e-10, atol=1e-10)
    with pytest.raises(RuntimeError, match="differentiable once"):
        torch.autograd.grad(forces[1].sum(), g.pos)


def test_the_fast_path_settings_and_the_training_kernels_limits(monkeypatch):
    """Where the kernels run (a CUDA device with Triton; not for a stress loss with trained parameters), the
    use_fast setting, no second derivatives without gradients, and the training kernels' bound on torsion
    series."""
    import xnn.ffnn.models.classical_fast as cf
    from xnn.ffnn.models import Dreiding
    from xnn.ffnn.models.classical_kernels import available
    from xnn.ffnn.models.classical_train import MAX_SERIES, bonded_terms_twice
    assert cf.fast_supported(torch.device("cuda"), torch.float64)
    assert not cf.fast_supported(torch.device("cuda"), torch.float16)
    assert not available(torch.device("cpu"))
    g, top, charges = _water_graph(True)
    model = Dreiding("dreiding", top, cutoff=8.0, charges=charges).eval()
    assert model.set_use_fast(False) is model and model.use_fast is False
    with torch.no_grad():
        assert not cf.needs_twice(model, g)
    assert not cf.needs_twice(model, g) and cf.needs_twice(model)
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: True)
    assert cf.wants_fast(model, True, g)
    # a trained parameter (of the force field, or given) with a cell gradient: the reference terms
    assert not cf.wants_fast(model, True, g, torch.ones(1, requires_grad=True))
    model.ff.params["bond_k"].requires_grad_(True)
    assert not cf.wants_fast(model, True, g)
    g.cell = g.cell.detach()
    assert cf.wants_fast(model, True, g)
    pos = torch.zeros(4, 3, dtype=torch.float64)
    idx = torch.tensor([[0], [1], [2], [3]])
    with pytest.raises(ValueError, match="at most"):
        bonded_terms_twice(pos, None, torch.zeros(4, dtype=torch.long), idx, None, "torsion", (), torch.float64,
                           1e-12, options={"FORM": 0, "S": MAX_SERIES + 1, "NMAX": 1})


def test_the_cpu_kernels_on_atoms_without_bonds(monkeypatch):
    """Atoms without bonded terms in a cell (no image shifts to find): the kernels' energies and forces are the
    eager terms'."""
    import xnn.ffnn.models.classical_fast as cf
    from xnn.ffnn.models import Dreiding, MolecularTopology
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    rng = np.random.default_rng(7)
    pos = (np.stack(np.meshgrid(*[np.arange(2)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) * 4.0
    pos = pos + rng.normal(scale=0.2, size=pos.shape)
    model = Dreiding("dreiding", MolecularTopology.from_bonds(["C_3"] * 8, []), cutoff=3.9,
                     charges=[0.1, -0.1] * 4).double().eval()
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.full((8,), 6),
         "cell": torch.eye(3, dtype=torch.float64) * 8.0, "pbc": torch.ones(3, dtype=torch.bool)}
    results = []
    for fast in (False, True):
        model.use_fast = fast
        out = ForceStressOutput(model, compute_forces=True)(structure_to_graph(s, 3.9))
        results.append((out["energy"].detach(), out["forces"].detach()))
    for a, b in zip(*results):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-10)
