"""The fused kernels of the classical force fields (OPLS, AMBER, CHARMM, DREIDING) against their eager
reference: periodic boxes of randomly rotated copies of each test module's molecule, wrapped into the cell,
and the same molecules as a cluster, with and without the switch, in float64 and float32. Energies (every
term), forces and stress agree to round-off. GPU only.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.ffnn.models import OPLS, Amber, Charmm, Dreiding, MolecularTopology

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the pair kernel runs on a GPU")


def _fixtures(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _box(pos, n_side, spacing, seed=0, unwrap=False):
    """Copies of ``pos`` on a cubic lattice, randomly rotated and displaced, wrapped into the cell (or not)."""
    rng = np.random.default_rng(seed)
    p1 = np.asarray(pos, float)
    p1 = p1 - p1.mean(0)
    out = []
    for site in np.ndindex(n_side, n_side, n_side):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        out.append(p1 @ q.T + (np.array(site) + 0.5) * spacing)
    L = n_side * spacing
    pos = np.concatenate(out) + 0.03 * rng.normal(size=(len(out) * len(p1), 3))
    return (pos if unwrap else pos % L), L


def _replicate(top, n_mol, n1):
    return MolecularTopology.from_bonds(
        top.types * n_mol, [(i + m * n1, j + m * n1) for m in range(n_mol) for i, j in top.bonds],
        impropers=[tuple(a + m * n1 for a in im) for m in range(n_mol) for im in top.impropers],
        bond_orders=list(top.bond_orders) * n_mol or None, charges=list(top.charges) * n_mol or None)


def _systems():
    """(name, model class, library, topology, atomic numbers, positions, box length, model kwargs)."""
    out = []
    opls = _fixtures("test_opls")
    pos, z, types, bonds = opls._butane()
    out.append(("opls", OPLS, "oplsaa", MolecularTopology.from_bonds(types, bonds), z, pos, {}))
    amber = _fixtures("test_amber")
    pos, z, types, bonds, charges = amber._nma()
    out.append(("amber", Amber, "ff94", MolecularTopology.from_bonds(types, bonds, charges=charges), z, pos, {}))
    charmm = _fixtures("test_charmm")
    pos, z, types, bonds, charges, impropers = charmm._nma()
    out.append(("charmm", Charmm, "charmm22",
                MolecularTopology.from_bonds(types, bonds, charges=charges, impropers=impropers), z, pos, {}))
    dreiding = _fixtures("test_dreiding")
    pos, z, top = dreiding._water_dimer(2.9)
    for lib in ("dreiding", "dreiding/X6"):
        out.append((lib, Dreiding, lib, top, z, pos, {"charges": [-0.8, 0.4, 0.4, -0.8, 0.4, 0.4]}))
    # a staggered-ish ethane: the torsion rules; a pyramidal ethylene: the inversions
    model, z = dreiding._ethane_model()
    out.append(("dreiding-ethane", Dreiding, "dreiding", model.topology, z, dreiding._ethane_pos(47.0),
                {"charges": [-0.18, -0.18] + [0.06] * 6}))
    out.append(("dreiding-ethane-morse", Dreiding, "dreiding", model.topology, z, dreiding._ethane_pos(47.0),
                {"charges": [-0.18, -0.18] + [0.06] * 6, "bond_style": "morse"}))
    pos, z, top = dreiding._ethylene(25.0)
    out.append(("dreiding-ethylene", Dreiding, "dreiding", top, z, pos, {"charges": [-0.2, -0.2] + [0.1] * 4}))
    return out


SYSTEMS = _systems()


@pytest.mark.parametrize("layout", ["box", "cluster", "small"])
@pytest.mark.parametrize("width", [0.0, 1.5])
@pytest.mark.parametrize("case", SYSTEMS, ids=[c[0] for c in SYSTEMS])
def test_the_kernels_match_the_eager_terms(case, width, layout):
    """``small``: one molecule in a cell shorter than the cutoff (each atom sees its own images)."""
    name, cls, lib, top1, z1, pos1, kw = case
    n1, cutoff = len(z1), 8.0
    n_side, spacing = (1, 7.5) if layout == "small" else (3, 9.0)
    periodic = layout != "cluster"
    pos, L = _box(pos1, n_side, spacing)
    n_mol = n_side ** 3
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, _replicate(top1, n_mol, n1), cutoff=cutoff, switch_width=width, **kw)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1) * n_mol)}
    if periodic:
        s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
    else:
        # the same molecules as a cluster: each unwrapped about its lattice site
        s["pos"] = torch.tensor(_box(pos1, n_side, spacing, unwrap=True)[0])
    for dtype, rtol, atol in ((torch.float64, 1e-11, 1e-10), (torch.float32, 2e-5, 2e-4)):
        results = []
        for fast in (False, True):
            m = model.to(device="cuda", dtype=dtype).eval()
            m.use_fast = fast
            g = structure_to_graph(s, cutoff, device="cuda")
            g.compute_dtype = dtype
            results.append(ForceStressOutput(m, compute_forces=True, compute_stress=periodic)(g))
        ref, got = results
        for key in ref:
            if key.startswith("e_") or key in ("energy", "forces", "stress"):
                a, b = ref[key].double(), got[key].double()
                scale = float(a.abs().max()) or 1.0
                assert torch.allclose(a, b, rtol=rtol, atol=atol * scale), \
                    (name, str(dtype), key, float((a - b).abs().max()), scale)


STEPPED = SYSTEMS


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("case", STEPPED, ids=[c[0] for c in STEPPED])
def test_the_stepper_follows_the_eager_model(case, dtype):
    """Steps on a neighbor list with a skin (plain, then replayed as a CUDA graph, then after a rebuild) give the
    energies and forces of the eager model, the padding of the capacity buffers included."""
    from xnn.ffnn.models import ClassicalStepper
    name, cls, lib, top1, z1, pos1, kw = case
    n1, n_side, cutoff = len(z1), 3, 8.0
    pos, L = _box(pos1, n_side, 9.0)
    n_mol = n_side ** 3
    kw = dict(kw)
    if "charges" in kw:
        kw["charges"] = list(kw["charges"]) * n_mol
    model = cls(lib, _replicate(top1, n_mol, n1), cutoff=cutoff, **kw).to(device="cuda", dtype=dtype).eval()
    stepper = ClassicalStepper(model, skin=1.0, warmup=2)
    rng = np.random.default_rng(1)
    reference = ForceStressOutput(model, compute_forces=True)
    rtol, atol = (1e-10, 1e-9) if dtype == torch.float64 else (2e-5, 2e-4)
    for step in range(7):
        moved = pos + rng.normal(scale=0.02 if step != 5 else 0.4, size=pos.shape)
        s = {"pos": torch.tensor(moved), "atomic_numbers": torch.tensor(list(z1) * n_mol),
             "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
        got = stepper(s)
        g = structure_to_graph(s, cutoff, device="cuda")
        g.compute_dtype = dtype
        want = reference(g)
        e, f = got["energy"].double(), got["forces"].double()
        e0, f0 = want["energy"].detach().double(), want["forces"].detach().double()
        assert torch.isfinite(e).all() and torch.isfinite(f).all(), (name, step)
        scale = float(f0.abs().max()) or 1.0
        assert torch.allclose(e, e0, rtol=rtol, atol=atol), (name, step, float((e - e0).abs().max()))
        assert torch.allclose(f, f0, rtol=rtol, atol=atol * scale), (name, step, float((f - f0).abs().max()))
    assert stepper.replays >= 2 and stepper.rebuilds >= 2


def test_the_stepper_refuses_a_cell_shorter_than_its_list():
    from xnn.ffnn.models import ClassicalStepper
    name, cls, lib, top1, z1, pos1, kw = SYSTEMS[0]
    pos, L = _box(pos1, 1, 7.5)
    model = cls(lib, top1, cutoff=8.0, **kw).to("cuda").eval()
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1)),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    with pytest.raises(ValueError, match="width"):
        ClassicalStepper(model)(s)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_blocks_without_an_active_pair_keep_the_stress(dtype):
    """Whole blocks of pairs the kernel skips (each pair once, ``i < k``: here every ``i > k`` direction first)
    add nothing to the cell gradient: energy, forces and stress are the eager terms'."""
    name, cls, lib, top1, z1, pos1, kw = SYSTEMS[0]
    n1, n_side, cutoff = len(z1), 2, 8.0
    pos, L = _box(pos1, n_side, 9.0)
    model = cls(lib, _replicate(top1, n_side ** 3, n1), cutoff=cutoff, **kw).to(device="cuda", dtype=dtype).eval()
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(list(z1) * n_side ** 3),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    g = structure_to_graph(s, cutoff, device="cuda")
    g.compute_dtype = dtype
    order = torch.argsort((g.edge_index[0] < g.edge_index[1]).to(torch.int8), stable=True)
    g.edge_index, g.cell_shifts = g.edge_index[:, order], g.cell_shifts[order]
    assert int((g.edge_index[0] > g.edge_index[1]).sum()) >= 4 * 128
    results = []
    for fast in (False, True):
        model.use_fast = fast
        out = ForceStressOutput(model, compute_stress=True)(g)
        results.append({k: out[k].detach().double() for k in ("energy", "forces", "stress")})
    torch.cuda.synchronize()
    rtol = 1e-11 if dtype == torch.float64 else 2e-5
    for key in ("energy", "forces", "stress"):
        a, b = results[0][key], results[1][key]
        assert float((a - b).abs().max()) <= rtol * float(a.abs().max()), (key, float((a - b).abs().max()))
