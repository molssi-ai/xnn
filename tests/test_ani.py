"""Tests for the faithful ANI model (registered as ``ani``) and its AEV.

Covers the symmetry-function / AEV building blocks, rotation/translation/
permutation invariance of energies and equivariance of forces, the ANI-1,
ANI-1x, and ANI-1ccx presets (AEV lengths and per-element architectures), self
atomic energies, config building with upstream key translation, and -- when
``torchani`` is available -- element-for-element AEV fidelity to
``torchani.AEVComputer`` and full energy/force fidelity via weight transplant of
both the pretrained ANI-1x and ANI-1ccx models.
"""
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicGraph, structure_to_graph
from xnn.common.data.neighborlist import build_neighbor_list
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.dnn.featurizers import AEV
from xnn.dnn.featurizers.aev import ANI2X_SPECIES, _angle_shifts, _even_shifts
from xnn.dnn.featurizers.symmetry_functions import build_triplets
from xnn.dnn.models.ani import (
    ANI,
    ANI1CCX_SELF_ENERGIES,
    ANI1X_HIDDEN,
    ANI1X_SELF_ENERGIES,
    ANI2X_HIDDEN,
    ANI2X_SELF_ENERGIES,
)

# H, C, N, O
SPECIES = [1, 6, 7, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=8, cutoff=5.2, R=None, shift=0.0, seed=1, perm=None):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 4, (n, 3)) + shift
    if R is not None:
        pos = pos @ R.T
    z = np.array(([8, 1, 1, 6, 7, 1, 6, 1] * n)[:n], dtype=np.int64)
    pos = pos.astype(np.float64)
    if perm is not None:
        pos, z = pos[perm], z[perm]
    return structure_to_graph({"pos": pos, "atomic_numbers": z}, cutoff)


# building blocks                                                             #
def test_build_triplets_matches_bruteforce():
    """The vectorised triplet builder matches a per-atom brute force."""
    rng = torch.Generator().manual_seed(0)
    for _ in range(50):
        n = int(torch.randint(1, 8, (1,), generator=rng))
        e = int(torch.randint(0, 20, (1,), generator=rng))
        if e == 0:
            continue
        ei = torch.stack([torch.randint(0, n, (e,), generator=rng),
                          torch.randint(0, n, (e,), generator=rng)])
        f, s, c = build_triplets(ei, n)

        want = set()
        for i in range(n):
            edges = (ei[1] == i).nonzero().flatten().tolist()
            for a in range(len(edges)):
                for b in range(a + 1, len(edges)):
                    want.add((i, tuple(sorted((edges[a], edges[b])))))
        got = {(int(cc), tuple(sorted((int(a), int(b)))))
               for a, b, cc in zip(f, s, c)}
        assert got == want


def test_aev_preset_dimensions():
    """ANI-1 and ANI-1x presets have the published AEV lengths (768 / 384)."""
    assert AEV.ani1(SPECIES).output_dim == 768
    assert AEV.ani1x(SPECIES).output_dim == 384
    # radial: n_shf * n_species; angular: n_grid * n_pairs
    a = AEV.ani1x(SPECIES)
    assert a.radial.output_dim == 16 * 4
    assert a.angular.output_dim == (4 * 8) * 10


def test_ani2x_aev_dimensions():
    """ANI-2x has the published 1008-length AEV over seven elements."""
    a = AEV.ani2x(ANI2X_SPECIES)
    assert a.output_dim == 1008
    # radial: 16 shifts * 7 species; angular: (8 radial * 4 angular) * 28 pairs
    assert a.radial.output_dim == 16 * 7
    assert a.angular.output_dim == (8 * 4) * 28
    assert math.isclose(a.radial.cutoff, 5.1)
    assert math.isclose(a.angular.cutoff, 3.5)


def test_shift_recipe_matches_torchani_grid():
    """The evenly-spaced-shift recipe reproduces torchani's ANI-1x ShfR/ShfZ."""
    shfr = _even_shifts(5.2, 16)
    assert math.isclose(shfr[0], 0.9) and math.isclose(shfr[-1], 4.93125)
    shfz = _angle_shifts(8)
    assert math.isclose(shfz[0], math.pi / 16)


# model behaviour                                                             #
def test_forward_and_forces():
    """ANI produces per-structure energy and conservative forces."""
    model = ForceStressOutput(ANI.ani1x(SPECIES))
    g = _graph()
    out = model(g)
    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (g.num_nodes, 3)
    assert torch.isfinite(out["forces"]).all()


def test_invariance_and_equivariance():
    """Energy is invariant to rotation/translation/permutation; forces rotate."""
    torch.manual_seed(0)
    model = ForceStressOutput(ANI.ani1x(SPECIES))
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    e0 = model(_graph())["energy"]
    e_rot = model(_graph(R=R.numpy()))["energy"]
    e_trans = model(_graph(shift=3.0))["energy"]
    perm = np.array([3, 0, 5, 1, 7, 2, 6, 4])
    e_perm = model(_graph(perm=perm))["energy"]

    assert torch.allclose(e0, e_rot, atol=1e-8)
    assert torch.allclose(e0, e_trans, atol=1e-8)
    assert torch.allclose(e0, e_perm, atol=1e-8)

    f0 = model(_graph())["forces"]
    f_rot = model(_graph(R=R.numpy()))["forces"]
    assert torch.allclose(f0 @ R.T, f_rot, atol=1e-8)


def test_self_energies_shift_total():
    """Per-species self energies add a constant offset to the total energy."""
    torch.manual_seed(0)
    m = ANI(SPECIES)
    g = _graph()
    e_before = m(g)["energy"].item()

    vals = {1: 0.5, 6: 1.0, 7: 2.0, 8: 3.0}
    for z, v in vals.items():
        m._self_energies_by_z[z] = v
    e_after = m(g)["energy"].item()

    counts = {z: int((g.atomic_numbers == z).sum()) for z in SPECIES}
    expected = sum(counts[z] * vals[z] for z in SPECIES)
    assert math.isclose(e_after - e_before, expected, rel_tol=1e-6)


def test_ani1x_architecture():
    """ANI-1x preset builds torchani's per-element widths and self energies."""
    m = ANI.ani1x(SPECIES)
    for z in SPECIES:
        outs = [l.out_features for l in m.element_nets.nets[str(z)]
                if hasattr(l, "out_features")]
        assert outs == list(ANI1X_HIDDEN[z]) + [1]
    for z in SPECIES:
        assert math.isclose(float(m._self_energies_by_z[z]),
                            ANI1X_SELF_ENERGIES[z])


def test_ani1ccx_architecture_matches_ani1x():
    """ANI-1ccx preset shares the ANI-1x architecture; only self energies differ."""
    ccx = ANI.ani1ccx(SPECIES)
    x = ANI.ani1x(SPECIES)
    assert ccx.featurizer.output_dim == x.featurizer.output_dim == 384
    for z in SPECIES:
        def widths(m, z=z):
            return [l.out_features for l in m.element_nets.nets[str(z)] if hasattr(l, "out_features")]
        assert widths(ccx) == widths(x)
        assert math.isclose(float(ccx._self_energies_by_z[z]),
                            ANI1CCX_SELF_ENERGIES[z])
        assert not math.isclose(float(ccx._self_energies_by_z[z]),
                                ANI1X_SELF_ENERGIES[z])


def test_ani2x_architecture():
    """ANI-2x preset builds torchani's seven-element widths and self energies."""
    m = ANI.ani2x()                       # default seven-element set
    assert m.species == ANI2X_SPECIES
    assert m.featurizer.output_dim == 1008
    for z in ANI2X_SPECIES:
        outs = [l.out_features for l in m.element_nets.nets[str(z)]
                if hasattr(l, "out_features")]
        assert outs == list(ANI2X_HIDDEN[z]) + [1]
        assert math.isclose(float(m._self_energies_by_z[z]),
                            ANI2X_SELF_ENERGIES[z])


def test_ani2x_energy_forces_on_halogens():
    """ANI-2x runs on an S/F/Cl-containing structure with finite forces."""
    from xnn.dnn.models.ani import ANI as _ANI
    model = ForceStressOutput(_ANI.ani2x())
    z = np.array([6, 16, 9, 17, 8, 7, 1, 1], dtype=np.int64)   # C S F Cl O N H H
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 4, (8, 3)).astype(np.float64)
    g = structure_to_graph({"pos": pos, "atomic_numbers": z}, 5.1)
    out = model(g)
    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (8, 3)
    assert torch.isfinite(out["forces"]).all()


# config                                                                      #
def test_registered():
    """ANI is discoverable through the model registry."""
    assert "ani" in available_models()


def test_from_config_preset_and_translation():
    """from_config honours presets, symbols, and upstream key spellings."""
    cfg = from_dict({"model": {
        "name": "ani",
        "extra": {"preset": "ani-1x", "species": ["H", "C", "N", "O"]},
    }})
    m = build_model(cfg.model)
    assert m.featurizer.output_dim == 384

    cfg_ccx = from_dict({"model": {
        "name": "ani", "extra": {"preset": "ani-1ccx"},
    }})
    m_ccx = build_model(cfg_ccx.model)
    assert m_ccx.featurizer.output_dim == 384
    assert math.isclose(float(m_ccx._self_energies_by_z[6]),
                        ANI1CCX_SELF_ENERGIES[6])

    # ani-2x defaults to its seven-element set (no species given)
    cfg_2x = from_dict({"model": {
        "name": "ani", "extra": {"preset": "ani-2x"},
    }})
    m_2x = build_model(cfg_2x.model)
    assert m_2x.species == ANI2X_SPECIES
    assert m_2x.featurizer.output_dim == 1008
    assert math.isclose(float(m_2x._self_energies_by_z[16]),
                        ANI2X_SELF_ENERGIES[16])

    # torchani/NeuroChem spellings Rcr/Rca translate to radial/angular cutoff
    cfg2 = from_dict({"model": {
        "name": "ani", "Rcr": 4.6, "Rca": 3.1, "atomic_numbers": [1, 6, 7, 8],
    }})
    m2 = build_model(cfg2.model)
    assert math.isclose(m2.featurizer.radial.cutoff, 4.6)
    assert math.isclose(m2.featurizer.angular.cutoff, 3.1)


def test_behler_parrinello_conventions():
    """radial_prefactor / angular_cos_factor default to torchani, override to BP."""
    ani = AEV.ani1x(SPECIES)
    assert math.isclose(ani.radial.prefactor, 0.25)
    assert math.isclose(ani.angular.cos_factor, 0.95)
    bp = AEV(SPECIES, radial_prefactor=1.0, angular_cos_factor=1.0)
    assert math.isclose(bp.radial.prefactor, 1.0)
    assert math.isclose(bp.angular.cos_factor, 1.0)


# torchani fidelity (only where torchani is installed)                          #
def _xnn_graph(Z, pos, cutoff):
    Z = torch.as_tensor(Z, dtype=torch.long)
    pos = torch.as_tensor(pos, dtype=torch.float64)
    ei, cs = build_neighbor_list(pos, cutoff)
    return AtomicGraph(pos=pos, atomic_numbers=Z, edge_index=ei, cell_shifts=cs,
                       batch=torch.zeros(len(Z), dtype=torch.long),
                       n_atoms=torch.tensor([len(Z)]))


@pytest.mark.parametrize("preset,grid", [
    ("ani1x", (5.2, 3.5, [16.0], 16, [8.0], [32.0], 4, 8)),
    ("ani1", (4.6, 3.1, [16.0], 32, [8.0], [8.0], 8, 8)),
])
def test_torchani_aev_fidelity(preset, grid):
    """xnn AEV matches torchani.AEVComputer element-for-element."""
    torchani = pytest.importorskip("torchani")
    Rcr, Rca, EtaR, nR, EtaA, Zeta, nA, nZ = grid
    def t(x):
        return torch.tensor(x, dtype=torch.float64)
    tani = torchani.AEVComputer(Rcr, Rca, t(EtaR), t(_even_shifts(Rcr, nR)),
                                t(EtaA), t(Zeta), t(_even_shifts(Rca, nA)),
                                t(_angle_shifts(nZ)), 4)
    aev = getattr(AEV, preset)(SPECIES)

    rng = np.random.default_rng(0)
    Z = np.array([6, 7, 8, 1, 1, 1], dtype=np.int64)
    pos = rng.uniform(0, 3, (6, 3))
    idx = {1: 0, 6: 1, 7: 2, 8: 3}
    x = aev(_xnn_graph(Z, pos, aev.cutoff))
    sp = torch.tensor([[idx[z] for z in Z]])
    _, ref = tani((sp, torch.as_tensor(pos[None])))
    assert torch.allclose(x, ref[0], atol=1e-10)


@pytest.mark.parametrize("preset,upstream", [
    ("ani1x", "ANI1x"),
    ("ani1ccx", "ANI1ccx"),
])
def test_torchani_energy_force_fidelity(preset, upstream):
    """Transplanting torchani's pretrained ANI-1x/ANI-1ccx weights reproduces E and F."""
    torchani = pytest.importorskip("torchani")
    model = getattr(torchani.models, upstream)(
        periodic_table_index=False).double()
    member = model.neural_networks[0]

    xa = getattr(ANI, preset)(SPECIES)
    nets = dict(member.named_children())
    zsym = {1: "H", 6: "C", 7: "N", 8: "O"}
    for z in SPECIES:
        src = [l for l in nets[zsym[z]] if isinstance(l, torch.nn.Linear)]
        dst = [l for l in xa.element_nets.nets[str(z)]
               if isinstance(l, torch.nn.Linear)]
        for s, d in zip(src, dst):
            d.weight.data = s.weight.data.clone()
            d.bias.data = s.bias.data.clone()

    rng = np.random.default_rng(1)
    Z = np.array([6, 7, 8, 1, 1, 1], dtype=np.int64)
    pos = rng.uniform(0, 3, (6, 3))
    idx = {1: 0, 6: 1, 7: 2, 8: 3}

    g = _xnn_graph(Z, pos, xa.cutoff)
    g.pos.requires_grad_(True)
    out = ForceStressOutput(xa)(g)

    coords = torch.as_tensor(pos[None]).clone().requires_grad_(True)
    sp = torch.tensor([[idx[z] for z in Z]])
    e = model.energy_shifter(member(model.aev_computer((sp, coords)))).energies
    f = -torch.autograd.grad(e.sum(), coords)[0][0]

    assert abs(out["energy"].item() - e.item()) < 1e-6
    assert torch.allclose(out["forces"].detach(), f, atol=1e-6)


def test_torchani_ani2x_aev_fidelity():
    """xnn ANI-2x AEV matches torchani.AEVComputer over all seven elements."""
    torchani = pytest.importorskip("torchani")
    def t(x):
        return torch.tensor(x, dtype=torch.float64)
    tani = torchani.AEVComputer(
        5.1, 3.5, t([19.7]), t(_even_shifts(5.1, 16, start=0.8)),
        t([12.5]), t([14.1]), t(_even_shifts(3.5, 8, start=0.8)),
        t(_angle_shifts(4)), 7)
    aev = AEV.ani2x(ANI2X_SPECIES)

    rng = np.random.default_rng(0)
    Z = np.array([6, 7, 8, 16, 9, 17, 1, 1], dtype=np.int64)
    pos = rng.uniform(0, 3, (8, 3))
    idx = {z: i for i, z in enumerate(ANI2X_SPECIES)}
    x = aev(_xnn_graph(Z, pos, aev.cutoff))
    sp = torch.tensor([[idx[z] for z in Z]])
    _, ref = tani((sp, torch.as_tensor(pos[None])))
    assert torch.allclose(x, ref[0], atol=1e-10)


def test_torchani_ani2x_energy_force_fidelity():
    """Transplanting torchani's pretrained ANI-2x weights reproduces E and F."""
    torchani = pytest.importorskip("torchani")
    model = torchani.models.ANI2x(periodic_table_index=False).double()
    member = model.neural_networks[0]

    xa = ANI.ani2x(ANI2X_SPECIES)
    nets = dict(member.named_children())
    zsym = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 9: "F", 17: "Cl"}
    for z in ANI2X_SPECIES:
        src = [l for l in nets[zsym[z]] if isinstance(l, torch.nn.Linear)]
        dst = [l for l in xa.element_nets.nets[str(z)]
               if isinstance(l, torch.nn.Linear)]
        for s, d in zip(src, dst):
            d.weight.data = s.weight.data.clone()
            d.bias.data = s.bias.data.clone()

    rng = np.random.default_rng(1)
    Z = np.array([6, 7, 8, 16, 9, 17, 1, 1, 1], dtype=np.int64)
    pos = rng.uniform(0, 3.5, (9, 3))
    idx = {z: i for i, z in enumerate(ANI2X_SPECIES)}

    g = _xnn_graph(Z, pos, xa.cutoff)
    g.pos.requires_grad_(True)
    out = ForceStressOutput(xa)(g)

    coords = torch.as_tensor(pos[None]).clone().requires_grad_(True)
    sp = torch.tensor([[idx[z] for z in Z]])
    e = model.energy_shifter(member(model.aev_computer((sp, coords)))).energies
    f = -torch.autograd.grad(e.sum(), coords)[0][0]

    # larger absolute energies (S/Cl cores ~ -400 Ha) than ANI-1x, so the
    # tolerance is looser in absolute terms but still ~1e-10 relative.
    assert abs(out["energy"].item() - e.item()) < 1e-5
    assert torch.allclose(out["forces"].detach(), f, atol=1e-5)


@pytest.mark.parametrize("preset", ["ani1x", "ani2x"])
def test_scriptable_and_lammps_export(tmp_path, preset):
    """The TorchScript core equals the eager model, and the LAMMPS export gives its energy and forces."""
    from xnn.common.deploy import export_to_lammps
    species = SPECIES if preset == "ani1x" else [1, 6, 7, 8, 9, 16, 17]
    model = getattr(ANI, preset)(species).eval()
    g = _graph(n=9, cutoff=model.cutoff)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12
    path = str(tmp_path / "ani_lammps.pt")
    export_to_lammps(model, model.cutoff, path)
    loaded = torch.jit.load(path)
    cell = torch.eye(3, dtype=g.pos.dtype) * 30.0
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, cell)
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-10
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-10


# AEV fast path (fused Triton kernels)                                         #

def _set_fast_terms(model, on):
    for term in (model.featurizer.radial, model.featurizer.angular):
        term.fast_active = on


def test_fast_aev_derivatives():
    """The fused AEV (both terms) against finite differences: first derivatives with respect
    to the edge vectors and second derivatives (the gradient's own derivatives with respect
    to the edge vectors and the incoming gradient, force training)."""
    pytest.importorskip("triton")
    device = "cuda" if torch.cuda.is_available() else "cpu"     # the interpreter on a CPU
    model = ANI.ani1x(SPECIES).double().to(device)
    g = _graph(n=6, cutoff=model.cutoff, seed=3).to(device)
    vec0 = g.edge_vectors().detach()
    _set_fast_terms(model, True)
    try:
        def f(vec):
            return model.featurizer.compute(g.atomic_numbers, g.edge_index, vec, g.num_nodes)
        vec = vec0.clone().requires_grad_(True)
        # the gathers before the kernels scatter their gradients with atomic adds on a GPU
        assert torch.autograd.gradcheck(f, (vec,), fast_mode=True, nondet_tol=1e-12)
        assert torch.autograd.gradgradcheck(f, (vec,), fast_mode=True, nondet_tol=1e-12)
    finally:
        _set_fast_terms(model, False)


def test_fast_aev_matches_the_reference():
    """The fused AEV, its gradient and the derivatives of that gradient equal the reference."""
    pytest.importorskip("triton")
    device = "cuda" if torch.cuda.is_available() else "cpu"     # the interpreter on a CPU
    model = ANI.ani1x(SPECIES).double().to(device)
    g = _graph(n=9, cutoff=model.cutoff, seed=4).to(device)
    vec0 = g.edge_vectors().detach()
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(g.num_nodes, model.featurizer.output_dim, dtype=torch.float64, generator=gen).to(device)
    out = {}
    for on in (False, True):
        _set_fast_terms(model, on)
        vec = vec0.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        aev = model.featurizer.compute(g.atomic_numbers, g.edge_index, vec, g.num_nodes)
        (gv,) = torch.autograd.grad((aev * ww).sum(), vec, create_graph=True)
        gw, gvv = torch.autograd.grad(gv.pow(2).sum(), (ww, vec))
        out[on] = (aev.detach(), gv.detach(), gw, gvv)
    _set_fast_terms(model, False)
    for ref, fast in zip(out[False], out[True]):
        assert (fast - ref).abs().max() <= 1e-12 * ref.abs().max()


def _fast_batch(model, dtype):
    from xnn.common.data import collate
    graphs = [_graph(n=14, cutoff=model.cutoff, seed=1), _graph(n=9, cutoff=model.cutoff, seed=2)]
    batch = collate(graphs).to("cuda")
    batch.pos = batch.pos.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("preset", ["ani1x", "ani2x"])
def test_fast_path_parity(dtype, preset):
    """The fused AEV gives the reference energies and forces."""
    pytest.importorskip("triton")
    species = SPECIES if preset == "ani1x" else [1, 6, 7, 8, 9, 16, 17]
    model = getattr(ANI, preset)(species).to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model).eval()
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(model, dtype))
        assert model.featurizer.angular.fast_active == use_fast
        out[use_fast] = {k: res[k].detach().double() for k in ("energy", "forces")}
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    for key, ref in out[False].items():
        err = (out[True][key] - ref).abs().max() / ref.abs().max()
        assert err < tol, (key, float(err))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fast_path_training(dtype):
    """Force training through the fused AEV gives the reference's weight gradients."""
    pytest.importorskip("triton")
    model = ANI.ani1x(SPECIES).to("cuda", dtype)
    grads = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast).train()
        model.zero_grad(set_to_none=True)
        out = ForceStressOutput(model).train()(_fast_batch(model, dtype))
        (out["energy"].pow(2).sum() + out["forces"].pow(2).sum()).backward()
        grads[use_fast] = {n: p.grad.detach().double().clone() for n, p in model.named_parameters()
                           if p.grad is not None}
    assert grads[False].keys() == grads[True].keys()
    tol = 1e-11 if dtype == torch.float64 else 1e-4
    for name, g in grads[False].items():
        err = (grads[True][name] - g).abs().max() / g.abs().max().clamp_min(1e-30)
        assert err < tol, (name, float(err))


def test_symmetry_functions_without_triplets_and_fast_support(monkeypatch):
    """A graph without neighbour triplets (two atoms) gives an all-zero angular descriptor on the
    reference and the fused path; ``forward`` on a graph equals ``compute``; the fused path is
    offered where Triton supports the device."""
    from xnn.dnn.fast import aev as fast_aev
    from xnn.dnn.featurizers.symmetry_functions import AngularSymmetryFunctions, RadialSymmetryFunctions
    angular = AngularSymmetryFunctions(SPECIES, cutoff=3.5).double()
    radial = RadialSymmetryFunctions(SPECIES, cutoff=3.5).double()
    pair = structure_to_graph({"pos": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]), "atomic_numbers": [6, 1]}, 3.5)
    for on in (False, True):
        angular.fast_active = on
        out = angular(pair)
        assert out.shape == (2, angular.output_dim) and not out.any()
    angular.fast_active = False
    g = _graph(n=6, cutoff=3.5)
    assert torch.equal(angular(g), angular.compute(g.atomic_numbers, g.edge_index, g.edge_vectors(), g.num_nodes))
    monkeypatch.setattr(fast_aev, "supported", lambda device, dtype: dtype == torch.float64)
    for term in (angular, radial):
        assert term.fast_supported(torch.device("cuda"), torch.float64)
        assert not term.fast_supported(torch.device("cuda"), torch.float16)


def test_fast_angular_general_grid_matches_the_reference():
    """A grid with several zeta values takes the flat-lane kernels (the presets the factored
    ones): values, gradient and both second derivatives equal the reference."""
    pytest.importorskip("triton")
    from xnn.dnn.featurizers.symmetry_functions import AngularSymmetryFunctions
    device = "cuda" if torch.cuda.is_available() else "cpu"     # the interpreter on a CPU
    term = AngularSymmetryFunctions(SPECIES, cutoff=3.5, etas=(4.0, 8.0), zetas=(8.0, 32.0), rs=(0.9, 1.6),
                                    theta_s=(0.2, 1.2, 2.2), cos_factor=0.95).double().to(device)
    g = _graph(n=8, cutoff=3.5, seed=5).to(device)
    vec0 = g.edge_vectors().detach()
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(g.num_nodes, term.output_dim, dtype=torch.float64, generator=gen).to(device)
    out = {}
    for on in (False, True):
        term.fast_active = on
        vec = vec0.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        aev = term.compute(g.atomic_numbers, g.edge_index, vec, g.num_nodes)
        (gv,) = torch.autograd.grad((aev * ww).sum(), vec, create_graph=True)
        gw, gvv = torch.autograd.grad(gv.pow(2).sum(), (ww, vec))
        out[on] = (aev.detach(), gv.detach(), gw, gvv)
    term.fast_active = False
    for ref, fast in zip(out[False], out[True]):
        assert (fast - ref).abs().max() <= 1e-12 * ref.abs().max()
