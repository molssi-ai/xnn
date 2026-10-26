"""SchNet tests: manuscript fidelity (NIPS 2017 / DTNN conventions),
physical invariances, TorchScript deployment, and config handling.

The centerpiece is ``_reference_energy``: an independent, loop-based
implementation of the paper's equations (embedding eq 3, Gaussian RBF
expansion, cfconv eq 2, the Fig. 2 interaction block and readout, and the
DTNN standardization) that reuses only the model's *parameters*. Given the
same weights, the model must reproduce it to float64 precision.
"""
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.common.models.ops import shifted_softplus
from xnn.gnn.models.schnet import SchNet

SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, 4, (n, 3)),
            "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _graph(n=6, cutoff=30.0, seed=0, periodic=False):
    s = _structure(n, seed)
    if periodic:
        s["cell"] = np.eye(3) * 6.0
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _small(cutoff_fn=None, **kw):
    torch.manual_seed(0)
    kw.setdefault("n_features", 16)
    kw.setdefault("n_interactions", 2)
    kw.setdefault("n_rbf", 20)
    kw.setdefault("cutoff", 5.0)
    return SchNet(cutoff_fn=cutoff_fn, **kw)


# registry / defaults

def test_registered():
    assert "schnet" in available_models()


def test_paper_defaults():
    """SchNet() is the NIPS 2017 architecture: F=64, T=3, RBF grid
    0..30 A every 0.1 A with gamma = 10 A^-2, ssp activations, no cutoff
    envelope."""
    m = SchNet()
    assert m.embedding.weight.shape[1] == 64
    assert len(m.interactions) == 3
    assert m.rbf.centers.shape == (301,)
    assert abs(float(m.rbf.centers[1] - m.rbf.centers[0]) - 0.1) < 1e-6
    assert m.rbf.gamma == 10.0
    assert m.cutoff_fn is None


def test_invalid_cutoff_fn():
    with pytest.raises(ValueError):
        SchNet(cutoff_fn="polynomial")


# manuscript equations

def test_shifted_softplus_is_paper_ssp():
    """ssp(x) = ln(0.5 e^x + 0.5) with ssp(0) = 0 (NIPS paper, sec. 4.2)."""
    x = torch.linspace(-6, 6, 101, dtype=torch.float64)
    ref = torch.log(0.5 * torch.exp(x) + 0.5)
    assert torch.allclose(shifted_softplus(x), ref, atol=1e-14)
    assert float(shifted_softplus(torch.tensor(0.0)).abs()) < 1e-15


def test_rbf_matches_paper():
    """e_k(r) = exp(-gamma (r - mu_k)^2) on the stated center grid."""
    m = SchNet()
    r = torch.tensor([0.7, 1.3, 2.9], dtype=torch.float32)
    mu = torch.linspace(0.0, 30.0, 301)
    ref = torch.exp(-10.0 * (r[:, None] - mu) ** 2)
    assert torch.allclose(m.rbf(r), ref, atol=1e-7)


def _reference_energy(model, Z, pos):
    """Loop-based re-implementation of the paper's forward pass (float64)."""
    def ssp(t):
        return torch.log(0.5 * torch.exp(t) + 0.5)

    def dense(lin, t):
        return t @ lin.weight.T + lin.bias

    n = len(Z)
    x = model.embedding.weight[Z]                                 # eq 3
    mu = model.rbf.centers
    for block in model.interactions:
        xw = dense(block.lin_in, x)                               # atom-wise
        conv = torch.zeros_like(x)
        for i in range(n):
            for j in range(n):                                    # cfconv, eq 2
                if i == j:
                    continue
                r = torch.linalg.norm(pos[i] - pos[j])
                e = torch.exp(-model.rbf.gamma * (r - mu) ** 2)
                fn = block.cfconv.filter_net
                W = ssp(dense(fn[2], ssp(dense(fn[0], e))))       # 2 dense+ssp
                if model.cutoff_fn is not None:
                    rc = model.cutoff_fn.cutoff
                    W = W * (0.5 * (math.cos(math.pi * float(r) / rc) + 1.0)
                             * (float(r) < rc))
                conv[i] = conv[i] + xw[j] * W
        v = dense(block.lin_out, ssp(dense(block.lin_mid, conv)))
        x = x + v                                                 # residual
    e_hat = dense(model.readout[2], ssp(dense(model.readout[0], x))).squeeze(-1)
    node_e = (model.energy_scale * e_hat + model.energy_shift
              + model.atom_ref.weight[Z, 0])                      # DTNN scale
    return node_e.sum()


@pytest.mark.parametrize("cutoff_fn", [None, "cosine"])
def test_forward_matches_paper_equations(cutoff_fn):
    """Same weights -> the model reproduces the independent equation-by-
    equation implementation to float64 precision."""
    torch.manual_seed(3)
    model = _small(cutoff_fn=cutoff_fn, cutoff=6.0).double()
    # make the zero-initialized head and standardization non-trivial
    torch.nn.init.normal_(model.readout[-1].weight)
    torch.nn.init.normal_(model.readout[-1].bias)
    model.set_energy_scale_shift(0.37, -1.42)
    model.set_atomic_energies(SPECIES, [-0.5, -1.0, -2.0])

    s = _structure()
    g = structure_to_graph(s, 6.0)  # 6.0 > any pair distance: complete graph
    with torch.no_grad():
        e_model = model(g)["energy"]
        e_ref = _reference_energy(model,
                                  torch.tensor(s["atomic_numbers"]),
                                  torch.tensor(s["pos"], dtype=torch.float64))
    assert abs(float(e_model) - float(e_ref)) < 1e-11


def test_zero_init_head_and_standardization():
    """Freshly built, E^hat = 0 so E = sum_i (shift + atom_ref[Z_i]) -- the
    DTNN 'good starting point'."""
    m = _small(energy_shift=-2.5)
    m.set_atomic_energies(SPECIES, [-1.0, -2.0, -3.0])
    g = _graph(cutoff=5.0)
    e = float(m(g)["energy"])
    expected = 6 * (-2.5) + 2 * (-1.0 - 2.0 - 3.0)  # Z = [1,6,8,1,6,8]
    assert abs(e - expected) < 1e-5


# physical properties

def test_energy_invariance_forces_equivariance():
    torch.manual_seed(1)
    model = ForceStressOutput(_small()).double()
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, (6, 3))
    z = [1, 6, 8, 1, 6, 8]
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    t = rng.normal(size=(1, 3))

    o0 = model(structure_to_graph({"pos": pos, "atomic_numbers": z}, 5.0))
    o1 = model(structure_to_graph({"pos": pos @ R.T + t,
                                   "atomic_numbers": z}, 5.0))
    Rt = torch.tensor(R, dtype=torch.float64)
    assert abs(float(o0["energy"]) - float(o1["energy"])) < 1e-10
    assert torch.allclose(o1["forces"].detach(),
                          o0["forces"].detach() @ Rt.T, atol=1e-10)


def test_permutation_invariance():
    torch.manual_seed(2)
    model = _small().double()
    s = _structure()
    perm = np.array([3, 1, 5, 0, 4, 2])
    e0 = model(structure_to_graph(s, 5.0))["energy"]
    e1 = model(structure_to_graph(
        {"pos": s["pos"][perm],
         "atomic_numbers": list(np.array(s["atomic_numbers"])[perm])},
        5.0))["energy"]
    assert abs(float(e0) - float(e1)) < 1e-11


def test_forces_are_minus_gradient():
    """ForceStressOutput's autograd forces match central finite differences
    of the SchNet energy (energy conservation by construction, eq 4)."""
    torch.manual_seed(4)
    model = ForceStressOutput(_small()).double()
    s = _structure()
    out = model(structure_to_graph(s, 5.0))
    f = out["forces"].detach().numpy()

    def energy(pos):
        return float(model(structure_to_graph(
            {"pos": pos, "atomic_numbers": s["atomic_numbers"]},
            5.0))["energy"])

    h = 1e-5
    for (i, k) in [(0, 0), (2, 1), (5, 2)]:
        p_plus, p_minus = s["pos"].copy(), s["pos"].copy()
        p_plus[i, k] += h
        p_minus[i, k] -= h
        fd = -(energy(p_plus) - energy(p_minus)) / (2 * h)
        assert abs(fd - f[i, k]) < 1e-7


def test_pes_smooth_across_cosine_cutoff():
    """With cutoff_fn='cosine' the energy is continuous when a neighbor
    crosses the finite cutoff (the deviation knob doing its job)."""
    torch.manual_seed(5)
    model = _small(cutoff_fn="cosine", cutoff=4.0).double()
    z = [1, 8]
    eps = 1e-6
    energies = []
    for d in (4.0 - eps, 4.0 + eps):
        pos = np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
        energies.append(float(model(structure_to_graph(
            {"pos": pos, "atomic_numbers": z}, 4.0))["energy"]))
    assert abs(energies[0] - energies[1]) < 1e-6


def test_size_extensivity_and_batching():
    torch.manual_seed(6)
    model = _small().double()
    s = _structure()
    far = {"pos": s["pos"] + 100.0, "atomic_numbers": s["atomic_numbers"]}
    both = {"pos": np.vstack([s["pos"], far["pos"]]),
            "atomic_numbers": list(s["atomic_numbers"]) * 2}
    e1 = float(model(structure_to_graph(s, 5.0))["energy"])
    e2 = float(model(structure_to_graph(far, 5.0))["energy"])
    e12 = float(model(structure_to_graph(both, 5.0))["energy"])
    assert abs(e12 - (e1 + e2)) < 1e-10

    from xnn.common.data import collate
    from xnn.common.data.dataset import AtomicDataset
    ds = AtomicDataset([s, far], 5.0)
    batch = collate([ds[0], ds[1]])
    eb = model(batch)["energy"]
    assert torch.allclose(eb, torch.tensor([e1, e2], dtype=torch.float64),
                          atol=1e-10)


def test_periodic_stress():
    model = ForceStressOutput(_small(), compute_stress=True)
    out = model(_graph(n=4, cutoff=5.0, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


# deployment

@pytest.mark.parametrize("cutoff_fn", [None, "cosine"])
def test_scriptable_and_lammps_export(tmp_path, cutoff_fn):
    from xnn.common.deploy import export_to_lammps

    model = _small(cutoff_fn=cutoff_fn).eval()
    g = _graph(n=6, cutoff=5.0, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12

    path = str(tmp_path / "schnet_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers,
                 g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-5
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-5


# config

def test_from_config_extras():
    cfg = from_dict({"model": {
        "name": "schnet", "cutoff": 5.0, "n_features": 16,
        "n_interactions": 2, "n_rbf": 25,
        "extra": {"cutoff_fn": "cosine", "gamma": 4.0,
                  "energy_shift": -1.5, "energy_scale": 0.2,
                  "species": SPECIES,
                  "atomic_energies": [-13.6, -1030.0, -2043.0]}}})
    m = build_model(cfg.model)
    assert m.rbf.gamma == 4.0
    assert m.cutoff_fn is not None
    assert float(m.energy_shift) == -1.5
    assert float(m.energy_scale) == pytest.approx(0.2)
    assert float(m.atom_ref.weight[6, 0]) == pytest.approx(-1030.0)


def test_schnetpack_key_translation():
    """schnetpack config spellings map onto the xnn core fields."""
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.5,
                               "n_atom_basis": 48, "n_gaussians": 30}})
    assert cfg.model.n_features == 48
    assert cfg.model.n_rbf == 30
    assert cfg.model.cutoff == 4.5
    assert cfg.data.cutoff == 4.5


def _fidelity(dtype):
    pytest.importorskip("schnetpack")
    import os
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(__file__), "schnet_fidelity.py")
    res = subprocess.run([sys.executable, script, dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return float(res.stdout.strip().splitlines()[-1])


def test_fidelity_vs_schnetpack_float64():
    """Transplanted weights give the schnetpack energies and forces at the float64 floor."""
    assert _fidelity("float64") < 1e-12


def test_fidelity_vs_schnetpack_float32():
    """The same in float32, at float32 round-off."""
    assert _fidelity("float32") < 1e-5


def test_float64_centers_exact_after_cast_and_load():
    """A float32 build cast to float64, or a float32 checkpoint, keeps exact float64 centers."""
    exact = torch.linspace(0.0, 30.0, 301, dtype=torch.float64)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        m32 = SchNet()
    finally:
        torch.set_default_dtype(old)
    assert torch.equal(m32.double().rbf.centers, exact)
    m64 = SchNet()
    m64.load_state_dict(SchNet().float().state_dict())
    assert torch.equal(m64.rbf.centers, exact)
    assert m64.rbf.gamma == 10.0


# fast path (fused continuous-filter convolution)

def test_fast_path_eligibility(monkeypatch):
    """The fused convolution needs a power-of-two width and plain dense filter layers."""
    import xnn.gnn.fast.schnet as fast
    from xnn.common.finetune import inject_lora
    monkeypatch.setattr(fast, "supported", lambda device, dtype: True)
    cuda = torch.device("cuda")
    m = SchNet(n_features=64, n_interactions=2, n_rbf=10, cutoff=5.0)
    assert m.interactions[0].cfconv.fast_supported(cuda, torch.float32)
    odd = SchNet(n_features=48, n_interactions=2, n_rbf=10, cutoff=5.0)
    assert not odd.interactions[0].cfconv.fast_supported(cuda, torch.float32)
    inject_lora(m, rank=2)
    assert not m.interactions[0].cfconv.fast_supported(cuda, torch.float32)


def _fast_batch(dtype):
    from xnn.common.data import collate
    graphs = [_graph(n=14, cutoff=5.0, seed=1, periodic=True), _graph(n=9, cutoff=5.0, seed=2)]
    batch = collate(graphs).to("cuda")
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cutoff_fn", [None, "cosine"])
def test_fast_path_parity(dtype, cutoff_fn):
    """The fused convolution gives the reference energies, forces and stress."""
    pytest.importorskip("triton")
    torch.manual_seed(0)
    model = SchNet(n_features=64, n_interactions=3, n_rbf=50, cutoff=5.0, cutoff_fn=cutoff_fn)
    with torch.no_grad():
        torch.nn.init.normal_(model.readout[-1].weight, std=0.3)
    model = model.to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model, compute_stress=True).eval()
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(dtype))
        assert all(b.cfconv.fast_active == use_fast for b in model.interactions)
        out[use_fast] = {k: res[k].detach().double() for k in ("energy", "forces", "stress")}
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    for key, ref in out[False].items():
        err = (out[True][key] - ref).abs().max() / ref.abs().max().clamp(min=1.0)
        assert err < tol, (key, float(err))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_path_off_in_training():
    """A training-mode evaluation (force training) runs the reference convolution."""
    pytest.importorskip("triton")
    model = SchNet(n_features=64, n_interactions=2, n_rbf=20, cutoff=5.0).to("cuda").train()
    model.set_use_fast(True)
    out = ForceStressOutput(model).train()(_fast_batch(torch.float64))
    assert not any(b.cfconv.fast_active for b in model.interactions)
    out["forces"].pow(2).sum().backward()
    assert model.interactions[0].cfconv.filter_net[0].weight.grad is not None
