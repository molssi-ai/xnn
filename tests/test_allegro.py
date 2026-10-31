"""Tests for the faithful Allegro (registered as ``allegro``).

Covers model equivariance / force co-rotation, periodic stress,
TorchScript/LAMMPS export, upstream yaml key translation, and -- when the
original ``allegro`` package is installed -- machine-precision
weight-transplant fidelity to it.
"""
import os

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402

SPECIES = [1, 6, 8]
TB, LAT, EE = [16, 32], [32], [16]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=6, cutoff=5.0, periodic=False, R=None):
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, (n, 3))
    if R is not None:
        pos = pos @ R.T
    s = {"pos": pos, "atomic_numbers": ([1, 6, 8] * n)[:n]}
    if periodic:
        s["cell"] = np.eye(3) * 6.0
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _build(num_layers=2, l_max=2, n_features=8, **extra):
    cfg = from_dict({"model": {
        "name": "allegro", "cutoff": 5.0, "n_features": n_features,
        "n_interactions": num_layers,
        "extra": {"species": SPECIES, "l_max": l_max, "avg_num_neighbors": 6.0,
                  "two_body_latent": TB, "latent": LAT, "edge_eng": EE, **extra},
    }})
    return build_model(cfg.model)


def test_registered():
    assert "allegro" in available_models()


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_model_equivariance(num_layers):
    model = ForceStressOutput(_build(num_layers=num_layers))
    rng = np.random.default_rng(3)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())
    o0 = model(_graph())
    o1 = model(_graph(R=R))
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-8
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T, atol=1e-7)


def test_periodic_stress():
    model = ForceStressOutput(_build(num_layers=1, l_max=1), compute_stress=True)
    out = model(_graph(n=4, cutoff=5.0, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


def test_scriptable_and_lammps_export(tmp_path):
    from xnn.common.deploy import export_to_lammps

    model = _build(num_layers=2, l_max=2).eval()
    g = _graph(n=6, cutoff=5.0, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12

    path = str(tmp_path / "allegro_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-10
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-10


def test_upstream_allegro_key_translation():
    """Keys copied verbatim from an upstream allegro yaml map to xnn names."""
    cfg = from_dict({"model": {
        "name": "allegro",
        "r_max": 4.5, "num_layers": 3, "num_tensor_features": 8,
        "num_bessels_per_basis": 10, "PolynomialCutoff_p": 5,
        "chemical_symbols": ["H", "O"], "l_max": 1, "parity": "o3_full",
        "two_body_latent_mlp_latent_dimensions": TB,
        "latent_mlp_latent_dimensions": LAT,
        "env_embed_mlp_latent_dimensions": [],
        "edge_eng_mlp_latent_dimensions": EE,
        "per_species_rescale_shifts": [-13.6, -2000.0],
        "per_species_rescale_scales": [3.0, 3.0],
        "avg_num_neighbors": 10.0,
    }})
    assert cfg.model.cutoff == 4.5
    assert cfg.model.n_interactions == 3
    assert cfg.model.n_rbf == 10
    m = build_model(cfg.model)
    assert m.species == [1, 8]
    assert len(m.tps) == 3
    assert m.atom_ref.weight[1].item() == -13.6
    assert m.atom_scale[8].item() == 3.0


def test_fidelity_vs_original_allegro():
    """Weight transplant from the original allegro gives identical E and F."""
    pytest.importorskip("allegro")
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.data.transforms import TypeMapper
    from nequip.model import model_from_config

    CUT, LMAX, NL, NF, NRBF, AVG = 5.0, 2, 2, 8, 8, 6.0
    E0, SIG = [0.5, -1.3, -2.1], [1.7, 0.9, 1.1]
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, (7, 3))
    Z = ([1, 6, 8] * 7)[:7]

    gm = model_from_config(dict(
        model_builders=["allegro.model.Allegro", "PerSpeciesRescale", "ForceOutput"],
        r_max=CUT, num_layers=NL, l_max=LMAX, parity="o3_full",
        num_tensor_features=NF, num_bessels_per_basis=NRBF,
        PolynomialCutoff_p=6.0, avg_num_neighbors=AVG,
        chemical_symbols=["H", "C", "O"],
        two_body_latent_mlp_latent_dimensions=TB,
        latent_mlp_latent_dimensions=LAT,
        env_embed_mlp_latent_dimensions=[], edge_eng_mlp_latent_dimensions=EE,
        per_species_rescale_shifts=E0, per_species_rescale_scales=SIG,
    ), initialize=True)
    seq = gm.model.func
    al = seq.allegro

    cfg = from_dict({"model": {"name": "allegro", "cutoff": CUT, "n_features": NF,
        "n_interactions": NL, "n_rbf": NRBF,
        "extra": {"species": SPECIES, "l_max": LMAX, "avg_num_neighbors": AVG,
                  "two_body_latent": TB, "latent": LAT, "edge_eng": EE,
                  "atomic_energies": E0, "atomic_scales": SIG}}})
    x = build_model(cfg.model)

    def copy_fcn(fcn, mod):  # upstream ScalarMLPFunction -> e3nn FullyConnectedNet
        sd = dict(mod.named_parameters())
        with torch.no_grad():
            for i in range(len(fcn.hs) - 1):
                getattr(fcn, f"layer{i}").weight.copy_(sd[f"_forward._weight_{i}"])

    with torch.no_grad():
        x.edge_feat.rbf.freqs.copy_(seq.radial_basis.bessel_weights * CUT)
        x.type_embeddings.copy_(seq.typeembed.type_embeddings)
        copy_fcn(x.basis_embed, seq.typeembed.basis_mlp)
        for i in range(NL):
            copy_fcn(x.latents[i], al.latents[i])
            copy_fcn(x.env_embed_mlps[i], al.env_embed_mlps[i])
            x.linears[i].w.copy_(al.linears[i].w)
        copy_fcn(x.final_latent, al.final_latent)
        copy_fcn(x.edge_eng, seq.edge_eng._module)
        x._resnet_params.copy_(al._latent_resnet_coefficients_params)

    tm = TypeMapper(chemical_symbols=["H", "C", "O"])
    dd = AtomicData.to_AtomicDataDict(tm(AtomicData.from_points(
        pos=torch.tensor(pos), r_max=CUT, atomic_numbers=torch.tensor(Z))))
    dd[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
    om = gm(dd)
    E_m = om[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
    F_m = -torch.autograd.grad(E_m, dd[AtomicDataDict.POSITIONS_KEY])[0]

    ox = ForceStressOutput(x)(
        structure_to_graph({"pos": pos, "atomic_numbers": Z}, CUT))
    assert abs(float(ox["energy"]) - float(E_m)) < 1e-12
    assert (ox["forces"].detach() - F_m).abs().max() < 1e-12


def test_fast_support_needs_cuda():
    from xnn.gnn.fast.allegro import supported
    assert not supported(torch.device("cpu"), torch.float64)


def _check_fast_contract(device):
    from e3nn import o3

    from xnn.gnn.fast.allegro_kernels import contract
    from xnn.gnn.models.allegro import _Contracter
    irs = [o3.Irrep("0e"), o3.Irrep("1o"), o3.Irrep("2e")]
    instr = [(i1, i2, ir) for i1, a in enumerate(irs) for i2, b in enumerate(irs[:2])
             for ir in a * b if ir.l <= 2]
    tp = _Contracter(irs, irs[:2], instr).to(device)
    k, i, j = torch.tensor(tp.__dict__["_pattern"][0], device=device).unbind(1)
    c = tp._w3j[k, i, j]
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(13, 9, dtype=torch.float64, generator=gen).to(device).requires_grad_(True)
    y = torch.randn(13, 4, dtype=torch.float64, generator=gen).to(device).requires_grad_(True)

    def f(x, y):
        return contract(x, y, c, tp.__dict__["_pattern"])

    with torch.no_grad():
        expect = torch.einsum("kij,ri,rj->rk", tp._w3j, x, y)
        assert torch.allclose(f(x, y), expect, rtol=0, atol=1e-13)
    assert torch.autograd.gradcheck(f, (x, y))
    assert torch.autograd.gradgradcheck(f, (x, y))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_fast_contract_gradients():
    """Values and every first and second derivative of the sparse contraction."""
    pytest.importorskip("triton")
    _check_fast_contract("cuda")


def test_fast_contract_gradients_interpreted():
    """The same under Triton's interpreter (chosen by conftest.py on machines without CUDA)."""
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("runs where Triton's interpreter is on (no CUDA device)")
    pytest.importorskip("triton")
    _check_fast_contract("cpu")


def _fast_batch(dtype):
    from xnn.common.data import collate
    batch = collate([_graph(n=9, periodic=True), _graph(n=6, periodic=True)]).to("cuda")
    batch.pos = batch.pos.to(dtype)
    batch.cell = batch.cell.to(dtype)
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("l_max", [1, 2])
def test_fast_path_parity(dtype, l_max):
    """The sparse tensor-product kernels give the reference energies, forces and stress."""
    pytest.importorskip("triton")
    torch.manual_seed(5)
    model = _build(num_layers=3, l_max=l_max).to("cuda", dtype).eval()
    wrapped = ForceStressOutput(model, compute_stress=True).eval()
    out = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast)
        res = wrapped(_fast_batch(dtype))
        assert all(tp.fast_active == use_fast for tp in model.tps)
        out[use_fast] = {k: res[k].detach().double() for k in ("energy", "forces", "stress")}
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    for key, ref in out[False].items():
        err = (out[True][key] - ref).abs().max() / ref.abs().max().clamp(min=1.0)
        assert err < tol, (key, float(err))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fast_path_training(dtype):
    """Force training through the sparse kernels gives the reference's weight gradients."""
    pytest.importorskip("triton")
    torch.manual_seed(6)
    model = _build(num_layers=3).to("cuda", dtype)
    grads = {}
    for use_fast in (False, True):
        model.set_use_fast(use_fast).train()
        model.zero_grad(set_to_none=True)
        out = ForceStressOutput(model).train()(_fast_batch(dtype))
        assert all(tp.fast_active == use_fast for tp in model.tps)
        (out["energy"].pow(2).sum() + out["forces"].pow(2).sum()).backward()
        grads[use_fast] = {name: p.grad.detach().double().clone() for name, p in model.named_parameters()
                           if p.grad is not None}
    assert grads[False].keys() == grads[True].keys()
    tol = 1e-11 if dtype == torch.float64 else 1e-4
    for name, g in grads[False].items():
        err = (grads[True][name] - g).abs().max() / g.abs().max().clamp_min(1e-30)
        assert err < tol, (name, float(err))
