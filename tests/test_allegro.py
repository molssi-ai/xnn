"""Tests for the faithful Allegro (registered as ``allegro``).

Covers model equivariance / force co-rotation, periodic stress,
TorchScript/LAMMPS export, upstream yaml key translation, and -- when the
original ``allegro`` package is installed -- machine-precision
weight-transplant parity against it.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from xnns.common.config import from_dict  # noqa: E402
from xnns.common.data import structure_to_graph  # noqa: E402
from xnns.common.models import ForceStressOutput, available_models, build_model  # noqa: E402

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
    from xnns.common.deploy import export_to_lammps

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
    """Keys copied verbatim from an upstream allegro yaml map to xnns names."""
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


def test_parity_vs_original_allegro():
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
