"""Tests for the faithful NequIP (registered as ``nequip``).

Covers the exact e3nn-Gate stand-in, model equivariance / force co-rotation,
periodic stress, the flexible number of layers, TorchScript/LAMMPS export,
upstream yaml key translation, and -- when the original ``nequip`` package is
installed -- machine-precision weight-transplant parity against it.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

import torch.nn.functional as F  # noqa: E402
from e3nn import o3  # noqa: E402

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.gnn.models.nequip import _Gate, nequip_hidden_irreps  # noqa: E402

SPECIES = [1, 6, 8]


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


def _proper_rotation(seed=3):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def _build(n_layers=2, l_max=2, n_features=16, **extra):
    cfg = from_dict({"model": {
        "name": "nequip", "cutoff": 5.0, "n_features": n_features,
        "n_interactions": n_layers,
        "extra": {"species": SPECIES, "l_max": l_max,
                  "avg_num_neighbors": 6.0, **extra},
    }})
    return build_model(cfg.model)


def test_registered_single_nequip():
    assert "nequip" in available_models()


def test_hidden_irreps_upstream_order():
    """Full-parity hidden irreps follow the upstream SimpleIrrepsConfig order."""
    assert nequip_hidden_irreps(8, 2, parity=True) == o3.Irreps(
        "8x0e+8x1e+8x2e+8x0o+8x1o+8x2o")
    assert nequip_hidden_irreps(8, 1, parity=False) == o3.Irreps("8x0e+8x1e")


def test_gate_matches_e3nn():
    """The scriptable _Gate reproduces e3nn.nn.Gate bit-for-bit and scripts."""
    from e3nn.nn import Gate

    scalars = o3.Irreps("4x0e+4x0o")
    gates = o3.Irreps("8x0e")
    gated = o3.Irreps("4x1o+4x2e")
    ref = Gate(scalars, [F.silu, torch.tanh], gates, [F.silu], gated)
    mine = _Gate(scalars, [F.silu, torch.tanh], gates, [F.silu], gated)
    assert mine.irreps_in == ref.irreps_in
    assert mine.irreps_out == ref.irreps_out
    x = torch.randn(5, ref.irreps_in.dim)
    assert torch.equal(mine(x), ref(x))
    scripted = torch.jit.script(mine)
    assert torch.equal(scripted(x), ref(x))


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_model_equivariance(n_layers):
    model = ForceStressOutput(_build(n_layers=n_layers))
    R = _proper_rotation()
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())
    o0 = model(_graph())
    o1 = model(_graph(R=R))
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-8
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T, atol=1e-7)


@pytest.mark.parametrize("n_layers", [0, 1, 4])
def test_flexible_num_layers(n_layers):
    """num_layers = 0..N must all build and run."""
    model = ForceStressOutput(_build(n_layers=n_layers))
    out = model(_graph())
    assert out["forces"].shape == (6, 3)
    assert len(model.model.layers) == n_layers
    if n_layers == 0:  # species-constant baseline: zero forces
        assert torch.allclose(out["forces"], torch.zeros_like(out["forces"]))


def test_periodic_stress():
    model = ForceStressOutput(_build(n_layers=1, l_max=1), compute_stress=True)
    out = model(_graph(n=4, cutoff=5.0, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


def test_per_species_scale_shift():
    """E = sigma_Z * eps + E0_Z; with zero layers the E0 sum is exact."""
    m = _build(n_layers=0, atomic_energies=[0.5, -1.3, -2.1], atomic_scales=0.0)
    out = ForceStressOutput(m)(_graph())
    expected = sum({1: 0.5, 6: -1.3, 8: -2.1}[int(z)]
                   for z in _graph().atomic_numbers)
    assert abs(float(out["energy"].detach()) - expected) < 1e-12


def test_nequip_scriptable_and_lammps_export(tmp_path):
    """NequIP must torch.jit.script cleanly and the LAMMPS artifact must
    reproduce the eager model's energy and forces on a periodic system."""
    from xnn.common.deploy import export_to_lammps

    model = _build(n_layers=2, l_max=2).eval()
    g = _graph(n=6, cutoff=5.0, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12  # scripted == eager

    path = str(tmp_path / "nequip_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-10
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-10


def test_upstream_nequip_key_translation():
    """Keys copied verbatim from an upstream NequIP yaml map to xnn names."""
    cfg = from_dict({"model": {
        "name": "nequip",
        "r_max": 4.5, "num_layers": 4, "num_features": 8, "num_basis": 10,
        "PolynomialCutoff_p": 5, "BesselBasis_trainable": False,
        "chemical_symbols": ["H", "O"], "l_max": 1,
        "per_species_rescale_shifts": [-13.6, -2000.0],
        "per_species_rescale_scales": [3.0, 3.0],
        "avg_num_neighbors": 10.0,
    }})
    assert cfg.model.cutoff == 4.5            # r_max -> typed core field
    assert cfg.data.cutoff == 4.5             # neighbor-list cutoff follows
    assert cfg.model.n_interactions == 4
    assert cfg.model.n_rbf == 10
    m = build_model(cfg.model)
    assert m.species == [1, 8]                # chemical symbols coerced to Z
    assert len(m.layers) == 4
    assert not isinstance(m.edge_feat.rbf.freqs, torch.nn.Parameter)
    assert m.atom_ref.weight[1].item() == -13.6
    assert m.atom_scale[8].item() == 3.0

    # the xnn canonical spelling wins when both are present
    both = from_dict({"model": {"name": "nequip", "cutoff": 4.0, "r_max": 9.0}})
    assert both.model.cutoff == 4.0


def test_parity_vs_original_nequip():
    """Weight transplant from the original nequip gives identical E and F."""
    pytest.importorskip("nequip")
    from nequip.data import AtomicData, AtomicDataDict
    from nequip.data.transforms import TypeMapper
    from nequip.model import model_from_config

    CUTOFF, LMAX, NF, NL, NRBF, AVG = 5.0, 2, 8, 2, 8, 6.0
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 4, (7, 3))
    Z = ([1, 6, 8] * 7)[:7]

    gm = model_from_config(dict(
        model_builders=["SimpleIrrepsConfig", "EnergyModel"],
        r_max=CUTOFF, num_layers=NL, l_max=LMAX, parity=True, num_features=NF,
        num_basis=NRBF, PolynomialCutoff_p=6, invariant_layers=2,
        invariant_neurons=64, avg_num_neighbors=AVG, use_sc=True, resnet=False,
        chemical_symbols=["H", "C", "O"],
    ), initialize=True)
    seq = gm.model

    x = _build(n_layers=NL, l_max=LMAX, n_features=NF)
    with torch.no_grad():
        x.edge_feat.rbf.freqs.copy_(seq.radial_basis.basis.bessel_weights)
        x.chemical_embedding.load_state_dict(
            seq.chemical_embedding.linear.state_dict())
        for i in range(NL):
            x.layers[i].conv.load_state_dict(
                getattr(seq, f"layer{i}_convnet").conv.state_dict())
        x.conv_to_output_hidden.load_state_dict(
            seq.conv_to_output_hidden.linear.state_dict())
        x.output_hidden_to_scalar.load_state_dict(
            seq.output_hidden_to_scalar.linear.state_dict())

    tm = TypeMapper(chemical_symbols=["H", "C", "O"])
    dd = AtomicData.to_AtomicDataDict(tm(AtomicData.from_points(
        pos=torch.tensor(pos), r_max=CUTOFF, atomic_numbers=torch.tensor(Z))))
    dd[AtomicDataDict.POSITIONS_KEY].requires_grad_(True)
    om = gm(dd)
    E_m = om[AtomicDataDict.TOTAL_ENERGY_KEY].sum()
    F_m = -torch.autograd.grad(E_m, dd[AtomicDataDict.POSITIONS_KEY])[0]

    ox = ForceStressOutput(x)(
        structure_to_graph({"pos": pos, "atomic_numbers": Z}, CUTOFF))
    assert abs(float(ox["energy"]) - float(E_m)) < 1e-12
    assert (ox["forces"].detach() - F_m).abs().max() < 1e-12
