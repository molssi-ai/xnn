"""Tests for the faithful MACE (registered as ``mace``).

Covers exact equivariance of the symmetric contraction, model equivariance /
force co-rotation, periodic stress, the flexible number of interaction layers
(T = 0..N), and pair repulsion.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from e3nn import o3  # noqa: E402

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.gnn.models.mace import SymmetricContraction, U_matrix_real  # noqa: E402

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


def _build(T=2, max_L=1, max_ell=3, correlation=3, n_features=16, **extra):
    cfg = from_dict({"model": {
        "name": "mace", "cutoff": 5.0, "n_features": n_features, "n_interactions": T,
        "extra": {"species": SPECIES, "max_ell": max_ell, "correlation": correlation,
                  "num_channels": n_features, "max_L": max_L, **extra},
    }})
    return build_model(cfg.model)


def test_registered_single_mace():
    models = available_models()
    assert "mace" in models
    assert "mace02" not in models  # only one MACE implementation


def test_u_matrix_shapes():
    coup = o3.Irreps("1x0e+1x1o+1x2e+1x3o")
    U = U_matrix_real(coup, "0e", 3, dtype=torch.float64)[-1]
    assert U.shape[:-1] == (coup.dim, coup.dim, coup.dim)  # nu=3 -> 3 input axes


def test_symmetric_contraction_equivariant():
    irreps_in = o3.Irreps("8x0e+8x1o+8x2e+8x3o")
    irreps_out = o3.Irreps("8x0e+8x1o")
    coupling = o3.Irreps([ir.ir for ir in irreps_in])
    sc = SymmetricContraction(irreps_in, irreps_out, correlation=3, num_elements=3)
    B, nf = 5, irreps_in.count((0, 1))
    x = torch.randn(B, nf, coupling.dim)
    y = torch.zeros(B, 3)
    y[torch.arange(B), torch.randint(0, 3, (B,))] = 1.0
    R = o3.rand_matrix()
    Dc, Do = coupling.D_from_matrix(R), irreps_out.D_from_matrix(R)
    out0 = sc(x, y)
    out1 = sc(torch.einsum("ij,bcj->bci", Dc, x), y)
    assert torch.allclose(out1, torch.einsum("ij,bj->bi", Do, out0), atol=1e-9)


@pytest.mark.parametrize("T", [1, 2, 3])
def test_model_equivariance(T):
    model = ForceStressOutput(_build(T=T))
    R = _proper_rotation()
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())
    o0 = model(_graph())
    o1 = model(_graph(R=R))
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-8
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T, atol=1e-7)


@pytest.mark.parametrize("T", [0, 1, 2, 4])
def test_flexible_num_interactions(T):
    """T = 0..N must all build and run (upstream MACE fixes T = 2)."""
    model = ForceStressOutput(_build(T=T, max_L=0, max_ell=2))
    out = model(_graph())
    assert out["forces"].shape == (6, 3)
    assert len(model.model.interactions) == T
    if T == 0:  # pure reference-energy baseline: constant energy, zero forces
        assert torch.allclose(out["forces"], torch.zeros_like(out["forces"]))


def test_periodic_stress():
    model = ForceStressOutput(_build(T=2, max_L=0, max_ell=2), compute_stress=True)
    out = model(_graph(n=4, cutoff=5.0, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


def test_pair_repulsion_runs():
    pytest.importorskip("ase")
    model = ForceStressOutput(_build(T=1, max_L=0, max_ell=2, pair_repulsion=True))
    out = model(_graph())
    assert out["forces"].shape == (6, 3)


def test_mace_scriptable_and_lammps_export(tmp_path):
    """MACE must torch.jit.script cleanly and the LAMMPS artifact must
    reproduce the eager model's energy and forces on a periodic system."""
    from xnn.common.deploy import export_to_lammps

    model = _build(T=2, max_L=1, max_ell=2).eval()
    # the scriptability placeholders must not leak into the state_dict
    assert not any("U_matrix_4" in k for k in model.state_dict())

    g = _graph(n=6, cutoff=5.0, periodic=True)
    scripted = torch.jit.script(model)
    d = (scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
         - model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors()))
    assert d.abs().max() < 1e-12  # scripted == eager

    path = str(tmp_path / "mace_lammps.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    ref = ForceStressOutput(model)(g)
    assert abs(float(out["total_energy"]) - float(ref["energy"])) < 1e-10
    assert (out["forces"] - ref["forces"].detach()).abs().max() < 1e-10


def test_mace_lammps_export_t0_zero_forces(tmp_path):
    """T=0 (pure reference energy) must export and yield zero forces."""
    from xnn.common.deploy import export_to_lammps

    model = _build(T=0, max_L=0, max_ell=2).eval()
    path = str(tmp_path / "mace_t0.pt")
    export_to_lammps(model, 5.0, path)
    loaded = torch.jit.load(path)
    g = _graph(n=6, cutoff=5.0, periodic=True)
    out = loaded(g.pos, g.edge_index, g.cell_shifts, g.atomic_numbers, g.cell[0])
    assert out["forces"].abs().max() == 0.0


def test_upstream_mace_key_translation():
    """Keys copied verbatim from an upstream MACE yaml are translated to xnn names."""
    cfg = from_dict({"model": {
        "name": "mace",
        "r_max": 5.5, "num_channels": 8, "num_interactions": 3,
        "num_radial_basis": 10, "num_cutoff_basis": 6,
        "atomic_numbers": "[1, 8]", "E0s": "{1: -13.6, 8: -2000.0}",
        "radial_MLP": "[16, 16]", "max_ell": 2, "max_L": 0,
    }})
    assert cfg.model.cutoff == 5.5           # r_max -> typed core field
    assert cfg.data.cutoff == 5.5            # ... so the neighbor-list cutoff follows
    assert "r_max" not in cfg.model.extra    # translated, not duplicated
    m = build_model(cfg.model)
    assert m.cutoff == 5.5
    assert len(m.interactions) == 3
    assert m.species == [1, 8]
    assert m.atom_ref.weight[1].item() == -13.6
    assert m.atom_ref.weight[8].item() == -2000.0

    # identical architecture whether xnn or upstream MACE names are used
    xnn_names = from_dict({"model": {
        "name": "mace", "cutoff": 5.5, "n_features": 8, "n_interactions": 3,
        "n_rbf": 10, "num_polynomial_cutoff": 6, "species": [1, 8],
        "atomic_energies": [-13.6, -2000.0], "radial_MLP": [16, 16],
        "max_ell": 2, "max_L": 0,
    }})
    m2 = build_model(xnn_names.model)
    assert {k: v.shape for k, v in m.state_dict().items()} == \
           {k: v.shape for k, v in m2.state_dict().items()}

    # the xnn canonical spelling wins when both are present
    both = from_dict({"model": {"name": "mace", "cutoff": 4.5, "r_max": 9.0}})
    assert both.model.cutoff == 4.5

    with pytest.raises(ValueError, match="E0s"):
        build_model(from_dict({"model": {"name": "mace", "E0s": "average"}}).model)


# ScaleShiftMACE energy expression, distance transforms, density blocks
def test_scale_shift_energy_expression():
    """E_i = E0_i + scale * E_int,i + shift, with ZBL inside the scale."""
    pytest.importorskip("ase")
    torch.manual_seed(0)
    base = _build(T=2, pair_repulsion=True)
    scaled = _build(T=2, pair_repulsion=True, scale=1.7, shift=-0.25)
    scaled.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        scaled.scale_shift.scale.fill_(1.7)
        scaled.scale_shift.shift.fill_(-0.25)
    g = _graph()
    e0 = base.atom_ref(g.atomic_numbers).squeeze(-1)
    inter = base(g)["node_energy"] - e0          # scale=1, shift=0 baseline
    expected = e0 + 1.7 * inter - 0.25
    assert torch.allclose(scaled(g)["node_energy"], expected, atol=1e-12)
    # the defaults are the exact identity (plain MACE)
    assert float(base.scale_shift.scale) == 1.0
    assert float(base.scale_shift.shift) == 0.0


def test_distance_transform_formulas():
    """Agnesi and Soft recomputed by hand from their published forms."""
    ase = pytest.importorskip("ase")
    import math

    from xnn.gnn.featurizers import (AgnesiDistanceTransform,
                                     SoftDistanceTransform)

    z = torch.tensor([1, 8])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    r = torch.tensor([0.97, 2.31])
    rc = ase.data.covalent_radii

    agnesi = AgnesiDistanceTransform()
    out = agnesi(r, z, edge_index)
    for k in range(2):
        r0 = 0.5 * (rc[1] + rc[8])
        u = float(r[k]) / r0
        expect = 1.0 / (1.0 + 1.0805 * u ** 0.9183 / (1.0 + u ** (0.9183 - 4.5791)))
        assert abs(float(out[k]) - expect) < 1e-12

    soft = SoftDistanceTransform()
    out = soft(r, z, edge_index)
    for k in range(2):
        r0 = rc[1] + rc[8]
        p0, p1 = 0.75 * r0, 4.0 / 3.0 * r0
        s = 0.5 * (1.0 + math.tanh(4.0 / (p1 - p0) * (float(r[k]) - 0.5 * (p0 + p1))))
        expect = p0 + (float(r[k]) - p0) * s
        assert abs(float(out[k]) - expect) < 1e-12

    with pytest.raises(NotImplementedError, match="distance_transform"):
        _build(T=1, distance_transform="Bogus")


def test_identity_transform_matches_previous_embedding():
    """distance_transform='None' reproduces the plain featurizer path."""
    model = _build(T=2)
    g = _graph()
    vec = g.edge_vectors()
    lengths, _, radial = model.edge_feat.embed(vec)
    r_t = model.distance_transform(lengths, g.atomic_numbers, g.edge_index)
    recomputed = model.edge_feat.rbf(r_t) * model.edge_feat.envelope(lengths)[:, None]
    assert torch.allclose(radial, recomputed, atol=0)


@pytest.mark.parametrize("first,rest", [
    ("RealAgnosticDensityInteractionBlock",
     "RealAgnosticDensityResidualInteractionBlock"),
])
def test_density_blocks_equivariance_and_script(first, rest):
    pytest.importorskip("ase")
    model = _build(T=2, interaction_first=first, interaction=rest,
                   distance_transform="Agnesi", pair_repulsion=True,
                   scale=0.8, shift=0.1)
    fmodel = ForceStressOutput(model)
    R = _proper_rotation()
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())
    o0, o1 = fmodel(_graph()), fmodel(_graph(R=R))
    assert abs(float(o0["energy"]) - float(o1["energy"])) < 1e-8
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T,
                          atol=1e-7)
    g = _graph()
    scripted = torch.jit.script(model)
    e_s = scripted.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
    e_e = model.node_energy(g.atomic_numbers, g.edge_index, g.edge_vectors())
    assert torch.allclose(e_s, e_e, atol=1e-14)


def test_from_config_scale_shift_transform_keys():
    pytest.importorskip("ase")
    cfg = from_dict({"model": {
        "name": "mace", "cutoff": 4.0, "n_features": 8, "n_interactions": 1,
        "species": SPECIES, "max_ell": 2, "scale": 0.9, "shift": 0.2,
        "distance_transform": "Agnesi", "pair_repulsion": True,
        "interaction_first": "RealAgnosticDensityResidualInteractionBlock",
        "interaction": "RealAgnosticDensityResidualInteractionBlock",
    }})
    m = build_model(cfg.model)
    assert float(m.scale_shift.scale) == 0.9
    assert float(m.scale_shift.shift) == 0.2
    assert type(m.distance_transform).__name__ == "AgnesiDistanceTransform"
    assert m.pair_repulsion


# foundation-model conversion: parity with upstream mace-torch
def _upstream_scale_shift_mace(flavor, heads=None):
    """Build a small upstream ScaleShiftMACE of the requested flavor."""
    import mace.modules as mm

    blocks = {
        "mp0": (mm.RealAgnosticResidualInteractionBlock,
                mm.RealAgnosticResidualInteractionBlock, "None", False),
        "off23": (mm.RealAgnosticInteractionBlock,
                  mm.RealAgnosticResidualInteractionBlock, "None", False),
        "0b": (mm.RealAgnosticInteractionBlock,
               mm.RealAgnosticResidualInteractionBlock, "Agnesi", True),
        "0b2": (mm.RealAgnosticDensityInteractionBlock,
                mm.RealAgnosticDensityResidualInteractionBlock, "Agnesi", True),
        "soft": (mm.RealAgnosticResidualInteractionBlock,
                 mm.RealAgnosticResidualInteractionBlock, "Soft", False),
    }
    first, rest, transform, zbl = blocks[flavor]
    n_heads = len(heads) if heads else 1
    e0 = np.array([[0.5, -1.3, -2.1], [0.1, -0.4, -0.9]])[:n_heads].squeeze(0) \
        if n_heads == 1 else np.array([[0.5, -1.3, -2.1], [0.1, -0.4, -0.9]])
    scale = 0.83 if n_heads == 1 else np.array([0.8, 1.1])
    shift = 0.11 if n_heads == 1 else np.array([0.1, -0.2])
    up = mm.ScaleShiftMACE(
        atomic_inter_scale=scale, atomic_inter_shift=shift,
        r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2,
        interaction_cls=rest, interaction_cls_first=first, num_interactions=2,
        num_elements=3, hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps(f"{16 * n_heads}x0e"), atomic_energies=e0,
        avg_num_neighbors=3.1, atomic_numbers=SPECIES, correlation=3,
        gate=torch.nn.functional.silu, radial_MLP=[16, 16],
        radial_type="bessel", distance_transform=transform,
        pair_repulsion=zbl, heads=heads, use_reduced_cg=False,
        apply_cutoff=True).double()
    torch.manual_seed(11)
    with torch.no_grad():
        for p in up.parameters():
            p.add_(0.05 * torch.randn_like(p))   # break symmetric inits
    return up


def _upstream_eval(up, g, head=None):
    """Evaluate an upstream model through its own data pipeline."""
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable, torch_geometric

    conf = Configuration(atomic_numbers=g.atomic_numbers.numpy(),
                         positions=g.pos.detach().numpy(), properties={},
                         property_weights={})
    ad = AtomicData.from_config(conf, z_table=AtomicNumberTable(SPECIES),
                                cutoff=float(up.r_max))
    batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
    d = batch.to_dict()
    if head is not None:
        d["head"] = torch.tensor([head])
    out = up(d, compute_force=True)
    return float(out["energy"]), out["forces"].detach()


@pytest.mark.parametrize("flavor", ["mp0", "off23", "0b", "0b2", "soft"])
def test_from_mace_torch_parity(flavor):
    """Every foundation-generation flavor converts to machine precision."""
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    from xnn.gnn.models.mace_foundation import from_mace_torch

    up = _upstream_scale_shift_mace(flavor)
    model = ForceStressOutput(from_mace_torch(up))
    g = _graph(cutoff=4.0)
    e_u, f_u = _upstream_eval(up, g)
    out = model(g)
    assert abs(float(out["energy"]) - e_u) < 1e-11
    assert (out["forces"].detach() - f_u).abs().max() < 1e-11


def test_from_mace_torch_multihead_slicing():
    """Multi-head checkpoints slice to each head exactly; head is required."""
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    from xnn.gnn.models.mace_foundation import from_mace_torch

    up = _upstream_scale_shift_mace("0b2", heads=["ha", "hb"])
    g = _graph(cutoff=4.0)
    with pytest.raises(ValueError, match="ha"):
        from_mace_torch(up)                    # ambiguous: must pick a head
    with pytest.raises(ValueError, match="unknown head"):
        from_mace_torch(up, head="nope")
    for idx, name in enumerate(["ha", "hb"]):
        model = ForceStressOutput(from_mace_torch(up, head=name))
        e_u, f_u = _upstream_eval(up, g, head=idx)
        out = model(g)
        assert abs(float(out["energy"]) - e_u) < 1e-11
        assert (out["forces"].detach() - f_u).abs().max() < 1e-11


def test_foundation_registry_and_errors():
    from xnn.gnn.models.mace_foundation import FOUNDATION_MODELS, _checkpoint_path

    assert "mace-mp-0-medium" in FOUNDATION_MODELS
    assert "mace-off23-small" in FOUNDATION_MODELS
    for url, license_ in FOUNDATION_MODELS.values():
        assert url.startswith("https://") and license_ in ("MIT", "ASL")
    with pytest.raises(FileNotFoundError, match="alias"):
        _checkpoint_path("not-a-model")


def test_from_foundation_cached_checkpoint():
    """Convert a real foundation checkpoint when one is already cached."""
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    from pathlib import Path

    from xnn.gnn.models.mace import MACE
    from xnn.gnn.models.mace_foundation import load_foundation

    # only use a file that is already on disk; tests never download
    # (the name is the upstream cache spelling: non-alphanumerics stripped)
    cached = Path.home() / ".cache" / "mace" / "MACEOFF23_smallmodel"
    if not cached.is_file():
        pytest.skip("no cached MACE-OFF23 small checkpoint")
    up = load_foundation(cached).double()
    model = MACE.from_foundation(cached, dtype=torch.float64)
    assert isinstance(model, MACE)
    assert model.species == [int(z) for z in up.atomic_numbers]
    g = _graph(cutoff=float(up.r_max))
    e_u, f_u = _upstream_eval_off(up, g)
    out = ForceStressOutput(model)(g)
    assert abs(float(out["energy"]) - e_u) < 1e-9
    assert (out["forces"].detach() - f_u).abs().max() < 1e-9


def _upstream_eval_off(up, g):
    """Upstream evaluation against the checkpoint's own element table."""
    from mace.data import AtomicData, Configuration
    from mace.tools import AtomicNumberTable, torch_geometric

    conf = Configuration(atomic_numbers=g.atomic_numbers.numpy(),
                         positions=g.pos.detach().numpy(), properties={},
                         property_weights={})
    zt = AtomicNumberTable([int(z) for z in up.atomic_numbers])
    ad = AtomicData.from_config(conf, z_table=zt, cutoff=float(up.r_max))
    batch = next(iter(torch_geometric.dataloader.DataLoader([ad], batch_size=1)))
    out = up(batch.to_dict(), compute_force=True)
    return float(out["energy"]), out["forces"].detach()
