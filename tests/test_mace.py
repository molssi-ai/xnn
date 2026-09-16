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
