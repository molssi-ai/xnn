"""Tests for the faithful BAMBOO graph equivariant transformer (``bamboo``).

Covers registration, rotation/translation/permutation invariance (energies,
forces, charges, dipoles), exact charge conservation (neutral and charged),
``energy == sum(node_energy)``, flexible GET depth, the reusable transformer
building blocks, upstream key translation, periodic stress, the optional
D3(CSO) dispersion, and -- when a clone of bytedance/bamboo is available
(path in ``BAMBOO_UPSTREAM_PATH``) -- machine-precision weight-transplant
parity against the original model.
"""
import os
import sys

import numpy as np
import pytest
import torch

from xnns.common.config import from_dict
from xnns.common.data import structure_to_graph
from xnns.common.models import ForceStressOutput, available_models, build_model
from xnns.hybrid.models.bamboo import BAMBOO, ELE_FACTOR
from xnns.transformer.attention import EdgeMultiheadAttention
from xnns.transformer.featurizers import ExpNormalSmearing

SPECIES = [3, 6, 7, 8, 9, 1]  # Li, C, N, O, F, H


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=7, cutoff=5.0, periodic=False, R=None, shift=0.0, seed=1,
           total_charge=None):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 4.0, (n, 3)) + shift
    if R is not None:
        pos = pos @ R.T
    s = {"pos": pos, "atomic_numbers": ([8, 1, 1, 6, 7, 9, 3] * n)[:n]}
    if periodic:
        s["cell"] = np.eye(3) * 8.0
        s["pbc"] = [True, True, True]
    g = structure_to_graph(s, cutoff)
    if total_charge is not None:
        g.total_charge = torch.tensor([float(total_charge)])
    return g


def _build(n_interactions=3, n_features=32, num_heads=8, seed=0, **extra):
    cfg = from_dict({"model": {
        "name": "bamboo", "cutoff": 5.0, "n_features": n_features, "n_rbf": 16,
        "n_interactions": n_interactions,
        "extra": {"num_heads": num_heads, **extra},
    }})
    model = build_model(cfg.model)
    # randomise the read-out heads so tests exercise the full network
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.mul_(1.0).add_(0.05 * torch.randn_like(p))
    return model


def _random_rotation(seed=0):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def test_registered():
    assert "bamboo" in available_models()


def test_energy_is_sum_of_node_energy():
    model = _build()
    out = model(_graph())
    assert torch.allclose(out["energy"], out["node_energy"].sum())


def test_charge_conservation_neutral_and_charged():
    model = _build()
    out = model(_graph())
    assert out["charges"].sum().abs() < 1e-10
    g = _graph(total_charge=1.0)
    out = model(g)
    assert (out["charges"].sum() - 1.0).abs() < 1e-10


def test_invariance_energy_forces_charges_dipole():
    model = ForceStressOutput(_build())
    g = _graph(seed=2)
    out = model(g)
    R = _random_rotation()
    gr = _graph(seed=2, R=R, shift=2.5)
    outr = model(gr)
    assert (out["energy"] - outr["energy"]).abs().max() < 1e-9
    assert (out["charges"] - outr["charges"]).abs().max() < 1e-9
    # forces are equivariant; charges/energy invariant
    f_rot = out["forces"].detach().numpy() @ R.T
    assert np.abs(f_rot - outr["forces"].detach().numpy()).max() < 1e-8
    # dipole magnitude is invariant
    assert (out["dipole"].norm(dim=-1) - outr["dipole"].norm(dim=-1)).abs().max() < 1e-8


def test_permutation_invariance():
    model = _build()
    g = _graph(seed=4)
    out = model(g)
    perm = torch.tensor([3, 0, 5, 1, 2, 4, 6])
    rng = np.random.default_rng(4)
    pos = rng.uniform(0, 4.0, (7, 3))
    Z = [8, 1, 1, 6, 7, 9, 3]
    s = {"pos": pos[perm.numpy()], "atomic_numbers": [Z[i] for i in perm.numpy()]}
    gp = structure_to_graph(s, 5.0)
    outp = model(gp)
    assert (out["energy"] - outp["energy"]).abs().max() < 1e-9


@pytest.mark.parametrize("n_interactions", [2, 3, 5])
def test_flexible_depth(n_interactions):
    model = _build(n_interactions=n_interactions)
    assert len(model.layers) == n_interactions
    out = ForceStressOutput(model)(_graph())
    assert out["forces"].shape == (7, 3)


def test_min_depth_rejected():
    with pytest.raises(ValueError):
        BAMBOO(n_layers=1)


def test_dim_divisible_by_heads():
    with pytest.raises(ValueError):
        BAMBOO(dim=64, num_heads=17)


def test_dispersion_toggle_changes_energy():
    g = _graph()
    m0 = _build(use_dispersion=False)
    m1 = _build(use_dispersion=True, seed=0)
    # copy the shared (non-dispersion) parameters so only the D3 term differs
    m1.load_state_dict(m0.state_dict(), strict=False)
    e0 = m0(g)["energy"]
    e1 = m1(g)["energy"]
    assert not torch.allclose(e0, e1)


def test_electrostatics_toggle():
    g = _graph()
    m = _build(use_electrostatics=True)
    out = m(g)
    assert out["energy_elec"].abs().sum() > 0
    m2 = _build(use_electrostatics=False)
    m2.load_state_dict(m.state_dict(), strict=False)
    out2 = m2(g)
    assert out2["energy_elec"].abs().sum() == 0


def test_periodic_stress_runs():
    model = ForceStressOutput(_build(), compute_stress=True)
    out = model(_graph(periodic=True))
    assert out["stress"].shape == (1, 3, 3)
    assert torch.isfinite(out["stress"]).all()


def test_batch_matches_single():
    from xnns.common.data import collate
    model = _build()
    g1 = _graph(seed=1)
    g2 = _graph(seed=2)
    e1 = model(g1)["energy"]
    e2 = model(g2)["energy"]
    batch = collate([g1, g2])
    eb = model(batch)["energy"]
    assert torch.allclose(eb, torch.cat([e1, e2]), atol=1e-9)


def test_upstream_bamboo_key_translation():
    """Upstream nn_params/gnn_params spellings map to the xnns canonical names."""
    cfg = from_dict({"model": {
        "name": "bamboo",
        "rcut": 6.0, "dim": 48, "num_rbf": 20, "n_layers": 4,
        "extra": {"num_heads": 12, "charge_ub": 3.0},
    }})
    assert cfg.model.cutoff == 6.0
    assert cfg.model.n_features == 48
    assert cfg.model.n_rbf == 20
    assert cfg.model.n_interactions == 4
    m = build_model(cfg.model)
    assert m.cutoff == 6.0 and m.dim == 48 and m.n_layers == 4
    assert m.num_heads == 12 and m.charge_ub == 3.0


def test_transformer_blocks_standalone():
    """The reusable transformer pieces work on their own."""
    rbf = ExpNormalSmearing(n_rbf=16, cutoff=5.0)
    r = torch.linspace(0.1, 6.0, 20)
    emb = rbf(r)
    assert emb.shape == (20, 16)
    assert emb[r >= 5.0].abs().max() < 1e-12  # smoothly zero past the cutoff

    attn = EdgeMultiheadAttention(dim=32, num_heads=8)
    feat = torch.randn(5, 32)
    center = torch.tensor([0, 1, 2, 3])
    neighbor = torch.tensor([1, 2, 3, 4])
    env = torch.rand(4)
    v, a = attn(feat, center, neighbor, env)
    assert v.shape == (4, 8, 4) and a.shape == (4, 8)


def test_ele_factor_constant():
    # Coulomb prefactor k_e e^2 in kcal/mol*Angstrom (bytedance/bamboo constant)
    assert abs(ELE_FACTOR - 332.0634945) < 1e-3


def _upstream_inputs(g, N, dtype):
    """Build the upstream ``predict`` input dict from an xnns graph."""
    pos = g.pos.detach().to(dtype)
    src, dst = g.edge_index[0], g.edge_index[1]
    up_edge_index = torch.stack([dst, src], 0)  # upstream row=center, col=neighbor
    edge_vec = (pos[dst] - pos[src]).detach()
    rows, cols = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                rows.append(i)
                cols.append(j)
    row_all = torch.tensor(rows)
    col_all = torch.tensor(cols)
    return {
        "atom_types": g.atomic_numbers.clone(),
        "edge_index": up_edge_index,
        "edge_cell_shift": edge_vec.clone(),
        "all_edge_index": torch.stack([row_all, col_all], 0),
        "all_edge_cell_shift": (pos[row_all] - pos[col_all]).detach(),
        "mol_ids": torch.zeros(N, dtype=torch.long),
        "total_charge": torch.zeros(1, dtype=dtype),
        "pos": pos.clone(),
    }


def test_parity_vs_original_bamboo():
    """Weight transplant from bytedance/bamboo gives identical E/F/q/dipole.

    Needs a clone of bytedance/bamboo whose path is in the
    ``BAMBOO_UPSTREAM_PATH`` environment variable (also requires
    ``torch_runstats``, which upstream imports).
    """
    path = os.environ.get("BAMBOO_UPSTREAM_PATH")
    if not path or not os.path.isdir(path):
        pytest.skip("set BAMBOO_UPSTREAM_PATH to a clone of bytedance/bamboo")
    pytest.importorskip("torch_runstats")
    sys.path.insert(0, path)
    try:
        import torch.nn as nn
        from models.bamboo_get import BambooGET
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot import upstream bamboo: {exc}")

    DIM, NRBF, NLAYERS, NHEADS, CUT = 64, 32, 3, 16, 5.0
    up = BambooGET(
        device=torch.device("cpu"),
        coul_disp_params={"coul_damping_beta": 18.7, "coul_damping_r0": 2.2,
                          "disp_cutoff": 10.0},
        nn_params={"dim": DIM, "num_rbf": NRBF, "rcut": CUT, "charge_ub": 2.0,
                   "act_fn": nn.SiLU(), "charge_mlp_layers": 2,
                   "energy_mlp_layers": 2},
        gnn_params={"n_layers": NLAYERS, "num_heads": NHEADS, "act_fn": nn.GELU()},
    ).eval()

    cfg = from_dict({"model": {"name": "bamboo", "cutoff": CUT, "n_features": DIM,
        "n_rbf": NRBF, "n_interactions": NLAYERS, "extra": {"num_heads": NHEADS}}})
    xm = build_model(cfg.model).eval()

    def copy(dst, src):
        dst.load_state_dict(src.state_dict())

    copy(xm.atom_emb, up.atom_embtab)
    copy(xm.dis_rbf, up.dis_rbf)
    copy(xm.rbf_proj, up.rbf_proj)
    copy(xm.energy_mlp, up.energy_mlp)
    copy(xm.charge_mlp, up.charge_mlp)
    copy(xm.electronegativity_mlp, up.pred_electronegativity_mlp)
    copy(xm.hardness_mlp, up.pred_electronegativity_hardness_mlp)
    for xl, ul in zip(xm.layers, [up.first_attn] + list(up.attns) + [up.last_attn]):
        copy(xl.attn.qkv_proj, ul.qkv_proj)
        copy(xl.attn.layer_norm, ul.layer_norm)
        copy(xl.output_proj, ul.output_proj)
        if xl.vec_proj is not None:
            copy(xl.vec_proj, ul.vec_proj)

    g = _graph(seed=3)
    N = g.num_nodes
    xout = ForceStressOutput(xm)(g)
    up_out = up.predict(_upstream_inputs(g, N, torch.get_default_dtype()))

    assert (abs(float(xout["energy"][0]) - float(up_out["energy"][0]))
            / abs(float(up_out["energy"][0]))) < 1e-12
    assert (xout["charges"] - up_out["charge"]).abs().max() < 1e-12
    # xnns gives the full conservative force = upstream (nn+coul) + qeq residual
    full = up_out["forces"] + up_out["qeq_force"]
    assert (xout["forces"].detach() - full).abs().max() < 1e-11
    assert (xout["dipole"][0] - up_out["dipole"][0]).abs().max() < 1e-11
