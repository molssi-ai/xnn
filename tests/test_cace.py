"""Tests for the faithful CACE (registered as ``cace``).

Covers the Cartesian angular basis, the symmetrized B-feature counts, model
equivariance / force co-rotation, flexible message-passing depth, periodic
stress, upstream constructor-key translation, and -- when the original
``cace`` package is installed -- machine-precision weight-transplant parity
against it.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")  # the xnn.gnn package import chain needs e3nn

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.gnn.featurizers.cartesian import (  # noqa: E402
    CartesianAngularBasis, lxlylz_list, n_lxlylz,
)
from xnn.gnn.models.cace import _Symmetrizer  # noqa: E402

SPECIES = [1, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=8, cutoff=4.5, periodic=False, R=None, seed=1):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 4, (n, 3))
    if R is not None:
        pos = pos @ R.T
    s = {"pos": pos, "atomic_numbers": ([1, 8] * n)[:n]}
    if periodic:
        s["cell"] = np.eye(3) * 5.0
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _build(num_mp=1, max_l=3, max_nu=3, **extra):
    cfg = from_dict({"model": {
        "name": "cace", "cutoff": 4.5, "n_interactions": num_mp, "n_rbf": 6,
        "extra": {"species": SPECIES, "n_atom_basis": 2, "n_radial_basis": 8,
                  "max_l": max_l, "max_nu": max_nu,
                  "avg_num_neighbors": 9.0, **extra},
    }})
    return build_model(cfg.model)


def test_registered():
    assert "cace" in available_models()


def test_angular_basis_is_the_monomials():
    """Each column equals x^lx * y^ly * z^lz of its (lx, ly, lz) triple."""
    basis = CartesianAngularBasis(4)
    vec = torch.randn(16, 3)
    vec = vec / vec.norm(dim=-1, keepdim=True)
    out = basis(vec)
    assert out.shape == (16, n_lxlylz(4))
    for i, (lx, ly, lz) in enumerate(lxlylz_list(4)):
        ref = vec[:, 0] ** lx * vec[:, 1] ** ly * vec[:, 2] ** lz
        assert torch.allclose(out[:, i], ref, atol=1e-12)


@pytest.mark.parametrize("max_l,max_nu,expected", [
    (3, 1, 1),        # nu=1 only: the l=0 channel
    (2, 2, 1 + 2),    # + one nu=2 invariant per l in 1..l_max
    (3, 3, 1 + 3 + 2),   # + nu=3 groups (l1,l2): (1,1), (1,2)
    (3, 4, 1 + 3 + 2 + 1),  # + nu=4 group (l1,l2,dl): (1,2,1)
])
def test_b_feature_count(max_l, max_nu, expected):
    """Number of angular invariants N_L matches the paper (fig 2)."""
    assert _Symmetrizer(max_nu, max_l).n_features == expected


@pytest.mark.parametrize("num_mp", [0, 1, 2])
def test_model_equivariance(num_mp):
    model = ForceStressOutput(_build(num_mp=num_mp))
    rng = np.random.default_rng(3)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    Rt = torch.tensor(R, dtype=torch.get_default_dtype())
    o0 = model(_graph())
    o1 = model(_graph(R=R))
    assert abs(float(o0["energy"].detach()) - float(o1["energy"].detach())) < 1e-8
    assert torch.allclose(o1["forces"].detach(), o0["forces"].detach() @ Rt.T,
                          atol=1e-7)


def test_flexible_num_layers():
    for t in range(3):
        model = _build(num_mp=t)
        assert len(model.interactions) == t
        out = ForceStressOutput(model)(_graph())
        assert out["forces"].shape == (8, 3)


@pytest.mark.parametrize("types", [["M", "Ar", "Bchi"], ["Bchi"], ["Ar"], ["M"]])
def test_message_types(types):
    model = _build(num_mp=1, message_types=types)
    layer = model.interactions[0]
    assert (layer.memory is not None) == ("M" in types)
    assert (layer.message_ar is not None) == ("Ar" in types)
    assert (layer.message_bchi is not None) == ("Bchi" in types)
    out = ForceStressOutput(model)(_graph())
    assert torch.isfinite(out["energy"]).all()


def test_unknown_message_type_rejected():
    with pytest.raises(ValueError, match="unknown message types"):
        _build(num_mp=1, message_types=["Ar", "bogus"])


def test_periodic_stress():
    model = ForceStressOutput(_build(num_mp=1, max_l=2, max_nu=2),
                              compute_stress=True)
    out = model(_graph(n=6, periodic=True))
    assert out["stress"].shape == (1, 3, 3)


def test_atomic_energies_shift():
    """set_atomic_energies adds exactly sum_i E0_{Z_i} to the total energy."""
    model = ForceStressOutput(_build(num_mp=0))
    g = _graph()
    e_before = float(model(g)["energy"])
    e0 = {1: -13.6, 8: -2000.0}
    model.model.set_atomic_energies([e0[z] for z in SPECIES])
    e_after = float(model(g)["energy"])
    expected = sum(e0[int(z)] for z in g.atomic_numbers)
    assert abs((e_after - e_before) - expected) < 1e-9


def test_upstream_cace_key_translation():
    """Keys spelled as in the upstream Cace(...) constructor map to xnn names."""
    cfg = from_dict({"model": {
        "name": "cace",
        "zs": [1, 8], "cutoff": 5.5, "num_message_passing": 2,
        "n_atom_basis": 3, "n_rbf": 6, "n_radial_basis": 12,
        "max_l": 3, "max_nu": 3, "type_message_passing": ["Bchi"],
        "embed_receiver_nodes": True, "avg_num_neighbors": 12.0,
        "atomic_energies": {1: -0.5, 8: -2.5},
    }})
    assert cfg.model.cutoff == 5.5
    assert cfg.model.n_interactions == 2
    assert cfg.model.n_rbf == 6
    assert cfg.data.cutoff == 5.5
    m = build_model(cfg.model)
    assert m.species == [1, 8]
    assert len(m.interactions) == 2
    assert m.interactions[0].message_ar is None
    assert m.interactions[0].message_bchi is not None
    assert m.n_radial_basis == 12
    assert m.embed_receiver is not m.embed_sender
    assert m.atom_ref.weight[1].item() == -0.5
    assert m.atom_ref.weight[8].item() == -2.5


def test_parity_vs_original_cace():
    """Weight transplant from the original cace gives identical E and F."""
    pytest.importorskip("cace")
    from cace.modules import BesselRBF as UpBessel
    from cace.modules import PolynomialCutoff as UpPoly
    from cace.modules.atomwise import Atomwise
    from cace.representations import Cace as UpCace

    CUT, NRBF, NRB, NAB, LMAX, NU, T, AVG = 4.5, 6, 8, 2, 3, 3, 2, 9.0
    types = ["M", "Ar", "Bchi"]
    torch.manual_seed(7)
    rb = UpBessel(cutoff=CUT, n_rbf=NRBF, trainable=True)
    up = UpCace(zs=SPECIES, n_atom_basis=NAB, cutoff=CUT, radial_basis=rb,
                cutoff_fn=UpPoly(cutoff=CUT, p=6), max_l=LMAX, max_nu=NU,
                num_message_passing=T, type_message_passing=types,
                n_radial_basis=NRB, avg_num_neighbors=AVG,
                embed_receiver_nodes=True)
    readout = Atomwise(n_layers=3, n_hidden=[32, 16], output_key="energy",
                       add_linear_nn=True)

    x = _build(num_mp=T, message_types=types, embed_receiver_nodes=True)

    for periodic in (False, True):
        g = _graph(periodic=periodic, seed=11)
        cell = g.cell[0] if periodic else torch.zeros(3, 3)
        data = {"positions": g.pos.clone().requires_grad_(True),
                "atomic_numbers": g.atomic_numbers,
                "edge_index": g.edge_index,
                "shifts": g.cell_shifts.to(torch.get_default_dtype()) @ cell,
                "batch": g.batch, "cell": cell.unsqueeze(0)}
        out = readout(up(data))  # also lazy-initializes Bchi hnet / Atomwise

        with torch.no_grad():
            x.embed_sender.copy_(up.node_embedding_sender.embedding_weights)
            x.embed_receiver.copy_(up.node_embedding_receiver.embedding_weights)
            x.rbf.freqs.copy_(rb.bessel_weights * CUT)
            x.radial_transform.weight.copy_(
                torch.stack(list(up.radial_transform.weights)))
            for t in range(T):
                memory, ar, bchi = up.message_passing_list[t]
                xi = x.interactions[t]
                xi.memory.memory_coef.copy_(torch.stack(list(memory.memory_coef)))
                xi.message_ar.prefactor.copy_(torch.stack(list(ar.prefactor)))
                xi.message_ar.inv_r0.copy_(torch.stack(list(ar.invr0)))
                xi.message_bchi.h.weight.copy_(bchi.hnet[0].linear.weight)
                xi.message_bchi.h.bias.copy_(bchi.hnet[0].linear.bias)
            for j, dense in enumerate(readout.outnet):
                x.readout_mlp[2 * j].weight.copy_(dense.linear.weight)
                x.readout_mlp[2 * j].bias.copy_(dense.linear.bias)
            x.readout_linear.weight.copy_(readout.linear_nn.linear.weight)
            x.readout_linear.bias.copy_(readout.linear_nn.linear.bias)

        e_up = out["energy"].sum()
        f_up = -torch.autograd.grad(e_up, data["positions"])[0]
        ox = ForceStressOutput(x)(g)
        # relative tolerances: the rand-initialized radial couplings are not
        # variance-preserving, so raw feature magnitudes are large
        assert abs(float(ox["energy"]) - float(e_up)) / abs(float(e_up)) < 1e-13
        assert ((ox["forces"].detach() - f_up).abs().max()
                / f_up.abs().max()) < 1e-12
