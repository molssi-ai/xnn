"""Tests for the Latent Ewald Summation (LES) long-range add-on.

Covers the :class:`~xnns.common.models.les.EwaldSummation` math (exact
rotation/translation/lattice-shift invariance, cubic and triclinic cells,
``1/r`` and ``1/r^6`` kernels, the analytic two-charge limit), the
:class:`~xnns.common.models.les.LatentEwald` wrapper around **every**
registered model (the ``node_features`` contract), batching consistency, the
``model.extra["long_range"]`` config hook, and -- when the original ``cace``
package is installed -- machine-precision parity against its
``EwaldPotential`` plus a whole-model CACE-LR weight-transplant check.
"""
import math

import numpy as np
import pytest
import torch

from xnns.common.config import from_dict
from xnns.common.data import structure_to_graph, collate
from xnns.common.models import (
    EwaldSummation,
    ForceStressOutput,
    LatentEwald,
    build_model,
)


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _rotation(seed=3):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return torch.tensor(R, dtype=torch.float64)


def _graph(periodic=True, R=None, shift=0.0, seed=0, n=10):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 5, (n, 3)) + shift
    s = {"pos": pos if R is None else pos @ R.numpy().T,
         "atomic_numbers": ([1, 8] * n)[:n]}
    if periodic:
        cell = np.eye(3) * 6.0
        s["cell"] = cell if R is None else cell @ R.numpy().T
        s["pbc"] = [True] * 3
    return structure_to_graph(s, 4.5)


# --------------------------------------------------------------------------
# EwaldSummation math
# --------------------------------------------------------------------------

@pytest.mark.parametrize("exponent", [1, 6])
@pytest.mark.parametrize("triclinic", [False, True])
def test_ewald_exact_invariances(exponent, triclinic):
    rng = np.random.default_rng(0)
    pos = torch.tensor(rng.uniform(0, 5, (10, 3)))
    q = torch.tensor(rng.normal(size=(10, 3)))
    cell = (torch.tensor([[6.0, 0, 0], [1.2, 5.8, 0], [-0.7, 0.9, 6.3]])
            if triclinic else torch.eye(3, dtype=torch.float64) * 6)
    ew = EwaldSummation(exponent=exponent)
    R = _rotation()
    e0 = ew.reciprocal(pos, q, cell)
    assert abs(float(e0) - float(ew.reciprocal(pos @ R.T, q, cell @ R.T))) < 1e-12
    # translation and shift by a full lattice vector
    assert abs(float(e0) - float(ew.reciprocal(pos + 2.34, q, cell))) < 1e-10
    assert abs(float(e0) - float(ew.reciprocal(pos + cell[0], q, cell))) < 1e-10


def test_ewald_two_charge_realspace_analytic():
    """Molecular fallback reproduces the analytic Gaussian-charge energy."""
    d, sigma = 4.0, 1.0
    pos = torch.tensor([[0.0, 0, 0], [d, 0, 0]])
    q = torch.tensor([[1.0], [-1.0]])
    ew = EwaldSummation(sigma=sigma, remove_self_interaction=False)
    e = float(ew.realspace(pos, q))
    # pair term (both directed pairs, upstream's 1e-6-shifted denominator)
    # plus the Gaussian self energy of both charges
    expected = (2 * 1.0 * -1.0 * math.erf(d / (sigma * math.sqrt(2)))
                / (d + 1e-6) / (4 * math.pi)
                + 2.0 / (sigma * (2 * math.pi) ** 1.5))
    assert abs(e - expected) < 1e-12


def test_ewald_realspace_matches_large_box_limit():
    """A big periodic box converges to the molecular direct sum.

    Requires a *neutral* hidden variable: the reciprocal sum omits ``k = 0``
    (tinfoil boundary conditions, paper), so a net q would add a background
    term absent from the isolated-cluster direct sum.
    """
    rng = np.random.default_rng(2)
    pos = torch.tensor(rng.uniform(14, 16, (6, 3)))  # cluster in the middle
    q = torch.tensor(rng.normal(size=(6, 1)))
    q = q - q.mean()
    ew = EwaldSummation(dl=1.0, sigma=1.0, remove_self_interaction=False)
    e_mol = float(ew.realspace(pos, q))
    e_box = float(ew.reciprocal(pos, q, torch.eye(3, dtype=torch.float64) * 30))
    assert abs(e_mol - e_box) < 1e-4  # periodic images + k-cutoff residual


# --------------------------------------------------------------------------
# LatentEwald wrapper
# --------------------------------------------------------------------------

MODEL_CONFIGS = {
    "cace": {"extra": {"species": [1, 8], "n_atom_basis": 2, "max_l": 2,
                       "max_nu": 2}},
    "mace": {"n_features": 8, "extra": {"species": [1, 8], "l_max": 2}},
    "nequip": {"n_features": 8, "extra": {"species": [1, 8], "l_max": 1}},
    "allegro": {"n_features": 8, "extra": {"species": [1, 8], "l_max": 1,
                "two_body_latent": [8, 16], "latent": [16], "edge_eng": [8],
                "avg_num_neighbors": 9.0}},
    "schnet": {"n_features": 16},
    "physnet": {"n_features": 16, "extra": {"use_dispersion": False}},
    "hdnnp": {"extra": {"species": [1, 8]}},
    "ani": {"extra": {"species": [1, 8]}},
}


def _wrapped(name, n_channels=2):
    over = dict(MODEL_CONFIGS[name])
    extra = dict(over.pop("extra", {}))
    cfg = from_dict({"model": {"name": name, "cutoff": 4.5, "n_interactions": 1,
                               "n_rbf": over.pop("n_rbf", 6),
                               "n_features": over.pop("n_features", 8),
                               "extra": extra}})
    base = build_model(cfg.model)
    torch.manual_seed(3)
    return LatentEwald(base, n_channels=n_channels)


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_wraps_every_model(name):
    if name in ("mace", "nequip", "allegro", "cace"):
        pytest.importorskip("e3nn")
    model = ForceStressOutput(_wrapped(name), compute_stress=True)
    out = model(_graph())
    assert torch.allclose(out["energy"], out["energy_sr"] + out["energy_lr"])
    # node energies still sum to the (combined) total
    assert abs(float(out["node_energy"].sum()) - float(out["energy"])) < 1e-10
    assert out["latent_charges"].shape == (10, 2)
    assert out["forces"].shape == (10, 3)
    assert out["stress"].shape == (1, 3, 3)
    assert torch.isfinite(out["forces"]).all()
    # rotation invariance of the combined model
    R = _rotation()
    out_rot = model(_graph(R=R))
    assert abs(float(out["energy"].detach()) - float(out_rot["energy"].detach())) < 1e-9
    assert torch.allclose(out_rot["forces"].detach(),
                          out["forces"].detach() @ R.T, atol=1e-9)
    # molecular structures fall back to the real-space sum
    out_mol = model(_graph(periodic=False))
    assert torch.isfinite(out_mol["energy_lr"]).all()


@pytest.mark.parametrize("periodic", [True, False])
def test_batching_matches_single_structures(periodic):
    """A batch equals per-structure evaluation (periodic and molecular).

    Mixed periodic/molecular batches are not covered: xnns ``collate`` keeps
    optional fields (like ``cell``) only when present in every structure.
    """
    pytest.importorskip("e3nn")
    model = _wrapped("cace")
    e1 = float(model(_graph(periodic=periodic, seed=1))["energy"])
    e2 = float(model(_graph(periodic=periodic, seed=2, n=8))["energy"])
    batch = collate([_graph(periodic=periodic, seed=1),
                     _graph(periodic=periodic, seed=2, n=8)])
    e_batch = model(batch)["energy"]
    assert abs(float(e_batch[0]) - e1) < 1e-10
    assert abs(float(e_batch[1]) - e2) < 1e-10


def test_config_hook_builds_wrapped_model():
    pytest.importorskip("e3nn")
    cfg = from_dict({"model": {"name": "cace", "cutoff": 4.5,
                               "n_interactions": 0, "n_rbf": 6,
                               "extra": {"species": [1, 8], "n_atom_basis": 2,
                                         "max_l": 2, "max_nu": 2,
                                         "long_range": {"n_channels": 3,
                                                        "sigma": 1.5, "dl": 3.0}}}})
    model = build_model(cfg.model)
    assert isinstance(model, LatentEwald)
    assert model.ewald.sigma == 1.5
    assert model.q_linear.out_features == 3
    assert model.cutoff == 4.5


def test_q_head_variants():
    """The q head is configurable: bias and the optional parallel linear."""
    pytest.importorskip("e3nn")
    base_cfg = {"name": "cace", "cutoff": 4.5, "n_interactions": 0, "n_rbf": 6,
                "extra": {"species": [1, 8], "n_atom_basis": 2, "max_l": 2,
                          "max_nu": 2}}
    base = build_model(from_dict({"model": base_cfg}).model)
    # water-script default: bias-free MLP + parallel linear
    m_water = LatentEwald(base, n_channels=4)
    assert m_water.q_linear is not None
    assert m_water.q_net[0].bias is None
    # charged-dimer variant: 1-channel MLP with bias, no parallel linear
    m_dimer = LatentEwald(base, n_channels=1, q_bias=True, q_add_linear=False)
    assert m_dimer.q_linear is None
    assert m_dimer.q_net[0].bias is not None
    out = ForceStressOutput(m_dimer)(_graph())
    assert out["latent_charges"].shape == (10, 1)
    assert torch.isfinite(out["energy"]).all()


def test_rejects_model_without_features():
    class Bare(torch.nn.Module):
        cutoff = 4.0
    with pytest.raises(TypeError, match="node_feature"):
        LatentEwald(Bare())


# --------------------------------------------------------------------------
# fidelity vs the original cace EwaldPotential
# --------------------------------------------------------------------------

def test_parity_vs_original_ewald():
    """The Ewald kernels match upstream cace.modules.EwaldPotential.

    float64 against upstream's dtype-safe orthorhombic reference loop and its
    real-space fallback; float32 against the full triclinic forward (the code
    path used in production, which is float32-only upstream). ``dl`` is
    chosen so no k shell lies exactly on the cutoff (where xnns resolves
    floating-point ties consistently and upstream truncates).
    """
    pytest.importorskip("cace")
    from cace.modules.ewald import EwaldPotential

    rng = np.random.default_rng(1)
    pos = torch.tensor(rng.uniform(0, 6, (12, 3)))
    q = torch.tensor(rng.normal(size=(12, 4)))
    box = torch.tensor([6.0, 6.0, 6.0], dtype=torch.float64)

    up = EwaldPotential(dl=1.9, sigma=1.0, remove_self_interaction=False)
    mine = EwaldSummation(dl=1.9, sigma=1.0, remove_self_interaction=False)
    pot, _ = up.compute_potential(pos, q, box)
    assert abs(float(mine.reciprocal(pos, q, torch.diag(box)))
               - float(pot.sum())) < 1e-12

    pot6, _ = EwaldPotential(dl=1.9, exponent=6,
                             remove_self_interaction=False).compute_potential(pos, q, box)
    mine6 = EwaldSummation(dl=1.9, exponent=6, remove_self_interaction=False)
    assert abs(float(mine6.reciprocal(pos, q, torch.diag(box)))
               - float(pot6.sum())) < 1e-12

    # self-interaction removal agrees for 1-channel q (upstream over-subtracts
    # the total once per channel for multi-channel q; xnns subtracts it once)
    q1 = q[:, :1]
    potr, _ = EwaldPotential(dl=1.9, remove_self_interaction=True).compute_potential(pos, q1, box)
    miner = EwaldSummation(dl=1.9, remove_self_interaction=True)
    assert abs(float(miner.reciprocal(pos, q1, torch.diag(box)))
               - float(potr.sum())) < 1e-12

    pot_rs, _ = up.compute_potential_realspace(pos, q)
    assert abs(float(mine.realspace(pos, q)) - float(pot_rs.sum())) < 1e-12

    # float32 triclinic against the production forward path
    torch.set_default_dtype(torch.float32)
    cell32 = torch.tensor([[6.0, 0, 0], [1.2, 5.8, 0], [-0.7, 0.9, 6.3]],
                          dtype=torch.float32)
    data = {"positions": pos.float(), "cell": cell32.unsqueeze(0),
            "q": q.float(), "batch": torch.zeros(12, dtype=torch.long)}
    up32 = EwaldPotential(dl=1.9, sigma=1.0, remove_self_interaction=False,
                          feature_key="q")
    e_up = float(up32(data)["ewald_potential"][0])
    e32 = float(EwaldSummation(dl=1.9, sigma=1.0, remove_self_interaction=False)
                .reciprocal(pos.float(), q.float(), cell32))
    assert abs(e32 - e_up) / abs(e_up) < 1e-5


def test_parity_vs_original_cace_lr():
    """Weight-transplanted LatentEwald(CACE) == upstream CACE-LR NNP.

    Runs in float32 -- upstream's triclinic Ewald path hard-codes a float32
    k-grid and cannot run in float64; the float64 math is covered by
    ``test_parity_vs_original_ewald`` and ``tests/test_cace.py``.
    """
    pytest.importorskip("cace")
    from cace.models.atomistic import NeuralNetworkPotential
    from cace.modules import BesselRBF as UpBessel
    from cace.modules import FeatureAdd
    from cace.modules import PolynomialCutoff as UpPoly
    from cace.modules.atomwise import Atomwise
    from cace.modules.ewald import EwaldPotential
    from cace.modules.forces import Forces
    from cace.representations import Cace as UpCace

    torch.set_default_dtype(torch.float32)
    CUT, NRBF, NRB, NAB, LMAX, NU, T, NQ = 4.5, 6, 8, 2, 3, 3, 1, 4
    torch.manual_seed(7)
    rep = UpCace(zs=[1, 8], n_atom_basis=NAB, cutoff=CUT,
                 radial_basis=UpBessel(cutoff=CUT, n_rbf=NRBF, trainable=True),
                 cutoff_fn=UpPoly(cutoff=CUT, p=6), max_l=LMAX, max_nu=NU,
                 num_message_passing=T, type_message_passing=["M", "Ar", "Bchi"],
                 n_radial_basis=NRB, avg_num_neighbors=9.0,
                 embed_receiver_nodes=True)
    sr = Atomwise(n_layers=3, n_hidden=[32, 16], output_key="SR_energy",
                  add_linear_nn=True)
    qhead = Atomwise(n_layers=3, n_hidden=[24, 12], n_out=NQ,
                     per_atom_output_key="q", output_key="tot_q",
                     residual=False, add_linear_nn=True, bias=False)
    ep = EwaldPotential(dl=1.9, sigma=1.0, feature_key="q",
                        output_key="ewald_potential",
                        remove_self_interaction=False, aggregation_mode="sum")
    eadd = FeatureAdd(feature_keys=["SR_energy", "ewald_potential"],
                      output_key="CACE_energy")
    nnp = NeuralNetworkPotential(
        representation=rep,
        output_modules=[sr, qhead, ep, eadd,
                        Forces(energy_key="CACE_energy",
                               forces_key="CACE_forces")])

    cfg = from_dict({"model": {
        "name": "cace", "cutoff": CUT, "n_interactions": T, "n_rbf": NRBF,
        "extra": {"species": [1, 8], "n_atom_basis": NAB, "n_radial_basis": NRB,
                  "max_l": LMAX, "max_nu": NU, "avg_num_neighbors": 9.0,
                  "embed_receiver_nodes": True,
                  "long_range": {"n_channels": NQ, "dl": 1.9, "sigma": 1.0}}}})
    x = build_model(cfg.model)

    rng = np.random.default_rng(11)
    pos = rng.uniform(0, 6, (12, 3))
    g = structure_to_graph({"pos": pos, "atomic_numbers": [1, 8] * 6,
                            "cell": np.eye(3) * 6.0, "pbc": [True] * 3}, CUT)
    cell = g.cell[0]
    data = {"positions": g.pos.clone().requires_grad_(True),
            "atomic_numbers": g.atomic_numbers, "edge_index": g.edge_index,
            "shifts": g.cell_shifts.to(cell.dtype) @ cell,
            "unit_shifts": g.cell_shifts.to(cell.dtype),
            "batch": g.batch, "cell": cell.unsqueeze(0)}
    out_up = nnp(data, training=True)  # lazy-inits the Atomwise heads

    with torch.no_grad():
        b = x.model
        b.embed_sender.copy_(rep.node_embedding_sender.embedding_weights)
        b.embed_receiver.copy_(rep.node_embedding_receiver.embedding_weights)
        b.rbf.freqs.copy_(rep.radial_basis.bessel_weights * CUT)
        b.radial_transform.weight.copy_(
            torch.stack(list(rep.radial_transform.weights)))
        for t, (memory, ar, bchi) in enumerate(rep.message_passing_list):
            xi = b.interactions[t]
            xi.memory.memory_coef.copy_(torch.stack(list(memory.memory_coef)))
            xi.message_ar.prefactor.copy_(torch.stack(list(ar.prefactor)))
            xi.message_ar.inv_r0.copy_(torch.stack(list(ar.invr0)))
            xi.message_bchi.h.weight.copy_(bchi.hnet[0].linear.weight)
            xi.message_bchi.h.bias.copy_(bchi.hnet[0].linear.bias)
        for j, dense in enumerate(sr.outnet):
            b.readout_mlp[2 * j].weight.copy_(dense.linear.weight)
            b.readout_mlp[2 * j].bias.copy_(dense.linear.bias)
        b.readout_linear.weight.copy_(sr.linear_nn.linear.weight)
        b.readout_linear.bias.copy_(sr.linear_nn.linear.bias)
        for j, dense in enumerate(qhead.outnet):
            x.q_net[2 * j].weight.copy_(dense.linear.weight)
        x.q_linear.weight.copy_(qhead.linear_nn.linear.weight)

    e_up = out_up["CACE_energy"].sum()
    f_up = -torch.autograd.grad(e_up, data["positions"])[0]
    ox = ForceStressOutput(x)(g)
    assert abs(float(ox["energy"]) - float(e_up)) / abs(float(e_up)) < 1e-5
    assert ((ox["forces"].detach() - f_up).abs().max()
            / f_up.abs().max()) < 1e-4
    q_up = out_up["q"]
    assert ((ox["latent_charges"].detach() - q_up).abs().max()
            / q_up.abs().max()) < 1e-5
