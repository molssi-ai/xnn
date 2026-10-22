"""Tests for the Latent Ewald Summation (LES) long-range add-on.

Covers the :class:`~xnn.common.models.les.EwaldSummation` math (exact
rotation/translation/lattice-shift invariance, cubic and triclinic cells,
``1/r`` and ``1/r^6`` kernels, the analytic two-charge limit), the
:class:`~xnn.common.models.les.LatentEwald` wrapper around **every**
registered model (the ``node_features`` contract), batching consistency, the
``model.extra["long_range"]`` config hook, and -- when the original ``cace``
package is installed -- machine-precision parity against its
``EwaldPotential`` plus a whole-model CACE-LR weight-transplant check.
"""
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import (
    EwaldSummation,
    ForceStressOutput,
    LatentEwald,
    build_model,
)
from xnn.common.models.les import LATENT_CHARGE_PER_E


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


# EwaldSummation math
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


# LatentEwald wrapper

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

    Mixed periodic/molecular batches are not covered: xnn ``collate`` keeps
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


def test_k_cutoff_follows_dl_on_a_live_module():
    """``dl`` can be retuned after construction (a convergence study).

    The k-space cutoff is derived from ``dl`` on every call rather than
    cached at construction: assigning ``ewald.dl`` on a model that is already
    built (or loaded from a checkpoint) must change the k-shell, not silently
    do nothing. Only the shell selection may depend on it -- a module retuned
    to a given ``dl`` has to agree bit for bit with one constructed with it.
    """
    torch.manual_seed(0)
    pos = torch.rand(40, 3) * 11.0
    q = torch.randn(40, 4) * 0.1
    cell = torch.eye(3) * 11.0

    retuned = EwaldSummation(dl=3.0, sigma=1.0)
    coarse = float(retuned.reciprocal(pos, q, cell))
    for dl in (2.0, 1.5, 1.0):
        retuned.dl = dl
        built = EwaldSummation(dl=dl, sigma=1.0)
        assert float(retuned.reciprocal(pos, q, cell)) == \
            float(built.reciprocal(pos, q, cell))
    # and the cutoff genuinely matters at this smearing: a coarse grid drops
    # part of the reciprocal sum (sigma = 1 leaves exp(-sigma^2 k_c^2 / 2)
    # = 0.11 at k_c = 2 pi / 3, against 0.007 at k_c = pi)
    retuned.dl = 1.0
    assert abs(float(retuned.reciprocal(pos, q, cell)) - coarse) > 1e-6


def test_dispersion_is_periodic_only():
    """``exponent = 6`` has no real-space branch, upstream or here.

    A structure without a cell takes the real-space sum, which implements the
    ``1/r`` kernel only, so a dispersion channel needs periodic training data
    rather than failing silently on molecular frames.
    """
    ew = EwaldSummation(dl=2.0, sigma=1.0, exponent=6)
    pos, q = torch.rand(9, 3) * 3.0, torch.randn(9, 4) * 0.1
    assert torch.isfinite(ew.reciprocal(pos, q, torch.eye(3) * 12.0))
    with pytest.raises(ValueError, match="exponent=1 only"):
        ew.realspace(pos, q)


# fidelity vs the original cace EwaldPotential
def test_parity_vs_original_ewald():
    """The Ewald kernels match upstream cace.modules.EwaldPotential.

    float64 against upstream's dtype-safe orthorhombic reference loop and its
    real-space fallback; float32 against the full triclinic forward (the code
    path used in production, which is float32-only upstream). ``dl`` is
    chosen so no k shell lies exactly on the cutoff (where xnn resolves
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
    # the total once per channel for multi-channel q; xnn subtracts it once)
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


# the optional net-charge constraint

SCHNET = {"name": "schnet", "cutoff": 4.5, "n_interactions": 1, "n_rbf": 6, "n_features": 8}


def _schnet_les(seed=3, **options):
    torch.manual_seed(seed)
    base = build_model(from_dict({"model": SCHNET}).model)
    return LatentEwald(base, n_channels=3, **options)


def _charged(seed, n, charge, periodic=False):
    rng = np.random.default_rng(seed)
    s = {"pos": rng.uniform(0, 5, (n, 3)), "atomic_numbers": ([1, 8] * n)[:n],
         "total_charge": charge}
    if periodic:
        s.update(cell=np.eye(3) * 6.0, pbc=[True] * 3)
    return structure_to_graph(s, 4.5)


def test_latent_charge_unit_is_the_derived_coulomb_constant():
    """Two opposite elementary charges 20 A apart interact with -14.3996/20 eV."""
    assert LATENT_CHARGE_PER_E == pytest.approx(math.sqrt(2 * math.pi * 14.399645), rel=1e-7)
    ewald = EwaldSummation(sigma=1.0, remove_self_interaction=True)
    q = torch.tensor([[1.0], [-1.0]]) * LATENT_CHARGE_PER_E
    pos = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    e = ewald(q, pos, torch.zeros(2, dtype=torch.long), 1, None, None)
    assert float(e) == pytest.approx(-14.399645 / 20.0, rel=1e-6)


@pytest.mark.parametrize("weights", ["uniform", "learned"])
@pytest.mark.parametrize("periodic", [False, True])
def test_charge_constraint_pins_every_structure(weights, periodic):
    """Per-structure sums in a batch; option off leaves the head's charges."""
    model = _schnet_les(constrain_charge=True, charge_channel=1, charge_weights=weights)
    batch = collate([_charged(1, 7, -1.0, periodic), _charged(2, 5, 0.0, periodic)])
    out = model(batch)
    q = out["latent_charges"]
    sums = torch.zeros(2, 3).index_add_(0, batch.batch, q)
    assert torch.allclose(sums[:, 1], torch.tensor([-1.0, 0.0]) * LATENT_CHARGE_PER_E, atol=1e-10)
    assert torch.allclose(sums[:, [0, 2]], torch.zeros(2, 2), atol=1e-10)
    # a batch equals the single structures
    single = model(_charged(1, 7, -1.0, periodic))
    assert torch.allclose(single["latent_charges"], q[:7], atol=1e-10)
    assert float(single["energy"]) == pytest.approx(float(out["energy"][0]), abs=1e-10)
    # the net charge changes the energy, so >TOTCHARGE / charge labels matter
    neutral = model(_charged(1, 7, 0.0, periodic))
    assert abs(float(neutral["energy_lr"]) - float(single["energy_lr"])) > 1e-6
    # off: the reference head, untouched
    plain = _schnet_les()
    assert not plain.constrain_charge and plain.charge_weight is None
    ref = plain(_charged(1, 7, -1.0, periodic))["latent_charges"]
    assert not torch.allclose(ref, single["latent_charges"])
    assert torch.allclose(ref, _schnet_les(constrain_charge=False)(_charged(1, 7, -1.0, periodic))["latent_charges"])


def test_charge_constraint_without_labels_is_neutral_and_forces_are_consistent():
    """No label means neutral; the forces are the energy gradient."""
    model = _schnet_les(constrain_charge=True)
    s = {"pos": np.random.default_rng(4).uniform(0, 5, (6, 3)), "atomic_numbers": [1, 8] * 3}
    g = structure_to_graph(s, 4.5)
    assert g.total_charge is None
    q = model(g)["latent_charges"]
    assert torch.allclose(q.sum(0), torch.zeros(3), atol=1e-10)
    charged = dict(s, total_charge=1.0)
    out = ForceStressOutput(model)(structure_to_graph(charged, 4.5))
    h = 1e-5
    for atom, comp in ((0, 0), (3, 2)):
        plus, minus = np.array(s["pos"]), np.array(s["pos"])
        plus[atom, comp] += h
        minus[atom, comp] -= h
        e_p = float(model(structure_to_graph(dict(charged, pos=plus), 4.5))["energy"])
        e_m = float(model(structure_to_graph(dict(charged, pos=minus), 4.5))["energy"])
        assert float(out["forces"][atom, comp]) == pytest.approx(-(e_p - e_m) / (2 * h), abs=1e-6)


def test_charge_constraint_options_are_validated_and_reach_the_config_hook():
    with pytest.raises(ValueError, match="charge_channel"):
        _schnet_les(constrain_charge=True, charge_channel=3)
    with pytest.raises(ValueError, match="charge_weights"):
        _schnet_les(constrain_charge=True, charge_weights="softmax")
    cfg = from_dict({"model": {**SCHNET, "extra": {"long_range": {
        "n_channels": 2, "constrain_charge": True, "charge_weights": "learned"}}}})
    model = build_model(cfg.model)
    assert isinstance(model, LatentEwald) and model.constrain_charge
    assert model.charge_weight is not None and model.charge_channel == 0


def test_trainer_warns_when_the_constraint_sees_no_charge_labels(tmp_path):
    """One warning when no training structure carries a charge label."""
    import warnings

    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer
    rng = np.random.default_rng(0)

    def structures(labeled):
        out = []
        for _ in range(3):
            s = {"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 8, 1, 8],
                 "energy": 0.0, "forces": np.zeros((4, 3))}
            if labeled:
                s["total_charge"] = 0.0
            out.append(s)
        return out

    def trainer(constrain, labeled):
        cfg = from_dict({"model": {**SCHNET, "extra": {"long_range": {
            "n_channels": 2, "constrain_charge": constrain}}},
            "data": {"batch_size": 3, "val_fraction": 0.34},
            "optim": {"epochs": 1}, "device": "cpu", "output_dir": str(tmp_path / "run")})
        return Trainer(cfg, AtomicDataset(structures(labeled), cfg.model.cutoff))

    with pytest.warns(UserWarning, match="no training structure carries"):
        trainer(True, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trainer(True, True)
        trainer(False, False)


def _dipole_about(q, pos, numbers, origin):
    from xnn.common.models.ops import ATOMIC_MASSES
    if origin == "com":
        m = np.asarray(ATOMIC_MASSES)[numbers][:, None]
        center = (m * pos).sum(0) / m.sum()
    elif origin == "centroid":
        center = pos.mean(0)
    else:
        center = np.zeros(3)
    return (q[:, None] * (pos - center)).sum(0)


@pytest.mark.parametrize("origin", ["com", "centroid", "origin"])
def test_les_charges_and_dipole_outputs(origin):
    """The charge channel in e, and its dipole about the chosen origin for molecules only."""
    model = _schnet_les(constrain_charge=True, charge_channel=1, dipole_origin=origin)
    graphs = [_charged(1, 7, -1.0), _charged(2, 5, 0.0), _charged(3, 6, 1.0, periodic=True)]
    batch = collate(graphs)
    out = model(batch)
    q, mu = out["les_charges"], out["les_dipole"]
    assert q.shape == (18,) and mu.shape == (3, 3)
    assert torch.allclose(q * LATENT_CHARGE_PER_E, out["latent_charges"][:, 1], atol=1e-12)
    sums = torch.zeros(3).index_add_(0, batch.batch, q)
    assert torch.allclose(sums, torch.tensor([-1.0, 0.0, 1.0]), atol=1e-10)
    start = 0
    for i, g in enumerate(graphs[:2]):
        n = g.pos.shape[0]
        ref = _dipole_about(q[start:start + n].detach().numpy(), g.pos.numpy(),
                            g.atomic_numbers.numpy(), origin)
        assert np.allclose(mu[i].detach().numpy(), ref, atol=1e-10)
        start += n
    assert torch.all(mu[2] == 0)
    # a neutral molecule's dipole does not depend on the origin
    neutral = _charged(2, 5, 0.0)
    mus = [_schnet_les(constrain_charge=True, charge_channel=1, dipole_origin=o)(neutral)["les_dipole"]
           for o in ("com", "centroid", "origin")]
    assert torch.allclose(mus[0], mus[1], atol=1e-10) and torch.allclose(mus[0], mus[2], atol=1e-10)
    # the charge solve exports the solved charges
    solved = _schnet_les(charge_solve=True)(graphs[0])
    assert torch.allclose(solved["les_charges"] * LATENT_CHARGE_PER_E,
                          solved["latent_charges"][:, 0], atol=1e-12)


def test_les_dipole_origin_is_validated_and_reaches_the_config_hook():
    with pytest.raises(ValueError, match="dipole_origin"):
        _schnet_les(dipole_origin="corner")
    cfg = from_dict({"model": {**SCHNET, "extra": {"long_range": {
        "n_channels": 1, "dipole_origin": "origin"}}}})
    model = build_model(cfg.model)
    assert isinstance(model, LatentEwald) and model.dipole_origin == "origin"
    assert model.dipole_origin != _schnet_les().dipole_origin == "com"


def test_les_charge_and_dipole_terms_fit_the_labels():
    """The les_* loss terms score the LES outputs against the charge and dipole labels."""
    from xnn.common.benchmark.metrics import prediction_pairs
    from xnn.common.train import weighted_loss
    rng = np.random.default_rng(7)
    structs = []
    for n, charge in ((5, 0.0), (4, 1.0)):
        s = {"pos": rng.uniform(0, 4, (n, 3)), "atomic_numbers": ([8, 1] * n)[:n],
             "total_charge": charge, "energy": 0.0, "forces": np.zeros((n, 3)),
             "charges": rng.normal(scale=0.3, size=n), "dipole": rng.normal(size=3)}
        s["charges"] += (charge - s["charges"].sum()) / n
        structs.append(s)
    cell = {"pos": rng.uniform(0, 5, (6, 3)), "atomic_numbers": [8, 1] * 3, "energy": 0.0,
            "forces": np.zeros((6, 3)), "cell": np.eye(3) * 6.0, "pbc": [True] * 3,
            "charges": rng.normal(scale=0.3, size=6)}
    batch = collate([structure_to_graph(s, 4.5) for s in structs + [cell]])
    assert batch.dipole_mask.tolist() == [True, True, False]
    model = _schnet_les(constrain_charge=True)
    pred = model(batch)
    _, plain = weighted_loss(pred, batch, 1.0, 0.0, 0.0, charge_weight=1.0, dipole_weight=1.0)
    assert "les_charge_mse" not in plain and "charge_mse" not in plain
    loss, logs = weighted_loss(pred, batch, 1.0, 0.0, 0.0, les_charge_weight=2.0,
                               les_dipole_weight=3.0)
    q_err = float(((pred["les_charges"] - batch.charges) ** 2).mean())
    mu_err = float(((pred["les_dipole"][:2] - batch.dipole[:2]) ** 2).mean(-1).mean())
    assert logs["les_charge_mse"] == pytest.approx(q_err, rel=1e-10)
    assert logs["les_dipole_mse"] == pytest.approx(mu_err, rel=1e-10)
    assert float(loss) == pytest.approx(plain["loss"] + 2 * q_err + 3 * mu_err, rel=1e-10)
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.q_net.parameters())
    pairs = prediction_pairs(pred, batch, ["les_dipole", "les_charges"])
    assert pairs["les_dipole"][0].shape == (6,) and pairs["les_charges"][0].shape == (15,)


def test_trainer_warns_when_les_terms_cannot_apply(tmp_path):
    """A warning for les_* weights without an LES wrapper or without labels."""
    import warnings

    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer
    rng = np.random.default_rng(0)

    def structures(labeled):
        out = []
        for _ in range(3):
            s = {"pos": rng.uniform(0, 4, (4, 3)), "atomic_numbers": [1, 8, 1, 8],
                 "energy": 0.0, "forces": np.zeros((4, 3))}
            if labeled:
                s["dipole"] = rng.normal(size=3)
            out.append(s)
        return out

    def trainer(les, labeled):
        extra = {"long_range": {"n_channels": 1}} if les else {}
        cfg = from_dict({"model": {**SCHNET, "extra": extra},
                         "data": {"batch_size": 3, "val_fraction": 0.34},
                         "optim": {"epochs": 1, "les_dipole_weight": 1.0},
                         "device": "cpu", "output_dir": str(tmp_path / "run")})
        return Trainer(cfg, AtomicDataset(structures(labeled), cfg.model.cutoff))

    with pytest.warns(UserWarning, match="no LES wrapper"):
        trainer(False, True)
    with pytest.warns(UserWarning, match="no training structure carries a 'dipole'"):
        trainer(True, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trainer(True, True)


@pytest.mark.parametrize("solve", [False, True])
def test_dielectric_screens_the_charge_channel_only(solve):
    """The charge channel's long-range energy is divided by the dielectric; the others are not."""
    kw = {"charge_solve": True} if solve else {"constrain_charge": True}
    vacuum = _schnet_les(charge_channel=1, **kw)
    screened = _schnet_les(charge_channel=1, dielectric=78.4, **kw)
    screened.load_state_dict({k: v for k, v in vacuum.state_dict().items()
                              if k != "log_dielectric"}, strict=False)
    assert float(screened.dielectric) == pytest.approx(78.4)
    g = _charged(1, 7, -1.0)
    out_v, out_s = vacuum(g), screened(g)
    # the same charges enter; only the charge channel's energy is scaled
    if not solve:
        assert torch.allclose(out_s["latent_charges"], out_v["latent_charges"], atol=1e-12)
        q = out_v["latent_charges"]
        ew = vacuum.ewald
        per_channel = [float(ew(q[:, [c]], g.pos, g.batch, 1, None, None)) for c in range(3)]
        expected = per_channel[0] + per_channel[1] / 78.4 + per_channel[2]
        assert float(out_s["energy_lr"]) == pytest.approx(expected, rel=1e-10)
        assert torch.allclose(out_s["les_charges"], out_v["les_charges"], atol=1e-12)
    else:
        # the solve sees the screened coupling: a large dielectric approaches the
        # coupling-free minimum chi.q + 1/2 J q^2 under the constraint
        assert not torch.allclose(out_s["les_charges"], out_v["les_charges"])
        free = _schnet_les(charge_channel=1, dielectric=1e12, charge_solve=True)
        free.load_state_dict({k: v for k, v in vacuum.state_dict().items()
                              if k != "log_dielectric"}, strict=False)
        out_f = free(g)
        features = free.model(g)["node_features"]
        chi = (free.q_net(features) + free.q_linear(features))[:, 1]
        q = out_f["latent_charges"][:, 1]
        grad = chi + out_f["hardness"] * q
        assert torch.allclose(grad, out_f["chemical_potential"].expand_as(grad), atol=1e-6)
    assert float(out_s["les_charges"].sum()) == pytest.approx(-1.0, abs=1e-8)


def test_dielectric_is_learnable_and_validated():
    with pytest.raises(ValueError, match="dielectric"):
        _schnet_les(dielectric=0.0)
    model = _schnet_les(constrain_charge=True, dielectric=10.0, learn_dielectric=True)
    assert isinstance(model.log_dielectric, torch.nn.Parameter)
    out = model(_charged(1, 7, 1.0))
    out["energy"].sum().backward()
    assert model.log_dielectric.grad is not None and float(model.log_dielectric.grad.abs()) > 0
    fixed = _schnet_les(dielectric=10.0)
    assert "log_dielectric" in fixed.state_dict() and not any(
        n == "log_dielectric" for n, _ in fixed.named_parameters())
    cfg = from_dict({"model": {**SCHNET, "extra": {"long_range": {
        "n_channels": 1, "dielectric": 78.4, "learn_dielectric": True}}}})
    built = build_model(cfg.model)
    assert float(built.dielectric) == pytest.approx(78.4)


@pytest.mark.parametrize("learn", [False, True])
def test_checkpoint_without_dielectric_loads_strictly(learn):
    """A state dict from before the dielectric existed loads with the built value."""
    model = _schnet_les(constrain_charge=True, dielectric=78.4, learn_dielectric=learn)
    state = {k: v for k, v in model.state_dict().items() if k != "log_dielectric"}
    fresh = _schnet_les(seed=5, constrain_charge=True, dielectric=78.4, learn_dielectric=learn)
    fresh.load_state_dict(state, strict=True)
    assert float(fresh.dielectric) == pytest.approx(78.4)
    g = _charged(1, 7, -1.0)
    assert torch.allclose(fresh(g)["energy"], model(g)["energy"], atol=1e-12)
    wrapped = ForceStressOutput(_schnet_les(seed=5, constrain_charge=True))
    wrapped.load_state_dict({"model." + k: v for k, v in state.items()}, strict=True)
    assert float(wrapped.model.dielectric) == pytest.approx(1.0)


def test_ase_calculator_reports_les_charges_and_dipole():
    pytest.importorskip("ase")
    from ase import Atoms

    from xnn.common.deploy import XNNCalculator

    model = ForceStressOutput(_schnet_les(constrain_charge=True)).eval()
    atoms = Atoms(numbers=[8, 1, 1], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    atoms.info["charge"] = -1.0
    atoms.calc = XNNCalculator(model, cutoff=4.5)
    assert atoms.get_charges().sum() == pytest.approx(-1.0, abs=1e-8)
    assert atoms.get_dipole_moment().shape == (3,)
    cell = Atoms(numbers=[8, 1, 1] * 2, positions=np.random.default_rng(1).uniform(0, 5, (6, 3)),
                 cell=np.eye(3) * 6.0, pbc=True)
    cell.calc = XNNCalculator(model, cutoff=4.5)
    cell.get_potential_energy()
    assert "charges" in cell.calc.results and "dipole" not in cell.calc.results


def _roundoff(ew, pos, q):
    """eps times the sum of the magnitudes of the terms in ``ew.realspace(pos, q)``: how far two
    summation orders of the same terms can drift apart."""
    d = torch.linalg.norm(pos[:, None] - pos[None], dim=-1)
    kernel = torch.special.erf(d / (ew.sigma * math.sqrt(2.0))) / (d + 1e-6)
    s = ((q[:, None] * q[None]).abs() * kernel[..., None]).sum() / (4.0 * math.pi)
    if not ew.remove_self_interaction:
        s = s + q.square().sum() / (ew.sigma * (2.0 * math.pi) ** 1.5)
    return torch.finfo(pos.dtype).eps * s.detach()


@pytest.mark.parametrize("remove_self", [False, True])
@pytest.mark.parametrize("channels", [1, 4])
def test_molecular_batch_matches_one_structure_at_a_time(remove_self, channels):
    """The batched molecular sum equals ``realspace`` structure by structure, in any atom order."""
    gen = torch.Generator().manual_seed(7)
    sizes = [1, 5, 12, 3, 30]
    batch = torch.cat([torch.full((n,), b) for b, n in enumerate(sizes)])
    perm = torch.randperm(batch.numel(), generator=gen)          # atoms not grouped by structure
    batch = batch[perm]
    pos = (torch.rand(batch.numel(), 3, generator=gen) * 6.0).requires_grad_()
    q = torch.randn(batch.numel(), channels, generator=gen)
    ew = EwaldSummation(sigma=1.0, remove_self_interaction=remove_self)
    batched = ew(q, pos, batch, len(sizes), None)
    single = torch.stack([ew.realspace(pos[batch == b], q[batch == b])
                          for b in range(len(sizes))])
    bound = torch.stack([_roundoff(ew, pos[batch == b], q[batch == b])
                         for b in range(len(sizes))])
    assert ((batched - single).abs() <= 4 * bound).all()
    g1, = torch.autograd.grad(batched.sum(), pos, create_graph=True)
    g2, = torch.autograd.grad(single.sum(), pos, create_graph=True)
    eps = torch.finfo(pos.dtype).eps
    assert (g1 - g2).abs().max() <= 8 * eps * g2.abs().max()
    # second derivatives (force training): finite here, while the per-structure sum's double
    # backward is NaN through its zero diagonal distances; checked against central
    # differences of the per-structure first derivatives
    h1, = torch.autograd.grad((g1 ** 2).sum(), pos)
    assert torch.isfinite(h1).all()

    def g_squared(p):
        e = torch.stack([ew.realspace(p[batch == b], q[batch == b]) for b in range(len(sizes))])
        g, = torch.autograd.grad(e.sum(), p)
        return (g ** 2).sum()

    h_fd = torch.zeros_like(h1)
    step = 1e-5
    for k in range(pos.numel()):
        dp = torch.zeros(pos.numel())
        dp[k] = step
        dp = dp.reshape(pos.shape)
        plus = g_squared((pos.detach() + dp).requires_grad_())
        minus = g_squared((pos.detach() - dp).requires_grad_())
        h_fd.view(-1)[k] = (plus - minus) / (2 * step)
    assert torch.allclose(h1, h_fd, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("remove_self", [False, True])
@pytest.mark.parametrize("cell", [None, "zero"])
def test_single_molecule_matches_realspace(remove_self, cell):
    """A batch of one molecule takes the dense sum: ``realspace`` values, finite Hessians."""
    gen = torch.Generator().manual_seed(5)
    pos = (torch.rand(25, 3, generator=gen) * 6.0).requires_grad_()
    q = torch.randn(25, 4, generator=gen)
    batch = torch.zeros(25, dtype=torch.long)
    ew = EwaldSummation(sigma=1.0, remove_self_interaction=remove_self)
    cells = None if cell is None else torch.zeros(1, 3, 3)
    pbc = None if cell is None else torch.zeros(1, 3, dtype=torch.bool)
    single = ew(q, pos, batch, 1, cells, pbc)
    reference = ew.realspace(pos, q)
    assert (single - reference).abs() <= 4 * _roundoff(ew, pos, q)
    g1, = torch.autograd.grad(single.sum(), pos, create_graph=True)
    g2, = torch.autograd.grad(reference, pos)
    eps = torch.finfo(pos.dtype).eps
    assert (g1 - g2).abs().max() <= 8 * eps * g2.abs().max()
    h1, = torch.autograd.grad((g1 ** 2).sum(), pos)
    gb, = torch.autograd.grad(ew.realspace_batch(pos, q, batch, 1).sum(), pos, create_graph=True)
    h2, = torch.autograd.grad((gb ** 2).sum(), pos)
    assert torch.isfinite(h1).all()
    assert (h1 - h2).abs().max() <= 64 * eps * h2.abs().max()


def test_mixed_batch_takes_each_structure_its_own_way():
    """Molecular structures in a mixed batch get the direct sum, periodic ones the reciprocal sum."""
    gen = torch.Generator().manual_seed(11)
    sizes = [4, 6, 5]
    batch = torch.cat([torch.full((n,), b) for b, n in enumerate(sizes)])
    pos = torch.rand(batch.numel(), 3, generator=gen) * 5.0
    q = torch.randn(batch.numel(), 2, generator=gen)
    cell = torch.zeros(3, 3, 3)
    cell[1] = torch.eye(3) * 7.0                                   # only structure 1 is periodic
    pbc = torch.tensor([[False] * 3, [True] * 3, [False] * 3])
    ew = EwaldSummation(sigma=1.0, dl=1.9)
    out = ew(q, pos, batch, 3, cell, pbc)
    m = [batch == b for b in range(3)]
    for b in (0, 2):
        diff = (out[b] - ew.realspace(pos[m[b]], q[m[b]])).abs()
        assert diff <= 4 * _roundoff(ew, pos[m[b]], q[m[b]])
    assert torch.equal(out[1], ew.reciprocal(pos[m[1]], q[m[1]], cell[1]))
