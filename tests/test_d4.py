"""Tests for the DFT-D4 dispersion model (:mod:`xnn.common.models.d4`).

Covers the physics of :class:`~xnn.common.models.d4.DFTD4` (exact invariances,
charge conservation, the neutral three-body C6, sign and decay of the two
terms, smooth switching), the :class:`~xnn.common.models.d4.D4Dispersion`
wrapper around **every** registered model (edge filtering for the core,
energy bookkeeping, forces/stress, LES nesting), the ``model.extra
["dispersion"]`` config hook, the ``total_charge`` data field, batching, all
four deploy channels (eager, TorchScript whole-system and pair-style ABIs,
:class:`~xnn.common.deploy.LAMMPSWrapper`, ASE calculator) and -- when the
upstream ``dftd4`` Python package is installed -- floating-point parity of
energies, gradients, virials, coordination numbers, EEQ charges,
polarizabilities and C6 coefficients for molecules, ions and crystals.
"""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import (
    D4Dispersion,
    DFTD4,
    ForceStressOutput,
    LatentEwald,
    build_model,
    c6_matrix,
)
from xnn.common.models.d4 import BOHR, HARTREE, PBE0_D4


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


# geometries (Angstrom)

# ASE g2 geometries (the reference values below were generated on these)
WATER = (np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                   [0.0, -0.763239, -0.477047]]), [8, 1, 1])
METHANOL = (np.array([[-0.0469, 0.6640, 0.0], [-0.0469, -0.7580, 0.0],
                      [-1.0900, 0.9691, 0.0], [0.4380, 1.0800, 0.8917],
                      [0.4380, 1.0800, -0.8917], [0.8570, -1.0680, 0.0]]),
            [6, 8, 1, 1, 1, 1])
AMMONIUM = (np.array([[0.0, 0.0, 0.0], [0.59, 0.59, 0.59], [-0.59, -0.59, 0.59],
                      [-0.59, 0.59, -0.59], [0.59, -0.59, -0.59]]), [7, 1, 1, 1, 1])
# small cutoffs keep the tests fast; the defaults are exercised in the
# upstream-parity tests
# cutoff_eeq pinned to the pair cutoff: the large regime's default (16 A) would
# otherwise widen the wrapper's neighbor list beyond these test cutoffs
FAST = dict(cutoff_pair=9.0, cutoff_triple=7.0, cutoff_cn=8.0, cutoff_eeq_cn=8.0, cutoff_eeq=9.0)


def _graph(pos, z, cutoff, cell=None, charge=None, R=None, shift=0.0):
    pos = np.asarray(pos, dtype=float) + shift
    s = {"pos": pos if R is None else pos @ R.T, "atomic_numbers": z}
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        s["cell"] = cell if R is None else cell @ R.T
        s["pbc"] = [True] * 3
    if charge is not None:
        s["total_charge"] = charge
    return structure_to_graph(s, cutoff)


def _rotation(seed=3):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def _cluster(n=10, seed=0, box=6.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box - 1, (n, 3)), [1, 8, 6, 7] * (n // 4) + [1] * (n % 4)


def _energy(model, pos, z, **kw):
    return float(model(_graph(pos, z, model.cutoff, **kw))["energy"])


# DFTD4 physics
def test_defaults_are_pbe0_d4():
    d4 = DFTD4()
    for k, v in PBE0_D4.items():
        assert float(getattr(d4, k)) == v
    assert d4.ga == 3.0 and d4.gc == 2.0 and d4.wf == 6.0
    # upstream real-space cutoffs (bohr) in Angstrom
    assert abs(d4.cutoff_pair - 60 * BOHR) < 1e-12
    assert abs(d4.cutoff_triple - 40 * BOHR) < 1e-12
    assert abs(d4.cutoff_cn - 30 * BOHR) < 1e-12
    assert abs(d4.cutoff_eeq_cn - 25 * BOHR) < 1e-12
    assert d4.cutoff == d4.cutoff_pair


def test_reference_data_shape_and_sanity():
    d4 = DFTD4()
    assert d4.refalpha.shape == (23, 7, 119)
    assert int(d4.nref[6]) == 7 and int(d4.nref[1]) == 2      # carbon, hydrogen
    assert bool((d4.refalpha >= 0).all())
    # C8/C6 factors: sqrt(0.5 sqrt(Z) <r4>/<r2>) is ~2.0 for H, ~3.1 for C
    assert 1.9 < float(d4.r4r2[1]) < 2.1 and 3.0 < float(d4.r4r2[6]) < 3.2


def test_water_matches_published_reference_values():
    """Energy and properties of water against the reference implementation.

    The values were generated with ``dftd4`` 4.2.0 (PBE0-D4, atomic units)
    and are hard-coded so the check runs without the upstream package.
    """
    d4 = D4Dispersion()
    out = d4(_graph(*WATER, d4.cutoff))
    assert abs(float(out["energy"]) / HARTREE - (-1.5761743e-04)) < 1e-10
    np.testing.assert_allclose(out["coordination_numbers"].numpy(),
                               [1.60884227, 0.80442113, 0.80442113], atol=1e-7)
    np.testing.assert_allclose(out["eeq_charges"].numpy(),
                               [-0.58639069, 0.29319534, 0.29319534], atol=1e-7)
    np.testing.assert_allclose(out["polarizabilities"].numpy(),
                               [6.71824515, 1.35296894, 1.35296894], atol=1e-7)
    c6 = c6_matrix(out["dynamic_polarizabilities"]).numpy()
    assert abs(c6[0, 0] - 24.68023585) < 1e-6 and abs(c6[0, 1] - 4.2011774) < 1e-6


@pytest.mark.parametrize("periodic", [False, True])
def test_exact_invariances(periodic):
    d4 = D4Dispersion(**FAST)
    pos, z = _cluster()
    cell = np.eye(3) * 6.0 if periodic else None
    e0 = _energy(d4, pos, z, cell=cell)
    R = _rotation()
    assert abs(e0 - _energy(d4, pos, z, cell=cell, R=R)) < 1e-11
    assert abs(e0 - _energy(d4, pos, z, cell=cell, shift=1.7)) < 1e-11
    if periodic:  # shift one atom by a lattice vector
        pos2 = pos.copy()
        pos2[3] += cell[1]
        assert abs(e0 - _energy(d4, pos2, z, cell=cell)) < 1e-11
    # permutation of atoms
    perm = np.random.default_rng(1).permutation(len(z))
    assert abs(e0 - _energy(d4, pos[perm], [z[i] for i in perm], cell=cell)) < 1e-11


def test_charges_sum_to_total_charge_and_shift_polarizabilities():
    d4 = D4Dispersion(**FAST)
    neutral = d4(_graph(*AMMONIUM, d4.cutoff, charge=0.0))
    cation = d4(_graph(*AMMONIUM, d4.cutoff, charge=1.0))
    assert abs(float(neutral["eeq_charges"].sum())) < 1e-10
    assert abs(float(cation["eeq_charges"].sum()) - 1.0) < 1e-10
    # removing an electron shrinks the atoms: smaller polarizabilities, less
    # binding dispersion (paper fig 2 / sec III.B)
    assert float(cation["polarizabilities"].sum()) < float(neutral["polarizabilities"].sum())
    assert float(cation["energy"]) > float(neutral["energy"])
    # the total charge is read from the graph (default neutral)
    assert abs(float(d4(_graph(*AMMONIUM, d4.cutoff))["eeq_charges"].sum())) < 1e-10


def test_two_body_attractive_and_three_body_repulsive():
    pos, z = _cluster(12, seed=5)
    out = D4Dispersion(**FAST)(_graph(pos, z, 9.0))
    assert float(out["energy_2body"]) < 0
    assert float(out["energy_3body"]) > 0
    assert abs(float(out["energy_3body"])) < 0.1 * abs(float(out["energy_2body"]))
    # s9 = 0 switches the ATM term off exactly
    off = D4Dispersion(s9=0.0, **FAST)(_graph(pos, z, 9.0))
    assert float(off["energy_3body"]) == 0.0
    assert abs(float(off["energy"]) - float(out["energy_2body"])) < 1e-14


def test_two_body_dimer_asymptote_is_c6_over_r6():
    """Far apart, the pair energy tends to ``-s6 C6 / r^6`` (BJ damping off)."""
    d4 = DFTD4(s9=0.0)
    for r in (12.0, 16.0):
        pos = np.array([[0.0, 0, 0], [r, 0, 0]])
        out = D4Dispersion(s9=0.0)(_graph(pos, [18, 18], d4.cutoff))
        c6 = float(c6_matrix(out["dynamic_polarizabilities"])[0, 1])
        r_au = r / BOHR
        r0 = float(d4.a1 * torch.sqrt(3 * d4.r4r2[18] ** 2) + d4.a2)
        expected = -(c6 / (r_au ** 6 + r0 ** 6)
                     + 1.20065498 * 3 * float(d4.r4r2[18]) ** 2 * c6 / (r_au ** 8 + r0 ** 8))
        assert abs(float(out["energy"]) / HARTREE - expected) < 1e-14


def test_switching_window_makes_energy_continuous_at_cutoff():
    z = [18, 18]
    sharp = D4Dispersion(s9=0.0, cutoff_pair=8.0, cutoff_cn=8.0, cutoff_eeq_cn=8.0,
                         cutoff_triple=8.0)
    smooth = D4Dispersion(s9=0.0, cutoff_pair=8.0, switch_width_pair=2.0,
                          cutoff_cn=8.0, cutoff_eeq_cn=8.0, cutoff_triple=8.0)
    e_in = [_energy(m, np.array([[0.0, 0, 0], [7.999, 0, 0]]), z) for m in (sharp, smooth)]
    e_out = [_energy(m, np.array([[0.0, 0, 0], [8.001, 0, 0]]), z) for m in (sharp, smooth)]
    assert abs(e_in[0] - e_out[0]) > 1e-8          # the sharp cutoff jumps
    assert abs(e_in[1] - e_out[1]) < 1e-12         # the switched one does not
    # and the switch leaves the short range untouched
    assert abs(_energy(sharp, np.array([[0.0, 0, 0], [4.0, 0, 0]]), z)
               - _energy(smooth, np.array([[0.0, 0, 0], [4.0, 0, 0]]), z)) < 1e-14


def test_forces_match_finite_differences():
    pos, z = _cluster(8, seed=2)
    model = ForceStressOutput(D4Dispersion(**FAST))
    out = model(_graph(pos, z, 9.0))
    h = 1e-4
    for i, a in [(0, 0), (3, 2), (5, 1)]:
        pp, pm = pos.copy(), pos.copy()
        pp[i, a] += h
        pm[i, a] -= h
        fd = -(_energy(model.model, pp, z) - _energy(model.model, pm, z)) / (2 * h)
        assert abs(float(out["forces"][i, a]) - fd) < 1e-7


def test_stress_matches_finite_strain():
    pos, z = _cluster(8, seed=4)
    cell = np.array([[6.0, 0, 0], [0.8, 5.7, 0], [-0.4, 0.6, 6.2]])
    model = ForceStressOutput(D4Dispersion(**FAST), compute_stress=True)
    out = model(_graph(pos, z, 9.0, cell=cell))
    vol = abs(np.linalg.det(cell))
    for (a, b) in [(0, 0), (1, 2), (2, 2)]:
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = 1e-4
        e_p = _energy(model.model, pos @ (np.eye(3) + eps), z, cell=cell @ (np.eye(3) + eps))
        e_m = _energy(model.model, pos @ (np.eye(3) - eps), z, cell=cell @ (np.eye(3) - eps))
        fd = (e_p - e_m) / (2e-4) / vol / (1 if a == b else 2)
        assert abs(float(out["stress"][0, a, b]) - fd) < 1e-7


@pytest.mark.parametrize("periodic", [False, True])
def test_batching_matches_single_structures(periodic):
    d4 = D4Dispersion(**FAST)
    cell = np.eye(3) * 6.0 if periodic else None
    graphs = [_graph(*_cluster(10, seed=1), 9.0, cell=cell, charge=0.0),
              _graph(*_cluster(7, seed=2), 9.0, cell=cell, charge=1.0)]
    singles = [d4(g) for g in graphs]
    batch = d4(collate(graphs))
    for b in range(2):
        assert abs(float(batch["energy"][b]) - float(singles[b]["energy"])) < 1e-12
    q = torch.cat([s["eeq_charges"] for s in singles])
    assert torch.allclose(batch["eeq_charges"], q, atol=1e-12)


def test_scripted_core_matches_eager():
    d4 = DFTD4(**FAST)
    scripted = torch.jit.script(d4)
    pos, z = _cluster(10)
    g = _graph(pos, z, 9.0, cell=np.eye(3) * 6.0)
    args = (g.atomic_numbers, g.pos, g.edge_index, g.edge_vectors(), g.batch, 1,
            g.cell, g.pbc, torch.zeros(1))
    a, b = d4.evaluate(*args), scripted.evaluate(*args)
    # the eager three-body term visits each triangle once and sums its thirds
    # into the three corners, the scripted loop once per corner: rounding only
    assert torch.allclose(a["node_energy"], b["node_energy"], atol=1e-13, rtol=0)
    assert torch.equal(a["charges"], b["charges"])


def test_trainable_parameters_and_gradients():
    model = D4Dispersion(trainable=True, **FAST)
    names = {n for n, _ in model.named_parameters()}
    assert names == {"term.s6", "term.s8", "term.a1", "term.a2", "term.s9"}
    e = model(_graph(*_cluster(8), 9.0))["energy"].sum()
    e.backward()
    assert model.term.s8.grad is not None and float(model.term.s8.grad) < 0  # more s8, more binding


def test_rejects_unsupported_elements():
    with pytest.raises(ValueError, match="1..103"):
        D4Dispersion()(_graph(np.array([[0.0, 0, 0], [2.0, 0, 0]]), [104, 1], 5.0))


# wrapper around every model, config hook, LES nesting

MODEL_CONFIGS = {
    "cace": {"extra": {"species": [1, 8], "n_atom_basis": 2, "max_l": 2, "max_nu": 2}},
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


def _config(name, dispersion=FAST, **more):
    over = dict(MODEL_CONFIGS[name])
    extra = dict(over.pop("extra", {}))
    extra["dispersion"] = dispersion
    extra.update(more)
    return from_dict({"model": {"name": name, "cutoff": 4.5, "n_interactions": 1,
                               "n_rbf": over.pop("n_rbf", 6),
                               "n_features": over.pop("n_features", 8),
                               "extra": extra}})


@pytest.mark.parametrize("name", sorted(MODEL_CONFIGS))
def test_wraps_every_model(name):
    if name in ("mace", "nequip", "allegro", "cace"):
        pytest.importorskip("e3nn")
    torch.manual_seed(3)
    model = build_model(_config(name).model)
    assert isinstance(model, D4Dispersion)
    base = model.model
    assert model.cutoff == max(base.cutoff, 9.0)
    pos, z = _cluster(10)
    pos = pos[: len(z)]
    z = [1, 8] * 5
    cell = np.eye(3) * 6.0
    wrapped = ForceStressOutput(model, compute_stress=True)
    out = wrapped(_graph(pos, z, model.cutoff, cell=cell))
    # the core sees exactly the edges of its own cutoff
    e_base = float(base(_graph(pos, z, base.cutoff, cell=cell))["energy"])
    assert abs(float(out["energy_sr"]) - e_base) < 1e-10
    assert abs(float(out["energy"]) - float(out["energy_sr"]) - float(out["energy_disp"])) < 1e-12
    assert abs(float(out["node_energy"].sum()) - float(out["energy"])) < 1e-10
    assert out["forces"].shape == (10, 3) and torch.isfinite(out["forces"]).all()
    assert out["stress"].shape == (1, 3, 3)
    assert out["eeq_charges"].shape == (10,)
    assert out["node_features"].shape[0] == 10   # passed through for LES


def test_config_hook_options_and_defaults():
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.5, "n_interactions": 1,
                               "n_rbf": 6, "n_features": 8,
                               "extra": {"dispersion": {"s8": 1.5, "cutoff_pair": 12.0,
                                                        "switch_width_pair": 2.0,
                                                        "s9": 0.0}}}})
    model = build_model(cfg.model)
    assert float(model.d4.s8) == 1.5 and float(model.d4.s9) == 0.0
    assert model.d4.cutoff_pair == 12.0 and model.d4.switch_width_pair == 2.0
    assert model.cutoff == model.d4.cutoff == max(12.0, 40 * BOHR)
    # ``dispersion: true`` gives the PBE0-D4 defaults with the upstream cutoffs
    cfg.model.extra["dispersion"] = True
    model = build_model(cfg.model)
    assert float(model.d4.s8) == PBE0_D4["s8"] and abs(model.cutoff - 60 * BOHR) < 1e-12
    # standalone model from config
    alone = build_model(from_dict({"model": {"name": "d4", "extra": {"s9": 0.0}}}).model)
    assert isinstance(alone, D4Dispersion) and alone.model is None
    assert alone.node_feature_dim == 3


def test_nests_with_les_in_both_orders():
    pytest.importorskip("e3nn")
    torch.manual_seed(0)
    both = build_model(_config("mace", long_range={"n_channels": 2, "dl": 3.0}).model)
    assert isinstance(both, LatentEwald) and isinstance(both.model, D4Dispersion)
    pos, z = _cluster(10)
    z = [1, 8] * 5
    out = ForceStressOutput(both)(_graph(pos, z, both.cutoff, cell=np.eye(3) * 6.0))
    assert abs(float(out["energy"]) - float(out["energy_sr"]) - float(out["energy_lr"])) < 1e-12
    assert "energy_disp" in out and torch.isfinite(out["forces"]).all()
    base = build_model(from_dict({"model": {"name": "schnet", "cutoff": 4.5,
                                            "n_interactions": 1, "n_rbf": 6,
                                            "n_features": 8}}).model)
    other = D4Dispersion(LatentEwald(base, n_channels=2, dl=3.0), **FAST)
    out2 = ForceStressOutput(other)(_graph(pos, z, other.cutoff, cell=np.eye(3) * 6.0))
    assert torch.isfinite(out2["energy"]).all()
    # LES on a standalone D4 uses the (CN, q, alpha) features
    les_d4 = LatentEwald(D4Dispersion(**FAST), n_channels=1)
    assert torch.isfinite(les_d4(_graph(pos, z, 9.0))["energy"]).all()


def test_total_charge_flows_through_the_data_layer():
    pytest.importorskip("ase")
    from ase import Atoms
    from xnn.common.data.ase_io import atoms_to_structure
    atoms = Atoms("NH4", positions=AMMONIUM[0])
    atoms.info["charge"] = 1.0
    s = atoms_to_structure(atoms)
    assert s["total_charge"] == 1.0
    g = structure_to_graph(s, 9.0)
    assert float(g.total_charge[0]) == 1.0
    batch = collate([g, structure_to_graph({"pos": WATER[0], "atomic_numbers": WATER[1],
                                            "charge": -1.0}, 9.0)])
    assert batch.total_charge.tolist() == [1.0, -1.0]
    # dropped when not every structure carries one
    assert collate([g, structure_to_graph({"pos": WATER[0], "atomic_numbers": WATER[1]}, 9.0)]
                   ).total_charge is None


# deploy channels
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("kind", ["standalone", "schnet"])
def test_torchscript_export_matches_eager(kind, periodic):
    from xnn.common.data import build_neighbor_list
    from xnn.common.deploy import LAMMPSWrapper, TorchScriptPotential
    torch.manual_seed(0)
    if kind == "standalone":
        model = D4Dispersion(**FAST)
    else:
        model = build_model(_config("schnet").model)
    pos, z = _cluster(10)
    z = [1, 8] * 5
    cell = np.eye(3) * 6.0 if periodic else None
    ref = ForceStressOutput(model, compute_stress=True)(_graph(pos, z, model.cutoff, cell=cell))
    scripted = torch.jit.script(TorchScriptPotential(model, model.cutoff).eval())
    cell_t = torch.tensor(cell) if periodic else None
    pbc_t = torch.tensor([periodic] * 3)
    out = scripted(torch.tensor(pos), torch.tensor(z), cell_t, pbc_t)
    assert abs(float(out["energy"]) - float(ref["energy"])) < 1e-12
    assert torch.allclose(out["forces"], ref["forces"], atol=1e-12)
    assert out["eeq_charges"].shape == (10, 1)
    assert abs(float(out["energy_disp"]) - float(ref["energy_disp"])) < 1e-12
    if periodic:
        assert torch.allclose(out["stress"], ref["stress"][0], atol=1e-12)
    # pair-style ABI and the LAMMPS wrapper on the same neighbor list
    ei, cs = build_neighbor_list(torch.tensor(pos), model.cutoff, cell_t, pbc_t if periodic else None)
    cell_l = cell_t if periodic else torch.zeros(3, 3)
    out_l = scripted.forward_lammps(torch.tensor(pos), ei, cs, torch.tensor(z), cell_l)
    assert abs(float(out_l["energy"]) - float(ref["energy"])) < 1e-12
    lammps = torch.jit.script(LAMMPSWrapper(model, model.cutoff).eval())
    out_w = lammps(torch.tensor(pos), ei, cs, torch.tensor(z), cell_l)
    assert abs(float(out_w["total_energy"]) - float(ref["energy"])) < 1e-12
    assert torch.allclose(out_w["forces"], ref["forces"], atol=1e-12)


def test_export_records_dispersion_metadata(tmp_path):
    from xnn.common.deploy import export_torchscript_potential
    model = D4Dispersion(**FAST)
    path = export_torchscript_potential(model, model.cutoff, str(tmp_path / "d4.pt"),
                                        total_charge=1.0)
    extra = {"cutoff": "", "dispersion": "", "long_range": "", "total_charge": ""}
    loaded = torch.jit.load(path, _extra_files=extra)
    assert extra["dispersion"] == b"True" and extra["long_range"] == b"False"
    assert extra["total_charge"] == b"1.0"
    out = loaded(torch.tensor(AMMONIUM[0]), torch.tensor(AMMONIUM[1]))
    assert abs(float(out["eeq_charges"].sum()) - 1.0) < 1e-10


def test_ase_calculator_matches_eager():
    pytest.importorskip("ase")
    from ase import Atoms
    from xnn.common.deploy import XNNCalculator
    model = D4Dispersion(**FAST)
    pos, z = _cluster(9, seed=7)
    atoms = Atoms(numbers=z, positions=pos, cell=np.eye(3) * 6.0, pbc=True)
    atoms.calc = XNNCalculator(ForceStressOutput(model, compute_stress=True), cutoff=model.cutoff)
    ref = ForceStressOutput(model, compute_stress=True)(_graph(pos, z, model.cutoff, cell=np.eye(3) * 6.0))
    assert abs(atoms.get_potential_energy() - float(ref["energy"])) < 1e-10
    assert np.abs(atoms.get_forces() - ref["forces"].detach().numpy()).max() < 1e-10
    s = ref["stress"][0].detach().numpy()
    assert np.abs(atoms.get_stress() - [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]).max() < 1e-10


def test_trainer_widens_dataset_cutoff_to_the_wrapper(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer
    frames = []
    for seed in range(4):
        pos, z = _cluster(8, seed=seed)
        a = Atoms(numbers=z, positions=pos)
        a.info["energy"] = float(seed)
        a.arrays["forces"] = np.zeros((8, 3))
        frames.append(a)
    path = tmp_path / "train.extxyz"
    write(str(path), frames)
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.5, "n_interactions": 1,
                               "n_rbf": 6, "n_features": 8,
                               "extra": {"dispersion": FAST}},
                     "data": {"train_path": str(path), "batch_size": 2, "val_fraction": 0.25},
                     "optim": {"epochs": 1}, "output_dir": str(tmp_path / "run"),
                     "device": "cpu"})
    train = AtomicDataset.from_file(str(path), cfg.data.cutoff)
    assert train.cutoff == 4.5
    trainer = Trainer(cfg, train)
    assert train.cutoff == 9.0                      # widened before any graph was built
    assert trainer.model.model.cutoff == 9.0


# parity with the upstream dftd4 package
def _upstream(pos, z, charge=0.0, cell=None, params=None):
    """Reference D4 results from the ``dftd4`` Python package, in atomic units.

    The dftd4 wheel bundles its own OpenMP runtime, which computes garbage
    EEQ charges once torch's thread pool is active in the same process; a
    single torch thread while it runs avoids the clash, and the water check
    below guards against it.
    """
    from dftd4.interface import DampingParam, DispersionModel
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = DispersionModel(np.asarray(z), np.asarray(pos) / BOHR, charge=charge,
                                lattice=None if cell is None else np.asarray(cell) / BOHR,
                                periodic=None if cell is None else np.array([True] * 3))
        param = DampingParam(**(params or PBE0_D4))
        res = model.get_dispersion(param, grad=True)
        props = model.get_properties()
        # environment guard: the water molecule's known charges
        probe = DispersionModel(np.array(WATER[1]), WATER[0] / BOHR).get_properties()
        assert abs(probe["partial charges"][0] + 0.58639069) < 1e-6, \
            "dftd4 returned wrong EEQ charges (OpenMP clash with torch)"
    finally:
        torch.set_num_threads(n_threads)
    return res, props


def _xnn(pos, z, charge=0.0, cell=None, **opts):
    model = D4Dispersion(**opts)
    g = _graph(pos, z, model.cutoff, cell=cell, charge=charge)
    out = ForceStressOutput(model, compute_stress=cell is not None)(g)
    return {k: v.detach() for k, v in out.items()}


def _compare(pos, z, charge=0.0, cell=None, tol_e=1e-13, **opts):
    res, props = _upstream(pos, z, charge, cell)
    out = _xnn(pos, z, charge, cell, **opts)
    energy = float(out["energy"]) / HARTREE
    assert abs(energy - float(res["energy"])) < tol_e * max(1.0, abs(energy) * 1e3)
    grad = -out["forces"].numpy() / HARTREE * BOHR
    assert np.abs(grad - res["gradient"]).max() < 1e-12
    np.testing.assert_allclose(out["coordination_numbers"].numpy(),
                               props["coordination numbers"], atol=1e-9)
    np.testing.assert_allclose(out["eeq_charges"].numpy(), props["partial charges"], atol=1e-10)
    # the covalent radii enter in a different floating-point order (Angstrom
    # -> bohr conversion), which the Gaussian CN weights amplify to ~1e-9
    np.testing.assert_allclose(out["polarizabilities"].numpy(), props["polarizabilities"],
                               rtol=1e-8)
    np.testing.assert_allclose(c6_matrix(out["dynamic_polarizabilities"]).numpy(),
                               props["c6 coefficients"], rtol=1e-7)
    if cell is not None:
        vol = abs(np.linalg.det(np.asarray(cell))) / BOHR ** 3
        # upstream reports the strain derivative dE/d(eps); xnn's stress is
        # (1/V) dE/d(eps)
        virial = out["stress"][0].numpy() / HARTREE * BOHR ** 3 * vol
        assert np.abs(virial - res["virial"]).max() < 1e-11


def test_parity_vs_dftd4_molecules():
    pytest.importorskip("dftd4")
    _compare(*WATER)
    _compare(*METHANOL)
    _compare(*AMMONIUM, charge=1.0)
    _compare(*AMMONIUM, charge=-1.0)
    pos, z = _cluster(12, seed=9)
    _compare(pos, z)


def test_parity_vs_dftd4_crystals():
    pytest.importorskip("dftd4")
    # rocksalt NaCl and a sheared silicon cell (2 atoms each)
    a = 5.64
    _compare(np.array([[0.0, 0, 0], [a / 2, 0, 0]]), [11, 17],
             cell=[[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    a = 5.43
    cell = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    cell[1, 0] += 0.5
    _compare(np.array([[0.0, 0, 0], [a / 4, a / 4, a / 4]]), [14, 14], cell=cell)
    # a small periodic water box (intact molecules: an isolated atom would
    # expose upstream's rounded CN cap, see test_parity_vs_dftd4_reduced_cutoffs)
    pos = np.concatenate([WATER[0] + shift for shift in
                          ([0.0, 0.0, 0.0], [3.1, 0.4, 0.2], [0.3, 3.0, 3.2])])
    _compare(pos, [8, 1, 1] * 3, cell=np.eye(3) * 6.2)


def test_parity_vs_dftd4_reduced_cutoffs():
    """Upstream and xnn agree with matching non-default real-space cutoffs."""
    pytest.importorskip("dftd4")
    from dftd4.interface import DampingParam, DispersionModel
    # a compact cluster: every atom keeps neighbors inside ~2 covalent radii.
    # For an *isolated* atom upstream's capped EEQ coordination number is
    # 1.8e-15 rather than 0 (the two log terms of the cap round differently),
    # which the 1e-14 regularizer of the EEQ right-hand side amplifies to
    # ~1e-9 in the charges; xnn evaluates the cap exactly and gets 0.
    pos, z = _cluster(12, seed=3)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = DispersionModel(np.asarray(z), pos / BOHR)
        model.set_realspace_cutoff(disp2=12.0, disp3=9.0, cn=10.0)
        e_ref = float(model.get_dispersion(DampingParam(**PBE0_D4), grad=False)["energy"])
    finally:
        torch.set_num_threads(n_threads)
    out = _xnn(pos, z, cutoff_pair=12.0 * BOHR, cutoff_triple=9.0 * BOHR,
               cutoff_cn=10.0 * BOHR)
    assert abs(float(out["energy"]) / HARTREE - e_ref) < 1e-13
