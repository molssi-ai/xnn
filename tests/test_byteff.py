"""ByteFF and ByteFF-Pol: molecular graphs, parameter heads, energies and forces."""
import logging
import math
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.gnn.featurizers.molgraph import BOND_ORDERS, molecular_graph  # noqa: E402
from xnn.hybnn.models import ByteFF, ByteFFPol  # noqa: E402

GRAPH = dict(atom_dim=8, feature_dim=4, node_mlp=(16, 16, 2), edge_mlp=(16, 16, 2), hidden=16, layers=2, heads=2,
             head_dim=4, ffn=(16, 2), out_dim=16)
# sigma and epsilon of an untrained head stay in (0, 3): energies of a few eV
SMALL = dict(graph=GRAPH, bonded=dict(pre=(16, 16, 2), out=(16, 2)),
             nonbonded=dict(pre=(16, 16, 2), out=(16, 2), tanh_output=1.5))
SMALL_POL = dict(graph=GRAPH, bonded=dict(pre=(16, 16, 2), out=(16, 2)), nonbonded=dict(pre=(16, 16, 2), out=(16, 2)))


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _coords(mol, seed=3, n=1):
    if mol.GetNumAtoms() == 1:
        return np.zeros((n, 1, 3))
    m = Chem.Mol(mol)
    ids = list(AllChem.EmbedMultipleConfs(m, n, randomSeed=seed))
    return np.stack([m.GetConformer(i).GetPositions() for i in ids])


def _graphs(z, coords, model):
    return collate([structure_to_graph({"pos": torch.as_tensor(c, dtype=torch.float64),
                                        "atomic_numbers": torch.as_tensor(z)}, cutoff=model.cutoff)
                    for c in coords])


def _z(mols):
    return [a.GetAtomicNum() for m in mols for a in m.GetAtoms()]


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_graph_features_and_terms():
    """Butane's terms, shells and equivalence classes; benzene's ring and aromatic features."""
    g = molecular_graph(_mol("CCCC"))
    assert (g.bonds.shape[0], g.angles.shape[0], g.propers.shape[0], g.impropers.shape[0]) == (13, 24, 27, 0)
    assert g.pairs14.shape[0] == 27 and g.pairs15.shape[0] == 18
    assert len(set(g.atom_class.tolist())) == 4                    # two carbon and two hydrogen classes
    assert np.all(g.angles[:, 0] < g.angles[:, 2])
    b = molecular_graph(_mol("c1ccccc1"))
    ring = b.atomic_numbers == 6
    assert np.all(b.min_ring[ring] == 6) and np.all(b.ring_degree[ring] == 2) and np.all(b.min_ring[~ring] == 0)
    aromatic = BOND_ORDERS.index(1.5)
    assert sorted(b.bond_order.tolist()).count(aromatic) == 6
    assert b.impropers.shape[0] == 6 and np.all(b.atomic_numbers[b.impropers[:, 0]] == 6)


def test_graph_standardization():
    """Sulfoxides become charge-separated; a 9-ring is not a ring; nitriles have linear torsions."""
    g = molecular_graph(_mol("CS(C)=O"))
    s = int(np.flatnonzero(g.atomic_numbers == 16)[0])
    o = int(np.flatnonzero(g.atomic_numbers == 8)[0])
    assert g.formal_charges[s] == 1 and g.formal_charges[o] == -1
    nine = molecular_graph(_mol("C1CCCCCCCC1"))
    assert not nine.bond_in_ring.any() and not nine.min_ring.any()
    eight = molecular_graph(_mol("C1CCCCCCC1"))
    assert np.all(eight.min_ring[eight.atomic_numbers == 6] == 8)
    nitrile = molecular_graph(_mol("CCC#N"))
    assert nitrile.proper_linear.any() and not nitrile.proper_linear.all()


def _byteff(structure=None, **kw):
    torch.manual_seed(0)
    return ByteFF(structure, **SMALL, **kw)


def test_byteff_forces_are_gradients():
    """Forces equal central finite differences of the energy."""
    mol = _mol("CC(=O)NC")
    model = _byteff(mol)
    pos = _coords(mol)[0]
    z = _z([mol])
    out = ForceStressOutput(model)(_graphs(z, [pos], model))
    h, i = 1e-5, 4
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[i, 1] += s * h
        e.append(float(model(_graphs(z, [p], model))["energy"][0]))
    f = float(out["forces"][i, 1])
    assert abs(-(e[0] - e[1]) / (2 * h) - f) < 1e-6 * max(1.0, abs(f))
    terms = ("e_bond", "e_angle", "e_torsion", "e_improper", "e_vdw", "e_coulomb", "e_vdw14", "e_coulomb14")
    assert torch.allclose(sum(out[k] for k in terms), out["energy"], rtol=1e-12, atol=1e-12)


def test_byteff_charges_symmetry_and_batch():
    """Charges sum to the formal charge, equivalent atoms share parameters, batches equal single runs."""
    mol = _mol("CC(=O)[O-]")
    model = _byteff(mol).eval()
    P = model.predicted_parameters()
    assert math.isclose(float(P["charge"].sum()), -1.0, abs_tol=1e-12)
    g = molecular_graph(mol)
    for cls in set(g.atom_class.tolist()):
        members = np.flatnonzero(g.atom_class == cls)
        for key in ("charge", "sigma", "epsilon"):
            assert torch.allclose(P[key][members], P[key][members[0]].expand(len(members)), atol=1e-12)
    coords = _coords(mol, n=2)
    z = _z([mol])
    batch = model(_graphs(z, coords, model))["energy"]
    single = torch.cat([model(_graphs(z, [c], model))["energy"] for c in coords])
    assert torch.allclose(batch, single, rtol=1e-12, atol=1e-12)


def test_ensemble_averages_parameters():
    """An ensemble's parameters are the mean of its members' head outputs."""
    model = _byteff(_mol("CCO"), members=2).eval()
    raw = model.predicted_parameters(per_member=True)
    P = model.predicted_parameters()
    assert torch.allclose(P["sigma"], raw["sigma"].mean(0))
    assert torch.allclose(P["bond_k"], raw["bond_k"].mean(0).squeeze(-1))


def test_parameters_track_weights():
    """Cached parameters are recomputed after the weights change."""
    model = _byteff(_mol("CCO")).eval()
    mol = _mol("CCO")
    z, pos = _z([mol]), _coords(mol)
    e0 = model(_graphs(z, pos, model))["energy"]
    with torch.no_grad():
        model.members[0].nonbonded.sigma.layers[-1].bias.add_(0.5)
    assert not torch.allclose(model(_graphs(z, pos, model))["energy"], e0)


def test_byteff_cutoff_periodic():
    """With a cutoff and a cell, the energy is finite and forces match finite differences."""
    mols = [_mol("O"), _mol("O")]
    model = _byteff(mols, cutoff=6.0, switch_width=1.0)
    pos = np.concatenate([_coords(mols[0])[0], _coords(mols[1])[0] + 3.0])
    z = _z(mols)

    def graph(p):
        return structure_to_graph({"pos": torch.as_tensor(p), "atomic_numbers": torch.as_tensor(z),
                                   "cell": torch.eye(3) * 12.0, "pbc": [True] * 3}, cutoff=6.0)

    f = ForceStressOutput(model)(graph(pos))["forces"]
    h = 1e-5
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[3, 0] += s * h
        e.append(float(model(graph(p))["energy"][0]))
    assert abs(-(e[0] - e[1]) / (2 * h) - float(f[3, 0])) < 1e-6 * max(1.0, abs(float(f[3, 0])))


def test_byteff_pol_cluster():
    """ByteFF-Pol on a lithium cluster: finite terms, attractive polarization, conservative forces."""
    mols = [_mol("COC(=O)OC"), _mol("[Li+]")]
    torch.manual_seed(1)
    model = ByteFFPol(mols, **SMALL_POL)
    pos = np.concatenate([_coords(mols[0])[0], np.array([[0.0, 0.0, 3.0]])])
    z = _z(mols)
    out = ForceStressOutput(model)(_graphs(z, [pos], model))
    assert float(out["e_polarization"][0]) < 0.0
    assert out["induced_dipoles"].shape == (len(z), 3)
    li = len(z) - 1
    h = 1e-5
    e = []
    for s in (1, -1):
        p = pos.copy()
        p[li, 2] += s * h
        e.append(float(model(_graphs(z, [p], model))["energy"][0]))
    f = float(out["forces"][li, 2])
    assert abs(-(e[0] - e[1]) / (2 * h) - f) < 1e-6 * max(1.0, abs(f))
    shifted = model(_graphs(z, [pos + 1.7], model))["energy"]
    assert torch.allclose(shifted, out["energy"], rtol=1e-12)


def test_float32_matches_float64():
    """The float32 model agrees with float64 to float32 round-off."""
    mol = _mol("c1ccccc1O")
    m64 = _byteff(mol).eval()
    pos, z = _coords(mol), _z([mol])
    e64 = m64(_graphs(z, pos, m64))["energy"]
    m32 = _byteff(mol).float().eval()
    torch.set_default_dtype(torch.float32)
    g = collate([structure_to_graph({"pos": torch.as_tensor(pos[0], dtype=torch.float32),
                                     "atomic_numbers": torch.as_tensor(z)}, cutoff=m32.cutoff)])
    e32 = m32(g)["energy"]
    assert abs(float(e32[0]) - float(e64[0])) < 1e-5 * max(1.0, abs(float(e64[0])))


def test_unsupported_element_and_unbound():
    """Unbound models and unsupported elements raise in evaluation."""
    model = _byteff().eval()
    mol = _mol("CCO")
    with pytest.raises(RuntimeError):
        model(_graphs(_z([mol]), _coords(mol), model))
    model.bind(_mol("[SiH4]"))
    si = _mol("[SiH4]")
    with pytest.raises(ValueError):
        model(_graphs(_z([si]), _coords(si), model))


def test_registry_and_config():
    """Both models are registered and build from a config."""
    assert {"byteff", "byteff_pol"} <= set(available_models())
    cfg = from_dict({"model": {"name": "byteff", "cutoff": 5.0, "extra": {**SMALL, "structure": "CCO"}}})
    model = build_model(cfg.model)
    assert isinstance(model, ByteFF) and model.n_atoms == 9


def test_cards_and_license_notice(caplog):
    """The released networks are registered under their non-commercial license, which is stated on load."""
    from xnn.common.models.hub.loader import _license_notice, _noticed
    from xnn.common.models.hub.registry import get_card
    card = get_card("byteff-pol-25")
    assert card.license == "CC-BY-NC-4.0" and card.format == "byteff2"
    _noticed.discard(card.name)
    with caplog.at_level(logging.WARNING):
        _license_notice(card)
    assert "no commercial use" in caplog.text


def _upstream():
    pytest.importorskip("torch_geometric")
    hybrid = pytest.importorskip("byteff2.model.hybrid_ff")
    data = pytest.importorskip("byteff2.data")
    return hybrid, data


def _upstream_config(polarizable):
    feat = dict(act="gelu", atom_embedding_dim=8, node_mlp_dims=[16, 16, 2], edge_mlp_dims=[16, 16, 2])
    gnn = dict(act="gelu", at_channels=4, ffn_dims=[16, 2], gnn_dims=[16, 16, 2], gnn_type="EGT", heads=2, jk="cat")
    bonded = dict(act="gelu", out_mlp_dims=[16, 2], post_mlp_dims=[16, 16, 1], pre_mlp_dims=[16, 16, 2],
                  tanh_output=20.0, type="MMBondedConj" if polarizable else "MMBonded")
    if polarizable:
        ff = [bonded, dict(act="gelu", out_mlp_dims=[16, 2], pre_mlp_dims=[16, 16, 2], type="ChargeVolume"),
              dict(act="gelu", charge14=0.5, disp_damping_factor=120.0, fix_li_alpha=1e-6, ind14=0.5,
                   li_damp_clip=0.2, out_mlp_dims=[16, 2], pol_damping_factor=0.39, pre_mlp_dims=[16, 16, 2],
                   s12=1.5, type="Exp6Pol", vdw14=0.5)]
        elements = [1, 3, 6, 7, 8, 9, 15, 16, 17]
    else:
        ff = [bonded, dict(act="gelu", out_mlp_dims=[16, 2], pre_mlp_dims=[16, 16, 2], tanh_output=20.0,
                           type="LJEs")]
        elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    return dict(graph_block=dict(feature_layer=feat, gnn_layer=gnn), ff_block=ff, supported_elements=elements)


def _mapped(mol):
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToSmiles(m)


@pytest.mark.parametrize("polarizable", [False, True])
def test_fidelity_vs_byteff2(tmp_path, polarizable):
    """Transplanted weights give the reference package's parameters, energies and forces to round-off."""
    hybrid, data = _upstream()
    from xnn.ffnn.models.oplslib import KCAL_TO_EV
    from xnn.hybnn.models.byteff_foundation import from_byteff2
    torch.manual_seed(5)
    up = hybrid.HybridFF(**_upstream_config(polarizable)).double().eval()
    torch.save({"model_state_dict": up.state_dict()}, tmp_path / "optimal.pt")
    model = from_byteff2(tmp_path / "optimal.pt", dtype=torch.float64)
    smiles = ["CCOC(=O)OC", "O"] if polarizable else ["CC(=O)Nc1ccc(O)cc1"]
    mols = [_mol(s) for s in smiles]
    coords = np.concatenate([_coords(m, seed=7)[0] + 3.5 * k for k, m in enumerate(mols)])[None]
    if polarizable:
        from byteff2.model.ff_layers import mm_tspol
        orig = mm_tspol.PolarizationSolver.compute_interaction_tensor

        def no_self(self, *args):
            T = orig(self, *args)
            n = T.shape[-1] // 3
            return T * (1.0 - torch.kron(torch.eye(n, dtype=T.dtype), torch.ones(3, 3, dtype=T.dtype)))
        mm_tspol.PolarizationSolver.compute_interaction_tensor = no_self
        try:
            batch = data.ClusterData("c", [_mapped(m) for m in mols], confdata={"coords": coords}, max_n_confs=1,
                                     float_dtype=torch.float64)
            preds = up(batch, cluster=True)
        finally:
            mm_tspol.PolarizationSolver.compute_interaction_tensor = orig
        e_ref, f_ref = preds["energy_cluster"], preds["forces_cluster"]
    else:
        batch = data.MonoData("x", _mapped(mols[0]), confdata={"coords": coords}, max_n_confs=1,
                              float_dtype=torch.float64)
        preds = up(batch)
        e_ref, f_ref = preds["energy"], preds["forces"]
    model.bind(mols)
    out = ForceStressOutput(model)(_graphs(_z(mols), coords, model))
    e = out["energy"] / KCAL_TO_EV
    f = out["forces"] / KCAL_TO_EV
    assert torch.allclose(e, e_ref.reshape(-1).detach(), rtol=1e-10, atol=1e-9)
    assert torch.allclose(f, f_ref[:, 0].detach(), rtol=1e-9, atol=1e-8)


def test_hypervalent_terms():
    """Bonds at hypervalent centers get fixed force constants and reference-geometry equilibrium values."""
    mol = _mol("F[P-](F)(F)(F)(F)F")
    with pytest.warns(UserWarning, match="hypervalent"):
        model = _byteff(mol)
    P = model.predicted_parameters()
    assert torch.all(P["bond_k"] == 1000.0) and torch.all(P["bond_x0"] == 0.0)
    pos = _coords(mol)[0]
    model.bind(mol, reference_positions=pos)
    P = model.predicted_parameters()
    r = np.linalg.norm(pos[model._bonds[1].numpy()] - pos[model._bonds[0].numpy()], axis=1)
    assert np.allclose(P["bond_x0"].detach().numpy(), r, atol=1e-12)
    assert torch.all(P["angle_k"] == 278.44)


def test_lone_ion():
    """A molecule without bonds (Li+) has its formal charge and no energy alone."""
    torch.manual_seed(2)
    model = ByteFFPol(_mol("[Li+]"), **SMALL_POL).eval()
    assert math.isclose(float(model.predicted_parameters()["charge"][0]), 1.0, abs_tol=1e-12)
    out = model(_graphs([3], np.zeros((1, 1, 3)), model))
    assert float(out["energy"][0]) == 0.0



@pytest.mark.parametrize("cutoff", [6.0])
def test_kernels_match_reference(cutoff, monkeypatch):
    """The fused kernels give the reference's terms, forces and training gradients."""
    from xnn.common import cpu_kernels
    if not cpu_kernels.available():
        pytest.skip("no C++ compiler or Triton")
    import xnn.ffnn.models.classical_fast as cf
    monkeypatch.setattr(cf, "fast_supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    mols = [_mol("CC(=O)Nc1ccc(O)cc1"), _mol("O")]
    model = _byteff(mols, cutoff=cutoff, switch_width=1.0 if cutoff else 0.0)
    pos = np.concatenate([_coords(mols[0])[0], _coords(mols[1])[0] + 4.0])
    # off planarity: at planar centers the improper's parameter derivative 1 - cos(2 phi) is a cancellation
    pos = pos + np.random.default_rng(0).normal(0.0, 0.03, pos.shape)
    s = {"pos": torch.as_tensor(pos), "atomic_numbers": torch.as_tensor(_z(mols))}
    if cutoff:
        s.update(cell=torch.eye(3) * 16.0, pbc=[True] * 3)
    results = []
    for fast in (False, True):
        model.use_fast = fast
        model.eval()
        out = ForceStressOutput(model)(structure_to_graph(dict(s), cutoff=model.cutoff))
        model.train()
        model.zero_grad()
        tr = ForceStressOutput(model).train()(structure_to_graph(dict(s), cutoff=model.cutoff))
        (tr["energy"].pow(2).sum() + tr["forces"].pow(2).sum()).backward()
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        results.append((out, grads))
    (ref, g_ref), (got, g_got) = results
    for key in ("energy", "forces", "e_bond", "e_angle", "e_torsion", "e_improper", "e_vdw", "e_coulomb",
                "e_vdw14", "e_coulomb14"):
        assert torch.allclose(got[key], ref[key], rtol=1e-10, atol=1e-10), key
    assert len(g_got) == len(g_ref) > 0
    for a, b in zip(g_got, g_ref):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-10)


def test_linear_torsions_at_collinearity():
    """Exactly collinear nitrile atoms give finite energies and forces."""
    mol = _mol("CC#N")
    model = _byteff(mol)
    pos = _coords(mol)[0]
    c, n = pos[1], pos[2]
    axis = (n - c) / np.linalg.norm(n - c)
    pos[2] = c + 1.16 * axis
    pos[0] = c - 1.46 * axis
    out = ForceStressOutput(model)(_graphs(_z([mol]), [pos], model))
    assert torch.isfinite(out["energy"]).all() and torch.isfinite(out["forces"]).all()



@pytest.mark.parametrize("name", ["byteff", "byteff_pol"])
def test_mdi_engine_serves_a_bound_model(tmp_path, name):
    """An MDI engine built from a checkpoint whose config binds water gives the model's energy."""
    from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine
    extra = {**(SMALL if name == "byteff" else SMALL_POL), "structure": "O"}
    cfg = from_dict({"model": {"name": name, "cutoff": 0.1, "extra": extra}})
    torch.manual_seed(0)
    model = ForceStressOutput(build_model(cfg.model)).to(torch.float64)
    path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, path)
    engine = MDIEngine.from_checkpoint(str(path))
    pos = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])
    engine.natoms = 3
    engine.atomic_numbers = np.array([8, 1, 1])
    engine.coords_bohr = pos / BOHR_TO_ANGSTROM
    engine.calculate()
    direct = model.eval()(_graphs([8, 1, 1], pos[None], model.model))["energy"]
    assert engine.energy * HARTREE_TO_EV == pytest.approx(float(direct[0]), rel=1e-10, abs=1e-12)


def test_pol_compiled_block_matches_reference(monkeypatch):
    """The compiled ByteFF-Pol energy block gives the eager block's energies, forces and training gradients."""
    if sys.version_info >= (3, 13):
        pytest.skip("torch.compile needs Python < 3.13")
    import xnn.common.models.compiled as cm
    monkeypatch.setattr(cm, "supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    mols = [_mol("O"), _mol("[Li+]")]
    torch.manual_seed(1)
    model = ByteFFPol(mols, **SMALL_POL)
    pos = np.concatenate([_coords(mols[0])[0], np.array([[0.0, 0.0, 2.2]])])
    pos = pos + np.random.default_rng(0).normal(0.0, 0.03, pos.shape)
    z = _z(mols)
    results = []
    for fast in (False, True):
        model.set_use_fast(fast)
        model.eval()
        out = ForceStressOutput(model)(_graphs(z, [pos], model))
        model.train()
        model.zero_grad()
        tr = ForceStressOutput(model).train()(_graphs(z, [pos], model))
        (tr["energy"].pow(2).sum() + tr["forces"].pow(2).sum()).backward()
        results.append((out, [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    (ref, g_ref), (got, g_got) = results
    for key in ("energy", "forces", "induced_dipoles", "e_polarization", "e_repulsion", "e_charge_transfer"):
        assert torch.allclose(got[key], ref[key], rtol=1e-10, atol=1e-12), key
    assert len(g_got) == len(g_ref) > 0
    for a, b in zip(g_got, g_ref):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-11)


def test_lammps_input_of_a_bound_molecule(tmp_path):
    """The LAMMPS inputs carry the bound system's terms with ByteFF's styles."""
    from xnn.ffnn.models import lammps_input, write_lammps_input
    mol = _mol("CC(=O)Nc1ccc(O)cc1")
    model = _byteff(mol).eval()
    pos = _coords(mol)[0]
    data, cmds, offset, files = lammps_input(model, pos)
    g = molecular_graph(mol)
    assert f"{g.n_atoms} atoms" in data and f"{g.bonds.shape[0]} bonds" in data
    assert f"{(~g.proper_linear).sum()} dihedrals" in data and f"{g.impropers.shape[0]} impropers" in data
    for style in ("dihedral_style fourier", "improper_style cvff", "pair_modify mix arithmetic", "boundary m m m"):
        assert style in cmds
    assert offset == 0.0 and files == {}
    assert write_lammps_input(model, pos, None, tmp_path, "mol").is_file()
    with pytest.raises(ValueError):
        lammps_input(_byteff(mol, cutoff=6.0, switch_width=1.0), pos, np.eye(3) * 20.0)


def test_byteff_compiled_terms_match_reference(monkeypatch):
    """Without a cutoff, the compiled terms block gives the eager block's energies, forces and training gradients."""
    if sys.version_info >= (3, 13):
        pytest.skip("torch.compile needs Python < 3.13")
    import xnn.common.models.compiled as cm
    monkeypatch.setattr(cm, "supported", lambda device, dtype: dtype in (torch.float32, torch.float64))
    mols = [_mol("CC(=O)N"), _mol("O")]
    model = _byteff(mols)
    pos = np.concatenate([_coords(mols[0])[0], _coords(mols[1])[0] + 3.5])
    pos = pos + np.random.default_rng(0).normal(0.0, 0.03, pos.shape)
    z = _z(mols)
    results = []
    for fast in (False, True):
        model.set_use_fast(fast)
        model.eval()
        out = ForceStressOutput(model)(_graphs(z, [pos], model))
        model.train()
        model.zero_grad()
        tr = ForceStressOutput(model).train()(_graphs(z, [pos], model))
        (tr["energy"].pow(2).sum() + tr["forces"].pow(2).sum()).backward()
        results.append((out, [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    (ref, g_ref), (got, g_got) = results
    for key in ("energy", "forces", "e_torsion", "e_vdw", "e_coulomb14"):
        assert torch.allclose(got[key], ref[key], rtol=1e-10, atol=1e-12), key
    assert len(g_got) == len(g_ref) > 0
    for a, b in zip(g_got, g_ref):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-11)



def _pol_box(seed=0, **kw):
    """Six waters, Li+ and Cl- on a 2 x 2 x 2 grid in a cubic cell of 9 Angstrom, cutoff 4.4."""
    mols = [_mol("O") for _ in range(6)] + [_mol("[Li+]"), _mol("[Cl-]")]
    rng = np.random.default_rng(seed)
    water = _coords(mols[0])[0]
    water = water - water.mean(0)
    pos = []
    for k, m in enumerate(mols):
        c = water if m.GetNumAtoms() == 3 else np.zeros((1, 3))
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        pos.append(c @ q.T + 4.5 * (np.array([k % 2, (k // 2) % 2, k // 4]) + 0.5) + 0.3 * rng.standard_normal(3))
    torch.manual_seed(2)
    options = {**dict(cutoff=4.4, polar_eps=1e-11, max_iter=500), **kw}
    model = ByteFFPol(mols, **SMALL_POL, **options)
    return model, np.concatenate(pos), _z(mols), 9.0


def _box_graph(model, pos, z, cell):
    cell = np.eye(3) * cell if np.ndim(cell) == 0 else cell
    return structure_to_graph({"pos": torch.as_tensor(pos), "atomic_numbers": torch.as_tensor(z),
                               "cell": torch.as_tensor(cell), "pbc": [True] * 3}, cutoff=model.cutoff)


def test_byteff_pol_periodic_forces_and_stress():
    """Periodic ByteFF-Pol: forces and stress are the derivatives of the energy (Ewald sums, iterative dipoles)."""
    model, pos, z, L = _pol_box()
    out = ForceStressOutput(model, compute_stress=True)(_box_graph(model, pos, z, L))
    assert float(out["e_polarization"][0]) < 0.0
    assert out["induced_dipoles"].shape == (len(z), 3)
    h = 1e-5
    for atom, axis in ((1, 0), (len(z) - 1, 2)):
        e = []
        for s in (1, -1):
            p = pos.copy()
            p[atom, axis] += s * h
            e.append(float(model(_box_graph(model, p, z, L))["energy"][0]))
        f = float(out["forces"][atom, axis])
        assert abs(-(e[0] - e[1]) / (2 * h) - f) < 1e-6 * max(1.0, abs(f))
    cell = np.eye(3) * L
    for a, b in ((0, 1), (2, 2)):
        e = []
        for s in (1, -1):
            eps = np.zeros((3, 3))
            eps[a, b] += 0.5 * s * h
            eps[b, a] += 0.5 * s * h
            g = _box_graph(model, pos @ (np.eye(3) + eps), z, cell @ (np.eye(3) + eps))
            e.append(float(model(g)["energy"][0]))
        assert float(out["stress"][0, a, b]) == pytest.approx((e[0] - e[1]) / (2 * h) / L ** 3, rel=1e-5, abs=1e-9)


def test_byteff_pol_ewald_split_invariance():
    """The periodic electrostatics and induction do not depend on how the Ewald sums are split."""
    energies = []
    for alpha, grid in ((1.0, [120] * 3), (1.15, [128] * 3)):
        model, pos, z, L = _pol_box(ewald_alpha=alpha, pme_grid=grid)
        out = model(_box_graph(model, pos, z, L))
        energies.append((float(out["e_electrostatics"][0]), float(out["e_polarization"][0])))
    (e1, p1), (e2, p2) = energies
    assert abs(e1 - e2) < 1e-7 * abs(e1)
    assert abs(p1 - p2) < 1e-6 * abs(p1)


def test_byteff_pol_periodic_images():
    """Moving a molecule, or every atom on its own, by lattice vectors leaves the energy unchanged."""
    model, pos, z, L = _pol_box()
    ref = model(_box_graph(model, pos, z, L))["energy"]
    moved = pos.copy()
    moved[3:6] += np.array([L, -L, 2 * L])
    scattered = pos + L * np.random.default_rng(4).integers(-2, 3, size=pos.shape)
    for p in (moved, scattered):
        assert torch.allclose(model(_box_graph(model, p, z, L))["energy"], ref, rtol=1e-10, atol=1e-12)


def test_byteff_pol_periodic_training_gradient():
    """The parameter gradient of a force loss through the iterative dipole solve matches finite differences
    (the solves converged far enough that their warm starts do not show in the differences)."""
    model, pos, z, L = _pol_box(long_range_correction=True, polar_eps=1e-13)
    model.train()
    fx = ForceStressOutput(model).train()
    weight = next(p for name, p in model.named_parameters() if "nonbonded" in name and p.dim() == 2)

    def loss():
        out = fx(_box_graph(model, pos, z, L))
        return out["energy"].pow(2).sum() * 1e-2 + out["forces"].pow(2).sum()

    model.zero_grad()
    loss().backward()
    g = float(weight.grad[0, 0])
    h = 1e-5
    values = []
    with torch.no_grad():
        for s in (1, -1):
            weight[0, 0] += s * h
            with torch.enable_grad():
                values.append(float(loss()))
            weight[0, 0] -= s * h
    assert g == pytest.approx((values[0] - values[1]) / (2 * h), rel=1e-5, abs=1e-8)


def test_byteff_pol_periodic_errors_and_config():
    """A cutoff needs a cell at least twice as wide; the config's all_pairs: false sets it."""
    model, pos, z, L = _pol_box()
    with pytest.raises(ValueError, match="twice the cutoff"):
        model(_box_graph(model, pos, z, 8.0))
    with pytest.raises(ValueError, match="periodic cells"):
        model(_graphs(z, [pos], model))
    cfg = from_dict({"model": {"name": "byteff_pol", "cutoff": 9.0,
                               "extra": {**SMALL_POL, "structure": "O", "all_pairs": False}}})
    built = build_model(cfg.model)
    assert isinstance(built, ByteFFPol) and not built.all_pairs and built.cutoff == 9.0


def test_byteff_pol_periodic_fast_path_matches_reference():
    """On a GPU, the periodic fast path (fused induction solve and mesh) gives the reference's energy, forces and
    stress."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("triton")
    model, pos, z, L = _pol_box()
    model = model.cuda().eval()
    graph = _box_graph(model, pos, z, L).to("cuda")
    results = []
    for fast in (False, True):
        model.set_use_fast(fast)
        results.append(ForceStressOutput(model, compute_stress=True).eval()(graph))
    ref, got = results
    for key in ("energy", "forces", "stress", "e_polarization", "induced_dipoles"):
        assert torch.allclose(got[key], ref[key], rtol=1e-9, atol=1e-11), key


@pytest.mark.parametrize("predictor", [None, "aspc"])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="records on a GPU"))])
def test_byteff_pol_step_graph(predictor, device):
    """Molecular dynamics steps of a periodic box on a step plan (fixed solve lengths after the warm-up, a
    pair-list rebuild, CUDA-graph replays on a GPU) equal fresh evaluations."""
    from xnn.common.deploy.step_graph import StepGraph
    if device == "cuda":
        pytest.importorskip("triton")
    model, pos, z, L = _pol_box(predictor=predictor)
    model = model.to(device).eval()
    ref = ForceStressOutput(model, compute_stress=True).eval()
    stepper = StepGraph(ForceStressOutput(model, compute_stress=True), model.cutoff, skin=0.3, warmup=2)
    velocity = np.random.default_rng(6).normal(scale=0.02, size=pos.shape)
    for step in range(10):
        s = {"pos": torch.tensor(pos + step * velocity, device=device),
             "atomic_numbers": torch.tensor(z, device=device), "cell": torch.eye(3, device=device) * L,
             "pbc": torch.ones(3, dtype=torch.bool, device=device)}
        got = stepper(s)
        want = ref(structure_to_graph(s, model.cutoff, device=device))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-8, atol=1e-9), (step, key)
    assert stepper.rebuilds >= 1
    if device == "cuda":
        assert stepper.replays > 0


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_byteff_pol_pair_kernels_match_reference(dtype):
    """The fused pair kernels of a step plan give the reference pair block's energies, field, ladders and
    vector-Jacobian products."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("triton")
    from xnn.hybnn.models.byteff import _pol_real_space
    from xnn.hybnn.models.byteff_kernels import pol_real_space
    torch.set_default_dtype(dtype)
    model, pos, z, L = _pol_box()
    model = model.to("cuda", dtype).eval()
    graph = _box_graph(model, pos, z, L).to("cuda")
    graph.pos, graph.cell = graph.pos.to(dtype), graph.cell.to(dtype)
    plan = model.plan_step(graph, 0.3, capacity=None)
    src, dst, shift = plan.pairs
    r = (graph.pos[dst] - graph.pos[src] + shift.to(graph.pos.dtype) @ graph.cell[0]).to(dtype).detach()
    R = model._rows(model._parameters_cached(), 1, dtype)
    atoms = torch.stack([R["charge"], R["damp"]] + [R["atom_" + k] for k in ("eps", "lam", "r0", "c6", "ct_eps",
                                                                             "ct_lam")], 1)
    consts = (model.cutoff ** 2, plan.ewald["alpha"], model.thole, model.coulomb_constant, model.repulsion_prefactor,
              model.wall, model.dispersion_damping)
    rr, rk = r.clone().requires_grad_(True), r.clone().requires_grad_(True)
    ref = _pol_real_space(rr, plan.valid, plan.scales, src, dst, atoms, graph.num_nodes, *consts)
    got = pol_real_space(rk, src, dst, plan.scales, atoms, plan.rows, plan.count, *consts)
    tol = dict(rtol=1e-10, atol=1e-12) if dtype == torch.float64 else dict(rtol=2e-5, atol=2e-6)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, **{k: v * float(b.abs().max()) if k == "atol" else v for k, v in tol.items()})
    gen = torch.Generator(device="cuda").manual_seed(0)
    cot = [torch.randn(t.shape, dtype=dtype, device="cuda", generator=gen) for t in ref]
    g_ref = torch.autograd.grad(sum((c * t).sum() for c, t in zip(cot, ref)), rr)[0]
    g_got = torch.autograd.grad(sum((c * t).sum() for c, t in zip(cot, got)), rk)[0]
    # float32 gradients sum many pair terms whose order the reference's atomic adds vary from run to run
    scale = float(g_ref.abs().max())
    gtol = (1e-10, 1e-12) if dtype == torch.float64 else (1e-4, 1e-4)
    assert torch.allclose(g_got, g_ref, rtol=gtol[0], atol=gtol[1] * scale)


@pytest.mark.parametrize("polarizable", [False, True])
def test_graphed_evaluation_of_molecule_batches(polarizable):
    """Without a cutoff a bound model's batch is captured unpadded as one CUDA graph and replayed: the same
    energies and forces as plain evaluations."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from xnn.common.deploy import GraphedEvaluation
    mols = [_mol("CCO"), _mol("O")]
    torch.manual_seed(3)
    model = (ByteFFPol(mols, **SMALL_POL) if polarizable else _byteff(mols)).cuda().eval()
    coords = np.concatenate([_coords(mols[0], n=4), _coords(mols[1], n=4) + 3.0], 1)
    z = _z(mols)
    plain = ForceStressOutput(model).eval()
    graphed = GraphedEvaluation(ForceStressOutput(model).eval())
    rng = np.random.default_rng(0)
    for step in range(5):
        batch = _graphs(z, coords + 0.02 * rng.standard_normal(coords.shape), model).to("cuda")
        got, want = graphed(batch), plain(batch)
        for key in ("energy", "forces"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-10, atol=1e-11), (step, key)
    assert graphed.graph is not None


def test_byteff_pol_step_graph_late_checks():
    """Molecular dynamics steps of a periodic box with late checks (each step's status read during the next
    one, the pair list built a step earlier) equal fresh evaluations, rebuilds included, with no failed step."""
    if not torch.cuda.is_available():
        pytest.skip("replays on a GPU")
    pytest.importorskip("triton")
    from xnn.common.deploy.step_graph import StepGraph
    model, pos, z, L = _pol_box(predictor="aspc")
    model = model.to("cuda").eval()
    ref = ForceStressOutput(model, compute_stress=True).eval()
    stepper = StepGraph(ForceStressOutput(model, compute_stress=True), model.cutoff, skin=0.3, warmup=2,
                        late_checks=True)
    velocity = np.random.default_rng(6).normal(scale=0.004, size=pos.shape)
    for step in range(16):
        s = {"pos": torch.tensor(pos + step * velocity, device="cuda"),
             "atomic_numbers": torch.tensor(z, device="cuda"), "cell": torch.eye(3, device="cuda") * L,
             "pbc": torch.ones(3, dtype=torch.bool, device="cuda")}
        got = stepper(s)
        want = ref(structure_to_graph(s, model.cutoff, device="cuda"))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-8, atol=1e-9), (step, key)
    assert stepper.settle() and stepper.late_failures == 0
    assert stepper.rebuilds >= 1 and stepper.replays > 0


def test_mdi_engine_takes_late_checks_and_unwraps_the_driver_positions():
    """An MDI engine on step plans reads ByteFF-Pol's steps late (the model's choice, which the caller can
    override), and unwraps positions a driver wraps back into the cell against the last step: the energies
    equal fresh evaluations of the wrapped positions."""
    from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine
    model, pos, z, L = _pol_box()
    model = model.eval()
    wrapped = ForceStressOutput(model, compute_stress=True)
    engine = MDIEngine(wrapped, model.cutoff, step_graph=True, skin=0.3)
    assert engine.steps.late_checks
    assert not MDIEngine(wrapped, model.cutoff, step_graph=True, late_checks=False).steps.late_checks
    with pytest.raises(ValueError, match="late_checks"):
        MDIEngine(wrapped, model.cutoff, late_checks="yes")
    engine.natoms, engine.atomic_numbers = len(z), np.asarray(z)
    engine.cell_bohr = np.eye(3) * L / BOHR_TO_ANGSTROM
    velocity = np.random.default_rng(4).normal(scale=0.01, size=pos.shape)
    for step in range(4):
        x = pos + step * velocity
        sent = x.copy()
        if step >= 2:
            # the driver wraps atom 0 back into the cell
            sent[0, 0] += L
        engine.coords_bohr = sent / BOHR_TO_ANGSTROM
        engine.calculate()
        assert np.allclose(engine._unwrapped, x, atol=1e-12)
        s = {"pos": torch.tensor(sent), "atomic_numbers": torch.tensor(z), "cell": torch.eye(3) * L,
             "pbc": torch.ones(3, dtype=torch.bool)}
        want = wrapped(structure_to_graph(s, model.cutoff))["energy"]
        assert engine.energy * HARTREE_TO_EV == pytest.approx(float(want.sum()), rel=1e-9, abs=1e-10)


def test_byteff_pol_torchscript_export(tmp_path):
    """The TorchScript export of a bound ByteFF-Pol model gives the reference energy, forces and charges of a
    cluster to round-off; other atoms, a cell and a model of periodic cells are refused."""
    from xnn.common.deploy import export_torchscript_potential
    mols = [_mol("O"), _mol("[Li+]")]
    torch.manual_seed(1)
    model = ByteFFPol(mols, **SMALL_POL).eval()
    pos = np.concatenate([_coords(mols[0])[0], np.array([[0.0, 0.0, 2.2]])])
    pos = pos + np.random.default_rng(2).normal(0.0, 0.03, pos.shape)
    z = _z(mols)
    want = ForceStressOutput(model).eval()(_graphs(z, [pos], model))
    path = export_torchscript_potential(model, 6.0, str(tmp_path / "byteff_pol.pt"))
    scripted = torch.jit.load(path)
    got = scripted(torch.tensor(pos), torch.tensor(z))
    assert torch.allclose(got["energy"], want["energy"].detach(), rtol=1e-12, atol=1e-12)
    assert torch.allclose(got["forces"], want["forces"].detach(), rtol=1e-10, atol=1e-11)
    assert torch.equal(got["charges"], want["charges"].detach())
    with pytest.raises(Exception, match="bound"):
        scripted(torch.tensor(pos), torch.tensor(z[::-1].copy()))
    with pytest.raises(Exception, match="cells"):
        scripted(torch.tensor(pos), torch.tensor(z), torch.eye(3, dtype=torch.float64) * 30.0,
                 torch.ones(3, dtype=torch.bool))
    periodic = _pol_box()[0]
    with pytest.raises(ValueError, match="periodic"):
        export_torchscript_potential(periodic, periodic.cutoff, str(tmp_path / "box.pt"))
