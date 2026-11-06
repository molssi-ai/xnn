"""Fidelity of xnn ByteFF and ByteFF-Pol to the reference package byteff2 (github.com/ByteDance-Seed/byteff2).

The reference package is an oracle only; it must be importable (``import byteff2``). Compared, in float64 and
float32: the molecular graphs; the predicted parameters, the energies (total and per term) and the forces of
molecules and of ByteFF-Pol clusters, for random networks of a small architecture (transplanted through
:func:`~xnn.hybnn.models.byteff_foundation.from_byteff2`) and for the released networks (ByteFF-26, ByteFF-Pol-25,
from the model hub cache, downloaded on first use). Periodic ByteFF-Pol is compared with the reference package's
OpenMM route (``byteff2.toolkit.openmmtool``) on OpenMM's Reference platform, which needs an OpenMM built with the
reference package's AMOEBA scale patches. Usage::

    python byteff_fidelity.py graphs
    python byteff_fidelity.py random|released float64|float32
    python byteff_fidelity.py periodic

The last line printed is the worst relative error.

Handled differences of the reference package: its float64 induction keeps a self term on the diagonal blocks of
the dipole tensor (float32 rounds it away, the paper has none), which is removed here; its torsion phase takes
the global default dtype, which is set before each run; its ensemble is evaluated with ``skip_ff=True`` and the
averaged parameters; it cannot evaluate a lone atom, which is skipped.
"""
import contextlib
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.ffnn.models.oplslib import KCAL_TO_EV
from xnn.gnn.featurizers.molgraph import molecular_graph
from xnn.hybnn.models.byteff_foundation import FOUNDATION_MODELS, from_byteff2

BYTEFF_SMILES = [
    "CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CS(C)=O", "CS(N)(=O)=O", "C[N+](=O)[O-]", "CC#N", "CC#C",
    "c1ccncc1", "CC(=O)[O-]", "[NH3+]CC(=O)[O-]", "COP(=O)(OC)OC", "FC(F)(F)c1ccc(Br)cc1I", "ClCCl",
    "C1CC2CCC1C2", "c1ccc2ccccc2c1", "O=C1NC(=O)C=C1", "CC(C)(C)OC(=O)N1CCCC1C(=O)O", "CN=[N+]=[N-]",
    "O=S(=O)([N-]c1ccccn1)c1ccccc1", "C1CCCCCCC1", "C1CCCCCCCC1", "OS(=O)(=O)O", "F[P-](F)(F)(F)(F)F",
    "c1ccc2[nH]ccc2c1", "O=c1cccc[nH]1", "CC=CC=O", "C[S+](C)C", "CC(=O)N(C)C", "OB(O)c1ccccc1",
    "c1cc[n+](C)cc1", "N#Cc1ccccc1C#N", "CSSC", "O=C(O)c1ccccc1C(=O)O", "C[NH+](C)C",
]
POL_SMILES = [
    "C1COC(=O)O1", "COC(=O)OC", "CCOC(=O)OC", "[Li+]", "F[P-](F)(F)(F)(F)F", "FS(=O)(=O)[N-]S(=O)(=O)F",
    "CC#N", "O", "CCO", "CS(C)=O", "CCOCCOC", "O=C1OCC(F)O1",
]
POL_CLUSTERS = [("CCOC(=O)OC", "CCOC(=O)OC"), ("CCOC(=O)OC", "[Li+]"), ("O", "O"),
                ("FS(=O)(=O)[N-]S(=O)(=O)F", "[Li+]"),
                ("COC(=O)OC", "COC(=O)OC", "COC(=O)OC", "[Li+]", "FS(=O)(=O)[N-]S(=O)(=O)F")]

BYTEFF_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
POL_ELEMENTS = [1, 3, 6, 7, 8, 9, 15, 16, 17]

# the reference package's model options of a small architecture (random networks) and of the released networks
_SMALL_GRAPH = dict(feature_layer=dict(act="gelu", atom_embedding_dim=8, node_mlp_dims=[16, 16, 2],
                                       edge_mlp_dims=[16, 16, 2]),
                    gnn_layer=dict(act="gelu", at_channels=4, ffn_dims=[16, 2], gnn_dims=[16, 16, 2], gnn_type="EGT",
                                   heads=2, jk="cat"))
_RELEASED_GRAPH = dict(feature_layer=dict(act="gelu", atom_embedding_dim=32, node_mlp_dims=[64, 256, 2],
                                          edge_mlp_dims=[64, 256, 2]),
                       gnn_layer=dict(act="gelu", at_channels=16, ffn_dims=[256, 2], gnn_dims=[256, 256, 4],
                                      gnn_type="EGT", heads=16, jk="cat"))


def upstream_options(polarizable: bool, released: bool) -> dict:
    """Options of the reference package's ``HybridFF``: a small architecture, or the released networks'."""
    if released:
        pre, out, lj_pre, lj_out = [256, 256, 3], [256, 4], [512, 512, 4], [256, 3]
        post = [256, 256, 1]
    else:
        pre, out, lj_pre, lj_out = [16, 16, 2], [16, 2], [16, 16, 2], [16, 2]
        post = [16, 16, 1]
    bonded = dict(act="gelu", out_mlp_dims=out, post_mlp_dims=post, pre_mlp_dims=pre, tanh_output=20.0,
                  type="MMBondedConj" if polarizable else "MMBonded")
    if polarizable:
        ff = [bonded, dict(act="gelu", out_mlp_dims=lj_out, pre_mlp_dims=pre, type="ChargeVolume"),
              dict(act="gelu", charge14=0.5, disp_damping_factor=120.0, fix_li_alpha=1e-6, ind14=0.5,
                   li_damp_clip=0.2, out_mlp_dims=lj_out, pol_damping_factor=0.39, pre_mlp_dims=pre, s12=1.5,
                   type="Exp6Pol", vdw14=0.5)]
    else:
        ff = [bonded, dict(act="gelu", out_mlp_dims=lj_out, pre_mlp_dims=lj_pre, tanh_output=20.0, type="LJEs")]
    return dict(graph_block=_RELEASED_GRAPH if released else _SMALL_GRAPH, ff_block=ff,
                supported_elements=POL_ELEMENTS if polarizable else BYTEFF_ELEMENTS)


def mol_h(smiles: str):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def mapped(mol) -> str:
    """The mapped SMILES the reference package reads (atom map numbers = xnn atom order + 1)."""
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    return Chem.MolToSmiles(m)


def coords_of(mol, n_conf: int, seed: int) -> np.ndarray:
    m = Chem.Mol(mol)
    if m.GetNumAtoms() == 1:
        return np.zeros((n_conf, 1, 3))
    ids = AllChem.EmbedMultipleConfs(m, n_conf, randomSeed=seed)
    assert len(ids) == n_conf, Chem.MolToSmiles(mol)
    rng = np.random.default_rng(seed)
    # off the force field's planar and linear minima, where parameter derivatives cancel
    return np.stack([m.GetConformer(i).GetPositions() + 0.03 * rng.standard_normal((m.GetNumAtoms(), 3))
                     for i in ids])


def cluster_coords(mols, n_conf: int, seed: int) -> np.ndarray:
    """Randomly oriented molecules about 4 Angstrom apart."""
    rng = np.random.default_rng(seed)
    parts = []
    for k, m in enumerate(mols):
        c = coords_of(m, n_conf, seed + k)
        c = c - c.mean(1, keepdims=True)
        for f in range(n_conf):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            c[f] = c[f] @ q
        shift = np.array([4.0 * (k % 2), 4.0 * ((k // 2) % 2), 4.0 * (k // 4)]) + 0.3 * rng.standard_normal(3)
        parts.append(c + shift)
    return np.concatenate(parts, 1)


def rel(a, b) -> float:
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-30)) if b.numel() else 0.0


def compare_graph(smiles: str) -> list:
    """The names of the graph arrays in which xnn and the reference package differ (empty: identical)."""
    from byteff2.data import GraphData
    mol = mol_h(smiles)
    up = GraphData("x", mapped(mol))
    g = molecular_graph(mol)
    n = g.n_atoms
    checks = {
        "node_features": (up.node_features.long().numpy(),
                          np.stack([g.atomic_numbers - 1, g.degree, g.formal_charges, g.ring_degree, g.min_ring], 1)),
        "edge_features": (up.edge_features.long().numpy(),
                          np.repeat(np.stack([g.bond_in_ring, g.bond_order], 1), 2, axis=0)),
        "bonds": (up.inc_node_bond.long().numpy(), g.bonds), "angles": (up.inc_node_angle.long().numpy(), g.angles),
        "propers": (up.inc_node_proper.long().numpy(), g.propers),
        "impropers": (up.inc_node_improper.long().numpy(), g.impropers),
        "angle_edges": (up.inc_edge_angle.long().numpy(), g.angle_edges),
        "proper_edges": (up.inc_edge_proper.long().numpy(), g.proper_edges),
        "improper_edges": (up.inc_edge_improper.long().numpy(), g.improper_edges),
        "atom_class": (up.inc_node_equiv.long().numpy(), g.atom_class),
        "edge_class": (up.inc_edge_equiv.long().numpy(), g.edge_class),
        "proper_mask": (up.proper_mask.numpy() > 0.5, ~g.proper_linear),
    }
    for k in (12, 13, 14, 15):
        checks[f"pairs{k}"] = (getattr(up, f"inc_node_nonbonded{k}").long().numpy(), getattr(g, f"pairs{k}"))
    excluded = {tuple(p) for k in (12, 13, 14) for p in getattr(g, f"pairs{k}").tolist()}
    far = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in excluded]).reshape(-1, 2)
    checks["pairs_far"] = (up.inc_node_nonbonded_all.long().numpy(), far)
    return [k for k, (a, b) in checks.items() if a.shape != b.shape or not np.array_equal(a.reshape(b.shape), b)]


@contextlib.contextmanager
def paper_induction():
    """The reference package's induction without its float64 self term (the i = i blocks of the dipole tensor)."""
    from byteff2.model.ff_layers import mm_tspol
    orig = mm_tspol.PolarizationSolver.compute_interaction_tensor

    def patched(self, *args):
        T = orig(self, *args)
        n = T.shape[-1] // 3
        keep = 1.0 - torch.kron(torch.eye(n, dtype=T.dtype, device=T.device),
                                torch.ones(3, 3, dtype=T.dtype, device=T.device))
        return T * keep
    mm_tspol.PolarizationSolver.compute_interaction_tensor = patched
    try:
        yield
    finally:
        mm_tspol.PolarizationSolver.compute_interaction_tensor = orig


@contextlib.contextmanager
def default_dtype(dtype):
    """The reference package builds its torsion phase in the global default dtype."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def random_models(polarizable: bool, dtype, seed: int = 5) -> tuple:
    """A random reference network of the small architecture and its transplant into xnn."""
    from byteff2.model.hybrid_ff import HybridFF
    torch.manual_seed(seed)
    up = HybridFF(**upstream_options(polarizable, released=False)).double()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "optimal.pt"
        torch.save({"model_state_dict": up.state_dict()}, path)
        model = from_byteff2(path, dtype=dtype)
    return up.to(dtype).eval(), model


def released_checkpoints(name: str) -> list:
    """The released checkpoint files of ``name`` in the model hub cache (downloaded on first use)."""
    from xnn.common.models.hub.cache import resolve_cache_dir
    from xnn.hybnn.models.byteff_foundation import from_foundation
    cards = FOUNDATION_MODELS[name]
    paths = [resolve_cache_dir(None) / c / "raw" / "optimal.pt" for c in cards]
    if not all(p.is_file() for p in paths):
        from_foundation(name)
    return paths


def released_upstream(path, polarizable: bool, dtype):
    from byteff2.model.hybrid_ff import HybridFF
    up = HybridFF(**upstream_options(polarizable, released=True))
    up.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["model_state_dict"])
    return up.to(dtype).eval()


def xnn_eval(model, mols, coords, dtype) -> dict:
    """Energies, forces and terms of conformations of the molecules ``mols`` (eV, Angstrom)."""
    model.bind(list(mols))
    z = torch.tensor([a.GetAtomicNum() for m in mols for a in m.GetAtoms()])
    graphs = [structure_to_graph({"pos": torch.as_tensor(c, dtype=dtype), "atomic_numbers": z},
                                 cutoff=model.cutoff) for c in coords]
    out = ForceStressOutput(model)(collate(graphs))
    return {k: v.detach() for k, v in out.items()}


TERMS = {"byteff": {"e_bond": "MMBonded.bond_energy", "e_angle": "MMBonded.angle_energy",
                    "e_torsion": "MMBonded.proper_energy", "e_improper": "MMBonded.improper_energy"},
         "pol": {"e_bond": "MMBondedConj.bond_energy", "e_angle": "MMBondedConj.angle_energy",
                 "e_torsion": "MMBondedConj.proper_energy", "e_improper": "MMBondedConj.improper_energy",
                 "e_repulsion": "Exp6Pol.rep_energy", "e_dispersion": "Exp6Pol.disp_energy",
                 "e_electrostatics": "Exp6Pol.perm_chg_energy", "e_polarization": "Exp6Pol.induction_energy",
                 "e_charge_transfer": "Exp6Pol.ct_energy"}}
PARAMETERS = {"byteff": {"bond_k": "PreMMBonded.bond_k", "bond_r0": "PreMMBonded.bond_r0",
                         "angle_k": "PreMMBonded.angle_k", "angle_d0": "PreMMBonded.angle_d0",
                         "improper_k": "PreMMBonded.improper_k", "charge": "PreLJEs.charges",
                         "sigma": "PreLJEs.sigma", "epsilon": "PreLJEs.epsilon"},
              "pol": {"bond_k1": "PreMMBondedConj.bond_k1", "bond_k2": "PreMMBondedConj.bond_k2",
                      "angle_k1": "PreMMBondedConj.angle_k1", "angle_k2": "PreMMBondedConj.angle_k2",
                      "improper_k": "PreMMBondedConj.improper_k", "charge": "PreChargeVolume.charges",
                      "volume": "PreChargeVolume.volumes", "c6": "PreExp6Pol.c6", "r0": "PreExp6Pol.rvdw",
                      "alpha": "PreExp6Pol.alpha", "damp": "PreExp6Pol.pol_damping", "lam": "PreExp6Pol.lambda",
                      "eps": "PreExp6Pol.eps", "ct_eps": "PreExp6Pol.ct_eps", "ct_lam": "PreExp6Pol.ct_lamb"}}


def compare_parameters(up_params: dict, model, kind: str) -> dict:
    """Largest relative difference of each predicted parameter (ensemble members averaged)."""
    raw = {k: v.mean(0) for k, v in model.predicted_parameters(per_member=True).items()}
    out = {k: rel(raw[k].reshape(-1), up_params[u].reshape(-1)) for k, u in PARAMETERS[kind].items()}
    prefix = "PreMMBonded" if kind == "byteff" else "PreMMBondedConj"
    out["proper_k"] = rel(model.predicted_parameters()["proper_k"].reshape(-1),
                          up_params[f"{prefix}.proper_k"].reshape(-1))
    return out


def compare_energies(up_e, up_f, up_terms: dict, out: dict, kind: str, n_conf: int) -> dict:
    """Relative differences of the energy and forces (to their largest value) and of each term (to the largest
    energy), in kcal/mol."""
    e = out["energy"].reshape(-1) / KCAL_TO_EV
    f = out["forces"].reshape(n_conf, -1, 3) / KCAL_TO_EV
    scale = up_e.abs().max().clamp(min=1e-30)
    res = {"energy": rel(e, up_e.reshape(-1)), "forces": rel(f, up_f.permute(1, 0, 2))}
    for k, u in TERMS[kind].items():
        if u in up_terms:
            res[k] = float((out[k].reshape(-1) / KCAL_TO_EV - up_terms[u].reshape(-1).double()).abs().max() / scale)
    if kind == "byteff":
        nb = sum(out[k] for k in ("e_vdw", "e_coulomb", "e_vdw14", "e_coulomb14")).reshape(-1) / KCAL_TO_EV
        res["e_nonbonded"] = float((nb - up_terms["LJEs.energy"].reshape(-1).double()).abs().max() / scale)
    return res


def _tensors(d: dict, suffix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and k.endswith(suffix):
            out[k[:len(k) - len(suffix)] if suffix else k] = v.detach()
    return out


def molecules(up, model, smiles: list, kind: str, dtype, n_conf: int = 2) -> dict:
    """Per molecule: parameters, energies, forces and terms of xnn against the reference package."""
    from byteff2.data import MonoData
    results = {}
    for smi in smiles:
        mol = mol_h(smi)
        if mol.GetNumAtoms() == 1:       # the reference package's evaluation needs a pair of atoms
            continue
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= set(POL_ELEMENTS if kind == "pol" else BYTEFF_ELEMENTS):
            continue
        coords = coords_of(mol, n_conf, 11)
        data = MonoData("x", mapped(mol), confdata={"coords": coords}, max_n_confs=n_conf, float_dtype=dtype)
        preds = up(data)
        up_params = _tensors(preds["ff_parameters"])
        out = xnn_eval(model, [mol], coords, dtype)
        r = compare_parameters(up_params, model, kind)
        r.update(compare_energies(preds["energy"].detach(), preds["forces"].detach(), up_params, out, kind, n_conf))
        r["_energy"], r["_forces"] = preds["energy"].detach().reshape(-1), preds["forces"].detach().permute(1, 0, 2)
        r["_xnn_energy"] = out["energy"].reshape(-1) / KCAL_TO_EV
        r["_xnn_forces"] = out["forces"].reshape(n_conf, -1, 3) / KCAL_TO_EV
        results[smi] = r
    return results


def ensemble(ups: list, model, smiles: list, dtype, n_conf: int = 2) -> dict:
    """The reference package's ensemble (averaged parameters) against an xnn model of several members."""
    from byteff2.data import MonoData
    from byteff2.model.hybrid_ff import EnsembleModel
    results = {}
    ens = EnsembleModel(ups)
    for smi in smiles:
        mol = mol_h(smi)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= set(BYTEFF_ELEMENTS):
            continue
        coords = coords_of(mol, n_conf, 11)
        data = MonoData("x", mapped(mol), confdata={"coords": coords}, max_n_confs=n_conf, float_dtype=dtype)
        params = _tensors(ens(data, skip_ff=True)["ff_parameters"])
        e, f = ups[0].ff_block(data, None, None, dict(params))
        out = xnn_eval(model, [mol], coords, dtype)
        r = compare_parameters(params, model, "byteff")
        r.update({"energy": rel(out["energy"] / KCAL_TO_EV, e.reshape(-1)),
                  "forces": rel(out["forces"].reshape(n_conf, -1, 3) / KCAL_TO_EV, f.permute(1, 0, 2))})
        results[smi] = r
    return results


def clusters(up, model, combos: list, dtype, n_conf: int = 2) -> dict:
    """ByteFF-Pol clusters: the whole cluster's energy, forces and terms."""
    from byteff2.data import ClusterData
    results = {}
    for combo in combos:
        mols = [mol_h(s) for s in combo]
        coords = cluster_coords(mols, n_conf, 17)
        data = ClusterData("c", [mapped(m) for m in mols], confdata={"coords": coords}, max_n_confs=n_conf,
                           float_dtype=dtype)
        preds = up(data, cluster=True)
        up_terms = _tensors(preds["ff_parameters"], "_cluster")
        out = xnn_eval(model, mols, coords, dtype)
        r = compare_energies(preds["energy_cluster"].detach(), preds["forces_cluster"].detach(), up_terms, out,
                             "pol", n_conf)
        r["_energy"] = preds["energy_cluster"].detach().reshape(-1)
        r["_forces"] = preds["forces_cluster"].detach().permute(1, 0, 2)
        r["_xnn_energy"] = out["energy"].reshape(-1) / KCAL_TO_EV
        r["_xnn_forces"] = out["forces"].reshape(n_conf, -1, 3) / KCAL_TO_EV
        results[".".join(combo)] = r
    return results


def run(source: str, dtype, quick: bool = False) -> dict:
    """Every comparison of one source (``"random"`` or ``"released"``) in one dtype: ``{section: {case: {quantity:
    relative difference}}}``; keys starting with ``_`` hold the energies and forces of both codes (kcal/mol)."""
    report = {}
    with default_dtype(dtype), paper_induction(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if source == "random":
            up, model = random_models(False, dtype)
            report["byteff"] = molecules(up, model, BYTEFF_SMILES[:6] if quick else BYTEFF_SMILES[:12], "byteff",
                                         dtype)
            from byteff2.model.hybrid_ff import HybridFF
            ups, paths = [], []
            with tempfile.TemporaryDirectory() as tmp:
                for seed in (5, 6, 7):
                    torch.manual_seed(seed)
                    ups.append(HybridFF(**upstream_options(False, released=False)).double())
                    paths.append(Path(tmp) / f"member{seed}.pt")
                    torch.save({"model_state_dict": ups[-1].state_dict()}, paths[-1])
                members = from_byteff2(paths, dtype=dtype)
            ups = [u.to(dtype).eval() for u in ups]
            report["byteff_ensemble"] = ensemble(ups, members, BYTEFF_SMILES[:4], dtype)
            up, model = random_models(True, dtype)
            report["pol"] = molecules(up, model, POL_SMILES[:4] if quick else POL_SMILES, "pol", dtype)
            report["pol_clusters"] = clusters(up, model, POL_CLUSTERS[:2] if quick else POL_CLUSTERS, dtype)
        elif source == "released":
            paths = released_checkpoints("byteff-26")
            ups = [released_upstream(p, False, dtype) for p in paths]
            single = from_byteff2(paths[0], dtype=dtype, supported_elements=BYTEFF_ELEMENTS)
            report["byteff"] = molecules(ups[0], single, BYTEFF_SMILES[:6] if quick else BYTEFF_SMILES, "byteff",
                                         dtype)
            members = from_byteff2(paths, dtype=dtype, supported_elements=BYTEFF_ELEMENTS)
            report["byteff_ensemble"] = ensemble(ups, members, BYTEFF_SMILES[:6] if quick else BYTEFF_SMILES, dtype)
            [path] = released_checkpoints("byteff-pol-25")
            up = released_upstream(path, True, dtype)
            model = from_byteff2(path, dtype=dtype, supported_elements=POL_ELEMENTS)
            report["pol"] = molecules(up, model, POL_SMILES[:4] if quick else POL_SMILES, "pol", dtype)
            report["pol_clusters"] = clusters(up, model, POL_CLUSTERS[:2] if quick else POL_CLUSTERS, dtype)
        else:
            raise ValueError(f"unknown source {source!r}")
    return report


def worst(report: dict) -> dict:
    """The largest difference of each quantity per section."""
    out = {}
    for section, rows in report.items():
        w = {}
        for r in rows.values():
            for k, v in r.items():
                if not k.startswith("_"):
                    w[k] = max(w.get(k, 0.0), v)
        out[section] = w
    return out


# periodic ByteFF-Pol against the reference package's OpenMM route

# OpenMM's 1 / (4 pi eps0) (CODATA 2018), in kcal Angstrom / (mol e^2)
KE_OPENMM = 1.0 / (4.0 * np.pi * 1e-6 * 8.8541878128e-12 / (1.602176634e-19 ** 2 * 6.02214076e23)) * 10.0 / 4.184
NONBONDED = ("e_repulsion", "e_dispersion", "e_charge_transfer", "e_electrostatics", "e_polarization")


def grid_box(smiles_counts: list, spacing: float, seed: int = 0) -> tuple:
    """Randomly oriented molecules on a cubic grid: the molecules, positions and the box side."""
    mols = [mol_h(s) for s, k in smiles_counts for _ in range(k)]
    rng = np.random.default_rng(seed)
    mols = [mols[i] for i in rng.permutation(len(mols))]
    side = int(np.ceil(len(mols) ** (1 / 3)))
    shapes, pos = {}, []
    for k, m in enumerate(mols):
        key = mapped(m)
        if key not in shapes:
            c = coords_of(m, 1, 11)[0]
            shapes[key] = c - c.mean(0)
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        site = np.array([k % side, (k // side) % side, k // side ** 2])
        pos.append(shapes[key] @ q.T + spacing * (site + 0.5) + 0.2 * rng.standard_normal(3))
    return mols, np.concatenate(pos), spacing * side


def _species(mols) -> list:
    """Runs of identical molecules: ``(name, mol, count)``."""
    species, names = [], {}
    for m in mols:
        key = mapped(m)
        names.setdefault(key, f"M{len(names)}")
        if species and species[-1][0] == names[key]:
            species[-1] = (species[-1][0], m, species[-1][2] + 1)
        else:
            species.append((names[key], m, 1))
    return species


class _Topology:
    """Stands in for the GROMACS topology file the reference package's route reads: the molecules and bonds."""

    def __init__(self, species: list, box: float):
        import openmm.app as app
        self.topology = app.Topology()
        chain = self.topology.addChain()
        self.box = box
        for name, mol, n in species:
            for _ in range(n):
                res = self.topology.addResidue(name, chain)
                atoms = [self.topology.addAtom(f"{a.GetSymbol()}{a.GetIdx()}",
                                               app.Element.getByAtomicNumber(a.GetAtomicNum()), res)
                         for a in mol.GetAtoms()]
                for b in mol.GetBonds():
                    self.topology.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])

    def createSystem(self, **kw):
        import openmm as omm
        s = omm.System()
        for a in self.topology.atoms():
            s.addParticle(a.element.mass)
        L = self.box / 10.0
        s.setDefaultPeriodicBoxVectors(omm.Vec3(L, 0, 0), omm.Vec3(0, L, 0), omm.Vec3(0, 0, L))
        return s


def _nonbonded_parameters(model, mol) -> dict:
    """xnn's predicted nonbonded parameters of one molecule in the units of the reference package's route."""
    model.bind([mol])
    p = {k: v.detach().numpy() for k, v in model.predicted_parameters().items()}
    n = mol.GetNumAtoms()

    def take(k, s=1.0):
        return (p[k][:n] * s).tolist()
    return {"charge": take("charge"), "alpha": take("alpha", 1e-3), "pol_damping": take("damp", 1e-3),
            "lamb": take("lam"), "eps": take("eps", 4.184), "Rvdw": take("r0", 0.1), "C6": take("c6", 4.184e-6),
            "ct_eps": take("ct_eps", 4.184e-4), "ct_lamb": take("ct_lam")}


def openmm_box(model, mols, pos, L: float, cutoff: float) -> dict:
    """The reference package's OpenMM system of a box on the Reference platform, the dipoles converged to 1e-12:
    energies per force (kcal/mol) without and with the long-range correction, forces, induced dipoles and the
    PME parameters."""
    import openmm as omm
    from byteff2.toolkit import openmmtool
    species = _species(mols)
    params = {name: _nonbonded_parameters(model, mol) for name, mol, _ in species}
    params["metadata"] = {"s12": model.wall / 10.0 * 4.184 ** (1 / 12), "disp_damping": model.dispersion_damping,
                          "thole": model.thole}
    topology = _Topology(species, L)
    orig = openmmtool.GromacsTopFile
    openmmtool.GromacsTopFile = lambda *a, **k: topology
    try:
        _, system = openmmtool.generate_byteffpol_system("box.top", params, unit_cell=[L / 10.0] * 3,
                                                         cutoff=cutoff / 10.0)
    finally:
        openmmtool.GromacsTopFile = orig
    out = {}
    for lrc in (False, True):
        for i, f in enumerate(system.getForces()):
            f.setForceGroup(i + 1)
            if isinstance(f, omm.AmoebaMultipoleForce):
                f.setMutualInducedTargetEpsilon(1e-12)
                f.setMutualInducedMaxIterations(2000)
                amoeba = f
            if isinstance(f, omm.CustomNonbondedForce):
                f.setUseLongRangeCorrection(lrc)
        ctx = omm.Context(system, omm.VerletIntegrator(1e-3), omm.Platform.getPlatformByName("Reference"))
        ctx.setPositions((np.asarray(pos) / 10.0).tolist())
        groups = {}
        for i, f in enumerate(system.getForces()):
            st = ctx.getState(getEnergy=True, groups={i + 1})
            groups[type(f).__name__] = st.getPotentialEnergy().value_in_unit(omm.unit.kilojoule_per_mole) / 4.184
        out["lrc" if lrc else "plain"] = groups
        if not lrc:
            st = ctx.getState(getForces=True, groups=set(range(1, system.getNumForces() + 1)))
            out["forces"] = np.asarray(st.getForces(asNumpy=True).value_in_unit(
                omm.unit.kilojoule_per_mole / omm.unit.angstrom)) / 4.184
            out["dipoles"] = np.asarray(amoeba.getInducedDipoles(ctx)) * 10.0
            alpha, nx, ny, nz = amoeba.getPMEParametersInContext(ctx)
            out["pme"] = (alpha / 10.0, [nx, ny, nz])
        del ctx
    return out


@contextlib.contextmanager
def openmm_arithmetic():
    """xnn evaluated with two shortcuts of OpenMM's AMOEBA PME code: sqrt(pi) truncated to 1.77245385091 and the
    Thole damping of the mutual-induction pairs in single precision (the permanent fields' in double)."""
    import xnn.ffnn.models.amoeba_ewald as AE
    import xnn.hybnn.models.byteff as BF
    from xnn.ffnn.models.amoeba import thole_ladder
    exact, apply, stash = AE.SQRT_PI, BF._ImplicitInduction.apply, {}

    def ladder(R, ai, aj, ti, tj, order=3):
        if order == 2:
            stash["args"] = (R, ai, aj, ti, tj)
        return thole_ladder(R, ai, aj, ti, tj, order)

    def single_damping(rhs, alpha, expol, r, D1, D2, *rest):
        R, ai, aj, ti, tj = stash["args"]
        ref = thole_ladder(R, ai, aj, ti, tj, 2)
        d = torch.minimum(ti, tj) * (R / (ai * aj) ** (1.0 / 6.0)) ** 3
        ex = torch.where(d < 50.0, torch.exp((-d).to(torch.float32)).to(R.dtype), torch.zeros_like(d))
        keep = (R < 10.0).to(R.dtype)
        D1 = D1 + keep * (-(1.0 - ex) / R ** 3 - ref[1])
        D2 = D2 + keep * (3.0 * (1.0 - (1.0 + d) * ex) / R ** 5 - ref[2])
        return apply(rhs, alpha, expol, r, D1, D2, *rest)

    AE.SQRT_PI = BF.SQRT_PI = 1.77245385091
    BF.thole_ladder, BF._ImplicitInduction.apply = ladder, single_damping
    try:
        yield
    finally:
        AE.SQRT_PI = BF.SQRT_PI = exact
        BF.thole_ladder, BF._ImplicitInduction.apply = thole_ladder, apply


def xnn_box(model, mols, pos, L: float) -> tuple:
    """xnn's nonbonded terms (kcal/mol), nonbonded forces, induced dipoles and Ewald coefficient and mesh."""
    model.bind(mols)
    s = {"pos": torch.as_tensor(pos), "atomic_numbers": model.atomic_numbers, "cell": torch.eye(3) * L,
         "pbc": [True] * 3}
    batch = collate([structure_to_graph(s, cutoff=model.cutoff)])
    batch.pos.requires_grad_(True)
    out = model(batch)
    f = -torch.autograd.grad(sum(out[k] for k in NONBONDED).sum(), batch.pos)[0]
    terms = {k: float(out[k][0].detach()) / KCAL_TO_EV for k in NONBONDED}
    ew = model._ewald_context(batch, torch.float64)
    return terms, f.detach().numpy() / KCAL_TO_EV, out["induced_dipoles"].detach().numpy(), \
        (float(ew["alpha_p_value"]), ew["groups"][0]["grid"])


def periodic_box(mols, pos, L: float, cutoff: float = 10.0) -> dict:
    """One box: xnn (float64, OpenMM's Coulomb constant and Reference mesh, dipoles converged to 1e-12) against the
    reference package's OpenMM route; relative differences and both codes' values."""
    [path] = released_checkpoints("byteff-pol-25")
    with default_dtype(torch.float64), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = openmm_box(from_byteff2(path, dtype=torch.float64, coulomb_constant=KE_OPENMM), mols, pos, L, cutoff)

        def model(lrc=False):
            return from_byteff2(path, dtype=torch.float64, coulomb_constant=KE_OPENMM, cutoff=cutoff,
                                polar_eps=1e-12, max_iter=2000, long_range_correction=lrc, pme_grid=o["pme"][1])
        x, fx, dx, pme = xnn_box(model(), mols, pos, L)
        xl = xnn_box(model(True), mols, pos, L)[0]
        with openmm_arithmetic():
            q, fq, dq, _ = xnn_box(model(), mols, pos, L)
    sr_o = o["plain"]["CustomNonbondedForce"] + o["plain"]["CustomBondForce"]
    sr_x = x["e_repulsion"] + x["e_dispersion"] + x["e_charge_transfer"]
    tail_o = o["lrc"]["CustomNonbondedForce"] - o["plain"]["CustomNonbondedForce"]
    tail_x = sum(xl[k] - x[k] for k in ("e_repulsion", "e_dispersion", "e_charge_transfer"))
    el_o = o["plain"]["AmoebaMultipoleForce"]
    el_x, el_q = (t["e_electrostatics"] + t["e_polarization"] for t in (x, q))
    return {"atoms": len(pos), "box": L, "alpha": (o["pme"][0], pme[0]), "mesh": (o["pme"][1], pme[1]),
            "short_range": (sr_o, sr_x, abs(sr_o - sr_x) / abs(sr_o)),
            "long_range_correction": (tail_o, tail_x, abs(tail_o - tail_x) / abs(tail_o)),
            "electrostatics_polarization": (el_o, el_x, abs(el_o - el_x) / abs(el_o)),
            "dipoles": rel(dx, o["dipoles"]), "forces": rel(fx, o["forces"]),
            "openmm_arithmetic": {"electrostatics_polarization": abs(el_o - el_q) / abs(el_o),
                                  "dipoles": rel(dq, o["dipoles"]), "forces": rel(fq, o["forces"])}}


PERIODIC_BOXES = [("water 343", [("O", 343)], 3.1, 0),
                  ("DMC 52 + LiFSI 6", [("COC(=O)OC", 52), ("[Li+]", 6), ("FS(=O)(=O)[N-]S(=O)(=O)F", 6)], 7.0, 1)]


def main():
    mode = sys.argv[1]
    if mode == "graphs":
        smiles = sorted(set(BYTEFF_SMILES + POL_SMILES))
        bad = {s: compare_graph(s) for s in smiles}
        for s, b in bad.items():
            if b:
                print("differs", s, b)
        print(f"graphs: {sum(not b for b in bad.values())} of {len(smiles)} identical")
        print(float(sum(bool(b) for b in bad.values())))
        return
    if mode == "periodic":
        top = 0.0
        for name, spec, spacing, seed in PERIODIC_BOXES:
            r = periodic_box(*grid_box(spec, spacing, seed=seed))
            print(f"{name}: {r['atoms']} atoms | short-range {r['short_range'][2]:.1e} | long-range correction "
                  f"{r['long_range_correction'][2]:.1e} | electrostatics + polarization "
                  f"{r['electrostatics_polarization'][2]:.1e} | dipoles {r['dipoles']:.1e} | forces {r['forces']:.1e}"
                  f" | with OpenMM's arithmetic: " + ", ".join(f"{k} {v:.1e}"
                                                              for k, v in r["openmm_arithmetic"].items()))
            top = max(top, r["short_range"][2], *r["openmm_arithmetic"].values())
        print(top)
        return
    dtype = {"float64": torch.float64, "float32": torch.float32}[sys.argv[2]]
    report = worst(run(mode, dtype))
    for section, w in report.items():
        print(f"{section}: " + ", ".join(f"{k} {v:.1e}" for k, v in w.items()))
    print(max(max(w["energy"], w["forces"]) for w in report.values()))


if __name__ == "__main__":
    main()
