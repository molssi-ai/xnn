"""The terms of single-atom ions in HIPPO: the charge transfer from the partners into an ion, its pair
correction and its many-body charge transfer; their formulas, derivatives, kernels, step plans and files."""
import copy
import math
import warnings

import numpy as np
import pytest
import torch

from test_amoeba_periodic import _KERNEL_DEVICES, BOX, CELL
from xnn.common.data import structure_to_graph
from xnn.common.deploy.step_graph import StepGraph
from xnn.common.models import ForceStressOutput
from xnn.ffnn.models import Hippo, MolecularTopology, builtin_amoeba, read_amoeba, read_tinker_prm, write_tinker_prm
from xnn.ffnn.models.amoeba import ion_terms
from xnn.ffnn.models.amoeba_lammps import write_lammps_amoeba_input
from xnn.ffnn.models.amoebalib import KCAL_TO_EV


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


CT = (700.0, 3.0)             # A (kcal/mol), B (1/A) of the charge transfer from water O into the ion
PAIR = (500.0, 300.0, 3.5)    # A (kcal/mol), C (kcal/mol A^6), beta (1/A) of the pair correction
X = 0.17


def _library(ions=True):
    """HIPPO water with a sodium-like ion; with ``ions`` its terms of an ion, else its pairwise charge
    transfer off (the same model without the ion terms)."""
    lib = builtin_amoeba("hippo-water-2021")
    lib.atom_types["NA"] = {"class": "NA", "element": 11, "mass": 22.99, "connections": 0, "comment": "ion"}
    lib.multipoles["NA"] = [{"frame": ("0", "0", "0"), "q": 1.0, "dipole": [0.0, 0.0, 0.0],
                             "quadrupole": [[0.0] * 3, [0.0] * 3, [0.0] * 3]}]
    lib.polarize["NA"] = {"alpha": 0.12, "thole": 0.0, "groups": []}
    lib.chgpen["NA"] = {"core": 1.0, "alpha": 3.0}
    lib.repulsion["NA"] = {"k": 10.0, "alpha": 4.5, "valence": 2.0}
    lib.dispersion["NA"] = {"c6": 2.0, "alpha": 4.0}
    # the ion's own HIPPO charge transfer, which its rows replace
    lib.chgtrn["NA"] = {"eps": 2.0, "alpha": 3.0} if ions else {"eps": 0.0, "alpha": 0.0}
    if ions:
        lib.ionct["NA-OW"] = {"a": CT[0], "b": CT[1]}
        lib.mbct["NA"] = {"x": X}
        lib.ionpair["NA-OW"] = {"a": PAIR[0], "c": PAIR[1], "beta": PAIR[2]}
    return lib


def _waters(n):
    return ["OW", "HW", "HW"] * n, [(3 * m, 3 * m + 1) for m in range(n)] + [(3 * m, 3 * m + 2) for m in range(n)]


def _topology(n_waters):
    types, bonds = _waters(n_waters)
    return MolecularTopology.from_bonds(types + ["NA"], bonds)


def _cluster():
    """An ion with four waters at 2.3-2.5 Angstrom and a fifth at 4.6, the hydrogens pointing away."""
    rng = np.random.default_rng(3)
    pos = []
    for d, u in [(2.35, (1, 1, 1)), (2.4, (1, -1, -1)), (2.3, (-1, 1, -1)), (2.45, (-1, -1, 1)), (4.6, (1, 0, 0.3))]:
        u = np.asarray(u, float) / np.linalg.norm(u)
        v = np.cross(u, [0.3, 0.5, 0.8])
        v /= np.linalg.norm(v)
        o = d * u
        a = math.radians(52.0)
        pos += [o, o + 0.96 * (math.cos(a) * u + math.sin(a) * v), o + 0.96 * (math.cos(a) * u - math.sin(a) * v)]
    pos.append([0.0, 0.0, 0.0])
    return np.asarray(pos) + rng.normal(scale=0.02, size=(16, 3))


def _box_structure():
    """The 64-water box with one water replaced by the ion at its oxygen."""
    pos = np.concatenate([np.delete(BOX, [0, 1, 2], axis=0), BOX[:1]])
    return pos


def _graph(pos, cell=None, cutoff=30.0):
    z = np.array([8, 1, 1] * ((len(pos) - 1) // 3) + [11])
    s = {"pos": np.asarray(pos, float), "atomic_numbers": z}
    if cell is not None:
        s["cell"], s["pbc"] = np.asarray(cell, float), np.ones(3, dtype=bool)
    return structure_to_graph(s, cutoff=cutoff)


def _cluster_model(ions=True, **kw):
    return Hippo(_library(ions), _topology(5), cutoff=30.0, polar_eps=1e-12, **kw)


def _box_model(ions=True, **kw):
    for key in ("ewald_cutoff", "repulsion_cutoff", "chgtrn_cutoff", "dispersion_cutoff"):
        kw.setdefault(key, 6.0)
    kw.setdefault("pme_grid", 48)
    kw.setdefault("dpme_grid", 48)
    return Hippo(_library(ions), _topology(63), cutoff=6.3, taper=5.4, polar_eps=1e-10, **kw)


def _mbct_energy(c, x, c0):
    """The many-body charge transfer of one ion from its pair charge transfers ``c`` (eV), as written."""
    c = np.asarray(c, float)
    w = c / (c + c0)
    s2 = c.sum() + sum(w[j] * w[k] * math.sqrt(c[j] * c[k]) for j in range(len(c)) for k in range(len(c)) if j != k)
    return x * ((1.0 + x) * s2 / (1.0 + x * w.sum()) - c.sum())


def test_ion_terms_are_their_formulas():
    """The charge transfer of the ion is the sum of its rows over the oxygens, in place of HIPPO's (whose ion
    parameters it ignores); the pair correction adds to the repulsion and the dispersion; the many-body term
    follows from the pair charge transfers; nothing else changes."""
    pos = _cluster()
    g = _graph(pos)
    full, base = _cluster_model(True)(g), _cluster_model(False)(g)
    R = np.linalg.norm(pos[0:15:3] - pos[15], axis=1)
    ct = -CT[0] * KCAL_TO_EV * np.exp(-CT[1] * R)
    x = PAIR[2] * R
    f6 = 1.0 - np.exp(-x) * sum(x ** k / math.factorial(k) for k in range(7))
    rep = PAIR[0] * KCAL_TO_EV * np.exp(-x)
    disp = -PAIR[1] * KCAL_TO_EV * f6 / R ** 6
    c0 = 0.229 / 4.184 * KCAL_TO_EV
    assert float(full["e_chgtrn"] - base["e_chgtrn"]) == pytest.approx(ct.sum(), rel=1e-12)
    assert float(full["e_repulsion"] - base["e_repulsion"]) == pytest.approx(rep.sum(), rel=1e-12)
    assert float(full["e_dispersion"] - base["e_dispersion"]) == pytest.approx(disp.sum(), rel=1e-12)
    assert float(full["e_mbct"]) == pytest.approx(_mbct_energy(-ct, X, c0), rel=1e-12)
    assert float(full["e_mbct"]) > 0.0
    for key in ("e_multipole", "e_polar"):
        assert torch.allclose(full[key], base[key], rtol=1e-13, atol=1e-13), key
    assert "e_mbct" not in base


def _star(c_values, x, c0=0.0, pairs_per_ligand=1):
    """``ion_terms`` of one ion with a ligand molecule per entry of ``c_values`` (its pair charge transfer at
    unit prefactor and zero exponent: ``c = weight``)."""
    n_lig = len(c_values)
    N = 1 + n_lig * pairs_per_ligand
    k = torch.arange(1, N)
    i = torch.zeros_like(k)
    weight = torch.tensor([c for c in c_values for _ in range(pairs_per_ligand)], dtype=torch.float64)
    weight = weight / pairs_per_ligand
    params = {"ionct_a": torch.ones(1), "ionct_b": torch.zeros(1), "mbct_x": torch.tensor([x]),
              "ionpair_a": torch.zeros(1), "ionpair_c": torch.zeros(1), "ionpair_beta": torch.ones(1)}
    cls = torch.tensor([0] + [1] * (N - 1))
    molecule = torch.tensor([0] + [1 + j for j in range(n_lig) for _ in range(pairs_per_ligand)])
    out = ion_terms(i, k, torch.ones(N - 1), weight, cls, N, 1, params, torch.tensor([[-1, 0], [-1, -1]]),
                    torch.full((2, 2), -1), torch.tensor([0, -1]), torch.tensor([0]),
                    torch.tensor([0] + [-1] * (N - 1)), molecule, 1 + n_lig, c0, True, False)
    return float(out["mbct"].sum()), float(out["ct"].sum())


def test_many_body_limits():
    """One bound ligand gives no many-body term, a weak one x^2 c (1 - w) / (1 + x w); n equal bound ligands
    give x n c (n - 1) / (1 + x n), the saturating ratio of charge equilibration; small x the geometric-mean
    three-body sum; a molecule's atoms add their charge transfer before the term."""
    x, c = 0.2, 0.3
    assert _star([c], x, c0=1e-300)[0] == pytest.approx(0.0, abs=1e-15)
    c0 = 0.05
    w = c / (c + c0)
    assert _star([c], x, c0)[0] == pytest.approx(x * x * c * (1 - w) / (1 + x * w), rel=1e-12)
    for n in (2, 4, 6):
        e, ct = _star([c] * n, x, c0=1e-300)
        assert e == pytest.approx(x * n * c * (n - 1) / (1 + x * n), rel=1e-12)
        assert ct == pytest.approx(-n * c, rel=1e-12)
    cs = [0.3, 0.1, 0.45]
    w = [v / (v + c0) for v in cs]
    three = sum(w[j] * w[k] * math.sqrt(cs[j] * cs[k]) for j in range(3) for k in range(j + 1, 3))
    small = 1e-7
    assert _star(cs, small, c0)[0] / small == pytest.approx(2.0 * three, rel=1e-5)
    assert _star(cs, x, c0, pairs_per_ligand=3)[0] == pytest.approx(_star(cs, x, c0)[0], rel=1e-12)


def test_ion_terms_derivatives():
    """First and second derivatives of the ion terms in the positions and the parameters, with molecules that
    transfer nothing (no row) and pairs in the switch."""
    rng = np.random.default_rng(5)
    pos = torch.tensor(np.concatenate([[[0.0, 0.0, 0.0]], rng.uniform(-4.0, 4.0, size=(9, 3))]))
    pos[1:] += 2.0 * pos[1:] / pos[1:].norm(dim=1, keepdim=True)
    k = torch.arange(1, 10)
    i = torch.zeros_like(k)
    # the ion, two waters (O class 1, H class 2) and a molecule without a row (it transfers nothing)
    cls = torch.tensor([0, 1, 2, 2, 1, 2, 2, 2, 2, 2])
    molecule = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    index = torch.full((3, 3), -1)
    index[0, 1] = 0
    names = ("ionct_a", "ionct_b", "ionpair_a", "ionpair_c", "ionpair_beta", "mbct_x")
    values = (torch.tensor([0.4]), torch.tensor([2.0]), torch.tensor([0.3]), torch.tensor([0.2]),
              torch.tensor([3.0]), torch.tensor([0.2]))

    def energy(p, *theta):
        R = (p[k] - p[i]).norm(dim=1)
        x = ((R - 4.0) / 1.0).clamp(0.0, 1.0)
        weight = 1.0 - x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)
        out = ion_terms(i, k, R, weight, cls, 10, 1, dict(zip(names, theta)), index, index,
                        torch.tensor([0, -1, -1]), torch.tensor([0]), torch.tensor([0] + [-1] * 9), molecule, 4,
                        0.01, True, True)
        return torch.stack([out[key].sum() for key in ("ct", "rep", "disp", "mbct")]).sum()

    args = (pos.clone().requires_grad_(True),) + tuple(v.clone().requires_grad_(True) for v in values)
    assert torch.autograd.gradcheck(energy, args)
    assert torch.autograd.gradgradcheck(energy, args)


def test_model_forces_and_stress_match_differences():
    """The forces of the model with the ion terms are the central differences of its energy, in a cluster and
    in a cell, where the stress is the strain derivative."""
    pos = _cluster()
    model = ForceStressOutput(_cluster_model(), compute_stress=False).eval()
    forces = model(_graph(pos))["forces"]
    h = 1e-5
    for atom, axis in ((15, 0), (15, 2), (0, 1), (13, 0)):
        p, m = pos.copy(), pos.copy()
        p[atom, axis] += h
        m[atom, axis] -= h
        fd = -(float(model(_graph(p))["energy"]) - float(model(_graph(m))["energy"])) / (2 * h)
        assert float(forces[atom, axis]) == pytest.approx(fd, rel=1e-6, abs=1e-8)
    box = _box_structure()
    cell_model = ForceStressOutput(_box_model(), compute_stress=True).eval()
    out = cell_model(_graph(box, CELL, 6.3))
    volume = float(np.linalg.det(CELL))
    e = 1e-4                  # the pairs crossing the real-space cutoff stay below the tolerance at this step
    for a, b in ((0, 0), (1, 2)):
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = e
        energies = []
        for sign in (1.0, -1.0):
            strain = np.eye(3) + sign * eps
            energies.append(float(cell_model(_graph(box @ strain, CELL @ strain, 6.3))["energy"]))
        fd = (energies[0] - energies[1]) / (2 * e) / volume / (1.0 if a == b else 2.0)
        assert float(out["stress"][0, a, b]) == pytest.approx(fd, abs=2e-6)


def test_ion_terms_follow_the_cell():
    """In a cell the ion terms take the minimum image: a translated box with every atom wrapped back, or a water
    moved by a lattice vector, has the same terms (the real-space ones: a mesh is not translation invariant)."""
    model = _box_model().eval()
    box = _box_structure()
    shifted = (box + np.array([3.1, -2.2, 5.7])) % np.diag(CELL)
    moved = box.copy()
    moved[30:33] += CELL[1]
    ref = model(_graph(box, CELL, 6.3))
    assert float(ref["e_mbct"]) > 0.0
    for pos in (shifted, moved):
        out = model(_graph(pos, CELL, 6.3))
        for key in ("e_mbct", "e_chgtrn", "e_repulsion"):
            assert float(out[key]) == pytest.approx(float(ref[key]), rel=1e-11), key


@pytest.mark.parametrize("device", _KERNEL_DEVICES)
def test_kernels_match_the_reference_with_ions(device):
    """With the ion terms, the fused kernels give the reference's energy, forces, stress, force-loss second
    derivatives and parameter gradients (the ion's HIPPO charge transfer is off in the kernels' table too)."""
    g = _graph(_box_structure(), CELL, 6.3).to(device)
    results = []
    for use_fast in (False, True):
        model = ForceStressOutput(_box_model(use_fast=use_fast), compute_stress=True).to(device).eval()
        results.append(model(g))
    for key in ("energy", "forces", "stress", "e_chgtrn", "e_mbct"):
        assert torch.allclose(results[1][key], results[0][key], rtol=1e-10, atol=1e-10), key
    weights = torch.tensor(np.random.default_rng(8).normal(size=(16, 3)), device=device)
    grads, second = [], []
    for use_fast in (False, True):
        model = _cluster_model(use_fast=use_fast, trainable=("ionct", "mbct", "ionpair", "chgtrn"))
        model = model.to(device).eval()
        graph = _graph(_cluster()).to(device)
        graph.pos.requires_grad_(True)
        energy = model(graph)["energy"].sum()
        forces = -torch.autograd.grad(energy, graph.pos, create_graph=True)[0]
        second.append(torch.autograd.grad((forces * weights).sum(), graph.pos, retain_graph=True)[0])
        energy.backward()
        grads.append({k: v.grad.clone() for k, v in model.ff.params.items() if v.grad is not None})
    assert torch.allclose(second[1], second[0], rtol=1e-9, atol=1e-9)
    assert set(grads[0]) == set(grads[1]) >= {"ionct_a", "ionct_b", "mbct_x", "ionpair_a", "chgtrn_eps"}
    for key in grads[0]:
        assert torch.allclose(grads[1][key], grads[0][key], rtol=1e-9, atol=1e-10), key
    # the ion's own charge-transfer parameters take no gradient: its rows replace them
    assert float(grads[1]["chgtrn_eps"][model.ff.class_names.index("NA")]) == 0.0


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="records on a GPU"))])
def test_steps_with_ions_equal_fresh_evaluations(device):
    """Steps on a plan, graph replays on a GPU included, equal fresh evaluations while the atoms move and the
    cell changes."""
    ref, model = (ForceStressOutput(_box_model(), compute_stress=True).to(device).eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
    box = _box_structure()
    velocity = np.random.default_rng(7).normal(scale=0.002, size=box.shape)
    for step in range(8):
        pos, cell = box + step * velocity, CELL.copy()
        if step >= 5:
            scale = 1.001 ** (step - 4)
            pos, cell = pos * scale, CELL * scale
        s = {"pos": torch.tensor(pos, device=device), "atomic_numbers": torch.tensor([8, 1, 1] * 63 + [11],
                                                                                     device=device),
             "cell": torch.tensor(cell, device=device), "pbc": torch.ones(3, dtype=torch.bool, device=device)}
        got, want = stepper(s), ref(structure_to_graph(s, 6.3, device=device))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-8), (key, step)


def test_library_files_keep_the_ion_terms(tmp_path):
    """The ion tables and the scale of the ligand weights survive the .frc, Tinker .prm and JSON files, and a
    trained model exports its values."""
    lib = _library()
    lib.settings["mbct-c0"] = 0.07
    lib.save_frc(tmp_path / "ions.frc", name="ions")
    lib.save(tmp_path / "ions.json")
    for back in (read_amoeba(tmp_path / "ions.frc"), read_amoeba(tmp_path / "ions.json")):
        assert back.ionct == lib.ionct and back.mbct == lib.mbct and back.ionpair == lib.ionpair
        assert back.setting("mbct-c0") == pytest.approx(0.07)
    numbers = write_tinker_prm(lib, tmp_path / "ions.prm")["classes"]
    back = read_tinker_prm(tmp_path / "ions.prm")
    na, ow = numbers["NA"], numbers["OW"]
    assert back.ionct[f"{na}-{ow}"] == pytest.approx(lib.ionct["NA-OW"])
    assert back.ionpair[f"{na}-{ow}"] == pytest.approx(lib.ionpair["NA-OW"])
    assert back.mbct[str(na)] == pytest.approx(lib.mbct["NA"])
    assert back.setting("mbct-c0") == pytest.approx(0.07)
    assert builtin_amoeba("hippo-water-2021").setting("mbct-c0") == pytest.approx(0.229 / 4.184)
    model = _cluster_model(trainable=("ionct", "mbct", "ionpair"))
    with torch.no_grad():
        model.ff.params["ionct_a"].mul_(1.5)
        model.ff.params["mbct_x"].add_(0.01)
    out = model.ff.export_library()
    assert out.ionct["NA-OW"]["a"] == pytest.approx(1.5 * CT[0]) and out.mbct["NA"]["x"] == pytest.approx(X + 0.01)
    assert out.ionpair["NA-OW"] == pytest.approx(lib.ionpair["NA-OW"])


def test_ion_terms_need_hippo_single_atoms_and_no_lammps(tmp_path):
    """The ion terms are HIPPO's, for single-atom ions, and LAMMPS's AMOEBA package has none of them."""
    from xnn.ffnn.models import Amoeba
    amoeba = builtin_amoeba("amoeba-water-2003")
    amoeba.mbct["OW"] = {"x": 0.1}
    with pytest.raises(ValueError, match="terms of HIPPO"):
        Amoeba(amoeba, MolecularTopology.from_bonds(*_waters(2)), cutoff=30.0)
    bonded = copy.deepcopy(_library())
    bonded.ionct["HW-OW"] = {"a": 1.0, "b": 1.0}
    with pytest.raises(ValueError, match="single-atom ions"):
        Hippo(bonded, _topology(5), cutoff=30.0)
    with pytest.raises(ValueError, match="ion terms"):
        write_lammps_amoeba_input(_cluster_model(), _cluster(), directory=tmp_path)


def test_a_fresh_evaluation_on_other_meshes_warns():
    """An evaluation off a plan whose cell picks other automatic meshes than the model's steps use warns once;
    with the meshes set it does not."""
    for meshes in ({"pme_grid": None, "dpme_grid": None}, {}):
        model = ForceStressOutput(_box_model(**meshes), compute_stress=True).eval()
        stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
        s = {"pos": torch.tensor(_box_structure()), "atomic_numbers": torch.tensor([8, 1, 1] * 63 + [11]),
             "cell": torch.tensor(CELL), "pbc": torch.ones(3, dtype=torch.bool)}
        for _ in range(2):
            stepper(s)
        grown = _graph(_box_structure(), CELL * 1.1, 6.3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model(grown)
        assert any("meshes" in str(w.message) for w in caught) == bool(meshes)
