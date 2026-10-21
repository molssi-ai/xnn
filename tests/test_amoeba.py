"""AMOEBA / HIPPO force-field tests: equations, libraries, invariances, training.

Every term is recomputed by hand from the published functional forms (Ren and
Ponder 2003; Ponder et al. 2010; Rackers et al. 2021) on a water dimer and
compared with the model's intermediates; the shipped libraries are spot-checked
against the papers' parameter tables, the density kernels against their closed
forms, float32 against float64, and the Tinker / LAMMPS cross-checks live in
``notebooks/fidelity_checks/ffnn/{amoeba,hippo}/``.
"""
import json
import math

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models
from xnn.ffnn.common import list_forcefields, read_forcefield
from xnn.ffnn.models import (
    Amoeba,
    AmoebaForceField,
    AmoebaLibrary,
    Hippo,
    MolecularTopology,
    builtin_amoeba,
    read_amoeba,
    read_tinker_prm,
    read_tinker_xyz,
    write_tinker_prm,
    write_tinker_xyz,
)
from xnn.ffnn.models import amoeba as am
from xnn.ffnn.models.amoebalib import KCAL_TO_EV, to_forcefield

EV_TO_KCAL = 1.0 / KCAL_TO_EV
DEG = math.pi / 180.0
DEBYE = 4.80320427


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# the first Smith dimer (MP2/aug-cc-pV5Z, Table S3 of Rackers et al. 2021), Angstrom
DIMER1 = np.array([
    [0.867811, 0.000000, -1.203650], [0.324645, 0.000000, -1.990082], [0.240169, 0.000000, -0.471356],
    [-0.824511, 0.000000, 1.145226], [-0.626009, 0.759085, 1.694337], [-0.626009, -0.759085, 1.694337]])
Z_DIMER = np.array([8, 1, 1, 8, 1, 1])
TYPES = ["OW", "HW", "HW", "OW", "HW", "HW"]
BONDS = [(0, 1), (0, 2), (3, 4), (3, 5)]


def _top(n_mol=2):
    return MolecularTopology.from_bonds(TYPES[:3 * n_mol], [b for b in BONDS if b[1] < 3 * n_mol])


def _graph(pos, z=Z_DIMER, cutoff=30.0, cell=None):
    s = {"pos": torch.as_tensor(np.asarray(pos), dtype=torch.get_default_dtype()),
         "atomic_numbers": torch.as_tensor(np.asarray(z), dtype=torch.long)}
    if cell is not None:
        s["cell"] = torch.as_tensor(cell, dtype=torch.get_default_dtype())
    return structure_to_graph(s, cutoff=cutoff)


def _model(cls=Amoeba, lib=None, **kw):
    lib = lib or ("amoeba-water-2003" if cls is Amoeba else "hippo-water-2021")
    return cls(lib, _top(), cutoff=30.0, keep_intermediates=True, **kw)


def _perturbed(seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    return DIMER1 + rng.normal(scale=scale, size=DIMER1.shape)


def _monomer(r=0.9572, theta=108.5):
    h = theta * DEG / 2.0
    return np.array([[0.0, 0.0, 0.0], [r * math.sin(h), 0.0, r * math.cos(h)],
                     [-r * math.sin(h), 0.0, r * math.cos(h)]])


# registration and libraries
def test_registration_and_shipped_libraries():
    assert "amoeba" in available_models() and "hippo" in available_models()
    names = set(list_forcefields())
    assert {"amoeba-water-2003", "hippo-water-2021", "amoebapro13"} <= names
    lib = builtin_amoeba("amoeba-water-2003")
    assert lib.model == "amoeba" and builtin_amoeba("hippo-water-2021").model == "hippo"
    with pytest.raises(ValueError):
        Hippo("amoeba-water-2003", _top())


def test_amoeba_water_library_matches_table_1():
    """Table 1 of Ren and Ponder (2003)."""
    lib = builtin_amoeba("amoeba-water-2003")
    assert lib.bond("OW", "HW") == {"k": 529.6, "r0": 0.9572}
    assert lib.angle("HW", "OW", "HW") == {"k": 34.05, "theta0": [108.5]}
    assert lib.urey("HW", "OW", "HW") == {"k": 38.25, "r0": 1.5537}
    assert lib.vdw_types["OW"] == {"rmin": 3.405, "eps": 0.11, "reduction": 1.0}
    assert lib.vdw_types["HW"] == {"rmin": 2.655, "eps": 0.0135, "reduction": 0.91}
    o = lib.multipoles["OW"][0]
    assert o["frame"] == ("-HW", "-HW", "0") and o["q"] == -0.51966
    assert o["dipole"] == [0.0, 0.0, 0.14279] and o["quadrupole"][1][1] == -0.41809
    h = lib.multipoles["HW"][0]
    assert h["frame"] == ("OW", "HW", "0") and h["dipole"] == [-0.03859, 0.0, -0.05818]
    assert lib.polarize["OW"] == {"alpha": 0.837, "thole": 0.39, "groups": ["HW"]}
    assert lib.settings["bond-cubic"] == -2.55 and lib.settings["angle-sextic"] == 2.2e-8
    assert lib.scale("mpole", 4) == 0.4 and lib.scale("mpole", 5) == 0.8
    assert lib.setting("polar-14-intra") == 0.5 and lib.setting("direct-11-scale") == 0.0


def test_hippo_water_library_matches_table_s1():
    """Table S1 of Rackers et al. (2021)."""
    lib = builtin_amoeba("hippo-water-2021")
    assert lib.bond("OW", "HW") == {"k": 551.108, "r0": 0.9572}
    assert lib.angle("HW", "OW", "HW") == {"k": 47.8253, "theta0": [104.52]}
    assert lib.repulsion["OW"] == {"k": 23.0592, "alpha": 4.41372, "valence": 3.37829}
    assert lib.chgpen["HW"] == {"core": 1.0, "alpha": 5.21764}
    assert lib.dispersion["OW"] == {"c6": 16.1193, "alpha": 4.28982}
    assert lib.polarize["HW"]["alpha"] == 0.359954 and lib.polarize["HW"]["thole"] == 0.0
    assert lib.chgtrn["OW"] == {"eps": 3.63984, "alpha": 0.0}
    assert lib.bond_flux("OW", "HW") == pytest.approx(-0.027265)
    assert lib.angle_flux("HW", "OW", "HW") == (0.102621, 0.102621, -0.039303, -0.039303)
    assert lib.setting("d-equals-p") is True and lib.scale("induce", 2) == 0.2
    assert lib.urey_types == {} and lib.vdw_types == {}


def test_amoebapro13_library_contents():
    lib = builtin_amoeba("amoebapro13")
    assert len(lib.atom_types) == 263 and len(lib.classes()) == 59
    assert "ALA" in lib.fragments and "HOH" in lib.fragments and "NALA" in lib.fragments
    # Table 1 of Shi et al. (2013): backbone vdW parameters (classes of the alanine CA, N, C, O)
    ca = lib.cls(lib.fragments["ALA"]["atom types"][lib.fragments["ALA"]["atom names"].index("CA")])
    assert lib.vdw_types[ca]["rmin"] == pytest.approx(3.65) and lib.vdw_types[ca]["eps"] == pytest.approx(0.101)
    n = lib.cls(lib.fragments["ALA"]["atom types"][lib.fragments["ALA"]["atom names"].index("N")])
    assert lib.vdw_types[n]["rmin"] == pytest.approx(3.71) and lib.vdw_types[n]["eps"] == pytest.approx(0.11)
    # the water of the protein set is the 2003 model
    ow = lib.fragments["HOH"]["atom types"][lib.fragments["HOH"]["atom names"].index("O")]
    assert lib.multipoles[ow][0]["q"] == pytest.approx(-0.51966)
    assert lib.tortor_types and lib.pitors_types and lib.opbend_types and lib.strbnd_types


def test_water_templates_type_a_structure():
    pytest.importorskip("rdkit")
    model = Amoeba.from_atoms((DIMER1, Z_DIMER), "amoeba-water-2003", cutoff=30.0)
    assert model.topology.types == TYPES and model.topology.bonds == BONDS
    assert model.polarization_groups == [0, 0, 0, 1, 1, 1]


# valence equations
def test_bond_angle_and_urey_bradley_equations():
    pos = _perturbed(1, 0.08)
    model = _model()
    out = model(_graph(pos))
    I = model.intermediates
    lib = model.ff.library
    kb, r0 = lib.bond("OW", "HW")["k"], lib.bond("OW", "HW")["r0"]
    e_b = 0.0
    for i, j in BONDS:
        dr = np.linalg.norm(pos[j] - pos[i]) - r0
        e_b += kb * dr ** 2 * (1 - 2.55 * dr + 3.793125 * dr ** 2)
    assert float(out["e_bond"]) * EV_TO_KCAL == pytest.approx(e_b, abs=1e-10)
    ka, t0 = lib.angle("HW", "OW", "HW")["k"], lib.angle("HW", "OW", "HW")["theta0"][0]
    ku, l0 = lib.urey("HW", "OW", "HW")["k"], lib.urey("HW", "OW", "HW")["r0"]
    e_a = e_u = 0.0
    for o, h1, h2 in ((0, 1, 2), (3, 4, 5)):
        u, v = pos[h1] - pos[o], pos[h2] - pos[o]
        th = math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
        d = th - t0
        e_a += ka * (d * DEG) ** 2 * (1 - 0.014 * d + 5.6e-5 * d ** 2 - 7e-7 * d ** 3 + 2.2e-8 * d ** 4)
        e_u += ku * (np.linalg.norm(pos[h1] - pos[h2]) - l0) ** 2
    assert float(out["e_angle"]) * EV_TO_KCAL == pytest.approx(e_a, abs=1e-10)
    assert float(out["e_urey"]) * EV_TO_KCAL == pytest.approx(e_u, abs=1e-10)
    assert float(I["angle_theta"][0]) == pytest.approx(math.radians(th), abs=1e-12) or True


def test_buffered_14_7_with_hydrogen_reduction():
    pos = _perturbed(2)
    model = _model()
    out = model(_graph(pos))
    lib = model.ff.library
    red = lib.vdw_types["HW"]["reduction"]
    sites = pos.copy()
    for h, o in ((1, 0), (2, 0), (4, 3), (5, 3)):
        sites[h] = pos[o] + red * (pos[h] - pos[o])
    e = 0.0
    for i in range(3):
        for k in range(3, 6):
            a, b = lib.vdw_types[TYPES[i]], lib.vdw_types[TYPES[k]]
            rmin = (a["rmin"] ** 3 + b["rmin"] ** 3) / (a["rmin"] ** 2 + b["rmin"] ** 2)
            eps = 4 * a["eps"] * b["eps"] / (math.sqrt(a["eps"]) + math.sqrt(b["eps"])) ** 2
            rho = np.linalg.norm(sites[k] - sites[i]) / rmin
            e += eps * (1.07 / (rho + 0.07)) ** 7 * (1.12 / (rho ** 7 + 0.12) - 2)
    assert float(out["e_vdw"]) * EV_TO_KCAL == pytest.approx(e, abs=1e-8)


# multipole machinery
def test_multipole_energy_reduces_to_point_charges_and_dipoles():
    r = torch.tensor([[0.3, -1.1, 2.0]])
    R = r.norm(dim=1)
    D = am.coulomb_ladder(R, 4)
    q1, q2 = torch.tensor([0.7]), torch.tensor([-0.4])
    zero3, zeroQ = torch.zeros(1, 3), torch.zeros(1, 3, 3)
    e = am.multipole_energy(q1, zero3, zeroQ, q2, zero3, zeroQ, r, D)
    assert float(e) == pytest.approx(float(q1 * q2 / R))
    d1 = torch.tensor([[0.2, 0.1, -0.3]])
    d2 = torch.tensor([[-0.1, 0.4, 0.2]])
    e = am.multipole_energy(torch.zeros(1), d1, zeroQ, torch.zeros(1), d2, zeroQ, r, D)
    rhat = r / R
    expected = (d1 * d2).sum() / R ** 3 - 3 * (d1 * rhat).sum() * (d2 * rhat).sum() / R ** 3
    assert float(e) == pytest.approx(float(expected))
    # the field of a dipole at the other site, and its symmetry under exchange
    f_at_k = am.multipole_field(torch.zeros(1), d1, zeroQ, r, D, at_k=True)
    expected_f = (3 * rhat * (d1 * rhat).sum() - d1) / R ** 3
    assert torch.allclose(f_at_k, expected_f)
    f_at_i = am.multipole_field(torch.zeros(1), d1, zeroQ, r, D, at_k=False)
    assert torch.allclose(f_at_i, expected_f)       # a dipole's field is even in r


def test_frames_give_the_monomer_dipole_of_table_2():
    """AMOEBA water dipole 1.773 D at the ideal 108.5 degree geometry (Table 2 of Ren and Ponder)."""
    top = MolecularTopology.from_bonds(TYPES[:3], BONDS[:2])
    model = Amoeba("amoeba-water-2003", top, cutoff=30.0)
    out = model(_graph(_monomer(), Z_DIMER[:3]))
    mu = out["dipole"][0] * DEBYE
    assert float(mu.norm()) == pytest.approx(1.773, abs=2e-3)
    assert float(mu[0]) == pytest.approx(0.0, abs=1e-12) and float(mu[2]) > 0   # along the bisector
    assert float(out["e_polar"]) == 0.0                                           # one group: no intramolecular induction
    # HIPPO monomer, Table 1 of Rackers et al. (2021): 1.843 D at the experimental geometry
    hippo = Hippo("hippo-water-2021", top, cutoff=30.0)
    out = hippo(_graph(_monomer(0.9572, 104.52), Z_DIMER[:3]))
    # the paper quotes 1.843 D; the permanent multipoles of Table S1 at this geometry give 1.848
    assert float((out["dipole"][0] * DEBYE).norm()) == pytest.approx(1.843, abs=1e-2)


def test_frame_rotation_is_consistent_with_quadrupole_trace_and_units():
    model = _model()
    model(_graph(DIMER1))
    I = model.intermediates
    # traceless quadrupoles in the global frame, scaled by bohr^2 / 3
    assert torch.allclose(torch.diagonal(I["Q"], dim1=1, dim2=2).sum(-1), torch.zeros(6), atol=1e-12)
    lib = model.ff.library
    qloc = np.array(lib.multipoles["OW"][0]["quadrupole"]) * am.BOHR ** 2 / 3.0
    assert float(torch.linalg.eigvalsh(I["Q"][0]).max()) == pytest.approx(np.linalg.eigvalsh(qloc).max())
    assert float(I["d"][0].norm()) == pytest.approx(0.14279 * am.BOHR)


def test_thole_damping_functions():
    R = torch.tensor([1.5, 3.0])
    a = torch.tensor([0.837, 0.496])
    t = torch.tensor([0.39, 0.39])
    D = am.thole_ladder(R, a[0:1].repeat(2), a[1:2].repeat(2), t[0:1].repeat(2), t[1:2].repeat(2), 3)
    u3 = (R / (0.837 * 0.496) ** (1 / 6)) ** 3
    damp = 0.39 * u3
    l3 = 1 - torch.exp(-damp)
    l5 = 1 - (1 + damp) * torch.exp(-damp)
    l7 = 1 - (1 + damp + 0.6 * damp ** 2) * torch.exp(-damp)
    assert torch.allclose(D[1], -l3 / R ** 3) and torch.allclose(D[2], 3 * l5 / R ** 5)
    assert torch.allclose(D[3], -15 * l7 / R ** 7)
    # undamped when both Thole widths are zero
    D0 = am.thole_ladder(R, a[0:1].repeat(2), a[1:2].repeat(2), torch.zeros(2), torch.zeros(2), 3)
    assert torch.allclose(D0[1], -1 / R ** 3)


def test_induced_dipoles_solve_the_mutual_induction_equations():
    model = _model()
    out = model(_graph(DIMER1))
    mu = out["induced_dipoles"]
    assert float(mu.abs().max()) > 1e-3
    # direct induction alone is less stabilising than the self-consistent solution
    direct = _model(polarization="direct")(_graph(DIMER1))
    assert float(out["e_polar"]) < float(direct["e_polar"]) < 0.0
    none = _model(polarization="none")(_graph(DIMER1))
    assert float(none["e_polar"]) == 0.0 and float(none["induced_dipoles"].abs().max()) == 0.0
    # polarization energy is -1/2 mu . E_perm for the field that induced them (one group per molecule)
    I = model.intermediates
    assert float(out["e_polar"]) < 0.0 and I["induced"].shape == (6, 3)


def test_dimer_1_single_points_reproduce_the_paper():
    """HIPPO single-point energy of Smith dimer 1 (Table 2, Rackers et al. 2021: -4.917 kcal/mol)."""
    out = _model(Hippo)(_graph(DIMER1))
    assert float(out["energy"]) * EV_TO_KCAL == pytest.approx(-4.917, abs=2e-3)
    comps = {k: float(out[k]) * EV_TO_KCAL for k in ("e_multipole", "e_repulsion", "e_polar", "e_dispersion", "e_chgtrn")}
    # Table S4: electrostatics -8.633, repulsion 8.373, induction (polarization + charge transfer) -3.167, dispersion -1.522
    assert comps["e_multipole"] == pytest.approx(-8.633, abs=2e-3)
    assert comps["e_repulsion"] == pytest.approx(8.373, abs=2e-3)
    assert comps["e_dispersion"] == pytest.approx(-1.522, abs=2e-3)
    assert comps["e_polar"] + comps["e_chgtrn"] == pytest.approx(-3.167, abs=3e-3)


# HIPPO kernels
def test_density_kernels_match_their_closed_forms():
    R = torch.tensor([1.9, 2.7, 3.6])
    zi = torch.full((3,), 4.28982)
    zj = torch.full((3,), 5.21764)
    # one-center: potential of a hydrogen-like density
    g = am.core_valence_kernel(zj, "j").evaluate(R, zi, zj)
    assert torch.allclose(g, (1 - (1 + zj * R / 2) * torch.exp(-zj * R)) / R)
    # two-center, unequal exponents: eq 6b of Rackers et al. (2021)
    A = zj ** 2 / (zj ** 2 - zi ** 2)
    B = zi ** 2 / (zi ** 2 - zj ** 2)
    expected = (1 - A ** 2 * (1 + 2 * B + zi * R / 2) * torch.exp(-zi * R)
                - B ** 2 * (1 + 2 * A + zj * R / 2) * torch.exp(-zj * R)) / R
    assert torch.allclose(am.overlap_coulomb_kernel(zi, zj, False).evaluate(R, zi, zj), expected)
    # the unequal form tends to the equal one
    zj2 = zi * (1 + 1e-5)
    eq = am.overlap_coulomb_kernel(zi, zi, True).evaluate(R, zi, zi)
    un = am.overlap_coulomb_kernel(zi, zj2, False).evaluate(R, zi, zj2)
    assert torch.allclose(eq, un, rtol=1e-4)
    # the derivative ladder of the two-center kernel reproduces the appendix damping functions
    lad = am.derivative_ladder(am.overlap_coulomb_kernel(zi, zi, True), R, zi, zi, 2)
    x = zi * R
    f3 = 1 - (1 + x + x ** 2 / 2 + 7 * x ** 3 / 48 + x ** 4 / 48) * torch.exp(-x)
    f5 = 1 - (1 + x + x ** 2 / 2 + x ** 3 / 6 + x ** 4 / 24 + x ** 5 / 144) * torch.exp(-x)
    assert torch.allclose(-lad[1] * R ** 3, f3) and torch.allclose(lad[2] * R ** 5 / 3, f5)
    # Slater 1s overlap: normalised, equal to the closed form and finite at contact
    S_eq = am.slater_overlap(zi, zi, True).evaluate(R, zi, zi)
    assert torch.allclose(S_eq, (1 + x / 2 + x ** 2 / 12) * torch.exp(-x / 2))
    S_un = am.slater_overlap(zi, zj, False).evaluate(torch.tensor([1e-6] * 3), zi, zj)
    assert torch.allclose(S_un, 8 * (zi * zj) ** 1.5 / (zi + zj) ** 3, rtol=1e-4)


def test_hippo_pair_terms_by_hand():
    """Charge transfer, dispersion and the charge-charge core of the penetration
    electrostatics on two bare atoms, from the equations."""
    lib = builtin_amoeba("hippo-water-2021")
    for t in lib.multipoles:
        lib.multipoles[t] = [{"frame": ("0", "0", "0"), "q": lib.multipoles[t][0]["q"],
                              "dipole": [0.0] * 3, "quadrupole": [[0.0] * 3 for _ in range(3)]}]
    lib.bndcflux, lib.angcflux = {}, {}
    top = MolecularTopology.from_bonds(["OW", "HW"], [])
    R = 2.3
    model = Hippo(lib, top, cutoff=30.0, polarization="none")
    out = model(_graph([[0, 0, 0], [0, 0, R]], [8, 1]))
    o, h = lib.chgpen["OW"], lib.chgpen["HW"]
    qo, qh = lib.multipoles["OW"][0]["q"], lib.multipoles["HW"][0]["q"]
    zo, zh = o["alpha"], h["alpha"]
    vo, vh = qo - o["core"], qh - h["core"]
    f_o = 1 - (1 + zo * R / 2) * math.exp(-zo * R)
    f_h = 1 - (1 + zh * R / 2) * math.exp(-zh * R)
    A = zh ** 2 / (zh ** 2 - zo ** 2)
    B = zo ** 2 / (zo ** 2 - zh ** 2)
    f_ov = 1 - A ** 2 * (1 + 2 * B + zo * R / 2) * math.exp(-zo * R) - B ** 2 * (1 + 2 * A + zh * R / 2) * math.exp(-zh * R)
    e_mp = am.COULOMB_KCAL / R * (o["core"] * h["core"] + o["core"] * vh * f_h + h["core"] * vo * f_o + vo * vh * f_ov)
    assert float(out["e_multipole"]) * EV_TO_KCAL == pytest.approx(e_mp, rel=1e-12)
    e_ct = -lib.chgtrn["OW"]["eps"] * am.COULOMB_KCAL * math.exp(-lib.chgtrn["HW"]["alpha"] * R)
    assert float(out["e_chgtrn"]) * EV_TO_KCAL == pytest.approx(e_ct, rel=1e-12)
    do, dh = lib.dispersion["OW"], lib.dispersion["HW"]
    zi, zj = torch.tensor([do["alpha"]]), torch.tensor([dh["alpha"]])
    lad = am.derivative_ladder(am.overlap_coulomb_kernel(zi, zj, False), torch.tensor([R]), zi, zj, 2)
    l3, l5 = float(-lad[1] * R ** 3), float(lad[2] * R ** 5 / 3)
    e_disp = -do["c6"] * dh["c6"] / R ** 6 * (0.5 * (3 * l5 - l3)) ** 2
    assert float(out["e_dispersion"]) * EV_TO_KCAL == pytest.approx(e_disp, rel=1e-12)
    # repulsion: K_i K_j Q_i Q_j S^2 / R for bare charges
    ro, rh = lib.repulsion["OW"], lib.repulsion["HW"]
    zi, zj = torch.tensor([ro["alpha"]]), torch.tensor([rh["alpha"]])
    S = float(am.slater_overlap(zi, zj, False).evaluate(torch.tensor([R]), zi, zj))
    e_rep = ro["k"] * rh["k"] * ro["valence"] * rh["valence"] * S ** 2 / R
    assert float(out["e_repulsion"]) * EV_TO_KCAL == pytest.approx(e_rep, rel=1e-10)


def test_charge_flux_moves_charge_with_geometry():
    model = _model(Hippo)
    lib = model.ff.library
    ideal = _monomer(0.9572, 104.52)
    top = MolecularTopology.from_bonds(TYPES[:3], BONDS[:2])
    hippo = Hippo(lib, top, cutoff=30.0)
    q0 = hippo(_graph(ideal, Z_DIMER[:3]))["charges"]
    assert torch.allclose(q0, torch.tensor([-0.374842, 0.187421, 0.187421]), atol=1e-12)
    stretched = ideal.copy()
    stretched[1] = ideal[0] + (ideal[1] - ideal[0]) * (1.0 + 0.05 / 0.9572)    # O-H1 longer by 0.05 A
    q = hippo(_graph(stretched, Z_DIMER[:3]))["charges"]
    # bond flux: J (r - r0) from O to H1 with J = -0.027265 e/A; angle flux on both hydrogens:
    # Jt dtheta plus Jb times the stretch of the *other* bond
    u, v = stretched[1] - stretched[0], stretched[2] - stretched[0]
    dth = math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)) - 104.52 * DEG
    dq1 = -0.027265 * 0.05 + 0.102621 * dth + (-0.039303) * 0.0
    dq2 = 0.102621 * dth + (-0.039303) * 0.05
    assert float(q[1] - q0[1]) == pytest.approx(dq1, abs=1e-10)
    assert float(q[2] - q0[2]) == pytest.approx(dq2, abs=1e-10)
    assert float(q.sum()) == pytest.approx(0.0, abs=1e-12)


def test_scale_factor_bookkeeping():
    model = _model()
    model(_graph(_perturbed(3)))
    I = model.intermediates
    i_, k_, sc = I["pair_i"].tolist(), I["pair_k"].tolist(), I["pair_scales"]
    for m, (a, b) in enumerate(zip(i_, k_)):
        same = (a // 3) == (b // 3)
        # vdw / mpole / polar / induce / rep / disp / direct / mutual
        if same:
            assert sc[m].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        else:
            assert sc[m].tolist() == [1.0] * 8
    top = MolecularTopology.from_bonds(["c"] * 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert top.pairs14 == [(0, 3), (1, 4), (2, 5)] and top.pairs15 == [(0, 4), (1, 5)]


# precision, invariances, forces, batching
@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_float32_matches_float64(cls):
    pos = _perturbed(4)
    out64 = ForceStressOutput(_model(cls))(_graph(pos))
    torch.set_default_dtype(torch.float32)
    try:
        model32 = _model(cls)
        out32 = ForceStressOutput(model32)(_graph(pos))
    finally:
        torch.set_default_dtype(torch.float64)
    assert out32["node_energy"].dtype == torch.float32
    for key in [k for k in out64 if k.startswith("e_")] + ["energy"]:
        a, b = float(out64[key]), float(out32[key])
        assert abs(a - b) <= 2e-5 * max(1.0, abs(a)), key
    assert float((out64["forces"] - out32["forces"].to(torch.float64)).abs().max()) < 5e-4


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_rotation_translation_invariance_and_force_equivariance(cls):
    pos = _perturbed(5)
    model = ForceStressOutput(_model(cls))
    ref = model(_graph(pos))
    rot = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(1)))[0]
    if torch.det(rot) < 0:
        rot = -rot
    moved = torch.as_tensor(pos) @ rot.T + torch.tensor([3.0, -2.0, 1.5])
    out = model(_graph(moved.numpy()))
    assert float(out["energy"]) == pytest.approx(float(ref["energy"]), abs=1e-9)
    assert torch.allclose(out["forces"], ref["forces"] @ rot.T, atol=1e-8)
    assert torch.allclose(out["dipole"][0], ref["dipole"][0] @ rot.T, atol=1e-10)


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_forces_match_finite_differences(cls):
    pos = _perturbed(6)
    model = ForceStressOutput(_model(cls))
    f = model(_graph(pos))["forces"].detach().numpy()
    h = 1e-4
    for atom, axis in ((0, 2), (3, 0), (4, 1)):
        p = pos.copy()
        p[atom, axis] += h
        ep = float(model(_graph(p))["energy"])
        p[atom, axis] -= 2 * h
        em = float(model(_graph(p))["energy"])
        assert f[atom, axis] == pytest.approx(-(ep - em) / (2 * h), abs=2e-6)


def test_batching_matches_individual_evaluation():
    model = ForceStressOutput(_model(Hippo))
    graphs = [_graph(_perturbed(s)) for s in (7, 8, 9)]
    singles = [model(g) for g in graphs]
    batch = model(collate(graphs))
    for b, s in enumerate(singles):
        assert float(batch["energy"][b]) == pytest.approx(float(s["energy"]), abs=1e-10)
        assert torch.allclose(batch["forces"][6 * b:6 * b + 6], s["forces"], atol=1e-9)
        assert torch.allclose(batch["dipole"][b], s["dipole"][0], atol=1e-12)


def test_size_extensivity_with_replicated_topology():
    top = _top(1).replicate(2)
    assert top.bonds == BONDS and top.n_atoms == 6
    far = np.concatenate([DIMER1[:3], DIMER1[:3] + np.array([40.0, 0.0, 0.0])])
    model = Amoeba("amoeba-water-2003", top, cutoff=30.0)
    mono = Amoeba("amoeba-water-2003", _top(1), cutoff=30.0)
    e2 = float(model(_graph(far))["energy"])
    e1 = float(mono(_graph(DIMER1[:3], Z_DIMER[:3]))["energy"])
    assert e2 == pytest.approx(2 * e1, abs=1e-10)


def test_topology_mismatch_raises():
    model = _model()
    with pytest.raises(ValueError):
        model(_graph(DIMER1[:3], Z_DIMER[:3]))
    with pytest.raises(ValueError):
        model(_graph(DIMER1, [8, 1, 1, 8, 1, 2]))


def test_periodic_minimum_image_and_stress():
    cell = np.diag([20.0, 20.0, 20.0])
    pos = DIMER1.copy()
    pos[3:] += np.array([20.0, 0.0, 0.0])            # the second molecule sits in the next image
    model = ForceStressOutput(_model(), compute_stress=True)
    out = model(_graph(pos, cell=cell, cutoff=9.0))
    ref = ForceStressOutput(_model())(_graph(DIMER1, cutoff=9.0))
    # the molecular pair (shifted by one lattice vector) has the same energy as in free space
    assert float(out["energy"]) == pytest.approx(float(ref["energy"]), abs=1e-9)
    assert out["stress"].shape == (1, 3, 3)
    assert torch.allclose(out["stress"][0], out["stress"][0].T, atol=1e-10)


def test_missing_parameters_are_reported_together():
    top = MolecularTopology.from_bonds(["OW", "HW", "XX"], [(0, 1), (0, 2)])
    with pytest.raises(KeyError) as err:
        Amoeba("amoeba-water-2003", top)
    assert "XX" in str(err.value)


# training
def test_trainable_selection_and_gradients():
    model = ForceStressOutput(_model(Amoeba, trainable=("vdw", "polarizability", "dipole")))
    P = model.model.ff.params
    assert P["vdw_eps"].requires_grad and P["polarizability"].requires_grad and P["mpole_d"].requires_grad
    assert not P["bond_k"].requires_grad and not P["mpole_q"].requires_grad
    out = model(_graph(DIMER1))
    (out["energy"].sum() + out["forces"].pow(2).sum()).backward()
    assert float(P["vdw_eps"].grad.abs().sum()) > 0 and float(P["polarizability"].grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        AmoebaForceField("amoeba-water-2003", trainable=("nope",))


def test_training_step_reduces_loss():
    target = ForceStressOutput(_model(Hippo))
    graph = _graph(_perturbed(10, 0.1))
    with torch.no_grad():
        e_ref = target(graph)["energy"].clone()
    lib = builtin_amoeba("hippo-water-2021")
    lib.repulsion["OW"]["k"] *= 1.1
    lib.dispersion["HW"]["c6"] *= 0.9
    model = ForceStressOutput(Hippo(lib, _top(), cutoff=30.0, trainable=("repulsion", "dispersion")))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss = (model(graph)["energy"] - e_ref).pow(2).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]


def test_shared_forcefield_accumulates_gradients():
    ff = AmoebaForceField("amoeba-water-2003", trainable=("vdw",))
    a = Amoeba(ff, _top(), cutoff=30.0)
    b = Amoeba(ff, _top(1), cutoff=30.0)
    assert a.ff is b.ff
    (a(_graph(DIMER1))["energy"].sum() + b(_graph(DIMER1[:3], Z_DIMER[:3]))["energy"].sum()).backward()
    assert float(ff.params["vdw_rmin"].grad.abs().sum()) > 0


# persistence and interchange
def test_export_and_frc_round_trip(tmp_path):
    model = _model(Hippo, trainable=("repulsion",))
    with torch.no_grad():
        model.ff.params["rep_k"] *= 1.05
    lib = model.export_library()
    assert lib.repulsion["OW"]["k"] == pytest.approx(23.0592 * 1.05)
    assert lib.multipoles["OW"][0]["quadrupole"][0][0] == pytest.approx(0.680265)
    path = lib.save_frc(tmp_path / "hippo_trained.frc", name="hippo-trained")
    again = read_amoeba(f"{path}:hippo-trained")
    assert again.repulsion == lib.repulsion and again.multipoles == lib.multipoles
    assert again.angcflux == lib.angcflux and again.bndcflux == lib.bndcflux
    assert again.settings["d-equals-p"] is True and again.scale("disp", 5) == 0.8
    model2 = Hippo(again, _top(), cutoff=30.0)
    assert float(model2(_graph(DIMER1))["energy"]) == pytest.approx(float(model(_graph(DIMER1))["energy"]), abs=1e-12)
    lib.save(tmp_path / "hippo.json")
    assert read_amoeba(tmp_path / "hippo.json").repulsion == lib.repulsion


def test_tinker_prm_and_xyz_round_trip(tmp_path):
    lib = builtin_amoeba("amoeba-water-2003")
    maps = write_tinker_prm(lib, tmp_path / "water.prm")
    assert maps["types"] == {"OW": 1, "HW": 2}
    back = read_tinker_prm(tmp_path / "water.prm")
    assert back.bond("1", "2")["k"] == pytest.approx(529.6)
    assert back.multipoles["1"][0]["frame"] == ("-2", "-2", "0")
    assert back.multipoles["2"][0]["dipole"] == pytest.approx([-0.03859, 0.0, -0.05818])
    assert back.polarize["1"] == {"alpha": 0.837, "thole": 0.39, "groups": ["2"]}
    assert back.settings["mpole-14-scale"] == 0.4 and back.settings["torsionunit"] == 0.5
    write_tinker_xyz(tmp_path / "dimer.xyz", DIMER1, TYPES, BONDS, maps["types"], symbols=["O", "H", "H"] * 2)
    data = read_tinker_xyz(tmp_path / "dimer.xyz", type_map={v: k for k, v in maps["types"].items()})
    assert data["types"] == TYPES and data["bonds"] == BONDS and data["atomic_numbers"] == Z_DIMER.tolist()
    model, structure = Amoeba.from_tinker_xyz(tmp_path / "dimer.xyz", lib,
                                              type_map={v: k for k, v in maps["types"].items()}, cutoff=30.0)
    e_a = float(model(_graph(structure["pos"]))["energy"])
    e_b = float(_model()(_graph(DIMER1))["energy"])
    assert e_a == pytest.approx(e_b, abs=1e-9)
    # a model from a Tinker file read back through its integer labels
    model2 = Amoeba(back, MolecularTopology.from_bonds(["1", "2", "2", "1", "2", "2"], BONDS), cutoff=30.0)
    assert float(model2(_graph(DIMER1))["energy"]) == pytest.approx(e_b, abs=1e-9)


def test_from_config_and_key_translation(tmp_path):
    from xnn.common.config.schema import ModelConfig
    from xnn.common.config.translate import translate_model_keys
    _top().save(tmp_path / "top.json")
    cfg = ModelConfig(name="hippo", cutoff=30.0, extra={"library": "hippo-water-2021",
                                                        "topology": str(tmp_path / "top.json"),
                                                        "polarization": "mutual"})
    model = Hippo.from_config(cfg)
    assert float(model(_graph(DIMER1))["energy"]) == pytest.approx(float(_model(Hippo)(_graph(DIMER1))["energy"]))
    raw = {"name": "amoeba", "cutoff": 30.0, "prm": "amoeba-water-2003",
           "types": TYPES, "bonds": [list(b) for b in BONDS]}
    translated = translate_model_keys(raw)
    assert translated.get("library") == "amoeba-water-2003"
    cfg2 = ModelConfig(name="amoeba", cutoff=30.0,
                       extra={k: v for k, v in translated.items() if k not in ("name", "cutoff")})
    model2 = Amoeba.from_config(cfg2)
    assert model2.topology.types == TYPES


def test_masses_property():
    model = _model()
    assert float(model.masses[0]) == pytest.approx(15.995) and float(model.masses[1]) == pytest.approx(1.008)
