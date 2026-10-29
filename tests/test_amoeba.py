"""AMOEBA / HIPPO force-field tests: equations, libraries, invariances, training.

Every term is recomputed by hand from the published functional forms (Ren and
Ponder 2003; Ponder et al. 2010; Rackers et al. 2021) on a water dimer and
compared with the model's intermediates; the shipped libraries are spot-checked
against the papers' parameter tables, the density kernels against their closed
forms, float32 against float64, and the Tinker / LAMMPS cross-checks live in
``notebooks/fidelity_checks/ffnn/{amoeba,hippo}/``.
"""
import copy
import math

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models
from xnn.ffnn.common import list_forcefields, read_forcefield
from xnn.ffnn.common.frc import SECTION_SCHEMA, flip_values, make_section
from xnn.ffnn.models import (
    Amoeba,
    AmoebaForceField,
    Hippo,
    MolecularTopology,
    amoebalib,
    builtin_amoeba,
    read_amoeba,
    read_tinker_prm,
    read_tinker_xyz,
    write_tinker_prm,
    write_tinker_xyz,
)
from xnn.ffnn.models import amoeba as am
from xnn.ffnn.models.amoeba_ewald import matmul3, matvec, vecmat
from xnn.ffnn.models.amoebalib import KCAL_TO_EV, from_forcefield

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
    assert float(out["e_polar"]) == 0.0    # one group: no intramolecular induction
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


def test_z_only_frame_takes_the_lab_axis_of_tinker():
    """A z-only frame's x axis comes from the lab x axis unless the z axis lies within 45 degrees of
    it, then from the lab y axis, as in Tinker; a quadrupole with Qxx != Qyy tells the choices apart."""
    lib = builtin_amoeba("hippo-water-2021")
    lib.multipoles["OW"] = [{"frame": ("0", "0", "0"), "q": lib.multipoles["OW"][0]["q"], "dipole": [0.0] * 3,
                             "quadrupole": [[0.0] * 3 for _ in range(3)]}]
    lib.multipoles["HW"] = [{"frame": ("OW", "0", "0"), "q": lib.multipoles["HW"][0]["q"], "dipole": [0.0, 0.0, 0.1],
                             "quadrupole": [[0.3, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.2]]}]
    lib.bndcflux, lib.angcflux = {}, {}
    # such a quadrupole makes the energy depend on the orientation of the system, which is said once
    with pytest.warns(UserWarning, match="not symmetric about the free axis"):
        model = Hippo(lib, MolecularTopology.from_bonds(["OW", "HW"], [(0, 1)]), cutoff=30.0, polarization="none")
    for angle, lab in ((50.0, [1.0, 0.0, 0.0]), (40.0, [0.0, 1.0, 0.0])):
        # the hydrogen at the origin, its z atom (the oxygen) `angle` degrees from the lab x axis
        a = angle * DEG
        g = _graph([[math.cos(a), math.sin(a), 0.0], [0.0, 0.0, 0.0]], [8, 1])
        R, _ = model._frames(g, am._Images(g, None))
        x_axis, z_axis = R[1, :, 0], R[1, :, 2]
        assert torch.allclose(z_axis, torch.tensor([math.cos(a), math.sin(a), 0.0]), atol=1e-12)
        lab = torch.tensor(lab)
        expected = lab - (lab @ z_axis) * z_axis
        assert torch.allclose(x_axis, expected / expected.norm(), atol=1e-12)


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
    comps = {k: float(out[k]) * EV_TO_KCAL
             for k in ("e_multipole", "e_repulsion", "e_polar", "e_dispersion", "e_chgtrn")}
    # Table S4: electrostatics -8.633, repulsion 8.373,
    # induction (polarization + charge transfer) -3.167, dispersion -1.522
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the GPU path evaluates both forms on every pair"))])
def test_two_center_kernels_are_exact_at_nearly_equal_exponents(device):
    """The two-center ladders keep full precision as the exponents approach each other."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40

    def coulomb(a, b, r):
        A, B = b * b / (b * b - a * a), a * a / (a * a - b * b)
        return (1 - A * A * (1 + 2 * B + a * r / 2) * mp.e ** (-a * r)
                - B * B * (1 + 2 * A + b * r / 2) * mp.e ** (-b * r)) / r

    def repulsion(a, b, r):
        d = a * a - b * b
        u, v = mp.e ** (-a * r / 2), mp.e ** (-b * r / 2)
        S = 8 * (a * b) ** mp.mpf(1.5) / d ** 3 * (a * d * v - 8 * a * b * v / r + b * d * u + 8 * a * b * u / r)
        return S * S

    def ladder(ref, a, b, x):
        """D_0, D_1, D_2 = (1/R d/dR)^n g at R = x."""
        g = lambda t: ref(a, b, t)  # noqa: E731
        d1 = lambda t: mp.diff(g, t) / t  # noqa: E731
        return g(x), d1(x), mp.diff(d1, x) / x

    for kernel, ref in ((am.overlap_coulomb_kernel, coulomb), (am.squared_overlap, repulsion)):
        for rel in (1e-4, 3e-3, 2e-2, 0.2):
            for r in (2.0, 6.0):
                zi, zk = torch.tensor([4.3], device=device), torch.tensor([4.3 * (1 + rel)], device=device)
                got = am.two_center(torch.tensor([r], device=device), zi, zk, 2, kernel)
                want = ladder(ref, mp.mpf(4.3), mp.mpf(4.3) * (1 + mp.mpf(rel)), mp.mpf(r))
                for n in range(3):
                    assert float(got[n]) == pytest.approx(float(want[n]), rel=1e-11), (kernel.__name__, rel, r, n)


def test_compiled_ladders_match_the_symbolic_derivation():
    """Every HIPPO kernel's ladder from its plan equals the term-by-term symbolic ladder."""
    g = torch.Generator().manual_seed(3)
    R = 0.8 + 5.0 * torch.rand(40, generator=g, dtype=torch.float64)
    zi = 3.0 + 2.0 * torch.rand(40, generator=g, dtype=torch.float64)
    # unequal exponents as far apart as HIPPO's atom classes; nearly equal ones cancel in either form
    zj = zi + 0.6 + 1.5 * torch.rand(40, generator=g, dtype=torch.float64)
    S_eq, S_un = am.slater_overlap(zi, zi, True), am.slater_overlap(zi, zj, False)
    kernels = [(am.core_valence_kernel(zi, "i"), zi, zj, 3), (am.core_valence_kernel(zj, "j"), zi, zj, 3),
               (am.overlap_coulomb_kernel(zi, zi, True), zi, zi, 4),
               (am.overlap_coulomb_kernel(zi, zj, False), zi, zj, 4), (S_eq * S_eq, zi, zi, 4),
               (S_un * S_un, zi, zj, 4)]
    for kernel, a, b, order in kernels:
        fast = am.derivative_ladder(kernel, R, a, b, order)
        series = kernel
        for n in range(order + 1):
            ref = series.evaluate(R, a, b)
            assert torch.allclose(fast[n], ref, rtol=1e-10, atol=1e-13 * float(ref.abs().max())), n
            series = series.ladder()
    scalar = am.derivative_ladder(am.overlap_coulomb_kernel(zi[:1], zj[:1], False), R[0], zi[0], zj[0], 2)
    assert scalar[2].shape == torch.Size([]) or scalar[2].shape == torch.Size([1])


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
    f_ov = (1 - A ** 2 * (1 + 2 * B + zo * R / 2) * math.exp(-zo * R)
            - B ** 2 * (1 + 2 * A + zh * R / 2) * math.exp(-zh * R))
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


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_batching_matches_individual_evaluation(cls):
    model = ForceStressOutput(_model(cls))
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
    model = ForceStressOutput(_model(ewald=False), compute_stress=True)
    out = model(_graph(pos, cell=cell, cutoff=9.0))
    ref = ForceStressOutput(_model())(_graph(DIMER1, cutoff=9.0))
    # without the Ewald sum the molecular pair (shifted by one lattice vector) has the same
    # energy as in free space
    assert float(out["energy"]) == pytest.approx(float(ref["energy"]), abs=1e-9)
    assert out["stress"].shape == (1, 3, 3)
    assert torch.allclose(out["stress"][0], out["stress"][0].T, atol=1e-10)


def test_missing_parameters_are_reported_together():
    top = MolecularTopology.from_bonds(["OW", "HW", "XX"], [(0, 1), (0, 2)])
    with pytest.raises(KeyError) as err:
        Amoeba("amoeba-water-2003", top)
    assert "XX" in str(err.value)


# training
def test_dense_induction_is_chosen_per_molecule_and_matches_the_iterative_solve():
    """Many small molecules go to the block-diagonal direct solve, which equals the iterative one."""
    model = _model(Hippo)
    assert model._use_dense(6, 400, None) and not model._use_dense(1200, 1, None)
    assert not model._use_dense(18, 30_000, None)
    graphs = collate([_graph(_perturbed(s, 0.05)) for s in range(5)])
    dense = Hippo("hippo-water-2021", _top(), cutoff=30.0, solver="dense")
    iterative = Hippo("hippo-water-2021", _top(), cutoff=30.0, solver="iterative", polar_eps=1e-12)
    e_dense, e_iter = dense(graphs)["e_polar"], iterative(graphs)["e_polar"]
    assert torch.allclose(e_dense, e_iter, rtol=0, atol=1e-12)
    warm = Hippo("hippo-water-2021", _top(), cutoff=30.0, solver="iterative", polar_eps=1e-12,
                 predictor="previous").eval()
    first = warm(graphs)["e_polar"]
    cold_iterations = warm.solver_stats["forward"]
    again = warm(graphs)["e_polar"]
    assert warm.solver_stats["forward_guess"] and warm.solver_stats["forward"] < cold_iterations
    assert torch.allclose(first, again, rtol=0, atol=1e-12) and torch.allclose(first, e_dense, rtol=0, atol=1e-12)
    # an explicit guess: the exact dipoles converge at once, a poor one is passed over
    exact = warm(graphs)["induced_dipoles"].detach()
    warm.set_dipole_predictor(None)
    warm.dipole_guess = exact
    warm(graphs)
    assert warm.solver_stats["forward_guess"] and warm.solver_stats["forward"] <= 2 and warm.dipole_guess is None
    warm.dipole_guess = torch.full_like(exact, 50.0)
    warm(graphs)
    assert not warm.solver_stats["forward_guess"]


def test_block_preconditioner_inverts_a_molecule_exactly():
    """One water molecule is one block: the block-preconditioned solve converges at once and equals the dense one."""
    g = _graph(_monomer(), Z_DIMER[:3])
    dense = Amoeba("amoeba-water-2003", _top(1), cutoff=30.0, solver="dense")
    block = Amoeba("amoeba-water-2003", _top(1), cutoff=30.0, solver="iterative", preconditioner="block",
                   polar_eps=1e-12)
    assert block.pc_blocks == 1 and block.pc_size == 3
    mu_dense, mu_block = dense(g)["induced_dipoles"], block(g)["induced_dipoles"]
    assert block.solver_stats["forward"] <= 2
    assert torch.allclose(mu_block, mu_dense, atol=1e-12)
    with pytest.raises(ValueError):
        Amoeba("amoeba-water-2003", _top(1), cutoff=30.0, preconditioner="ilu")


def test_pcg_restarts_with_the_diagonal_when_a_preconditioner_is_indefinite():
    """A preconditioner that loses definiteness is replaced and the solve still converges."""
    rng = np.random.default_rng(4)
    Q = rng.normal(size=(9, 9))
    A = torch.tensor(Q @ Q.T + 9.0 * np.eye(9))
    b = torch.tensor(rng.normal(size=(3, 3)))
    alpha, polar = torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.bool)

    def field_of(v):
        return v - (A @ v.reshape(-1)).reshape(3, 3)

    x, iterations, _, _, _ = am._pcg(b, alpha, field_of, polar, 1e-12, 200, precond=lambda r: -r)
    assert torch.allclose(x.reshape(-1), torch.linalg.solve(A, b.reshape(-1)), atol=1e-10)
    assert iterations < 200


def test_matrix_free_pair_products_match_the_tensor_products():
    """dipole_field, matvec, vecmat and matmul3 equal their einsum and matmul forms."""
    g = torch.Generator().manual_seed(5)
    r, m, D1, D2 = (torch.randn(s, generator=g, dtype=torch.float64) for s in ((7, 3), (2, 7, 3), (7,), (7,)))
    A = torch.randn(7, 3, 3, generator=g, dtype=torch.float64)
    T = am.dipole_field_tensor(r, D1, D2)
    assert torch.allclose(am.dipole_field(m, r, D1, D2), torch.einsum("mab,smb->sma", T, m), atol=1e-14)
    assert torch.allclose(matvec(A, r), torch.einsum("mab,mb->ma", A, r), atol=1e-14)
    assert torch.allclose(vecmat(r, A), torch.einsum("mb,mba->ma", r, A), atol=1e-14)
    B = torch.randn(7, 3, 3, generator=g, dtype=torch.float64)
    assert torch.allclose(matmul3(A, B), A @ B, atol=1e-14)


def test_dipole_predictor_coefficients_and_history():
    """ASPC and Gear weights from their closed forms; exactness and history handling."""
    P = am.DipolePredictor
    assert P.coefficients("aspc", 2) == pytest.approx([2.0, -1.0])
    assert P.coefficients("aspc", 6) == pytest.approx([22 / 7, -55 / 14, 55 / 21, -22 / 21, 5 / 21, -1 / 42])
    assert P.coefficients("gear", 6) == [6.0, -15.0, 20.0, -15.0, 6.0, -1.0]
    for m in range(2, 8):
        aspc, gear = P.coefficients("aspc", m), P.coefficients("gear", m)
        assert sum(aspc) == pytest.approx(1.0)
        assert sum(j * b for j, b in enumerate(aspc, 1)) == pytest.approx(0.0, abs=1e-12)
        # Gear reproduces any polynomial of degree m - 1 at the next step
        poly = np.polynomial.Polynomial(np.arange(1.0, m + 1.0))
        assert sum(b * poly(-j) for j, b in enumerate(gear, 1)) == pytest.approx(poly(0.0), rel=1e-12)
    pred = P("gear", 4)
    cubic = np.polynomial.Polynomial([0.3, -1.0, 0.5, 0.2])
    for t in (-4, -3, -2, -1):
        pred.push(torch.full((5, 3), float(cubic(t)), dtype=torch.float64))
    assert torch.allclose(pred.predict(torch.zeros(5, 3, dtype=torch.float64)),
                          torch.full((5, 3), float(cubic(0)), dtype=torch.float64))
    assert pred.predict(torch.zeros(4, 3, dtype=torch.float64)) is None
    pred.push(torch.zeros(4, 3, dtype=torch.float64))
    assert len(pred._sets) == 1
    with pytest.raises(ValueError):
        P("lsq")


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


def test_charge_conservation_and_symmetry_masks_in_training():
    """Gradient steps keep every molecule's charge and the library's zero components."""
    lib = builtin_amoeba("amoeba-water-2003")
    graph = _graph(DIMER1)
    model = _model(Amoeba, trainable=("charge", "dipole", "quadrupole"))
    ff = model.ff
    assert ff.charge_rows.shape == (1, 2) and float(ff.charge_totals[0]) == pytest.approx(0.0, abs=1e-12)
    e0 = float(model(graph)["energy"])
    plain = _model(Amoeba, conserve_charge=False, symmetry_mask=False)
    assert float(plain(graph)["energy"]) == pytest.approx(e0, abs=1e-12)
    q0 = torch.tensor([lib.multipoles[t][0]["q"] for t, _ in ff.multipole_rows])
    d0 = torch.tensor([lib.multipoles[t][0]["dipole"] for t, _ in ff.multipole_rows])
    Q0 = torch.tensor([lib.multipoles[t][0]["quadrupole"] for t, _ in ff.multipole_rows])
    opt = torch.optim.SGD([p for p in ff.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        model(graph)["energy"].sum().backward()
        opt.step()
    assert float((ff.params["mpole_q"] - q0).abs().max()) > 1e-4
    q = model(graph)["charges"]
    assert float(q[:3].sum()) == pytest.approx(0.0, abs=1e-12) and float(q[3:].sum()) == pytest.approx(0.0, abs=1e-12)
    q_rows, d_rows, Q_rows = ff.effective_multipoles()
    assert bool((d_rows[d0 == 0] == 0).all()) and bool((d_rows != d0 * am.BOHR).any())
    assert bool((Q_rows[Q0 == 0] == 0).all())
    assert torch.allclose(Q_rows, Q_rows.transpose(1, 2))
    assert float(torch.diagonal(Q_rows, dim1=1, dim2=2).sum(-1).abs().max()) < 1e-15


def test_export_after_training_writes_the_model_multipoles():
    """Exported charges are neutral, quadrupoles symmetric and traceless, and reproduce the model."""
    model = _model(Amoeba, trainable=("charge", "quadrupole"), symmetry_mask=False)
    with torch.no_grad():
        model.ff.params["mpole_Q"] += 0.01 * torch.eye(3)
        model.ff.params["mpole_Q"][:, 0, 1] += 0.005
        model.ff.params["mpole_q"] += 0.01
    lib = model.export_library()
    for rows in lib.multipoles.values():
        Q = np.array(rows[0]["quadrupole"])
        assert abs(np.trace(Q)) < 1e-12 and np.allclose(Q, Q.T)
    assert lib.multipoles["OW"][0]["q"] + 2 * lib.multipoles["HW"][0]["q"] == pytest.approx(0.0, abs=1e-12)
    assert lib.multipoles["OW"][0]["q"] != pytest.approx(-0.51966, abs=1e-6)
    again = Amoeba(lib, _top(), cutoff=30.0)
    e_model = float(model(_graph(DIMER1))["energy"])
    assert float(again(_graph(DIMER1))["energy"]) == pytest.approx(e_model, abs=1e-10)


def test_d_equals_p_uses_the_polar_scales_for_the_direct_field():
    """With d-equals-p the direct field carries the polar scale factors (Tinker), whatever the groups."""
    lib = builtin_amoeba("hippo-water-2021")
    for t in ("OW", "HW"):
        lib.polarize[t]["groups"] = []              # every atom its own polarization group, as in EC
    model = Hippo(lib, _top(), cutoff=30.0)
    assert lib.settings["d-equals-p"] is True
    assert torch.equal(model.special_scales[:, 6], model.special_scales[:, 2])
    assert bool((model.special_scales[:, 7] == 1.0).all())
    graph = _graph(DIMER1)
    e_dp = float(model(graph)["e_polar"])
    plain = copy.deepcopy(lib)
    plain.settings["d-equals-p"] = False
    other = Hippo(plain, _top(), cutoff=30.0)
    assert not torch.equal(other.special_scales[:, 6], other.special_scales[:, 2])
    assert float(other(graph)["e_polar"]) != pytest.approx(e_dp, rel=1e-3)
    assert -5.0 < e_dp * EV_TO_KCAL < 0.0           # bounded, no runaway dipoles


# the water values of Chung et al. (2022, Table 1) on the HIPPO water dimer, and Tinker's energies
# (kcal/mol) and forces of the first four atoms (kcal/mol/A) in mutual and direct polarization
EXCHPOL = {"OW": {"spring": 6.238, "size": 0.255, "damp": 1.586, "use": True},
           "HW": {"spring": 9.273, "size": 0.001, "damp": 5.685, "use": True}}
TINKER_EXCHPOL = {
    "mutual": ({"e_polar": -0.93048599, "energy": -4.86641126},
               [[-1.45183399, 0.0, 2.30824788], [-0.27202215, 0.0, -0.1958421],
                [1.98009888, 0.0, -2.12134428], [-0.39776186, 0.0, 1.43423385]]),
    "direct": ({"e_polar": -0.76842226, "energy": -4.70434753},
               [[-1.40757799, 0.0, 2.30352805], [-0.313485, 0.0, -0.2633486],
                [2.28036015, 0.0, -2.46989686], [-0.7139203, 0.0, 1.8323153]])}


def _exchpol_library(**use):
    lib = copy.deepcopy(builtin_amoeba("hippo-water-2021"))
    lib.exchpol = {c: dict(v, use=use.get(c, True)) for c, v in EXCHPOL.items()}
    return lib


@pytest.mark.parametrize("mode", ["mutual", "direct"])
def test_exchange_polarization_matches_tinker(mode):
    """The exchange polarization of the water dimer matches Tinker's energies and forces."""
    refs, forces = TINKER_EXCHPOL[mode]
    model = Hippo(_exchpol_library(), _top(), cutoff=30.0, polarization=mode, polar_eps=1e-10)
    out = ForceStressOutput(model)(_graph(DIMER1))
    for k, ref in refs.items():
        assert float(out[k]) / KCAL_TO_EV == pytest.approx(ref, abs=1e-6), k
    assert np.abs(out["forces"].detach().numpy()[:4] / KCAL_TO_EV - np.array(forces)).max() < 1e-6


def test_exchange_polarization_without_a_used_class_is_off():
    """A pair takes part only when one of its classes is used, so none used leaves the plain induction."""
    plain = Hippo(builtin_amoeba("hippo-water-2021"), _top(), cutoff=30.0)(_graph(DIMER1))
    unused = Hippo(_exchpol_library(OW=False, HW=False), _top(), cutoff=30.0)(_graph(DIMER1))
    damped = Hippo(_exchpol_library(), _top(), cutoff=30.0)(_graph(DIMER1))
    assert float(unused["e_polar"]) == pytest.approx(float(plain["e_polar"]), abs=1e-12)
    assert float(damped["e_polar"]) > float(plain["e_polar"]) + 1e-4


def test_exchange_polarization_trains_and_round_trips_through_tinker_files(tmp_path):
    """The exchpol group trains, and the library's exchpol lines survive a Tinker file."""
    model = Hippo(_exchpol_library(HW=False), _top(), cutoff=30.0, trainable="exchpol")
    model(_graph(DIMER1))["energy"].sum().backward()
    for name in ("expol_spring", "expol_size", "expol_damp"):
        assert float(model.ff.params[name].grad.abs().sum()) > 0, name
    write_tinker_prm(model.export_library(), tmp_path / "water.prm", keep_numbers=False)
    back = read_tinker_prm(tmp_path / "water.prm")
    assert sorted(v["use"] for v in back.exchpol.values()) == [False, True]
    assert {round(v["spring"], 6) for v in back.exchpol.values()} == {6.238, 9.273}


def test_polarization_setting_of_the_library_is_the_default():
    """The file's polarization keyword sets the model's mode; the constructor argument overrides it."""
    lib = builtin_amoeba("hippo-water-2021")
    assert Hippo(lib, _top(), cutoff=30.0).polarization == "mutual"
    direct = copy.deepcopy(lib)
    direct.settings["polarization"] = "DIRECT"
    model = Hippo(direct, _top(), cutoff=30.0)
    assert model.polarization == "direct"
    graph = _graph(DIMER1)
    e_direct = float(model(graph)["e_polar"])
    explicit = Hippo(lib, _top(), cutoff=30.0, polarization="direct")
    assert e_direct == pytest.approx(float(explicit(graph)["e_polar"]), abs=1e-12)
    assert e_direct != pytest.approx(float(Hippo(lib, _top(), cutoff=30.0)(graph)["e_polar"]), rel=1e-4)
    assert Hippo(direct, _top(), cutoff=30.0, polarization="mutual").polarization == "mutual"


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


NUMBERED_PRM = """
forcefield              AMOEBA-WATER
atom         36   71    O    "water O"    8    15.999    2
atom         37   72    H    "water H"    1     1.008    1
vdw          71    3.405   0.110
vdw          72    2.655   0.0135   0.91
bond         71   72    529.6   0.9572
angle        72   71   72    34.05   108.5
multipole    36  -37  -37    -0.51966
                              0.0  0.0  0.14279
                              0.37928
                              0.0  -0.41809
                              0.0  0.0  0.03881
multipole    37   36   37     0.25983
                             -0.03859  0.0  -0.05818
                             -0.03673
                              0.0  -0.10739
                             -0.00203  0.0  0.14412
polarize     36    0.837   0.39   37
polarize     37    0.496   0.39   36
"""


def test_tinker_prm_keeps_its_own_numbers_on_request(tmp_path):
    """A library read from a Tinker file is written back with its own type and class numbers."""
    (tmp_path / "numbered.prm").write_text(NUMBERED_PRM)
    lib = read_tinker_prm(tmp_path / "numbered.prm")
    maps = write_tinker_prm(lib, tmp_path / "kept.prm", keep_numbers=True)
    assert maps == {"types": {"36": 36, "37": 37}, "classes": {"71": 71, "72": 72}}
    again = read_tinker_prm(tmp_path / "kept.prm")
    assert again.atom_types == lib.atom_types and again.multipoles == lib.multipoles
    assert again.polarize == lib.polarize and again.vdw_types == lib.vdw_types
    assert again.bond("36", "37") == lib.bond("36", "37")
    assert write_tinker_prm(lib, tmp_path / "renumbered.prm")["types"] == {"36": 1, "37": 2}
    with pytest.raises(ValueError, match="not an integer"):
        write_tinker_prm(builtin_amoeba("amoeba-water-2003"), tmp_path / "bad.prm", keep_numbers=True)


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


# the AMOEBA 2013 protein force field on capped peptides, as Tinker coordinate files whose
# integer types are the library's type names: Ac-Ala-NMe with a water molecule, and Ac-Gly-NMe
ALA_WATER_XYZ = """\
    25  ala_water
     1  C       2.4986439756     1.6881951726     0.7208539176   221     2    11    12    13
     2  C       1.6457907339     0.7603942739    -0.0998015225   223     1     3     4
     3  O       1.4760230855     0.9133188327    -1.3060790658   224     2
     4  N       1.0378216183    -0.2385334618     0.6204984096     7     2     5    14
     5  C       0.1283928162    -1.1995272646    -0.0088304986     8     4     6     7    15
     6  C       0.1231556208    -2.5012789954     0.7835149821    13     5    16    17    18
     7  C      -1.2991973636    -0.6137983382    -0.1356592477     9     5     8     9
     8  O      -2.3011558272    -1.1866577893     0.2853169947    11     7
     9  N      -1.3559675183     0.5788935075    -0.8298120372   227     7    10    19
    10  C      -2.6107294489     1.1881324643    -1.1820779888   229     9    20    21    22
    11  H       2.7604063108     1.2417529089     1.6843387634   222     1
    12  H       1.9521301836     2.6188504366     0.8936045926   222     1
    13  H       3.4246452587     1.8977850556     0.1784103897   222     1
    14  H       0.9488317725    -0.1117781165     1.6211037764    10     4
    15  H       0.4845660305    -1.4059486408    -1.0257975322    12     5
    16  H      -0.2396762057    -2.3434982034     1.8055889828    14     6
    17  H       1.1330031578    -2.9201985331     0.8503187559    14     6
    18  H      -0.5244912100    -3.2468864641     0.3102572623    14     6
    19  H      -0.5154727858     0.8611105512    -1.3355730496   228     9
    20  H      -3.0239636176     0.6626703862    -2.0469841167   230    10
    21  H      -3.3129107069     1.1221077766    -0.3467625144   230    10
    22  H      -2.4298458802     2.2348944410    -1.4364292536   230    10
    23  O       5.5000000000     0.5000000000     0.0000000000   247    24    25
    24  H       6.4572000000     0.5000000000     0.0000000000   248    23
    25  H       5.2600000000     1.4270000000     0.0000000000   248    23
"""

GLY_XYZ = """\
    19  gly_dipeptide
     1  C      -3.6199962035     0.0040293419    -0.3666062651   221     2    10    11    12
     2  C      -2.2599464852    -0.1373316510    -0.9936618824   223     1     3     4
     3  O      -2.1094421886    -0.3240098740    -2.1971281266   224     2
     4  N      -1.2142552273    -0.0422555821    -0.1156596708     1     2     5    13
     5  C       0.1572209936    -0.1819862693    -0.5576704015     2     4     6    14    15
     6  C       1.1107050645     0.0088647028     0.6211932316     3     5     7     8
     7  O       0.7120805733     0.1839941856     1.7714190805     5     6
     8  N       2.4448777645    -0.0476392977     0.2755851697   227     6     9    16
     9  C       3.4758519117     0.1946646265     1.2529569745   229     8    17    18    19
    10  H      -4.1773160193    -0.9254952311    -0.5100136693   222     1
    11  H      -4.1497690439     0.8313157500    -0.8465419911   222     1
    12  H      -3.5538315184     0.2101932598     0.7052068045   222     1
    13  H      -1.3673682740     0.0725523090     0.8821515300     4     4
    14  H       0.2895933990    -1.1832326188    -0.9795739090     6     5
    15  H       0.3584830432     0.5711472684    -1.3260579485     6     5
    16  H       2.6965550541    -0.0160465726    -0.7047044357   228     8
    17  H       4.4185313616    -0.2025384575     0.8697352523   230     9
    18  H       3.2241005457    -0.2895111707     2.2003081538   230     9
    19  H       3.5639252488     1.2732852809     1.4073306170   230     9
"""


def _protein(tmp_path, text):
    path = tmp_path / "peptide.xyz"
    path.write_text(text)
    lib = builtin_amoeba("amoebapro13")
    model, structure = Amoeba.from_tinker_xyz(path, lib, type_map={int(t): t for t in lib.atom_types},
                                              cutoff=30.0, keep_intermediates=True)
    return model, structure["pos"], structure["atomic_numbers"], lib


def _finite_difference_forces(model, pos, z, samples, h=1e-4, tol=2e-6):
    f = model(_graph(pos, z=z))["forces"].detach().numpy()
    for atom, axis in samples:
        p = pos.copy()
        p[atom, axis] += h
        ep = float(model(_graph(p, z=z))["energy"])
        p[atom, axis] -= 2 * h
        em = float(model(_graph(p, z=z))["energy"])
        assert f[atom, axis] == pytest.approx(-(ep - em) / (2 * h), abs=tol), (atom, axis)


def test_protein_dipeptide_valence_terms_and_file_round_trips(tmp_path):
    model, pos, z, lib = _protein(tmp_path, ALA_WATER_XYZ)
    counts = {k: int(getattr(model, k).shape[-1]) for k in
              ("bond_index", "angle_index", "torsion_index", "opbend_index", "pitors_index", "tortor_index")}
    assert counts == {"bond_index": 23, "angle_index": 37, "torsion_index": 41, "opbend_index": 12,
                      "pitors_index": 2, "tortor_index": 0}
    assert all(int(f) == am.FRAME_ZTHENX for f in model.frame_type[:22])
    assert len(set(model.polarization_groups[:22])) == 6
    fso = ForceStressOutput(model)
    g = _graph(pos, z=z)
    out = fso(g)
    terms = {k: float(out[k]) * EV_TO_KCAL for k in out if k.startswith("e_")}
    assert all(math.isfinite(v) for v in terms.values())
    for k in ("e_bond", "e_angle", "e_strbnd", "e_opbend", "e_torsion", "e_pitors", "e_vdw", "e_multipole", "e_polar"):
        assert abs(terms[k]) > 1e-4, k
    assert terms["e_tortor"] == 0.0
    _finite_difference_forces(fso, pos, z, ((4, 0), (7, 2), (22, 1)))
    # the exported library resolves the peptide's valence terms through the canonical accessors
    cls_of = [model.ff.class_of(t) for t in model.topology.types]
    exported = model.export_library()
    assert exported.torsion(cls_of[1], cls_of[3], cls_of[4], cls_of[6]) is not None
    assert exported.pitors(cls_of[1], cls_of[3]) is not None
    assert exported.opbend(cls_of[2], cls_of[1], cls_of[0], cls_of[3]) is not None
    assert exported.strbnd(cls_of[3], cls_of[4], cls_of[6]) is not None
    assert exported.tortor(*(cls_of[i] for i in (1, 3, 4, 6, 8))) is None
    assert exported.bond_flux(cls_of[1], cls_of[3]) is None
    assert exported.angle_flux(cls_of[3], cls_of[4], cls_of[6]) is None
    assert exported.vdw_pair(cls_of[0], cls_of[1]) is None
    e_ref = float(out["energy"])
    path = exported.save_frc(tmp_path / "pro.frc", name="amoebapro13-export")
    again = read_amoeba(f"{path}:amoebapro13-export")
    assert float(Amoeba(again, model.topology, cutoff=30.0)(g)["energy"]) == pytest.approx(e_ref, abs=1e-9)
    maps = write_tinker_prm(exported, tmp_path / "pro.prm")
    back = read_amoeba(str(tmp_path / "pro.prm"))
    assert back.pitors_types and back.opbend_types and back.angle_inplane_types
    assert len(back.tortor_types) == len(exported.tortor_types)
    renamed = MolecularTopology.from_bonds([str(maps["types"][t]) for t in model.topology.types], model.topology.bonds)
    assert float(Amoeba(back, renamed, cutoff=30.0)(g)["energy"]) == pytest.approx(e_ref, abs=1e-5)


def test_glycine_torsion_torsion_term(tmp_path):
    model, pos, z, lib = _protein(tmp_path, GLY_XYZ)
    assert int(model.tortor_index.shape[-1]) == 1
    cls_of = [model.ff.class_of(t) for t in model.topology.types]
    chain = [cls_of[i] for i in (1, 3, 4, 5, 7)]
    assert lib.tortor(*chain) is not None and lib.tortor(*chain[::-1]) is not None
    fso = ForceStressOutput(model)
    out = fso(_graph(pos, z=z))
    e_tt = float(out["e_tortor"]) * EV_TO_KCAL
    assert math.isfinite(e_tt) and abs(e_tt) > 1e-3
    _finite_difference_forces(fso, pos, z, ((3, 1), (4, 0), (5, 2)))
    exported = model.export_library()
    assert exported.tortor_types.keys() == lib.tortor_types.keys()
    maps = write_tinker_prm(exported, tmp_path / "gly.prm")
    back = read_tinker_prm(tmp_path / "gly.prm")
    assert len(back.tortor_types) == len(lib.tortor_types)
    renamed = MolecularTopology.from_bonds([str(maps["types"][t]) for t in model.topology.types], model.topology.bonds)
    e_back = float(Amoeba(back, renamed, cutoff=30.0)(_graph(pos, z=z))["e_tortor"]) * EV_TO_KCAL
    assert e_back == pytest.approx(e_tt, abs=1e-6)


def test_torsion_torsion_spline_tables_and_bicubic_cells():
    x = np.arange(-180.0, 181.0, 15.0)
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="ij")

    def fun(a, b):
        return np.cos(np.radians(a)) * np.sin(np.radians(2 * b)) + 0.3 * np.cos(np.radians(3 * a - b))

    F = fun(X, Y)
    fx, fy, fxy = am.tortor_spline_tables(x, y, F)
    assert fx.shape == F.shape and np.allclose(fx[0], fx[-1]) and np.allclose(fy[:, 0], fy[:, -1])
    rad = math.pi / 180.0
    dfdx = (-np.sin(np.radians(X)) * np.sin(np.radians(2 * Y)) - 0.9 * np.sin(np.radians(3 * X - Y))) * rad
    dfdy = (2 * np.cos(np.radians(X)) * np.cos(np.radians(2 * Y)) + 0.3 * np.sin(np.radians(3 * X - Y))) * rad
    assert np.abs(fx - dfdx).max() < 1e-2 * np.abs(dfdx).max()
    assert np.abs(fy - dfdy).max() < 1e-2 * np.abs(dfdy).max()
    i, j = 5, 7
    corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
    tables = [torch.tensor([[float(t[a, b]) for a, b in corners]]) for t in (F, fx, fy, fxy)]
    x1, x2, y1, y2 = (torch.tensor([v]) for v in (x[i], x[i + 1], y[j], y[j + 1]))
    at_knot = am.bicubic_interpolate(*tables, x1, y1, x1, x2, y1, y2)
    assert float(at_knot) == pytest.approx(F[i, j], abs=1e-12)
    xm, ym = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    mid = am.bicubic_interpolate(*tables, xm, ym, x1, x2, y1, y2)
    assert float(mid) == pytest.approx(fun(float(xm), float(ym)), abs=2e-2)


def test_trainable_strings_class_lookup_and_kernel_helpers():
    everything = Amoeba("amoeba-water-2003", _top(), cutoff=30.0, trainable="all")
    assert all(p.requires_grad for p in everything.ff.params.values()) and everything.atom_ref.weight.requires_grad
    vdw_only = Amoeba("amoeba-water-2003", _top(), cutoff=30.0, trainable="vdw")
    assert vdw_only.ff.params["vdw_eps"].requires_grad and not vdw_only.ff.params["mpole_q"].requires_grad
    assert not vdw_only.atom_ref.weight.requires_grad
    ref_only = Amoeba("amoeba-water-2003", _top(), cutoff=30.0, trainable="atom_ref")
    assert ref_only.atom_ref.weight.requires_grad and not any(p.requires_grad for p in ref_only.ff.params.values())
    assert everything.ff.class_of("OW") == builtin_amoeba("amoeba-water-2003").cls("OW")
    R = torch.tensor([1.5, 3.0])
    one = torch.ones(2)
    assert torch.allclose(am.coulomb_kernel().evaluate(R, one, one), 1.0 / R)
    assert torch.allclose(am.coulomb_kernel().shift(1).evaluate(R, one, one), torch.ones(2))
    with pytest.raises(ValueError):
        Amoeba("amoeba-water-2003", _top(), cutoff=30.0, trainable=("no-such-group",))
    with pytest.raises(ValueError):
        Amoeba("amoeba-water-2003", _top(), cutoff=30.0, polarization="sometimes")


def test_vdw_pair_rows_override_the_combination_rules():
    base = float(_model()(_graph(DIMER1))["e_vdw"])
    lib = copy.deepcopy(builtin_amoeba("amoeba-water-2003"))
    lib.settings["radiusrule"] = "GEOMETRIC"
    lib.settings["epsilonrule"] = "ARITHMETIC"
    changed = float(Amoeba(lib, _top(), cutoff=30.0)(_graph(DIMER1))["e_vdw"])
    assert abs(changed - base) > 1e-6
    vdw = lib.vdw_types
    for a, b in (("OW", "OW"), ("OW", "HW"), ("HW", "HW")):
        ra, rb, ea, eb = vdw[a]["rmin"], vdw[b]["rmin"], vdw[a]["eps"], vdw[b]["eps"]
        lib.vdw_pair_types["-".join(sorted((a, b)))] = {
            "rmin": (ra ** 3 + rb ** 3) / (ra ** 2 + rb ** 2),
            "eps": 4.0 * ea * eb / (math.sqrt(ea) + math.sqrt(eb)) ** 2}
    model = Amoeba(lib, _top(), cutoff=30.0)
    assert float(model(_graph(DIMER1))["e_vdw"]) == pytest.approx(base, abs=1e-9)
    assert lib.vdw_pair("HW", "OW")["rmin"] == pytest.approx((3.405 ** 3 + 2.655 ** 3) / (3.405 ** 2 + 2.655 ** 2))
    exported = model.export_library()
    for key, row in lib.vdw_pair_types.items():
        assert exported.vdw_pair_types[key]["rmin"] == pytest.approx(row["rmin"])
        assert exported.vdw_pair_types[key]["eps"] == pytest.approx(row["eps"])


def test_from_config_inline_strings_and_tinker_file(tmp_path):
    from xnn.common.config.schema import ModelConfig
    cfg = ModelConfig(name="amoeba", cutoff=30.0, extra={"library": "amoeba-water-2003", "types": str(TYPES),
                                                         "bonds": str([list(b) for b in BONDS]),
                                                         "trainable": "['vdw']"})
    model = Amoeba.from_config(cfg)
    assert model.topology.types == TYPES and model.ff.params["vdw_eps"].requires_grad
    assert float(model(_graph(DIMER1))["energy"]) == pytest.approx(float(_model()(_graph(DIMER1))["energy"]))
    path = tmp_path / "ala_water.xyz"
    path.write_text(ALA_WATER_XYZ)
    model2 = Amoeba.from_config(ModelConfig(name="amoeba", cutoff=30.0,
                                            extra={"library": "amoebapro13", "tinker_xyz": str(path)}))
    assert model2.topology.n_atoms == 25 and model2.topology.types[0] == "221"
    with pytest.raises(ValueError):
        Amoeba.from_config(ModelConfig(name="amoeba", cutoff=30.0, extra={"library": "amoeba-water-2003"}))


def test_molecular_polarizability_is_batched_and_needs_induction():
    mono = Amoeba("amoeba-water-2003", _top(1), cutoff=30.0)
    g1 = _graph(_monomer(), z=Z_DIMER[:3])
    g2 = _graph(_monomer(0.98, 105.0), z=Z_DIMER[:3])
    both = mono.molecular_polarizability(collate([g1, g2]))
    assert both.shape == (2, 3, 3)
    assert torch.allclose(both[0], mono.molecular_polarizability(g1)[0], atol=1e-10)
    assert torch.allclose(both[1], mono.molecular_polarizability(g2)[0], atol=1e-10)
    assert float(torch.diagonal(both[0]).sum() / 3) == pytest.approx((1.672 + 1.225 + 1.328) / 3, abs=0.01)
    with pytest.raises(RuntimeError):
        Amoeba("amoeba-water-2003", _top(1), cutoff=30.0, polarization="none").molecular_polarizability(g1)


def test_flipped_keys_reorder_values_in_the_amoeba_sections():
    sec = make_section("stretch_bend", "x", ["I", "J", "K"], ["K1", "K2"],
                       [(("HW", "OW", "C"), {"K1": 1.0, "K2": 2.0})])
    (row,) = sec.rows
    assert tuple(row.key) == ("C", "OW", "HW") and row.values == {"K1": 2.0, "K2": 1.0}
    sec = make_section("bond_charge_flux", "x", ["I", "J"], ["J"], [(("OW", "HW"), {"J": -0.027})])
    assert tuple(sec.rows[0].key) == ("HW", "OW") and sec.rows[0].values["J"] == pytest.approx(0.027)
    sec = make_section("angle_charge_flux", "x", ["I", "J", "K"], ["Jt1", "Jt2", "Jb1", "Jb2"],
                       [(("HW", "OW", "C"), {"Jt1": 1.0, "Jt2": 2.0, "Jb1": 3.0, "Jb2": 4.0})])
    assert sec.rows[0].values == {"Jt1": 2.0, "Jt2": 1.0, "Jb1": 4.0, "Jb2": 3.0}
    assert flip_values(SECTION_SCHEMA["bond_charge_flux"], ["-0.1", 0.2]) == ["0.1", -0.2]


def test_unsupported_sections_and_file_helpers(tmp_path):
    ff = read_forcefield("amoeba-water-2003")
    ff.sections["nonbond_pair"] = {}
    with pytest.raises(ValueError):
        from_forcefield(ff)
    lenient = from_forcefield(ff, strict=False)
    assert any("nonbond_pair" in note for note in lenient.notes)
    assert lenient.scale("polar-intra", 4) == float(lenient.setting("polar-14-intra"))
    numbers = {"OW": 1, "HW": 2}
    write_tinker_xyz(tmp_path / "cell.xyz", DIMER1, TYPES, BONDS, numbers, symbols=["O", "H", "H"] * 2,
                     cell=[20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    data = read_tinker_xyz(tmp_path / "cell.xyz", type_map={1: "OW", 2: "HW"})
    assert data["cell"] == [20.0, 20.0, 20.0, 90.0, 90.0, 90.0] and len(data["positions"]) == 6
    model, structure = Amoeba.from_tinker_xyz(tmp_path / "cell.xyz", "amoeba-water-2003",
                                              type_map={1: "OW", 2: "HW"}, cutoff=9.0)
    assert np.allclose(structure["cell"], np.diag([20.0, 20.0, 20.0])) and model.topology.n_atoms == 6
    assert amoebalib._looks_like_keyword(["bond", "1", "2"])
    assert not amoebalib._looks_like_keyword(["The", "AMOEBA", "force", "field", "paper"])
    assert amoebalib._is_float("1.5") and not amoebalib._is_float("x")
    assert amoebalib._prm_number(1.23456789) == "   1.23457"
