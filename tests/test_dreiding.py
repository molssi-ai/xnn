"""DREIDING force-field tests: rules, equations, invariances, training.

Verification is self-contained, in the clean-room convention this code base
uses: every generator rule and energy term is recomputed by hand from the
published DREIDING functional form (Mayo, Olafson & Goddard, *J. Phys.
Chem.* 94, 8897, 1990) and compared against the model's intermediates.
Specifically checked against the paper:

* Table I bond radii and angles, Table II van der Waals parameters and
  Table III force constants, as read from the SEAMM ``dreiding.frc``;
* eq 6 (``R0_IJ = R0_I + R0_J - 0.01``), eqs 7-9 (bond-order scaling),
  eq 10a/10'/11 (angles), eq 12, eqs 13-23 (the nine torsion rules),
  eq 28a-c (inversions), eqs 31'/32' (van der Waals), eqs 35-36
  (combination rules), eq 37 (electrostatics) and eq 38 (hydrogen bonds);
* the eclipsed-ethane torsion barrier of exactly ``V_JK = 2.0`` kcal/mol
  (eq 14) and the hydrogen-bond minimum of exactly ``-D_hb`` at
  ``R = R_hb`` for a linear donor-hydrogen-acceptor arrangement (eq 38).

An independent cross-check against LAMMPS's DREIDING styles (bond
harmonic, angle cosine/squared, dihedral harmonic, improper umbrella, pair
lj/cut / buck / hbond/dreiding/lj) lives in
``examples/fidelity_checks/dreiding_verification.ipynb``; it needs LAMMPS
and is not part of this suite.
"""
import json
import math

import pytest
import torch

from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models
from xnn.common.train import weighted_loss
from xnn.ffnn.models import (Dreiding, DreidingForceField, DreidingLibrary,
                             MolecularTopology, read_dreiding, torsion_rule,
                             hybridization, TORSION_RULES)
from xnn.ffnn.models.dreiding import KE
from xnn.ffnn.models.oplslib import KCAL_TO_EV

EV_TO_KCAL = 1.0 / KCAL_TO_EV
DEG = math.pi / 180.0


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# small builders
def _graph(pos, z, cutoff, cell=None):
    """Build a single-structure graph from raw arrays."""
    s = {"pos": torch.as_tensor(pos, dtype=torch.get_default_dtype()),
         "atomic_numbers": torch.as_tensor(z, dtype=torch.long)}
    if cell is not None:
        s["cell"] = torch.as_tensor(cell, dtype=torch.get_default_dtype())
        s["pbc"] = torch.tensor([True, True, True])
    return structure_to_graph(s, cutoff=cutoff)


def _ethane_pos(phi_deg=60.0, r_cc=1.53, r_ch=1.09, ang_deg=109.471):
    """Ethane with a prescribed H-C-C-H dihedral (0 = eclipsed)."""
    ang = math.radians(ang_deg)
    pos = [[0.0, 0.0, 0.0], [r_cc, 0.0, 0.0]]
    s, c = r_ch * math.sin(math.pi - ang), r_ch * math.cos(math.pi - ang)
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0
        pos.append([-c, s * math.cos(a), s * math.sin(a)])
    for k in range(3):
        a = 2.0 * math.pi * k / 3.0 + math.radians(phi_deg)
        pos.append([r_cc + c, s * math.cos(a), s * math.sin(a)])
    return pos


def _ethane_model(**kwargs):
    """A DREIDING ethane model plus its element list."""
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 7)
    model = Dreiding(kwargs.pop("ffield", "dreiding"), top,
                     cutoff=kwargs.pop("cutoff", 14.0), **kwargs)
    return model, [6, 6] + [1] * 6


def _ethylene(pyramid_deg=0.0):
    """Ethylene with one H rotated out of plane by ``pyramid_deg``."""
    d, dh = 1.34, 1.08
    ang = math.radians(120.0)
    phi = math.radians(pyramid_deg)
    y = dh * math.sin(math.pi - ang)
    x = dh * math.cos(math.pi - ang)
    pos = [[0.0, 0.0, 0.0], [d, 0.0, 0.0], [-x, y, 0.0], [-x, -y, 0.0],
           [d + x, y * math.cos(phi), y * math.sin(phi)], [d + x, -y, 0.0]]
    types = ["C_2", "C_2"] + ["H_"] * 4
    bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    top = MolecularTopology.from_bonds(types, bonds,
                                       bond_orders=[2.0, 1.0, 1.0, 1.0, 1.0])
    return pos, [6, 6, 1, 1, 1, 1], top


def _water_dimer(r_oo=2.75):
    """Two waters with a linear O-H...O bridge along x."""
    pos = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
           [r_oo, 0.0, 0.0], [r_oo + 0.3, 0.90, 0.0], [r_oo + 0.3, -0.90, 0.0]]
    types = ["O_3", "H__HB", "H__HB", "O_3", "H__HB", "H__HB"]
    bonds = [(0, 1), (0, 2), (3, 4), (3, 5)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 4)
    return pos, [8, 1, 1, 8, 1, 1], top


# registration and the parameter library
def test_registration():
    assert "dreiding" in available_models()


def test_library_matches_paper_tables():
    lib = read_dreiding("dreiding")
    # Table I: bond radii (A) and angles (deg)
    for name, r0, th0 in [("H_", 0.330, 180.0), ("C_3", 0.770, 109.471),
                          ("C_R", 0.700, 120.0), ("C_2", 0.670, 120.0),
                          ("C_1", 0.602, 180.0), ("N_3", 0.702, 106.7),
                          ("O_3", 0.660, 104.51), ("S_3", 1.040, 92.1),
                          ("Zn", 1.330, 109.471)]:
        assert lib.radius[name] == pytest.approx(r0, abs=1e-9)
        assert lib.theta0[name] == pytest.approx(th0, abs=1e-9)
    # Table II: van der Waals R0 (A) and D0 (kcal/mol)
    for name, r0, d0 in [("H_", 3.195, 0.0152), ("C_3", 3.8983, 0.0951),
                         ("N_3", 3.6621, 0.0774), ("O_3", 3.4046, 0.0957),
                         ("Cl", 3.9503, 0.2833), ("C_33", 4.1524, 0.2500)]:
        assert lib.vdw_r0[name] == pytest.approx(r0, abs=1e-9)
        assert lib.vdw_d0[name] == pytest.approx(d0, abs=1e-9)
    # Table III: the global valence force constants
    assert lib.bond_k1 == 700.0 and lib.bond_d1 == 70.0
    assert lib.angle_k == 100.0 and lib.delta == 0.01
    # Table III inversions: 40 kcal/mol/rad^2, planar centers at Psi0 = 0
    for center in ("C_2", "C_R", "N_2", "N_R", "O_2", "O_R", "B_2"):
        assert lib.oop[center] == (40.0, 0.0)
    assert lib.oop["C_31"] == (40.0, 54.74)
    # Table V: hydrogen bond, the no-charges convention
    assert lib.hbond_d0 == 9.0 and lib.hbond_r0 == 2.75
    assert lib.form == "lj" and lib.combination == "arithmetic"


def test_x6_library_and_zeta():
    lib = read_dreiding("dreiding/X6")
    assert lib.form == "x6"
    # Table II scaling parameters (zeta)
    for name, zeta in [("H_", 12.382), ("C_3", 14.034), ("N_3", 13.843),
                       ("O_3", 13.483), ("F_", 14.444), ("Cl", 13.861),
                       ("Al3", 12.0)]:
        assert lib.x6_zeta[name] == pytest.approx(zeta, abs=1e-9)
    # both variants share the same R0 / D0
    lj = read_dreiding("dreiding")
    assert lib.vdw_r0 == pytest.approx(lj.vdw_r0)
    assert lib.vdw_d0 == pytest.approx(lj.vdw_d0)


def test_hybridization_and_element_mnemonics():
    # the third character of the five-character label encodes the geometry
    for name, h in [("C_3", "3"), ("C_2", "2"), ("C_1", "1"), ("C_R", "R"),
                    ("N_3", "3"), ("O_R", "R"), ("Al3", "3"), ("Si3", "3"),
                    ("C_R1", "R"), ("C_33", "3"),
                    ("H_", "0"), ("H__HB", "0"), ("Cl", "0"), ("Na", "0"),
                    ("Zn", "0"), ("Br", "0")]:
        assert hybridization(name) == h
    lib = read_dreiding("dreiding")
    for name, el in [("C_3", "C"), ("H__HB", "H"), ("Al3", "Al"),
                     ("Si3", "Si"), ("Cl", "Cl"), ("Se3", "Se"), ("I_", "I")]:
        assert lib.element(name) == el
    assert lib.oxygen_column("S_3") and lib.oxygen_column("O_3")
    assert not lib.oxygen_column("N_3") and not lib.oxygen_column("C_3")


def test_library_json_roundtrip(tmp_path):
    lib = read_dreiding("dreiding")
    path = lib.save(tmp_path / "d.json")
    back = read_dreiding(path)
    assert back.radius == lib.radius and back.theta0 == lib.theta0
    assert back.vdw_r0 == lib.vdw_r0 and back.vdw_d0 == lib.vdw_d0
    assert back.oop == lib.oop and back.torsion_v == lib.torsion_v
    assert back.form == lib.form and back.templates == lib.templates
    assert json.loads(path.read_text())["format"] == "xnn-dreiding-1"


# the torsion rule engine (eqs 14-23)
def test_torsion_rules_against_the_paper():
    lib = read_dreiding("dreiding")

    def rule(i, j, k, l, order=1.0):
        return torsion_rule(lib, i, j, k, l, order)

    # (a) sp3 - sp3 single bond: V = 2.0, n = 3, phi0 = 180
    assert rule("H_", "C_3", "C_3", "H_") == "a"
    # (b) sp2/resonant - sp3 single bond with an sp2 outer atom (acetate)
    assert rule("O_2", "C_2", "C_3", "H_") == "b"
    assert rule("H_", "C_3", "C_R", "C_R") == "b"
    # (c) sp2 = sp2 double bond
    assert rule("H_", "C_2", "C_2", "H_", 2.0) == "c"
    # (d) resonance bond (order 1.5)
    assert rule("C_R", "C_R", "C_R", "C_R", 1.5) == "d"
    # (e) single bond between two sp2 centers (butadiene)
    assert rule("H_", "C_2", "C_2", "H_", 1.0) == "e"
    # (f) exocyclic single bond between two aromatic atoms (biphenyl)
    assert rule("C_R", "C_R", "C_R", "C_R", 1.0) == "f"
    # (g) a central sp1, monovalent or metal atom carries no torsion
    assert rule("H_", "C_1", "C_3", "H_") is None
    assert rule("C_3", "C_3", "H_", "H_") is None
    assert rule("H_", "C_3", "Zn", "H_") is None
    # (h) sp3 - sp3, both of the oxygen column (HOOH): n = 2, phi0 = 90
    assert rule("H__HB", "O_3", "O_3", "H__HB") == "h"
    assert rule("H_", "S_3", "S_3", "H_") == "h"
    # (i) oxygen-column sp3 bonded to an sp2/resonant atom of another column
    assert rule("O_2", "C_2", "O_3", "H_") == "i"
    assert rule("C_R", "C_R", "O_3", "H_") == "i"
    # (j) the propene exception: a non-sp2 outer atom on the sp2 side
    assert rule("H_", "C_2", "C_3", "H_") == "j"
    # the rule table itself carries the published (V, n, phi0)
    assert TORSION_RULES["a"] == (2.0, 3, 180.0)
    assert TORSION_RULES["b"] == (1.0, 6, 0.0)
    assert TORSION_RULES["c"] == (45.0, 2, 180.0)
    assert TORSION_RULES["d"] == (25.0, 2, 180.0)
    assert TORSION_RULES["e"] == (5.0, 2, 180.0)
    assert TORSION_RULES["f"] == (10.0, 2, 180.0)
    assert TORSION_RULES["h"] == (2.0, 2, 90.0)
    assert TORSION_RULES["i"] == (2.0, 2, 180.0)
    assert TORSION_RULES["j"] == (2.0, 3, 180.0)


def test_torsion_rules_are_symmetric_under_reversal():
    lib = read_dreiding("dreiding")
    quads = [("H_", "C_3", "C_3", "H_", 1.0), ("O_2", "C_2", "C_3", "H_", 1.0),
             ("H_", "C_2", "C_2", "H_", 2.0), ("C_R", "C_R", "C_R", "C_R", 1.5),
             ("H__HB", "O_3", "O_3", "H__HB", 1.0),
             ("C_R", "C_R", "O_3", "H_", 1.0), ("H_", "C_2", "C_3", "H_", 1.0)]
    for i, j, k, l, o in quads:
        assert torsion_rule(lib, i, j, k, l, o) == \
            torsion_rule(lib, l, k, j, i, o)


def test_torsion_barrier_is_split_over_the_central_bond():
    # eq 13 note: V_JK is the total barrier of the J-K bond, renormalized by
    # the number of I,L combinations -- 2/9 kcal/mol for each of ethane's nine
    model, _ = _ethane_model()
    assert model.dihedral_index.shape[1] == 9
    assert model.dihedral_weight.tolist() == pytest.approx([1.0 / 9.0] * 9)
    v = float(model.ff.params["torsion_v"][0]) * EV_TO_KCAL
    assert v == pytest.approx(2.0, abs=1e-12)


# energy expressions, recomputed by hand
def test_bond_energy_and_radius_additivity():
    model, z = _ethane_model(keep_intermediates=True)
    pos = _ethane_pos()
    pos[1][0] = 1.75                       # stretch the C-C bond
    model(_graph(pos, z, 14.0))
    inter = model.intermediates
    # eq 6: R0_CC = 0.770 + 0.770 - 0.01, R0_CH = 0.770 + 0.330 - 0.01
    assert float(inter["bond_r0"][0]) == pytest.approx(1.53, abs=1e-12)
    assert float(inter["bond_r0"][1]) == pytest.approx(1.09, abs=1e-12)
    # eq 4a with eq 7: E = 1/2 * 700 * (R - R0)^2
    e_cc = float(inter["e_bond"][0]) * EV_TO_KCAL
    assert e_cc == pytest.approx(0.5 * 700.0 * (1.75 - 1.53) ** 2, rel=1e-9)


def test_bond_order_scales_force_constant():
    # eq 9a: K(n) = n K(1); the double bond of ethylene is twice as stiff
    pos, z, top = _ethylene()
    model = Dreiding("dreiding", top, cutoff=14.0, keep_intermediates=True)
    pos[1][0] = 1.34 + 0.1
    model(_graph(pos, z, 14.0))
    inter = model.intermediates
    cc = [m for m, b in enumerate(top.bonds) if b == (0, 1)][0]
    e = float(inter["e_bond"][cc]) * EV_TO_KCAL
    # R0 = 0.670 + 0.670 - 0.01 = 1.33, so the stretch is 1.44 - 1.33
    assert e == pytest.approx(0.5 * 2.0 * 700.0 * (1.44 - 1.33) ** 2, rel=1e-9)


def test_morse_bond_matches_harmonic_curvature():
    # eq 5a/5b: the Morse form shares the harmonic curvature at R = R0
    types, bonds = ["C_3", "C_3"], [(0, 1)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0])
    harm = Dreiding("dreiding", top, cutoff=14.0, keep_intermediates=True)
    mors = Dreiding("dreiding", top, cutoff=14.0, bond_style="morse",
                    keep_intermediates=True)
    # the two forms agree to O(alpha dr): alpha = 2.236/A here, so a 1e-4 A
    # stretch differs by ~2e-4 relative and a 0.3 A stretch by ~50%
    for dr, rel in ((1e-4, 1e-3), (0.3, 5e-1)):
        g = _graph([[0, 0, 0], [1.53 + dr, 0, 0]], [6, 6], 14.0)
        eh = float(harm(g)["e_bond"]) * EV_TO_KCAL
        em = float(mors(g)["e_bond"]) * EV_TO_KCAL
        assert em == pytest.approx(eh, rel=rel)
    # and the exact Morse value, with alpha = sqrt(k / 2D)
    alpha = math.sqrt(700.0 / (2.0 * 70.0))
    g = _graph([[0, 0, 0], [1.93, 0, 0]], [6, 6], 14.0)
    assert float(mors(g)["e_bond"]) * EV_TO_KCAL == pytest.approx(
        70.0 * (math.exp(-alpha * 0.4) - 1.0) ** 2, rel=1e-9)
    # a very long bond approaches the dissociation energy
    g = _graph([[0, 0, 0], [12.0, 0, 0]], [6, 6], 14.0)
    assert float(mors(g)["e_bond"]) * EV_TO_KCAL == pytest.approx(70.0, rel=1e-6)


def test_angle_harmonic_cosine_and_linear_forms():
    # eq 10a: E = 1/2 (K / sin^2 theta0) (cos theta - cos theta0)^2
    r, theta = 1.09, 100.0
    pos = [[0, 0, 0], [r, 0, 0],
           [r * math.cos(math.radians(theta)),
            r * math.sin(math.radians(theta)), 0]]
    top = MolecularTopology.from_bonds(["C_3", "H_", "H_"],
                                       [(0, 1), (0, 2)], bond_orders=[1.0] * 2)
    model = Dreiding("dreiding", top, cutoff=14.0, keep_intermediates=True)
    out = model(_graph(pos, [6, 1, 1], 14.0))
    th0 = math.radians(109.471)
    want = 0.5 * (100.0 / math.sin(th0) ** 2) \
        * (math.cos(math.radians(theta)) - math.cos(th0)) ** 2
    assert float(out["e_angle"]) * EV_TO_KCAL == pytest.approx(want, rel=1e-12)
    # eq 11: the plain theta form, as an option
    harm = Dreiding("dreiding", top, cutoff=14.0, angle_style="harmonic")
    want_h = 0.5 * 100.0 * (math.radians(theta) - th0) ** 2
    assert float(harm(_graph(pos, [6, 1, 1], 14.0))["e_angle"]) * EV_TO_KCAL \
        == pytest.approx(want_h, rel=1e-12)
    # eq 10': linear centers (theta0 = 180) use K (1 + cos theta)
    lin_top = MolecularTopology.from_bonds(["C_1", "O_1", "O_1"],
                                           [(0, 1), (0, 2)],
                                           bond_orders=[2.0, 2.0])
    lin = Dreiding("dreiding", lin_top, cutoff=14.0)
    assert bool(lin.angle_linear.all())
    pos2 = [[0, 0, 0], [1.16, 0, 0],
            [1.16 * math.cos(math.radians(theta)),
             1.16 * math.sin(math.radians(theta)), 0]]
    want_l = 100.0 * (1.0 + math.cos(math.radians(theta)))
    assert float(lin(_graph(pos2, [6, 8, 8], 14.0))["e_angle"]) * EV_TO_KCAL \
        == pytest.approx(want_l, rel=1e-12)


def test_eclipsed_ethane_torsion_barrier_is_the_published_total():
    # eq 14: V_JK = 2.0 kcal/mol as the *total* barrier of the C-C bond
    model, z = _ethane_model()
    e = {}
    for phi in (0.0, 60.0):
        out = model(_graph(_ethane_pos(phi), z, 14.0))
        e[phi] = float(out["e_torsion"]) * EV_TO_KCAL
        # an ideal geometry puts bonds exactly at their minima; the angles
        # land within 4e-6 rad of theta0 because the file rounds the
        # tetrahedral angle to 109.471 deg, leaving a residue ~1e-8 kcal/mol
        assert float(out["e_bond"]) == pytest.approx(0.0, abs=1e-15)
        assert abs(float(out["e_angle"]) * EV_TO_KCAL) < 1e-7
    assert e[60.0] == pytest.approx(0.0, abs=1e-10)
    assert e[0.0] == pytest.approx(2.0, abs=1e-10)
    # and the shape is the pure 3-fold cosine of eq 13
    for phi in (17.0, 41.0, 88.0):
        out = model(_graph(_ethane_pos(phi), z, 14.0))
        want = 0.5 * 2.0 * (1.0 + math.cos(3.0 * math.radians(phi)))
        assert float(out["e_torsion"]) * EV_TO_KCAL == pytest.approx(want,
                                                                     abs=1e-10)


def test_torsion_phase_of_rule_h():
    # eq 21: the oxygen-column sp3-sp3 torsion has n = 2, phi0 = 90, so HOOH
    # is at a minimum near 90 degrees and at maxima at 0 and 180
    types = ["O_3", "O_3", "H__HB", "H__HB"]
    bonds = [(0, 1), (0, 2), (1, 3)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 3)
    model = Dreiding("dreiding", top, cutoff=14.0)
    assert model.dihedral_index.shape[1] == 1
    r, ang = 1.31, math.radians(104.51)
    for phi, want in ((90.0, 0.0), (0.0, 2.0), (180.0, 2.0), (45.0, 1.0)):
        p0 = [0.0, 0.0, 0.0]
        p1 = [r, 0.0, 0.0]
        p2 = [-r * math.cos(math.pi - ang), r * math.sin(math.pi - ang), 0.0]
        a = math.radians(phi)
        p3 = [r + r * math.cos(math.pi - ang),
              r * math.sin(math.pi - ang) * math.cos(a),
              r * math.sin(math.pi - ang) * math.sin(a)]
        # atom order is (H2, O0, O1, H3) for the dihedral, so pass positions
        # in topology order: O0, O1, H2, H3
        out = model(_graph([p0, p1, p2, p3], [8, 8, 1, 1], 14.0))
        assert float(out["e_torsion"]) * EV_TO_KCAL == pytest.approx(want,
                                                                     abs=1e-9)


def test_inversion_planar_and_nonplanar_forms():
    # eq 28c: E = K (1 - cos Psi) at a planar center, averaged over the three
    # axis choices with weight 1/3 (so a planar molecule gives exactly zero)
    pos, z, top = _ethylene(0.0)
    model = Dreiding("dreiding", top, cutoff=14.0, keep_intermediates=True)
    assert model.inversion_index.shape[1] == 6        # 2 centers x 3 axes
    out = model(_graph(pos, z, 14.0))
    assert float(out["e_inversion"]) == pytest.approx(0.0, abs=1e-18)
    # pyramidalize one hydrogen and recompute eq 28c by hand
    pos, z, top = _ethylene(25.0)
    out = model(_graph(pos, z, 14.0))
    sin_psi = model.intermediates["inversion_sin"]
    want = sum(40.0 * (1.0 - math.sqrt(1.0 - float(s) ** 2)) / 3.0
               for s in sin_psi)
    assert float(out["e_inversion"]) * EV_TO_KCAL == pytest.approx(want,
                                                                   rel=1e-12)
    assert float(out["e_inversion"]) > 0.0
    # the term is even in Psi (eq 30)
    a = model(_graph(_ethylene(25.0)[0], z, 14.0))["e_inversion"]
    b = model(_graph(_ethylene(-25.0)[0], z, 14.0))["e_inversion"]
    assert float(a) == pytest.approx(float(b), rel=1e-12)


def test_inversion_nonplanar_center_has_two_minima():
    # eq 28a: a center with Psi0 != 0 (C_31, Psi0 = 54.74) is minimal at +-Psi0
    lib = read_dreiding("dreiding")
    assert lib.oop["C_31"][1] == 54.74
    k, psi0 = lib.oop["C_31"]
    for psi in (54.74, -54.74):
        want = 0.5 * (k / math.sin(math.radians(psi0)) ** 2) \
            * (math.cos(math.radians(psi)) - math.cos(math.radians(psi0))) ** 2
        assert want == pytest.approx(0.0, abs=1e-20)
    want0 = 0.5 * (k / math.sin(math.radians(psi0)) ** 2) \
        * (1.0 - math.cos(math.radians(psi0))) ** 2
    assert want0 > 0.0


def test_vdw_lennard_jones_and_combination_rule():
    # eq 31': E = D0 [rho^-12 - 2 rho^-6]; eq 36a/36c combination
    lib = read_dreiding("dreiding")
    r = 3.6
    types, bonds = ["C_3", "O_3"], []
    top = MolecularTopology.from_bonds(types, bonds)
    model = Dreiding("dreiding", top, cutoff=14.0)
    out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 8], 14.0))
    d0 = math.sqrt(lib.vdw_d0["C_3"] * lib.vdw_d0["O_3"])       # eq 36a
    r0 = 0.5 * (lib.vdw_r0["C_3"] + lib.vdw_r0["O_3"])          # eq 36c
    rho = r / r0
    want = d0 * (rho ** -12 - 2.0 * rho ** -6)
    assert float(out["e_vdw"]) * EV_TO_KCAL == pytest.approx(want, rel=1e-12)
    # at R = R0 a homonuclear pair sits exactly at -D0
    two = Dreiding("dreiding", MolecularTopology.from_bonds(["C_3", "C_3"], []),
                   cutoff=14.0)
    out = two(_graph([[0, 0, 0], [lib.vdw_r0["C_3"], 0, 0]], [6, 6], 14.0))
    assert float(out["e_vdw"]) * EV_TO_KCAL == pytest.approx(
        -lib.vdw_d0["C_3"], rel=1e-12)


def test_vdw_exponential_6_form():
    # eq 32': E = D0 [6/(zeta-6) exp(zeta (1-rho)) - zeta/(zeta-6) rho^-6]
    lib = read_dreiding("dreiding/X6")
    r = 3.6
    top = MolecularTopology.from_bonds(["C_3", "C_3"], [])
    model = Dreiding("dreiding/X6", top, cutoff=14.0)
    out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 6], 14.0))
    d0, r0, z = lib.vdw_d0["C_3"], lib.vdw_r0["C_3"], lib.x6_zeta["C_3"]
    rho = r / r0
    want = d0 * (6.0 / (z - 6.0) * math.exp(z * (1.0 - rho))
                 - z / (z - 6.0) * rho ** -6)
    assert float(out["e_vdw"]) * EV_TO_KCAL == pytest.approx(want, rel=1e-12)
    # the X6 minimum is at R0 with depth D0, like the LJ form
    out = model(_graph([[0, 0, 0], [r0, 0, 0]], [6, 6], 14.0))
    assert float(out["e_vdw"]) * EV_TO_KCAL == pytest.approx(-d0, rel=1e-12)


def test_coulomb_uses_the_published_constant():
    # eq 37: E = 322.0637 Q_i Q_j / R kcal/mol  (332.0637 in the paper's text)
    top = MolecularTopology.from_bonds(["C_3", "C_3"], [])
    model = Dreiding("dreiding", top, cutoff=14.0, charges=[0.4, -0.4])
    r = 5.0
    out = model(_graph([[0, 0, 0], [r, 0, 0]], [6, 6], 14.0))
    assert float(out["e_coulomb"]) * EV_TO_KCAL == pytest.approx(
        332.0637 * 0.4 * (-0.4) / r, rel=1e-12)
    assert KE == pytest.approx(332.0637 * KCAL_TO_EV, rel=1e-15)


def test_hydrogen_bond_minimum_and_angle_factor():
    # eq 38: E = D_hb [5 (R_hb/R)^12 - 6 (R_hb/R)^10] cos^4(theta_DHA),
    # so a linear bridge at R = R_hb sits exactly at -D_hb
    pos, z, top = _water_dimer(2.75)
    model = Dreiding("dreiding", top, cutoff=14.0, keep_intermediates=True)
    out = model(_graph(pos, z, 14.0))
    assert float(out["e_hbond"]) * EV_TO_KCAL == pytest.approx(-9.0, rel=1e-12)
    # the full expression at another distance, summed over the active triplets
    pos, z, top = _water_dimer(3.1)
    out = model(_graph(pos, z, 14.0))
    inter = model.intermediates
    want = 0.0
    for r_da, c in zip(inter["hb_r"], inter["hb_cos"]):
        if float(c) >= 0.0:
            continue                       # the paper restricts theta > 90 deg
        rho = 2.75 / float(r_da)
        want += 9.0 * (5.0 * rho ** 12 - 6.0 * rho ** 10) * float(c) ** 4
    assert float(out["e_hbond"]) * EV_TO_KCAL == pytest.approx(want, rel=1e-12)
    # switching the term off removes it entirely
    off = Dreiding("dreiding", top, cutoff=14.0, hbond=False)
    assert float(off(_graph(pos, z, 14.0))["e_hbond"]) == 0.0


def test_fourteen_pairs_are_included_in_full():
    # "in DREIDING, the default is to include the full value for all 1,4
    # terms"; only 1,2 and 1,3 pairs are excluded
    model, z = _ethane_model(keep_intermediates=True)
    top = model.topology
    assert set(map(tuple, model.excl_index.t().tolist())) == set(top.exclusions)
    assert all(p not in top.exclusions for p in top.pairs14)
    model(_graph(_ethane_pos(0.0), z, 14.0))
    w = model.intermediates["nb_weight"]
    src = model.intermediates["type_index"]
    assert float(w.min()) == 0.0 and float(w.max()) == 1.0
    # the nine 1,4 H-H pairs survive (each direction of the edge list)
    assert int((w > 0).sum()) == 2 * len(top.pairs14)


# invariances, batching, periodicity
def test_rotation_translation_invariance_and_force_equivariance():
    model, z = _ethane_model()
    pos = torch.tensor(_ethane_pos(37.0))
    out = ForceStressOutput(model)
    e0 = float(out(_graph(pos, z, 14.0))["energy"])
    a = 0.7
    R = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                      [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]],
                     dtype=torch.float64)
    moved = pos @ R.T + torch.tensor([3.0, -1.0, 2.0], dtype=torch.float64)
    g0 = _graph(pos, z, 14.0)
    g1 = _graph(moved, z, 14.0)
    o0, o1 = out(g0), out(g1)
    assert float(o1["energy"]) == pytest.approx(e0, rel=1e-12)
    assert torch.allclose(o1["forces"], o0["forces"] @ R.T, atol=1e-10)


def test_forces_match_finite_differences():
    pos, z, top = _ethylene(18.0)
    model = Dreiding("dreiding", top, cutoff=14.0,
                     charges=[-0.1, -0.1, 0.05, 0.05, 0.05, 0.05])
    out = ForceStressOutput(model)
    p = torch.tensor(pos, dtype=torch.float64)
    f = out(_graph(p, z, 14.0))["forces"].detach()
    h = 1e-6
    for atom in (0, 3, 4):
        for ax in range(3):
            pp, pm = p.clone(), p.clone()
            pp[atom, ax] += h
            pm[atom, ax] -= h
            fd = -(float(out(_graph(pp, z, 14.0))["energy"])
                   - float(out(_graph(pm, z, 14.0))["energy"])) / (2 * h)
            assert float(f[atom, ax]) == pytest.approx(fd, abs=1e-6)


def test_batching_matches_individual_evaluation():
    model, z = _ethane_model()
    graphs = [_graph(_ethane_pos(phi), z, 14.0) for phi in (0.0, 23.0, 60.0)]
    singles = torch.cat([model(g)["energy"] for g in graphs])
    batched = model(collate(graphs))["energy"]
    assert torch.allclose(batched, singles, atol=1e-12)
    for key in ("e_bond", "e_angle", "e_torsion", "e_vdw"):
        s = torch.cat([model(g)[key] for g in graphs])
        assert torch.allclose(model(collate(graphs))[key], s, atol=1e-12)


def test_size_extensivity_with_replicated_topology():
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 7)
    one = Dreiding("dreiding", top, cutoff=6.0)
    two = Dreiding("dreiding", top.replicate(2), cutoff=6.0)
    assert len(two.topology.bond_orders) == 2 * len(top.bond_orders)
    pos = torch.tensor(_ethane_pos(41.0))
    far = torch.cat([pos, pos + torch.tensor([60.0, 0.0, 0.0])])
    z = [6, 6] + [1] * 6
    e1 = float(one(_graph(pos, z, 6.0))["energy"])
    e2 = float(two(_graph(far, z * 2, 6.0))["energy"])
    assert e2 == pytest.approx(2.0 * e1, rel=1e-12)


def test_periodic_minimum_image_and_stress():
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 7)
    model = Dreiding("dreiding", top, cutoff=5.0)
    out = ForceStressOutput(model, compute_stress=True)
    pos = torch.tensor(_ethane_pos(55.0))
    z = [6, 6] + [1] * 6
    cell = torch.eye(3, dtype=torch.float64) * 14.0
    e0 = float(out(_graph(pos, z, 5.0, cell=cell))["energy"])
    # wrap the molecule across the boundary: bonded terms must not notice
    wrapped = pos.clone()
    wrapped[:, 0] = torch.remainder(wrapped[:, 0] - 0.7, 14.0)
    e1 = float(out(_graph(wrapped, z, 5.0, cell=cell))["energy"])
    assert e1 == pytest.approx(e0, rel=1e-10)
    s = out(_graph(pos, z, 5.0, cell=cell))["stress"]
    assert s.shape[-2:] == (3, 3)
    assert torch.allclose(s, s.transpose(-1, -2), atol=1e-12)


def test_topology_mismatch_raises():
    model, z = _ethane_model()
    with pytest.raises(ValueError, match="atoms"):
        model(_graph(_ethane_pos()[:6], z[:6], 14.0))
    with pytest.raises(ValueError, match="atomic numbers"):
        model(_graph(_ethane_pos(), [6, 7] + [1] * 6, 14.0))


def test_switching_function():
    top = MolecularTopology.from_bonds(["C_3", "C_3"], [])
    plain = Dreiding("dreiding", top, cutoff=6.0)
    smooth = Dreiding("dreiding", top, cutoff=6.0, switch_width=1.5)
    for r, same in ((4.0, True), (5.5, False)):
        g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], 6.0)
        a, b = float(plain(g)["energy"]), float(smooth(g)["energy"])
        assert (a == pytest.approx(b, rel=1e-12)) is same
    g = _graph([[0, 0, 0], [5.999, 0, 0]], [6, 6], 6.0)
    assert abs(float(smooth(g)["energy"])) \
        < 1e-7 * abs(float(plain(g)["energy"]))
    with pytest.raises(ValueError, match="switch_width"):
        Dreiding("dreiding", top, cutoff=6.0, switch_width=7.0)


def test_missing_parameters_are_reported_together():
    top = MolecularTopology.from_bonds(["C_3", "Xx_", "Yy_"],
                                       [(0, 1), (0, 2)], bond_orders=[1.0] * 2)
    with pytest.raises(KeyError) as err:
        Dreiding("dreiding", top, cutoff=10.0)
    msg = str(err.value)
    assert "Xx_" in msg and "Yy_" in msg


# typing, config, training
@pytest.mark.parametrize("smiles,expect", [
    ("CC", ["C_3", "C_3"] + ["H_"] * 6),
    ("C=C", ["C_2", "C_2"] + ["H_"] * 4),
    ("c1ccccc1", ["C_R"] * 6 + ["H_"] * 6),
    ("CO", ["C_3", "O_3"] + ["H_"] * 3 + ["H__HB"]),
])
def test_from_atoms_typing_and_bond_orders(smiles, expect):
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    model = Dreiding.from_atoms(mol, "dreiding", cutoff=12.0)
    assert model.topology.types == expect
    orders = model.topology.bond_orders
    assert len(orders) == len(model.topology.bonds)
    if smiles == "C=C":
        assert max(orders) == 2.0
    if smiles == "c1ccccc1":
        assert sorted(set(orders)) == [1.0, 1.5]
        # every aromatic carbon is a planar inversion center
        assert model.inversion_index.shape[1] == 18      # 6 centers x 3 axes


def test_from_atoms_gasteiger_charges():
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    model = Dreiding.from_atoms(mol, "dreiding", charges="gasteiger",
                                cutoff=12.0)
    assert model.has_coulomb
    assert float(model.charge.sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(model.charge[1]) < 0.0          # the oxygen
    with pytest.raises(ValueError, match="charge scheme"):
        Dreiding.from_atoms(mol, "dreiding", charges="mulliken")


def test_from_config_and_key_translation(tmp_path):
    from xnn.common.config.loaders import from_dict
    from xnn.common.models import build_model
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]]
    # upstream spellings: r_max -> cutoff, forcefield -> ffield, orders ->
    # bond_orders
    cfg = from_dict({"model": {"name": "dreiding", "r_max": 12.0,
                               "forcefield": "dreiding/X6", "types": types,
                               "bonds": bonds, "orders": [1.0] * 7,
                               "bond_style": "morse",
                               "trainable": ["radius", "vdw_d0"]}})
    assert cfg.model.cutoff == 12.0
    model = build_model(cfg.model)
    assert isinstance(model, Dreiding)
    assert model.ff.form == "x6" and model.bond_style == "morse"
    assert model.ff.params["radius"].requires_grad
    assert not model.ff.params["angle_k"].requires_grad
    # the same model from a saved topology file
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 7)
    path = tmp_path / "top.json"
    top.save(path)
    cfg2 = from_dict({"model": {"name": "dreiding", "cutoff": 12.0,
                                "topology": str(path)}})
    m2 = build_model(cfg2.model)
    assert m2.topology.bond_orders == [1.0] * 7
    z = [6, 6] + [1] * 6
    g = _graph(_ethane_pos(33.0), z, 12.0)
    assert float(m2(g)["energy"]) == pytest.approx(
        float(Dreiding("dreiding", top, cutoff=12.0)(g)["energy"]), rel=1e-12)
    with pytest.raises(ValueError, match="topology"):
        build_model(from_dict({"model": {"name": "dreiding"}}).model)


def test_trainable_selection_and_gradients():
    model, z = _ethane_model(trainable=("radius", "torsion_v", "charge"))
    model.charge.data.fill_(0.1)
    out = model(_graph(_ethane_pos(12.0), z, 14.0))
    out["energy"].pow(2).sum().backward()
    P = model.ff.params
    assert P["radius"].requires_grad and P["radius"].grad is not None
    assert P["torsion_v"].requires_grad and P["torsion_v"].grad is not None
    assert model.charge.requires_grad and model.charge.grad is not None
    assert not P["angle_k"].requires_grad and not P["vdw_r0"].requires_grad
    frozen, _ = _ethane_model()
    assert not any(p.requires_grad for p in frozen.parameters())
    every, _ = _ethane_model(trainable="all")
    assert all(p.requires_grad for p in every.ff.params.values())
    assert every.charge.requires_grad
    with pytest.raises(ValueError, match="unknown trainable"):
        _ethane_model(trainable=("not_a_group",))


def test_training_step_reduces_loss():
    # refit the generators against energies from a perturbed force field --
    # the "retarget DREIDING to a new system" task
    torch.manual_seed(0)
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 7)
    target_lib = read_dreiding("dreiding")
    target_lib.radius["C_3"] = 0.80
    target_lib.torsion_v["a"] = 3.1
    target = Dreiding(target_lib, top, cutoff=14.0)
    model = Dreiding("dreiding", top, cutoff=14.0,
                     trainable=("radius", "torsion_v"))
    graphs = [_graph(torch.tensor(_ethane_pos(phi))
                     + 0.05 * torch.randn(8, 3), [6, 6] + [1] * 6, 14.0)
              for phi in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)]
    batch = collate(graphs)
    want = target(batch)["energy"].detach()
    opt = torch.optim.Adam([p for p in model.parameters()
                            if p.requires_grad], lr=3e-3)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = (model(batch)["energy"] - want).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.05 * losses[0]
    # the fit moves the right generators toward the right values
    assert float(model.ff.params["radius"][model.ff.type_index("C_3")]) > 0.775


def test_shared_forcefield_accumulates_gradients():
    ff = DreidingForceField("dreiding", trainable=("vdw_d0",))
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    ethane = Dreiding(ff, MolecularTopology.from_bonds(
        types, bonds, bond_orders=[1.0] * 7), cutoff=14.0)
    pos, z, eth_top = _ethylene(5.0)
    ethylene = Dreiding(ff, eth_top, cutoff=14.0)
    assert ethane.ff is ethylene.ff
    ethane(_graph(_ethane_pos(11.0), [6, 6] + [1] * 6, 14.0))["energy"] \
        .pow(2).sum().backward()
    g1 = ff.params["vdw_d0"].grad.clone()
    ethylene(_graph(pos, z, 14.0))["energy"].pow(2).sum().backward()
    assert float((ff.params["vdw_d0"].grad - g1).abs().max()) > 0.0


def test_export_library_roundtrip(tmp_path):
    model, z = _ethane_model(trainable="all")
    with torch.no_grad():
        model.ff.params["radius"][model.ff.type_index("C_3")] = 0.81
        model.ff.params["torsion_v"][0] *= 1.5
    lib = model.export_library()
    assert lib.radius["C_3"] == pytest.approx(0.81, abs=1e-12)
    assert lib.torsion_v["a"] == pytest.approx(3.0, abs=1e-12)
    assert lib.vdw_r0["C_3"] == pytest.approx(3.8983, abs=1e-9)
    assert lib.templates == read_dreiding("dreiding").templates
    # a model rebuilt from the exported library reproduces the energies
    back = Dreiding(read_dreiding(lib.save(tmp_path / "t.json")),
                    model.topology, cutoff=14.0)
    g = _graph(_ethane_pos(29.0), z, 14.0)
    assert float(back(g)["energy"]) == pytest.approx(float(model(g)["energy"]),
                                                     rel=1e-12)


def test_masses_property():
    model, _ = _ethane_model()
    m = model.masses
    assert m.shape == (8,)
    assert float(m[0]) == pytest.approx(12.011, abs=1e-9)
    assert float(m[2]) == pytest.approx(1.008, abs=1e-9)


def test_dataset_training_smoke():
    types = ["C_3", "C_3"] + ["H_"] * 6
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    top = MolecularTopology.from_bonds(types, bonds, bond_orders=[1.0] * 7)
    model = ForceStressOutput(
        Dreiding("dreiding", top, cutoff=12.0, trainable=("vdw_d0",)))
    z = [6, 6] + [1] * 6
    structures = []
    for phi in (0.0, 30.0, 60.0):
        p = torch.tensor(_ethane_pos(phi))
        structures.append({"pos": p, "atomic_numbers": torch.tensor(z),
                           "energy": torch.tensor(0.0),
                           "forces": torch.zeros(8, 3)})
    ds = AtomicDataset(structures, cutoff=12.0)
    batch = collate([ds[i] for i in range(len(ds))])
    pred = model(batch)
    loss, parts = weighted_loss(pred, batch, 1.0, 1.0, 0.0)
    loss.backward()
    assert "energy_mse" in parts and "force_mse" in parts
    assert model.model.ff.params["vdw_d0"].grad is not None
