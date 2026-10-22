"""ReaxFF / ReaxFF-nn (``ffnn``): equation-level references and behavior.

Every check here is self-contained: a small parameter library is built with
:func:`~xnn.ffnn.models.ffield.template_library` and the model's terms are
compared against direct scalar evaluations of the published equations
(van Duin et al., J. Phys. Chem. A 105, 9396, 2001; Xue et al., PCCP 23,
19457, 2021) -- the clean-room convention this code base uses when a
third-party reference cannot be redistributed (see the fidelity notes in the
documentation). Also covered: EEM charge equilibration (analytic two-atom
solution, neutrality, total-charge constraint), autograd forces against
finite differences, rotation/translation invariance, size extensivity,
batching, valence/torsion enumeration, trainability, library round-trips
(ReaxFF-nn JSON and the SEAMM ``.frc`` format), the shipped published fields
and config plumbing.
"""
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate, structure_to_graph
from xnn.common.models import ForceStressOutput, build_model
from xnn.ffnn.models import ReaxFF, read_ffield, template_library
from xnn.ffnn.models.ffield import (
    dedup_torsion_types,
    default_pair_cutoff,
    lammps_pair_parameters,
    resolve_torsion,
    to_forcefield,
    write_ffield,
)
from xnn.ffnn.models.reaxff import (
    KCAL_TO_EV,
    KE,
    LAMMPS_C_ELE,
    LAMMPS_EV_TO_KCAL,
    LAMMPS_PI,
    LAMMPS_QEQ_KE,
    LAMMPS_THB_CUT,
    nonbonded_taper,
    reverse_edge_permutation,
    taper_up,
)


@pytest.fixture(autouse=True)
def _f64():
    """Run every test in float64 and restore the previous default dtype."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _graph(pos, z, cutoff, cell=None):
    """Build a single-structure graph from raw arrays."""
    s = {"pos": np.asarray(pos, dtype=float), "atomic_numbers": np.asarray(z)}
    if cell is not None:
        s["cell"] = np.asarray(cell, dtype=float)
    return structure_to_graph(s, cutoff=cutoff)


def _methanol():
    """A methanol-like CH3-OH toy geometry (Angstrom)."""
    pos = [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.60, 0.90, 0.0],
           [-0.60, -0.50, 0.80], [-0.60, -0.50, -0.80], [1.80, 0.85, 0.0]]
    z = [6, 8, 1, 1, 1, 1]
    return np.array(pos), np.array(z)


def _model(nn, **kwargs):
    """A CHO model on the template library."""
    return ReaxFF(template_library(["C", "H", "O"], nn=nn), nn=nn, **kwargs)


# equation-level references
def test_uncorrected_bond_order_equation():
    """A C2 dimer reproduces eq 2 of van Duin 2001 (scalar recomputation)."""
    lib = template_library(["C"], nn=False)
    model = ReaxFF(lib, nn=False, keep_intermediates=True)
    r = 1.35
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], model.cutoff)
    model(g)
    it = model.intermediates
    p = lib.p
    botol = 0.01 * p["cutoff"]
    # like-species pair radii inherit the atomic values (combination rules)
    e1 = (1.0 + botol) * math.exp(p["bo1_C-C"] * (r / p["rosi_C"]) ** p["bo2_C-C"])
    e2 = math.exp(p["bo3_C-C"] * (r / p["ropi_C"]) ** p["bo4_C-C"])
    e3 = math.exp(p["bo5_C-C"] * (r / p["ropp_C"]) ** p["bo6_C-C"])
    assert e1 > 2 * botol and e2 > 2 * botol  # away from the switching window
    # the model inflates r by a 1e-9 numerical guard; compare at 1e-8
    assert torch.allclose(it["eterm1"], torch.full((2,), e1), atol=1e-8)
    assert torch.allclose(it["bop_si"], torch.full((2,), e1 - botol), atol=1e-8)
    assert torch.allclose(it["bop_pi"], torch.full((2,), e2), atol=1e-8)
    assert torch.allclose(it["bop_pp"], torch.full((2,), e3), atol=1e-8)


@pytest.mark.parametrize("conventions", ["lammps", "reaxff-nn"])
def test_bond_order_corrections_and_bond_energy_equations(conventions):
    """Corrections f1/f4/f5 (eq 3) and E_bond (eq 5) for a dimer, by hand."""
    lib = template_library(["C"], nn=False)
    model = ReaxFF(lib, nn=False, conventions=conventions,
                   keep_intermediates=True)
    r = 1.35
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], model.cutoff)
    out = model(g)
    it = model.intermediates
    p = lib.p

    bop = float(it["bop"][0])
    deltap = bop                       # one bond per atom in a dimer
    val, valboc = p["val_C"], p["valboc_C"]
    dv = deltap - val
    f2 = 2.0 * math.exp(-p["boc1"] * dv)
    f3 = (-1.0 / p["boc2"]) * math.log(math.exp(-p["boc2"] * dv))
    f1 = (val + f2) / (val + f2 + f3)
    d_boc = deltap - valboc
    df = p["boc4_C"] * bop ** 2 - d_boc
    f4 = 1.0 / (1.0 + math.exp(-p["boc3_C"] * df + p["boc5_C"]))
    bo0 = bop * f1 * f4 * f4
    assert abs(float(it["bo0"][0]) - bo0) < 1e-12

    bopi = float(it["bop_pi"][0]) * f1 * f1 * f4 * f4
    bopp = float(it["bop_pp"][0]) * f1 * f1 * f4 * f4
    bosi = bo0 - bopi - bopp
    desi = p["Desi_C-C"] * KCAL_TO_EV
    depi = p["Depi_C-C"] * KCAL_TO_EV
    depp = p["Depp_C-C"] * KCAL_TO_EV
    guard = 0.0 if conventions == "lammps" else 1e-9
    esi = desi * bosi * math.exp(p["be1_C-C"] * (1.0 - (bosi + guard) ** p["be2_C-C"]))
    e_bond = -(esi + depi * bopi + depp * bopp)   # one bond
    assert abs(float(out["e_bond"][0]) - e_bond) < 1e-10


@pytest.mark.parametrize("conventions", ["lammps", "reaxff-nn"])
def test_eem_two_atom_analytic(conventions):
    """EEM charges of a heteronuclear dimer match the analytic solution."""
    lib = template_library(["C", "O"], nn=False)
    model = ReaxFF(lib, nn=False, conventions=conventions)
    r = 2.0
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 8], model.cutoff)
    out = model(g)
    p = lib.p
    gamma = math.sqrt(p["gamma_C"] * p["gamma_O"])
    tap = float(nonbonded_taper(torch.tensor(r), model.cutoff))
    kernel = tap / (r ** 3 + (1.0 / gamma) ** 3) ** (1.0 / 3.0)
    if conventions == "lammps":
        # fix qeq/reaxff solves with 14.4 eV A; the energy uses C_ele and
        # converts the self energy with 23.02 kcal/mol per eV
        k_solve, k_coul = LAMMPS_QEQ_KE, LAMMPS_C_ELE * KCAL_TO_EV
        self_scale = LAMMPS_EV_TO_KCAL * KCAL_TO_EV
    else:
        k_solve = k_coul = KE
        self_scale = 1.0
    # stationarity of E(q) = sum_i (chi_i q_i + mu_i q_i^2) + H q_C q_O
    # under q_C + q_O = 0:  q_C = (chi_O - chi_C) / (2 mu_C + 2 mu_O - 2 H)
    h = k_solve * kernel
    q_c = (p["chi_O"] - p["chi_C"]) / (2 * p["mu_C"] + 2 * p["mu_O"] - 2 * h)
    q = out["charges"]
    assert abs(float(q[0]) - q_c) < 1e-9
    assert abs(float(q.sum())) < 1e-12
    # E_self and E_Coulomb from the same charges, by hand
    e_self = q_c * p["chi_C"] + q_c ** 2 * p["mu_C"] \
        + (-q_c) * p["chi_O"] + q_c ** 2 * p["mu_O"]
    assert abs(float(out["e_self"][0]) - self_scale * e_self) < 1e-9
    assert abs(float(out["e_coulomb"][0]) - k_coul * kernel * q_c * (-q_c)) < 1e-9


def test_total_charge_constraint():
    """A per-structure ``total_charge`` shifts the EEM solution."""
    model = _model(nn=False)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    g.total_charge = torch.tensor([1.0])
    q = model(g)["charges"]
    assert abs(float(q.sum()) - 1.0) < 1e-10


def test_vdw_dimer_analytic():
    """Beyond the bond cutoff only vdW + Coulomb remain (eqs 12-13)."""
    lib = template_library(["C"], nn=False)
    model = ReaxFF(lib, nn=False)
    r = 4.0                                    # beyond any bond cutoff
    g = _graph([[0, 0, 0], [r, 0, 0]], [6, 6], model.cutoff)
    out = model(g)
    p = lib.p
    assert float(out["e_bond"][0]) == 0.0
    tap = float(nonbonded_taper(torch.tensor(r), model.cutoff))
    gammaw = p["gammaw_C"]
    f13 = (r ** p["vdw1"] + (1.0 / gammaw) ** p["vdw1"]) ** (1.0 / p["vdw1"])
    ex = math.exp(0.5 * p["alfa_C"] * (1.0 - f13 / (2.0 * p["rvdw_C"])))
    e_vdw = tap * p["Devdw_C"] * KCAL_TO_EV * (ex ** 2 - 2.0 * ex)
    assert abs(float(out["e_vdw"][0]) - e_vdw) < 1e-9
    # a like-species dimer is charge-neutral by symmetry
    assert torch.allclose(out["charges"], torch.zeros(2), atol=1e-12)


def test_angle_geometry_and_count():
    """Water: one valence angle, theta from the law of cosines."""
    model = _model(nn=False, keep_intermediates=True)
    d, ang = 0.96, math.radians(104.5)
    pos = [[0, 0, 0], [d, 0, 0], [d * math.cos(ang), d * math.sin(ang), 0]]
    g = _graph(pos, [8, 1, 1], model.cutoff)
    model(g)
    it = model.intermediates
    assert it["theta"].shape == (1,)                    # exactly one angle
    assert int(it["angle_center"][0]) == 0              # centered on oxygen
    assert abs(float(it["theta"][0]) - ang) < 1e-8


def test_angle_energy_equation_water():
    """The ReaxFF-nn valence-angle energy of water reproduces eqs 8a-8d."""
    lib = template_library(["O", "H"], nn=False)
    model = ReaxFF(lib, nn=False, conventions="reaxff-nn",
                   keep_intermediates=True)
    d, ang = 0.96, math.radians(104.5)
    pos = [[0, 0, 0], [d, 0, 0], [d * math.cos(ang), d * math.sin(ang), 0]]
    g = _graph(pos, [8, 1, 1], model.cutoff)
    out = model(g)
    it = model.intermediates
    p = lib.p

    b_src_all = g.edge_index[0][it["bond_mask"]]
    b_dst_all = g.edge_index[1][it["bond_mask"]]

    def edge(i, j):
        """Index of the directed bonded edge dst=i, src=j."""
        return int(((b_dst_all == i) & (b_src_all == j)).nonzero()[0])

    e01, e02 = edge(0, 1), edge(0, 2)   # the two O-H arms
    # per-atom quantities for the center (O = atom 0)
    delta_o = float(it["delta"][0])
    dpi_o = float(it["dpi"][0])
    nlp_o = float(it["nlp"][0])
    pbo_o = math.exp(sum(-(float(b) + 1e-9) ** 8
                         for b, dsti in zip(it["bo"], b_dst_all)
                         if int(dsti) == 0))
    dang_o = delta_o - p["valang_O"]
    sbo = dpi_o - (1.0 - pbo_o) * (dang_o + p["val8"] * nlp_o)
    if 0.0 < sbo <= 1.0:
        sbo3 = sbo ** p["val9"]
    elif 1.0 < sbo < 2.0:
        sbo3 = 2.0 - (2.0 - sbo) ** p["val9"]
    else:
        sbo3 = 0.0 if sbo <= 0 else 2.0
    theta0 = math.radians(180.0 - p["theta0_H-O-H"]
                          * (1.0 - math.exp(-p["val10"] * (2.0 - sbo3))))
    # the two O-H arms are symmetric; use their explicit edges
    bo_oh = float(it["bo"][e01])
    assert abs(bo_oh - float(it["bo"][e02])) < 1e-9
    fbo = float(taper_up(it["bo0"], model.atol, 2 * model.atol)[e01])
    f7arm = 1.0 - math.exp(-p["val3_O"] * (bo_oh + 1e-9) ** p["val4_H-O-H"])
    exp6 = math.exp(p["val6"] * dang_o)
    exp7 = math.exp(-p["val7_H-O-H"] * dang_o)
    f8 = p["val5_O"] - (p["val5_O"] - 1.0) * (2.0 + exp6) / (1.0 + exp6 + exp7)
    val1 = p["val1_H-O-H"] * KCAL_TO_EV
    eang = (fbo * fbo) * (f7arm * f7arm) * f8 * val1 \
        * (1.0 - math.exp(-p["val2_H-O-H"] * (theta0 - ang) ** 2))
    assert abs(float(out["e_angle"][0]) - eang) < 1e-9


def test_torsion_enumeration_chain():
    """An H-C-C-H chain has exactly the brute-force set of torsions."""
    model = _model(nn=True, keep_intermediates=True)
    # staggered ethane-like fragment: 2 C + 6 H
    c1, c2 = np.array([0.0, 0, 0]), np.array([1.54, 0, 0])
    hs = []
    for base, sgn in ((c1, 1.0), (c2, -1.0)):
        for k in range(3):
            a = 2 * math.pi * k / 3 + (0.0 if sgn > 0 else math.pi / 3)
            hs.append(base + [sgn * -0.36, 1.02 * math.cos(a), 1.02 * math.sin(a)])
    pos = np.vstack([c1, c2] + hs)
    z = np.array([6, 6, 1, 1, 1, 1, 1, 1])
    g = _graph(pos, z, model.cutoff)
    model(g)
    it = model.intermediates
    tors = set()
    for i, j, k, l in zip(it["tor_i"].tolist(), it["tor_j"].tolist(),
                          it["tor_k"].tolist(), it["tor_l"].tolist()):
        tors.add((i, j, k, l) if (i, j, k, l) < (l, k, j, i) else (l, k, j, i))
    # only the C-C bond is a heavy central bond: 3 H x 3 H torsions
    assert len(tors) == 9
    assert all(j in (0, 1) and k in (0, 1) for _, j, k, _ in tors)
    # angles: 3 HCH + 1 HCC per carbon... brute force: pairs of neighbors
    angs = set()
    for i, j, k in zip(it["angle_i"].tolist(), it["angle_center"].tolist(),
                       it["angle_k"].tolist()):
        angs.add((min(i, k), j, max(i, k)))
    assert len(angs) == 2 * (3 + 3)     # C(3H + C) neighbors: C(4,2)=6 pairs


def test_hydrogen_bond_term():
    """A linear O-H..O motif gives the ReaxFF-nn H-bond energy."""
    lib = template_library(["O", "H"], nn=False)
    model = ReaxFF(lib, nn=False, conventions="reaxff-nn",
                   keep_intermediates=True)
    # donor O-H at 0.98, acceptor O at 1.9 from H, collinear
    pos = [[0, 0, 0], [0.98, 0, 0], [2.88, 0, 0], [3.4, 0.9, 0]]
    z = [8, 1, 8, 1]                      # second H saturates the acceptor
    g = _graph(pos, z, model.cutoff)
    out = model(g)
    it = model.intermediates
    p = lib.p
    assert float(out["e_hbond"][0]) < 0.0
    # hand evaluation for the O0-H1..O2 triple
    triples = list(zip(it["hb_x"].tolist(), it["hb_h"].tolist(),
                       it["hb_z"].tolist()))
    idx = triples.index((0, 1, 2))
    rij, rjk, rik = 0.98, 1.90, 2.88
    cos_th = (rij + rjk ** 2 - rik ** 2) / (2 * rij * rjk)  # library convention
    bo_oh = None
    b_dst = g.edge_index[1][it["bond_mask"]]
    b_src = g.edge_index[0][it["bond_mask"]]
    for e in range(len(b_dst)):
        if int(b_dst[e]) == 0 and int(b_src[e]) == 1:
            bo_oh = float(it["bo0"][e])
    fhb = float(taper_up(torch.tensor([bo_oh]), model.hbtol, 2 * model.hbtol)[0])
    exphb1 = 1.0 - math.exp(-p["hb1_O-H-O"] * bo_oh)
    hbsum = p["rohb_O-H-O"] / rjk + rjk / p["rohb_O-H-O"] - 2.0
    exphb2 = math.exp(-p["hb2_O-H-O"] * hbsum)
    ehb = fhb * p["Dehb_O-H-O"] * KCAL_TO_EV * exphb1 * exphb2 \
        * (0.5 - 0.5 * cos_th) ** 2       # frhb = 1 well below hb_short
    assert abs(float(it["ehb"][idx]) - ehb) < 1e-8


# LAMMPS conventions (pair_style reaxff + fix qeq/reaxff)
def _bonded_edge(g, it, i, j):
    """Index of the directed bonded edge dst=i, src=j."""
    b_src = g.edge_index[0][it["bond_mask"]]
    b_dst = g.edge_index[1][it["bond_mask"]]
    return int(((b_dst == i) & (b_src == j)).nonzero()[0])


def _water_dimer():
    """A hydrogen-bonded water dimer: O0-H1..O3, H1..O3 = 1.94 Angstrom."""
    pos = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0],
           [2.90, 0.0, 0.0], [3.20, 0.90, 0.20], [3.20, -0.50, 0.80]]
    return np.array(pos), np.array([8, 1, 1, 8, 1, 1])


def test_default_conventions():
    """Classical libraries default to the LAMMPS conventions, ReaxFF-nn to its own."""
    assert _model(nn=False).conventions == "lammps"
    assert _model(nn=True).conventions == "reaxff-nn"
    with pytest.raises(ValueError):
        _model(nn=False, conventions="gulp")


def test_lammps_pair_parameter_rules():
    """Radii take the arithmetic mean, vdW the geometric; positive overrides win."""
    p = {"rosi_A": 1.0, "rosi_B": 2.0, "ropi_A": -0.1, "ropi_B": 1.0,
         "ropp_A": 1.0, "ropp_B": 1.0, "Devdw_A": 0.1, "Devdw_B": 0.4,
         "rvdw_A": 1.0, "rvdw_B": 4.0, "alfa_A": 9.0, "alfa_B": 16.0,
         "rosi_A-B": -1.0, "ropi_A-B": -1.0, "ropp_A-B": 0.7, "Devdw_A-B": 0.3,
         "rvdw_A-B": -1.0, "alfa_A-B": 11.0}
    lammps_pair_parameters(p, ["A", "B"], ["A-B"])
    assert p["rosi_A-B"] == p["rosi_B-A"] == pytest.approx(1.5)
    assert p["ropi_A-B"] == pytest.approx(0.45)
    assert p["ropp_A-B"] == pytest.approx(0.7)
    assert p["Devdw_A-B"] == pytest.approx(0.3)
    assert p["rvdw_A-B"] == pytest.approx(2.0)
    assert p["alfa_A-B"] == pytest.approx(11.0)
    assert p["rosi_B-B"] == pytest.approx(2.0)


def test_lammps_bond_list():
    """Bonds reach as far as the bond order does; H has no pi channel."""
    lib = read_ffield("CHO_cho_2008")
    r = 2.2                                     # beyond the 2.0 A X-H heuristic
    g = _graph([[0, 0, 0], [r, 0, 0]], [8, 1], 10.0)
    model = ReaxFF(lib, conventions="lammps", keep_intermediates=True)
    model(g)
    it = model.intermediates
    assert int(it["bond_mask"].sum()) == 2
    p = lib.p
    botol = 0.01 * p["cutoff"]
    e1 = (1.0 + botol) * math.exp(p["bo1_O-H"] * (r / p["rosi_O-H"]) ** p["bo2_O-H"])
    assert torch.allclose(it["bop_si"], torch.full((2,), e1 - botol), atol=1e-10)
    assert torch.all(it["bop_pi"] == 0.0) and torch.all(it["bop_pp"] == 0.0)
    nn_conv = ReaxFF(lib, conventions="reaxff-nn", keep_intermediates=True)
    nn_conv(g)
    assert int(nn_conv.intermediates["bond_mask"].sum()) == 0


def test_lammps_angle_energy_equation_water():
    """The LAMMPS valence-angle energy of water, by hand."""
    lib = template_library(["O", "H"], nn=False)
    model = ReaxFF(lib, keep_intermediates=True)
    d, ang = 0.96, math.radians(104.5)
    pos = [[0, 0, 0], [d, 0, 0], [d * math.cos(ang), d * math.sin(ang), 0]]
    g = _graph(pos, [8, 1, 1], model.cutoff)
    out = model(g)
    it = model.intermediates
    p = lib.p
    assert it["theta"].shape == (1,)
    e01 = _bonded_edge(g, it, 0, 1)
    b_dst = g.edge_index[1][it["bond_mask"]]
    delta_o, dpi_o, nlp_o = (float(it[k][0]) for k in ("delta", "dpi", "nlp"))
    pbo = math.exp(-sum(float(b) ** 8 for b, i in zip(it["bo0"], b_dst) if int(i) == 0))
    delta_e = delta_o - p["vale_O"]
    vlpex = delta_e - 2.0 * int(delta_e / 2.0)
    sbo = dpi_o - (1.0 - pbo) * (delta_o - p["valang_O"]
                                 + p["val8"] * (0.0 if vlpex >= 0 else nlp_o))
    if 0.0 < sbo <= 1.0:
        sbo2 = sbo ** p["val9"]
    elif 1.0 < sbo < 2.0:
        sbo2 = 2.0 - (2.0 - sbo) ** p["val9"]
    else:
        sbo2 = 0.0 if sbo <= 0 else 2.0
    theta0 = (180.0 - p["theta0_H-O-H"] * (1.0 - math.exp(-p["val10"] * (2.0 - sbo2)))) \
        * LAMMPS_PI / 180.0
    boa = float(it["bo0"][e01]) - LAMMPS_THB_CUT
    f7arm = 1.0 - math.exp(-p["val3_O"] * boa ** p["val4_H-O-H"])
    dang = delta_o - p["valang_O"]
    exp6 = math.exp(p["val6"] * dang)
    exp7 = math.exp(-p["val7_H-O-H"] * dang)
    f8 = p["val5_O"] - (p["val5_O"] - 1.0) * (2.0 + exp6) / (1.0 + exp6 + exp7)
    eang = f7arm ** 2 * f8 * p["val1_H-O-H"] * KCAL_TO_EV \
        * (1.0 - math.exp(-p["val2_H-O-H"] * (theta0 - ang) ** 2))
    assert abs(float(out["e_angle"][0]) - eang) < 1e-9


def test_lammps_hydrogen_atoms_center_angles_and_torsions():
    """A hydrogen-bonded water dimer has H-centered angles and torsions."""
    model = ReaxFF("CHOLiNaKCsFClI_Fedkin_2019", species=["H", "O"],
                   keep_intermediates=True)
    pos, z = _water_dimer()
    g = _graph(pos, z, model.cutoff)
    out = model(g)
    it = model.intermediates
    assert 1 in it["angle_center"].tolist()               # O0-H1..O3
    assert 1 in it["tor_j"].tolist() + it["tor_k"].tolist()
    assert float(out["e_hbond"][0]) < 0.0
    nn_conv = ReaxFF("CHOLiNaKCsFClI_Fedkin_2019", species=["H", "O"],
                     conventions="reaxff-nn", keep_intermediates=True)
    nn_conv(g)
    assert 1 not in nn_conv.intermediates["angle_center"].tolist()


def test_lammps_hydrogen_bond_term():
    """A bent O-H..O motif gives the LAMMPS H-bond energy, by hand."""
    lib = template_library(["O", "H"], nn=False)
    model = ReaxFF(lib, keep_intermediates=True)
    t = math.radians(15.0)
    h = np.array([0.98, 0.0, 0.0])
    o2 = h + 1.9 * np.array([math.cos(t), math.sin(t), 0.0])
    pos = [[0, 0, 0], h, o2, o2 + [0.52, 0.8, 0.0]]
    g = _graph(pos, [8, 1, 8, 1], model.cutoff)
    model(g)
    it = model.intermediates
    p = lib.p
    triples = list(zip(it["hb_x"].tolist(), it["hb_h"].tolist(), it["hb_z"].tolist()))
    idx = triples.index((0, 1, 2))
    bo_oh = float(it["bo0"][_bonded_edge(g, it, 1, 0)])
    cos_th = math.cos(math.pi - t)                        # X-H-Z angle, 165 deg
    ehb = p["Dehb_O-H-O"] * KCAL_TO_EV * (1.0 - math.exp(-p["hb1_O-H-O"] * bo_oh)) \
        * math.exp(-p["hb2_O-H-O"] * (p["rohb_O-H-O"] / 1.9 + 1.9 / p["rohb_O-H-O"] - 2.0)) \
        * (0.5 - 0.5 * cos_th) ** 2
    assert abs(float(it["ehb"][idx]) - ehb) < 1e-10


def test_lammps_forces_hold_charges_fixed():
    """LAMMPS-convention forces are the gradient at fixed charges."""
    model = _model(nn=False)
    fs = ForceStressOutput(model)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    forces = fs(g)["forces"].detach().numpy()
    q0 = model(g)["charges"].detach()
    model._eem_charges = lambda *args: q0

    def energy(pp):
        return float(model(_graph(pp, z, model.cutoff))["energy"][0])

    d = 1e-5
    for (i, c) in [(0, 0), (1, 1), (3, 2), (5, 0)]:
        pp, pm = pos.copy(), pos.copy()
        pp[i, c] += d
        pm[i, c] -= d
        assert abs(forces[i, c] + (energy(pp) - energy(pm)) / (2 * d)) < 1e-6


def test_fedkin_field_ships():
    """The Fedkin 2019 electrolyte-water field loads with its H-bond roles."""
    lib = read_ffield("CHOLiNaKCsFClI_Fedkin_2019")
    assert lib.name == "reaxff/CHOLiNaKCsFClI_Fedkin_2019"
    assert set(lib.spec) == {"H", "Li", "C", "O", "F", "Na", "Cl", "K", "I", "Cs"}
    # its general-block slots 35/36 hold unrelated settings, not thresholds
    assert lib.p["acut"] == pytest.approx(1e-4) and lib.p["hbtol"] == pytest.approx(1e-4)
    assert lib.p["theta0_H-O-H"] == pytest.approx(85.8)
    assert lib.p["Dehb_O-H-O"] == pytest.approx(-3.58)
    model = ReaxFF(lib, species=["H", "O", "Na", "Cl"])
    assert model.conventions == "lammps"
    assert model.hb_flag.tolist() == [1, 2, 0, 2]
    assert model.vdw_type == 1


def _parse_ffield(text):
    """Blocks of a written ffield: general values and per-block token rows."""
    lines = [line.split("!")[0].split() for line in text.splitlines()]
    k = 1
    n = int(lines[k][0])
    general = [float(t[0]) for t in lines[k + 1:k + 1 + n]]
    k += 1 + n
    n = int(lines[k][0])
    k += 4
    atoms = [lines[k + 4 * i:k + 4 * i + 4] for i in range(n)]
    k += 4 * n
    n = int(lines[k][0])
    k += 2
    bonds = [lines[k + 2 * i:k + 2 * i + 2] for i in range(n)]
    k += 2 * n
    blocks = {}
    for name in ("offd", "angles", "torsions", "hbonds"):
        n = int(lines[k][0])
        blocks[name] = lines[k + 1:k + 1 + n]
        k += 1 + n
    return general, atoms, bonds, blocks


def test_write_ffield_layout():
    """The written ffield has LAMMPS's block counts, columns and values."""
    lib = read_ffield("CHO_cho_2008")
    general, atoms, bonds, blocks = _parse_ffield(write_ffield(lib))
    assert len(general) == 39
    assert general[0] == 50.0 and general[5] == 70.0      # boc1, kc2
    assert [a[0][0] for a in atoms] == lib.spec
    assert all(len(a[0]) == 9 and all(len(r) == 8 for r in a[1:]) for a in atoms)
    assert len(bonds) == 6 and all(len(b[0]) == 10 and len(b[1]) == 8 for b in bonds)
    assert [len(blocks[k]) for k in ("offd", "angles", "torsions", "hbonds")] \
        == [len(lib.offd), len(lib.angs), 26, 1]
    assert all(len(r) == 11 for r in blocks["torsions"])
    assert sum(r[0] == "0" and r[3] == "0" for r in blocks["torsions"]) == 5
    index = {sym: str(i + 1) for i, sym in enumerate(lib.spec)}
    ch = next(b for b in bonds if b[0][:2] == [index["C"], index["H"]])
    assert float(ch[0][2]) == lib.p["Desi_C-H"]
    o = atoms[lib.spec.index("O")]
    assert float(o[1][7]) == 2.0                            # hydrogen-bond flag
    assert blocks["hbonds"][0][:3] == [index["O"], index["H"], index["O"]]
    assert float(blocks["hbonds"][0][4]) == lib.p["Dehb_O-H-O"]
    assert "70.0000" in write_ffield(lib, digits=4)
    with pytest.raises(ValueError):
        write_ffield(template_library(["C", "H"], nn=True))


def test_export_library_keeps_bonds_and_energies(tmp_path):
    """A model's exported library lists only its bonds and gives its energies."""
    model = ReaxFF("CHOLiNaKCsFClI_Fedkin_2019", species=["H", "O", "Na", "Cl"])
    pos, z = _water_dimer()
    g = _graph(pos, z, model.cutoff)
    lib = model.export_library()
    assert len(lib.bonds) == 10                 # every H/O/Na/Cl pair has a row
    path = to_forcefield(lib, name="reaxff/refit").write(tmp_path / "refit.frc")
    e1 = float(model(g)["energy"][0])
    e2 = float(ReaxFF(str(path))(g)["energy"][0])
    assert abs(e1 - e2) < 1e-10
    general, atoms, bonds, _ = _parse_ffield(write_ffield(lib, tmp_path / "ffield"))
    assert len(atoms) == 4 and len(bonds) == 10
    assert general[11:13] == [0.0, model.cutoff]       # the nonbonded taper


# invariances, forces, batching
@pytest.mark.parametrize("nn", [False, True])
def test_rotation_translation_invariance(nn):
    """Energy is invariant and forces are equivariant under rigid motions."""
    model = _model(nn=nn)
    fs = ForceStressOutput(model)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    out = fs(g)
    # rotation + translation
    a, b, c = 0.3, -1.1, 0.7
    rz = np.array([[math.cos(a), -math.sin(a), 0],
                   [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)],
                   [0, math.sin(b), math.cos(b)]])
    rot = rz @ rx
    g2 = _graph(pos @ rot.T + c, z, model.cutoff)
    out2 = fs(g2)
    assert abs(float(out["energy"][0] - out2["energy"][0])) < 1e-9
    f_rot = out["forces"].detach().numpy() @ rot.T
    assert np.abs(out2["forces"].detach().numpy() - f_rot).max() < 1e-8


@pytest.mark.parametrize("nn", [False, True])
def test_forces_match_finite_differences(nn):
    """Autograd forces equal central finite differences of the energy."""
    model = _model(nn=nn, conventions="reaxff-nn")
    fs = ForceStressOutput(model)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    forces = fs(g)["forces"].detach().numpy()

    def energy(pp):
        return float(model(_graph(pp, z, model.cutoff))["energy"][0])

    d = 1e-5
    for (i, c) in [(0, 0), (1, 1), (3, 2), (5, 0)]:
        pp, pm = pos.copy(), pos.copy()
        pp[i, c] += d
        pm[i, c] -= d
        f_num = -(energy(pp) - energy(pm)) / (2 * d)
        assert abs(forces[i, c] - f_num) < 1e-6


@pytest.mark.parametrize("nn", [False, True])
def test_batching_matches_individual(nn):
    """A collated batch reproduces the per-structure energies and terms."""
    model = _model(nn=nn)
    pos, z = _methanol()
    g1 = _graph(pos, z, model.cutoff)
    g2 = _graph(pos * 1.05 + 3.0, z, model.cutoff)
    batch = collate([g1, g2])
    out = model(batch)
    for i, g in enumerate((g1, g2)):
        single = model(g)
        assert abs(float(out["energy"][i] - single["energy"][0])) < 1e-10
        for key in ("e_bond", "e_angle", "e_vdw", "e_coulomb", "e_hbond"):
            assert abs(float(out[key][i] - single[key][0])) < 1e-10


def test_size_extensivity():
    """Two far-separated copies have exactly twice the energy of one."""
    model = _model(nn=True)
    pos, z = _methanol()
    e1 = float(model(_graph(pos, z, model.cutoff))["energy"][0])
    pos2 = np.vstack([pos, pos + np.array([50.0, 0, 0])])
    z2 = np.concatenate([z, z])
    e2 = float(model(_graph(pos2, z2, model.cutoff))["energy"][0])
    assert abs(e2 - 2 * e1) < 1e-9


def test_energy_decomposition_sums_to_total():
    """The per-term decomposition plus atomic offsets equals the energy."""
    model = _model(nn=True)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    out = model(g)
    terms = sum(float(v[0]) for k, v in out.items() if k.startswith("e_"))
    atomic = -float(model.params["atomic"][model.z_to_index[g.atomic_numbers]].sum())
    assert abs(terms + atomic - float(out["energy"][0])) < 1e-10
    assert abs(float(out["node_energy"].sum() - out["energy"][0])) < 1e-10


def test_periodic_stress_and_bond_orders():
    """A periodic diamond-like cell runs with stress and symmetric BOs."""
    model = ReaxFF(template_library(["C"], nn=True), keep_intermediates=True)
    fs = ForceStressOutput(model, compute_stress=True)
    a = 3.57
    frac = np.array([[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.25],
                     [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0, 0.5, 0.5], [0.25, 0.75, 0.75]])
    cell = np.eye(3) * a
    g = _graph(frac @ cell, [6] * 8, model.cutoff, cell=cell)
    out = fs(g)
    assert torch.isfinite(out["stress"]).all()
    it = model.intermediates
    src = g.edge_index[0][it["bond_mask"]]
    dst = g.edge_index[1][it["bond_mask"]]
    shifts = g.cell_shifts[it["bond_mask"]]
    rev = reverse_edge_permutation(src, dst, shifts)
    assert torch.allclose(it["bo0"], it["bo0"][rev], atol=1e-12)


# training, libraries, config
def test_trainable_selection_and_step():
    """Only requested classical groups (plus nn weights) receive gradients."""
    model = _model(nn=True, trainable=("Desi", "ang_val1"))
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    loss = model(g)["energy"].pow(2).sum()
    loss.backward()
    assert model.params["Desi"].requires_grad
    assert model.params["Desi"].grad is not None
    assert model.params["ang_val1"].requires_grad
    assert not model.params["be1"].requires_grad
    assert model.weights["fewi"].grad is not None
    # frozen-by-default classical model exposes no trainable classical params
    frozen = _model(nn=False)
    assert not any(p.requires_grad for p in frozen.params.values())
    with pytest.raises(ValueError, match="unknown trainable group"):
        _model(nn=False, trainable=("bond", "angle"))


def test_training_step_reduces_loss():
    """A few Adam steps on the nn weights reduce an energy-fitting loss."""
    model = _model(nn=True)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    target = torch.tensor([-30.0])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-2)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = (model(g)["energy"] - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_library_json_roundtrip(tmp_path):
    """export_library -> save -> read -> identical energies."""
    model = _model(nn=True)
    pos, z = _methanol()
    g = _graph(pos, z, model.cutoff)
    e1 = float(model(g)["energy"][0])
    path = tmp_path / "ffield.json"
    model.export_library().save(path)
    model2 = ReaxFF(str(path))
    assert model2.nn and model2.species == model.species
    assert abs(float(model2(g)["energy"][0]) - e1) < 1e-12


def test_frc_roundtrip_of_template_library(tmp_path):
    """A classical library survives a trip through the .frc format exactly."""
    lib = template_library(["C", "H", "O"], nn=False)
    path = to_forcefield(lib, name="reaxff/template_CHO").write(tmp_path / "t.frc")
    parsed = read_ffield(str(path))
    assert parsed.name == "reaxff/template_CHO"
    assert parsed.spec == lib.spec and parsed.bonds == lib.bonds
    assert parsed.angs == lib.angs and parsed.torp == lib.torp
    assert parsed.hbs == lib.hbs
    shared = set(lib.p) & set(parsed.p)
    # the per-species reference energy has no column in the format
    assert shared >= {k for k in lib.p
                      if not k.startswith(("n.u.", "atomic_"))}
    assert max(abs(lib.p[k] - parsed.p[k]) for k in shared) == 0.0
    # the seed's own thresholds survive (published fields, with zeros in
    # those slots, get the customary 1e-4 instead)
    assert parsed.p["acut"] == lib.p["acut"] == 0.001
    assert read_ffield("CHO_cho_2008").p["acut"] == pytest.approx(1e-4)
    # and the models agree
    g = _graph(*_methanol(), 10.0)
    e1 = ReaxFF(lib, nn=False)(g)["energy"]
    e2 = ReaxFF(parsed, nn=False)(g)["energy"]
    assert float((e1 - e2).abs()) < 1e-10
    # network weights cannot go in a .frc file
    with pytest.raises(ValueError):
        to_forcefield(template_library(["C", "H"], nn=True))


def test_published_ffield_parses():
    """The published CHO combustion field loads from its shipped .frc file.

    Guards the SEAMM-name -> ReaxFF-parameter map against a real, externally
    authored field. Reference values are read off the file's own column
    headers (bond and angle keys keep the file's orientation).
    """
    lib = read_ffield("CHO_cho_2008")
    assert lib.name == "reaxff/CHO_cho_2008"
    assert lib.spec == ["H", "C", "O"]
    assert set(lib.bonds) == {"H-H", "C-H", "C-C", "O-H", "O-C", "O-O"}
    assert lib.hbs == ["O-H-O"]
    p = lib.p
    assert p["vdw1"] == pytest.approx(1.5591)          # general: PvdW,1
    assert p["cutoff"] == pytest.approx(0.1)           # general: BO_cutoff
    assert p["rosi_C"] == pytest.approx(1.3825)        # atomic: R0,alpha
    assert p["gammaw_O"] == pytest.approx(7.7719)      # atomic: gamma,w
    assert p["ropp_C"] == pytest.approx(1.2104)        # atomic: R0,pi-pi
    assert p["ovun2_H"] == pytest.approx(-15.7683)     # atomic: Povun,2
    assert p["Desi_C-H"] == pytest.approx(170.232)     # bond: De,sigma
    assert p["bo1_C-C"] == pytest.approx(-0.0750)      # bond: Pbo_1
    assert p["rvdw_O-C"] == pytest.approx(1.8523)      # off-diagonal: RvdW
    assert p["theta0_C-C-C"] == pytest.approx(67.2326)  # angle: Theta0
    assert p["rohb_O-H-O"] == pytest.approx(1.9682)    # hbond: Rhb
    assert p["Dehb_O-H-O"] == pytest.approx(-4.4628)   # hbond: Ehb
    # the heat increments are carried but, as in LAMMPS, not part of the energy
    assert lib.heat_increment["C"] == pytest.approx(199.03)
    assert p["acore2_C"] == pytest.approx(0.0)        # atomic: Acore,2
    assert "atomic_C" not in p
    model = ReaxFF(lib, nn=False)
    g = _graph([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], [6, 1], model.cutoff)
    assert torch.isfinite(model(g)["energy"]).all()


def test_all_shipped_reaxff_fields_load():
    """Every ReaxFF field shipped with xnn parses and builds a model."""
    from xnn.ffnn.common import list_forcefields
    names = [n for n in list_forcefields() if n.startswith("reaxff/")]
    assert len(names) >= 12
    for name in names:
        lib = read_ffield(name)
        assert lib.spec and lib.bonds
        ReaxFF(lib, nn=False)


def test_torsion_types_keep_central_bond_distinct():
    """``C-O-C-H`` and ``H-O-C-C`` are different torsions and both are kept.

    Published fields list them with different parameters; only the full
    reversal ``l-k-j-i`` is a duplicate, and the lookup prefers the reversal
    over a central-bond swap.
    """
    assert dedup_torsion_types(["C-O-C-H", "H-O-C-C", "H-C-O-C"]) \
        == ["C-O-C-H", "H-O-C-C"]
    p = {"V2_C-O-C-H": 1.0, "V2_H-O-C-C": 2.0, "V2_X-C-C-X": 9.0}
    torp = ["C-O-C-H", "H-O-C-C", "X-C-C-X"]
    assert resolve_torsion(p, torp, "C-O-C-H", "V2") == 1.0
    assert resolve_torsion(p, torp, "H-C-O-C", "V2") == 1.0    # reversal
    assert resolve_torsion(p, torp, "C-C-O-H", "V2") == 2.0    # reversal
    assert resolve_torsion(p, torp, "H-C-C-H", "V2") == 9.0    # wildcard
    assert resolve_torsion(p, torp, "H-O-O-H", "V2") == 0.0
    lib = read_ffield("CHO_cho_2008")
    assert len(lib.torp) == 26
    assert resolve_torsion(lib.p, lib.torp, "C-O-C-H", "V2") \
        != resolve_torsion(lib.p, lib.torp, "H-O-C-C", "V2")


def test_default_pair_cutoffs():
    """The heuristic fallback cutoffs follow the hydrogen-count rule."""
    assert default_pair_cutoff("C", "C", "rcut") == 2.5
    assert default_pair_cutoff("C", "H", "rcut") == 2.0
    assert default_pair_cutoff("C", "C", "rcuta") == 1.95
    assert default_pair_cutoff("O", "H", "rcuta") == 1.75
    assert default_pair_cutoff("H", "H", "rcuta") == 1.35


def test_from_config_and_key_translation(tmp_path):
    """build_model constructs ReaxFF from a config with translated keys."""
    path = tmp_path / "ffield.json"
    template_library(["C", "H", "O"], nn=True).save(path)
    cfg = from_dict({"model": {
        "name": "reaxff",
        "libfile": str(path),          # translated spelling of `ffield`
        "vdwcut": 9.0,                 # translated spelling of `cutoff`
        "hbshort": 6.0,
        "hblong": 7.0,
        "trainable": ["Desi"],
    }})
    assert cfg.model.cutoff == 9.0
    model = build_model(cfg.model)
    assert isinstance(model, ReaxFF)
    assert model.cutoff == 9.0 and model.hb_short == 6.0
    assert model.params["Desi"].requires_grad
    pos, z = _methanol()
    out = model(_graph(pos, z, model.cutoff))
    assert torch.isfinite(out["energy"]).all()
    # a shipped .frc field by name, through the translated "frc" key
    cfg2 = from_dict({"model": {"name": "reaxff", "frc": "CHO_cho_2008",
                                "cutoff": 10.0}})
    model2 = build_model(cfg2.model)
    assert model2.species == ["H", "C", "O"]
    assert torch.isfinite(model2(_graph(pos, z, 10.0))["energy"]).all()


def test_dataset_training_smoke(tmp_path):
    """ReaxFF-nn trains through the standard AtomicDataset/loss pipeline."""
    from xnn.common.train.losses import weighted_loss
    model = _model(nn=True)
    fs = ForceStressOutput(model)
    pos, z = _methanol()
    rng = np.random.default_rng(0)
    structures = []
    for _ in range(4):
        p = pos + 0.05 * rng.standard_normal(pos.shape)
        structures.append({"pos": p, "atomic_numbers": z,
                           "energy": -30.0 + rng.standard_normal(),
                           "forces": rng.standard_normal(pos.shape)})
    ds = AtomicDataset(structures, cutoff=model.cutoff)
    batch = collate([ds[i] for i in range(len(ds))])
    out = fs(batch)
    loss, logs = weighted_loss(out, batch, energy_weight=1.0,
                               force_weight=0.1, stress_weight=0.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.weights["fmwi"].grad is not None
