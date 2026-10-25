"""Periodic AMOEBA / HIPPO: Ewald and particle-mesh sums, switching, the iterative induction
and two 64-water boxes against Tinker (energies by term, forces, internal virial).

The boxes are a 4 x 4 x 4 lattice of randomly oriented, jittered water molecules in a
12.4294 Angstrom cell evaluated with the shipped water libraries; the references were
computed with Tinker 8.11.3 on the same coordinates (cutoffs 6 Angstrom, taper 5.4, its
default Ewald coefficient, 48^3 meshes and Tinker's default 16^3 ones, induced dipoles
converged to 1e-10 Debye).
"""
import copy
import math

import numpy as np
import pytest
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, MDIEngine
from xnn.common.deploy.step_graph import StepGraph
from xnn.common.models import ForceStressOutput
from xnn.common.models.ops import cell_volume
from xnn.ffnn.models import Amoeba, Hippo, MolecularTopology, builtin_amoeba
from xnn.ffnn.models import amoeba_ewald as ew
from xnn.ffnn.models.amoebalib import KCAL_TO_EV

EV_TO_KCAL = 1.0 / KCAL_TO_EV
L = 12.4294
CELL = np.diag([L, L, L])
Z = np.tile([8, 1, 1], 64)
# the box: 64 waters, atoms O H H per molecule, Angstrom
BOX = np.array([
    [1.649419, 1.509868, 1.506883], [1.405334, 0.668217, 1.121815], [2.585452, 1.427806, 1.689477],
    [1.630881, 1.544056, 4.648204], [0.794787, 1.990466, 4.781980], [1.631879, 0.837745, 5.294234],
    [1.534103, 1.347511, 7.696783], [0.937998, 1.769048, 7.077754], [1.044031, 0.594520, 8.027036],
    [1.827367, 1.446697, 11.077956], [1.554838, 2.270458, 11.482157], [1.470789, 0.768414, 11.651556],
    [1.678763, 4.831683, 1.420845], [1.829304, 3.897440, 1.276763], [2.356761, 5.090562, 2.044970],
    [1.685183, 4.691688, 4.566827], [1.295663, 5.546647, 4.749998], [0.953324, 4.077377, 4.623717],
    [1.881198, 4.538646, 8.152294], [2.452663, 5.170008, 8.589377], [1.663327, 4.950767, 7.316279],
    [1.702848, 4.594621, 10.872477], [2.466536, 4.027138, 10.767717], [1.201722, 4.193904, 11.582780],
    [1.405864, 7.617932, 1.408433], [0.473797, 7.644092, 1.192103], [1.753577, 6.895983, 0.884871],
    [1.592241, 7.585841, 4.686473], [1.411783, 6.988214, 5.412082], [1.147348, 7.192848, 3.935568],
    [1.721671, 7.836633, 7.745427], [1.099524, 7.410076, 8.334675], [1.421935, 7.599893, 6.867734],
    [1.445813, 7.776976, 10.945550], [1.946171, 7.509845, 11.716598], [2.042576, 8.343652, 10.456690],
    [1.425620, 10.977324, 1.641878], [1.840296, 11.828199, 1.499445], [1.004376, 10.772890, 0.807018],
    [1.858651, 10.818372, 4.698623], [2.636600, 10.685624, 4.156954], [1.696095, 9.964078, 5.098610],
    [1.695758, 10.822218, 7.977843], [1.903373, 11.137257, 7.098140], [2.304608, 10.098088, 8.123339],
    [1.725846, 10.755432, 10.532518], [1.262898, 11.527689, 10.207664], [2.552181, 11.097042, 10.874137],
    [4.495599, 1.668395, 1.695555], [3.545431, 1.777175, 1.655802], [4.687981, 1.578134, 2.628869],
    [4.816872, 1.661720, 4.869582], [4.562607, 0.965085, 4.264370], [4.004045, 2.135874, 5.044855],
    [4.355462, 1.388465, 7.891151], [4.105972, 1.108535, 7.010454], [3.530340, 1.640907, 8.305492],
    [4.929673, 1.404941, 10.506729], [4.105269, 1.137546, 10.913042], [4.670880, 2.022414, 9.822635],
    [4.518691, 4.893920, 1.786076], [5.210827, 5.545867, 1.675875], [3.735374, 5.406806, 1.985061],
    [4.625165, 4.645285, 4.533190], [4.272234, 3.899657, 5.018690], [5.187570, 5.096898, 5.162458],
    [4.735548, 4.772416, 7.861569], [4.910136, 4.859418, 8.798682], [5.385763, 5.336502, 7.442930],
    [4.522977, 4.779336, 10.987194], [4.568115, 5.025193, 11.911179], [4.444718, 3.825383, 10.996234],
    [4.559665, 7.963770, 1.424534], [5.280093, 8.488331, 1.773891], [4.197297, 8.495489, 0.715876],
    [4.949439, 7.938849, 4.755931], [5.833429, 8.294191, 4.663603], [4.423062, 8.679845, 5.056071],
    [4.774338, 7.737755, 7.206263], [4.746479, 8.561202, 6.719035], [4.801017, 8.003554, 8.125431],
    [4.700880, 7.798968, 10.653186], [4.275688, 7.968097, 11.493924], [4.869755, 6.856785, 10.654897],
    [4.685964, 10.937443, 1.559085], [5.395917, 11.104167, 2.179087], [3.915088, 11.334409, 1.964546],
    [4.778984, 10.971183, 4.676835], [4.148631, 10.667341, 4.023714], [5.437814, 10.277857, 4.715237],
    [4.670347, 10.885437, 7.669195], [3.737629, 11.050300, 7.807359], [4.767413, 9.940748, 7.789082],
    [4.568539, 10.791727, 10.794532], [5.038865, 11.559924, 11.118411], [5.226882, 10.294081, 10.309594],
    [7.598648, 1.524250, 1.521567], [6.926246, 1.931227, 2.067897], [8.011746, 2.256532, 1.064023],
    [7.585815, 1.654623, 4.775183], [8.055400, 2.457089, 5.002711], [8.255562, 0.971025, 4.794282],
    [7.866347, 1.385315, 7.600321], [8.018653, 2.257999, 7.237750], [8.348917, 0.796713, 7.019885],
    [7.822376, 1.694064, 10.859573], [6.920898, 1.815152, 11.157745], [7.776597, 1.796854, 9.909010],
    [7.893371, 4.989574, 1.675805], [7.168546, 5.155777, 2.278497], [8.531576, 4.502062, 2.196630],
    [7.867263, 4.672672, 4.557958], [7.808379, 5.571497, 4.234108], [8.578122, 4.694498, 5.198610],
    [7.871830, 4.889534, 7.717487], [8.547792, 4.304022, 8.058782], [7.047692, 4.441596, 7.908221],
    [7.912488, 4.506762, 11.046881], [7.390903, 4.778015, 11.802263], [7.578424, 5.035479, 10.322249],
    [8.122455, 7.915208, 1.579516], [8.234943, 7.060098, 1.994686], [8.190154, 8.542318, 2.299503],
    [7.629838, 7.842973, 4.559293], [7.461816, 6.905598, 4.462712], [8.095615, 8.086957, 3.759446],
    [7.889010, 7.808545, 7.775033], [7.246026, 8.328858, 8.256778], [8.503905, 7.507529, 8.444007],
    [8.019888, 7.751533, 10.609767], [7.583033, 8.573961, 10.831132], [7.682215, 7.526176, 9.742920],
    [7.629029, 10.863688, 1.583578], [7.752105, 10.271621, 0.841594], [6.943051, 10.451236, 2.108506],
    [7.677891, 10.763016, 4.639837], [7.418333, 9.928421, 4.249564], [6.995596, 10.948773, 5.284973],
    [7.605506, 10.849094, 7.684368], [7.964757, 11.672911, 8.013755], [7.369863, 11.036318, 6.775714],
    [7.600670, 10.688383, 10.808902], [8.384370, 11.236654, 10.770882], [7.212177, 10.883845, 11.661603],
    [10.816199, 1.600547, 1.620378], [10.329893, 1.635430, 2.444102], [11.601931, 2.122390, 1.783280],
    [10.855368, 1.208219, 4.814026], [10.105775, 1.784358, 4.963711], [10.767169, 0.522705, 5.476237],
    [11.011342, 1.832345, 7.603056], [10.824940, 2.683114, 8.000143], [10.917724, 1.980997, 6.662115],
    [10.788621, 1.443988, 10.915882], [10.121786, 1.066303, 11.489395], [11.563559, 0.900507, 11.058468],
    [10.661883, 4.545137, 1.427940], [10.433395, 5.381478, 1.833599], [9.831083, 4.073886, 1.365284],
    [10.879085, 4.627320, 4.792178], [9.947634, 4.829101, 4.703213], [11.318872, 5.277447, 4.244315],
    [10.537797, 4.657755, 7.780294], [10.068097, 4.434895, 6.976585], [10.823144, 3.814923, 8.133060],
    [10.967609, 4.860672, 10.550783], [10.454921, 5.483113, 10.035079], [10.315123, 4.359472, 11.039959],
    [10.833000, 7.697252, 1.583646], [11.362990, 8.167881, 2.226958], [10.365951, 8.384542, 1.108536],
    [10.971304, 8.016064, 4.716798], [11.532861, 8.143580, 3.952190], [10.863300, 7.067055, 4.779642],
    [10.795035, 7.753941, 8.071860], [10.543850, 7.159409, 7.364987], [10.033458, 8.321637, 8.189986],
    [10.977923, 7.848318, 10.988382], [11.840340, 7.441929, 10.902853], [10.939746, 8.483125, 10.272984],
    [10.951589, 10.714751, 1.567910], [10.673428, 11.520553, 1.132547], [10.247245, 10.091105, 1.391283],
    [10.985342, 10.740294, 4.718064], [10.230553, 11.263024, 4.988764], [11.299307, 11.167031, 3.920848],
    [10.747412, 10.598472, 8.055545], [10.569968, 9.856410, 8.633550], [11.089990, 11.278193, 8.635935],
    [10.902121, 10.780252, 10.730251], [10.070128, 10.328030, 10.590554], [10.808681, 11.195244, 11.587737],
]).reshape(-1, 3)

# Tinker 8.11.3 energy terms (kcal/mol), internal virial (kcal/mol, Tinker's sign) and the
# forces of the first four atoms (kcal/mol/A) of the boxes
TINKER_AMOEBA = {"e_bond": 0.00000001, "e_angle": 11.11097557, "e_urey": 3.87760444, "e_vdw": 91.74039763, "e_multipole": 34.56619814, "e_polar": -76.68072254, "energy": 64.61445325}   # noqa: E501
TINKER_AMOEBA_VIRIAL = np.array([[-651.307, 8.948, 5.89], [8.948, -580.646, 39.172], [5.89, 39.172, -558.31]])
TINKER_AMOEBA_FORCES = np.array([[-10.73049941, 10.45227510, -7.48333535], [-20.20383982, 16.71993053, -3.96736761], [-114.34227554, -61.52887798, 24.75483643], [2.93538084, 4.92715067, 2.39439932]])   # noqa: E501
TINKER_HIPPO = {"e_bond": 0.00000001, "e_angle": 0.00000000, "e_repulsion": 624.27637121, "e_dispersion": -233.11745121, "e_multipole": -183.53106497, "e_polar": -87.82485316, "e_chgtrn": -125.76374408, "energy": -5.96074220}   # noqa: E501
TINKER_HIPPO_VIRIAL = np.array([[-374.733, 34.825, -23.782], [34.825, -320.221, 61.47], [-23.782, 61.47, -455.051]])
TINKER_HIPPO_FORCES = np.array([[-4.27527556, 9.53332856, -1.34967508], [-6.45852711, 7.15980308, 3.56915377], [-10.62015902, -6.93775281, 1.99761812], [5.26340457, 5.54326675, 1.33678951]])   # noqa: E501
# the HIPPO box with the exchange polarization of the water values of Chung et al. (2022, Table 1)
EXCHPOL = {"OW": {"spring": 6.238, "size": 0.255, "damp": 1.586, "use": True},
           "HW": {"spring": 9.273, "size": 0.001, "damp": 5.685, "use": True}}
TINKER_HIPPO_EXCHPOL = {"e_polar": -81.44311527, "energy": 0.4209957}
TINKER_HIPPO_EXCHPOL_FORCES = np.array([[-4.19956136, 9.31200863, -1.31334641], [-6.55655573, 7.32607884, 3.40826096], [-10.67063455, -6.85262682, 1.96538361], [5.21699362, 5.47697107, 1.3604798]])   # noqa: E501
# the same at Tinker's default 16^3 meshes
TINKER_AMOEBA_16 = {"e_multipole": 34.15078415, "e_polar": -76.55900824, "energy": 64.32075356}
TINKER_AMOEBA_16_VIRIAL = np.array([[-651.823, 8.983, 5.918], [8.983, -581.457, 39.36], [5.918, 39.36, -559.051]])
TINKER_AMOEBA_16_FORCES = np.array([[-10.75798799, 10.38233496, -7.5004989], [-20.22193768, 16.69408847, -3.94680373], [-114.32809796, -61.53079366, 24.72361458], [2.84960679, 4.88508647, 2.40856126]])   # noqa: E501
TINKER_HIPPO_16 = {"e_dispersion": -233.54472332, "e_multipole": -183.82005897, "e_polar": -87.73205422, "energy": -6.58420937}   # noqa: E501
TINKER_HIPPO_16_VIRIAL = np.array([[-375.845, 34.784, -23.864], [34.784, -321.171, 61.643], [-23.864, 61.643, -455.898]])   # noqa: E501
TINKER_HIPPO_16_FORCES = np.array([[-4.35234624, 9.51461307, -1.36353251], [-6.46510101, 7.14793217, 3.59537636], [-10.59393288, -6.95899593, 1.97105238], [5.16639248, 5.49522924, 1.33101038]])   # noqa: E501
TOL_TERM = 1e-5        # kcal/mol
TOL_FORCE = 1e-5       # kcal/mol/A
TOL_VIRIAL = 2e-3      # kcal/mol (the reference carries three decimals)
# at 16^3 (alpha h = 0.5) the multipole mesh sums of the two codes differ by a few 1e-3 kcal/mol
TOL_COARSE_TERM, TOL_COARSE_FORCE, TOL_COARSE_VIRIAL = 5e-3, 1e-3, 0.05


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _topology(n=64):
    return MolecularTopology.from_bonds(["OW", "HW", "HW"] * n,
                                        [(3 * m, 3 * m + 1) for m in range(n)] + [(3 * m, 3 * m + 2) for m in range(n)])


def _graph(pos, cell=CELL, cutoff=6.3):
    s = {"pos": np.asarray(pos, float), "atomic_numbers": Z[:len(pos)]}
    if cell is not None:
        s["cell"] = np.asarray(cell, float)
    return structure_to_graph(s, cutoff=cutoff)


def _box(cls, **kw):
    lib = kw.pop("library", "amoeba-water-2003" if cls is Amoeba else "hippo-water-2021")
    kw.setdefault("cutoff", 6.3)             # 0.3 A beyond the vdW cutoff for the reduced hydrogen sites
    for key in ("ewald_cutoff", "vdw_cutoff", "repulsion_cutoff", "chgtrn_cutoff", "dispersion_cutoff"):
        kw.setdefault(key, 6.0)
    kw.setdefault("taper", 5.4)
    kw.setdefault("pme_grid", 48)
    kw.setdefault("dpme_grid", 48)
    kw.setdefault("polar_eps", 1e-10)
    return cls(lib, _topology(), keep_intermediates=True, **kw)


def test_tinker_defaults_for_the_coefficient_and_the_mesh():
    alpha = ew.ewald_coefficient(6.0)
    assert math.erfc(alpha * 6.0) / 6.0 == pytest.approx(1e-8, rel=1e-6)
    assert ew.pme_grid_size([18.643] * 3) == [24, 24, 24]
    assert ew.pme_grid_size([12.4294] * 3) == [16, 16, 16]
    assert ew.pme_grid_size([15.24] * 3) == [20, 20, 20]
    assert ew.pme_grid_size([30.48] * 3, density=0.8) == [30, 30, 30]


def test_switching_is_a_c2_step():
    r = torch.linspace(4.0, 7.0, 301, requires_grad=True)
    s = ew.switching(r, 5.4, 6.0)
    ds = torch.autograd.grad(s.sum(), r, create_graph=True)[0]
    d2s = torch.autograd.grad(ds.sum(), r)[0]
    assert torch.all(s[r.detach() <= 5.4] == 1.0) and torch.all(s[r.detach() >= 6.0] == 0.0)
    assert torch.all(ds <= 1e-12)
    def at(x):
        return int(torch.argmin((r.detach() - x).abs()))
    assert abs(float(ds[at(5.4)])) < 1e-9 and abs(float(ds[at(6.0)])) < 1e-9
    assert abs(float(d2s[at(5.4)])) < 1e-6 and abs(float(d2s[at(6.0)])) < 1e-6
    assert float(s[at(5.7)]) == pytest.approx(0.5, abs=1e-12)


def test_ewald_ladder_reduces_to_coulomb_and_differentiates():
    from xnn.ffnn.models.amoeba import coulomb_ladder
    R = torch.tensor([1.5, 3.0, 5.5], requires_grad=True)
    zero = torch.zeros(3)
    for a, b in zip(ew.ewald_ladder(R, zero, 4), coulomb_ladder(R, 4)):
        assert torch.allclose(a, b, rtol=1e-13)
    alpha = torch.full((3,), 0.6)
    D = ew.ewald_ladder(R, alpha, 3)
    for n in range(3):
        dD = torch.autograd.grad(D[n].sum(), R, retain_graph=True)[0]
        assert torch.allclose(dD / R.detach(), D[n + 1], rtol=1e-10)
    assert float(D[0][2]) < 1e-5


def _random_multipoles(n, seed=3):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, generator=g)
    q = q - q.mean()
    d = 0.3 * torch.randn(n, 3, generator=g)
    Q = 0.1 * torch.randn(n, 3, 3, generator=g)
    Q = 0.5 * (Q + Q.transpose(1, 2))
    Q = Q - torch.diag_embed(torch.diagonal(Q, dim1=1, dim2=2).mean(1)[:, None].expand(n, 3))
    return q, d, Q


def _explicit_reciprocal(q, d, Q, pos, cell, alpha, bound=8):
    V = float(cell_volume(cell))
    recip = 2 * math.pi * torch.linalg.inv(cell).t()
    m = torch.cartesian_prod(*[torch.arange(-bound, bound + 1, dtype=torch.float64)] * 3)
    m = m[(m != 0).any(1)]
    k = m @ recip
    k2 = (k * k).sum(1)
    w = 4 * math.pi / k2 * torch.exp(-k2 / (4 * alpha * alpha))
    phase = pos @ k.t()
    rho = ((q[:, None] - 1j * (d @ k.t()) - torch.einsum("ka,nab,kb->nk", k, Q, k)) * torch.exp(-1j * phase)).sum(0)
    energy = (w * (rho.real ** 2 + rho.imag ** 2)).sum() / (2 * V)
    eik = torch.exp(1j * phase)
    phi = ((w * rho)[None] * eik).sum(1).real / V
    dphi = torch.einsum("nk,ka->na", (w * rho)[None] * eik * 1j, k.to(torch.complex128)).real / V
    kc = k.to(torch.complex128)
    d2phi = -torch.einsum("nk,ka,kb->nab", (w * rho)[None] * eik, kc, kc).real / V
    return energy, phi, dphi, d2phi


def test_multipole_pme_matches_the_explicit_reciprocal_sum():
    cell = torch.tensor([[7.0, 0.0, 0.0], [0.5, 6.5, 0.0], [0.3, -0.4, 7.5]])
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(10, 3, generator=g) @ cell
    q, d, Q = _random_multipoles(10)
    alpha = 0.7
    e_ref, phi_ref, dphi_ref, d2phi_ref = _explicit_reciprocal(q, d, Q, pos, cell, alpha)
    phi, dphi, d2phi = ew.multipole_pme(q, d, Q, pos, cell, alpha, [32, 32, 32], 8)
    e = ew.multipole_reciprocal_energy(q, d, Q, pos, cell, alpha, [32, 32, 32], 8).sum()
    assert float(e) == pytest.approx(float(e_ref), rel=1e-7)
    assert torch.allclose(phi, phi_ref, atol=1e-7)
    assert torch.allclose(dphi, dphi_ref, atol=1e-7)
    assert torch.allclose(d2phi, d2phi_ref, atol=1e-6)


def test_dispersion_ewald_matches_the_lattice_sum():
    L6 = 6.0
    cell = torch.eye(3) * L6
    g = torch.Generator().manual_seed(2)
    pos = torch.rand(8, 3, generator=g) * L6
    c = 1.0 + torch.rand(8, generator=g)
    alpha, rc = 1.2, 5.0
    nmax = 8
    shifts = torch.cartesian_prod(*[torch.arange(-nmax, nmax + 1, dtype=torch.float64)] * 3) * L6
    rij = pos[None, :, None, :] - pos[None, None, :, :] + shifts[:, None, None, :]
    dist2 = (rij * rij).sum(-1)
    rmax = nmax * L6
    mask = (dist2 > 1e-8) & (dist2 < rmax * rmax)
    inv_r6 = torch.where(mask, 1.0 / dist2.clamp_min(1e-8) ** 3, torch.zeros_like(dist2))
    direct = -0.5 * (c[:, None] * c[None, :] * inv_r6).sum()
    # the continuum beyond the sphere of radius rmax
    direct = direct - 0.5 * float(c.sum()) ** 2 / L6 ** 3 * 4 * math.pi / (3 * rmax ** 3)
    near = mask & (dist2 < rc * rc)
    R = torch.sqrt(dist2[near])
    cc = (c[:, None] * c[None, :]).expand_as(dist2)[near]
    real = -0.5 * (cc / R ** 6 * ew.dispersion_real_kernel(R, torch.full_like(R, alpha))).sum()
    recip = ew.dispersion_pme(c, pos, cell, alpha, [48, 48, 48], 6).sum()
    assert float(real + recip) == pytest.approx(float(direct), rel=2e-7)


@pytest.mark.parametrize("kind", ["block", "neumann", "block-neumann"])
def test_preconditioners_reproduce_the_diagonal_solve(kind):
    """Every preconditioner gives the diagonal one's energies and forces; the block one in fewer iterations."""
    base = ForceStressOutput(_box(Amoeba, pme_grid=24, polar_eps=1e-10)).eval()
    other = ForceStressOutput(_box(Amoeba, pme_grid=24, polar_eps=1e-10, preconditioner=kind)).eval()
    g = _graph(BOX)
    a, b = base(g), other(g)
    assert float(b["energy"][0]) == pytest.approx(float(a["energy"][0]), abs=1e-9)
    assert torch.allclose(b["forces"], a["forces"], atol=1e-8)
    if kind == "block":
        assert other.model.solver_stats["forward"] < base.model.solver_stats["forward"]


def test_predictor_and_adjoint_warm_start_save_iterations_and_keep_the_results():
    """Along a short trajectory ASPC and the adjoint memory cut iterations; energies and forces stay."""
    cold = ForceStressOutput(_box(Amoeba, pme_grid=24)).eval()
    hot = ForceStressOutput(_box(Amoeba, pme_grid=24, predictor="aspc")).eval()
    velocity = np.random.default_rng(5).normal(scale=0.004, size=BOX.shape)      # Angstrom per step
    first_adjoint = None
    for step in range(7):
        g = _graph(BOX + step * velocity)
        a, b = cold(g), hot(g)
        assert float(b["energy"][0]) == pytest.approx(float(a["energy"][0]), abs=1e-8)
        assert torch.allclose(b["forces"], a["forces"], atol=1e-7)
        if first_adjoint is None:
            first_adjoint = cold.model.solver_stats["adjoint"]
    assert hot.model.solver_stats["forward_guess"]
    assert hot.model.solver_stats["forward"] <= 0.75 * cold.model.solver_stats["forward"]
    assert cold.model.solver_stats["adjoint_guess"] and cold.model.solver_stats["adjoint"] < first_adjoint


def test_a_batch_of_different_cells_matches_the_structures_one_by_one():
    """Strained copies of a box in one batch (one mesh plan, one batched FFT) equal separate evaluations."""
    model = ForceStressOutput(_box(Hippo, pme_grid=24, dpme_grid=24), compute_stress=True)
    strains = [np.eye(3), np.diag([1.02, 0.99, 1.0]), np.array([[1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.98]])]
    graphs = [_graph(BOX @ s, cell=CELL @ s) for s in strains]
    together = model(collate(graphs))
    for b, g in enumerate(graphs):
        alone = model(g)
        assert float(together["energy"][b]) == pytest.approx(float(alone["energy"][0]), abs=1e-9)
        rows = slice(b * len(BOX), (b + 1) * len(BOX))
        assert torch.allclose(together["forces"][rows], alone["forces"], atol=1e-8)
        assert torch.allclose(together["stress"][b], alone["stress"][0], atol=1e-9)


def test_mesh_self_moments_match_a_brute_force_position_average():
    """The spurious mesh self-interaction of one multipole equals the average over its position."""
    cell = torch.tensor([[7.0, 0.0, 0.0], [0.5, 6.0, 0.0], [-0.3, 0.4, 6.5]])
    alpha, mesh, order = 0.9, [12, 12, 10], 5
    s0, s2, s4 = ew.mesh_self_moments(cell, alpha, mesh, order, "coulomb")
    # the exact reciprocal self term of one unit charge over the mesh vectors: the bare kernel, no spline moduli
    g = (ew._influence(cell, alpha, mesh, order, "coulomb") / (mesh[0] * mesh[1] * mesh[2])
         / ew.spline_moduli(mesh, order, torch.float64, cell.device, least_squares=True))
    exact_self = 0.5 * float(g.sum())
    n = 12
    frac = torch.stack(torch.meshgrid(*[(torch.arange(n) + 0.5) / n] * 3, indexing="ij"), -1).reshape(-1, 3)
    pos = (frac / torch.tensor(mesh, dtype=torch.float64)) @ cell    # positions spanning one mesh cell
    one, zero3, zero33 = torch.ones(1), torch.zeros(1, 3), torch.zeros(1, 3, 3)
    d = torch.tensor([[0.3, -0.2, 0.5]])
    Q = torch.tensor([[[0.2, 0.1, 0.0], [0.1, -0.3, 0.2], [0.0, 0.2, 0.1]]])
    e_q = e_d = e_Q = 0.0
    for r in pos:
        r = r[None]
        e_q += float(ew.multipole_reciprocal_energy(one, zero3, zero33, r, cell, alpha, mesh, order))
        e_d += float(ew.multipole_reciprocal_energy(torch.zeros(1), d, zero33, r, cell, alpha, mesh, order))
        e_Q += float(ew.multipole_reciprocal_energy(one, zero3, Q, r, cell, alpha, mesh, order))
    e_q, e_d, e_Q = e_q / len(pos), e_d / len(pos), e_Q / len(pos)
    assert float(s0) == pytest.approx(e_q - exact_self, abs=1e-7)
    assert abs(float(s0)) > 1e-6                                     # a coarse mesh: the spurious part is visible
    exact_d = 0.5 * float((g * (ew.mesh_wavevectors(cell, mesh)[0] @ d[0]) ** 2).sum())
    assert float(torch.einsum("a,ab,b->", d[0], s2, d[0])) == pytest.approx(e_d - exact_d, abs=1e-7)
    k = ew.mesh_wavevectors(cell, mesh)[0]
    kQk = torch.einsum("...a,ab,...b->...", k, Q[0], k)
    exact_Q = 0.5 * float((g * (1.0 - kQk) ** 2).sum())
    spurious_Q = float(s0 - 2.0 * torch.einsum("ab,ab->", Q[0], s2) + torch.einsum("ab,abcd,cd->", Q[0], s4, Q[0]))
    assert spurious_Q == pytest.approx(e_Q - exact_Q, abs=1e-6)
    r0, _, _ = ew.mesh_self_moments(cell, alpha, mesh, 4, "r6")
    g6 = (ew._influence(cell, alpha, mesh, 4, "r6") / (mesh[0] * mesh[1] * mesh[2])
          / ew.spline_moduli(mesh, 4, torch.float64, cell.device, least_squares=True))
    e_6 = sum(float(-ew.dispersion_pme(one, r[None], cell, alpha, mesh, 4) + alpha ** 6 / 12.0) for r in pos) / len(pos)
    assert float(r0) == pytest.approx(e_6 - 0.5 * float(g6.sum()), abs=1e-7)
    fine = ew.mesh_self_moments(torch.eye(3) * 6.0, 1.2, [64, 64, 64], 5, "coulomb")
    # a fine mesh (alpha h = 0.11): the spurious moments are below 1e-6 of the analytic self coefficients
    assert abs(float(fine[0])) < 1e-6 and float(fine[1].abs().max()) < 1e-6 and float(fine[2].abs().max()) < 1e-4


def test_mesh_self_correction_subtracts_the_spurious_self_energy():
    """With mesh_self the dispersion energy drops by the spurious self moment times sum c^2."""
    model = _box(Hippo, polarization="none")
    model(_graph(BOX))
    c = model.ff.params["disp_c6"][model.top_class].detach()
    s0 = ew.mesh_self_moments(torch.tensor(CELL), ew.ewald_coefficient(6.0), [16] * 3, 4, "r6")[0]
    full = _box(Hippo, polarization="none", dpme_grid=16, mesh_self=True)
    plain_model = _box(Hippo, polarization="none", dpme_grid=16)
    e_full, e_plain = float(full(_graph(BOX))["e_dispersion"]), float(plain_model(_graph(BOX))["e_dispersion"])
    assert e_full - e_plain == pytest.approx(float(s0 * (c * c).sum()), rel=1e-9)
    assert full.intermediates["ewald"]["mesh_self"] is True


def _explicit_reciprocal_energy(q, d, Q, pos, cell, alpha, bound):
    V = float(torch.linalg.det(cell))
    recip = 2 * math.pi * torch.linalg.inv(cell).t()
    m = torch.cartesian_prod(*[torch.arange(-bound, bound + 1, dtype=torch.float64)] * 3)
    m = m[(m != 0).any(1)]
    k = m @ recip
    k2 = (k * k).sum(1)
    w = 4 * math.pi / k2 * torch.exp(-k2 / (4 * alpha * alpha))
    phase = pos @ k.t()
    rho = ((q[:, None] - 1j * (d @ k.t()) - torch.einsum("ka,nab,kb->nk", k, Q, k)) * torch.exp(-1j * phase)).sum(0)
    return float((w * (rho.real ** 2 + rho.imag ** 2)).sum() / (2 * V))


def test_amoeba_box_matches_tinker():
    model = ForceStressOutput(_box(Amoeba), compute_stress=True)
    out = model(_graph(BOX))
    for k, ref in TINKER_AMOEBA.items():
        assert float(out[k]) * EV_TO_KCAL == pytest.approx(ref, abs=TOL_TERM), k
    forces = out["forces"].detach().numpy() * EV_TO_KCAL
    assert np.abs(forces[:4] - TINKER_AMOEBA_FORCES).max() < TOL_FORCE
    virial = -out["stress"][0].detach().numpy() * L ** 3 * EV_TO_KCAL
    assert np.abs(virial + TINKER_AMOEBA_VIRIAL).max() < TOL_VIRIAL
    info = model.model.intermediates["ewald"]
    assert info["grids"][0][1] == [48, 48, 48] and info["alpha"] == pytest.approx(ew.ewald_coefficient(6.0))


def test_hippo_box_matches_tinker():
    model = ForceStressOutput(_box(Hippo), compute_stress=True)
    out = model(_graph(BOX))
    for k, ref in TINKER_HIPPO.items():
        assert float(out[k]) * EV_TO_KCAL == pytest.approx(ref, abs=TOL_TERM), k
    forces = out["forces"].detach().numpy() * EV_TO_KCAL
    assert np.abs(forces[:4] - TINKER_HIPPO_FORCES).max() < TOL_FORCE
    virial = -out["stress"][0].detach().numpy() * L ** 3 * EV_TO_KCAL
    assert np.abs(virial + TINKER_HIPPO_VIRIAL).max() < TOL_VIRIAL


def _exchpol_library():
    lib = copy.deepcopy(builtin_amoeba("hippo-water-2021"))
    lib.exchpol = copy.deepcopy(EXCHPOL)
    return lib


def test_hippo_box_with_exchange_polarization_matches_tinker():
    """Exchange polarization in a cell, cut by the switch of the repulsion, matches Tinker's."""
    out = ForceStressOutput(_box(Hippo, library=_exchpol_library()))(_graph(BOX))
    for k, ref in TINKER_HIPPO_EXCHPOL.items():
        assert float(out[k]) * EV_TO_KCAL == pytest.approx(ref, abs=TOL_TERM), k
    assert np.abs(out["forces"].detach().numpy()[:4] * EV_TO_KCAL - TINKER_HIPPO_EXCHPOL_FORCES).max() < TOL_FORCE


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_boxes_match_tinker_on_its_default_coarse_meshes(cls):
    """At Tinker's default 16^3 meshes the dispersion sum matches Tinker's, the multipole sums nearly."""
    refs, forces, virial_ref = ((TINKER_AMOEBA_16, TINKER_AMOEBA_16_FORCES, TINKER_AMOEBA_16_VIRIAL) if cls is Amoeba
                                else (TINKER_HIPPO_16, TINKER_HIPPO_16_FORCES, TINKER_HIPPO_16_VIRIAL))
    model = ForceStressOutput(_box(cls, pme_grid=None, dpme_grid=None), compute_stress=True)
    out = model(_graph(BOX))
    for k, ref in refs.items():
        tol = TOL_TERM if k == "e_dispersion" else TOL_COARSE_TERM
        assert float(out[k]) * EV_TO_KCAL == pytest.approx(ref, abs=tol), k
    assert np.abs(out["forces"].detach().numpy()[:4] * EV_TO_KCAL - forces).max() < TOL_COARSE_FORCE
    virial = -out["stress"][0].detach().numpy() * L ** 3 * EV_TO_KCAL
    assert np.abs(virial + virial_ref).max() < TOL_COARSE_VIRIAL
    assert model.model.intermediates["ewald"]["grids"][0][1] == [16, 16, 16]


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_box_forces_and_stress_are_derivatives(cls):
    model = _box(cls)
    fso = ForceStressOutput(model, compute_stress=True)
    out = fso(_graph(BOX))
    f = out["forces"].detach().numpy()
    h = 1e-4
    for atom, axis in ((0, 0), (37, 2)):
        p = BOX.copy()
        p[atom, axis] += h
        ep = float(model(_graph(p))["energy"])
        p[atom, axis] -= 2 * h
        em = float(model(_graph(p))["energy"])
        assert f[atom, axis] == pytest.approx(-(ep - em) / (2 * h), abs=2e-6)
    # a symmetric shear strain of e on xy and yx moves the energy by 2 e V sigma_xy; the step is
    # large enough for the pairs crossing the real-space cutoff (1e-8 of a pair energy) not to matter
    e = 1e-4
    energies = []
    for sign in (1, -1):
        eps = np.eye(3)
        eps[0, 1] += sign * e
        eps[1, 0] += sign * e
        energies.append(float(model(_graph(BOX @ eps.T, cell=CELL @ eps.T))["energy"]))
    assert float(out["stress"][0, 0, 1]) == pytest.approx((energies[0] - energies[1]) / (4 * e) / L ** 3, abs=2e-6)


def test_iterative_solve_matches_the_dense_one():
    pos = BOX[:24] - BOX[:24].mean(0)
    dense = ForceStressOutput(Amoeba("amoeba-water-2003", _topology(8), cutoff=30.0, solver="dense"))
    iterative = ForceStressOutput(Amoeba("amoeba-water-2003", _topology(8), cutoff=30.0,
                                         solver="iterative", polar_eps=1e-12))
    a = dense(_graph(pos, cell=None, cutoff=30.0))
    b = iterative(_graph(pos, cell=None, cutoff=30.0))
    assert float(a["energy"]) == pytest.approx(float(b["energy"]), abs=1e-10)
    assert torch.allclose(a["forces"], b["forces"], atol=1e-8)
    assert torch.allclose(a["induced_dipoles"], b["induced_dipoles"], atol=1e-10)
    with pytest.raises(ValueError):
        Amoeba("amoeba-water-2003", _topology(), cutoff=6.0, solver="dense")(_graph(BOX))


def test_ewald_total_is_independent_of_the_coefficients():
    base = float(_box(Amoeba, pme_grid=48)(_graph(BOX))["energy"]) * EV_TO_KCAL
    other = float(_box(Amoeba, ewald_alpha=0.58, pewald_alpha=0.72, pme_grid=48)(_graph(BOX))["energy"]) * EV_TO_KCAL
    assert other == pytest.approx(base, abs=5e-3)
    truncated = float(_box(Amoeba, ewald=False)(_graph(BOX))["energy"]) * EV_TO_KCAL
    assert abs(truncated - base) > 1.0


def test_batches_of_cells_and_a_cell_with_a_molecule():
    model = _box(Amoeba)
    single = float(model(_graph(BOX))["energy"])
    pair = model(collate([_graph(BOX), _graph(BOX + L / 16)]))["energy"]      # a mesh spacing: the same spline weights
    assert float(pair[0]) == pytest.approx(single, abs=1e-9) and float(pair[1]) == pytest.approx(single, abs=1e-9)
    dimer = Amoeba("amoeba-water-2003", _topology(2), cutoff=6.3, ewald_cutoff=6.0, vdw_cutoff=6.0)
    pos = BOX[:6] - BOX[:6].mean(0)
    vacuum = float(dimer(_graph(pos, cell=None))["energy"])
    both = dimer(collate([_graph(pos, cell=np.diag([40.0, 40.0, 40.0])), _graph(pos, cell=None)]))["energy"]
    assert float(both[1]) == pytest.approx(vacuum, abs=1e-10)
    # periodic images and the mesh error of a dilute cell
    assert float(both[0]) == pytest.approx(vacuum, abs=5e-3)


def test_training_step_through_the_implicit_solve():
    model = _box(Amoeba, trainable=("vdw", "polarizability"), polar_eps=1e-9)
    fso = ForceStressOutput(model)
    fso.train()
    out = fso(_graph(BOX))
    target = torch.tensor(TINKER_AMOEBA_FORCES * KCAL_TO_EV)
    loss = ((out["energy"][0] - TINKER_AMOEBA["energy"] * KCAL_TO_EV + 0.01) ** 2
            + ((out["forces"][:4] - target) ** 2).mean())
    loss.backward()
    grads = {k: p.grad for k, p in model.ff.params.items() if p.requires_grad}
    present = {k: g for k, g in grads.items() if g is not None}
    assert present and all(torch.isfinite(g).all() for g in present.values())
    assert float(present["polarizability"].abs().max()) > 0 and float(present["vdw_eps"].abs().max()) > 0


def test_config_keys_in_tinker_spelling():
    from xnn.common.config.schema import ModelConfig
    from xnn.common.config.translate import translate_model_keys
    raw = {"name": "amoeba", "cutoff": 6.0, "prm": "amoeba-water-2003", "types": ["OW", "HW", "HW"],
           "bonds": [[0, 1], [0, 2]], "ewald-alpha": 0.6, "pme-grid": [16, 16, 16], "vdw-taper": 5.4,
           "polar-eps": 1e-9}
    t = translate_model_keys(raw)
    assert t["ewald_alpha"] == 0.6 and t["pme_grid"] == [16, 16, 16]
    assert t["vdw_taper"] == 5.4 and t["polar_eps"] == 1e-9
    cfg = ModelConfig(name="amoeba", cutoff=6.0, extra={k: v for k, v in t.items() if k not in ("name", "cutoff")})
    model = Amoeba.from_config(cfg)
    assert model.ewald_alpha == 0.6 and model.pme_grid == [16, 16, 16]
    assert model.vdw_taper == 5.4 and model.polar_eps == 1e-9


def _slab_graph(gap, cutoff=6.3):
    cell = np.diag([L, L, L + gap])
    return structure_to_graph({"pos": BOX, "atomic_numbers": Z, "cell": cell, "pbc": [True, True, False]},
                              cutoff=cutoff)


def test_slab_energies_are_independent_of_the_vacuum_gap():
    # the box as a slab: periodic in x and y, a vacuum gap along z; the mesh keeps 2.6 points per Angstrom
    outs = [_box(Amoeba, pme_grid=[32, 32, nz])(_slab_graph(gap)) for gap, nz in ((2 * L, 96), (3 * L, 128))]
    for k in ("e_multipole", "e_polar", "energy"):
        assert float(outs[0][k]) * EV_TO_KCAL == pytest.approx(float(outs[1][k]) * EV_TO_KCAL, abs=1e-2), k
    assert abs(float(outs[0]["e_multipole"]) - float(_box(Amoeba)(_graph(BOX))["e_multipole"])) * EV_TO_KCAL > 0.1
    model = _box(Amoeba, pme_grid=[32, 32, 96])
    fso = ForceStressOutput(model)
    f = fso(_slab_graph(2 * L))["forces"].detach().numpy()
    h = 1e-4
    for atom, axis in ((0, 2), (77, 0)):
        p = BOX.copy()
        p[atom, axis] += h
        cell = np.diag([L, L, 3 * L])
        ep = float(model(structure_to_graph({"pos": p, "atomic_numbers": Z, "cell": cell,
                                             "pbc": [True, True, False]}, cutoff=6.3))["energy"])
        p[atom, axis] -= 2 * h
        em = float(model(structure_to_graph({"pos": p, "atomic_numbers": Z, "cell": cell,
                                             "pbc": [True, True, False]}, cutoff=6.3))["energy"])
        assert f[atom, axis] == pytest.approx(-(ep - em) / (2 * h), abs=2e-6)
    with pytest.raises(ValueError):
        _box(Amoeba)(structure_to_graph({"pos": BOX, "atomic_numbers": Z, "cell": CELL,
                                         "pbc": [True, False, False]}, cutoff=6.3))


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_second_derivatives_through_the_iterative_solve(cls):
    model = _box(cls, polar_eps=1e-11)
    g = _graph(BOX)
    g.pos.requires_grad_(True)
    energy = model(g)["energy"].sum()
    forces = -torch.autograd.grad(energy, g.pos, create_graph=True)[0]
    atom, axis = 5, 1
    row = torch.autograd.grad(forces[atom, axis], g.pos)[0]        # d F_(atom, axis) / d positions
    fso = ForceStressOutput(model)
    h = 1e-3
    for other, ax2 in ((5, 1), (6, 0), (40, 2)):
        p = BOX.copy()
        p[other, ax2] += h
        fp = float(fso(_graph(p))["forces"][atom, axis])
        p[other, ax2] -= 2 * h
        fm = float(fso(_graph(p))["forces"][atom, axis])
        assert float(row[other, ax2]) == pytest.approx((fp - fm) / (2 * h), rel=1e-4, abs=1e-3)


def test_use_fast_validates_and_keeps_the_reference_on_the_cpu():
    """use_fast is validated, reached by set_use_fast, and inactive on the CPU."""
    from xnn.common.models.fast import set_use_fast
    with pytest.raises(ValueError):
        _box(Amoeba, use_fast="always")
    ref, fast = _box(Amoeba), _box(Amoeba)
    set_use_fast(ForceStressOutput(fast), True)
    assert fast.use_fast is True and ref.use_fast is False
    fast.eval()
    assert not fast._fused_active(torch.ones(3, dtype=torch.float64))
    e_ref = float(ref.eval()(_graph(BOX))["energy"])
    assert float(fast(_graph(BOX))["energy"]) == e_ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused path runs on a GPU")
@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_fused_pair_arithmetic_matches_the_reference_on_the_gpu(cls):
    """Energies, forces and stress agree with the reference over eager, compiling and compiled calls."""
    g = _graph(BOX).to("cuda")
    ref = ForceStressOutput(_box(cls), compute_stress=True).to("cuda").eval()
    fast = ForceStressOutput(_box(cls, use_fast=True), compute_stress=True).to("cuda").eval()
    want = ref(g)
    for _ in range(3):
        got = fast(g)
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key], rtol=1e-10, atol=1e-10), key


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
@pytest.mark.parametrize("periodic", [True, False])
def test_forward_on_a_plan_matches_the_plain_forward(cls, periodic):
    """A recorded plan reproduces the plain evaluation, also after the atoms move."""
    model = _box(cls).eval()
    cell = CELL if periodic else None
    g = _graph(BOX, cell=cell, cutoff=7.3)
    padded = model.plan_step(g, skin=1.0, capacity=int(g.edge_index.shape[1]))
    rng = np.random.default_rng(3)
    cases = [(model.plan_step(_graph(BOX, cell=cell)), 0.0), (padded, 0.0), (padded, 0.02), (padded, 0.02)]
    for plan, scale in cases:
        moved = BOX + rng.normal(scale=scale, size=BOX.shape)
        results = []
        for p in (None, plan):
            g = _graph(moved if p is None else BOX, cell=cell)
            g.pos = torch.tensor(moved, dtype=g.pos.dtype).requires_grad_(True)
            out = model(g, p)
            forces = -torch.autograd.grad(out["energy"].sum(), g.pos)[0]
            results.append((out["energy"].detach(), forces, out["induced_dipoles"].detach()))
        assert not bool(plan.outdated(g.pos))
        for a, b in zip(*results):
            assert torch.allclose(a, b, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_a_solve_of_fixed_length_matches_the_tested_solve(cls):
    """Solves of fixed length on a plan give the tested solve's results and report their convergence."""
    model = _box(cls).eval()
    g = _graph(BOX)
    plan = model.plan_step(g)
    results = []
    for fixed in (None, {"forward": 100, "adjoint": 100}):
        model.set_dipole_predictor(None)
        model._adjoints.clear()
        plan.solve_iterations = fixed
        g.pos = g.pos.detach().requires_grad_(True)
        out = model(g, plan)
        forces = -torch.autograd.grad(out["energy"].sum(), g.pos)[0]
        results.append((out["energy"].detach(), forces, out["induced_dipoles"].detach()))
    assert bool(plan.converged)
    for a, b in zip(*results):
        assert torch.equal(a, b)
    plan.solve_iterations = {"forward": 1, "adjoint": 1}
    with torch.no_grad():
        model(g, plan)
    assert not bool(plan.converged)


def _walk(cls, device, steps, **kw):
    """Plain steps and steps on a plan along a random walk of the box."""
    ref, model = (ForceStressOutput(_box(cls, **kw).to(device), compute_stress=True).eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.3, warmup=3)
    rng = np.random.default_rng(5)
    pos = BOX.copy()
    for _ in range(steps):
        pos = pos + rng.normal(scale=0.03, size=BOX.shape)
        s = {"pos": torch.tensor(pos, device=device), "atomic_numbers": torch.tensor(Z, device=device),
             "cell": torch.tensor(CELL, device=device), "pbc": torch.ones(3, dtype=torch.bool, device=device)}
        got = stepper(s)
        want = ref(structure_to_graph(s, 6.3, device=device))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-8), key
    assert stepper.rebuilds >= 1 and stepper.full_rebuilds == 0 and stepper.plan.solve_iterations
    return stepper


@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_steps_on_a_plan_follow_the_plain_steps(cls):
    """Steps on a plan follow the plain steps through rebuilds of the pair list and fixed solves."""
    _walk(cls, "cpu", 6)


def test_steps_on_a_plan_follow_the_plain_steps_with_exchange_polarization():
    """Steps on a plan follow the plain steps with exchange polarization."""
    _walk(Hippo, "cpu", 4, library=_exchpol_library())


def test_steps_on_a_plan_follow_an_atom_wrapped_into_the_cell():
    """An atom wrapped into the cell makes the steps on a plan record their bonded image shifts again."""
    ref, model = (ForceStressOutput(_box(Amoeba), compute_stress=True).eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
    for wrap in (0.0, 0.0, L):
        pos = BOX.copy()
        pos[0, 0] += wrap
        s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(Z), "cell": torch.tensor(CELL),
             "pbc": torch.ones(3, dtype=torch.bool)}
        got, want = stepper(s), ref(structure_to_graph(s, 6.3))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-8), key
    assert stepper.full_rebuilds == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cls", [Amoeba, Hippo])
def test_steps_replayed_as_a_cuda_graph_follow_the_plain_steps(cls):
    """Steps replayed as a CUDA graph follow the plain steps on the GPU."""
    stepper = _walk(cls, "cuda", 10, use_fast=True)
    assert stepper.recordings >= 1 and stepper.replays >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_steps_with_exchange_polarization_replayed_as_a_cuda_graph_follow_the_plain_steps():
    """Steps with exchange polarization replayed as a CUDA graph follow the plain steps on the GPU."""
    stepper = _walk(Hippo, "cuda", 10, use_fast=True, library=_exchpol_library())
    assert stepper.recordings >= 1 and stepper.replays >= 1


def test_the_engine_evaluates_steps_on_a_plan():
    """The MDI engine on step plans gives the plain engine's results."""
    engines = [MDIEngine(ForceStressOutput(_box(Amoeba), compute_stress=True), 6.3, step_graph=flag, skin=0.5)
               for flag in (False, True)]
    for engine in engines:
        engine.natoms, engine.atomic_numbers = len(BOX), Z.astype(np.int64)
        engine.cell_bohr = CELL / BOHR_TO_ANGSTROM
    for shift in (0.0, 0.01):
        for engine in engines:
            engine.coords_bohr = (BOX + shift) / BOHR_TO_ANGSTROM
            engine.calculate()
        plain, planned = engines
        assert planned.energy == pytest.approx(plain.energy, rel=1e-10)
        np.testing.assert_allclose(planned.forces, plain.forces, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(planned.stress, plain.stress, rtol=1e-8, atol=1e-12)
    assert engines[1].steps is not None


def test_the_engine_replays_steps_by_default_only_on_a_gpu():
    """The engine's step graph is on by default only on a GPU, and its option is validated."""
    model = ForceStressOutput(_box(Amoeba), compute_stress=True)
    assert MDIEngine(model, 6.3).steps is None
    with pytest.raises(ValueError, match="step_graph"):
        MDIEngine(model, 6.3, step_graph="yes")
    if torch.cuda.is_available():
        assert MDIEngine(model, 6.3, device="cuda").steps is not None
