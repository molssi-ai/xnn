"""Tests for the faithful ANI model (registered as ``ani``) and its AEV.

Covers the symmetry-function / AEV building blocks, rotation/translation/
permutation invariance of energies and equivariance of forces, the ANI-1,
ANI-1x, and ANI-1ccx presets (AEV lengths and per-element architectures), self
atomic energies, config building with upstream key translation, and -- when
``torchani`` is available -- element-for-element AEV parity against
``torchani.AEVComputer`` and full energy/force parity via weight transplant of
both the pretrained ANI-1x and ANI-1ccx models.
"""
import math

import numpy as np
import pytest
import torch

from xnns.common.config import from_dict
from xnns.common.data import structure_to_graph
from xnns.common.data.neighborlist import build_neighbor_list
from xnns.common.data import AtomicGraph
from xnns.common.models import ForceStressOutput, available_models, build_model
from xnns.dnn.featurizers import AEV
from xnns.dnn.featurizers.aev import _angle_shifts, _even_shifts
from xnns.dnn.featurizers.symmetry_functions import build_triplets
from xnns.dnn.models.ani import (
    ANI, ANI1CCX_SELF_ENERGIES, ANI1X_HIDDEN, ANI1X_SELF_ENERGIES)

# H, C, N, O
SPECIES = [1, 6, 7, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _graph(n=8, cutoff=5.2, R=None, shift=0.0, seed=1, perm=None):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 4, (n, 3)) + shift
    if R is not None:
        pos = pos @ R.T
    z = np.array(([8, 1, 1, 6, 7, 1, 6, 1] * n)[:n], dtype=np.int64)
    pos = pos.astype(np.float64)
    if perm is not None:
        pos, z = pos[perm], z[perm]
    return structure_to_graph({"pos": pos, "atomic_numbers": z}, cutoff)


# --------------------------------------------------------------------------- #
# building blocks                                                             #
# --------------------------------------------------------------------------- #

def test_build_triplets_matches_bruteforce():
    """The vectorised triplet builder matches a per-atom brute force."""
    rng = torch.Generator().manual_seed(0)
    for _ in range(50):
        n = int(torch.randint(1, 8, (1,), generator=rng))
        e = int(torch.randint(0, 20, (1,), generator=rng))
        if e == 0:
            continue
        ei = torch.stack([torch.randint(0, n, (e,), generator=rng),
                          torch.randint(0, n, (e,), generator=rng)])
        f, s, c = build_triplets(ei, n)

        want = set()
        for i in range(n):
            edges = (ei[1] == i).nonzero().flatten().tolist()
            for a in range(len(edges)):
                for b in range(a + 1, len(edges)):
                    want.add((i, tuple(sorted((edges[a], edges[b])))))
        got = {(int(cc), tuple(sorted((int(a), int(b)))))
               for a, b, cc in zip(f, s, c)}
        assert got == want


def test_aev_preset_dimensions():
    """ANI-1 and ANI-1x presets have the published AEV lengths (768 / 384)."""
    assert AEV.ani1(SPECIES).output_dim == 768
    assert AEV.ani1x(SPECIES).output_dim == 384
    # radial: n_shf * n_species; angular: n_grid * n_pairs
    a = AEV.ani1x(SPECIES)
    assert a.radial.output_dim == 16 * 4
    assert a.angular.output_dim == (4 * 8) * 10


def test_shift_recipe_matches_torchani_grid():
    """The evenly-spaced-shift recipe reproduces torchani's ANI-1x ShfR/ShfZ."""
    shfr = _even_shifts(5.2, 16)
    assert math.isclose(shfr[0], 0.9) and math.isclose(shfr[-1], 4.93125)
    shfz = _angle_shifts(8)
    assert math.isclose(shfz[0], math.pi / 16)


# --------------------------------------------------------------------------- #
# model behaviour                                                             #
# --------------------------------------------------------------------------- #

def test_forward_and_forces():
    """ANI produces per-structure energy and conservative forces."""
    model = ForceStressOutput(ANI.ani1x(SPECIES))
    g = _graph()
    out = model(g)
    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (g.num_nodes, 3)
    assert torch.isfinite(out["forces"]).all()


def test_invariance_and_equivariance():
    """Energy is invariant to rotation/translation/permutation; forces rotate."""
    torch.manual_seed(0)
    model = ForceStressOutput(ANI.ani1x(SPECIES))
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    e0 = model(_graph())["energy"]
    e_rot = model(_graph(R=R.numpy()))["energy"]
    e_trans = model(_graph(shift=3.0))["energy"]
    perm = np.array([3, 0, 5, 1, 7, 2, 6, 4])
    e_perm = model(_graph(perm=perm))["energy"]

    assert torch.allclose(e0, e_rot, atol=1e-8)
    assert torch.allclose(e0, e_trans, atol=1e-8)
    assert torch.allclose(e0, e_perm, atol=1e-8)

    f0 = model(_graph())["forces"]
    f_rot = model(_graph(R=R.numpy()))["forces"]
    assert torch.allclose(f0 @ R.T, f_rot, atol=1e-8)


def test_self_energies_shift_total():
    """Per-species self energies add a constant offset to the total energy."""
    torch.manual_seed(0)
    m = ANI(SPECIES)
    g = _graph()
    e_before = m(g)["energy"].item()

    vals = {1: 0.5, 6: 1.0, 7: 2.0, 8: 3.0}
    for z, v in vals.items():
        m._self_energies_by_z[z] = v
    e_after = m(g)["energy"].item()

    counts = {z: int((g.atomic_numbers == z).sum()) for z in SPECIES}
    expected = sum(counts[z] * vals[z] for z in SPECIES)
    assert math.isclose(e_after - e_before, expected, rel_tol=1e-6)


def test_ani1x_architecture():
    """ANI-1x preset builds torchani's per-element widths and self energies."""
    m = ANI.ani1x(SPECIES)
    for z in SPECIES:
        outs = [l.out_features for l in m.element_nets.nets[str(z)]
                if hasattr(l, "out_features")]
        assert outs == list(ANI1X_HIDDEN[z]) + [1]
    for z in SPECIES:
        assert math.isclose(float(m._self_energies_by_z[z]),
                            ANI1X_SELF_ENERGIES[z])


def test_ani1ccx_architecture_matches_ani1x():
    """ANI-1ccx preset shares the ANI-1x architecture; only self energies differ."""
    ccx = ANI.ani1ccx(SPECIES)
    x = ANI.ani1x(SPECIES)
    assert ccx.featurizer.output_dim == x.featurizer.output_dim == 384
    for z in SPECIES:
        widths = lambda m: [l.out_features for l in m.element_nets.nets[str(z)]
                            if hasattr(l, "out_features")]
        assert widths(ccx) == widths(x)
        assert math.isclose(float(ccx._self_energies_by_z[z]),
                            ANI1CCX_SELF_ENERGIES[z])
        assert not math.isclose(float(ccx._self_energies_by_z[z]),
                                ANI1X_SELF_ENERGIES[z])


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #

def test_registered():
    """ANI is discoverable through the model registry."""
    assert "ani" in available_models()


def test_from_config_preset_and_translation():
    """from_config honours presets, symbols, and upstream key spellings."""
    cfg = from_dict({"model": {
        "name": "ani",
        "extra": {"preset": "ani-1x", "species": ["H", "C", "N", "O"]},
    }})
    m = build_model(cfg.model)
    assert m.featurizer.output_dim == 384

    cfg_ccx = from_dict({"model": {
        "name": "ani", "extra": {"preset": "ani-1ccx"},
    }})
    m_ccx = build_model(cfg_ccx.model)
    assert m_ccx.featurizer.output_dim == 384
    assert math.isclose(float(m_ccx._self_energies_by_z[6]),
                        ANI1CCX_SELF_ENERGIES[6])

    # torchani/NeuroChem spellings Rcr/Rca translate to radial/angular cutoff
    cfg2 = from_dict({"model": {
        "name": "ani", "Rcr": 4.6, "Rca": 3.1, "atomic_numbers": [1, 6, 7, 8],
    }})
    m2 = build_model(cfg2.model)
    assert math.isclose(m2.featurizer.radial.cutoff, 4.6)
    assert math.isclose(m2.featurizer.angular.cutoff, 3.1)


def test_behler_parrinello_conventions():
    """radial_prefactor / angular_cos_factor default to torchani, override to BP."""
    ani = AEV.ani1x(SPECIES)
    assert math.isclose(ani.radial.prefactor, 0.25)
    assert math.isclose(ani.angular.cos_factor, 0.95)
    bp = AEV(SPECIES, radial_prefactor=1.0, angular_cos_factor=1.0)
    assert math.isclose(bp.radial.prefactor, 1.0)
    assert math.isclose(bp.angular.cos_factor, 1.0)


# --------------------------------------------------------------------------- #
# torchani parity (only where torchani is installed)                          #
# --------------------------------------------------------------------------- #

def _xnns_graph(Z, pos, cutoff):
    Z = torch.as_tensor(Z, dtype=torch.long)
    pos = torch.as_tensor(pos, dtype=torch.float64)
    ei, cs = build_neighbor_list(pos, cutoff)
    return AtomicGraph(pos=pos, atomic_numbers=Z, edge_index=ei, cell_shifts=cs,
                       batch=torch.zeros(len(Z), dtype=torch.long),
                       n_atoms=torch.tensor([len(Z)]))


@pytest.mark.parametrize("preset,grid", [
    ("ani1x", (5.2, 3.5, [16.0], 16, [8.0], [32.0], 4, 8)),
    ("ani1", (4.6, 3.1, [16.0], 32, [8.0], [8.0], 8, 8)),
])
def test_torchani_aev_parity(preset, grid):
    """xnns AEV matches torchani.AEVComputer element-for-element."""
    torchani = pytest.importorskip("torchani")
    Rcr, Rca, EtaR, nR, EtaA, Zeta, nA, nZ = grid
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    tani = torchani.AEVComputer(Rcr, Rca, t(EtaR), t(_even_shifts(Rcr, nR)),
                                t(EtaA), t(Zeta), t(_even_shifts(Rca, nA)),
                                t(_angle_shifts(nZ)), 4)
    aev = getattr(AEV, preset)(SPECIES)

    rng = np.random.default_rng(0)
    Z = np.array([6, 7, 8, 1, 1, 1], dtype=np.int64)
    pos = rng.uniform(0, 3, (6, 3))
    idx = {1: 0, 6: 1, 7: 2, 8: 3}
    x = aev(_xnns_graph(Z, pos, aev.cutoff))
    sp = torch.tensor([[idx[z] for z in Z]])
    _, ref = tani((sp, torch.as_tensor(pos[None])))
    assert torch.allclose(x, ref[0], atol=1e-10)


@pytest.mark.parametrize("preset,upstream", [
    ("ani1x", "ANI1x"),
    ("ani1ccx", "ANI1ccx"),
])
def test_torchani_energy_force_parity(preset, upstream):
    """Transplanting torchani's pretrained ANI-1x/ANI-1ccx weights reproduces E and F."""
    torchani = pytest.importorskip("torchani")
    model = getattr(torchani.models, upstream)(
        periodic_table_index=False).double()
    member = model.neural_networks[0]

    xa = getattr(ANI, preset)(SPECIES)
    nets = dict(member.named_children())
    zsym = {1: "H", 6: "C", 7: "N", 8: "O"}
    for z in SPECIES:
        src = [l for l in nets[zsym[z]] if isinstance(l, torch.nn.Linear)]
        dst = [l for l in xa.element_nets.nets[str(z)]
               if isinstance(l, torch.nn.Linear)]
        for s, d in zip(src, dst):
            d.weight.data = s.weight.data.clone()
            d.bias.data = s.bias.data.clone()

    rng = np.random.default_rng(1)
    Z = np.array([6, 7, 8, 1, 1, 1], dtype=np.int64)
    pos = rng.uniform(0, 3, (6, 3))
    idx = {1: 0, 6: 1, 7: 2, 8: 3}

    g = _xnns_graph(Z, pos, xa.cutoff)
    g.pos.requires_grad_(True)
    out = ForceStressOutput(xa)(g)

    coords = torch.as_tensor(pos[None]).clone().requires_grad_(True)
    sp = torch.tensor([[idx[z] for z in Z]])
    e = model.energy_shifter(member(model.aev_computer((sp, coords)))).energies
    f = -torch.autograd.grad(e.sum(), coords)[0][0]

    assert abs(out["energy"].item() - e.item()) < 1e-6
    assert torch.allclose(out["forces"].detach(), f, atol=1e-6)
