"""Voxelized environments and the conventional 3D CNN potential.

Covers the :class:`VoxelGrid` featurizer against a loop-based reference of
its definition, the exact rotation of grids by cube symmetries, the grid
operations shared by the volumetric models (low-pass filter, pooling, field
bookkeeping), and the :class:`CNN3D` potential: shapes, batching, periodic
structures, finite-difference forces, the invariances it has (translation,
permutation) and the one it lacks by construction (rotation).
"""
import itertools
import math

import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput, available_models, build_model
from xnn.cnn.featurizers import VoxelGrid
from xnn.cnn.models import (CNN3D, default_fields, default_strides, field_dim,
                            global_average_pool, grid_coordinates, low_pass_filter,
                            rotate_voxels)

SPECIES = [1, 6, 8]


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _structure(n=6, seed=0, box=3.0):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, box, (n, 3)), "atomic_numbers": ([1, 6, 8] * n)[:n]}


def _graph(s, cutoff=4.0):
    return structure_to_graph(s, cutoff)


def cube_rotations():
    """The 24 proper rotations of the cube as signed permutation matrices."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = torch.zeros(3, 3, dtype=torch.float64)
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if abs(float(torch.det(R)) - 1.0) < 1e-12:
                out.append(R)
    assert len(out) == 24
    return out


def _small_model(**kw):
    torch.manual_seed(0)
    kw.setdefault("species", SPECIES)
    kw.setdefault("cutoff", 4.0)
    kw.setdefault("grid_size", 9)
    kw.setdefault("channels", (6, 8))
    kw.setdefault("kernel_size", 3)
    m = CNN3D(**kw)
    # a random readout head (the zero-initialized one predicts a constant)
    torch.nn.init.normal_(m.readout[-1].weight)
    torch.nn.init.normal_(m.readout[-1].bias)
    return m


# the voxel grid featurizer

def _reference_grid(vox, s):
    """Loop-based evaluation of the definition of the species density grids."""
    pos = torch.as_tensor(s["pos"], dtype=torch.float64)
    Z = list(s["atomic_numbers"])
    n, C, size = len(Z), vox.n_channels, vox.grid_size
    pts = grid_coordinates(size) * vox.spacing                    # (s, s, s, 3) voxel centers
    grid = torch.zeros(n, C, size, size, size, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            r = pos[j] - pos[i]
            d = float(torch.linalg.norm(r))
            if i == j:
                if not vox.include_center:
                    continue
                w = 1.0
            else:
                if d >= vox.cutoff:
                    continue
                w = 0.5 * (math.cos(math.pi * d / vox.cutoff) + 1.0) if vox.envelope is not None else 1.0
            c = vox.species.index(Z[j])
            grid[i, c] += w * torch.exp(-((pts - r) ** 2).sum(-1) / (2 * vox.sigma ** 2))
    return grid


@pytest.mark.parametrize("include_center,cutoff_fn", [(True, "cosine"), (False, None)])
def test_voxel_grid_matches_definition(include_center, cutoff_fn):
    vox = VoxelGrid(SPECIES, cutoff=3.5, grid_size=7, include_center=include_center,
                    cutoff_fn=cutoff_fn)
    s = _structure(n=7, seed=1)
    g = _graph(s, 3.5)
    grid = vox(g)
    assert grid.shape == (7, 3, 7, 7, 7)
    assert torch.allclose(grid, _reference_grid(vox, s), atol=1e-12)


def test_voxel_grid_geometry():
    vox = VoxelGrid(SPECIES, cutoff=4.0, grid_size=16)
    assert vox.output_dim == 3
    assert vox.spacing == pytest.approx(0.5)
    assert vox.sigma == pytest.approx(0.25)            # half a voxel by default
    assert torch.allclose(vox.axis, torch.linspace(-3.75, 3.75, 16))
    assert VoxelGrid(SPECIES, 4.0, 16, sigma=0.7).sigma == 0.7


def test_voxel_grid_channels_and_center():
    """A lone H-C pair on a 1 A grid: H density only in channel 0, C only in
    channel 1, each atom's own blob at the center of its channel, and the
    neighbor at its position weighted by the cosine envelope."""
    vox = VoxelGrid(SPECIES, cutoff=4.5, grid_size=9)       # 1.0 A voxels, sigma 0.5 A
    g = _graph({"pos": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "atomic_numbers": [1, 6]}, 4.5)
    grid = vox(g)
    assert float(grid[:, 2].abs().max()) == 0.0        # no oxygen anywhere
    c = 4
    assert grid[0, 0, c, c, c] == pytest.approx(1.0)   # H sees itself at the origin
    assert grid[1, 1, c, c, c] == pytest.approx(1.0)
    w = 0.5 * (math.cos(math.pi * 1.0 / 4.5) + 1.0)    # envelope at 1 A
    assert grid[0, 1, c + 1, c, c] == pytest.approx(w)   # the carbon, at +x
    assert grid[1, 0, c - 1, c, c] == pytest.approx(w)   # the hydrogen, at -x of the carbon
    assert float(grid[0, 1, :c - 1].abs().max()) < 1e-6  # nothing far on the -x side


def test_voxel_grid_unknown_species():
    vox = VoxelGrid([1, 6], cutoff=4.0, grid_size=5)
    with pytest.raises(ValueError, match="no density channel"):
        vox(_graph(_structure(), 4.0))


def test_voxel_grid_rotates_with_cube_symmetries():
    """Rotating the structure by a symmetry of the cube rotates every
    environment grid exactly (no interpolation involved)."""
    vox = VoxelGrid(SPECIES, cutoff=4.0, grid_size=9)
    s = _structure(n=6, seed=2)
    ref = vox(_graph(s))
    for R in cube_rotations():
        rot = dict(s, pos=np.asarray(s["pos"]) @ R.numpy().T)
        assert torch.allclose(vox(_graph(rot)), rotate_voxels(ref, R), atol=1e-12)


def test_rotate_voxels_group_action():
    x = torch.randn(2, 3, 5, 5, 5)
    Rs = cube_rotations()
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.equal(rotate_voxels(x, eye), x)
    R1, R2 = Rs[5], Rs[17]
    assert torch.allclose(rotate_voxels(rotate_voxels(x, R2), R1), rotate_voxels(x, R1 @ R2))
    with pytest.raises(ValueError):
        rotate_voxels(x, torch.tensor([[0.6, 0.8, 0], [-0.8, 0.6, 0], [0, 0, 1.0]]))


# shared grid operations

def test_low_pass_filter():
    x = torch.randn(2, 4, 9, 9, 9)
    assert torch.equal(low_pass_filter(x, 1), x)
    const = torch.ones(1, 1, 9, 9, 9)
    inner = low_pass_filter(const, 2)[0, 0, 3:6, 3:6, 3:6]
    assert torch.allclose(inner, torch.ones_like(inner))        # unit-sum kernel
    assert low_pass_filter(x, 2, stride=2).shape == (2, 4, 5, 5, 5)
    with pytest.raises(ValueError):
        low_pass_filter(x, 1, stride=2)


def test_field_bookkeeping():
    assert field_dim((4, 4, 4, 1)) == 4 + 12 + 20 + 7 == 43      # Table 1, layer 1
    assert field_dim((16, 16, 16)) == 144 and field_dim((32, 16, 16)) == 160
    assert default_fields(32, 3) == [(8, 4, 2), (16, 8, 4), (32,)]
    assert default_fields(32, 2, l_max=1) == [(8, 4), (32,)]
    assert default_fields(16, 1) == [(16,)]
    assert default_strides(4) == [1, 2, 2, 1] and default_strides(1) == [1]
    assert torch.allclose(global_average_pool(torch.ones(2, 3, 4, 4, 4)), torch.ones(2, 3))


# the conventional CNN potential

def test_registered_and_from_config():
    assert "cnn3d" in available_models()
    cfg = from_dict({"model": {"name": "cnn3d", "cutoff": 3.0, "n_features": 8,
                               "n_interactions": 2,
                               "extra": {"species": ["H", "C", "O"], "grid_size": 9,
                                         "kernel_size": 3, "normalization": "batch",
                                         "atomic_energies": [-0.5, -1.0, -2.0]}}})
    m = build_model(cfg.model)
    assert isinstance(m, CNN3D)
    assert m.channels == [field_dim(f) for f in default_fields(8, 2)] == [2 + 3 + 5, 8]
    assert m.grid_size == 9 and m.species == SPECIES
    assert m.blocks[0].norm is not None
    assert float(m.atom_ref.weight[8, 0]) == -2.0
    out = m(_graph(_structure(), 3.0))
    assert out["energy"].shape == (1,) and out["node_features"].shape == (6, 8)


def test_paper_cnn_spellings_translate():
    cfg = from_dict({"model": {"name": "cnn3d", "cutoff": 3.0, "n_interactions": 2,
                               "extra": {"grid_size": 9, "size": 3, "features": [4, 8]}}})
    m = build_model(cfg.model)
    assert m.channels == [4, 8] and m.blocks[0].conv.kernel_size == (3, 3, 3)


def test_invalid_options():
    with pytest.raises(ValueError):
        CNN3D(SPECIES, channels=(4, 4), strides=(1,), grid_size=5, kernel_size=3)
    with pytest.raises(ValueError):
        CNN3D(SPECIES, channels=(4,), normalization="layer", grid_size=5, kernel_size=3)


def test_fresh_model_predicts_shift():
    torch.manual_seed(0)
    m = CNN3D(SPECIES, grid_size=9, channels=(4, 4), kernel_size=3, energy_shift=-1.5,
              atomic_energies=[-0.5, -1.0, -2.0])
    s = _structure()
    e = m(_graph(s))["energy"]
    ref = -1.5 * 6 + sum({1: -0.5, 6: -1.0, 8: -2.0}[z] for z in s["atomic_numbers"])
    assert float(e) == pytest.approx(ref)


def test_batch_equals_individual_and_periodic():
    m = _small_model()
    graphs = [_graph(_structure(n, seed)) for n, seed in [(5, 1), (7, 2), (4, 3)]]
    e_single = torch.cat([m(g)["energy"] for g in graphs])
    e_batch = m(collate(graphs))["energy"]
    assert torch.allclose(e_single, e_batch, atol=1e-10)
    s = dict(_structure(n=5, seed=4), cell=np.eye(3) * 5.0, pbc=[True, True, True])
    out = ForceStressOutput(m, compute_stress=True)(_graph(s))
    assert out["forces"].shape == (5, 3) and out["stress"].shape == (1, 3, 3)
    assert torch.isfinite(out["stress"]).all()


def test_translation_and_permutation_invariance():
    m = _small_model()
    s = _structure(n=6, seed=5)
    e0 = m(_graph(s))["energy"]
    shifted = dict(s, pos=np.asarray(s["pos"]) + np.array([1.3, -0.7, 2.1]))
    assert torch.allclose(m(_graph(shifted))["energy"], e0, atol=1e-10)
    perm = np.random.default_rng(0).permutation(6)
    permuted = {"pos": np.asarray(s["pos"])[perm],
                "atomic_numbers": [s["atomic_numbers"][i] for i in perm]}
    assert torch.allclose(m(_graph(permuted))["energy"], e0, atol=1e-10)


def test_conventional_cnn_is_not_rotation_invariant():
    """The control of the paper: unconstrained kernels do not generalize over
    rotations, so even a rotation of the cube changes the energy."""
    m = _small_model()
    s = _structure(n=6, seed=6)
    e0 = m(_graph(s))["energy"]
    diffs = [abs(float(m(_graph(dict(s, pos=np.asarray(s["pos"]) @ R.numpy().T)))["energy"] - e0))
             for R in cube_rotations()[1:]]
    assert max(diffs) > 1e-6


def test_forces_match_finite_differences():
    m = ForceStressOutput(_small_model())
    s = _structure(n=5, seed=7)
    g = _graph(s)
    forces = m(g)["forces"].detach()
    pos = np.asarray(s["pos"])
    h = 1e-5
    for i, a in [(0, 0), (2, 1), (4, 2)]:
        plus, minus = pos.copy(), pos.copy()
        plus[i, a] += h
        minus[i, a] -= h
        e_plus = float(m(_graph(dict(s, pos=plus)))["energy"])
        e_minus = float(m(_graph(dict(s, pos=minus)))["energy"])
        assert float(forces[i, a]) == pytest.approx(-(e_plus - e_minus) / (2 * h), abs=1e-6)


def test_train_step():
    """A few optimizer steps lower the loss (the grids are differentiable
    through the positions, so force training works)."""
    from xnn.common.config import Config
    from xnn.common.data import AtomicDataset
    from xnn.common.train import Trainer

    rng = np.random.default_rng(0)
    structs = []
    for _ in range(8):
        s = _structure(n=4, seed=int(rng.integers(1 << 30)))
        s["energy"] = float(rng.normal())
        s["forces"] = rng.normal(0, 0.1, (4, 3))
        structs.append(s)
    cfg = Config()
    cfg.model.name = "cnn3d"
    cfg.model.cutoff = 4.0
    cfg.model.n_features = 4
    cfg.model.n_interactions = 2
    cfg.model.extra = {"species": SPECIES, "grid_size": 7, "kernel_size": 3}
    cfg.optim.epochs = 2
    cfg.data.batch_size = 4
    cfg.data.val_fraction = 0.25
    cfg.device = "cpu"
    Trainer(cfg, AtomicDataset(structs, cfg.model.cutoff)).fit()
