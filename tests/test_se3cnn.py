"""3D steerable CNN (Weiler et al., NeurIPS 2018): manuscript fidelity.

The kernel basis is checked against the paper's equations: the
steerability constraint of eq 12 under random continuous rotations, the
change of basis ``Q`` of eq 14 against the numerical null-space solution
the paper describes (Sec. 4.4.1), the basis counts of the bandlimited
shells, and exact equivariance of the sampled kernels, the convolution,
the gated nonlinearity, the batch normalization and the whole potential
under the rotations of the cube. Rotation invariance of the energy under
arbitrary rotations is approximate (the bandlimit) and is compared with
the conventional CNN, which has none.
"""
import math

import numpy as np
import pytest
import torch

e3nn = pytest.importorskip("e3nn")
from e3nn import o3  # noqa: E402

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput, available_models, build_model  # noqa: E402
from xnn.cnn.models import (CNN3D, GatedBlock, SteerableBatchNorm, SteerableCNN,  # noqa: E402
                            SteerableConv3d, angular_kernel_basis, field_dim,
                            n_basis_kernels, rotate_fields, rotate_voxels,
                            steerable_kernel_basis)
from xnn.cnn.models.steerable import BANDLIMITS, shell_bandlimits  # noqa: E402

from test_cnn3d import SPECIES, _graph, _structure, cube_rotations  # noqa: E402


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _D(l, R):
    return o3.wigner_D(l, *o3.matrix_to_angles(R.to(torch.float64)))


# the kernel basis (Sec. 4.2 and 4.4.1)

@pytest.mark.parametrize("l_in,l_out", [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)])
def test_angular_basis_is_steerable(l_in, l_out):
    """eq 12: kappa(r x) = D^j(r) kappa(x) D^l(r)^T for every coupled J."""
    torch.manual_seed(0)
    x = torch.randn(40, 3)
    for J in range(abs(l_in - l_out), l_in + l_out + 1):
        for _ in range(3):
            R = o3.rand_matrix()
            lhs = angular_kernel_basis(l_in, l_out, J, x @ R.T)          # kappa(R x)
            rhs = torch.einsum("ij,jkp,lk->ilp", _D(l_out, R),
                               angular_kernel_basis(l_in, l_out, J, x), _D(l_in, R))
            assert torch.allclose(lhs, rhs, atol=1e-10)


@pytest.mark.parametrize("l_in,l_out,J", [(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 3)])
def test_change_of_basis_is_the_numerical_solution(l_in, l_out, J):
    """The Clebsch-Gordan Q^J spans the one-dimensional null space of the
    linear constraint of eq 14, [D^j (x) D^l](r) Q = Q D^J(r), solved
    numerically for several random rotations as the paper does."""
    torch.manual_seed(1)
    n = (2 * l_out + 1) * (2 * l_in + 1)
    rows = []
    for _ in range(5):
        R = o3.rand_matrix()
        tensor = torch.kron(_D(l_out, R), _D(l_in, R))
        rows.append(torch.kron(tensor, torch.eye(2 * J + 1))
                    - torch.kron(torch.eye(n), _D(J, R).T.contiguous()))
    A = torch.cat(rows)
    _, sing, vh = torch.linalg.svd(A)
    null = vh[sing < 1e-9]
    assert null.shape[0] == 1                                     # a unique solution
    Q = o3.wigner_3j(l_out, l_in, J, dtype=torch.float64).reshape(n, 2 * J + 1).reshape(-1)
    assert abs(float(null[0] @ Q / Q.norm())) == pytest.approx(1.0, abs=1e-9)


def test_basis_counts_and_bandlimits():
    """Sec. 4.4.1: shells at radii 0 .. s//2, J <= J_max(m) per shell."""
    assert shell_bandlimits(5, "compromise") == ([0.0, 1.0, 2.0], [0, 3, 5])
    assert shell_bandlimits(7, "conservative")[1] == [0, 2, 4, 6]
    assert shell_bandlimits(5, [0, 1, 1])[1] == [0, 1, 1]
    with pytest.raises(ValueError):
        shell_bandlimits(5, "loose")
    with pytest.raises(ValueError):
        shell_bandlimits(7, [0, 2])
    assert n_basis_kernels(0, 0, 5) == 3           # J = 0 on every shell
    assert n_basis_kernels(0, 1, 5) == 2           # J = 1 needs J_max >= 1: shells 1 and 2
    assert n_basis_kernels(1, 1, 5) == 1 + 3 + 3   # J = 0, 1, 2
    assert n_basis_kernels(2, 3, 5) == 0 + 3 + 5   # J = 1 .. 5
    assert n_basis_kernels(1, 1, 5, "conservative") == 1 + 3 + 3
    assert n_basis_kernels(2, 3, 5, "conservative") == 0 + 2 + 4
    assert steerable_kernel_basis(3, 3, 1) is not None       # J = 0 at the origin
    assert steerable_kernel_basis(0, 3, 1) is None           # J = 3 is bandlimited away
    for name, limits in BANDLIMITS.items():
        assert limits[0] == 0 and all(a < b for a, b in zip(limits, limits[1:]))


def test_sampled_basis_shapes_norms_and_shells():
    basis = steerable_kernel_basis(1, 2, 5)
    assert basis.shape == (n_basis_kernels(1, 2, 5), 5, 3, 5, 5, 5)
    assert basis.dtype == torch.float64
    assert torch.allclose(basis.flatten(1).norm(dim=1), torch.ones(basis.shape[0]))
    # the first shell (radius 0) is bandlimited to J = 0, which l = 1 -> 2 lacks,
    # so every basis kernel vanishes at the center voxel
    assert float(basis[:, :, :, 2, 2, 2].abs().max()) < 1e-12


def test_sampled_basis_rotates_exactly_on_the_grid():
    """kappa(R^-1 x) = D^j(R)^T kappa(x) D^l(R) for the rotations of the cube."""
    for l_in, l_out in [(0, 1), (1, 1), (1, 2), (2, 2)]:
        basis = steerable_kernel_basis(l_in, l_out, 5)
        for R in cube_rotations():
            lhs = rotate_voxels(basis, R)
            rhs = torch.einsum("ji,bjkxyz,kl->bilxyz", _D(l_out, R), basis, _D(l_in, R))
            assert torch.allclose(lhs, rhs, atol=1e-10)


# the layers

def _field_stack(fields, size=9, batch=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, field_dim(fields), size, size, size)


def test_steerable_conv_equivariance():
    torch.manual_seed(0)
    fields_in, fields_out = (2, 1, 1), (2, 2, 1)
    conv = SteerableConv3d(fields_in, fields_out, kernel_size=5, n_gates=3)
    assert conv.in_channels == field_dim(fields_in) and conv.out_channels == field_dim(fields_out) + 3
    assert conv.kernel().shape == (conv.out_channels, conv.in_channels, 5, 5, 5)
    x = _field_stack(fields_in)
    y = conv(x)
    assert y.shape == (2, conv.out_channels, 9, 9, 9)
    # the gates are three more scalar fields, appended after the stack
    for R in cube_rotations():
        y_rot = conv(rotate_fields(x, R, fields_in))
        ref = torch.cat([rotate_fields(y[:, :field_dim(fields_out)], R, fields_out),
                         rotate_voxels(y[:, field_dim(fields_out):], R)], dim=1)
        assert torch.allclose(y_rot, ref, atol=1e-9)


def test_steerable_conv_weight_layout_and_scaling():
    """One weight per (output field, input field, basis kernel); blocks with
    no basis kernel contribute zero."""
    conv = SteerableConv3d((2,), (1, 1), kernel_size=5)
    assert conv.weight["0_0"].shape == (1, 2, n_basis_kernels(0, 0, 5))
    assert conv.weight["1_0"].shape == (1, 2, n_basis_kernels(0, 1, 5))
    conv = SteerableConv3d((1,), (1, 0, 0, 1), kernel_size=1)   # 0 -> 3 has no kernel at size 1
    assert "1_0" not in conv.weight
    k = conv.kernel()
    assert k.shape == (8, 1, 1, 1, 1) and float(k[1:].abs().max()) == 0.0


def test_gated_block_equivariance_and_shapes():
    torch.manual_seed(0)
    fields_in, fields_out = (2, 1), (3, 2, 1)
    block = GatedBlock(fields_in, fields_out, kernel_size=3, activation="relu")
    assert block.n_gates == 3 and block.bias.shape == (3 + 3,)
    x = _field_stack(fields_in)
    y = block(x)
    assert y.shape == (2, field_dim(fields_out), 9, 9, 9)
    for R in cube_rotations():
        assert torch.allclose(block(rotate_fields(x, R, fields_in)), rotate_fields(y, R, fields_out), atol=1e-9)
    # a strided block (low-pass filter, then subsampling) stays exactly
    # equivariant when the grid keeps a center voxel (9 -> 5)
    block = GatedBlock(fields_in, fields_out, kernel_size=3, stride=2)
    y = block(x)
    assert y.shape == (2, field_dim(fields_out), 5, 5, 5)
    for R in cube_rotations():
        assert torch.allclose(block(rotate_fields(x, R, fields_in)), rotate_fields(y, R, fields_out), atol=1e-9)
    # without gates and activation the block is linear
    lin = GatedBlock(fields_in, fields_out, kernel_size=3, activation=None, gate_activation=None)
    assert lin.n_gates == 0 and lin.bias is None
    assert torch.allclose(lin(2 * x), 2 * lin(x))


def test_gates_multiply_non_scalar_fields():
    """Fig. 5: the non-scalar fields of the convolution output are scaled by
    the sigmoid of their gate; the scalars pass through the activation."""
    torch.manual_seed(0)
    block = GatedBlock((1,), (1, 1), kernel_size=3, activation="relu")
    x = _field_stack((1,))
    y = block.conv(x)                                            # scalar, vector (3), gate
    vector = y[:, 1:4] * torch.sigmoid(y[:, 4:5] + block.bias[1])
    scalar = torch.relu(y[:, :1] + block.bias[0])
    assert torch.allclose(block(x), torch.cat([scalar, vector], dim=1))


def test_batch_norm_equivariance_and_statistics():
    torch.manual_seed(0)
    fields = (2, 1, 1)
    bn = SteerableBatchNorm(fields)
    x = _field_stack(fields, batch=4) * 3 + 1
    for mode in (True, False):
        bn.train(mode)
        y = bn(x)
        for R in cube_rotations()[:6]:
            assert torch.allclose(bn(rotate_fields(x, R, fields)), rotate_fields(y, R, fields), atol=1e-9)
    bn.train()
    y = bn(x)
    # scalars are standardized; the vector field has unit mean squared norm (eq 17)
    assert torch.allclose(y[:, :2].mean(dim=(0, 2, 3, 4)), torch.zeros(2), atol=1e-10)
    assert torch.allclose(y[:, :2].pow(2).mean(dim=(0, 2, 3, 4)), torch.ones(2), atol=1e-4)
    assert float(y[:, 2:5].pow(2).sum(1).mean()) == pytest.approx(1.0, abs=1e-4)
    assert bn.running_mean.shape == (2,) and bn.running_var.shape == (4,)


# the potential

def _small_model(**kw):
    torch.manual_seed(0)
    kw.setdefault("species", SPECIES)
    kw.setdefault("cutoff", 4.0)
    kw.setdefault("grid_size", 9)
    kw.setdefault("fields", [(3, 2, 1), (4,)])
    kw.setdefault("kernel_size", 3)
    m = SteerableCNN(**kw)
    torch.nn.init.normal_(m.readout[-1].weight)
    torch.nn.init.normal_(m.readout[-1].bias)
    return m


def test_registered_and_from_config():
    assert "se3cnn" in available_models()
    cfg = from_dict({"model": {"name": "se3cnn", "cutoff": 3.0, "n_features": 8,
                               "n_interactions": 3,
                               "extra": {"species": ["H", "C", "O"], "grid_size": 9,
                                         "kernel_size": 3, "l_max": 1,
                                         "bandlimit": "conservative", "normalization": "batch",
                                         "atomic_energies": {"H": -0.5, "C": -1.0, "O": -2.0}}}})
    m = build_model(cfg.model)
    assert isinstance(m, SteerableCNN)
    assert m.fields == [(2, 1), (4, 2), (8,)]
    assert [b.stride for b in m.blocks] == [1, 2, 1]
    assert m.blocks[0].norm is not None
    assert m.blocks[1].conv.weight["0_0"].shape[-1] == n_basis_kernels(0, 0, 3, "conservative")
    assert float(m.atom_ref.weight[6, 0]) == -1.0
    out = m(_graph(_structure(), 3.0))
    assert out["energy"].shape == (1,) and out["node_features"].shape == (6, 8)


def test_paper_spellings_translate():
    cfg = from_dict({"model": {"name": "se3cnn", "cutoff": 3.0, "n_interactions": 2,
                               "extra": {"grid_size": 9, "size": 3,
                                         "features": [[2, 1], [4]]}}})
    m = build_model(cfg.model)
    assert m.fields == [(2, 1), (4,)] and m.blocks[0].conv.kernel_size == 3


def test_invalid_architectures():
    with pytest.raises(ValueError, match="scalar"):
        SteerableCNN(SPECIES, fields=[(2, 1), (2, 1)], grid_size=5, kernel_size=3)
    with pytest.raises(ValueError):
        SteerableCNN(SPECIES, fields=[(2,), (2,)], strides=[1], grid_size=5, kernel_size=3)


def test_fresh_model_predicts_shift():
    torch.manual_seed(0)
    m = SteerableCNN(SPECIES, grid_size=9, fields=[(2, 1), (4,)], kernel_size=3,
                     energy_shift=-1.5, atomic_energies=[-0.5, -1.0, -2.0])
    s = _structure()
    ref = -1.5 * 6 + sum({1: -0.5, 6: -1.0, 8: -2.0}[z] for z in s["atomic_numbers"])
    assert float(m(_graph(s))["energy"]) == pytest.approx(ref)


def test_energy_invariant_under_cube_rotations_forces_rotate():
    """Exact SE(3) equivariance on the grid: the energy is the same for the 24
    rotations of the cube and the forces co-rotate."""
    m = ForceStressOutput(_small_model())
    s = _structure(n=6, seed=8)
    out = m(_graph(s))
    e0, f0 = out["energy"].detach(), out["forces"].detach()
    assert float(f0.abs().max()) > 1e-6
    for R in cube_rotations():
        rot = m(_graph(dict(s, pos=np.asarray(s["pos"]) @ R.numpy().T)))
        assert torch.allclose(rot["energy"].detach(), e0, atol=1e-9)
        assert torch.allclose(rot["forces"].detach(), f0 @ R.T, atol=1e-8)


def test_approximate_invariance_under_arbitrary_rotations():
    """Away from the grid symmetries the invariance holds to the bandlimit:
    the energy spread over random rotations is a small fraction of the
    spread over structures, unlike the conventional CNN's."""
    torch.manual_seed(0)
    structs = [_structure(n=6, seed=20 + k) for k in range(6)]
    Rs = [o3.rand_matrix().numpy() for _ in range(8)]

    def spread(model):
        e = torch.tensor([[float(model(_graph(dict(s, pos=np.asarray(s["pos"]) @ R.T)))["energy"])
                           for R in Rs] for s in structs])
        return float(e.std(dim=1).mean() / e.mean(dim=1).std())

    steerable = spread(_small_model(fields=[(4, 2, 1), (4,)]))
    torch.manual_seed(0)
    cnn = CNN3D(SPECIES, cutoff=4.0, grid_size=9, channels=(10, 4), kernel_size=3)
    torch.nn.init.normal_(cnn.readout[-1].weight)
    torch.nn.init.normal_(cnn.readout[-1].bias)
    conventional = spread(cnn)
    assert steerable < 0.1
    assert steerable < 0.5 * conventional


def test_translation_permutation_batch_periodic():
    m = _small_model()
    s = _structure(n=6, seed=5)
    e0 = m(_graph(s))["energy"]
    shifted = dict(s, pos=np.asarray(s["pos"]) + np.array([1.3, -0.7, 2.1]))
    assert torch.allclose(m(_graph(shifted))["energy"], e0, atol=1e-10)
    perm = np.random.default_rng(0).permutation(6)
    permuted = {"pos": np.asarray(s["pos"])[perm],
                "atomic_numbers": [s["atomic_numbers"][i] for i in perm]}
    assert torch.allclose(m(_graph(permuted))["energy"], e0, atol=1e-10)
    graphs = [_graph(_structure(n, seed)) for n, seed in [(5, 1), (7, 2), (4, 3)]]
    e_single = torch.cat([m(g)["energy"] for g in graphs])
    assert torch.allclose(m(collate(graphs))["energy"], e_single, atol=1e-10)
    periodic = dict(_structure(n=5, seed=4), cell=np.eye(3) * 5.0, pbc=[True, True, True])
    out = ForceStressOutput(m, compute_stress=True)(_graph(periodic))
    assert out["stress"].shape == (1, 3, 3) and torch.isfinite(out["stress"]).all()


def test_forces_match_finite_differences():
    m = ForceStressOutput(_small_model())
    s = _structure(n=5, seed=7)
    forces = m(_graph(s))["forces"].detach()
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
    cfg.model.name = "se3cnn"
    cfg.model.cutoff = 4.0
    cfg.model.n_features = 4
    cfg.model.n_interactions = 2
    cfg.model.extra = {"species": SPECIES, "grid_size": 7, "kernel_size": 3, "l_max": 1}
    cfg.optim.epochs = 2
    cfg.data.batch_size = 4
    cfg.data.val_fraction = 0.25
    cfg.device = "cpu"
    Trainer(cfg, AtomicDataset(structs, cfg.model.cutoff)).fit()
