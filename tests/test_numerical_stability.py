"""Numerical edge cases for derivatives and periodic geometry."""
import numpy as np
import pytest
import torch
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import D4Dispersion, ForceStressOutput
from xnn.common.models.ops import shifted_softplus, Softplus
from xnn.dnn.featurizers import AtomCenteredSymmetryFunctions
from xnn.ffnn.models.geometry import minimum_image_vectors, pair_vectors
from xnn.ffnn.models.topology import guess_bonds
from xnn.hybnn.models.spookynet import _exact_softplus


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('fn', [shifted_softplus, Softplus(), _exact_softplus],
                         ids=['shifted_softplus', 'Softplus', 'SpookyNet'])
def test_softplus_derivatives(fn, dtype):
    """Softplus has analytic derivatives at zero and finite extreme derivatives."""
    for implementation in (fn, torch.jit.script(fn)):
        x = torch.tensor([-1000., -30., -1., 0., 1., 30., 1000.],
                         dtype=dtype, requires_grad=True)
        value = implementation(x)
        first, = torch.autograd.grad(value.sum(), x, create_graph=True)
        second, = torch.autograd.grad(first.sum(), x)
        assert torch.isfinite(value).all()
        torch.testing.assert_close(first, torch.sigmoid(x))
        torch.testing.assert_close(second, torch.sigmoid(x) * torch.sigmoid(-x))
        assert (first[3].item(), second[3].item()) == pytest.approx((0.5, 0.25))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('kind', [3, 8, 9])
def test_acsf_collinear_derivatives_are_finite(dtype, kind):
    """Angular descriptors have finite derivatives on collinear distinct atoms."""
    descriptor = AtomCenteredSymmetryFunctions(
        [1], [{'kind': 'cosine', 'r_cut': 4.}],
        [{'element': 1, 'type': kind, 'neighbors': [1, 1], 'theta_s': 100.,
          'eta': 0.001 if kind == 8 else 0.1, 'lambda': 1., 'zeta': 2.}]).to(dtype)
    pos = torch.tensor([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
                       dtype=dtype, requires_grad=True)
    graph = structure_to_graph({'pos': pos, 'atomic_numbers': [1, 1, 1]}, 4.)
    features = descriptor(graph)
    assert torch.isfinite(features).all()
    gradient, = torch.autograd.grad(features.sum(), pos, create_graph=True)
    assert torch.isfinite(gradient).all(), gradient
    curvature, = torch.autograd.grad(gradient.square().sum(), pos)
    assert torch.isfinite(curvature).all(), curvature


def test_bonded_vectors_choose_nearest_triclinic_image():
    """The bonded vector length agrees with exhaustive periodic image enumeration."""
    cell = torch.tensor([[3., 0., 0.], [1.5, 3., 0.], [0., 0., 3.]], dtype=torch.float64)
    pos = torch.stack([torch.zeros(3, dtype=torch.float64),
                       torch.tensor([0.49, 0.49, 0.], dtype=torch.float64) @ cell])
    graph = structure_to_graph({'pos': pos, 'atomic_numbers': [6, 6],
                               'cell': cell, 'pbc': [True] * 3}, 3.)
    actual = pair_vectors(graph, torch.tensor([0]), torch.tensor([1])).norm()
    shifts = torch.cartesian_prod(*[torch.arange(-2, 3) for _ in range(3)]).double()
    expected = torch.linalg.norm(pos[1] + shifts @ cell, dim=1).min()
    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)
    assert guess_bonds(pos.numpy(), [6, 6], cell.numpy()) == [(0, 1)]


@pytest.mark.parametrize('skew', [0., 1.5])
@pytest.mark.parametrize('periodic', [[True, True, True], [True, False, True]])
def test_minimum_image_matches_enumeration(skew, periodic):
    """The image search matches enumeration on reduced cells with partial periodicity."""
    cell = torch.tensor([[3., 0., 0.], [skew, 3., 0.], [0., 0., 3.]], dtype=torch.float64)
    vec = torch.tensor([[0.49, 0.49, 0.1], [3.2, -2.1, 0.4]], dtype=torch.float64) @ cell
    pbc = torch.tensor(periodic)
    actual = minimum_image_vectors(vec, cell, pbc)
    shifts = torch.cartesian_prod(*[torch.arange(-15, 16) if p else torch.zeros(1, dtype=torch.long)
                                    for p in periodic]).double()
    expected = torch.linalg.norm(vec[:, None] + shifts @ cell, dim=-1).min(dim=1).values
    torch.testing.assert_close(actual.norm(dim=-1), expected)


def test_minimum_image_exact_on_reduced_cells():
    """Every image of random vectors is found on random Minkowski-reduced cells."""
    from ase.geometry import minkowski_reduce
    rng = np.random.default_rng(0)
    shifts = torch.cartesian_prod(*[torch.arange(-4, 5)] * 3).double()
    for _ in range(5):
        lengths = rng.uniform(8., 20., 3)
        raw = np.diag(lengths)
        raw[1, 0], raw[2, 0], raw[2, 1] = rng.uniform(-.5, .5, 3) * lengths[[0, 0, 1]]
        cell = torch.tensor(np.array(minkowski_reduce(raw)[0]), dtype=torch.float64)
        vec = torch.tensor(rng.normal(size=(500, 3)) * 12., dtype=torch.float64)
        expected = torch.linalg.norm(vec[:, None] + shifts @ cell, dim=-1).min(dim=1).values
        torch.testing.assert_close(minimum_image_vectors(vec, cell).norm(dim=-1), expected)


@pytest.mark.parametrize('skew', [1.5, 10.5])
def test_unreduced_cell_warns(skew):
    """An unreduced cell warns from the image search, a reduced one does not; bond guessing is exact on both."""
    import warnings
    cell = torch.tensor([[3., 0., 0.], [skew, 3., 0.], [0., 0., 3.]], dtype=torch.float64)
    vec = torch.tensor([[0.49, 0.49, 0.1]], dtype=torch.float64) @ cell
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        minimum_image_vectors(vec, cell)
        bonds = guess_bonds(torch.stack([torch.zeros(3, dtype=torch.float64), vec[0]]).numpy(), [6, 6], cell.numpy())
    assert sum('not reduced' in str(w.message) for w in caught) == (1 if skew > 1.5 else 0)
    assert bonds == [(0, 1)]


def test_pair_vectors_batched_structures():
    """A batch mixing orthorhombic, triclinic and molecular structures matches the per-structure result."""
    rng = np.random.default_rng(1)
    cells = [torch.eye(3, dtype=torch.float64) * 7.,
             torch.tensor([[6., 0., 0.], [2.5, 6., 0.], [-1., 2., 6.]], dtype=torch.float64), None]
    graphs, vecs = [], []
    for cell in cells:
        pos = torch.tensor(rng.uniform(0., 6., (8, 3)), dtype=torch.float64)
        s = {'pos': pos, 'atomic_numbers': [6] * 8}
        if cell is not None:
            s.update(cell=cell, pbc=[True] * 3)
        graphs.append(structure_to_graph(s, 2.))
        plain = pos[1:] - pos[:-1]
        vecs.append(plain if cell is None else minimum_image_vectors(plain, cell))
    batch = collate(graphs)
    a = torch.cat([torch.arange(7) + 8 * i for i in range(3)])
    torch.testing.assert_close(pair_vectors(batch, a, a + 1), torch.cat(vecs))


def test_minimum_image_position_and_cell_gradients():
    """Image selection preserves first and second position and cell derivatives."""
    cell = torch.tensor([[3., 0., 0.], [1.5, 3., 0.], [0., 0., 3.]],
                        dtype=torch.float64, requires_grad=True)
    vec = torch.tensor([[2.205, 1.47, 0.1]], dtype=torch.float64, requires_grad=True)

    def energy(vectors, lattice):
        return minimum_image_vectors(vectors, lattice).square().sum()

    assert torch.autograd.gradcheck(energy, (vec, cell))
    assert torch.autograd.gradgradcheck(energy, (vec, cell))


def test_nonperiodic_zero_cell():
    """A molecular cell placeholder does not require an invertible lattice."""
    vec = torch.tensor([[2., 3., 4.]])
    torch.testing.assert_close(minimum_image_vectors(
        vec, torch.zeros(3, 3), torch.zeros(3, dtype=torch.bool)), vec)


@pytest.mark.parametrize('skew', [0., 1.5])
def test_dense_d4_is_invariant_to_individual_lattice_translations(skew):
    """Whole-cell translations preserve periodic charges, energy, forces and stress."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        cell = torch.eye(3) * 6.
        cell[1, 0] = skew
        pos = torch.tensor([[0., 0., 0.119262], [0., 0.763239, -0.477047],
                            [0., -0.763239, -0.477047]]) + 3.
        model = ForceStressOutput(D4Dispersion(
            regime='dense', cutoff_pair=9., cutoff_triple=7.,
            cutoff_cn=8., cutoff_eeq_cn=8., cutoff_eeq=9.), compute_stress=True).eval()
        outputs = []
        for images in (0, 5):
            translated = pos.clone()
            translated[1] += images * (cell[0] - cell[1])
            graph = structure_to_graph({'pos': translated, 'atomic_numbers': [8, 1, 1],
                                        'cell': cell, 'pbc': [True] * 3}, model.cutoff)
            outputs.append(model(graph))
        for key in ('eeq_charges', 'energy', 'forces', 'stress'):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=1e-9, atol=1e-10)
    finally:
        torch.set_default_dtype(old)


def test_les_periodic_energy_is_rotation_invariant():
    """A cyclic rotation of a periodic cubic cell preserves the LES energy."""
    from xnn.common.models.les import EwaldSummation
    pos = torch.tensor([[1., 1., 1.], [2., 1., 1.]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 6.
    rotation = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=torch.float64)
    q = torch.tensor([1., -1.], dtype=torch.float64)
    batch = torch.zeros(2, dtype=torch.long)
    pbc = torch.ones(1, 3, dtype=torch.bool)
    model = EwaldSummation()
    pos.requires_grad_()
    original = model(q, pos, batch, 1, cell[None], pbc)
    rotated = model(q, pos @ rotation.T, batch, 1, (cell @ rotation.T)[None], pbc)
    torch.testing.assert_close(original, rotated, rtol=1e-10, atol=1e-12)
    force, = torch.autograd.grad(original.sum(), pos)
    rotated_force, = torch.autograd.grad(rotated.sum(), pos)
    torch.testing.assert_close(force, rotated_force, rtol=1e-10, atol=1e-12)
