"""Hessians and harmonic analysis of any model (xnn.common.models.vibrations).

On a model of harmonic springs the Hessians and Hessian-vector products of a batch equal autograd's dense
Hessian per structure, stay differentiable in the parameters, and the harmonic analysis gives the closed-form
frequencies of a diatomic and of a linear symmetric triatomic, the right number of vibrations for atoms, linear
and bent molecules, rotation-invariant frequencies, imaginary modes as negative numbers and the zero-point
energy.
"""
import math
import warnings

import numpy as np
import pytest
import torch
from torch import nn

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import (
    ForceStressOutput,
    exact_curvature,
    harmonic_analysis,
    hessian,
    hessian_vector_product,
)
from xnn.common.models.ops import ATOMIC_MASSES
from xnn.common.models.vibrations import HC, WAVENUMBER


class Springs(nn.Module):
    """Springs between given atom pairs (global indices), ``k/2 (r - d)^2 + c (r - d)^3`` (eV, Angstrom)."""

    def __init__(self, pairs, d0, k=10.0, c=0.0):
        super().__init__()
        self.register_buffer("pairs", torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2))
        self.register_buffer("d0", torch.as_tensor(d0, dtype=torch.float64))
        self.k = nn.Parameter(torch.tensor(float(k), dtype=torch.float64))
        self.c = c
        self.position_response = False

    def forward(self, data):
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        x = (data.pos[i] - data.pos[j]).norm(dim=-1) - self.d0
        e = 0.5 * self.k * x * x + self.c * x ** 3
        return {"energy": torch.zeros(data.num_graphs, dtype=e.dtype).index_add(0, data.batch[i], e)}


def _graph(pos, z):
    return structure_to_graph({"pos": torch.as_tensor(pos, dtype=torch.float64), "atomic_numbers": np.asarray(z)},
                              0.1)


BENT = (np.array([[0.0, 0.0, 0.1], [0.0, 0.77, -0.45], [0.02, -0.76, -0.47]]), [8, 1, 1])


def _bent_model(scale=1.0, c=0.3):
    d0 = [0.95, 0.95, 1.5]
    return Springs([[0, 1], [0, 2], [1, 2]], [scale * d for d in d0], c=c)


def test_hessians_of_a_batch_match_dense_autograd():
    """Each structure's Hessian of a batch, and several Hessian-vector products at once, equal the dense autograd
    Hessian of its energy; the model may come wrapped for forces."""
    pos, z = BENT
    model = Springs([[0, 1], [0, 2], [1, 2], [3, 4]], [0.95, 0.95, 1.5, 1.1], c=0.3)
    batch = collate([_graph(pos, z), _graph([[0, 0, 0], [0.2, 0.1, 1.2]], [6, 8])])

    def energy(x, b):
        g = collate([_graph(pos, z), _graph([[0, 0, 0], [0.2, 0.1, 1.2]], [6, 8])])
        g.pos = x
        return model(g)["energy"][b]

    x0 = batch.pos.detach().clone()
    dense = [torch.autograd.functional.hessian(lambda x, b=b: energy(x, b), x0).reshape(15, 15) for b in range(2)]
    blocks = [dense[0][:9, :9], dense[1][9:, 9:]]
    for H, ref in zip(hessian(ForceStressOutput(model), batch), blocks):
        assert torch.allclose(H, ref, rtol=0.0, atol=1e-12)
    v = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64).reshape(2, 5, 3)
    hv = hessian_vector_product(model, batch, v)
    full = dense[0] + dense[1]
    for k in range(2):
        assert torch.allclose(hv[k].reshape(-1), full @ v[k].reshape(-1), rtol=0.0, atol=1e-12)
    assert torch.allclose(hessian_vector_product(model, batch, v[1]), hv[1], rtol=0.0, atol=1e-14)


def test_hessian_is_differentiable_in_the_parameters():
    """With ``create_graph`` the Hessian (and a curvature) carries the parameter gradient: the Hessian of
    springs at rest is linear in their force constant."""
    pos, z = BENT
    model = _bent_model(c=0.0)
    (H,) = hessian(model, _graph(pos, z), create_graph=True)
    (g,) = torch.autograd.grad(H.trace(), model.k)
    assert float(g) == pytest.approx(float(H.trace()) / float(model.k), rel=1e-12)
    u = torch.ones(3, 3, dtype=torch.float64)
    hu = hessian_vector_product(model, _graph(pos, z), u, create_graph=True)
    (gu,) = torch.autograd.grad((u * hu).sum(), model.k)
    assert float(gu) == pytest.approx(float(u.reshape(-1) @ H.detach() @ u.reshape(-1)) / float(model.k),
                                      rel=1e-12)


def test_position_independent_energy_has_zero_hessian():
    """A model whose energy does not depend on the positions (with or without gradients) has zero Hessians."""
    class Constant(nn.Module):
        def __init__(self, trainable):
            super().__init__()
            self.e = nn.Parameter(torch.ones(1, dtype=torch.float64), requires_grad=trainable)

        def forward(self, data):
            return {"energy": self.e.expand(data.num_graphs)}

    pos, z = BENT
    for trainable in (True, False):
        (H,) = hessian(Constant(trainable), _graph(pos, z))
        assert H.shape == (9, 9) and float(H.abs().max()) == 0.0


def test_exact_curvature_switches_and_restores():
    """The models' ``position_response`` is on during the evaluations and returns to its value afterwards, also
    after an error."""
    model = _bent_model()
    seen = []
    forward = model.forward
    model.forward = lambda data: seen.append(model.position_response) or forward(data)
    hessian(model, _graph(*BENT))
    assert seen == [True] and model.position_response is False
    with pytest.raises(RuntimeError):
        with exact_curvature(model):
            assert model.position_response is True
            raise RuntimeError
    assert model.position_response is False


def test_diatomic_and_linear_triatomic_frequencies():
    """A diatomic has one vibration at ``sqrt(k / mu)``; a linear X-Y-X of two springs has 3 n - 5 = 4 modes,
    the stretches at ``sqrt(k / m_X)`` and ``sqrt(k (1 / m_X + 2 / m_Y))`` and two zero bends; the zero-point
    energy is ``h c / 2`` times the sum of the wavenumbers."""
    k = 12.0
    pos = [[0.0, 0.0, 0.0], [0.3, 0.4, 1.0]]
    model = Springs([[0, 1]], [math.dist(*pos)], k=k)
    (H,) = hessian(model, _graph(pos, [6, 8]))
    out = harmonic_analysis(H, torch.tensor(pos), atomic_numbers=torch.tensor([6, 8]))
    mu = ATOMIC_MASSES[6] * ATOMIC_MASSES[8] / (ATOMIC_MASSES[6] + ATOMIC_MASSES[8])
    assert out["linear"] and out["frequencies"].shape == (1,)
    assert float(out["frequencies"][0]) == pytest.approx(math.sqrt(k / mu) * WAVENUMBER, rel=1e-10)
    assert float(out["zero_point_energy"]) == pytest.approx(0.5 * HC * float(out["frequencies"][0]), rel=1e-12)

    pos = [[0.0, 0.0, -1.16], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]
    model = Springs([[0, 1], [1, 2]], [1.16, 1.16], k=k)
    (H,) = hessian(model, _graph(pos, [8, 6, 8]))
    m = torch.tensor([ATOMIC_MASSES[8], ATOMIC_MASSES[6], ATOMIC_MASSES[8]])
    out = harmonic_analysis(H, torch.tensor(pos), masses=m)
    f = out["frequencies"]
    assert out["linear"] and f.shape == (4,)
    assert torch.allclose(f[:2], torch.zeros(2, dtype=f.dtype), atol=1e-4)
    mo, mc = float(m[0]), float(m[1])
    assert float(f[2]) == pytest.approx(math.sqrt(k / mo) * WAVENUMBER, rel=1e-10)
    assert float(f[3]) == pytest.approx(math.sqrt(k * (1 / mo + 2 / mc)) * WAVENUMBER, rel=1e-10)
    # the displacements are the Cartesian modes: d . H d is the eigenvalue
    d = out["displacements"].reshape(4, -1)
    assert torch.allclose(torch.einsum("ki,ij,kj->k", d, H, d), out["eigenvalues"], rtol=1e-10, atol=1e-10)


def test_bent_molecule_frequencies_are_rotation_invariant():
    """A bent triatomic has three vibrations; translations and rotations are projected out, so the frequencies of
    a rotated and shifted copy are the same; a single atom has none."""
    pos, z = BENT
    model = _bent_model()
    (H,) = hessian(model, _graph(pos, z))
    ref = harmonic_analysis(H, torch.tensor(pos), atomic_numbers=torch.tensor(z))
    assert not ref["linear"] and ref["frequencies"].shape == (3,)
    assert bool((ref["frequencies"] > 100.0).all())
    P = ref["projected_hessian"]
    assert torch.allclose(P, P.T, atol=1e-12) and P.shape == (9, 9)
    R, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    moved = pos @ R.T + [1.0, -2.0, 0.5]
    (H2,) = hessian(model, _graph(moved, z))
    out = harmonic_analysis(H2, torch.tensor(moved), atomic_numbers=torch.tensor(z))
    assert torch.allclose(out["frequencies"], ref["frequencies"], rtol=1e-9)
    single = harmonic_analysis(torch.zeros(3, 3, dtype=torch.float64), torch.zeros(1, 3), masses=torch.ones(1))
    assert single["frequencies"].numel() == 0 and float(single["zero_point_energy"]) == 0.0


def test_imaginary_modes_and_missing_masses():
    """A stretched spring of negative force constant gives an imaginary (negative) wavenumber that does not
    enter the zero-point energy; the analysis needs masses or atomic numbers."""
    pos, z = BENT
    model = _bent_model(c=0.0)
    with torch.no_grad():
        model.k.fill_(-5.0)
    (H,) = hessian(model, _graph(pos, z))
    out = harmonic_analysis(H, torch.tensor(pos), atomic_numbers=torch.tensor(z))
    f = out["frequencies"]
    assert float(f[0]) < 0.0
    assert float(out["zero_point_energy"]) == pytest.approx(0.5 * HC * float(f.clamp(min=0).sum()), rel=1e-12)
    with pytest.raises(ValueError, match="masses"):
        harmonic_analysis(H, torch.tensor(pos))


def test_models_with_approximate_second_derivatives_warn():
    """A model that declares its second derivatives incomplete warns; the others do not."""
    pos, z = BENT
    model = _bent_model()
    model.exact_second_derivatives = False
    with pytest.warns(RuntimeWarning, match="approximate"):
        hessian(model, _graph(pos, z))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hessian_vector_product(_bent_model(), _graph(pos, z), torch.ones(3, 3))
