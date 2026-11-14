"""Shared pieces of the graph-parameterized force fields: induced dipoles, MLP sizes, inductor options and
license notices."""
import logging

import pytest
import torch

from xnn.common.models.blocks import MLP
from xnn.ffnn.models.terms import induced_dipoles, polarization_system, solve_dipoles


def test_induced_dipoles_solve_the_polarization_system():
    """The dipoles solve ``A mu = E`` and the per-atom energies are ``-mu . E / 2``."""
    gen = torch.Generator().manual_seed(0)
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.1, 0.3, 0.0], [0.4, 2.5, 0.6], [2.6, 2.2, -1.1]],
                       dtype=torch.float64)[None]
    q = torch.tensor([0.4, -0.3, 0.2, -0.3], dtype=torch.float64)
    alpha = 0.5 + torch.rand(4, generator=gen, dtype=torch.float64)
    scale = 1.0 - torch.eye(4, dtype=torch.float64)
    energy, mu, field = induced_dipoles(pos, q, alpha, alpha, scale, thole=0.39)
    A, E = polarization_system(pos, q, alpha, alpha, scale, 0.39)
    assert torch.allclose(field, E, rtol=1e-14, atol=0.0)
    assert torch.allclose(A @ mu.reshape(1, -1, 1), E.reshape(1, -1, 1), rtol=1e-12, atol=1e-14)
    assert torch.allclose(energy, -0.5 * (mu * E).sum(-1), rtol=1e-14, atol=0.0)


def test_dipole_solve_of_an_indefinite_system():
    """A system that is not positive definite (a polarization catastrophe) is solved by LU."""
    A = torch.tensor([[2.0, 0.5, 0.0], [0.5, -1.0, 0.3], [0.0, 0.3, 3.0]], dtype=torch.float64)
    E = torch.tensor([[[1.0, -2.0, 0.5]]], dtype=torch.float64)
    mu = solve_dipoles(A[None], E)
    assert mu.shape == (1, 1, 3)
    assert torch.allclose(A @ mu.reshape(3), E.reshape(3), rtol=1e-14, atol=1e-15)


def test_mlp_needs_two_sizes():
    """An MLP without an output size raises."""
    with pytest.raises(ValueError, match="input and an output"):
        MLP([4])


def test_inductor_options_keep_known_names():
    """Only the inductor options this PyTorch knows are passed on; none known gives ``None``."""
    from xnn.common.models.compiled import _inductor_options
    options = {"max_autotune": False, "triton.cudagraphs": False, "triton.no_such_option": 1, "no_such": 2}
    assert _inductor_options(options) == {"max_autotune": False, "triton.cudagraphs": False}
    assert _inductor_options({"no_such": 2}) is None
    assert _inductor_options(None) is None


def test_academic_software_license_notice(caplog):
    """A model under the Academic Software License states its terms once when it loads."""
    from xnn.common.models.hub.card import ModelCard
    from xnn.common.models.hub.loader import _license_notice, _noticed
    card = ModelCard(name="asl-licensed-test-model", license="ASL")
    _noticed.discard(card.name)
    with caplog.at_level(logging.WARNING):
        _license_notice(card)
        _license_notice(card)
    assert caplog.text.count("Academic Software License") == 1 and card.name in caplog.text
    _noticed.discard(card.name)
