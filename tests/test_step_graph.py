"""Steps on a step plan (xnn.common.deploy.StepGraph): a fixed solve length that does not converge, the
checks that lower the lengths, and steps whose cell changes."""
import numpy as np
import pytest
import torch

from test_amoeba_periodic import BOX, CELL, Z, _box
from xnn.common.data import structure_to_graph
from xnn.common.deploy.step_graph import StepGraph
from xnn.common.models import ForceStressOutput
from xnn.ffnn.models import Amoeba, Hippo


@pytest.fixture(autouse=True)
def _f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)



def test_a_fixed_length_too_short_is_raised():
    """A step whose fixed-length solve does not converge is evaluated again with converged solves (its
    dipoles taken back out of the predictor) and equals the plain step; the next fixed length lies above
    the one that failed. With ``shorten`` 0 the lengths are never lowered, and a failed length stops
    being a floor after four times ``shorten`` converged steps."""
    ref, model = (ForceStressOutput(_box(Amoeba), compute_stress=True).eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
    velocity = np.random.default_rng(5).normal(scale=0.004, size=BOX.shape)
    for step in range(5):
        pos = BOX + step * velocity
        s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(Z), "cell": torch.tensor(CELL),
             "pbc": torch.ones(3, dtype=torch.bool)}
        if step == 2:
            assert stepper.plan.solve_iterations
            stepper.plan.solve_iterations = dict(stepper.plan.solve_iterations, forward=1)
        got, want = stepper(s), ref(structure_to_graph(s, 6.3))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-8), key
        if step == 2:
            assert stepper.floor["forward"] == 1 and stepper.plan.solve_iterations is None
    assert stepper.plan.solve_iterations["forward"] >= 3
    stepper.needed, stepper.shorten = [], 0
    stepper._shorten([1, 1])
    assert stepper.needed == []
    stepper.shorten, stepper.settled = 2, 8
    stepper._shorten([1, 1])
    assert stepper.floor == {} and stepper.needed == [[1, 1]]


@pytest.mark.parametrize("cls", (Amoeba, Hippo))
@pytest.mark.parametrize("stress", (False, True))
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="records on a GPU"))])
def test_steps_follow_a_changing_cell(cls, stress, device):
    """Steps whose cell changes (a barostat scaling the molecules' centers with the box, then a cell that
    shrinks without the atoms) equal fresh evaluations at every step: the plan's mesh terms follow the cell,
    replayed steps included, and the pair list is rebuilt once the images may have moved past the skin.
    Without the stress the cell carries no gradient and the plan keeps its mesh terms across steps."""
    ref, model = (ForceStressOutput(_box(cls), compute_stress=stress).to(device).eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
    rng = np.random.default_rng(7)
    velocity = rng.normal(scale=0.002, size=BOX.shape)
    centers = BOX.reshape(-1, 3, 3)[:, 0]
    rebuilds = None
    for step in range(12):
        pos, cell = BOX + step * velocity, CELL.copy()
        if 6 <= step < 10:
            # the box grows by 0.1 % per step and every molecule moves with its oxygen's scaled position
            scale = 1.001 ** (step - 5)
            pos = (pos.reshape(-1, 3, 3) + ((scale - 1.0) * centers)[:, None, :]).reshape(-1, 3)
            cell = CELL * scale
        elif step >= 10:
            # the cell alone shrinks by more than the skin can take: the images come closer, the atoms
            # stay where they are
            pos = BOX + 9 * velocity
            cell = CELL - np.diag([0.4, 0.4, 0.4]) * (step - 9)
        if step == 10:
            rebuilds = stepper.rebuilds
        s = {"pos": torch.tensor(pos, device=device), "atomic_numbers": torch.tensor(Z, device=device),
             "cell": torch.tensor(cell, device=device), "pbc": torch.ones(3, dtype=torch.bool, device=device)}
        got, want = stepper(s), ref(structure_to_graph(s, 6.3, device=device))
        for key in ("energy", "forces", "stress") if stress else ("energy", "forces"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-7), (key, step)
    assert stepper.rebuilds > rebuilds
    if device == "cuda":
        assert stepper.replays > 0

