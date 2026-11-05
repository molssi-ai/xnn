"""Steps on a step plan (xnn.common.deploy.StepGraph) on the CPU: a fixed solve length that does not
converge, and the checks that lower the lengths."""
import numpy as np
import pytest
import torch

from test_amoeba_periodic import BOX, CELL, Z, _box
from xnn.common.data import structure_to_graph
from xnn.common.deploy.step_graph import StepGraph
from xnn.common.models import ForceStressOutput
from xnn.ffnn.models import Amoeba


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
