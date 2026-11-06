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



VALENCE_TERMS = ("e_bond", "e_angle", "e_strbnd", "e_urey", "e_opbend", "e_torsion", "e_pitors", "e_tortor", "e_ref")


def _struct(pos, device):
    return {"pos": torch.tensor(pos, device=device), "atomic_numbers": torch.tensor(Z, device=device),
            "cell": torch.tensor(CELL, device=device), "pbc": torch.ones(3, dtype=torch.bool, device=device)}


def _valence_reference(core, s, device):
    """The valence energy and forces of a plain evaluation (its valence terms differentiated)."""
    graph = structure_to_graph(s, 6.3, device=device)
    graph.pos.requires_grad_(True)
    out = core(graph)
    energy = sum(out[key] for key in VALENCE_TERMS)
    return energy.detach(), -torch.autograd.grad(energy.sum(), graph.pos)[0]


@pytest.mark.parametrize("cls", (Amoeba, Hippo))
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="records on a GPU"))])
def test_inner_and_outer_steps_split_the_terms(cls, device):
    """Multiple time steps on one plan: inner (valence) steps equal the valence terms of a plain evaluation,
    also with an atom wrapped by a lattice vector between outer steps, and an inner plus an outer step at the
    same positions equal the whole step (energy, forces, stress)."""
    ref = ForceStressOutput(_box(cls), compute_stress=True).to(device).eval()
    model = ForceStressOutput(_box(cls, predictor="aspc"), compute_stress=True).to(device).eval()
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
    with pytest.raises(ValueError):
        stepper(_struct(BOX, device), terms="valence")
    with pytest.raises(ValueError):
        stepper(_struct(BOX, device), terms="bonded")
    velocity = np.random.default_rng(11).normal(scale=0.001, size=BOX.shape)
    pos = BOX.copy()
    for outer in range(6):
        s = _struct(pos, device)
        slow, fast = stepper(s, terms="nonbonded"), stepper(s, terms="valence")
        want = ref(structure_to_graph(s, 6.3, device=device))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(slow[key] + fast[key], want[key].detach(), rtol=1e-9, atol=1e-7), (key, outer)
        for inner in range(3):
            pos = pos + velocity
            if (outer, inner) == (3, 1):
                pos[1, 0] += CELL[0, 0]
            s = _struct(pos, device)
            fast = stepper(s, terms="valence")
            energy, forces = _valence_reference(ref.model, s, device)
            assert torch.allclose(fast["energy"], energy, rtol=1e-10, atol=1e-9), (outer, inner)
            assert torch.allclose(fast["forces"], forces, rtol=1e-9, atol=1e-8), (outer, inner)
    if device == "cuda":
        assert stepper.valence_replays > 0 and stepper.replays > 0


def test_a_cell_tensor_changed_in_place_is_followed():
    """The same cell tensor passed at every step is compared with the plan's once, and followed when it
    changes in place."""
    ref, model = (ForceStressOutput(_box(Hippo), compute_stress=True).eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2)
    cell = torch.tensor(CELL)
    s = {"pos": torch.tensor(BOX), "atomic_numbers": torch.tensor(Z), "cell": cell,
         "pbc": torch.ones(3, dtype=torch.bool)}
    for step in range(5):
        if step >= 3:
            cell.mul_(1.0005)
        got, want = stepper(s), ref(structure_to_graph(s, 6.3))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-7), (key, step)
    assert stepper._seen_cell[0] is cell


@pytest.mark.parametrize("cls", (Amoeba, Hippo))
def test_late_checks_count_failed_steps_and_recover(cls):
    """With late checks a replayed step returns before its status is read. Steps that move smoothly equal
    fresh evaluations. An atom pushed past half the skin in one step outdates the pair list: that step has
    already returned (no pair is missing yet), the next one reads it with a warning, rebuilds and equals
    the fresh evaluation. A step read as unconverged raises the solve lengths. The default stepper reads
    every status at once and builds the pair list one step later."""
    if not torch.cuda.is_available():
        pytest.skip("replays on a GPU")
    ref, model = (ForceStressOutput(_box(cls), compute_stress=True).to("cuda").eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2, late_checks=True)
    assert stepper._ahead() == 3 and StepGraph(model, 6.3)._ahead() == 2
    velocity = np.random.default_rng(8).normal(scale=0.002, size=BOX.shape)
    kick = np.zeros_like(BOX)
    for step in range(14):
        if step == 7:
            kick[0, 0] = 0.3
        s = _struct(BOX + step * velocity + kick, "cuda")
        if step == 8:
            rebuilds = stepper.rebuilds
            with pytest.warns(RuntimeWarning, match="pair list was outdated"):
                got = stepper(s)
            assert stepper.late_failures == 1 and stepper.rebuilds > rebuilds
        elif step == 11:
            assert stepper.previous is not None and stepper.plan.solve_iterations
            # the last step's status, as read on the host, says its solves did not converge
            event, host = stepper.previous[:2]
            event.synchronize()
            host[2] = 0.0
            with pytest.warns(RuntimeWarning, match="did not converge"):
                got = stepper(s)
            assert stepper.late_failures == 2 and stepper.plan.solve_iterations is None and stepper.floor
        else:
            got = stepper(s)
        want = ref(structure_to_graph(s, 6.3, device="cuda"))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-7), (key, step)
    assert stepper.replays > 0 and stepper.settle()


def test_late_checks_rebuild_behind_the_replay():
    """With late checks a cell too small for a grid finds its pair lists on the device: a rebuild takes them
    behind the step's replay and the steps equal fresh evaluations. A list that does not hold (an atom
    wrapped back into the cell while it was found) is built again in full by the next step, with a warning;
    the wrapping step itself returned before its check, a late failure."""
    if not torch.cuda.is_available():
        pytest.skip("replays on a GPU")
    ref, model = (ForceStressOutput(_box(Amoeba), compute_stress=True).to("cuda").eval() for _ in range(2))
    stepper = StepGraph(model, 6.3, skin=0.5, warmup=2, late_checks=True)
    velocity = np.random.default_rng(9).normal(scale=0.008, size=BOX.shape)
    wrap = np.zeros_like(BOX)
    for step in range(16):
        if step == 11:
            # a rebuild at this step, while atom 0 is wrapped back into the cell by one edge
            stepper.pending = True
            wrap[0, 0] = CELL[0, 0]
        s = _struct(BOX + step * velocity + wrap, "cuda")
        if step == 11:
            assert stepper.gridded and stepper.cuda_graph is not None and stepper.previous is not None
            overlapped, failures = stepper.overlapped, stepper.late_failures
            stepper(s)
            assert stepper.overlapped == overlapped + 1 and stepper.late_failures == failures
            continue
        if step == 12:
            full = stepper.full_rebuilds + stepper.recordings
            with pytest.warns(RuntimeWarning, match="did not hold"):
                got = stepper(s)
            assert stepper.late_failures == failures + 1 and not stepper.stale
            assert stepper.full_rebuilds + stepper.recordings > full
        else:
            got = stepper(s)
        want = ref(structure_to_graph(s, 6.3, device="cuda"))
        for key in ("energy", "forces", "stress"):
            assert torch.allclose(got[key], want[key].detach(), rtol=1e-9, atol=1e-7), (key, step)
        if step == 10:
            assert stepper.overlapped > 0 and stepper.late_failures == 0
    assert stepper.settle()
