"""The ReaxFF MD stepper (candidate lists with a skin, flagged terms on the kernels, warm-started charges, CUDA
graph replays) against the model's eager evaluation along a short trajectory: energies, forces and charges
at every step, through the recorded graph and a list rebuild, periodic boxes and a cluster, float64 and
float32; on a GPU and, without graphs and with the kernels as C++ loops, on the CPU.
"""
import numpy as np
import pytest
import torch

from xnn.common import cpu_kernels
from xnn.common.data import structure_to_graph
from xnn.ffnn.models import ReaxFF, ReaxFFStepper

DEVICES = [pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")),
           pytest.param("cpu", marks=pytest.mark.skipif(not cpu_kernels.available(), reason="no C++ compiler"))]

METHANOL = (np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.60, 0.90, 0.0], [-0.60, -0.50, 0.80],
                      [-0.60, -0.50, -0.80], [1.80, 0.85, 0.0]]), [6, 8, 1, 1, 1, 1])
WATER = (np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]), [8, 1, 1])


def _box(mol, n_side, spacing, seed=0):
    rng = np.random.default_rng(seed)
    p1, z1 = mol
    p1 = p1 - p1.mean(0)
    out = []
    for site in np.ndindex(n_side, n_side, n_side):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        out.append(p1 @ (q * np.sign(np.diag(r))).T + (np.array(site) + 0.5) * spacing)
    pos = np.concatenate(out) + 0.03 * rng.normal(size=(len(out) * len(p1), 3))
    return pos, list(z1) * len(out), n_side * spacing


CASES = [("cho-2008", lambda: ReaxFF("CHO_cho_2008", conventions="lammps"), METHANOL, True),
         ("fedkin-water", lambda: ReaxFF("CHOLiNaKCsFClI_Fedkin_2019", species=["H", "O"], conventions="lammps"),
          WATER, True),
         ("cho-2008-cluster", lambda: ReaxFF("CHO_cho_2008", conventions="lammps"), METHANOL, False)]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("device", DEVICES)
def test_the_stepper_follows_the_model(case, dtype, device):
    name, build, mol, periodic = case
    pos0, z, L = _box(mol, 3, 6.5)
    if periodic:
        pos0 = pos0 % L
    rng = np.random.default_rng(1)
    step = rng.normal(size=pos0.shape)
    step *= 0.04 / np.linalg.norm(step, axis=1, keepdims=True)
    model = build().to(device=device, dtype=dtype).eval()
    model.use_fast = False
    stepper = ReaxFFStepper(model)
    rtol = 1e-9 if dtype == torch.float64 else 3e-5
    for t in range(12):
        s = {"pos": torch.tensor(pos0 + t * step), "atomic_numbers": torch.tensor(z)}
        if periodic:
            s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
        got = stepper(s)
        g = structure_to_graph(s, model.cutoff, device=device)
        g.compute_dtype = dtype
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        out = model(g)
        forces = -torch.autograd.grad(out["energy"].sum(), p)[0]
        ref = {"energy": out["energy"].detach(), "forces": forces, "charges": out["charges"].detach()}
        for key in ("energy", "forces", "charges"):
            a, b = ref[key].double().reshape(-1), got[key].double().reshape(-1)
            scale = float(a.abs().max()) or 1.0
            # float32 charges: conjugate gradients to a 1e-6 residual against a dense float32 solve, both off by
            # the condition number times the precision
            tol = 1e-4 if key == "charges" and dtype == torch.float32 else rtol
            assert float((a - b).abs().max()) <= tol * scale, (name, str(dtype), t, key,
                                                              float((a - b).abs().max()), scale)
    assert stepper.rebuilds > 1
    if device == "cuda":
        assert stepper.replays > 0 and stepper.refills >= 1


@pytest.mark.skipif(not cpu_kernels.available(), reason="no C++ compiler")
def test_the_cpu_stepper_with_the_one_pass_search_follows_the_model():
    """A cell over twice cutoff plus skin wide: the CPU searches every pair in one pass, its bond and
    hydrogen-bond candidates from the rows of that list; energies, forces and charges of the model through a
    rebuild."""
    pos0, z, L = _box(WATER, 3, 6.5)
    pos0 = pos0 % L
    rng = np.random.default_rng(2)
    step = rng.normal(size=pos0.shape)
    step *= 0.05 / np.linalg.norm(step, axis=1, keepdims=True)
    model = ReaxFF("CHOLiNaKCsFClI_Fedkin_2019", species=["H", "O"], conventions="lammps", vdw_cutoff=8.0)
    model = model.double().eval()
    model.use_fast = False
    stepper = ReaxFFStepper(model)
    for t in range(5):
        s = {"pos": torch.tensor(pos0 + t * step), "atomic_numbers": torch.tensor(z),
             "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
        got = stepper(s)
        g = structure_to_graph(s, model.cutoff)
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        out = model(g)
        forces = -torch.autograd.grad(out["energy"].sum(), p)[0]
        ref = {"energy": out["energy"].detach(), "forces": forces, "charges": out["charges"].detach()}
        for key in ("energy", "forces", "charges"):
            a, b = ref[key].double().reshape(-1), got[key].double().reshape(-1)
            scale = float(a.abs().max()) or 1.0
            assert float((a - b).abs().max()) <= 1e-9 * scale, (t, key, float((a - b).abs().max()), scale)
    assert stepper.rebuilds > 1 and stepper.refills >= 1
    assert int(stepper.lists["hbonds"]["n"]) > 0


def test_the_stepper_refuses_other_fields_trained_parameters_and_small_cells():
    """The ReaxFF-nn conventions, a parameter requiring a gradient and a cell no wider than cutoff plus skin
    are refused."""
    from xnn.ffnn.models import template_library
    with pytest.raises(ValueError, match="LAMMPS conventions"):
        ReaxFFStepper(ReaxFF(template_library(["C", "H", "O"], nn=False), nn=False, conventions="reaxff-nn"))
    model = ReaxFF("CHO_cho_2008", conventions="lammps")
    model.params["val"].requires_grad_(True)
    with pytest.raises(ValueError, match="fixed parameters"):
        ReaxFFStepper(model)
    model.params["val"].requires_grad_(False)
    pos, z, L = _box(WATER, 1, 6.5)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    with pytest.raises(ValueError, match="width"):
        ReaxFFStepper(model)(s)
