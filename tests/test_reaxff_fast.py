"""The ReaxFF fast path (the kernels of reaxff_kernels, their backward generated) against the eager terms:
energies of every term, forces, charges, stress-free first derivatives in the trained parameters, on periodic
boxes of methanol / water copies and on clusters, for the CHO template in the ReaxFF-nn conventions (classical
and neural bond orders) and two published fields in the LAMMPS conventions (the 2008 combustion field, the Fedkin
electrolyte field with its inner wall), float64 and float32. GPU only.
"""
import numpy as np
import pytest
import torch

from xnn.common.data import structure_to_graph
from xnn.ffnn.models import ReaxFF, template_library

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels run on a GPU")

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


CASES = [("template-nn-conventions", lambda: ReaxFF(template_library(["C", "H", "O"], nn=False), nn=False,
                                                    conventions="reaxff-nn"), METHANOL),
         ("template-neural", lambda: ReaxFF(template_library(["C", "H", "O"], nn=True), nn=True,
                                            conventions="reaxff-nn"), METHANOL),
         ("cho-2008", lambda: ReaxFF("CHO_cho_2008", conventions="lammps"), METHANOL),
         ("fedkin-water", lambda: ReaxFF("CHOLiNaKCsFClI_Fedkin_2019", species=["H", "O"],
                                         conventions="lammps"), WATER)]


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_the_nonbonded_kernel_matches_the_eager_terms(case, periodic):
    name, build, mol = case
    pos, z, L = _box(mol, 3, 6.5)
    s = {"pos": torch.tensor(pos if not periodic else pos % L), "atomic_numbers": torch.tensor(z)}
    if periodic:
        s.update(cell=torch.eye(3, dtype=torch.float64) * L, pbc=torch.ones(3, dtype=torch.bool))
    model = build()
    for p in model.params.values():
        p.requires_grad_(p.is_floating_point() and p.dim() > 0)
    model.params["vdw1"].requires_grad_(False)
    for dtype, rtol in ((torch.float64, 1e-10), (torch.float32, 3e-5)):
        m = model.to(device="cuda", dtype=dtype).eval()
        results = []
        for fast in (False, True):
            m.use_fast = fast
            g = structure_to_graph(s, m.cutoff, device="cuda")
            g.compute_dtype = dtype
            p = g.pos.detach().clone().requires_grad_(True)
            g.pos = p
            out = m(g)
            params = [v for v in m.params.values() if v.requires_grad]
            grads = torch.autograd.grad(out["energy"].sum(), [p] + params, allow_unused=True)
            results.append((out, grads))
        (ref, g_ref), (got, g_got) = results
        for key in ref:
            if key.startswith("e_") or key in ("energy", "charges"):
                a, b = ref[key].detach().double(), got[key].detach().double()
                scale = float(a.abs().max()) or 1.0
                assert float((a - b).abs().max()) <= rtol * scale, (name, str(dtype), key, float((a - b).abs().max()))
        for j, (a, b) in enumerate(zip(g_ref, g_got)):
            if a is None and b is None:
                continue
            a = torch.zeros_like(b) if a is None else a
            b = torch.zeros_like(a) if b is None else b
            a, b = a.double(), b.double()
            scale = float(a.abs().max()) or 1.0
            # float32 parameter gradients sum thousands of terms in another order
            gtol = rtol if dtype == torch.float64 or j == 0 else 3e-4
            assert float((a - b).abs().max()) <= gtol * scale, (name, str(dtype), "grad", j,
                                                               float((a - b).abs().max()), scale)


def test_the_conjugate_gradient_charges_on_the_kernel_match(monkeypatch):
    """Large structures solve the EEM charges by conjugate gradients; the fast path's products on one kernel give
    the eager solve's charges, energies and forces."""
    import xnn.ffnn.models.reaxff as rx
    monkeypatch.setattr(rx, "EEM_DENSE_MAX", 0)
    pos, z, L = _box(METHANOL, 3, 6.5)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    m = ReaxFF("CHO_cho_2008", conventions="lammps").to(device="cuda", dtype=torch.float64).eval()
    res = []
    for fast in (False, True):
        m.use_fast = fast
        g = structure_to_graph(s, m.cutoff, device="cuda")
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        out = m(g)
        (f,) = torch.autograd.grad(-out["energy"].sum(), p)
        res.append((out["energy"].detach(), out["charges"].detach(), f))
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-10), float((a - b).abs().max())


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_the_kernels_give_the_eager_stress(case):
    """The stress of a periodic box (the cell adjoint of the pair kernel, the cell through the bond vectors of
    the others) is the eager terms'."""
    from xnn.common.models import ForceStressOutput
    name, build, mol = case
    pos, z, L = _box(mol, 3, 6.5)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z),
         "cell": torch.eye(3, dtype=torch.float64) * L + 0.3 * torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                                                                              [0.0, 0.0, 0.0]], dtype=torch.float64),
         "pbc": torch.ones(3, dtype=torch.bool)}
    model = build()
    for dtype, rtol in ((torch.float64, 1e-10), (torch.float32, 3e-5)):
        m = model.to(device="cuda", dtype=dtype).eval()
        g = structure_to_graph(s, m.cutoff, device="cuda")
        g.compute_dtype = dtype
        w = ForceStressOutput(m, compute_stress=True).eval()
        res = []
        for fast in (False, True):
            m.use_fast = fast
            out = w(g)
            res.append({k: out[k].detach().double() for k in ("energy", "forces", "stress")})
        for key in ("energy", "forces", "stress"):
            a, b = res[0][key], res[1][key]
            scale = float(a.abs().max()) or 1.0
            assert float((a - b).abs().max()) <= rtol * scale, (name, str(dtype), key, float((a - b).abs().max()))
