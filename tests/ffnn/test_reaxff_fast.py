"""The ReaxFF fast path (the kernels of reaxff_kernels, their backward generated) against the eager terms:
energies of every term, forces, charges, stress-free first derivatives in the trained parameters, on periodic
boxes of methanol / water copies and on clusters, for the CHO template in the ReaxFF-nn conventions (classical
and neural bond orders) and two published fields in the LAMMPS conventions (the 2008 combustion field, the Fedkin
electrolyte field with its inner wall), float64 and float32; on a GPU and, as C++ loops (xnn.common.cpu_kernels), on
the CPU.
"""
import numpy as np
import pytest
import torch

from xnn.common import cpu_kernels
from xnn.common.data import structure_to_graph
from xnn.ffnn.models import ReaxFF, template_library

DEVICES = [pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")),
           pytest.param("cpu", marks=pytest.mark.skipif(not cpu_kernels.available(), reason="no C++ compiler"))]


@pytest.fixture(params=DEVICES)
def device(request, monkeypatch):
    """The device of a test; on the CPU the model takes the kernels (its default there is the eager terms)."""
    if request.param == "cpu":
        import xnn.ffnn.models.classical_fast as cf
        monkeypatch.setattr(cf, "fast_supported", lambda d, dtype: dtype in (torch.float32, torch.float64))
    return request.param

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
def test_the_nonbonded_kernel_matches_the_eager_terms(case, periodic, device):
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
        m = model.to(device=device, dtype=dtype).eval()
        results = []
        for fast in (False, True):
            m.use_fast = fast
            g = structure_to_graph(s, m.cutoff, device=device)
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


def test_the_conjugate_gradient_charges_on_the_kernel_match(monkeypatch, device):
    """Large structures solve the EEM charges by conjugate gradients; the fast path's products on one kernel give
    the eager solve's charges, energies and forces."""
    import xnn.ffnn.models.reaxff as rx
    monkeypatch.setattr(rx, "EEM_DENSE_MAX", 0)
    pos, z, L = _box(METHANOL, 3, 6.5)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z),
         "cell": torch.eye(3, dtype=torch.float64) * L, "pbc": torch.ones(3, dtype=torch.bool)}
    m = ReaxFF("CHO_cho_2008", conventions="lammps").to(device=device, dtype=torch.float64).eval()
    res = []
    for fast in (False, True):
        m.use_fast = fast
        g = structure_to_graph(s, m.cutoff, device=device)
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        out = m(g)
        (f,) = torch.autograd.grad(-out["energy"].sum(), p)
        res.append((out["energy"].detach(), out["charges"].detach(), f))
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-10), float((a - b).abs().max())


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_the_kernels_give_the_eager_stress(case, device):
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
        m = model.to(device=device, dtype=dtype).eval()
        g = structure_to_graph(s, m.cutoff, device=device)
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


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_a_force_loss_trains_through_the_kernels(case, device):
    """In training the kernels are differentiable twice: the gradients of an energy and force loss in the
    positions and every trained parameter (network weights included) are the eager terms' in float64; in
    float32 both are as far from the float64 gradients (these second derivatives cancel)."""
    name, build, mol = case
    pos, z, L = _box(mol, 2, 6.5)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z), "cell": torch.eye(3, dtype=torch.float64) * L,
         "pbc": torch.ones(3, dtype=torch.bool)}
    model = build()
    for p in model.params.values():
        p.requires_grad_(p.is_floating_point() and p.dim() > 0)
    model.params["vdw1"].requires_grad_(False)

    def gradients(dtype, fast):
        m = model.to(device=device, dtype=dtype).train()
        m.use_fast = fast
        g = structure_to_graph(s, m.cutoff, device=device)
        g.compute_dtype = dtype
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        e = m(g)["energy"].sum()
        (f,) = torch.autograd.grad(-e, p, create_graph=True)
        loss = 1e-3 * e * e + (f * f).sum()
        params = [v for v in m.parameters() if v.requires_grad]
        out = torch.autograd.grad(loss, [p] + params, allow_unused=True)
        return [None if x is None else x.detach().double() for x in out]

    ref = gradients(torch.float64, False)
    fast64 = gradients(torch.float64, True)
    eager32, fast32 = gradients(torch.float32, False), gradients(torch.float32, True)
    for j, (r, a, b, c) in enumerate(zip(ref, fast64, eager32, fast32)):
        if r is None and a is None:
            continue
        r = torch.zeros_like(a) if r is None else r
        a, b, c = (torch.zeros_like(r) if x is None else x for x in (a, b, c))
        scale = float(r.abs().max()) or 1.0
        assert float((a - r).abs().max()) <= 1e-9 * scale, (name, "float64", j, float((a - r).abs().max()), scale)
        err32 = float((b - r).abs().max())
        # float32 round-off of a second derivative through the dihedral's cancelling law of cosines (the
        # kernels contract to fused multiply-adds) reaches 1e-5 relative, and the atomic sums' order varies
        # from run to run
        assert float((c - r).abs().max()) <= max(5.0 * err32, 1e-4 * scale), (name, "float32", j, err32,
                                                                               float((c - r).abs().max()), scale)


@pytest.mark.parametrize("shielded", [False, True])
def test_the_inner_wall_forms_on_the_kernels(shielded, device):
    """A library with an inner wall (LAMMPS's van der Waals form 3, or form 2 without the shielding): the
    kernels' van der Waals energies and forces are the eager terms'."""
    import copy

    from xnn.ffnn.models import read_ffield
    lib = copy.deepcopy(read_ffield("CHO_cho_2008"))
    for sp in lib.spec:
        lib.p.update({f"rcore2_{sp}": 1.6, f"acore2_{sp}": 12.0, f"ecore2_{sp}": 0.2})
        if not shielded:
            lib.p[f"gammaw_{sp}"] = 0.4
    m = ReaxFF(lib, conventions="lammps").to(device=device, dtype=torch.float64).eval()
    assert m.vdw_type == (3 if shielded else 2)
    pos, z, L = _box(METHANOL, 2, 6.5)
    s = {"pos": torch.tensor(pos % L), "atomic_numbers": torch.tensor(z), "cell": torch.eye(3, dtype=torch.float64) * L,
         "pbc": torch.ones(3, dtype=torch.bool)}
    res = []
    for fast in (False, True):
        m.use_fast = fast
        g = structure_to_graph(s, m.cutoff, device=device)
        p = g.pos.detach().clone().requires_grad_(True)
        g.pos = p
        out = m(g)
        (f,) = torch.autograd.grad(-out["energy"].sum(), p)
        res.append((out["e_vdw"].detach(), out["energy"].detach(), f))
    for a, b in zip(*res):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-10), float((a - b).abs().max())



def test_the_incoming_gradients_are_differentiated_through_the_kernels(device):
    """A force loss of an energy scaled by a trained factor: its gradient in the factor (through the incoming
    gradients of every term kernel's energies, the nonbonded one included) is the force loss at factor one,
    as in the eager terms."""
    pos, z, _ = _box(METHANOL, 2, 6.5)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(z)}
    model = ReaxFF("CHO_cho_2008", conventions="lammps").to(device=device, dtype=torch.float64).train()
    w = torch.tensor(np.random.default_rng(5).normal(size=pos.shape), device=device)
    results = []
    for fast in (False, True):
        model.use_fast = fast
        g = structure_to_graph(s, model.cutoff, device=device)
        g.pos = g.pos.detach().clone().requires_grad_(True)
        factor = torch.ones((), dtype=torch.float64, device=device, requires_grad=True)
        energy = factor * model(g)["energy"].sum()
        (forces,) = torch.autograd.grad(-energy, g.pos, create_graph=True)
        loss = (forces * w).sum()
        (grad,) = torch.autograd.grad(loss, factor)
        assert torch.allclose(grad, loss.detach(), rtol=1e-10, atol=1e-10)
        results.append(grad)
    assert torch.allclose(results[0], results[1], rtol=1e-9, atol=1e-9)


def test_the_compressed_row_products_match_a_dense_matrix(device):
    """The charge matrix's product by rows (the stepper's), its values per entry or per pair through an index,
    is the dense matrix's product with two right-hand sides."""
    from xnn.ffnn.models.reaxff_kernels import eem_rows
    rng = np.random.default_rng(4)
    N, P = 9, 20
    i, k = rng.integers(0, N, P), rng.integers(0, N, P)
    keep = i != k
    i, k = i[keep], k[keep]
    g_pair = torch.tensor(rng.normal(size=len(i)), device=device)
    dense = torch.zeros(N, N, dtype=torch.float64, device=device)
    dense.index_put_((torch.tensor(i, device=device), torch.tensor(k, device=device)), g_pair, accumulate=True)
    dense = dense + dense.T
    rows, cols = np.concatenate([i, k]), np.concatenate([k, i])
    order = np.argsort(rows, kind="stable")
    ptr = torch.tensor(np.searchsorted(rows[order], np.arange(N + 1)), dtype=torch.int32, device=device)
    col = torch.tensor(cols[order], dtype=torch.int32, device=device)
    pair = torch.tensor(order % len(i), dtype=torch.int32, device=device)
    x = torch.tensor(rng.normal(size=(N, 2)), device=device)
    for product in (eem_rows(ptr, col, g_pair[pair.long()]), eem_rows(ptr, col, g_pair, pair)):
        assert torch.allclose(product(x), dense @ x, rtol=1e-12, atol=1e-12)
