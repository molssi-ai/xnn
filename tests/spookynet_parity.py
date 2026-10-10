"""Parity of xnn SpookyNet with the reference implementation (github.com/OUnke/SpookyNet).

Random-weight reference models are saved in the reference format and loaded
through :meth:`SpookyNet.from_reference_checkpoint`; energies, forces,
charges and dipoles of molecules in several charge and spin states, of a
batch and of a periodic cell are compared, first with the xnn D4 data and
then with the reference code's D4 tables loaded into the xnn term (its
covalent radii are float32-rounded), so the second comparison is of the code
alone. The physical constants are set to the reference code's values. Usage::

    python spookynet_parity.py float64|float32 [checkpoint.pth]
    python spookynet_parity.py tables          # the effect of each D4 table

With a checkpoint path the published parameters are compared instead of
random ones. The last line printed is the worst relative error of the code
comparison.
"""
import sys
import warnings
import tempfile

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.hybrid.models.spookynet import SpookyNet, electron_configurations

REF_BOHR = 0.5291772105638411
REF_HARTREE = 27.211386024367243
REF_KE = 14.399645351950548


def reference_model(dtype, seed, **kw):
    from spookynet import SpookyNet as Reference

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Reference(activation="swish", basis_functions="exp-bernstein", use_irreps=True,
                      use_nonlinear_embedding=False, zero_init=False, **kw).to(torch.float64)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=g, dtype=p.dtype)
            if name.startswith(("zbl_", "d4_")) or name.endswith("_alpha"):
                p.add_(0.05 * noise)
            elif name.endswith(".alpha"):
                p.copy_(1.0 + 0.1 * noise)
            elif name.endswith(".beta"):
                p.copy_(1.702 + 0.1 * noise)
            elif p.dim() == 2 and "element" not in name:
                p.copy_(noise / np.sqrt(p.shape[1]))
            else:
                p.copy_(0.3 * noise)
    if getattr(model, "use_d4_dispersion", False):
        model.d4_dispersion_energy._compute_refc6()
    model = model.to(dtype)
    model.eval()            # the reference eval() returns None
    return model


def convert(reference):
    with tempfile.NamedTemporaryFile(suffix=".pth") as fh:
        reference.save(fh.name)
        model = SpookyNet.from_reference_checkpoint(fh.name)
    use_reference_constants(model)
    return model


def adopt_reference_tables(model, ref, only=None):
    """Load the reference code's D4 data (Z <= 86) into the xnn term, for a code-only comparison."""
    with torch.no_grad():
        model.nuclear_embedding.electron_config.copy_(ref.nuclear_embedding.electron_config)
        model.radial_basis.log_binom.copy_(ref.radial_basis_functions.logc.flip(0))
    d4, r4 = model.dispersion, ref.d4_dispersion_energy
    if d4 is None:
        return
    n_ref = d4.refcn.shape[0]
    z = torch.arange(r4.zeff.shape[0])

    def put(name, value):
        if only is not None and name not in only:
            return
        buf = getattr(d4, name)
        buf[..., z] = value.to(buf.dtype)

    with torch.no_grad():
        put("rcov", 4.0 / 3.0 * r4.rcov)
        put("en", r4.en)
        put("zeff", r4.zeff)
        put("hardness", r4.gam)
        put("r4r2", r4.sqrt_r4r2 ** 2)          # the reference stores sqrt(Q)
        put("refq", r4.refq[:, :n_ref].t())
        put("refcn", r4.cn[:, :n_ref, 0].t())
        put("refh", r4.refh[:, :n_ref].t())
        put("ascale", r4.ascale[:, :n_ref].t())
        put("hcount", r4.hcount[:, :n_ref].t())
        put("alphaiw", r4.alphaiw[:, :n_ref, :].permute(2, 1, 0))
        if only is None or "secondary" in only:
            d4.refsys[..., z] = r4.refsys[:, :n_ref].t()
            n_sec = r4.sscale.shape[0]
            d4.sscale[:n_sec] = r4.sscale.to(d4.sscale.dtype)
            d4.secaiw[:, :n_sec] = r4.secaiw.t().to(d4.secaiw.dtype)
        if only is None or "cp_weights" in only:
            d4.cp_weights.copy_(r4.casimir_polder_weights)


def use_reference_constants(model):
    if model.repulsion is not None:
        model.repulsion.k_e = REF_KE
    if model.electrostatics is not None:
        model.electrostatics.k_e = REF_KE
    if model.dispersion is not None:
        model.dispersion.bohr, model.dispersion.hartree = REF_BOHR, REF_HARTREE


def structures():
    rng = np.random.default_rng(1)
    out = []
    for n, q, s in ((5, 0, 0), (7, 1, 0), (6, -1, 1), (4, 0, 2), (8, 2, 1)):
        pos = []
        while len(pos) < n:
            p = rng.uniform(0, 3.4, 3)
            if all(np.linalg.norm(p - o) > 0.95 for o in pos):
                pos.append(p)
        z = [[1, 6, 7, 8, 16][i % 5] for i in range(n)]
        out.append({"pos": np.array(pos), "atomic_numbers": z, "total_charge": float(q),
                    "spin_multiplicity": float(s + 1)})
    return out


def periodic_structure():
    rng = np.random.default_rng(2)
    pos = []
    while len(pos) < 6:
        p = rng.uniform(0, 5.0, 3)
        if all(np.linalg.norm(p - o) > 1.0 for o in pos):
            pos.append(p)
    return {"pos": np.array(pos), "atomic_numbers": [8, 1, 1, 6, 1, 7],
            "cell": np.diag([5.0, 5.5, 6.0]), "pbc": [True] * 3,
            "total_charge": 0.0, "spin_multiplicity": 1.0}


def reference_outputs(ref, graph, dtype):
    """Run the reference model on the pairs of an xnn graph (both conventions checked)."""
    n_graphs = graph.num_graphs
    pos = graph.pos.to(dtype).detach().clone().requires_grad_(True)
    if ref.lr_cutoff is None:
        same = graph.batch[:, None] == graph.batch[None, :]
        same &= ~torch.eye(len(graph.batch), dtype=torch.bool)
        idx_i, idx_j = same.nonzero().T
        cell = offsets = None
    elif graph.cell is None:
        idx_i, idx_j = graph.edge_index[1], graph.edge_index[0]
        cell = offsets = None
    else:
        idx_i, idx_j = graph.edge_index[1], graph.edge_index[0]
        offsets = -graph.cell_shifts.to(dtype)
        cell = graph.cell.to(dtype)
        vij = pos[idx_j] + offsets @ cell[0] - pos[idx_i]
        assert torch.allclose(vij, -graph.edge_vectors().to(dtype), atol=1e-10)
    energy, forces, dipole, f, ea, qa, e_rep, e_ele, e_vdw, *_ = ref(
        Z=graph.atomic_numbers, Q=graph.total_charge.to(dtype),
        S=graph.spin_multiplicity.to(dtype) - 1.0, R=pos, idx_i=idx_i, idx_j=idx_j, cell=cell,
        cell_offsets=offsets, num_batch=n_graphs, batch_seg=graph.batch, create_graph=False)

    def per_structure(x):
        return x.detach().new_zeros(n_graphs).index_add(0, graph.batch, x.detach())

    return {"energy": energy.detach(), "forces": forces.detach(), "charges": qa.detach(),
            "dipole": dipole.detach(), "energy_repulsion": per_structure(e_rep),
            "energy_electrostatics": per_structure(e_ele),
            "energy_dispersion": per_structure(e_vdw)}


def compare(model, ref, graph, dtype, label):
    out = ForceStressOutput(model)(graph)
    want = reference_outputs(ref, graph, dtype)
    errors = {}
    for key in ("energy", "forces", "charges", "dipole"):
        a, b = out[key].detach().to(torch.float64), want[key].to(torch.float64)
        errors[key] = float((a - b).abs().max()) / max(1.0, float(b.abs().max()))
    terms = {key.split("_")[1][:4]: float((out[key].detach().double() - want[key].double()).abs().max())
             for key in ("energy_repulsion", "energy_electrostatics", "energy_dispersion")}
    print(label, " ".join(f"{k} {v:.2e}" for k, v in errors.items()),
          "| abs terms", " ".join(f"{k} {v:.1e}" for k, v in terms.items()))
    return max(errors.values())


def check_tables(model, ref):
    """The element descriptors and the D4 reference C6 of both codes."""
    config = torch.as_tensor(electron_configurations())
    d_cfg = float((config - ref.nuclear_embedding.electron_config.double().cpu()).abs().max())
    print("electron configurations", f"{d_cfg:.2e}")
    if model.dispersion is None:
        return d_cfg
    d4 = model.dispersion
    with torch.no_grad():
        alpha = d4.reference_polarizabilities(torch.nn.functional.softplus(d4._scaleq))
        z = torch.arange(1, 87)
        a = alpha[:, :, z].permute(2, 1, 0).double()                         # (Z, ref, 23)
        c6 = (3.0 / np.pi) * torch.einsum("aik,bjk,k->abij", a, a, d4.cp_weights.double())
        want = ref.d4_dispersion_energy.refc6[1:87, 1:87].double()
        n = min(c6.shape[-1], want.shape[-1])
        nref = d4.nref[z]
        slots = torch.arange(n)
        valid = slots[None, :] < nref[:, None]                                 # (Z, ref)
        mask = valid[:, None, :, None] & valid[None, :, None, :]
        diff = torch.where(mask, c6[..., :n, :n] - want[..., :n, :n], torch.zeros(()))
        d_c6 = float(diff.abs().max()) / float(want.abs().max())
    print("D4 reference C6", f"{d_c6:.2e}")
    r4 = ref.d4_dispersion_energy
    pairs = {"rcov": (d4.rcov[z], 4.0 / 3.0 * r4.rcov[z]), "en": (d4.en[z], r4.en[z]),
             "zeff": (d4.zeff[z], r4.zeff[z]), "hardness": (d4.hardness[z], r4.gam[z]),
             "r4r2": (d4.r4r2[z], r4.sqrt_r4r2[z] ** 2), "cp_weights": (d4.cp_weights, r4.casimir_polder_weights),
             "refq": (d4.refq[:, z].t(), r4.refq[z, :d4.refq.shape[0]]),
             "refcn": (d4.refcn[:, z].t(), r4.cn[z, :d4.refcn.shape[0], 0])}
    for name, (a, b) in pairs.items():
        a, b = a.double(), b.double()
        if name in ("refq", "refcn"):
            a = torch.where(valid, a, torch.zeros_like(a))
            b = torch.where(valid, b, torch.zeros_like(b))
        diff = (a - b).abs()
        worst_z = (int(z[diff.reshape(len(z), -1).max(1).values.argmax()])
                   if diff.shape[0] == len(z) else 0)
        print(f"  D4 table {name}: max abs diff {float(diff.max()):.2e} (Z={worst_z})")
    return max(d_cfg, d_c6)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    mode = sys.argv[1]
    dtype = {"float64": torch.float64, "float32": torch.float32}[mode.replace("tables", "") or "float64"]
    torch.set_default_dtype(dtype)
    if mode.startswith("tables"):
        kw = dict(num_features=16, num_basis_functions=8, num_modules=2, cutoff=4.0)
        ref = reference_model(dtype, seed=9, **kw)
        g = structure_to_graph(structures()[1], 4.0)
        model = convert(ref).to(dtype)
        for z in (1, 6, 8, 16, 54, 55, 56):
            print(f"Z={z} r4r2 xnn {float(model.dispersion.r4r2[z]):.6f} reference "
                  f"{float(ref.d4_dispersion_energy.sqrt_r4r2[z]):.6f}")
        for name in ("rcov", "en", "zeff", "hardness", "r4r2", "refq", "refcn", "refh",
                     "ascale", "hcount", "alphaiw", "secondary", "cp_weights"):
            model = convert(ref).to(dtype)
            adopt_reference_tables(model, ref, only={name})
            compare(model, ref, g, dtype, f"only {name}:")
        return
    worst = 0.0
    if len(sys.argv) > 2:
        from spookynet import SpookyNet as Reference
        ref = Reference(load_from=sys.argv[2]).to(dtype)
        ref.eval()
        model = SpookyNet.from_reference_checkpoint(sys.argv[2]).to(dtype)
        use_reference_constants(model)
        worst = check_tables(model, ref)
        graphs = [structure_to_graph(s, model.cutoff) for s in structures()[:3]]
        print("checkpoint: xnn tables")
        for i, g in enumerate(graphs):
            compare(model, ref, g, dtype, f"checkpoint structure {i}")
        adopt_reference_tables(model, ref)
        print("checkpoint: the tables stored in the checkpoint")
        worst = 0.0
        for i, g in enumerate(graphs):
            worst = max(worst, compare(model, ref, g, dtype, f"checkpoint structure {i}"))
        print(worst)
        return
    cases = [
        ("molecules", dict(num_features=16, num_basis_functions=8, num_modules=2, cutoff=4.0,
                           exp_weighting=True)),
        ("lr_cutoff", dict(num_features=12, num_basis_functions=6, num_modules=2, cutoff=3.5,
                           lr_cutoff=6.0, exp_weighting=False)),
    ]
    for name, kw in cases:
        ref = reference_model(dtype, seed=len(name), **kw)
        model = convert(ref).to(dtype)
        check_tables(model, ref)
        graphs = [structure_to_graph(s, model.cutoff) for s in structures()]
        print(f"{name}: xnn D4 data (dftd4 4.2.0)")
        for i, g in enumerate(graphs):
            compare(model, ref, g, dtype, f"{name} structure {i}")
        adopt_reference_tables(model, ref)
        print(f"{name}: the reference code's D4 data")
        for i, g in enumerate(graphs):
            worst = max(worst, compare(model, ref, g, dtype, f"{name} structure {i}"))
        worst = max(worst, compare(model, ref, collate(graphs), dtype, f"{name} batch"))
        if model.lr_cutoff is not None:
            g = structure_to_graph(periodic_structure(), model.cutoff)
            worst = max(worst, compare(model, ref, g, dtype, f"{name} periodic"))
    print(worst)


if __name__ == "__main__":
    main()
