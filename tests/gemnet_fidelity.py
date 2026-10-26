"""Fidelity of xnn GemNet to the reference implementation (TUM-DAML/gemnet_pytorch).

``python gemnet_fidelity.py <checkout> random|pretrained`` prints per-structure
errors and, on its last line, the worst relative error (energies relative to
``max(|E|, 1)``, forces relative to ``max |F|``). ``random`` transplants random
weights and scaling factors of all four variants in float64; ``pretrained``
transplants the published GemNet-Q and GemNet-T weights and compares them on COLL
test molecules in float64 (last line) and float32. The reference code runs
unmodified; ``setup`` only makes it importable here (stand-ins for
``torch_scatter.scatter`` and ``numba.njit``, ``np.math``, float64 Bessel zeros).
"""
import json
import math
import os
import sys
import tempfile
import types

import numpy as np
import torch


def _scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """The two reductions of ``torch_scatter.scatter`` the reference uses (dim 0)."""
    if dim != 0:
        raise NotImplementedError(dim)
    size = int(index.max()) + 1 if dim_size is None else int(dim_size)
    out = src.new_zeros((size,) + tuple(src.shape[1:])).index_add(0, index, src)
    if reduce == "mean":
        count = torch.bincount(index, minlength=size).clamp(min=1).to(src.dtype)
        out = out / count.view(-1, *([1] * (src.dim() - 1)))
    return out


def _jn_zeros_float64(n, k):
    """First ``k`` zeros of j_0 .. j_{n-1} by Brent's method between interlacing zeros."""
    from scipy.optimize import brentq
    from scipy.special import spherical_jn

    zeros = np.zeros((n, k))
    brackets = np.arange(1, k + n) * np.pi
    zeros[0] = brackets[:k]
    for l in range(1, n):
        brackets = np.array([brentq(lambda x: spherical_jn(l, x), lo, hi, xtol=1e-15)   # noqa: B023
                             for lo, hi in zip(brackets[:-1], brackets[1:])])
        zeros[l] = brackets[:k]
    return zeros


def setup(upstream, float64_zeros=True):
    """Import the reference ``GemNet`` and ``DataContainer`` from a checkout."""
    if "torch_scatter" not in sys.modules:
        try:
            import torch_scatter  # noqa: F401
        except ImportError:
            mod = types.ModuleType("torch_scatter")
            mod.scatter = _scatter
            sys.modules["torch_scatter"] = mod
    try:
        import numba  # noqa: F401
    except ImportError:
        mod = types.ModuleType("numba")
        mod.njit = lambda *args, **kw: args[0] if args and callable(args[0]) else (lambda f: f)
        sys.modules["numba"] = mod
    if not hasattr(np, "math"):
        np.math = math
    if not hasattr(np, "bool"):
        np.bool = bool
    sys.path.insert(0, upstream)
    import gemnet.model.layers.basis_utils as basis_utils
    if float64_zeros:
        basis_utils.Jn_zeros = _jn_zeros_float64
    from gemnet.model.gemnet import GemNet as RefGemNet
    from gemnet.training.data_container import DataContainer
    return RefGemNet, DataContainer


def xnn_kwargs(kw):
    """xnn GemNet options for a reference ``model_kwargs`` dict."""
    return dict(n_features=kw["emb_size_atom"], n_edge_features=kw["emb_size_edge"],
                n_interactions=kw["num_blocks"], n_rbf=kw["num_radial"],
                n_spherical=kw["num_spherical"], cutoff=kw["cutoff"],
                int_cutoff=kw["int_cutoff"], p=kw["envelope_exponent"],
                n_triplet_features=kw["emb_size_trip"], n_quadruplet_features=kw["emb_size_quad"],
                n_rbf_features=kw["emb_size_rbf"], n_cbf_features=kw["emb_size_cbf"],
                n_sbf_features=kw["emb_size_sbf"], n_bilinear_triplet=kw["emb_size_bil_trip"],
                n_bilinear_quadruplet=kw["emb_size_bil_quad"],
                n_before_skip=kw["num_before_skip"], n_after_skip=kw["num_after_skip"],
                n_concat=kw["num_concat"], n_atom_residual=kw["num_atom"],
                quadruplets=not kw["triplets_only"], direct_forces=kw.get("direct_forces", False),
                forces_coupled=kw.get("forces_coupled", False))


@torch.no_grad()
def transplant(ref, model):
    """Copy the reference model's weights and scaling factors into an xnn GemNet."""
    L, N = model.n_spherical, model.n_rbf

    def dense(dst, src):
        dst.weight.copy_(src.weight)

    def residuals(dst, src):
        for d, s in zip(dst, src, strict=True):
            dense(d.lin1, s.dense_mlp[0])
            dense(d.lin2, s.dense_mlp[1])

    def scale(dst, src):
        dst.scale_factor.fill_(float(src.scale_factor))

    def atom_block(dst, src):
        dense(dst.lin_rbf, src.dense_rbf)
        dense(dst.dense, src.layers[0])
        residuals(dst.residuals, src.layers[1:])
        scale(dst.scale_sum, src.scale_sum)

    model.rbf.freqs.copy_(ref.rbf_basis.frequencies)
    model.atom_embedding.weight.zero_()
    model.atom_embedding.weight[1:94].copy_(ref.atom_emb.embeddings.weight)   # rows by Z - 1
    dense(model.edge_embedding.dense, ref.edge_emb.dense)
    for name in ("rbf3", "rbf_h", "rbf_out"):
        dense(getattr(model, "lin_" + name), getattr(ref, "mlp_" + name))
    w = ref.mlp_cbf3.weight                                                   # (L, N, J)
    model.lin_cbf3.weight.copy_(w.permute(2, 0, 1).reshape(w.shape[2], L * N))
    if model.quadruplets:
        dense(model.lin_rbf4, ref.mlp_rbf4)
        dense(model.lin_cbf4, ref.mlp_cbf4)
        w = ref.mlp_sbf4.weight                                               # (L^2, N, J)
        # the reference orders each degree's orders as m = 0, 1, .., l, -l, .., -1
        perm = [l * l + (m if m >= 0 else 2 * l + 1 + m)
                for l in range(L) for m in range(-l, l + 1)]
        model.lin_sbf4.weight.copy_(w[perm].permute(2, 0, 1).reshape(w.shape[2], L * L * N))
    for blk, rb in zip(model.interactions, ref.int_blocks, strict=True):
        dense(blk.dense_ca, rb.dense_ca)
        pairs = [(blk.trip, rb.trip_interaction, "dense_ba", "mlp_cbf")]
        if blk.quad is not None:
            pairs.append((blk.quad, rb.quad_interaction, "dense_db", "mlp_sbf"))
        for mine, theirs, first, bil in pairs:
            dense(getattr(mine, first), getattr(theirs, first))
            dense(mine.lin_rbf, theirs.mlp_rbf)
            dense(mine.down, theirs.down_projection)
            dense(mine.up_ca, theirs.up_projection_ca)
            dense(mine.up_ac, theirs.up_projection_ac)
            mine.bilinear.copy_(getattr(theirs, bil).weight.permute(1, 0, 2))  # (F, J, O) -> (J, F, O)
            scale(mine.scale_rbf, theirs.scale_rbf)
        scale(blk.trip.scale_cbf_sum, rb.trip_interaction.scale_cbf_sum)
        if blk.quad is not None:
            dense(blk.quad.lin_cbf, rb.quad_interaction.mlp_cbf)
            scale(blk.quad.scale_cbf, rb.quad_interaction.scale_cbf)
            scale(blk.quad.scale_sbf_sum, rb.quad_interaction.scale_sbf_sum)
        residuals(blk.before_skip, rb.layers_before_skip)
        residuals(blk.after_skip, rb.layers_after_skip)
        residuals(blk.residual_m, rb.residual_m)
        atom_block(blk.atom_update, rb.atom_update)
        dense(blk.concat.dense, rb.concat_layer.dense)
    for out, rb in zip(model.outputs, ref.out_blocks, strict=True):
        atom_block(out.atom, rb)
        dense(out.final, rb.out_energy)
        if out.forces is not None:
            scale(out.forces.scale_rbf, rb.scale_rbf)
            dense(out.forces.dense, rb.seq_forces[0])
            residuals(out.forces.residuals, rb.seq_forces[1:])
            dense(out.forces.final, rb.out_forces)
    model.scaling_fitted.fill_(True)


def container(DataContainer, molecules, kw):
    """A reference DataContainer over in-memory molecules (labels zero)."""
    path = os.path.join(tempfile.mkdtemp(), "molecules.npz")
    np.savez(path, N=np.array([len(z) for z, _ in molecules]),
             Z=np.concatenate([z for z, _ in molecules]),
             R=np.concatenate([r for _, r in molecules]),
             F=np.zeros((sum(len(z) for z, _ in molecules), 3)), E=np.zeros(len(molecules)))
    return DataContainer(path, cutoff=kw["cutoff"], int_cutoff=kw["int_cutoff"],
                         triplets_only=kw["triplets_only"])


def compare(ref, model, data, molecules, dtype, label):
    """Worst relative energy/force error of xnn vs the reference over ``molecules``."""
    from xnn.common.data import structure_to_graph
    from xnn.common.models import ForceStressOutput

    wrapped = ForceStressOutput(model)
    worst = 0.0
    for i, (z, r) in enumerate(molecules):
        inputs = data[[i]]
        inputs["R"] = torch.tensor(r, dtype=dtype)
        e_ref, f_ref = ref(inputs)
        e_ref, f_ref = float(e_ref.reshape(-1)[0]), f_ref.detach().reshape(-1, 3).double()
        g = structure_to_graph({"pos": np.asarray(r, dtype=np.float64), "atomic_numbers": z},
                               model.cutoff)
        if dtype == torch.float32:
            g.pos = g.pos.float()
        out = wrapped(g)
        e, f = float(out["energy"]), out["forces"].detach().double()
        de = abs(e - e_ref) / max(abs(e_ref), 1.0)
        df = float((f - f_ref).abs().max() / f_ref.abs().max().clamp(min=1e-12))
        worst = max(worst, de, df)
        print(f"{label} mol {i:2d} ({len(z):2d} atoms): E {e_ref:+.10f}  dE_rel {de:.2e}  "
              f"dF_rel {df:.2e}")
    return worst


def random_molecules(n_mol=3, seed=0):
    """Random molecules with pairs inside and beyond the embedding cutoff."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_mol):
        n = int(rng.integers(6, 11))
        while True:
            r = rng.uniform(0, 7.0, (n, 3))
            d = np.linalg.norm(r[:, None] - r[None], axis=-1) + 10 * np.eye(n)
            if d.min() > 0.9:
                break
        out.append((rng.choice([1, 6, 7, 8], n), r))
    return out


def coll_molecules(upstream, count, start=0):
    path = os.path.join(upstream, "data", "coll_v1.2_test.npz")
    if not os.path.exists(path):
        return []
    with np.load(path) as f:
        n, z, r = f["N"], f["Z"], f["R"]
    offsets = np.concatenate([[0], np.cumsum(n)])
    return [(z[offsets[i]:offsets[i + 1]], r[offsets[i]:offsets[i + 1]].astype(np.float64))
            for i in range(start, start + count)]


def run_random(upstream):
    from xnn.gnn.models.gemnet import GemNet

    RefGemNet, DataContainer = setup(upstream, float64_zeros=True)
    torch.set_default_dtype(torch.float64)
    base = dict(num_spherical=7, num_radial=6, num_blocks=2, emb_size_atom=32, emb_size_edge=24,
                emb_size_trip=16, emb_size_quad=8, emb_size_rbf=8, emb_size_cbf=8,
                emb_size_sbf=12, emb_size_bil_trip=16, emb_size_bil_quad=10, num_before_skip=1,
                num_after_skip=2, num_concat=1, num_atom=2, cutoff=5.0, int_cutoff=10.0,
                envelope_exponent=5)
    molecules = random_molecules() + coll_molecules(upstream, 3)
    worst = 0.0
    for triplets_only in (False, True):
        for direct, coupled in ((False, False), (True, False), (True, True)):
            kw = dict(base, triplets_only=triplets_only, direct_forces=direct,
                      forces_coupled=coupled)
            model = GemNet(**xnn_kwargs(kw)).double()
            rng = np.random.default_rng(1)
            factors = {name: float(rng.uniform(0.5, 2.0)) for name in model.scale_factors()}
            for q in ("QuadInteraction_1_had_rbf", "QuadInteraction_1_had_cbf",
                      "QuadInteraction_1_sum_sbf", "QuadInteraction_2_had_rbf",
                      "QuadInteraction_2_had_cbf", "QuadInteraction_2_sum_sbf"):
                factors.setdefault(q, 1.0)
            scale_file = os.path.join(tempfile.mkdtemp(), "scaling_factors.json")
            with open(scale_file, "w") as f:
                json.dump(factors, f)
            torch.manual_seed(2)
            ref = RefGemNet(**kw, scale_file=scale_file).double()
            transplant(ref, model)
            data = container(DataContainer, molecules, kw)
            label = ("T" if triplets_only else "Q") + ("d" if direct else "") + ("c" if coupled else "")
            worst = max(worst, compare(ref, model, data, molecules, torch.float64, label))
    return worst


def run_pretrained(upstream, count=20):
    from xnn.gnn.models.gemnet import GemNet

    molecules = coll_molecules(upstream, count)
    worst64 = 0.0
    for dtype in (torch.float64, torch.float32):
        for name in ("GemNet-Q", "GemNet-T"):
            for mod in [m for m in sys.modules if m == "gemnet" or m.startswith("gemnet.")]:
                del sys.modules[mod]                      # fresh basis (zeros of this dtype)
            RefGemNet, DataContainer = setup(upstream, float64_zeros=dtype == torch.float64)
            torch.set_default_dtype(dtype)
            folder = os.path.join(upstream, "pretrained", name)
            with open(os.path.join(folder, "model_kwargs.json")) as f:
                kw = json.load(f)
            kw["scale_file"] = os.path.join(upstream, "pretrained", "scaling_factors.json")
            ref = RefGemNet(**kw)
            ref.load_state_dict(torch.load(os.path.join(folder, "model.pth"), map_location="cpu"))
            ref = ref.to(dtype)
            model = GemNet(**xnn_kwargs(kw)).to(dtype)
            transplant(ref, model)
            data = container(DataContainer, molecules, kw)
            label = f"{name} {str(dtype)[6:]}"
            err = compare(ref, model, data, molecules, dtype, label)
            print(f"{label}: worst relative error {err:.2e}")
            if dtype == torch.float64:
                worst64 = max(worst64, err)
    return worst64


if __name__ == "__main__":
    upstream, mode = sys.argv[1], sys.argv[2]
    worst = run_random(upstream) if mode == "random" else run_pretrained(upstream)
    print(f"{worst:.3e}")
