"""Parity of the xnn ET and TensorNet with the reference code (torchmd/torchmd-net).

    python torchmdnet_parity.py <path to a torchmd-net checkout> et|tensornet [float64|float32]

Builds randomly initialized xnn models, transplants their weights into the
reference modules (``TorchMD_ET`` / ``TensorNet`` with the ``EquivariantScalar``
/ ``Scalar`` output, the ``Atomref`` prior and the ``TorchMD_Net`` wrapper) and
compares energies and forces on random molecules, a batch and, for TensorNet,
charged structures, in the requested precision. The last printed line is the
worst relative error (energies relative to ``max(|E|, 1)``, forces to the
largest component). The reference package is used as an external oracle only;
its neighbor search (a warp kernel) is replaced by a plain PyTorch all-pairs
list with the same conventions (both directions, self pairs, ``pos[i] -
pos[j]``), and TensorNet runs its pure-PyTorch path (warp is not imported).
"""
import sys

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.tensornet import TensorNet
from xnn.trfnn.models.et import EquivariantTransformer


def setup(upstream_path):
    """Import the reference package from ``upstream_path`` and patch its neighbor search.

    The reference's neighbor search and TensorNet kernels are warp custom ops whose
    registration needs a newer torch than xnn pins; both modules are replaced by
    stubs before the import, so the pure-PyTorch paths run and warp is not needed.
    """
    import types
    sys.path.insert(0, upstream_path)
    ops = types.ModuleType("torchmdnet.extensions.ops")

    def get_neighbor_pairs_kernel(*args, **kwargs):
        raise NotImplementedError("the parity harness replaces the neighbor search")

    ops.get_neighbor_pairs_kernel = get_neighbor_pairs_kernel
    sys.modules["torchmdnet.extensions.ops"] = ops
    sys.modules["torchmdnet.extensions.warp_ops"] = types.ModuleType("torchmdnet.extensions.warp_ops")
    import torchmdnet.models.utils as utils

    def all_pairs(self, pos, batch=None, box=None):
        n = pos.shape[0]
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=pos.device)
        i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        i, j = i.flatten(), j.flatten()
        vec = pos[i] - pos[j]
        dist = vec.norm(dim=-1)
        keep = (batch[i] == batch[j]) & ((dist < self.cutoff_upper) | (i == j))
        if not self.loop:
            keep = keep & (i != j)
        return torch.stack([i[keep], j[keep]]), dist[keep], vec[keep]

    utils.OptimizedDistance.forward = all_pairs
    import torchmdnet.models.tensornet as tn
    assert not tn.OPT, "the reference TensorNet must run its pure-PyTorch path"


def _copy(dst, src):
    with torch.no_grad():
        dst.copy_(src.to(dst.dtype))


def _copy_module(dst, src):
    for (name, p), (name_src, q) in zip(dst.named_parameters(), src.named_parameters()):
        assert name == name_src, (name, name_src)
        _copy(p, q)
    for (name, b), (name_src, c) in zip(dst.named_buffers(), src.named_buffers()):
        assert name == name_src, (name, name_src)
        _copy(b, c)


def build_et(model, dtype):
    """The reference TorchMD_Net around a TorchMD_ET with the xnn weights."""
    from torchmdnet.models.model import TorchMD_Net
    from torchmdnet.models.output_modules import EquivariantScalar
    from torchmdnet.models.torchmd_et import TorchMD_ET
    from torchmdnet.priors import Atomref

    F = model.n_features
    rep = TorchMD_ET(hidden_channels=F, num_layers=len(model.layers), num_rbf=model.rbf.n_rbf,
                     rbf_type="expnorm", trainable_rbf=isinstance(model.rbf.means, torch.nn.Parameter),
                     activation="silu", attn_activation="silu",
                     neighbor_embedding=model.neighbor_embedding is not None,
                     num_heads=model.num_heads, distance_influence=model.distance_influence,
                     cutoff_lower=0.0, cutoff_upper=model.cutoff,
                     max_z=model.embedding.num_embeddings, vector_cutoff=model.vector_cutoff,
                     dtype=dtype)
    out = EquivariantScalar(F, activation="silu", dtype=dtype)
    prior = Atomref(max_z=model.atom_ref.num_embeddings, trainable=False)
    ref = TorchMD_Net(rep, out, prior_model=[prior], std=model.energy_scale.clone(),
                      derivative=True, dtype=dtype)
    _copy(rep.embedding.weight, model.embedding.weight)
    _copy(rep.distance_expansion.means, model.rbf.means)
    _copy(rep.distance_expansion.betas, model.rbf.betas)
    if model.neighbor_embedding is not None:
        _copy_module(rep.neighbor_embedding, model.neighbor_embedding)
    for layer, src in zip(rep.attention_layers, model.layers):
        for name in ("layernorm", "q_proj", "k_proj", "v_proj", "o_proj", "vec_proj",
                     "dk_proj", "dv_proj"):
            if getattr(src, name) is not None:
                _copy_module(getattr(layer, name), getattr(src, name))
    _copy_module(rep.out_norm, model.out_norm)
    for geb, block in zip(out.output_network, model.readout.blocks):
        _copy(geb.vec1_proj.weight, block.lin_v.weight[:block.n_norm])
        _copy(geb.vec2_proj.weight, block.lin_v.weight[block.n_norm:])
        _copy_module(geb.update_net.layers[0], block.net[0])
        _copy_module(geb.update_net.layers[2], block.net[2])
    _copy(prior.atomref.weight, model.atom_ref.weight)
    return ref.eval()


def build_tensornet(model, dtype):
    """The reference TorchMD_Net around a TensorNet with the xnn weights."""
    from torchmdnet.models.model import TorchMD_Net
    from torchmdnet.models.output_modules import Scalar
    from torchmdnet.models.tensornet import TensorNet as RefTensorNet
    from torchmdnet.priors import Atomref

    F = model.n_features
    rep = RefTensorNet(hidden_channels=F, num_layers=len(model.layers), num_rbf=model.rbf.n_rbf,
                       rbf_type="expnorm", trainable_rbf=isinstance(model.rbf.means, torch.nn.Parameter),
                       activation="silu", cutoff_lower=0.0, cutoff_upper=model.cutoff,
                       max_z=model.tensor_embedding.emb.num_embeddings,
                       equivariance_invariance_group=model.equivariance_group,
                       static_shapes=False, dtype=dtype)
    out = Scalar(F, activation="silu", dtype=dtype)
    prior = Atomref(max_z=model.atom_ref.num_embeddings, trainable=False)
    ref = TorchMD_Net(rep, out, prior_model=[prior], std=model.energy_scale.clone(),
                      derivative=True, dtype=dtype)
    _copy(rep.distance_expansion.means, model.rbf.means)
    _copy(rep.distance_expansion.betas, model.rbf.betas)
    _copy_module(rep.tensor_embedding, model.tensor_embedding)
    for layer, src in zip(rep.layers, model.layers):
        _copy_module(layer, src)
    _copy_module(rep.out_norm, model.out_norm)
    _copy_module(rep.linear, model.linear)
    _copy_module(out.output_network.layers[0], model.readout[0])
    _copy_module(out.output_network.layers[2], model.readout[2])
    _copy(prior.atomref.weight, model.atom_ref.weight)
    return ref.eval()


def reference_outputs(ref, graph, dtype):
    z = graph.atomic_numbers
    pos = graph.pos.to(dtype).clone()
    q = graph.total_charge.to(dtype) if graph.total_charge is not None else None
    energy, forces = ref(z, pos, graph.batch, q=q)
    return {"energy": energy.detach().reshape(-1), "forces": forces.detach()}


def xnn_outputs(model, graph):
    out = ForceStressOutput(model)(graph)
    return {"energy": out["energy"].detach(), "forces": out["forces"].detach()}


def random_molecule(n, seed, total_charge=None):
    rng = np.random.default_rng(seed)
    s = {"pos": rng.uniform(-2.5, 2.5, (n, 3)), "atomic_numbers": rng.choice([1, 1, 6, 7, 8], size=n)}
    if total_charge is not None:
        s["total_charge"] = float(total_charge)
    return s


def compare(model, ref, graph, dtype, label):
    a, b = xnn_outputs(model, graph), reference_outputs(ref, graph, dtype)
    worst = 0.0
    for key in a:
        scale = a[key].abs().max().item() if key == "forces" else max(a[key].abs().max().item(), 1.0)
        err = (a[key].to(b[key].dtype) - b[key]).abs().max().item() / scale
        print(f"  {label:28s} {key:7s} max rel err {err:.3e}")
        worst = max(worst, err)
    return worst


def main():
    upstream, which = sys.argv[1], sys.argv[2]
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        sys.argv[3] if len(sys.argv) > 3 else "float64"]
    setup(upstream)
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    worst = 0.0
    if which == "et":
        variants = [("vector_cutoff", dict(vector_cutoff=True)),
                    ("original cutoff", dict(vector_cutoff=False)),
                    ("no neighbor embedding", dict(neighbor_embedding=False, distance_influence="keys"))]
        cases = [((5, 0), None), ((9, 1), None), ((14, 2), None)]
    else:
        variants = [("O(3)", dict(equivariance_group="O(3)")),
                    ("SO(3)", dict(equivariance_group="SO(3)")),
                    ("O(3), no interaction", dict(n_interactions=0))]
        cases = [((5, 0), None), ((9, 1), 1.0), ((14, 2), -1.0)]
    for name, options in variants:
        if which == "et":
            model = EquivariantTransformer(n_features=32, n_interactions=2, n_rbf=16, num_heads=4,
                                           cutoff=5.0, energy_scale=1.7, **options)
            ref = build_et(model, dtype)
        else:
            kw = dict(n_features=16, n_interactions=2, n_rbf=16, cutoff=4.5, energy_scale=1.7)
            kw.update(options)
            model = TensorNet(**kw)
            ref = build_tensornet(model, dtype)
        with torch.no_grad():
            model.atom_ref.weight.normal_()
            for p in model.parameters():
                if p.dim() == 1:
                    p.add_(0.1 * torch.randn_like(p))
        model.eval()
        ref = (build_et if which == "et" else build_tensornet)(model, dtype)
        print(f"{which} {name} ({dtype}):")
        for (n, seed), q in cases:
            graph = structure_to_graph(random_molecule(n, seed, q), model.cutoff)
            worst = max(worst, compare(model, ref, graph, dtype, f"{n} atoms, q={q}"))
        batch = collate([structure_to_graph(random_molecule(n, 10 + n, 1.0 if which != "et" else None),
                                            model.cutoff) for n in (6, 11)])
        worst = max(worst, compare(model, ref, batch, dtype, "batch of two"))
    print(f"worst relative error ({which}, {dtype}): {worst:.3e}")
    print(worst)


if __name__ == "__main__":
    main()
