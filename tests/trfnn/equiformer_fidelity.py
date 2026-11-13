"""Fidelity of the xnn Equiformer and EquiformerV2 to the reference codes.

    python equiformer_fidelity.py v1 <path to atomicarchitects/equiformer> [float64|float32]
    python equiformer_fidelity.py v2 [float64|float32]

``v1`` compares :class:`~xnn.trfnn.models.equiformer.Equiformer` with
``GraphAttentionTransformerMD17`` / ``DotProductAttentionTransformerMD17`` of
the authors' code (its ``nets`` package is loaded from the checkout without its
``__init__``, which imports the OC20 trainer); ``v2`` compares
:class:`~xnn.trfnn.models.equiformer_v2.EquiformerV2` with
``EquiformerV2Backbone`` and its scalar / vector heads from the installed
``fairchem-core`` 1.x. The reference codes are external oracles only, with these
harness patches:

* v1: ``torch_cluster.radius_graph`` and ``torch_scatter.scatter`` are stubs (the
  graph is the xnn neighbor list, the sum is ``index_add``); the GemNet radial
  basis comes from fairchem; the atom one-hot follows the weights' dtype (the
  reference casts it to float32). Periodic structures run the reference blocks
  with the periodic edge vectors ``pos[src] - pos[dst] + offset`` of its OC20
  model, since the MD17 model builds its own (molecular) graph.
* v2: ``generate_graph`` returns the xnn neighbor list, and the random edge frames
  of ``init_edge_rot_mat`` are replaced by the deterministic frames of xnn (the
  model is invariant to the roll up to the sphere sampling). The reference
  detaches the frames from the positions, so its autograd forces are compared
  with the xnn forces of detached frames; the full xnn forces are checked against
  finite differences.

Weights of randomly initialized xnn models (with random biases) are copied into
the reference modules; energies and forces are compared on random molecules, a
batch and periodic cells. The last printed line is the worst relative error
(energies relative to ``max(|E|, 1)``, forces to the largest component).
"""
import os
import sys
import types

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.trfnn.models.equiformer import Equiformer
from xnn.trfnn.models.equiformer_v2 import EdgeFrames, EquiformerV2, edge_rotation


def _copy(dst, src):
    with torch.no_grad():
        dst.copy_(src.detach().reshape(dst.shape).to(dst.dtype))


def _randomize_biases(model, seed=1):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or name.endswith("affine_weight") or "norm" in name:
                p.add_(0.1 * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def random_structure(n, seed, periodic=False, elements=(1, 1, 6, 7, 8)):
    rng = np.random.default_rng(seed)
    if periodic:
        cell = np.diag(rng.uniform(5.5, 7.0, 3)) + rng.uniform(-0.4, 0.4, (3, 3))
        frac = rng.uniform(0, 1, (n, 3))
        s = {"pos": frac @ cell, "cell": cell, "pbc": [True, True, True]}
    else:
        s = {"pos": rng.uniform(-2.2, 2.2, (n, 3))}
    s["atomic_numbers"] = rng.choice(elements, size=n)
    return s


def compare(a, b, label):
    worst = 0.0
    for key in a:
        x, y = a[key].double(), b[key].double()
        scale = x.abs().max().item() if key == "forces" else max(x.abs().max().item(), 1.0)
        err = (x - y).abs().max().item() / scale
        print(f"  {label:44s} {key:7s} max rel err {err:.3e}")
        worst = max(worst, err)
    return worst


# v1: atomicarchitects/equiformer

_EDGES = {}


def _stub_module(name):
    """An empty module registered under ``name``, with a spec so ``find_spec`` accepts it."""
    import importlib.machinery
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = module
    return module


def _stub_scatter():
    """``torch_scatter`` (a compiled extension the references import) as a sum by ``index_add``."""
    import torch_geometric  # noqa: F401  (imported first: it probes for the compiled extensions)
    if "torch_scatter" in sys.modules:
        return
    scatter_mod = _stub_module("torch_scatter")

    def scatter(src, index, dim=0, dim_size=None, reduce="sum", **kwargs):
        assert dim == 0 and reduce == "sum"
        size = int(index.max()) + 1 if dim_size is None else dim_size
        out = torch.zeros((size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        return out.index_add(0, index, src)

    scatter_mod.scatter = scatter


def _fairchem_modules():
    """Make fairchem modules importable without the package ``__init__`` files.

    Those import the trainers, the ASE calculator and their dependencies; the
    model modules themselves need only torch, e3nn and torch_geometric.
    """
    import importlib.util
    _stub_scatter()
    root = list(importlib.util.find_spec("fairchem").submodule_search_locations)[0]
    for name in ("fairchem.core", "fairchem.core.models", "fairchem.core.models.scn",
                 "fairchem.core.models.gemnet", "fairchem.core.models.equiformer_v2"):
        if name not in sys.modules:
            module = _stub_module(name)
            module.__path__ = [os.path.join(root, *name.split(".")[1:])]


def setup_v1(path):
    """Load the reference ``nets`` modules with stubs for the compiled graph libraries."""
    _stub_scatter()
    cluster = _stub_module("torch_cluster")

    def radius_graph(pos, r, batch=None, max_num_neighbors=32, **kwargs):
        return _EDGES["edge_index"]

    cluster.radius_graph = radius_graph
    _fairchem_modules()
    import fairchem.core.models.gemnet.layers.radial_basis as rb
    for name in ("ocpmodels", "ocpmodels.models", "ocpmodels.models.gemnet",
                 "ocpmodels.models.gemnet.layers"):
        if name not in sys.modules:
            _stub_module(name)
    sys.modules["ocpmodels.models.gemnet.layers.radial_basis"] = rb
    nets = types.ModuleType("nets")
    nets.__path__ = [f"{path}/nets"]
    sys.modules["nets"] = nets
    import nets.dp_attention_transformer_md17 as dp
    import nets.graph_attention_transformer as gat
    import nets.graph_attention_transformer_md17 as md17

    def embed(self, node_atom):
        onehot = torch.nn.functional.one_hot(node_atom, self.max_atom_type).to(
            self.atom_type_lin.tp.weight.dtype)
        return self.atom_type_lin(onehot), onehot, onehot

    gat.NodeEmbeddingNetwork.forward = embed
    return md17, dp


def _linear_rs(dst, src):
    """An e3nn LinearRS / FCTP of the reference from an :class:`IrrepsLinear` (same flat layout)."""
    _copy(dst.tp.weight, src.weight)
    if src.bias is not None:
        _copy(dst.bias[0], src.bias)
    else:
        assert len(dst.bias) == 0


def _radial(dst, src):
    """A reference RadialProfile (last layer without bias, plus an offset) from a RadialMLP."""
    for k in (0, 1, 3, 4):
        _copy(dst.net[k].weight, src.net[k].weight)
        _copy(dst.net[k].bias, src.net[k].bias)
    _copy(dst.net[6].weight, src.net[6].weight)
    _copy(dst.offset, src.net[6].bias)


def _norm(dst, src):
    _copy(dst.affine_weight, src.affine_weight)
    _copy(dst.affine_bias, src.affine_bias)




def build_v1(model, spec, md17, dp):
    """The reference MD17 model with the xnn weights; ``spec`` holds the options of :func:`small_v1`."""
    n_types = model.atom_embed.weight.shape[0]
    kwargs = dict(irreps_in=f"{n_types}x0e", irreps_node_embedding=str(model.irreps_node_embedding),
                  num_layers=len(model.blocks), irreps_node_attr="1x0e",
                  irreps_sh=str(model.irreps_sh), max_radius=model.cutoff,
                  number_of_basis=spec["n_rbf"], fc_neurons=spec["radial_hidden"],
                  basis_type=spec["basis"], irreps_feature=str(model.irreps_feature),
                  irreps_head=spec["irreps_head"], num_heads=spec["num_heads"],
                  irreps_pre_attn=None, rescale_degree=False,
                  nonlinear_message=spec["nonlinear"], irreps_mlp_mid=spec["irreps_mlp_mid"],
                  norm_layer="layer", alpha_drop=0.0, proj_drop=0.0, out_drop=0.0,
                  drop_path_rate=0.0)
    if spec["attention"] == "mlp":
        ref = md17.GraphAttentionTransformerMD17(**kwargs)
    else:
        ref = dp.DotProductAttentionTransformerMD17(**kwargs)
    # the reference creates its Bessel frequencies in float32: cast it as a float64 user would
    ref = ref.to(model.atom_embed.weight.dtype)
    ref.edge_deg_embed.scale_scatter.avg_aggregate_num = model.edge_deg_embed.avg_degree
    ref.scale_scatter.avg_aggregate_num = model.avg_num_nodes
    _copy(ref.atom_embed.atom_type_lin.tp.weight, model.atom_embed.weight)
    _copy(ref.atom_embed.atom_type_lin.bias[0], model.atom_embed.bias)
    if spec["basis"] == "exp":
        _copy(ref.rbf.means, model.rbf.means)
        _copy(ref.rbf.betas, model.rbf.betas)
    elif spec["basis"] == "gaussian":
        for name in ("mean", "std", "weight", "bias"):
            _copy(getattr(ref.rbf, name), getattr(model.rbf, name))
    else:
        _copy(ref.rbf.rbf.frequencies, model.rbf.rbf.freqs)
    deg, mine = ref.edge_deg_embed, model.edge_deg_embed
    _linear_rs(deg.exp, mine.exp)
    _radial(deg.rad, mine.rad)
    _linear_rs(deg.proj, mine.proj)
    for blk, src in zip(ref.blocks, model.blocks):
        _norm(blk.norm_1, src.norm_1)
        _norm(blk.norm_2, src.norm_2)
        att = blk.ga if spec["attention"] == "mlp" else blk.dpa
        core = src.ga.core
        _linear_rs(att.merge_src, src.ga.merge_src)
        _linear_rs(att.merge_dst, src.ga.merge_dst)
        _linear_rs(att.proj, src.ga.proj)
        if spec["attention"] == "mlp" and spec["nonlinear"]:
            _radial(att.sep_act.dtp_rad, core.sep_act.rad)
            _linear_rs(att.sep_act.lin, core.sep_act.lin)
            _linear_rs(att.sep_alpha, core.sep_alpha)
            _copy(att.sep_value.dtp.tp.weight, core.sep_value.dtp.weight)
            _linear_rs(att.sep_value.lin, core.sep_value.lin)
            _copy(att.alpha_dot, core.scores.alpha_dot)
        elif spec["attention"] == "mlp":
            _radial(att.sep.dtp_rad, core.sep.rad)
            _linear_rs(att.sep.lin, core.sep.lin)
            _copy(att.alpha_dot, core.scores.alpha_dot)
        else:
            _linear_rs(att.query, core.query)
            _radial(att.key_value.dtp_rad, core.key_value.rad)
            _linear_rs(att.key_value.lin, core.key_value.lin)
        _linear_rs(blk.ffn.fctp_1, src.ffn.lin_1)
        _linear_rs(blk.ffn.fctp_2, src.ffn.lin_2)
        if src.ffn_shortcut is not None:
            _linear_rs(blk.ffn_shortcut, src.ffn_shortcut)
    _norm(ref.norm, model.norm)
    _linear_rs(ref.head[0], model.head[0])
    _linear_rs(ref.head[2], model.head[2])
    n_ref = sum(p.numel() for p in ref.parameters())
    n_mine = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("atom_ref"))
    assert n_ref == n_mine, (n_ref, n_mine)
    return ref.eval()


def v1_reference(ref, graph):
    """Energies and forces of the reference v1 model on an xnn graph."""
    from e3nn import o3
    pos = graph.pos.detach().clone().requires_grad_(True)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    if graph.cell is None:
        _EDGES["edge_index"] = graph.edge_index
        energy, forces = ref(node_atom=graph.atomic_numbers, pos=pos, batch=graph.batch)
        return {"energy": energy.detach().reshape(-1), "forces": forces.detach()}
    # the periodic edge vectors of the reference OC20 model, through the same blocks
    shifts = torch.einsum("ei,eij->ej", graph.cell_shifts.to(pos.dtype), graph.cell[graph.batch[src]])
    edge_vec = pos[src] - pos[dst] - shifts
    sh = o3.spherical_harmonics(l=ref.irreps_edge_attr, x=edge_vec, normalize=True,
                                normalization="component")
    emb, _, _ = ref.atom_embed(graph.atomic_numbers)
    rbf = ref.rbf(edge_vec.norm(dim=1))
    x = emb + ref.edge_deg_embed(emb, sh, rbf, src, dst, graph.batch)
    attr = torch.ones_like(x.narrow(1, 0, 1))
    for blk in ref.blocks:
        x = blk(node_input=x, node_attr=attr, edge_src=src, edge_dst=dst, edge_attr=sh,
                edge_scalars=rbf, batch=graph.batch)
    x = ref.norm(x, batch=graph.batch)
    energy = ref.scale_scatter(ref.head(x), graph.batch, dim=0)
    forces = -torch.autograd.grad(energy.sum(), pos)[0]
    return {"energy": energy.detach().reshape(-1), "forces": forces.detach()}


def xnn_outputs(model, graph):
    out = ForceStressOutput(model)(graph)
    return {"energy": out["energy"].detach(), "forces": out["forces"].detach()}


def graphs(cutoff, dtype):
    """The test cases: two molecules, a batch of two and a triclinic cell."""
    cases = [("molecule 6", [random_structure(6, 1, elements=(1, 6, 8))]),
             ("molecule 11", [random_structure(11, 2)]),
             ("batch 5+9", [random_structure(5, 3), random_structure(9, 4)]),
             ("periodic cell 8", [random_structure(8, 5, periodic=True)])]
    for label, structures in cases:
        graph = collate([structure_to_graph(s, cutoff) for s in structures])
        graph.pos = graph.pos.to(dtype)
        if graph.cell is not None:
            graph.cell = graph.cell.to(dtype)
        yield label, graph


V1_VARIANTS = [
    ("MLP attention, non-linear, L=2, exp", dict(l_max=2, e3=False, basis="exp", attention="mlp",
                                                 nonlinear=True)),
    ("MLP attention, linear messages", dict(l_max=2, e3=False, basis="exp", attention="mlp",
                                            nonlinear=False)),
    ("dot-product attention", dict(l_max=2, e3=False, basis="exp", attention="dot_product",
                                   nonlinear=False)),
    ("E(3), L=2, Gaussian basis", dict(l_max=2, e3=True, basis="gaussian", attention="mlp",
                                       nonlinear=True)),
    ("L=3, Bessel basis", dict(l_max=3, e3=False, basis="bessel", attention="mlp", nonlinear=True)),
]


def small_v1(spec, seed=0):
    """A small xnn Equiformer (64 element types, as the reference MD17 model) and its full spec."""
    from xnn.trfnn.models.equiformer import paper_irreps
    torch.manual_seed(seed)
    irreps = paper_irreps(spec["l_max"], spec["e3"], n_features=32)
    spec = dict(spec, n_rbf=8, radial_hidden=[16, 16], **irreps)
    model = Equiformer(n_interactions=2, cutoff=4.0, n_rbf=8,
                       radial_basis={"exp": "expnorm"}.get(spec["basis"], spec["basis"]),
                       radial_hidden=[16, 16], irreps_feature="48x0e", attention=spec["attention"],
                       nonlinear_message=spec["nonlinear"], n_elements=64, avg_degree=9.0,
                       avg_num_nodes=7.0, **irreps)
    _randomize_biases(model)
    return model, spec


def run_v1(path, dtype):
    md17, dp = setup_v1(path)
    worst = 0.0
    for name, spec in V1_VARIANTS:
        model, spec = small_v1(spec)
        ref = build_v1(model, spec, md17, dp)
        for label, graph in graphs(model.cutoff, dtype):
            worst = max(worst, compare(xnn_outputs(model, graph), v1_reference(ref, graph),
                                       f"{name} | {label}"))
    return worst


# v2: fairchem-core 1.x

def setup_v2():
    """Patch the fairchem graph builder and the random edge frames."""
    _fairchem_modules()
    import fairchem.core.models.equiformer_v2.equiformer_v2 as eqv2
    from fairchem.core.models.base import GraphData

    def generate_graph(self, data, **kwargs):
        g = data.xnn_graph
        vec = -g.edge_vectors()
        neighbors = torch.zeros(g.num_graphs, dtype=torch.long, device=vec.device)
        neighbors.index_add_(0, g.batch[g.edge_index[1]],
                             torch.ones(g.num_edges, dtype=torch.long, device=vec.device))
        zeros = torch.zeros(g.num_edges, 3, dtype=vec.dtype, device=vec.device)
        return GraphData(edge_index=g.edge_index, edge_distance=vec.norm(dim=-1),
                         edge_distance_vec=vec, cell_offsets=zeros, offset_distances=zeros,
                         neighbors=neighbors, batch_full=g.batch,
                         atomic_numbers_full=g.atomic_numbers.long(), node_offset=0)

    eqv2.EquiformerV2Backbone.generate_graph = generate_graph
    eqv2.EquiformerV2Backbone._init_edge_rot_mat = (
        lambda self, data, edge_index, vec: edge_rotation(vec).detach())
    return eqv2


V2_VARIANTS = [
    ("SLN, separable S2, grid MLP, L=4 M=2", dict(l_max=4, m_max=2)),
    ("rms_norm_sh, gate, L=3 M=3", dict(l_max=3, m_max=3, norm_type="rms_norm_sh", activation="gate",
                                       use_grid_mlp=False)),
    ("layer_norm, S2, no renorm, L=3 M=1", dict(l_max=3, m_max=1, norm_type="layer_norm",
                                               activation="s2", use_grid_mlp=False,
                                               use_attn_renorm=False)),
    ("separable S2 without grid MLP, m-shared radial", dict(l_max=4, m_max=2, use_grid_mlp=False,
                                                           use_m_share_rad=True)),
    ("shared atom-edge embedding, direct forces", dict(l_max=4, m_max=2, share_atom_edge_embedding=True,
                                                      direct_forces=True)),
]


SMALL_V2 = dict(n_features=16, n_interactions=2, cutoff=4.0, num_heads=2, attn_hidden_channels=12,
                attn_alpha_channels=8, attn_value_channels=6, ffn_hidden_channels=20, edge_channels=12)


def small_v2(options, seed=0, sizes=SMALL_V2):
    """An xnn EquiformerV2 with the reference's 600 Gaussians and 90 element types (small by default)."""
    torch.manual_seed(seed)
    model = EquiformerV2(n_rbf=600, grid_resolution=18, n_elements=90, avg_degree=9.0,
                         avg_num_nodes=7.0, **sizes, **options)
    _randomize_biases(model)
    return model


def _v2_key(prefix, name):
    """The xnn parameter of a reference parameter (the element embeddings sit one level deeper)."""
    for owner in ("ga.", "edge_degree_embedding.", "force_block."):
        for kind in ("source_embedding", "target_embedding"):
            name = name.replace(owner + kind, owner + "edge_scalars." + kind)
    if prefix == "backbone" and name.startswith(("source_embedding", "target_embedding")):
        name = "shared_edge_scalars." + name
    return name


def build_v2(model, options, eqv2, sizes=SMALL_V2):
    """The reference backbone and heads with the xnn weights."""
    from fairchem.core.models.equiformer_v2.heads import EqV2ScalarHead, EqV2VectorHead
    activation = options.get("activation", "sep_s2")
    backbone = eqv2.EquiformerV2Backbone(
        use_pbc=True, regress_forces=True, otf_graph=False, max_neighbors=10000,
        max_radius=model.cutoff, max_num_elements=90, num_layers=len(model.blocks),
        sphere_channels=sizes["n_features"], attn_hidden_channels=sizes["attn_hidden_channels"],
        num_heads=sizes["num_heads"], attn_alpha_channels=sizes["attn_alpha_channels"],
        attn_value_channels=sizes["attn_value_channels"],
        ffn_hidden_channels=sizes["ffn_hidden_channels"],
        norm_type=options.get("norm_type", "layer_norm_sh"), lmax_list=[options["l_max"]],
        mmax_list=[options["m_max"]], grid_resolution=18, edge_channels=sizes["edge_channels"],
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=options.get("share_atom_edge_embedding", False),
        use_m_share_rad=options.get("use_m_share_rad", False), distance_function="gaussian",
        use_s2_act_attn=False, use_attn_renorm=options.get("use_attn_renorm", True),
        use_gate_act=activation == "gate", use_grid_mlp=options.get("use_grid_mlp", True),
        use_sep_s2_act=activation == "sep_s2", alpha_drop=0.0, drop_path_rate=0.0, proj_drop=0.0,
        weight_init="normal", avg_num_nodes=model.avg_num_nodes,
        avg_degree=model.edge_degree_embedding.avg_degree)
    energy_head = EqV2ScalarHead(backbone)
    force_head = EqV2VectorHead(backbone) if model.direct_forces else None
    mine = dict(model.named_parameters())
    copied = set()
    for prefix, module in (("backbone", backbone), ("energy", energy_head), ("force", force_head)):
        if module is None:
            continue
        for name, p in module.named_parameters():
            key = _v2_key(prefix, name)
            assert key in mine, (prefix, name, key)
            _copy(p, mine[key])
            copied.add(key)
    missing = [k for k in mine if k not in copied and not k.startswith("atom_ref")]
    assert not missing, missing
    _copy(backbone.distance_expansion.offset, model.distance_expansion.centers)
    return backbone.eval(), energy_head.eval(), None if force_head is None else force_head.eval()


def v2_reference(parts, graph):
    backbone, energy_head, force_head = parts
    pos = graph.pos.detach().clone().requires_grad_(True)
    graph.pos = pos
    data = types.SimpleNamespace(pos=pos, atomic_numbers=graph.atomic_numbers, batch=graph.batch,
                                 natoms=torch.bincount(graph.batch), xnn_graph=graph)
    emb = backbone(data)
    energy = energy_head(data, emb)["energy"]
    out = {"energy": energy.detach().reshape(-1)}
    if force_head is not None:
        out["forces"] = force_head(data, emb)["forces"].detach()
    else:
        out["forces"] = -torch.autograd.grad(energy.sum(), pos)[0].detach()
    return out


def xnn_v2_outputs(model, graph, detach_frames):
    original = EdgeFrames.forward
    if detach_frames:
        EdgeFrames.forward = lambda self, vec: [d.detach() for d in original(self, vec)]
    try:
        return xnn_outputs(model, graph)
    finally:
        EdgeFrames.forward = original


def run_v2(dtype):
    eqv2 = setup_v2()
    worst = 0.0
    for name, options in V2_VARIANTS:
        model = small_v2(options)
        parts = build_v2(model, options, eqv2)
        for label, graph in graphs(model.cutoff, dtype):
            # the reference detaches its frames: compare its autograd forces with those of
            # detached xnn frames (the complete xnn forces are checked by finite differences)
            mine = xnn_v2_outputs(model, graph, detach_frames=not model.direct_forces)
            worst = max(worst, compare(mine, v2_reference(parts, graph), f"{name} | {label}"))
    return worst


def main():
    which = sys.argv[1]
    args = sys.argv[2:]
    path = args.pop(0) if which == "v1" else None
    dtype = {"float32": torch.float32, "float64": torch.float64}[args[0] if args else "float64"]
    torch.set_default_dtype(dtype)
    worst = run_v1(path, dtype) if which == "v1" else run_v2(dtype)
    print(f"{worst:.6e}")


if __name__ == "__main__":
    main()
