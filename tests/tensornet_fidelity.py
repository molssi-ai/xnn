"""Fidelity of the xnn TensorNet to the reference implementation (torchmd/torchmd-net, MIT).

    python tensornet_fidelity.py <path to a torchmd-net checkout> [float64|float32]

Builds randomly initialized xnn :class:`~xnn.gnn.models.tensornet.TensorNet`
models, builds the reference ``TorchMD_Net`` from its public classes (the
``TensorNet`` representation on its pure-PyTorch path, the ``Scalar`` output,
the ``Atomref`` prior, the standard deviation as ``std``), loads the xnn
weights into it through its state dict and compares energies and forces on the
same neighbor lists: molecules, charged molecules, a batch of molecules, a water
box and a batch of two cells. The models have the default sizes (``F = 128``,
two interaction layers, 32 radial functions, 4.5 Angstrom), in four variants:
the defaults (``O(3)``), ``SO(3)``, the embedding alone and trainable radial
functions. The last printed line is the worst relative error. The reference is
used as an external oracle only.

Both codes see one neighbor list: the xnn graph's pairs plus one self pair per
atom, which the reference's neighbor list carries and xnn adds as per-atom
terms (:func:`reference_inputs`, through the patched search of
``torchmdnet_fidelity.setup``). The fixed radial centers and widths are not
transplanted: the reference evaluates its own formula in the model's dtype
(it builds them in float32 otherwise), so a model built in float32 and cast to
float64 is compared with exact constants.
"""
import os
import sys

import torch
from torch import nn

import torchmdnet_fidelity as upstream
from painn_fidelity import random_molecule, water_box
from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.tensornet import TensorNet

#: the defaults
DEFAULT = dict()
SO3 = dict(equivariance_group="SO(3)")
EMBEDDING = dict(n_interactions=0)
TRAINABLE_RBF = dict(trainable_rbf=True)

KEYS = ("energy", "forces")


def build_model(settings, dtype, seed=0, built_in=None) -> TensorNet:
    """A random xnn TensorNet in ``dtype`` (built in ``built_in`` and cast, when given).

    The biases and layer-norm weights are perturbed and the reference energies
    drawn at random, so that no term of the comparison is trivially zero.
    """
    default = torch.get_default_dtype()
    torch.set_default_dtype(built_in or dtype)
    try:
        torch.manual_seed(seed)
        model = TensorNet(energy_scale=1.7, **settings)
        with torch.no_grad():
            model.atom_ref.weight.normal_()
            for p in model.parameters():
                if p.dim() == 1:
                    p.add_(0.1 * torch.randn_like(p))
    finally:
        torch.set_default_dtype(default)
    return model.to(dtype).eval()


def _reference_name(name: str) -> str:
    """The reference state-dict key of an xnn TensorNet key."""
    if name == "energy_scale":
        return "std"
    if name == "atom_ref.weight":
        return "prior_model.0.atomref.weight"
    if name.startswith("readout."):
        return "output_model.output_network.layers." + name[len("readout."):]
    if name.startswith("rbf."):
        return "representation_model.distance_expansion." + name[len("rbf."):]
    return "representation_model." + name


def reference_state_dict(model: TensorNet, ref: nn.Module) -> dict:
    """The xnn weights under the reference names, the reference's own fixed buffers otherwise."""
    template = ref.state_dict()
    trainable_rbf = isinstance(model.rbf.means, nn.Parameter)
    sd = {}
    for name, value in model.state_dict().items():
        if name == "energy_shift":
            # the reference adds its mean per structure, xnn its shift per atom
            assert float(value) == 0.0, "the harness compares models without an energy shift"
            continue
        if name.startswith("rbf.") and not trainable_rbf:
            continue
        target = _reference_name(name)
        assert target in template, (name, target)
        assert template[target].shape == value.shape, (name, template[target].shape, value.shape)
        sd[target] = value.to(template[target].dtype)
    for name in template:
        if name not in sd:
            sd[name] = template[name]
    return sd


def build_reference(model: TensorNet) -> nn.Module:
    """The reference ``TorchMD_Net`` with the sizes of ``model``, carrying its weights."""
    from torchmdnet.models.model import TorchMD_Net
    from torchmdnet.models.output_modules import Scalar
    from torchmdnet.models.tensornet import TensorNet as RefTensorNet
    from torchmdnet.priors import Atomref

    dtype = model.linear.weight.dtype
    F = model.n_features
    # the reference builds some layers in the default dtype
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        rep = RefTensorNet(hidden_channels=F, num_layers=len(model.layers), num_rbf=model.rbf.n_rbf,
                           rbf_type="expnorm", trainable_rbf=isinstance(model.rbf.means, nn.Parameter),
                           activation="silu", cutoff_lower=0.0, cutoff_upper=model.cutoff,
                           max_z=model.tensor_embedding.emb.num_embeddings,
                           equivariance_invariance_group=model.equivariance_group,
                           static_shapes=False, dtype=dtype)
        out = Scalar(F, activation="silu", dtype=dtype)
        prior = Atomref(max_z=model.atom_ref.num_embeddings, trainable=False)
        ref = TorchMD_Net(rep, out, prior_model=[prior], std=model.energy_scale.detach().cpu().clone(),
                          derivative=True, dtype=dtype)
    finally:
        torch.set_default_dtype(default)
    ref = ref.to(dtype)
    basis = ref.representation_model.distance_expansion
    if not isinstance(basis.means, nn.Parameter):
        # the reference builds its fixed centers and widths in float32 whatever the
        # model's dtype (a float64 model carries them float32-rounded); its own
        # formula evaluated in the model's dtype gives the exact constants
        basis.dtype = dtype
        means, betas = basis._initial_params()
        with torch.no_grad():
            basis.means.copy_(means)
            basis.betas.copy_(betas)
    ref.load_state_dict(reference_state_dict(model, ref), strict=True)
    return ref.to(model.linear.weight.device).eval()


def reference_inputs(graph) -> dict:
    """The reference inputs of ``graph``: the arguments of its forward and the pairs.

    The reference forms ``pos[i] - pos[j]`` with ``i`` the receiver (xnn
    ``dst``) and ``j`` the neighbor (xnn ``src``), the xnn edge vector; the
    periodic image adds ``cell_shift @ cell`` as in xnn. One self pair per atom
    (zero vector) is appended.
    """
    src, dst = graph.edge_index[0], graph.edge_index[1]
    pos = graph.pos
    if graph.cell is None:
        shift = pos.new_zeros((src.numel(), 3))
    else:
        cell = graph.cell.to(pos.dtype)[graph.batch[src]]
        shift = torch.einsum("ei,eij->ej", graph.cell_shifts.to(pos.dtype), cell)
    loops = torch.arange(pos.shape[0], device=pos.device)
    pairs = (torch.cat([dst, loops]), torch.cat([src, loops]),
             torch.cat([shift, pos.new_zeros((loops.numel(), 3))]).detach())
    q = graph.total_charge.to(pos.dtype) if graph.total_charge is not None else None
    return {"z": graph.atomic_numbers, "pos": pos.detach().clone(), "batch": graph.batch,
            "q": q, "num_systems": int(graph.n_atoms.numel()), "pairs": pairs}


def reference_call(ref, inputs):
    """Energies ``(B,)`` and forces of the reference, positions as a fresh leaf."""
    upstream.PAIRS = inputs["pairs"]
    energy, forces = ref(inputs["z"], inputs["pos"].detach().clone(), inputs["batch"],
                         q=inputs["q"], num_systems=inputs["num_systems"])
    return energy.reshape(-1), forces


def reference_outputs(ref, graph) -> dict:
    energy, forces = reference_call(ref, reference_inputs(graph))
    return {"energy": energy.detach(), "forces": forces.detach()}


def xnn_outputs(model, graph) -> dict:
    out = ForceStressOutput(model).eval()(graph)
    return {k: out[k].detach() for k in KEYS}


def charged(structure, q):
    return dict(structure, total_charge=float(q))


def compare(model, ref, graph, label) -> float:
    """Worst error: energies relative to ``max(|E|, 1)``, forces to their largest component."""
    a, b = xnn_outputs(model, graph), reference_outputs(ref, graph)
    errors = {}
    for key in KEYS:
        scale = b[key].abs().max().item()
        if key == "energy":
            scale = max(scale, 1.0)
        errors[key] = (a[key].to(b[key].dtype) - b[key]).abs().max().item() / scale
    print(f"  {label:34s} " + "  ".join(f"{k} {e:.2e}" for k, e in errors.items()))
    return max(errors.values())


def model_and_reference(settings, dtype, seed=0, built_in=None):
    """A random xnn model and the reference carrying its weights."""
    model = build_model(settings, dtype, seed, built_in)
    return model, build_reference(model)


def run(dtype) -> float:
    torch.set_default_dtype(dtype)
    name = str(dtype).replace("torch.", "")
    worst = 0.0
    variants = (("defaults, O(3)", DEFAULT), ("SO(3)", SO3), ("embedding alone", EMBEDDING),
                ("trainable radial functions", TRAINABLE_RBF))
    for label, settings in variants:
        print(f"{label} ({name}):")
        model, ref = model_and_reference(settings, dtype)
        cutoff = model.cutoff
        for n, seed in ((5, 0), (9, 1), (21, 2)):
            graph = structure_to_graph(random_molecule(n, seed), cutoff)
            worst = max(worst, compare(model, ref, graph, f"molecule of {n} atoms"))
        for n, seed, q in ((9, 3, 1.0), (14, 4, -2.0)):
            graph = structure_to_graph(charged(random_molecule(n, seed), q), cutoff)
            worst = max(worst, compare(model, ref, graph, f"molecule of {n} atoms, q = {q:+.0f}"))
        batch = collate([structure_to_graph(charged(random_molecule(n, 10 + n), q), cutoff)
                         for n, q in ((6, 0.0), (12, 1.0), (17, -1.0))])
        worst = max(worst, compare(model, ref, batch, "batch of three molecules, charged"))
        if settings in (DEFAULT, SO3):
            graph = structure_to_graph(water_box(4, 0), cutoff)
            worst = max(worst, compare(model, ref, graph, "water box of 192 atoms"))
            batch = collate([structure_to_graph(water_box(k, k), cutoff) for k in (3, 4)])
            worst = max(worst, compare(model, ref, batch, "batch of two cells"))
        if dtype == torch.float64 and settings in (DEFAULT, TRAINABLE_RBF):
            print(f"{label}, built in float32 and cast to float64:")
            model, ref = model_and_reference(settings, dtype, built_in=torch.float32)
            graph = structure_to_graph(random_molecule(21, 2), cutoff)
            worst = max(worst, compare(model, ref, graph, "molecule of 21 atoms"))
    print(f"worst relative error ({name}): {worst:.3e}")
    return worst


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ["TORCHMDNET_UPSTREAM_PATH"]
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        sys.argv[2] if len(sys.argv) > 2 else "float64"]
    upstream.setup(path)
    print(run(dtype))


if __name__ == "__main__":
    main()
