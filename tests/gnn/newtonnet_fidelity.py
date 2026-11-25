"""Fidelity of the xnn NewtonNet to the authors' code (THGLab/NewtonNet), used as an external oracle only.

    python newtonnet_fidelity.py [float32|float64]

The authors' code is only run: models are built from its public classes, the xnn weights are loaded
into them by parameter name and the outputs are compared on the same structures. Which comparison runs
depends on the installed version:

* the current code (``InteractionNet`` blocks): the xnn defaults, layer normalization, the cosine
  envelope (the reference's ``CosineCutoff`` swapped into its edge embedding) and direct forces;
  energies and forces of molecules and batches, energies, forces and stress of water boxes wider than
  twice the cutoff (the reference uses the minimum image);
* the paper-era code (tag v1.0.0, ``DynamicsCalculator`` blocks): the paper's model
  (``latent_force``, ``displacement``). The reference blocks are driven one by one with the paper's
  Bessel basis (eq 4) and their displacement update is taken to the paper's eq 13 and 14 with the
  block's own networks: the v1.0.0 code scales the force change instead of the force features and
  keeps the previous displacement. Energies, forces and latent forces of molecules are compared.

xnn takes ``r_ij = r_j - r_i`` for the directions; the current code takes the opposite sign, which turns
the force features over and leaves the energy unchanged, so the last layer of the direct-force network
changes sign in the transplant. The last printed line is the worst relative error; in float32, the
worst error in units of its bound (see :data:`F32_BOUND`).
"""
import copy
import dataclasses
import math
import sys

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.newtonnet import NewtonNet

#: the xnn defaults (the current code)
DEFAULT = dict()
LAYER_NORM = dict(layer_norm=True)
COSINE = dict(cutoff_function="cosine")
DIRECT = dict(direct_forces=True)
#: the paper's model (compared with the paper-era code)
PAPER = dict(latent_force=True, displacement=True, rbf_bias=True, rbf_prefactor=math.sqrt(2 / 5.0),
             readout_widths=[128, 64])

F32_BOUND = 1e-5
FLOOR_FACTOR = 4.0


def reference_version() -> str:
    """``"current"``, ``"v1"`` (the paper-era code) or ``""`` when the authors' code is missing."""
    try:
        from newtonnet.models import newtonnet as nnmod
    except Exception:  # noqa: BLE001 - absent or broken install
        return ""
    if hasattr(nnmod, "InteractionNet"):
        return "current"
    return "v1" if hasattr(nnmod, "DynamicsCalculator") else ""


def build_model(settings, dtype, seed=0, F=32, T=3, n_rbf=20, element_scales=True) -> NewtonNet:
    """A random xnn NewtonNet (PyTorch's initialization, small random biases, so that dense cells keep
    energies and forces of order one); every parameter is nonzero, so that each one is exercised.

    ``element_scales=False`` keeps the per-element scale and shift at one and zero (the paper-era code
    has neither).
    """
    torch.manual_seed(seed)
    model = NewtonNet(n_features=F, n_interactions=T, n_rbf=n_rbf, cutoff=5.0, **settings).to(dtype)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or "norm" in name:
                p.uniform_(-0.1, 0.1)
        if element_scales:
            model.atom_scale.weight.uniform_(0.5, 1.5)
            model.atom_ref.weight.normal_(std=0.5)
        if model.force_scale is not None:
            model.force_scale.weight.uniform_(0.5, 1.5)
    return model.eval()


def _copy(dst, src_state, mapping, negate=()):
    """Load ``dst`` keys from xnn tensors: ``mapping`` maps reference names to xnn names."""
    state = dst.state_dict()
    for ref_name, xnn_name in mapping.items():
        value = src_state[xnn_name].detach().clone()
        if ref_name in negate:
            value = -value
        if state[ref_name].shape != value.shape:
            raise ValueError(f"{ref_name}: reference {tuple(state[ref_name].shape)} vs xnn {tuple(value.shape)}")
        state[ref_name] = value.to(state[ref_name].dtype)
    missing = [k for k in state if k not in mapping and "frequencies" not in k]
    if missing:
        raise ValueError(f"reference parameters without an xnn counterpart: {missing}")
    dst.load_state_dict(state)


def build_current(model: NewtonNet, settings):
    """The current reference code with the weights of ``model``."""
    from newtonnet.layers import representations
    from newtonnet.models.newtonnet import NewtonNet as Reference

    props = ["energy", "direct_force"] if model.direct_forces else ["energy", "gradient_force", "stress"]
    ref = Reference(cutoff=model.cutoff, n_features=model.n_features, n_basis=model.rbf.freqs.numel(),
                    n_interactions=model.n_interactions, activation="swish",
                    layer_norm=bool(settings.get("layer_norm", False)), output_properties=props)
    if settings.get("cutoff_function") == "cosine":
        ref.embedding_layers.edge_embedding.envelope = representations.CosineCutoff()
    ref = ref.to(model.embedding.weight.dtype)
    mapping = {"embedding_layers.node_embedding.weight": "embedding.weight"}
    for k in range(model.n_interactions):
        p, q = f"interaction_layers.{k}.", f"interactions.{k}."
        for a, b in (("message_nodepart.0", "phi_a.0"), ("message_nodepart.2", "phi_a.2")):
            mapping[p + a + ".weight"] = q + b + ".weight"
            mapping[p + a + ".bias"] = q + b + ".bias"
        mapping[p + "message_edgepart.weight"] = q + "phi_rbf.weight"
        for a, b in (("equiv_message1.0", "phi_f.0"), ("equiv_message1.2", "phi_f.2"),
                     ("equiv_message2.0", "phi_p.0"), ("equiv_message2.2", "phi_p.2"),
                     ("equiv_update", "mix")):
            mapping[p + a + ".weight"] = q + b + ".weight"
        if settings.get("layer_norm"):
            mapping[p + "layer_norm.weight"] = q + "norm.weight"
            mapping[p + "layer_norm.bias"] = q + "norm.bias"
    for a, b in (("0", "0"), ("2", "2"), ("4", "4")):
        mapping[f"output_layers.0.layers.{a}.weight"] = f"readout.{b}.weight"
        mapping[f"output_layers.0.layers.{a}.bias"] = f"readout.{b}.bias"
    mapping["scalers.0.scale.weight"] = "atom_scale.weight"
    mapping["scalers.0.shift.weight"] = "atom_ref.weight"
    negate = ()
    if model.direct_forces:
        for a in ("0", "2", "4"):
            mapping[f"output_layers.1.layers.{a}.weight"] = f"force_head.{a}.weight"
            mapping[f"output_layers.1.layers.{a}.bias"] = f"force_head.{a}.bias"
        mapping["scalers.1.scale.weight"] = "force_scale.weight"
        # the opposite direction convention turns the force features over
        negate = ("output_layers.1.layers.4.weight", "output_layers.1.layers.4.bias")
    _copy(ref, model.state_dict(), mapping, negate)
    freqs = ref.embedding_layers.edge_embedding.embedding.frequencies
    if (freqs.double() - model.rbf.freqs.double()).abs().max() > 1e-6:
        raise ValueError("Bessel frequencies differ")
    ref.eval()      # the reference's train() returns None, so eval() cannot be chained
    return ref


def current_outputs(ref, graph) -> dict:
    """Energy, forces (gradient or direct) and stress of the current reference."""
    B = graph.num_graphs
    cell = (graph.cell.to(graph.pos.dtype).reshape(B, 3, 3) if graph.cell is not None
            else graph.pos.new_zeros((B, 3, 3)))
    out = ref(graph.atomic_numbers, graph.pos.detach().clone(), cell, graph.batch)
    res = {"energy": out.energy.detach().reshape(-1), "n_edges": out.edge_index.shape[1]}
    if "direct_force" in vars(out):
        res["forces"] = out.direct_force.detach()
    else:
        res["forces"] = out.gradient_force.detach()
        if graph.cell is not None:
            res["stress"] = out.stress.detach().reshape(B, 3, 3)
    return res


def xnn_outputs(model, graph) -> dict:
    out = ForceStressOutput(model, compute_stress=graph.cell is not None).eval()(graph)
    res = {k: out[k].detach() for k in ("energy", "forces", "stress", "latent_forces") if out.get(k) is not None}
    res["n_edges"] = graph.edge_index.shape[1]
    return res


def build_v1(model: NewtonNet):
    """The paper-era reference with the weights of ``model`` (elements up to Z = 9)."""
    from newtonnet.layers.activations import get_activation_by_string
    from newtonnet.models.newtonnet import NewtonNet as Reference

    ref = Reference(resolution=model.rbf.freqs.numel(), n_features=model.n_features,
                    activation=get_activation_by_string("swish"), n_interactions=model.n_interactions,
                    dropout=0.0, max_z=10, cutoff=model.cutoff, cutoff_network="poly", normalizer=(0.0, 1.0),
                    normalize_atomic=False, requires_dr=True, device=torch.device("cpu"), create_graph=True,
                    shared_interactions=False, double_update_latent=True, layer_norm=False,
                    aggregration="sum").to(model.embedding.weight.dtype)
    xs = model.state_dict()
    xs = dict(xs, **{"embedding.weight": xs["embedding.weight"][:10]})
    mapping = {"embedding.weight": "embedding.weight"}
    for k in range(model.n_interactions):
        p, q = f"dycalc.{k}.", f"interactions.{k}."
        mapping[p + "phi_rbf.weight"] = q + "phi_rbf.weight"
        mapping[p + "phi_rbf.bias"] = q + "phi_rbf.bias"
        mapping[p + "phi_f.weight"] = q + "phi_F.weight"
        for a, b, bias in (("phi_a", "phi_a", True), ("phi_f_scale", "phi_f", True), ("phi_r", "phi_r", True),
                           ("phi_r_ext", "phi_rj", False), ("phi_e", "phi_u", True)):
            for i, j in (("0", "0"), ("1", "2")):
                mapping[f"{p}{a}.{i}.weight"] = f"{q}{b}.{j}.weight"
                if bias:
                    mapping[f"{p}{a}.{i}.bias"] = f"{q}{b}.{j}.bias"
    for i, j in (("0", "0"), ("1", "2"), ("2", "4")):
        mapping[f"atomic_energy.environment.{i}.weight"] = f"readout.{j}.weight"
        mapping[f"atomic_energy.environment.{i}.bias"] = f"readout.{j}.bias"
    state = ref.state_dict()
    for k in state:
        if k not in mapping and "frequencies" not in k and "cutoff_network" not in k and "inverse_normalize" not in k:
            raise ValueError(f"reference parameter without an xnn counterpart: {k}")
    for ref_name, xnn_name in mapping.items():
        state[ref_name] = xs[xnn_name].detach().clone().to(state[ref_name].dtype)
    ref.load_state_dict(state)
    ref.eval()
    return ref


def v1_outputs(ref, model: NewtonNet, mol) -> dict:
    """The paper's model from the paper-era blocks, for one molecule (energy, forces, latent forces).

    The blocks are called one by one with the paper's basis ``sqrt(2 / r_c) sin(n pi r / r_c) / r``,
    and each block's displacement and energy update are taken to eq 13 to 16 with its own networks.
    """
    dtype = model.embedding.weight.dtype
    Z = torch.as_tensor(mol["atomic_numbers"]).long().unsqueeze(0)
    R = torch.as_tensor(mol["pos"], dtype=dtype).unsqueeze(0).clone().requires_grad_(True)
    A = Z.shape[1]
    N = torch.stack([torch.tensor([j for j in range(A) if j != i]) for i in range(A)]).unsqueeze(0)
    Rj = R[0][N[0]].unsqueeze(0)                                     # (1, A, A-1, 3)
    vec = Rj - R.unsqueeze(2)
    d = vec.norm(dim=-1)
    NM = (d < model.cutoff).to(dtype)
    unit = vec / d.unsqueeze(-1)
    n = torch.arange(1, model.rbf.freqs.numel() + 1, dtype=dtype)
    rc = model.cutoff
    rbf = math.sqrt(2 / rc) * torch.sin(n * math.pi * d.unsqueeze(-1) / rc) / d.unsqueeze(-1)
    a = ref.embedding(Z)
    F = model.n_features
    latent = R.new_zeros((1, A, 3))
    f = R.new_zeros((1, A, 3, F))
    dr = R.new_zeros((1, A, 3, F))
    e = R.new_zeros((1, A, F))
    for block in ref.dycalc:
        a1, latent1, f1, dr1, e1 = block(a, rbf, d, unit, N, NM, latent, f, dr, e)
        # the block's own message and networks: a after eq 7, then eq 13 to 16 as printed
        x = block.phi_a(a)
        m = x.unsqueeze(2) * x[0][N[0]].unsqueeze(0) * block.phi_rbf(rbf) * block.cutoff_network(d).unsqueeze(-1)
        a_new = a + (m * NM.unsqueeze(-1)).sum(2)
        phi_r = block.phi_r(a_new).unsqueeze(2)
        dr_paper = dr1 - dr - phi_r * (f1 - f) + phi_r * f1
        a = a_new - block.phi_e(a_new) * (f1 * dr_paper).sum(2)
        latent, f, dr, e = latent1, f1, dr_paper, e1
    energy = ref.atomic_energy(a).sum()
    forces = -torch.autograd.grad(energy, R)[0]
    return {"energy": energy.detach().reshape(1), "forces": forces[0].detach(), "latent_forces": latent[0].detach()}


def random_molecule(n, seed, elements=(1, 1, 6, 7, 8)):
    """A random cluster, no two atoms closer than 0.9 Angstrom."""
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(-0.9, 0.9, 3) * n ** (1 / 3) * 1.4
        if all(np.linalg.norm(p - q) > 0.9 for q in pos):
            pos.append(p)
    return {"pos": np.array(pos), "atomic_numbers": rng.choice(elements, size=n)}


def water_box(n_side, seed, spacing=3.1):
    """``n_side**3`` randomly oriented water molecules on a cubic grid (about 1 g/cm^3)."""
    rng = np.random.default_rng(seed)
    angle = math.radians(104.5)
    mol = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0],
                    [0.9572 * math.cos(angle), 0.9572 * math.sin(angle), 0.0]])
    pos = []
    for site in np.ndindex(n_side, n_side, n_side):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        center = (np.array(site) + 0.5 + rng.uniform(-0.1, 0.1, 3)) * spacing
        pos.append(mol @ q.T + center)
    return {"pos": np.concatenate(pos), "atomic_numbers": np.tile([8, 1, 1], n_side ** 3),
            "cell": np.eye(3) * n_side * spacing, "pbc": [True] * 3}


def _errors(a, b, keys):
    out = {}
    for key in keys:
        if key not in b:
            continue
        if not (torch.isfinite(a[key]).all() and torch.isfinite(b[key]).all()):
            out[key] = math.inf
            continue
        scale = b[key].abs().max().item()
        if key == "energy":
            scale = max(scale, 1.0)
        out[key] = (a[key].to(b[key].dtype) - b[key]).abs().max().item() / max(scale, 1e-300)
    return out


def compare(label, a, b, exact=None, keys=("energy", "forces", "stress", "latent_forces")) -> float:
    """Worst error over ``keys`` (energies relative to ``max(|E|, 1)``, the others to their largest
    component); with ``exact`` (the reference in float64), in units of the float32 bound."""
    if "n_edges" in a and "n_edges" in b and a["n_edges"] != b["n_edges"]:
        raise AssertionError(f"{label}: {a['n_edges']} xnn edges vs {b['n_edges']} reference edges")
    errors = _errors(a, b, keys)
    scores = {}
    if exact is not None:
        floor = _errors(b, exact, keys)
        scores = {k: errors[k] / max(F32_BOUND, FLOOR_FACTOR * floor.get(k, 0.0)) for k in errors}
    line = "  ".join(f"{k} {e:.2e}" for k, e in errors.items())
    line += f"  (max |F| {b['forces'].abs().max().item():.2g})"
    if scores:
        line += f"  (of bound {max(scores.values()):.2f})"
    print(f"  {label:34s} {line}")
    return max((scores or errors).values())


def check_current(model, ref, exact, name, graph) -> float:
    """Compare one graph with the current reference (``exact``: its float64 copy, for float32 bounds)."""
    b = current_outputs(ref, graph)
    c = None
    if exact is not None:
        wide = dataclasses.replace(graph, pos=graph.pos.double(),
                                   cell=None if graph.cell is None else graph.cell.double())
        c = current_outputs(exact, wide)
    return compare(name, xnn_outputs(model, graph), b, c)


def run_current(dtype) -> float:
    cutoff = 5.0
    worst = 0.0
    variants = (("defaults", DEFAULT), ("layer normalization", LAYER_NORM), ("cosine envelope", COSINE),
                ("direct forces", DIRECT))
    for label, settings in variants:
        print(f"{label} ({dtype}):")
        model = build_model(settings, dtype)
        ref = build_current(model, settings)
        exact = copy.deepcopy(ref).double() if dtype == torch.float32 else None

        for n, seed in ((5, 0), (9, 1), (21, 2)):
            graph = structure_to_graph(random_molecule(n, seed), cutoff)
            worst = max(worst, check_current(model, ref, exact, f"molecule of {n} atoms", graph))
        batch = collate([structure_to_graph(random_molecule(n, 10 + n), cutoff) for n in (6, 12, 17)])
        worst = max(worst, check_current(model, ref, exact, "batch of three molecules", batch))
        box = structure_to_graph(water_box(4, 0), cutoff)
        worst = max(worst, check_current(model, ref, exact, "water box of 192 atoms", box))
        cells = collate([structure_to_graph(water_box(k, k), cutoff) for k in (4, 5)])
        worst = max(worst, check_current(model, ref, exact, "batch of two cells", cells))
    return worst


def run_v1(dtype) -> float:
    worst = 0.0
    print(f"paper's model ({dtype}):")
    model = build_model(PAPER, dtype, element_scales=False)
    ref = build_v1(model)
    exact = copy.deepcopy(ref).double() if dtype == torch.float32 else None
    wide_model = copy.deepcopy(model).double() if dtype == torch.float32 else None
    for n, seed in ((5, 0), (9, 1), (21, 2)):
        mol = random_molecule(n, seed)
        graph = structure_to_graph(mol, model.cutoff)
        b = v1_outputs(ref, model, mol)
        c = v1_outputs(exact, wide_model, mol) if exact is not None else None
        worst = max(worst, compare(f"molecule of {n} atoms", xnn_outputs(model, graph), b, c,
                                   keys=("energy", "forces", "latent_forces")))
    return worst


def run(dtype) -> float:
    torch.set_default_dtype(dtype)
    version = reference_version()
    if not version:
        raise RuntimeError("the NewtonNet reference code is not installed")
    worst = run_current(dtype) if version == "current" else run_v1(dtype)
    if dtype == torch.float32:
        print(f"worst error in units of the float32 bound: {worst:.3f}")
    else:
        print(f"worst relative error ({dtype}): {worst:.3e}")
    return worst


def main():
    dtype = {"float32": torch.float32, "float64": torch.float64}[sys.argv[1] if len(sys.argv) > 1 else "float64"]
    print(run(dtype))


if __name__ == "__main__":
    main()
