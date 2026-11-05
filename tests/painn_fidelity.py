"""Fidelity of the xnn PaiNN to the reference implementation (schnetpack, MIT).

    python painn_fidelity.py [float32|float64]

Builds randomly initialized xnn :class:`~xnn.gnn.models.painn.PaiNN` models
with the dipole and polarizability heads, builds the reference ``schnetpack``
potential from its public classes (``NeuralNetworkPotential`` with
``PairwiseDistances``, the ``PaiNN`` representation with the Bessel basis and
the cosine cutoff, ``Atomwise``, ``DipoleMoment``, ``Polarizability``,
``Forces``), loads the xnn weights into it through its state dict and compares
energies, forces, latent charges, dipole moments and polarizability tensors
on the same neighbor lists: molecules and a batch of molecules (centers of
mass at the origin, as the paper assumes for the tensorial outputs), and
energies and forces of water boxes and a batch of cells. The models have
``F = 128`` and ``T = 3``, in four variants: the defaults (20 Gaussians to
5 Angstrom, latent charges), the paper's options (20 Bessel functions,
atomic dipoles), shared interactions, and shared interactions and filters.
The last printed line is the worst relative error; in float32, the worst
error in units of its bound (see :data:`F32_BOUND`). The reference is used as
an external oracle only.

The feature layouts differ by fixed permutations (see
:func:`reference_state_dict`): the reference orders the message filters
``(s, vs, vv)`` and the update network ``(ss, vv, sv)`` and puts ``V`` before
``U`` in its vector mixing; xnn uses ``(s, vv, vs)``, ``(vv, sv, ss)`` and
``[U, V]``. Both stack the filters of all blocks in one layer.
"""
import copy
import dataclasses
import math
import sys

import numpy as np
import torch
from torch import nn

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.painn import ATOMIC_MASSES, PaiNN

#: the defaults, with both tensorial heads
DEFAULT = dict(dipole=True, polarizability=True)
#: the paper's options: the Bessel basis and atomic dipoles (eq 13)
PAPER = dict(DEFAULT, radial_basis="bessel", atomic_dipoles=True)
#: weight sharing across the blocks
SHARED = dict(DEFAULT, shared_interactions=True)
SHARED_ALL = dict(DEFAULT, shared_interactions=True, shared_filters=True)

KEYS = ("energy", "forces", "charges", "dipole", "polarizability")

#: float32 errors pass within this bound, or within FLOOR_FACTOR times the reference's own
#: float32 error where that is larger: a small output whose terms cancel (the polarizability of
#: a few atoms) carries more float32 round-off, and how much depends on the summation order
F32_BOUND = 1e-5
FLOOR_FACTOR = 4.0


def build_model(settings, dtype, seed=0, built_in=None) -> PaiNN:
    """A random xnn PaiNN in ``dtype`` (built in ``built_in`` and cast, when given).

    The readout and head weights are scaled up so that energies, charges,
    dipoles and polarizabilities are not small.
    """
    default = torch.get_default_dtype()
    torch.set_default_dtype(built_in or dtype)
    try:
        torch.manual_seed(seed)
        model = PaiNN(**settings)
        with torch.no_grad():
            nn.init.normal_(model.readout[-1].weight, std=0.3)
            for head in (model.dipole_head, model.polarizability_head):
                if head is not None:
                    last = head[-1] if isinstance(head, nn.Sequential) else head.blocks[-1].net[-1]
                    nn.init.normal_(last.weight, std=0.3)
    finally:
        torch.set_default_dtype(default)
    return model.to(dtype).eval()


def build_reference(model: PaiNN) -> nn.Module:
    """The schnetpack potential with the sizes of ``model``, carrying its weights."""
    import schnetpack.nn as snn
    from schnetpack.atomistic import Atomwise, DipoleMoment, Forces, PairwiseDistances, Polarizability
    from schnetpack.model import NeuralNetworkPotential
    from schnetpack.representation import PaiNN as SpkPaiNN

    F = model.n_features
    n_rbf = model.filters.in_features
    bessel = hasattr(model.rbf, "freqs")
    basis = snn.radial.BesselRBF if bessel else snn.radial.GaussianRBF
    # the reference builds its buffers through float32 constructors
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        rep = SpkPaiNN(n_atom_basis=F, n_interactions=len(model.interactions),
                       radial_basis=basis(n_rbf=n_rbf, cutoff=model.cutoff),
                       cutoff_fn=snn.cutoff.CosineCutoff(model.cutoff),
                       shared_interactions=model.shared_interactions,
                       shared_filters=model.shared_filters)
        outputs = [Atomwise(n_in=F, output_key="energy")]
        if model.dipole_head is not None:
            outputs.append(DipoleMoment(n_in=F, use_vector_representation=model.atomic_dipoles,
                                        return_charges=True,
                                        correct_charges=model.correct_charges))
        if model.polarizability_head is not None:
            outputs.append(Polarizability(n_in=F))
        ref = NeuralNetworkPotential(rep, input_modules=[PairwiseDistances()],
                                     output_modules=outputs + [Forces()],
                                     do_postprocessing=False)
    finally:
        torch.set_default_dtype(default)
    dtype = model.embedding.weight.dtype
    ref = ref.to(dtype)
    ref.load_state_dict(reference_state_dict(model, ref, dtype), strict=True)
    return ref.to(model.embedding.weight.device)


def _rows(w, F, order):
    """Reorder the three ``F``-row groups of a ``(3F, ...)`` tensor."""
    groups = torch.split(w, F, dim=0)
    return torch.cat([groups[k] for k in order], dim=0)


def reference_state_dict(model: PaiNN, ref: nn.Module, dtype) -> dict:
    """The xnn weights under the reference's state-dict keys.

    The radial basis (Bessel frequencies ``n pi / r_cut``, or Gaussian
    centers and widths) and the cutoff are written exactly in the working
    dtype. Every parameter of the reference must be mapped; a buffer that is
    not is kept as built and reported.
    """
    def t(x):
        return x.detach().to("cpu", dtype)

    F = model.n_features
    n_rbf = model.filters.in_features
    rep = "representation."
    sd = {rep + "embedding.weight": t(model.embedding.weight),
          rep + "cutoff_fn.cutoff": torch.tensor([model.cutoff], dtype=dtype)}
    if hasattr(model.rbf, "freqs"):
        freqs = torch.arange(1, n_rbf + 1, dtype=torch.float64) * math.pi / model.cutoff
        sd[rep + "radial_basis.freqs"] = freqs.to(dtype)
    else:
        centers = torch.linspace(0.0, model.cutoff, n_rbf, dtype=torch.float64)
        sd[rep + "radial_basis.offsets"] = centers.to(dtype)
        sd[rep + "radial_basis.widths"] = torch.full((n_rbf,), model.cutoff / (n_rbf - 1),
                                                     dtype=torch.float64).to(dtype)
    # message filters, blocks stacked in both codes: xnn (s, vv, vs) per block ->
    # reference (s, vs, vv)
    for name in ("weight", "bias"):
        blocks = torch.split(getattr(model.filters, name), model.filter_width)
        sd[rep + "filter_net." + name] = torch.cat([t(_rows(b, F, (0, 2, 1))) for b in blocks])
    for k, block in enumerate(model.interactions):
        phi, net, lin_v = block.message.phi, block.update.net, block.update.lin_v.weight
        pre = f"{rep}interactions.{k}.interatomic_context_net."
        sd[pre + "0.weight"], sd[pre + "0.bias"] = t(phi[0].weight), t(phi[0].bias)
        sd[pre + "1.weight"] = t(_rows(phi[2].weight, F, (0, 2, 1)))
        sd[pre + "1.bias"] = t(_rows(phi[2].bias, F, (0, 2, 1)))
        pre = f"{rep}mixing.{k}."
        # xnn [U, V] -> reference [V, U]
        sd[pre + "mu_channel_mix.weight"] = t(torch.cat([lin_v[F:], lin_v[:F]], 0))
        sd[pre + "intraatomic_context_net.0.weight"] = t(net[0].weight)
        sd[pre + "intraatomic_context_net.0.bias"] = t(net[0].bias)
        # xnn (vv, sv, ss) -> reference (ss, vv, sv)
        sd[pre + "intraatomic_context_net.1.weight"] = t(_rows(net[2].weight, F, (2, 0, 1)))
        sd[pre + "intraatomic_context_net.1.bias"] = t(_rows(net[2].bias, F, (2, 0, 1)))
    for i, j in ((0, 0), (1, 2)):
        sd[f"output_modules.0.outnet.{i}.weight"] = t(model.readout[j].weight)
        sd[f"output_modules.0.outnet.{i}.bias"] = t(model.readout[j].bias)
    heads = [h for h in (model.dipole_head, model.polarizability_head) if h is not None]
    for m, head in enumerate(heads, start=1):
        if isinstance(head, nn.Sequential):                 # latent charges alone (eq 12)
            for i, lin in enumerate(head[::2]):
                sd[f"output_modules.{m}.outnet.{i}.weight"] = t(lin.weight)
                sd[f"output_modules.{m}.outnet.{i}.bias"] = t(lin.bias)
            continue
        for k, block in enumerate(head.blocks):
            pre = f"output_modules.{m}.outnet.{k}."
            sd[pre + "mix_vectors.weight"] = t(block.lin_v.weight)
            for i, j in ((0, 0), (1, 2)):
                sd[pre + f"scalar_net.{i}.weight"] = t(block.net[j].weight)
                sd[pre + f"scalar_net.{i}.bias"] = t(block.net[j].bias)
    template = ref.state_dict()
    parameters = {name for name, _ in ref.named_parameters()}
    unknown = sorted(set(sd) - set(template))
    unmapped = sorted(set(template) - set(sd))
    if unknown or parameters & set(unmapped):
        raise KeyError(f"state-dict map out of date: unknown {unknown}, unmapped {unmapped}")
    for name in unmapped:
        print(f"  reference buffer kept as built: {name} = {template[name].tolist()}")
        sd[name] = template[name]
    return sd


def reference_inputs(graph) -> dict:
    """The reference input dict for ``graph``, with fresh leaf positions.

    The reference forms ``R[j] - R[i] + offset`` with ``i`` the center
    (xnn ``dst``) and ``j`` the neighbor (xnn ``src``); xnn forms
    ``pos[dst] - pos[src] + shift @ cell``, so the offsets are minus the xnn
    image shifts.
    """
    import schnetpack.properties as properties

    src, dst = graph.edge_index[0], graph.edge_index[1]
    if graph.cell is None:
        offsets = graph.pos.new_zeros((src.numel(), 3))
    else:
        cell = graph.cell.to(graph.pos.dtype)[graph.batch[src]]
        offsets = -torch.einsum("ei,eij->ej", graph.cell_shifts.to(graph.pos.dtype), cell)
    return {
        properties.Z: graph.atomic_numbers,
        properties.R: graph.pos.detach().clone(),
        properties.idx_i: dst,
        properties.idx_j: src,
        properties.offsets: offsets,
        properties.n_atoms: graph.n_atoms,
        properties.idx_m: graph.batch,
    }


def reference_outputs(ref, graph) -> dict:
    import schnetpack.properties as properties

    out = ref(reference_inputs(graph))
    res = {"energy": out["energy"].detach(), "forces": out["forces"].detach()}
    if properties.dipole_moment in out:
        res["charges"] = out[properties.partial_charges].detach().reshape(-1)
        res["dipole"] = out[properties.dipole_moment].detach()
    if properties.polarizability in out:
        res["polarizability"] = out[properties.polarizability].detach()
    return res


def xnn_outputs(model, graph) -> dict:
    out = ForceStressOutput(model).eval()(graph)
    return {k: out[k].detach() for k in KEYS if out.get(k) is not None}


def random_molecule(n, seed):
    """A random H/C/N/O cluster, no two atoms closer than 0.9 Angstrom, center of mass at 0."""
    rng = np.random.default_rng(seed)
    pos = []
    while len(pos) < n:
        p = rng.uniform(-0.9, 0.9, 3) * n ** (1 / 3) * 1.4
        if all(np.linalg.norm(p - q) > 0.9 for q in pos):
            pos.append(p)
    z = rng.choice([1, 1, 6, 7, 8], size=n)
    pos = np.array(pos)
    m = np.asarray(ATOMIC_MASSES)[z][:, None]
    return {"pos": pos - (m * pos).sum(0) / m.sum(), "atomic_numbers": z}


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
    n = n_side ** 3
    return {"pos": np.concatenate(pos), "atomic_numbers": np.tile([8, 1, 1], n),
            "cell": np.eye(3) * n_side * spacing, "pbc": [True] * 3}


def compare(model, ref, graph, label, keys=KEYS, exact=None) -> float:
    """Worst error over ``keys``: energies relative to ``max(|E|, 1)``, the others to their largest component.

    With ``exact`` (the reference in float64), the worst error in units of its float32 bound
    instead: :data:`F32_BOUND`, or :data:`FLOOR_FACTOR` times the reference's own float32 error
    (its float32 against its float64 evaluation) where that is larger.
    """
    a, b = xnn_outputs(model, graph), reference_outputs(ref, graph)
    if exact is not None:
        wide = dataclasses.replace(graph, pos=graph.pos.double(),
                                   cell=None if graph.cell is None else graph.cell.double())
        c = reference_outputs(exact, wide)
    errors, scores = {}, {}
    for key in keys:
        scale = b[key].abs().max().item()
        if key == "energy":
            scale = max(scale, 1.0)
        errors[key] = (a[key].to(b[key].dtype) - b[key]).abs().max().item() / scale
        if exact is not None:
            floor = (b[key].double() - c[key]).abs().max().item() / scale
            scores[key] = errors[key] / max(F32_BOUND, FLOOR_FACTOR * floor)
    line = "  ".join(f"{k} {e:.2e}" for k, e in errors.items())
    if exact is not None:
        line += f"  (of bound {max(scores.values()):.2f})"
    print(f"  {label:34s} {line}")
    return max((scores or errors).values())


def model_and_reference(settings, dtype, seed=0, built_in=None):
    """A random xnn model and the reference carrying its weights."""
    model = build_model(settings, dtype, seed, built_in)
    return model, build_reference(model).eval()


def run(dtype) -> float:
    torch.set_default_dtype(dtype)
    name = str(dtype).replace("torch.", "")
    cutoff = 5.0
    worst = 0.0
    variants = (("defaults", DEFAULT), ("paper's options", PAPER),
                ("shared interactions", SHARED), ("shared interactions and filters", SHARED_ALL))
    for label, settings in variants:
        print(f"{label} ({name}):")
        model, ref = model_and_reference(settings, dtype)
        # float32: the reference's own round-off, from the same weights in float64
        exact = copy.deepcopy(ref).double() if dtype == torch.float32 else None
        for n, seed in ((5, 0), (9, 1), (21, 2)):
            graph = structure_to_graph(random_molecule(n, seed), cutoff)
            worst = max(worst, compare(model, ref, graph, f"molecule of {n} atoms", exact=exact))
        batch = collate([structure_to_graph(random_molecule(n, 10 + n), cutoff) for n in (6, 12, 17)])
        worst = max(worst, compare(model, ref, batch, "batch of three molecules", exact=exact))
        if settings in (DEFAULT, PAPER):
            ef = ("energy", "forces")
            graph = structure_to_graph(water_box(4, 0), cutoff)
            worst = max(worst, compare(model, ref, graph, "water box of 192 atoms", ef, exact))
            batch = collate([structure_to_graph(water_box(k, k), cutoff) for k in (3, 4)])
            worst = max(worst, compare(model, ref, batch, "batch of two cells", ef, exact))
        if dtype == torch.float64 and settings in (DEFAULT, PAPER):
            print(f"{label}, built in float32 and cast to float64:")
            model, ref = model_and_reference(settings, dtype, built_in=torch.float32)
            graph = structure_to_graph(random_molecule(21, 2), cutoff)
            worst = max(worst, compare(model, ref, graph, "molecule of 21 atoms"))
    if dtype == torch.float32:
        print(f"worst error in units of the float32 bound: {worst:.3f}")
    else:
        print(f"worst relative error ({name}): {worst:.3e}")
    return worst


def main():
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        sys.argv[1] if len(sys.argv) > 1 else "float64"]
    print(run(dtype))


if __name__ == "__main__":
    main()
