"""Parity of the xnn PaiNN with the reference implementation (schnetpack).

    python painn_spk_parity.py [float32|float64] [bessel|gaussian]

Builds a randomly initialized xnn :class:`~xnn.gnn.models.painn.PaiNN` with
the dipole and polarizability heads, transplants its weights into the
reference ``schnetpack`` modules (``PaiNN`` representation, ``Atomwise``,
``DipoleMoment``, ``Polarizability``) and compares energies, forces, latent
charges, dipole moments and polarizability tensors on random molecules in
the requested precision. The last printed line is the worst relative error
over every quantity (energies relative to ``|E|``, the others to the largest
component). The reference package is used as an external oracle only.

The feature layouts differ by a fixed permutation (documented in
:func:`transplant`): the reference splits the message filters into
``(s, vs, vv)`` and the update network into ``(ss, vv, sv)``, and its
``mu_channel_mix`` puts the normed vectors first; xnn uses ``(s, vv, vs)``,
``(vv, sv, ss)`` and ``[U, V]``.
"""
import sys

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.common.models import ForceStressOutput
from xnn.gnn.models.painn import ATOMIC_MASSES, PaiNN


def build_reference(n_features, n_interactions, n_rbf, cutoff, radial_basis):
    """The schnetpack PaiNN plus its energy, dipole and polarizability outputs."""
    import schnetpack.nn as snn
    from schnetpack.atomistic import Atomwise, DipoleMoment, Polarizability
    from schnetpack.representation import PaiNN as SpkPaiNN

    if radial_basis == "bessel":
        rbf = snn.radial.BesselRBF(n_rbf=n_rbf, cutoff=cutoff)
    else:
        # the reference builds its widths through torch.FloatTensor, which only
        # accepts a float32 source; exact_widths() redoes them in the working dtype
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            rbf = snn.radial.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
        finally:
            torch.set_default_dtype(default)
    rep = SpkPaiNN(n_atom_basis=n_features, n_interactions=n_interactions,
                   radial_basis=rbf, cutoff_fn=snn.cutoff.CosineCutoff(cutoff))
    energy = Atomwise(n_in=n_features, output_key="energy", aggregation_mode="sum")
    dipole = DipoleMoment(n_in=n_features, use_vector_representation=True,
                          return_charges=True, correct_charges=True)
    alpha = Polarizability(n_in=n_features)
    return torch.nn.ModuleDict({"rep": rep, "energy": energy, "dipole": dipole,
                                "alpha": alpha})


def exact_widths(ref) -> None:
    """Recompute the reference Gaussian widths in the working dtype.

    The reference builds them through a float32 tensor, which would put a
    1e-8 relative error into a float64 comparison; the centers are rebuilt
    the same way, since a float32 build rounded them too.
    """
    rbf = ref["rep"].radial_basis
    if hasattr(rbf, "widths"):
        with torch.no_grad():
            n, cutoff = rbf.offsets.numel(), float(ref["rep"].cutoff)
            rbf.offsets.copy_(torch.linspace(0.0, cutoff, n, dtype=rbf.offsets.dtype))
            rbf.widths.copy_((rbf.offsets[1] - rbf.offsets[0]).abs() * torch.ones_like(rbf.offsets))


def _copy(dst, src):
    with torch.no_grad():
        dst.copy_(src.to(dst.dtype))


def _rows(w, F, order):
    """Reorder the three ``F``-row groups of a ``(3F, ...)`` tensor."""
    groups = torch.split(w, F, dim=0)
    return torch.cat([groups[k] for k in order], dim=0)


def transplant(model: PaiNN, ref: torch.nn.ModuleDict) -> None:
    """Copy the xnn weights into the reference modules (in place)."""
    F = model.n_features
    rep = ref["rep"]
    _copy(rep.embedding.weight, model.embedding.weight)
    # message filters: xnn (s, vv, vs) -> reference (s, vs, vv), one block after another
    filt_w, filt_b = [], []
    for block in model.interactions:
        filt_w.append(_rows(block.message.filter.weight, F, (0, 2, 1)))
        filt_b.append(_rows(block.message.filter.bias, F, (0, 2, 1)))
    _copy(rep.filter_net.weight, torch.cat(filt_w, 0))
    _copy(rep.filter_net.bias, torch.cat(filt_b, 0))
    for block, inter, mix in zip(model.interactions, rep.interactions, rep.mixing):
        phi = block.message.phi
        _copy(inter.interatomic_context_net[0].weight, phi[0].weight)
        _copy(inter.interatomic_context_net[0].bias, phi[0].bias)
        _copy(inter.interatomic_context_net[1].weight, _rows(phi[2].weight, F, (0, 2, 1)))
        _copy(inter.interatomic_context_net[1].bias, _rows(phi[2].bias, F, (0, 2, 1)))
        # update: xnn [U, V] -> reference [mu_V (normed), mu_W (scaled)] = [V, U]
        lin_v = block.update.lin_v.weight
        _copy(mix.mu_channel_mix.weight, torch.cat([lin_v[F:], lin_v[:F]], 0))
        net = block.update.net
        _copy(mix.intraatomic_context_net[0].weight, net[0].weight)
        _copy(mix.intraatomic_context_net[0].bias, net[0].bias)
        # xnn (vv, sv, ss) -> reference (ss, vv, sv)
        _copy(mix.intraatomic_context_net[1].weight, _rows(net[2].weight, F, (2, 0, 1)))
        _copy(mix.intraatomic_context_net[1].bias, _rows(net[2].bias, F, (2, 0, 1)))
    out = ref["energy"].outnet
    _copy(out[0].weight, model.readout[0].weight)
    _copy(out[0].bias, model.readout[0].bias)
    _copy(out[1].weight, model.readout[2].weight)
    _copy(out[1].bias, model.readout[2].bias)
    for head, name in ((model.dipole_head, "dipole"), (model.polarizability_head, "alpha")):
        for block, geb in zip(head.blocks, ref[name].outnet):
            _copy(geb.mix_vectors.weight, block.lin_v.weight)
            _copy(geb.scalar_net[0].weight, block.net[0].weight)
            _copy(geb.scalar_net[0].bias, block.net[0].bias)
            _copy(geb.scalar_net[1].weight, block.net[2].weight)
            _copy(geb.scalar_net[1].bias, block.net[2].bias)


def reference_outputs(ref, graph):
    """Energies, forces, charges, dipoles and polarizabilities of the reference."""
    import schnetpack.properties as properties

    R = graph.pos.clone().requires_grad_(True)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    inputs = {
        properties.Z: graph.atomic_numbers,
        properties.R: R,
        properties.Rij: R[src] - R[dst],          # r_j - r_i with j = src, i = dst
        properties.idx_i: dst,
        properties.idx_j: src,
        properties.n_atoms: graph.n_atoms,
        properties.idx_m: graph.batch,
    }
    inputs = ref["rep"](inputs)
    inputs = ref["energy"](inputs)
    inputs = ref["dipole"](inputs)
    inputs = ref["alpha"](inputs)
    energy = inputs["energy"]
    (grad,) = torch.autograd.grad(energy.sum(), R)
    return {"energy": energy.detach(), "forces": -grad.detach(),
            "charges": inputs[properties.partial_charges].detach().reshape(-1),
            "dipole": inputs[properties.dipole_moment].detach(),
            "polarizability": inputs[properties.polarizability].detach()}


def xnn_outputs(model, graph):
    out = ForceStressOutput(model)(graph)
    return {k: out[k].detach() for k in ("energy", "forces", "charges", "dipole",
                                        "polarizability")}


def random_molecule(n, seed):
    """A random H/C/O/N cluster with the center of mass at the origin."""
    rng = np.random.default_rng(seed)
    z = rng.choice([1, 1, 6, 7, 8], size=n)
    pos = rng.uniform(-2.5, 2.5, (n, 3))
    m = np.asarray(ATOMIC_MASSES)[z][:, None]
    pos = pos - (m * pos).sum(0) / m.sum()
    return {"pos": pos, "atomic_numbers": z}


def compare(model, ref, graph):
    a, b = xnn_outputs(model, graph), reference_outputs(ref, graph)
    worst = 0.0
    for key in a:
        scale = a[key].abs().max().item()
        err = (a[key].to(b[key].dtype) - b[key]).abs().max().item() / max(scale, 1e-300)
        print(f"  {key:15s} max rel err {err:.3e}  (scale {scale:.3e})")
        worst = max(worst, err)
    return worst


def main():
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        sys.argv[1] if len(sys.argv) > 1 else "float64"]
    radial_basis = sys.argv[2] if len(sys.argv) > 2 else "bessel"
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)
    F, T, n_rbf, cutoff = 32, 3, 20, 5.0
    model = PaiNN(n_features=F, n_interactions=T, n_rbf=n_rbf, cutoff=cutoff,
                  radial_basis=radial_basis, dipole=True, polarizability=True).eval()
    # random non-trivial standardization is left at identity: the reference has none
    ref = build_reference(F, T, n_rbf, cutoff, radial_basis).to(dtype).eval()
    exact_widths(ref)
    transplant(model, ref)
    worst = 0.0
    for n, seed in ((5, 0), (9, 1), (14, 2)):
        graph = structure_to_graph(random_molecule(n, seed), cutoff)
        print(f"molecule of {n} atoms ({dtype}, {radial_basis}):")
        worst = max(worst, compare(model, ref, graph))
    batch = collate([structure_to_graph(random_molecule(n, 10 + n), cutoff) for n in (6, 11)])
    print("batch of two molecules:")
    worst = max(worst, compare(model, ref, batch))
    print(f"worst relative error ({dtype}, {radial_basis}): {worst:.3e}")
    print(worst)


if __name__ == "__main__":
    main()
