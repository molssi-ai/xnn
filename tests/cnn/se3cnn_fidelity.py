"""Fidelity of the xnn 3D steerable CNN to the reference implementation
(mariogeiger/se3cnn), as a harness for the tests and the benchmarks.

Both implementations sample the same steerable kernel spaces, in different
real bases of the irreducible representations. ``change_of_basis(l)`` reads
the orthogonal ``U_l`` with ``Y_xnn = c U_l Y_ref`` off sampled harmonics;
conjugated with ``U_out`` and ``U_in``, every reference basis kernel is an
xnn basis kernel times a factor (a sign and a norm), so ``transplant_conv``
divides the xnn weights by these factors. Gates, batch normalization (by
the norm for non-scalar fields), the low-pass filter and the pooling act on
whole fields, so a trunk with scalar inputs and scalar outputs needs the
change of basis only inside its convolutions.

``build_reference_trunk(model)`` builds the reference blocks of a
:class:`~xnn.cnn.models.SteerableCNN` with its weights, and
:class:`ReferencePipeline` runs the xnn featurizer, a reference trunk and the
xnn energy head, so energies and forces compare the trunks alone.
``compare_all(dtype)`` returns the worst relative error of every compared
quantity. The reference caches its sampled harmonics in ``cache/`` under the
working directory.

    python tests/cnn/se3cnn_fidelity.py            # prints both precisions
"""
from __future__ import annotations

import collections
import collections.abc
import contextlib
import io
from functools import lru_cache, partial

import torch

_PATCHED = False


def setup() -> None:
    """Import patches for the reference packages (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    import se3cnn  # noqa: F401
    _PATCHED = True


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-300))


def _quiet():
    """The reference prints a line for every cache file it writes."""
    return contextlib.redirect_stdout(io.StringIO())


def _rs(fields) -> list:
    """xnn multiplicities ``(m_0, m_1, ...)`` -> reference ``[(m, l), ...]``."""
    return [(int(m), l) for l, m in enumerate(fields)]


def _harmonics(l: int, n: int = 64):
    """Both real harmonics of order ``l`` at ``n`` random unit vectors, ``(2l + 1, n)`` each."""
    from se3cnn.SO3 import spherical_harmonics_xyz

    from xnn.cnn.models.steerable import angular_kernel_basis

    g = torch.Generator().manual_seed(l)
    pts = torch.randn(n, 3, generator=g, dtype=torch.float64)
    pts = pts / pts.norm(dim=1, keepdim=True)
    y_ref = spherical_harmonics_xyz(l, pts).to(torch.float64)
    # the 0 -> l block of the xnn kernel basis is the e3nn harmonic of order l up to a constant
    y_xnn = angular_kernel_basis(0, l, l, pts)[:, 0]
    return y_xnn, y_ref


@lru_cache(maxsize=None)
def change_of_basis(l: int) -> torch.Tensor:
    """The orthogonal ``U_l`` (float64) with ``Y_xnn = c U_l Y_ref``."""
    setup()
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    y_xnn, y_ref = _harmonics(l)
    u = torch.linalg.lstsq(y_ref.T, y_xnn.T).solution.T
    a, _, bt = torch.linalg.svd(u)
    return a @ bt                                   # the orthogonal factor of the polar decomposition


def basis_residual(l: int) -> float:
    """Relative misfit of ``Y_xnn = c U_l Y_ref`` with the best constant ``c``."""
    y_xnn, y_ref = _harmonics(l)
    rotated = change_of_basis(l) @ y_ref
    c = (y_xnn * rotated).sum() / rotated.pow(2).sum()
    return _relative(c * rotated, y_xnn)


def to_xnn_basis(x: torch.Tensor, fields) -> torch.Tensor:
    """Fields ``(B, field_dim, ...)`` in the reference basis -> the xnn basis."""
    from xnn.cnn.models.steerable import field_slices

    pieces = []
    for l, start, stop in field_slices(fields):
        part = x[:, start:stop]
        if l > 0:
            u = change_of_basis(l).to(x)
            f = part.reshape(x.shape[0], -1, 2 * l + 1, *x.shape[2:])
            part = torch.einsum("ij,bmj...->bmi...", u, f).reshape(part.shape)
        pieces.append(part)
    return torch.cat(pieces, dim=1)


@torch.no_grad()
def transplant_conv(conv, kernel) -> float:
    """Write the weights of an xnn ``SteerableConv3d`` into a reference ``SE3Kernel``.

    The reference block ``(i, j)`` conjugated into the xnn basis,
    ``U_out K_b U_in^T``, equals ``f_b`` times the xnn basis kernel ``b``;
    the reference weight is the xnn weight divided by ``f_b``. Returns the
    worst relative misfit of these proportionalities.
    """
    assert [tuple(b) for b in kernel.Rs_out] == conv.blocks_out, (kernel.Rs_out, conv.blocks_out)
    assert [tuple(b) for b in kernel.Rs_in] == conv.blocks_in, (kernel.Rs_in, conv.blocks_in)
    pieces, worst = [], 0.0
    for i, (_, l_out) in enumerate(kernel.Rs_out):
        for j, (_, l_in) in enumerate(kernel.Rs_in):
            ref_basis = getattr(kernel, f"kernel_{i}_{j}")
            name = conv._basis_keys[i * len(conv.blocks_in) + j][2]
            if ref_basis is None:
                assert name is None, (i, j)
                continue
            ours = getattr(conv, name).double().flatten(1)
            conj = torch.einsum("ij,bjkxyz,lk->bilxyz", change_of_basis(l_out),
                                ref_basis.double(), change_of_basis(l_in)).flatten(1)
            assert conj.shape == ours.shape, (i, j, conj.shape, ours.shape)
            factor = (conj * ours).sum(1) / ours.pow(2).sum(1)
            worst = max(worst, _relative(factor[:, None] * ours, conj))
            pieces.append((conv.weight[f"{i}_{j}"].double() / factor).flatten())
    kernel.weight.copy_(torch.cat(pieces).to(kernel.weight))
    return worst


@torch.no_grad()
def _copy_batch_norm(norm, ref) -> None:
    """xnn ``SteerableBatchNorm`` statistics (and affine terms) -> the reference ones."""
    ref.running_mean.copy_(norm.running_mean)
    ref.running_var.copy_(norm.running_var)
    if getattr(ref, "weight", None) is not None:
        ref.weight.copy_(norm.weight)
        ref.bias.copy_(norm.bias)


def reference_block(block, fused_norm: bool = False, bandlimit: str = "compromise",
                    shell_width: float = 0.6):
    """The reference ``GatedBlock`` of an xnn :class:`~xnn.cnn.models.GatedBlock`, weights copied.

    An xnn block with batch normalization becomes the reference ``SE3BatchNorm``
    followed by the block. ``fused_norm`` uses the reference's own
    ``normalization="batch"`` instead, its ``SE3BNConvolution``, which folds the
    mean of the scalar inputs into a bias of the convolution; the fold is exact
    without zero padding only, and the fused form has no affine terms.
    """
    setup()
    from se3cnn.batchnorm import SE3BatchNorm
    from se3cnn.blocks import GatedBlock
    from se3cnn.kernel import gaussian_window_wrapper

    conv = block.conv
    with _quiet():
        # checkpoint=False: the default recomputes the gates through torch.utils.checkpoint,
        # whose reentrant form refuses torch.autograd.grad (the forces) and double backward
        ref = GatedBlock(block.fields_in, block.fields_out, conv.kernel_size,
                         radial_window=partial(gaussian_window_wrapper, mode=bandlimit, sigma=shell_width),
                         activation=(block.act, block.gate_act), stride=block.stride,
                         padding=conv.padding, normalization="batch" if fused_norm else None,
                         bias=block.bias is not None,
                         smooth_stride=block.smooth_stride or block.stride == 1, checkpoint=False)
    transplant_conv(conv, ref.conv.kernel)
    with torch.no_grad():
        b0 = 0
        if ref.scalar_act is not None and ref.scalar_act.bias is not None:
            b0 = ref.scalar_act.bias.numel()
            ref.scalar_act.bias.copy_(block.bias[:b0])
        if ref.gate_act is not None and ref.gate_act.bias is not None:
            ref.gate_act.bias.copy_(block.bias[b0:b0 + ref.gate_act.bias.numel()])
    if block.norm is None:
        assert not fused_norm
        return ref
    if fused_norm:
        assert torch.all(block.norm.weight == 1) and torch.all(block.norm.bias == 0)
        _copy_batch_norm(block.norm, ref.conv)
        return ref
    norm = SE3BatchNorm([(m, 2 * l + 1) for l, m in enumerate(block.fields_in)],
                        eps=block.norm.eps, momentum=block.norm.momentum, affine=block.norm.affine)
    _copy_batch_norm(block.norm, norm)
    return torch.nn.Sequential(norm, ref)


class SpatialMean(torch.nn.Module):
    """Global average pooling ``(B, C, X, Y, Z) -> (B, C)``, the pooling of the reference examples."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], x.shape[1], -1).mean(-1)


def build_reference_trunk(model, bandlimit: str = "compromise", shell_width: float = 0.6):
    """The reference blocks and pooling of a :class:`~xnn.cnn.models.SteerableCNN`, its weights copied."""
    blocks = [reference_block(b, bandlimit=bandlimit, shell_width=shell_width) for b in model.blocks]
    return torch.nn.Sequential(*blocks, SpatialMean())


class ReferencePipeline(torch.nn.Module):
    """The xnn featurizer, a reference trunk and the xnn energy head of a grid potential.

    Returns the outputs of :meth:`~xnn.cnn.models.base.VoxelPotential.forward`,
    so :class:`~xnn.common.models.ForceStressOutput` adds the forces.
    """

    def __init__(self, model, trunk):
        super().__init__()
        self.model, self.trunk = model, trunk
        self.cutoff = model.cutoff

    def forward(self, data):
        m = self.model
        features = self.trunk(m.featurizer(data))
        node_energy = (m.energy_scale * m.readout(features).squeeze(-1) + m.energy_shift
                       + m.atom_ref(data.atomic_numbers).squeeze(-1))
        return {"node_energy": node_energy, "energy": m.aggregate_energy(node_energy, data),
                "node_features": features}


def random_batch(n_atoms, cutoff: float, dtype: torch.dtype, seeds=(0, 1)):
    """A batch of random H/C/N/O molecules (no two atoms closer than 0.9 Angstrom)."""
    import os
    import sys

    # the random molecules come from the PaiNN harness of the graph tests
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gnn"))
    from painn_fidelity import random_molecule
    from xnn.common.data import collate, structure_to_graph

    g = collate([structure_to_graph(random_molecule(n_atoms, s), cutoff) for s in seeds])
    g.pos = g.pos.to(dtype)
    return g


def energy_forces(module, n_atoms, cutoff, dtype):
    """Energies and forces of ``module`` (evaluation mode) on :func:`random_batch`."""
    from xnn.common.models import ForceStressOutput

    out = ForceStressOutput(module).eval()(random_batch(n_atoms, cutoff, dtype))
    return out["energy"].detach(), out["forces"].detach()


@torch.no_grad()
def _randomize(module, g, norm_affine: bool = True) -> None:
    """Nonzero biases, batch-norm terms and energy head, so every path carries signal."""
    from xnn.cnn.models import GatedBlock, SteerableBatchNorm

    def rand(t, scale, shift=0.0):
        t.copy_(shift + scale * torch.randn(t.shape, generator=g, dtype=t.dtype))

    for m in module.modules():
        if isinstance(m, GatedBlock) and m.bias is not None:
            rand(m.bias, 0.5)
        if isinstance(m, SteerableBatchNorm):
            rand(m.running_mean, 0.3)
            m.running_var.copy_(1 + torch.rand(m.running_var.shape, generator=g, dtype=m.running_var.dtype))
            if norm_affine:
                rand(m.weight, 0.3, 1.0)
                rand(m.bias, 0.3)
    readout = getattr(module, "readout", None)
    if readout is not None:
        rand(readout[-1].weight, 1.0)
        rand(readout[-1].bias, 1.0)


def compare_all(dtype: torch.dtype = torch.float64, seed: int = 0) -> dict[str, float]:
    """Worst relative errors of every compared quantity, in ``dtype``.

    Everything is built in float64 (the weights transplanted there) and then
    cast, so both implementations round the same float64 parameters.
    """
    setup()
    from se3cnn.convolution import SE3Convolution

    from xnn.cnn.models import GatedBlock, SteerableCNN, SteerableConv3d, field_dim
    from xnn.cnn.models.steerable import field_slices

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    errors = {}
    try:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        size, grid = 5, 9
        fields_in, fields_out = (2, 1, 1), (2, 2, 1)
        errors["change_of_basis"] = max(basis_residual(l) for l in (1, 2, 3))

        def field_input(fields, offset=0.0):
            """Random fields in the reference basis; ``offset`` shifts the scalar ones."""
            x = torch.randn(3, field_dim(fields), grid, grid, grid, generator=g)
            for l, start, stop in field_slices(fields):
                if l == 0:
                    x[:, start:stop] += offset
            return x.to(dtype)

        def compare(mine, theirs, x, f_in, f_out):
            return _relative(mine(to_xnn_basis(x, f_in)), to_xnn_basis(theirs(x), f_out))

        # one convolution
        conv = SteerableConv3d(fields_in, fields_out, size)
        with _quiet():
            ref = SE3Convolution(_rs(fields_in), _rs(fields_out), size, padding=size // 2)
        errors["kernel_basis"] = transplant_conv(conv, ref.kernel)
        errors["conv"] = compare(conv.to(dtype), ref.to(dtype), field_input(fields_in),
                                 fields_in, fields_out)

        # gated blocks: plain, low-pass strided, batch normalization (train, then eval)
        for key, opts in (("gated", {}), ("gated_stride", {"stride": 2}),
                          ("gated_bn", {"normalization": "batch"})):
            block = GatedBlock(fields_in, fields_out, size, activation="ssp", **opts)
            _randomize(block, g)
            ref = reference_block(block)
            block, ref = block.to(dtype), ref.to(dtype)
            x = field_input(fields_in, offset=0.7)
            errors[key] = compare(block.train(), ref.train(), x, fields_in, fields_out)
            if "normalization" in opts:
                errors[key] = max(errors[key], compare(block.eval(), ref.eval(), x, fields_in, fields_out))

        # the reference's fused SE3BNConvolution (no zero padding, unit affine terms)
        block = GatedBlock(fields_in, fields_out, size, padding=0, activation="ssp", normalization="batch")
        _randomize(block, g, norm_affine=False)
        ref = reference_block(block, fused_norm=True)
        block, ref = block.to(dtype), ref.to(dtype)
        errors["gated_bn_fused"] = compare(block.train(), ref.train(), field_input(fields_in, offset=0.7),
                                           fields_in, fields_out)

        # whole trunks on voxelized molecules, then energies and forces
        n_atoms, cutoff = 5, 3.0
        arch = dict(species=[1, 6, 7, 8], cutoff=cutoff, grid_size=grid,
                    fields=[(3, 2, 1), (4, 2, 1), (6,)], kernel_size=size)
        for key, opts in (("trunk", {}), ("trunk_bn", {"normalization": "batch"})):
            model = SteerableCNN(**arch, **opts)
            _randomize(model, g)
            trunk = build_reference_trunk(model)
            model, trunk = model.to(dtype), trunk.to(dtype)
            grids = model.featurizer(random_batch(n_atoms, cutoff, dtype))
            errors[key] = _relative(model.train().trunk(grids), trunk.train()(grids))
            if key == "trunk":
                e, f = energy_forces(model, n_atoms, cutoff, dtype)
                e_ref, f_ref = energy_forces(ReferencePipeline(model, trunk), n_atoms, cutoff, dtype)
                errors["energy"] = _relative(e, e_ref)
                errors["forces"] = _relative(f, f_ref)
    finally:
        torch.set_default_dtype(old)
    return errors


if __name__ == "__main__":
    for dtype in (torch.float32, torch.float64):
        out = compare_all(dtype)
        print(dtype, {k: f"{v:.2e}" for k, v in out.items()})
        print("worst", max(out.values()))
