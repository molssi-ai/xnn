"""The fast path of the MACE symmetric contraction (the product basis).

The contraction is a polynomial of degree ``correlation`` in the node features
with per-element weights over a coupling basis ``U``. cuEquivariance evaluates
the same polynomial space in one fused kernel, with its own coupling basis, so
xnn's weights reach it through a fixed linear change of basis: for each output
irrep and order, the kernel's weights are ``M @ w`` with ``M`` fitted once, in
float64, by matching the two polynomials on random features (exact, as both
bases span the same space of equivariant symmetric contractions).
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from . import _cueq


def eligible(sc) -> bool:
    """Whether a :class:`~xnn.gnn.models.mace.SymmetricContraction` fits the kernel."""
    muls = {mul for mul, _ in sc.irreps_in} | {mul for mul, _ in sc.irreps_out}
    return len(muls) == 1


class SymmetricContractionKernel:
    """cuEquivariance evaluation of one ``SymmetricContraction`` on one device and dtype.

    Parameters
    ----------
    sc : xnn.gnn.models.mace.SymmetricContraction
        The reference module (its ``U`` buffers define the basis of its weights).
    device : torch.device
        The CUDA device the kernel runs on.
    dtype : torch.dtype
        float32 or float64.
    """

    def __init__(self, sc, device: torch.device, dtype: torch.dtype):
        import cuequivariance as cue
        import cuequivariance_torch as cuet

        self.correlation = sc.contractions[0].correlation
        self.mul = next(iter(sc.irreps_in)).mul
        degrees = range(1, self.correlation + 1)
        poly = cue.descriptors.symmetric_contraction(
            _cueq.cue_irreps(sc.irreps_in), _cueq.cue_irreps(sc.irreps_out), degrees)
        self.n_weights = poly.inputs[0].dim // self.mul
        self.f = cuet.SegmentedPolynomial(poly.polynomial, method="uniform_1d",
                                          math_dtype=dtype).to(device)
        self.transpose_in = cuet.TransposeIrrepsLayout(
            poly.inputs[1].irreps, source=cue.ir_mul, target=poly.inputs[1].layout, device=device)
        self.transpose_out = cuet.TransposeIrrepsLayout(
            poly.outputs[0].irreps, source=poly.outputs[0].layout, target=cue.mul_ir, device=device)
        self.basis_map = fit_basis_map(sc).to(device=device, dtype=dtype)

    def __call__(self, x: Tensor, y: Tensor, weights: List[Tensor]) -> Tensor:
        """Evaluate the contraction.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``(B, mul, coupling_dim)`` (the reference layout).
        y : torch.Tensor
            One-hot element attributes ``(B, num_elements)``.
        weights : list of torch.Tensor
            The reference weights, ``(num_elements, n_paths, mul)`` per output
            irrep and order, in the order of :func:`reference_weights`.

        Returns
        -------
        torch.Tensor
            ``(B, irreps_out.dim)`` in the reference (mul, ir) layout.
        """
        w = torch.cat(weights, dim=1)                                   # (E, K, mul)
        w = torch.einsum("ak,eku->eau", self.basis_map, w).reshape(w.shape[0], -1)
        elements = torch.argmax(y, dim=-1)
        x_in = self.transpose_in(x.transpose(1, 2).reshape(x.shape[0], -1))
        out = self.f([w, x_in], input_indices={0: elements})[0]
        return self.transpose_out(out)


def reference_weights(sc) -> List[Tensor]:
    """The reference weights, per output irrep and then per order (1, 2, ...)."""
    return [w for c in sc.contractions for w in c.weights]


@torch.no_grad()
def _reference_columns(sc, x: Tensor) -> Tensor:
    """The reference contraction as a linear map of its weights, at multiplicity one.

    Returns ``J`` of shape ``(R, out_dim, K)`` with ``out[r] = J[r] @ w`` for the
    concatenated weights ``w`` (of one element and channel) and features ``x``
    ``(R, coupling_dim)``.
    """
    blocks = []
    out_dims = []
    for c in sc.contractions:
        dim = 2 * c.lmax_out + 1
        out_dims.append(dim)
        for nu in range(1, c.correlation + 1):
            U = c._U(nu).detach().to(device="cpu", dtype=torch.float64)
            if c.lmax_out == 0:
                U = U.unsqueeze(0)                                       # (1, D, ..., D, K)
            T = torch.einsum("...ik,ri->r...k", U, x)
            for _ in range(nu - 1):
                T = torch.einsum("r...ik,ri->r...k", T, x)
            blocks.append((len(out_dims) - 1, T.reshape(x.shape[0], dim, -1)))
    offsets = [0]
    for d in out_dims:
        offsets.append(offsets[-1] + d)
    n_cols = sum(T.shape[-1] for _, T in blocks)
    J = torch.zeros(x.shape[0], offsets[-1], n_cols, dtype=torch.float64)
    col = 0
    for ci, T in blocks:
        J[:, offsets[ci]:offsets[ci + 1], col:col + T.shape[-1]] = T
        col += T.shape[-1]
    return J


@torch.no_grad()
def fit_basis_map(sc) -> Tensor:
    """The change of basis ``M`` (kernel weights ``= M @`` reference weights), in float64.

    Raises
    ------
    RuntimeError
        If the kernel's polynomials do not reproduce the reference ones.
    """
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    corr = sc.contractions[0].correlation
    irreps_in = _cueq.cue_irreps(sc.irreps_in).set_mul(1)
    irreps_out = _cueq.cue_irreps(sc.irreps_out).set_mul(1)
    poly = cue.descriptors.symmetric_contraction(irreps_in, irreps_out, range(1, corr + 1))
    n_fast = poly.inputs[0].dim
    dim_in = poly.inputs[1].dim
    f = cuet.SegmentedPolynomial(poly.polynomial, method="naive")
    gen = torch.Generator().manual_seed(0)
    out_dim = poly.outputs[0].dim
    n_samples = max(32, 4 * n_fast // out_dim + 8)
    x = torch.randn(n_samples, dim_in, generator=gen, dtype=torch.float64)
    J_ref = _reference_columns(sc, x)                                    # (R, out, K)
    # the kernel at every weight vector of the identity: class a uses weights e_a
    eye = torch.eye(n_fast, dtype=torch.float64)
    classes = torch.arange(n_fast).repeat(n_samples)
    out = f([eye, x.repeat_interleave(n_fast, dim=0)], input_indices={0: classes})[0]
    J_fast = out.reshape(n_samples, n_fast, out_dim).transpose(1, 2)    # (R, out, A)
    A = J_fast.reshape(-1, n_fast)
    B = J_ref.reshape(-1, J_ref.shape[-1])
    M = torch.linalg.lstsq(A, B).solution
    # the coupling basis is the module's own buffer: exact in a float64 model
    # (exact_float64_constants rebuilds it after a cast or a load), so the fit
    # reproduces it to ~1e-15, and float32-rounded in a float32 one, which the
    # fit reproduces to ~1e-8. The check guards against a mismatched layout,
    # which shows as an O(1) difference.
    _cueq.check_close(B, A @ M, "symmetric contraction", tol=1e-6)
    return M
