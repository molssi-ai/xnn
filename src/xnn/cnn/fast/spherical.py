"""Fast path of the SphericalCNN correlations (Triton kernels).

The reference layers transform with zero-padded spectra ``(b, 2b - 1, 2b - 1)``:
the band of each FFT axis is cut out and pasted back with ``cat`` and zero fills,
the contractions over ``beta`` and the spectral products are einsums over the
padded entries (most of them zero) with layout copies around them. Here the
spectra are compact and blocked by degree (see
:mod:`~xnn.cnn.fast.spherical_kernels`): the forward contraction reads the FFT
output at the band positions and writes the blocks, and the inverse contraction
writes the whole FFT grid of the inverse transform (zeros off the band included),
both as Triton kernels. A block of degree ``l`` is the matrix of that degree's
spectral product as a view, so the product is one complex matrix product per
degree with the filter spectrum, ``(B (2l + 1), C_in (2l + 1)) @ (C_in (2l + 1),
C_out (2l + 1))`` for an ``SO(3)`` input and ``(B (2l + 1), C_in) @ (C_in, C_out (2l
+ 1))`` for an ``S^2`` one, whose result is the block of the output spectrum. The
FFTs, the filter spectra (a small matrix product of the filter values with the
basis) and everything outside the correlations stay as in the reference.

The contractions are linear and adjoint to each other, so each one's backward is
the other kernel and differentiable again; autograd differentiates the matrix
products, keeping views of the compact spectra. Force training runs through the
fused layers. The parameters and the ``state_dict`` are the reference ones.
"""
import functools
from typing import Dict

import torch
from torch import Tensor

from xnn.common.models.fast import AutoPolicy

#: Edge counts above which ``use_fast="auto"`` takes the kernels in evaluation mode (energy and
#: forces): every size, from a single molecule on they are faster than the reference.
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)

#: The same for training mode (a step with a force loss, second derivatives included).
AUTO_POLICY_TRAINING = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)


def supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Triton kernels on a CUDA device in float32 or float64."""
    from xnn.common.models.dispersion_fast._triton import supported as triton_supported
    return triton_supported(device, dtype)


def _cached(owner, name: str, sources, build):
    """``build()``, kept on ``owner`` until a tensor in ``sources`` changes (device, dtype, values)."""
    stamp = tuple((t.data_ptr(), t._version, t.device, t.dtype) for t in sources)
    cache = owner.__dict__.setdefault("_fast_cache", {})
    hit = cache.get(name)
    if hit is None or hit[0] != stamp:
        hit = (stamp, build())
        cache[name] = hit
    return hit[1]


def _plan(transform, name: str, dtype: torch.dtype, J: int, S: int, padded: bool):
    """The kernels' plan of ``transform`` with its table ``name`` (``d_weighted`` or ``d_degree``)."""
    from .spherical_kernels import SpectralPlan

    source = getattr(transform, name)
    so3 = source.dim() == 4
    L = transform.b_spec
    width = 2 * L - 1

    def build():
        # [j, l, m, n] -> [(m, n), j, l]: one (J, L) matrix per frequency
        table = (source.permute(2, 3, 0, 1).reshape(width * width, J, L) if so3
                 else source.permute(2, 0, 1))
        return table.to(dtype).contiguous()

    table = _cached(transform, f"{name}_{dtype}", (source,), build)
    return SpectralPlan(table, J, L, S, so3, padded)


def analyze(transform, x: Tensor, padded: bool = False) -> Tensor:
    """The degree-blocked spectra of grid signals: the fast counterpart of ``transform.analyze``.

    Parameters
    ----------
    transform : S2Transform or SO3Transform
        The reference transform (its tables are used).
    x : Tensor
        Real grid signals ``(..., C, 2b, 2b)`` or ``(..., C, 2b, 2b, 2b)``.
    padded : bool, optional
        Blocks padded to ``2 b_spec - 1`` entries per index (see
        :mod:`~xnn.cnn.fast.spherical_kernels`).

    Returns
    -------
    Tensor
        Spectra ``(B C K, 2)``, blocked by degree (``B`` the product of the dimensions before
        the channels).
    """
    from .spherical_kernels import to_spectrum

    size = 2 * transform.b_grid
    plan = _plan(transform, "d_weighted", x.dtype, size, size, padded)
    spec = torch.fft.fft2(x, dim=(-2, -1)) if plan.so3 else torch.fft.fft(x, dim=-1)
    return to_spectrum(torch.view_as_real(spec), plan)


def synthesize(transform, z: Tensor, lead, padded: bool = False) -> Tensor:
    """The real grid signals of degree-blocked spectra: the fast counterpart of
    ``transform.synthesize``.

    Parameters
    ----------
    transform : S2Transform or SO3Transform
        The reference transform.
    z : Tensor
        Spectra ``(B C K, 2)``.
    lead : tuple of int
        The leading dimensions ``(..., C)`` of the signals.
    padded : bool, optional
        Whether the blocks of ``z`` are padded.

    Returns
    -------
    Tensor
        ``(*lead, 2b, 2b)`` or ``(*lead, 2b, 2b, 2b)``.
    """
    from .spherical_kernels import to_grid

    size = 2 * transform.b_grid
    plan = _plan(transform, "d_degree", z.dtype, size, size, padded)
    grid = torch.view_as_complex(to_grid(z, plan, lead))
    # the inverse FFT without its 1 / size factors, which the reference multiplies back
    dims = (-2, -1) if plan.so3 else (-1,)
    return torch.fft.ifftn(grid, dim=dims, norm="forward").real


def filter_spectrum(layer, dtype: torch.dtype) -> Tensor:
    """The conjugated filter spectra ``conj(psi)`` of a correlation layer, complex
    ``(n_in, n_out, K)`` with the entries of each channel pair ordered degree by degree
    (``(k, n)`` within a degree of an ``SO(3)`` filter)."""
    real, imag = layer.basis_real, layer.basis_imag
    n_points = real.shape[0]

    def build():
        # the layer's scaling times conj of the basis at the compact entries, as (real,
        # imaginary) pairs
        r = real.reshape(n_points, -1).index_select(1, layer.compact)
        i = imag.reshape(n_points, -1).index_select(1, layer.compact)
        return (layer.scaling * torch.stack([r, -i], dim=-1).reshape(n_points, -1)).to(dtype)

    basis = _cached(layer, f"basis_{dtype}", (real, imag), build)
    weights = layer.kernel if layer.kernel.dtype == dtype else layer.kernel.to(dtype)
    return torch.view_as_complex((weights.reshape(-1, n_points) @ basis).reshape(layer.n_in, layer.n_out, -1, 2))


@functools.lru_cache(maxsize=None)
def _matrix_index(n_in: int, n_out: int, L: int, so3: bool, padded: bool) -> Tensor:
    """Positions in the flattened ``(n_in, n_out, K)`` filter spectra of the entries of the
    per-degree matrices of :func:`degree_matrices` (``n_in n_out K`` where an entry is padding)."""
    from .spherical_kernels import compact_size

    K = compact_size(L, so3)
    pair = (torch.arange(n_in)[:, None] * n_out + torch.arange(n_out)[None, :]) * K      # (i, o)
    parts = []
    for l in range(L):
        d = 2 * l + 1
        t = torch.arange(2 * L - 1) - (L - 1) if padded else torch.arange(d) - l
        ok = t.abs() <= l
        if so3:
            # rows (i, n), columns (o, k) of psi^l_{kn}
            start = l * (4 * l * l - 1) // 3
            idx = pair[:, None, :, None] + start + (t[None, None, None, :] + l) * d + (t[None, :, None, None] + l)
            ok = ok[None, :, None, None] & ok[None, None, None, :]
        else:
            # rows i, columns (o, n)
            idx = pair[:, :, None] + l * l + t[None, None, :] + l
            ok = ok[None, None, :]
        parts.append(torch.where(ok, idx, torch.full_like(idx, n_in * n_out * K)).reshape(-1))
    return torch.cat(parts)


_INDEX_ON_DEVICE: Dict[tuple, Tensor] = {}


def degree_matrices(w: Tensor, L: int, so3: bool, padded: bool = False) -> Tensor:
    """The filter spectra ``w`` of :func:`filter_spectrum` as the per-degree matrices of the
    spectral product, one gather.

    ``SO(3)`` filters give ``W_l[(i, n), (o, k)] = w[i, o, (l, k, n)]``, ``S^2`` filters
    ``W_l[i, (o, n)] = w[i, o, (l, n)]``; concatenated degree after degree, flattened
    (complex), or, ``padded``, ``(L, n_in (2L - 1), n_out (2L - 1))`` / ``(L, n_in, n_out (2L -
    1))`` with zeros outside ``|k|, |n| <= l``.
    """
    n_in, n_out = int(w.shape[0]), int(w.shape[1])
    key = (n_in, n_out, int(L), bool(so3), bool(padded), w.device)
    index = _INDEX_ON_DEVICE.get(key)
    if index is None:
        index = _matrix_index(*key[:5]).to(w.device)
        _INDEX_ON_DEVICE[key] = index
    pairs = torch.view_as_real(w).reshape(-1, 2)
    if padded:
        pairs = torch.cat([pairs, pairs.new_zeros(1, 2)])
    out = torch.view_as_complex(pairs.index_select(0, index))
    if padded:
        width = 2 * L - 1
        return out.view(L, n_in * width if so3 else n_in, n_out * width)
    return out


def spectral_product(f: Tensor, w: Tensor, L: int, batch: int, so3: bool, padded: bool = False) -> Tensor:
    """The spectral product of a correlation: a complex matrix product per degree.

    Parameters
    ----------
    f : Tensor
        Degree-blocked input spectra ``(B C_in K, 2)``: ``SO(3)`` spectra (``so3``) or ``S^2``
        ones.
    w : Tensor
        The filter's matrices of :func:`degree_matrices`.
    L : int
        Bandwidth.
    batch : int
        The batch ``B``.
    so3 : bool
        Whether the input is an ``SO(3)`` spectrum (``z^l = f^l w^l``, summed over ``(i, n)``)
        or an ``S^2`` one (``z^l_{mn} = f^l_m w^l_n``, summed over ``i``).
    padded : bool, optional
        Padded blocks (``f`` and ``w`` alike): one batched matrix product over the degrees.

    Returns
    -------
    Tensor
        Degree-blocked ``SO(3)`` spectra ``(B C_out K_SO3, 2)``, padded with ``f``.
    """
    fc = torch.view_as_complex(f)
    if padded:
        z = torch.matmul(fc.view(L, batch * (2 * L - 1), -1), w)
        return torch.view_as_real(z.reshape(-1))
    from .spherical_kernels import compact_size

    K = compact_size(L, so3)
    n_in = fc.shape[0] // (batch * K)
    n_out = w.shape[0] // (n_in * K)
    sizes = [(2 * l + 1) ** 2 if so3 else 2 * l + 1 for l in range(L)]
    # split (not slices): its backward is one concatenation instead of a zero-filled
    # full-size gradient per degree
    f_blocks = fc.split([batch * n_in * size for size in sizes])
    w_blocks = w.split([n_in * n_out * size for size in sizes])
    blocks = []
    for l, (f_l, w_l) in enumerate(zip(f_blocks, w_blocks)):
        d = 2 * l + 1
        f_l = f_l.view(batch * d, -1)
        blocks.append((f_l @ w_l.view(f_l.shape[1], -1)).reshape(-1))
    return torch.view_as_real(torch.cat(blocks))


#: Batches (atoms) up to which float32 evaluation runs the correlations on padded blocks. A
#: call on a single molecule is bound by the host: one batched product of all degrees saves
#: the launches of the per-degree products. From two molecules on, in float64 and in
#: training, the padding's extra work and memory cost more than the launches save.
PADDED_MAX_BATCH = 32


def correlation(layer, x: Tensor, so3: bool) -> Tensor:
    """The output of an ``S^2`` (``so3=False``) or ``SO(3)`` correlation layer on its input ``x``."""
    batch = x.shape[0]
    padded = batch <= PADDED_MAX_BATCH and x.dtype == torch.float32 and not layer.training
    f = analyze(layer.analysis, x, padded)
    w = degree_matrices(filter_spectrum(layer, x.dtype), layer.b_out, so3, padded)
    z = spectral_product(f, w, layer.b_out, batch, so3, padded)
    return synthesize(layer.synthesis, z, (batch, layer.n_out), padded) + layer.bias
