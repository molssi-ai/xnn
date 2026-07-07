"""Grimme DFT-D3 dispersion with Becke-Johnson damping, for PhysNet.

A faithful PyTorch translation of the TensorFlow implementation bundled with
the original PhysNet (MMunibas/PhysNet, ``neural_network/grimme_d3``), which
in turn ports Grimme's reference code (Grimme et al., J. Chem. Phys. 132,
154104, 2010; only BJ damping -- zero-damping introduces spurious repulsion).
The C6/covalent-radius/r2r4 reference tables are shipped compressed in
``d3_tables.npz`` (identical values to upstream's ``tables/*.npy``).

Everything operates on an edge list (pairs ``idx_i`` center / ``idx_j``
neighbor, both directions present) with distances in **bohr**, returning the
per-atom dispersion energy in **hartree**, exactly as upstream. The
:class:`~xnns.dnn.models.physnet.PhysNet` model wraps this with the
angstrom/eV conversions and (optionally) learnable ``s6/s8/a1/a2``.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

# conversion factors used in grimme d3 code
d3_autoang = 0.52917726  # bohr -> angstrom
d3_autoev = 27.21138505  # hartree -> eV

# global parameters (the values here are the standard for HF)
d3_s6 = 1.0000
d3_s8 = 0.9171
d3_a1 = 0.3385
d3_a2 = 2.8830
d3_k1 = 16.000
d3_k2 = 4 / 3
d3_k3 = -4.000
d3_maxc = 5  # maximum number of coordination complexes

_tables = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "d3_tables.npz"))
d3_c6ab = torch.from_numpy(_tables["c6ab"])    # (95, 95, 5, 5, 3)
d3_r0ab = torch.from_numpy(_tables["r0ab"])    # (95, 95)
d3_rcov = torch.from_numpy(_tables["rcov"])    # (95,)
d3_r2r4 = torch.from_numpy(_tables["r2r4"])    # (95,)


def _scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` entries into ``dim_size`` bins (TF ``segment_sum``)."""
    out = src.new_zeros(dim_size)
    return out.index_add(0, index, src)


def _smootherstep(r: Tensor, cutoff: float) -> Tensor:
    """Smooth step from 1 to 0 over the last bohr before ``cutoff``."""
    cuton = cutoff - 1
    x = (cutoff - r) / (cutoff - cuton)
    step = 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3
    return torch.where(r <= cuton, torch.ones_like(x),
                       torch.where(r >= cutoff, torch.zeros_like(x), step))


def _ncoord(Zi: Tensor, Zj: Tensor, r: Tensor, idx_i: Tensor, n_atoms: int,
            cutoff: float | None = None, k1: float = d3_k1,
            rcov: Tensor = d3_rcov) -> Tensor:
    """Fractional coordination numbers via an inverse damping function."""
    rco = (rcov[Zi] + rcov[Zj]).to(r.dtype)
    rr = rco / r
    damp = 1.0 / (1.0 + torch.exp(-k1 * (rr - 1.0)))
    if cutoff is not None:
        damp = damp * _smootherstep(r, cutoff)
    return _scatter_add(damp, idx_i, n_atoms)


def _getc6(Zi: Tensor, Zj: Tensor, nci: Tensor, ncj: Tensor,
           c6ab: Tensor, k3: float = d3_k3) -> Tensor:
    """Interpolate the C6 coefficient from the reference-system table.

    Gaussian-weighted average over the (up to 5 x 5) reference coordination
    numbers, ported statement-for-statement from upstream (including its
    ``c6mem``/``r_save`` bookkeeping) so the numerics match exactly.
    """
    c6ab_ = c6ab[Zi, Zj].to(nci.dtype)  # (E, 5, 5, 3)
    c6mem = -1.0e99 * torch.ones_like(nci)
    r_save = 1.0e99 * torch.ones_like(nci)
    rsum = torch.zeros_like(nci)
    csum = torch.zeros_like(nci)
    for i in range(d3_maxc):
        for j in range(d3_maxc):
            cn0 = c6ab_[:, i, j, 0]
            cn1 = c6ab_[:, i, j, 1]
            cn2 = c6ab_[:, i, j, 2]
            r = (cn1 - nci) ** 2 + (cn2 - ncj) ** 2
            r_save = torch.where(r < r_save, r, r_save)
            c6mem = torch.where(r < r_save, cn0, c6mem)
            tmp1 = torch.exp(k3 * r)
            rsum = rsum + torch.where(cn0 > 0.0, tmp1, torch.zeros_like(tmp1))
            csum = csum + torch.where(cn0 > 0.0, tmp1 * cn0,
                                      torch.zeros_like(tmp1))
    return torch.where(rsum > 0.0, csum / rsum, c6mem)


def edisp(Z: Tensor, r: Tensor, idx_i: Tensor, idx_j: Tensor,
          cutoff: float | None = None, s6=d3_s6, s8=d3_s8, a1=d3_a1,
          a2=d3_a2, k1: float = d3_k1, k3: float = d3_k3,
          c6ab: Tensor | None = None, rcov: Tensor | None = None,
          r2r4: Tensor | None = None) -> Tensor:
    """Per-atom D3(BJ) dispersion energy.

    Parameters
    ----------
    Z : Tensor
        Atomic numbers, shape ``(N,)``.
    r : Tensor
        Pair distances in **bohr**, shape ``(E,)`` (both edge directions
        present; each pair's energy carries the usual factor 1/2).
    idx_i, idx_j : Tensor
        Center / neighbor atom index of each pair, shape ``(E,)``.
    cutoff : float or None, optional
        Long-range cutoff in bohr; ``None`` (default) means no cutoff. With a
        cutoff the energy expression is force-shifted so both the energy and
        its derivative vanish smoothly at ``cutoff``.
    s6, s8, a1, a2 : float or Tensor, optional
        D3(BJ) functional parameters (may be learnable tensors).
    c6ab, rcov, r2r4 : Tensor or None, optional
        Reference tables; ``None`` (default) uses the module-level CPU copies.
        Pass device-resident copies (e.g. registered buffers) on GPU.

    Returns
    -------
    Tensor
        Dispersion energy per atom in **hartree**, shape ``(N,)``.
    """
    c6ab = d3_c6ab if c6ab is None else c6ab
    rcov = d3_rcov if rcov is None else rcov
    r2r4 = d3_r2r4 if r2r4 is None else r2r4
    n_atoms = Z.shape[0]
    Zi, Zj = Z[idx_i], Z[idx_j]
    nc = _ncoord(Zi, Zj, r, idx_i, n_atoms, cutoff=cutoff, k1=k1, rcov=rcov)
    nci, ncj = nc[idx_i], nc[idx_j]
    c6 = _getc6(Zi, Zj, nci, ncj, c6ab, k3=k3)
    c8 = 3 * c6 * r2r4[Zi].to(c6.dtype) * r2r4[Zj].to(c6.dtype)

    r2 = r ** 2
    r6 = r2 ** 3
    r8 = r6 * r2

    # Becke-Johnson damping
    tmp = a1 * torch.sqrt(c8 / c6) + a2
    tmp2 = tmp ** 2
    tmp6 = tmp2 ** 3
    tmp8 = tmp6 * tmp2
    if cutoff is None:
        e6 = 1 / (r6 + tmp6)
        e8 = 1 / (r8 + tmp8)
    else:
        cut2 = cutoff ** 2
        cut6 = cut2 ** 3
        cut8 = cut6 * cut2
        cut6tmp6 = cut6 + tmp6
        cut8tmp8 = cut8 + tmp8
        e6 = 1 / (r6 + tmp6) - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * (r / cutoff - 1)
        e8 = 1 / (r8 + tmp8) - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * (r / cutoff - 1)
        e6 = torch.where(r < cutoff, e6, torch.zeros_like(e6))
        e8 = torch.where(r < cutoff, e8, torch.zeros_like(e8))
    e6 = -0.5 * s6 * c6 * e6
    e8 = -0.5 * s8 * c8 * e8
    return _scatter_add(e6 + e8, idx_i, n_atoms)
