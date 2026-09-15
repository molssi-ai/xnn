"""Grimme D3 dispersion correction with Becke-Johnson damping.

An independent PyTorch implementation of DFT-D3(BJ) (Grimme, Antony,
Ehrlich & Krieg, J. Chem. Phys. 132, 154104, 2010; BJ damping from Grimme,
Ehrlich & Goerigk, J. Comput. Chem. 32, 1456, 2011), written for
:class:`~xnns.dnn.models.physnet.PhysNet` and verified numerically against
the TensorFlow D3 module bundled with the original PhysNet
(MMunibas/PhysNet); see ``tests/test_physnet.py``. Only BJ damping is
provided, matching PhysNet (zero damping is repulsive at short range and
was deliberately left out there).

The element-pair reference data -- the C6 interpolation table, covalent
radii pre-scaled for coordination counting, and the multipole expectation
factors entering C8 -- are Grimme's standard D3 tables, shipped compressed
in ``d3_tables.npz`` next to this file (byte-identical values to the
``tables/*.npy`` files distributed with PhysNet).

Everything operates on an edge list: ``idx_i`` holds the central atom and
``idx_j`` the neighbor of every pair, with both directions present, so each
pair energy carries a factor 1/2. Distances are expected in **bohr** and
:func:`edisp` returns per-atom energies in **hartree**. The reference
tables default to the module-level CPU copies but may be passed in
explicitly (e.g. as registered buffers) so callers control device
residency; :class:`~xnns.dnn.models.physnet.PhysNet` wraps this module
with the angstrom/eV conversions and optionally learnable ``s6/s8/a1/a2``.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

# unit conversions, with the exact values used in Grimme's reference code
d3_autoang = 0.52917726  # multiply to go from bohr to angstrom
d3_autoev = 27.21138505  # multiply to go from hartree to eV

# D3(BJ) damping parameters of the Hartree-Fock parametrization; PhysNet
# initializes its (optionally learnable) copies from these
d3_s6 = 1.0000
d3_s8 = 0.9171
d3_a1 = 0.3385
d3_a2 = 2.8830

# fixed constants of the D3 model: steepness of the coordination-number
# counting function (k1) and exponent of the C6 interpolation weights (k3)
_K1 = 16.0
_K3 = -4.0

with np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "d3_tables.npz")) as _tables:
    #: C6 reference systems, ``(Zi, Zj, ref_i, ref_j) -> (C6, CN_i, CN_j)``
    d3_c6ab = torch.from_numpy(_tables["c6ab"])  # (95, 95, 5, 5, 3)
    #: covalent radii in bohr, pre-scaled for coordination counting
    d3_rcov = torch.from_numpy(_tables["rcov"])  # (95,)
    #: element factors ``sqrt(Q)`` entering ``C8 = 3 C6 Q_i Q_j``
    d3_r2r4 = torch.from_numpy(_tables["r2r4"])  # (95,)


def _scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum edge values into ``dim_size`` per-atom bins."""
    return src.new_zeros(dim_size).index_add(0, index, src)


def _taper(r: Tensor, cutoff: float) -> Tensor:
    """C2-continuous switch: 1 up to ``cutoff - 1`` bohr, 0 beyond ``cutoff``.

    Over the final bohr the quintic smootherstep polynomial
    ``10 u^3 - 15 u^4 + 6 u^5`` (in ``u = cutoff - r``) takes the value from
    1 down to 0 with vanishing first and second derivatives at both ends.
    """
    u = cutoff - r
    ramp = u ** 3 * (10.0 + u * (6.0 * u - 15.0))
    return torch.where(u >= 1.0, torch.ones_like(u),
                       torch.where(u <= 0.0, torch.zeros_like(u), ramp))


def _ncoord(z_i: Tensor, z_j: Tensor, r: Tensor, idx_i: Tensor, n_atoms: int,
            cutoff: float | None = None, rcov: Tensor = d3_rcov) -> Tensor:
    """Fractional coordination number of every atom.

    Each neighbor contributes a logistic count
    ``1 / (1 + exp(-k1 (R_cov / r - 1)))`` that goes from ~1 inside the
    summed covalent radius ``R_cov`` to ~0 outside; the counts are summed
    per central atom.

    Parameters
    ----------
    z_i, z_j : Tensor
        Atomic numbers of the central/neighbor atom of each pair, ``(E,)``.
    r : Tensor
        Pair distances in bohr, ``(E,)``.
    idx_i : Tensor
        Central-atom index of each pair, ``(E,)``.
    n_atoms : int
        Number of atoms ``N``.
    cutoff : float or None, optional
        If given, counts are tapered to zero at this radius (bohr) so the
        coordination number is smooth under a finite neighbor list.
    rcov : Tensor, optional
        Covalent-radius table; defaults to the module-level CPU copy.

    Returns
    -------
    Tensor
        Coordination numbers, shape ``(N,)``.
    """
    radius_sum = (rcov[z_i] + rcov[z_j]).to(r.dtype)
    count = 1.0 / (1.0 + torch.exp(-_K1 * (radius_sum / r - 1.0)))
    if cutoff is not None:
        count = count * _taper(r, cutoff)
    return _scatter_add(count, idx_i, n_atoms)


def _getc6(z_i: Tensor, z_j: Tensor, cn_i: Tensor, cn_j: Tensor,
           table: Tensor) -> Tensor:
    """C6 coefficient of each pair, interpolated over reference systems.

    D3 tabulates C6 values for up to 5 x 5 reference environments per
    element pair, each tagged with the pair of coordination numbers it was
    computed at. The molecular C6 is a normalized Gaussian-weighted average
    of the reference values, with weights ``exp(k3 * d^2)`` where ``d`` is
    the Euclidean distance between the actual and the reference coordination
    numbers. Unused table slots carry a non-positive C6 and are masked out
    of both sums; a pair with no valid reference at all yields the same
    (large negative, physically inert) sentinel as the reference code.

    Parameters
    ----------
    z_i, z_j : Tensor
        Atomic numbers of each pair, ``(E,)``.
    cn_i, cn_j : Tensor
        Coordination numbers of each pair's atoms, ``(E,)``.
    table : Tensor
        C6 reference table of shape ``(95, 95, 5, 5, 3)``.

    Returns
    -------
    Tensor
        Interpolated C6 coefficients, shape ``(E,)``.
    """
    refs = table[z_i, z_j].to(cn_i.dtype)  # (E, 5, 5, 3)
    c6_ref = refs[..., 0]
    dist2 = ((refs[..., 1] - cn_i[:, None, None]) ** 2
             + (refs[..., 2] - cn_j[:, None, None]) ** 2)
    gauss = torch.exp(_K3 * dist2)
    weight = torch.where(c6_ref > 0.0, gauss, torch.zeros_like(gauss))
    norm = weight.sum(dim=(-2, -1))
    total = (weight * c6_ref).sum(dim=(-2, -1))
    # The no-reference sentinel exceeds the float32 range; build it as a
    # float64 tensor and cast so float32 saturates to -inf instead of
    # raising on the scalar-conversion path.
    sentinel = torch.tensor(-1.0e99, dtype=torch.float64,
                            device=norm.device).to(norm.dtype)
    return torch.where(norm > 0.0, total / norm, sentinel)


def edisp(Z: Tensor, r: Tensor, idx_i: Tensor, idx_j: Tensor,
          cutoff: float | None = None, s6=d3_s6, s8=d3_s8, a1=d3_a1,
          a2=d3_a2, c6ab: Tensor | None = None, rcov: Tensor | None = None,
          r2r4: Tensor | None = None) -> Tensor:
    """Per-atom D3(BJ) dispersion energy.

    Evaluates ``-1/2 sum_j [s6 C6 / (r^6 + R0^6) + s8 C8 / (r^8 + R0^8)]``
    for every atom, with the BJ damping radius ``R0 = a1 sqrt(C8/C6) + a2``.

    Parameters
    ----------
    Z : Tensor
        Atomic numbers, shape ``(N,)``.
    r : Tensor
        Pair distances in **bohr**, shape ``(E,)`` (both edge directions
        present; the factor 1/2 below accounts for the double counting).
    idx_i, idx_j : Tensor
        Central/neighbor atom index of each pair, shape ``(E,)``.
    cutoff : float or None, optional
        Long-range cutoff in bohr; ``None`` (default) applies no cutoff.
        When set, the pair energies are force-shifted so both the energy
        and its derivative go to zero smoothly at ``cutoff``, and the
        coordination numbers are tapered accordingly.
    s6, s8, a1, a2 : float or Tensor, optional
        D3(BJ) parameters, by default the Hartree-Fock values (may be
        scalar tensors, e.g. learnable parameters).
    c6ab, rcov, r2r4 : Tensor or None, optional
        Reference tables; ``None`` (default) uses the module-level CPU
        copies. Pass device-resident copies (e.g. registered buffers) when
        running on an accelerator.

    Returns
    -------
    Tensor
        Dispersion energy per atom in **hartree**, shape ``(N,)``.
    """
    if c6ab is None:
        c6ab = d3_c6ab
    if rcov is None:
        rcov = d3_rcov
    if r2r4 is None:
        r2r4 = d3_r2r4
    n_atoms = Z.shape[0]
    z_i, z_j = Z[idx_i], Z[idx_j]

    cn = _ncoord(z_i, z_j, r, idx_i, n_atoms, cutoff=cutoff, rcov=rcov)
    c6 = _getc6(z_i, z_j, cn[idx_i], cn[idx_j], c6ab)
    # C8 from C6 via the tabulated multipole factors
    c8 = 3 * c6 * r2r4[z_i].to(c6.dtype) * r2r4[z_j].to(c6.dtype)

    # BJ damping: the denominators saturate at R0^n instead of diverging
    r0 = a1 * torch.sqrt(c8 / c6) + a2
    r0_2 = r0 ** 2
    r0_6 = r0_2 ** 3
    r0_8 = r0_6 * r0_2
    r_2 = r ** 2
    r_6 = r_2 ** 3
    r_8 = r_6 * r_2

    if cutoff is None:
        f6 = 1 / (r_6 + r0_6)
        f8 = 1 / (r_8 + r0_8)
    else:
        # force-shifted form: subtract the kernel value at the cutoff plus a
        # linear term chosen so that energy and derivative vanish there
        cut_2 = cutoff ** 2
        cut_6 = cut_2 ** 3
        cut_8 = cut_6 * cut_2
        den_6 = cut_6 + r0_6
        den_8 = cut_8 + r0_8
        f6 = 1 / (r_6 + r0_6) - 1 / den_6 + 6 * cut_6 / den_6 ** 2 * (r / cutoff - 1)
        f8 = 1 / (r_8 + r0_8) - 1 / den_8 + 8 * cut_8 / den_8 ** 2 * (r / cutoff - 1)
        f6 = torch.where(r < cutoff, f6, torch.zeros_like(f6))
        f8 = torch.where(r < cutoff, f8, torch.zeros_like(f8))

    e6 = -0.5 * s6 * c6 * f6
    e8 = -0.5 * s8 * c8 * f8
    return _scatter_add(e6 + e8, idx_i, n_atoms)
