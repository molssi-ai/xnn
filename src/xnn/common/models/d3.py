"""Grimme DFT-D3 London dispersion: a geometry-dependent correction for any xnn model.

Implements the D3 model of Grimme, Antony, Ehrlich & Krieg, *J. Chem. Phys.*
**132**, 154104 (2010) with the damping functions of that paper (zero
damping) and of Grimme, Ehrlich & Goerigk, *J. Comput. Chem.* **32**, 1456
(2011) (rational Becke-Johnson damping), in pure PyTorch, as an additive
energy term that combines with every xnn model family and deploys through
every channel (PyTorch, TorchScript, ASE, LAMMPS) -- the same way
:mod:`~xnn.common.models.d4` does for the charge-dependent successor D4.

The model, equation by equation (atomic units; 2010 paper numbering)
----------------------------------------------------------------------
* **Coordination number** (eq 15): ``CN_A = sum_B 1 / (1 + exp(-k1 (k2 (R_A,cov
  + R_B,cov) / R_AB - 1)))`` with ``k1 = 16`` and the 4/3-scaled Pyykko
  covalent radii, no electronegativity factor.
* **CN-dependent C6** (eq 16): a Gaussian-weighted average over the TD-DFT
  reference pairs, ``C6^AB = sum_ij C6,ref^AB(CN_i, CN_j) L_ij / sum_ij
  L_ij`` with ``L_ij = exp(-k3 [(CN_A - CN_i)^2 + (CN_B - CN_j)^2])``,
  ``k3 = 4``; here written as a product of per-atom weights.
* **C8** (eqs 6, 9): ``C8 = 3 C6 sqrt(Q_A Q_B)`` with the tabulated
  ``sqrt(0.5 sqrt(Z) <r^4>/<r^2>)`` factors.
* **Two-body energy** (eqs 3-4): ``E = -sum_AB sum_n=6,8 s_n C_n / R^n f_n``
  with the damping function selected by ``damping``:

  - ``"zero"`` (2010 eq 4, Chai & Head-Gordon): ``f_n = 1 / (1 + 6 (R /
    (sr_n R_0^AB))^-alp_n)``, ``alp_6 = alp``, ``alp_8 = alp + 2``, with the
    tabulated pair cutoff radii ``R_0^AB`` of sec II.D;
  - ``"bj"`` (2011 eqs 5-7, rational): ``E = -sum_AB sum_n s_n C_n / (R^n +
    (a1 R_0 + a2)^n)`` with ``R_0 = sqrt(C8/C6)``;
  - ``"mzero"`` (Smith *et al.* 2016): zero damping with the shifted argument
    ``R / (sr_n R_0) + bet R_0``;
  - ``"op"`` (Witte *et al.* 2017, optimized power): ``E = -sum s_n C_n R^bet /
    (R^(n+bet) + (a1 R_0 + a2)^(n+bet))``.

* **Three-body ATM energy** (eqs 11-14): ``E = s9 sum_ABC C9 (3 cos cos cos + 1)
  / (R_AB R_BC R_CA)^3 f_d,3`` with ``C9 = sqrt(C6 C6 C6)`` and zero damping
  built on the 4/3-scaled pair cutoff radii and exponent ``alp + 2`` (off by
  default, as recommended in the paper and as in the reference code).

Fidelity
--------
An independent implementation, written from the two papers and the observed
behavior of the reference code `simple-dftd3 <https://github.com/dftd3/simple-dftd3>`_
(not copied, and never imported here). The conventions
the papers leave to the code are reproduced so that energies, gradients and
virials match ``s-dftd3`` to floating-point precision for every damping
function, molecular and periodic (``tests/test_d3.py`` and
``examples/fidelity_checks/d3_verification.ipynb``): the exponential counting
function and its 40 bohr cutoff, the highest-CN fallback of the Gaussian
weights, the 60 / 40 bohr pair / triple cutoffs with optional quintic
switching windows, and the ``triple_scale`` bookkeeping of the ATM term.

One reference-data file ships next to this module, ``d3_reference.npz``,
extracted from the reference code by ``tools/build_d3_reference.py``. It
holds two sets of reference systems, selected with ``references``:

* ``"2010"`` -- Grimme's original reference systems (Z <= 94, up to five per
  element), the tables of the original D3 codes and of the PhysNet
  TensorFlow code. Default of the legacy functional API below (:func:`edisp`,
  :func:`_ncoord`, :func:`_getc6`), which
  :class:`~xnn.dnn.models.physnet.PhysNet` and BAMBOO's D3(CSO) use, so their
  upstream parity holds for every element;
* ``"2024"`` -- the current reference code (simple-dftd3 >= 1.1.0), which
  re-parametrized Fr-Pu with up to seven references and added Am-Lr.
  Default of :class:`DFTD3`, verified against ``s-dftd3``.

The two sets are identical for Z <= 86 and stored once; the original Fr-Pu
references are kept as patches. The 95-entry radius tables of the PhysNet
code (``rcov``, ``r2r4``) are single-precision values that the reference-code
radii do not reproduce (seventh digit), so the legacy API keeps them verbatim
as literals.

Units and cutoffs
-----------------
Public interfaces take Angstrom and return eV; internally atomic units with
the CODATA-2018 factors of :mod:`~xnn.common.models.dispersion`. The D3
cutoffs (default: the upstream 60 / 40 / 40 bohr for pairs / triples / CN)
set the neighbor-list radius of :class:`D3Dispersion`; for condensed-phase
MLIP training shorter cutoffs with a switching window are the sensible
choice, the defaults reproduce ``s-dftd3`` exactly.
"""
# NOTE: no ``from __future__ import annotations`` -- TorchScript resolves the
# annotations of the exported methods at compile time.
import math
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn

from .dispersion import (
    BOHR,
    HARTREE,
    DispersionCorrection,
    evaluate_on_graph,
    gaussian_reference_weights,
    options_from_extra,
    switching_function,
    three_body_energy,
    three_body_energy_chunked,
)
from .ops import scatter_sum
from .registry import register_model

_HERE = os.path.dirname(os.path.abspath(__file__))

# legacy functional API (PhysNet / BAMBOO): D3(BJ) as in the PhysNet TF code

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

# Covalent radii (bohr, pre-scaled by 4/3 for the counting function) and
# sqrt(Q) factors of the PhysNet TensorFlow tables. They are single-precision
# values that the reference-code radii do not reproduce bit-for-bit (they
# differ by ~5e-7), so they are kept verbatim to preserve the TF parity of
# :func:`edisp`; ``d3_reference.npz`` supplies the C6 reference systems.
_LEGACY_RCOV = np.array([
    0.0, 0.80628306, 1.159032, 3.0235617, 2.3684566, 1.9401187, 1.889726,
    1.7889405, 1.5873698, 1.6125661, 1.6881553, 3.5274885, 3.1495433,
    2.8471873, 2.62042, 2.771598, 2.5700274, 2.4944384, 2.4188492, 4.434557,
    3.8802373, 3.3511143, 3.0739543, 3.048758, 2.771598, 2.6960092, 2.62042,
    2.5196347, 2.4944384, 2.544831, 2.7464018, 2.821991, 2.7464018, 2.89758,
    2.771598, 2.8723836, 2.9479725, 4.7621093, 4.20779, 3.7038631, 3.5022922,
    3.325918, 3.124347, 2.89758, 2.8471873, 2.8471873, 2.7212055, 2.89758,
    3.0991507, 3.2251322, 3.1747396, 3.1747396, 3.0991507, 3.325918,
    3.3007212, 5.266036, 4.434557, 4.081808, 3.7038631, 3.9810228, 3.9558265,
    3.93063, 3.9054337, 3.8046484, 3.8298447, 3.8046484, 3.779452, 3.7542558,
    3.7542558, 3.7290595, 3.855041, 3.6786668, 3.4518995, 3.3007212,
    3.0991507, 2.9731688, 2.9227762, 2.7967944, 2.821991, 2.8471873,
    3.325918, 3.2755249, 3.2755249, 3.4267032, 3.3007212, 3.4770958,
    3.577881, 5.0644655, 4.560539, 4.20779, 3.9810228, 3.8298447, 3.855041,
    3.8802373, 3.9054337,
], dtype=np.float32).astype(np.float64)

_LEGACY_R2R4 = np.array([
    0.0, 2.007349, 1.5663713, 5.0198693, 3.8537903, 3.644466, 3.1049283,
    2.7117524, 2.5936167, 2.3882525, 2.2152252, 6.5858555, 5.46296, 5.652167,
    4.882849, 4.2972755, 4.041089, 3.7293236, 3.4467728, 7.9776278,
    7.0762396, 6.6084404, 6.287914, 6.077287, 5.546431, 5.8049116, 5.584156,
    5.4137454, 5.284972, 5.2259283, 5.098171, 6.1214967, 5.5408373, 5.066969,
    4.870051, 4.5908966, 4.311763, 9.554617, 8.673961, 7.972102, 7.434399,
    6.5871186, 6.195362, 6.015173, 5.816234, 5.657104, 5.526407, 5.442633,
    5.582854, 7.020819, 6.4681554, 5.980891, 5.8168664, 5.5332184, 5.2547703,
    11.022045, 10.1567955, 9.351678, 9.069261, 8.972411, 8.9009285, 8.859848,
    8.8173685, 8.793177, 7.8996964, 8.805884, 8.424392, 8.542892, 8.475834,
    8.450909, 8.473393, 7.8352566, 8.207028, 7.7055907, 7.32756, 7.0388737,
    6.6897874, 6.0545006, 5.8875203, 5.706615, 5.784507, 7.797807, 7.2644386,
    6.78152, 6.6788316, 6.390243, 6.0952797, 11.791561, 11.109977, 9.513778,
    8.67197, 8.771407, 8.654027, 8.539235, 8.850247,
], dtype=np.float32).astype(np.float64)


def _load_reference() -> dict:
    """Materialize the reference-code tables of ``d3_reference.npz``."""
    with np.load(os.path.join(_HERE, "d3_reference.npz")) as f:
        return {k: f[k] for k in f.files}


_REFERENCE = _load_reference()

#: Selectable sets of reference systems: ``"2010"`` (Grimme's original, Z <= 94,
#: PhysNet / original D3 codes) and ``"2024"`` (current reference code,
#: simple-dftd3 >= 1.1.0, actinides re-parametrized and extended to Lr).
REFERENCE_SETS = ("2010", "2024")


def reference_tables(references: str = "2024") -> Dict[str, np.ndarray]:
    """Reference-system tables of one reference set (fresh copies).

    Returns ``nref (104,)`` reference counts, ``refcn (7, 104)`` reference
    coordination numbers (``-1`` in unused slots) and ``c6 (7, 7, 104, 104)``
    indexed ``[ref_i, ref_j, Z_i, Z_j]``. The ``"2010"`` set has no data for
    elements beyond Z = 94 (``nref = 0``).
    """
    if references not in REFERENCE_SETS:
        raise ValueError(f"references must be one of {REFERENCE_SETS}, got {references!r}")
    ref = _REFERENCE
    nref, refcn, c6 = ref["nref"].copy(), ref["refcn"].copy(), ref["c6"].copy()
    if references == "2010":
        zs = ref["original_z"]
        nref[zs] = ref["original_nref"]
        refcn[:, zs] = ref["original_refcn"]
        c6[:, :, zs, :] = ref["original_c6"]
        c6[:, :, :, zs] = ref["original_c6"].transpose(1, 0, 3, 2)
        beyond = int(ref["original_max_z"]) + 1
        nref[beyond:] = 0
        refcn[:, beyond:] = -1.0
        c6[:, :, beyond:, :] = 0.0
        c6[:, :, :, beyond:] = 0.0
    return {"nref": nref, "refcn": refcn, "c6": c6}


def legacy_c6_table(references: str = "2010", max_z: int = 95) -> Tensor:
    """C6 reference systems in the layout of the PhysNet TensorFlow code.

    ``(Zi, Zj, ref_i, ref_j) -> (C6, CN_i, CN_j)`` with ``-1`` in all three
    channels of an unused reference slot and zeros in the ``Z = 0`` padding
    row and column; the slot count ``R`` is the largest number of references
    of any element up to ``max_z - 1`` (5 for ``"2010"``, 7 for ``"2024"``).
    With the default arguments this is Grimme's original table, bit for bit.
    """
    tab = reference_tables(references)
    nref = tab["nref"][:max_z]
    n_slots = int(nref.max())
    valid = np.arange(n_slots)[None, :] < nref[:, None]              # (Z, R)
    mask = valid[:, None, :, None] & valid[None, :, None, :]         # (Z, Z, R, R)
    c6 = tab["c6"][:n_slots, :n_slots, :max_z, :max_z].transpose(2, 3, 0, 1)
    cn = tab["refcn"][:n_slots, :max_z].T                            # (Z, R)
    table = np.stack([
        np.where(mask, c6, -1.0),
        np.where(mask, np.broadcast_to(cn[:, None, :, None], mask.shape), -1.0),
        np.where(mask, np.broadcast_to(cn[None, :, None, :], mask.shape), -1.0),
    ], axis=-1)
    table[0] = 0.0
    table[:, 0] = 0.0
    return torch.from_numpy(np.ascontiguousarray(table))


#: Grimme's original C6 reference systems (PhysNet layout),
#: ``(Zi, Zj, ref_i, ref_j) -> (C6, CN_i, CN_j)``, shape ``(95, 95, 5, 5, 3)``
d3_c6ab = legacy_c6_table()
#: covalent radii in bohr, pre-scaled for coordination counting
d3_rcov = torch.from_numpy(_LEGACY_RCOV)  # (95,)
#: element factors ``sqrt(Q)`` entering ``C8 = 3 C6 Q_i Q_j``
d3_r2r4 = torch.from_numpy(_LEGACY_R2R4)  # (95,)


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
            cutoff: Optional[float] = None, rcov: Tensor = d3_rcov) -> Tensor:
    """Fractional coordination number of every atom (2010 paper eq 15).

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
        coordination number is smooth under a finite neighbor list (the
        PhysNet convention).
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
    numbers (2010 paper eq 16). Unused table slots carry a non-positive C6
    and are masked out of both sums; a pair with no valid reference at all
    yields the same (large negative, physically inert) sentinel as the
    reference code.

    Parameters
    ----------
    z_i, z_j : Tensor
        Atomic numbers of each pair, ``(E,)``.
    cn_i, cn_j : Tensor
        Coordination numbers of each pair's atoms, ``(E,)``.
    table : Tensor
        C6 reference table of shape ``(95, 95, R, R, 3)``; ``R = 5`` for
        :data:`d3_c6ab`, see :func:`legacy_c6_table` for the 2024 set.

    Returns
    -------
    Tensor
        Interpolated C6 coefficients, shape ``(E,)``.
    """
    refs = table[z_i, z_j].to(cn_i.dtype)  # (E, R, R, 3)
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
          cutoff: Optional[float] = None, s6=d3_s6, s8=d3_s8, a1=d3_a1,
          a2=d3_a2, c6ab: Optional[Tensor] = None, rcov: Optional[Tensor] = None,
          r2r4: Optional[Tensor] = None) -> Tensor:
    """Per-atom D3(BJ) dispersion energy, PhysNet's force-shifted variant.

    Evaluates ``-1/2 sum_j [s6 C6 / (r^6 + R0^6) + s8 C8 / (r^8 + R0^8)]``
    for every atom, with the BJ damping radius ``R0 = a1 sqrt(C8/C6) + a2``.
    This is the form PhysNet (Unke & Meuwly 2019) uses; the general D3
    model with its damping variants, three-body term and the reference
    code's cutoffs is :class:`DFTD3`.

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


# the general D3 model (reference-code semantics)

# model constants
_KCN = 16.0                    # steepness k1 of the counting function (eq 15)
_WF = 4.0                      # Gaussian exponent k3 of the CN weights (eq 16)
_RS9 = 4.0 / 3.0               # scaling of the pair radii in the ATM damping
_MAX_Z = 103                   # reference data end at Lr

# upstream default real-space cutoffs, in bohr
_CUTOFF_PAIR_AU = 60.0
_CUTOFF_TRIPLE_AU = 40.0
_CUTOFF_CN_AU = 40.0

#: PBE0 parameters of the two papers: rational damping (2011, table 2) and
#: zero damping (2010, table IV). ``s9 = 0`` as recommended in the 2010 paper
#: and as the reference code's default; set ``s9 = 1`` for D3(BJ)-ATM.
PBE0_D3BJ = {"s6": 1.0, "s8": 1.2177, "a1": 0.4145, "a2": 4.8593, "s9": 0.0, "alp": 14.0}
PBE0_D3ZERO = {"s6": 1.0, "s8": 0.928, "rs6": 1.287, "rs8": 1.0, "s9": 0.0, "alp": 14.0}

_DAMPINGS = ("bj", "zero", "mzero", "op")


class DFTD3(nn.Module):
    """The D3 dispersion model as a TorchScript-compatible energy evaluator.

    Holds the damping parameters, the model constants and the reference data
    and evaluates the geometry-dependent dispersion energy of one or several
    structures from positions, atomic numbers and a neighbor list. Positions
    are in Angstrom, energies in eV; coordination numbers and C6 coefficients
    (hartree bohr^6) are reported in atomic units.

    Parameters
    ----------
    damping : str, optional
        ``"bj"`` (rational, the 2011 default and this class's), ``"zero"``
        (the 2010 form), ``"mzero"`` or ``"op"``.
    s6, s8, s9 : float, optional
        Scaling of the C6, C8 and C9 terms; by default 1, the PBE0 value of
        the chosen damping (1.2177 for BJ, 0.928 for zero) and 0 (no
        three-body term, as upstream).
    a1, a2 : float, optional
        Rational-damping radii parameters (PBE0: 0.4145, 4.8593 bohr); used
        by ``"bj"`` and ``"op"``.
    rs6, rs8 : float, optional
        Zero-damping radii scalings (PBE0: 1.287, 1.0); used by ``"zero"``
        and ``"mzero"``.
    alp : float, optional
        Zero-damping steepness ``alpha_6`` (14; ``alpha_8 = alp + 2``), also
        setting the ATM exponent.
    bet : float, optional
        The extra parameter of ``"mzero"`` (shift) and ``"op"`` (power), 0.
    cutoff_pair, cutoff_triple, cutoff_cn : float, optional
        Real-space cutoffs in Angstrom of the two-body sum, the three-body
        sum and the coordination number; by default upstream's 60, 40 and
        40 bohr.
    switch_width_pair, switch_width_triple : float, optional
        Widths (Angstrom) of quintic switching windows at the pair / triple
        cutoffs, by default 0 (sharp, as upstream).
    trainable : bool, optional
        Make ``s6, s8, s9, a1, a2, rs6, rs8, bet`` learnable parameters, by
        default ``False``.
    checkpoint_triplets : bool, optional
        Evaluate the three-body term in recompute blocks of centers
        (:func:`~xnn.common.models.dispersion.three_body_energy_chunked`),
        by default ``True``; bounds the memory by one block at every
        derivative order at the cost of re-evaluating it in the backward
        passes. Eager only.
    triplet_chunk : int or None, optional
        Triplets per recompute block; ``None`` (default) sizes it from the
        free device memory.
    references : str, optional
        Set of reference systems: ``"2024"`` (default, the current reference
        code: Fr-Pu re-parametrized with up to seven references, Am-Lr added)
        or ``"2010"`` (Grimme's original references, Z <= 94, the tables of
        the original D3 codes and of PhysNet). Identical for Z <= 86.

    Attributes
    ----------
    cutoff : float
        The largest of the cutoffs (Angstrom), the neighbor-list radius.
    n_features : int
        Width of the per-atom descriptors ``evaluate`` reports as
        ``"node_features"`` (the coordination number and the homoatomic C6).
    references : str
        The selected set of reference systems.
    """

    bohr: float
    hartree: float
    kcn: float
    wf: float
    rs9: float
    max_z: int
    damping: str
    alp: float
    cutoff: float
    cutoff_pair: float
    cutoff_triple: float
    cutoff_cn: float
    switch_width_pair: float
    switch_width_triple: float
    n_features: int
    references: str
    checkpoint_triplets: bool
    triplet_chunk: Optional[int]

    def __init__(self, damping: str = "bj", s6: float = 1.0, s8: Optional[float] = None,
                 s9: float = 0.0, a1: float = 0.4145, a2: float = 4.8593,
                 rs6: float = 1.287, rs8: float = 1.0, alp: float = 14.0,
                 bet: float = 0.0,
                 cutoff_pair: float = _CUTOFF_PAIR_AU * BOHR,
                 cutoff_triple: float = _CUTOFF_TRIPLE_AU * BOHR,
                 cutoff_cn: float = _CUTOFF_CN_AU * BOHR,
                 switch_width_pair: float = 0.0, switch_width_triple: float = 0.0,
                 trainable: bool = False, references: str = "2024",
                 checkpoint_triplets: bool = True, triplet_chunk: Optional[int] = None):
        super().__init__()
        self.checkpoint_triplets = bool(checkpoint_triplets)
        self.triplet_chunk = triplet_chunk
        damping = damping.lower()
        if damping in ("rational", "d3bj"):
            damping = "bj"
        if damping not in _DAMPINGS:
            raise ValueError(f"damping must be one of {_DAMPINGS}, got {damping!r}")
        self.damping = damping
        if s8 is None:
            s8 = PBE0_D3BJ["s8"] if damping in ("bj", "op") else PBE0_D3ZERO["s8"]
        self.bohr, self.hartree = BOHR, HARTREE
        self.kcn, self.wf, self.rs9, self.max_z = _KCN, _WF, _RS9, _MAX_Z
        self.alp = float(alp)
        self.cutoff_pair = float(cutoff_pair)
        self.cutoff_triple = float(cutoff_triple)
        self.cutoff_cn = float(cutoff_cn)
        self.cutoff = max(self.cutoff_pair, self.cutoff_triple, self.cutoff_cn)
        self.switch_width_pair = float(switch_width_pair)
        self.switch_width_triple = float(switch_width_triple)
        self.n_features = 2

        dt = torch.get_default_dtype()
        for name, value in [("s6", s6), ("s8", s8), ("s9", s9), ("a1", a1), ("a2", a2),
                            ("rs6", rs6), ("rs8", rs8), ("bet", bet)]:
            t = torch.tensor(float(value), dtype=dt)
            if trainable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

        references = str(references)          # YAML may deliver the year as an int
        if references not in REFERENCE_SETS:
            raise ValueError(f"references must be one of {REFERENCE_SETS}, got {references!r}")
        self.references = references
        ref = _REFERENCE
        tab = reference_tables(references)
        self.register_buffer("rcov", torch.tensor(
            4.0 / 3.0 * ref["covalent_radius_aa"] / BOHR, dtype=dt), persistent=False)
        zz = np.arange(ref["r4r2_raw"].shape[0], dtype=np.float64)
        self.register_buffer("r4r2", torch.tensor(
            np.sqrt(0.5 * ref["r4r2_raw"] * np.sqrt(zz)), dtype=dt), persistent=False)
        self.register_buffer("rvdw", torch.tensor(ref["rvdw_aa"] / BOHR, dtype=dt),
                             persistent=False)
        self.register_buffer("nref", torch.tensor(tab["nref"], dtype=torch.long),
                             persistent=False)
        self.register_buffer("refcn", torch.tensor(tab["refcn"], dtype=dt),
                             persistent=False)
        self.register_buffer("c6ref", torch.tensor(tab["c6"], dtype=dt), persistent=False)

    # building blocks (atomic units; TorchScript-compatible)
    def coordination_numbers(self, z: Tensor, edge_index: Tensor, r: Tensor,
                             n_atoms: int) -> Tensor:
        """Exponential-count coordination numbers (2010 paper eq 15), ``(N,)``.

        Uses the edges within ``cutoff_cn``; ``r`` in bohr.
        """
        src, dst = edge_index[0], edge_index[1]
        rc = self.rcov[z[dst]] + self.rcov[z[src]]
        count = 1.0 / (1.0 + torch.exp(-self.kcn * (rc / r - 1.0)))
        keep = r <= self.cutoff_cn / self.bohr
        return scatter_sum(torch.where(keep, count, torch.zeros_like(count)), dst, n_atoms)

    def reference_weights(self, z: Tensor, cn: Tensor) -> Tensor:
        """Gaussian CN weights of the reference systems (eq 16), ``(N, 7)``."""
        refcn = self.refcn[:, z].t()
        n_slots = refcn.shape[1]
        valid = (torch.arange(n_slots, device=z.device)[None, :] < self.nref[z][:, None])
        return gaussian_reference_weights(cn, refcn, valid, self.wf)

    def _species_vectors(self, z: Tensor, weights: Tensor) -> Tensor:
        """``V[i, b, Z] = sum_a W_ia C6ref[a, b, Z_i, Z]`` for every atom, ``(N, 7, Zmax+1)``.

        Contracting ``V`` with the weights of a partner atom gives the pair
        C6 (eq 16); grouping the atoms by species keeps the reference-table
        gathers small.
        """
        n_atoms = z.shape[0]
        n_slots, n_z = self.refcn.shape[0], self.refcn.shape[1]
        v = torch.zeros((n_atoms, n_slots, n_z), dtype=weights.dtype, device=z.device)
        species = torch.unique(z)
        for i in range(species.shape[0]):
            zi = int(species[i])
            rows = torch.nonzero(z == zi).squeeze(1)
            block = self.c6ref[:, :, zi, :]                       # (7, 7, Zmax+1)
            v = v.index_put((rows,), torch.einsum("na,abz->nbz", weights[rows], block))
        return v

    def pair_c6(self, z: Tensor, weights: Tensor, species_vectors: Tensor,
                idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """C6 of the pairs ``(idx_i, idx_j)`` (eq 16), ``(E,)``."""
        v = species_vectors[idx_i]                                # (E, 7, Zmax+1)
        zj = z[idx_j].view(-1, 1, 1).expand(-1, v.shape[1], 1)
        v_j = torch.gather(v, 2, zj).squeeze(2)                   # (E, 7): V[i, :, Z_j]
        return (v_j * weights[idx_j]).sum(-1)

    def c6_matrix(self, z: Tensor, weights: Tensor, species_vectors: Tensor) -> Tensor:
        """Dense pair C6 matrix, ``(N, N)`` in hartree bohr^6."""
        v = species_vectors[:, :, z]                              # (N, 7, N): V[i, b, Z_j]
        return torch.einsum("jb,ibj->ij", weights, v)

    def pair_radius(self, zi: Tensor, zj: Tensor) -> Tensor:
        """Critical radius ``R_0^AB`` of the two-body damping, bohr, ``(E,)``.

        Rational and optimized-power damping build it from the C8/C6 ratio
        (2011 eq 7), zero and modified-zero damping use the tabulated pair
        cutoff radii (2010 sec II.D).
        """
        if self.damping == "bj" or self.damping == "op":
            return self.a1 * torch.sqrt(3.0 * self.r4r2[zi] * self.r4r2[zj]) + self.a2
        return self.rvdw[zi, zj]

    def two_body_energy(self, z: Tensor, edge_index: Tensor, r: Tensor, c6: Tensor,
                        n_atoms: int) -> Tensor:
        """Per-atom two-body energy in hartree for the selected damping.

        Every directed edge contributes half of its pair energy to its
        center, so the sum over atoms is the sum over unordered pairs.
        """
        src, dst = edge_index[0], edge_index[1]
        zi, zj = z[dst], z[src]
        rr = 3.0 * self.r4r2[zi] * self.r4r2[zj]                  # C8 / C6
        r0 = self.pair_radius(zi, zj)
        r2 = r * r
        r6, r8 = r2 ** 3, r2 ** 4
        if self.damping == "bj":
            t6 = 1.0 / (r6 + r0 ** 6)
            t8 = 1.0 / (r8 + r0 ** 8)
        elif self.damping == "zero":
            t6 = 1.0 / (1.0 + 6.0 * (self.rs6 * r0 / r) ** self.alp) / r6
            t8 = 1.0 / (1.0 + 6.0 * (self.rs8 * r0 / r) ** (self.alp + 2.0)) / r8
        elif self.damping == "mzero":
            t6 = 1.0 / (1.0 + 6.0 * (r / (self.rs6 * r0) + self.bet * r0) ** (-self.alp)) / r6
            t8 = 1.0 / (1.0 + 6.0 * (r / (self.rs8 * r0) + self.bet * r0) ** (-(self.alp + 2.0))) / r8
        else:  # optimized power
            rb = r ** self.bet
            ab = r0 ** self.bet
            t6 = rb / (rb * r6 + ab * r0 ** 6)
            t8 = rb / (rb * r8 + ab * r0 ** 8)
        sw = switching_function(r, self.cutoff_pair / self.bohr, self.switch_width_pair / self.bohr)
        e_pair = -0.5 * c6 * sw * (self.s6 * t6 + self.s8 * rr * t8)
        return scatter_sum(e_pair, dst, n_atoms)

    # evaluation
    @torch.jit.unused
    def _three_body_chunked(self, z: Tensor, edge_index: Tensor, edge_vec: Tensor, r: Tensor,
                            c6_mat: Tensor, n_atoms: int) -> Tensor:
        """Checkpointed ATM term (eager only), memory bounded by one block of centers."""
        return three_body_energy_chunked(z, edge_index, edge_vec, r, self.rs9 * self.rvdw,
                                         self.s9, (self.alp + 2.0) / 3.0,
                                         self.cutoff_triple / self.bohr,
                                         self.switch_width_triple / self.bohr, n_atoms,
                                         c6_mat=c6_mat, chunk=self.triplet_chunk)

    @torch.jit.export
    def evaluate(self, atomic_numbers: Tensor, pos: Tensor, edge_index: Tensor,
                 edge_vec: Tensor, batch: Tensor, num_graphs: int,
                 cell: Tensor, pbc: Tensor, total_charge: Tensor) -> Dict[str, Tensor]:
        """Dispersion energy and D3 properties of a batch of structures.

        The TorchScript-compatible core shared by
        :class:`~xnn.common.models.dispersion.DispersionCorrection` and the
        deploy wrappers; the signature matches
        :meth:`~xnn.common.models.d4.DFTD4.evaluate` (D3 has no charge
        dependence, so ``cell``, ``pbc`` and ``total_charge`` are accepted
        and ignored -- periodicity enters through the edge list). Inputs in
        Angstrom; the neighbor list must reach :attr:`cutoff`.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)`` in eV, ``"energy_2body"`` /
            ``"energy_3body"`` ``(B,)`` in eV, ``"coordination_numbers"``
            ``(N,)``, ``"c6_matrix"`` ``(N, N)`` in hartree bohr^6 (empty for
            more than 20000 atoms) and ``"node_features"`` ``(N, 2)`` (the
            CN and the homoatomic C6).
        """
        z = atomic_numbers
        n_atoms = z.shape[0]
        if bool((z > self.max_z).any()) or bool((z < 1).any()):
            raise ValueError("D3 reference data cover atomic numbers 1..103")
        if bool((self.nref[z] == 0).any()):
            raise ValueError("no D3 reference systems for an element of the input "
                             "(Z > 94 with references='2010'?)")
        vec_au = edge_vec / self.bohr
        r = torch.linalg.norm(vec_au, dim=-1)

        cn = self.coordination_numbers(z, edge_index, r, n_atoms)
        weights = self.reference_weights(z, cn)
        vectors = self._species_vectors(z, weights)

        in_pair = r <= self.cutoff_pair / self.bohr
        pair_index = edge_index[:, in_pair]
        c6_edge = self.pair_c6(z, weights, vectors, pair_index[1], pair_index[0])
        e2 = self.two_body_energy(z, pair_index, r[in_pair], c6_edge, n_atoms)
        node_energy = e2
        e3 = torch.zeros_like(e2)

        dense = n_atoms <= 20000
        c6_mat = (self.c6_matrix(z, weights, vectors) if dense
                  else torch.zeros((0, 0), dtype=r.dtype, device=r.device))
        if bool(self.s9 != 0.0):
            if not dense:
                raise ValueError("the D3 three-body term needs the dense C6 matrix "
                                 "(more than 20000 atoms); set s9 = 0")
            in_triple = r <= self.cutoff_triple / self.bohr
            if self.checkpoint_triplets and not torch.jit.is_scripting():
                e3 = self._three_body_chunked(z, edge_index[:, in_triple], vec_au[in_triple],
                                              r[in_triple], c6_mat, n_atoms)
            else:
                e3 = three_body_energy(z, edge_index[:, in_triple], vec_au[in_triple],
                                       r[in_triple], c6_mat, self.rs9 * self.rvdw, self.s9,
                                       (self.alp + 2.0) / 3.0, self.cutoff_triple / self.bohr,
                                       self.switch_width_triple / self.bohr, n_atoms)
            node_energy = node_energy + e3
        if dense:
            c6_self = c6_mat.diagonal()
        else:
            c6_self = self.pair_c6(z, weights, vectors, torch.arange(n_atoms, device=z.device),
                                   torch.arange(n_atoms, device=z.device))
        return {
            "node_energy": node_energy * self.hartree,
            "energy_2body": scatter_sum(e2, batch, num_graphs) * self.hartree,
            "energy_3body": scatter_sum(e3, batch, num_graphs) * self.hartree,
            "coordination_numbers": cn,
            "c6_matrix": c6_mat,
            "node_features": torch.stack([cn, c6_self], dim=1),
        }

    @torch.jit.ignore
    def forward(self, data) -> Dict[str, Tensor]:
        """Evaluate a (batched) :class:`~xnn.common.data.AtomicGraph`.

        Adds ``"energy"`` ``(B,)`` to the keys of :meth:`evaluate`.
        """
        return evaluate_on_graph(self, data)


_D3_KEYS = ("damping", "s6", "s8", "s9", "a1", "a2", "rs6", "rs8", "alp", "bet", "references",
            "cutoff_pair", "cutoff_triple", "cutoff_cn", "switch_width_pair",
            "switch_width_triple", "trainable", "checkpoint_triplets", "triplet_chunk")


@register_model("d3")
class D3Dispersion(DispersionCorrection):
    """DFT-D3 dispersion as an xnn potential, standalone or wrapped around a model.

    See :class:`~xnn.common.models.dispersion.DispersionCorrection` for the
    wrapper semantics and :class:`DFTD3` for the options. Enable from a config
    with ``model.extra["dispersion"] = {"name": "d3", ...}`` or wrap
    directly::

        model = D3Dispersion(build_model(cfg.model), damping="bj", s9=1.0,
                             cutoff_pair=12.0, switch_width_pair=2.0)

    Parameters
    ----------
    model : InteratomicPotential or None, optional
        The short-range model to correct; ``None`` for pure dispersion.
    **d3_options
        Keyword arguments of :class:`DFTD3`.

    Attributes
    ----------
    d3 : DFTD3
        The dispersion evaluator (alias of ``term``).
    """

    def __init__(self, model=None, **d3_options):
        super().__init__(DFTD3(**d3_options), model)

    @property
    def d3(self) -> DFTD3:
        """The :class:`DFTD3` evaluator."""
        return self.term

    @classmethod
    def from_config(cls, cfg) -> "D3Dispersion":
        """Build a standalone D3 model; :class:`DFTD3` options come from ``cfg.extra``.

        ``cfg.cutoff`` is ignored: the neighbor-list radius follows from the
        D3 cutoffs. To *correct* another model with D3 put the same keys
        (plus ``name: d3``) under that model's ``extra["dispersion"]``.
        """
        return cls(None, **options_from_extra(cfg.extra or {}, _D3_KEYS))


def d3_options_from_extra(extra: dict) -> dict:
    """Pick the :class:`DFTD3` keyword arguments out of a config ``extra`` dict."""
    return options_from_extra(extra, _D3_KEYS)
