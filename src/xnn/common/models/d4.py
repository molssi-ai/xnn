"""DFT-D4 London dispersion: a charge-dependent correction for any xnn model.

Implements the D4 model of Caldeweyher *et al.*, *J. Chem. Phys.* **150**,
154122 (2019) (doi:10.1063/1.5090222) in pure PyTorch, as an additive
energy term that combines with every xnn model family -- a GNN, SchNet, a
descriptor network or a classical force field -- and deploys through the same
channels (PyTorch, TorchScript, ASE, LAMMPS). In the paper's terms this is the
default *D4 model*: EEQ partial charges, BJ (rational) damping for the
two-body term and the approximate Axilrod-Teller-Muto three-body term
(``bj-eeq-atm``).

The model, equation by equation (atomic units; the paper's numbering)
----------------------------------------------------------------------
* **Coordination number** (eq 6): an error-function count of neighbors,
  weighted by a Pauling-electronegativity factor,

      CN_A = sum_B  delta^EN_AB/2 (1 + erf(-k0 (R_AB - R^cov_AB) / R^cov_AB)),
      delta^EN_AB = k1 exp(-(abs(EN_A - EN_B) + k2)^2 / k3),

  with ``k0 = 7.5``, ``k1 = 4.10451``, ``k2 = 19.08857``, ``k3 = 2 * 11.28174^2``
  and ``R^cov_AB`` the sum of the (4/3-scaled) Pyykko covalent radii.
* **EEQ partial charges** (eqs 11-16): the electronegativity-equilibration
  charges of Gaussian charge densities of width ``a_A``, obtained from the
  linear system ``[[A, 1], [1^T, 0]] [q, lambda] = [X, q_tot]`` with
  ``A_AA = J_A + 2 gamma_AA / sqrt(pi)`` (``gamma_AB = (a_A^2 + a_B^2)^-1/2``),
  ``A_AB = erf(gamma_AB R_AB) / R_AB`` and ``X_A = -EN_A + kappa_A sqrt(mCN_A)``,
  where ``mCN`` is the plain (electronegativity-free) error-function CN of
  eq 14, softly capped at 8. For periodic structures the ``1/r`` matrix is
  Ewald-summed.
* **Charge scaling** (eqs 2-4): every reference polarizability is scaled by
  ``zeta(z, z_ref) = exp(beta1 [1 - exp(gamma_A [1 - z_ref / z])])`` with
  ``z = Z_eff + q`` (``beta1 = 3``, ``gamma_A`` twice the element's chemical
  hardness).
* **Reference polarizabilities** (eq 5): the atom-in-molecule dynamic
  polarizabilities are partitioned out of the TD-DFT (PBE38/daug-def2-QZVP)
  polarizabilities of the reference systems ``A_m X_n`` by subtracting the
  charge-scaled contribution of the ``X_n`` atoms.
* **Gaussian CN weighting** (eqs 7-8): ``alpha_A(i omega) = sum_ref
  W_A,ref alpha_A,ref(i omega)`` with normalized weights
  ``sum_j^{N^s} exp(-beta2 j (CN_A - CN_A,ref)^2)`` (``beta2 = 6``; ``N^s``
  grows where reference CNs cluster, fig 4).
* **Casimir-Polder integration** (eqs 1, 9): ``C6^AB = 3/pi int alpha_A(i w)
  alpha_B(i w) dw`` on the fixed 23-point trapezoid grid.
* **Two-body energy** (eqs 18-21): ``E = -sum_AB sum_n=6,8 s_n C_n^AB /
  (R_AB^n + R_0^n)`` with ``R_0 = a1 sqrt(C8/C6) + a2`` and
  ``C8 = 3 C6 sqrt(Q_A Q_B)`` (``Q`` the ``<r^4>/<r^2>`` factors).
* **Three-body ATM energy** (eqs 22-27): ``E = s9 sum_ABC C9^ABC (3 cos
  cos cos + 1) / (R_AB R_BC R_CA)^3 / (1 + 6 (R_0^ABC / R_ABC)^16)`` with
  ``C9 = sqrt(C6 C6 C6)`` built from *neutral* (``q = 0``) polarizabilities.

Fidelity
--------
An independent implementation, written from the paper and from the
observed behavior of the reference code `dftd4 <https://github.com/dftd4/dftd4>`_
(not copied, and never imported here). The conventions
that the paper leaves to the code are reproduced so that energies, forces,
virials, charges, coordination numbers, C6 coefficients and polarizabilities
match ``dftd4`` to floating-point precision (``tests/test_d4.py`` and
``examples/fidelity_checks/d4_verification.ipynb``): the ``N^s`` bookkeeping
of eq 8 and its fallback to the highest-CN reference when every Gaussian
weight underflows, the soft CN cap in the EEQ model (and its absence in the
D4 CN), the effective nuclear charges of the reference systems, the ``q =
0`` C6 coefficients of the ATM term, the real-space cutoffs (60 / 40 / 30 /
25 bohr for pairs / triples / CN / EEQ-CN) with the optional quintic
switching windows, and the Ewald conventions of the periodic EEQ (the
automatic splitting parameter, the fixed real- and reciprocal-space windows,
the Wigner-Seitz image averaging).

The element and reference data (``d4_reference.npz`` next to this file) are
the numerical values published with the D4 method (TD-DFT polarizabilities,
reference coordination numbers and charges, element constants), extracted
from the dftd4 / multicharge / mctc-lib sources by
``tools/build_d4_reference.py``.

Units and cutoffs
-----------------
Like every xnn model the public interfaces take Angstrom and return eV;
internally the model works in atomic units with CODATA-2018 conversion
factors. The D4 cutoffs (default: the upstream 60 / 40 / 30 / 25 bohr) set
the neighbor-list radius of :class:`D4Dispersion` (``model.cutoff``), so a
wrapped short-range model receives only the edges within *its* own cutoff.
For condensed-phase MLIP training, shorter D4 cutoffs (10-15 Angstrom) with
a switching window are the sensible choice; the defaults reproduce
``dftd4`` exactly.
"""
# NOTE: deliberately no ``from __future__ import annotations``: TorchScript
# resolves the annotations of the exported methods at compile time.
import math
import os
from typing import Dict, List, Optional, Tuple

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
    edge_cell_shifts,
    three_body_energy,
    three_body_energy_chunked,
)
from .eeq import EEQReuse, EEQSystem, eeq_charges_large, ewald_alpha, reciprocal_vectors
from .ops import cell_volume, scatter_sum
from .registry import register_model

# D4 model constants
_KCN = 7.5                     # steepness k0 of the CN counting function
_K4 = 4.10451                  # electronegativity factor k1 (paper eq 6)
_K5 = 19.08857                 # k2
_K6 = 2.0 * 11.28174 ** 2      # k3
_EEQ_CN_MAX = 8.0              # soft cap of the EEQ coordination number
_EEQ_CN_REG = 1.0e-14          # regularizer of sqrt(CN) in the EEQ RHS
_WS_TOL = 0.01                 # bohr^2, tie tolerance of equivalent images
_EWALD_REP = 2                 # +-2 real / reciprocal lattice windows
_MAX_Z_EEQ = 103               # EEQ parameters exist up to Lr

# upstream default real-space cutoffs, in bohr
_CUTOFF_PAIR_AU = 60.0
_CUTOFF_TRIPLE_AU = 40.0
_CUTOFF_CN_AU = 30.0
_CUTOFF_EEQ_CN_AU = 25.0
_MIN_CUTOFF_EEQ_AU = 20.0      # shortest real-space range of the large-regime EEQ split
#: ``regime="auto"`` switches from the dense (bit-exact) EEQ path to the
#: large-system operator above these atom counts (dense memory: about 7 kB
#: per pair for a periodic cell, 64 B per pair for a molecule)
AUTO_LARGE_PERIODIC = 1500
AUTO_LARGE_MOLECULAR = 6000
REGIMES = ("auto", "dense", "large")

# PBE0-D4 (bj-eeq-atm) damping parameters, paper / dftd4 parameter set
PBE0_D4 = {"s6": 1.0, "s8": 1.20065498, "a1": 0.40085597, "a2": 5.02928789,
           "s9": 1.0, "alp": 16.0}

# Casimir-Polder frequency grid (hartree) and trapezoid weights (paper eq 9)
_FREQ = [0.000001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
_CP_WEIGHTS = [0.5 * ((_FREQ[k] - _FREQ[k - 1] if k > 0 else 0.0)
                      + (_FREQ[k + 1] - _FREQ[k] if k < len(_FREQ) - 1 else 0.0))
               for k in range(len(_FREQ))]


def _load_reference() -> dict:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "d4_reference.npz")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _zeta_np(a: float, c: np.ndarray, qref: np.ndarray, qmod: np.ndarray) -> np.ndarray:
    """Charge-scaling function of paper eq 2 (NumPy, for the reference setup)."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scaled = np.exp(a * (1.0 - np.exp(c * (1.0 - qref / qmod))))
    return np.where(qmod < 0.0, np.exp(a), scaled)


def _erf_count(r: Tensor, rc: Tensor, kcn: float) -> Tensor:
    """Error-function counting function ``(1 + erf(-k0 (r - rc) / rc)) / 2``."""
    return 0.5 * (1.0 + torch.erf(-kcn * (r - rc) / rc))


def _ewald_rec_term(g: float, alpha: float, vol: float) -> float:
    """Magnitude of a reciprocal-space Ewald term at wave-vector length ``g``."""
    return 4.0 * math.pi * math.exp(-0.25 * g * g / (alpha * alpha)) / (vol * g * g)


def _ewald_dir_term(r: float, alpha: float) -> float:
    """Magnitude of a real-space Ewald term at distance ``r``."""
    return math.erfc(alpha * r) / r


def _ewald_balance(alpha: float, rlen: float, dlen: float, vol: float) -> float:
    """Decay of the reciprocal sum (4 -> 5 shortest G) minus that of the real
    sum (2 -> 3 shortest lattice vectors); its root is the splitting parameter."""
    return ((_ewald_rec_term(4.0 * rlen, alpha, vol) - _ewald_rec_term(5.0 * rlen, alpha, vol))
            - (_ewald_dir_term(2.0 * dlen, alpha) - _ewald_dir_term(3.0 * dlen, alpha)))


def _ewald_alpha(rlen: float, dlen: float, vol: float) -> float:
    """Ewald splitting parameter balancing real- and reciprocal-space decay.

    Doubles ``alpha`` from 1e-8 until the balance changes sign, then bisects
    to ``sqrt(eps)``; the fixed +-2 summation windows of the periodic EEQ
    matrix are converged to that tolerance with this choice (the upstream
    recipe; 0.25 is its fallback when the search fails).
    """
    tol = 1.4901161193847656e-08
    alpha = 1.0e-8
    d = _ewald_balance(alpha, rlen, dlen, vol)
    n_up = 0
    while d < -tol and n_up < 2000:
        alpha = 2.0 * alpha
        d = _ewald_balance(alpha, rlen, dlen, vol)
        n_up += 1
    if n_up == 0 or n_up >= 2000:
        return 0.25
    left = 0.5 * alpha
    n_up = 0
    while d < tol and n_up < 2000:
        alpha = 2.0 * alpha
        d = _ewald_balance(alpha, rlen, dlen, vol)
        n_up += 1
    if n_up >= 2000:
        return 0.25
    right = alpha
    alpha = 0.5 * (left + right)
    d = _ewald_balance(alpha, rlen, dlen, vol)
    n_bisect = 0
    while abs(d) > tol and n_bisect <= 30:
        if d < 0.0:
            left = alpha
        else:
            right = alpha
        alpha = 0.5 * (left + right)
        d = _ewald_balance(alpha, rlen, dlen, vol)
        n_bisect += 1
    if n_bisect > 30:
        return 0.25
    return alpha


def c6_matrix(alpha_iw: Tensor) -> Tensor:
    """Pairwise ``C6`` from dynamic polarizabilities (paper eqs 1, 9).

    Parameters
    ----------
    alpha_iw : Tensor
        Atom-in-molecule dynamic polarizabilities on the 23-point imaginary
        frequency grid, shape ``(N, 23)``, in bohr^3.

    Returns
    -------
    Tensor
        ``C6`` coefficients for all pairs, shape ``(N, N)``, in hartree bohr^6.
    """
    w = torch.tensor(_CP_WEIGHTS, dtype=alpha_iw.dtype, device=alpha_iw.device)
    return (3.0 / math.pi) * (alpha_iw * w) @ alpha_iw.T


class DFTD4(nn.Module):
    """The D4 dispersion model as a TorchScript-compatible energy evaluator.

    Holds the damping parameters, the model constants and the reference
    data, and evaluates the charge- and geometry-dependent dispersion energy
    of one or several structures from positions, atomic numbers and a
    neighbor list. Positions are in Angstrom, energies in eV; coordination
    numbers, charges (e), polarizabilities (bohr^3) and C6 coefficients
    (hartree bohr^6) are reported in the customary atomic units of the D4
    literature.

    Parameters
    ----------
    s6, s8, a1, a2, s9, alp : float, optional
        BJ / ATM damping parameters, by default the PBE0-D4 values of the
        paper (``s6 = 1``, ``s8 = 1.20065498``, ``a1 = 0.40085597``,
        ``a2 = 5.02928789``, ``s9 = 1``, ``alp = 16``). ``s9 = 0`` switches the
        three-body term off.
    ga, gc, wf : float, optional
        Model constants: charge-scaling height ``beta1`` (3.0), charge-scaling
        steepness multiplier of the chemical hardness (2.0), and the
        Gaussian weighting exponent ``beta2`` (6.0).
    cutoff_pair, cutoff_triple, cutoff_cn, cutoff_eeq_cn : float, optional
        Real-space cutoffs in Angstrom of the two-body sum, the three-body sum,
        the D4 coordination number and the EEQ coordination number; by default
        the upstream 60, 40, 30 and 25 bohr.
    switch_width_pair, switch_width_triple : float, optional
        Widths (Angstrom) of quintic switching windows ending at the pair /
        triple cutoffs, by default 0 (sharp cutoffs, as upstream). A window
        of a few Angstrom keeps energy and forces continuous under a finite
        cutoff, which matters when D4 supplements an MLIP in MD.
    trainable : bool, optional
        Make ``s6, s8, a1, a2, s9`` learnable ``nn.Parameter``s, by default
        ``False`` (fixed buffers).
    regime : str, optional
        How the EEQ charges are computed: ``"dense"`` builds the ``(N, N)``
        interaction matrix as the reference code does (bit-exact
        ``dftd4`` parity; memory grows as ``N^2``), ``"large"`` uses the
        matrix-free operator, iterative / factorized solve and implicit
        differentiation of :mod:`~xnn.common.models.eeq` (memory ``O(E + N
        N_G)``, agreement with ``dense`` to about 1e-10 hartree, eager
        only), and ``"auto"`` (default) picks ``large`` above
        :data:`AUTO_LARGE_PERIODIC` / :data:`AUTO_LARGE_MOLECULAR` atoms.
        TorchScript always runs ``dense``.
    cutoff_eeq : float or None, optional
        Real-space range (Angstrom) of the large-regime Ewald split, which
        sets the splitting parameter (a longer range means fewer reciprocal
        vectors). By default the largest of the other cutoffs, so the
        neighbor list is not widened; the large regime needs at least 20 bohr
        (10.6 A), below which the ``erf(gamma r)`` part of the kernel is not
        screened, and raises otherwise.
    eeq_solver : str, optional
        Large-regime solver: ``"auto"`` (LU of the assembled matrix up to
        :data:`~xnn.common.models.eeq.LU_MAX_ATOMS` atoms, conjugate
        gradients above), ``"lu"`` or ``"cg"``.
    checkpoint_triplets : bool, optional
        Evaluate the three-body term in recompute blocks of centers
        (:func:`~xnn.common.models.dispersion.three_body_energy_chunked`,
        :mod:`~xnn.common.models.recompute`), by default ``True``: memory is
        bounded by one block at every derivative order (force training
        included) and no ``(N, N)`` C6 matrix is needed, at the cost of
        re-evaluating each block once per order of differentiation. Eager
        only; TorchScript uses the plain chunk loop.
    recompute_pairs : bool, optional
        Evaluate the two-body term in recompute blocks of edges, by default
        ``True``: the per-edge ``(E, 23)`` polarizability products of the
        Casimir-Polder C6 (about 1.5 kB per edge when retained for the
        backward pass, 14 million edges at 20 000 atoms with a 12 A cutoff)
        are then transient. Eager only.
    triplet_chunk : int or None, optional
        Triplets per three-body recompute block; ``None`` (default) sizes it
        from the free device memory (2^20 to 2^24). Larger blocks amortize
        the per-block set-up, which is paid in both passes.
    triplet_cache : float or None, optional
        Keep each three-body block's triples from the forward to the backward
        pass so they are enumerated once per step: a budget in GB, ``None``
        (default) for a quarter of the free device memory, 0 to disable.
        Exact (the backward pass uses the same triples); 14 bytes per triple,
        e.g. 0.5 GB for 5000 water atoms at an 8 A triple cutoff. Eager only.

    Notes
    -----
    For molecular dynamics and geometry optimization,
    :meth:`enable_eeq_reuse` carries the large-regime EEQ solve from one
    step to the next (see :class:`~xnn.common.models.eeq.EEQReuse`).

    Attributes
    ----------
    cutoff : float
        The largest of the cutoffs (Angstrom) -- the neighbor-list radius
        this module needs.
    regime : str
        The requested regime (``"auto"``, ``"dense"`` or ``"large"``).
    """

    bohr: float
    hartree: float
    kcn: float
    k4: float
    k5: float
    k6: float
    eeq_cn_max: float
    eeq_cn_reg: float
    ws_tol: float
    ewald_rep: int
    max_z: int
    cutoff: float
    cutoff_pair: float
    cutoff_triple: float
    cutoff_cn: float
    cutoff_eeq_cn: float
    switch_width_pair: float
    switch_width_triple: float
    ga: float
    gc: float
    wf: float
    alp: float
    max_ngw: int
    n_features: int
    regime: str
    cutoff_eeq: float
    eeq_solver: str
    checkpoint_triplets: bool
    recompute_pairs: bool
    triplet_chunk: Optional[int]
    triplet_cache: Optional[float]
    auto_large_periodic: int
    auto_large_molecular: int

    def __init__(self, s6: float = 1.0, s8: float = 1.20065498,
                 a1: float = 0.40085597, a2: float = 5.02928789,
                 s9: float = 1.0, alp: float = 16.0,
                 ga: float = 3.0, gc: float = 2.0, wf: float = 6.0,
                 cutoff_pair: float = _CUTOFF_PAIR_AU * BOHR,
                 cutoff_triple: float = _CUTOFF_TRIPLE_AU * BOHR,
                 cutoff_cn: float = _CUTOFF_CN_AU * BOHR,
                 cutoff_eeq_cn: float = _CUTOFF_EEQ_CN_AU * BOHR,
                 switch_width_pair: float = 0.0,
                 switch_width_triple: float = 0.0,
                 trainable: bool = False, regime: str = "auto",
                 cutoff_eeq: Optional[float] = None, eeq_solver: str = "auto",
                 checkpoint_triplets: bool = True, recompute_pairs: bool = True,
                 triplet_chunk: Optional[int] = None,
                 triplet_cache: Optional[float] = None):
        super().__init__()
        regime = str(regime).lower()
        if regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
        if eeq_solver not in ("auto", "lu", "cg"):
            raise ValueError(f"eeq_solver must be 'auto', 'lu' or 'cg', got {eeq_solver!r}")
        self.regime = regime
        self.auto_large_periodic, self.auto_large_molecular = AUTO_LARGE_PERIODIC, AUTO_LARGE_MOLECULAR
        self.eeq_solver = str(eeq_solver)
        self.checkpoint_triplets = bool(checkpoint_triplets)
        self.recompute_pairs = bool(recompute_pairs)
        self.triplet_chunk = triplet_chunk
        self.triplet_cache = None if triplet_cache is None else float(triplet_cache)
        # model constants, held on the instance so the exported methods can
        # read them under TorchScript
        self.bohr, self.hartree = BOHR, HARTREE
        self.kcn, self.k4, self.k5, self.k6 = _KCN, _K4, _K5, _K6
        self.eeq_cn_max, self.eeq_cn_reg = _EEQ_CN_MAX, _EEQ_CN_REG
        self.ws_tol, self.ewald_rep = _WS_TOL, _EWALD_REP
        self.max_z = _MAX_Z_EEQ
        self.n_features = 3
        self.ga, self.gc, self.wf, self.alp = float(ga), float(gc), float(wf), float(alp)
        self.cutoff_pair = float(cutoff_pair)
        self.cutoff_triple = float(cutoff_triple)
        self.cutoff_cn = float(cutoff_cn)
        self.cutoff_eeq_cn = float(cutoff_eeq_cn)
        # the large-regime real-space range defaults to the neighbor-list
        # radius the other terms need, so it never widens the graph on its own
        base_cutoff = max(self.cutoff_pair, self.cutoff_triple,
                          self.cutoff_cn, self.cutoff_eeq_cn)
        self.cutoff_eeq = float(cutoff_eeq) if cutoff_eeq is not None else base_cutoff
        self.cutoff = max(base_cutoff, self.cutoff_eeq)
        self.switch_width_pair = float(switch_width_pair)
        self.switch_width_triple = float(switch_width_triple)

        dt = torch.get_default_dtype()
        for name, value in [("s6", s6), ("s8", s8), ("a1", a1), ("a2", a2),
                            ("s9", s9)]:
            t = torch.tensor(float(value), dtype=dt)
            if trainable:
                setattr(self, name, nn.Parameter(t))
            else:
                self.register_buffer(name, t)

        ref = _load_reference()
        # element tables, index = atomic number (row 0 unused)
        self.register_buffer("rcov", torch.tensor(
            4.0 / 3.0 * ref["covalent_radius_aa"] / BOHR, dtype=dt), persistent=False)
        self.register_buffer("en", torch.tensor(ref["pauling_en"], dtype=dt),
                             persistent=False)
        self.register_buffer("zeff", torch.tensor(ref["zeff"], dtype=dt),
                             persistent=False)
        self.register_buffer("hardness", torch.tensor(ref["hardness"], dtype=dt),
                             persistent=False)
        z = np.arange(ref["r4r2_raw"].shape[0], dtype=np.float64)
        self.register_buffer("r4r2", torch.tensor(
            np.sqrt(0.5 * ref["r4r2_raw"] * np.sqrt(z)), dtype=dt), persistent=False)
        for key in ("eeq_chi", "eeq_eta", "eeq_kcnchi", "eeq_rad"):
            self.register_buffer(key, torch.tensor(
                np.nan_to_num(ref[key], nan=0.0), dtype=dt), persistent=False)
        # reference systems: (7, 119) tables, 7 = max references per element
        self.register_buffer("nref", torch.tensor(ref["nref"], dtype=torch.long),
                             persistent=False)
        self.register_buffer("ngw", torch.tensor(ref["ngw"], dtype=torch.long),
                             persistent=False)
        self.max_ngw = int(ref["ngw"].max())
        self.register_buffer("refcn", torch.tensor(ref["refcn"], dtype=dt),
                             persistent=False)
        self.register_buffer("refq", torch.tensor(ref["refq"], dtype=dt),
                             persistent=False)
        # charge-scaled, partitioned reference polarizabilities (paper eq 5)
        alpha = self._reference_polarizabilities(ref)
        self.register_buffer("refalpha", torch.tensor(alpha, dtype=dt),
                             persistent=False)
        self.register_buffer("cp_weights", torch.tensor(_CP_WEIGHTS, dtype=dt),
                             persistent=False)

    # setup
    def _reference_polarizabilities(self, ref: dict) -> np.ndarray:
        """Atom-in-molecule reference polarizabilities ``alpha_A,ref(i w)``.

        Paper eq 5: from the TD-DFT polarizability of the reference molecule
        ``A_m X_n`` subtract ``n`` charge-scaled polarizabilities of the
        secondary system ``X_l`` (its charge in the reference is ``refh``),
        scale by ``1/m`` and clamp at zero. Returns shape ``(23, 7, 119)``.
        """
        refsys = ref["refsys"]                       # (7, 119) secondary index
        sys_present = refsys > 0
        is_ = np.where(sys_present, refsys, 0)
        zeff_x = ref["zeff"][is_]                    # (7, 119)
        hard_x = ref["hardness"][is_] * self.gc
        scale = _zeta_np(self.ga, hard_x, zeff_x, ref["refh"] + zeff_x)
        aiw_x = ref["sscale"][is_] * ref["secaiw"][:, is_] * scale  # (23, 7, 119)
        alpha = ref["ascale"] * (ref["alphaiw"] - ref["hcount"] * aiw_x)
        alpha = np.where(sys_present, np.maximum(alpha, 0.0), 0.0)
        return alpha

    # building blocks (atomic units; all TorchScript-compatible)
    def coordination_numbers(self, z: Tensor, edge_index: Tensor, r: Tensor,
                             n_atoms: int) -> Tuple[Tensor, Tensor]:
        """D4 and EEQ coordination numbers of every atom (paper eqs 6, 14).

        Parameters
        ----------
        z : Tensor
            Atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge list ``[src, dst]`` of shape ``(2, E)``; the count of
            ``src`` is added to the center ``dst``.
        r : Tensor
            Edge lengths in bohr, shape ``(E,)``.
        n_atoms : int
            Number of atoms ``N``.

        Returns
        -------
        cn_d4 : Tensor
            The electronegativity-weighted CN entering the Gaussian
            weighting, shape ``(N,)`` (edges within ``cutoff_cn``).
        cn_eeq : Tensor
            The plain CN of the charge model, softly capped at 8, shape
            ``(N,)`` (edges within ``cutoff_eeq_cn``).
        """
        src, dst = edge_index[0], edge_index[1]
        zi, zj = z[dst], z[src]
        rc = self.rcov[zi] + self.rcov[zj]
        count = _erf_count(r, rc, self.kcn)
        en_factor = self.k4 * torch.exp(-((self.en[zi] - self.en[zj]).abs() + self.k5) ** 2 / self.k6)
        in_cn = r <= self.cutoff_cn / self.bohr
        cn_d4 = scatter_sum(torch.where(in_cn, en_factor * count,
                                        torch.zeros_like(count)), dst, n_atoms)
        in_eeq = r <= self.cutoff_eeq_cn / self.bohr
        cn_raw = scatter_sum(torch.where(in_eeq, count, torch.zeros_like(count)),
                             dst, n_atoms)
        # soft cap (upstream ``log_cn_cut``): cn' = log(1 + e^max) - log(1 + e^(max - cn))
        cn_eeq = (math.log(1.0 + math.exp(self.eeq_cn_max))
                  - torch.log(1.0 + torch.exp(self.eeq_cn_max - cn_raw)))
        return cn_d4, cn_eeq

    def _coulomb_matrix_molecular(self, z: Tensor, pos: Tensor) -> Tensor:
        """EEQ interaction matrix ``A`` of an isolated structure (paper eq 12).

        ``pos`` in bohr, shape ``(N, 3)``; returns ``(N, N)``.
        """
        n = pos.shape[0]
        rad2 = self.eeq_rad[z] ** 2
        gamma = torch.rsqrt(rad2[:, None] + rad2[None, :])
        eye = torch.eye(n, dtype=torch.bool, device=pos.device)
        d = pos[None, :, :] - pos[:, None, :]
        r2 = (d * d).sum(-1)
        r = torch.sqrt(torch.where(eye, torch.ones_like(r2), r2))
        off = torch.where(eye, torch.zeros_like(r), torch.erf(gamma * r) / r)
        diag = self.eeq_eta[z] + math.sqrt(2.0 / math.pi) / self.eeq_rad[z]
        return off + torch.diag(diag)

    def _coulomb_matrix_periodic(self, z: Tensor, pos: Tensor, cell: Tensor) -> Tensor:
        """Ewald-summed EEQ interaction matrix of a periodic structure.

        Every pair (including an atom with its own images) is represented by
        its Wigner-Seitz image(s) -- the translations of ``pos_j - pos_i``
        within ``+-1`` cells that realize the minimum distance, averaged when
        several tie within 0.01 bohr^2 -- and summed over ``+-2`` cells in real
        space with the ``erf(gamma r)/r - erf(alpha r)/r`` kernel and over
        ``+-2`` reciprocal cells (``G != 0``) with ``4 pi / V exp(-G^2 / 4
        alpha^2) / G^2 cos(G . r)``. The diagonal carries
        ``J_A + sqrt(2/pi) / a_A - 2 alpha / sqrt(pi)``. The neutralizing
        background of a charged cell is omitted: it is a constant added to
        every matrix element, which the charge constraint absorbs.
        """
        n = pos.shape[0]
        device, dtype = pos.device, pos.dtype
        vol = cell_volume(cell)
        recip = 2.0 * math.pi * torch.linalg.inv(cell).t()
        alpha = _ewald_alpha(float(torch.linalg.norm(recip, dim=1).min()),
                             float(torch.linalg.norm(cell, dim=1).min()), float(vol))
        one = torch.arange(-1, 2, device=device)
        t27 = torch.cartesian_prod(one, one, one).to(dtype) @ cell           # (27, 3)
        two = torch.arange(-self.ewald_rep, self.ewald_rep + 1, device=device)
        grid = torch.cartesian_prod(two, two, two)
        t125 = grid.to(dtype) @ cell                                          # (125, 3)
        nonzero = (grid != 0).any(dim=1)
        g124 = grid[nonzero].to(dtype) @ recip                                # (124, 3)
        g2 = (g124 * g124).sum(-1)
        g_fac = 4.0 * math.pi / vol * torch.exp(-0.25 * g2 / (alpha * alpha)) / g2

        rad2 = self.eeq_rad[z] ** 2
        gamma = torch.rsqrt(rad2[:, None] + rad2[None, :])                    # (N, N)

        # Wigner-Seitz images of every pair vector
        d = pos[None, :, :] - pos[:, None, :]                                  # (N, N, 3)
        d_img = d[:, :, None, :] + t27[None, None, :, :]                       # (N, N, 27, 3)
        r2_img = (d_img * d_img).sum(-1)
        thr = 1.4901161193847656e-08                                            # sqrt(eps)
        r2_img = torch.where(r2_img < thr, torch.full_like(r2_img, math.inf), r2_img)
        r2_min = r2_img.min(dim=-1, keepdim=True).values
        selected = (r2_img - r2_min) <= self.ws_tol                            # (N, N, 27)
        n_img = selected.sum(dim=-1).to(dtype)                                 # (N, N)
        idx = torch.nonzero(selected)                                          # (P, 3)
        ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]
        vec = d_img[ii, jj, kk]                                                # (P, 3)
        gam_p = gamma[ii, jj]

        # real space: sum over +-2 translations of the screened kernel
        vt = vec[:, None, :] + t125[None, :, :]                                # (P, 125, 3)
        r2_pt = (vt * vt).sum(-1)
        # the translation cancelling an image vector gives r = 0 (an atom
        # with itself): mask it *before* the square root so no infinite
        # derivative leaks through the masked branch
        keep = r2_pt >= thr * thr
        r_safe = torch.sqrt(torch.where(keep, r2_pt, torch.ones_like(r2_pt)))
        kern = (torch.erf(gam_p[:, None] * r_safe) - torch.erf(alpha * r_safe)) / r_safe
        a_dir = torch.where(keep, kern, torch.zeros_like(kern)).sum(-1)       # (P,)
        # reciprocal space
        a_rec = (torch.cos(vec @ g124.t()) * g_fac[None, :]).sum(-1)          # (P,)
        contrib = (a_dir + a_rec) / n_img[ii, jj]
        amat = scatter_sum(contrib, ii * n + jj, n * n).view(n, n)
        diag = (self.eeq_eta[z] + math.sqrt(2.0 / math.pi) / self.eeq_rad[z]
                - 2.0 * alpha / math.sqrt(math.pi))
        return amat + torch.diag(diag)

    def eeq_charges(self, z: Tensor, pos: Tensor, cn_eeq: Tensor,
                    total_charge: Tensor, cell: Tensor, periodic: bool) -> Tensor:
        """EEQ partial charges of one structure (paper eqs 11-16).

        Parameters
        ----------
        z : Tensor
            Atomic numbers, shape ``(N,)``.
        pos : Tensor
            Positions in bohr, shape ``(N, 3)``.
        cn_eeq : Tensor
            Capped error-function coordination numbers, shape ``(N,)``.
        total_charge : Tensor
            Scalar total charge of the structure.
        cell : Tensor
            Lattice vectors as rows in bohr, shape ``(3, 3)`` (ignored when
            ``periodic`` is ``False``).
        periodic : bool
            Whether to Ewald-sum the Coulomb matrix.

        Returns
        -------
        Tensor
            Partial charges, shape ``(N,)``, summing to ``total_charge``.
        """
        n = pos.shape[0]
        if periodic:
            amat = self._coulomb_matrix_periodic(z, pos, cell)
        else:
            amat = self._coulomb_matrix_molecular(z, pos)
        # electronegativity RHS with the CN-dependent shift (paper eq 13)
        x = -self.eeq_chi[z] + self.eeq_kcnchi[z] * cn_eeq / torch.sqrt(cn_eeq + self.eeq_cn_reg)
        ones = torch.ones((n, 1), dtype=pos.dtype, device=pos.device)
        zero = torch.zeros((1, 1), dtype=pos.dtype, device=pos.device)
        full = torch.cat([torch.cat([amat, ones], dim=1),
                          torch.cat([ones.t(), zero], dim=1)], dim=0)
        rhs = torch.cat([x, total_charge.reshape(1).to(pos.dtype)])
        sol = torch.linalg.solve(full, rhs)
        return sol[:n]

    def select_regime(self, n_atoms: int, periodic: bool) -> str:
        """Resolve ``"auto"`` to ``"dense"`` or ``"large"`` for one structure."""
        if self.regime != "auto":
            return self.regime
        limit = self.auto_large_periodic if periodic else self.auto_large_molecular
        return "large" if n_atoms > limit else "dense"

    @torch.jit.unused
    def enable_eeq_reuse(self, enabled: bool = True, **options) -> None:
        """Carry the large-regime EEQ solve over between consecutive structures.

        For molecular dynamics and geometry optimization, where one structure
        follows the previous one closely; see
        :class:`~xnn.common.models.eeq.EEQReuse`, which receives ``options``
        (``tol``, ``refresh``, ``maxiter``). Results agree with the fresh
        solve to the solver tolerance; the dense regime is not affected.
        ``enable_eeq_reuse(False)`` switches it off again. Eager only.
        """
        # kept out of the module's attributes that TorchScript would inspect
        self.__dict__["_eeq_reuse"] = EEQReuse(**options) if enabled else None

    @torch.jit.unused
    def _eeq_charges_large(self, z: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor,
                           cn_eeq: Tensor, total_charge: Tensor, cell: Tensor,
                           periodic: bool) -> Tensor:
        """Large-regime EEQ charges of one structure (eager only).

        ``edge_index`` / ``edge_vec`` are the structure's edges within
        ``cutoff_eeq`` in local numbering and bohr. See
        :mod:`~xnn.common.models.eeq`.
        """
        if periodic and self.cutoff_eeq < _MIN_CUTOFF_EEQ_AU * self.bohr:
            raise ValueError(
                f"the large EEQ regime needs cutoff_eeq >= {_MIN_CUTOFF_EEQ_AU:.0f} bohr "
                f"({_MIN_CUTOFF_EEQ_AU * self.bohr:.1f} A) of neighbor list, got "
                f"{self.cutoff_eeq:.2f} A; raise cutoff_eeq or use regime='dense'")
        rad = self.eeq_rad[z]
        diag = self.eeq_eta[z] + math.sqrt(2.0 / math.pi) / rad
        x = -self.eeq_chi[z] + self.eeq_kcnchi[z] * cn_eeq / torch.sqrt(cn_eeq + self.eeq_cn_reg)
        if periodic:
            alpha = ewald_alpha(self.cutoff_eeq / self.bohr)
            grid, gvec, gfac = reciprocal_vectors(cell.detach(), alpha)
            diag = diag - 2.0 * alpha / math.sqrt(math.pi)
            system = EEQSystem(diag, rad, pos, edge_index, edge_vec, alpha, gvec, gfac, grid,
                               solver=self.eeq_solver, reuse=self.__dict__.get("_eeq_reuse"))
            return eeq_charges_large(system, pos, edge_vec, rad, diag, x, total_charge, cell)
        system = EEQSystem(diag, rad, pos, solver=self.eeq_solver,
                           reuse=self.__dict__.get("_eeq_reuse"))
        return eeq_charges_large(system, pos, None, rad, diag, x, total_charge)

    @torch.jit.unused
    def _two_body_chunked(self, z: Tensor, edge_index: Tensor, r: Tensor, alpha_iw: Tensor,
                          n_atoms: int, chunk: int = 1 << 20) -> Tensor:
        """:meth:`two_body_energy` in recompute blocks of ``chunk`` edges (eager only)."""
        from .recompute import recompute
        energy = torch.zeros(n_atoms, dtype=r.dtype, device=r.device)
        n_edges = int(edge_index.shape[1])

        def block(ei, r_, alpha, s6, s8, a1, a2):
            src, dst = ei[0], ei[1]
            zi, zj = z[dst], z[src]
            c6 = (3.0 / math.pi) * (alpha[dst] * alpha[src] * self.cp_weights).sum(-1)
            rr = 3.0 * self.r4r2[zi] * self.r4r2[zj]
            r0 = a1 * torch.sqrt(rr) + a2
            r2 = r_ * r_
            t6 = 1.0 / (r2 ** 3 + r0 ** 6)
            t8 = 1.0 / (r2 ** 4 + r0 ** 8)
            sw = switching_function(r_, self.cutoff_pair / self.bohr, self.switch_width_pair / self.bohr)
            return scatter_sum(-0.5 * c6 * sw * (s6 * t6 + s8 * rr * t8), dst, n_atoms)

        for e0 in range(0, n_edges, chunk):
            e1 = min(e0 + chunk, n_edges)
            energy = energy + recompute(block, edge_index[:, e0:e1], r[e0:e1], alpha_iw,
                                        self.s6, self.s8, self.a1, self.a2)
        return energy

    @torch.jit.unused
    def _three_body_chunked(self, z: Tensor, pos: Tensor, edge_index: Tensor, edge_vec: Tensor,
                            r: Tensor, alpha_neutral: Tensor, cell: Tensor, batch: Tensor,
                            n_atoms: int) -> Tensor:
        """Recompute-block ATM term with per-edge C6 (eager only).

        The pair C6 of every directed edge within the three-body cutoff is
        formed once from the neutral-atom polarizabilities (differentiable, so
        the coordination-number dependence reaches the forces), and the blocks
        gather scalars; see :func:`~xnn.common.models.dispersion.three_body_energy_chunked`.
        """
        alpha_a = (3.0 / math.pi) * alpha_neutral * self.cp_weights
        c6_edge = (alpha_a[edge_index[1]] * alpha_neutral[edge_index[0]]).sum(-1)
        shifts = edge_cell_shifts(pos, edge_index, edge_vec, cell, batch)
        r0_atom = (3.0 ** 0.25) * torch.sqrt(self.r4r2[z])      # rho_A: R0_AB = a1 rho_A rho_B + a2
        return three_body_energy_chunked(z, edge_index, edge_vec, r, None,
                                         self.s9, self.alp / 3.0, self.cutoff_triple / self.bohr,
                                         self.switch_width_triple / self.bohr, n_atoms,
                                         r0_atom=r0_atom, a1=self.a1, a2=self.a2,
                                         c6_edge=c6_edge, edge_shift=shifts,
                                         chunk=self.triplet_chunk,
                                         triplet_cache=self.triplet_cache)

    def reference_weights(self, z: Tensor, cn: Tensor, q: Tensor) -> Tensor:
        """Charge-scaled Gaussian weights of the reference systems (eqs 2-4, 8).

        Parameters
        ----------
        z : Tensor
            Atomic numbers, shape ``(N,)``.
        cn : Tensor
            D4 coordination numbers, shape ``(N,)``.
        q : Tensor
            Partial charges, shape ``(N,)`` (zeros for the ATM term).

        Returns
        -------
        Tensor
            ``W_A,ref zeta(z_A, z_A,ref)`` for the 7 reference slots of each
            atom, shape ``(N, 7)`` (zero in unused slots).
        """
        refcn = self.refcn[:, z].t()                                  # (N, 7)
        ngw = self.ngw[:, z].t()
        n_slots = refcn.shape[1]
        valid = (torch.arange(n_slots, device=z.device)[None, :]
                 < self.nref[z][:, None])
        # sqrt(tiny) of the working precision, as upstream
        eps_norm = 1.0842021724855044e-19 if cn.dtype == torch.float32 else 1.4916681462400413e-154
        weights = gaussian_reference_weights(cn, refcn, valid, self.wf, ngw, eps_norm)
        # charge scaling of paper eqs 2-4
        zi = self.zeff[z][:, None]
        gi = (self.hardness[z] * self.gc)[:, None]
        qmod = q[:, None] + zi
        qref = self.refq[:, z].t() + zi
        negative = qmod < 0.0
        safe = torch.where(negative, torch.ones_like(qmod), qmod)
        scaled = torch.exp(self.ga * (1.0 - torch.exp(gi * (1.0 - qref / safe))))
        zeta = torch.where(negative, torch.full_like(qmod, math.exp(self.ga)), scaled)
        return weights * zeta

    def dynamic_polarizabilities(self, z: Tensor, weights: Tensor) -> Tensor:
        """Atom-in-molecule ``alpha_A(i omega)`` on the 23-point grid (eq 7).

        Returns shape ``(N, 23)`` in bohr^3; column 0 (``omega = 1e-6``) is the
        static polarizability.
        """
        refalpha = self.refalpha[:, :, z]                             # (23, 7, N)
        return torch.einsum("nr,krn->nk", weights, refalpha)

    def pair_c6(self, alpha_i: Tensor, alpha_j: Tensor) -> Tensor:
        """``C6`` of pairs from their dynamic polarizabilities (eq 9), ``(E,)``."""
        return (3.0 / math.pi) * (alpha_i * alpha_j * self.cp_weights).sum(-1)

    def two_body_energy(self, z: Tensor, edge_index: Tensor, r: Tensor,
                        alpha_iw: Tensor, n_atoms: int) -> Tensor:
        """Per-atom BJ-damped two-body energy in hartree (eqs 18-21).

        Every directed edge contributes half of its pair energy to its
        center, so the sum over atoms is the sum over unordered pairs.
        """
        src, dst = edge_index[0], edge_index[1]
        zi, zj = z[dst], z[src]
        c6 = self.pair_c6(alpha_iw[dst], alpha_iw[src])
        rr = 3.0 * self.r4r2[zi] * self.r4r2[zj]                     # C8 / C6
        r0 = self.a1 * torch.sqrt(rr) + self.a2
        r2 = r * r
        r6, r8 = r2 ** 3, r2 ** 4
        t6 = 1.0 / (r6 + r0 ** 6)
        t8 = 1.0 / (r8 + r0 ** 8)
        sw = switching_function(r, self.cutoff_pair / self.bohr, self.switch_width_pair / self.bohr)
        e_pair = -0.5 * c6 * sw * (self.s6 * t6 + self.s8 * rr * t8)
        return scatter_sum(e_pair, dst, n_atoms)

    def pair_radius_table(self) -> Tensor:
        """BJ critical radii ``a1 sqrt(3 Q_A Q_B) + a2`` for all element pairs, bohr."""
        return self.a1 * torch.sqrt(3.0 * self.r4r2[:, None] * self.r4r2[None, :]) + self.a2

    # evaluation
    @torch.jit.export
    def evaluate(self, atomic_numbers: Tensor, pos: Tensor, edge_index: Tensor,
                 edge_vec: Tensor, batch: Tensor, num_graphs: int,
                 cell: Tensor, pbc: Tensor, total_charge: Tensor) -> Dict[str, Tensor]:
        """Dispersion energy and D4 properties of a batch of structures.

        The TorchScript-compatible core shared by :meth:`forward` and the
        deploy wrappers. Inputs in Angstrom; the neighbor list must reach
        :attr:`cutoff` (edges beyond a term's own cutoff are ignored by it).

        Parameters
        ----------
        atomic_numbers : Tensor
            Atomic numbers, shape ``(N,)``.
        pos : Tensor
            Cartesian positions in Angstrom, shape ``(N, 3)``.
        edge_index : Tensor
            Edge list ``[src, dst]``, shape ``(2, E)``, both directions of
            every pair present.
        edge_vec : Tensor
            Edge vectors ``pos[dst] - pos[src] (+ shift)`` in Angstrom,
            shape ``(E, 3)``.
        batch : Tensor
            Structure index of every atom, shape ``(N,)``.
        num_graphs : int
            Number of structures ``B``.
        cell : Tensor
            Lattice vectors as rows in Angstrom, shape ``(B, 3, 3)``; an
            all-zero cell marks a molecular structure.
        pbc : Tensor
            Periodicity flags, shape ``(B, 3)``; a structure is periodic
            when any flag is set and its cell is nonzero.
        total_charge : Tensor
            Total charge per structure, shape ``(B,)``.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)`` in eV, plus the atomic units
            quantities ``"coordination_numbers"`` ``(N,)``, ``"charges"``
            ``(N,)``, ``"polarizabilities"`` ``(N,)`` (static, bohr^3) and
            ``"dynamic_polarizabilities"`` ``(N, 23)`` (for
            :func:`c6_matrix`), and ``"energy_2body"`` / ``"energy_3body"``
            ``(B,)`` in eV.
        """
        z = atomic_numbers
        n_atoms = z.shape[0]
        if bool((z > self.max_z).any()) or bool((z < 1).any()):
            raise ValueError("D4 with EEQ charges supports atomic numbers 1..103")
        pos_au = pos / self.bohr
        vec_au = edge_vec / self.bohr
        r = torch.linalg.norm(vec_au, dim=-1)

        cn_d4, cn_eeq = self.coordination_numbers(z, edge_index, r, n_atoms)

        charges: List[Tensor] = []
        in_eeq = r <= self.cutoff_eeq / self.bohr
        for b in range(num_graphs):
            members = torch.nonzero(batch == b).squeeze(1)
            cell_b = cell[b] / self.bohr
            periodic = bool(pbc[b].any()) and bool(cell_b.abs().sum() > 1e-8)
            regime = self.select_regime(int(members.shape[0]), periodic)
            if regime == "large" and not torch.jit.is_scripting():
                # the structure's EEQ-range edges in local numbering
                local = torch.full((n_atoms,), -1, dtype=torch.long, device=z.device)
                local[members] = torch.arange(members.shape[0], device=z.device)
                sel = in_eeq & (batch[edge_index[1]] == b)
                charges.append(self._eeq_charges_large(
                    z[members], pos_au[members], local[edge_index[:, sel]], vec_au[sel],
                    cn_eeq[members], total_charge[b], cell_b, periodic))
            else:
                charges.append(self.eeq_charges(z[members], pos_au[members],
                                                cn_eeq[members], total_charge[b],
                                                cell_b, periodic))
        q = torch.cat(charges)

        weights = self.reference_weights(z, cn_d4, q)
        alpha_iw = self.dynamic_polarizabilities(z, weights)

        in_pair = r <= self.cutoff_pair / self.bohr
        if self.recompute_pairs and not torch.jit.is_scripting():
            e2 = self._two_body_chunked(z, edge_index[:, in_pair], r[in_pair], alpha_iw, n_atoms)
        else:
            e2 = self.two_body_energy(z, edge_index[:, in_pair], r[in_pair],
                                      alpha_iw, n_atoms)
        node_energy = e2
        e3 = torch.zeros_like(e2)
        if bool(self.s9 != 0.0):
            # the ATM term uses C6 coefficients of the *neutral* atoms
            weights_neutral = self.reference_weights(z, cn_d4, torch.zeros_like(q))
            alpha_neutral = self.dynamic_polarizabilities(z, weights_neutral)
            in_triple = r <= self.cutoff_triple / self.bohr
            if self.checkpoint_triplets and not torch.jit.is_scripting():
                e3 = self._three_body_chunked(z, pos_au, edge_index[:, in_triple], vec_au[in_triple],
                                              r[in_triple], alpha_neutral, cell / self.bohr, batch,
                                              n_atoms)
            else:
                if n_atoms > 20000:
                    raise ValueError("the scripted D4 three-body term needs the dense C6 "
                                     "matrix (more than 20000 atoms); set s9 = 0")
                c6_neutral = (3.0 / math.pi) * (alpha_neutral * self.cp_weights) @ alpha_neutral.t()
                e3 = three_body_energy(z, edge_index[:, in_triple], vec_au[in_triple],
                                       r[in_triple], c6_neutral, self.pair_radius_table(),
                                       self.s9, self.alp / 3.0, self.cutoff_triple / self.bohr,
                                       self.switch_width_triple / self.bohr, n_atoms)
            node_energy = node_energy + e3
        return {
            "node_energy": node_energy * self.hartree,
            "energy_2body": scatter_sum(e2, batch, num_graphs) * self.hartree,
            "energy_3body": scatter_sum(e3, batch, num_graphs) * self.hartree,
            "coordination_numbers": cn_d4,
            "charges": q,
            "polarizabilities": alpha_iw[:, 0],
            "dynamic_polarizabilities": alpha_iw,
            "node_features": torch.stack([cn_d4, q, alpha_iw[:, 0]], dim=1),
        }

    @torch.jit.ignore
    def forward(self, data) -> Dict[str, Tensor]:
        """Evaluate a (batched) :class:`~xnn.common.data.AtomicGraph`.

        Adds ``"energy"`` ``(B,)`` to the keys of :meth:`evaluate`.
        """
        return evaluate_on_graph(self, data)


@register_model("d4")
class D4Dispersion(DispersionCorrection):
    """DFT-D4 dispersion as an xnn potential, standalone or wrapped around a model.

    Standalone (``model=None``) it is the pure D4 dispersion energy -- what
    ``dftd4`` computes -- and is registered as the model ``"d4"``. Given a
    short-range ``model`` it adds the D4 energy to that model's prediction
    (see :class:`~xnn.common.models.dispersion.DispersionCorrection` for the
    wrapper semantics). Enable from a config with ``model.extra["dispersion"]``
    or wrap directly::

        model = D4Dispersion(build_model(cfg.model), cutoff_pair=12.0,
                             switch_width_pair=2.0)
        out = ForceStressOutput(model, compute_stress=True)(graph)

    Parameters
    ----------
    model : InteratomicPotential or None, optional
        The short-range model to correct; ``None`` for pure dispersion.
    **d4_options
        Keyword arguments of :class:`DFTD4` (damping parameters, cutoffs,
        switching widths, ``trainable``).

    Attributes
    ----------
    d4 : DFTD4
        The dispersion evaluator (alias of ``term``).

    Notes
    -----
    Besides the wrapper's common outputs, ``forward`` adds the per-atom
    ``"eeq_charges"``, ``"polarizabilities"`` and
    ``"dynamic_polarizabilities"``; standalone, ``"node_features"`` is the
    per-atom (CN, EEQ charge, static polarizability) triple.
    """

    def __init__(self, model: Optional[nn.Module] = None, **d4_options):
        super().__init__(DFTD4(**d4_options), model)

    @property
    def d4(self) -> DFTD4:
        """The :class:`DFTD4` evaluator."""
        return self.term

    @classmethod
    def from_config(cls, cfg) -> "D4Dispersion":
        """Build a standalone D4 model; options are read from ``cfg.extra``.

        Recognized keys are the :class:`DFTD4` arguments (``s6, s8, a1, a2,
        s9, alp, ga, gc, wf, cutoff_pair, cutoff_triple, cutoff_cn,
        cutoff_eeq_cn, switch_width_pair, switch_width_triple, trainable``).
        ``cfg.cutoff`` is ignored: the neighbor-list radius follows from the
        D4 cutoffs. To *correct* another model with D4 put the same keys
        under that model's ``extra["dispersion"]`` instead.
        """
        return cls(None, **options_from_extra(cfg.extra or {}, _D4_KEYS))


_D4_KEYS = ("s6", "s8", "a1", "a2", "s9", "alp", "ga", "gc", "wf",
            "cutoff_pair", "cutoff_triple", "cutoff_cn", "cutoff_eeq_cn",
            "switch_width_pair", "switch_width_triple", "trainable",
            "regime", "cutoff_eeq", "eeq_solver", "checkpoint_triplets", "recompute_pairs",
            "triplet_chunk", "triplet_cache")


def d4_options_from_extra(extra: dict) -> dict:
    """Pick the :class:`DFTD4` keyword arguments out of a config ``extra`` dict."""
    return options_from_extra(extra, _D4_KEYS)
