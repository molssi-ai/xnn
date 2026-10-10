"""SpookyNet (Unke et al., Nat. Commun. 2021): electronic states and nonlocal effects.

An implementation of the SpookyNet potential of

* O. T. Unke, S. Chmiela, M. Gastegger, K. T. Schuett, H. E. Sauceda,
  K.-R. Mueller, "SpookyNet: Learning force fields with electronic degrees
  of freedom and nonlocal effects", *Nat. Commun.* **12**, 7273 (2021),

written from the paper's Methods section on the xnn abstractions
(:func:`~xnn.common.models.ops.scatter_sum`,
:class:`~xnn.common.featurizers.MollifierCutoff`,
:class:`~xnn.hybrid.featurizers.ExponentialBernsteinRBF`, the D4 tables of
:mod:`~xnn.common.models.d4`, the Ewald sums of
:mod:`~xnn.common.models.electrostatics`) and consistent with the behavior of
the authors' reference code (github.com/OUnke/SpookyNet), without copying it.

It belongs to the hybrid family: graph message passing combined with a
transformer (self-attention) update and a physics-based split of the energy,
like :class:`~xnn.hybrid.models.bamboo.BAMBOO`.

Inputs. Besides atomic numbers and positions the model reads the total
charge ``Q`` (``AtomicGraph.total_charge``, neutral when absent) and the
number of unpaired electrons ``S = spin_multiplicity - 1``
(``AtomicGraph.spin_multiplicity``, singlet when absent).

Representation (eqs 1-3, 9-11). Every atom starts from
``x0 = e_Z + e_Q + e_S``. The nuclear embedding ``e_Z = M d_Z + e~_Z`` maps
a 20-entry descriptor of the ground-state electron configuration
(:func:`electron_configurations`) through a learned matrix and adds a free
element vector. The electronic embeddings spread ``Q`` and ``S`` over the
atoms by an attention-like weighting (eq 10)::

    q_i = linear(e_Z,i),  a_i = softplus(q_i . k / sqrt(F)) / sum_j softplus(q_j . k / sqrt(F)),
    e_Psi,i = resmlp(a_i |Psi| v)

with separate keys and values ``k, v`` for positive and negative charges.
``T`` interaction modules then refine the features (eq 11)::

    x~ = residual(x),  x' = residual(x~ + l + n),  y = resmlp(x')

and the atomic descriptors are ``f = sum_t y_t``. Here ``residual`` is the
pre-activation residual block ``x + W2 silu(W1 silu(x))`` (eq 7),
``resmlp`` a residual block followed by an activation and a linear layer
(eq 8) and ``silu`` the generalized SiLU ``alpha x sigmoid(beta x)`` with
learned per-feature ``alpha`` and ``beta`` (eq 6, :class:`Swish`).

Local interaction (eqs 12-17). Within the cutoff ``r_c`` every pair
contributes through basis functions that look like s, p and d orbitals,
``g_lm(r) = rho_k(r) Y_lm(r / |r|)`` with the exponential Bernstein radial
functions ``rho_k`` (times the cutoff ``exp(-r^2 / (r_c^2 - r^2))``) and the
real spherical harmonics without their normalization constant::

    s_i = sum_j resmlp_s(x~_j) o G_s rho(r_ij)
    p_i = sum_j resmlp_p(x~_j) o G_p rho(r_ij) Y_1(r_ij)          (3 x F)
    d_i = sum_j resmlp_d(x~_j) o G_d rho(r_ij) Y_2(r_ij)          (5 x F)
    l_i = resmlp_l(resmlp_c(x~_i) + s_i + <P1 p_i, P2 p_i> + <D1 d_i, D2 d_i>)

so angular information enters at linear cost in the number of neighbors.

Nonlocal interaction (eqs 18-20). Self-attention over all atoms of a
structure, ``n = attention(resmlp_q(x~), resmlp_k(x~), resmlp_v(x~))``,
evaluated with the linear-scaling FAVOR+ approximation (positive orthogonal
random features drawn once at initialization, ``f = F`` of them) or exactly
(``attention="exact"``, eq 19).

Energy (eqs 4, 21-27). Atomic energies and partial charges come from one
linear layer on ``f`` plus element biases; the charges are shifted so that
they sum to ``Q`` (eq 24). Three physical terms are added: a ZBL-like
nuclear repulsion with learnable, positive parameters (eq 22), point-charge
electrostatics that switch from ``1 / sqrt(r^2 + 1)`` to ``1 / r`` between
``r_c / 4`` and ``3 r_c / 4`` (eq 23, 25) and the two-body D4 dispersion with
the model's charges, learnable ``s8, a1, a2`` and a learnable scale ``s_q``
of the reference charges (eq 27). The dipole moment is ``sum_i q_i r_i``
(eq 26).

Conventions worth knowing:

* The electronic embedding follows the reference code: the value is
  ``|Psi| v_+-`` (the paper writes ``Psi v_+-``, the same model with ``v_-``
  negated), the keys use the sign pattern of ``Psi`` (``e / max(e, 1)``) and
  the weights are normalized with ``+1e-8``; ``k`` and ``v`` start
  orthogonal, since zero values would leave the branch without a gradient.
* The radial functions are ordered by the paper's ``k`` (the power of
  ``exp(-gamma r)``); the reference code stores them in reverse order. Its
  extra weighting ``exp(-gamma r)`` is the ``exp_weighting`` option.
* FAVOR+ subtracts, for stability, the largest key projection of the whole
  structure; the reference code's batched path takes the largest over a
  zero-padded matrix instead, which differs (by the ``1e-4`` stabilizer of
  the features) only when every projection of a structure is negative. Here a
  structure's energy never depends on the other structures of a batch.
* The D4 reference polarizabilities are recomputed from ``s_q`` in every
  evaluation (differentiable), with the dftd4 reference data shipped with
  :mod:`~xnn.common.models.d4`; physical constants are CODATA 2018 (the
  attributes ``k_e``, ``bohr`` and ``hartree`` of the term modules).
* Without ``lr_cutoff`` the electrostatics and dispersion run over all pairs
  of every structure (molecules). With ``lr_cutoff`` they use the neighbor
  list with the reference code's shifted kernels, which also serves periodic
  structures; ``ewald=True`` replaces the shifted Coulomb kernel by the Ewald
  sum of the periodic structures (the bare ``1 / r`` lattice sum plus the
  short-range damping of eq 23).
"""
# NOTE: no `from __future__ import annotations` -- TorchScript resolves the
# annotations of the scripted methods.
import functools
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import MollifierCutoff
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.d4 import DFTD4, _erf_count, _load_reference
from xnn.common.models.dispersion import BOHR, HARTREE, gaussian_reference_weights
from xnn.common.models.electrostatics import (COULOMB_CONSTANT, all_pairs, coulomb_ewald,
                                              ewald_parameters)
from xnn.common.models.ops import glorot_orthogonal_, scatter_sum, softplus_inverse, structure_sum
from xnn.common.models.registry import register_model
from xnn.hybrid.featurizers import ExponentialBernsteinRBF

#: Number of element rows (Z = 0 to 86): the electron-configuration descriptor
#: and the reference models cover the elements up to Rn.
MAX_Z = 87

_SUBSHELLS = ("1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s",
              "4f", "5d", "6p")
_CAPACITY = {"s": 2, "p": 6, "d": 10, "f": 14}
# ground states that deviate from the Madelung filling order (Z <= 86)
_MADELUNG_EXCEPTIONS = {
    24: {"4s": 1, "3d": 5}, 29: {"4s": 1, "3d": 10},
    41: {"5s": 1, "4d": 4}, 42: {"5s": 1, "4d": 5}, 44: {"5s": 1, "4d": 7},
    45: {"5s": 1, "4d": 8}, 46: {"5s": 0, "4d": 10}, 47: {"5s": 1, "4d": 10},
    57: {"4f": 0, "5d": 1}, 58: {"4f": 1, "5d": 1}, 64: {"4f": 7, "5d": 1},
    78: {"6s": 1, "5d": 9}, 79: {"6s": 1, "5d": 10},
}
_PERIOD_ENDS = (2, 10, 18, 36, 54, 86)


def electron_configurations(max_z: int = MAX_Z - 1) -> np.ndarray:
    """Element descriptors ``d_Z`` of the nuclear embedding (paper eq 9).

    Row ``Z`` holds the atomic number, the occupations of the subshells
    1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d and 6p of the
    neutral ground state, and the valence occupations (the ``n``-s and
    ``n``-p shells of the element's period ``n``, the ``(n-1)``-d and the
    ``(n-2)``-f shells); every column is divided by its largest value. Row 0
    (no element) is zero.

    Parameters
    ----------
    max_z : int, optional
        Last element, at most 86 (Rn), by default 86.

    Returns
    -------
    numpy.ndarray
        Shape ``(max_z + 1, 20)``, float64, entries in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``max_z`` exceeds 86.
    """
    if max_z > 86:
        raise ValueError(f"the descriptor covers Z <= 86, got max_z={max_z}")
    table = np.zeros((max_z + 1, 20), dtype=np.float64)
    for z in range(1, max_z + 1):
        occupation, left = {}, z
        for shell in _SUBSHELLS:
            occupation[shell] = min(left, _CAPACITY[shell[-1]])
            left -= occupation[shell]
        occupation.update(_MADELUNG_EXCEPTIONS.get(z, {}))
        n = 1 + sum(z > end for end in _PERIOD_ENDS)
        valence = [occupation.get(f"{n}s", 0), occupation.get(f"{n}p", 0),
                   occupation.get(f"{n - 1}d", 0), occupation.get(f"{n - 2}f", 0)]
        table[z] = [z] + [occupation[s] for s in _SUBSHELLS] + valence
    return table / table.max(axis=0)


def smooth_switch(r: Tensor, r_on: float, r_off: float) -> Tensor:
    """Switch from 1 (``r <= r_on``) to 0 (``r >= r_off``), smooth to every order (eq 25).

    ``sigma(1 - x) / (sigma(1 - x) + sigma(x))`` with ``x = (r - r_on) /
    (r_off - r_on)`` and ``sigma(x) = exp(-1 / x)`` for ``x > 0`` (0 else).
    """
    x = (r - r_on) / (r_off - r_on)
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    # the dummy arguments keep the unused branch of torch.where finite, so
    # the gradient stays finite as well
    sigma_x = torch.where(x > 0, torch.exp(-1.0 / torch.where(x > 0, x, ones)), zeros)
    y = 1.0 - x
    sigma_y = torch.where(y > 0, torch.exp(-1.0 / torch.where(y > 0, y, ones)), zeros)
    middle = sigma_y / torch.where((x > 0) & (x < 1), sigma_x + sigma_y, ones)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, middle))


def orthogonal_random_features(n_features: int, dim: int,
                               generator: Optional[torch.Generator] = None) -> Tensor:
    """Projection matrix of positive orthogonal random features (FAVOR+).

    Blocks of ``dim`` mutually orthogonal unit vectors (QR of Gaussian
    matrices) are stacked to ``n_features`` rows and each row is rescaled to
    the length of an independent Gaussian vector, so the rows are marginally
    ``N(0, I)`` but orthogonal within a block (Choromanski *et al.*, ICLR
    2021).

    Parameters
    ----------
    n_features : int
        Number of random features ``m``.
    dim : int
        Dimension ``d`` of the queries and keys.
    generator : torch.Generator, optional
        Source of randomness.

    Returns
    -------
    Tensor
        ``(dim, n_features)``, float64.
    """
    blocks = []
    for start in range(0, n_features, dim):
        gauss = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(gauss)
        blocks.append(q.t()[:min(dim, n_features - start)])
    rows = torch.cat(blocks)
    lengths = torch.randn(n_features, dim, generator=generator,
                          dtype=torch.float64).norm(dim=1, keepdim=True)
    return (lengths * rows).t().contiguous()


@functools.lru_cache(maxsize=1)
def _d4_tables() -> Tuple[Dict[str, float], Dict[str, Tensor]]:
    """Constants and float64 (CPU) tables of the D4 term, taken from :class:`DFTD4`."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        d4 = DFTD4(regime="dense", s9=0.0)
    finally:
        torch.set_default_dtype(previous)
    constants = {name: float(getattr(d4, name))
                 for name in ("ga", "gc", "wf", "kcn", "k4", "k5", "k6")}
    tables = {name: getattr(d4, name).detach().clone()
              for name in ("rcov", "en", "zeff", "hardness", "r4r2", "refcn", "refq",
                           "cp_weights", "nref", "ngw")}
    ref = _load_reference()
    tables["refsys"] = torch.tensor(ref["refsys"], dtype=torch.long)
    for name in ("refh", "sscale", "secaiw", "ascale", "alphaiw", "hcount"):
        tables[name] = torch.tensor(ref[name], dtype=torch.float64)
    return constants, tables


def _restore_tables(module: nn.Module, tables: Dict[str, Tensor]) -> bool:
    """Copy exact float64 values into the float64 floating buffers named in ``tables``."""
    changed = False
    with torch.no_grad():
        for name, exact in tables.items():
            buf = module._buffers.get(name)
            if buf is None or buf.dtype != torch.float64 or not exact.is_floating_point():
                continue
            exact = exact.to(buf.device)
            if not torch.equal(buf, exact):
                buf.copy_(exact)
                changed = True
    return changed


def _exact_softplus(x: Tensor) -> Tensor:
    return F.relu(x) + torch.log1p(torch.exp(-x.abs()))


def _structure_slots(batch: Tensor, num_graphs: int) -> Tuple[Tensor, int]:
    """Position of every atom within its structure and the largest structure size."""
    counts = torch.bincount(batch, minlength=num_graphs)
    order = torch.argsort(batch, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.empty_like(batch)
    slot[order] = torch.arange(batch.shape[0], device=batch.device) - starts[batch[order]]
    return slot, int(counts.max())


class Swish(nn.Module):
    """Generalized SiLU ``alpha x sigmoid(beta x)`` with learned per-feature ``alpha, beta`` (eq 6).

    Parameters
    ----------
    n_features : int
        Number of features.
    alpha, beta : float, optional
        Initial values, by default 1.0 and 1.702 (close to GELU).
    """

    def __init__(self, n_features: int, alpha: float = 1.0, beta: float = 1.702):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((n_features,), float(alpha)))
        self.beta = nn.Parameter(torch.full((n_features,), float(beta)))

    def forward(self, x: Tensor) -> Tensor:
        return self.alpha * x * torch.sigmoid(self.beta * x)


class _Residual(nn.Module):
    """Pre-activation residual block ``x + W2 silu(W1 silu(x))`` (eq 7); ``W2`` starts at zero."""

    def __init__(self, n_features: int, bias: bool = True):
        super().__init__()
        self.activation1 = Swish(n_features)
        self.linear1 = nn.Linear(n_features, n_features, bias=bias)
        self.activation2 = Swish(n_features)
        self.linear2 = nn.Linear(n_features, n_features, bias=bias)
        glorot_orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        if bias:
            nn.init.zeros_(self.linear1.bias)
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear2(self.activation2(self.linear1(self.activation1(x))))


class _ResidualMLP(nn.Module):
    """``linear(silu(residual(x)))`` (eq 8), with a stack of ``n_residual`` residual blocks."""

    def __init__(self, n_features: int, n_residual: int = 1, bias: bool = True,
                 zero_init: bool = False):
        super().__init__()
        self.residual = nn.ModuleList([_Residual(n_features, bias) for _ in range(n_residual)])
        self.activation = Swish(n_features)
        self.linear = nn.Linear(n_features, n_features, bias=bias)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
        else:
            glorot_orthogonal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.residual:
            x = block(x)
        return self.linear(self.activation(x))


class _NuclearEmbedding(nn.Module):
    """``e_Z = M d_Z + e~_Z`` (eq 9); both terms start at zero."""

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("electron_config", torch.tensor(
            electron_configurations(), dtype=torch.get_default_dtype()), persistent=False)
        self.element_embedding = nn.Parameter(torch.zeros(MAX_Z, n_features))
        self.config_linear = nn.Linear(20, n_features, bias=False)
        nn.init.zeros_(self.config_linear.weight)

    @torch.jit.ignore
    def exact_constants(self) -> bool:
        """Restore the float64 descriptor table exactly after a cast; whether it changed."""
        return _restore_tables(self, {"electron_config": torch.tensor(electron_configurations())})

    def forward(self, z: Tensor) -> Tensor:
        table = self.element_embedding + self.config_linear(self.electron_config)
        return table[z]


class _ElectronicEmbedding(nn.Module):
    """Charge or spin embedding ``e_Psi`` (eq 10).

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_residual : int
        Residual blocks of the output network.
    charge : bool
        ``True`` for the total charge (separate keys and values for positive
        and negative charges), ``False`` for the spin (``S >= 0``).
    """

    def __init__(self, n_features: int, n_residual: int, charge: bool):
        super().__init__()
        self.charge = charge
        n_in = 2 if charge else 1
        self.linear_q = nn.Linear(n_features, n_features)
        self.linear_k = nn.Linear(n_in, n_features, bias=False)
        self.linear_v = nn.Linear(n_in, n_features, bias=False)
        self.resmlp = _ResidualMLP(n_features, n_residual, bias=False, zero_init=True)
        for lin in (self.linear_q, self.linear_k, self.linear_v):
            glorot_orthogonal_(lin.weight)
        nn.init.zeros_(self.linear_q.bias)

    def forward(self, e_z: Tensor, psi: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        """Embedding ``(N, F)`` of the per-structure values ``psi`` ``(B,)``."""
        if self.charge:
            e = F.relu(torch.stack([psi, -psi], dim=-1))
        else:
            e = psi.abs().unsqueeze(-1)
        k = self.linear_k(e / torch.clamp(e, min=1.0))[batch]
        v = self.linear_v(e)[batch]
        q = self.linear_q(e_z)
        a = _exact_softplus((k * q).sum(-1) / math.sqrt(q.shape[-1]))
        norm = scatter_sum(a, batch, num_graphs)[batch]
        return self.resmlp((a / (norm + 1e-8)).unsqueeze(-1) * v)


class _LocalInteraction(nn.Module):
    """s-, p- and d-orbital-like message passing within the cutoff (eq 12)."""

    def __init__(self, n_features: int, n_rbf: int, n_residual: int):
        super().__init__()
        self.n_features = n_features
        self.radial_s = nn.Linear(n_rbf, n_features, bias=False)
        self.radial_p = nn.Linear(n_rbf, n_features, bias=False)
        self.radial_d = nn.Linear(n_rbf, n_features, bias=False)
        self.resmlp_c = _ResidualMLP(n_features, n_residual)
        self.resmlp_s = _ResidualMLP(n_features, n_residual)
        self.resmlp_p = _ResidualMLP(n_features, n_residual)
        self.resmlp_d = _ResidualMLP(n_features, n_residual)
        self.projection_p = nn.Linear(n_features, 2 * n_features, bias=False)
        self.projection_d = nn.Linear(n_features, 2 * n_features, bias=False)
        self.resmlp_l = _ResidualMLP(n_features, n_residual, zero_init=True)
        for lin in (self.radial_s, self.radial_p, self.radial_d,
                    self.projection_p, self.projection_d):
            glorot_orthogonal_(lin.weight)

    def forward(self, x: Tensor, rho: Tensor, y1: Tensor, y2: Tensor, src: Tensor,
                dst: Tensor, use_p: bool, use_d: bool) -> Tensor:
        """Local features ``l`` ``(N, F)`` from ``x~`` and the edge basis.

        ``rho`` ``(E, K)`` holds the radial functions times the cutoff, ``y1``
        ``(E, 3)`` and ``y2`` ``(E, 5)`` the angular functions of every edge
        ``j -> i`` (``src`` = ``j``, ``dst`` = ``i``).
        """
        n = x.shape[0]
        total = self.resmlp_c(x) + scatter_sum(
            self.radial_s(rho) * self.resmlp_s(x)[src], dst, n)
        if use_p:
            g = (self.radial_p(rho) * self.resmlp_p(x)[src]).unsqueeze(1) * y1.unsqueeze(-1)
            p = scatter_sum(g, dst, n)                                       # (N, 3, F)
            parts = torch.split(self.projection_p(p), self.n_features, dim=-1)
            total = total + (parts[0] * parts[1]).sum(1)
        if use_d:
            g = (self.radial_d(rho) * self.resmlp_d(x)[src]).unsqueeze(1) * y2.unsqueeze(-1)
            d = scatter_sum(g, dst, n)                                       # (N, 5, F)
            parts = torch.split(self.projection_d(d), self.n_features, dim=-1)
            total = total + (parts[0] * parts[1]).sum(1)
        return self.resmlp_l(total)


class _NonlocalInteraction(nn.Module):
    """Self-attention over all atoms of each structure (eqs 18-20).

    Parameters
    ----------
    n_features : int
        Feature width ``F`` (queries, keys and values).
    n_residual : int
        Residual blocks of the query / key / value networks.
    n_random_features : int or None
        FAVOR+ random features ``m``; ``None`` evaluates the exact attention.
    generator : torch.Generator, optional
        Source of the random features.
    """

    exact: bool

    def __init__(self, n_features: int, n_residual: int, n_random_features: Optional[int],
                 generator: Optional[torch.Generator] = None):
        super().__init__()
        self.resmlp_q = _ResidualMLP(n_features, n_residual, zero_init=True)
        self.resmlp_k = _ResidualMLP(n_features, n_residual, zero_init=True)
        self.resmlp_v = _ResidualMLP(n_features, n_residual, zero_init=True)
        self.exact = n_random_features is None
        omega = (torch.zeros(n_features, 0, dtype=torch.float64) if self.exact else
                 orthogonal_random_features(int(n_random_features), n_features, generator))
        self.register_buffer("omega", omega.to(torch.get_default_dtype()))

    def _features(self, x: Tensor, u: Tensor, shift: Tensor) -> Tensor:
        """Positive random features ``(exp(u - |x'|^2 / 2 - shift) + 1e-4) / sqrt(m)``.

        ``u = omega^T x'`` are the projections of ``x' = x / d^(1/4)`` and
        ``shift`` a stabilizing constant (it cancels up to the ``1e-4``).
        """
        h = (x * x).sum(-1, keepdim=True) / (2.0 * math.sqrt(x.shape[-1]))
        return (torch.exp(u - h - shift) + 1e-4) / math.sqrt(self.omega.shape[-1])

    def forward(self, x: Tensor, batch: Tensor, num_graphs: int, slot: Tensor,
                n_max: int) -> Tensor:
        """Nonlocal features ``n`` ``(N, F)``."""
        q = self.resmlp_q(x)
        k = self.resmlp_k(x)
        v = self.resmlp_v(x)
        if self.exact:
            return self._exact(q, k, v, batch, num_graphs, slot, n_max)
        scale = x.shape[-1] ** 0.25
        u_q = torch.matmul(q / scale, self.omega)
        u_k = torch.matmul(k / scale, self.omega)
        # the stabilizing shifts: the largest projection of every query, the
        # largest key projection of the whole structure
        phi_q = self._features(q, u_q, u_q.max(dim=-1, keepdim=True).values)
        if num_graphs == 1:
            phi_k = self._features(k, u_k, u_k.max())
            numerator = torch.matmul(phi_q, torch.matmul(phi_k.t(), v))
            denominator = torch.matmul(phi_q, phi_k.sum(0))
            return numerator / (denominator + 1e-8).unsqueeze(-1)
        m = u_k.shape[-1]
        padded = u_k.new_full((num_graphs, n_max, m), -math.inf)
        padded[batch, slot] = u_k
        k_shift = padded.flatten(1).max(dim=1).values[batch].unsqueeze(-1)
        phi_k = self._features(k, u_k, k_shift)
        pk = phi_k.new_zeros((num_graphs, n_max, m))
        pk[batch, slot] = phi_k
        pv = v.new_zeros((num_graphs, n_max, v.shape[-1]))
        pv[batch, slot] = v
        pq = phi_q.new_zeros((num_graphs, n_max, m))
        pq[batch, slot] = phi_q
        numerator = torch.bmm(pq, torch.bmm(pk.transpose(1, 2), pv))[batch, slot]
        denominator = (pq * pk.sum(1, keepdim=True)).sum(-1)[batch, slot]
        return numerator / (denominator + 1e-8).unsqueeze(-1)

    def _exact(self, q: Tensor, k: Tensor, v: Tensor, batch: Tensor, num_graphs: int,
               slot: Tensor, n_max: int) -> Tensor:
        """Exact softmax attention within every structure (eq 19)."""
        d = q.shape[-1]
        pq = q.new_zeros((num_graphs, n_max, d))
        pq[batch, slot] = q
        pk = k.new_zeros((num_graphs, n_max, d))
        pk[batch, slot] = k
        pv = v.new_zeros((num_graphs, n_max, v.shape[-1]))
        pv[batch, slot] = v
        valid = torch.zeros((num_graphs, n_max), dtype=torch.bool, device=q.device)
        valid[batch, slot] = True
        dot = torch.bmm(pq, pk.transpose(1, 2))
        mask = valid.unsqueeze(1) & valid.unsqueeze(2)
        shift = torch.where(mask, dot, torch.full_like(dot, -math.inf)).flatten(1).max(dim=1).values
        a = torch.exp((dot - shift[:, None, None]) / math.sqrt(d))
        a = torch.where(mask, a, torch.zeros_like(a))
        a = a / (a.sum(-1, keepdim=True) + 1e-8)
        return torch.bmm(a, pv)[batch, slot]


class _InteractionModule(nn.Module):
    """One refinement step ``x -> (x', y)`` (eq 11)."""

    def __init__(self, n_features: int, n_rbf: int, n_residual: int, nonlocal_: bool,
                 n_random_features: Optional[int], generator: Optional[torch.Generator]):
        super().__init__()
        self.residual_pre = nn.ModuleList([_Residual(n_features) for _ in range(n_residual)])
        self.local_interaction = _LocalInteraction(n_features, n_rbf, n_residual)
        self.nonlocal_interaction = (_NonlocalInteraction(n_features, n_residual,
                                                          n_random_features, generator)
                                     if nonlocal_ else None)
        self.residual_post = nn.ModuleList([_Residual(n_features) for _ in range(n_residual)])
        self.resmlp_y = _ResidualMLP(n_features, n_residual)

    def forward(self, x: Tensor, rho: Tensor, y1: Tensor, y2: Tensor, src: Tensor,
                dst: Tensor, batch: Tensor, num_graphs: int, slot: Tensor, n_max: int,
                use_p: bool, use_d: bool, use_nonlocal: bool) -> Tuple[Tensor, Tensor]:
        for block in self.residual_pre:
            x = block(x)
        h = x + self.local_interaction(x, rho, y1, y2, src, dst, use_p, use_d)
        nonlocal_interaction = self.nonlocal_interaction
        if nonlocal_interaction is not None and use_nonlocal:
            h = h + nonlocal_interaction(x, batch, num_graphs, slot, n_max)
        for block in self.residual_post:
            h = block(h)
        return h, self.resmlp_y(h)


class _ZBLRepulsion(nn.Module):
    """Learnable ZBL-like nuclear repulsion (eq 22).

    ``E = k_e sum_{i<j} Z_i Z_j / r f_cut(r) sum_k c_k exp(-a_k r (Z_i^p + Z_j^p) / d)``;
    all parameters are kept positive by a softplus and the ``c_k`` are
    normalized to sum to one, so the bare Coulomb repulsion is recovered as
    ``r -> 0``. The parameters start from the universal ZBL values; ``_adiv``
    holds ``1 / d``.
    """

    k_e: float

    def __init__(self):
        super().__init__()
        self.k_e = COULOMB_CONSTANT
        self._adiv = nn.Parameter(torch.tensor(float(softplus_inverse(1.0 / (0.8854 * BOHR)))))
        self._apow = nn.Parameter(torch.tensor(float(softplus_inverse(0.23))))
        self._c = nn.Parameter(torch.tensor(softplus_inverse(
            np.array([0.18180, 0.50990, 0.28020, 0.02817])), dtype=torch.get_default_dtype()))
        self._a = nn.Parameter(torch.tensor(softplus_inverse(
            np.array([3.20000, 0.94230, 0.40280, 0.20160])), dtype=torch.get_default_dtype()))

    def forward(self, zf: Tensor, r: Tensor, fcut: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """Per-atom repulsion energies ``(N,)`` in eV over the local edges."""
        zp = zf ** F.softplus(self._apow)
        a = (zp[dst] + zp[src]) * F.softplus(self._adiv)
        c = F.softplus(self._c)
        c = c / c.sum()
        screening = (c * torch.exp(-F.softplus(self._a) * (a * r).unsqueeze(-1))).sum(-1)
        pair = 0.5 * self.k_e * zf[dst] * zf[src] / r * screening * fcut
        return scatter_sum(pair, dst, zf.shape[0])


class _Electrostatics(nn.Module):
    """Damped point-charge electrostatics (eqs 23 and 25).

    Parameters
    ----------
    cutoff : float
        The local cutoff ``r_c``; the damping switches off between ``r_c / 4``
        and ``3 r_c / 4``.
    lr_cutoff : float or None
        Optional truncation of the kernels (the reference code's shifted
        forms, zero value and slope at ``lr_cutoff``).
    """

    k_e: float
    r_on: float
    r_off: float
    lr_cutoff: Optional[float]

    def __init__(self, cutoff: float, lr_cutoff: Optional[float]):
        super().__init__()
        self.k_e = COULOMB_CONSTANT
        self.r_on = 0.25 * float(cutoff)
        self.r_off = 0.75 * float(cutoff)
        self.lr_cutoff = None if lr_cutoff is None else float(lr_cutoff)

    def kernel(self, r: Tensor) -> Tensor:
        """``f / sqrt(r^2 + 1) + (1 - f) / r`` (shifted when ``lr_cutoff`` is set)."""
        f = smooth_switch(r, self.r_on, self.r_off)
        coulomb = 1.0 / r
        damped = 1.0 / torch.sqrt(r * r + 1.0)
        lr = self.lr_cutoff
        if lr is not None:
            c3 = (lr * lr + 1.0) ** 1.5
            coulomb = coulomb + r / (lr * lr) - 2.0 / lr
            damped = damped + r * lr / c3 - (2.0 * lr * lr + 1.0) / c3
            inside = r < lr
            coulomb = torch.where(inside, coulomb, torch.zeros_like(r))
            damped = torch.where(inside, damped, torch.zeros_like(r))
        return f * damped + (1.0 - f) * coulomb

    def forward(self, q: Tensor, src: Tensor, dst: Tensor, r: Tensor) -> Tensor:
        """Per-atom energies ``(N,)`` in eV over ordered pairs (both directions present)."""
        pair = 0.5 * self.k_e * q[src] * q[dst] * self.kernel(r)
        return scatter_sum(pair, dst, q.shape[0])

    def damping(self, q: Tensor, src: Tensor, dst: Tensor, r: Tensor) -> Tensor:
        """Per-atom short-range part ``k_e f (1 / sqrt(r^2 + 1) - 1 / r)`` (for the Ewald path)."""
        f = smooth_switch(r, self.r_on, self.r_off)
        pair = 0.5 * self.k_e * q[src] * q[dst] * f * (1.0 / torch.sqrt(r * r + 1.0) - 1.0 / r)
        return scatter_sum(pair, dst, q.shape[0])


class _ChargeD4(nn.Module):
    """Two-body D4 dispersion with the model's charges (eq 27).

    ``E = -sum_{i<j} sum_{n=6,8} s_n C_n^ij f_damp,n(r_ij)`` with the
    rational (BJ) damping, ``C6`` from the Casimir-Polder integral of the
    charge- and CN-dependent D4 polarizabilities and ``C8 = 3 C6 Q_i Q_j``.
    ``s8``, ``a1``, ``a2`` and the scale ``s_q`` of the reference charges are
    learnable (kept positive by a softplus), ``s6 = 1`` is fixed; the
    starting values are the Hartree-Fock parameters. The reference tables
    are those of :class:`~xnn.common.models.d4.DFTD4`.

    Parameters
    ----------
    lr_cutoff : float or None
        Optional truncation (Angstrom): the damped ``r^-n`` terms are shifted
        to zero value and slope there and the coordination number is switched
        off over the last 0.529 bohr, as in the reference code.
    """

    bohr: float
    hartree: float
    ga: float
    gc: float
    wf: float
    kcn: float
    k4: float
    k5: float
    k6: float
    lr_cutoff: Optional[float]

    def __init__(self, s6: float = 1.0, s8: float = 1.61679827, a1: float = 0.44959224,
                 a2: float = 3.35743605, lr_cutoff: Optional[float] = None):
        super().__init__()
        self.bohr, self.hartree = BOHR, HARTREE
        self.lr_cutoff = None if lr_cutoff is None else float(lr_cutoff)
        constants, tables = _d4_tables()
        self.ga, self.gc, self.wf = constants["ga"], constants["gc"], constants["wf"]
        self.kcn, self.k4, self.k5, self.k6 = (constants["kcn"], constants["k4"],
                                               constants["k5"], constants["k6"])
        dtype = torch.get_default_dtype()
        for name, table in tables.items():
            self.register_buffer(name, table.clone() if not table.is_floating_point()
                                 else table.to(dtype, copy=True), persistent=False)
        self.register_buffer("_s6", torch.tensor(float(softplus_inverse(s6)), dtype=dtype))
        self._s8 = nn.Parameter(torch.tensor(float(softplus_inverse(s8)), dtype=dtype))
        self._a1 = nn.Parameter(torch.tensor(float(softplus_inverse(a1)), dtype=dtype))
        self._a2 = nn.Parameter(torch.tensor(float(softplus_inverse(a2)), dtype=dtype))
        self._scaleq = nn.Parameter(torch.tensor(float(softplus_inverse(1.0)), dtype=dtype))

    @torch.jit.ignore
    def exact_constants(self) -> bool:
        """Restore the float64 reference tables exactly after a cast; whether any changed."""
        return _restore_tables(self, _d4_tables()[1])

    def _zeta(self, hardness: Tensor, qref: Tensor, qmod: Tensor) -> Tensor:
        """Charge scaling ``exp(ga (1 - exp(gamma (1 - qref / qmod))))``, ``exp(ga)`` for ``qmod <= 0``."""
        positive = qmod > 1e-8
        safe = torch.where(positive, qmod, torch.ones_like(qmod))
        scaled = torch.exp(self.ga * (1.0 - torch.exp(hardness * self.gc * (1.0 - qref / safe))))
        return torch.where(positive, scaled, torch.full_like(qmod, math.exp(self.ga)))

    def reference_polarizabilities(self, scale_q: Tensor) -> Tensor:
        """Atom-in-molecule reference ``alpha(i omega)`` ``(23, 7, 119)`` with the hydrogen charges scaled."""
        present = self.refsys > 0
        secondary = torch.where(present, self.refsys, torch.zeros_like(self.refsys))
        zeff_x = self.zeff[secondary]
        scale = self._zeta(self.hardness[secondary], zeff_x, self.refh * scale_q + zeff_x)
        alpha_x = self.sscale[secondary] * self.secaiw[:, secondary] * scale
        alpha = self.ascale * (self.alphaiw - self.hcount * alpha_x)
        return torch.where(present, torch.clamp(alpha, min=0.0), torch.zeros_like(alpha))

    def forward(self, z: Tensor, q: Tensor, src: Tensor, dst: Tensor, r_angstrom: Tensor) -> Tensor:
        """Per-atom dispersion energies ``(N,)`` in eV over ordered pairs."""
        n = z.shape[0]
        r = r_angstrom / self.bohr
        zi, zj = z[dst], z[src]
        rc = self.rcov[zi] + self.rcov[zj]
        count = _erf_count(r, rc, self.kcn) * self.k4 * torch.exp(
            -((self.en[zi] - self.en[zj]).abs() + self.k5) ** 2 / self.k6)
        lr = self.lr_cutoff
        if lr is not None:
            cut = lr / self.bohr
            count = count * smooth_switch(r, cut - self.bohr, cut)
        cn = scatter_sum(count, dst, n)
        refcn = self.refcn[:, z].t()
        valid = torch.arange(refcn.shape[1], device=z.device)[None, :] < self.nref[z][:, None]
        weights = gaussian_reference_weights(cn, refcn, valid, self.wf, self.ngw[:, z].t(), 1e-8)
        scale_q = F.softplus(self._scaleq)
        zeff = self.zeff[z][:, None]
        zeta = self._zeta(self.hardness[z][:, None], zeff + self.refq[:, z].t() * scale_q,
                          zeff + q[:, None])
        alpha_ref = self.reference_polarizabilities(scale_q)[:, :, z]          # (23, 7, N)
        alpha_iw = torch.einsum("nr,krn->nk", [weights * zeta, alpha_ref])     # (N, 23)
        c6 = (3.0 / math.pi) * (alpha_iw[dst] * alpha_iw[src] * self.cp_weights).sum(-1)
        rr = 3.0 * self.r4r2[zi] * self.r4r2[zj]
        r0 = F.softplus(self._a1) * torch.sqrt(rr) + F.softplus(self._a2)
        r0_6 = r0 ** 6
        r0_8 = r0 ** 8
        t6 = 1.0 / (r ** 6 + r0_6)
        t8 = 1.0 / (r ** 8 + r0_8)
        if lr is not None:
            cut = lr / self.bohr
            c6c, c8c = cut ** 6, cut ** 8
            tail = r / cut - 1.0
            t6 = t6 - 1.0 / (c6c + r0_6) + 6.0 * c6c / (c6c + r0_6) ** 2 * tail
            t8 = t8 - 1.0 / (c8c + r0_8) + 8.0 * c8c / (c8c + r0_8) ** 2 * tail
            inside = r < cut
            t6 = torch.where(inside, t6, torch.zeros_like(t6))
            t8 = torch.where(inside, t8, torch.zeros_like(t8))
        pair = -0.5 * self.hartree * c6 * (F.softplus(self._s6) * t6
                                           + F.softplus(self._s8) * rr * t8)
        return scatter_sum(pair, dst, n)


@register_model("spookynet")
class SpookyNet(InteratomicPotential):
    """SpookyNet potential with charge/spin embeddings, nonlocal attention and physical terms.

    The defaults are the paper's architecture: ``F = 128`` features,
    ``T = 6`` interaction modules, ``K = 16`` exponential Bernstein radial
    functions, a cutoff of 10 bohr (5.29 Angstrom), FAVOR+ attention with
    ``F`` random features, and the repulsion, electrostatics and D4 terms.
    See the module docstring for the equations.

    Parameters
    ----------
    n_features : int, optional
        Feature width ``F``, by default 128.
    n_interactions : int, optional
        Interaction modules ``T``, by default 6.
    n_rbf : int, optional
        Radial functions ``K``, by default 16.
    cutoff : float, optional
        Local cutoff ``r_c`` (Angstrom), by default 10 bohr.
    lr_cutoff : float or None, optional
        Cutoff of the electrostatics and dispersion; ``None`` (default) sums
        over all pairs of every (molecular) structure. Required for periodic
        structures. The neighbor-list radius ``cutoff`` attribute is the
        larger of the two.
    n_residual : int, optional
        Residual blocks in every residual stack and residual MLP, by default 1.
    charge_embedding, spin_embedding : bool, optional
        Include ``e_Q`` and ``e_S`` (eq 1), by default ``True``.
    nonlocal_interactions : bool, optional
        Include the attention blocks (eq 18), by default ``True``.
    attention : str, optional
        ``"favor"`` (default; FAVOR+, eq 20) or ``"exact"`` (eq 19).
    n_random_features : int or None, optional
        FAVOR+ features, by default ``n_features``.
    zbl_repulsion, electrostatics, d4_dispersion : bool, optional
        Include the physical terms of eq 4, by default ``True``.
    exp_weighting : bool, optional
        Weight the radial functions by ``exp(-gamma r)`` (reference-code
        option), by default ``False``.
    ewald : bool, optional
        Ewald-sum the electrostatics of periodic structures (needs
        ``lr_cutoff``, the real-space cutoff), by default ``False``.
    ewald_accuracy : float, optional
        Relative accuracy of the Ewald sum, by default 1e-6.
    seed : int or None, optional
        Seed of the FAVOR+ random features (``None``: the global generator).
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``.
    atomic_energies : array-like or None, optional
        Per-species energies loaded into ``atom_ref`` (the element energy
        biases of eq 21).

    Attributes
    ----------
    cutoff : float
        Neighbor-list radius.
    local_cutoff : float
        The cutoff of the local interactions and the repulsion.
    atom_ref, charge_ref : torch.nn.Embedding
        Element energy and charge biases (eqs 21, 24), starting at zero.
    output : torch.nn.Linear
        Bias-free map of ``f`` to the atomic energy and charge.
    use_p_orbitals, use_d_orbitals, use_nonlocal : bool
        Evaluation switches (all ``True``): setting one to ``False`` drops
        the p- or d-like terms of eq 12 or the nonlocal features ``n`` of
        eq 11, the decompositions of the paper's Fig. 2.
    """

    head_modules = ("output", "atom_ref", "charge_ref")
    lr_cutoff: Optional[float]
    use_p_orbitals: bool
    use_d_orbitals: bool
    use_nonlocal: bool
    ewald: bool
    ewald_alpha: float
    ewald_k_cutoff: float

    def __init__(self, n_features: int = 128, n_interactions: int = 6, n_rbf: int = 16,
                 cutoff: float = 10.0 * BOHR, lr_cutoff: Optional[float] = None,
                 n_residual: int = 1, charge_embedding: bool = True,
                 spin_embedding: bool = True, nonlocal_interactions: bool = True,
                 attention: str = "favor", n_random_features: Optional[int] = None,
                 zbl_repulsion: bool = True, electrostatics: bool = True,
                 d4_dispersion: bool = True, exp_weighting: bool = False,
                 ewald: bool = False, ewald_accuracy: float = 1.0e-6,
                 seed: Optional[int] = None, species=None, atomic_energies=None):
        super().__init__()
        if attention not in ("favor", "exact"):
            raise ValueError(f"attention must be 'favor' or 'exact', got {attention!r}")
        if ewald and lr_cutoff is None:
            raise ValueError("ewald=True needs lr_cutoff (the real-space cutoff)")
        self.local_cutoff = float(cutoff)
        self.lr_cutoff = None if lr_cutoff is None else float(lr_cutoff)
        self.cutoff = max(self.local_cutoff, self.lr_cutoff or 0.0)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.use_p_orbitals = True
        self.use_d_orbitals = True
        self.use_nonlocal = True
        self.ewald = bool(ewald)
        self.ewald_alpha, self.ewald_k_cutoff = 0.0, 0.0
        if self.ewald:
            self.ewald_alpha, self.ewald_k_cutoff = ewald_parameters(self.lr_cutoff,
                                                                     float(ewald_accuracy))
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        m = None if attention == "exact" else int(n_random_features or n_features)

        self.nuclear_embedding = _NuclearEmbedding(n_features)
        self.charge_embedding = (_ElectronicEmbedding(n_features, n_residual, charge=True)
                                 if charge_embedding else None)
        self.spin_embedding = (_ElectronicEmbedding(n_features, n_residual, charge=False)
                               if spin_embedding else None)
        self.radial_basis = ExponentialBernsteinRBF(n_rbf, exp_weighting=exp_weighting)
        self.cutoff_fn = MollifierCutoff(self.local_cutoff)
        self.interactions = nn.ModuleList([
            _InteractionModule(n_features, n_rbf, n_residual, nonlocal_interactions, m, generator)
            for _ in range(n_interactions)])
        self.output = nn.Linear(n_features, 2, bias=False)
        glorot_orthogonal_(self.output.weight)
        self.atom_ref = nn.Embedding(MAX_Z, 1)
        self.charge_ref = nn.Embedding(MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        nn.init.zeros_(self.charge_ref.weight)
        self.repulsion = _ZBLRepulsion() if zbl_repulsion else None
        self.electrostatics = (_Electrostatics(self.local_cutoff, self.lr_cutoff)
                               if electrostatics else None)
        self.dispersion = _ChargeD4(lr_cutoff=self.lr_cutoff) if d4_dispersion else None
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)

    def _apply(self, fn, recurse=True):
        # constant tables built in float32 are rebuilt exactly once float64
        out = super()._apply(fn, recurse)
        for module in self.modules():
            restore = getattr(module, "exact_constants", None)
            if module is not self and callable(restore):
                restore()
        return out

    @torch.jit.ignore
    def set_atomic_energies(self, species, values) -> None:
        """Set the element energy biases ``atom_ref`` of ``species`` to ``values``.

        Raises
        ------
        ValueError
            If the number of values does not match the number of species.
        """
        ae = torch.as_tensor(values, dtype=self.atom_ref.weight.dtype)
        species = [int(z) for z in species]
        if ae.numel() != len(species):
            raise ValueError(f"got {ae.numel()} atomic energies for {len(species)} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(species), 0] = ae

    def _long_range_pairs(self, edge_index: Tensor, r_all: Tensor, pos: Tensor,
                          batch: Tensor, periodic: bool) -> Tuple[Tensor, Tensor, Tensor]:
        """Ordered pairs ``(src, dst)`` and distances of the electrostatics and dispersion."""
        lr = self.lr_cutoff
        if lr is not None:
            keep = r_all < lr
            return edge_index[0][keep], edge_index[1][keep], r_all[keep]
        if periodic:
            raise ValueError("SpookyNet sums the long-range terms over all pairs of a "
                             "molecule; periodic structures need lr_cutoff")
        dst, src = all_pairs(batch)
        r = torch.linalg.norm(pos[dst] - pos[src], dim=-1)
        return src, dst, r

    def _evaluate(self, z: Tensor, edge_index: Tensor, edge_vec: Tensor, pos: Tensor,
                  cell: Optional[Tensor], pbc: Optional[Tensor], batch: Tensor,
                  num_graphs: int, total_charge: Tensor, n_unpaired: Tensor) -> Dict[str, Tensor]:
        """Features, energies and charges of a batch (the scriptable core)."""
        dtype = self.output.weight.dtype
        n = z.shape[0]
        r_all = torch.linalg.norm(edge_vec, dim=-1)
        local = r_all < self.local_cutoff
        src, dst = edge_index[0][local], edge_index[1][local]
        r = r_all[local]
        # the paper's r_ij = r_j - r_i is the negative of the xnn edge vector
        u = -edge_vec[local] / r.unsqueeze(-1)
        ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
        y1 = torch.stack([uy, uz, ux], dim=-1)
        sqrt3 = math.sqrt(3.0)
        y2 = torch.stack([sqrt3 * ux * uy, sqrt3 * uy * uz, 0.5 * (3.0 * uz * uz - 1.0),
                          sqrt3 * ux * uz, 0.5 * sqrt3 * (ux * ux - uy * uy)], dim=-1)
        fcut = self.cutoff_fn(r)
        rho = self.radial_basis(r) * fcut.unsqueeze(-1)

        e_z = self.nuclear_embedding(z)
        x = e_z
        charge_embedding = self.charge_embedding
        if charge_embedding is not None:
            x = x + charge_embedding(e_z, total_charge, batch, num_graphs)
        spin_embedding = self.spin_embedding
        if spin_embedding is not None:
            x = x + spin_embedding(e_z, n_unpaired, batch, num_graphs)

        slot = torch.arange(n, device=z.device)
        n_max = n
        if num_graphs > 1:
            slot, n_max = _structure_slots(batch, num_graphs)
        f = torch.zeros_like(x)
        for module in self.interactions:
            x, y = module(x, rho, y1, y2, src, dst, batch, num_graphs, slot, n_max,
                          self.use_p_orbitals, self.use_d_orbitals, self.use_nonlocal)
            f = f + y

        out = self.output(f)
        e_nn = out[:, 0] + self.atom_ref(z).squeeze(-1)
        q = out[:, 1] + self.charge_ref(z).squeeze(-1)
        counts = torch.bincount(batch, minlength=num_graphs).to(dtype)
        q = q + ((total_charge - scatter_sum(q, batch, num_graphs)) / counts)[batch]

        zero = torch.zeros_like(e_nn)
        e_rep, e_ele, e_vdw = zero, zero, zero
        repulsion = self.repulsion
        if repulsion is not None:
            e_rep = repulsion(z.to(dtype), r, fcut, src, dst)
        electrostatics = self.electrostatics
        dispersion = self.dispersion
        if electrostatics is not None or dispersion is not None:
            periodic_atoms = torch.zeros(n, dtype=torch.bool, device=z.device)
            if cell is not None and pbc is not None:
                periodic_atoms = pbc.any(dim=1)[batch]
            lr_src, lr_dst, lr_r = self._long_range_pairs(edge_index, r_all, pos, batch,
                                                          bool(periodic_atoms.any()))
            if electrostatics is not None:
                if self.ewald and cell is not None and pbc is not None \
                        and bool(periodic_atoms.any()):
                    e_ele = self._ewald(q, pos, cell, pbc, batch, num_graphs,
                                        lr_src, lr_dst, lr_r, periodic_atoms)
                else:
                    e_ele = electrostatics(q, lr_src, lr_dst, lr_r)
            if dispersion is not None:
                e_vdw = dispersion(z, q, lr_src, lr_dst, lr_r)
        return {"node_features": f, "node_energy": e_nn + e_rep + e_ele + e_vdw,
                "charges": q, "node_repulsion": e_rep, "node_electrostatics": e_ele,
                "node_dispersion": e_vdw}

    def _ewald(self, q: Tensor, pos: Tensor, cell: Tensor,
               pbc: Tensor, batch: Tensor, num_graphs: int, src: Tensor, dst: Tensor,
               r: Tensor, periodic_atoms: Tensor) -> Tensor:
        """Ewald sum for the periodic structures, the shifted kernel for the others."""
        electrostatics = self.electrostatics
        assert electrostatics is not None
        periodic_pair = periodic_atoms[dst]
        free = ~periodic_pair
        node = electrostatics(q, src[free], dst[free], r[free])
        on = periodic_pair
        node = node + electrostatics.damping(q, src[on], dst[on], r[on])
        lr = self.lr_cutoff
        assert lr is not None
        for i in range(num_graphs):
            flags = pbc[i]
            if not bool(flags.any()):
                continue
            if not bool(flags.all()):
                raise ValueError("ewald=True needs structures periodic in all three directions")
            mask = batch == i
            idx = mask.nonzero().squeeze(1)
            on_edge = mask[dst]
            remap = torch.cumsum(mask.to(torch.long), 0) - 1
            local_index = torch.stack([remap[src[on_edge]], remap[dst[on_edge]]])
            part = coulomb_ewald(q[idx], pos[idx], cell[i], local_index, r[on_edge],
                                 self.ewald_alpha, lr, self.ewald_k_cutoff, electrostatics.k_e)
            node = node.index_add(0, idx, part.to(node.dtype))
        return node

    @torch.jit.export
    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor,
                                     edge_vec: Tensor, pos: Tensor, cell: Tensor, pbc: Tensor,
                                     total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript core of one structure: descriptors, atomic energies and charges.

        Parameters
        ----------
        atomic_numbers : Tensor
            ``(N,)``.
        edge_index : Tensor
            ``(2, E)`` within ``self.cutoff``.
        edge_vec : Tensor
            ``(E, 3)``.
        pos : Tensor
            ``(N, 3)``.
        cell : Tensor
            ``(3, 3)`` (ignored unless ``pbc`` has a true entry).
        pbc : Tensor
            ``(3,)`` bool.
        total_charge : float
            Net charge of the structure.
        spin_multiplicity : float
            Its spin multiplicity ``2S + 1``.

        Returns
        -------
        tuple of Tensor
            ``node_features (N, F)``, ``node_energy (N,)`` (all terms) and
            ``charges (N,)``.
        """
        n = atomic_numbers.shape[0]
        device = atomic_numbers.device
        dtype = self.output.weight.dtype
        batch = torch.zeros(n, dtype=torch.long, device=device)
        cell_b: Optional[Tensor] = None
        pbc_b: Optional[Tensor] = None
        if bool(pbc.any()):
            cell_b = cell.unsqueeze(0)
            pbc_b = pbc.unsqueeze(0)
        out = self._evaluate(atomic_numbers, edge_index, edge_vec, pos, cell_b, pbc_b, batch, 1,
                             torch.full((1,), total_charge, dtype=dtype, device=device),
                             torch.full((1,), spin_multiplicity - 1.0, dtype=dtype,
                                        device=device))
        return out["node_features"], out["node_energy"], out["charges"]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies, charges and the dipole moment of a (batched) graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched graph built with ``self.cutoff``. ``total_charge``
            (``None``: neutral) and ``spin_multiplicity`` (``None``: singlet)
            set the electronic state.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``,
            ``"node_features"`` (the descriptors ``f``, ``(N, F)``),
            ``"charges"`` ``(N,)``, ``"dipole"`` ``(B, 3)`` (``sum q_i r_i``,
            e Angstrom) and the per-structure terms ``"energy_repulsion"``,
            ``"energy_electrostatics"`` and ``"energy_dispersion"``.
        """
        dtype = self.output.weight.dtype
        b = data.num_graphs
        device = data.atomic_numbers.device
        q_tot = (data.total_charge.to(dtype) if data.total_charge is not None
                 else torch.zeros(b, dtype=dtype, device=device))
        n_unpaired = (data.spin_multiplicity.to(dtype) - 1.0
                      if data.spin_multiplicity is not None
                      else torch.zeros(b, dtype=dtype, device=device))
        out = self._evaluate(data.atomic_numbers, data.edge_index, data.edge_vectors(),
                             data.pos.to(dtype), data.cell, data.pbc, data.batch, b, q_tot,
                             n_unpaired)
        charges = out["charges"]
        return {
            "node_energy": out["node_energy"],
            "energy": self.aggregate_energy(out["node_energy"], data),
            "node_features": out["node_features"],
            "charges": charges,
            "dipole": scatter_sum(charges.unsqueeze(-1) * data.pos.to(dtype), data.batch, b),
            "energy_repulsion": structure_sum(out["node_repulsion"], data.batch, b),
            "energy_electrostatics": structure_sum(out["node_electrostatics"], data.batch, b),
            "energy_dispersion": structure_sum(out["node_dispersion"], data.batch, b),
        }

    @classmethod
    def from_config(cls, cfg) -> "SpookyNet":
        """Build a :class:`SpookyNet` from a configuration object.

        Core fields: ``cfg.n_features``, ``cfg.n_interactions``,
        ``cfg.n_rbf``, ``cfg.cutoff``. The other constructor arguments are
        read from ``cfg.extra``; the reference code's spellings
        (``num_features``, ``num_modules``, ``use_zbl_repulsion``, ...) are
        translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        SpookyNet
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        lr_cutoff = extra.get("lr_cutoff")
        n_random = extra.get("n_random_features")
        seed = extra.get("seed")
        return cls(
            n_features=cfg.n_features, n_interactions=cfg.n_interactions, n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            lr_cutoff=None if lr_cutoff is None else float(lr_cutoff),
            n_residual=int(extra.get("n_residual", 1)),
            charge_embedding=bool(extra.get("charge_embedding", True)),
            spin_embedding=bool(extra.get("spin_embedding", True)),
            nonlocal_interactions=bool(extra.get("nonlocal_interactions", True)),
            attention=str(extra.get("attention", "favor")),
            n_random_features=None if n_random is None else int(n_random),
            zbl_repulsion=bool(extra.get("zbl_repulsion", True)),
            electrostatics=bool(extra.get("electrostatics", True)),
            d4_dispersion=bool(extra.get("d4_dispersion", True)),
            exp_weighting=bool(extra.get("exp_weighting", False)),
            ewald=bool(extra.get("ewald", False)),
            ewald_accuracy=float(extra.get("ewald_accuracy", 1.0e-6)),
            seed=None if seed is None else int(seed),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
        )

    @classmethod
    def from_reference_checkpoint(cls, source) -> "SpookyNet":
        """Load a model saved by the reference code (``SpookyNet.save``).

        Reads the hyperparameters and the state dict of the reference
        format (the current layout: linear electronic embeddings, irreducible
        d functions, exponential Bernstein radial basis, swish activations),
        renames the parameters and reverses the order of the radial
        functions. The model is returned in eval mode, in float32 or float64
        as stored.

        Parameters
        ----------
        source : str, pathlib.Path or dict
            Path of the ``.pth`` file, or its loaded content.

        Returns
        -------
        SpookyNet
            The model.

        Raises
        ------
        ValueError
            For a checkpoint layout this implementation does not cover.
        """
        state = source
        if not isinstance(source, dict):
            state = torch.load(source, map_location="cpu", weights_only=True)
        hp = {k: v for k, v in state.items() if k != "state_dict"}
        # (key, required value, value of the older layouts that lack the key)
        requirements = [("activation", "swish", "swish"),
                        ("basis_functions", "exp-bernstein", "exp-bernstein"),
                        ("use_irreps", True, False), ("use_nonlinear_embedding", False, True),
                        ("Zmax", MAX_Z, MAX_Z)]
        for key, value, legacy in requirements:
            if hp.get(key, legacy) != value:
                raise ValueError(f"reference checkpoint has {key}={hp.get(key)!r}; "
                                 f"only {key}={value!r} is supported")
        residuals = {int(v) for k, v in hp.items() if k.startswith("num_residual")}
        if len(residuals) > 1:
            raise ValueError(f"different residual depths {sorted(residuals)} are not supported")
        sd = state["state_dict"]
        dtype = sd["output.weight"].dtype
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = cls(n_features=int(hp["num_features"]), n_interactions=int(hp["num_modules"]),
                        n_rbf=int(hp["num_basis_functions"]), cutoff=float(hp["cutoff"]),
                        lr_cutoff=hp.get("lr_cutoff"),
                        n_residual=residuals.pop() if residuals else 1,
                        zbl_repulsion=bool(hp["use_zbl_repulsion"]),
                        electrostatics=bool(hp["use_electrostatics"]),
                        d4_dispersion=bool(hp["use_d4_dispersion"]),
                        exp_weighting=bool(hp["exp_weighting"]))
        finally:
            torch.set_default_dtype(previous)
        model.load_state_dict(reference_state_dict(sd))
        return model.eval()


def reference_state_dict(sd: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Translate a reference-code state dict to :class:`SpookyNet` parameter names.

    Parameters
    ----------
    sd : dict of str to Tensor
        The ``state_dict`` of a reference-code model.

    Returns
    -------
    dict of str to Tensor
        The state dict of the equivalent :class:`SpookyNet` (constant tables
        and the reference code's cached buffers are dropped; the radial
        weights are reordered).
    """
    renames = [
        (r"\.stack\.", "."),
        (r"^magmom_embedding\.", "spin_embedding."),
        (r"^module\.", "interactions."),
        (r"_embedding\.resblock\.", "_embedding.resmlp."),
        (r"\.local_interaction\.resblock_x\.", ".local_interaction.resmlp_c."),
        (r"\.resblock_([spdqkv])\.", r".resmlp_\1."),
        (r"\.local_interaction\.resblock\.", ".local_interaction.resmlp_l."),
        (r"^(interactions\.\d+)\.resblock\.", r"\1.resmlp_y."),
        (r"\.attention\.omega$", ".omega"),
        (r"^zbl_repulsion_energy\.", "repulsion."),
        (r"^d4_dispersion_energy\.", "dispersion."),
        (r"^radial_basis_functions\._alpha$", "radial_basis.gamma_raw"),
    ]
    scalars = ("repulsion._adiv", "repulsion._apow", "dispersion._s6", "dispersion._s8",
               "dispersion._a1", "dispersion._a2", "dispersion._scaleq", "radial_basis.gamma_raw")
    dropped = re.compile(r"keep_prob|nuclear_embedding\.(electron_config|embedding)"
                         r"|radial_basis_functions\.(logc|n|v)"
                         r"|d4_dispersion_energy\.(?!_s6$|_s8$|_a1$|_a2$|_scaleq$).*")
    out: Dict[str, Tensor] = {}
    zbl: Dict[str, List[Tensor]] = {"c": [], "a": []}
    for key in sorted(sd):
        value = sd[key]
        if key == "element_bias":
            out["atom_ref.weight"] = value[:, :1].clone()
            out["charge_ref.weight"] = value[:, 1:].clone()
            continue
        if dropped.fullmatch(key):
            continue
        coefficient = re.fullmatch(r"zbl_repulsion_energy\._([ca])\d", key)
        if coefficient is not None:
            zbl[coefficient.group(1)].append(value.reshape(1))
            continue
        new = key
        for pattern, replacement in renames:
            new = re.sub(pattern, replacement, new)
        if re.search(r"\.radial_[spd]\.weight$", new):
            value = value.flip(-1)
        if new in scalars:
            value = value.reshape(())
        out[new] = value
    if zbl["c"]:
        out["repulsion._c"] = torch.cat(zbl["c"])
        out["repulsion._a"] = torch.cat(zbl["a"])
    return out
