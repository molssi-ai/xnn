"""Ewald and particle-mesh Ewald sums for atomic multipoles and r^-6 dispersion.

The periodic electrostatics of the AMOEBA / HIPPO models (``amoeba.py``):

* the real-space ladder of the screened Coulomb kernel ``erfc(alpha R) / R``
  (:func:`ewald_ladder`), in the same ``D_n = (1/R d/dR)^n g`` form as the
  plain Coulomb ladder, so every pair term of the vacuum model carries over
  by swapping the kernel; a zero ``alpha`` gives the Coulomb ladder, which
  is how molecular structures of a mixed batch keep their vacuum terms;
* the reciprocal-space part on a B-spline mesh after Essmann *et al.*,
  *J. Chem. Phys.* **103**, 8577 (1995): the multipoles are spread with the
  cardinal B-splines and their first and second derivatives (a point dipole
  is the derivative of a point charge), the mesh potential is formed with
  the same influence function as for point charges, and the potential, the
  field and the field gradient are interpolated back onto the atoms with
  the same splines, so that the reciprocal energy is ``1/2 sum_i [q_i phi +
  d_i . grad phi + Q_i : grad grad phi]`` and the reciprocal field drives
  the induced dipoles (:func:`multipole_pme`); the influence function
  carries the least-squares spline moduli
  (:func:`~xnn.common.models.pme.least_squares_moduli`) rather than the
  Euler exponential-spline ones of the paper, which at coarse meshes
  leaves a far smaller mean error in the energies and reproduces Tinker's;
* the self-interaction terms of Gaussian-smeared multipoles, half the
  interaction of a multipole with its own smeared kernel ``erf(alpha R) / R``
  at zero separation, ``-alpha / sqrt(pi) [q^2 + 2 alpha^2 / 3 |d|^2 +
  8 alpha^4 / 5 Q:Q]`` for the quadrupole convention of the model, and
  the self field ``4 alpha^3 / (3 sqrt(pi)) d`` that the mesh field of a
  dipole at its own site must be corrected by;
* the Ewald sum of the ``-c_i c_j / R^6`` dispersion (Essmann *et al.*,
  eq. 5.1 and following): the real-space kernel ``g(alpha R) / R^6`` with
  ``g(x) = e^{-x^2} (1 + x^2 + x^4 / 2)``, the reciprocal kernel
  ``pi^(3/2) alpha^3 / 3 [(1 - 2 b^2) e^{-b^2} + 2 b^3 sqrt(pi) erfc(b)]``,
  ``b = k / (2 alpha)``, including the ``k = 0`` term, and the self term
  ``alpha^6 / 12 sum c_i^2`` (:func:`dispersion_pme`);
* the C2 polynomial switch between a taper distance and a cutoff
  (:func:`switching`): the unique quintic with unit value and zero first
  and second derivatives at the taper, and zero value and derivatives at
  the cutoff, ``1 - 10 x^3 + 15 x^4 - 6 x^5`` in the reduced distance;
* Tinker-compatible defaults: the Ewald coefficient that makes
  ``erfc(alpha r_c) / r_c`` equal ``1e-8`` at the real-space cutoff
  (:func:`ewald_coefficient`) and the mesh of at least 1.2 points per
  Angstrom (0.8 for dispersion) rounded up to an even size with prime
  factors 2, 3 and 5, at least 16 (:func:`pme_grid_size`).

Everything is written in differentiable tensor operations: forces follow
from the positions through the spline weights, the stress from the cell
through the reciprocal vectors, the volume and the fractional coordinates.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from xnn.common.models.ops import cell_volume
from xnn.common.models.pme import bspline_weights, mesh_wavevectors, spline_moduli

SQRT_PI = math.sqrt(math.pi)


def region(name: str):
    """Mark a function as a named region of the PyTorch profiler, which also
    becomes an NVTX range in Nsight timelines under
    ``torch.autograd.profiler.emit_nvtx``; it costs a few microseconds per
    call when no profiler runs."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if torch.compiler.is_compiling():
                return fn(*args, **kwargs)
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)
        return inner
    return wrap


_FUSED = contextvars.ContextVar("amoeba_fused", default=False)


@contextlib.contextmanager
def fused(active: bool):
    """Evaluate the :func:`fusable` functions called inside compiled (or not).

    Parameters
    ----------
    active : bool
        Whether they run compiled.
    """
    token = _FUSED.set(bool(active))
    try:
        yield
    finally:
        _FUSED.reset(token)


def fused_active() -> bool:
    """Whether the :func:`fusable` functions run compiled here (inside :func:`fused`)."""
    return _FUSED.get()


def _compile_options() -> dict:
    """The inductor options of the fused blocks: its combo kernels merge the independent small
    kernels of a block (the valence terms, the frames) into shared launches, where this version
    of PyTorch has them."""
    import torch._inductor.config as inductor_config
    return {"combo_kernels": True} if hasattr(inductor_config, "combo_kernels") else {}


def fusable(fn):
    """Mark a pure function of per-pair tensors for the fused path.

    Inside :func:`fused` it runs through ``torch.compile`` with dynamic sizes,
    which turns its chain of small elementwise operations into a few generated
    kernels in the forward and in the backward pass (independent ones merged
    into shared launches by the inductor's combo kernels). The first fused call runs
    eagerly, so evaluations that happen once never pay for the compilation.
    A compiled function has no double backward: the fused path serves
    evaluations with first derivatives (molecular dynamics, inference).
    """
    state: dict = {}

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if torch.compiler.is_compiling() or not _FUSED.get():
            return fn(*args, **kwargs)
        if "compiled" not in state:
            state["compiled"] = torch.compile(fn, dynamic=True, options=_compile_options())
            return fn(*args, **kwargs)
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            # no compilation can run inside a CUDA-graph capture: the compiled code of an earlier
            # call serves while its guards hold, the eager function otherwise
            return torch._dynamo.run(fn)(*args, **kwargs)
        return state["compiled"](*args, **kwargs)
    return inner


# defaults

def ewald_coefficient(cutoff: float, accuracy: float = 1e-8) -> float:
    """The Ewald coefficient for a real-space cutoff.

    The value ``alpha`` at which the screened interaction at the cutoff,
    ``erfc(alpha r_c) / r_c``, equals ``accuracy`` (bisection).

    Parameters
    ----------
    cutoff : float
        Real-space cutoff in Angstrom.
    accuracy : float, optional
        The screened interaction at the cutoff, by default ``1e-8``.

    Returns
    -------
    float
        ``alpha`` in 1/Angstrom.
    """
    lo, hi = 0.0, 0.5
    while math.erfc(hi * cutoff) / cutoff > accuracy:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid * cutoff) / cutoff > accuracy:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fft_friendly(n: int) -> bool:
    if n % 2:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def fft_mesh_size(n: int, minimum: int = 16) -> int:
    """The smallest even size of at least ``n`` and ``minimum`` whose prime
    factors are 2, 3 and 5 (Tinker rounds a requested mesh up the same way).

    Parameters
    ----------
    n : int
        The requested number of mesh points along one axis.
    minimum : int, optional
        The smallest mesh, by default 16.

    Returns
    -------
    int
        The mesh size.
    """
    n = max(int(n), int(minimum))
    while not _fft_friendly(n):
        n += 1
    return n


def pme_grid_size(lengths: Sequence[float], density: float = 1.2, minimum: int = 16) -> List[int]:
    """Mesh dimensions from the cell lengths: at least ``density`` points per
    Angstrom along each axis, rounded up to an even size whose prime factors
    are 2, 3 and 5, and at least ``minimum``.

    Parameters
    ----------
    lengths : sequence of float
        The three lattice-vector lengths in Angstrom.
    density : float, optional
        Points per Angstrom, by default 1.2 (0.8 is used for dispersion).
    minimum : int, optional
        The smallest mesh, by default 16.

    Returns
    -------
    list of int
        ``(K_1, K_2, K_3)``.
    """
    return [fft_mesh_size(math.ceil(float(length) * density - 1e-8), minimum) for length in lengths]


# switching and the real-space ladder

@fusable
def switching(R: Tensor, taper: float, cutoff: float) -> Tensor:
    """The C2 switch ``S(R)``: one below ``taper``, zero beyond ``cutoff``,
    ``1 - 10 x^3 + 15 x^4 - 6 x^5`` with ``x = (R - taper) / (cutoff - taper)``
    in between (unit value and zero first and second derivatives at the
    taper, zero value and derivatives at the cutoff).

    Parameters
    ----------
    R : Tensor
        Distances ``(M,)``.
    taper, cutoff : float
        The switching window in Angstrom (``taper < cutoff``).

    Returns
    -------
    Tensor
        ``S(R)`` ``(M,)``.
    """
    if taper >= cutoff:
        return (R < cutoff).to(R.dtype)
    x = ((R - taper) / (cutoff - taper)).clamp(0.0, 1.0)
    x3 = x * x * x
    return 1.0 - x3 * (10.0 - 15.0 * x + 6.0 * x * x)


@fusable
def ewald_ladder(R: Tensor, alpha: Tensor, order: int) -> List[Tensor]:
    """The ladder ``D_n = (1/R d/dR)^n [erfc(alpha R) / R]`` for ``n`` from 0 to ``order``.

    ``D_n = (-1)^n B_n`` with ``B_0 = erfc(alpha R) / R`` and
    ``B_n = [(2n - 1) B_{n-1} + 2^n alpha^(2n-1) e^{-alpha^2 R^2} / sqrt(pi)] / R^2``;
    a zero ``alpha`` gives the plain Coulomb ladder.

    Parameters
    ----------
    R : Tensor
        Distances ``(M,)``.
    alpha : Tensor
        Ewald coefficient per pair ``(M,)`` (zero for pairs of a molecular
        structure).
    order : int
        The highest derivative.

    Returns
    -------
    list of Tensor
        ``[D_0, ..., D_order]``.
    """
    inv2 = 1.0 / (R * R)
    expo = torch.exp(-(alpha * R) ** 2) / SQRT_PI
    b = torch.erfc(alpha * R) / R
    out = [b]
    for n in range(1, order + 1):
        b = ((2 * n - 1) * b + (2.0 ** n) * alpha ** (2 * n - 1) * expo) * inv2
        out.append(b if n % 2 == 0 else -b)
    return out


# B-splines with derivatives and the mesh

def bspline_with_derivatives(t: Tensor, order: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Cardinal B-spline weights ``M_p(t + j)`` and their first and second
    derivatives with respect to the scaled coordinate, for ``j`` from 0 to ``p - 1``.

    ``M_p'(x) = M_{p-1}(x) - M_{p-1}(x - 1)`` and
    ``M_p''(x) = M_{p-2}(x) - 2 M_{p-2}(x - 1) + M_{p-2}(x - 2)``.

    Parameters
    ----------
    t : Tensor
        Fractional offsets in ``[0, 1)``, shape ``(...)``.
    order : int
        Spline order ``p >= 4``.

    Returns
    -------
    tuple of Tensor
        ``(w, dw, d2w)``, each ``(..., p)``.
    """
    w = bspline_weights(t, order)
    w1 = bspline_weights(t, order - 1)
    w2 = bspline_weights(t, order - 2)
    pad = torch.nn.functional.pad
    dw = pad(w1, (0, 1)) - pad(w1, (1, 0))
    d2w = pad(w2, (0, 2)) - 2.0 * pad(w2, (1, 1)) + pad(w2, (2, 0))
    return w, dw, d2w


def matvec(A: Tensor, v: Tensor) -> Tensor:
    """``A v`` for stacks of ``3 x 3`` matrices ``(M, 3, 3)`` and vectors ``(M, 3)``.

    An elementwise product and sum: ``einsum`` would map it to a batched
    matrix-vector product, which the GPU libraries run as many launches of
    tiny problems far below the memory bandwidth.
    """
    return (A * v[:, None, :]).sum(-1)


def vecmat(v: Tensor, A: Tensor) -> Tensor:
    """``v^T A`` for stacks of vectors ``(M, 3)`` and matrices ``(M, 3, 3)``, as ``matvec``."""
    return (v[:, :, None] * A).sum(-2)


def matmul3(A: Tensor, B: Tensor) -> Tensor:
    """``A B`` for stacks of ``3 x 3`` matrices ``(M, 3, 3)``, as ``matvec``."""
    return (A[:, :, :, None] * B[:, None, :, :]).sum(-2)


def _outer3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """``(N, p, p, p)`` products of three per-axis weight sets ``(N, p)``."""
    return (a[:, :, None, None] * b[:, None, :, None] * c[:, None, None, :])


@fusable
def first_stencils(w: Tensor, dw: Tensor) -> dict:
    """The 3D spline weights ``"0"`` and their first derivatives ``"a_"``, each
    ``(N, p^3)``, from the per-axis weights and derivatives ``(N, 3, p)``."""
    n = w.shape[0]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    dx, dy, dz = dw[:, 0], dw[:, 1], dw[:, 2]
    return {key: _outer3(a, b, c).reshape(n, -1) for key, (a, b, c) in
            {"0": (wx, wy, wz), "0_": (dx, wy, wz), "1_": (wx, dy, wz), "2_": (wx, wy, dz)}.items()}


@fusable
def second_stencils(w: Tensor, dw: Tensor, d2w: Tensor) -> dict:
    """The second derivatives ``"ab"`` of the 3D spline weights, each ``(N, p^3)``."""
    n = w.shape[0]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    dx, dy, dz = dw[:, 0], dw[:, 1], dw[:, 2]
    return {key: _outer3(a, b, c).reshape(n, -1) for key, (a, b, c) in
            {"00": (d2w[:, 0], wy, wz), "11": (wx, d2w[:, 1], wz), "22": (wx, wy, d2w[:, 2]),
             "01": (dx, dy, wz), "02": (dx, wy, dz), "12": (wx, dy, dz)}.items()}


@fusable
def stencil_coefficients(q: Optional[Tensor], d: Optional[Tensor], Q: Optional[Tensor], jac: Tensor,
                         st: dict) -> Tensor:
    """The mesh coefficients ``(N, p^3)`` of multipoles ``q``, ``d``, ``Q`` (any of
    which may be ``None``) on the stencils ``st`` of their atoms; ``jac`` maps
    Cartesian to fractional mesh derivatives."""
    coeff = None
    if q is not None:
        coeff = q[:, None] * st["0"]
    if d is not None:
        df = vecmat(d, jac)                                               # d^f_a = sum_beta d_beta J[beta, a]
        term = df[:, 0:1] * st["0_"] + df[:, 1:2] * st["1_"] + df[:, 2:3] * st["2_"]
        coeff = term if coeff is None else coeff + term
    if Q is not None:
        Qf = matmul3(matmul3(jac.transpose(1, 2), Q), jac)                # J^T Q J
        term = (Qf[:, 0, 0, None] * st["00"] + Qf[:, 1, 1, None] * st["11"] + Qf[:, 2, 2, None] * st["22"]
                + 2.0 * (Qf[:, 0, 1, None] * st["01"] + Qf[:, 0, 2, None] * st["02"]
                         + Qf[:, 1, 2, None] * st["12"]))
        coeff = term if coeff is None else coeff + term
    return coeff


@fusable
def stencil_potential(at: Tensor, jac: Tensor, st: dict, hessian: bool):
    """The potential ``(N,)``, its Cartesian gradient ``(N, 3)`` and, with
    ``hessian``, its Hessian ``(N, 3, 3)`` at the atoms, from the mesh
    potential ``at`` ``(N, p^3)`` at their stencil points."""
    phi = (at * st["0"]).sum(-1)
    dphi_f = torch.stack([(at * st["0_"]).sum(-1), (at * st["1_"]).sum(-1), (at * st["2_"]).sum(-1)], dim=1)
    dphi = matvec(jac, dphi_f)                                            # grad_beta = sum_a J[beta, a] d_a
    if not hessian:
        return phi, dphi, None
    h00 = (at * st["00"]).sum(-1)
    h11 = (at * st["11"]).sum(-1)
    h22 = (at * st["22"]).sum(-1)
    h01 = (at * st["01"]).sum(-1)
    h02 = (at * st["02"]).sum(-1)
    h12 = (at * st["12"]).sum(-1)
    h = torch.stack([torch.stack([h00, h01, h02], -1), torch.stack([h01, h11, h12], -1),
                     torch.stack([h02, h12, h22], -1)], -2)
    return phi, dphi, matmul3(matmul3(jac, h), jac.transpose(1, 2))   # J h J^T


def _influence(cell: Tensor, alpha: float, mesh: List[int], order: int, kind: str, bare: bool = False) -> Tensor:
    """The mesh influence function in FFT order: for ``kind="coulomb"``
    ``K^3 / (pi V) exp(-pi^2 m^2 / alpha^2) / m^2 B(m)`` (zero at ``m = 0``), for
    ``kind="r6"`` ``K^3 / V pi^(3/2) alpha^3 / 3 f(b) B(m)`` with
    ``f(b) = (1 - 2 b^2) e^{-b^2} + 2 b^3 sqrt(pi) erfc(b)``, ``b = k / (2 alpha)``
    (``f(0) = 1``) and ``B(m)`` the least-squares spline moduli (one with
    ``bare``, the exact reciprocal kernel)."""
    dtype, device = cell.dtype, cell.device
    k1, k2, k3 = mesh[0], mesh[1], mesh[2]
    volume = cell_volume(cell)
    _, k2sq = mesh_wavevectors(cell, mesh)
    moduli = 1.0 if bare else _moduli(mesh, order, dtype, device)
    scale = (k1 * k2 * k3) / volume
    if kind == "coulomb":
        m2 = k2sq / (4.0 * math.pi * math.pi)
        safe = torch.where(m2 > 0, m2, torch.ones_like(m2))
        g = scale / math.pi * torch.exp(-(math.pi ** 2) * safe / (alpha * alpha)) / safe * moduli
        return torch.where(m2 > 0, g, torch.zeros_like(g))
    b = torch.sqrt(k2sq + 1e-30) / (2.0 * alpha)      # the offset keeps the k = 0 cell derivative finite
    f = (1.0 - 2.0 * b * b) * torch.exp(-b * b) + 2.0 * b ** 3 * SQRT_PI * torch.erfc(b)
    return scale * (math.pi ** 1.5) * alpha ** 3 / 3.0 * f * moduli


_MODULI_CACHE: dict = {}


def _moduli(mesh: List[int], order: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """The least-squares spline moduli of a mesh, cached (they depend on the mesh and the order only)."""
    key = (tuple(mesh), order, dtype, str(device))
    if key not in _MODULI_CACHE:
        if len(_MODULI_CACHE) > 64:
            _MODULI_CACHE.clear()
        _MODULI_CACHE[key] = spline_moduli(list(mesh), order, dtype, device, least_squares=True)
    return _MODULI_CACHE[key]


def _mesh_sizes(mesh: List[int], dtype: torch.dtype, device: torch.device) -> Tensor:
    """The mesh dimensions as a tensor, cached: a new one is a copy from the host
    at every evaluation, which a CUDA graph cannot record."""
    key = ("sizes", tuple(mesh), dtype, str(device))
    if key not in _MODULI_CACHE:
        if len(_MODULI_CACHE) > 64:
            _MODULI_CACHE.clear()
        _MODULI_CACHE[key] = torch.tensor(list(mesh), dtype=dtype, device=device)
    return _MODULI_CACHE[key]


class MeshPlan:
    """The fixed particle-mesh data of a group of periodic structures on one mesh.

    The spline weights, their derivative stencils and the Jacobian of the
    scaled fractional coordinates depend on the positions and the cells
    only; the influence functions on the cells, the mesh, the spline order
    and the Ewald coefficient. A plan computes each of them once, so the
    permanent multipole terms, every iteration of the induced-dipole solve
    and the adjoint solve of the backward pass share them. The structures
    of a group sit side by side, each on its own mesh, and one batched real
    FFT serves them all. Everything stays differentiable with respect to the
    positions and the cells.

    Parameters
    ----------
    pos : Tensor
        Positions ``(N, 3)`` of the atoms of the group.
    cells : Tensor
        Lattice vectors as rows ``(G, 3, 3)``, one per structure.
    batch : Tensor
        Structure of every atom ``(N,)``, from zero to ``G - 1``.
    mesh : list of int
        Mesh dimensions ``(K_1, K_2, K_3)``, shared by the structures.
    order : int
        B-spline order.
    kernels : bool, optional
        Run the spreading, interpolation and their adjoint as fused kernels
        (:mod:`.amoeba_kernels`) on spline tables made once, instead of the
        ``(N, p^3)`` stencils; the result is the same to round-off, with
        first derivatives.
    """

    @region("mesh.plan")
    def __init__(self, pos: Tensor, cells: Tensor, batch: Tensor, mesh: List[int], order: int,
                 kernels: bool = False, influence_cache: Optional[dict] = None):
        dtype, device = pos.dtype, pos.device
        # with ``kernels`` the spreading and interpolation run as fused kernels on the 1D splines
        # (xnn.ffnn.models.amoeba_kernels) instead of forming the (N, p^3) stencils
        self.kernels = bool(kernels)
        # the influence functions of a structure whose cell does not move (a step plan's) are kept
        # across evaluations in this dict, unless the cell carries a gradient
        self._cache = influence_cache
        self.mesh = [int(mesh[0]), int(mesh[1]), int(mesh[2])]
        self.order = int(order)
        self.cells = cells
        self.batch = batch
        self.groups = int(cells.shape[0])
        k1, k2, k3 = self.mesh
        sizes = _mesh_sizes(self.mesh, dtype, device)
        # the cell-only quantities (inverse, volume, per-atom inverse) of a cell that does not move
        # and carries no gradient are kept with the influence functions across evaluations
        geometry = ("geometry", tuple(self.mesh), dtype)
        fixed = influence_cache is not None and not cells.requires_grad
        if fixed and geometry in influence_cache:
            inv, inv_n, self.volume = influence_cache[geometry]
        else:
            inv = torch.linalg.inv_ex(cells)[0]                                   # (G, 3, 3)
            # gathers by index_select: their backward adds atomically, without sorting the indices
            inv_n = inv.index_select(0, batch)
            self.volume = cell_volume(cells)
            if fixed:
                influence_cache[geometry] = (inv.detach(), inv_n.detach(), self.volume.detach())
        u = vecmat(pos, inv_n) * sizes
        self.jac = inv_n * sizes                                            # (N, 3, 3): J[n, beta, a]
        self.inv = inv
        self._stencils: dict = {}
        self._influence: dict = {}
        self._self: dict = {}
        if self.kernels:
            # the scaled coordinates keep their graph; the spline tables are constants of the
            # evaluation, since the adjoint kernel returns the gradient with respect to ``u``
            from . import amoeba_kernels as km
            self.u = u
            self.tables = km.spline_tables(u.detach(), self.mesh, self.order)
            self.jac_g = (inv * sizes).contiguous()                          # (G, 3, 3): J of every structure
            self._zero = self.tables[0].new_zeros(self.tables[0].shape[0])    # the absent charges of a pass
            self._paired: Optional[tuple] = None
            return
        base = torch.floor(u)
        self._splines = bspline_with_derivatives(u - base, self.order)      # (N, 3, p) each
        offsets = torch.arange(self.order, dtype=dtype, device=device)
        index = (base[:, :, None] - offsets).remainder(sizes[:, None]).to(torch.long)
        n = pos.shape[0]
        self.flat = (batch[:, None, None, None] * (k1 * k2 * k3) + index[:, 0, :, None, None] * (k2 * k3)
                     + index[:, 1, None, :, None] * k3 + index[:, 2, None, None, :]).reshape(n, -1)
        self.start = base.remainder(sizes).to(torch.long)                   # (N, 3): the stencil start

    def stencils(self, second: bool = False) -> dict:
        """The 3D spline weights ``"0"``, their first derivatives ``"a_"`` and, with
        ``second``, the second derivatives ``"ab"``, each ``(N, p^3)``."""
        w, dw, d2w = self._splines
        st = self._stencils
        if "0" not in st:
            st.update(first_stencils(w, dw))
        if second and "00" not in st:
            st.update(second_stencils(w, dw, d2w))
        return st

    def influence(self, alpha: float, kind: str) -> Tensor:
        """The influence function of every structure on the half spectrum of the
        real FFT, ``(G, K_1, K_2, K_3 // 2 + 1)`` (see :func:`_influence`)."""
        key = (float(alpha), kind)
        shared = (float(alpha), kind, tuple(self.mesh), self.order, self.cells.dtype)
        if (key not in self._influence and self._cache is not None and shared in self._cache
                and not self.cells.requires_grad):
            self._influence[key] = self._cache[shared]
        if key not in self._influence:
            dtype, device = self.cells.dtype, self.cells.device
            k1, k2, k3 = self.mesh
            h3 = k3 // 2 + 1
            axes = [torch.fft.fftfreq(k, d=1.0 / k, dtype=dtype, device=device) for k in (k1, k2)]
            axes.append(torch.fft.fftfreq(k3, d=1.0 / k3, dtype=dtype, device=device)[:h3])
            moduli = _moduli(self.mesh, self.order, dtype, device)[:, :, :h3]
            # the inverse transforms run unscaled (norm="forward"), so the 1 / (K1 K2 K3) of the
            # inverse FFT is folded in here: one multiply fewer per transform
            scale = (1.0 / self.volume)[:, None, None, None]

            def kernel(frequencies):
                grid = torch.meshgrid(frequencies[0], frequencies[1], frequencies[2], indexing="ij")
                miller = torch.stack(grid, dim=-1)                            # (K1, K2, H3, 3)
                kvec = 2.0 * math.pi * torch.einsum("xyzc,gdc->gxyzd", miller, self.inv)
                k2sq = (kvec * kvec).sum(-1)                                   # (G, K1, K2, H3)
                if kind == "coulomb":
                    m2 = k2sq / (4.0 * math.pi * math.pi)
                    safe = torch.where(m2 > 0, m2, torch.ones_like(m2))
                    g = scale / math.pi * torch.exp(-(math.pi ** 2) * safe / (alpha * alpha)) / safe * moduli
                    return torch.where(m2 > 0, g, torch.zeros_like(g))
                b = torch.sqrt(k2sq + 1e-30) / (2.0 * alpha)
                f = (1.0 - 2.0 * b * b) * torch.exp(-b * b) + 2.0 * b ** 3 * SQRT_PI * torch.erfc(b)
                return scale * (math.pi ** 1.5) * alpha ** 3 / 3.0 * f * moduli

            g = kernel(axes)
            # a Nyquist index of an even axis stands for both -K/2 and +K/2, whose wave vectors
            # differ in a non-orthogonal cell: the two aliases are averaged, the symmetric kernel
            # the full complex transform applies through its real part
            flipped = [torch.where(a == -(k // 2), torch.full_like(a, k // 2), a) if k % 2 == 0 else a
                       for a, k in zip(axes, (k1, k2, k3))]
            if any(k % 2 == 0 for k in (k1, k2, k3)):
                g = 0.5 * (g + kernel(flipped))
            self._influence[key] = g
            if self._cache is not None and not self.cells.requires_grad:
                self._cache[shared] = g.detach()
        return self._influence[key]

    @region("mesh.convolve")
    def convolve(self, coeff: Tensor, influence: Tensor) -> Tensor:
        """Spread the stencil coefficients ``(N, p^3)`` onto the meshes, apply the
        influence function and return the mesh potential at the stencil points."""
        k1, k2, k3 = self.mesh
        grid = torch.zeros(self.groups * k1 * k2 * k3, dtype=coeff.dtype, device=coeff.device)
        grid = grid.index_add(0, self.flat.reshape(-1), coeff.reshape(-1))
        spectrum = torch.fft.rfftn(grid.view(self.groups, k1, k2, k3), dim=(1, 2, 3))
        potential = torch.fft.irfftn(spectrum * influence, s=(k1, k2, k3), dim=(1, 2, 3), norm="forward").reshape(-1)
        return potential.index_select(0, self.flat.reshape(-1)).view(self.flat.shape)

    @region("mesh.potential")
    def potential(self, q: Optional[Tensor], d: Optional[Tensor], Q: Optional[Tensor], alpha: float,
                  hessian: bool = True) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """The mesh potential, its gradient and (with ``hessian``) its Hessian at the
        atoms of the group, for multipoles ``q``, ``d``, ``Q`` (any of which may be
        ``None``); see :func:`multipole_pme`."""
        if self.kernels:
            return self._potential_kernels(q, d, Q, alpha, hessian)
        st = self.stencils(second=hessian or Q is not None)
        coeff = stencil_coefficients(q, d, Q, self.jac, st)
        at = self.convolve(coeff, self.influence(alpha, "coulomb"))
        return stencil_potential(at, self.jac, st, hessian)

    def _potential_kernels(self, q, d, Q, alpha: float, hessian: bool):
        from . import amoeba_kernels as km
        w, start = self.tables
        n = w.shape[0]
        zeros = w.new_zeros
        order = 2 if Q is not None else (1 if d is not None else 0)
        if not self.cells.requires_grad:
            # the Jacobians are constants: Cartesian multipoles in, Cartesian derivatives out
            phi, dphi, hess = km.MeshConvolutionCartesian.apply(
                self.u, self._zero if q is None else q, d, Q, self.jac_g, self.batch, self.influence(alpha, "coulomb"),
                w, start, self.mesh, self.groups, order, 2 if hessian else 1)
            return phi, dphi, hess if hessian else None
        c0 = zeros(n) if q is None else q
        c1 = zeros(n, 3) if d is None else vecmat(d, self.jac)
        if Q is None:
            c2 = zeros(n, 6)
        else:
            Qf = matmul3(matmul3(self.jac.transpose(1, 2), Q), self.jac)    # J^T Q J
            c2 = torch.stack([Qf[:, 0, 0], Qf[:, 1, 1], Qf[:, 2, 2], 2.0 * Qf[:, 0, 1], 2.0 * Qf[:, 0, 2],
                              2.0 * Qf[:, 1, 2]], 1)
        at = km.MeshConvolution.apply(self.u, c0, c1, c2, self.influence(alpha, "coulomb"), w, start, self.batch,
                                      self.mesh, self.groups, order, 2 if hessian else 1)
        phi = at[:, 0]
        dphi = matvec(self.jac, at[:, 1:4])
        if not hessian:
            return phi, dphi, None
        h = at[:, 4:10]
        hf = torch.stack([torch.stack([h[:, 0], h[:, 3], h[:, 4]], -1), torch.stack([h[:, 3], h[:, 1], h[:, 5]], -1),
                          torch.stack([h[:, 4], h[:, 5], h[:, 2]], -1)], -2)
        return phi, dphi, matmul3(matmul3(self.jac, hf), self.jac.transpose(1, 2))

    def field_into(self, d: Tensor, alpha: float, acc: Tensor, coef: float, guard: Optional[Tensor] = None) -> Tensor:
        """``acc += coef d - grad(mesh potential of the dipoles d)`` in place, the mesh part of the
        induction solve's field on the kernels (no graph: the solve runs without gradients; the
        Jacobians constants). One spread, two FFTs and one gather, no intermediate per atom. With
        the scalars ``guard`` of a conjugate-gradient iteration the spread and the gather do nothing
        once the solve has converged."""
        from . import amoeba_kernels as km
        w, start = self.tables
        k1, k2, k3 = self.mesh
        grid = km.spread_cartesian(self._zero, d, None, self.jac_g, w, start, self.batch, self.mesh, self.groups, 1,
                                   guard)
        pot = torch.fft.irfftn(torch.fft.rfftn(grid, dim=(1, 2, 3)) * self.influence(alpha, "coulomb"),
                               s=(k1, k2, k3), dim=(1, 2, 3), norm="forward")
        return km.gather_field(pot, self.jac_g, w, start, self.batch, self.mesh, acc, d, coef, guard)

    def paired_field_into(self, d: Tensor, alpha: float, acc: Tensor, coef: float,
                          guard: Optional[Tensor] = None) -> Tensor:
        """:meth:`field_into` for two sets of dipoles stacked ``(2 N, 3)`` (the paired solve of the
        induction): the second set spreads into a second batch of grids, both are transformed
        together, and one gather serves both. The doubled tables are made once per plan."""
        from . import amoeba_kernels as km
        if self._paired is None or self._paired[0] != float(alpha):
            w, start = self.tables
            self._paired = (float(alpha), torch.cat([w, w]), torch.cat([start, start]),
                            torch.cat([self.batch, self.batch + self.groups]),
                            torch.cat([self.jac_g, self.jac_g]).contiguous(),
                            self.influence(alpha, "coulomb").repeat(2, 1, 1, 1),
                            self._zero.new_zeros(2 * self._zero.shape[0]))
        _, w, start, batch, jac, influence, zero = self._paired
        k1, k2, k3 = self.mesh
        grid = km.spread_cartesian(zero, d, None, jac, w, start, batch, self.mesh, 2 * self.groups, 1, guard)
        pot = torch.fft.irfftn(torch.fft.rfftn(grid, dim=(1, 2, 3)) * influence, s=(k1, k2, k3), dim=(1, 2, 3),
                               norm="forward")
        return km.gather_field(pot, jac, w, start, batch, self.mesh, acc, d, coef, guard)

    def dipole_adjoint(self, mu: Tensor, lam: Tensor, alpha: float, scale: Optional[float] = None) -> Tensor:
        """The gradient with respect to the scaled coordinates of ``sum(lam . E(mu))``, the mesh field of
        dipoles ``mu`` probed by ``lam`` (the mesh part of the induction's implicit backward), as the
        backward of the mesh convolution forms it but without the gathers that term does not need: the
        potentials of ``mu`` and of the probes, then one adjoint kernel. With ``lam`` equal to ``scale``
        times ``mu`` one spread and FFT pair serves both."""
        from . import amoeba_kernels as km
        w, start = self.tables
        k1, k2, k3 = self.mesh
        influence = self.influence(alpha, "coulomb")

        def potential(d):
            grid = km.spread_cartesian(self._zero, d, None, self.jac_g, w, start, self.batch, self.mesh,
                                       self.groups, 1)
            return torch.fft.irfftn(torch.fft.rfftn(grid, dim=(1, 2, 3)) * influence, s=(k1, k2, k3),
                                    dim=(1, 2, 3), norm="forward")

        pot = potential(mu)
        # the field is minus the potential's gradient: the probes are -lam
        probe = -lam
        psi = -scale * pot if scale is not None else potential(probe)
        return km.adjoint_cartesian(pot, (self._zero, probe, None), 1, psi, (self._zero, mu, None), 1, self.jac_g,
                                    w, start, self.batch, self.mesh)

    @region("mesh.dispersion")
    def dispersion(self, c: Tensor, alpha: float) -> Tensor:
        """Per-atom reciprocal energies of the ``-c_i c_j / R^6`` sum, self term
        included (see :func:`dispersion_pme`)."""
        if self.kernels:
            from . import amoeba_kernels as km
            w, start = self.tables
            n = w.shape[0]
            influence = self.influence(alpha, "r6")
            phi6 = km.MeshConvolution.apply(self.u, c, w.new_zeros(n, 3), w.new_zeros(n, 6), influence, w, start,
                                            self.batch, self.mesh, self.groups, 0, 0)[:, 0]
        else:
            w0 = self.stencils()["0"]
            at = self.convolve(c[:, None] * w0, self.influence(alpha, "r6"))
            phi6 = (at * w0).sum(-1)
        return -(0.5 * c * phi6 - alpha ** 6 / 12.0 * c * c)

    def self_moments(self, alpha: float, kind: str, max_rank: int = 4):
        """:func:`mesh_self_moments` of every structure, stacked: ``(G,)``,
        ``(G, 3, 3)`` and ``(G, 3, 3, 3, 3)`` (``None`` beyond ``max_rank``)."""
        key = (float(alpha), kind, int(max_rank))
        if key not in self._self:
            per = [mesh_self_moments(self.cells[g], alpha, self.mesh, self.order, kind, max_rank)
                   for g in range(self.groups)]
            self._self[key] = tuple(None if per[0][r] is None else torch.stack([m[r] for m in per])
                                    for r in range(3))
        return self._self[key]


def multipole_pme(q: Tensor, d: Tensor, Q: Tensor, pos: Tensor, cell: Tensor, alpha: float,
                  mesh: List[int], order: int, hessian: bool = True
                  ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Reciprocal-space potential of Gaussian-smeared multipoles at the atoms,
    with its gradient and (optionally) its Hessian.

    The sources ``q_k delta - d_k . grad delta + Q_k : grad grad delta`` are
    spread as ``q_k W + d_k . grad_k W + Q_k : grad_k grad_k W`` (``W`` the
    spline weight, differentiated with respect to the atom position); the
    potential is interpolated back with the same weights. A one-structure
    :class:`MeshPlan`; use a plan directly to reuse the setup.

    Parameters
    ----------
    q : Tensor
        Charges ``(N,)``.
    d : Tensor
        Dipoles ``(N, 3)`` in e Angstrom.
    Q : Tensor
        Quadrupoles ``(N, 3, 3)`` (traceless, divided by three), or ``None``.
    pos : Tensor
        Positions ``(N, 3)``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    alpha : float
        Ewald coefficient.
    mesh : list of int
        Mesh dimensions ``(K_1, K_2, K_3)``.
    order : int
        B-spline order (at least 4; 5 keeps the second derivatives continuous).
    hessian : bool, optional
        Whether to return the Hessian (needed for quadrupole energies).

    Returns
    -------
    tuple of Tensor
        ``phi`` ``(N,)``, ``grad phi`` ``(N, 3)`` and ``grad grad phi``
        ``(N, 3, 3)`` (``None`` without ``hessian``), in ``e / Angstrom``
        and its derivatives (multiply by the Coulomb constant).
    """
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    plan = MeshPlan(pos, cell[None], batch, mesh, order)
    return plan.potential(q, d, Q, alpha, hessian=hessian)


def multipole_reciprocal_energy(q: Tensor, d: Tensor, Q: Tensor, pos: Tensor, cell: Tensor,
                                alpha: float, mesh: List[int], order: int) -> Tensor:
    """Per-atom reciprocal energies ``1/2 [q phi + d.grad phi + Q:grad grad phi]``
    of one periodic structure, in ``e^2 / Angstrom`` (without the self terms).

    Parameters
    ----------
    q, d, Q, pos, cell, alpha, mesh, order
        As in :func:`multipole_pme`.

    Returns
    -------
    Tensor
        ``(N,)``.
    """
    phi, dphi, d2phi = multipole_pme(q, d, Q, pos, cell, alpha, mesh, order, hessian=True)
    return 0.5 * (q * phi + (d * dphi).sum(-1) + (Q * d2phi).sum((-1, -2)))


def multipole_self_energy(q: Tensor, d: Tensor, Q: Tensor, alpha: Tensor) -> Tensor:
    """Per-atom self energies ``-alpha / sqrt(pi) [q^2 + 2 alpha^2 / 3 |d|^2 +
    8 alpha^4 / 5 Q:Q]`` in ``e^2 / Angstrom``: half the interaction of the
    multipole with its own smeared kernel ``erf(alpha R) / R`` at zero
    separation, whose ladder there is ``2 alpha / sqrt(pi)``,
    ``-4 alpha^3 / (3 sqrt(pi))``, ``8 alpha^5 / (5 sqrt(pi))``.

    Parameters
    ----------
    q, d, Q : Tensor
        Multipoles ``(N,)``, ``(N, 3)``, ``(N, 3, 3)``.
    alpha : Tensor
        Ewald coefficient per atom ``(N,)``.

    Returns
    -------
    Tensor
        ``(N,)``.
    """
    a2 = alpha * alpha
    return -alpha / SQRT_PI * (q * q + (2.0 / 3.0) * a2 * (d * d).sum(-1)
                               + 1.6 * a2 * a2 * (Q * Q).sum((-1, -2)))


def self_field_coefficient(alpha: Tensor) -> Tensor:
    """``4 alpha^3 / (3 sqrt(pi))``: the mesh field of a Gaussian dipole at its
    own site is ``-`` this times the dipole, so the correction adds it back."""
    return 4.0 * alpha ** 3 / (3.0 * SQRT_PI)


def _alias_moments(size: int, order: int, dtype: torch.dtype, device: torch.device,
                   n_alias: int = 24) -> Tensor:
    """Per-axis alias sums ``sum_j sinc(m / K + j)^(2 p) (m + j K)^n`` for the
    powers ``n`` from zero to four, ``(5, K)`` in FFT order: the position
    average of the squared transform of a spline weight and of its
    frequency moments."""
    m = torch.fft.fftfreq(size, d=1.0 / size, dtype=dtype, device=device)
    j = torch.arange(-n_alias, n_alias + 1, dtype=dtype, device=device)
    weight = torch.sinc(m[:, None] / size + j[None, :]) ** (2 * order)
    f = m[:, None] + j[None, :] * size
    return torch.stack([(weight * f ** n).sum(1) for n in range(5)])


def mesh_self_moments(cell: Tensor, alpha: float, mesh: List[int], order: int, kind: str,
                      max_rank: int = 4) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """Moments of the spurious, position-averaged mesh self-interaction.

    A multipole spread on the mesh interacts with itself through the mesh
    sum; the spline interpolation makes that interaction differ from the
    exact reciprocal-space self term by an amount that, averaged over the
    position of the atom within a mesh cell, is
    ``S0 q^2 - 2 q Q:S2 + d.S2.d + Q:S4:Q`` with the moments returned here
    (the alias sums of the squared spline transform minus the exact term).
    They vanish as the mesh is refined. Subtracting the spurious part
    removes the mean discretisation error of the self-interaction from the
    energy, the field and the dispersion sum (Ballenegger, Cerda and Holm,
    Comput. Phys. Commun. 182, 1919, 2011).

    Parameters
    ----------
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    alpha : float
        Ewald coefficient.
    mesh : list of int
        Mesh dimensions.
    order : int
        B-spline order.
    kind : str
        ``"coulomb"`` or ``"r6"`` (dispersion, scalar moment only).
    max_rank : int, optional
        Highest moment wanted: 0 (charges), 2 (dipoles) or 4 (quadrupoles).

    Returns
    -------
    tuple of Tensor
        ``S0`` (scalar), ``S2`` ``(3, 3)`` and ``S4`` ``(3, 3, 3, 3)`` in
        Cartesian components (``None`` beyond ``max_rank``), in the units
        of the energy per squared multipole.
    """
    dtype, device = cell.dtype, cell.device
    k1, k2, k3 = mesh[0], mesh[1], mesh[2]
    g = _influence(cell, alpha, mesh, order, kind) / (k1 * k2 * k3)
    # the exact reciprocal self term carries the bare kernel, without the spline moduli of the mesh sum
    g0 = _influence(cell, alpha, mesh, order, kind, bare=True) / (k1 * k2 * k3)
    mu = [_alias_moments(k, order, dtype, device) for k in (k1, k2, k3)]
    freq = [torch.fft.fftfreq(k, d=1.0 / k, dtype=dtype, device=device) for k in (k1, k2, k3)]
    cache: dict = {}

    def weighted(axes) -> Tensor:
        counts = (axes.count(0), axes.count(1), axes.count(2))
        if counts not in cache:
            alias = (mu[0][counts[0]][:, None, None] * mu[1][counts[1]][None, :, None]
                     * mu[2][counts[2]][None, None, :])
            exact = (freq[0][:, None, None] ** counts[0] * freq[1][None, :, None] ** counts[1]
                     * freq[2][None, None, :] ** counts[2])
            cache[counts] = 0.5 * (g * alias - g0 * exact).sum()
        return cache[counts]

    s0 = weighted(())
    if kind == "r6" or max_rank < 2:
        return s0, None, None
    inv = torch.linalg.inv_ex(cell)[0]                     # k_x = 2 pi sum_a inv[x, a] f_a
    two_pi = 2.0 * math.pi
    s2 = torch.stack([weighted((a, b)) for a in range(3) for b in range(3)]).reshape(3, 3) * two_pi ** 2
    s2c = torch.einsum("ab,xa,yb->xy", s2, inv, inv)
    if max_rank < 4:
        return s0, s2c, None
    s4 = torch.stack([weighted((a, b, c, d)) for a in range(3) for b in range(3) for c in range(3)
                      for d in range(3)]).reshape(3, 3, 3, 3) * two_pi ** 4
    s4c = torch.einsum("abcd,xa,yb,zc,wd->xyzw", s4, inv, inv, inv, inv)
    return s0, s2c, s4c


def mesh_self_energy(q: Tensor, d: Tensor, Q: Optional[Tensor], moments) -> Tensor:
    """Per-atom spurious mesh self-interaction of the multipoles,
    ``S0 q^2 - 2 q Q:S2 + d.S2.d + Q:S4:Q``, to be subtracted from the mesh
    energy in addition to the analytic self energy.

    Parameters
    ----------
    q, d, Q : Tensor
        Multipoles ``(N,)``, ``(N, 3)``, ``(N, 3, 3)`` (``Q`` may be ``None``).
    moments : tuple of Tensor
        From :func:`mesh_self_moments` with ``kind="coulomb"``.

    Returns
    -------
    Tensor
        ``(N,)`` in ``e^2 / Angstrom``.
    """
    s0, s2, s4 = moments
    e = q * q * s0 + torch.einsum("na,ab,nb->n", d, s2, d)
    if Q is not None:
        e = e - 2.0 * q * torch.einsum("nab,ab->n", Q, s2) + torch.einsum("nab,abcd,ncd->n", Q, s4, Q)
    return e


def dispersion_pme(c: Tensor, pos: Tensor, cell: Tensor, alpha: float, mesh: List[int],
                   order: int) -> Tensor:
    """Per-atom reciprocal energies of the ``-c_i c_j / R^6`` Ewald sum of one
    periodic structure, self term included: ``-[1/2 c_i phi6(r_i) - alpha^6 / 12 c_i^2]``
    with ``phi6`` the mesh potential of the ``r^-6`` kernel (``k = 0`` included).

    Parameters
    ----------
    c : Tensor
        Dispersion coefficients ``(N,)`` (the pair energy is ``-c_i c_j / R^6``).
    pos : Tensor
        Positions ``(N, 3)``.
    cell : Tensor
        Lattice vectors as rows ``(3, 3)``.
    alpha : float
        The dispersion Ewald coefficient.
    mesh : list of int
        Mesh dimensions.
    order : int
        B-spline order.

    Returns
    -------
    Tensor
        ``(N,)`` in the units of ``c_i c_j / R^6``.
    """
    batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
    return MeshPlan(pos, cell[None], batch, mesh, order).dispersion(c, alpha)


@fusable
def dispersion_real_kernel(R: Tensor, alpha: Tensor) -> Tensor:
    """``g(alpha R) = e^{-x^2} (1 + x^2 + x^4 / 2)``, the real-space share of
    ``1 / R^6`` (one for a zero ``alpha``).

    Parameters
    ----------
    R : Tensor
        Distances ``(M,)``.
    alpha : Tensor
        Dispersion Ewald coefficient per pair ``(M,)``.

    Returns
    -------
    Tensor
        ``(M,)``.
    """
    x2 = (alpha * R) ** 2
    return torch.exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2)


__all__ = ["fused", "fused_active", "fusable", "matvec", "vecmat", "matmul3", "ewald_coefficient", "fft_mesh_size",
           "pme_grid_size", "switching", "ewald_ladder",
           "bspline_with_derivatives", "multipole_pme", "multipole_reciprocal_energy",
           "multipole_self_energy", "self_field_coefficient", "dispersion_pme",
           "dispersion_real_kernel", "mesh_self_moments", "mesh_self_energy", "MeshPlan"]
