"""Second derivatives of a model's energy with respect to the positions, and harmonic vibrational analysis.

:func:`hessian_vector_product` differentiates the energy gradient once more along a direction (one more
backward pass); :func:`hessian` forms the full matrix of every structure of a batch from ``3 n`` such products.
With ``create_graph=True`` both stay differentiable, so a loss on curvatures (mode curvatures, frequencies) can
be trained. :func:`harmonic_analysis` turns a Hessian into harmonic frequencies, normal modes and the
zero-point energy.

Units are those of the models: energies in eV, positions in Angstrom, masses in u.

Models whose energies come from a self-consistent density tell these functions how to get exact second
derivatives: a ``position_response`` attribute is switched on during the evaluation (see
:func:`exact_curvature`), or an ``energy_gradient(data, create_graph)`` method returns the positions and a
gradient whose derivatives are exact (``create_graph`` tells it the Hessian will be differentiated once more).
``differentiable_hessian = False`` marks Hessians that cannot be differentiated once more, and
``exact_second_derivatives = False`` a model whose plain autograd Hessian is approximate (a warning).
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import warnings
from typing import Dict, Iterator, List, Optional

import torch
from torch import Tensor, nn

from ..data import AtomicGraph
from .ops import ATOMIC_MASSES
from .outputs import ForceStressOutput

_EV = 1.602176634e-19          # J
_DALTON = 1.66053906660e-27    # kg
_C_CM = 2.99792458e10          # cm / s
_PLANCK = 6.62607015e-34       # J s
#: Wavenumber (cm^-1) of a mass-weighted Hessian eigenvalue of 1 eV / (Angstrom^2 u).
WAVENUMBER = math.sqrt(_EV / (_DALTON * 1e-20)) / (2.0 * math.pi * _C_CM)
#: ``h c`` in eV cm: the energy of a vibrational quantum of 1 cm^-1.
HC = _PLANCK * _C_CM / _EV


@contextlib.contextmanager
def exact_curvature(model: nn.Module) -> Iterator[nn.Module]:
    """Switch on, within the block, what a model's exact second derivatives need.

    Every submodule with a ``position_response`` attribute (the NDDO methods of
    :mod:`xnn.sqmnn`, whose SCF density must then follow the positions) has it
    set to ``True``; the previous values return on exit.

    Parameters
    ----------
    model : torch.nn.Module
        The model.

    Yields
    ------
    torch.nn.Module
        The same model.
    """
    mods = [m for m in model.modules() if hasattr(m, "position_response")]
    old = [m.position_response for m in mods]
    try:
        for m in mods:
            m.position_response = True
        yield model
    finally:
        for m, v in zip(mods, old):
            m.position_response = v


def _energy_gradient(model: nn.Module, data: AtomicGraph, create_graph: bool = False):
    """The positions (a leaf requiring gradients) and the energy gradient with its graph."""
    while isinstance(model, ForceStressOutput):
        model = model.model
    if create_graph and getattr(model, "differentiable_hessian", True) is False:
        raise NotImplementedError(f"the Hessians of {type(model).__name__} cannot be differentiated once more "
                                  "(parameter gradients of curvatures); use create_graph=False")
    own = getattr(model, "energy_gradient", None)
    if callable(own):
        # a model that forms a gradient with exact derivatives itself (the tight-binding methods)
        return own(data, create_graph)
    approximate = sorted({type(m).__name__ for m in model.modules()
                          if getattr(m, "exact_second_derivatives", True) is False})
    if approximate:
        warnings.warn(f"the second derivatives of {', '.join(approximate)} lack the response of the electronic "
                      "density: the Hessian is approximate", RuntimeWarning, stacklevel=3)
    pos = data.pos if data.pos.requires_grad else data.pos.detach().requires_grad_(True)
    graph = dataclasses.replace(data, pos=pos)
    with torch.enable_grad(), exact_curvature(model):
        energy = model(graph)["energy"]
        grad = None
        if energy.requires_grad:
            (grad,) = torch.autograd.grad(energy.sum(), pos, create_graph=True, allow_unused=True)
    return pos, grad


def _directional(pos: Tensor, grad: Optional[Tensor], v: Tensor, create_graph: bool, retain_graph: bool) -> Tensor:
    """The derivative of ``grad`` along ``v``: ``H v``."""
    if grad is None or not grad.requires_grad:
        return torch.zeros_like(pos)
    with torch.enable_grad():
        (hv,) = torch.autograd.grad(grad, pos, v.to(grad.dtype), create_graph=create_graph,
                                    retain_graph=retain_graph, allow_unused=True)
    return hv if hv is not None else torch.zeros_like(pos)


def hessian_vector_product(model: nn.Module, data: AtomicGraph, v: Tensor, create_graph: bool = False) -> Tensor:
    """The Hessian of the energy times a direction, ``H v``, for every structure of a batch.

    Parameters
    ----------
    model : torch.nn.Module
        The potential (bare or wrapped in
        :class:`~xnn.common.models.ForceStressOutput`).
    data : AtomicGraph
        The structures.
    v : Tensor
        The direction of every atom ``(N, 3)`` (Angstrom), or ``K``
        directions ``(K, N, 3)``, all from one evaluation of the model.
    create_graph : bool, optional
        Keep the product differentiable (for a loss on curvatures, whose
        backward reaches the model's parameters); default ``False``.

    Returns
    -------
    Tensor
        ``H v`` of the shape of ``v``, in eV / Angstrom^2 times the unit of
        ``v``; the structures of a batch do not couple.

    Examples
    --------
    The curvatures of a structure along modes ``u`` ``(K, N, 3)`` are
    ``u_k . H u_k``::

        hu = hessian_vector_product(model, graph, u, create_graph=True)
        curvatures = (u * hu).sum((1, 2))
    """
    pos, grad = _energy_gradient(model, data, create_graph)
    if v.dim() == 2:
        return _directional(pos, grad, v, create_graph, create_graph)
    k = v.shape[0]
    return torch.stack([_directional(pos, grad, v[i], create_graph, create_graph or i < k - 1) for i in range(k)])


def hessian(model: nn.Module, data: AtomicGraph, create_graph: bool = False) -> List[Tensor]:
    """The Hessian of the energy with respect to the positions of every structure of a batch.

    Parameters
    ----------
    model : torch.nn.Module
        The potential (bare or wrapped in
        :class:`~xnn.common.models.ForceStressOutput`).
    data : AtomicGraph
        The structures.
    create_graph : bool, optional
        Keep the Hessians differentiable; default ``False``.

    Returns
    -------
    list of Tensor
        One ``(3 n, 3 n)`` matrix per structure in eV / Angstrom^2, rows and
        columns ordered atom by atom (``x``, ``y``, ``z`` of the first atom
        first).
    """
    pos, grad = _energy_gradient(model, data, create_graph)
    batch, n_struct = data.batch, data.num_graphs
    counts = torch.bincount(batch, minlength=n_struct)
    start = torch.cumsum(counts, 0) - counts
    local = torch.arange(pos.shape[0], device=pos.device) - start[batch]
    n_max = int(counts.max()) if counts.numel() > 0 else 0
    # one product per atom slot and direction gives that column of every structure
    columns = []
    for j in range(n_max):
        at = local == j
        for c in range(3):
            v = torch.zeros_like(pos.detach())
            v[at, c] = 1.0
            columns.append(_directional(pos, grad, v, create_graph, True))
    out = []
    for b in range(n_struct):
        idx = (batch == b).nonzero().flatten()
        cols = [columns[k].index_select(0, idx).reshape(-1) for k in range(3 * idx.numel())]
        out.append(torch.stack(cols, 1) if cols else pos.new_zeros(0, 0))
    return out


def harmonic_analysis(hessian: Tensor, positions: Tensor, masses: Optional[Tensor] = None,
                      atomic_numbers: Optional[Tensor] = None,
                      linear_tolerance: float = 1e-5) -> Dict[str, object]:
    """Harmonic frequencies, normal modes and zero-point energy of one structure.

    The Hessian is symmetrized and mass-weighted. Translations and rotations
    (about the center of mass) are projected out, and the remaining
    ``3 n - 6`` vibrations (``3 n - 5`` for a linear molecule, detected from
    the moments of inertia) are found by diagonalizing the Hessian in their
    space. The result is differentiable with respect to ``hessian``.

    Parameters
    ----------
    hessian : Tensor
        The Hessian ``(3 n, 3 n)`` in eV / Angstrom^2 (as from :func:`hessian`).
    positions : Tensor
        The positions ``(n, 3)`` in Angstrom.
    masses : Tensor, optional
        The atomic masses ``(n,)`` in u.
    atomic_numbers : Tensor, optional
        The atomic numbers ``(n,)``, for the standard atomic weights when
        ``masses`` is not given.
    linear_tolerance : float, optional
        A molecule is linear when its smallest principal moment of inertia is
        below this fraction of the largest; default 1e-5.

    Returns
    -------
    dict
        ``frequencies``
            Harmonic wavenumbers ``(n_vib,)`` in cm^-1, ascending; imaginary
            ones as negative numbers.
        ``eigenvalues``
            The eigenvalues of the projected mass-weighted Hessian
            ``(n_vib,)`` in eV / (Angstrom^2 u).
        ``modes``
            The normal modes ``(n_vib, n, 3)``, orthonormal in mass-weighted
            coordinates.
        ``displacements``
            The modes in Cartesian coordinates (``modes`` divided by the square
            roots of the masses): ``d . H d`` is the mode's eigenvalue.
        ``zero_point_energy``
            ``h c / 2`` times the sum of the real wavenumbers, in eV.
        ``mass_weighted_hessian``, ``projected_hessian``
            The mass-weighted Hessian before and after the projection
            ``(3 n, 3 n)``.
        ``linear``
            Whether the molecule was taken as linear.
    """
    dtype, device = hessian.dtype, hessian.device
    if masses is None:
        if atomic_numbers is None:
            raise ValueError("harmonic_analysis needs masses or atomic_numbers")
        masses = torch.as_tensor([ATOMIC_MASSES[int(z)] for z in atomic_numbers.tolist()], dtype=dtype,
                                 device=device)
    m = masses.to(dtype=dtype, device=device)
    x = positions.detach().to(dtype=dtype, device=device)
    n = x.shape[0]
    s = m.sqrt().repeat_interleave(3)
    Hmw = 0.5 * (hessian + hessian.transpose(0, 1)) / (s[:, None] * s[None, :])

    r = x - (m[:, None] * x).sum(0) / m.sum()
    eye = torch.eye(3, dtype=dtype, device=device)
    inertia = (m * (r * r).sum(1)).sum() * eye - torch.einsum("i,ij,ik->jk", m, r, r)
    moments = torch.linalg.eigvalsh(inertia)
    linear = n == 2 or (n > 2 and bool(moments[0] <= linear_tolerance * moments[2]))
    n_external = 3 if n == 1 else (5 if linear else 6)
    sm = m.sqrt()[:, None]
    external = [(sm * eye[k]).reshape(-1) for k in range(3)]
    external += [(sm * torch.linalg.cross(eye[k].expand_as(r), r)).reshape(-1) for k in range(3)]
    U, _, _ = torch.linalg.svd(torch.stack(external, 1), full_matrices=True)
    E, B = U[:, :n_external], U[:, n_external:]

    K = B.transpose(0, 1) @ Hmw @ B
    if K.shape[0] > 0:
        lam, Y = torch.linalg.eigh(K)
    else:
        lam, Y = K.new_zeros(0), K
    modes = (B @ Y).transpose(0, 1).reshape(-1, n, 3)
    frequencies = torch.sign(lam) * lam.abs().sqrt() * WAVENUMBER
    projector = torch.eye(3 * n, dtype=dtype, device=device) - E @ E.transpose(0, 1)
    return {
        "frequencies": frequencies,
        "eigenvalues": lam,
        "modes": modes,
        "displacements": modes / m.sqrt()[None, :, None],
        "zero_point_energy": 0.5 * HC * frequencies.clamp(min=0.0).sum(),
        "mass_weighted_hessian": Hmw,
        "projected_hessian": projector @ Hmw @ projector,
        "linear": linear,
    }
