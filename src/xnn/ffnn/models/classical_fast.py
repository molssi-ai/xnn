"""The fast nonbonded pair terms of the classical force fields (OPLS, AMBER, CHARMM, DREIDING): the half
neighbor list, the exclusion weights, the per-type-pair table over the atom types a topology uses, and
the call of the pair kernel of :mod:`.classical_kernels`. Each model supplies its table and charges; the
numbers are those of its eager nonbonded block to round-off.
"""
from typing import Optional

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph

from .geometry import TINY


def fast_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the pair kernel runs on ``device`` in ``dtype`` (CUDA and Triton)."""
    if device.type != "cuda" or dtype not in (torch.float32, torch.float64):
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def used_types(module, top_type: Tensor) -> tuple:
    """The library type indices a bound topology uses ``(T,)`` and each atom's index among them ``(n,)``,
    made once per device."""
    key = str(top_type.device)
    cache = module.__dict__.setdefault("_used_types", {})
    if key not in cache:
        cache[key] = torch.unique(top_type, return_inverse=True)
    return cache[key]


def exclusion_rows(module, excl_index: Tensor, n: int) -> tuple:
    """The excluded partners of every atom of one structure as compressed rows ``(ptr (n + 1,), cols, width)``
    (int32; each pair ``i < k`` of ``excl_index`` in the row of ``i``; ``width`` the longest row), made once
    per device."""
    key = str(excl_index.device)
    cache = module.__dict__.setdefault("_exclusion_rows", {})
    if key not in cache:
        i, k = excl_index[0], excl_index[1]
        order = torch.argsort(i * n + k)
        counts = torch.bincount(i, minlength=n)
        ptr = torch.zeros(n + 1, dtype=torch.int32, device=i.device)
        ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)
        width = int(counts.max()) if counts.numel() else 0
        cache[key] = (ptr, k[order].to(torch.int32).contiguous(), width)
    return cache[key]


def fast_nonbonded(data: AtomicGraph, n: int, B: int, compact: Tensor, q: Tensor, table: Tensor, ke: float,
                   excl_index: Tensor, cutoff: float, width: float, form: int, module=None,
                   twice: bool = False) -> tuple:
    """The per-atom van der Waals and Coulomb energies ``(N,)`` each, in the model's dtype, from the pair
    kernel on the graph's directed neighbor list (each pair once, ``i < k``), without the excluded pairs
    ``excl_index`` ``(2, E)`` of one structure (``i < k``), with the compact types ``compact`` ``(n,)`` of one
    structure, the charges ``q`` ``(N,)`` and the ``(T, T, 3)`` table of the form; ``twice`` the kernels
    differentiable twice (:mod:`.classical_train`)."""
    from . import classical_kernels as kk
    from . import classical_train as kt
    rows = exclusion_rows(module, excl_index, n) if excl_index.shape[1] > 0 else None
    shifts = data.cell_shifts if data.cell is not None else None
    pairs = kt.pair_terms_twice if twice else kk.pair_terms
    return pairs(data.pos, data.cell, data.batch, data.edge_index[0], data.edge_index[1], shifts, rows, n,
                         compact.repeat(B), q, table, data.model_dtype, ke, TINY, cutoff, width, form)


# the vectors of each kind of bonded term, as (from, to) rows of its atom indices
VECTORS = {"bond": ((0, 1),), "morse": ((0, 1),), "angle": ((1, 0), (1, 2)), "torsion": ((0, 1), (1, 2), (2, 3)),
           "inversion": ((0, 1), (0, 2), (0, 3)), "hbond": ((0, 2), (1, 0))}


def image_shifts(data: AtomicGraph, terms: list) -> list:
    """The image shifts ``(V, M, 3)`` of the vectors of every ``(idx, kind)`` in ``terms`` from one search
    (:func:`~xnn.ffnn.models.geometry.pair_image_shifts`), or ``None`` each without cells."""
    from .geometry import pair_image_shifts
    if data.cell is None:
        return [None] * len(terms)
    a = torch.cat([idx[f] for idx, kind in terms for f, _ in VECTORS[kind]])
    b = torch.cat([idx[t] for idx, kind in terms for _, t in VECTORS[kind]])
    found = pair_image_shifts(data, a, b)
    if found is None:
        return [None] * len(terms)
    out, start = [], 0
    for idx, kind in terms:
        V, M = len(VECTORS[kind]), int(idx.shape[1])
        out.append(found[start:start + V * M].reshape(V, M, 3))
        start += V * M
    return out


def fast_bonded(data: AtomicGraph, idx: Tensor, kind: str, params: tuple, shifts: Optional[Tensor],
                options: Optional[dict] = None, extra: tuple = (0.0, 0.0), twice: bool = False) -> Tensor:
    """The per-atom energies ``(N,)`` of one class of bonded terms from the kernels of
    :mod:`.classical_kernels` (``kind``, ``params`` and ``extra`` as there), in the model's dtype; ``twice``
    the kernels differentiable twice (:mod:`.classical_train`)."""
    from . import classical_kernels as kk
    from . import classical_train as kt
    dtype = data.model_dtype
    bound = min(0.9999999999, 1.0 - 4.0 * torch.finfo(dtype).eps)
    terms = kt.bonded_terms_twice if twice else kk.bonded_terms
    return terms(data.pos, data.cell, data.batch, idx, shifts, kind, params, dtype, TINY, bound, options,
                           extra)


def fast_pairs14(data: AtomicGraph, pairs: Tensor, shifts: Optional[Tensor], compact: Tensor, B: int, q: Tensor,
                 table: Tensor, ke: float, twice: bool = False) -> tuple:
    """The per-atom van der Waals and Coulomb energies ``(N,)`` each of the 1,4 pairs ``pairs`` ``(2, P)``
    (``i < k``, of the batch) with their image shifts ``(1, P, 3)`` from the pair kernel, without a cutoff, the
    tables and the Coulomb constant already scaled."""
    from . import classical_kernels as kk
    from . import classical_train as kt
    terms = kt.pair_terms_twice if twice else kk.pair_terms
    return terms(data.pos, data.cell, data.batch, pairs[0], pairs[1], None if shifts is None else shifts[0],
                         None, 1, compact.repeat(B), q, table, data.model_dtype, ke, TINY, 1.0e15, 0.0, 0)


def evaluate_spec(data: AtomicGraph, spec: dict, module) -> dict:
    """The per-atom energies ``{name: (N,)}`` of a model's kernel spec (its ``_fast_spec``): ``"bonded"`` a list
    of ``(name, kind, idx, params, options)`` (:func:`fast_bonded`), ``"pairs14"`` ``(names, pairs, compact,
    q, table, ke)`` or ``None`` (:func:`fast_pairs14`), ``"nonbonded"`` ``(names, compact, q, table, ke,
    excl_index, n, cutoff, width, form)`` (:func:`fast_nonbonded`); one search for every term's image shifts."""
    bonded, p14, nb = spec["bonded"], spec.get("pairs14"), spec.get("nonbonded")
    found = image_shifts(data, [(t[2], t[1]) for t in bonded] + ([(p14[1], "bond")] if p14 is not None else []))
    twice = needs_twice(module, data)
    out = {}
    for (name, kind, idx, params, options), shifts in zip(bonded, found):
        out[name] = fast_bonded(data, idx, kind, params, shifts, options, twice=twice)
    B = data.num_graphs
    if p14 is not None:
        names, pairs, compact, q, table, ke = p14
        out[names[0]], out[names[1]] = fast_pairs14(data, pairs, found[-1], compact, B, q, table, ke, twice)
    if nb is not None:
        names, compact, q, table, ke, excl_index, n, cutoff, width, form = nb
        out[names[0]], out[names[1]] = fast_nonbonded(data, n, B, compact, q, table, ke, excl_index, cutoff, width,
                                                      form, module, twice)
    return out


def lj_table(A: Tensor, B: float, R: Tensor) -> Tensor:
    """The ``(T, T, 3)`` table of the 12-6 form ``A (x^2 - B x)``, ``x = (R^2 / r^2)^3``."""
    return torch.stack([A, torch.full_like(A, float(B)), R * R], -1)


def x6_table(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """The ``(T, T, 3)`` table of the exponential-6 form ``a exp(-c r) - b / r^6``."""
    return torch.stack([a, b, c], -1)


def needs_twice(module, data: Optional[AtomicGraph] = None) -> bool:
    """Whether an evaluation takes the kernels differentiable twice (:mod:`.classical_train`): gradients on
    and no cell gradient (the cell is a constant of those kernels; with a cell gradient, the stress, the
    kernels differentiable once run, which carry it)."""
    if not torch.is_grad_enabled():
        return False
    return data is None or data.cell is None or not data.cell.requires_grad


def _trains(module, *params: Optional[Tensor]) -> bool:
    """Whether second derivatives in the parameters may be asked for: the model training, or a parameter of
    the force field or one of ``params`` requiring a gradient."""
    if module.training:
        return True
    ff = getattr(module, "ff", None)
    if ff is not None and any(p.requires_grad for p in ff.params.values()):
        return True
    return any(p is not None and p.requires_grad for p in params)


def wants_fast(module, use_fast, data: AtomicGraph, *params: Optional[Tensor]) -> bool:
    """Whether an evaluation takes the kernels: ``use_fast`` on, a CUDA device with Triton, no intermediates
    kept, and, where the kernels must be differentiable twice (training, trained parameters), no cell
    gradient (a stress loss takes the reference terms)."""
    if not use_fast or getattr(module, "keep_intermediates", False):
        return False
    if not fast_supported(data.pos.device, data.model_dtype):
        return False
    if torch.is_grad_enabled() and _trains(module, *params) and data.cell is not None and data.cell.requires_grad:
        return False
    return True


class ClassicalFastPath:
    """The ``use_fast`` setting of a classical force field with kernel terms (OPLS, AMBER, CHARMM, DREIDING)."""

    def set_use_fast(self, use_fast):
        """Set when the kernels run (see ``use_fast`` of the model).

        Parameters
        ----------
        use_fast : bool or str
            ``True``, ``False`` or ``"auto"``.

        Returns
        -------
        nn.Module
            The model.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self


__all__ = ["ClassicalFastPath", "needs_twice", "fast_supported", "used_types", "exclusion_rows", "fast_nonbonded",
           "image_shifts", "fast_bonded", "fast_pairs14", "evaluate_spec", "lj_table", "x6_table", "wants_fast"]
