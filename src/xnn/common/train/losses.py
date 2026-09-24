"""Weighted energy / force / stress loss."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..data import AtomicGraph


def _residual_sq(diff: Tensor, delta: float) -> Tensor:
    """Element-wise squared error, optionally with linear tails beyond ``delta``.

    With ``delta <= 0`` this is plain ``diff ** 2``. Otherwise it is the Huber
    function **scaled to agree with the squared error** in the quadratic
    region, ``diff ** 2`` for ``|diff| <= delta`` and
    ``2 * delta * |diff| - delta ** 2`` beyond it. That scaling (twice the
    textbook Huber, which uses ``0.5 * diff ** 2``) is deliberate: it keeps
    ``energy_weight`` / ``force_weight`` and the learning rate meaning the same
    thing whether or not the tails are clipped, so switching a run to Huber
    does not silently halve its loss. Compare with MACE, which uses the
    textbook form, if you port ``delta`` values between the two.

    Parameters
    ----------
    diff : Tensor
        Residuals ``pred - target``, any shape.
    delta : float
        Crossover from quadratic to linear. ``<= 0`` disables the tails.

    Returns
    -------
    Tensor
        Element-wise error, the same shape as ``diff``.
    """
    if delta <= 0:
        return diff ** 2
    a = diff.abs()
    return torch.where(a <= delta, diff ** 2, 2.0 * delta * a - delta ** 2)


def _mean(err: Tensor, w: Optional[Tensor]) -> Tensor:
    """Mean of ``err``, weighted by ``w`` broadcast over the trailing axes.

    Parameters
    ----------
    err : Tensor
        Element-wise errors whose first axis indexes the weighted unit
        (structures for energy and stress, atoms for forces).
    w : Tensor or None
        Non-negative weights of shape ``(len(err),)``. ``None`` gives the plain
        mean, which is what uniform weights reduce to exactly.

    Returns
    -------
    Tensor
        Scalar weighted mean.
    """
    if w is None:
        return err.mean()
    shape = (-1,) + (1,) * (err.dim() - 1)
    per_element = err.numel() / err.shape[0]          # 1 for energy, 3 for forces
    return (w.reshape(shape) * err).sum() / (w.sum() * per_element)


def weighted_loss(pred: dict[str, Tensor], data: AtomicGraph,
                  energy_weight: float, force_weight: float,
                  stress_weight: float, *,
                  huber_delta: float = 0.0,
                  huber_delta_energy: Optional[float] = None,
                  huber_delta_forces: Optional[float] = None,
                  huber_delta_stress: Optional[float] = None,
                  ) -> tuple[Tensor, dict[str, float]]:
    """Compute a weighted sum of energy, force and stress loss terms.

    Each term is included only when the target is present in ``data`` and its
    weight is positive (and, for forces and stress, when the corresponding key
    is present in ``pred``). The individual terms are:

    - Energy: per-atom energy error, i.e. the difference between the predicted
      and target total energy divided by the number of atoms in each structure,
      then averaged over structures. Dividing by the atom count makes the term
      size-extensive and well-scaled across structures of different sizes.
    - Forces: per-atom force error, averaged over every atom and Cartesian
      component in the batch.
    - Stress: error between predicted and target stress tensors.

    **Per-structure weighting.** When ``data.weight`` is set (shape ``(B,)``),
    every term becomes a weighted mean instead of a plain one: structure ``b``
    contributes in proportion to ``weight[b]``, and for the force term that
    weight is spread to each of its atoms. Uniform weights reproduce the
    unweighted loss exactly, so this is a no-op unless the dataset asks for it.
    It exists because the two terms otherwise disagree about what a sample is:
    energy counts each *structure* once while forces count each *atom* once, so
    a set mixing small and large structures, or dense scans with sparse
    sampling, silently allocates the fit. Set ``weight`` in the structure dicts
    (see :func:`~xnn.common.data.dataset.to_graph`) to allocate it on purpose.

    **Huber tails.** With ``huber_delta > 0`` the squared error is replaced by
    a Huber-like function that is quadratic up to ``delta`` and linear beyond,
    which caps the pull of a few large residuals. The three residuals have very
    different natural scales (per-atom energies in eV, forces in eV/A), so a
    single ``huber_delta`` is rarely right for all of them; the per-term
    arguments override it. See :func:`_residual_sq` for the exact form and how
    it compares with MACE's.

    Parameters
    ----------
    pred : dict of str to Tensor
        Model outputs. Must contain ``"energy"``; may also contain
        ``"forces"`` and ``"stress"``.
    data : AtomicGraph
        The (possibly batched) target graph. Reads ``energy``, ``forces``,
        ``stress``, ``n_atoms`` (per-structure atom counts), ``batch`` (the
        structure index of every atom) and the optional ``weight``.
    energy_weight : float
        Weight applied to the energy term. The term is skipped when this is
        not positive or ``data.energy`` is ``None``.
    force_weight : float
        Weight applied to the force term. The term is skipped when this is
        not positive, ``data.forces`` is ``None``, or ``pred`` lacks
        ``"forces"``.
    stress_weight : float
        Weight applied to the stress term. The term is skipped when this is
        not positive, ``data.stress`` is ``None``, or ``pred`` lacks
        ``"stress"``.
    huber_delta : float, optional
        Default crossover from quadratic to linear for every term. ``0.0``
        (the default) means plain squared error throughout.
    huber_delta_energy, huber_delta_forces, huber_delta_stress : float, optional
        Per-term overrides of ``huber_delta``. ``None`` falls back to it.

    Returns
    -------
    tuple of (Tensor, dict of str to float)
        A pair ``(loss, logs)`` where ``loss`` is the scalar weighted total
        loss (a 0-dim tensor carrying gradients) and ``logs`` maps ``"loss"``
        to the detached total plus, for each included term, its detached value
        under ``"energy_mse"``, ``"force_mse"`` and/or ``"stress_mse"``. Those
        keys keep their names when Huber tails are on, where they hold the
        Huber value rather than a mean square.
    """
    logs: dict[str, float] = {}
    loss = pred["energy"].new_zeros(())
    w = data.weight
    d_e = huber_delta if huber_delta_energy is None else huber_delta_energy
    d_f = huber_delta if huber_delta_forces is None else huber_delta_forces
    d_s = huber_delta if huber_delta_stress is None else huber_delta_stress

    if data.energy is not None and energy_weight > 0:
        # per-atom energy error -> size-extensive, well-scaled across structures
        n = data.n_atoms.to(pred["energy"].dtype)
        e_loss = _mean(_residual_sq((pred["energy"] - data.energy) / n, d_e), w)
        loss = loss + energy_weight * e_loss
        logs["energy_mse"] = float(e_loss.detach())

    if data.forces is not None and force_weight > 0 and "forces" in pred:
        # a structure's weight applies to each of its atoms
        w_atom = None if w is None else w[data.batch]
        f_loss = _mean(_residual_sq(pred["forces"] - data.forces, d_f), w_atom)
        loss = loss + force_weight * f_loss
        logs["force_mse"] = float(f_loss.detach())

    if data.stress is not None and stress_weight > 0 and "stress" in pred:
        s_loss = _mean(_residual_sq(pred["stress"] - data.stress, d_s), w)
        loss = loss + stress_weight * s_loss
        logs["stress_mse"] = float(s_loss.detach())

    logs["loss"] = float(loss.detach())
    return loss, logs
