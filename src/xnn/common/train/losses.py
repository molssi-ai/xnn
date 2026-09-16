"""Weighted energy / force / stress loss."""
from __future__ import annotations

import torch
from torch import Tensor

from ..data import AtomicGraph


def weighted_loss(pred: dict[str, Tensor], data: AtomicGraph,
                  energy_weight: float, force_weight: float,
                  stress_weight: float) -> tuple[Tensor, dict[str, float]]:
    """Compute a weighted sum of energy, force and stress MSE losses.

    Each term is included only when the target is present in ``data`` and its
    weight is positive (and, for forces and stress, when the corresponding key
    is present in ``pred``). The individual mean-squared-error terms are:

    - Energy: per-atom energy MSE, i.e. the squared difference between the
      predicted and target total energy divided by the number of atoms in each
      structure, then averaged. Dividing by the atom count makes the term
      size-extensive and well-scaled across structures of different sizes.
    - Forces: mean squared error between predicted and target per-atom forces.
    - Stress: mean squared error between predicted and target stress tensors.

    Parameters
    ----------
    pred : dict of str to Tensor
        Model outputs. Must contain ``"energy"``; may also contain
        ``"forces"`` and ``"stress"``.
    data : AtomicGraph
        The (possibly batched) target graph. Reads ``energy``, ``forces``,
        ``stress`` and ``n_atoms`` (per-structure atom counts).
    energy_weight : float
        Weight applied to the per-atom energy MSE term. The term is skipped
        when this is not positive or ``data.energy`` is ``None``.
    force_weight : float
        Weight applied to the force MSE term. The term is skipped when this is
        not positive, ``data.forces`` is ``None``, or ``pred`` lacks
        ``"forces"``.
    stress_weight : float
        Weight applied to the stress MSE term. The term is skipped when this is
        not positive, ``data.stress`` is ``None``, or ``pred`` lacks
        ``"stress"``.

    Returns
    -------
    tuple of (Tensor, dict of str to float)
        A pair ``(loss, logs)`` where ``loss`` is the scalar weighted total
        loss (a 0-dim tensor carrying gradients) and ``logs`` maps
        ``"loss"`` to the detached total plus, for each included term, its
        detached MSE under ``"energy_mse"``, ``"force_mse"`` and/or
        ``"stress_mse"``.
    """
    logs: dict[str, float] = {}
    loss = pred["energy"].new_zeros(())

    if data.energy is not None and energy_weight > 0:
        # per-atom energy MSE -> size-extensive, well-scaled across structures
        n = data.n_atoms.to(pred["energy"].dtype)
        e_loss = torch.mean(((pred["energy"] - data.energy) / n) ** 2)
        loss = loss + energy_weight * e_loss
        logs["energy_mse"] = float(e_loss.detach())

    if data.forces is not None and force_weight > 0 and "forces" in pred:
        f_loss = torch.mean((pred["forces"] - data.forces) ** 2)
        loss = loss + force_weight * f_loss
        logs["force_mse"] = float(f_loss.detach())

    if data.stress is not None and stress_weight > 0 and "stress" in pred:
        s_loss = torch.mean((pred["stress"] - data.stress) ** 2)
        loss = loss + stress_weight * s_loss
        logs["stress_mse"] = float(s_loss.detach())

    logs["loss"] = float(loss.detach())
    return loss, logs
