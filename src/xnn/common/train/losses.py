"""Weighted energy / force / stress (and dipole / polarizability / charge) loss."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

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


def _first_output(pred: dict[str, Tensor]) -> Tensor:
    """The energy, or the property of a model without one: the dtype and device of the loss."""
    return pred["energy"] if "energy" in pred else pred["properties"]


def _masked(w: Optional[Tensor], mask: Optional[Tensor], like: Tensor,
            term: Optional[Tensor] = None) -> Optional[Tensor]:
    """Per-structure weights restricted to the structures that carry a label.

    Parameters
    ----------
    w : Tensor or None
        Per-structure weights ``(B,)``; ``None`` means all equal.
    mask : Tensor or None
        Bool ``(B,)``, the structures with the label; ``None`` means all.
    like : Tensor
        Gives the dtype of the result.
    term : Tensor or None, optional
        Per-structure weights ``(B,)`` of this one term (``energy_weight``,
        ``force_weight`` or ``stress_weight``); ``None`` means 1.

    Returns
    -------
    Tensor or None
        ``w`` times ``term`` with the unlabelled structures at zero, or
        ``None`` when no weights and no mask apply (the plain mean).
    """
    if term is not None:
        term = term.to(like.dtype)
        w = term if w is None else w * term
    if mask is None:
        return w
    m = mask.to(like.dtype)
    return m if w is None else w * m


def _mean(err: Tensor, w: Optional[Tensor], sync: bool = True) -> Tensor:
    """Mean of ``err``, weighted by ``w`` broadcast over the trailing axes.

    Parameters
    ----------
    err : Tensor
        Element-wise errors whose first axis indexes the weighted unit
        (structures for energy and stress, atoms for forces).
    w : Tensor or None
        Non-negative weights of shape ``(len(err),)``. ``None`` gives the plain
        mean, which is what uniform weights reduce to exactly.
    sync : bool, optional
        ``False``: a zero weight sum gives 0 instead of 0 / 0 (see
        :func:`weighted_loss`).

    Returns
    -------
    Tensor
        Scalar weighted mean.
    """
    if w is None:
        return err.mean()
    shape = (-1,) + (1,) * (err.dim() - 1)
    per_element = err.numel() / err.shape[0]          # 1 for energy, 3 for forces
    total = w.sum()
    if not sync:
        # a term without labelled structures is 0, not 0 / 0 (the host cannot check)
        total = total.clamp_min(torch.finfo(err.dtype).tiny)
    return (w.reshape(shape) * err).sum() / (total * per_element)


def _count(err: Tensor, w: Optional[Tensor], sync: bool = True):
    """The weight :func:`_mean` divides by, in units of its first axis.

    Parameters
    ----------
    err : Tensor
        Element-wise errors, as passed to :func:`_mean`.
    w : Tensor or None
        Their weights, as passed to :func:`_mean`.

    Returns
    -------
    float
        ``len(err)`` when unweighted, ``w.sum()`` otherwise, so that a mean
        over several batches is the sum of ``mean * count`` over the sum of
        ``count``.
    """
    if w is None:
        return float(err.shape[0])
    return float(w.sum().detach()) if sync else w.sum().detach()


def _terms(pred: dict[str, Tensor], data: AtomicGraph, w: Optional[Tensor],
           energy_weight: float, force_weight: float, stress_weight: float,
           d_e: float, d_f: float, d_s: float,
           dipole_weight: float = 0.0, polarizability_weight: float = 0.0,
           d_t: float = 0.0, charge_weight: float = 0.0,
           counts: Optional[dict] = None, les_dipole_weight: float = 0.0,
           les_charge_weight: float = 0.0, sync: bool = True,
           property_weight: float = 0.0) -> tuple[Tensor, dict[str, float]]:
    """The weighted energy, force, stress, dipole, polarizability, charge and property
    terms over the structures weighted by ``w``.

    Parameters
    ----------
    pred, data
        As in :func:`weighted_loss`.
    w : Tensor or None
        Per-structure weights ``(B,)`` (``None`` = uniform). A zero weight
        drops a structure from every term, so a head's terms are the means
        over its own structures.
    counts : dict or None, optional
        Filled, when given, as described in :func:`weighted_loss`.
    sync : bool, optional
        As in :func:`weighted_loss`.

    Returns
    -------
    tuple of (Tensor, dict of str to float)
        The summed loss and the detached values of the included terms.
    """
    logs: dict[str, float] = {}
    loss = _first_output(pred).new_zeros(())

    def present(w_t: Optional[Tensor]) -> bool:
        return w_t is None or not sync or bool(w_t.any())

    def value(t: Tensor):
        return float(t.detach()) if sync else t.detach()

    if data.energy is not None and energy_weight > 0 and "energy" in pred:
        w_e = _masked(w, getattr(data, "energy_mask", None), pred["energy"],
                      getattr(data, "energy_weight", None))
        if present(w_e):
            # per-atom energy error -> size-extensive, well-scaled across structures
            n = data.n_atoms.to(pred["energy"].dtype)
            err = _residual_sq((pred["energy"] - data.energy) / n, d_e)
            e_loss = _mean(err, w_e, sync)
            loss = loss + energy_weight * e_loss
            logs["energy_mse"] = value(e_loss)
            if counts is not None:
                counts["energy_mse"] = (_count(err, w_e, sync), energy_weight)

    if data.forces is not None and force_weight > 0 and "forces" in pred:
        w_f = _masked(w, getattr(data, "forces_mask", None), pred["forces"],
                      getattr(data, "force_weight", None))
        # a structure's weights (and label mask) apply to each of its atoms,
        # times the atom's own weight
        w_atom = None if w_f is None else w_f[data.batch]
        a = getattr(data, "atom_force_weight", None)
        if a is not None:
            a = a.to(pred["forces"].dtype)
            w_atom = a if w_atom is None else w_atom * a
        if present(w_atom):
            err = _residual_sq(pred["forces"] - data.forces, d_f)
            f_loss = _mean(err, w_atom, sync)
            loss = loss + force_weight * f_loss
            logs["force_mse"] = value(f_loss)
            if counts is not None:
                counts["force_mse"] = (_count(err, w_atom, sync), force_weight)

    if data.stress is not None and stress_weight > 0 and "stress" in pred:
        w_s = _masked(w, getattr(data, "stress_mask", None), pred["stress"],
                      getattr(data, "stress_weight", None))
        if present(w_s):
            err = _residual_sq(pred["stress"] - data.stress, d_s)
            s_loss = _mean(err, w_s, sync)
            loss = loss + stress_weight * s_loss
            logs["stress_mse"] = value(s_loss)
            if counts is not None:
                counts["stress_mse"] = (_count(err, w_s, sync), stress_weight)

    # the tensorial labels of molecules (PaiNN, PhysNet, AIMNet2 dipoles, the
    # LES latent dipole): per-structure means over the 3 (dipole) or 9
    # (polarizability) components
    for key, label, weight in (("dipole", "dipole", dipole_weight),
                               ("polarizability", "polarizability", polarizability_weight),
                               ("les_dipole", "dipole", les_dipole_weight)):
        target = getattr(data, label, None)
        if target is None or weight <= 0 or key not in pred:
            continue
        w_t = _masked(w, getattr(data, label + "_mask", None), pred[key])
        if present(w_t):
            err = _residual_sq(pred[key] - target, d_t)
            t_loss = _mean(err, w_t, sync)
            loss = loss + weight * t_loss
            logs[key + "_mse"] = value(t_loss)
            if counts is not None:
                counts[key + "_mse"] = (_count(err, w_t, sync), weight)

    # per-atom partial charges (the charge networks of the 3G / 4G HDNNPs, the
    # LES latent charges), averaged over the atoms of the labelled structures
    # like the forces
    for key, name, weight in (("charges", "charge_mse", charge_weight),
                              ("les_charges", "les_charge_mse", les_charge_weight)):
        if data.charges is None or weight <= 0 or key not in pred:
            continue
        w_q = _masked(w, getattr(data, "charges_mask", None), pred[key])
        if present(w_q):
            w_atom = None if w_q is None else w_q[data.batch]
            err = _residual_sq(pred[key] - data.charges, d_t)
            q_loss = _mean(err, w_atom, sync)
            loss = loss + weight * q_loss
            logs[name] = value(q_loss)
            if counts is not None:
                counts[name] = (_count(err, w_atom, sync), weight)

    # intensive per-structure targets (MEGNet): the plain mean over the labelled
    # structures and the components of each, with no division by the atom count
    if data.properties is not None and property_weight > 0 and "properties" in pred:
        w_p = _masked(w, getattr(data, "properties_mask", None), pred["properties"])
        if present(w_p):
            target = data.properties.to(pred["properties"].dtype)
            err = _residual_sq(pred["properties"] - target.reshape(pred["properties"].shape), d_t)
            p_loss = _mean(err, w_p, sync)
            loss = loss + property_weight * p_loss
            logs["property_mse"] = value(p_loss)
            if counts is not None:
                counts["property_mse"] = (_count(err, w_p, sync), property_weight)
    return loss, logs


def weighted_loss(pred: dict[str, Tensor], data: AtomicGraph,
                  energy_weight: float, force_weight: float,
                  stress_weight: float, *,
                  dipole_weight: float = 0.0,
                  polarizability_weight: float = 0.0,
                  charge_weight: float = 0.0,
                  les_dipole_weight: float = 0.0,
                  les_charge_weight: float = 0.0,
                  property_weight: float = 0.0,
                  huber_delta: float = 0.0,
                  huber_delta_energy: Optional[float] = None,
                  huber_delta_forces: Optional[float] = None,
                  huber_delta_stress: Optional[float] = None,
                  head_weights: Optional[Mapping[int, Sequence[float]]] = None,
                  head_names: Optional[Sequence[str]] = None,
                  counts: Optional[dict] = None,
                  sync: bool = True,
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
    - Dipole and polarizability (``dipole_weight`` /
      ``polarizability_weight`` > 0): the mean squared error over the three
      (nine) components of the predicted and target ``dipole``
      (``polarizability``) of each structure, the tensorial targets of PaiNN
      (and the dipoles of PhysNet and AIMNet2). These terms take the global
      ``huber_delta`` and are not weighted per head.
    - Charges (``charge_weight`` > 0): the squared error of the predicted
      ``charges`` of every atom against ``data.charges``, averaged over the
      atoms like the force term (the charge networks of the 3G and 4G
      HDNNPs, and any model with a ``"charges"`` output). It takes the global
      ``huber_delta`` and is not weighted per head.
    - LES charges and dipole (``les_charge_weight`` / ``les_dipole_weight``
      > 0): the same two terms for the ``les_charges`` and ``les_dipole``
      outputs of :class:`~xnn.common.models.les.LatentEwald`, against the
      same ``charges`` and ``dipole`` labels.
    - Properties (``property_weight`` > 0): the squared error of the
      predicted ``properties`` of every structure against
      ``data.properties``, averaged over the structures and the components,
      without division by the atom count (intensive targets such as a band
      gap). It takes the global ``huber_delta`` and is not weighted per head.

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

    **Term and atom weights.** ``data.energy_weight`` / ``force_weight`` /
    ``stress_weight`` (shape ``(B,)``) multiply ``weight`` for one term only,
    and ``data.atom_force_weight`` (shape ``(N,)``) multiplies the weight of
    each atom's force error. The force term is then
    ``sum_i u_i |dF_i|^2 / (3 sum_i u_i)`` with
    ``u_i = weight * force_weight * atom_force_weight``, so a weighted atom
    counts like an atom of an unweighted structure. An atom of weight 0 stays
    in the graph as environment; only its force label is dropped. A term
    whose weights are all zero in a batch is skipped.

    **Partially labelled batches.** A batch may mix structures with and without
    energy, force or stress labels (``data.energy_mask`` /
    ``data.forces_mask`` / ``data.stress_mask``, set by
    :func:`~xnn.common.data.dataset.collate`): those terms then average over
    the labelled structures only, and a batch without any is skipped for that
    term.

    **Several heads.** When the batch carries ``data.head`` with more than one
    head, the loss is the sum over heads of that head's weighted terms, each
    averaged over the head's own structures (the multi-head replay objective,
    Tompa *et al.* eq. 3), and ``head_weights`` gives a head its own energy,
    force and stress weights (eq. 4). A single-head batch reduces to the plain
    loss above; with ``head_names`` given, its logs still carry the head's
    prefix, so every batch of a multi-head run logs its terms under the same
    keys.

    **Dataset-level metrics.** Each term is a weighted mean, and a mean over
    batches of unequal size (or with unequal numbers of labels) is not the
    mean of the batch means. Pass a dict as ``counts`` to have it filled with
    ``{key: (count, term_weight)}`` for every term key of ``logs``: the
    weight the term's mean divides by (structures, atoms or their summed
    weights) and the term's loss weight. The mean of a term over many batches
    is then ``sum(logs[key] * count) / sum(count)``, exactly the value one
    batch holding all the structures would give, and the loss over them is
    ``sum(term_weight * mean)`` over the keys.

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
        structure index of every atom), the optional ``weight`` and the
        optional ``head``.
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
    dipole_weight, polarizability_weight : float, optional
        Weights of the dipole and polarizability terms, skipped the same way
        (``data.dipole`` / ``data.polarizability`` and the ``"dipole"`` /
        ``"polarizability"`` predictions). Default ``0.0``.
    charge_weight : float, optional
        Weight of the partial-charge term, skipped the same way
        (``data.charges`` and the ``"charges"`` prediction). Default ``0.0``.
    les_dipole_weight, les_charge_weight : float, optional
        Weights of the LES dipole and charge terms (``data.dipole`` /
        ``data.charges`` against the ``"les_dipole"`` / ``"les_charges"``
        predictions). Default ``0.0``.
    property_weight : float, optional
        Weight of the property term (``data.properties`` against the
        ``"properties"`` prediction). Default ``0.0``.
    huber_delta : float, optional
        Default crossover from quadratic to linear for every term. ``0.0``
        (the default) means plain squared error throughout.
    huber_delta_energy, huber_delta_forces, huber_delta_stress : float, optional
        Per-term overrides of ``huber_delta``. ``None`` falls back to it.
    head_weights : mapping of int to sequence of float, optional
        ``{head_index: (energy_weight, force_weight, stress_weight)}`` for the
        heads whose weights differ from the global ones.
    head_names : sequence of str, optional
        Head names for the per-head log keys (``"<head>/energy_mse"``);
        indices are used when omitted.
    counts : dict or None, optional
        When given, filled with ``{key: (count, term_weight)}`` for every term
        key of ``logs`` (see *Dataset-level metrics* above). Defaults to
        ``None``.
    sync : bool, optional
        ``True`` (the default): terms without a labelled structure are left
        out and ``logs`` and ``counts`` hold Python numbers. ``False``: no
        value is read back to the host, so the loss can run inside a captured
        CUDA graph. Every term whose labels the batch carries is included (one
        whose structures are all masked out is exactly 0) and ``logs`` and
        ``counts`` hold 0-dimensional tensors. Single-head batches only.

    Returns
    -------
    tuple of (Tensor, dict of str to float)
        A pair ``(loss, logs)`` where ``loss`` is the scalar weighted total
        loss (a 0-dim tensor carrying gradients) and ``logs`` maps ``"loss"``
        to the detached total plus, for each included term, its detached value
        under ``"energy_mse"``, ``"force_mse"``, ``"stress_mse"``,
        ``"dipole_mse"``, ``"polarizability_mse"``, ``"charge_mse"``,
        ``"les_dipole_mse"`` and/or ``"les_charge_mse"`` (with a
        ``"<head>/"`` prefix per head in a multi-head batch). Those keys keep
        their names when Huber tails are on, where they hold the Huber value
        rather than a mean square.
    """
    d_e = huber_delta if huber_delta_energy is None else huber_delta_energy
    d_f = huber_delta if huber_delta_forces is None else huber_delta_forces
    d_s = huber_delta if huber_delta_stress is None else huber_delta_stress
    weights = (energy_weight, force_weight, stress_weight)

    if not sync:
        if data.head is not None:
            raise ValueError("weighted_loss(sync=False) takes single-head batches only")
        loss, logs = _terms(pred, data, data.weight, energy_weight, force_weight, stress_weight,
                            d_e, d_f, d_s, dipole_weight, polarizability_weight, huber_delta,
                            charge_weight, counts, les_dipole_weight, les_charge_weight, sync=False,
                            property_weight=property_weight)
        logs["loss"] = loss.detach()
        return loss, logs

    heads = [] if data.head is None else [int(h) for h in torch.unique(data.head).tolist()]
    def _name(h):
        return head_names[h] if head_names is not None and h < len(head_names) else str(h)

    if len(heads) <= 1:
        w_e, w_f, w_s = (head_weights or {}).get(heads[0], weights) if heads else weights
        part_counts = {} if counts is not None else None
        loss, logs = _terms(pred, data, data.weight, w_e, w_f, w_s, d_e, d_f, d_s,
                            dipole_weight, polarizability_weight, huber_delta,
                            charge_weight, part_counts, les_dipole_weight,
                            les_charge_weight, property_weight=property_weight)
        if heads and head_names is not None:
            # one head of a multi-head run: log under its name, like mixed batches
            name = _name(heads[0])
            logs = {f"{name}/{k}": v for k, v in logs.items()}
            logs[f"{name}/loss"] = float(loss.detach())
            if part_counts is not None:
                part_counts = {f"{name}/{k}": v for k, v in part_counts.items()}
        if counts is not None:
            counts.update(part_counts)
        logs["loss"] = float(loss.detach())
        return loss, logs

    loss = _first_output(pred).new_zeros(())
    logs: dict[str, float] = {}
    for h in heads:
        mask = (data.head == h).to(_first_output(pred).dtype)
        w = mask if data.weight is None else data.weight * mask
        w_e, w_f, w_s = (head_weights or {}).get(h, weights)
        part_counts = {} if counts is not None else None
        part, part_logs = _terms(pred, data, w, w_e, w_f, w_s, d_e, d_f, d_s,
                                 dipole_weight, polarizability_weight, huber_delta,
                                 charge_weight, part_counts, les_dipole_weight,
                                 les_charge_weight, property_weight=property_weight)
        loss = loss + part
        name = _name(h)
        logs.update({f"{name}/{k}": v for k, v in part_logs.items()})
        if counts is not None:
            counts.update({f"{name}/{k}": v for k, v in part_counts.items()})
        logs[f"{name}/loss"] = float(part.detach())
    logs["loss"] = float(loss.detach())
    return loss, logs
