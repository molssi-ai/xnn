"""Error metrics for benchmarking, plus prediction collection.

A name -> callable registry mirrors the model registry (see
``models.registry``) so new metrics are added without touching the core, and
user-defined metrics register the same way. The built-in metrics are MAE, MSE
and RMSE; each takes two flat tensors ``(pred, target)`` and returns a Python
float.

:func:`collect_predictions` runs a model over a data loader once and returns
the paired prediction/target tensors for each requested target quantity
(``energy`` / ``forces`` / ``stress``), which :func:`score` then reduces with
the selected metrics. Energy is scored per atom -- the same size-extensive
normalization the training loss uses (see ``train.losses.weighted_loss``).
"""
from __future__ import annotations

import importlib
from typing import Callable

import torch
from torch import Tensor

from ..data import AtomicGraph
from ..models import scatter_sum

Metric = Callable[[Tensor, Tensor], float]

_METRICS: dict[str, Metric] = {}


def register_metric(name: str) -> Callable[[Metric], Metric]:
    """Return a decorator registering an error metric under ``name``.

    Parameters
    ----------
    name : str
        Name under which to register the metric. Lookups are case-insensitive
        (the name is lowercased internally).

    Returns
    -------
    Callable[[Metric], Metric]
        A decorator that registers the callable it wraps and returns it
        unchanged.

    Raises
    ------
    KeyError
        When applied, if ``name`` is already registered to a different callable.

    Examples
    --------
    >>> @register_metric("maxae")
    ... def max_abs_error(pred, target):
    ...     return float((pred - target).abs().max())
    """
    def deco(fn: Metric) -> Metric:
        key = name.lower()
        if key in _METRICS and _METRICS[key] is not fn:
            raise KeyError(f"metric '{name}' already registered")
        _METRICS[key] = fn
        return fn
    return deco


def get_metric(name: str) -> Metric:
    """Look up a registered metric by name.

    Parameters
    ----------
    name : str
        Metric name (case-insensitive).

    Returns
    -------
    Metric
        The registered callable.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    key = name.lower()
    if key not in _METRICS:
        raise KeyError(
            f"unknown metric '{name}'. registered: {available_metrics()}")
    return _METRICS[key]


def available_metrics() -> list[str]:
    """List the names of all currently registered metrics.

    Returns
    -------
    list of str
        The registered metric names, sorted alphabetically.
    """
    return sorted(_METRICS)


def load_custom_metric(name: str, path: str) -> Metric:
    """Import a ``module:function`` callable and register it under ``name``.

    Lets a benchmark config pull in user-defined metrics without any code
    changes to xnns: give the dotted import path of a callable with the
    ``(pred, target) -> float`` signature (see :data:`Metric`).

    Parameters
    ----------
    name : str
        Name to register the imported callable under.
    path : str
        Import path of the form ``"package.module:function"``.

    Returns
    -------
    Metric
        The imported (and now registered) callable.

    Raises
    ------
    ValueError
        If ``path`` does not contain the ``module:function`` separator.
    """
    if ":" not in path:
        raise ValueError(
            f"custom metric '{name}' path must be 'module:function', got {path!r}")
    module_name, func_name = path.split(":", 1)
    fn = getattr(importlib.import_module(module_name), func_name)
    return register_metric(name)(fn)


@register_metric("mae")
def mae(pred: Tensor, target: Tensor) -> float:
    """Mean absolute error, ``mean(|pred - target|)``."""
    return float((pred - target).abs().mean())


@register_metric("mse")
def mse(pred: Tensor, target: Tensor) -> float:
    """Mean squared error, ``mean((pred - target) ** 2)``."""
    return float(((pred - target) ** 2).mean())


@register_metric("rmse")
def rmse(pred: Tensor, target: Tensor) -> float:
    """Root mean squared error, ``sqrt(mean((pred - target) ** 2))``."""
    return float(((pred - target) ** 2).mean().sqrt())


def collect_predictions(model, loader, device, targets: list[str],
                        atomic_energies: Tensor | None = None,
                        energy_per_atom: bool = True,
                        ) -> dict[str, tuple[Tensor, Tensor]]:
    """Run ``model`` over ``loader`` once, pairing predictions with targets.

    The model is put in eval mode but gradients are left enabled, because
    force predictions differentiate the energy with respect to positions (the
    same reason :meth:`Trainer.evaluate` avoids ``torch.no_grad``). Each
    prediction/target pair is detached and moved to CPU before being stacked.

    Energy handling is controlled by ``atomic_energies`` and
    ``energy_per_atom``. When ``atomic_energies`` is given, the per-element
    reference energy of every atom is subtracted from both the predicted and
    reference total energy, turning them into *atomization* (interaction)
    energies -- the physically meaningful quantity to report (see
    :mod:`~xnns.common.benchmark.energy`). When ``energy_per_atom`` is set, the
    energy is then divided by the per-structure atom count, the size-extensive
    normalization the training loss uses, so energy metrics are comparable
    across differently sized structures. Both are applied identically to the
    prediction and the reference, so difference metrics (MAE/MSE/RMSE) are
    invariant to the E0 offset while their reported values become meaningful.

    Parameters
    ----------
    model : torch.nn.Module
        A model whose forward returns a dict with an ``"energy"`` key and,
        when forces/stress are requested, ``"forces"`` / ``"stress"`` keys
        (e.g. a :class:`~xnns.common.models.ForceStressOutput`).
    loader : torch.utils.data.DataLoader
        Loader yielding batched :class:`AtomicGraph` objects.
    device : torch.device
        Device to evaluate on.
    targets : list of str
        Which quantities to collect; a subset of ``"energy"``, ``"forces"``
        and ``"stress"``. A target is skipped for a batch that lacks the
        reference value or the corresponding prediction.
    atomic_energies : torch.Tensor or None, optional
        A ``Z``-indexed lookup of per-element reference energies (as built by
        :func:`~xnns.common.benchmark.energy.build_e0_lookup`). When given,
        energies are scored on an atomization basis. Defaults to ``None`` (raw
        total energy).
    energy_per_atom : bool, optional
        Whether to divide the (atomization) energy by the atom count before
        scoring. Defaults to ``True``.

    Returns
    -------
    dict of str to (Tensor, Tensor)
        Maps each target that had at least one value to a
        ``(prediction, reference)`` pair of flat CPU tensors of equal length.
        Targets with no data are omitted.
    """
    model.eval()
    preds: dict[str, list[Tensor]] = {t: [] for t in targets}
    refs: dict[str, list[Tensor]] = {t: [] for t in targets}

    for data in loader:
        data = data.to(device)
        out = model(data)
        if "energy" in targets and data.energy is not None and "energy" in out:
            e_pred, e_ref = out["energy"], data.energy
            if atomic_energies is not None:
                e0 = atomic_energies.to(e_pred.device, e_pred.dtype)
                e0_sum = scatter_sum(
                    e0[data.atomic_numbers], data.batch, data.num_graphs)
                e_pred = e_pred - e0_sum
                e_ref = e_ref - e0_sum
            if energy_per_atom:
                n = data.n_atoms.to(e_pred.dtype)
                e_pred = e_pred / n
                e_ref = e_ref / n
            preds["energy"].append(e_pred.detach().cpu())
            refs["energy"].append(e_ref.detach().cpu())
        if "forces" in targets and data.forces is not None and "forces" in out:
            preds["forces"].append(out["forces"].detach().reshape(-1).cpu())
            refs["forces"].append(data.forces.detach().reshape(-1).cpu())
        if "stress" in targets and data.stress is not None and "stress" in out:
            preds["stress"].append(out["stress"].detach().reshape(-1).cpu())
            refs["stress"].append(data.stress.detach().reshape(-1).cpu())

    out_pairs: dict[str, tuple[Tensor, Tensor]] = {}
    for t in targets:
        if preds[t]:
            out_pairs[t] = (torch.cat(preds[t]), torch.cat(refs[t]))
    return out_pairs


def score(pairs: dict[str, tuple[Tensor, Tensor]],
          metrics: list[str]) -> dict[str, float]:
    """Reduce prediction/target pairs to a flat ``{target_metric: value}`` map.

    Parameters
    ----------
    pairs : dict of str to (Tensor, Tensor)
        Prediction/target pairs keyed by target, as returned by
        :func:`collect_predictions`.
    metrics : list of str
        Names of registered metrics to apply to every target.

    Returns
    -------
    dict of str to float
        One entry per ``(target, metric)`` combination, keyed
        ``f"{target}_{metric}"`` (e.g. ``"energy_mae"``, ``"forces_rmse"``).
    """
    funcs = {m: get_metric(m) for m in metrics}
    result: dict[str, float] = {}
    for target, (pred, ref) in pairs.items():
        for name, fn in funcs.items():
            result[f"{target}_{name}"] = fn(pred, ref)
    return result
