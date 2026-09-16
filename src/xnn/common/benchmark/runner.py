"""Score pre-trained models on one dataset and tabulate their errors.

Benchmarking does one thing: for every model in the config it builds the
architecture, loads the entry's ``checkpoint`` into it, scores it on the
benchmark dataset with the configured metrics, and writes a comparison table.
It does not train or evaluate during training -- produce the checkpoints with
``xnn train`` (or any other route) first.

Model building reuses the model registry and
:class:`~xnn.common.models.ForceStressOutput`, so a benchmarked model is built
and run exactly as it would be in a single run. The benchmark dataset is built
once per cutoff, so models sharing a cutoff share one neighbor-list build.
"""
from __future__ import annotations

import dataclasses
import os

import torch
from torch.utils.data import DataLoader

from ..data import AtomicDataset, collate
from ..models import build_model, ForceStressOutput
from ..train.trainer import resolve_device
from . import metrics as _metrics
from . import report
from .config import BenchmarkConfig, ModelEntry
from .energy import build_e0_lookup

# Sentinel so the (possibly None) E0 lookup is built exactly once, lazily.
_UNSET = object()


class Benchmark:
    """Run a :class:`BenchmarkConfig` and collect a comparison table.

    Parameters
    ----------
    cfg : BenchmarkConfig
        The benchmark configuration: the pre-trained models, the dataset, the
        metrics/targets to score, and where to write results.

    Attributes
    ----------
    cfg : BenchmarkConfig
        The configuration passed in.
    device : torch.device
        The resolved compute device.
    rows : list of dict
        The accumulated result rows after :meth:`run` (one per model).
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.rows: list[dict] = []
        self._datasets: dict[float, AtomicDataset] = {}
        self._e0 = _UNSET
        # Register any user-defined metrics up front so scoring can find them.
        for cm in cfg.custom_metrics:
            _metrics.load_custom_metric(cm["name"], cm["path"])

    # -- data -------------------------------------------------------------
    def _dataset(self, cutoff: float) -> AtomicDataset:
        """Return the benchmark dataset for a given cutoff (cached per cutoff).

        The dataset path is resolved from the ``data`` section: ``test_path``
        (the natural held-out benchmark set), falling back to ``val_path`` then
        ``train_path``. Several models with the same cutoff share one build.

        Parameters
        ----------
        cutoff : float
            Neighbor-list cutoff radius (the model's cutoff).

        Returns
        -------
        AtomicDataset
            The dataset all models of this cutoff are scored on.

        Raises
        ------
        ValueError
            If no dataset path is configured.
        """
        if cutoff in self._datasets:
            return self._datasets[cutoff]

        d = self.cfg.data
        path = d.test_path or d.val_path or d.train_path
        if path is None:
            raise ValueError(
                "no benchmark dataset configured; set data.test_path (or "
                "data.val_path / data.train_path)")
        keys = dict(energy_key=d.energy_key, forces_key=d.forces_key,
                    stress_key=d.stress_key)
        self._datasets[cutoff] = AtomicDataset.from_file(path, cutoff, **keys)
        return self._datasets[cutoff]

    def _loader(self, dataset) -> DataLoader:
        """Wrap a dataset in a non-shuffled evaluation loader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to load.

        Returns
        -------
        torch.utils.data.DataLoader
            A loader using the benchmark's batch size / worker count.
        """
        return DataLoader(dataset, batch_size=self.cfg.data.batch_size,
                          shuffle=False, collate_fn=collate,
                          num_workers=self.cfg.data.num_workers)

    # -- per-model model handling ----------------------------------------
    def _load_model(self, entry: ModelEntry):
        """Build the model and load the entry's checkpoint weights.

        The architecture is taken from the checkpoint itself when it embeds a
        config (checkpoints written by :meth:`~xnn.common.train.Trainer.save`
        carry their :class:`Config`), so an entry needs only a ``checkpoint``;
        the entry's own ``model`` section is used only for checkpoints that do
        not embed a config. Force and stress heads are enabled only when
        ``"forces"`` / ``"stress"`` are among the benchmark targets, so autograd
        work matches what is actually scored.

        Parameters
        ----------
        entry : ModelEntry
            The model entry, carrying the required ``checkpoint`` path.

        Returns
        -------
        tuple of (ForceStressOutput, Config)
            The wrapped model with weights loaded (on :attr:`device`) and the
            per-model configuration it was built from (its ``model.cutoff``
            selects the dataset).

        Raises
        ------
        ValueError
            If the entry has no ``checkpoint`` (there is nothing to score).
        FileNotFoundError
            If the ``checkpoint`` path does not exist.
        """
        if entry.checkpoint is None:
            raise ValueError(
                f"[{entry.label}] no 'checkpoint' given; benchmarking scores "
                f"pre-trained models, so every entry needs a checkpoint path "
                f"(train one first with 'xnn train').")
        if not os.path.exists(entry.checkpoint):
            raise FileNotFoundError(
                f"[{entry.label}] checkpoint not found: {entry.checkpoint}")

        state = torch.load(entry.checkpoint, map_location=self.device,
                           weights_only=False)
        # Prefer the architecture embedded in the checkpoint (xnn-trained
        # checkpoints store their Config), so entries need only a checkpoint.
        stored = state.get("cfg") if isinstance(state, dict) else None
        arch = dataclasses.asdict(stored.model) if stored is not None else None
        cfg = entry.to_config(self.cfg, arch)

        model = ForceStressOutput(
            build_model(cfg.model),
            compute_forces="forces" in self.cfg.targets,
            compute_stress="stress" in self.cfg.targets,
        ).to(self.device)
        sd = state["model"] if isinstance(state, dict) and "model" in state \
            else state
        model.load_state_dict(sd)
        return model, cfg

    # -- scoring ----------------------------------------------------------
    def _atomic_energies(self, dataset):
        """Build (once) the ``Z``-indexed E0 lookup for atomization scoring.

        Cached across models: the E0s are a property of the elements and the
        dataset, not of the model, so they are built lazily on first use (from
        the benchmark dataset, which the ``"average"`` fit needs) and reused.

        Parameters
        ----------
        dataset : AtomicDataset
            The benchmark dataset, used only to fit E0s when
            ``atomic_energies="average"``.

        Returns
        -------
        torch.Tensor or None
            The E0 lookup, or ``None`` when energies are scored as raw totals.
        """
        if self._e0 is _UNSET:
            self._e0 = build_e0_lookup(
                self.cfg.atomic_energies, self.cfg.species, dataset)
        return self._e0

    def _score(self, model, loader, entry: ModelEntry, atomic_energies) -> dict:
        """Score a model over the benchmark loader and build one result row.

        Parameters
        ----------
        model : ForceStressOutput
            The wrapped model to evaluate.
        loader : torch.utils.data.DataLoader
            The benchmark loader to score over.
        entry : ModelEntry
            The model entry (for the row's ``model`` label).
        atomic_energies : torch.Tensor or None
            E0 lookup for atomization-energy scoring, or ``None`` for raw
            total energy.

        Returns
        -------
        dict
            A row with ``model``, ``n_params`` and one ``target_metric`` column
            per scored quantity.
        """
        pairs = _metrics.collect_predictions(
            model, loader, self.device, self.cfg.targets,
            atomic_energies=atomic_energies,
            energy_per_atom=self.cfg.energy_per_atom)
        scores = _metrics.score(pairs, self.cfg.metrics)
        n_params = sum(p.numel() for p in model.parameters())
        return {"model": entry.label, "n_params": n_params, **scores}

    # -- driver -----------------------------------------------------------
    def run(self) -> list[dict]:
        """Score every model on the benchmark dataset and write the results.

        For each model entry the architecture is built and its checkpoint
        weights loaded (:meth:`_load_model`), then the model is scored on the
        benchmark dataset (:meth:`_score`), appending one row to :attr:`rows`.
        The accumulated table is printed and written to every configured output
        format.

        Returns
        -------
        list of dict
            The result rows (also stored on :attr:`rows`).
        """
        for entry in self.cfg.models:
            model, cfg = self._load_model(entry)
            dataset = self._dataset(cfg.model.cutoff)
            e0 = self._atomic_energies(dataset)
            self.rows.append(
                self._score(model, self._loader(dataset), entry, e0))

        if self.rows:
            self._report()
        return self.rows

    def _column_units(self) -> dict[str, str]:
        """Map each metric column to its physical unit for the printed table.

        Units are labels only (xnn is unit-agnostic). Defaults are ``eV/atom``
        (or ``eV`` when :attr:`BenchmarkConfig.energy_per_atom` is off) for
        energy, ``eV/A`` for forces and ``eV/A**3`` for stress; any
        :attr:`BenchmarkConfig.units` entry overrides its target. The unit for a
        target is applied to all of that target's ``<target>_<metric>`` columns.

        Returns
        -------
        dict of str to str
            Column name -> unit, for the metric columns that have a unit.
        """
        per_target = {
            "energy": "eV/atom" if self.cfg.energy_per_atom else "eV",
            "forces": "eV/A",
            "stress": "eV/A**3",
        }
        per_target.update(self.cfg.units or {})
        cols: dict[str, str] = {}
        for col in report.columns(self.rows):
            for target, unit in per_target.items():
                if unit and col.startswith(f"{target}_"):
                    cols[col] = unit
        return cols

    def _report(self) -> None:
        """Print the results table and write it to every configured format.

        The printed table and the written files carry the same unit-annotated
        headers (e.g. ``energy_mae [eV/atom]``); :attr:`rows` itself keeps plain
        keys for programmatic use.
        """
        out = self.cfg.output
        os.makedirs(out.dir, exist_ok=True)
        units = self._column_units()
        print("\n" + report.format_table(self.rows, units) + "\n")
        labeled = report.apply_units(self.rows, units)
        paths = report.write_all(labeled, out.formats, out.dir, out.filename)
        for p in paths:
            print("wrote", p)


def run_benchmark(cfg: BenchmarkConfig) -> list[dict]:
    """Convenience wrapper: build a :class:`Benchmark` and run it.

    Parameters
    ----------
    cfg : BenchmarkConfig
        The benchmark configuration.

    Returns
    -------
    list of dict
        The result rows produced by :meth:`Benchmark.run`.
    """
    return Benchmark(cfg).run()
