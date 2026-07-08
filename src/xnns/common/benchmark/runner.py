"""Drive a multi-model benchmark through its train / evaluate / benchmark phases.

The three phases are independent switches; any combination the config lists is
run, so the same driver covers every workflow asked of it:

- ``[train, evaluate, benchmark]`` -- train each model, score it on the
  validation split, then score and tabulate it on the test split.
- ``[evaluate, benchmark]`` -- score pre-trained models on validation and test.
- ``[train, benchmark]`` -- train each model, then tabulate test scores.
- ``[benchmark]`` -- tabulate test scores for pre-trained models only.

A model is *trained* only when the ``train`` phase is active **and** the entry
has no ``checkpoint``; an entry that carries a checkpoint is treated as
pre-trained and its weights are loaded as-is. Training reuses the unchanged
:class:`~xnns.common.train.Trainer`; scoring reuses the model registry and
:class:`~xnns.common.models.ForceStressOutput`, so a benchmarked model is built
and run exactly as it would be in a single run.

Data splits are built once per cutoff (models sharing a cutoff share the split)
using the same seeded ``random_split`` the trainer uses, so a model's training
split and benchmark split are consistent.
"""
from __future__ import annotations

import copy
import os

import torch
from torch.utils.data import DataLoader, Subset, random_split

from ..data import AtomicDataset, collate
from ..models import build_model, ForceStressOutput
from ..train import Trainer
from ..train.trainer import resolve_device
from . import metrics as _metrics
from . import report
from .config import BenchmarkConfig, ModelEntry


class Benchmark:
    """Run a :class:`BenchmarkConfig` and collect a comparison table.

    Parameters
    ----------
    cfg : BenchmarkConfig
        The benchmark configuration: the models, shared data/optim sections,
        the phases to run, the metrics/targets to score, and where to write
        results.

    Attributes
    ----------
    cfg : BenchmarkConfig
        The configuration passed in.
    device : torch.device
        The resolved compute device.
    rows : list of dict
        The accumulated result rows after :meth:`run` (one per model per
        scored split).
    """

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.rows: list[dict] = []
        self._splits: dict[float, tuple] = {}
        # Register any user-defined metrics up front so scoring can find them.
        for cm in cfg.custom_metrics:
            _metrics.load_custom_metric(cm["name"], cm["path"])

    # -- data -------------------------------------------------------------
    def _splits_for(self, cutoff: float):
        """Return the ``(train, val, test)`` datasets for a given cutoff.

        Splits are cached per cutoff, so several models with the same cutoff
        share one neighbor-list build and one split. Explicit ``val_path`` /
        ``test_path`` files are used when given; otherwise validation and test
        subsets are carved out of the training set with the same seeded
        ``random_split`` the trainer uses (:attr:`BenchmarkConfig.seed`).

        Parameters
        ----------
        cutoff : float
            Neighbor-list cutoff radius (the model's cutoff).

        Returns
        -------
        tuple
            ``(train_set, val_set, test_set)``; any of ``val_set`` /
            ``test_set`` may be ``None`` when neither a path nor a split
            fraction provides it. ``train_set`` may be ``None`` only when no
            ``train_path`` is configured.
        """
        if cutoff in self._splits:
            return self._splits[cutoff]

        d = self.cfg.data
        keys = dict(energy_key=d.energy_key, forces_key=d.forces_key,
                    stress_key=d.stress_key)
        train = (AtomicDataset.from_file(d.train_path, cutoff, **keys)
                 if d.train_path else None)
        val = (AtomicDataset.from_file(d.val_path, cutoff, **keys)
               if d.val_path else None)
        test = (AtomicDataset.from_file(d.test_path, cutoff, **keys)
                if d.test_path else None)

        f_val = d.val_fraction if val is None else 0.0
        f_test = d.test_fraction if test is None else 0.0
        if train is not None and (f_val > 0 or f_test > 0):
            n = len(train)
            n_val = max(1, int(n * f_val)) if f_val > 0 else 0
            n_test = max(1, int(n * f_test)) if f_test > 0 else 0
            parts = random_split(
                train, [n - n_val - n_test, n_val, n_test],
                generator=torch.Generator().manual_seed(self.cfg.seed))
            train = parts[0]
            val = parts[1] if n_val else val
            test = parts[2] if n_test else test

        self._splits[cutoff] = (train, val, test)
        return self._splits[cutoff]

    def _loader(self, dataset) -> DataLoader | None:
        """Wrap a dataset in a non-shuffled evaluation loader.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset or None
            The dataset (or ``Subset``) to load; ``None`` yields ``None``.

        Returns
        -------
        torch.utils.data.DataLoader or None
            A loader using the benchmark's batch size / worker count, or
            ``None`` when ``dataset`` is ``None``.
        """
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=self.cfg.data.batch_size,
                          shuffle=False, collate_fn=collate,
                          num_workers=self.cfg.data.num_workers)

    # -- per-model model handling ----------------------------------------
    def _wrap(self, run_cfg):
        """Build a model from a run config, wrapped for force/stress output.

        Force and stress heads are enabled only when ``"forces"`` / ``"stress"``
        are among the benchmark targets, so autograd work matches what is
        actually scored.

        Parameters
        ----------
        run_cfg : Config
            The per-model run configuration.

        Returns
        -------
        ForceStressOutput
            The wrapped, uninitialized (freshly built) model on
            :attr:`device`.
        """
        base = build_model(run_cfg.model)
        wrapped = ForceStressOutput(
            base,
            compute_forces="forces" in self.cfg.targets,
            compute_stress="stress" in self.cfg.targets,
        )
        return wrapped.to(self.device)

    def _obtain_model(self, entry: ModelEntry, run_cfg):
        """Get the model to score: load a checkpoint or train from scratch.

        An entry with a ``checkpoint`` is treated as pre-trained and its
        weights are loaded. Otherwise, when the ``train`` phase is active, the
        model is trained with :class:`~xnns.common.train.Trainer` and its best
        (or last) checkpoint is loaded back. A model with neither a checkpoint
        nor an active training phase cannot be scored.

        Parameters
        ----------
        entry : ModelEntry
            The model entry (carries the optional checkpoint path).
        run_cfg : Config
            The per-model run configuration.

        Returns
        -------
        ForceStressOutput
            The wrapped model with weights loaded, on :attr:`device`.

        Raises
        ------
        ValueError
            If the entry has no checkpoint and the ``train`` phase is not
            active (there is nothing to benchmark).
        FileNotFoundError
            If a configured ``checkpoint`` path does not exist.
        """
        if entry.checkpoint is not None:
            if not os.path.exists(entry.checkpoint):
                raise FileNotFoundError(
                    f"[{entry.label}] checkpoint not found: {entry.checkpoint}")
            model = self._wrap(run_cfg)
            self._load_state(model, entry.checkpoint)
            return model

        if "train" not in self.cfg.phases:
            raise ValueError(
                f"[{entry.label}] no checkpoint given and 'train' is not in "
                f"phases {self.cfg.phases}; nothing to benchmark. Provide a "
                f"'checkpoint:' path or add 'train' to phases.")

        ckpt = self._train(entry, run_cfg)
        model = self._wrap(run_cfg)
        self._load_state(model, ckpt)
        return model

    def _train(self, entry: ModelEntry, run_cfg) -> str:
        """Train one model and return the path of the checkpoint to score.

        The trainer carves no splits of its own: the benchmark's pre-built
        train/validation datasets are passed explicitly and the run config's
        split fractions are zeroed, so the training split matches the split the
        model is later benchmarked on. The best-validation checkpoint is
        preferred; the final-epoch checkpoint is used when no validation split
        exists.

        Parameters
        ----------
        entry : ModelEntry
            The model entry being trained (used for its label in logging).
        run_cfg : Config
            The per-model run configuration (its ``output_dir`` receives the
            checkpoints).

        Returns
        -------
        str
            Path to ``best.pt`` if it was written, otherwise ``last.pt``.
        """
        train, val, _ = self._splits_for(run_cfg.model.cutoff)
        if train is None:
            raise ValueError(
                f"[{entry.label}] training requested but no 'data.train_path' "
                f"is configured.")
        print(f"[{entry.label}] training -> {run_cfg.output_dir}")
        # Zero the split fractions and pass val explicitly so the Trainer
        # reuses the benchmark's split instead of carving its own.
        train_cfg = copy.deepcopy(run_cfg)
        train_cfg.data.val_fraction = 0.0
        train_cfg.data.test_fraction = 0.0
        Trainer(train_cfg, train, val_set=val, test_set=None).fit()

        best = os.path.join(run_cfg.output_dir, "best.pt")
        return best if os.path.exists(best) else os.path.join(
            run_cfg.output_dir, "last.pt")

    def _load_state(self, model, ckpt_path: str) -> None:
        """Load a saved ``state_dict`` into a wrapped model.

        Parameters
        ----------
        model : ForceStressOutput
            The wrapped model to load weights into.
        ckpt_path : str
            Path to a checkpoint written by
            :meth:`~xnns.common.train.Trainer.save` (a dict with a ``"model"``
            key holding the state dict).
        """
        state = torch.load(ckpt_path, map_location=self.device,
                           weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)

    # -- scoring ----------------------------------------------------------
    def _score(self, model, loader, entry: ModelEntry, split: str) -> dict:
        """Score a model over a loader and build one result row.

        Parameters
        ----------
        model : ForceStressOutput
            The wrapped model to evaluate.
        loader : torch.utils.data.DataLoader
            The split loader to score over.
        entry : ModelEntry
            The model entry (for the row's ``model`` label).
        split : str
            Split name recorded in the row (``"val"`` / ``"test"``).

        Returns
        -------
        dict
            A row with ``model``, ``split``, ``n_params`` and one
            ``target_metric`` column per scored quantity.
        """
        pairs = _metrics.collect_predictions(
            model, loader, self.device, self.cfg.targets)
        scores = _metrics.score(pairs, self.cfg.metrics)
        n_params = sum(p.numel() for p in model.parameters())
        return {"model": entry.label, "split": split,
                "n_params": n_params, **scores}

    # -- driver -----------------------------------------------------------
    def run(self) -> list[dict]:
        """Execute the configured phases for every model and write results.

        For each model entry the weights are obtained (loaded or trained via
        :meth:`_obtain_model`), then the ``evaluate`` phase scores the
        validation split and the ``benchmark`` phase scores the test split
        (falling back to validation when no test split exists). Each scored
        split appends a row to :attr:`rows`. When the ``benchmark`` phase is
        active the accumulated table is printed and written to every configured
        output format.

        Returns
        -------
        list of dict
            The result rows (also stored on :attr:`rows`).
        """
        phases = self.cfg.phases
        for entry in self.cfg.models:
            run_cfg = entry.to_run_config(self.cfg)
            model = self._obtain_model(entry, run_cfg)
            _, val, test = self._splits_for(run_cfg.model.cutoff)

            if "evaluate" in phases:
                loader = self._loader(val)
                if loader is not None:
                    self.rows.append(self._score(model, loader, entry, "val"))
                else:
                    print(f"[{entry.label}] evaluate: no validation split, "
                          f"skipping")

            if "benchmark" in phases:
                split_name, dataset = ("test", test) if test is not None \
                    else ("val", val)
                loader = self._loader(dataset)
                if loader is not None:
                    self.rows.append(
                        self._score(model, loader, entry, split_name))
                else:
                    print(f"[{entry.label}] benchmark: no test or validation "
                          f"split, skipping")

        if "benchmark" in phases and self.rows:
            self._report()
        return self.rows

    def _report(self) -> None:
        """Print the results table and write it to every configured format."""
        out = self.cfg.output
        os.makedirs(out.dir, exist_ok=True)
        print("\n" + report.format_table(self.rows) + "\n")
        paths = report.write_all(self.rows, out.formats, out.dir, out.filename)
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
