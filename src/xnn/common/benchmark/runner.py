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

Two independent kinds of parallelism are available, and they combine:

* **Across the dataset** (one model, several GPUs or nodes). Launched through
  a distributed launcher exactly like training (``torchrun --nproc-per-node 4
  -m xnn benchmark ...``), every rank scores a disjoint strided shard of the
  dataset and the predictions are gathered before the metrics are computed, so
  the numbers equal those of a single-process run. Only rank 0 writes.
* **Across models** (one GPU per model). :func:`select_entries` restricts a run
  to some entries (``--models`` / ``--shard``), each such run writes its rows
  to a *part* file, and :func:`merge_parts` assembles the final table from the
  parts. On a cluster the parts are the tasks of a Slurm job array; on one node
  :func:`run_parallel` spawns one worker process per GPU and merges itself.
"""
from __future__ import annotations

import dataclasses
import glob
import json
import os
import re
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

from .. import distributed
from ..data import AtomicDataset, collate
from ..models import build_model, ForceStressOutput
from ..train.trainer import resolve_device
from . import metrics as _metrics
from . import report
from .config import BenchmarkConfig, ModelEntry
from .energy import build_e0_lookup

# Sentinel so the (possibly None) E0 lookup is built exactly once, lazily.
_UNSET = object()

# Sub-directory of output.dir holding the per-task row files of a split run.
PARTS_DIR = "parts"


class Benchmark:
    """Run a :class:`BenchmarkConfig` and collect a comparison table.

    Parameters
    ----------
    cfg : BenchmarkConfig
        The benchmark configuration: the pre-trained models, the dataset, the
        metrics/targets to score, and where to write results.
    entries : list of ModelEntry or None, optional
        The subset of ``cfg.models`` this run scores (see
        :func:`select_entries`). ``None`` (the default) scores every model and
        writes the final table; a subset writes a part file named after
        ``part`` instead, for :func:`merge_parts` to assemble later.
    part : str or None, optional
        The part file's tag; defaults to one derived from the selected labels.

    Attributes
    ----------
    cfg : BenchmarkConfig
        The configuration passed in.
    entries : list of ModelEntry
        The model entries this run scores.
    device : torch.device
        The resolved compute device (``cuda:LOCAL_RANK`` per rank when
        launched distributed on GPUs).
    distributed : bool
        Whether this process is one rank of a distributed launch, in which
        case the dataset is sharded across ranks and predictions are gathered.
    rank : int
        This process's rank (0 in a single-process run).
    is_main : bool
        Whether this is rank 0, the only rank that prints and writes.
    rows : list of dict
        The accumulated result rows after :meth:`run` (one per model).
    """

    def __init__(self, cfg: BenchmarkConfig, entries: list[ModelEntry] | None = None,
                 part: str | None = None):
        self.cfg = cfg
        self.entries = list(cfg.models) if entries is None else list(entries)
        self.part = part if entries is None or part is not None else part_tag(entries)
        self.distributed = distributed.is_distributed()
        self.device, self._owns_pg = distributed.init_process_group(
            resolve_device(cfg.device))
        self.rank = distributed.rank()
        self.is_main = self.rank == 0
        self.rows: list[dict] = []
        self._datasets: dict[float, AtomicDataset] = {}
        self._e0 = _UNSET
        # Register any user-defined metrics up front so scoring can find them.
        for cm in cfg.custom_metrics:
            _metrics.load_custom_metric(cm["name"], cm["path"])

    # data
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

        In a distributed run the loader walks only this rank's strided shard
        (:func:`~xnn.common.distributed.shard_indices`), with no padding, so
        the ranks' shards partition the dataset exactly.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to load.

        Returns
        -------
        torch.utils.data.DataLoader
            A loader using the benchmark's batch size / worker count.
        """
        sampler = distributed.shard_indices(len(dataset)) if self.distributed else None
        return DataLoader(dataset, batch_size=self.cfg.data.batch_size,
                          shuffle=False, sampler=sampler, collate_fn=collate,
                          num_workers=self.cfg.data.num_workers)

    # per-model model handling
    def _load_model(self, entry: ModelEntry):
        """Build the model and load the entry's checkpoint weights.

        The ``checkpoint`` is anything the model hub resolves (a trainer
        ``best.pt``, a portable model directory, a registered model name, URL
        or Zenodo DOI; see :func:`~xnn.common.models.hub.fetch_model`). The
        architecture is taken from the checkpoint itself when it embeds a
        config (checkpoints written by :meth:`~xnn.common.train.Trainer.save`
        carry their :class:`Config`), so an entry needs only a ``checkpoint``;
        the entry's own ``model`` section is used only for checkpoints that do
        not embed a config. Force and stress heads are enabled only when
        ``"forces"`` / ``"stress"`` are among the benchmark targets, so autograd
        work matches what is actually scored. In a distributed run rank 0
        fetches first, so a download happens once and the other ranks read
        the cached copy.

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
        from ..models.hub import fetch_model, load_checkpoint
        if self.distributed and not self.is_main:
            distributed.barrier()
        try:
            ck = load_checkpoint(fetch_model(entry.checkpoint))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"[{entry.label}] checkpoint not found: {entry.checkpoint} ({e})") from e
        finally:
            if self.distributed and self.is_main:
                distributed.barrier()
        # Prefer the architecture embedded in the checkpoint (xnn-trained
        # checkpoints store their Config), so entries need only a checkpoint.
        arch = dataclasses.asdict(ck.config.model) if ck.config is not None else None
        cfg = entry.to_config(self.cfg, arch)

        model = ForceStressOutput(
            build_model(cfg.model),
            compute_forces="forces" in self.cfg.targets,
            compute_stress="stress" in self.cfg.targets,
        ).to(self.device)
        model.load_state_dict(ck.state_dict)
        return model, cfg

    # scoring
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

    def _gather(self, pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]
                ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Concatenate every rank's prediction/reference pairs (distributed only).

        Each target is gathered in rank order with
        :func:`~xnn.common.distributed.all_gather_cat`; a rank whose shard
        holds no labelled structure for a target contributes an empty piece.
        The metrics are permutation-invariant, so the resulting order (rank 0's
        structures, then rank 1's, ...) does not matter, and every rank ends
        up with the full vectors and the same scores.

        Parameters
        ----------
        pairs : dict of str to (Tensor, Tensor)
            This rank's pairs as returned by
            :func:`~xnn.common.benchmark.metrics.collect_predictions`.

        Returns
        -------
        dict of str to (Tensor, Tensor)
            The gathered pairs; targets without any data on any rank are
            omitted, as in a single-process run.
        """
        if not self.distributed:
            return pairs
        empty = torch.empty(0)
        out = {}
        for t in self.cfg.targets:
            pred, ref = pairs.get(t, (empty, empty))
            pred = distributed.all_gather_cat(pred, self.device)
            ref = distributed.all_gather_cat(ref, self.device)
            if pred.numel():
                out[t] = (pred, ref)
        return out

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
        scores = _metrics.score(self._gather(pairs), self.cfg.metrics)
        n_params = sum(p.numel() for p in model.parameters())
        return {"model": entry.label, "n_params": n_params, **scores}

    # driver
    def run(self) -> list[dict]:
        """Score every selected model on the benchmark dataset and write the results.

        For each model entry the architecture is built and its checkpoint
        weights loaded (:meth:`_load_model`), then the model is scored on the
        benchmark dataset (:meth:`_score`), appending one row to :attr:`rows`.
        A full run prints the accumulated table and writes it to every
        configured output format; a run restricted to a subset of the entries
        prints its table and writes a part file instead (see
        :func:`merge_parts`). Only rank 0 of a distributed run writes.

        Returns
        -------
        list of dict
            The result rows (also stored on :attr:`rows`).
        """
        for entry in self.entries:
            model, cfg = self._load_model(entry)
            dataset = self._dataset(cfg.model.cutoff)
            e0 = self._atomic_energies(dataset)
            self.rows.append(
                self._score(model, self._loader(dataset), entry, e0))

        if self.rows and self.is_main:
            if self.part is None:
                self._report()
            else:
                print("\n" + report.format_table(self.rows, self._column_units()) + "\n")
                print("wrote", write_part(self.cfg, self.rows, self.part))
        distributed.destroy_process_group(self._owns_pg)
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
        return column_units(self.cfg, self.rows)

    def _report(self) -> None:
        """Print the results table and write it to every configured format.

        The printed table and the written files carry the same unit-annotated
        headers (e.g. ``energy_mae [eV/atom]``); :attr:`rows` itself keeps plain
        keys for programmatic use.
        """
        write_report(self.cfg, self.rows)


def column_units(cfg: BenchmarkConfig, rows: list[dict]) -> dict[str, str]:
    """Map each metric column of ``rows`` to its unit label (see :meth:`Benchmark._column_units`).

    Parameters
    ----------
    cfg : BenchmarkConfig
        Supplies ``energy_per_atom`` and the ``units`` overrides.
    rows : list of dict
        The result rows whose columns are labelled.

    Returns
    -------
    dict of str to str
        Column name -> unit, for the metric columns that have a unit.
    """
    per_target = {
        "energy": "eV/atom" if cfg.energy_per_atom else "eV",
        "forces": "eV/A",
        "stress": "eV/A**3",
    }
    per_target.update(cfg.units or {})
    cols: dict[str, str] = {}
    for col in report.columns(rows):
        for target, unit in per_target.items():
            if unit and col.startswith(f"{target}_"):
                cols[col] = unit
    return cols


def write_report(cfg: BenchmarkConfig, rows: list[dict]) -> list[str]:
    """Print the results table and write it to every configured format.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Supplies the output directory, filename and formats.
    rows : list of dict
        The result rows (plain keys; units are added to the written headers).

    Returns
    -------
    list of str
        The paths written, one per format.
    """
    out = cfg.output
    os.makedirs(out.dir, exist_ok=True)
    units = column_units(cfg, rows)
    print("\n" + report.format_table(rows, units) + "\n")
    labeled = report.apply_units(rows, units)
    paths = report.write_all(labeled, out.formats, out.dir, out.filename)
    for p in paths:
        print("wrote", p)
    return paths


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


# splitting a benchmark over several runs
def select_entries(cfg: BenchmarkConfig, models: list[str] | str | None = None,
                   shard: str | None = None) -> list[ModelEntry]:
    """Pick the model entries one run of a split benchmark scores.

    Parameters
    ----------
    cfg : BenchmarkConfig
        The full benchmark configuration.
    models : list of str, str or None, optional
        Labels (or zero-based positions) of the entries to keep, as a list or
        one comma-separated string. ``None`` keeps all.
    shard : str or None, optional
        ``"I/N"`` keeps every ``N``-th entry starting at ``I`` (zero-based),
        so ``N`` runs with ``I = 0 .. N-1`` cover the list; ``"slurm"`` reads
        ``I`` and ``N`` from the Slurm job-array variables
        (``SLURM_ARRAY_TASK_ID`` relative to ``SLURM_ARRAY_TASK_MIN``, and
        ``SLURM_ARRAY_TASK_COUNT``). Applied after ``models``.

    Returns
    -------
    list of ModelEntry
        The selected entries, in config order.

    Raises
    ------
    ValueError
        For an unknown label, a malformed ``shard``, or ``"slurm"`` outside a
        job array.
    """
    entries = list(cfg.models)
    if models is not None:
        wanted = [m.strip() for m in (models.split(",") if isinstance(models, str) else models)]
        wanted = [m for m in wanted if m]
        by_label = {e.label: e for e in entries}
        picked = []
        for m in wanted:
            if m in by_label:
                picked.append(by_label[m])
            elif m.isdigit() and int(m) < len(entries):
                picked.append(entries[int(m)])
            else:
                raise ValueError(
                    f"unknown model {m!r}; the config has {list(by_label)}")
        entries = picked

    if shard is not None:
        if shard.strip().lower() == "slurm":
            env = os.environ
            if "SLURM_ARRAY_TASK_ID" not in env:
                raise ValueError("--shard slurm needs a Slurm job array "
                                 "(SLURM_ARRAY_TASK_ID is not set)")
            i = int(env["SLURM_ARRAY_TASK_ID"]) - int(env.get("SLURM_ARRAY_TASK_MIN", "0"))
            n = int(env["SLURM_ARRAY_TASK_COUNT"])
        else:
            m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", shard)
            if m is None:
                raise ValueError(f"shard must look like I/N (got {shard!r}) or be 'slurm'")
            i, n = int(m.group(1)), int(m.group(2))
        if not 0 <= i < n:
            raise ValueError(f"shard index {i} is outside 0..{n - 1}")
        entries = entries[i::n]
    return entries


def part_tag(entries: list[ModelEntry]) -> str:
    """A filesystem-safe tag naming a subset of entries (their labels joined by ``+``)."""
    return "+".join(re.sub(r"[^\w.\-#]+", "_", e.label) for e in entries)


def _parts_dir(cfg: BenchmarkConfig) -> str:
    return os.path.join(cfg.output.dir, PARTS_DIR)


def write_part(cfg: BenchmarkConfig, rows: list[dict], tag: str) -> str:
    """Write the rows of one run of a split benchmark to its part file.

    Parts are JSON row lists with plain (unit-free) keys under
    ``<output.dir>/parts/<filename>.<tag>.json``; :func:`merge_parts` reads
    them back.

    Parameters
    ----------
    cfg : BenchmarkConfig
        Supplies the output directory and filename.
    rows : list of dict
        The rows this run produced.
    tag : str
        Distinguishes this run's file (see :func:`part_tag`).

    Returns
    -------
    str
        The path written.
    """
    os.makedirs(_parts_dir(cfg), exist_ok=True)
    path = os.path.join(_parts_dir(cfg), f"{cfg.output.filename}.{tag}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path


def merge_parts(cfg: BenchmarkConfig) -> list[dict]:
    """Assemble the final table from the part files of a split benchmark.

    Every ``parts/<filename>.*.json`` under the output directory is read, the
    rows are put in the config's model order, and the table is printed and
    written in every configured format exactly as a single full run would.
    A model without a part (a failed task) is reported on stderr and left
    out; a part whose label is not in the config (stale, from an earlier
    config) is ignored with a note. Rerun the merge after fixing the tasks.

    Parameters
    ----------
    cfg : BenchmarkConfig
        The full benchmark configuration the parts were produced with.

    Returns
    -------
    list of dict
        The merged rows, in config order.

    Raises
    ------
    FileNotFoundError
        If there are no part files to merge.
    """
    pattern = os.path.join(_parts_dir(cfg), f"{cfg.output.filename}.*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no part files match {pattern}; run the parts first "
                                f"(xnn benchmark --shard / --models)")
    found: dict[str, dict] = {}
    for path in files:
        with open(path) as f:
            for row in json.load(f):
                found[row["model"]] = row
    labels = [e.label for e in cfg.models]
    missing = [l for l in labels if l not in found]
    stale = [l for l in found if l not in labels]
    if missing:
        print(f"merge: no part for {missing}; their rows are left out", file=sys.stderr)
    if stale:
        print(f"merge: ignoring parts for models not in the config: {stale}", file=sys.stderr)
    rows = [found[l] for l in labels if l in found]
    if rows:
        write_report(cfg, rows)
    return rows


def _visible_gpus() -> list[str]:
    """The CUDA device ids a worker may be pinned to (honouring CUDA_VISIBLE_DEVICES)."""
    if not torch.cuda.is_available():
        return []
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [d for d in env.split(",") if d.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def run_parallel(raw: dict, workers: int | None = None) -> list[dict]:
    """Score the models of a benchmark concurrently, one worker process per model.

    The in-node counterpart of a Slurm job array: the effective config is
    written to ``<output.dir>/parts/config.yaml``, and ``xnn benchmark
    --models <label>`` is spawned for each entry, at most ``workers`` at a
    time. With GPUs available each worker is pinned to one of them through
    ``CUDA_VISIBLE_DEVICES`` (worker ``k`` takes the ``k``-th visible GPU,
    cycling), so the default of one worker per GPU scores the models in
    parallel without sharing devices. Each worker's output goes to
    ``parts/<label>.log``. When all workers have finished the parts are
    merged into the final table with :func:`merge_parts`.

    Parameters
    ----------
    raw : dict
        The benchmark config as a plain dict (after any command-line
        overrides), as :func:`~xnn.common.benchmark.from_dict` takes it.
    workers : int or None, optional
        How many models to score at once. Defaults to the number of visible
        GPUs; on a machine without GPUs it must be given.

    Returns
    -------
    list of dict
        The merged rows, in config order.

    Raises
    ------
    ValueError
        If run inside a distributed launch (the two fan-outs are exclusive
        at this level; combine them through a job array instead), or if no
        worker count can be inferred.
    RuntimeError
        If any worker fails; the message names its log file.
    """
    import yaml
    from .config import from_dict

    if distributed.is_distributed():
        raise ValueError("--parallel spawns its own workers and cannot run under a "
                         "distributed launcher; use torchrun for one model at a time "
                         "(for example one job-array task per model)")
    cfg = from_dict(raw)
    gpus = _visible_gpus() if resolve_device(cfg.device).type == "cuda" else []
    if workers is None or workers <= 0:
        if not gpus:
            raise ValueError("--parallel needs an explicit worker count on a CPU-only run "
                             "(for example --parallel 4)")
        workers = len(gpus)

    parts = _parts_dir(cfg)
    os.makedirs(parts, exist_ok=True)
    # workers see only their own GPU, so their device is simply "cuda"
    child_raw = dict(raw, device="cuda" if gpus else "cpu")
    config_path = os.path.join(parts, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(child_raw, f, sort_keys=False)

    pending = list(cfg.models)
    # each running worker: (process, entry, log path, open log handle, gpu slot)
    running: list[tuple] = []
    free_slots = list(range(workers))
    failed: list[tuple[str, str]] = []
    while pending or running:
        while pending and free_slots:
            entry, slot = pending.pop(0), free_slots.pop(0)
            env = dict(os.environ)
            where = ""
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[slot % len(gpus)]
                where = f" on GPU {env['CUDA_VISIBLE_DEVICES']}"
            log_path = os.path.join(parts, f"{part_tag([entry])}.log")
            log = open(log_path, "w")
            proc = subprocess.Popen(
                [sys.executable, "-m", "xnn", "benchmark", "--config", config_path,
                 "--models", entry.label],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, entry, log_path, log, slot))
            print(f"started {entry.label}{where} (log: {log_path})", flush=True)
        # hand a finished worker's slot to the next model as soon as it is free
        done = [r for r in running if r[0].poll() is not None]
        if not done:
            time.sleep(0.2)
            continue
        for proc, entry, log_path, log, slot in done:
            running.remove((proc, entry, log_path, log, slot))
            log.close()
            free_slots.append(slot)
            if proc.returncode != 0:
                failed.append((entry.label, log_path))
                print(f"FAILED {entry.label} (see {log_path})", flush=True)
            else:
                print(f"finished {entry.label}", flush=True)

    if failed:
        raise RuntimeError("benchmark workers failed: "
                           + "; ".join(f"{label} (see {log})" for label, log in failed))
    return merge_parts(cfg)
