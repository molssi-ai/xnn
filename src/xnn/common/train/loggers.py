"""Training loggers: where the trainer's metrics go (console, CSV, TensorBoard or your own).

A logger receives flat ``{name: value}`` dictionaries with the optimizer step
they belong to. The :class:`~xnn.common.train.trainer.Trainer` sends one per
epoch (``epoch``, ``lr``, ``train/<term>``, ``val/<term>``), one after the
test evaluation (``test/<term>``) and, with ``log_every_n_steps > 0``, the
training batch values (``train_step/<term>``). Subclass :class:`Logger` to
send them anywhere else.
"""
from __future__ import annotations

import csv
import os

__all__ = ["Logger", "ConsoleLogger", "CSVLogger", "TensorBoardLogger", "LOGGERS",
           "build_loggers", "metric_units"]

#: units of the scored quantities, from the energy unit ``E`` and the length unit ``L`` of the data
_UNITS = {"energy": "{E}/atom", "forces": "{E}/{L}", "stress": "{E}/{L}^3", "dipole": "e*{L}",
          "polarizability": "{L}^3", "charges": "e", "les_dipole": "e*{L}", "les_charges": "e"}
#: the quantity whose squared error each loss term averages
_TERMS = {"energy_mse": "energy", "force_mse": "forces", "stress_mse": "stress", "dipole_mse": "dipole",
          "polarizability_mse": "polarizability", "charge_mse": "charges",
          "les_dipole_mse": "les_dipole", "les_charge_mse": "les_charges"}


def metric_units(energy_unit: str = "eV", length_unit: str = "A") -> dict:
    """Units of the scored quantities (energy per atom, forces, stress, dipoles, ...).

    Parameters
    ----------
    energy_unit : str, optional
        Unit of the data's energies. Defaults to ``"eV"``.
    length_unit : str, optional
        Unit of the data's positions. Defaults to ``"A"`` (Angstrom).

    Returns
    -------
    dict of str to str
        Quantity names (``"energy"``, ``"forces"``, ...) mapped to units, e.g.
        ``{"energy": "eV/atom", "forces": "eV/A", ...}``.
    """
    return {k: v.format(E=energy_unit, L=length_unit) for k, v in _UNITS.items()}


def _squared(unit: str) -> str:
    """``unit`` squared factor by factor: ``eV/A`` gives ``eV^2/A^2``."""
    def square(factor):
        base, _, power = factor.partition("^")
        if not power:
            return f"{factor}^2"
        return f"{base}^{2 * int(power)}" if power.isdigit() else f"({factor})^2"
    return "/".join("*".join(square(f) for f in part.split("*")) for part in unit.split("/"))


def _with_unit(text: str, unit: str | None) -> str:
    return f"{text} {unit}" if unit else text


def _summary(metrics: dict, units: dict | None = None) -> str:
    """Format a metrics dict as ``loss (term values; per-head losses)``.

    Parameters
    ----------
    metrics : dict of str to float
        Dataset-level metrics as returned by ``Trainer._dataset_metrics``.
    units : dict of str to str, optional
        Units of the scored quantities (:func:`metric_units`); each term is
        labelled with the square of its quantity's unit. The weighted loss
        mixes quantities and stays without a unit.

    Returns
    -------
    str
        For example ``"1.2340e-02 (energy 1.100e-04 eV^2/atom^2, force 1.200e-03 eV^2/A^2)"``.
    """
    units = units or {}

    def unit(key):
        quantity = units.get(_TERMS.get(key, ""))
        return _squared(quantity) if quantity else None
    parts = [_with_unit(f"{k[:-4]} {v:.3e}", unit(k)) for k, v in sorted(metrics.items())
             if k.endswith("_mse") and "/" not in k]
    heads = [f"{k[:-5]} {v:.3e}" for k, v in sorted(metrics.items())
             if k.endswith("/loss")]
    detail = "; ".join(x for x in (", ".join(parts), ", ".join(heads)) if x)
    return f"{metrics.get('loss', 0):.4e}" + (f" ({detail})" if detail else "")


def _errors(metrics: dict, units: dict | None = None) -> str:
    """Format the error metrics as ``mae: energy 1.2e-03 eV/atom, forces 4.5e-02 eV/A | rmse: ...``."""
    units = units or {}
    out = []
    for kind in ("mae", "rmse"):
        parts = [_with_unit(f"{k[:-len(kind) - 1]} {v:.3e}", units.get(k[:-len(kind) - 1].split("/")[-1]))
                 for k, v in sorted(metrics.items()) if k.endswith("_" + kind)]
        if parts:
            out.append(f"{kind}: " + ", ".join(parts))
    return " | ".join(out)


def _split(metrics: dict, prefix: str) -> dict:
    """The entries of ``metrics`` under ``prefix/``, with the prefix removed."""
    n = len(prefix) + 1
    return {k[n:]: v for k, v in metrics.items() if k.startswith(prefix + "/")}


class Logger:
    """Base class of the training loggers; override :meth:`log_metrics` (and optionally the others).

    The trainer calls the hooks on rank 0 only, so a logger never needs to
    know about distributed runs.
    """

    def setup(self, output_dir: str) -> None:
        """Called once by the trainer with ``cfg.output_dir``, before anything is logged.

        Parameters
        ----------
        output_dir : str
            The run's output directory, the default place for log files.
        """

    def set_units(self, units: dict) -> None:
        """Called by the trainer, before anything is logged, with the units of the data.

        Parameters
        ----------
        units : dict of str to str
            Quantity names mapped to units, as returned by :func:`metric_units`.
            The loss terms are mean squares of these quantities.
        """

    def log_hyperparams(self, params: dict) -> None:
        """Record the run configuration (the config as a nested dict) at the start of ``fit``.

        Parameters
        ----------
        params : dict
            The full configuration, as written to a checkpoint's ``config.yaml``.
        """

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Record scalar metrics.

        Parameters
        ----------
        metrics : dict of str to float
            Metric names mapped to values.
        step : int
            The number of optimizer steps taken so far.
        """
        raise NotImplementedError

    def finalize(self, status: str) -> None:
        """Flush and close at the end of ``fit``.

        Parameters
        ----------
        status : str
            ``"success"``, or ``"failed"`` when ``fit`` raised.
        """


class ConsoleLogger(Logger):
    """Prints the epoch summary, the test loss and any step values to stdout.

    Every loss term and error carries its unit: the units of the data
    (``data.energy_unit``, ``data.length_unit``, eV and Angstrom by default)
    for the errors and their squares for the loss terms. The weighted total
    loss mixes quantities and is printed without a unit.

    Parameters
    ----------
    errors : bool, optional
        Also print the error metrics (MAE / RMSE) of the validation set, or of
        the training set without one, on a second line. Defaults to ``True``.
    """

    def __init__(self, errors: bool = True):
        self.errors = errors
        self.units = metric_units()

    def set_units(self, units: dict) -> None:
        """Label the printed values with ``units`` (see :meth:`Logger.set_units`)."""
        self.units = dict(units)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Print ``metrics`` in the trainer's line format (see :class:`Logger`)."""
        tr, va, te = (_split(metrics, p) for p in ("train", "val", "test"))
        batch = _split(metrics, "train_step")
        if "epoch" in metrics:
            msg = f"epoch {int(metrics['epoch']):4d} | train loss {_summary(tr, self.units)}"
            if va:
                msg += f" | val loss {_summary(va, self.units)}"
            errors = _errors(va or tr, self.units) if self.errors else ""
            if errors:
                msg += f"\n{'':10s}{'val' if va else 'train'} {errors}"
            print(msg)
        elif te:
            errors = _errors(te, self.units) if self.errors else ""
            print(f"test loss {_summary(te, self.units)}" + (f"\n{'':10s}test {errors}" if errors else ""))
        elif batch:
            print(f"step {step:7d} | train loss {_summary(batch, self.units)}")
        else:
            print(f"step {step:7d} | "
                  + ", ".join(f"{k} {v:.4e}" for k, v in sorted(metrics.items())))


class CSVLogger(Logger):
    """Writes every logged dictionary as one row of a CSV file (``step`` plus one column per metric).

    Parameters
    ----------
    save_dir : str or None, optional
        Directory of ``metrics.csv`` and ``hparams.yaml``; defaults to the
        run's ``output_dir``.
    filename : str, optional
        Name of the metrics file. Defaults to ``"metrics.csv"``.
    """

    def __init__(self, save_dir: str | None = None, filename: str = "metrics.csv"):
        self.save_dir = save_dir
        self.filename = filename
        self._rows: list[dict] = []
        self._columns: list[str] = []

    def setup(self, output_dir: str) -> None:
        """Default :attr:`save_dir` to ``output_dir``."""
        if self.save_dir is None:
            self.save_dir = output_dir

    @property
    def path(self) -> str:
        """Path of the metrics file."""
        return os.path.join(self.save_dir or ".", self.filename)

    def log_hyperparams(self, params: dict) -> None:
        """Write ``params`` to ``hparams.yaml`` next to the metrics file."""
        import yaml
        os.makedirs(self.save_dir or ".", exist_ok=True)
        with open(os.path.join(self.save_dir or ".", "hparams.yaml"), "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Append a row; the file is rewritten whenever a new column appears."""
        row = {"step": step, **metrics}
        self._rows.append(row)
        new = [k for k in row if k not in self._columns]
        os.makedirs(self.save_dir or ".", exist_ok=True)
        if new or len(self._rows) == 1:
            self._columns += new
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._columns, restval="")
                writer.writeheader()
                writer.writerows(self._rows)
        else:
            with open(self.path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._columns, restval="").writerow(row)


class TensorBoardLogger(Logger):
    """Writes the metrics as TensorBoard scalars through ``torch.utils.tensorboard.SummaryWriter``.

    Needs the ``tensorboard`` package (``pip install "xnns[tensorboard]"``).
    The writer is created on first use, so constructing the logger on every
    rank of a distributed run creates no files.

    Parameters
    ----------
    save_dir : str or None, optional
        The writer's log directory; defaults to ``<output_dir>/tensorboard``.
    **writer_kwargs
        Passed to ``SummaryWriter`` (``flush_secs``, ``comment``, ...).
    """

    def __init__(self, save_dir: str | None = None, **writer_kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except ImportError as exc:
            raise ImportError("TensorBoardLogger needs the tensorboard package: "
                              "pip install tensorboard") from exc
        self.save_dir = save_dir
        self._writer_kwargs = writer_kwargs
        self._writer = None

    def setup(self, output_dir: str) -> None:
        """Default :attr:`save_dir` to ``<output_dir>/tensorboard``."""
        if self.save_dir is None:
            self.save_dir = os.path.join(output_dir, "tensorboard")

    @property
    def experiment(self):
        """The ``SummaryWriter``, for logging histograms, images or text directly."""
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.save_dir, **self._writer_kwargs)
        return self._writer

    def log_hyperparams(self, params: dict) -> None:
        """Show ``params`` as YAML in TensorBoard's text tab."""
        import yaml
        text = yaml.safe_dump(params, sort_keys=False)
        self.experiment.add_text("hparams", "    " + text.replace("\n", "\n    "))

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Add one scalar per metric at ``step``."""
        for k, v in metrics.items():
            self.experiment.add_scalar(k, v, global_step=step)

    def finalize(self, status: str) -> None:
        """Flush and close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


#: Logger names accepted by ``cfg.loggers``.
LOGGERS = {"console": ConsoleLogger, "csv": CSVLogger, "tensorboard": TensorBoardLogger}


def build_loggers(spec) -> list[Logger]:
    """Turn a logger specification into a list of loggers.

    Parameters
    ----------
    spec : Logger, str, list of Logger or str, False or None
        A logger, a name from :data:`LOGGERS`, a list of either, or
        ``False`` / ``None`` / ``[]`` for none.

    Returns
    -------
    list of Logger
        The loggers, in the order given.
    """
    if spec is None or spec is False:
        return []
    if isinstance(spec, (Logger, str)):
        spec = [spec]
    out = []
    for item in spec:
        if isinstance(item, str):
            name = item.lower()
            if name not in LOGGERS:
                raise ValueError(f"unknown logger {item!r}; use one of {sorted(LOGGERS)} "
                                 "or a Logger instance")
            item = LOGGERS[name]()
        elif not isinstance(item, Logger):
            raise TypeError(f"a logger must be a Logger or a name, got {type(item).__name__}")
        out.append(item)
    return out
