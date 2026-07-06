"""Training loop: batch training, device selection, validation, checkpointing.

Keeps the moving parts explicit rather than hiding them in a framework, so the
pipeline is easy to follow and extend.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, random_split

from ..config import Config
from ..data import AtomicDataset, collate
from ..models import build_model, ForceStressOutput
from .losses import weighted_loss


def resolve_device(name: str) -> torch.device:
    """Resolve a device specification string into a ``torch.device``.

    Parameters
    ----------
    name : str
        Device name. The special value ``"auto"`` selects ``"cuda"`` when a
        CUDA device is available and falls back to ``"cpu"`` otherwise. Any
        other value is passed through to ``torch.device`` unchanged
        (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``).

    Returns
    -------
    torch.device
        The resolved device.
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    """Batch training loop with validation, scheduling and checkpointing.

    The trainer builds the model from configuration, wraps it in
    :class:`ForceStressOutput` (enabling force and/or stress heads only when
    the corresponding loss weight is positive), sets up the data loaders,
    optimizer and learning-rate scheduler, then runs the full training
    pipeline via :meth:`fit`.

    Parameters
    ----------
    cfg : Config
        Full run configuration. Fields used include ``device``, ``seed``,
        ``model`` (model config), ``optim`` (learning rate, weight decay,
        epochs, scheduler name and the energy/force/stress loss weights),
        ``data`` (batch size, number of workers, validation fraction) and
        ``output_dir`` (where checkpoints are written).
    train_set : AtomicDataset
        Training dataset of atomic structures.
    val_set : AtomicDataset or None, optional
        Explicit validation dataset. When ``None`` and
        ``cfg.data.val_fraction > 0``, a validation split is carved out of
        ``train_set`` using ``random_split`` seeded by ``cfg.seed``. When
        ``None`` and the fraction is zero, no validation is performed.
    test_set : AtomicDataset or None, optional
        Explicit held-out test dataset, evaluated once at the end of
        :meth:`fit`. When ``None`` and ``cfg.data.test_fraction > 0``, a test
        split is carved out of ``train_set`` (alongside the validation split,
        from the same seeded permutation). When ``None`` and the fraction is
        zero, no test evaluation is performed.

    Attributes
    ----------
    cfg : Config
        The configuration passed in.
    device : torch.device
        The resolved training device.
    model : ForceStressOutput
        The model wrapped with force/stress output heads, moved to ``device``.
    train_loader : torch.utils.data.DataLoader
        Shuffled loader over the training set. Batch training is simply
        ``batch_size > 1``; set the batch size to 1 to disable it.
    val_loader : torch.utils.data.DataLoader or None
        Non-shuffled loader over the validation set, or ``None`` if there is
        no validation set.
    test_loader : torch.utils.data.DataLoader or None
        Non-shuffled loader over the test set, or ``None`` if there is no
        test set.
    opt : torch.optim.Adam
        The Adam optimizer.
    sched : torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau or None
        The learning-rate scheduler, or ``None`` when disabled.
    """

    def __init__(self, cfg: Config, train_set: AtomicDataset,
                 val_set: AtomicDataset | None = None,
                 test_set: AtomicDataset | None = None):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

        torch.manual_seed(cfg.seed)

        base = build_model(cfg.model)
        self.model = ForceStressOutput(
            base,
            compute_forces=cfg.optim.force_weight > 0,
            compute_stress=cfg.optim.stress_weight > 0,
        ).to(self.device)

        # Carve val/test splits out of the training set for whichever of the
        # two was not given explicitly (a single seeded permutation, so the
        # train/val split is unchanged by adding a test fraction of zero).
        f_val = cfg.data.val_fraction if val_set is None else 0.0
        f_test = cfg.data.test_fraction if test_set is None else 0.0
        if f_val > 0 or f_test > 0:
            n = len(train_set)
            n_val = max(1, int(n * f_val)) if f_val > 0 else 0
            n_test = max(1, int(n * f_test)) if f_test > 0 else 0
            splits = random_split(
                train_set, [n - n_val - n_test, n_val, n_test],
                generator=torch.Generator().manual_seed(cfg.seed))
            train_set = splits[0]
            val_set = splits[1] if n_val else val_set
            test_set = splits[2] if n_test else test_set

        # batch training is just batch_size > 1; set to 1 to disable.
        def _loader(ds, shuffle):
            return DataLoader(ds, batch_size=cfg.data.batch_size,
                              shuffle=shuffle, collate_fn=collate,
                              num_workers=cfg.data.num_workers)
        self.train_loader = _loader(train_set, shuffle=True)
        self.val_loader = _loader(val_set, False) if val_set is not None else None
        self.test_loader = _loader(test_set, False) if test_set is not None else None

        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay)
        self.sched = self._make_scheduler(cfg.optim.scheduler)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _make_scheduler(self, name: str):
        """Construct the learning-rate scheduler named in the config.

        Parameters
        ----------
        name : str
            Scheduler identifier. ``"cosine"`` builds a
            ``CosineAnnealingLR`` over ``cfg.optim.epochs``; ``"plateau"``
            builds a ``ReduceLROnPlateau`` with ``patience=10``. Any other
            value disables scheduling.

        Returns
        -------
        torch.optim.lr_scheduler.CosineAnnealingLR or torch.optim.lr_scheduler.ReduceLROnPlateau or None
            The scheduler instance, or ``None`` when scheduling is disabled.
        """
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt, T_max=self.cfg.optim.epochs)
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=10)
        return None

    def _step(self, data, train: bool):
        """Run one forward/loss step on a batch, optionally back-propagating.

        Moves the batch to the training device, runs the model forward,
        computes the weighted energy/force/stress loss, and (when ``train``)
        performs a single optimizer step.

        Parameters
        ----------
        data : AtomicGraph
            A (possibly batched) atomic graph produced by the collate
            function.
        train : bool
            When ``True``, zero the gradients, back-propagate the loss and
            step the optimizer. When ``False`` (validation), only the forward
            pass and loss computation are performed; note that gradients are
            still enabled because force predictions require them.

        Returns
        -------
        dict of str to float
            The scalar loss logs for this batch, as returned by
            :func:`weighted_loss` (e.g. ``"loss"`` plus any of
            ``"energy_mse"``, ``"force_mse"``, ``"stress_mse"``).
        """
        data = data.to(self.device)
        o = self.cfg.optim
        pred = self.model(data)
        loss, logs = weighted_loss(
            pred, data, o.energy_weight, o.force_weight, o.stress_weight)
        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return logs

    def fit(self):
        """Run the full training loop over ``cfg.optim.epochs`` epochs.

        For each epoch the model is trained over the whole training loader and,
        if a validation loader exists, evaluated over it. The best validation
        loss seen so far is tracked and its model saved to ``best.pt`` in the
        output directory. The scheduler is stepped every epoch: a
        ``ReduceLROnPlateau`` scheduler is stepped with the current loss, while
        any other scheduler is stepped without arguments. A per-epoch summary
        is printed. After the final epoch the model is saved to ``last.pt``
        and, if a test loader exists, evaluated once on the test set (with the
        final-epoch weights; to test the best checkpoint instead, load
        ``best.pt`` into ``self.model`` and call :meth:`evaluate`).

        Returns
        -------
        dict
            Final metrics: ``{"train": ..., "val": ..., "test": ...}``, each a
            dict of averaged losses (empty when that split does not exist).
        """
        best = float("inf")
        tr, va = {}, {}
        for epoch in range(self.cfg.optim.epochs):
            self.model.train()
            tr = self._avg(self._step(d, True) for d in self.train_loader)

            va = {}
            if self.val_loader is not None:
                self.model.eval()
                # forces need grad even at eval -> no torch.no_grad()
                va = self._avg(self._step(d, False) for d in self.val_loader)
                metric = va.get("loss", tr["loss"])
                if metric < best:
                    best = metric
                    self.save(os.path.join(self.cfg.output_dir, "best.pt"))

            if isinstance(self.sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.sched.step(va.get("loss", tr["loss"]))
            elif self.sched is not None:
                self.sched.step()

            self._log(epoch, tr, va)
        self.save(os.path.join(self.cfg.output_dir, "last.pt"))

        te = {}
        if self.test_loader is not None:
            te = self.evaluate()
            print(f"test loss {te.get('loss', 0):.4e}")
        return {"train": tr, "val": va, "test": te}

    def evaluate(self, loader=None):
        """Evaluate the current model over a data loader without training.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader or None, optional
            Loader to evaluate over. Defaults to ``self.test_loader``.

        Returns
        -------
        dict of str to float
            Averaged losses over the loader (as in :meth:`_step`), or an empty
            dict when there is no loader.
        """
        loader = self.test_loader if loader is None else loader
        if loader is None:
            return {}
        self.model.eval()
        # forces need grad even at eval -> no torch.no_grad()
        return self._avg(self._step(d, False) for d in loader)

    @staticmethod
    def _avg(logs_iter):
        """Average a sequence of per-batch log dictionaries.

        Parameters
        ----------
        logs_iter : iterable of dict of str to float
            An iterable yielding per-batch log dictionaries (as returned by
            :meth:`_step`). Keys need not be present in every dictionary.

        Returns
        -------
        dict of str to float
            Each key mapped to the mean of its values across the batches. An
            empty iterable yields an empty dictionary (division guarded so an
            empty iterable does not raise).
        """
        agg, n = {}, 0
        for logs in logs_iter:
            n += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
        return {k: v / max(n, 1) for k, v in agg.items()}

    @staticmethod
    def _log(epoch, tr, va):
        """Print a one-line summary of an epoch's train and validation loss.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        tr : dict of str to float
            Averaged training logs; its ``"loss"`` entry is reported.
        va : dict of str to float
            Averaged validation logs. When non-empty, its ``"loss"`` entry is
            appended to the message; when empty, only training loss is shown.

        Returns
        -------
        None
        """
        msg = f"epoch {epoch:4d} | train loss {tr.get('loss', 0):.4e}"
        if va:
            msg += f" | val loss {va.get('loss', 0):.4e}"
        print(msg)

    def save(self, path: str):
        """Serialize the model state dict and configuration to disk.

        Parameters
        ----------
        path : str
            Destination file path. The saved checkpoint is a dictionary with
            keys ``"model"`` (the model ``state_dict``) and ``"cfg"`` (the
            :class:`Config` used for the run).

        Returns
        -------
        None
        """
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, path)
