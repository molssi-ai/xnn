"""Training loop: batch training, device selection, validation, checkpointing.

Keeps the moving parts explicit rather than hiding them in a framework, so the
pipeline is easy to follow and extend.

Multi-GPU / multi-node data parallelism uses native PyTorch DDP and is driven
purely by the environment: when the process was spawned by a distributed
launcher that sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` (``torchrun
--nproc-per-node N -m xnn train ...``, Slurm + torchrun, or an
``accelerate launch`` configured for multi-GPU), the trainer initializes the
process group, shards the data loaders with ``DistributedSampler``, wraps the
model in ``DistributedDataParallel``, all-reduces the logged metrics, and
writes checkpoints from rank 0 only. A plain ``python`` / ``xnn`` invocation
runs the unchanged single-process pipeline.

Sharded strategies (FSDP, DeepSpeed ZeRO) are deliberately not used: force and
stress losses back-propagate through gradients taken with
``create_graph=True`` (see ``ForceStressOutput``), a double backward that DDP
supports but sharded wrappers do not.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, random_split

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

    When spawned by a distributed launcher (``torchrun -m xnn train ...`` on
    one or many nodes; see :meth:`_init_distributed`) the same pipeline runs
    data-parallel: the model is wrapped in ``DistributedDataParallel``, each
    loader is sharded with a ``DistributedSampler``, metrics are all-reduced
    so every rank sees global averages, and only rank 0 logs and writes
    checkpoints.

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
    distributed : bool
        Whether this process is part of a distributed launch (``WORLD_SIZE``
        in the environment is greater than one).
    rank : int
        This process's global rank; 0 in a single-process run.
    is_main : bool
        Whether this is rank 0, the only rank that logs and saves.
    device : torch.device
        The resolved training device (``cuda:LOCAL_RANK`` per rank when
        distributed on GPUs).
    model : ForceStressOutput or DistributedDataParallel
        The model wrapped with force/stress output heads, moved to ``device``
        (and wrapped in DDP when distributed); :attr:`module` always gives the
        bare :class:`ForceStressOutput`.
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
        self.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.device = self._init_distributed(resolve_device(cfg.device))
        self.rank = dist.get_rank() if self.distributed else 0
        self.is_main = self.rank == 0

        torch.manual_seed(cfg.seed)

        base = build_model(cfg.model)
        self.model = ForceStressOutput(
            base,
            compute_forces=cfg.optim.force_weight > 0,
            compute_stress=cfg.optim.stress_weight > 0,
        ).to(self.device)
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=(
                    [self.device.index] if self.device.type == "cuda" else None))

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
        # Distributed runs shard every loader across ranks; the sampler then
        # owns the shuffling (the two DataLoader options are exclusive).
        def _loader(ds, shuffle):
            sampler = (DistributedSampler(ds, shuffle=shuffle, seed=cfg.seed)
                       if self.distributed else None)
            return DataLoader(ds, batch_size=cfg.data.batch_size,
                              shuffle=shuffle and sampler is None,
                              sampler=sampler, collate_fn=collate,
                              num_workers=cfg.data.num_workers)
        self.train_loader = _loader(train_set, shuffle=True)
        self.val_loader = _loader(val_set, False) if val_set is not None else None
        self.test_loader = _loader(test_set, False) if test_set is not None else None

        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay)
        self.sched = self._make_scheduler(cfg.optim.scheduler)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _init_distributed(self, dev: torch.device) -> torch.device:
        """Join the process group of a distributed launcher, if there is one.

        A launcher such as ``torchrun`` (or ``accelerate launch``) exports
        ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` into every process it
        spawns; ``self.distributed`` reflects whether that happened. In a
        distributed run each rank is pinned to one CUDA device selected by
        ``LOCAL_RANK`` (the configured device only chooses cpu vs cuda), and
        the process group is initialized with the matching backend -- NCCL on
        GPUs, Gloo on CPUs.

        Parameters
        ----------
        dev : torch.device
            The device resolved from the configuration.

        Returns
        -------
        torch.device
            The device this rank should train on: ``dev`` unchanged in a
            single-process run, ``cuda:LOCAL_RANK`` (or ``dev`` on CPU) in a
            distributed one.
        """
        self._owns_pg = False
        if not self.distributed:
            return dev
        if dev.type == "cuda":
            dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            torch.cuda.set_device(dev)
        if not dist.is_initialized():
            dist.init_process_group("nccl" if dev.type == "cuda" else "gloo")
            self._owns_pg = True
        return dev

    @property
    def module(self) -> ForceStressOutput:
        """The bare model, unwrapped from ``DistributedDataParallel`` if any."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

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
        # Evaluation steps run the bare module: they are not followed by a
        # backward pass, so the DDP wrapper's gradient sync must not be armed.
        model = self.model if train else self.module
        pred = model(data)
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
            if self.distributed:
                # reseed the sampler so each epoch shuffles differently
                self.train_loader.sampler.set_epoch(epoch)
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

            if self.is_main:
                self._log(epoch, tr, va)
        self.save(os.path.join(self.cfg.output_dir, "last.pt"))

        te = {}
        if self.test_loader is not None:
            te = self.evaluate()
            if self.is_main:
                print(f"test loss {te.get('loss', 0):.4e}")
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()
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

    def _avg(self, logs_iter):
        """Average a sequence of per-batch log dictionaries.

        In a distributed run the per-rank sums and batch counts are further
        summed over all ranks with an all-reduce, so every rank returns the
        same global averages -- the best-checkpoint decision and the plateau
        scheduler then stay in lockstep across ranks.

        Parameters
        ----------
        logs_iter : iterable of dict of str to float
            An iterable yielding per-batch log dictionaries (as returned by
            :meth:`_step`). Keys need not be present in every dictionary.

        Returns
        -------
        dict of str to float
            Each key mapped to the mean of its values across the batches (and
            across ranks when distributed). An empty iterable yields an empty
            dictionary (division guarded so an empty iterable does not raise).
        """
        agg, n = {}, 0
        for logs in logs_iter:
            n += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
        if self.distributed and dist.is_initialized():
            keys = sorted(agg)
            # NCCL reduces on the rank's GPU; Gloo reduces on CPU.
            t = torch.tensor(
                [float(n)] + [agg[k] for k in keys], dtype=torch.float64,
                device=self.device if self.device.type == "cuda" else "cpu")
            dist.all_reduce(t)
            n = t[0].item()
            agg = {k: t[i + 1].item() for i, k in enumerate(keys)}
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

        Only rank 0 writes (in a single-process run that is the only rank);
        the state dict is taken from the bare module, so checkpoint keys are
        identical with and without DDP. A barrier keeps the other ranks from
        racing ahead of the write.

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
        if self.is_main:
            torch.save({"model": self.module.state_dict(), "cfg": self.cfg}, path)
        if self.distributed and dist.is_initialized():
            dist.barrier()
