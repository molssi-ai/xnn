"""Training loop: batch training, device selection, validation, checkpointing.

Keeps the moving parts explicit rather than hiding them in a framework, so the
pipeline is easy to follow and extend.

Multi-GPU / multi-node data parallelism uses native PyTorch DDP and is driven
purely by the environment: when the process was spawned by a distributed
launcher that sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` (``torchrun
--nproc-per-node N -m xnn train ...``, Slurm + torchrun, or any launcher
that exports the same variables), the trainer initializes the
process group, shards the data loaders with ``DistributedSampler``, wraps the
model in ``DistributedDataParallel``, all-reduces the logged metrics, and
writes checkpoints from rank 0 only. A plain ``python`` / ``xnn`` invocation
runs the unchanged single-process pipeline.

Sharded strategies (FSDP, DeepSpeed ZeRO) are deliberately not used: force and
stress losses back-propagate through gradients taken with
``create_graph=True`` (see ``ForceStressOutput``), a double backward that DDP
supports but sharded wrappers do not.

Fine-tuning a pretrained model is the same loop: ``model.extra["pretrained"]``
starts from any hub model, ``extra["lora"]`` adapts it with low-rank
updates, ``extra["heads"]`` plus a replay set trains a second readout on the
pretraining distribution, and ``atomic_energies: estimated`` re-sets the
per-element references from the training data (see
:mod:`xnn.common.finetune`).
"""
from __future__ import annotations

import contextlib
import os
import warnings

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, random_split

from .. import distributed
from ..config import Config
from ..data import AtomicDataset, collate
from ..models import ForceStressOutput
from ..models.registry import prepare_model
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


def _check_charge_labels(base, train_set) -> None:
    """Warn when a LatentEwald net-charge constraint will never see a charge label.

    With ``constrain_charge`` on, a structure without ``total_charge`` is
    pinned to neutrality. A training set in which no structure carries the
    key at all is either genuinely neutral, in which case the warning is
    harmless, or unlabeled, in which case every charged structure would be
    fitted wrong, silently; this makes it visible once, at start.
    """
    from ..models.les import LatentEwald
    les = [m for m in base.modules() if isinstance(m, LatentEwald)]
    structures = getattr(train_set, "structures", None)
    if not les or not structures:
        return
    if (any(m.constrain_charge or m.charge_solve for m in les)
            and not any(s.get("total_charge", s.get("charge")) is not None for s in structures)):
        warnings.warn(
            "the model ties its latent charges to each structure's total_charge, "
            "but no training structure carries a 'total_charge' (or 'charge') label: all "
            f"{len(structures)} are treated as neutral. Label charged structures, or turn "
            "constrain_charge / charge_solve off.", stacklevel=3)
    if (any(m.fragments for m in les)
            and not any(s.get("fragment_charges") is not None for s in structures)):
        warnings.warn(
            "the charge solve constrains every fragment, but no training structure carries "
            "a 'fragment_charges' label: fragment charges come from the ion_charges table "
            "alone (monatomic ions), every other fragment is neutral.", stacklevel=3)


def _structures(dataset, what: str) -> list[dict]:
    """The structure dicts behind a dataset (an ``AtomicDataset`` or a list of dicts)."""
    if isinstance(dataset, AtomicDataset):
        return dataset.structures
    if isinstance(dataset, (list, tuple)):
        return list(dataset)
    raise TypeError(f"{what} must be an AtomicDataset (or a list of structure dicts) for "
                    f"multi-head training, got {type(dataset).__name__}")


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

    Fine-tuning options of the config are applied while building: the model
    section's ``pretrained`` / ``lora`` / ``heads`` entries (see
    :func:`~xnn.common.models.registry.prepare_model`), the reference-energy
    markers ``atomic_energies: estimated`` / ``average`` (resolved on the
    training structures and written back into ``cfg.model`` as numbers),
    ``optim.freeze`` / ``optim.train_only`` and, for a multi-head model, the
    head labels of the datasets and the replay set.

    Parameters
    ----------
    cfg : Config
        Full run configuration. Fields used include ``device``, ``seed``,
        ``model`` (model config), ``optim`` (learning rate, weight decay,
        epochs, scheduler name and the energy/force/stress loss weights),
        ``data`` (batch size, number of workers, validation fraction) and
        ``output_dir`` (where checkpoints are written). ``cfg.model`` is
        replaced by the resolved model config (a ``pretrained`` source is
        expanded into the architecture it holds), so the saved checkpoints
        rebuild the model from their weights alone.
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
    replay_set : AtomicDataset or None, optional
        Structures of the pretraining distribution for the replay head of a
        multi-head model (``cfg.data.replay_head``). They are element-filtered
        and subsampled as ``cfg.data.replay_filter`` / ``replay_samples`` say,
        relabelled by the pretrained model when ``cfg.data.replay_pseudolabel``
        is set, split by ``val_fraction`` like the training set, and trained
        alongside it. Requires ``model.extra["heads"]``.

    Attributes
    ----------
    cfg : Config
        The configuration passed in, with ``model`` resolved.
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
    heads : list of str or None
        The head names of a multi-head model, ``None`` otherwise.
    ema : torch.optim.swa_utils.AveragedModel or None
        The exponential moving average of the weights when
        ``cfg.optim.ema_decay > 0``; :attr:`eval_module` is what validation,
        testing and the checkpoints use.
    train_loader : torch.utils.data.DataLoader
        Shuffled loader over the training set. Batch training is simply
        ``batch_size > 1``; set the batch size to 1 to disable it.
    val_loader : torch.utils.data.DataLoader or None
        Non-shuffled loader over the validation set, or ``None`` if there is
        no validation set.
    test_loader : torch.utils.data.DataLoader or None
        Non-shuffled loader over the test set, or ``None`` if there is no
        test set.
    opt : torch.optim.Adam or torch.optim.AdamW
        The optimizer over the trainable parameters.
    sched : torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau or None
        The learning-rate scheduler, or ``None`` when disabled.
    """

    def __init__(self, cfg: Config, train_set: AtomicDataset,
                 val_set: AtomicDataset | None = None,
                 test_set: AtomicDataset | None = None,
                 replay_set: AtomicDataset | None = None):
        from ..finetune.freeze import freeze_parameters
        from ..finetune.heads import find_multihead, label_head
        from ..finetune.reference import reference_markers

        self.cfg = cfg
        self.distributed = distributed.is_distributed()
        self.device = self._init_distributed(resolve_device(cfg.device))
        self.rank = distributed.rank()
        self.is_main = self.rank == 0

        torch.manual_seed(cfg.seed)

        # reference-energy markers need data: they are resolved below, after
        # the model exists, and the config then records the numbers
        model_cfg, markers = reference_markers(cfg.model)
        base, cfg.model = prepare_model(model_cfg)
        # A wrapper such as the D4 dispersion correction can need a larger
        # neighbor list than the core model's cutoff that the config carries;
        # graphs are built lazily, so widening the datasets' radius here (before
        # any graph exists) keeps data and model consistent.
        for ds in (train_set, val_set, test_set, replay_set):
            model_cutoff = getattr(base, "cutoff", None)
            if (ds is not None and model_cutoff is not None
                    and hasattr(ds, "cutoff") and model_cutoff > ds.cutoff):
                ds.cutoff = float(model_cutoff)
                if hasattr(ds, "_cache"):
                    ds._cache.clear()

        o = cfg.optim
        self._head_weights = {}
        multi = find_multihead(base)
        self.heads = list(multi.heads) if multi is not None else None
        if self.heads:
            self._head_weights = self._resolve_head_weights(multi)
        force_on = o.force_weight > 0 or any(w[1] > 0 for w in self._head_weights.values())
        stress_on = o.stress_weight > 0 or any(w[2] > 0 for w in self._head_weights.values())
        self.model = ForceStressOutput(base, compute_forces=force_on,
                                       compute_stress=stress_on).to(self.device)
        if self.is_main:
            _check_charge_labels(base, train_set)

        # multi-head: the target sets belong to one head, the replay set to another
        replay_train = replay_val = None
        if self.heads:
            target = self._target_head(multi)
            for ds in (train_set, val_set, test_set):
                if ds is not None:
                    label_head(_structures(ds, "the dataset"), multi.index(target))
            if replay_set is not None:
                replay_train, replay_val = self._prepare_replay(multi, train_set, replay_set)
        elif replay_set is not None:
            raise ValueError("a replay set needs a multi-head model: set model.heads "
                             "(for example [pt_head, Default])")

        self._resolve_reference_energies(base, multi, markers, train_set)

        if o.freeze or o.train_only:
            n_train, n_total = freeze_parameters(self.module, o.freeze, o.train_only)
            if self.is_main:
                print(f"trainable parameters: {n_train} of {n_total}")

        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=(
                    [self.device.index] if self.device.type == "cuda" else None),
                # a batch without one head leaves that head's parameters unused
                find_unused_parameters=bool(self.heads and len(self.heads) > 1))

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
        if replay_train is not None:
            train_set = ConcatDataset([train_set, replay_train])
            if replay_val is not None:
                val_set = replay_val if val_set is None else ConcatDataset([val_set, replay_val])

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

        self._params = [p for p in self.model.parameters() if p.requires_grad]
        if not self._params:
            raise ValueError("no trainable parameters: every parameter is frozen")
        optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[o.optimizer.lower()]
        self.opt = optimizer(self._params, lr=o.lr, weight_decay=o.weight_decay)
        self.sched = self._make_scheduler(o.scheduler)
        self.ema = None
        if o.ema_decay > 0:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            if not 0.0 < o.ema_decay < 1.0:
                raise ValueError(f"optim.ema_decay must lie in (0, 1), got {o.ema_decay}")
            self.ema = AveragedModel(self.module, multi_avg_fn=get_ema_multi_avg_fn(o.ema_decay),
                                     use_buffers=False)
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _init_distributed(self, dev: torch.device) -> torch.device:
        """Join the process group of a distributed launcher, if there is one.

        A launcher such as ``torchrun`` exports
        ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` into every process it
        spawns; ``self.distributed`` reflects whether that happened. In a
        distributed run each rank is pinned to one CUDA device selected by
        ``LOCAL_RANK`` (the configured device only chooses cpu vs cuda), and
        the process group is initialized with the matching backend -- NCCL on
        GPUs, Gloo on CPUs. The work is :func:`xnn.common.distributed.init_process_group`,
        shared with the benchmark runner.

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
        dev, self._owns_pg = distributed.init_process_group(dev)
        return dev

    @property
    def module(self) -> ForceStressOutput:
        """The bare model, unwrapped from ``DistributedDataParallel`` if any."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    @property
    def eval_module(self) -> ForceStressOutput:
        """The weights used for validation, testing and checkpoints.

        The exponential moving average when ``cfg.optim.ema_decay > 0``
        (its buffers synced from the training model), otherwise
        :attr:`module`.
        """
        if self.ema is None:
            return self.module
        with torch.no_grad():
            for avg, cur in zip(self.ema.module.buffers(), self.module.buffers()):
                if avg.shape == cur.shape:
                    avg.copy_(cur)
        return self.ema.module

    def _target_head(self, multi) -> str:
        """The head the training structures belong to (``cfg.data.head`` or the first non-replay one)."""
        if self.cfg.data.head is not None:
            return self.cfg.data.head
        for name in multi.heads:
            if name != self.cfg.data.replay_head:
                return name
        return multi.heads[0]

    def _resolve_head_weights(self, multi) -> dict[int, tuple[float, float, float]]:
        """``cfg.optim.head_weights`` as ``{head_index: (energy, force, stress)}``."""
        o = self.cfg.optim
        out = {}
        for name, spec in (o.head_weights or {}).items():
            spec = dict(spec or {})
            unknown = sorted(set(spec) - {"energy_weight", "force_weight", "stress_weight"})
            if unknown:
                raise ValueError(f"optim.head_weights[{name!r}]: unknown key(s) {unknown}")
            out[multi.index(str(name))] = (
                float(spec.get("energy_weight", o.energy_weight)),
                float(spec.get("force_weight", o.force_weight)),
                float(spec.get("stress_weight", o.stress_weight)))
        return out

    def _prepare_replay(self, multi, train_set, replay_set):
        """Filter, subsample, (pseudo)label and split the replay set.

        Returns
        -------
        tuple
            ``(replay_train, replay_val)`` datasets labelled with the replay
            head (``replay_val`` is ``None`` without a validation fraction).
        """
        from ..finetune.heads import label_head
        from ..finetune.reference import species_of
        from ..finetune.replay import pseudolabel, select_replay

        d = self.cfg.data
        structures = _structures(replay_set, "replay_set")
        species = species_of(_structures(train_set, "train_set")) if d.replay_filter != "none" else None
        selected = select_replay(structures, species, n=d.replay_samples, mode=d.replay_filter,
                                 seed=self.cfg.seed)
        if not selected:
            raise ValueError("the replay set is empty after the element filter "
                             f"({d.replay_filter}); use replay_filter: none or another set")
        if d.replay_pseudolabel:
            selected = pseudolabel(multi, selected, batch_size=d.batch_size, device=self.device,
                                   head=d.replay_head)
        label_head(selected, multi.index(d.replay_head))
        if self.is_main:
            print(f"replay head {d.replay_head!r}: {len(selected)} structures"
                  f"{' (pseudolabelled)' if d.replay_pseudolabel else ''}")
        cutoff = float(getattr(replay_set, "cutoff", self.cfg.data.cutoff))
        ds = AtomicDataset(selected, cutoff)
        if d.val_fraction > 0 and len(ds) > 1:
            n_val = max(1, int(len(ds) * d.val_fraction))
            train, val = random_split(ds, [len(ds) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(self.cfg.seed + 1))
            return train, val
        return ds, None

    def _resolve_reference_energies(self, base, multi, markers, train_set) -> None:
        """Estimate the reference energies the config marked, set them, record them."""
        if not markers:
            return
        from ..finetune.heads import head_options
        from ..finetune.reference import (average_atomic_energies, estimate_atomic_energies,
                                          get_atomic_energies, set_atomic_energies)

        structures = _structures(train_set, "train_set")
        extra = dict(self.cfg.model.extra or {})
        # the recorded values cover every species of the model (a per-species
        # list in the config must be complete), the estimate only those present
        def complete(single, e0):
            full = get_atomic_energies(single)
            full.update(e0)
            return full

        heads_spec = extra.get("heads")
        if multi is not None and isinstance(heads_spec, (list, tuple)):
            heads_spec = {name: {} for name in heads_spec}
        for head, marker in markers:
            if multi is not None:
                head = head if head is not None else self._target_head(multi)
                target_structures = [s for s in structures
                                     if int(s.get("head", 0)) == multi.index(head)]
                scope = multi.using(head)
            else:
                target_structures = structures
                scope = contextlib.nullcontext(base)
            with scope as single:
                if marker == "estimated":
                    e0 = estimate_atomic_energies(single, target_structures,
                                                  batch_size=self.cfg.data.batch_size,
                                                  device=self.device)
                else:
                    e0 = average_atomic_energies(target_structures)
                set_atomic_energies(single, e0)
                recorded = complete(single, e0)
            if self.is_main:
                where = f"head {head!r}" if multi is not None else "the model"
                print(f"{marker} reference energies for {where}: "
                      + ", ".join(f"{z}: {v:.4f}" for z, v in e0.items()))
            if multi is not None:
                heads_spec[head] = {**head_options(heads_spec, head), "atomic_energies": recorded}
            else:
                extra["atomic_energies"] = recorded
        if multi is not None:
            extra["heads"] = heads_spec
        self.cfg.model.extra = extra

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
        performs a single optimizer step (with gradient clipping and the EMA
        update when configured).

        Parameters
        ----------
        data : AtomicGraph
            A (possibly batched) atomic graph produced by the collate
            function.
        train : bool
            When ``True``, zero the gradients, back-propagate the loss and
            step the optimizer. When ``False`` (validation), only the forward
            pass and loss computation are performed on :attr:`eval_module`;
            note that gradients are still enabled because force predictions
            require them.

        Returns
        -------
        dict of str to float
            The scalar loss logs for this batch, as returned by
            :func:`weighted_loss` (e.g. ``"loss"`` plus any of
            ``"energy_mse"``, ``"force_mse"``, ``"stress_mse"``, per head
            for a multi-head batch).
        """
        data = data.to(self.device)
        o = self.cfg.optim
        # Evaluation steps run the bare module (the EMA weights when kept):
        # they are not followed by a backward pass, so the DDP wrapper's
        # gradient sync must not be armed.
        model = self.model if train else self.eval_module
        pred = model(data)
        loss, logs = weighted_loss(
            pred, data, o.energy_weight, o.force_weight, o.stress_weight,
            huber_delta=o.huber_delta,
            huber_delta_energy=o.huber_delta_energy,
            huber_delta_forces=o.huber_delta_forces,
            huber_delta_stress=o.huber_delta_stress,
            head_weights=self._head_weights or None, head_names=self.heads)
        if train:
            self.opt.zero_grad()
            loss.backward()
            if o.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self._params, o.clip_grad)
            self.opt.step()
            if self.ema is not None:
                self.ema.update_parameters(self.module)
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
        distributed.destroy_process_group(self._owns_pg)
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
        if self.distributed:
            keys = sorted(agg)
            n, *sums = distributed.all_reduce_sum(
                [float(n)] + [agg[k] for k in keys], self.device)
            agg = dict(zip(keys, sums))
        return {k: v / max(n, 1) for k, v in agg.items()}

    @staticmethod
    def _log(epoch, tr, va):
        """Print a one-line summary of an epoch's train and validation loss.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        tr : dict of str to float
            Averaged training logs; its ``"loss"`` entry is reported, plus the
            per-head losses of a multi-head run.
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
        heads = [k[:-5] for k in sorted(tr) if k.endswith("/loss")]
        if heads:
            msg += " | " + " ".join(f"{h} {tr[h + '/loss']:.3e}" for h in heads)
        print(msg)

    def save(self, path: str):
        """Serialize the model state dict and configuration to disk.

        Only rank 0 writes (in a single-process run that is the only rank);
        the state dict is taken from the bare module (the EMA weights when
        kept), so checkpoint keys are identical with and without DDP. A
        barrier keeps the other ranks from racing ahead of the write.

        Parameters
        ----------
        path : str
            Destination file path. The saved checkpoint is a dictionary with
            keys ``"model"`` (the model ``state_dict``) and ``"cfg"`` (the
            :class:`Config` used for the run, with the resolved model
            section).

        Returns
        -------
        None
        """
        if self.is_main:
            torch.save({"model": self.eval_module.state_dict(), "cfg": self.cfg}, path)
        distributed.barrier()

    def save_pretrained(self, save_directory: str, **card_fields):
        """Write the current model as a portable model directory.

        The layout :func:`~xnn.common.models.hub.from_pretrained` loads and
        the model hub caches (``card.json``, ``config.yaml``, ``model.pt``),
        with no pickled objects, so the directory can be shared, uploaded to
        Zenodo, or copied to another machine. Only rank 0 writes. A
        multi-head or LoRA model is written as such (its config rebuilds it);
        ``from_pretrained`` then serves one head with the LoRA updates folded
        in.

        Parameters
        ----------
        save_directory : str
            Output directory.
        **card_fields
            Model card fields (``name``, ``description``, ``license``,
            ``citation``, ...); see :class:`~xnn.common.models.hub.ModelCard`.

        Returns
        -------
        pathlib.Path or None
            The directory on rank 0, ``None`` on the other ranks.
        """
        from ..models.hub import save_pretrained
        out = None
        if self.is_main:
            out = save_pretrained(self.eval_module, save_directory, config=self.cfg, **card_fields)
        distributed.barrier()
        return out
