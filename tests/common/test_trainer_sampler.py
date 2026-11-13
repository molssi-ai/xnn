"""Shuffling of the trainer's loaders and the custom batch-sampler hook."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from xnn.common.config import from_dict
from xnn.common.data import AtomicDataset, collate
from xnn.common.train import Trainer
from xnn.common.train.trainer import _set_epoch

CUTOFF = 3.0


def _dataset(n, start=0):
    rng = np.random.default_rng(start)
    structs = [{"pos": rng.uniform(0.0, 3.0, (3, 3)), "atomic_numbers": [1, 8, 1],
                "energy": float(start + i), "forces": rng.normal(size=(3, 3))}
               for i in range(n)]
    return AtomicDataset(structs, CUTOFF)


def _cfg(tmp_path, epochs=1):
    return from_dict({"model": {"name": "schnet", "n_features": 8, "n_interactions": 1,
                                "cutoff": CUTOFF},
                      "optim": {"epochs": epochs},
                      "data": {"batch_size": 3, "val_fraction": 0.0},
                      "device": "cpu", "output_dir": str(tmp_path)})


def _order(loader):
    return [int(e) for d in loader for e in d.energy.round().tolist()]


class _Batches:
    """A batch sampler that reshuffles from its epoch and records what it yields."""

    def __init__(self, n, size):
        self.n, self.size, self.epoch = n, size, 0
        self.told, self.yielded = [], []

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.told.append(epoch)

    def _batches(self):
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.epoch))
        return [perm[i:i + self.size].tolist() for i in range(0, self.n, self.size)]

    def __iter__(self):
        batches = self._batches()
        self.yielded.append([i for b in batches for i in b])
        return iter(batches)

    def __len__(self):
        return (self.n + self.size - 1) // self.size


def test_builtin_loader_reshuffles_every_epoch(tmp_path):
    """The single-process training loader visits every structure in a new order each epoch."""
    trainer = Trainer(_cfg(tmp_path), _dataset(12))
    first, second = _order(trainer.train_loader), _order(trainer.train_loader)
    assert sorted(first) == sorted(second) == list(range(12))
    assert first != second


def test_fit_reshuffles_a_custom_batch_sampler(tmp_path):
    """fit() tells a custom batch sampler every epoch and trains on its batches."""
    ds = _dataset(10)
    sampler = _Batches(len(ds), 3)
    trainer = Trainer(_cfg(tmp_path, epochs=3), ds, train_batch_sampler=sampler)
    assert trainer.train_loader.batch_sampler is sampler
    seen = []
    step = trainer._step

    def record(data, train):
        if train:
            seen.extend(int(e) for e in data.energy.round().tolist())
        return step(data, train)

    trainer._step = record
    trainer.fit()
    assert sampler.told == [0, 1, 2]
    assert seen == [i for order in sampler.yielded for i in order]
    assert len({tuple(order) for order in sampler.yielded}) == 3


def test_val_and_test_batch_samplers_are_used(tmp_path):
    """The validation and test loaders batch as their samplers say."""
    val, test = _dataset(4, 100), _dataset(5, 200)
    trainer = Trainer(_cfg(tmp_path), _dataset(6), val, test,
                      val_batch_sampler=[[3, 0, 1], [2]], test_batch_sampler=[[4], [0, 1, 2, 3]])
    assert [d.energy.round().tolist() for d in trainer.val_loader] == [[103, 100, 101], [102]]
    assert [len(d.energy) for d in trainer.test_loader] == [1, 4]


def test_batch_samplers_refuse_renumbered_datasets(tmp_path):
    """A batch sampler is refused with a carved split or a replay set."""
    cfg = _cfg(tmp_path)
    ds = _dataset(6)
    cfg.data.val_fraction = 0.25
    with pytest.raises(ValueError, match="train_batch_sampler"):
        Trainer(cfg, ds, train_batch_sampler=_Batches(6, 3))
    with pytest.raises(ValueError, match="val_batch_sampler"):
        Trainer(cfg, ds, val_batch_sampler=[[0]])
    cfg.data.val_fraction = 0.0
    with pytest.raises(ValueError, match="train_batch_sampler"):
        Trainer(cfg, ds, replay_set=_dataset(2), train_batch_sampler=_Batches(6, 3))
    with pytest.raises(ValueError, match="test_batch_sampler"):
        Trainer(cfg, ds, test_batch_sampler=[[0]])


def test_set_epoch_reaches_wrapped_and_plain_samplers():
    """_set_epoch tells a batch sampler, a wrapper's sampler, and each object once."""
    batches = _Batches(6, 3)
    _set_epoch(DataLoader(_dataset(6), batch_sampler=batches, collate_fn=collate), 4)
    assert batches.told == [4]

    class Wrapper:
        def __init__(self, s):
            self.sampler = self.batch_sampler = s

    _set_epoch(Wrapper(batches), 5)
    assert batches.told == [4, 5]
    _set_epoch(DataLoader(_dataset(6), batch_size=2, shuffle=True, collate_fn=collate), 1)
