"""Unit tests of the trainer's float64 weight EMA and the optimizer choices."""
import torch
from torch import nn

from xnn.common.train.trainer import _EMA


class _Two(nn.Module):
    def __init__(self):
        super().__init__()
        self.big = nn.Parameter(torch.tensor([-616.0, -205.0], dtype=torch.float32))
        self.small = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))


def _drift(ema, model, steps, step):
    with torch.no_grad():
        for _ in range(steps):
            model.big += step
            model.small += step
            ema.update_parameters(model)


def test_ema_tracks_large_parameters_in_float32():
    model = _Two()
    ema = _EMA(model, 0.999)
    _drift(ema, model, 1, 0.0)                     # first update copies
    assert torch.equal(ema.module.big, model.big)
    start = {n: getattr(model, n).detach().clone() for n in ("big", "small")}
    _drift(ema, model, 10_000, 1e-3)               # a drift of about 10 per parameter
    for name in ("big", "small"):
        avg, cur = getattr(ema.module, name), getattr(model, name)
        assert avg.dtype == torch.float32
        # the EMA of a linear drift lags by step * d / (1 - d); the step is measured, since
        # the float32 parameter itself accumulates its 1e-3 increments inexactly at 616
        step = (cur - start[name]) / 10_000
        assert torch.allclose(cur - avg, step * 0.999 / 0.001, atol=2e-3), name
        assert float((cur - start[name]).abs().min()) > 9.0


def test_ema_first_update_copies_and_decays():
    model = _Two()
    ema = _EMA(model, 0.9)
    _drift(ema, model, 1, 0.0)
    with torch.no_grad():
        model.small.fill_(1.5)
    ema.update_parameters(model)
    assert torch.allclose(ema.module.small, torch.tensor([0.5 * 0.9 + 1.5 * 0.1], dtype=torch.float32))


def test_amsgrad_is_a_valid_optimizer():
    from xnn.common.config import Config

    cfg = Config()
    cfg.optim.optimizer = "amsgrad"
    cfg.__post_init__()
    cfg.optim.optimizer = "sgd"
    try:
        cfg.__post_init__()
    except ValueError as err:
        assert "amsgrad" in str(err)
    else:
        raise AssertionError("an unknown optimizer must be rejected")
