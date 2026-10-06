"""The shared process-group helpers, outside and inside a launch.

The single-process behaviour (identity collectives, a full-range shard) is
checked directly; the collective semantics are checked in a 2-process Gloo
launch of a tiny script, as a user's ``torchrun`` would start it.
"""
import subprocess
import sys

import torch

from xnn.common import distributed


def test_single_process_defaults(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert distributed.world_size() == 1
    assert not distributed.is_distributed()
    assert distributed.rank() == 0 and distributed.is_main()
    dev, owns = distributed.init_process_group(torch.device("cpu"))
    assert dev.type == "cpu" and owns is False
    assert list(distributed.shard_indices(5)) == [0, 1, 2, 3, 4]
    assert distributed.all_reduce_sum([1.0, 2.5], dev) == [1.0, 2.5]
    t = torch.arange(6.0).reshape(2, 3)
    assert torch.equal(distributed.all_gather_cat(t, dev), torch.arange(6.0))
    distributed.barrier()
    distributed.destroy_process_group(owns)


_SCRIPT = """
import torch
from xnn.common import distributed

assert distributed.is_distributed() and distributed.world_size() == 2
dev, owns = distributed.init_process_group(torch.device("cpu"))
assert owns
r = distributed.rank()

# strided shards partition the dataset with no padding
idx = list(distributed.shard_indices(5))
assert idx == ([0, 2, 4] if r == 0 else [1, 3]), idx

assert distributed.all_reduce_sum([1.0, float(r)], dev) == [2.0, 1.0]

# ragged gather in rank order, including an empty piece whose dtype is unknown
piece = torch.tensor([10.0, 11.0, 12.0], dtype=torch.float64) if r == 0 else torch.empty(0)
got = distributed.all_gather_cat(piece, dev)
assert got.dtype == torch.float64 and got.tolist() == [10.0, 11.0, 12.0], got
piece = torch.full((r + 1,), float(r))
got = distributed.all_gather_cat(piece, dev)
assert got.tolist() == [0.0, 1.0, 1.0], got

distributed.barrier()
distributed.destroy_process_group(owns)
print("ok", r)
"""


def test_collectives_two_cpu_processes(tmp_path):
    script = tmp_path / "collectives.py"
    script.write_text(_SCRIPT)
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2",
         "--rdzv-backend", "c10d", "--rdzv-endpoint", "localhost:0", str(script)],
        capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ok") == 2
