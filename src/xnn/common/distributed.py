"""Process-group helpers shared by training and benchmarking.

xnn's multi-GPU / multi-node support is native ``torch.distributed`` driven
purely by the environment: a launcher such as ``torchrun`` (or Slurm plus
torchrun) exports ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` into every process
it spawns, and the code paths here switch on those variables. A plain
``python`` / ``xnn`` invocation sees ``WORLD_SIZE`` unset and runs the
unchanged single-process pipeline. No framework wrapper is involved; these are
the handful of primitives the :class:`~xnn.common.train.Trainer` and the
:class:`~xnn.common.benchmark.Benchmark` share so they cannot drift apart.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world_size() -> int:
    """Number of processes in the distributed launch (1 when not launched)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    """Whether this process was spawned by a distributed launcher."""
    return world_size() > 1


def rank() -> int:
    """This process's global rank (0 when not launched distributed)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main() -> bool:
    """Whether this process is rank 0 (the one that prints and writes)."""
    return rank() == 0


def init_process_group(device: torch.device) -> tuple[torch.device, bool]:
    """Join the launcher's process group, if there is one.

    In a distributed run each rank is pinned to one CUDA device selected by
    ``LOCAL_RANK`` (the configured device only chooses cpu vs cuda), and the
    process group is initialized with the matching backend: NCCL on GPUs,
    Gloo on CPUs. A process group that already exists (for example created
    by a caller that runs several xnn components in one process) is reused.

    Parameters
    ----------
    device : torch.device
        The device resolved from the configuration.

    Returns
    -------
    tuple of (torch.device, bool)
        The device this rank should compute on (``device`` unchanged in a
        single-process run, ``cuda:LOCAL_RANK`` or ``device`` on CPU in a
        distributed one), and whether this call created the process group,
        in which case the caller owns it and should pass the flag to
        :func:`destroy_process_group` when done.
    """
    if not is_distributed():
        return device, False
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    owns = False
    if not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        owns = True
    return device, owns


def destroy_process_group(owns: bool) -> None:
    """Tear down the process group if ``owns`` says this caller created it."""
    if owns and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all ranks (a no-op outside a distributed run)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def shard_indices(n: int) -> range:
    """The dataset indices this rank scores: every ``WORLD_SIZE``-th one.

    A strided shard, ``range(RANK, n, WORLD_SIZE)``, so that ranks hold
    disjoint subsets whose union is exactly the dataset. Unlike
    ``torch.utils.data.DistributedSampler`` nothing is padded or repeated,
    so metrics gathered across ranks equal the single-process ones; the
    price is that shards may differ in length by one, which evaluation does
    not mind (training does, which is why the trainer keeps the sampler).
    Usable directly as a ``DataLoader`` ``sampler``.

    Parameters
    ----------
    n : int
        Dataset length.

    Returns
    -------
    range
        The indices for this rank (the full ``range(n)`` when not distributed).
    """
    return range(rank(), n, world_size())


def _comm_device(device: torch.device) -> torch.device:
    """Where collectives run: the rank's GPU under NCCL, the CPU under Gloo."""
    return device if device.type == "cuda" else torch.device("cpu")


def all_reduce_sum(values: list[float], device: torch.device) -> list[float]:
    """Sum a list of scalars over all ranks (identity outside a distributed run).

    Parameters
    ----------
    values : list of float
        Per-rank partial sums, in the same order on every rank.
    device : torch.device
        This rank's compute device, which selects where the collective runs.

    Returns
    -------
    list of float
        The global sums, identical on every rank.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device=_comm_device(device))
    dist.all_reduce(t)
    return t.tolist()


def all_gather_cat(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Concatenate a flat tensor from every rank, in rank order.

    Ranks may hold different lengths, including zero (a rank whose shard has
    no labelled structure for a target), so the lengths are exchanged first
    and each contribution is padded to the longest, gathered, and trimmed.
    The dtype travels with the lengths so an empty contribution can be typed
    like the others. Outside a distributed run the tensor is returned as is.

    Parameters
    ----------
    t : torch.Tensor
        This rank's one-dimensional contribution (any device).
    device : torch.device
        This rank's compute device, which selects where the collective runs.

    Returns
    -------
    torch.Tensor
        The concatenation over ranks, on the CPU.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t.reshape(-1).cpu()
    world = dist.get_world_size()
    comm = _comm_device(device)
    t = t.reshape(-1)

    # lengths and dtype of every rank's piece (an empty piece carries no dtype)
    metas: list = [None] * world
    dist.all_gather_object(metas, (t.numel(), str(t.dtype) if t.numel() else None))
    sizes = [m[0] for m in metas]
    dtypes = {m[1] for m in metas if m[1] is not None}
    if len(dtypes) > 1:
        raise RuntimeError(f"ranks hold different dtypes for one quantity: {sorted(dtypes)}")
    dtype = getattr(torch, dtypes.pop().removeprefix("torch.")) if dtypes else t.dtype
    longest = max(sizes)
    if longest == 0:
        return torch.empty(0, dtype=dtype)

    padded = torch.zeros(longest, dtype=dtype, device=comm)
    padded[:t.numel()] = t.to(comm, dtype)
    pieces = [torch.empty(longest, dtype=dtype, device=comm) for _ in range(world)]
    dist.all_gather(pieces, padded)
    return torch.cat([p[:n] for p, n in zip(pieces, sizes)]).cpu()
