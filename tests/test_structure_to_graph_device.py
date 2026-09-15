"""structure_to_graph builds where it is told to.

The neighbour list is the expensive part and deployment rebuilds it every step,
so building it on the CPU and copying the result to the model's device pays for
it in the wrong place. Both deployment wrappers pass ``device``; a dataset
caching graphs does not, and still gets CPU tensors.
"""

import pytest
import torch

from xnns.common.data import structure_to_graph

CUT = 5.0


def struct(n=40, periodic=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = 10.0
    s = {
        "pos": torch.rand((n, 3), generator=g, dtype=torch.float64) * L,
        "atomic_numbers": torch.full((n,), 8, dtype=torch.long),
    }
    if periodic:
        s["cell"] = torch.eye(3, dtype=torch.float64) * L
        s["pbc"] = torch.ones(3, dtype=torch.bool)
    return s


def test_defaults_to_cpu():
    g = structure_to_graph(struct(), CUT)
    assert g.pos.device.type == "cpu"
    assert g.edge_index.device.type == "cpu"


@pytest.mark.parametrize("periodic", [True, False])
def test_every_tensor_lands_on_the_requested_device(periodic):
    """Including the ones built here rather than coerced from the input --
    a stray CPU tensor would force a copy on the first model call."""
    dev = torch.device("cpu")   # same code path as cuda; no GPU needed in CI
    g = structure_to_graph(struct(periodic=periodic), CUT, device=dev)
    for name in ("pos", "atomic_numbers", "edge_index", "cell_shifts",
                 "batch", "n_atoms"):
        t = getattr(g, name)
        assert t is None or t.device == dev, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_builds_on_cuda_without_touching_the_cpu_copy():
    dev = torch.device("cuda:0")
    g = structure_to_graph(struct(), CUT, device=dev)
    for name in ("pos", "atomic_numbers", "edge_index", "cell_shifts",
                 "batch", "n_atoms"):
        t = getattr(g, name)
        assert t is None or t.device.type == "cuda", name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_same_graph_either_way():
    """Where it is built must not change what is built."""
    s = struct()
    cpu = structure_to_graph(s, CUT)
    gpu = structure_to_graph(s, CUT, device=torch.device("cuda:0"))
    as_set = lambda ei, cs: {  # noqa: E731
        (int(a), int(b), int(x[0]), int(x[1]), int(x[2]))
        for a, b, x in zip(ei[0], ei[1], cs)
    }
    assert as_set(cpu.edge_index, cpu.cell_shifts) == as_set(
        gpu.edge_index.cpu(), gpu.cell_shifts.cpu()
    )
