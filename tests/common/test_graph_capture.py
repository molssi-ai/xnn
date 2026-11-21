"""CUDA graph recording with the cuBLAS workspaces cleared (xnn.common.models.fast.graph_capture).

The workspaces are cleared before and after every recording, also when the recording fails, and the options of
``torch.cuda.graph`` pass through. On a GPU, graphs of matrix products record and replay their results while
other graphs are recorded and freed and the allocator releases its memory in between.
"""
import contextlib

import pytest
import torch

from xnn.common.models import fast


def test_workspaces_are_cleared_around_every_recording(monkeypatch):
    """Clear, record (with the given options), clear; the second clear also after a failed recording."""
    calls = []
    # a CPU-only build of PyTorch has no such function: set it all the same
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append("clear"), raising=False)

    @contextlib.contextmanager
    def graph(g, **kwargs):
        calls.append(("record", g, tuple(sorted(kwargs.items()))))
        yield
        calls.append("recorded")

    monkeypatch.setattr(torch.cuda, "graph", graph)
    with fast.graph_capture("g", pool=1, capture_error_mode="thread_local"):
        calls.append("body")
    assert calls == ["clear", ("record", "g", (("capture_error_mode", "thread_local"), ("pool", 1))), "body",
                     "recorded", "clear"]
    calls.clear()
    with pytest.raises(RuntimeError):
        with fast.graph_capture("h"):
            raise RuntimeError("capture failed")
    assert calls[0] == "clear" and calls[-1] == "clear"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graphs_of_products_replay_after_others_are_freed():
    """Graphs of float64 matrix products (cuBLAS with a workspace), each on a side stream or in its own pool,
    replay the products of new inputs after earlier graphs are freed and the cache is emptied."""
    a = torch.randn(64, 64, dtype=torch.float64, device="cuda")
    b = torch.randn(64, 64, dtype=torch.float64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        (a @ b).sum()                       # a workspace of the side stream, made outside any graph
    torch.cuda.current_stream().wait_stream(stream)
    kept = []
    for i in range(6):
        g = torch.cuda.CUDAGraph()
        options = {"stream": stream} if i % 2 else {"pool": torch.cuda.graph_pool_handle()}
        with fast.graph_capture(g, **options):
            out = a @ b
        kept.append((g, out))
        if len(kept) > 2:
            del kept[0]                     # a freed graph and its pool
            torch.cuda.empty_cache()
        a.copy_(torch.randn_like(a))
        for graph, result in kept:
            graph.replay()
            torch.cuda.synchronize()
            assert torch.allclose(result, a @ b, rtol=1e-12, atol=1e-12)
