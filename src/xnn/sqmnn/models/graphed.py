"""CUDA graphs of a differentiable function and of its derivatives (the sqmnn fast path).

:class:`GraphedFunction` records, for one batch composition, the kernels of ``fn(*inputs)`` and of its
vector-Jacobian products, first order and (on request) differentiable first order with its own derivative, on
fixed buffers. :func:`replay` applies them inside autograd: the outputs, gradients, gradients taken with
``create_graph`` (force training, Hessians) and the gradients of those come from replays where the graphs are
current, otherwise from the launched function, so every derivative is that of the reference.
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor


def _side_stream_runs(work: Callable[[], object], times: int = 2) -> None:
    """Launch ``work`` on a side stream before a capture (the allocator and the libraries settle there)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(times):
            work()
    torch.cuda.current_stream().wait_stream(side)


def synchronizes(work: Callable[[], object]) -> bool:
    """Whether ``work`` synchronizes with the host or copies from it (which a capture cannot hold and which would
    spoil the device's state if met during one): a dry run under the synchronization debug mode ``"error"``."""
    try:
        torch.cuda.synchronize()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # the debug mode announces itself as a prototype
            torch.cuda.set_sync_debug_mode("error")
        work()
        return False
    except RuntimeError:
        return True
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.cuda.set_sync_debug_mode(0)


def _zeros_for_none(grads: Sequence[Optional[Tensor]], like: Sequence[Tensor]) -> List[Tensor]:
    return [g if g is not None else torch.zeros_like(x) for g, x in zip(grads, like)]


class GraphedFunction:
    """CUDA graphs of ``fn(*inputs) -> tuple of tensors`` for one composition.

    * ``forward``: the outputs from the inputs (copied into fixed buffers);
    * ``vjp``: the gradients of the inputs that require them, for cotangents of the differentiable outputs;
    * after :meth:`capture_second`: ``vjp2``, the same gradients as a differentiable function of the inputs and
      the cotangents (``create_graph``), and ``hvp``, their gradients for given cotangents.

    Every graph has its own memory pool (they replay in any order). The generations tell whether the buffers a
    derivative needs still hold the evaluation it belongs to.
    """

    def __init__(self, fn: Callable[..., tuple], inputs: Sequence[Tensor]):
        self.fn = fn
        self.static = [x.detach().clone().requires_grad_(x.requires_grad) for x in inputs]
        self.grad_index = [i for i, x in enumerate(inputs) if x.requires_grad]
        self.generation, self.generation2 = 0, 0
        self.second = False
        self.wants_second = False
        self.second_tried = False
        with torch.enable_grad():
            self._capture_first()

    def _gin(self) -> List[Tensor]:
        return [self.static[i] for i in self.grad_index]

    def _capture_first(self) -> None:
        gin = self._gin()

        def work():
            outs = self.fn(*self.static)
            diff = [o for o in outs if o.requires_grad]
            if diff and gin:
                torch.autograd.grad(diff, gin, [torch.ones_like(o) for o in diff], allow_unused=True)
        _side_stream_runs(work)
        self.forward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph):
            self.outs = tuple(self.fn(*self.static))
        self.diff_index = [j for j, o in enumerate(self.outs) if o.requires_grad]
        self.cot = [torch.zeros_like(self.outs[j]) for j in self.diff_index]
        self.vjp_graph = None
        if self.diff_index and gin:
            self.vjp_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.vjp_graph):
                g = torch.autograd.grad([self.outs[j] for j in self.diff_index], gin, self.cot, retain_graph=True,
                                        allow_unused=True)
                self.grad_in = _zeros_for_none(g, gin)

    def capture_second(self) -> bool:
        """Capture the differentiable vector-Jacobian product and its derivative (after a forward replay, on the
        main thread); ``False`` where they would synchronize (the launched function then serves them)."""
        self.wants_second = False
        self.second_tried = True
        if self.vjp_graph is None:
            return False
        gin = self._gin()
        diff = [self.outs[j] for j in self.diff_index]
        with torch.enable_grad():
            cot2 = [torch.zeros_like(o).requires_grad_(True) for o in diff]

            def work():
                g = torch.autograd.grad(diff, gin, cot2, create_graph=True, retain_graph=True, allow_unused=True)
                live = [x for x in g if x is not None and x.requires_grad]
                if live:
                    torch.autograd.grad(live, gin + cot2, [torch.ones_like(x) for x in live], retain_graph=True,
                                        allow_unused=True)
            if synchronizes(work):
                return False
            _side_stream_runs(work)
            self.cot2 = cot2
            self.vjp2_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.vjp2_graph):
                g = torch.autograd.grad(diff, gin, cot2, create_graph=True, retain_graph=True, allow_unused=True)
                self.grad2 = _zeros_for_none(g, gin)
            self.live_index = [i for i, x in enumerate(self.grad2) if x.requires_grad]
            self.cot3 = [torch.zeros_like(self.grad2[i]) for i in self.live_index]
            self.hvp_graph = None
            if self.live_index:
                self.hvp_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.hvp_graph):
                    h = torch.autograd.grad([self.grad2[i] for i in self.live_index], gin + cot2, self.cot3,
                                            retain_graph=True, allow_unused=True)
                    self.grad3 = _zeros_for_none(h, gin + cot2)
        self.second = True
        return True

    def run(self, inputs: Sequence[Tensor]) -> int:
        # through .data: the buffers are saved by the captured autograd graph, whose version checks a copy would
        # trip when a derivative graph is captured from it
        with torch.no_grad():
            for buf, x in zip(self.static, inputs):
                buf.data.copy_(x)
        self.forward_graph.replay()
        self.generation += 1
        return self.generation


def replay(graph: GraphedFunction, inputs: Sequence[Tensor]) -> tuple:
    """The outputs of ``graph.fn(*inputs)`` from the graphs, differentiable to second order."""
    return _Replay.apply(graph, *inputs)


def _launched_vjp(graph: GraphedFunction, inputs: Sequence[Tensor], cots: Sequence[Tensor], create_graph: bool):
    """The gradients of the inputs that require them, from the launched function (the reference)."""
    with torch.enable_grad():
        xs = [x if x.requires_grad else x.detach().requires_grad_(i in graph.grad_index)
              for i, x in enumerate(inputs)]
        outs = graph.fn(*xs)
        gin = [xs[i] for i in graph.grad_index]
        g = torch.autograd.grad([outs[j] for j in graph.diff_index], gin, cots, create_graph=create_graph,
                                allow_unused=True)
    return _zeros_for_none(g, gin)


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph: GraphedFunction, *inputs):
        ctx.graph, ctx.generation = graph, graph.run(inputs)
        ctx.save_for_backward(*inputs)
        outs = tuple(o.detach().clone() for o in graph.outs)
        ctx.mark_non_differentiable(*[o for j, o in enumerate(outs) if j not in graph.diff_index])
        return outs

    @staticmethod
    def backward(ctx, *grads):
        graph, inputs = ctx.graph, ctx.saved_tensors
        cots = [grads[j] if grads[j] is not None else torch.zeros_like(graph.outs[j]) for j in graph.diff_index]
        current = ctx.generation == graph.generation
        out: List[Optional[Tensor]] = [None] * len(inputs)
        if torch.is_grad_enabled():
            if graph.second and current:
                g = _ReplayVJP.apply(graph, ctx.generation, len(inputs), *inputs, *cots)
            else:
                # the launched form; the next evaluation of the composition captures the graphs of this derivative
                graph.wants_second = not graph.second_tried
                g = _launched_vjp(graph, inputs, cots, True)
        elif current and graph.vjp_graph is not None:
            for buf, c in zip(graph.cot, cots):
                buf.data.copy_(c)
            graph.vjp_graph.replay()
            g = [x.clone() for x in graph.grad_in]
        else:
            g = _launched_vjp(graph, inputs, cots, False)
        for i, x in zip(graph.grad_index, g):
            out[i] = x
        return (None, *out)


class _ReplayVJP(torch.autograd.Function):
    """The gradients of the graphed function's inputs (those that require them) as a function of all inputs and
    the cotangents."""

    @staticmethod
    def forward(ctx, graph: GraphedFunction, generation: int, n_in: int, *args):
        cots = args[n_in:]
        for buf, c in zip(graph.cot2, cots):
            buf.data.copy_(c)
        graph.vjp2_graph.replay()
        graph.generation2 += 1
        ctx.graph, ctx.n_in = graph, n_in
        ctx.generations = (generation, graph.generation2)
        ctx.save_for_backward(*args)
        return tuple(x.detach().clone() for x in graph.grad2)

    @staticmethod
    def backward(ctx, *cc):
        graph, n_in = ctx.graph, ctx.n_in
        args = ctx.saved_tensors
        inputs, cots = args[:n_in], args[n_in:]
        cc = [c if c is not None else torch.zeros_like(x) for c, x in zip(cc, graph.grad2)]
        current = ctx.generations == (graph.generation, graph.generation2)
        if current and not torch.is_grad_enabled() and graph.hvp_graph is not None:
            for buf, i in zip(graph.cot3, graph.live_index):
                buf.data.copy_(cc[i])
            graph.hvp_graph.replay()
            h = [x.clone() for x in graph.grad3]
        else:
            h = _launched_hvp(graph, inputs, cots, cc)
        out: List[Optional[Tensor]] = [None] * n_in
        for k, i in enumerate(graph.grad_index):
            out[i] = h[k]
        return (None, None, None, *out, *h[len(graph.grad_index):])


def _launched_hvp(graph: GraphedFunction, inputs: Sequence[Tensor], cots: Sequence[Tensor],
                  cc: Sequence[Tensor]):
    """Second derivatives from the launched function: the gradients of ``<cc, vjp(inputs, cots)>`` with respect
    to the inputs that require them and to the cotangents."""
    create = torch.is_grad_enabled()
    with torch.enable_grad():
        xs = [x if x.requires_grad else x.detach().requires_grad_(i in graph.grad_index)
              for i, x in enumerate(inputs)]
        cs = [c if c.requires_grad else c.detach().requires_grad_(True) for c in cots]
        outs = graph.fn(*xs)
        g_in = [xs[i] for i in graph.grad_index]
        g = torch.autograd.grad([outs[j] for j in graph.diff_index], g_in, cs, create_graph=True, allow_unused=True)
        live = [(x, c) for x, c in zip(g, cc) if x is not None and x.requires_grad]
        if not live:
            return [torch.zeros_like(x) for x in list(g_in) + cs]
        h = torch.autograd.grad([x for x, _ in live], g_in + cs, [c for _, c in live], create_graph=create,
                                allow_unused=True)
    return _zeros_for_none(h, list(g_in) + cs)
