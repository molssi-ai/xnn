"""Low-rank adaptation (LoRA) of the linear layers of any xnn potential.

LoRA (Hu *et al.*, arXiv:2106.09685) freezes a pretrained weight ``W`` and
learns a low-rank update ``W + (alpha / r) B A`` with ``A`` of shape
``(r, d_in)`` and ``B`` of shape ``(d_out, r)``, ``B = 0`` at the start so
the adapted model equals the pretrained one. For the equivariant linear
layers of MACE and NequIP the update must keep its block structure: an
``o3.Linear`` is a scalar matrix per irrep block that never mixes irreps, and
the low-rank update is applied block by block with the same rank
(Tompa *et al.*, arXiv:2606.12704, eq. 2; the path-wise decomposition of
Wang *et al.*, ELoRA, ICML 2025, restricted to linear layers), so the adapted
layer stays an equivariant linear layer.

Two adapters cover every linear layer in xnn:

* :class:`LoRAAdapter` for any module with a dense 2-D weight
  (``torch.nn.Linear``, the layers of ``e3nn.nn.FullyConnectedNet``, PhysNet's
  dense layers), whichever axis is the input;
* :class:`LoRAEquivariantLinear` for ``e3nn.o3.Linear``.

Both compute the adapted weight and evaluate the *original* module with it
(``torch.func.functional_call`` / e3nn's external ``weight`` argument), so
the adapted layer is the base layer's own code with a different weight: one
code path for training and inference, and :func:`merge_lora` folds the update
into the base weights for deployment at no inference cost.

Enable from a config with ``model.extra["lora"]``::

    model:
      pretrained: mace-mp-0-small
      lora: {rank: 4, alpha: 1.0}        # or lora: 4, or lora: true (rank 4)
"""
from __future__ import annotations

import fnmatch
import logging
import math
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

#: Default rank of the low-rank update.
DEFAULT_RANK = 4
#: Default LoRA scaling numerator (the update is scaled by ``alpha / rank``).
DEFAULT_ALPHA = 1.0
#: Default standard deviation of the Gaussian input factor at initialization.
DEFAULT_INIT_STD = 1e-3


class LoRAAdapter(nn.Module):
    """Low-rank update of one dense weight matrix of any module.

    The base module is evaluated unchanged with its weight replaced by
    ``W + (alpha / rank) * delta``, ``delta = lora_in @ lora_out`` arranged in
    the weight's layout; ``lora_in`` (input side, Gaussian) has shape
    ``(d_in, rank)`` and ``lora_out`` (output side, zero at the start)
    ``(rank, d_out)``. The base weight is frozen.

    Parameters
    ----------
    base : torch.nn.Module
        The module to adapt.
    rank : int, optional
        Rank of the update, by default 4.
    alpha : float, optional
        Scaling numerator, by default 1.0.
    weight : str, optional
        Name of the 2-D weight parameter, by default ``"weight"``.
    in_axis : int, optional
        Which axis of the weight is the input dimension: 1 for
        ``torch.nn.Linear`` (``(d_out, d_in)``), 0 for the ``(d_in, d_out)``
        layout of e3nn's fully connected layers. By default 1.
    init_std : float, optional
        Standard deviation of ``lora_in`` at initialization, by default 1e-3.

    Attributes
    ----------
    base : torch.nn.Module
        The adapted module (frozen weight).
    lora_in, lora_out : torch.nn.Parameter
        The two factors.
    scaling : float
        ``alpha / rank``.
    """

    def __init__(self, base: nn.Module, rank: int = DEFAULT_RANK, alpha: float = DEFAULT_ALPHA,
                 *, weight: str = "weight", in_axis: int = 1, init_std: float = DEFAULT_INIT_STD):
        super().__init__()
        w = getattr(base, weight, None)
        if not torch.is_tensor(w) or w.dim() != 2:
            raise TypeError(f"{type(base).__name__}.{weight} is not a 2-D weight")
        if in_axis not in (0, 1):
            raise ValueError("in_axis must be 0 or 1")
        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self.base = base
        self.weight_name = weight
        self.in_axis = int(in_axis)
        self.rank = rank
        self.scaling = float(alpha) / rank
        n_in, n_out = w.shape[self.in_axis], w.shape[1 - self.in_axis]
        self.lora_in = nn.Parameter(
            torch.randn(n_in, rank, dtype=w.dtype, device=w.device) * float(init_std))
        self.lora_out = nn.Parameter(torch.zeros(rank, n_out, dtype=w.dtype, device=w.device))
        w.requires_grad_(False)

    def delta(self) -> Tensor:
        """The unscaled update, in the layout of the base weight."""
        d = self.lora_in @ self.lora_out                  # (d_in, d_out)
        return d if self.in_axis == 0 else d.t()

    def adapted_weight(self) -> Tensor:
        """``W + scaling * delta``."""
        return getattr(self.base, self.weight_name) + self.scaling * self.delta()

    def forward(self, *args, **kwargs):
        """Evaluate the base module with the adapted weight."""
        return torch.func.functional_call(
            self.base, {self.weight_name: self.adapted_weight()}, args, kwargs)

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Fold the update into the base weight and return the base module."""
        getattr(self.base, self.weight_name).add_(self.scaling * self.delta())
        return self.base

    def __getattr__(self, name: str):
        # attributes of the adapted layer (irreps, dimensions, ...) stay reachable
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base":
                raise
            return getattr(self.base, name)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:g}, weight={self.weight_name!r}"


class LoRAEquivariantLinear(nn.Module):
    """Block-wise low-rank update of an ``e3nn.o3.Linear``.

    An ``o3.Linear`` holds one weight block ``W_p`` of shape
    ``(mul_in, mul_out)`` per path ``p`` (an input irrep block connected to an
    output block of the same irrep), applied as ``path_weight * W_p`` on the
    irrep's channels. Each block gets its own rank-``r`` update acting on the
    effective map: ``path_weight * W_p + scaling * A_p B_p``, with ``A_p`` of
    shape ``(mul_in, r)`` Gaussian and ``B_p`` of shape ``(r, mul_out)`` zero
    at the start. Every block stays a scalar matrix on one irrep, so the
    adapted layer is equivariant. The base linear is evaluated with the
    adapted flat weight through its external ``weight`` argument. Bias terms
    (scalar outputs) are left unchanged.

    Parameters
    ----------
    base : e3nn.o3.Linear
        The layer to adapt (internal, shared weights).
    rank : int, optional
        Rank of every block's update, by default 4.
    alpha : float, optional
        Scaling numerator, by default 1.0.
    init_std : float, optional
        Standard deviation of the input factors at initialization, by default
        1e-3.

    Attributes
    ----------
    base : e3nn.o3.Linear
        The adapted layer (frozen weight).
    lora_in, lora_out : torch.nn.ParameterList
        The factors, one pair per weight path, in instruction order.
    paths : list of tuple
        ``(offset, numel, path_shape, path_weight)`` of each adapted path in
        the flat weight.
    """

    def __init__(self, base: nn.Module, rank: int = DEFAULT_RANK, alpha: float = DEFAULT_ALPHA,
                 *, init_std: float = DEFAULT_INIT_STD):
        super().__init__()
        if not getattr(base, "internal_weights", False) or base.weight.dim() != 1:
            raise TypeError("LoRAEquivariantLinear needs an o3.Linear with internal, "
                            "shared weights")
        rank = int(rank)
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / rank
        w = base.weight
        self.paths: list[tuple[int, int, tuple[int, int], float]] = []
        lora_in, lora_out = [], []
        offset = 0
        for ins in base.instructions:
            if ins.i_in < 0:                       # a bias: stored in base.bias
                continue
            numel = math.prod(ins.path_shape)
            if numel:
                mul_in, mul_out = ins.path_shape
                lora_in.append(nn.Parameter(
                    torch.randn(mul_in, rank, dtype=w.dtype, device=w.device) * float(init_std)))
                lora_out.append(nn.Parameter(
                    torch.zeros(rank, mul_out, dtype=w.dtype, device=w.device)))
                self.paths.append((offset, numel, tuple(ins.path_shape), float(ins.path_weight)))
            offset += numel
        if offset != base.weight_numel:
            raise RuntimeError("unexpected weight layout of the o3.Linear")
        self.lora_in = nn.ParameterList(lora_in)
        self.lora_out = nn.ParameterList(lora_out)
        w.requires_grad_(False)

    def delta(self) -> Tensor:
        """The unscaled update of the flat weight (every path, in layout order)."""
        pieces = [((a @ b) / pw).flatten()
                  for (_, _, _, pw), a, b in zip(self.paths, self.lora_in, self.lora_out)]
        if not pieces:
            return torch.zeros_like(self.base.weight)
        return torch.cat(pieces)

    def adapted_weight(self) -> Tensor:
        """``W + scaling * delta`` as a flat weight."""
        return self.base.weight + self.scaling * self.delta()

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the base linear with the adapted weight."""
        return self.base(x, weight=self.adapted_weight())

    @torch.no_grad()
    def merge(self) -> nn.Module:
        """Fold the update into the base weight and return the base linear."""
        self.base.weight.add_(self.scaling * self.delta())
        return self.base

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base":
                raise
            return getattr(self.base, name)

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:g}, paths={len(self.paths)}"


LORA_CLASSES = (LoRAAdapter, LoRAEquivariantLinear)

# (class, factory) pairs; the first class an instance matches wins
_TARGETS: list[tuple[type, Callable[..., nn.Module]]] = []
_builtin_done = False


def register_lora_target(cls: type, weight: str = "weight", in_axis: int = 1) -> None:
    """Make instances of ``cls`` eligible for :func:`inject_lora`.

    Parameters
    ----------
    cls : type
        A module class with a dense 2-D weight parameter.
    weight : str, optional
        Name of that parameter, by default ``"weight"``.
    in_axis : int, optional
        Axis of the weight that is the input dimension (1 for
        ``torch.nn.Linear``'s ``(d_out, d_in)``), by default 1.
    """
    def factory(base, rank, alpha, init_std):
        return LoRAAdapter(base, rank, alpha, weight=weight, in_axis=in_axis, init_std=init_std)
    _TARGETS.append((cls, factory))


def _builtin_targets() -> None:
    global _builtin_done
    if _builtin_done:
        return
    _builtin_done = True
    register_lora_target(nn.Linear, "weight", 1)
    try:
        from e3nn import o3
        from e3nn.nn._fc import _Layer
    except Exception:                 # e3nn is optional
        return
    _TARGETS.append((o3.Linear, lambda base, rank, alpha, init_std: LoRAEquivariantLinear(
        base, rank, alpha, init_std=init_std)))
    register_lora_target(_Layer, "weight", 0)


def _factory_for(module: nn.Module) -> Optional[Callable[..., nn.Module]]:
    _builtin_targets()
    for cls, factory in _TARGETS:
        if isinstance(module, cls):
            return factory
    return None


def lora_options(spec: Any) -> dict[str, Any]:
    """Normalize a ``model.extra["lora"]`` entry to :func:`inject_lora` keywords.

    ``True`` gives the defaults, an int sets the rank, a mapping may hold
    ``rank``, ``alpha``, ``init_std``, ``exclude`` (module-name patterns),
    ``freeze`` and ``trainable`` (parameter-name patterns kept trainable).
    """
    if spec is True:
        return {}
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return {"rank": int(spec)}
    if isinstance(spec, dict):
        known = {"rank", "alpha", "init_std", "exclude", "freeze", "trainable"}
        unknown = sorted(set(spec) - known)
        if unknown:
            raise ValueError(f"lora: unknown option(s) {unknown}; use {sorted(known)}")
        return dict(spec)
    raise TypeError(f"lora must be true, a rank or a mapping, got {spec!r}")


def inject_lora(model: nn.Module, rank: int = DEFAULT_RANK, alpha: float = DEFAULT_ALPHA,
                init_std: float = DEFAULT_INIT_STD, *, exclude: Sequence[str] = (),
                freeze: bool = True, trainable: Sequence[str] = ()) -> nn.Module:
    """Replace every eligible linear layer of ``model`` by a LoRA adapter, in place.

    Eligible layers are ``torch.nn.Linear``, ``e3nn.o3.Linear``, the layers of
    ``e3nn.nn.FullyConnectedNet`` and every class registered with
    :func:`register_lora_target`. The adapted model computes exactly what
    ``model`` did until the adapters are trained.

    Parameters
    ----------
    model : torch.nn.Module
        The model (bare, multi-head or wrapped).
    rank : int, optional
        Rank of every update, by default 4.
    alpha : float, optional
        Scaling numerator, by default 1.0.
    init_std : float, optional
        Standard deviation of the Gaussian factor at initialization.
    exclude : sequence of str, optional
        ``fnmatch`` patterns of module names to leave alone
        (``["readouts*"]`` keeps the readouts frozen and unadapted).
    freeze : bool, optional
        Freeze every parameter that is not a LoRA factor (the usual LoRA
        setting), by default True.
    trainable : sequence of str, optional
        Parameter-name patterns kept trainable besides the LoRA factors
        (``["*atom_ref*"]`` lets the reference energies move).

    Returns
    -------
    torch.nn.Module
        ``model``.

    Raises
    ------
    ValueError
        If no layer was eligible.
    """
    count = 0

    def visit(parent: nn.Module, prefix: str) -> None:
        nonlocal count
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LORA_CLASSES):
                continue
            if any(fnmatch.fnmatchcase(full, pat) for pat in exclude):
                continue
            factory = _factory_for(child)
            if factory is not None:
                try:
                    adapted = factory(child, rank, alpha, init_std)
                except TypeError as exc:
                    logger.debug("LoRA skips %s: %s", full, exc)
                    visit(child, full)
                    continue
                setattr(parent, name, adapted)
                count += 1
            else:
                visit(child, full)

    visit(model, "")
    if count == 0:
        raise ValueError(f"{type(model).__name__} has no linear layer LoRA can adapt")
    if freeze:
        for name, p in model.named_parameters():
            p.requires_grad = (_is_lora_name(name)
                               or any(fnmatch.fnmatchcase(name, pat) for pat in trainable))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA: %d layers adapted with rank %d; %d of %d parameters trainable (%.1f%%)",
                count, rank, n_train, n_total, 100.0 * n_train / max(n_total, 1))
    return model


def _is_lora_name(name: str) -> bool:
    parts = name.split(".")
    return "lora_in" in parts or "lora_out" in parts


def lora_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Every LoRA adapter inside ``model``."""
    for m in model.modules():
        if isinstance(m, LORA_CLASSES):
            yield m


def has_lora(model: nn.Module) -> bool:
    """Whether ``model`` holds any LoRA adapter."""
    return next(lora_modules(model), None) is not None


def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """The LoRA factors of ``model`` (the parameters LoRA trains)."""
    for name, p in model.named_parameters():
        if _is_lora_name(name):
            yield p


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRA update into its base layer, in place.

    The adapters are replaced by their base modules carrying the merged
    weights, so the result has the original architecture and ``state_dict``
    layout (and loads where the pretrained model did), at the pretrained
    model's inference cost. Every parameter is made trainable again.

    Parameters
    ----------
    model : torch.nn.Module
        A model with LoRA adapters (a model without any is returned unchanged).

    Returns
    -------
    torch.nn.Module
        ``model``.
    """
    def visit(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, LORA_CLASSES):
                setattr(parent, name, child.merge())
            else:
                visit(child)

    merged = has_lora(model)
    visit(model)
    if merged:
        for p in model.parameters():
            p.requires_grad = True
    return model
