"""Shared E(3)-equivariant building blocks (e3nn) for the GNN potentials.

Holds the pieces used by more than one equivariant model (NequIP / MACE /
Allegro): irreps helpers, the tensor-product path test, a TorchScript-safe
normalized scalar activation and the per-edge radial network. Model-specific blocks (MACE's interaction /
product blocks, NequIP's convnet layer, ...) live in the model modules.
Requires e3nn.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies the
# annotations that TorchScript needs to resolve, breaking `torch.jit.script`.
import math
from typing import Final, List, Optional, Tuple

import torch
from e3nn import nn as e3nn_nn
from e3nn import o3
from e3nn.math import normalize2mom
from torch import Tensor, nn
from torch.nn import functional as F


def shifted_softplus(x: Tensor) -> Tensor:
    """Shifted softplus ``log(0.5 e^x + 0.5)`` (NequIP's ``ShiftedSoftPlus``)."""
    return F.softplus(x) - math.log(2.0)


# scalar nonlinearities by their upstream config spellings; shared by the MACE
# readout gate (``GATES``) and the NequIP gate/radial-MLP choices
SCALAR_ACTIVATIONS = {
    "silu": F.silu,
    "tanh": torch.tanh,
    "abs": torch.abs,
    "ssp": shifted_softplus,
    "None": None,
    None: None,
}


def species_irreps(n_species: int) -> o3.Irreps:
    """Irreps of the one-hot species (node attribute) channels.

    Parameters
    ----------
    n_species : int
        Number of distinct chemical species.

    Returns
    -------
    o3.Irreps
        ``n_species`` even scalar irreps, i.e. ``o3.Irreps([(n_species, (0,
        1))])`` (written ``{n_species}x0e``).
    """
    return o3.Irreps([(n_species, (0, 1))])


def hidden_irreps(mul: int, l_max: int) -> o3.Irreps:
    """Hidden feature irreps: ``mul`` copies of the spherical-harmonic irreps.

    Parameters
    ----------
    mul : int
        Multiplicity (number of channels) per irrep degree.
    l_max : int
        Maximum degree ``l``; the degrees/parities follow
        ``o3.Irreps.spherical_harmonics(l_max)`` (i.e. ``0e, 1o, 2e, ...``).

    Returns
    -------
    o3.Irreps
        ``mul`` copies of each spherical-harmonic irrep up to ``l_max``.
    """
    return o3.Irreps([(mul, ir) for _, ir in o3.Irreps.spherical_harmonics(l_max)])


def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps,
    sort_instructions: bool = True,
) -> Tuple[o3.Irreps, List]:
    """(uvu) tensor-product output irreps + instructions, keeping only target paths.

    Enumerates the ``uvu`` tensor-product paths between ``irreps1`` and
    ``irreps2``, keeping only those whose output irrep lies in
    ``target_irreps``, then sorts the output irreps and remaps the instruction
    indices accordingly. Used to build the convolution
    :class:`e3nn.o3.TensorProduct` in the MACE and NequIP interaction blocks.

    Parameters
    ----------
    irreps1 : e3nn.o3.Irreps
        First operand irreps (node features).
    irreps2 : e3nn.o3.Irreps
        Second operand irreps (edge spherical-harmonic attributes).
    target_irreps : e3nn.o3.Irreps
        Only output irreps present in this set are retained.
    sort_instructions : bool, optional
        Also sort the instructions by output index (the ``mace-torch``
        convention; the original ``nequip`` keeps enumeration order). The
        choice fixes the tensor-product weight layout, so it must match the
        upstream code weights are transplanted from. Default is ``True``.

    Returns
    -------
    tuple of (e3nn.o3.Irreps, list)
        The sorted output irreps and the corresponding list of tensor-product
        instructions ``(i, j, k, "uvu", True)`` referencing the sorted indices.
    """
    wanted = o3.Irreps(target_irreps)
    kept: List[Tuple[int, o3.Irrep]] = []
    paths = []
    for a, (mul, ir_node) in enumerate(o3.Irreps(irreps1)):
        for b, (_, ir_edge) in enumerate(o3.Irreps(irreps2)):
            for ir_prod in ir_node * ir_edge:
                if ir_prod not in wanted:
                    continue
                # one uvu path per (input pair, product irrep); its output slot
                # is the position of the entry appended to `kept`
                paths.append((a, b, len(kept), "uvu", True))
                kept.append((mul, ir_prod))
    irreps_sorted, remap, _ = o3.Irreps(kept).sort()
    paths = [(a, b, remap[slot], mode, weighted)
             for a, b, slot, mode, weighted in paths]
    if sort_instructions:
        paths.sort(key=lambda path: path[2])
    return irreps_sorted, paths


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """Whether some tensor-product path ``irreps_in1 x irreps_in2 -> ir_out`` exists.

    Behaves like ``nequip.utils.tp_utils.tp_path_exists`` (independent
    implementation): used to prune hidden irreps that no tensor-product path
    can populate (e.g. ``0o`` features in the first NequIP layer, where the
    node features are still all ``0e``).

    Parameters
    ----------
    irreps_in1, irreps_in2 : e3nn.o3.Irreps
        The two tensor-product operand irreps.
    ir_out : e3nn.o3.Irrep or str
        The candidate output irrep.

    Returns
    -------
    bool
        ``True`` if any pair of input irreps couples to ``ir_out``.
    """
    target = o3.Irrep(ir_out)
    for mul1, ir1 in o3.Irreps(irreps_in1):
        if mul1 == 0:  # empty entries cannot contribute a path
            continue
        for mul2, ir2 in o3.Irreps(irreps_in2):
            if mul2 > 0 and target in ir1 * ir2:
                return True
    return False


class ScalarActivation(nn.Module):
    """Scriptable stand-in for :class:`e3nn.nn.Activation` on all-scalar irreps.

    ``e3nn.nn.Activation`` (0.4.4) does not compile under ``torch.jit.script``
    on torch 2.x, but for all-scalar irreps -- the only case the readouts and
    gates in this package need -- it reduces to applying the second-moment-
    normalized activation (:func:`e3nn.math.normalize2mom`) elementwise. This
    module does exactly that, so it is numerically identical to the e3nn
    original while remaining TorchScript-compatible. It carries no state, so
    swapping it in leaves the ``state_dict`` layout untouched.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Irreps of the activated features; every entry must be ``l = 0``.
    act : callable or None
        Scalar activation; ``None`` means identity.

    Raises
    ------
    ValueError
        If ``irreps_in`` contains any ``l > 0`` irrep.
    """

    has_act: Final[bool]

    def __init__(self, irreps_in: o3.Irreps, act):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        if irreps_in.lmax > 0:
            raise ValueError(
                f"gate activation needs all-scalar irreps, got {irreps_in}"
            )
        self.has_act = act is not None
        if act is not None:
            self.act = normalize2mom(act)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the normalized scalar activation (identity when ``act`` is None).

        Parameters
        ----------
        x : torch.Tensor
            All-scalar features.

        Returns
        -------
        torch.Tensor
            The activated features.
        """
        if self.has_act:
            return self.act(x)
        return x


#: Rows (edges) from which a radial network in evaluation mode recomputes its
#: hidden layers in the backward pass instead of keeping their activations
#: (``recompute="auto"``). Measured on an L40S with the D4 of the production
#: water model, 5k-41k water atoms: the peak falls by 22% for the production
#: model (32 channels) and 5-7% for MACE-OFF23 medium and MACE-MP-0b2 large; the
#: step takes 0-5% longer (one forward of the hidden layers, 4-5% of a step).
RECOMPUTE_MIN_EDGES = 1_000_000


class RadialNet(e3nn_nn.FullyConnectedNet):
    """Per-edge radial network that can recompute its hidden layers in the backward.

    An :class:`e3nn.nn.FullyConnectedNet` (same parameters, ``state_dict`` keys
    and outputs). When a gradient flows through it (forces, stress), the hidden
    layers' activations, ``E x hidden`` per layer whatever the tensor product's
    size, are kept for the backward pass; with recomputation only the last hidden
    output is kept and the hidden layers run again in the backward, giving the
    same values. TorchScript always keeps the activations.

    With ``out_index`` and ``out_scale`` it returns the output columns
    ``out_index`` times ``out_scale``, mapping the last layer's matrix once
    instead of every output row (the column order and scale of a fused tensor
    product's weights; equal to mapping the output, to round-off).

    Attributes
    ----------
    recompute : str
        ``"auto"`` (default): recompute in evaluation mode from
        :data:`RECOMPUTE_MIN_EDGES` edges; ``"on"``: whenever a gradient is
        needed; ``"off"``: never. Set with :func:`set_recompute_radial`.
    """

    recompute: str

    def __init__(self, hs, act=None, variance_in=1, variance_out=1, out_act=False):
        super().__init__(hs, act, variance_in, variance_out, out_act)
        self.recompute = "auto"
        # the (ignored) input of the matrix products that carry a scalar factor
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def forward(self, x: Tensor, out_index: Optional[Tensor] = None,
                out_scale: Optional[Tensor] = None) -> Tensor:
        """Apply the network to ``x`` of shape ``(E, hs[0])``; with ``out_scale``
        (and ``out_index``), its output columns mapped as in the class notes.

        Each layer is one matrix product whose scalar factor carries the layer's
        normalization and the constant of the previous activation, then the
        activation: the same values as the layers applied one by one, in fewer
        kernels. TorchScript applies the layers one by one.
        """
        if not torch.jit.is_scripting():
            return self._forward_folded(x, out_index, out_scale)
        for layer in self:
            x = layer(x)
        if out_scale is not None:
            if out_index is not None:
                x = x.index_select(1, out_index)
            x = x * out_scale
        return x

    @torch.jit.unused
    def _recomputes(self, x: Tensor) -> bool:
        if self.recompute == "off" or len(self) < 2:
            return False
        if not (torch.is_grad_enabled() and x.requires_grad):
            return False
        if self.recompute == "on":
            return True
        return not self.training and x.shape[0] >= RECOMPUTE_MIN_EDGES

    @staticmethod
    def _parts(layer: nn.Module):
        """``(weight, activation, constant after it, h_in var_in)`` of an e3nn layer.

        A LoRA wrapper gives its adapted weight. e3nn's layer computes
        ``act(x @ w / sqrt(h_in var_in)) * sqrt(var_out)`` with ``act = cst * f``,
        or ``x @ w / sqrt(h_in var_in / var_out)`` without an activation.
        """
        weight = layer.adapted_weight() if hasattr(layer, "adapted_weight") else layer.weight
        act, const = layer.act, math.sqrt(layer.var_out)
        if act is not None and hasattr(act, "cst") and hasattr(act, "f"):
            const *= 1.0 if act._is_id else float(act.cst)
            act = act.f
        return weight, act, const, layer.h_in * layer.var_in

    @torch.jit.unused
    def _matmul(self, x: Tensor, w: Tensor, alpha: float) -> Tensor:
        if self.training and torch.is_grad_enabled():
            # training: the factor on the (small) matrix, so the gradients are plain
            # products (addmm's backward multiplies each by alpha in a kernel of its
            # own, which a force loss records and differentiates again)
            return torch.mm(x, w * alpha if alpha != 1.0 else w)
        zero = self._zero if self._zero.dtype == x.dtype else self._zero.to(x.dtype)
        return torch.addmm(zero, x, w, beta=0, alpha=alpha)

    @torch.jit.unused
    def _plan(self):
        """The folded layers, or ``None`` when they are not e3nn's (a replaced block).

        ``(hidden, last)``: per hidden layer ``(layer, adapted, activation, factor)``
        and for the last one ``(layer, adapted, activation, constant after it,
        factor)``, the factor carrying the layer's normalization and the constant
        of the previous activation. Built once for the current layer objects
        (replacing a layer, as a LoRA wrapper does, builds it again); the weights
        are read at every call.
        """
        layers = tuple(self._modules.values())
        key = tuple(id(layer) for layer in layers)
        hit = self.__dict__.get("_plan_cache")
        if hit is not None and hit[0] == key:
            return hit[1]
        plan = None
        if all(hasattr(layer, "h_in") and hasattr(layer, "act") for layer in layers):
            hidden = []
            c = 1.0
            for layer in layers[:-1]:
                _, act, const, d = self._parts(layer)
                hidden.append((layer, hasattr(layer, "adapted_weight"), act, c / math.sqrt(d)))
                c = const
            _, act, const, d = self._parts(layers[-1])
            plan = (hidden, (layers[-1], hasattr(layers[-1], "adapted_weight"), act, const, c / math.sqrt(d)))
        # the layers themselves are held, so their ids stay theirs while cached
        self.__dict__["_plan_cache"] = (key, plan)
        return plan

    @staticmethod
    def _weight(layer: nn.Module, adapted: bool) -> Tensor:
        return layer.adapted_weight() if adapted else layer.weight

    @torch.jit.unused
    def _hidden(self, x: Tensor) -> Tensor:
        """The output of the hidden layers, owing the constant of the last activation."""
        for layer, adapted, act, factor in self._plan()[0]:
            x = self._matmul(x, self._weight(layer, adapted), factor)
            if act is not None:
                x = act(x)
        return x

    @torch.jit.unused
    def _forward_folded(self, x: Tensor, out_index: Optional[Tensor],
                        out_scale: Optional[Tensor]) -> Tensor:
        plan = self._plan()
        if plan is None:
            # not e3nn's layers (a replaced block): as written, the output mapped
            for layer in self:
                x = layer(x)
            if out_scale is not None:
                if out_index is not None:
                    x = x.index_select(1, out_index)
                x = x * out_scale
            return x
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if self._recomputes(x):
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(self._hidden, x, use_reentrant=False)
        else:
            h = self._hidden(x)
        layer, adapted, act, const, alpha = plan[1]
        w = self._weight(layer, adapted)
        if act is None:
            # the output scale folds into the product as well; a column map
            # (order and scale) applies to the last matrix, once
            if out_scale is not None:
                if out_index is not None:
                    w = w.index_select(1, out_index)
                w = w * out_scale
            y = self._matmul(h, w, alpha * const)
        else:
            y = act(self._matmul(h, w, alpha))
            if const != 1.0:
                y = y * const
            if out_scale is not None:
                if out_index is not None:
                    y = y.index_select(1, out_index)
                y = y * out_scale
        return y.reshape(lead + y.shape[-1:])

def set_recompute_radial(model: nn.Module, recompute="auto") -> nn.Module:
    """Choose when the radial networks of ``model`` recompute their hidden layers.

    Parameters
    ----------
    model : torch.nn.Module
        Any model; every :class:`RadialNet` in it is set, and every other module
        that declares ``radial_recompute`` (the TensorNet filter networks, the PaiNN
        filters).
    recompute : bool or str
        ``"auto"`` (default), ``True`` (whenever a gradient is needed) or
        ``False`` (never); see :class:`RadialNet`.

    Returns
    -------
    torch.nn.Module
        ``model``.
    """
    modes = {"auto": "auto", True: "on", False: "off"}
    if recompute not in modes:
        raise ValueError(f"recompute must be 'auto', True or False, got {recompute!r}")
    for module in model.modules():
        if isinstance(module, RadialNet) or getattr(type(module), "radial_recompute", False):
            module.recompute = modes[recompute]
    return model
