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
from typing import Final, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import nn as e3nn_nn
from e3nn import o3
from e3nn.math import normalize2mom


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

    def forward(self, x: Tensor) -> Tensor:
        """Apply the network to ``x`` of shape ``(E, hs[0])``."""
        if not torch.jit.is_scripting() and self._recomputes(x):
            return self._forward_recomputed(x)
        for layer in self:
            x = layer(x)
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

    @torch.jit.unused
    def _forward_recomputed(self, x: Tensor) -> Tensor:
        from torch.utils.checkpoint import checkpoint
        layers = list(self)

        def hidden(h: Tensor) -> Tensor:
            for layer in layers[:-1]:
                h = layer(h)
            return h

        return layers[-1](checkpoint(hidden, x, use_reentrant=False))


def set_recompute_radial(model: nn.Module, recompute="auto") -> nn.Module:
    """Choose when the radial networks of ``model`` recompute their hidden layers.

    Parameters
    ----------
    model : torch.nn.Module
        Any model; every :class:`RadialNet` in it is set.
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
        if isinstance(module, RadialNet):
            module.recompute = modes[recompute]
    return model
