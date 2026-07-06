"""Shared E(3)-equivariant building blocks (e3nn) for the GNN potentials.

Holds the pieces used by more than one equivariant model (NequIP / MACE /
Allegro): irreps helpers, the tensor-product path test, and a TorchScript-safe
normalized scalar activation. Model-specific blocks (MACE's interaction /
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
    irreps_out_list, instructions = [], []
    for i, (mul, ir_in) in enumerate(o3.Irreps(irreps1)):
        for j, (_, ir_edge) in enumerate(o3.Irreps(irreps2)):
            for ir_out in ir_in * ir_edge:
                if ir_out in o3.Irreps(target_irreps):
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    irreps_out, permut, _ = o3.Irreps(irreps_out_list).sort()
    instructions = [
        (a, b, permut[c], mode, train) for a, b, c, mode, train in instructions
    ]
    if sort_instructions:
        instructions = sorted(instructions, key=lambda x: x[2])
    return irreps_out, instructions


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """Whether some tensor-product path ``irreps_in1 x irreps_in2 -> ir_out`` exists.

    Mirrors ``nequip.utils.tp_utils.tp_path_exists``: used to prune hidden
    irreps that no tensor-product path can populate (e.g. ``0o`` features in
    the first NequIP layer, where the node features are still all ``0e``).

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
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
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
