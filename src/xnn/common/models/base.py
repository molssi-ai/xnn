"""The contract every model obeys.

A model is an ``nn.Module`` that maps an :class:`AtomicGraph` to a dict with at
least:

    {"node_energy": (N,), "energy": (B,)}

Models do **not** compute forces/stress themselves -- that is added uniformly by
:class:`xnn.common.models.outputs.ForceStressOutput` via autograd, so every model gets
correct, conservative forces for free and the code lives in exactly one place.
"""
from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn

from ..data import AtomicGraph


class InteratomicPotential(nn.Module):
    """Abstract base class every interatomic-potential model must subclass.

    A model is an :class:`torch.nn.Module` that maps an :class:`AtomicGraph`
    to a dict of tensors with at least the keys ``"node_energy"`` (shape
    ``(N,)``, per-atom energies) and ``"energy"`` (shape ``(B,)``, per-structure
    energies). Models do **not** compute forces or stress themselves; those are
    added uniformly by
    :class:`xnn.common.models.outputs.ForceStressOutput` via autograd, so every
    model gets correct conservative forces for free.

    Notes
    -----
    The radial cutoff (in Angstrom, used to build neighbor lists) is set as the
    instance attribute ``cutoff`` in each subclass ``__init__``. It is
    intentionally not declared as a bare class annotation here, so subclasses
    remain compatible with :func:`torch.jit.script`.
    """

    # `cutoff` (radial cutoff in Angstrom, used to build neighbor lists) is set
    # as an instance attribute in each subclass __init__. It is intentionally
    # not a bare class annotation here, so subclasses stay torch.jit.script-able.

    @abstractmethod
    def forward(self, data: AtomicGraph) -> dict[str, torch.Tensor]:
        """Compute per-atom and per-structure energies for a graph.

        Parameters
        ----------
        data : AtomicGraph
            The batched atomic graph to evaluate.

        Returns
        -------
        dict of str to torch.Tensor
            A mapping containing at least ``"node_energy"`` of shape ``(N,)``
            (per-atom energies) and ``"energy"`` of shape ``(B,)`` (per-structure
            energies), where ``N`` is the number of atoms and ``B`` the number of
            structures in the batch.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a concrete subclass.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, cfg) -> "InteratomicPotential":
        """Construct a model instance from a configuration dataclass.

        Parameters
        ----------
        cfg : ModelConfig
            The model configuration dataclass describing hyperparameters.

        Returns
        -------
        InteratomicPotential
            A newly constructed model instance.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a concrete subclass.
        """
        raise NotImplementedError

    def aggregate_energy(self, node_energy: torch.Tensor,
                         data: AtomicGraph) -> torch.Tensor:
        """Sum per-atom energies into per-structure energies (locality).

        Parameters
        ----------
        node_energy : torch.Tensor
            Per-atom energies of shape ``(N,)``.
        data : AtomicGraph
            The batched graph, providing ``num_graphs`` and the ``batch`` vector
            that maps each atom to its structure index.

        Returns
        -------
        torch.Tensor
            Per-structure total energies of shape ``(B,)``, where
            ``B == data.num_graphs``.
        """
        energy = torch.zeros(data.num_graphs, dtype=node_energy.dtype,
                             device=node_energy.device)
        energy.index_add_(0, data.batch, node_energy)
        return energy
