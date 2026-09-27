"""Differentiate energy to get conservative forces and the stress tensor.

Wrap *any* registered model::

    model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
    out = model(graph)   # out has energy, forces, (stress)

Forces: ``F = -dE/dr`` (autograd w.r.t. positions).

Stress: symmetric-strain trick -- introduce eps (B,3,3)=0, displace positions
and cell by eps, then sigma = (1/V) dE/deps. This matches the NequIP/MACE
convention and is what ASE/LAMMPS expect.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .ops import cell_volume

from ..data import AtomicGraph


class ForceStressOutput(nn.Module):
    """Wrap a model to add conservative forces and the stress tensor.

    Wrapping any registered model augments its output dict with autograd-derived
    quantities, so forces and stress are computed uniformly in one place rather
    than in each model.

    Forces are the negative gradient of the energy with respect to positions,
    ``F = -dE/dr``. The stress is obtained with the symmetric-strain trick:
    a strain tensor ``eps`` of shape ``(B, 3, 3)`` initialised to zero is
    introduced, positions and cell are displaced by ``eps``, and the stress is
    ``sigma = (1/V) dE/deps``. This matches the NequIP/MACE convention expected
    by ASE and LAMMPS.

    Parameters
    ----------
    model : torch.nn.Module
        The wrapped model whose energy output is differentiated. Its ``cutoff``
        attribute, if present, is exposed as ``self.cutoff``.
    compute_forces : bool, optional
        Whether to compute forces (default ``True``).
    compute_stress : bool, optional
        Whether to compute the stress tensor (default ``False``). Stress is only
        computed for periodic structures (those with a non-``None`` cell).

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped model.
    compute_forces : bool
        Whether forces are computed.
    compute_stress : bool
        Whether stress is computed.
    cutoff : float or None
        The wrapped model's radial cutoff, or ``None`` if it has none.

    Examples
    --------
    >>> model = ForceStressOutput(build_model(cfg.model), compute_stress=True)
    >>> out = model(graph)   # out has energy, forces, (stress)
    """

    def __init__(self, model: nn.Module, compute_forces: bool = True,
                 compute_stress: bool = False):
        super().__init__()
        self.model = model
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress
        self.cutoff = getattr(model, "cutoff", None)

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Run the wrapped model and add forces and stress via autograd.

        Parameters
        ----------
        data : AtomicGraph
            The batched atomic graph to evaluate.

        Returns
        -------
        dict of str to torch.Tensor
            The wrapped model's output dict, augmented with:

            ``energy``
                Per-structure energies of shape ``(B,)`` (from the model).
            ``forces``
                Present when ``compute_forces`` is set; per-atom forces of shape
                ``(N, 3)`` computed as ``F = -dE/dr`` (autograd with respect to
                positions).
            ``stress``
                Present when ``compute_stress`` is set and the batch has a cell;
                the symmetric stress tensor of shape ``(B, 3, 3)`` computed as
                ``sigma = (1/V) dE/deps`` via the symmetric-strain trick, using
                the NequIP/MACE convention.

        Notes
        -----
        If the model's energy is independent of positions or strain (e.g. a
        ``T = 0`` MACE whose energy is only a per-atom reference), the
        corresponding gradient is unused and the associated forces/stress default
        to zeros instead of raising.
        """
        create_graph = self.training  # needed for force-loss backprop

        strain = None
        if self.compute_stress and data.cell is not None:
            strain = torch.zeros((data.num_graphs, 3, 3), dtype=data.pos.dtype,
                                 device=data.pos.device, requires_grad=True)
            sym = 0.5 * (strain + strain.transpose(-1, -2))
            # displace positions and cell by the strain of their structure
            data.pos = data.pos + torch.einsum(
                "ni,nij->nj", data.pos, sym[data.batch])
            data.cell = data.cell + torch.einsum("bij,bjk->bik", data.cell, sym)

        if self.compute_forces:
            data.pos.requires_grad_(True)

        out = self.model(data)
        energy = out["energy"]

        grad_outputs = torch.ones_like(energy)
        inputs, need = [], []
        if self.compute_forces:
            inputs.append(data.pos); need.append("pos")
        if strain is not None:
            inputs.append(strain); need.append("strain")

        # A model whose energy is independent of positions/strain (e.g. a T=0
        # MACE whose energy is only the per-atom reference) yields a None
        # (unused) gradient or an energy that needs no grad at all -- in both
        # cases forces/stress are zero, so default to zeros instead of failing.
        g = {k: None for k in need}
        if inputs and energy.requires_grad:
            grads = torch.autograd.grad(
                [energy], inputs, grad_outputs=[grad_outputs],
                create_graph=create_graph, retain_graph=True, allow_unused=True,
            )
            g = dict(zip(need, grads))
        if self.compute_forces:
            out["forces"] = -g["pos"] if g.get("pos") is not None else torch.zeros_like(data.pos)
        if strain is not None:
            if g.get("strain") is not None:
                volume = cell_volume(data.cell).clamp(min=1e-8)  # (B,)
                out["stress"] = g["strain"] / volume[:, None, None]
            else:
                out["stress"] = torch.zeros_like(strain)
        return out
