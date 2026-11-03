"""CPU fast path of the atom-centered symmetry functions (HDNNP): compiled blocks.

RuNNer evaluates the symmetry functions in fused Fortran loops; the reference
(:class:`~xnn.dnn.featurizers.acsf.AtomCenteredSymmetryFunctions`) evaluates the
radial functions of every edge and the angular functions of every triplet as a
sequence of tensor operations over ``(edges, functions)`` and ``(triplets,
functions)`` blocks, each read and written in full. On a CPU those two blocks
(the same methods) run compiled by ``torch.compile`` with dynamic sizes: inductor
fuses each into a few C++ / OpenMP loops, forward and backward. The neighbor
pairs and triplets are listed as in the reference. The compiled blocks are
differentiable once, so they serve evaluation (energies, forces, stress); a
training step keeps the reference.
"""
from __future__ import annotations

from xnn.common.models.dispersion_fast.atm_cpu import available
from xnn.common.models.fast import AutoPolicy

#: ``use_fast="auto"`` in evaluation: off on CPUs until measured (``cpu_min_edges``).
AUTO_POLICY = AutoPolicy()

#: Training mode: the reference (the compiled blocks are differentiable once).
AUTO_POLICY_TRAINING = AutoPolicy()

__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "available"]
