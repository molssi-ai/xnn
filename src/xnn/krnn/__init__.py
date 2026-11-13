"""Kernel-regression potentials (``krnn``).

Models whose energy is a kernel expansion over stored reference geometries
rather than a neural network:

* :class:`~xnn.krnn.models.sgdml.SGDML` and :class:`~xnn.krnn.models.sgdml.GDML`,
  gradient-domain machine-learning force fields (Chmiela et al. 2017, 2018,
  2019): a global, energy-conserving kernel model of one molecule, fitted in
  closed form to forces, with the molecule's permutational symmetries (sGDML).

The descriptor lives in :mod:`xnn.krnn.featurizers`, the Matern kernels in
:mod:`xnn.krnn.kernels`, the fit, the symmetry search and the model-file
reader in :mod:`xnn.krnn.common`, and the fused GPU kernels of the expansion in
:mod:`xnn.krnn.fast`.
"""
from . import common, fast, featurizers, kernels, models  # noqa: F401  (registers the kernel models)

__all__ = ["common", "fast", "featurizers", "kernels", "models"]
