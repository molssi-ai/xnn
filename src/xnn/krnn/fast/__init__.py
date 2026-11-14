"""Fast paths of the kernel-regression models (fused Triton kernels).

:mod:`xnn.krnn.fast.sgdml` evaluates the GDML / sGDML kernel expansion with fused
kernels; ``use_fast`` on the model chooses between it and the reference.
"""
from .sgdml import AUTO_POLICY, AUTO_POLICY_TRAINING, supported

__all__ = ["AUTO_POLICY", "AUTO_POLICY_TRAINING", "supported"]
