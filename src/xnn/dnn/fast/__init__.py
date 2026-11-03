"""Fast paths of the dense-network family (Triton kernels).

The atomic environment vectors of ANI (:mod:`xnn.dnn.fast.aev`) have a fused GPU
implementation next to their reference one; :class:`~xnn.common.models.fast.FastPathModule`
explains the contract and ``use_fast`` the choice.
"""
