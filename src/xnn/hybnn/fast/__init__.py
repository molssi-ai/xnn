"""Fast paths of the hybrid-network family (Triton kernels).

The local message sums of SpookyNet (:mod:`xnn.hybnn.fast.spookynet`) and the attention
messages of BAMBOO (:mod:`xnn.hybnn.fast.bamboo`) have a fused GPU implementation next to their
reference one; :class:`~xnn.common.models.fast.FastPathModule` explains the contract and
``use_fast`` the choice.
"""
