"""Fast paths of the dispersion terms (DFT-D3, DFT-D4).

Profiled on periodic water boxes (5k-24k atoms, energy + forces + stress), the
reference D4 spends its time in two places: the large-regime EEQ solve above
about 10k atoms (85-90% of a step, nearly all of it recomputing structure
factors in the conjugate-gradient loop) and the three-body ATM term below that
and in molecular dynamics with the EEQ reuse (about 80% of a step); D3 spends
it in its three-body term and in the dense ``(N, N)`` C6 matrix that term reads.
The fast paths replace exactly those:

* :class:`~xnn.common.models.dispersion_fast.eeq.FastEEQSystem`: the EEQ
  operator with factorized reciprocal phases (no trigonometry per product), a
  sparse real-space matrix, a fused molecular kernel and lockstep solves;
* :func:`~xnn.common.models.dispersion_fast.atm.atm_energy`: the ATM energy and
  its gradient as fused Triton kernels, with the pair C6 from per-atom factors
  (no ``(N, N)`` matrix, so D3's three-body term has no atom limit).

They follow the ``use_fast`` convention of :mod:`xnn.common.models.fast`; the
parameters, buffers and outputs of the modules are unchanged, except that the
fast D3 path does not form the dense ``"c6_matrix"`` output (it is returned
empty, as above 20000 atoms; :meth:`~xnn.common.models.d3.DFTD3.c6_matrix`
computes it on request).
"""
from ..fast import AutoPolicy
from ._triton import available, supported

#: ``use_fast="auto"``: minimum edges of the neighbor graph per GPU model. None:
#: measured on A100, A30 and V100 in float32 and float64, the fast paths win at
#: every size, from 1.2-2x on 24-atom clusters to 3-11x at 1536 atoms and
#: 7-15x above 10k atoms
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0)

__all__ = ["AUTO_POLICY", "available", "supported"]
