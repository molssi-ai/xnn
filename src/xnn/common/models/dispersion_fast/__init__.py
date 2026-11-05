"""Fast paths of the dispersion terms (D3, D4).

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
  its first and second derivatives as fused Triton kernels, with the pair C6
  from per-atom factors (no ``(N, N)`` matrix, so D3's three-body term has no
  atom limit);
* :mod:`~xnn.common.models.dispersion_fast.atm_cpu`: on CPUs, the three-body
  term on compiled tiles (``torch.compile``), with an analytic gradient for
  :mod:`~xnn.common.models.dispersion_fast.d3_direct` and
  :mod:`~xnn.common.models.dispersion_fast.d4_direct`, which form the forces
  and stress of a standalone D3 or D4 term (D4: molecules) without autograd;
* :mod:`~xnn.common.models.dispersion_fast.cpu`: optional C++ / OpenMP kernels
  for the whole of D3 and for D4's polarizabilities and two- and three-body
  terms, the loops of the Fortran codes (off by default, see
  :mod:`xnn.common.models.cpu_extension`).

They follow the ``use_fast`` convention of :mod:`xnn.common.models.fast`; the
parameters, buffers and outputs of the modules are unchanged, except that the
fast D3 paths do not form the dense ``"c6_matrix"`` output (it is returned
empty, as above 20000 atoms; :meth:`~xnn.common.models.d3.DFTD3.c6_matrix`
computes it on request).
"""
from ..fast import AutoPolicy
from ._triton import available, supported

#: ``use_fast="auto"``: minimum edges of the neighbor graph per GPU model. None:
#: measured on A100, A30 and V100 in float32 and float64, the fast paths win at
#: every size, from 1.2-2x on 24-atom clusters to 3-11x at 1536 atoms and
#: 7-15x above 10k atoms
AUTO_POLICY = AutoPolicy(default_min_edges=0, default_min_edges_float64=0,
                         # on a CPU the compiled three-body tiles (D3 and D4): faster than the reference
                         # at every size measured (0.06-0.94 on 8 threads), but the first call compiles for
                         # 10-20 s, which only pays from a few thousand edges (about 75 atoms) on
                         cpu_min_edges=5000, cpu_min_edges_float64=5000)

#: ``AUTO_POLICY`` for a term whose optional C++ CPU kernels are on and apply (D3 and D4,
#: :mod:`~xnn.common.models.dispersion_fast.cpu`): on a CPU every size takes them
AUTO_POLICY_CPU_KERNELS = AutoPolicy(default_min_edges=0, default_min_edges_float64=0,
                                     cpu_min_edges=0, cpu_min_edges_float64=0)

__all__ = ["AUTO_POLICY", "AUTO_POLICY_CPU_KERNELS", "available", "supported"]
