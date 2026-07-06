.. _fidelity:

********************************
Model Fidelity to Upstream Codes
********************************

The equivariant models in xnns are not approximations or "inspired-by"
re-implementations: they are faithful, self-contained reproductions of the
reference codes, verified numerically block by block and end to end. Each
needs only ``e3nn`` — no ``mace-torch``, ``nequip``, ``cuequivariance``, or
``opt_einsum_fx`` at runtime.

Equivariance itself is verified in the test suite (rotate the inputs → the
energy is invariant and the forces co-rotate, to ~1e-7; see
``tests/test_gnn.py`` and ``tests/test_mace.py``).

MACE
====
A faithful re-implementation of `ACEsuit/mace <https://github.com/ACEsuit/mace>`_
(``mace-torch``):

- the real ``RealAgnostic(Residual)InteractionBlock`` and the paper's
  *learned symmetric contraction* over Clebsch–Gordan paths (``correlation``
  order);
- the CG coupling basis (``U_matrix_real``) is bit-identical to
  ``mace-torch``, and the contraction reproduces it to ~1e-16 given the same
  weights;
- adds ZBL ``pair_repulsion``, and makes the message-passing depth fully
  flexible (``num_interactions`` = T = 0..N, vs. upstream's fixed 2).

The notebooks in ``examples/gnn/mace/`` verify it block by block and end to
end against ``mace-torch`` on Argon MD data.

NequIP
======
A faithful re-implementation of the upstream ``EnergyModel`` of
`mir-group/nequip <https://github.com/mir-group/nequip>`_:

- the real ``InteractionBlock``, with upstream parameter names, so state
  dicts transplant directly;
- per-layer ``tp_path_exists`` irreps pruning and the gated nonlinearity;
- NequIP's radial conventions: trainable Bessel with the :math:`2/r_{\max}`
  prefactor, :math:`1/\sqrt{\langle n_\text{neigh}\rangle}` message
  normalization, and the :math:`\mathbf{r}_j - \mathbf{r}_i` edge
  orientation;
- the per-species energy scale/shift.

Given the same weights it reproduces ``nequip`` to ~1e-16 in energies,
forces, and stress (``tests/test_nequip.py``). TorchScript export required a
scriptable, bit-exact stand-in for e3nn's ``Gate``
(``xnns.gnn.models.nequip._Gate``), which the e3nn 0.4.4 original cannot
provide on torch 2.x.

Allegro
=======
A faithful re-implementation of the original `mir-group/allegro
<https://github.com/mir-group/allegro>`_ (v0.3.0, the e3nn-era reference,
default ``uuulin`` mode):

- the two-body product type embedding and the per-channel weightless
  Wigner-3j tensor products with the embedded-environment density trick;
- the strided channel-mixing linears with the same flat weight layout, so
  state dicts transplant directly;
- the cumulative-softmax latent resnet;
- Allegro's radial conventions: trainable "normalized sinc" Bessel with the
  :math:`r_{\max}/\pi` prefactor,
  :math:`1/\sqrt{\langle n_\text{neigh}\rangle - 1}` environment and
  :math:`1/\sqrt{\langle n_\text{neigh}\rangle}` energy-sum normalization;
- the per-species scale/shift.

Given the same weights it reproduces ``allegro`` to ~1e-15 in energies and
forces (``tests/test_allegro.py``).

Why this matters
================
Fidelity means results published with the reference codes can be reproduced,
weights can be transplanted in either direction (see
:ref:`howto-transplant`), and the xnns implementations can serve as readable,
single-dependency references for how these architectures actually work — the
``01_*_block_by_block_vs_original.ipynb`` notebooks double as annotated tours
of each architecture.
