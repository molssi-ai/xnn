.. _howto-transplant:

**************************************
Transplant Weights from Upstream Codes
**************************************

Every state-of-the-art model from the literature, which is implemented in xnns,
is verified against its reference code by weight transplant: MACE, NequIP,
Allegro, CACE, PhysNet, ANI, BAMBOO, and the LES long-range add-on all load
upstream weights and reproduce the upstream energies and forces to round-off
error (see :ref:`fidelity` for what each implementation matches, and the
per-model notes below for the achieved precision). The one exception is SchNet:
it is a clean-room build from the manuscripts, so there is no upstream code (or
weights) to transplant; its verification reference is an independent NumPy
implementation of the papers' equations
(``examples/fidelity_checks/schnet_verification.ipynb``).

The pattern
===========
1. Build the xnns model with the *same architecture hyperparameters* as the
   upstream model (cutoff, ``l_max``/``max_ell``, channels, layers, radial
   basis size, ``avg_num_neighbors``, per-species energies/scales).
2. Map the upstream state dict onto the xnns parameter names.
3. ``load_state_dict`` and verify on a batch.

Transplants work in *both* directions: the same mapping loads xnns-trained
weights back into the reference code (see the lock-step MD example below).

How much work step 2 is depends on the model:

- **NequIP / Allegro**: the parameter names already match (the interaction
  block uses upstream names), and Allegro's strided channel-mixing linears
  keep the same flat weight layout as upstream, so a state dict trained with
  the reference code loads nearly as-is (and vice versa).
- **MACE**: the implementation reproduces upstream block by block, so whole
  models transplant through a simple name map.
- **CACE**: same, with one gotcha: upstream lazily initializes some layers
  (the ``Bchi`` transform and the readout MLP), so run one forward pass on a
  sample batch before reading the upstream state dict.
- **PhysNet**: the transplant crosses frameworks (the original is
  TensorFlow 1.x); each TF variable is assigned from / to the corresponding
  torch parameter as a NumPy array.
- **ANI**: torchani's *pretrained* ANI-1x, ANI-1ccx, and ANI-2x weights
  transplant into the :meth:`ANI.ani1x` / :meth:`ANI.ani1ccx` /
  :meth:`ANI.ani2x` presets, for a single network or the full 8-model
  ensemble.
- **BAMBOO / LES**: plain name maps onto the xnns modules; for BAMBOO note
  the documented force-convention difference (xnns returns the conservative
  ``-dE/dr``, equal to upstream ``forces + qeq_force``).

Worked examples
===============
The block-by-block notebooks in ``examples/fidelity_checks`` perform full
transplants and check every intermediate tensor:

- ``mace_verification.ipynb``: transplants a whole ``mace-torch`` model and
  reproduces its energy and forces to ~1e-15
  (``examples/gnn/mace/recreate_mace_architecture.ipynb`` is a longer
  companion tutorial that rebuilds the architecture step by step in both
  codes).
- ``nequip_verification.ipynb``: ends with a whole-model weight transplant
  (~1e-16 agreement in energies, forces, and stress).
- ``allegro_verification.ipynb``: the same for Allegro (~1e-15).
- ``cace_verification.ipynb``: whole-model transplant from the original
  ``cace`` package, ~1e-16 relative in energies and forces, molecular and
  periodic.
- ``physnet_verification.ipynb``: transplants the TF1 graph's variables into
  the pure-PyTorch xnns model; energies, forces, and corrected charges match
  to ~1e-15 (float64).
- ``ani_verification.ipynb``: transplants torchani's pretrained ANI-1x /
  ANI-1ccx / ANI-2x ensembles; the AEV matches element for element to ~1e-16,
  ensemble energies to ~1e-8 Ha and forces to ~2e-7 Ha/Å.
- ``bamboo_verification.ipynb``: transplants ``bytedance/bamboo`` weights;
  every GET layer, the charges, dipole, and component energies match to
  ~1e-15.
- ``les_verification.ipynb``: transplants the upstream CACE-LR latent-charge
  head and Ewald settings into ``LatentEwald(CACE)``; each kernel matches to
  ~1e-16 in float64 and the whole model to float32 round-off.

A reverse transplant in production: the MD notebook
``examples/dnn/physnet/physnet_argon_density_md.ipynb`` loads *xnns-trained*
PhysNet weights back into the original TF1 graph and propagates both engines
through the same NVE trajectory in lock step.

The classical force fields need no transplant machinery at all: the force
field *is* its parameter library. :class:`~xnns.ffnn.models.reaxff.ReaxFF`
loads ``ffield`` text and ReaxFF-nn JSON libraries directly, and
``ReaxFF.export_library()`` writes trained parameters back out in the same
portable format. :class:`~xnns.ffnn.models.opls.OPLS` likewise loads its
built-in sets, native JSON libraries, and GROMACS ``oplsaa.ff``-style
``.itp`` files (translated to OPLS conventions at load time), and
``OPLS.export_library()`` round-trips trained parameters through the native
JSON format.

Tests
=====
Parity with upstream given identical weights is also enforced in the test
suite whenever the reference package is available: ``tests/test_nequip.py``,
``tests/test_allegro.py``, ``tests/test_cace.py``, ``tests/test_ani.py``
(needs ``torchani``, the ``[ani]`` extra), ``tests/test_les.py``,
``tests/test_physnet.py`` (needs TensorFlow and an upstream clone via
``PHYSNET_UPSTREAM_PATH``), and ``tests/test_bamboo.py`` (upstream clone via
``BAMBOO_UPSTREAM_PATH``).
