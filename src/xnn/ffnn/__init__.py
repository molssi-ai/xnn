"""ffnn: learnable classical force fields.

Classical force-field functional forms whose parameters (and, optionally,
small embedded neural networks) are trainable by gradient descent -- the
force field *is* the model. The first member is ReaxFF, the bond-order
reactive force field of van Duin et al. (2001), together with its
machine-learned variant ReaxFF-nn (Guo et al. 2020, Xue et al. 2021). The
second is OPLS, the fixed-topology all-atom/united-atom force field of
Jorgensen et al. (1996), covering its variants (OPLS-AA, OPLS-UA, L-OPLS)
through interchangeable parameter libraries; DREIDING (Mayo et al. 1990)
generates its valence terms from per-atom rules, AMBER (Cornell et al.
1995, ff94) and CHARMM22 (MacKerell et al. 1998) assign their charges from
residue templates, and AMOEBA (Ren and Ponder 2003; Ponder et al. 2010; Shi
et al. 2013) and HIPPO (Rackers et al. 2021) are the polarizable atomic
multipole force fields with induced dipoles.

Parameter libraries for the whole family are read from the MolSSI/SEAMM
``.frc`` force-field format (:mod:`xnn.ffnn.common.frc`), which also
carries the SMARTS templates that type a structure
(:mod:`xnn.ffnn.common.typing`); the OPLS-AA distribution and a dozen
published ReaxFF fields ship with xnn
(:func:`~xnn.ffnn.common.frc.list_forcefields`).

Importing this package registers the family's models with the shared model
registry (``xnn.common.models``).
"""
from . import common  # noqa: F401
from . import models  # noqa: F401

__all__ = ["common", "models"]
