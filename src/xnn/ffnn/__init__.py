"""ffnn: learnable classical force fields.

Classical force-field functional forms whose parameters (and, optionally,
small embedded neural networks) are trainable by gradient descent -- the
force field *is* the model. The first member is ReaxFF, the bond-order
reactive force field of van Duin et al. (2001), together with its
machine-learned variant ReaxFF-nn (Guo et al. 2020, Xue et al. 2021). The
second is OPLS, the fixed-topology all-atom/united-atom force field of
Jorgensen et al. (1996), covering its variants (OPLS-AA, OPLS-UA, L-OPLS)
through interchangeable parameter libraries.

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
