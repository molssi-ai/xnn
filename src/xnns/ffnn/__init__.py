"""ffnn: learnable classical force fields.

Classical force-field functional forms whose parameters (and, optionally,
small embedded neural networks) are trainable by gradient descent -- the
force field *is* the model. The first member is ReaxFF, the bond-order
reactive force field of van Duin et al. (2001), together with its
machine-learned variant ReaxFF-nn (Guo et al. 2020, Xue et al. 2021).

Importing this package registers the family's models with the shared model
registry (``xnns.common.models``).
"""
from . import models  # noqa: F401

__all__ = ["models"]
