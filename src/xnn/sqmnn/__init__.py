"""sqmnn: semiempirical quantum-mechanical models (NDDO methods) as xnn models.

The MNDO family of semiempirical Hartree-Fock methods -- PM3, PM6, PM6-ORG
and PM7 -- in PyTorch, with the published parameters (reproducing the
reference program MOPAC) or with any group of parameters trainable by
gradient descent through the standard xnn training pipeline.

    common/  element data, constants, the MOPAC ``EXTERNAL`` reader and writer,
             the published values (downloaded from MOPAC on first use)
    models/  integrals, SCF, the methods (:class:`~xnn.sqmnn.models.PM6`, ...)

The published parameter values are not shipped with xnn; see
:mod:`xnn.sqmnn.common.published`. Importing this package registers the models (``pm3``, ``pm6``, ``pm6_org``,
``pm7``) with the shared model registry.
"""
from . import (
    common,  # noqa: F401
    models,  # noqa: F401
)

__all__ = ["common", "models"]
