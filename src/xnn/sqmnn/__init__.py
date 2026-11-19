"""sqmnn: semiempirical quantum-mechanical models as xnn models.

The MNDO family of semiempirical Hartree-Fock methods (PM3, PM6, PM6-ORG,
PM7), with the published parameters reproducing the reference program MOPAC
or with any group of parameters trainable by gradient descent, and the GFN
extended tight-binding methods (GFN0-xTB, GFN1-xTB, GFN2-xTB) reproducing the xtb
program.

    common/  element data, constants, parameter readers and writers, the
             published values (downloaded on first use)
    models/  integrals, SCF, the methods (:class:`~xnn.sqmnn.models.PM6`,
             :class:`~xnn.sqmnn.models.GFN2`, ...)

The published parameter values are not shipped with xnn; see
:mod:`xnn.sqmnn.common.published` and :mod:`xnn.sqmnn.common.xtb_published`.
Importing this package registers the models (``pm3``, ``pm6``, ``pm6_org``,
``pm7``, ``gfn0_xtb``, ``gfn1_xtb``, ``gfn2_xtb``) with the shared model registry.
"""
from . import (
    common,  # noqa: F401
    models,  # noqa: F401
)

__all__ = ["common", "models"]
