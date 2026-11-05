"""PM3, PM6, PM6-ORG and PM7: the core-core terms and corrections of each method.

Common core-core form: ``E_AB = Z_A Z_B gamma_AB s(R) + G(R)`` with
``gamma_AB = (s_A s_A | s_B s_B)`` evaluated with the core additive terms.

* **PM3** (Stewart 1989): ``s = 1 + exp(-alpha_A R) + exp(-alpha_B R)``
  (``R exp(-alpha R)`` for N and O bonded to H), Gaussians
  ``G = Z_A Z_B / R sum_k K_k exp(-L_k (R - M_k)^2)`` of both atoms; pairs with
  diatomic parameters use ``s = 1 + 2 x_AB exp(-alpha_AB R)`` instead.
* **PM6** (Stewart 2007, eqs 6-11): ``s = 1 + 2 x_AB exp(-alpha_AB (R +
  0.0003 R^6))`` with the C-H, N-H, O-H (``exp(-alpha R^2)``), C-C (triple
  bonds) and Si-O special forms; pairs without diatomic parameters use
  ``|10 exp(-2.18 R) Z_A Z_B gamma| + Z_A Z_B gamma``; one Gaussian per atom;
  the repulsion ``1e-8 / x^12`` of the unpolarizable cores, ``x = R / (Z_A^(1/3)
  + Z_B^(1/3))``.
* **PM7** (Stewart 2013): PM6's form; missing diatomic parameters are the
  means of the like-pair values; a further O-H term.
* **PM6-ORG** (Stewart & Stewart 2023): PM6's form with Gaussian steric
  terms for the pairs of H, C, N, O and S.

Each threshold (``ax <= 25`` Gaussian cut-offs, ``x < 3``) and the numerical
constants follow the reference program MOPAC.
"""
from __future__ import annotations

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.models import register_model

from .core import pm3_core_core, pm6_core_core
from .corrections import batch_corrections, correction_tables
from .nddo import NDDO


class _PM6Family(NDDO):
    """Core-core terms shared by PM6, PM7 and PM6-ORG."""

    average_missing_pairs = False

    #: Gaussian steric terms of the H, C, N, O and S pairs in the scale factors (PM6-ORG).
    steric_pairs = False

    def core_core(self, za, zb, r, gab, tore_a, tore_b):
        return pm6_core_core(self.core_parameters(), za, zb, r, gab, tore_a, tore_b, self.average_missing_pairs,
                             self.steric_pairs)


def _corrections(model, data: AtomicGraph, pos: Tensor) -> Tensor:
    return batch_corrections(model.correction_kind, data.atomic_numbers.tolist(), pos, data.batch, data.num_graphs,
                             correction_tables(pos.device), model.peptide_barrier)


@register_model("pm3")
class PM3(NDDO):
    """PM3 (Stewart, J. Comput. Chem. 10, 209 and 221, 1989); see :class:`NDDO`."""

    default_parameters = "pm3"
    method = "PM3"
    correction_kind = 0
    core_form = 1
    peptide_barrier = 7.1853

    def core_core(self, za, zb, r, gab, tore_a, tore_b):
        return pm3_core_core(self.core_parameters(), za, zb, r, gab, tore_a, tore_b)

    def corrections(self, data, pos):
        return _corrections(self, data, pos)


@register_model("pm6")
class PM6(_PM6Family):
    """PM6 (Stewart, J. Mol. Model. 13, 1173, 2007); see :class:`NDDO`."""

    default_parameters = "pm6"
    method = "PM6"
    correction_kind = 1

    def corrections(self, data, pos):
        return _corrections(self, data, pos)


@register_model("pm7")
class PM7(_PM6Family):
    """PM7 (Stewart, J. Mol. Model. 19, 1, 2013); see :class:`NDDO`.

    PM6's Hamiltonian with the PM7 parameters, the transition-metal integral
    balance, like-pair means for missing diatomic parameters, and the
    dispersion and hydrogen-bond terms of the method.
    """

    default_parameters = "pm7"
    method = "PM7"
    correction_kind = 2
    average_missing_pairs = True
    feather = True

    balance_integrals = True

    def corrections(self, data, pos):
        return _corrections(self, data, pos)


@register_model("pm6_org")
class PM6ORG(_PM6Family):
    """PM6-ORG (Stewart & Stewart, J. Mol. Model. 29, 284, 2023); see :class:`NDDO`.

    PM6's Hamiltonian reparameterized for proteins, with Gaussian steric
    terms in the core-core repulsion of the H, C, N, O and S pairs and the
    D3 dispersion, H4 hydrogen-bond and H-H repulsion corrections (D3H4).
    """

    default_parameters = "pm6_org"
    method = "PM6-ORG"
    correction_kind = 3
    feather = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from xnn.common.models.d3 import DFTD3
        par = self.global_table
        self.d3 = DFTD3(damping="zero", s6=float(par[7]), rs6=float(par[9]), alp=float(par[8]),
                        s8=float(par[10]), rs8=1.0, references="2010", cutoff_pair=15.0,
                        cutoff_triple=15.0, cutoff_cn=1.0e4)

    steric_pairs = True

    def corrections(self, data, pos):
        from xnn.common.models.electrostatics import all_pairs
        from xnn.common.models.ops import scatter_sum  # noqa: F401
        src, dst = all_pairs(data.batch)
        edge_index = torch.stack([src, dst])
        vec = pos[dst] - pos[src]
        b = data.num_graphs
        zero_cell = torch.zeros((b, 3, 3), dtype=pos.dtype, device=pos.device)
        no_pbc = torch.zeros((b, 3), dtype=torch.bool, device=pos.device)
        charge = torch.zeros(b, dtype=pos.dtype, device=pos.device)
        d3 = self.d3.to(pos.dtype).evaluate(data.atomic_numbers, pos, edge_index, vec, data.batch, b,
                                            zero_cell, no_pbc, charge)["node_energy"]
        return d3 * 23.060547830619029 + _corrections(self, data, pos)
