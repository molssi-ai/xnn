"""D3(CSO) dispersion for BAMBOO (Schroeder, Creon & Schwabe 2015).

BAMBOO adds a Grimme-D3 dispersion correction in the *CSO* reformulation
(Schroeder et al., J. Chem. Theory Comput. 11, 3163, 2015), which keeps only
the ``C6`` term and replaces the Becke-Johnson denominator with a smooth,
coordination-independent damping::

    E_disp = - sum_{i<j} C6_ij * m_ij / (r_ij^6 + R_c^6)
    m_ij   = s6 + a1 / (1 + exp(r_ij - 2.5 * R0_ij))

with ``R0_ij = sqrt(3 * r2r4_i * r2r4_j)`` and a fixed inner cutoff
``R_c = 7.75`` bohr. The coordination-number-dependent ``C6_ij`` is
interpolated exactly as in ordinary D3.

Implementation note: this reuses the standard Grimme-D3 reference tables and
the ``C6``/coordination-number machinery already shipped with xnn
(:mod:`xnn.common.models.d3`) rather than the upstream ``dftd3.pt`` blob, so the
dispersion here is *not* on the machine-precision fidelity path (the paper
excludes dispersion from the DFT training set entirely and only adds it during
MD, so it never enters the trained BAMBOO weights). It is provided for
MD/inference use and is **off by default**. Distances are handled in bohr and
the returned per-atom energy is converted to BAMBOO's native kcal/mol.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from xnn.common.models import d3

#: hartree -> kcal/mol (bytedance/bamboo ``utils/constant.py``).
HARTREE_KCAL_MOL = 627.5094740631


class D3CSODispersion(nn.Module):
    """Grimme D3(CSO) dispersion energy on a neighbour graph.

    Parameters
    ----------
    s6 : float, optional
        Global ``C6`` scaling, by default 1.0 (upstream ``DFTD3CSO``).
    a1 : float, optional
        Damping amplitude, by default 0.86 (upstream ``DFTD3CSO``).
    rc_bohr : float, optional
        Inner-damping cutoff ``R_c`` in bohr, by default 7.75.
    disp_cutoff : float or None, optional
        Real-space cutoff in Angstrom for the coordination-number damping;
        ``None`` disables it. Default 10.0.
    references : str, optional
        D3 reference systems, ``"2010"`` (default; Grimme's original tables,
        as in upstream BAMBOO) or ``"2024"`` (current ``simple-dftd3``
        references, Fr-Pu re-parametrized); see
        :func:`xnn.common.models.d3.legacy_c6_table`.

    Attributes
    ----------
    s6, a1, rc_bohr : float
        The damping parameters above.
    """

    def __init__(self, s6: float = 1.0, a1: float = 0.86, rc_bohr: float = 7.75,
                 disp_cutoff: float | None = 10.0, references: str = "2010"):
        super().__init__()
        self.s6 = s6
        self.a1 = a1
        self.rc_bohr = rc_bohr
        self.disp_cutoff = disp_cutoff
        dt = torch.get_default_dtype()
        # device-resident copies of the standard D3 reference tables
        self.register_buffer("_c6ab", d3.legacy_c6_table(str(references)).to(dt),
                             persistent=False)
        self.register_buffer("_rcov", d3.d3_rcov.to(dt), persistent=False)
        self.register_buffer("_r2r4", d3.d3_r2r4.to(dt), persistent=False)

    def forward(self, atomic_numbers: Tensor, edge_vec: Tensor,
                edge_index: Tensor, n_atoms: int) -> Tensor:
        """Per-atom D3(CSO) dispersion energy, in kcal/mol.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_vec : Tensor
            Edge displacement vectors ``(E, 3)`` in Angstrom (both directions).
        edge_index : Tensor
            Edge index ``(2, E)`` as ``[src, dst]``.
        n_atoms : int
            Number of atoms ``N``.

        Returns
        -------
        Tensor
            Per-atom dispersion energy ``(N,)`` in kcal/mol.
        """
        idx_j, idx_i = edge_index[0], edge_index[1]
        r_bohr = edge_vec.norm(dim=-1) / d3.d3_autoang
        cutoff_bohr = (None if self.disp_cutoff is None
                       else self.disp_cutoff / d3.d3_autoang)

        Zi, Zj = atomic_numbers[idx_i], atomic_numbers[idx_j]
        nc = d3._ncoord(Zi, Zj, r_bohr, idx_i, n_atoms, cutoff=cutoff_bohr,
                        rcov=self._rcov)
        nci, ncj = nc[idx_i], nc[idx_j]
        c6 = d3._getc6(Zi, Zj, nci, ncj, self._c6ab)

        r0 = torch.sqrt(3.0 * self._r2r4[Zi].to(c6.dtype) * self._r2r4[Zj].to(c6.dtype))
        damp = self.s6 + self.a1 / (1.0 + torch.exp(r_bohr - 2.5 * r0))
        t = 1.0 / (r_bohr ** 6 + self.rc_bohr ** 6)
        e_pair = -0.5 * c6 * damp * t                     # hartree, per direction
        if cutoff_bohr is not None:
            e_pair = torch.where(r_bohr < cutoff_bohr, e_pair,
                                 torch.zeros_like(e_pair))
        e_atom = d3._scatter_add(e_pair, idx_i, n_atoms)
        return e_atom * HARTREE_KCAL_MOL
