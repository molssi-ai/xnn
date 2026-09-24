"""Shared machinery of the dispersion add-ons (DFT-D3, DFT-D4).

The two Grimme dispersion models share most of their structure: a real-space
cutoff with an optional quintic switching window, Gaussian coordination-number
weights over tabulated reference systems (D3 paper eq 16, D4 paper eq 8), an
Axilrod-Teller-Muto three-body term with zero damping over the atom triples
of a neighbor list (D3 paper eqs 11-14, D4 paper eqs 22-27), and the same way
of attaching themselves to a short-range model. Those pieces live here, once;
:mod:`~xnn.common.models.d3` and :mod:`~xnn.common.models.d4` hold what is
specific to each model (the coordination-number counting function, the
reference data, the damping functions, D4's charge scaling and EEQ charges).

:class:`DispersionCorrection` is the model-agnostic wrapper both add-ons
derive from: standalone it *is* the dispersion energy, given a short-range
``model`` it adds the dispersion energy to that model's prediction while
handing the model only the edges within its own cutoff. The deploy channels
(:mod:`~xnn.common.deploy.torchscript`, :mod:`~xnn.common.deploy.lammps`)
peel the wrapper off through the same interface.
"""
# NOTE: no ``from __future__ import annotations`` -- TorchScript resolves the
# annotations of the scripted functions at compile time.
import math
from dataclasses import replace
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from ..data import AtomicGraph
from .base import InteratomicPotential
from .ops import build_triplets, scatter_sum

# CODATA 2018, derived from h, m_e, c, alpha and e exactly as the reference
# codes (mctc-lib) do: a_0 = hbar / (m_e c alpha), E_h = m_e c^2 alpha^2. The
# tabulated 0.529177210903 / 27.211386245988 differ in the 12th digit, which
# the coordination numbers would show at the 1e-11 level.
BOHR = 0.5291772109044924      # Angstrom per bohr
HARTREE = 27.21138624593551    # eV per hartree



def switching_function(r: Tensor, cutoff: float, width: float) -> Tensor:
    """Quintic switching window: 1 below ``cutoff - width``, 0 above ``cutoff``.

    ``width <= 0`` gives the sharp cutoff (the upstream default). Inside the
    window ``x = (cutoff - r) / width`` runs from 0 to 1 and the switch is
    ``x^3 (10 - 15 x + 6 x^2)``, with zero slope at both ends.

    Parameters
    ----------
    r : Tensor
        Distances, any shape.
    cutoff : float
        End of the window (same unit as ``r``).
    width : float
        Width of the window; ``<= 0`` disables it.

    Returns
    -------
    Tensor
        The switch value in ``[0, 1]``, same shape as ``r``.
    """
    if width <= 0.0:
        return torch.ones_like(r)
    w = min(width, cutoff)
    x = torch.clamp((cutoff - r) / w, 0.0, 1.0)
    return x ** 3 * (10.0 + x * (6.0 * x - 15.0))


def gaussian_reference_weights(cn: Tensor, refcn: Tensor, valid: Tensor, wf: float,
                               ngw: Optional[Tensor] = None,
                               eps_norm: float = 0.0) -> Tensor:
    """Normalized Gaussian weights of the reference systems in the coordination number.

    D3 paper eq 16 / D4 paper eq 8: ``W_ref = sum_j^{N_ref} exp(-wf j (CN -
    CN_ref)^2) / norm``. When every Gaussian underflows (a coordination number
    far from all references) the reference(s) with the largest CN get weight
    one, as the reference codes do.

    Parameters
    ----------
    cn : Tensor
        Coordination numbers, shape ``(N,)``.
    refcn : Tensor
        Reference coordination numbers per atom, shape ``(N, R)``.
    valid : Tensor
        Boolean mask of the used reference slots, shape ``(N, R)``.
    wf : float
        Gaussian exponent (D3: 4, D4: 6).
    ngw : Tensor or None, optional
        Number of Gaussians per reference (D4's ``N^s``), shape ``(N, R)``;
        ``None`` (D3) uses a single Gaussian per reference.
    eps_norm : float, optional
        Norm below which the fallback applies (D4 uses ``sqrt(tiny)``, D3
        exactly zero).

    Returns
    -------
    Tensor
        Weights, shape ``(N, R)``, zero in unused slots.
    """
    dcn2 = (cn[:, None] - refcn) ** 2
    if ngw is None:
        gw = torch.exp(-wf * dcn2)
    else:
        n_max = int(ngw.max())
        j = torch.arange(1, n_max + 1, device=cn.device)
        gauss = torch.exp(-wf * j.to(cn.dtype)[None, None, :] * dcn2[:, :, None])
        use = j[None, None, :] <= ngw[:, :, None]
        gw = torch.where(use, gauss, torch.zeros_like(gauss)).sum(-1)
    gw = torch.where(valid, gw, torch.zeros_like(gw))
    norm = gw.sum(dim=1)
    ok = norm > eps_norm
    weights = gw / torch.where(ok, norm, torch.ones_like(norm))[:, None]
    neg_inf = torch.full_like(refcn, -math.inf)
    max_cn = torch.where(valid, refcn, neg_inf).max(dim=1).values
    fallback = (valid & ((refcn - max_cn[:, None]).abs() < 1e-12)).to(cn.dtype)
    return torch.where(ok[:, None], weights, fallback)


def three_body_energy(z: Tensor, edge_index: Tensor, edge_vec: Tensor, r: Tensor,
                      c6_mat: Tensor, r0_table: Tensor, s9: Tensor, alp3: float,
                      cutoff: float, width: float, n_atoms: int) -> Tensor:
    """Per-atom Axilrod-Teller-Muto three-body energy with zero damping.

    ``E = s9 sum_ABC C9 (3 cos cos cos + 1) / (R_AB R_BC R_CA)^3 / (1 + 6
    (R_0 / R)^alp3)`` with ``C9 = sqrt(C6_AB C6_BC C6_CA)``, ``R_0`` the
    geometric product of the three pair critical radii and ``R`` that of the
    three distances. Triples are enumerated as pairs of edges sharing a center;
    each geometric triangle is visited once per corner (three times for
    distinct atoms, and correspondingly for the self-image triangles of a
    periodic cell), so a third of the triangle energy is assigned to the
    center at every visit -- which reproduces the atom-resolved bookkeeping
    of the reference codes exactly. The triplet arithmetic runs in chunks so
    memory stays bounded for long cutoffs.

    Parameters
    ----------
    z : Tensor
        Atomic numbers, shape ``(N,)``.
    edge_index : Tensor
        Edges ``[src, dst]`` within the three-body cutoff, shape ``(2, E)``.
    edge_vec : Tensor
        Edge vectors in bohr, shape ``(E, 3)``.
    r : Tensor
        Edge lengths in bohr, shape ``(E,)``.
    c6_mat : Tensor
        Dense pair ``C6`` matrix, shape ``(N, N)``.
    r0_table : Tensor
        Pair critical radii by element, shape ``(Z_max + 1, Z_max + 1)``, bohr.
    s9 : Tensor
        Three-body scaling (scalar tensor, may be learnable).
    alp3 : float
        Damping exponent (``alp / 3`` in D4, ``(alp + 2) / 3`` in D3).
    cutoff, width : float
        Three-body cutoff and switching width, bohr.
    n_atoms : int
        Number of atoms.

    Returns
    -------
    Tensor
        Per-atom three-body energies in hartree, shape ``(N,)``.
    """
    e1, e2, center = build_triplets(edge_index, n_atoms)
    energy = torch.zeros(n_atoms, dtype=r.dtype, device=r.device)
    n_trip = e1.shape[0]
    chunk = 1048576                      # triplets per block (a literal: TorchScript)
    start = 0
    while start < n_trip:
        stop = min(start + chunk, n_trip)
        a, b, c = e1[start:stop], e2[start:stop], center[start:stop]
        j, k = edge_index[0][a], edge_index[0][b]
        v_ij, v_ik = edge_vec[a], edge_vec[b]
        r2_ij, r2_ik = r[a] ** 2, r[b] ** 2
        v_jk = v_ij - v_ik
        r2_jk = (v_jk * v_jk).sum(-1)
        keep = (r2_jk <= cutoff * cutoff) & (r2_jk > 2.220446049250313e-16)
        r2_jk = torch.where(keep, r2_jk, torch.ones_like(r2_jk))
        r_jk = torch.sqrt(r2_jk)

        c9 = s9 * torch.sqrt((c6_mat[c, j] * c6_mat[c, k] * c6_mat[j, k]).abs())
        zc, zj, zk = z[c], z[j], z[k]
        r0 = r0_table[zc, zj] * r0_table[zc, zk] * r0_table[zj, zk]

        prod2 = r2_ij * r2_ik * r2_jk
        prod1 = torch.sqrt(prod2)
        prod3 = prod2 * prod1
        prod5 = prod3 * prod2
        damp = 1.0 / (1.0 + 6.0 * (r0 / prod1) ** alp3)
        angular = (0.375 * (r2_ij + r2_jk - r2_ik) * (r2_ij - r2_jk + r2_ik)
                   * (-r2_ij + r2_jk + r2_ik) / prod5 + 1.0 / prod3)
        sw = (switching_function(r[a], cutoff, width) * switching_function(r[b], cutoff, width)
              * switching_function(r_jk, cutoff, width))
        e_tri = torch.where(keep, c9 * angular * damp * sw / 3.0, torch.zeros_like(prod1))
        energy = energy + scatter_sum(e_tri, c, n_atoms)
        start = stop
    return energy


class DispersionCorrection(InteratomicPotential):
    """A dispersion term as an xnn potential, standalone or wrapped around a model.

    Standalone (``model=None``) the potential is the dispersion energy alone;
    given a short-range ``model`` it adds the dispersion energy to that
    model's prediction: the wrapper's ``cutoff`` is the larger of the model's
    and the term's cutoffs (the neighbor-list radius the data pipeline uses),
    and the wrapped model only ever sees the edges within its own cutoff.
    :class:`~xnn.common.models.d4.D4Dispersion` and
    :class:`~xnn.common.models.d3.D3Dispersion` are the two instances; enable
    either from a config with ``model.extra["dispersion"]`` (see
    :func:`~xnn.common.models.registry.build_model`).

    Parameters
    ----------
    term : torch.nn.Module
        The dispersion evaluator (:class:`~xnn.common.models.d4.DFTD4` or
        :class:`~xnn.common.models.d3.DFTD3`): exposes ``cutoff`` (Angstrom)
        and ``evaluate(atomic_numbers, pos, edge_index, edge_vec, batch,
        num_graphs, cell, pbc, total_charge)`` returning at least
        ``"node_energy"`` (eV), ``"energy_2body"``, ``"energy_3body"``,
        ``"coordination_numbers"`` and ``"node_features"``.
    model : InteratomicPotential or None, optional
        The short-range model to correct; ``None`` for pure dispersion.

    Attributes
    ----------
    term : torch.nn.Module
        The dispersion evaluator.
    model : InteratomicPotential or None
        The wrapped model.
    cutoff : float
        Neighbor-list radius (Angstrom): ``max(model.cutoff, term.cutoff)``.
    node_feature_dim : int
        The wrapped model's feature width, or the term's own per-atom
        descriptors standalone (so LES can wrap a pure dispersion model too).

    Notes
    -----
    ``forward`` returns the combined ``"energy"`` / ``"node_energy"`` and adds
    ``"energy_sr"`` (the wrapped model's energy), ``"energy_disp"``,
    ``"energy_2body"``, ``"energy_3body"`` ``(B,)`` and the term's per-atom
    quantities (``"coordination_numbers"``, and for D4 ``"eeq_charges"``,
    ``"polarizabilities"``, ``"dynamic_polarizabilities"``; for D3
    ``"c6_matrix"``). The total charge of each structure is read from
    ``data.total_charge`` (``(B,)``; zero when absent).
    """

    # output keys of ``term.evaluate`` that are renamed in the wrapper's output
    _RENAME = {"charges": "eeq_charges"}

    def __init__(self, term: nn.Module, model: Optional[InteratomicPotential] = None):
        super().__init__()
        self.term = term
        self.model = model
        inner_cutoff = float(getattr(model, "cutoff", 0.0)) if model is not None else 0.0
        self.cutoff = max(inner_cutoff, float(term.cutoff))
        self.node_feature_dim = (int(getattr(model, "node_feature_dim", 0))
                                 if model is not None else int(term.n_features))

    def inner_graph(self, data: AtomicGraph) -> AtomicGraph:
        """``data`` restricted to the edges within the wrapped model's cutoff."""
        inner_cutoff = float(getattr(self.model, "cutoff", self.cutoff))
        if inner_cutoff >= self.cutoff:
            return data
        with torch.no_grad():
            keep = torch.linalg.norm(data.edge_vectors(), dim=-1) < inner_cutoff
        return replace(data, edge_index=data.edge_index[:, keep],
                       cell_shifts=data.cell_shifts[keep])

    def dispersion(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Evaluate the dispersion term on a (batched) graph; see :func:`evaluate_on_graph`."""
        return evaluate_on_graph(self.term, data)

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Wrapped-model prediction plus the dispersion energy.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph, built with this wrapper's ``cutoff``.

        Returns
        -------
        dict of str to Tensor
            See the class notes.
        """
        disp = self.dispersion(data)
        node_disp = disp.pop("node_energy")
        energy_disp = disp.pop("energy")
        features = disp.pop("node_features")
        if self.model is None:
            out: Dict[str, Tensor] = {
                "node_energy": node_disp, "energy": energy_disp,
                "node_features": features,
                "energy_sr": torch.zeros_like(energy_disp)}
        else:
            out = dict(self.model(self.inner_graph(data)))
            out["energy_sr"] = out["energy"]
            out["energy"] = out["energy"] + energy_disp
            out["node_energy"] = out["node_energy"] + node_disp
        out["energy_disp"] = energy_disp
        for key, value in disp.items():
            out[self._RENAME.get(key, key)] = value
        return out


def evaluate_on_graph(term: nn.Module, data: AtomicGraph) -> Dict[str, Tensor]:
    """Run a dispersion term's ``evaluate`` on a (batched) :class:`AtomicGraph`.

    Fills in an all-zero cell / periodicity for molecular batches, reads
    ``data.total_charge`` when present (zeros otherwise) and adds
    ``"energy"`` ``(B,)`` to the keys the term returns.

    Parameters
    ----------
    term : torch.nn.Module
        :class:`~xnn.common.models.d3.DFTD3` or :class:`~xnn.common.models.d4.DFTD4`.
    data : AtomicGraph
        The batched graph (built with the term's cutoff).

    Returns
    -------
    dict of str to Tensor
        The term's outputs plus ``"energy"``.
    """
    b = data.num_graphs
    dtype, device = data.pos.dtype, data.pos.device
    cell = (data.cell if data.cell is not None
            else torch.zeros((b, 3, 3), dtype=dtype, device=device))
    pbc = (data.pbc if data.pbc is not None
           else torch.zeros((b, 3), dtype=torch.bool, device=device))
    charge = getattr(data, "total_charge", None)
    if charge is None:
        charge = torch.zeros(b, dtype=dtype, device=device)
    out = term.evaluate(data.atomic_numbers, data.pos, data.edge_index,
                        data.edge_vectors(), data.batch, b, cell, pbc, charge.to(dtype))
    out["energy"] = scatter_sum(out["node_energy"], data.batch, b)
    return out


def options_from_extra(extra: dict, keys) -> dict:
    """Pick the recognized keyword arguments out of a config ``extra`` dict.

    Parameters
    ----------
    extra : dict
        A ``ModelConfig.extra`` mapping (or its ``dispersion`` sub-mapping).
    keys : iterable of str
        The accepted option names.

    Returns
    -------
    dict
        The recognized options; unknown keys (such as ``name``) are ignored.
    """
    return {k: extra[k] for k in keys if k in extra}
