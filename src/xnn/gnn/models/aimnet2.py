"""AIMNet2 (Anstine, Zubatyuk & Isayev, *Chem. Sci.* **16**, 10228, 2025).

The atoms-in-molecules neural network potential for neutral and charged
organic and elemental-organic molecules, written from the paper on the xnn
GNN abstractions. The total energy is (paper eq 1)

    ``U = U_local + U_disp + U_coul``

with ``U_local`` the output of a message-passing network over a radial
Gaussian basis, ``U_coul`` the Coulomb energy of the partial charges the
network predicts (made exact in the net charge by *neural charge
equilibration*, eq 6) and ``U_disp`` a DFT-D3(BJ) correction. Block by block:

* **radial basis** (eq 2): ``g_ijs = exp(-eta (r_ij - r_s)^2) f_c(r_ij)``,
  ``n_rbf`` Gaussians on ``[rbf_start, cutoff)`` under a cosine cutoff
  (:class:`~xnn.common.featurizers.GaussianRBF`,
  :class:`~xnn.common.featurizers.CosineCutoff`);
* **atomic embedding** (eq 3): one learned ``(n_features, n_rbf)`` matrix
  ``a_ds`` per element, a feature vector *per radial shell*;
* **convolution** (eqs 4-5): the scalar features ``v_isd = sum_j g_ijs
  a_jds`` and the vector features ``v_ihd = | sum_js g_ijs u_ij a_jds
  w_dsh |^2`` (the squared norm of a learned combination of the shells along
  the unit bond vectors ``u_ij``), see :class:`ShellConvolution`;
* **message passing**: each pass feeds the atom's own embedding and its
  convolved environment to an MLP that returns an embedding update, a
  partial-charge update and a non-negative weight. From the second pass on
  the charges enter the input too, through the same convolution. Every
  charge update is followed by the equilibration (eq 6)

  ``q_i = q~_i + f_i / sum_j f_j (Q - sum_j q~_j)``

  that redistributes the surplus charge according to the predicted weights
  ``f``; the final pass yields the AIM vector;
* **readout**: an MLP from the AIM vector to the atomic energy, plus a
  per-element shift (the standard xnn ``atom_ref``, kept in float64 as the
  reference code does since the shifts are O(1000 eV));
* **electrostatics**: the published models learn the Coulomb interaction
  within ``coulomb_sr_cutoff`` implicitly, so the short-range part is
  subtracted under a smooth envelope and the Coulomb energy of the
  predicted charges is added back: the full ``1/r`` sum over every pair
  (``coulomb="simple"``, molecules), a damped shifted-force truncation at
  ``lr_cutoff`` (``coulomb="dsf"``, what the paper uses for its periodic CO2
  simulation), or the Ewald sum of a periodic cell with the real-space part
  over the neighbor list within ``lr_cutoff`` and the reciprocal part either
  over the lattice vectors (``coulomb="ewald"``) or on a B-spline charge
  mesh (``coulomb="pme"``), both to the relative accuracy
  ``ewald_accuracy`` (:mod:`~xnn.common.models.electrostatics`).

The open-shell variant (AIMNet2-NSE, ``charge_channels=2``) carries two
charge channels equilibrated separately to ``(Q +- (M - 1)) / 2`` for a
structure of net charge ``Q`` and spin multiplicity ``M``
(``AtomicGraph.spin_multiplicity``); their sum is the partial charge and
their difference the spin density.

The D3(BJ) correction the published models were fitted without and are
served with is not part of this class: the foundation loader
(:mod:`~xnn.gnn.models.aimnet2_foundation`) records it as the config's
``subtracted_dispersion`` and the model hub wraps the model in the shared
:class:`~xnn.common.models.d3.D3Dispersion` term, as for every xnn model.

The network and its Coulomb sums are tensor code that compiles under
TorchScript: :meth:`AIMNet2.node_features_energy_charges` is the core the
deploy wrapper (:class:`~xnn.common.deploy.TorchScriptPotential`) exports,
with the net charge and spin multiplicity fixed at export time.

Fidelity: an independent implementation that follows the conventions of the
reference code `isayevlab/aimnetcentral
<https://github.com/isayevlab/aimnetcentral>`_ where the paper leaves them
open (Gaussian centers ``r_s = rbf_start + s (cutoff - rbf_start) / n_rbf``
with ``eta = (n_rbf / (cutoff - rbf_start))^2``, the squared vector norm,
the ``1e-6`` regularization of the equilibration weights, the shifted-force
Coulomb with its self term, the Ewald/PME sums with their self and
background terms and the Kolafa-Perram parameter balance, the float64
accumulation of the pair sums), so that the published checkpoints
transplant weight for weight and reproduce the reference energies, forces
and charges to float32 round-off, and the Ewald and PME energies to the
requested accuracy (``tests/test_aimnet2.py``,
``examples/fidelity_checks/aimnet2_verification.ipynb``). One documented
difference: the reference chooses the Ewald real-space cutoff per structure
from the accuracy, xnn keeps it at ``lr_cutoff`` (the radius of the graph
the data pipeline builds) and balances the splitting parameter to it.
Units: eV, Angstrom, elementary charges.
"""
import ast
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, GaussianRBF
from xnn.common.models.electrostatics import (COULOMB_CONSTANT, coulomb_direct, coulomb_dsf,
                                              coulomb_ewald, coulomb_pme, ewald_parameters)
from xnn.common.models.ops import scatter_sum, structure_sum
from xnn.common.models.registry import register_model
from ..featurizers import MollifierCutoff
from .base import GNNPotential

#: Electrostatics options: the all-pairs sum of molecules, the damped
#: shifted-force truncation, the Ewald and particle-mesh Ewald sums of
#: periodic cells, or none.
COULOMB_METHODS = ("simple", "dsf", "ewald", "pme", None)
#: The methods whose neighbor list reaches ``lr_cutoff``.
LONG_RANGE_METHODS = ("dsf", "ewald", "pme")
#: Envelopes that switch off the implicitly learned short-range Coulomb energy.
SR_ENVELOPES = {"exp": MollifierCutoff, "cosine": CosineCutoff}
#: Regularization of the equilibration weights, ``sum_j f_j + eps`` (eq 6).
NQE_EPSILON = 1.0e-6


def _mlp(sizes: Sequence[int], final_activation: bool) -> nn.Sequential:
    """Linear layers with GELU between them (and after the last one if asked).

    Weights are Xavier-normal and biases zero, the initialization of the
    reference code.
    """
    layers: list[nn.Module] = []
    for i in range(1, len(sizes)):
        linear = nn.Linear(sizes[i - 1], sizes[i])
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i < len(sizes) - 1 or final_activation:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ShellConvolution(nn.Module):
    """The AIMNet2 convolution of per-atom features over the radial shells (paper eqs 4-5).

    For center ``i`` with neighbors ``j`` carrying features ``a_jcs`` (one
    value per channel ``c`` and radial shell ``s``) and the radial basis
    ``g_ijs`` of the pair, the scalar features are the shell-wise sums

        ``v_ics = sum_j g_ijs a_jcs``

    and the vector features the squared norms of ``n_combinations`` learned
    combinations of the shells along the unit bond vectors ``u_ij``,

        ``v_ich = | sum_s w_csh sum_j g_ijs u_ij a_jcs |^2``.

    Both are flattened and concatenated (channel-major) into a vector of
    ``n_channels * (n_shells + n_combinations)`` entries per atom. The atomic
    embeddings (one value per shell) and the partial charges (the same value
    in every shell) go through the same block.

    Parameters
    ----------
    n_channels : int
        Feature channels ``c`` of the convolved quantity.
    n_shells : int
        Radial shells ``s`` (the number of Gaussian basis functions).
    n_combinations : int
        Learned shell combinations ``h`` of the vector features.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The combination weights ``w_csh``, shape ``(n_channels, n_shells,
        n_combinations)``. Initialized to standardized random vectors over
        the shells (the reference code picks a maximally spread set of
        sinusoids; any spread set serves, and trained weights transplant).
    output_dim : int
        ``n_channels * (n_shells + n_combinations)``.
    """

    n_channels: int
    n_shells: int
    n_combinations: int
    output_dim: int

    def __init__(self, n_channels: int, n_shells: int, n_combinations: int):
        super().__init__()
        weight = torch.randn(n_channels, n_shells, n_combinations)
        weight = weight - weight.mean(dim=1, keepdim=True)
        weight = weight / weight.std(dim=1, keepdim=True)
        self.weight = nn.Parameter(weight)
        self.n_channels = n_channels
        self.n_shells = n_shells
        self.n_combinations = n_combinations
        self.output_dim = n_channels * (n_shells + n_combinations)

    def forward(self, feats: Tensor, g: Tensor, gu: Tensor, dst: Tensor,
                num_nodes: int) -> Tensor:
        """Convolve neighbor features into per-center environment features.

        Parameters
        ----------
        feats : Tensor
            Features of the neighbor (source) atom of every edge, shape
            ``(E, n_channels, n_shells)``.
        g : Tensor
            Radial basis of every edge, shape ``(E, n_shells)``.
        gu : Tensor
            Radial basis times the unit bond vector, shape ``(E, n_shells, 3)``.
        dst : Tensor
            Center atom of every edge, shape ``(E,)``.
        num_nodes : int
            Number of atoms ``N``.

        Returns
        -------
        Tensor
            Environment features, shape ``(N, output_dim)``.
        """
        scalar = scatter_sum(feats * g[:, None, :], dst, num_nodes)        # (N, C, S)
        # one Cartesian component at a time keeps the edge temporaries at
        # (E, C, S) instead of (E, C, S, 3)
        vector = torch.stack(
            [scatter_sum(feats * gu[:, None, :, d], dst, num_nodes) for d in range(3)],
            dim=-1)                                                          # (N, C, S, 3)
        mixed = torch.einsum("csh,ncsd->nchd", [self.weight, vector])
        norms = mixed.square().sum(-1)                                       # (N, C, H)
        return torch.cat([scalar.flatten(1), norms.flatten(1)], dim=-1)


@register_model("aimnet2")
class AIMNet2(GNNPotential):
    """The AIMNet2 potential (Anstine *et al.* 2025); see the module docstring.

    Subclasses :class:`~xnn.gnn.models.base.GNNPotential` for the species
    bookkeeping and the per-element reference energy ``atom_ref`` (the
    paper's atomic energy shifts, kept in float64 whatever the model dtype).

    Parameters
    ----------
    species : list of int
        Atomic numbers of the supported elements, in channel order.
    cutoff : float, optional
        Radius of the local environment (the Gaussian basis and its cosine
        cutoff), by default 5.0 Angstrom. The neighbor-list radius
        ``self.cutoff`` is this, or ``lr_cutoff`` with a long-range Coulomb
        method (``"dsf"``, ``"ewald"``, ``"pme"``).
    n_features : int, optional
        Embedding channels ``d`` per radial shell, by default 16.
    n_rbf : int, optional
        Gaussian basis functions (radial shells ``s``), by default 16.
    rbf_start : float, optional
        Center of the first Gaussian, by default 0.8 Angstrom; the centers
        are ``rbf_start + s (cutoff - rbf_start) / n_rbf``.
    gaussian_width : float or None, optional
        Exponent ``eta`` of the Gaussians; ``None`` (default) uses the
        reference choice ``(n_rbf / (cutoff - rbf_start))^2`` (the inverse
        squared spacing).
    hidden : sequence of sequences of int or None, optional
        Hidden widths of the MLP of each message pass; the number of passes
        is their count (at least two: the first pass predicts the charges,
        the last the AIM vector). ``None`` gives the published architecture
        ``[[512, 380], [512, 380], [512, 380, 380]]`` for ``n_passes``.
    n_passes : int, optional
        Number of message passes when ``hidden`` is ``None``, by default 3.
    n_vector_combinations : int, optional
        Learned shell combinations ``h`` of the vector features, by default 12.
    aim_size : int, optional
        Width of the AIM vector (the ``node_features``), by default 256.
    readout_hidden : sequence of int, optional
        Hidden widths of the energy MLP, by default ``(128, 128)``.
    charge_channels : int, optional
        1 (closed shell, default) or 2 (AIMNet2-NSE: one channel per spin).
    coulomb : str or None, optional
        ``"simple"`` (default): the all-pairs Coulomb energy of molecular
        structures; ``"dsf"``: the damped shifted-force truncation at
        ``lr_cutoff`` over the neighbor list (periodic structures, large
        molecules); ``"ewald"`` / ``"pme"``: the Ewald sum of periodic
        structures, its real-space part over the neighbor list within
        ``lr_cutoff`` and its reciprocal part over the lattice vectors or on
        a particle mesh (non-periodic structures of the batch get the
        all-pairs sum); ``None``: no electrostatics (AIMNet2-Pd).
    coulomb_sr_cutoff : float, optional
        Radius within which the implicitly learned Coulomb energy is
        subtracted, by default 4.6 Angstrom.
    coulomb_sr_envelope : str, optional
        Its envelope, ``"exp"`` (the mollifier, default) or ``"cosine"``.
    lr_cutoff : float, optional
        Real-space cutoff of the long-range methods, by default 15.0 Angstrom.
    dsf_alpha : float, optional
        Damping parameter of ``coulomb="dsf"``, by default 0.2 per Angstrom.
    ewald_accuracy : float, optional
        Target relative accuracy of the Ewald and PME sums, by default
        ``1e-6``; sets the splitting parameter and the reciprocal cutoff
        (:func:`~xnn.common.models.electrostatics.ewald_parameters`) and the
        PME mesh.
    pme_spline_order : int, optional
        B-spline order of the PME charge assignment, by default 4 (cubic).
    atomic_energies : array-like or None, optional
        Per-element energy shifts, one per entry of ``species``.

    Raises
    ------
    ValueError
        For fewer than two passes, an unknown ``coulomb`` method or
        envelope, ``charge_channels`` outside ``{1, 2}``, an accuracy
        outside ``(0, 1)`` or a spline order below 3.
    """

    coulomb: Optional[str]
    aev_cutoff: float
    cutoff: float
    coulomb_sr_cutoff: float
    lr_cutoff: float
    dsf_alpha: float
    ewald_accuracy: float
    ewald_alpha: float
    ewald_k_cutoff: float
    pme_spline_order: int
    coulomb_constant: float
    n_features: int
    n_rbf: int
    charge_channels: int
    aim_size: int

    # one readout head = the energy MLP and the reference energies (the charge
    # passes stay shared; see MultiHead)
    head_modules = ("readout", "atom_ref")

    def __init__(
        self,
        species: list[int],
        cutoff: float = 5.0,
        n_features: int = 16,
        n_rbf: int = 16,
        rbf_start: float = 0.8,
        gaussian_width: Optional[float] = None,
        hidden: Optional[Sequence[Sequence[int]]] = None,
        n_passes: int = 3,
        n_vector_combinations: int = 12,
        aim_size: int = 256,
        readout_hidden: Sequence[int] = (128, 128),
        charge_channels: int = 1,
        coulomb: Optional[str] = "simple",
        coulomb_sr_cutoff: float = 4.6,
        coulomb_sr_envelope: str = "exp",
        lr_cutoff: float = 15.0,
        dsf_alpha: float = 0.2,
        ewald_accuracy: float = 1.0e-6,
        pme_spline_order: int = 4,
        atomic_energies=None,
    ):
        super().__init__(species, cutoff)
        # registered again below, after the network, so that the first
        # floating parameter (what sets the model's dtype when it is served)
        # is a network weight and the float64 shifts stay a special case
        del self.atom_ref
        if hidden is None:
            if n_passes < 2:
                raise ValueError("AIMNet2 needs at least two passes (charges, then AIM)")
            hidden = [[512, 380]] * (n_passes - 1) + [[512, 380, 380]]
        hidden = [list(h) for h in hidden]
        if len(hidden) < 2:
            raise ValueError("AIMNet2 needs at least two passes (charges, then AIM)")
        if charge_channels not in (1, 2):
            raise ValueError("charge_channels must be 1 (closed shell) or 2 (NSE)")
        if coulomb not in COULOMB_METHODS:
            raise ValueError(f"coulomb must be one of {COULOMB_METHODS}, got {coulomb!r}")
        if coulomb_sr_envelope not in SR_ENVELOPES:
            raise ValueError(f"coulomb_sr_envelope must be one of {sorted(SR_ENVELOPES)}")
        if not 0.0 < float(ewald_accuracy) < 1.0:
            raise ValueError(f"ewald_accuracy must lie in (0, 1), got {ewald_accuracy}")
        if int(pme_spline_order) < 3:
            raise ValueError(f"pme_spline_order must be at least 3, got {pme_spline_order}")

        self.aev_cutoff = float(cutoff)
        self.n_features = int(n_features)
        self.n_rbf = int(n_rbf)
        self.charge_channels = int(charge_channels)
        self.aim_size = int(aim_size)
        self.node_feature_dim = int(aim_size)
        self.coulomb = coulomb
        self.coulomb_sr_cutoff = float(coulomb_sr_cutoff)
        self.lr_cutoff = float(lr_cutoff)
        self.dsf_alpha = float(dsf_alpha)
        self.ewald_accuracy = float(ewald_accuracy)
        self.pme_spline_order = int(pme_spline_order)
        self.ewald_alpha, self.ewald_k_cutoff = ewald_parameters(self.lr_cutoff, self.ewald_accuracy)
        self.coulomb_constant = COULOMB_CONSTANT
        # the neighbor list reaches the Coulomb cutoff when a long-range sum needs it
        self.cutoff = (max(self.aev_cutoff, self.lr_cutoff) if coulomb in LONG_RANGE_METHODS
                       else self.aev_cutoff)

        if gaussian_width is None:
            gaussian_width = (n_rbf / (cutoff - rbf_start)) ** 2
        self.rbf = GaussianRBF(n_rbf, cutoff, gamma=gaussian_width, start=rbf_start,
                               endpoint=False)
        self.envelope = CosineCutoff(cutoff)

        # one embedding vector per element, repeated over the radial shells
        # (orthogonal across elements, as the reference code initializes it)
        n_species = len(self.species)
        embed = torch.empty(n_species, n_features)
        if n_species > 1:
            nn.init.orthogonal_(embed)
        else:
            nn.init.normal_(embed)
        self.embedding = nn.Embedding(n_species, n_features * n_rbf)
        with torch.no_grad():
            self.embedding.weight.copy_(embed.repeat_interleave(n_rbf, dim=1))

        self.conv_a = ShellConvolution(n_features, n_rbf, n_vector_combinations)
        self.conv_q = ShellConvolution(charge_channels, n_rbf, n_vector_combinations)

        n_embed = n_features * n_rbf
        in_first = n_embed + self.conv_a.output_dim
        in_rest = in_first + charge_channels + self.conv_q.output_dim
        n_update = n_embed + 2 * charge_channels
        passes = [_mlp([in_first, *hidden[0], n_update], final_activation=False)]
        for widths in hidden[1:-1]:
            passes.append(_mlp([in_rest, *widths, n_update], final_activation=True))
        passes.append(_mlp([in_rest, *hidden[-1], aim_size], final_activation=True))
        self.passes = nn.ModuleList(passes)
        self.readout = _mlp([aim_size, *readout_hidden, 1], final_activation=False)
        self.sr_envelope = SR_ENVELOPES[coulomb_sr_envelope](self.coulomb_sr_cutoff)

        # the per-element shifts, O(1000 eV): float64 whatever the model dtype
        self.atom_ref = nn.Embedding(200, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.atom_ref.double()
        if atomic_energies is not None:
            self.set_atomic_energies(atomic_energies)

    def _apply(self, fn, recurse=True):
        # a cast of the model (float32 serving, half precision) leaves the
        # shifts in float64 with their exact values (the parent's cast would
        # round them to the new precision first: 1e-4 eV on a 2000 eV shift,
        # 3e-3 eV on iodine's); a device move keeps them where the model went
        exact = self.atom_ref.weight.data
        out = super()._apply(fn, recurse)
        weight = self.atom_ref.weight
        if weight.dtype != torch.float64:
            weight.data = exact.to(device=weight.device, dtype=torch.float64)
        return out

    def equilibrate(self, charges: Tensor, weights: Tensor, total: Tensor, batch: Tensor,
                    num_graphs: int, eps: float = NQE_EPSILON) -> Tensor:
        """Neural charge equilibration (paper eq 6).

        Parameters
        ----------
        charges : Tensor
            Charges before equilibration ``(N, C)``.
        weights : Tensor
            Non-negative weights ``f`` ``(N, C)``.
        total : Tensor
            Target total per structure and channel ``(B, C)``.
        batch : Tensor
            Structure index per atom ``(N,)``.
        num_graphs : int
            Number of structures ``B``.
        eps : float, optional
            Regularization of the weight sum, by default :data:`NQE_EPSILON`.

        Returns
        -------
        Tensor
            Charges ``(N, C)`` whose structure sums equal ``total`` (up to the
            regularization of the weight sum).
        """
        weight_sum = scatter_sum(weights, batch, num_graphs) + eps
        surplus = total - scatter_sum(charges, batch, num_graphs)
        return charges + weights / weight_sum[batch] * surplus[batch]

    def channel_totals(self, total_charge: Tensor, spin_multiplicity: Tensor) -> Tensor:
        """The net charge of every structure split over the charge channels, ``(B, C)``.

        One channel carries ``Q``; two channels carry ``(Q +- (M - 1)) / 2``.

        Parameters
        ----------
        total_charge : Tensor
            Net charges ``(B,)``.
        spin_multiplicity : Tensor
            Multiplicities ``(B,)`` (read by the two-channel models only).
        """
        if self.charge_channels == 1:
            return total_charge[:, None]
        half_spin = 0.5 * (spin_multiplicity - 1.0)
        return torch.stack([0.5 * total_charge + half_spin, 0.5 * total_charge - half_spin],
                           dim=-1)

    def _channel_totals(self, data: AtomicGraph, dtype: torch.dtype) -> Tensor:
        """:meth:`channel_totals` of a graph (``None`` = neutral, singlet)."""
        b = data.num_graphs
        q = data.total_charge
        q = (torch.zeros(b, dtype=dtype, device=data.pos.device) if q is None
             else q.to(dtype))
        m = data.spin_multiplicity
        m = (torch.ones(b, dtype=dtype, device=data.pos.device) if m is None
             else m.to(dtype))
        return self.channel_totals(q, m)

    def _coulomb_long_range(self, charges: Tensor, pos: Tensor, cell: Optional[Tensor],
                            pbc: Optional[Tensor], batch: Tensor, num_graphs: int,
                            edge_index: Tensor, r: Tensor) -> Tensor:
        """Per-atom Coulomb energies of the predicted charges by the chosen method (eV, float64)."""
        method = self.coulomb
        assert method is not None
        periodic = torch.zeros(charges.shape[0], dtype=torch.bool, device=charges.device)
        if cell is not None and pbc is not None:
            periodic = pbc.any(dim=1)[batch]
        if method == "simple":
            if bool(periodic.any()):
                raise ValueError("coulomb='simple' sums every pair of a molecule; periodic "
                                 "structures need coulomb='dsf', 'ewald' or 'pme'")
            return coulomb_direct(charges, pos, batch)
        if method == "dsf":
            return coulomb_dsf(charges, edge_index, r, self.dsf_alpha, self.lr_cutoff)
        # ewald / pme: the lattice sum for the periodic structures of the
        # batch, the all-pairs sum for the others
        node = torch.zeros(charges.shape[0], dtype=torch.float64, device=charges.device)
        if cell is not None and pbc is not None:
            for i in range(num_graphs):
                flags = pbc[i]
                if not bool(flags.any()):
                    continue
                if not bool(flags.all()):
                    raise ValueError("coulomb='ewald'/'pme' need structures periodic in all "
                                     "three directions; use coulomb='dsf' for slabs and wires")
                mask = batch == i
                idx = mask.nonzero().squeeze(1)
                on_edge = mask[edge_index[0]]
                remap = torch.cumsum(mask.to(torch.long), 0) - 1
                local = remap[edge_index[:, on_edge]]
                if method == "ewald":
                    part = coulomb_ewald(charges[idx], pos[idx], cell[i], local, r[on_edge],
                                         self.ewald_alpha, self.lr_cutoff, self.ewald_k_cutoff)
                else:
                    part = coulomb_pme(charges[idx], pos[idx], cell[i], local, r[on_edge],
                                       self.ewald_alpha, self.lr_cutoff, self.ewald_accuracy,
                                       self.pme_spline_order)
                node = node.index_add(0, idx, part)
        free = ~periodic
        if bool(free.any()):
            idx = free.nonzero().squeeze(1)
            node = node.index_add(0, idx, coulomb_direct(charges[idx], pos[idx], batch[idx]))
        return node

    def _evaluate(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                  pos: Tensor, cell: Optional[Tensor], pbc: Optional[Tensor], batch: Tensor,
                  num_graphs: int, totals: Tensor) -> Dict[str, Tensor]:
        """The network and the electrostatics on raw tensors (scriptable).

        Parameters
        ----------
        atomic_numbers : Tensor
            ``(N,)``.
        edge_index : Tensor
            ``(2, E)`` as ``[src, dst]`` within ``self.cutoff`` (both
            directions, periodic images included).
        edge_vec : Tensor
            ``pos[dst] - pos[src]`` (plus the image shift), ``(E, 3)``, in the
            model dtype.
        pos : Tensor
            Positions ``(N, 3)`` (any float dtype).
        cell, pbc : Tensor or None
            ``(B, 3, 3)`` lattice vectors as rows and ``(B, 3)`` flags.
        batch : Tensor
            Structure index per atom ``(N,)``.
        num_graphs : int
            ``B``.
        totals : Tensor
            Channel targets of the charge equilibration ``(B, C)``.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` and ``"node_coulomb"`` ``(N,)`` in float64,
            ``"node_features"`` ``(N, aim_size)``, ``"charges"`` ``(N,)`` and,
            for two channels, ``"spin_charges"`` ``(N,)``.
        """
        index = self.z_to_index[atomic_numbers]
        if bool((index < 0).any()):
            raise ValueError("an element of the structure is not among the model's species")
        num_nodes = atomic_numbers.shape[0]

        # the local environment: edges within the Gaussian-basis cutoff (the
        # graph may reach further for the long-range Coulomb sum)
        r_all = torch.linalg.norm(edge_vec, dim=-1)
        if self.cutoff > self.aev_cutoff:
            local = r_all < self.aev_cutoff
            edge_local, vec_local, r = edge_index[:, local], edge_vec[local], r_all[local]
        else:
            edge_local, vec_local, r = edge_index, edge_vec, r_all
        src, dst = edge_local[0], edge_local[1]
        g = self.rbf(r) * self.envelope(r)[:, None]                        # (E, S)
        unit = -vec_local / r[:, None]                                       # r_j - r_i
        gu = g[:, :, None] * unit[:, None, :]                                # (E, S, 3)

        n_feat, n_rbf, n_ch = self.n_features, self.n_rbf, self.charge_channels
        a = self.embedding(index).view(num_nodes, n_feat, n_rbf)
        q: Optional[Tensor] = None
        aim = a.new_zeros((num_nodes, self.aim_size))
        last = len(self.passes) - 1
        for i, mlp in enumerate(self.passes):
            parts = [a.flatten(1), self.conv_a(a[src], g, gu, dst, num_nodes)]
            if q is not None:
                parts.append(q)
                parts.append(self.conv_q(q[src][:, :, None].expand(-1, -1, n_rbf),
                                         g, gu, dst, num_nodes))
            out = mlp(torch.cat(parts, dim=-1))
            if i == last:
                aim = out
            else:
                pieces = out.split([n_ch, n_ch, n_feat * n_rbf], dim=-1)
                dq, f, da = pieces[0], pieces[1], pieces[2]
                if q is None:
                    q_pre = dq
                else:
                    q_pre = q + dq
                q = self.equilibrate(q_pre, f.square(), totals, batch, num_graphs)
                a = a + da.view(num_nodes, n_feat, n_rbf)
        assert q is not None
        charges = q.sum(-1)

        node_energy = (self.readout(aim).squeeze(-1).to(torch.float64)
                       + self.atom_ref(atomic_numbers).squeeze(-1))
        node_coulomb = torch.zeros_like(node_energy)
        if self.coulomb is not None:
            # the network has learned the short-range Coulomb energy: take it
            # out under the envelope and add the full (or truncated) sum
            pair = (0.5 * self.coulomb_constant * self.sr_envelope(r)
                    * charges[src] * charges[dst] / r)
            node_coulomb = -scatter_sum(pair.to(torch.float64), dst, num_nodes)
            node_coulomb = node_coulomb + self._coulomb_long_range(
                charges, pos, cell, pbc, batch, num_graphs, edge_index, r_all)
            node_energy = node_energy + node_coulomb

        result = {"node_energy": node_energy, "node_coulomb": node_coulomb,
                  "node_features": aim, "charges": charges}
        if n_ch == 2:
            result["spin_charges"] = q[:, 0] - q[:, 1]
        return result

    @torch.jit.export
    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor,
                                     edge_vec: Tensor, pos: Tensor, cell: Tensor, pbc: Tensor,
                                     total_charge: float,
                                     spin_multiplicity: float) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript core of one structure: AIM features, atomic energies and charges.

        The entry point :class:`~xnn.common.deploy.TorchScriptPotential`
        exports; the net charge and spin multiplicity are fixed at export.

        Parameters
        ----------
        atomic_numbers : Tensor
            ``(N,)``.
        edge_index : Tensor
            ``(2, E)`` within ``self.cutoff``.
        edge_vec : Tensor
            ``(E, 3)``.
        pos : Tensor
            ``(N, 3)``.
        cell : Tensor
            ``(3, 3)`` (ignored unless ``pbc`` has a true entry).
        pbc : Tensor
            ``(3,)`` bool.
        total_charge : float
            Net charge of the structure.
        spin_multiplicity : float
            Its spin multiplicity (two-channel models).

        Returns
        -------
        tuple of Tensor
            ``node_features (N, aim_size)``, ``node_energy (N,)`` (float64,
            Coulomb energy included) and ``charges (N,)``.
        """
        n = atomic_numbers.shape[0]
        device = atomic_numbers.device
        dtype = self.embedding.weight.dtype
        batch = torch.zeros(n, dtype=torch.long, device=device)
        totals = self.channel_totals(torch.full((1,), total_charge, dtype=dtype, device=device),
                                     torch.full((1,), spin_multiplicity, dtype=dtype,
                                                device=device))
        cell_b: Optional[Tensor] = None
        pbc_b: Optional[Tensor] = None
        if bool(pbc.any()):
            cell_b = cell.unsqueeze(0)
            pbc_b = pbc.unsqueeze(0)
        out = self._evaluate(atomic_numbers, edge_index, edge_vec, pos, cell_b, pbc_b, batch, 1,
                             totals)
        return out["node_features"], out["node_energy"], out["charges"]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict energies, partial charges and the dipole of a (batched) graph.

        Parameters
        ----------
        data : xnn.common.data.AtomicGraph
            The input graph, built with ``self.cutoff``. ``total_charge``
            (``None`` = neutral) sets the net charge the partial charges sum
            to; ``spin_multiplicity`` (``None`` = 1) is read by the two-channel
            (NSE) models.

        Returns
        -------
        dict of str to torch.Tensor
            ``"node_energy"`` ``(N,)`` and ``"energy"`` ``(B,)`` in float64
            (as every xnn model), ``"node_features"`` ``(N, aim_size)`` (the
            AIM vector), ``"charges"`` ``(N,)``, ``"dipole"`` ``(B, 3)`` in
            e Angstrom, ``"energy_coulomb"`` ``(B,)`` (the electrostatic
            part of the energy; zeros without a Coulomb term) and, for the
            two-channel models, ``"spin_charges"`` ``(N,)``.

        Raises
        ------
        ValueError
            For an element outside ``species``, a periodic structure with
            ``coulomb="simple"`` (use ``"dsf"``, ``"ewald"`` or ``"pme"``),
            or a partially periodic structure with ``"ewald"`` / ``"pme"``.
        """
        z = data.atomic_numbers
        index = self.z_to_index[z]
        if bool((index < 0).any()):
            missing = sorted(set(z[index < 0].tolist()))
            raise ValueError(f"elements {missing} are not among the model's species "
                             f"{self.species}")
        num_graphs, batch = data.num_graphs, data.batch
        dtype = self.embedding.weight.dtype
        totals = self._channel_totals(data, dtype)                           # (B, C)
        out = self._evaluate(z, data.edge_index, data.edge_vectors(), data.pos, data.cell,
                             data.pbc, batch, num_graphs, totals)
        charges = out["charges"]
        result = {
            "node_energy": out["node_energy"],
            "energy": self.aggregate_energy(out["node_energy"], data),
            "node_features": out["node_features"],
            "charges": charges,
            "dipole": scatter_sum(charges[:, None] * data.pos.to(dtype), batch, num_graphs),
            "energy_coulomb": structure_sum(out["node_coulomb"], batch, num_graphs),
        }
        if "spin_charges" in out:
            result["spin_charges"] = out["spin_charges"]
        return result

    @classmethod
    def from_config(cls, cfg) -> "AIMNet2":
        """Construct an :class:`AIMNet2` from a core model config.

        Core fields: ``cfg.cutoff`` (the local cutoff), ``cfg.n_features``
        (embedding channels), ``cfg.n_rbf`` (radial shells) and
        ``cfg.n_interactions`` (message passes, when ``extra["hidden"]``
        does not list the MLP widths of every pass). The remaining options
        are the constructor arguments, read from ``cfg.extra``; the
        reference code's spellings (``nfeature``, ``nshifts_s``, ``rc_s``,
        ``ncomb_v``, ``num_charge_channels``) are translated at config-load
        time by :mod:`xnn.common.config.translate`.

        Alternatively ``extra["foundation"]`` names a published AIMNet2
        checkpoint (see :meth:`from_foundation`); the bare network is
        returned and every architecture key comes from the checkpoint, while
        the Coulomb options (``coulomb``, ``lr_cutoff``, ``dsf_alpha``,
        ``ewald_accuracy``, ``pme_spline_order``) may be overridden.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        AIMNet2
            The instantiated model.

        Raises
        ------
        ValueError
            If ``cfg.cutoff`` differs from a foundation checkpoint's
            neighbor-list cutoff.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        foundation = extra.get("foundation")
        if foundation is not None:
            options = {k: extra[k] for k in COULOMB_OPTIONS if k in extra}
            model = cls.from_foundation(foundation, dtype=extra.get("dtype"), dispersion=False,
                                        **options)
            if abs(float(cfg.cutoff) - float(model.cutoff)) > 1e-9:
                raise ValueError(
                    f"model.cutoff={cfg.cutoff} does not match the foundation "
                    f"checkpoint's neighbor-list cutoff {model.cutoff}; set "
                    f"cutoff: {model.cutoff} in the config")
            return model
        species = coerce_species(extra.get("species"), default=[1, 6, 7, 8])
        atomic_energies = coerce_per_species(
            extra.get("atomic_energies"), species, "atomic_energies (AIMNet2 atomic shifts)")

        def seq(value):
            return ast.literal_eval(value) if isinstance(value, str) else value

        hidden = seq(extra.get("hidden"))
        return cls(
            species=species,
            cutoff=cfg.cutoff,
            n_features=cfg.n_features,
            n_rbf=cfg.n_rbf,
            rbf_start=float(extra.get("rbf_start", 0.8)),
            gaussian_width=extra.get("gaussian_width"),
            hidden=hidden,
            n_passes=int(cfg.n_interactions),
            n_vector_combinations=int(extra.get("n_vector_combinations", 12)),
            aim_size=int(extra.get("aim_size", 256)),
            readout_hidden=seq(extra.get("readout_hidden", (128, 128))),
            charge_channels=int(extra.get("charge_channels", 1)),
            coulomb=extra.get("coulomb", "simple"),
            coulomb_sr_cutoff=float(extra.get("coulomb_sr_cutoff", 4.6)),
            coulomb_sr_envelope=extra.get("coulomb_sr_envelope", "exp"),
            lr_cutoff=float(extra.get("lr_cutoff", 15.0)),
            dsf_alpha=float(extra.get("dsf_alpha", 0.2)),
            ewald_accuracy=float(extra.get("ewald_accuracy", 1.0e-6)),
            pme_spline_order=int(extra.get("pme_spline_order", 4)),
            atomic_energies=atomic_energies,
        )

    @classmethod
    def from_foundation(cls, source, dtype=None, cache_dir=None, dispersion=None,
                        **model_options):
        """Load a published AIMNet2 model.

        Downloads (and caches) the requested checkpoint if needed and
        converts it weight for weight into this implementation; see
        :mod:`xnn.gnn.models.aimnet2_foundation` for the registry and the
        conversion. The published models are served with the D3(BJ)
        dispersion they were trained without, so by default the returned
        potential is the network wrapped in the shared
        :class:`~xnn.common.models.d3.D3Dispersion` term.

        Parameters
        ----------
        source : str, Path or Mapping
            A registered name or alias (``"aimnet2"``, ``"aimnet2-nse"``,
            ``"aimnet2-wb97m-d3-2"``, ...; see
            :data:`~xnn.gnn.models.aimnet2_foundation.FOUNDATION_MODELS`),
            a checkpoint URL or local path (the reference ``.pt`` artifact,
            or a directory of ``config.json`` + ``ensemble_<k>.safetensors``),
            or an already-loaded artifact mapping.
        dtype : torch.dtype or str, optional
            Final dtype; ``None`` keeps the checkpoint's float32 (the
            per-element shifts stay float64 either way).
        cache_dir : str or Path, optional
            Model hub cache directory.
        dispersion : dict, str, bool or None, optional
            ``None`` (default) adds the D3(BJ) term recorded for the model,
            ``False`` returns the bare :class:`AIMNet2`; see
            :func:`~xnn.common.models.hub.load_pretrained`.
        **model_options
            Deployment knobs applied before the model is built, e.g.
            ``coulomb="dsf"`` with ``lr_cutoff=15.0``, or ``coulomb="ewald"``
            / ``"pme"`` with ``ewald_accuracy=1e-6``, for periodic
            structures.

        Returns
        -------
        InteratomicPotential
            The :class:`AIMNet2` model, inside its dispersion term unless
            ``dispersion=False``.
        """
        from .aimnet2_foundation import foundation_to_xnn
        return foundation_to_xnn(source, dtype=dtype, cache_dir=cache_dir,
                                 dispersion=dispersion, **model_options)


#: The constructor options a deployment may override on a published model.
COULOMB_OPTIONS = ("coulomb", "lr_cutoff", "dsf_alpha", "ewald_accuracy", "pme_spline_order")
