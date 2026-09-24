"""PhysNet (Unke & Meuwly 2019): message-passing HDNN with explicit physics.

A faithful, self-contained PyTorch translation of the original TensorFlow
implementation (https://github.com/MMunibas/PhysNet,
``neural_network/NeuralNetwork.py`` and ``layers/``) on the xnn abstractions:
it subclasses :class:`~xnn.common.models.base.InteratomicPotential` directly
(PhysNet is a message-passing high-dimensional NN -- it has no hand-crafted
descriptor, so :class:`~xnn.dnn.models.base.DescriptorPotential` does not
apply) and gets forces/stress from the shared
:class:`~xnn.common.models.outputs.ForceStressOutput`. Given the same weights
it reproduces the original TensorFlow graph to machine precision
(``tests/test_physnet.py`` and the block-by-block notebook).

Architecture (paper eqs 3-15, J. Chem. Theory Comput. 15, 3678, 2019):

* nuclear charges are embedded into ``F``-vectors (a 95-row table indexed by
  ``Z`` directly -- all elements up to Pu, no species list needed; eq 3);
* distances are expanded in ``K`` radial basis functions
  ``g_k(r) = phi(r) exp(-beta_k (exp(-r) - mu_k)^2)`` with learnable centers
  and widths (softplus-reparametrized for positivity) and the smooth cutoff
  ``phi`` (eqs 7-8);
* ``num_blocks`` modules refine the features: an interaction layer computes
  the message ``v`` from gated features and the distance-based attention mask
  ``G g(r_ij)`` (eqs 5-6), followed by pre-activation residual blocks (eq 4);
* every module feeds an output block whose zero-initialized linear head
  predicts per-atom energy and partial-charge contributions; module outputs
  are summed and scaled/shifted per element (eqs 9-10);
* predicted charges are corrected to the exact total charge (eq 14) and enter
  a damped/switched Coulomb term (eqs 12-13 -- the code form: shielded
  ``1/sqrt(r^2+1)`` below ``sr_cut/2``, smoothstep-switched to ``1/r``, and
  force-shifted at ``lr_cutoff`` when one is set);
* Grimme D3(BJ) dispersion (:mod:`~xnn.common.models.d3`, an independent
  implementation verified against the upstream TF module, tables included)
  with optionally learnable ``s6/s8/a1/a2`` completes the total energy
  (eq 12).

Upstream conventions preserved: shifted-softplus activation, semi-orthogonal
Glorot weight init with zero biases, zero-initialized ``k2f``/output heads,
per-element scale/shift tables of length 95, ``kehalf`` Coulomb constant in
eV*Angstrom units, and the non-hierarchicality penalty returned as
``"nh_loss"``. Dropout (upstream ``keep_prob``, default 1.0 = off) is not
implemented.

The neighbor-list radius (``self.cutoff``) is ``lr_cutoff`` when set,
otherwise ``sr_cut``: radial-basis features vanish identically beyond
``sr_cut`` because of the ``phi`` envelope, so feeding the longer-range edge
list to the interaction blocks is mathematically identical to upstream's
separate short-range index list. Without ``lr_cutoff`` upstream evaluates
electrostatics/dispersion over *all* pairs; in xnn the pair list is the
graph's, so set ``lr_cutoff`` (or a large ``cutoff``) to capture long-range
terms explicitly.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from xnn.common.data import AtomicGraph
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import scatter_sum, shifted_softplus
from xnn.common.models.registry import register_model
from ...common.models import d3

MAX_Z = 95  # element-indexed tables cover Z = 0..94 (through Pu)
KEHALF = 7.199822675975274  # ke/2 in eV*A/e^2; halved since edges come in pairs


def softplus_inverse(x):
    """Return ``y`` such that ``softplus(y) = x``.

    Evaluated as ``x + log(1 - exp(-x))`` (i.e. ``log(expm1(x))`` rearranged
    so the exponential never overflows for large ``x``).
    """
    return x + np.log(-np.expm1(-x))


# PhysNet's activation ``log(exp(x) + 1) - log(2)`` is the shared exact
# shifted softplus (matches TF's softplus bit-for-bit); re-exported here so
# ``from xnn.dnn.models.physnet import shifted_softplus`` keeps working.


def semi_orthogonal_glorot_weights(n_in: int, n_out: int,
                                   scale: float = 2.0) -> Tensor:
    """Random (semi-)orthogonal weights rescaled to Glorot variance.

    Port of upstream ``layers/util.py``: a random orthogonal matrix (QR of a
    standard-normal matrix) cropped to ``(n_in, n_out)`` and rescaled so its
    entries have variance ``scale / (n_in + n_out)``.

    Returns
    -------
    Tensor
        Weight matrix of shape ``(n_in, n_out)`` in the default dtype.
    """
    dim = max(n_in, n_out)
    q, r = torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))  # uniform over the orthogonal group
    w = q[:n_in, :n_out]
    w = w * torch.sqrt(scale / ((n_in + n_out) * w.var()))
    return w.to(torch.get_default_dtype())


class _Dense(nn.Module):
    """Upstream ``DenseLayer``: linear with semi-orthogonal Glorot init, zero
    bias, and an optional activation applied after."""

    def __init__(self, n_in: int, n_out: int, activation: bool = False,
                 use_bias: bool = True, zero_init: bool = False):
        super().__init__()
        if zero_init:
            weight = torch.zeros(n_in, n_out)
        else:
            weight = semi_orthogonal_glorot_weights(n_in, n_out)
        self.weight = nn.Parameter(weight)  # (n_in, n_out), upstream layout
        self.bias = nn.Parameter(torch.zeros(n_out)) if use_bias else None
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        if self.activation:
            y = shifted_softplus(y)
        return y


class _Residual(nn.Module):
    """Pre-activation residual block ``x + W2 act(W1 act(x) + b1) + b2``
    (paper eq 4, upstream ``ResidualLayer``)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.dense = _Dense(n_features, n_features, activation=True)
        self.residual = _Dense(n_features, n_features, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.residual(self.dense(shifted_softplus(x)))


class _RBF(nn.Module):
    """Radial basis ``g_k(r) = phi(r) exp(-beta_k (exp(-r) - mu_k)^2)``
    (paper eqs 7-8, upstream ``RBFLayer``).

    Centers ``mu_k`` (equally spaced on ``[exp(-cutoff), 1]``) and the shared
    width are stored pre-softplus so positivity is guaranteed while training.
    """

    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        # initialization convention: the (post-softplus) centers tile
        # [exp(-cutoff), 1], the range exp(-r) sweeps on [0, cutoff], on a
        # uniform grid; all basis functions start from one shared width
        # beta = 1/(2 delta)^2 with delta = (1 - exp(-cutoff)) / n_rbf, so
        # neighboring Gaussians overlap at about half height
        grid = np.linspace(1.0, np.exp(-cutoff), n_rbf)
        delta = (1.0 - np.exp(-cutoff)) / n_rbf
        self.centers = nn.Parameter(torch.as_tensor(
            softplus_inverse(grid), dtype=torch.get_default_dtype()))
        self.widths = nn.Parameter(torch.full(
            (n_rbf,), float(softplus_inverse((0.5 / delta) ** 2)),
            dtype=torch.get_default_dtype()))

    def cutoff_fn(self, r: Tensor) -> Tensor:
        """Smooth cutoff ``phi(r) = 1 - 6x^5 + 15x^4 - 10x^3`` (paper eq 8)."""
        x = r / self.cutoff
        env = 1 - 6 * x ** 5 + 15 * x ** 4 - 10 * x ** 3
        return torch.where(x < 1, env, torch.zeros_like(x))

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances ``(E,)`` into ``(E, n_rbf)``."""
        r = r.unsqueeze(-1)
        g = torch.exp(-F.softplus(self.widths)
                      * (torch.exp(-r) - F.softplus(self.centers)) ** 2)
        return self.cutoff_fn(r) * g


class _InteractionLayer(nn.Module):
    """The message computation (paper eqs 5-6, upstream ``InteractionLayer``).

    ``x' = u * x + W act(m)``, where the proto-message sums the transformed
    central-atom features and the attention-masked neighbor features
    ``m~ = act(W_I act(x_i) + b_I) + sum_j G g(r_ij) * act(W_J act(x_j) + b_J)``
    and is refined by residual blocks. ``G`` (``k2f``) is zero-initialized so
    messages initially see only the central atom.
    """

    def __init__(self, n_rbf: int, n_features: int, num_residual: int):
        super().__init__()
        self.k2f = _Dense(n_rbf, n_features, use_bias=False, zero_init=True)
        self.dense_i = _Dense(n_features, n_features, activation=True)
        self.dense_j = _Dense(n_features, n_features, activation=True)
        self.residuals = nn.ModuleList(
            [_Residual(n_features) for _ in range(num_residual)])
        self.dense = _Dense(n_features, n_features)
        self.u = nn.Parameter(torch.ones(n_features))

    def forward(self, x: Tensor, rbf: Tensor, idx_i: Tensor,
                idx_j: Tensor) -> Tensor:
        xa = shifted_softplus(x)
        g = self.k2f(rbf)
        m = self.dense_i(xa) + scatter_sum(
            g * self.dense_j(xa)[idx_j], idx_i, x.shape[0])
        for residual in self.residuals:
            m = residual(m)
        return self.u * x + self.dense(shifted_softplus(m))


class _InteractionBlock(nn.Module):
    """Interaction layer + atom-wise residual refinements (paper fig 1B/C)."""

    def __init__(self, n_rbf: int, n_features: int, num_residual_atomic: int,
                 num_residual_interaction: int):
        super().__init__()
        self.interaction = _InteractionLayer(n_rbf, n_features,
                                             num_residual_interaction)
        self.residuals = nn.ModuleList(
            [_Residual(n_features) for _ in range(num_residual_atomic)])

    def forward(self, x: Tensor, rbf: Tensor, idx_i: Tensor,
                idx_j: Tensor) -> Tensor:
        x = self.interaction(x, rbf, idx_i, idx_j)
        for residual in self.residuals:
            x = residual(x)
        return x


class _OutputBlock(nn.Module):
    """Residual refinements + zero-initialized linear head predicting the
    per-atom ``(energy, charge)`` contribution of one module (paper eq 9)."""

    def __init__(self, n_features: int, num_residual: int):
        super().__init__()
        self.residuals = nn.ModuleList(
            [_Residual(n_features) for _ in range(num_residual)])
        self.dense = _Dense(n_features, 2, use_bias=False, zero_init=True)

    def forward(self, x: Tensor) -> Tensor:
        for residual in self.residuals:
            x = residual(x)
        return self.dense(shifted_softplus(x))


@register_model("physnet")
class PhysNet(InteratomicPotential):
    """Faithful PhysNet (Unke & Meuwly 2019): energies, forces, and charges.

    See the module docstring for the architecture walk-through. All options
    are read from ``ModelConfig.extra`` (see :meth:`from_config`); upstream
    ``train.py`` spellings are translated by
    :mod:`xnn.common.config.translate`.

    Parameters
    ----------
    cutoff : float, optional
        Short-range cutoff ``sr_cut`` of the neural-network interactions and
        the radial basis, by default 10.0 (the paper's value).
    lr_cutoff : float or None, optional
        Long-range cutoff for the electrostatic/dispersion terms (upstream
        ``lr_cut``); the Coulomb term is force-shifted so energy and forces
        vanish smoothly there. ``None`` (default) evaluates the long-range
        terms un-damped on the graph's edge list.
    n_features : int, optional
        Feature-space width ``F``, by default 128.
    n_rbf : int, optional
        Number of radial basis functions ``K``, by default 64.
    num_blocks : int, optional
        Number of stacked module blocks, by default 5 (the paper; the
        upstream code default is 3).
    num_residual_atomic : int, optional
        Residual blocks for atom-wise refinements, by default 2.
    num_residual_interaction : int, optional
        Residual blocks refining the proto-message, by default 3 (the paper;
        the upstream code default is 2).
    num_residual_output : int, optional
        Residual blocks in the output blocks, by default 1.
    use_electrostatics : bool, optional
        Add the switched/shielded Coulomb energy of the predicted partial
        charges (paper eqs 12-13), by default ``True``.
    use_dispersion : bool, optional
        Add Grimme D3(BJ) dispersion, by default ``True``.
    s6, s8, a1, a2 : float or None, optional
        D3(BJ) parameters. ``None`` (default) makes them learnable
        (softplus-reparametrized, initialized to the HF values), a number
        fixes them.
    d3_references : str, optional
        D3 reference systems: ``"2010"`` (default; Grimme's original tables,
        as in upstream PhysNet) or ``"2024"`` (the current ``simple-dftd3``
        references, which re-parametrize Fr-Pu). Identical for Z <= 86; see
        :func:`xnn.common.models.d3.legacy_c6_table`.
    energy_shift, energy_scale : float, optional
        Initial value of the per-element energy shift/scale tables
        (upstream ``Eshift``/``Escale``), by default 0 and 1.
    charge_shift, charge_scale : float, optional
        Initial value of the per-element charge shift/scale tables, by
        default 0 and 1.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``/``atomic_scales``; the
        model itself handles all elements up to Z = 94.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``Eshift`` (aligned with
        ``species``), like upstream's dataset-regression initialization.
    atomic_scales : array-like or None, optional
        Per-species initial ``Escale`` values (aligned with ``species``).

    Notes
    -----
    ``forward`` additionally returns ``"charges"`` (corrected partial
    charges, summing exactly to the total charge -- 0 unless the graph
    carries a ``total_charge`` attribute), ``"dipole"`` (eq 15) and
    ``"nh_loss"`` (the non-hierarchicality penalty, paper eq 18/19) for
    training-loop use.
    """

    def __init__(
        self,
        cutoff: float = 10.0,
        lr_cutoff: float | None = None,
        n_features: int = 128,
        n_rbf: int = 64,
        num_blocks: int = 5,
        num_residual_atomic: int = 2,
        num_residual_interaction: int = 3,
        num_residual_output: int = 1,
        use_electrostatics: bool = True,
        use_dispersion: bool = True,
        s6: float | None = None,
        s8: float | None = None,
        a1: float | None = None,
        a2: float | None = None,
        d3_references: str = "2010",
        energy_shift: float = 0.0,
        energy_scale: float = 1.0,
        charge_shift: float = 0.0,
        charge_scale: float = 1.0,
        species=None,
        atomic_energies=None,
        atomic_scales=None,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.sr_cut = cutoff
        self.lr_cut = lr_cutoff
        # neighbor-list radius: long-range cutoff when set (rbf features
        # vanish beyond sr_cut anyway, see module docstring)
        self.cutoff = lr_cutoff if lr_cutoff is not None else cutoff
        self.n_features = n_features
        self.node_feature_dim = n_features  # invariant features (for e.g. LES)
        self.use_electrostatics = use_electrostatics
        self.use_dispersion = use_dispersion
        self.kehalf = KEHALF

        self.embeddings = nn.Parameter(
            torch.empty(MAX_Z, n_features).uniform_(-math.sqrt(3), math.sqrt(3)))
        self.rbf_layer = _RBF(n_rbf, cutoff)
        self.interaction_blocks = nn.ModuleList([
            _InteractionBlock(n_rbf, n_features, num_residual_atomic,
                              num_residual_interaction)
            for _ in range(num_blocks)])
        self.output_blocks = nn.ModuleList([
            _OutputBlock(n_features, num_residual_output)
            for _ in range(num_blocks)])

        self.Eshift = nn.Parameter(torch.full((MAX_Z,), float(energy_shift)))
        self.Escale = nn.Parameter(torch.full((MAX_Z,), float(energy_scale)))
        self.Qshift = nn.Parameter(torch.full((MAX_Z,), float(charge_shift)))
        self.Qscale = nn.Parameter(torch.full((MAX_Z,), float(charge_scale)))

        # D3 parameters: learnable through a softplus unless fixed
        for name, value, default in [("s6", s6, d3.d3_s6), ("s8", s8, d3.d3_s8),
                                     ("a1", a1, d3.d3_a1), ("a2", a2, d3.d3_a2)]:
            if value is None:
                setattr(self, f"_{name}", nn.Parameter(torch.tensor(
                    float(softplus_inverse(default)))))
            else:
                self.register_buffer(f"_{name}",
                                     torch.tensor(float(value)))
            setattr(self, f"_{name}_learnable", value is None)
        if use_dispersion:
            dt = torch.get_default_dtype()
            self.register_buffer("_d3_c6ab", d3.legacy_c6_table(str(d3_references)).to(dt),
                                 persistent=False)
            self.register_buffer("_d3_rcov", d3.d3_rcov.to(dt), persistent=False)
            self.register_buffer("_d3_r2r4", d3.d3_r2r4.to(dt), persistent=False)

        if atomic_energies is not None or atomic_scales is not None:
            if species is None:
                raise ValueError(
                    "species is required to map atomic_energies/atomic_scales")
            with torch.no_grad():
                if atomic_energies is not None:
                    ae = torch.as_tensor(atomic_energies, dtype=self.Eshift.dtype)
                    self.Eshift[torch.tensor(list(species))] = ae
                if atomic_scales is not None:
                    sc = torch.as_tensor(atomic_scales, dtype=self.Escale.dtype)
                    self.Escale[torch.tensor(list(species))] = sc

    # D3 parameters (softplus-positive when learnable, as upstream)
    def _d3_param(self, name: str) -> Tensor:
        raw = getattr(self, f"_{name}")
        return F.softplus(raw) if getattr(self, f"_{name}_learnable") else raw

    @property
    def s6(self) -> Tensor:
        """Effective D3 ``s6`` coefficient."""
        return self._d3_param("s6")

    @property
    def s8(self) -> Tensor:
        """Effective D3 ``s8`` coefficient."""
        return self._d3_param("s8")

    @property
    def a1(self) -> Tensor:
        """Effective D3 ``a1`` coefficient."""
        return self._d3_param("a1")

    @property
    def a2(self) -> Tensor:
        """Effective D3 ``a2`` coefficient."""
        return self._d3_param("a2")

    def atomic_properties(self, atomic_numbers: Tensor, edge_index: Tensor,
                          edge_vec: Tensor):
        """Scaled atomic energies/charges before the long-range terms.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 is the neighbor ``j``, row 1 the
            center ``i`` (upstream ``idx_j``/``idx_i``).
        edge_vec : Tensor
            Edge vectors ``pos[i] - pos[j]``, shape ``(E, 3)``.

        Returns
        -------
        tuple of Tensor
            Per-atom energies ``(N,)``, raw (uncorrected) per-atom charges
            ``(N,)``, edge distances ``(E,)``, the scalar
            non-hierarchicality penalty, and the final per-atom feature
            vectors ``(N, n_features)``.
        """
        idx_j, idx_i = edge_index[0], edge_index[1]
        Dij = edge_vec.norm(dim=-1)
        rbf = self.rbf_layer(Dij)
        x = self.embeddings[atomic_numbers]

        # every module contributes an additive (energy, charge) pair per atom
        energy = x.new_zeros(x.shape[0])
        charge = x.new_zeros(x.shape[0])
        head_sq = []
        for interaction, head in zip(self.interaction_blocks,
                                     self.output_blocks):
            x = interaction(x, rbf, idx_i, idx_j)
            eq = head(x)  # (N, 2): energy column 0, charge column 1
            energy = energy + eq[:, 0]
            charge = charge + eq[:, 1]
            head_sq.append(eq ** 2)

        # non-hierarchicality penalty (paper eqs 18/19): push every module to
        # contribute less than its predecessor; the small constant keeps the
        # ratio finite when both contributions vanish (zero-init heads)
        nh_loss = x.new_zeros(())
        for prev_sq, cur_sq in zip(head_sq, head_sq[1:]):
            nh_loss = nh_loss + torch.mean(cur_sq / (cur_sq + prev_sq + 1e-7))

        energy = self.Escale[atomic_numbers] * energy + self.Eshift[atomic_numbers]
        charge = self.Qscale[atomic_numbers] * charge + self.Qshift[atomic_numbers]
        return energy, charge, Dij, nh_loss, x

    def scaled_charges(self, Qa: Tensor, batch: Tensor, num_graphs: int,
                       total_charge: Tensor | None = None) -> Tensor:
        """Correct the raw charges to the exact total charge (paper eq 14)."""
        n_per = torch.bincount(batch, minlength=num_graphs).to(Qa.dtype)
        if total_charge is None:
            total_charge = torch.zeros(num_graphs, dtype=Qa.dtype,
                                       device=Qa.device)
        q_sum = scatter_sum(Qa, batch, num_graphs)
        return Qa + ((total_charge - q_sum) / n_per)[batch]

    def _switch(self, Dij: Tensor) -> Tensor:
        """Weight of the bare Coulomb kernel: quintic smoothstep in the
        distance, rising from 0 at ``r = 0`` to exactly 1 at ``sr_cut / 2``
        and beyond (where electrostatics are purely ``1/r``)."""
        half = self.sr_cut / 2
        y = Dij / half
        ramp = y ** 3 * (y * (6.0 * y - 15.0) + 10.0)
        return torch.where(Dij < half, ramp, torch.ones_like(Dij))

    def electrostatic_energy_per_atom(self, Dij: Tensor, Qa: Tensor,
                                      idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """Switched, shielded Coulomb energy per atom (paper eqs 12-13).

        At short range the divergent ``1/r`` is traded for the bounded
        ``1/sqrt(r^2 + 1)``; :meth:`_switch` blends the two so the kernel is
        smooth everywhere. When ``lr_cut`` is set, both kernels are
        force-shifted (value and slope zero at the cutoff) and pairs beyond
        it are dropped.
        """
        q_pair = self.kehalf * Qa[idx_i] * Qa[idx_j]
        r_bound = torch.sqrt(Dij * Dij + 1.0)
        w = self._switch(Dij)
        w_bar = 1.0 - w
        if self.lr_cut is None:
            kernel_bare = 1.0 / Dij
            kernel_bound = 1.0 / r_bound
            e_pair = q_pair * (w_bar * kernel_bound + w * kernel_bare)
        else:
            rc = self.lr_cut
            rc_sq = rc * rc
            kernel_bare = 1.0 / Dij + Dij / rc_sq - 2.0 / rc
            kernel_bound = 1.0 / r_bound + r_bound / rc_sq - 2.0 / rc
            e_pair = q_pair * (w_bar * kernel_bound + w * kernel_bare)
            e_pair = torch.where(Dij <= rc, e_pair, torch.zeros_like(e_pair))
        return scatter_sum(e_pair, idx_i, Qa.shape[0])

    def dispersion_energy_per_atom(self, atomic_numbers: Tensor, Dij: Tensor,
                                   idx_i: Tensor, idx_j: Tensor) -> Tensor:
        """Grimme D3(BJ) dispersion per atom, in eV (paper eq 12)."""
        cutoff = self.lr_cut / d3.d3_autoang if self.lr_cut is not None else None
        return d3.d3_autoev * d3.edisp(
            atomic_numbers, Dij / d3.d3_autoang, idx_i, idx_j, cutoff=cutoff,
            s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2,
            c6ab=self._d3_c6ab, rcov=self._d3_rcov, r2r4=self._d3_r2r4)

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict energies, corrected charges, and the dipole for a graph.

        Parameters
        ----------
        data : xnn.common.data.AtomicGraph
            The input atomic graph. If it carries a ``total_charge``
            attribute (per-structure tensor), charges are corrected to it;
            otherwise neutral structures are assumed.

        Returns
        -------
        dict of str to torch.Tensor
            ``"node_energy"`` ``(N,)`` and ``"energy"`` ``(B,)`` as for every
            xnn model, plus ``"charges"`` ``(N,)`` (corrected partial
            charges), ``"dipole"`` ``(B, 3)`` (paper eq 15) and ``"nh_loss"``
            (scalar regularization term).
        """
        idx_j, idx_i = data.edge_index[0], data.edge_index[1]
        Ea, Qa, Dij, nh_loss, features = self.atomic_properties(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        Qa = self.scaled_charges(Qa, data.batch, data.num_graphs,
                                 getattr(data, "total_charge", None))
        if self.use_electrostatics:
            Ea = Ea + self.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j)
        if self.use_dispersion:
            Ea = Ea + self.dispersion_energy_per_atom(
                data.atomic_numbers, Dij, idx_i, idx_j)
        dipole = scatter_sum(Qa.unsqueeze(-1) * data.pos, data.batch,
                             data.num_graphs)
        return {"node_energy": Ea,
                "energy": self.aggregate_energy(Ea, data),
                "charges": Qa, "dipole": dipole, "nh_loss": nh_loss,
                "node_features": features}

    @classmethod
    def from_config(cls, cfg) -> "PhysNet":
        """Construct a :class:`PhysNet` from a core model config.

        Core fields: ``cfg.cutoff`` -> ``sr_cut``, ``cfg.n_features`` -> ``F``,
        ``cfg.n_rbf`` -> ``K``, ``cfg.n_interactions`` -> ``num_blocks``.
        Everything else is read from ``cfg.extra``; upstream ``train.py``
        spellings are translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        PhysNet
            The instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        lr_cutoff = extra.get("lr_cutoff")

        def opt_float(key):
            v = extra.get(key)
            return None if v is None else float(v)

        return cls(
            cutoff=cfg.cutoff,
            lr_cutoff=None if lr_cutoff is None else float(lr_cutoff),
            n_features=cfg.n_features,
            n_rbf=cfg.n_rbf,
            num_blocks=cfg.n_interactions,
            num_residual_atomic=int(extra.get("num_residual_atomic", 2)),
            num_residual_interaction=int(
                extra.get("num_residual_interaction", 3)),
            num_residual_output=int(extra.get("num_residual_output", 1)),
            use_electrostatics=bool(extra.get("use_electrostatics", True)),
            use_dispersion=bool(extra.get("use_dispersion", True)),
            s6=opt_float("s6"), s8=opt_float("s8"),
            a1=opt_float("a1"), a2=opt_float("a2"),
            d3_references=str(extra.get("d3_references", "2010")),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            charge_shift=float(extra.get("charge_shift", 0.0)),
            charge_scale=float(extra.get("charge_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
            atomic_scales=coerce_per_species(
                extra.get("atomic_scales"), species or [], "atomic_scales"),
        )
