"""NewtonNet (Haghighatlari et al., Digital Discovery 2022): Newtonian message passing.

The message passing network of

* M. Haghighatlari, J. Li, X. Guan, O. Zhang, A. Das, C. J. Stein, F. Heidar-Zadeh, M. Liu,
  M. Head-Gordon, L. Bertels, H. Hao, I. Leven, T. Head-Gordon, "NewtonNet: a Newtonian message passing
  network for deep learning of interatomic potentials and forces", Digital Discovery 1, 333 (2022),

written from the paper's equations on the xnn abstractions. Every atom carries ``F`` invariant features
``a_i`` and ``F`` equivariant force features ``f_i`` in ``R^3``; ``T`` interaction blocks refine both.

Symmetric message (eq 5 to 7), with ``r_ij = r_j - r_i`` and its unit vector ``u_ij``::

    m_ij = phi_a(a_i) o phi_a(a_j) o phi_rbf(e(r_ij)) e_cut(r_ij)
    a_i <- a_i + sum_j m_ij

``e`` is the radial basis (``sin(n pi r / r_c) / (r / r_c)`` by default, the paper's
``sqrt(2 / r_c) sin(n pi r / r_c) / r`` with ``rbf_prefactor``) and ``e_cut`` the polynomial envelope of
degree ``p = 9`` (or a cosine cutoff). Every ``phi`` is a two-layer network ``W act(W x + b) + b``.

Force features (eq 11 and 12). By default each block adds directional terms and carries the force
features of the neighbors along::

    f_i <- f_i + sum_j phi_f(m_ij) u_ij + phi_p(m_ij) o f_j

With ``latent_force=True`` the directions are scaled by learned force magnitudes first (eq 8 to 10),
``F_ij = phi_F(m_ij) u_ij``, accumulated into the latent forces ``F_i <- F_i + sum_j F_ij`` (output
``"latent_forces"``, used by the force-direction loss of eq 23), and ``f_i <- f_i + sum_j phi_f(m_ij) F_ij``.

Energy update. By default the force features are contracted with a linear mix ``U`` of themselves::

    a_i <- a_i + <f_i, U f_i>

With ``displacement=True`` the paper's displacement features (eq 13 to 16) are used instead, and the
force features of the neighbors are not carried::

    dr_i <- sum_j phi_r'(m_ij) o dr_j + phi_r(a_i) o f_i
    a_i  <- a_i - phi_u(a_i) o <f_i, dr_i>

``latent_force=True`` with ``displacement=True`` is the model of the paper. The networks multiplying
neighbor quantities (``phi_f`` without latent forces, ``phi_p``, ``phi_r'``) and ``phi_F`` carry no bias,
so every edge term vanishes smoothly at the cutoff.

Readout (eq 20 and 21): ``E_i = phi_out(a_i)``, a network ``F -> F -> F -> 1`` by default
(``readout_widths``), scaled and shifted per element (``atom_scale``, ``atom_ref``) and by the xnn
standardization (``energy_scale``, ``energy_shift``), then summed. With ``direct_forces=True`` a second
network gives forces from the force features, ``F_i = s_Z sum_k phi_d(a_i)_k f_ik``.

Works for molecules and periodic systems through ``data.edge_vectors()`` (energy, forces, stress).
"""
# NOTE: no `from __future__ import annotations` here: TorchScript needs the annotations of the
# scriptable cores (`node_energy` is what LAMMPS deployment uses).
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, PolynomialCutoff, make_radial_basis
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.constants import ExactFloat64Constants
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import gather_rows, init_linears_, make_activation, scatter_sum
from xnn.common.models.registry import register_model

#: Embedding rows: atomic numbers 1 to 118, row 0 a zero padding row.
N_ELEMENTS = 119


def _mlp(n_in: int, n_hidden: int, n_out: int, activation, bias: bool = True) -> nn.Sequential:
    """The two-layer network ``W act(W x + b) + b`` of every ``phi``."""
    return nn.Sequential(nn.Linear(n_in, n_hidden, bias=bias), make_activation(activation),
                         nn.Linear(n_hidden, n_out, bias=bias))


class _Interaction(nn.Module, FastPathModule):
    """One NewtonNet interaction block: message, force features and the energy update.

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions.
    activation : str or torch.nn.Module
        Nonlinearity of every network.
    latent_force : bool
        Scale the directions by the force magnitudes ``phi_F(m_ij)`` and accumulate the latent forces.
    displacement : bool
        The paper's displacement features and energy change instead of the contraction ``<f, U f>``.
    layer_norm : bool
        Normalize the invariant features at the end of the block.
    rbf_bias : bool
        Give ``phi_rbf`` a bias (it is applied before the envelope).

    Notes
    -----
    A fused GPU implementation (:mod:`xnn.gnn.fast.newtonnet`) forms the edge sums of the force and
    displacement features when :attr:`fast_active` is set (see :meth:`NewtonNet.set_use_fast`).
    """

    def __init__(self, n_features: int, n_rbf: int, activation, latent_force: bool, displacement: bool,
                 layer_norm: bool, rbf_bias: bool):
        super().__init__()
        F = n_features
        self.latent_force = latent_force
        self.displacement = displacement
        self.rbf_bias = rbf_bias
        self.phi_a = _mlp(F, F, F, activation)
        self.phi_rbf = nn.Linear(n_rbf, F, bias=rbf_bias)
        # a magnitude that vanishes with the message lets phi_f carry a bias
        self.phi_F = nn.Linear(F, 1, bias=False) if latent_force else nn.Identity()
        self.phi_f = _mlp(F, F, F, activation, bias=latent_force)
        self.phi_p = nn.Identity() if displacement else _mlp(F, F, F, activation, bias=False)
        self.mix = nn.Identity() if displacement else nn.Linear(F, F, bias=False)
        self.phi_r = _mlp(F, F, F, activation) if displacement else nn.Identity()
        self.phi_rj = _mlp(F, F, F, activation, bias=False) if displacement else nn.Identity()
        self.phi_u = _mlp(F, F, F, activation) if displacement else nn.Identity()
        self.norm = nn.LayerNorm(F) if layer_norm else nn.Identity()
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Triton on ``device`` in float32 or float64."""
        from xnn.gnn.fast.newtonnet import supported
        return supported(device, dtype)

    def forward(self, a: Tensor, f: Tensor, dr: Tensor, latent: Tensor, rbf: Tensor, cut: Tensor,
                direction: Tensor, edge_index: Tensor, first: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Update ``(a, f, dr, latent)`` of every atom.

        Parameters
        ----------
        a : Tensor
            Invariant features ``(N, F)``.
        f : Tensor
            Force features ``(N, 3, F)``.
        dr : Tensor
            Displacement features ``(N, 3, F)`` (zeros unless ``displacement``).
        latent : Tensor
            Latent forces ``(N, 3)`` (zeros unless ``latent_force``).
        rbf : Tensor
            Radial basis ``(E, n_rbf)``, already times the envelope unless ``rbf_bias``.
        cut : Tensor
            Cutoff envelope ``(E,)``.
        direction : Tensor
            Unit vectors ``(r_j - r_i) / |r_j - r_i|`` ``(E, 3)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the center ``i``.
        first : bool, optional
            The first block: ``f`` and ``dr`` are zero, so the terms carrying the neighbors' values are skipped.

        Returns
        -------
        tuple of Tensor
            The updated ``(a, f, dr, latent)``.
        """
        src, dst = edge_index[0], edge_index[1]
        n = a.shape[0]
        x = self.phi_a(a)
        # a bias-free radial layer takes the envelope through its input (one product for all blocks)
        e = self.phi_rbf(rbf) * cut.unsqueeze(-1) if self.rbf_bias else self.phi_rbf(rbf)
        m = gather_rows(x, dst) * gather_rows(x, src) * e
        a = a + scatter_sum(m, dst, n)
        if self.fast_active and not torch.jit.is_scripting():
            f, dr, latent = self._features_fast(a, f, dr, latent, m, direction, edge_index, first)
        else:
            if self.latent_force:
                force = self.phi_F(m) * direction                                # F_ij (E, 3)
                latent = latent + scatter_sum(force, dst, n)
            else:
                force = direction
            vec = self.phi_f(m).unsqueeze(1) * force.unsqueeze(-1)             # (E, 3, F)
            if not self.displacement and not first:
                vec = vec + self.phi_p(m).unsqueeze(1) * gather_rows(f, src)
            f = f + scatter_sum(vec, dst, n)
            if self.displacement:
                dr_new = self.phi_r(a).unsqueeze(1) * f
                if not first:
                    dr_new = dr_new + scatter_sum(self.phi_rj(m).unsqueeze(1) * gather_rows(dr, src), dst, n)
                dr = dr_new
        if self.displacement:
            a = a - self.phi_u(a) * (f * dr).sum(dim=1)
        else:
            a = a + (f * self.mix(f)).sum(dim=1)
        return self.norm(a), f, dr, latent

    @torch.jit.unused
    def _features_fast(self, a: Tensor, f: Tensor, dr: Tensor, latent: Tensor, m: Tensor, direction: Tensor,
                       edge_index: Tensor, first: bool) -> Tuple[Tensor, Tensor, Tensor]:
        """The force and displacement features (and latent forces) with the fused edge sums."""
        from xnn.gnn.fast.newtonnet_kernels import edge_sum
        src, dst = edge_index[0], edge_index[1]
        n = a.shape[0]
        carry = None if first else (self.phi_rj if self.displacement else self.phi_p)
        c_f = self.phi_f(m)
        c_x = carry(m) if carry is not None else None
        if self.latent_force:
            force = self.phi_F(m) * direction
            latent = latent + scatter_sum(force, dst, n)
        else:
            force = direction
        if self.displacement:
            f = f + edge_sum(n, src, dst, c_f, force)
            dr_new = self.phi_r(a).unsqueeze(1) * f
            if c_x is not None:
                dr_new = dr_new + edge_sum(n, src, dst, None, None, c_x, dr)
            return f, dr_new, latent
        return f + edge_sum(n, src, dst, c_f, force, c_x, f if c_x is not None else None), dr, latent


@register_model("newtonnet")
class NewtonNet(ExactFloat64Constants, InteratomicPotential):
    """NewtonNet interatomic potential (see the module docstring for the equations).

    The defaults follow the authors' current code: ``F = 128`` features, ``T = 3`` blocks, 20 Bessel
    functions ``sin(n pi r / r_c) / (r / r_c)`` with the degree-9 polynomial envelope at 5 Angstrom,
    SiLU networks, the contraction ``<f, U f>`` for the energy update and a readout ``F -> F -> F -> 1``.
    The model of the paper is ``latent_force=True, displacement=True, rbf_bias=True,
    rbf_prefactor=sqrt(2 / cutoff), readout_widths=[128, 64]``.

    Parameters
    ----------
    n_features : int, optional
        Feature width ``F``, by default 128.
    n_interactions : int, optional
        Number of interaction blocks ``T``, by default 3.
    n_rbf : int, optional
        Number of radial basis functions, by default 20.
    cutoff : float, optional
        Neighbor-list radius and cutoff of the envelope (Angstrom), by default 5.0.
    activation : str, optional
        Nonlinearity of every network (a name of :func:`~xnn.common.models.ops.make_activation`),
        by default ``"silu"``.
    radial_basis : str, optional
        Any basis of :func:`~xnn.common.featurizers.make_radial_basis`, by default ``"bessel"``.
    rbf_prefactor : float or None, optional
        Normalization of the Bessel functions ``prefactor sin(n pi r / r_c) / r``; ``None`` (default)
        uses ``r_c``, i.e. ``sin(n pi x) / x`` with ``x = r / r_c``. The paper's is ``sqrt(2 / r_c)``.
    trainable_rbf : bool, optional
        Learn the parameters of the radial basis, by default ``False``.
    cutoff_function : str, optional
        ``"polynomial"`` (default) or ``"cosine"``.
    num_polynomial_cutoff : int, optional
        Degree ``p`` of the polynomial envelope, by default 9.
    rbf_bias : bool, optional
        Give the radial layer ``phi_rbf`` a bias, by default ``False``.
    latent_force : bool, optional
        Force magnitudes ``phi_F`` and the latent forces of eq 8 to 10, by default ``False``.
    displacement : bool, optional
        The displacement features and energy change of eq 13 to 16 instead of the contraction
        ``<f, U f>``, by default ``False``.
    layer_norm : bool, optional
        Layer normalization of the invariant features after every block, by default ``False``.
    readout_widths : list of int or None, optional
        Hidden widths of the energy network; ``None`` (default) means ``[F, F]``.
    direct_forces : bool, optional
        Predict forces directly from the force features (output ``"forces"``) instead of as the
        energy gradient, by default ``False``.
    weight_init : str, optional
        Initialization of every dense layer (a name of :data:`~xnn.common.models.ops.WEIGHT_INITS`;
        biases start at zero), by default ``"torch"`` (PyTorch's own).
    energy_shift : float, optional
        Additive per-atom energy standardization, by default 0.0.
    energy_scale : float, optional
        Multiplicative per-atom energy standardization, by default 1.0.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``; the model handles every element up to Z = 118.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.

    Attributes
    ----------
    embedding : torch.nn.Embedding
        Atomic-number embedding ``g(Z)`` (row 0 is a zero padding row).
    rbf : torch.nn.Module
        The radial basis.
    cutoff_fn : torch.nn.Module
        The envelope multiplying every message.
    interactions : torch.nn.ModuleList
        The ``T`` interaction blocks.
    readout : torch.nn.Sequential
        The energy network ``phi_out``.
    atom_scale, atom_ref : torch.nn.Embedding
        Learnable per-element scale (initialized to one) and shift (zero) of the atomic energies.
    force_head : torch.nn.Module or None
        The direct-force network, when requested.
    force_scale : torch.nn.Embedding or None
        Learnable per-element scale of the direct forces.
    """

    head_modules = ("readout", "energy_scale", "energy_shift", "atom_scale", "atom_ref")

    def __init__(self, n_features: int = 128, n_interactions: int = 3, n_rbf: int = 20, cutoff: float = 5.0,
                 activation: str = "silu", radial_basis: str = "bessel", rbf_prefactor: Optional[float] = None,
                 trainable_rbf: bool = False, cutoff_function: str = "polynomial",
                 num_polynomial_cutoff: int = 9, rbf_bias: bool = False, latent_force: bool = False,
                 displacement: bool = False, layer_norm: bool = False,
                 readout_widths: Optional[List[int]] = None, direct_forces: bool = False,
                 weight_init: str = "torch", energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        if cutoff_function not in ("polynomial", "cosine"):
            raise ValueError(f"cutoff_function must be 'polynomial' or 'cosine', got {cutoff_function!r}")
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.n_interactions = n_interactions
        self.latent_force = bool(latent_force)
        self.displacement = bool(displacement)
        self.rbf_bias = bool(rbf_bias)
        self.direct_forces = bool(direct_forces)
        self.embedding = nn.Embedding(N_ELEMENTS, n_features, padding_idx=0)
        if radial_basis == "bessel":
            prefactor = self.cutoff if rbf_prefactor is None else float(rbf_prefactor)
            self.rbf = make_radial_basis("bessel", n_rbf, cutoff, trainable_rbf, prefactor=prefactor)
        else:
            self.rbf = make_radial_basis(radial_basis, n_rbf, cutoff, trainable_rbf)
        self.cutoff_fn = (PolynomialCutoff(cutoff, p=num_polynomial_cutoff) if cutoff_function == "polynomial"
                          else CosineCutoff(cutoff))
        self.interactions = nn.ModuleList([
            _Interaction(n_features, n_rbf, activation, self.latent_force, self.displacement, layer_norm,
                         rbf_bias) for _ in range(n_interactions)])
        widths = [n_features] + list(readout_widths if readout_widths is not None
                                     else [n_features, n_features]) + [1]
        layers: List[nn.Module] = []
        for k in range(len(widths) - 1):
            layers.append(nn.Linear(widths[k], widths[k + 1]))
            if k < len(widths) - 2:
                layers.append(make_activation(activation))
        self.readout = nn.Sequential(*layers)
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        self.atom_scale = nn.Embedding(N_ELEMENTS, 1)
        nn.init.ones_(self.atom_scale.weight)
        self.atom_ref = nn.Embedding(N_ELEMENTS, 1)
        nn.init.zeros_(self.atom_ref.weight)
        self.force_head: Optional[nn.Module] = None
        self.force_scale: Optional[nn.Embedding] = None
        if self.direct_forces:
            self.force_head = nn.Sequential(
                nn.Linear(n_features, n_features), make_activation(activation),
                nn.Linear(n_features, n_features), make_activation(activation),
                nn.Linear(n_features, n_features))
            self.force_scale = nn.Embedding(N_ELEMENTS, 1)
            nn.init.ones_(self.force_scale.weight)
        init_linears_(self, weight_init)
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)
        # the fused edge sums (see set_use_fast); "auto" takes them where measured faster
        self.use_fast = "auto"

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "NewtonNet":
        """Choose the implementation of the edge sums of the force and displacement features.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the fused GPU kernels of :mod:`xnn.gnn.fast.newtonnet` where they are
            available and faster for the input (a CUDA device, Triton, a graph above the size crossover
            measured for evaluation, ``AUTO_POLICY``, or for training, ``AUTO_POLICY_TRAINING``); ``True``:
            wherever they are available; ``False``: always the reference implementation. The kernels give
            the reference's numbers to round-off and are differentiable to any order, so force training
            takes them as well.

        Returns
        -------
        NewtonNet
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> None:
        """Switch the fused edge sums on or off for one evaluation."""
        from xnn.common.models.fast import fast_modules, select
        from xnn.gnn.fast.newtonnet import AUTO_POLICY, AUTO_POLICY_TRAINING
        blocks = self.__dict__.get("_fast_blocks")
        if blocks is None:
            blocks = fast_modules(self)
            self.__dict__["_fast_blocks"] = blocks
        like = self.embedding.weight
        policy = AUTO_POLICY_TRAINING if self.training else AUTO_POLICY
        select(blocks, self.use_fast, policy, like.device, like.dtype, num_edges)

    @torch.jit.ignore
    def set_energy_scale_shift(self, scale: float, shift: float) -> None:
        """Set the per-atom energy standardization from training statistics.

        Parameters
        ----------
        scale : float
            Standard deviation of the training-set energy per atom.
        shift : float
            Mean training-set energy per atom.
        """
        with torch.no_grad():
            self.energy_scale.fill_(float(scale))
            self.energy_shift.fill_(float(shift))

    @torch.jit.ignore
    def set_atomic_energies(self, species, values) -> None:
        """Initialize the per-element reference energies ``atom_ref``.

        Parameters
        ----------
        species : list of int
            Atomic numbers the values refer to, in order.
        values : array-like
            One reference energy per entry of ``species``.

        Raises
        ------
        ValueError
            If the number of values does not match the number of species.
        """
        ae = torch.as_tensor(values, dtype=self.atom_ref.weight.dtype)
        if ae.numel() != len(list(species)):
            raise ValueError(f"got {ae.numel()} atomic energies for {len(list(species))} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(list(species)), 0] = ae

    @torch.jit.export
    def representation(self, atomic_numbers: Tensor, edge_index: Tensor,
                       edge_vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript-compatible trunk: invariant features, force features and latent forces.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the center ``i``.
        edge_vec : Tensor
            Edge vectors ``pos[i] - pos[j]`` ``(E, 3)`` (with any periodic image shift applied).

        Returns
        -------
        tuple of Tensor
            ``a`` ``(N, F)``, ``f`` ``(N, 3, F)`` and the latent forces ``(N, 3)`` (zeros unless
            ``latent_force``) after the last block.
        """
        r = torch.linalg.norm(edge_vec, dim=-1)
        # the paper's r_ij = r_j - r_i is the negative of the xnn edge vector
        direction = -edge_vec / r.clamp(min=1e-12).unsqueeze(-1)
        cut = self.cutoff_fn(r)
        rbf = self.rbf(r)
        if not self.rbf_bias:
            rbf = rbf * cut.unsqueeze(-1)
        a = self.embedding(atomic_numbers)
        f = a.new_zeros((a.shape[0], 3, a.shape[1]))
        dr = torch.zeros_like(f)
        latent = a.new_zeros((a.shape[0], 3))
        for k, block in enumerate(self.interactions):
            a, f, dr, latent = block(a, f, dr, latent, rbf, cut, direction, edge_index, k == 0)
        return a, f, latent

    def _energy(self, a: Tensor, atomic_numbers: Tensor) -> Tensor:
        """Per-atom energies from the invariant features."""
        y = self.readout(a).squeeze(-1) * self.atom_scale(atomic_numbers).squeeze(-1)
        return self.energy_scale * y + self.energy_shift + self.atom_ref(atomic_numbers).squeeze(-1)

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: invariant features and per-atom energies.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.

        Returns
        -------
        tuple of Tensor
            The features ``(N, F)`` and the per-atom energy ``(N,)``.

        Raises
        ------
        RuntimeError
            With ``direct_forces``: its forces are not the gradient of this energy.
        """
        if self.direct_forces:
            raise RuntimeError("a NewtonNet with direct_forces has no energy-only core "
                               "(its forces are not the energy gradient)")
        a, _, _ = self.representation(atomic_numbers, edge_index, edge_vec)
        return a, self._energy(a, atomic_numbers)

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        """Per-atom energy ``(N,)`` (the LAMMPS / TorchScript entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies and, when requested, latent and direct forces.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``, ``"node_features"`` ``(N, F)``,
            ``"node_vectors"`` (the force features, ``(N, 3, F)``) and, with ``latent_force``,
            ``"latent_forces"`` ``(N, 3)``; with ``direct_forces``, ``"forces"`` ``(N, 3)``.
        """
        self._select_fast(int(data.edge_index.shape[1]))
        a, f, latent = self.representation(data.atomic_numbers, data.edge_index, data.edge_vectors())
        node_energy = self._energy(a, data.atomic_numbers)
        out = {"node_energy": node_energy, "energy": self.aggregate_energy(node_energy, data),
               "node_features": a, "node_vectors": f}
        if self.latent_force:
            out["latent_forces"] = latent
        if self.force_head is not None and self.force_scale is not None:
            out["forces"] = ((self.force_head(a).unsqueeze(1) * f).sum(-1)
                             * self.force_scale(data.atomic_numbers))
        return out

    @classmethod
    def from_config(cls, cfg) -> "NewtonNet":
        """Build a :class:`NewtonNet` from a configuration object.

        Core fields: ``cfg.n_features``, ``cfg.n_interactions``, ``cfg.n_rbf``, ``cfg.cutoff``.
        Everything else is read from ``cfg.extra`` (``activation``, ``radial_basis``, ``rbf_prefactor``,
        ``trainable_rbf``, ``cutoff_function``, ``num_polynomial_cutoff``, ``rbf_bias``, ``latent_force``,
        ``displacement``, ``layer_norm``, ``readout_widths``, ``direct_forces``, ``weight_init``,
        ``energy_shift``, ``energy_scale``, ``species``, ``atomic_energies``); other codes' key
        spellings are translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        NewtonNet
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        species = coerce_species(extra.get("species")) if extra.get("species") is not None else None
        cutoff_function = str(extra.get("cutoff_function", "polynomial"))
        widths = extra.get("readout_widths")
        prefactor = extra.get("rbf_prefactor")
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            activation=str(extra.get("activation", "silu")),
            radial_basis=str(extra.get("radial_basis", "bessel")),
            rbf_prefactor=None if prefactor is None else float(prefactor),
            trainable_rbf=bool(extra.get("trainable_rbf", False)),
            # the reference code's value spelling of the polynomial envelope
            cutoff_function="polynomial" if cutoff_function == "poly" else cutoff_function,
            num_polynomial_cutoff=int(extra.get("num_polynomial_cutoff", 9)),
            rbf_bias=bool(extra.get("rbf_bias", False)),
            latent_force=bool(extra.get("latent_force", False)),
            displacement=bool(extra.get("displacement", False)),
            layer_norm=bool(extra.get("layer_norm", False)),
            readout_widths=None if widths is None else [int(w) for w in widths],
            direct_forces=bool(extra.get("direct_forces", False)),
            weight_init=str(extra.get("weight_init", "torch")),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(extra.get("atomic_energies"), species or [], "atomic_energies"),
        )
