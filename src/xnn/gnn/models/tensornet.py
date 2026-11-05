"""TensorNet (Simeon & De Fabritiis, NeurIPS 2023), the Cartesian-tensor model of TorchMD-Net 2.0.

An implementation of

* G. Simeon, G. De Fabritiis, "TensorNet: Cartesian Tensor Representations
  for Efficient Learning of Molecular Potentials", NeurIPS 2023
  (arXiv:2306.06482), as released and described in TorchMD-Net 2.0 (Pelaez
  *et al.*, *J. Chem. Theory Comput.* **20**, 4076 (2024), section 2.1.1),

written from the papers on the xnn abstractions
(:class:`~xnn.common.featurizers.ExpNormalSmearing`,
:class:`~xnn.common.featurizers.CosineCutoff`,
:func:`~xnn.common.models.ops.scatter_sum`) and consistent with the
conventions of the authors' code (torchmd/torchmd-net: tensor layout,
feature order, self-interaction, the ``O(3)`` / ``SO(3)`` variants), without
copying it. Parameter names follow the reference so that weights transplant.

Representation. Every atom carries ``F`` rank-2 Cartesian tensors
``X_i in R^{3 x 3}``, each the sum of its irreducible parts: a scalar
``I_i`` times the identity, a skew-symmetric ``A_i`` (a vector) and a
symmetric traceless ``S_i``. The embedding builds them from the neighbors::

    I_i = sum_j f_ij W^I e_rbf(d_ij),  A_i = [sum_j f_ij W^A e_rbf(d_ij) r_ij / |r_ij|]_x,
    S_i = sum_j f_ij W^S e_rbf(d_ij) (r_ij r_ij^T / |r_ij|^2 - 1/3 1)

with ``f_ij = phi(d_ij) W^Z [z_i, z_j]`` (the cutoff times a pair-type
embedding), then rescales each part by an MLP of the layer-normed Frobenius
norms. Every interaction layer normalizes ``X_i`` by ``1 + |X_i|^2``, mixes the
parts linearly into ``Y_i``, passes messages ``M_i = sum_j f_ij o (I_j, A_j,
S_j)`` with edge-wise filters ``f_ij`` (an MLP of ``e_rbf(d_ij)`` times the
cutoff), forms the tensor products ``C_i = Y_i M_i + M_i Y_i`` (the ``O(3)``
model) or ``C_i = 2 Y_i M_i`` (``SO(3)``, sensitive to chirality), normalizes,
mixes again into ``dX_i`` and updates ``X_i <- X_i + dX_i + dX_i^2``. A total
charge ``q`` scales the two products by ``1 + 0.1 q``. The readout concatenates
``3 I_i^2``, ``|A_i|^2`` and ``|S_i|^2``, applies a layer norm, a linear layer
and a two-layer MLP, standardized by ``energy_scale`` / ``energy_shift`` and
shifted by the per-element reference ``atom_ref``.

Conventions worth knowing:

* the sums over neighbors include ``j = i`` (the reference neighbor list
  carries self pairs at ``d_ii = 0``), which contribute to ``I`` in the
  embedding and to every message; the model appends one self pair per atom
  (zero distance and direction, cutoff one) to the pairs of the graph;
* ``r_ij`` points from the neighbor to the center atom, which is the xnn edge
  vector ``pos[dst] - pos[src]``; the sign only matters for the ``SO(3)``
  variant;
* the reference standardizes with the dataset mean per *structure* and the
  standard deviation per atom; xnn applies both per atom.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import CosineCutoff, make_radial_basis
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.constants import ExactFloat64Constants
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import gather_rows, make_activation, scatter_sum
from xnn.common.models.registry import register_model

_GROUPS = ("O(3)", "SO(3)")

#: Pairs (self pairs included) from which an interaction layer in evaluation mode
#: recomputes its filter network in the backward pass instead of keeping its activations
#: (``recompute="auto"``, see :func:`~xnn.gnn.models.set_recompute_radial`). Recomputing
#: lowers the peak of the fused layers by about a quarter and costs 13-35 % of the time
#: of an evaluation, so it is left to graphs where memory binds (with the defaults about
#: 18 GB at a million pairs in float32); ``True`` takes it at any size.
RECOMPUTE_MIN_PAIRS = 1_000_000


def skew_tensor(vector: Tensor) -> Tensor:
    """The skew-symmetric tensor ``[v]_x`` of a vector, ``(N, 3, F) -> (N, 3, 3, F)``."""
    zero = torch.zeros_like(vector[:, 0])
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    rows = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1)
    return rows.reshape(vector.shape[0], 3, 3, vector.shape[2])


def axial_vector(tensor: Tensor) -> Tensor:
    """The vector ``v`` with ``[v]_x = A`` of a skew tensor, ``(N, 3, 3, F) -> (N, 3, F)``."""
    return 0.5 * torch.stack([tensor[:, 2, 1] - tensor[:, 1, 2],
                              tensor[:, 0, 2] - tensor[:, 2, 0],
                              tensor[:, 1, 0] - tensor[:, 0, 1]], dim=1)


def decompose(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Split ``(N, 3, 3, F)`` tensors into the scalar ``(N, F)``, skew and symmetric traceless parts."""
    skew = 0.5 * (tensor - tensor.transpose(1, 2))
    scalar = tensor.diagonal(dim1=1, dim2=2).mean(dim=-1)
    eye = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
    symmetric = tensor - skew - scalar[:, None, None, :] * eye[None, :, :, None]
    return scalar, skew, symmetric


def compose(scalar: Tensor, skew: Tensor, symmetric: Tensor) -> Tensor:
    """Rebuild ``(N, 3, 3, F)`` tensors from their irreducible parts."""
    eye = torch.eye(3, dtype=skew.dtype, device=skew.device)
    return scalar[:, None, None, :] * eye[None, :, :, None] + skew + symmetric


def frobenius(tensor: Tensor) -> Tensor:
    """Squared Frobenius norm of every ``3 x 3`` tensor, ``(N, 3, 3, F) -> (N, F)``."""
    return (tensor * tensor).sum(dim=(1, 2))


def tensor_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Feature-wise matrix product of two ``(N, 3, 3, F)`` tensors."""
    # the products summed over the inner index (elementwise kernels): a batched matrix
    # product of N F separate 3 x 3 matrices runs far below the device's speed
    return (a.unsqueeze(3) * b.unsqueeze(1)).sum(dim=2)


class _TensorEmbedding(nn.Module, FastPathModule):
    """The tensor embedding layer (section 3.2 of the TensorNet paper).

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions.
    n_elements : int
        Size of the type embedding table (indexed by ``Z``).
    activation : str
        Nonlinearity of the norm MLP.

    Notes
    -----
    A fused GPU implementation (:mod:`xnn.gnn.fast.tensornet`) evaluates the
    sums over the pairs when :attr:`fast_active` is set (see
    :meth:`TensorNet.set_use_fast`).
    """

    def __init__(self, n_features: int, n_rbf: int, n_elements: int, activation: str):
        super().__init__()
        self.n_features = n_features
        self.distance_proj1 = nn.Linear(n_rbf, n_features)
        self.distance_proj2 = nn.Linear(n_rbf, n_features)
        self.distance_proj3 = nn.Linear(n_rbf, n_features)
        self.emb = nn.Embedding(n_elements, n_features)
        self.emb2 = nn.Linear(2 * n_features, n_features)
        self.act = make_activation(activation)
        self.linears_tensor = nn.ModuleList(
            [nn.Linear(n_features, n_features, bias=False) for _ in range(3)])
        self.linears_scalar = nn.ModuleList([
            nn.Linear(n_features, 2 * n_features), nn.Linear(2 * n_features, 3 * n_features)])
        self.init_norm = nn.LayerNorm(n_features)
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Triton on ``device`` and a power-of-two feature width."""
        from xnn.gnn.fast.tensornet import supported
        n = self.n_features
        return 16 <= n <= 512 and n & (n - 1) == 0 and supported(device, dtype)

    def _pair_terms(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
        """The center and neighbor terms of the pair embedding, per atom ``(N, F)`` each.

        ``emb2`` is affine, so ``emb2([z_i, z_j]) = A_i + B_j`` with
        ``A = emb2([z, 0])`` and ``B = emb2([0, z]) - emb2([0, 0])``; one call of
        the module itself (an adapter on it included) evaluates them.
        """
        n = emb.shape[0]
        zero = torch.zeros_like(emb)
        out = self.emb2(torch.cat([torch.cat([emb, zero], dim=1), torch.cat([zero, emb], dim=1),
                                   torch.zeros_like(emb[:1]).repeat(1, 2)], dim=0))
        return out[:n], out[n:2 * n] - out[2 * n:]

    def forward(self, atomic_numbers: Tensor, src: Tensor, dst: Tensor, cut: Tensor,
                unit: Tensor, rbf: Tensor) -> Tensor:
        """Embed the neighborhoods into ``(N, 3, 3, F)`` tensors.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        src, dst : Tensor
            Neighbor ``j`` and center ``i`` of every pair, self pairs included,
            ``(P,)`` each.
        cut : Tensor
            The cosine cutoff of every pair ``(P,)``.
        unit : Tensor
            Unit vectors ``(r_i - r_j) / |r_i - r_j|`` (zero for a self pair),
            ``(P, 3)``.
        rbf : Tensor
            Radial basis of the pair distances ``(P, n_rbf)``.
        """
        n = atomic_numbers.shape[0]
        center, neighbor = self._pair_terms(self.emb(atomic_numbers))
        if self.fast_active and not torch.jit.is_scripting():
            return self._forward_fast(center, neighbor, cut, unit, rbf, src, dst)
        pair = gather_rows(center, dst) + gather_rows(neighbor, src)      # (P, F)
        weight = cut.unsqueeze(-1) * pair
        scalar = scatter_sum(weight * self.distance_proj1(rbf), dst, n)
        vector = scatter_sum((weight * self.distance_proj2(rbf)).unsqueeze(1)
                             * unit.unsqueeze(-1), dst, n)            # (N, 3, F)
        outer = unit.unsqueeze(2) * unit.unsqueeze(1)                 # (P, 3, 3)
        full = scatter_sum((weight * self.distance_proj3(rbf))[:, None, None, :]
                           * outer.unsqueeze(-1), dst, n)             # (N, 3, 3, F)
        skew = skew_tensor(vector)
        eye = torch.eye(3, dtype=full.dtype, device=full.device)
        symmetric = full - full.diagonal(dim1=1, dim2=2).mean(dim=-1)[:, None, None, :] * eye[None, :, :, None]

        norm = self.init_norm(frobenius(compose(scalar, skew, symmetric)))
        for linear in self.linears_scalar:
            norm = self.act(linear(norm))
        norm = norm.reshape(n, 3, self.n_features)
        scalar = self.linears_tensor[0](scalar) * norm[:, 0]
        skew = self.linears_tensor[1](skew) * norm[:, 1, None, None, :]
        symmetric = self.linears_tensor[2](symmetric) * norm[:, 2, None, None, :]
        return compose(scalar, skew, symmetric)

    @torch.jit.unused
    def _forward_fast(self, center: Tensor, neighbor: Tensor, cut: Tensor, unit: Tensor,
                      rbf: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """The layer on the fused pair sums and the compact parts (scalar, vector, six entries)."""
        from xnn.gnn.fast.tensornet import stage
        from xnn.gnn.fast.tensornet_kernels import embedding
        rows = embedding(center, neighbor, cut, self.distance_proj1(rbf),
                         self.distance_proj2(rbf), self.distance_proj3(rbf), unit, src, dst)
        training = self.training
        I, v, S, squared = stage("embedding_parts", rows, training)(rows)
        norm = self.init_norm(squared)
        for linear in self.linears_scalar:
            norm = self.act(linear(norm))
        norm = norm.reshape(rows.shape[0], 3, self.n_features)
        return stage("compose_rows", rows, training)(self.linears_tensor[0](I) * norm[:, 0],
                                                     self.linears_tensor[1](v) * norm[:, 1, None, :],
                                                     self.linears_tensor[2](S) * norm[:, 2, None, :])


class _Interaction(nn.Module, FastPathModule):
    """One interaction layer (section 3.3 of the TensorNet paper).

    Parameters
    ----------
    n_features : int
        Feature width ``F``.
    n_rbf : int
        Number of radial basis functions.
    activation : str
        Nonlinearity of the edge filter MLP.
    o3 : bool
        ``True`` for the ``O(3)`` products ``YM + MY``, ``False`` for the
        ``SO(3)`` product ``2 YM``.

    Attributes
    ----------
    recompute : str
        ``"auto"`` (default): in evaluation mode from :data:`RECOMPUTE_MIN_PAIRS`
        pairs, the filter network keeps only its input for the backward pass and
        runs again there (the same values); ``"on"``: whenever a gradient is
        needed; ``"off"``: never. Set with
        :func:`~xnn.gnn.models.set_recompute_radial`.

    Notes
    -----
    A fused GPU implementation (:mod:`xnn.gnn.fast.tensornet`) evaluates the
    messages when :attr:`fast_active` is set (see :meth:`TensorNet.set_use_fast`).
    """

    recompute: str
    #: recomputes its filter network on request (see set_recompute_radial)
    radial_recompute = True

    def __init__(self, n_features: int, n_rbf: int, activation: str, o3: bool):
        super().__init__()
        self.n_features = n_features
        self.o3 = o3
        self.recompute = "auto"
        self.act = make_activation(activation)
        self.linears_scalar = nn.ModuleList([
            nn.Linear(n_rbf, n_features), nn.Linear(n_features, 2 * n_features),
            nn.Linear(2 * n_features, 3 * n_features)])
        self.linears_tensor = nn.ModuleList(
            [nn.Linear(n_features, n_features, bias=False) for _ in range(6)])
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """Triton on ``device`` and a power-of-two feature width."""
        from xnn.gnn.fast.tensornet import supported
        n = self.n_features
        return 16 <= n <= 512 and n & (n - 1) == 0 and supported(device, dtype)

    def _filters(self, rbf: Tensor) -> Tensor:
        """The edge filter MLP, ``(E, n_rbf) -> (E, 3 F)``."""
        if not torch.jit.is_scripting() and self._recomputes(rbf):
            return self._filters_recomputed(rbf)
        return self._filter_network(rbf)

    def _filter_network(self, rbf: Tensor) -> Tensor:
        for linear in self.linears_scalar:
            rbf = self.act(linear(rbf))
        return rbf

    @torch.jit.unused
    def _recomputes(self, rbf: Tensor) -> bool:
        if self.recompute == "off" or not (torch.is_grad_enabled() and rbf.requires_grad):
            return False
        if self.recompute == "on":
            return True
        return not self.training and rbf.shape[0] >= RECOMPUTE_MIN_PAIRS

    @torch.jit.unused
    def _filters_recomputed(self, rbf: Tensor) -> Tensor:
        from torch.utils.checkpoint import checkpoint
        return checkpoint(self._filter_network, rbf, use_reentrant=False)

    def forward(self, X: Tensor, src: Tensor, dst: Tensor, cut: Tensor, rbf: Tensor,
                q: Tensor) -> Tensor:
        """Update the tensors ``X`` of shape ``(N, 3, 3, F)``.

        Parameters
        ----------
        X : Tensor
            Current tensors ``(N, 3, 3, F)``.
        src, dst : Tensor
            Neighbor ``j`` and center ``i`` of every pair, self pairs included,
            ``(P,)`` each.
        cut : Tensor
            The cosine cutoff of every pair ``(P,)``.
        rbf : Tensor
            Radial basis of the pair distances ``(P, n_rbf)``.
        q : Tensor
            Total charge of every atom's structure, ``(N,)``.
        """
        if self.fast_active and not torch.jit.is_scripting():
            return self._forward_fast(X, src, dst, cut, rbf, q)
        n, F = X.shape[0], self.n_features
        filters = (self._filters(rbf) * cut.unsqueeze(-1)).reshape(-1, 3, F)

        X = X / (frobenius(X) + 1.0)[:, None, None, :]
        scalar, skew, symmetric = decompose(X)
        scalar = self.linears_tensor[0](scalar)
        skew = self.linears_tensor[1](skew)
        symmetric = self.linears_tensor[2](symmetric)
        Y = compose(scalar, skew, symmetric)

        # messages from the neighbors and, through the self pairs, from the atom itself
        vector = axial_vector(skew)
        m_scalar = scatter_sum(filters[:, 0] * gather_rows(scalar, src), dst, n)
        m_vector = scatter_sum(filters[:, 1].unsqueeze(1) * gather_rows(vector, src), dst, n)
        m_symmetric = scatter_sum(filters[:, 2][:, None, None, :] * gather_rows(symmetric, src),
                                  dst, n)
        message = compose(m_scalar, skew_tensor(m_vector), m_symmetric)

        charge_scale = (1.0 + 0.1 * q)[:, None, None, None]
        if self.o3:
            C = charge_scale * (tensor_matmul(Y, message) + tensor_matmul(message, Y))
        else:
            C = 2.0 * tensor_matmul(Y, message)
        scalar, skew, symmetric = decompose(C)
        norm = frobenius(C) + 1.0
        scalar = self.linears_tensor[3](scalar / norm)
        skew = self.linears_tensor[4](skew / norm[:, None, None, :])
        symmetric = self.linears_tensor[5](symmetric / norm[:, None, None, :])
        dX = compose(scalar, skew, symmetric)
        return X + dX + charge_scale * tensor_matmul(dX, dX)

    @torch.jit.unused
    def _forward_fast(self, X: Tensor, src: Tensor, dst: Tensor, cut: Tensor, rbf: Tensor,
                      q: Tensor) -> Tensor:
        """The layer on the fused messages and the compact parts (scalar, vector, six entries)."""
        from xnn.gnn.fast.tensornet import stage
        from xnn.gnn.fast.tensornet_kernels import messages
        training = self.training
        Xn, I, v, S = stage("interaction_parts", X, training)(X)
        I = self.linears_tensor[0](I)
        v = self.linears_tensor[1](v)
        S = self.linears_tensor[2](S)
        m = messages(torch.cat([I.unsqueeze(1), v, S], dim=1), self._filters(rbf), cut, src, dst)
        product = "interaction_product_o3" if self.o3 else "interaction_product_so3"
        I, v, S = stage(product, X, training)(I, v, S, m, q)
        return stage("interaction_update", X, training)(Xn, self.linears_tensor[3](I),
                                                        self.linears_tensor[4](v),
                                                        self.linears_tensor[5](S), q)


@register_model("tensornet")
class TensorNet(ExactFloat64Constants, InteratomicPotential):
    """TensorNet interatomic potential (rank-2 Cartesian tensor features).

    See the module docstring for the equations. The defaults are the
    reference architecture: ``F = 128``, two interaction layers, 32 fixed
    exponential-normal radial functions with a cosine cutoff at 4.5 Angstrom,
    ``O(3)`` products.

    Parameters
    ----------
    n_features : int, optional
        Number of tensor features ``F``, by default 128.
    n_interactions : int, optional
        Number of interaction layers (``0`` keeps the embedding alone), by
        default 2.
    n_rbf : int, optional
        Number of radial basis functions, by default 32.
    cutoff : float, optional
        Neighbor-list radius and cosine cutoff (Angstrom), by default 4.5.
    radial_basis : str, optional
        Any basis of :func:`~xnn.common.featurizers.make_radial_basis`:
        ``"expnorm"`` (default), ``"gaussian"`` or ``"bessel"``.
    trainable_rbf : bool, optional
        Learn the parameters of the basis (the exponential-normal centers and
        widths, the Bessel frequencies), by default ``False``.
    activation : str, optional
        Nonlinearity of the filter MLPs and the readout, by default ``"silu"``.
    equivariance_group : str, optional
        ``"O(3)"`` (default; energies invariant under rotations and
        reflections) or ``"SO(3)"`` (rotations only, so enantiomers can
        differ).
    n_elements : int, optional
        Size of the embedding table, indexed by ``Z``, by default 128.
    energy_shift : float, optional
        Additive per-atom energy standardization, by default 0.0.
    energy_scale : float, optional
        Multiplicative per-atom energy standardization, by default 1.0.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref``.

    Attributes
    ----------
    cutoff : float
        Neighbor-list cutoff radius.
    rbf : torch.nn.Module
        The radial basis.
    cutoff_fn : torch.nn.Module
        The cosine cutoff of the embedding and the filters.
    tensor_embedding : torch.nn.Module
        The tensor embedding layer.
    layers : torch.nn.ModuleList
        The interaction layers.
    out_norm : torch.nn.LayerNorm
        Layer norm of the ``(3 I^2, |A|^2, |S|^2)`` readout features.
    linear : torch.nn.Linear
        The ``3F -> F`` readout projection.
    readout : torch.nn.Sequential
        Atom-wise energy network ``F -> F/2 -> 1``.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference, initialized to zero.
    """

    head_modules = ("readout", "energy_scale", "energy_shift", "atom_ref")

    def __init__(self, n_features: int = 128, n_interactions: int = 2, n_rbf: int = 32,
                 cutoff: float = 4.5, radial_basis: str = "expnorm",
                 trainable_rbf: bool = False, activation: str = "silu",
                 equivariance_group: str = "O(3)", n_elements: int = 128,
                 energy_shift: float = 0.0, energy_scale: float = 1.0,
                 species=None, atomic_energies=None):
        super().__init__()
        if equivariance_group not in _GROUPS:
            raise ValueError(f"equivariance_group must be one of {_GROUPS}, "
                             f"got {equivariance_group!r}")
        self.cutoff = float(cutoff)
        if species is not None:
            self.species = [int(z) for z in species]
        self.n_features = n_features
        self.node_feature_dim = n_features
        self.n_interactions = n_interactions
        self.equivariance_group = equivariance_group
        self.rbf = make_radial_basis(radial_basis, n_rbf, cutoff, trainable_rbf)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.tensor_embedding = _TensorEmbedding(n_features, n_rbf, n_elements, activation)
        self.layers = nn.ModuleList([
            _Interaction(n_features, n_rbf, activation, equivariance_group == "O(3)")
            for _ in range(n_interactions)])
        self.out_norm = nn.LayerNorm(3 * n_features)
        self.linear = nn.Linear(3 * n_features, n_features)
        self.act = make_activation(activation)
        self.readout = nn.Sequential(
            nn.Linear(n_features, n_features // 2), make_activation(activation),
            nn.Linear(n_features // 2, 1))
        for module in self.readout:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        self.atom_ref = nn.Embedding(n_elements, 1)
        nn.init.zeros_(self.atom_ref.weight)
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)
        # the fused embedding and interaction layers (see set_use_fast); "auto" takes them
        # where measured faster
        self.use_fast = "auto"

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "TensorNet":
        """Choose the implementation of the embedding and interaction layers.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the fused GPU kernels of
            :mod:`xnn.gnn.fast.tensornet` where they are available and faster
            for the input (a CUDA device, Triton, a graph above the size
            crossover measured for evaluation, ``AUTO_POLICY``, or for training,
            ``AUTO_POLICY_TRAINING``); ``True``: wherever they are available; ``False``:
            always the reference implementation. The kernels are twice
            differentiable, so force training takes them as well. In float32
            the kernels agree with the reference to about 1e-6 relative, in
            float64 to about 1e-14.

        Returns
        -------
        TensorNet
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> None:
        """Switch the fused layers on or off for one evaluation."""
        from xnn.common.models.fast import fast_modules, select
        from xnn.gnn.fast.tensornet import AUTO_POLICY, AUTO_POLICY_TRAINING
        layers = self.__dict__.get("_fast_layers")
        if layers is None:
            layers = fast_modules(self)
            self.__dict__["_fast_layers"] = layers
        like = self.linear.weight
        policy = AUTO_POLICY_TRAINING if self.training else AUTO_POLICY
        select(layers, self.use_fast, policy, like.device, like.dtype, num_edges)

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
            raise ValueError(
                f"got {ae.numel()} atomic energies for {len(list(species))} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(list(species)), 0] = ae

    @torch.jit.export
    def tensors(self, atomic_numbers: Tensor, edge_index: Tensor, edge_vec: Tensor,
                total_charge: Tensor) -> Tensor:
        """TorchScript-compatible trunk: the tensor features ``X`` of every atom.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 the neighbor ``j``, row 1 the
            center ``i``.
        edge_vec : Tensor
            Edge vectors ``pos[i] - pos[j]`` of shape ``(E, 3)`` (with any
            periodic image shift applied).
        total_charge : Tensor
            Total charge of every atom's structure, ``(N,)``.

        Returns
        -------
        Tensor
            The tensors ``X`` of shape ``(N, 3, 3, F)`` after the last layer.
        """
        r = torch.linalg.norm(edge_vec, dim=-1)
        unit = edge_vec / r.clamp(min=1e-12).unsqueeze(-1)
        # the sums over neighbors include the atom itself: one self pair per atom
        # (r = 0, no direction, cutoff 1) appended to the pairs
        n = atomic_numbers.shape[0]
        loops = torch.arange(n, dtype=edge_index.dtype, device=edge_index.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        r = torch.cat([r, r.new_zeros((n,))])
        unit = torch.cat([unit, unit.new_zeros((n, 3))])
        cut = self.cutoff_fn(r)
        rbf = self.rbf(r)
        X = self.tensor_embedding(atomic_numbers, src, dst, cut, unit, rbf)
        for layer in self.layers:
            X = layer(X, src, dst, cut, rbf, total_charge)
        return X

    def _features(self, X: Tensor) -> Tensor:
        """Invariant per-atom features ``(N, F)`` from the tensors."""
        scalar, skew, symmetric = decompose(X)
        x = torch.cat([3.0 * scalar * scalar, frobenius(skew), frobenius(symmetric)], dim=-1)
        return self.act(self.linear(self.out_norm(x)))

    def _energy(self, x: Tensor, atomic_numbers: Tensor) -> Tensor:
        """Per-atom energies from the invariant features."""
        return (self.energy_scale * self.readout(x).squeeze(-1) + self.energy_shift
                + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core of a neutral system: features and per-atom energies.

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
            The invariant features ``(N, F)`` and the per-atom energy ``(N,)``.
        """
        q = torch.zeros(atomic_numbers.shape[0], dtype=edge_vec.dtype, device=edge_vec.device)
        x = self._features(self.tensors(atomic_numbers, edge_index, edge_vec, q))
        return x, self._energy(x, atomic_numbers)

    @torch.jit.export
    def node_features_energy_charges(self, atomic_numbers: Tensor, edge_index: Tensor,
                                     edge_vec: Tensor, pos: Tensor, cell: Tensor,
                                     pbc: Tensor, total_charge: float,
                                     spin_multiplicity: float
                                     ) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript-compatible core of a charged system (the deployment ABI).

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_vec : Tensor
            Edge vectors ``(E, 3)``.
        pos, cell, pbc : Tensor
            Unused (part of the shared ABI).
        total_charge : float
            Net charge of the system.
        spin_multiplicity : float
            Unused.

        Returns
        -------
        tuple of Tensor
            The invariant features ``(N, F)``, the per-atom energy ``(N,)``
            and an empty ``(0,)`` tensor (the model predicts no charges).
        """
        q = torch.full((atomic_numbers.shape[0],), total_charge, dtype=edge_vec.dtype,
                       device=edge_vec.device)
        x = self._features(self.tensors(atomic_numbers, edge_index, edge_vec, q))
        return x, self._energy(x, atomic_numbers), torch.zeros(0, dtype=x.dtype, device=x.device)

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy ``(N,)`` of a neutral system (the LAMMPS entry point)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energies of a batched graph.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph; its ``total_charge`` (per structure, zero
            when absent) enters the interaction layers.

        Returns
        -------
        dict of str to Tensor
            ``"node_energy"`` ``(N,)``, ``"energy"`` ``(B,)``,
            ``"node_features"`` (the invariant readout features, ``(N, F)``)
            and ``"node_tensors"`` (``(N, 3, 3, F)``).
        """
        self._select_fast(int(data.edge_index.shape[1]))
        edge_vec = data.edge_vectors()
        if data.total_charge is None:
            q = torch.zeros(data.num_nodes, dtype=edge_vec.dtype, device=edge_vec.device)
        else:
            q = data.total_charge.to(edge_vec.dtype)[data.batch]
        X = self.tensors(data.atomic_numbers, data.edge_index, edge_vec, q)
        x = self._features(X)
        node_energy = self._energy(x, data.atomic_numbers)
        return {"node_energy": node_energy,
                "energy": self.aggregate_energy(node_energy, data),
                "node_features": x, "node_tensors": X}

    @classmethod
    def from_config(cls, cfg) -> "TensorNet":
        """Build a :class:`TensorNet` from a configuration object.

        Core fields: ``cfg.n_features``, ``cfg.n_interactions``, ``cfg.n_rbf``,
        ``cfg.cutoff``. Everything else is read from ``cfg.extra``
        (``radial_basis``, ``trainable_rbf``, ``activation``,
        ``equivariance_group``, ``n_elements``, ``energy_shift``,
        ``energy_scale``, ``species``, ``atomic_energies``); the reference
        code's key spellings are translated by
        :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        TensorNet
            Instantiated model.
        """
        from xnn.common.config.coerce import coerce_per_species, coerce_species
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            radial_basis=str(extra.get("radial_basis", "expnorm")),
            trainable_rbf=bool(extra.get("trainable_rbf", False)),
            activation=str(extra.get("activation", "silu")),
            equivariance_group=str(extra.get("equivariance_group", "O(3)")),
            n_elements=int(extra.get("n_elements", 128)),
            energy_shift=float(extra.get("energy_shift", 0.0)),
            energy_scale=float(extra.get("energy_scale", 1.0)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
        )
