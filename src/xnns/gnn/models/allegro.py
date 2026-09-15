"""Allegro (Musaelian et al. 2023): strictly local equivariant potential.

A faithful, self-contained re-implementation of the original
mir-group/allegro (v0.3.0, the e3nn-era reference of the paper) on the xnns
equivariant-GNN abstractions: it subclasses
:class:`~xnns.gnn.models.base.EquivariantGNN` (species bookkeeping, per-species
energy shift ``atom_ref``, shared edge featurizer) and adds the genuinely
Allegro-specific pieces -- the two-body scalar embedding, the per-edge scalar /
tensor latent tracks, and the iterated weighted-environment tensor products
(paper eqs 7-16). Given the same weights it reproduces the original package to
machine precision (``tests/test_allegro.py`` and the block-by-block notebook),
needing only ``e3nn`` -- no ``nequip``/``allegro``/``opt_einsum_fx``.

Upstream conventions preserved (defaults of the reference ``uuulin`` mode):

* Bessel basis is *trainable* with the Allegro prefactor ``r_max / pi``
  (the "normalized sinc" ``AllegroBesselBasis``);
* spherical harmonics on ``r_j - r_i`` (flipped from the xnns edge vector);
* the environment sum is normalized by ``1/sqrt(avg_num_neighbors - 1)``
  (the current edge is subtracted out of its own environment) and the
  edgewise energy sum by ``1/sqrt(avg_num_neighbors)``;
* per-channel ("uuu") weightless tensor products whose Wigner-3j blocks are
  scaled by ``sqrt(2 l_out + 1)``, followed by a strided linear mix with
  ``1/sqrt(mul * n_paths)`` normalization;
* variance-preserving scalar MLPs (:class:`e3nn.nn.FullyConnectedNet` is
  transplant-identical to upstream's ``ScalarMLPFunction``);
* the cumulative-softmax latent resnet coefficients;
* per-species scale/shift ``E_i = sigma_Z eps_i + mu_Z`` of the deployed
  upstream model (``atom_ref`` shift + ``atom_scale`` buffer, as in the xnns
  NequIP).

The tensor-only :meth:`Allegro.node_energy` core compiles under
``torch.jit.script`` for the LAMMPS/TorchScript exporters in
:mod:`xnns.common.deploy` (the ``pair_allegro`` deployment path).
"""
# NOTE: no `from __future__ import annotations` -- TorchScript needs resolvable
# class-level annotations.
import math
from typing import Final, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3
from e3nn.nn import FullyConnectedNet

from xnns.common.data import AtomicGraph
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model
from .base import EquivariantGNN
from .blocks import tp_path_exists


class _ChannelWeighter(nn.Module):
    """Weight single-multiplicity irreps into ``mul_out`` channels.

    Numerically identical to allegro's ``MakeWeightedChannels``: every output
    channel ``u`` carries one learned weight per irrep entry ``r``, and that
    weight (times the constant ``alpha``) multiplies all ``2l + 1`` components
    of irrep ``r``. Here the broadcast is done with a per-component Long
    lookup buffer that records which irrep entry each flattened component
    belongs to; the flat weights are gathered through it to full feature
    width and then scaled by ``alpha``.

    Parameters
    ----------
    irreps : e3nn.o3.Irreps
        Input irreps, one multiplicity each.
    mul_out : int
        Number of output channels (``num_tensor_features``).
    alpha : float, optional
        Constant folded into the weights (the env-sum normalization). Kept as
        a floating buffer so that module dtype conversions round it the same
        way they round any other floating buffer.
    """

    weight_numel: Final[int]
    mul_out: Final[int]
    _num_irreps: Final[int]

    def __init__(self, irreps: o3.Irreps, mul_out: int, alpha: float = 1.0):
        super().__init__()
        irreps = o3.Irreps(irreps)
        self._num_irreps = len(irreps)
        self.mul_out = mul_out
        self.weight_numel = len(irreps) * mul_out
        # Component j of the flattened feature vector belongs to irrep entry
        # _which_irrep[j] (entry index repeated over its 2l + 1 components).
        which = torch.arange(self._num_irreps).repeat_interleave(
            torch.tensor([mul_ir.ir.dim for mul_ir in irreps])
        )
        self.register_buffer("_which_irrep", which, persistent=False)
        self.register_buffer("_alpha", torch.tensor(alpha), persistent=False)

    def forward(self, edge_attr: Tensor, weights: Tensor) -> Tensor:
        """Apply per-irrep channel weights.

        Parameters
        ----------
        edge_attr : torch.Tensor
            Edge attributes of shape ``(E, irreps.dim)``.
        weights : torch.Tensor
            Flat weights of shape ``(E, weight_numel)`` (channel-major).

        Returns
        -------
        torch.Tensor
            Weighted channels of shape ``(E, mul_out, irreps.dim)``.
        """
        w = weights.reshape(-1, self.mul_out, self._num_irreps)
        w = w.index_select(-1, self._which_irrep) * self._alpha
        return edge_attr.unsqueeze(-2) * w


class _Contracter(nn.Module):
    """Per-channel ("uuu") weightless tensor product (allegro's ``Contracter``).

    Contracts the strided features ``x1[z,u,i]`` with ``x2[z,u,j]`` through a
    block Wigner-3j tensor -- one output block per instruction, each scaled by
    ``sqrt(2 l_out + 1)`` (component normalization; every output block has a
    single path, exactly as in the upstream strided codegen).

    Parameters
    ----------
    irs_in1, irs_in2 : list of e3nn.o3.Irrep
        Per-entry irreps of the two (strided) operands.
    instructions : list of (int, int, e3nn.o3.Irrep)
        ``(i_in1, i_in2, ir_out)`` per output block, in output order.
    """

    def __init__(self, irs_in1: List[o3.Irrep], irs_in2: List[o3.Irrep],
                 instructions: List[Tuple[int, int, o3.Irrep]]):
        super().__init__()
        dim1 = sum(ir.dim for ir in irs_in1)
        dim2 = sum(ir.dim for ir in irs_in2)
        off1 = [0]
        for ir in irs_in1:
            off1.append(off1[-1] + ir.dim)
        off2 = [0]
        for ir in irs_in2:
            off2.append(off2[-1] + ir.dim)
        self.dim_out = sum(ir.dim for _, _, ir in instructions)
        w3j = torch.zeros(self.dim_out, dim1, dim2)
        k = 0
        for i1, i2, ir_out in instructions:
            block = o3.wigner_3j(ir_out.l, irs_in1[i1].l, irs_in2[i2].l)  # (k, i, j)
            block = block * math.sqrt(2 * ir_out.l + 1)
            w3j[k:k + ir_out.dim, off1[i1]:off1[i1 + 1], off2[i2]:off2[i2 + 1]] = block
            k += ir_out.dim
        self.register_buffer("_w3j", w3j)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Contract ``(E, mul, dim1) x (E, mul, dim2) -> (E, mul, dim_out)``."""
        return torch.einsum("zui,zuj,kij->zuk", x1, x2, self._w3j)


class _StridedLinear(nn.Module):
    """Equivariant linear mix in the strided layout (allegro's strided ``Linear``).

    Mixes, per channel block, the consecutive same-irrep entries of the
    tensor-product output into each output irrep with a
    ``1/sqrt(mul * n_paths)`` normalization. The internal weight vector uses
    the *upstream flat layout* (output-entry order, each group ``[v, u, n]``)
    so state dicts transplant directly.

    Parameters
    ----------
    irs_in : list of e3nn.o3.Irrep
        Per-entry irreps of the (strided) input, in order.
    irs_out : list of e3nn.o3.Irrep
        Output irreps (distinct entries).
    mul : int
        Channel multiplicity shared by input and output.
    """

    _starts: Final[List[int]]
    _stops: Final[List[int]]
    _ns: Final[List[int]]
    _dims: Final[List[int]]
    _norms: Final[List[float]]
    _wnums: Final[List[int]]
    mul: Final[int]

    def __init__(self, irs_in: List[o3.Irrep], irs_out: List[o3.Irrep], mul: int):
        super().__init__()
        self.mul = mul
        off = [0]
        for ir in irs_in:
            off.append(off[-1] + ir.dim)
        starts: List[int] = []
        stops: List[int] = []
        ns: List[int] = []
        dims: List[int] = []
        norms: List[float] = []
        wnums: List[int] = []
        for ir_out in irs_out:
            idx = [i for i, ir in enumerate(irs_in) if ir == ir_out]
            assert idx == list(range(idx[0], idx[-1] + 1)), \
                "inputs per output must be consecutive"
            starts.append(off[idx[0]])
            stops.append(off[idx[-1] + 1])
            ns.append(len(idx))
            dims.append(ir_out.dim)
            norms.append(1.0 / math.sqrt(mul * len(idx)))
            wnums.append(mul * len(idx) * mul)
        self._starts, self._stops, self._ns = starts, stops, ns
        self._dims, self._norms, self._wnums = dims, norms, wnums
        self.w = nn.Parameter(
            torch.empty(sum(wnums)).uniform_(-math.sqrt(3), math.sqrt(3)))

    def forward(self, x: Tensor) -> Tensor:
        """Mix ``(E, mul, dim_in) -> (E, mul, dim_out)``."""
        outs: List[Tensor] = []
        w_index = 0
        for i in range(len(self._starts)):
            piece = x[:, :, self._starts[i]:self._stops[i]]
            piece = piece.reshape(-1, self.mul, self._ns[i], self._dims[i])
            wg = self.w[w_index:w_index + self._wnums[i]].reshape(
                self.mul, self.mul, self._ns[i])
            w_index += self._wnums[i]
            outs.append(torch.einsum("vun,zuni->zvi", wg, piece) * self._norms[i])
        return torch.cat(outs, dim=-1)


@register_model("allegro")
class Allegro(EquivariantGNN):
    """Faithful Allegro (Musaelian et al. 2023): strictly local pair energies.

    The energy is decomposed into pair energies,
    ``E = sum_i [ sigma_Z_i (sum_j E_ij / sqrt(lambda)) + mu_Z_i ]``
    (paper eqs 5-6). Each ordered pair ``ij`` carries an invariant scalar
    latent and an equivariant tensor latent that interact at every layer: the
    scalar latent generates the weights embedding atom ``i``'s environment, a
    per-channel tensor product couples the pair tensors with that embedded
    environment (eqs 11-14), its scalar outputs feed back into the scalar
    latent (eq 15), and a linear layer mixes the tensor channels (eq 16).
    There is **no message passing** -- every pair energy is a function of the
    fixed local environment, which is what makes Allegro scalable.

    All architecture options are read from ``ModelConfig.extra`` (see
    :meth:`from_config`); upstream allegro yaml spellings are translated to
    the xnns names at config-load time by
    :mod:`xnns.common.config.translate`.

    Parameters
    ----------
    species : list of int
        Atomic numbers of the supported elements, in channel order.
    cutoff : float, optional
        Radial cutoff ``r_max``, by default 6.0.
    l_max : int, optional
        Maximum rotation order of the tensor track, by default 1.
    parity : str, optional
        ``"o3_full"`` (default; all irreps up to ``l_max``),
        ``"o3_restricted"`` (only SH irreps) or ``"so3"`` (no parity).
    n_rbf : int, optional
        Bessel basis size, by default 8.
    num_layers : int, optional
        Number of tensor-product layers (>= 1), by default 2.
    num_tensor_features : int, optional
        Channel multiplicity of the tensor track, by default 32.
    two_body_latent : list of int, optional
        Hidden+output widths of the two-body scalar MLP (upstream
        ``two_body_latent_mlp_latent_dimensions``), default ``[32, 64, 128]``.
    latent : list of int, optional
        Hidden+output widths of the later latent MLPs (upstream
        ``latent_mlp_latent_dimensions``), default ``[128]``. The final width
        must match ``two_body_latent[-1]`` when ``latent_resnet`` is on.
    env_embed : list of int, optional
        Hidden widths of the environment-embedding MLPs (upstream
        ``env_embed_mlp_latent_dimensions``), default ``[]`` (linear).
    edge_eng : list of int, optional
        Hidden widths of the final edge-energy MLP (upstream
        ``edge_eng_mlp_latent_dimensions``), default ``[32]``.
    initial_scalar_embedding_dim : int or None, optional
        Width of the two-body radial-chemical product embedding; ``None``
        (default) uses ``two_body_latent[0]`` (upstream default). Must be even.
    avg_num_neighbors : float or None, optional
        Environment/energy-sum normalization; ``None`` (default) disables it.
    latent_resnet : bool, optional
        Cumulative-softmax residual updates of the scalar latent, default True.
    num_polynomial_cutoff : int, optional
        Polynomial cutoff degree ``p``, by default 6.
    trainable_rbf : bool, optional
        Learnable Bessel frequencies (upstream default), by default True.
    atomic_energies, atomic_scales : array-like or None, optional
        Per-species energy shifts / scales (as in the xnns NequIP).
    """

    num_layers: Final[int]
    latent_resnet: Final[bool]
    _n_scalar_outs: Final[List[int]]
    _edge_w_numel: Final[int]
    _env_w_numel: Final[int]
    _energy_factor: Final[float]

    def __init__(
        self,
        species: List[int],
        cutoff: float = 6.0,
        l_max: int = 1,
        parity: str = "o3_full",
        n_rbf: int = 8,
        num_layers: int = 2,
        num_tensor_features: int = 32,
        two_body_latent: Optional[List[int]] = None,
        latent: Optional[List[int]] = None,
        env_embed: Optional[List[int]] = None,
        edge_eng: Optional[List[int]] = None,
        initial_scalar_embedding_dim: Optional[int] = None,
        avg_num_neighbors: Optional[float] = None,
        latent_resnet: bool = True,
        num_polynomial_cutoff: int = 6,
        trainable_rbf: bool = True,
        atomic_energies=None,
        atomic_scales=None,
    ):
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if parity not in ("o3_full", "o3_restricted", "so3"):
            raise ValueError(f"unknown parity setting {parity!r}")
        # EquivariantGNN gives species/z_to_index/node_attr/atom_ref/edge_feat;
        # the featurizer uses Allegro's radial conventions (trainable
        # "normalized sinc" Bessel with prefactor r_max/pi, degree-p envelope).
        super().__init__(species, cutoff, l_max, n_rbf,
                         p=num_polynomial_cutoff, trainable_rbf=trainable_rbf,
                         rbf_prefactor=cutoff / math.pi)
        self.irreps_sh = o3.Irreps.spherical_harmonics(
            l_max, p=1 if parity == "so3" else -1)
        two_body_latent = list(two_body_latent or [32, 64, 128])
        latent = list(latent or [128])
        env_embed = list(env_embed or [])
        edge_eng = list(edge_eng or [32])
        self.num_layers = num_layers
        self.latent_resnet = latent_resnet
        self.avg_num_neighbors = avg_num_neighbors

        # -- two-body radial x chemical product embedding (ProductTypeEmbedding)
        embed_dim = initial_scalar_embedding_dim or two_body_latent[0]
        if embed_dim % 2 != 0:
            raise ValueError("initial_scalar_embedding_dim must be even")
        self.type_embeddings = nn.Parameter(
            torch.randn(2, len(self.species), embed_dim // 2))
        self.basis_embed = FullyConnectedNet([n_rbf, embed_dim], None)

        # -- irreps of the tensor track: forward growth + backward pruning of
        #    unreachable/unneeded paths, exactly as upstream
        env_irs: List[o3.Irrep] = [ir for _, ir in self.irreps_sh]
        if parity == "o3_full":
            allowed = [o3.Irrep(l, p) for l in range(l_max + 1) for p in (1, -1)]
        else:
            allowed = list(env_irs)
        tps_irreps: List[List[o3.Irrep]] = [list(env_irs)]
        for layer in range(num_layers):
            want = [o3.Irrep(0, 1)] if layer == num_layers - 1 else allowed
            arg_now = o3.Irreps([(1, i) for i in tps_irreps[-1]])
            tps_irreps.append(
                [ir for ir in want
                 if tp_path_exists(arg_now, self.irreps_sh, ir)])
        out_irs = tps_irreps[-1]
        pruned = [out_irs]
        for arg in reversed(tps_irreps[:-1]):
            keep = [a for a in arg
                    if any(any(o in out_irs for o in a * e) for e in env_irs)]
            pruned.append(keep)
            out_irs = keep
        tps_irreps = list(reversed(pruned))

        # -- weighters, TPs, linears, MLPs
        self._edge_weighter = _ChannelWeighter(self.irreps_sh, num_tensor_features)
        env_alpha = 1.0
        if avg_num_neighbors is not None:
            # the edge itself is subtracted from its own environment sum
            env_alpha = 1.0 / math.sqrt(avg_num_neighbors - 1)
        self._env_weighter = _ChannelWeighter(self.irreps_sh, num_tensor_features,
                                              alpha=env_alpha)
        self._energy_factor = (1.0 / math.sqrt(avg_num_neighbors)
                               if avg_num_neighbors is not None else 1.0)

        self.tps = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.latents = nn.ModuleList()
        self.env_embed_mlps = nn.ModuleList()
        n_scalar_outs: List[int] = []
        for layer, (arg_irs, out_irs) in enumerate(zip(tps_irreps[:-1], tps_irreps[1:])):
            instr: List[Tuple[int, int, o3.Irrep]] = []
            full_out: List[o3.Irrep] = []
            n_scalar = 0
            for ir_out in out_irs:
                for i1, ir1 in enumerate(arg_irs):
                    for i2, ir2 in enumerate(env_irs):
                        if ir_out in ir1 * ir2:
                            if ir_out == o3.Irrep(0, 1):
                                n_scalar += 1
                            instr.append((i1, i2, ir_out))
                            full_out.append(ir_out)
            n_scalar_outs.append(n_scalar)
            self.tps.append(_Contracter(arg_irs, env_irs, instr))
            self.linears.append(_StridedLinear(full_out, out_irs, num_tensor_features))
            if layer == 0:
                self.latents.append(FullyConnectedNet(
                    [embed_dim] + two_body_latent, F.silu))
            else:
                self.latents.append(FullyConnectedNet(
                    [self.latents[-1].hs[-1]
                     + num_tensor_features * n_scalar_outs[-2]] + latent, F.silu))
            n_w = self._env_weighter.weight_numel
            if layer == 0:
                n_w += self._edge_weighter.weight_numel
            self.env_embed_mlps.append(FullyConnectedNet(
                [self.latents[-1].hs[-1]] + env_embed + [n_w],
                F.silu if env_embed else None))
        self._n_scalar_outs = n_scalar_outs
        self._edge_w_numel = self._edge_weighter.weight_numel
        self._env_w_numel = self._env_weighter.weight_numel

        self.final_latent = FullyConnectedNet(
            [self.latents[-1].hs[-1] + num_tensor_features * n_scalar_outs[-1]]
            + latent, F.silu)
        self.edge_eng = FullyConnectedNet(
            [self.final_latent.hs[-1]] + edge_eng + [1], F.silu)
        # invariant per-atom features (for e.g. LES): the final per-edge
        # scalar latents summed onto their centre atoms
        self.node_feature_dim = self.final_latent.hs[-1]

        # cumulative-softmax resnet coefficients (default zeros -> equal weights)
        self.register_buffer("_resnet_params", torch.zeros(num_layers + 1))

        # per-species scale/shift (deployed upstream PerSpeciesRescale)
        self.register_buffer("atom_scale", torch.ones(200))
        if atomic_scales is not None:
            sc = torch.as_tensor(atomic_scales, dtype=self.atom_scale.dtype)
            if sc.numel() != len(self.species):
                raise ValueError(
                    f"got {sc.numel()} atomic scales for {len(self.species)} species")
            self.atom_scale[torch.tensor(self.species)] = sc
        if atomic_energies is not None:
            self.set_atomic_energies(atomic_energies)

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: tensors in, features + energies out.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : torch.Tensor
            Edge index ``(2, E)``; row 0 is the neighbour, row 1 the centre.
        edge_vec : torch.Tensor
            Edge vectors ``pos[dst] - pos[src]``, shape ``(E, 3)``.

        Returns
        -------
        tuple of torch.Tensor
            The invariant node features ``(N, node_feature_dim)`` -- the
            final per-edge scalar latents summed onto their centre atoms
            (what :class:`~xnns.common.models.les.LatentEwald` consumes) --
            and the per-atom energy ``(N,)``.
        """
        num_atoms = atomic_numbers.shape[0]
        center = edge_index[1]                     # xnns dst == allegro centre
        # Allegro/NequIP orientation: Y_l on r_j - r_i (neighbour - centre)
        _, edge_sh, edge_radial = self.edge_feat.embed(-edge_vec)

        # two-body scalar embedding: (center type || neighbor type) * radial
        types = self.z_to_index[atomic_numbers].clamp(min=0)
        center_embed = self.type_embeddings[0][types[edge_index[1]]]
        neighbor_embed = self.type_embeddings[1][types[edge_index[0]]]
        edge_invariants = (torch.cat((center_embed, neighbor_embed), dim=-1)
                           * self.basis_embed(edge_radial))

        # resnet coefficients: cumulative softmax over the layer parameters
        coeff = (self._resnet_params - self._resnet_params.max()).exp()
        cumsum = coeff.cumsum(dim=0) + 1e-12

        latents = torch.zeros(1, device=edge_vec.device, dtype=edge_vec.dtype)
        latent_in = edge_invariants
        features = edge_sh
        layer: int = 0
        for latent_mlp, env_mlp, tp, linear in zip(
                self.latents, self.env_embed_mlps, self.tps, self.linears):
            new_latents = latent_mlp(latent_in)
            if self.latent_resnet and layer > 0:
                latents = ((cumsum[layer - 1] / cumsum[layer]).sqrt() * latents
                           + (coeff[layer] / cumsum[layer]).sqrt() * new_latents)
            else:
                latents = new_latents
            weights = env_mlp(latents)
            w_index: int = 0
            if layer == 0:
                edge_w = weights.narrow(-1, 0, self._edge_w_numel)
                w_index += self._edge_w_numel
                features = self._edge_weighter(features, edge_w)
            env_w = weights.narrow(-1, w_index, self._env_w_numel)
            env_edges = self._env_weighter(edge_sh, env_w)
            local_env = scatter_sum(env_edges, center, num_atoms)[center]
            local_env = local_env - env_edges   # env of all *other* neighbours
            features = tp(features, local_env)
            scalars = features[:, :, :self._n_scalar_outs[layer]].reshape(
                features.shape[0], -1)
            features = linear(features)
            latent_in = torch.cat((latents, scalars), dim=-1)
            layer += 1

        new_latents = self.final_latent(latent_in)
        if self.latent_resnet:
            latents = ((cumsum[layer - 1] / cumsum[layer]).sqrt() * latents
                       + (coeff[layer] / cumsum[layer]).sqrt() * new_latents)
        else:
            latents = new_latents

        edge_energy = self.edge_eng(latents).squeeze(-1) * self._energy_factor
        eps = scatter_sum(edge_energy, center, num_atoms)
        features = scatter_sum(latents, center, num_atoms)
        return features, (self.atom_scale[atomic_numbers] * eps
                          + self.atom_ref(atomic_numbers).squeeze(-1))

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy, shape ``(N,)`` (thin wrapper over
        :meth:`node_features_energy`; the deploy wrappers call this)."""
        out = self.node_features_energy(atomic_numbers, edge_index, edge_vec)
        return out[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict per-node and total energy for an atomic graph.

        Parameters
        ----------
        data : xnns.common.data.AtomicGraph
            The input atomic graph.

        Returns
        -------
        dict of str to torch.Tensor
            ``"node_energy"`` (per-atom energies) and ``"energy"``
            (per-structure totals).
        """
        features, node_energy = self.node_features_energy(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        return {"node_energy": node_energy,
                "energy": self.aggregate_energy(node_energy, data),
                "node_features": features}

    @classmethod
    def from_config(cls, cfg) -> "Allegro":
        """Construct an :class:`Allegro` from a core model config.

        Reads the Allegro hyper-parameters from ``cfg.extra``; upstream yaml
        spellings are translated by :mod:`xnns.common.config.translate` and
        value forms coerced by :mod:`xnns.common.config.coerce`.

        Parameters
        ----------
        cfg : xnns.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        Allegro
            The instantiated model.
        """
        from xnns.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=[1, 6, 8])
        avg = extra.get("avg_num_neighbors")
        if isinstance(avg, str):
            try:
                avg = float(avg)
            except ValueError:
                raise ValueError(
                    f"avg_num_neighbors = {avg!r}: pass the training-set "
                    "average as a number ('auto' needs the dataset)") from None
        return cls(
            species=species,
            cutoff=cfg.cutoff,
            l_max=extra.get("l_max", 1),
            parity=extra.get("parity", "o3_full"),
            n_rbf=cfg.n_rbf,
            num_layers=cfg.n_interactions,
            num_tensor_features=cfg.n_features,
            two_body_latent=extra.get("two_body_latent"),
            latent=extra.get("latent"),
            env_embed=extra.get("env_embed"),
            edge_eng=extra.get("edge_eng"),
            initial_scalar_embedding_dim=extra.get("initial_scalar_embedding_dim"),
            avg_num_neighbors=avg,
            latent_resnet=extra.get("latent_resnet", True),
            num_polynomial_cutoff=extra.get("num_polynomial_cutoff", 6),
            trainable_rbf=extra.get("trainable_rbf", True),
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species, "atomic_energies"),
            atomic_scales=coerce_per_species(
                extra.get("atomic_scales"), species, "atomic_scales"),
        )
