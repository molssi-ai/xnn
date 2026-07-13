"""NequIP (Batzner et al. 2022): E(3)-equivariant message-passing potential.

A faithful, self-contained NequIP built on the xnns equivariant-GNN
abstractions: it subclasses :class:`~xnns.gnn.models.base.EquivariantGNN`
(species bookkeeping, per-element reference energy ``atom_ref``, and the
:class:`~xnns.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge featurizer)
and adds the genuinely NequIP-specific pieces -- the :class:`InteractionBlock`
convolution, the gated :class:`ConvNetLayer`, and the per-species energy
scale/shift of the deployed upstream model.

Architecture (the upstream ``EnergyModel``): one-hot species -> linear chemical
embedding -> ``num_layers`` convnet layers (each an equivariant convolution of
the node features with the edge spherical harmonics, radially weighted, with an
element-dependent self-connection and a gated equivariant nonlinearity) -> two
linear atom-wise readouts to a per-atom energy, finally per-species
scale/shifted (``E_i = sigma_Z * eps_i + E0_Z``).

Given the same weights this reproduces the original ``nequip`` package
(mir-group/nequip) to machine precision -- the interaction blocks even share
the upstream parameter names (``linear_1``/``fc``/``tp``/``linear_2``/``sc``)
so ``load_state_dict`` transplants work directly (see ``tests/test_nequip.py``
and ``examples/fidelity_checks/nequip_verification.ipynb``).
Only ``e3nn`` is required -- no ``nequip`` / ``torch_runstats``.

Upstream conventions preserved here:

* the Bessel radial basis is *trainable* and normalized by ``2/r_max``
  (MACE uses fixed weights and ``sqrt(2/r_max)``);
* messages are divided by ``sqrt(avg_num_neighbors)`` (MACE divides by the
  full count);
* the spherical harmonics are evaluated on ``r_j - r_i`` (neighbour minus
  centre) -- the opposite orientation to the xnns/MACE edge vector, so the
  model flips the edge vectors internally;
* hidden feature irreps carry both parities per ``l`` when ``parity=True``
  and are pruned per layer to irreps reachable by some tensor-product path.

The model is TorchScript-deployable like MACE: the tensor-only
:meth:`NequIP.node_energy` core compiles under ``torch.jit.script`` (used by
the LAMMPS/TorchScript exporters in :mod:`xnns.common.deploy`). The e3nn
``Gate`` itself does not script on torch 2.x, so :class:`_Gate` re-implements
it exactly (same sorted input layout, same ``normalize2mom`` activations).
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies the
# class-level attribute annotations that TorchScript needs to resolve, breaking
# `torch.jit.script`.
from typing import Dict, Final, List, Optional, Tuple

import torch
from torch import Tensor, nn

from e3nn import o3
from e3nn.nn import FullyConnectedNet

from xnns.common.data import AtomicGraph
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model
from .base import EquivariantGNN
from .blocks import SCALAR_ACTIVATIONS as ACTS
from .blocks import (
    ScalarActivation,
    tp_out_irreps_with_instructions,
    tp_path_exists,
)


def nequip_hidden_irreps(num_features: int, l_max: int, parity: bool = True) -> o3.Irreps:
    """The NequIP hidden feature irreps (upstream ``feature_irreps_hidden``).

    ``num_features`` channels of every ``l = 0..l_max``; with ``parity`` both
    parities are included, in the upstream order (all even, then all odd).

    Parameters
    ----------
    num_features : int
        Channel multiplicity per irrep.
    l_max : int
        Maximum rotation order of the hidden features.
    parity : bool, optional
        Include odd-parity irreps (the full O(3) model), by default ``True``.

    Returns
    -------
    e3nn.o3.Irreps
        E.g. ``32x0e+32x1e+32x2e+32x0o+32x1o+32x2o`` for
        ``num_features=32, l_max=2, parity=True``.
    """
    return o3.Irreps(
        [(num_features, (l, p))
         for p in ((1, -1) if parity else (1,))
         for l in range(l_max + 1)]
    )


def _act_parity(act) -> int:
    """Parity of a scalar activation (+1 even, -1 odd), as e3nn's Activation does.

    Parameters
    ----------
    act : callable
        The scalar activation function.

    Returns
    -------
    int
        ``+1`` if ``act(-x) == act(x)``, ``-1`` if ``act(-x) == -act(x)``,
        ``0`` otherwise.
    """
    x = torch.linspace(0, 10, 256)
    a1, a2 = act(x), act(-x)
    if (a1 - a2).abs().max() < 1e-5:
        return 1
    if (a1 + a2).abs().max() < 1e-5:
        return -1
    return 0


class _Gate(nn.Module):
    """Scriptable, numerically exact stand-in for :class:`e3nn.nn.Gate` (0.4.4).

    The e3nn ``Gate`` does not compile under ``torch.jit.script`` on torch 2.x
    (its ``Activation`` submodule breaks), but its maths is simple: the input
    is the *sorted* concatenation of scalars, gate scalars and gated tensors
    (e3nn's ``_Sortcut`` layout); scalars and gates pass through second-moment
    normalized activations; each gated irrep is multiplied channel-wise by its
    activated gate. This module precomputes the sorted slice layout in
    ``__init__`` and reproduces the e3nn ``Gate`` bit-for-bit (checked in
    ``tests/test_nequip.py``). It carries no state, so the ``state_dict``
    layout matches the (parameter-free) e3nn original.

    Scope: one entry per distinct irrep in ``irreps_scalars``/``irreps_gated``
    and a single (merged) ``irreps_gates`` entry -- exactly what
    :class:`ConvNetLayer` produces.

    Parameters
    ----------
    irreps_scalars : e3nn.o3.Irreps
        The ``l = 0`` output irreps, activated directly (one act per entry).
    act_scalars : list of callable
        Activation per ``irreps_scalars`` entry.
    irreps_gates : e3nn.o3.Irreps
        Scalar gates, one channel per gated irrep channel (single entry).
    act_gates : list of callable
        Activation per ``irreps_gates`` entry.
    irreps_gated : e3nn.o3.Irreps
        The ``l > 0`` irreps, multiplied by the activated gates.

    Attributes
    ----------
    irreps_in : e3nn.o3.Irreps
        The (sorted, simplified) input irreps an upstream linear must produce.
    irreps_out : e3nn.o3.Irreps
        Output irreps: the scalars followed by the gated irreps.
    """

    scalar_starts: Final[List[int]]
    scalar_stops: Final[List[int]]
    gate_start: Final[int]
    gate_stop: Final[int]
    gated_starts: Final[List[int]]
    gated_stops: Final[List[int]]
    gated_muls: Final[List[int]]
    gated_dims: Final[List[int]]
    has_gates: Final[bool]

    def __init__(self, irreps_scalars, act_scalars, irreps_gates, act_gates,
                 irreps_gated):
        super().__init__()
        irreps_scalars = o3.Irreps(irreps_scalars).simplify()
        irreps_gates = o3.Irreps(irreps_gates).simplify()
        irreps_gated = o3.Irreps(irreps_gated).simplify()
        if len(irreps_scalars) != len(act_scalars) or len(irreps_gates) != len(act_gates):
            raise ValueError("one activation per irreps entry is required")
        if len(irreps_gates) > 1:
            raise ValueError(f"expected a single merged gates entry, got {irreps_gates}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(
                f"needs one gate per gated channel: {irreps_gates} vs {irreps_gated}"
            )

        # e3nn _Sortcut layout: sort the concatenated groups, then locate each
        # original entry's slice inside the sorted feature vector
        unsorted = irreps_scalars + irreps_gates + irreps_gated
        srt, perm, _ = unsorted.sort()
        offsets = [0]
        for mul, ir in srt:
            offsets.append(offsets[-1] + mul * ir.dim)
        slices = [(offsets[perm[k]], offsets[perm[k] + 1]) for k in range(len(unsorted))]
        n_s, n_g = len(irreps_scalars), len(irreps_gates)
        self.scalar_starts = [s for s, _ in slices[:n_s]]
        self.scalar_stops = [e for _, e in slices[:n_s]]
        self.has_gates = n_g > 0
        self.gate_start = slices[n_s][0] if self.has_gates else 0
        self.gate_stop = slices[n_s][1] if self.has_gates else 0
        self.gated_starts = [s for s, _ in slices[n_s + n_g:]]
        self.gated_stops = [e for _, e in slices[n_s + n_g:]]
        self.gated_muls = [mul for mul, _ in irreps_gated]
        self.gated_dims = [ir.dim for _, ir in irreps_gated]

        self.scalar_acts = nn.ModuleList(
            [ScalarActivation(o3.Irreps([mul_ir]), act)
             for mul_ir, act in zip(irreps_scalars, act_scalars)]
        )
        self.gate_act = ScalarActivation(
            irreps_gates, act_gates[0] if self.has_gates else None
        )

        # output irreps: activated scalars (parity may flip for even acts on
        # odd scalars, as in e3nn's Activation; None keeps the input parity),
        # then gates (x) gated
        out = []
        for (mul, ir), act in zip(irreps_scalars, act_scalars):
            if act is None:
                p_out = ir.p
            else:
                p_act = _act_parity(act)
                p_out = p_act if ir.p == -1 else ir.p
            if p_out == 0:
                raise ValueError(f"activation on odd scalars must have parity: {ir}")
            out.append((mul, (0, p_out)))
        p_gate = irreps_gates[0].ir.p if self.has_gates else 1
        out += [(mul, (ir.l, ir.p * p_gate)) for mul, ir in irreps_gated]
        self.irreps_out = o3.Irreps(out)
        self.irreps_in = srt.simplify()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated nonlinearity.

        Parameters
        ----------
        x : torch.Tensor
            Features of shape ``(N, irreps_in.dim)`` in the sorted layout.

        Returns
        -------
        torch.Tensor
            Activated features of shape ``(N, irreps_out.dim)``.
        """
        outs: List[Tensor] = []
        for i, act in enumerate(self.scalar_acts):
            outs.append(act(x[:, self.scalar_starts[i]:self.scalar_stops[i]]))
        if self.has_gates:
            gates = self.gate_act(x[:, self.gate_start:self.gate_stop])
            c = 0
            for i in range(len(self.gated_starts)):
                mul, d = self.gated_muls[i], self.gated_dims[i]
                piece = x[:, self.gated_starts[i]:self.gated_stops[i]]
                piece = piece.reshape(-1, mul, d) * gates[:, c:c + mul].unsqueeze(-1)
                outs.append(piece.reshape(-1, mul * d))
                c += mul
        return torch.cat(outs, dim=-1)


class InteractionBlock(nn.Module):
    """The NequIP equivariant convolution (upstream ``nequip.nn.InteractionBlock``).

    Messages are the tensor product of the neighbour node features with the
    edge spherical harmonics, weighted per edge by a radial MLP acting on the
    invariant radial embedding::

        m_ij  = TP(linear_1(h_j), Y(r_ij); w = fc(radial_ij))
        h_i'  = linear_2( sum_j m_ij / sqrt(avg_num_neighbors) ) + sc(h_i, z_i)

    where ``sc`` is the element-dependent self-connection
    (:class:`e3nn.o3.FullyConnectedTensorProduct` with the one-hot species).
    Member names match upstream (``linear_1``/``fc``/``tp``/``linear_2``/
    ``sc``) so state dicts transplant directly.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Irreps of the input node features.
    irreps_out : e3nn.o3.Irreps
        Irreps of the output node features (the gate's input irreps).
    irreps_node_attr : e3nn.o3.Irreps
        Irreps of the one-hot species node attributes.
    irreps_edge_attr : e3nn.o3.Irreps
        Irreps of the spherical-harmonic edge attributes.
    n_radial : int
        Width of the invariant radial edge embedding feeding ``fc``.
    invariant_layers : int, optional
        Hidden layers of the radial MLP, by default 2 (upstream
        ``example.yaml``; the bare upstream module defaults to 1).
    invariant_neurons : int, optional
        Hidden width of the radial MLP, by default 64.
    avg_num_neighbors : float or None, optional
        Divide the aggregated message by ``sqrt(avg_num_neighbors)``;
        ``None`` (default) disables the normalization.
    use_sc : bool, optional
        Include the self-connection, by default ``True``.
    nonlinearity_scalars : dict, optional
        Upstream-style ``{"e": <name>}`` choice of the radial-MLP
        nonlinearity, by default ``{"e": "silu"}``.
    """

    avg_num_neighbors: Optional[float]
    use_sc: Final[bool]

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps,
                 irreps_node_attr: o3.Irreps, irreps_edge_attr: o3.Irreps,
                 n_radial: int, invariant_layers: int = 2,
                 invariant_neurons: int = 64,
                 avg_num_neighbors: Optional[float] = None, use_sc: bool = True,
                 nonlinearity_scalars: Dict[str, str] = {"e": "silu"}):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.avg_num_neighbors = (
            float(avg_num_neighbors) if avg_num_neighbors is not None else None
        )
        self.use_sc = use_sc

        self.linear_1 = o3.Linear(irreps_in, irreps_in,
                                  internal_weights=True, shared_weights=True)

        # uvu tensor-product paths node (x) sh -> irreps_out; the original
        # nequip keeps enumeration order for the instructions (weight layout)
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            irreps_in, irreps_edge_attr, irreps_out, sort_instructions=False)
        self.tp = o3.TensorProduct(
            irreps_in, irreps_edge_attr, irreps_mid, instructions,
            shared_weights=False, internal_weights=False,
        )
        self.fc = FullyConnectedNet(
            [n_radial] + invariant_layers * [invariant_neurons] + [self.tp.weight_numel],
            ACTS[nonlinearity_scalars["e"]],
        )
        self.linear_2 = o3.Linear(irreps_mid.simplify(), irreps_out,
                                  internal_weights=True, shared_weights=True)
        if use_sc:
            self.sc = o3.FullyConnectedTensorProduct(
                irreps_in, irreps_node_attr, irreps_out
            )

    def forward(self, x: Tensor, node_attrs: Tensor, edge_index: Tensor,
                edge_sh: Tensor, edge_radial: Tensor) -> Tensor:
        """Compute one convolution update.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(N, irreps_in.dim)``.
        node_attrs : torch.Tensor
            One-hot species node attributes of shape ``(N, n_species)``.
        edge_index : torch.Tensor
            Edge index of shape ``(2, E)``; features are gathered from row 0
            (the neighbour) and scattered onto row 1 (the centre).
        edge_sh : torch.Tensor
            Spherical-harmonic edge attributes, shape ``(E, irreps_sh.dim)``.
        edge_radial : torch.Tensor
            Invariant radial edge embedding, shape ``(E, n_radial)``.

        Returns
        -------
        torch.Tensor
            Updated node features of shape ``(N, irreps_out.dim)``.
        """
        weight = self.fc(edge_radial)
        x_in = x
        x = self.linear_1(x)
        edge_feats = self.tp(x[edge_index[0]], edge_sh, weight)
        # divide first for numerics; the scatter is linear (upstream convention)
        avg: Optional[float] = self.avg_num_neighbors
        if avg is not None:
            edge_feats = edge_feats.div(avg ** 0.5)
        out = scatter_sum(edge_feats, edge_index[1], x.shape[0])
        out = self.linear_2(out)
        if self.use_sc:
            out = out + self.sc(x_in, node_attrs)
        return out


class ConvNetLayer(nn.Module):
    """One NequIP layer: :class:`InteractionBlock` + gated nonlinearity (+ resnet).

    Mirrors ``nequip.nn.ConvNetLayer``: the desired hidden irreps are pruned to
    those reachable by a tensor-product path from the current features and the
    edge attributes, split into scalars (activated directly) and gated irreps
    (multiplied by activated scalar gates), and the convolution outputs the
    gate's input irreps. The residual ("resnet") update is applied only when
    the layer preserves the feature irreps.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Irreps of the incoming node features.
    feature_irreps_hidden : e3nn.o3.Irreps
        The desired hidden irreps (see :func:`nequip_hidden_irreps`).
    irreps_node_attr : e3nn.o3.Irreps
        Irreps of the one-hot species node attributes.
    irreps_edge_attr : e3nn.o3.Irreps
        Irreps of the spherical-harmonic edge attributes.
    n_radial : int
        Width of the invariant radial edge embedding.
    resnet : bool, optional
        Residual update when the irreps allow it, by default ``False``.
    nonlinearity_scalars, nonlinearity_gates : dict, optional
        Per-parity activation names (upstream defaults
        ``{"e": "silu", "o": "tanh"}``).
    **conv_kwargs
        Forwarded to :class:`InteractionBlock` (``invariant_layers``,
        ``invariant_neurons``, ``avg_num_neighbors``, ``use_sc``).

    Attributes
    ----------
    conv : InteractionBlock
        The convolution (named ``conv`` to match upstream state dicts).
    equivariant_nonlin : _Gate
        The gated nonlinearity.
    irreps_out : e3nn.o3.Irreps
        Irreps of the output node features.
    """

    resnet: Final[bool]

    def __init__(self, irreps_in: o3.Irreps, feature_irreps_hidden: o3.Irreps,
                 irreps_node_attr: o3.Irreps, irreps_edge_attr: o3.Irreps,
                 n_radial: int, resnet: bool = False,
                 nonlinearity_scalars: Dict[str, str] = {"e": "silu", "o": "tanh"},
                 nonlinearity_gates: Dict[str, str] = {"e": "silu", "o": "tanh"},
                 **conv_kwargs):
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        acts_s = {1: ACTS[nonlinearity_scalars["e"]], -1: ACTS[nonlinearity_scalars["o"]]}
        acts_g = {1: ACTS[nonlinearity_gates["e"]], -1: ACTS[nonlinearity_gates["o"]]}

        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in feature_irreps_hidden
             if ir.l == 0 and tp_path_exists(irreps_in, irreps_edge_attr, ir)]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in feature_irreps_hidden
             if ir.l > 0 and tp_path_exists(irreps_in, irreps_edge_attr, ir)]
        )
        ir_gate = "0e" if tp_path_exists(irreps_in, irreps_edge_attr, "0e") else "0o"
        n_gated = sum(mul for mul, _ in irreps_gated)
        irreps_gates = o3.Irreps([(n_gated, ir_gate)]) if n_gated > 0 else o3.Irreps([])

        self.equivariant_nonlin = _Gate(
            irreps_scalars, [acts_s[ir.p] for _, ir in irreps_scalars],
            irreps_gates, [acts_g[o3.Irrep(ir_gate).p]] if n_gated > 0 else [],
            irreps_gated,
        )
        self.conv = InteractionBlock(
            irreps_in, self.equivariant_nonlin.irreps_in.simplify(),
            irreps_node_attr, irreps_edge_attr, n_radial,
            nonlinearity_scalars=nonlinearity_scalars, **conv_kwargs,
        )
        self.irreps_out = self.equivariant_nonlin.irreps_out
        self.resnet = bool(resnet) and self.irreps_out == irreps_in

    def forward(self, x: Tensor, node_attrs: Tensor, edge_index: Tensor,
                edge_sh: Tensor, edge_radial: Tensor) -> Tensor:
        """Apply one message-passing layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(N, irreps_in.dim)``.
        node_attrs : torch.Tensor
            One-hot species node attributes.
        edge_index : torch.Tensor
            Edge index of shape ``(2, E)``.
        edge_sh : torch.Tensor
            Spherical-harmonic edge attributes.
        edge_radial : torch.Tensor
            Invariant radial edge embedding.

        Returns
        -------
        torch.Tensor
            Updated node features of shape ``(N, irreps_out.dim)``.
        """
        old_x = x
        x = self.equivariant_nonlin(
            self.conv(x, node_attrs, edge_index, edge_sh, edge_radial)
        )
        if self.resnet:
            x = old_x + x
        return x


@register_model("nequip")
class NequIP(EquivariantGNN):
    """Faithful NequIP (Batzner et al. 2022) with a flexible number of layers.

    Subclasses :class:`~xnns.gnn.models.base.EquivariantGNN`, inheriting
    species bookkeeping, the per-element reference energy ``atom_ref`` (the
    NequIP per-species *shift*) and the shared
    :class:`~xnns.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge
    featurizer (configured with the NequIP radial conventions), and adds the
    NequIP layers and readout. All architecture options are read from
    ``ModelConfig.extra`` (see :meth:`NequIP.from_config`); upstream NequIP
    yaml spellings (``r_max``, ``num_layers``, ``num_basis``, ...) are
    translated to the xnns names at config-load time by the key-translation
    registry in :mod:`xnns.common.config.translate`.

    Parameters
    ----------
    species : list of int
        Atomic numbers of the supported elements, in channel order.
    cutoff : float, optional
        Radial cutoff ``r_max`` in angstrom, by default 4.0.
    l_max : int, optional
        Maximum rotation order of the hidden features and edge spherical
        harmonics, by default 2.
    parity : bool, optional
        Use both parities per ``l`` (the full O(3) model), by default ``True``.
    n_rbf : int, optional
        Number of Bessel basis functions (``num_basis``), by default 8.
    n_layers : int, optional
        Number of convnet layers (``num_layers``), by default 3. ``0`` gives a
        pure per-species baseline.
    num_features : int, optional
        Channel multiplicity of the hidden irreps, by default 32.
    invariant_layers : int, optional
        Hidden layers of the radial MLP, by default 2.
    invariant_neurons : int, optional
        Hidden width of the radial MLP, by default 64.
    avg_num_neighbors : float or None, optional
        Message normalization ``sqrt(avg_num_neighbors)``; ``None`` (default)
        disables it. Pass the training-set average (upstream ``auto``).
    use_sc : bool, optional
        Use the element-dependent self-connection, by default ``True``.
    resnet : bool, optional
        Residual updates between layers of equal irreps, by default ``False``.
    nonlinearity_scalars, nonlinearity_gates : dict or None, optional
        Per-parity activation names, upstream defaults
        ``{"e": "silu", "o": "tanh"}``.
    num_polynomial_cutoff : int, optional
        Degree ``p`` of the polynomial cutoff envelope, by default 6.
    trainable_rbf : bool, optional
        Learnable Bessel frequencies (upstream default), by default ``True``.
    conv_to_output_hidden : int or None, optional
        Width of the scalar readout hidden layer; ``None`` (default) uses the
        upstream ``max(1, num_features // 2)``.
    atomic_energies : torch.Tensor or None, optional
        Per-species energy *shifts* (upstream ``per_species_rescale_shifts``,
        in the same order as ``species``), used to initialise ``atom_ref``.
    atomic_scales : torch.Tensor or None, optional
        Per-species energy *scales* (upstream ``per_species_rescale_scales``
        with any global rescale folded in). Default is 1 for every species.

    Raises
    ------
    ValueError
        If ``n_layers`` is negative.
    """

    def __init__(
        self,
        species: List[int],
        cutoff: float = 4.0,
        l_max: int = 2,
        parity: bool = True,
        n_rbf: int = 8,
        n_layers: int = 3,
        num_features: int = 32,
        invariant_layers: int = 2,
        invariant_neurons: int = 64,
        avg_num_neighbors: Optional[float] = None,
        use_sc: bool = True,
        resnet: bool = False,
        nonlinearity_scalars: Optional[Dict[str, str]] = None,
        nonlinearity_gates: Optional[Dict[str, str]] = None,
        num_polynomial_cutoff: int = 6,
        trainable_rbf: bool = True,
        conv_to_output_hidden: Optional[int] = None,
        atomic_energies: Optional[Tensor] = None,
        atomic_scales: Optional[Tensor] = None,
    ):
        if n_layers < 0:
            raise ValueError("n_layers (num_layers) must be >= 0")
        # EquivariantGNN gives species/z_to_index/node_attr/atom_ref/edge_feat/
        # irreps_sh; the featurizer uses the NequIP radial conventions
        # (trainable Bessel with prefactor 2/r_max, degree-p envelope).
        super().__init__(species, cutoff, l_max, n_rbf,
                         p=num_polynomial_cutoff, trainable_rbf=trainable_rbf,
                         rbf_prefactor=2.0 / cutoff)
        # the SH *values* are parity-independent; only the irreps bookkeeping
        # changes when parity=False (all-even SE(3) variant)
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max, p=-1 if parity else 1)

        nonlinearity_scalars = nonlinearity_scalars or {"e": "silu", "o": "tanh"}
        nonlinearity_gates = nonlinearity_gates or {"e": "silu", "o": "tanh"}
        feature_irreps_hidden = nequip_hidden_irreps(num_features, l_max, parity)

        irreps = o3.Irreps([(num_features, (0, 1))])
        self.chemical_embedding = o3.Linear(self.node_attr_irreps, irreps)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = ConvNetLayer(
                irreps, feature_irreps_hidden, self.node_attr_irreps,
                self.irreps_sh, n_rbf, resnet=resnet,
                nonlinearity_scalars=nonlinearity_scalars,
                nonlinearity_gates=nonlinearity_gates,
                invariant_layers=invariant_layers,
                invariant_neurons=invariant_neurons,
                avg_num_neighbors=avg_num_neighbors, use_sc=use_sc,
            )
            self.layers.append(layer)
            irreps = layer.irreps_out
        # output block: two atom-wise linears (scalar hidden layer -> energy)
        hidden = conv_to_output_hidden if conv_to_output_hidden is not None \
            else max(1, num_features // 2)
        irreps_out_hidden = o3.Irreps([(hidden, (0, 1))])
        self.conv_to_output_hidden = o3.Linear(irreps, irreps_out_hidden)
        # invariant features (for e.g. LES): the scalar conv-to-output layer
        self.node_feature_dim = irreps_out_hidden.dim
        self.output_hidden_to_scalar = o3.Linear(irreps_out_hidden, o3.Irreps("1x0e"))

        # per-species scale/shift (upstream PerSpeciesScaleShift, deployed form)
        self.register_buffer("atom_scale", torch.ones(200))
        if atomic_scales is not None:
            sc = torch.as_tensor(atomic_scales, dtype=self.atom_scale.dtype)
            if sc.numel() != len(self.species):
                raise ValueError(
                    f"got {sc.numel()} atomic scales for {len(self.species)} species"
                )
            self.atom_scale[torch.tensor(self.species)] = sc
        if atomic_energies is not None:
            self.set_atomic_energies(atomic_energies)

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: tensors in, features + energies out.

        The single implementation reused by :meth:`node_energy` (the deploy
        entry point) and :meth:`forward`; it avoids the
        :class:`~xnns.common.data.AtomicGraph` dataclass. Embeds the species,
        runs the convnet layers, and reads out both the invariant
        conv-to-output features (what
        :class:`~xnns.common.models.les.LatentEwald` consumes) and the
        per-atom energy with the per-species scale/shift applied.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : torch.Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbour)
            and row 1 the destination (centre) node of each edge.
        edge_vec : torch.Tensor
            Edge displacement vectors ``pos[dst] - pos[src]`` of shape
            ``(E, 3)`` (already accounting for periodic cell shifts).

        Returns
        -------
        tuple of torch.Tensor
            The invariant conv-to-output features ``(N, node_feature_dim)``
            and the per-atom energy ``(N,)``.
        """
        node_attrs = self.node_attr(atomic_numbers)
        # NequIP evaluates Y_l on r_j - r_i (neighbour minus centre); the xnns
        # edge vector points the other way (centre minus neighbour), so flip.
        _, edge_sh, edge_radial = self.edge_feat.embed(-edge_vec)
        x = self.chemical_embedding(node_attrs)
        for layer in self.layers:
            x = layer(x, node_attrs, edge_index, edge_sh, edge_radial)
        features = self.conv_to_output_hidden(x)
        eps = self.output_hidden_to_scalar(features).squeeze(-1)
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

        Thin wrapper over :meth:`node_energy` (the scriptable tensor core):
        builds the edge displacement vectors from the graph and aggregates the
        per-node energies to a per-structure total.

        Parameters
        ----------
        data : xnns.common.data.AtomicGraph
            The input atomic graph (atomic numbers, edge index, positions, ...).

        Returns
        -------
        dict of str to torch.Tensor
            ``"node_energy"`` (per-node energies) and ``"energy"`` (per-structure
            total energy).
        """
        edge_vec = data.edge_vectors()
        features, node_energy = self.node_features_energy(
            data.atomic_numbers, data.edge_index, edge_vec)
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy,
                "node_features": features}

    @classmethod
    def from_config(cls, cfg) -> "NequIP":
        """Construct a :class:`NequIP` from a core model config.

        Reads the NequIP-specific hyper-parameters from ``cfg.extra`` (falling
        back to the upstream defaults) and the shared fields from ``cfg``
        directly. Only the xnns canonical key names are read here; upstream
        NequIP yaml spellings (``r_max``, ``num_layers``, ``num_basis``,
        ``chemical_symbols``, ``per_species_rescale_shifts``, ...) are
        translated to these names at config-load time by
        :mod:`xnns.common.config.translate`. Values copied from an upstream
        yaml are coerced: ``species`` accepts atomic numbers or chemical
        symbols (also as a string form), and ``atomic_energies`` /
        ``atomic_scales`` accept a list aligned with ``species``, a
        ``{Z: value}`` dict, a single number (broadcast), or the string form
        of any of these.

        Parameters
        ----------
        cfg : xnns.common.config.schema.ModelConfig
            The core model config, whose ``extra`` dict carries the NequIP
            architecture options.

        Returns
        -------
        NequIP
            The instantiated model.
        """
        from xnns.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=[1, 6, 8])

        if extra.get("global_rescale_scale") is not None:
            raise ValueError(
                "global_rescale_scale is not applied by the xnns NequIP; fold "
                "it into the per-species values (atomic_scales = "
                "global_rescale_scale * per_species_rescale_scales, likewise "
                "atomic_energies) as the deployed upstream model does"
            )

        avg = extra.get("avg_num_neighbors")
        if isinstance(avg, str):
            try:
                avg = float(avg)
            except ValueError:
                raise ValueError(
                    f"avg_num_neighbors = {avg!r} is not supported; pass the "
                    "training-set average as a number (upstream 'auto' needs "
                    "the dataset)"
                ) from None

        # readout hidden width: an int, or the upstream irreps string
        # (`conv_to_output_hidden_irreps_out`, e.g. "16x0e")
        hidden = extra.get("conv_to_output_hidden")
        if isinstance(hidden, str):
            hidden = o3.Irreps(hidden).count(o3.Irrep(0, 1))

        return cls(
            species=species,
            cutoff=cfg.cutoff,
            l_max=extra.get("l_max", 2),
            parity=extra.get("parity", True),
            n_rbf=cfg.n_rbf,
            n_layers=cfg.n_interactions,
            num_features=cfg.n_features,
            invariant_layers=extra.get("invariant_layers", 2),
            invariant_neurons=extra.get("invariant_neurons", 64),
            avg_num_neighbors=avg,
            use_sc=extra.get("use_sc", True),
            resnet=extra.get("resnet", False),
            nonlinearity_scalars=extra.get("nonlinearity_scalars"),
            nonlinearity_gates=extra.get("nonlinearity_gates"),
            num_polynomial_cutoff=extra.get("num_polynomial_cutoff", 6),
            trainable_rbf=extra.get("trainable_rbf", True),
            conv_to_output_hidden=hidden,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species,
                "atomic_energies (NequIP per_species_rescale_shifts)"),
            atomic_scales=coerce_per_species(
                extra.get("atomic_scales"), species,
                "atomic_scales (NequIP per_species_rescale_scales)"),
        )
