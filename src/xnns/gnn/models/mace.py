"""MACE (Batatia et al. 2022): higher body-order equivariant message passing.

A faithful, self-contained MACE built on the xnns equivariant-GNN abstractions:
it subclasses :class:`~xnns.gnn.models.base.EquivariantGNN` (species bookkeeping,
per-element reference energy ``atom_ref``, and the
:class:`~xnns.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge featurizer) and
adds the genuinely MACE-specific pieces -- the real
``RealAgnostic(Residual)InteractionBlock`` and a *learned symmetric contraction*
over Clebsch-Gordan paths (``correlation`` order).

The CG ``U`` basis (:func:`U_matrix_real`) is bit-identical to ``mace-torch`` and
the symmetric contraction reproduces it to ~1e-16 given the same weights
(see ``tests/test_gnn.py``). Only ``e3nn`` is required -- no ``mace-torch``,
``cuequivariance`` or ``opt_einsum_fx``.

The model is TorchScript-deployable: the tensor-only :meth:`MACE.node_energy`
core compiles under ``torch.jit.script`` (used by the LAMMPS/TorchScript
exporters in :mod:`xnns.common.deploy`) and reproduces the eager model to
machine precision (see ``tests/test_mace.py``).

Difference from upstream MACE: ``num_interactions`` (the number of message-passing
layers *T*) is fully flexible -- ``T = 0`` (a pure ``atom_ref``/pair-repulsion
baseline) through any ``T = N`` -- rather than being fixed to 2. All architecture
options are read from ``ModelConfig.extra`` (see :meth:`MACE.from_config`);
upstream MACE-CLI spellings (``r_max``, ``atomic_numbers``, ``E0s``, ...) are
translated to the xnns names at config-load time by the key-translation registry
in :mod:`xnns.common.config.translate`.

Vendored maths (CG coupling + symmetric contraction) is adapted from
ACEsuit/mace (MIT licence; authors Ilyes Batatia, Gregor Simm; CG based on e3nn
by Mario Geiger), simplified to plain ``torch.einsum`` with no codegen/cueq deps.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies the
# class-level attribute annotations that TorchScript needs to resolve (e.g.
# `dims: List[int]` on _ReshapeIrreps), breaking `torch.jit.script`.
from typing import Final, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3
from e3nn import nn as e3nn_nn

from xnns.common.data import AtomicGraph
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model
from .base import EquivariantGNN
from .blocks import SCALAR_ACTIVATIONS as GATES
from .blocks import ScalarActivation as _ScalarActivation
from .blocks import hidden_irreps as _hidden_irreps
from .blocks import tp_out_irreps_with_instructions


# ===========================================================================
# Clebsch-Gordan symmetric coupling basis (the ``U`` tensors)
# ===========================================================================
def _wigner_nj(irrepss, normalization: str = "component", filter_ir_mid=None, dtype=None):
    """Recursively couple ``len(irrepss)`` copies of irreps into each output irrep.

    Builds the generalized (nested) Clebsch-Gordan coupling of a list of input
    :class:`e3nn.o3.Irreps` by repeatedly contracting Wigner-3j symbols, one
    input at a time. The result enumerates every coupling path that yields each
    reachable output irrep.

    Parameters
    ----------
    irrepss : list of e3nn.o3.Irreps
        The list of input irreps (one entry per body factor) to couple together.
    normalization : {"component", "norm"}, optional
        Clebsch-Gordan normalization convention. ``"component"`` scales each
        3j block by ``ir_out.dim ** 0.5``; ``"norm"`` scales by
        ``ir_left.dim ** 0.5 * ir.dim ** 0.5``. Defaults to ``"component"``.
    filter_ir_mid : list of e3nn.o3.Irrep or None, optional
        If given, restricts the intermediate/output irreps to this set (used to
        keep the coupling tractable at high correlation order). ``None`` keeps
        every reachable irrep.
    dtype : torch.dtype or None, optional
        Dtype for the coupling tensors. ``None`` uses the tensor default.

    Returns
    -------
    list of tuple
        A list sorted by output irrep, each entry ``(ir_out, path, C)`` where
        ``ir_out`` is the coupled output :class:`e3nn.o3.Irrep`, ``path`` is a
        ``(depth, start, stop)`` slice descriptor, and ``C`` is the coupling
        tensor of shape ``(ir_out.dim, in_0.dim, ..., in_last.dim)``.
    """
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, (0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left, normalization=normalization, filter_ir_mid=filter_ir_mid, dtype=dtype
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue
                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5
                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(ir_out.dim, *(irr.dim for irr in irrepss_left), ir.dim)
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim, *(irr.dim for irr in irrepss_left), irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [(ir_out, (len(irrepss_left), sl.start, sl.stop), E)]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(irreps_in, irreps_out, correlation: int, normalization: str = "component",
                  filter_ir_mid=None, dtype=None):
    """Symmetric coupling basis of ``correlation`` copies of ``irreps_in`` -> ``irreps_out``.

    Assembles the generalized Clebsch-Gordan ``U`` basis for the MACE symmetric
    contraction: it couples ``correlation`` copies of ``irreps_in`` (via
    :func:`_wigner_nj`) and stacks the paths reaching each target irrep along a
    trailing ``n_paths`` axis. The result is bit-identical to ``mace-torch``.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        The input irreps; ``correlation`` copies are coupled together.
    irreps_out : e3nn.o3.Irreps
        The target output irreps to keep coupling paths for.
    correlation : int
        Correlation order (number of copies of ``irreps_in`` to couple), i.e.
        the body order minus one. When ``correlation == 4`` the intermediate
        irreps are filtered (upstream restriction) for tractability.
    normalization : {"component", "norm"}, optional
        Clebsch-Gordan normalization convention passed through to
        :func:`_wigner_nj`. Defaults to ``"component"``.
    filter_ir_mid : list of e3nn.o3.Irrep or None, optional
        Intermediate-irrep filter passed to :func:`_wigner_nj`; overridden
        internally when ``correlation == 4``.
    dtype : torch.dtype or None, optional
        Dtype for the coupling tensors. ``None`` uses the tensor default.

    Returns
    -------
    list
        ``[..., U]`` where the final element ``U`` is the coupling tensor of
        shape ``([out.dim,] in.dim, ..., in.dim, n_paths)`` (leading output axis
        squeezed away for scalar outputs). When no coupling path exists, ``U``
        is an all-zeros tensor of the appropriate shape.
    """
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation
    if correlation == 4:  # upstream restricts the 4-body intermediates for tractability
        filter_ir_mid = [(i, 1 if i % 2 == 0 else -1) for i in range(12)]

    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])
    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    try:
        out += [last_ir, stack]  # noqa: F821 - last_ir unbound => no coupling (fallback)
    except (NameError, UnboundLocalError):
        first_dim = irreps_out.dim
        size = ([first_dim] if first_dim != 1 else []) + [o3.Irreps(irreps_in).dim] * correlation + [1]
        out = [str(irreps_out)[:-2], torch.zeros(size, dtype=dtype)]
    return out


# ===========================================================================
# Symmetric contraction (MACE Eq. 10-11): the learned product basis
# ===========================================================================
_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


class _Contraction(nn.Module):
    """Contract ``correlation`` symmetric powers of the input into one ``irrep_out``.

    Implements the per-output-irrep piece of the MACE symmetric contraction
    (Batatia et al. 2022, Eq. 10-11). For each correlation order ``nu`` from 1
    to ``correlation`` it registers the generalized Clebsch-Gordan basis
    ``U_matrix_{nu}`` (from :func:`U_matrix_real`) as a buffer and a
    per-element learnable weight over the coupling paths. The ``forward``
    contracts these via a Horner-style nested :func:`torch.einsum`.

    The einsum equations are precomputed in ``__init__`` and the ``U`` buffers
    are accessed statically (empty non-persistent placeholders fill the unused
    orders up to 4) so ``forward`` is ``torch.jit.script``-compatible for
    LAMMPS/TorchScript deployment. This caps ``correlation`` at 4, matching the
    practical upstream MACE range (its intermediate-irrep filter also special-
    cases ``correlation == 4``).

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Input node-feature irreps; the ``0e`` multiplicity defines
        ``num_features`` (the channel count carried through the contraction).
    irrep_out : e3nn.o3.Irreps
        The single output irrep this contraction produces.
    correlation : int
        Maximum correlation order (body order minus one) to sum over. Must be
        in ``1..4``.
    num_elements : int
        Number of chemical elements; the learnable weights are indexed
        per-element (element-dependent product basis).

    Attributes
    ----------
    num_features : int
        Number of ``0e`` channels in ``irreps_in``.
    coupling_irreps : e3nn.o3.Irreps
        The irrep list (multiplicities stripped) used to build the ``U`` basis.
    correlation : int
        The maximum correlation order.
    lmax_out : int
        The maximum ``l`` of ``irrep_out``.
    weights : torch.nn.ParameterList
        One ``(num_elements, n_paths, num_features)`` weight per correlation
        order.

    Raises
    ------
    NotImplementedError
        If ``correlation`` is outside ``1..4``.
    """

    correlation: int
    eq_main: str
    eqs_weighting: List[str]
    eqs_contract: List[str]

    def __init__(self, irreps_in: o3.Irreps, irrep_out: o3.Irreps, correlation: int,
                 num_elements: int):
        super().__init__()
        if not 1 <= correlation <= 4:
            raise NotImplementedError(
                f"correlation={correlation} is not supported; the scriptable "
                "symmetric contraction covers 1..4 (the practical MACE range)"
            )
        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([ir.ir for ir in irreps_in])
        self.correlation = correlation
        self.lmax_out = o3.Irreps(irrep_out).lmax
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U = U_matrix_real(self.coupling_irreps, irrep_out, nu, dtype=dtype)[-1]
            self.register_buffer(f"U_matrix_{nu}", U)
        # empty placeholders keep the static buffer references in `forward`
        # compilable; non-persistent, so the state_dict layout is unchanged
        for nu in range(correlation + 1, 5):
            self.register_buffer(f"U_matrix_{nu}", torch.zeros(0, dtype=dtype),
                                 persistent=False)
        self.weights = nn.ParameterList([])
        for nu in range(1, correlation + 1):
            n_paths = self._U(nu).size(-1)
            self.weights.append(
                nn.Parameter(torch.randn(num_elements, n_paths, self.num_features) / n_paths)
            )
        # precompute the einsum equations (TorchScript cannot build them)
        L = min(self.lmax_out, 1)
        g = "".join(_ALPHABET[: correlation + L - 1])
        self.eq_main = f"{g}ik,ekc,bci,be->bc{g}"
        self.eqs_weighting = []
        self.eqs_contract = []
        for nu in range(1, correlation):
            gw = "".join(_ALPHABET[: nu + L])
            gf = "".join(_ALPHABET[: nu - 1 + L])
            self.eqs_weighting.append(f"{gw}k,ekc,be->bc{gw}")
            self.eqs_contract.append(f"bc{gf}i,bci->bc{gf}")

    def _U(self, nu: int) -> Tensor:
        """Return the registered coupling basis buffer for correlation order ``nu``.

        Parameters
        ----------
        nu : int
            Correlation order (1-based).

        Returns
        -------
        torch.Tensor
            The ``U_matrix_{nu}`` buffer.
        """
        return getattr(self, f"U_matrix_{nu}")

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Evaluate the symmetric contraction for one output irrep.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(B, num_features, coupling_dim)`` where
            ``coupling_dim`` is the dimension of the coupling irreps.
        y : torch.Tensor
            Per-node one-hot element attributes of shape ``(B, num_elements)``,
            selecting the element-dependent weights.

        Returns
        -------
        torch.Tensor
            Contracted features of shape ``(B, num_features * irrep_out.dim)``.
        """
        # x: (B, num_features, coupling_dim); y: (B, num_elements)
        ws: List[Tensor] = []
        for w in self.weights:
            ws.append(w)
        us = [self.U_matrix_1, self.U_matrix_2, self.U_matrix_3, self.U_matrix_4]
        corr = self.correlation
        out = torch.einsum(self.eq_main, us[corr - 1], ws[corr - 1], x, y)
        for nu in range(corr - 1, 0, -1):
            c = torch.einsum(self.eqs_weighting[nu - 1], us[nu - 1], ws[nu - 1], y)
            c = c + out
            out = torch.einsum(self.eqs_contract[nu - 1], c, x)
        return out.reshape(out.shape[0], -1)


class SymmetricContraction(nn.Module):
    """Per-element symmetric contraction over all output irreps (the MACE product basis).

    The learned, higher-body-order product basis of MACE (Batatia et al. 2022,
    Eq. 10-11). It holds one :class:`_Contraction` per output irrep and
    concatenates their results, reproducing ``mace-torch`` to ~1e-16 given the
    same weights.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Input node-feature irreps fed to every per-irrep contraction.
    irreps_out : e3nn.o3.Irreps
        Target output irreps; one :class:`_Contraction` is created per entry.
    correlation : int
        Maximum correlation order (body order minus one).
    num_elements : int
        Number of chemical elements (weights are element-dependent).

    Attributes
    ----------
    contractions : torch.nn.ModuleList
        One :class:`_Contraction` per output irrep.
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, correlation: int,
                 num_elements: int):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.contractions = nn.ModuleList([
            _Contraction(self.irreps_in, o3.Irreps(str(mul_ir.ir)), correlation, num_elements)
            for mul_ir in self.irreps_out
        ])

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Run every per-irrep contraction and concatenate the results.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(B, num_features, coupling_dim)``.
        y : torch.Tensor
            Per-node one-hot element attributes of shape ``(B, num_elements)``.

        Returns
        -------
        torch.Tensor
            The concatenated contracted features spanning all output irreps.
        """
        outs: List[Tensor] = []
        for c in self.contractions:
            outs.append(c(x, y))
        return torch.cat(outs, dim=-1)


# ===========================================================================
# Irreps helpers + equivariant blocks
# (the shared uvu path helper `tp_out_irreps_with_instructions` lives in
#  .blocks; MACE uses its default sorted-instruction convention)
# ===========================================================================
class _ReshapeIrreps(nn.Module):
    """Flat ``(N, irreps.dim)`` -> ``(N, mul, sum_ir_dim)`` (uniform mul assumed).

    Reshapes a flat irreps tensor into a channel-first layout, separating the
    multiplicity axis from the per-irrep dimension. A uniform multiplicity
    across irreps is assumed.

    Parameters
    ----------
    irreps : e3nn.o3.Irreps
        The irreps describing the flat input layout.

    Attributes
    ----------
    dims : list of int
        The dimension ``ir.dim`` of each irrep.
    muls : list of int
        The multiplicity ``mul`` of each irrep.
    """

    dims: List[int]
    muls: List[int]

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = [ir.dim for _, ir in self.irreps]
        self.muls = [mul for mul, _ in self.irreps]

    def forward(self, tensor: Tensor) -> Tensor:
        """Reshape flat irreps features to a channel-first layout.

        Parameters
        ----------
        tensor : torch.Tensor
            Flat features of shape ``(N, irreps.dim)``.

        Returns
        -------
        torch.Tensor
            Reshaped features of shape ``(N, mul, sum_ir_dim)``.
        """
        ix = 0
        out: List[Tensor] = []
        batch = tensor.shape[0]
        for mul, d in zip(self.muls, self.dims):
            out.append(tensor[:, ix : ix + mul * d].reshape(batch, mul, d))
            ix += mul * d
        return torch.cat(out, dim=-1)


class _EquivariantProductBasis(nn.Module):
    """Raise body order via the symmetric contraction, with optional residual (sc).

    Wraps a :class:`SymmetricContraction` followed by an equivariant
    :class:`e3nn.o3.Linear`, optionally adding the interaction block's
    self-connection ``sc`` as a residual. This is the MACE product-basis stage
    that increases body order.

    Parameters
    ----------
    node_feats_irreps : e3nn.o3.Irreps
        Irreps of the incoming node features fed to the contraction.
    target_irreps : e3nn.o3.Irreps
        Output irreps of the contraction and the following linear map.
    correlation : int
        Maximum correlation order (body order minus one).
    num_elements : int
        Number of chemical elements (element-dependent weights).
    use_sc : bool, optional
        If ``True`` (default), add the self-connection residual ``sc`` when it
        is provided.
    """

    def __init__(self, node_feats_irreps: o3.Irreps, target_irreps: o3.Irreps,
                 correlation: int, num_elements: int, use_sc: bool = True):
        super().__init__()
        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            node_feats_irreps, target_irreps, correlation, num_elements
        )
        self.linear = o3.Linear(target_irreps, target_irreps,
                                internal_weights=True, shared_weights=True)

    def forward(self, node_feats: Tensor, sc: Optional[Tensor], node_attrs: Tensor) -> Tensor:
        """Apply the symmetric contraction, linear map, and optional residual.

        Parameters
        ----------
        node_feats : torch.Tensor
            Reshaped node features from the interaction block.
        sc : torch.Tensor or None
            The self-connection tensor to add as a residual, or ``None``.
        node_attrs : torch.Tensor
            Per-node one-hot element attributes selecting the element-dependent
            contraction weights.

        Returns
        -------
        torch.Tensor
            The higher-body-order node features (with residual added when
            ``use_sc`` is set and ``sc`` is not ``None``).
        """
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class _InteractionBase(nn.Module):
    """Shared machinery for the MACE message-passing interaction blocks.

    Sets up the common convolution path used by both the residual and
    non-residual variants: an equivariant ``linear_up`` on the input features, a
    radial-MLP-weighted ``uvu`` :class:`e3nn.o3.TensorProduct` between node
    features and edge spherical-harmonic attributes, a ``linear`` back to the
    target irreps, and a :class:`_ReshapeIrreps`. Subclasses implement
    ``_setup`` (the skip/self connection) and ``forward``.

    Parameters
    ----------
    node_attrs_irreps : e3nn.o3.Irreps
        Irreps of the per-node one-hot element attributes.
    node_feats_irreps : e3nn.o3.Irreps
        Irreps of the incoming node features.
    edge_attrs_irreps : e3nn.o3.Irreps
        Irreps of the edge spherical-harmonic attributes.
    edge_feats_irreps : e3nn.o3.Irreps
        Irreps of the (scalar) radial edge features feeding the radial MLP.
    target_irreps : e3nn.o3.Irreps
        Target irreps of the message (the block output irreps).
    hidden_irreps : e3nn.o3.Irreps
        Hidden irreps used by the residual variant's self-connection.
    avg_num_neighbors : float
        Average neighbour count used to normalise the aggregated message.
    radial_MLP : list of int
        Hidden layer widths of the radial MLP that produces the tensor-product
        weights.
    """

    avg_num_neighbors: float

    def __init__(self, node_attrs_irreps, node_feats_irreps, edge_attrs_irreps,
                 edge_feats_irreps, target_irreps, hidden_irreps, avg_num_neighbors,
                 radial_MLP):
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.radial_MLP = list(radial_MLP)
        self._common_setup()
        self._setup()

    def _common_setup(self):
        """Build the convolution path shared by all interaction variants.

        Constructs ``linear_up``, the ``uvu`` convolution tensor product
        ``conv_tp`` with its radial-MLP weight generator ``conv_tp_weights``,
        the output ``linear`` map, and the ``reshape`` module. Sets
        ``irreps_out`` to ``target_irreps``.
        """
        self.linear_up = o3.Linear(self.node_feats_irreps, self.node_feats_irreps,
                                   internal_weights=True, shared_weights=True)
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps, self.edge_attrs_irreps, irreps_mid,
            instructions=instructions, shared_weights=False, internal_weights=False,
        )
        self.conv_tp_weights = e3nn_nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps] + self.radial_MLP + [self.conv_tp.weight_numel],
            F.silu,
        )
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(irreps_mid, self.irreps_out,
                                internal_weights=True, shared_weights=True)
        self.reshape = _ReshapeIrreps(self.irreps_out)


class RealAgnosticInteractionBlock(_InteractionBase):
    """Non-residual interaction: the skip connection is applied to the message.

    A :class:`_InteractionBase` whose ``skip_tp`` mixes the aggregated,
    normalised message with the node element attributes (so the self-connection
    acts on the *message* rather than the input features). Returns ``None`` in
    place of a separate self-connection.
    """

    def _setup(self):
        """Build the skip tensor product acting on the output message."""
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

    def forward(self, node_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor,
                edge_feats: Tensor, edge_index: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute one message-passing update (non-residual variant).

        Parameters
        ----------
        node_attrs : torch.Tensor
            Per-node one-hot element attributes.
        node_feats : torch.Tensor
            Incoming node features, shape ``(num_nodes, node_feats_irreps.dim)``.
        edge_attrs : torch.Tensor
            Edge spherical-harmonic attributes.
        edge_feats : torch.Tensor
            Scalar radial edge features feeding the radial MLP.
        edge_index : torch.Tensor
            Edge index of shape ``(2, num_edges)`` (``[senders, receivers]``).

        Returns
        -------
        tuple of (torch.Tensor, None)
            The reshaped message features and ``None`` (no separate
            self-connection for the non-residual block).
        """
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(mji, edge_index[1], num_nodes)
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return self.reshape(message), None


class RealAgnosticResidualInteractionBlock(_InteractionBase):
    """Residual interaction: self-connection computed from the *input* features.

    A :class:`_InteractionBase` whose ``skip_tp`` builds a self-connection from
    the *input* node features and element attributes. That self-connection is
    returned alongside the message so the downstream product basis can add it as
    a residual.
    """

    def _setup(self):
        """Build the skip tensor product acting on the input node features."""
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )

    def forward(self, node_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor,
                edge_feats: Tensor, edge_index: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute one message-passing update (residual variant).

        Parameters
        ----------
        node_attrs : torch.Tensor
            Per-node one-hot element attributes.
        node_feats : torch.Tensor
            Incoming node features, shape ``(num_nodes, node_feats_irreps.dim)``.
        edge_attrs : torch.Tensor
            Edge spherical-harmonic attributes.
        edge_feats : torch.Tensor
            Scalar radial edge features feeding the radial MLP.
        edge_index : torch.Tensor
            Edge index of shape ``(2, num_edges)`` (``[senders, receivers]``).

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The reshaped message features and the self-connection ``sc``
            computed from the input features (to be added as a residual).
        """
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)
        message = scatter_sum(mji, edge_index[1], num_nodes)
        message = self.linear(message) / self.avg_num_neighbors
        return self.reshape(message), sc


class _LinearReadout(nn.Module):
    """Linear equivariant readout mapping node features to a scalar output.

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Irreps of the input node features.
    irrep_out : e3nn.o3.Irreps, optional
        Output irreps, by default a single scalar ``"1x0e"`` (per-node energy).
    """

    def __init__(self, irreps_in: o3.Irreps, irrep_out: o3.Irreps = o3.Irreps("1x0e")):
        super().__init__()
        self.linear = o3.Linear(irreps_in, irrep_out)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear readout.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            The linear projection onto ``irrep_out``.
        """
        return self.linear(x)


class _NonLinearReadout(nn.Module):
    """Gated non-linear equivariant readout (linear -> gate -> linear).

    Parameters
    ----------
    irreps_in : e3nn.o3.Irreps
        Irreps of the input node features.
    MLP_irreps : e3nn.o3.Irreps
        Hidden irreps of the readout MLP.
    gate : callable
        Scalar activation applied (second-moment normalized, exactly as
        :class:`e3nn.nn.Activation` would) between the two linear layers.
    irrep_out : e3nn.o3.Irreps, optional
        Output irreps, by default a single scalar ``"1x0e"`` (per-node energy).
    """

    def __init__(self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate,
                 irrep_out: o3.Irreps = o3.Irreps("1x0e")):
        super().__init__()
        self.linear_1 = o3.Linear(irreps_in, MLP_irreps)
        self.non_linearity = _ScalarActivation(MLP_irreps, gate)
        self.linear_2 = o3.Linear(MLP_irreps, irrep_out)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated non-linear readout.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            The projection onto ``irrep_out`` after the gated non-linearity.
        """
        return self.linear_2(self.non_linearity(self.linear_1(x)))


class _ZBLPairRepulsion(nn.Module):
    """Ziegler-Biersack-Littmark short-range repulsion with a polynomial cutoff.

    A screened-Coulomb pairwise repulsion between atoms, summed per receiving
    node and multiplied by a smooth polynomial envelope that goes to zero at the
    covalent-radii sum. Requires ``ase`` for element data.

    Parameters
    ----------
    p : int, optional
        Degree of the polynomial cutoff envelope, by default 6.

    Attributes
    ----------
    c : torch.Tensor
        The four ZBL universal screening coefficients.
    p : torch.Tensor
        The polynomial cutoff degree (integer buffer).
    covalent_radii : torch.Tensor
        Per-element covalent radii (from :mod:`ase.data`) used for the cutoff.

    Raises
    ------
    ImportError
        If ``ase`` is not installed.
    """

    def __init__(self, p: int = 6):
        super().__init__()
        try:
            import ase.data
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ImportError('pair_repulsion needs ase: pip install "xnns[ase]"') from e
        self.register_buffer("c", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype())
        )

    @staticmethod
    def _envelope(x: Tensor, r_max: Tensor, p: Tensor) -> Tensor:
        """Smooth polynomial cutoff envelope, zero for ``x >= r_max``.

        Parameters
        ----------
        x : torch.Tensor
            Pairwise distances.
        r_max : torch.Tensor
            Per-pair cutoff radius (the covalent-radii sum).
        p : torch.Tensor
            Polynomial degree.

        Returns
        -------
        torch.Tensor
            The envelope value, masked to zero where ``x >= r_max``.
        """
        t = x / r_max
        env = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * t**p
               + p * (p + 2.0) * t**(p + 1) - (p * (p + 1.0) / 2) * t**(p + 2))
        return env * (x < r_max)

    def forward(self, lengths: Tensor, atomic_numbers: Tensor, edge_index: Tensor,
                num_nodes: int) -> Tensor:
        """Compute the per-node ZBL pair-repulsion energy.

        Parameters
        ----------
        lengths : torch.Tensor
            Edge lengths of shape ``(E, 1)``.
        atomic_numbers : torch.Tensor
            Per-node atomic numbers ``Z``.
        edge_index : torch.Tensor
            Edge index of shape ``(2, E)`` (``[senders, receivers]``).
        num_nodes : int
            Number of nodes to scatter the pair energies onto.

        Returns
        -------
        torch.Tensor
            Per-node repulsion energy of shape ``(num_nodes,)``.
        """
        x = lengths  # (E, 1)
        Z_u = atomic_numbers[edge_index[0]].unsqueeze(-1).to(torch.int64)
        Z_v = atomic_numbers[edge_index[1]].unsqueeze(-1).to(torch.int64)
        a = 0.4543 * 0.529 / (torch.pow(Z_u, 0.300) + torch.pow(Z_v, 0.300))
        r_a = x / a
        phi = (self.c[0] * torch.exp(-3.2 * r_a) + self.c[1] * torch.exp(-0.9423 * r_a)
               + self.c[2] * torch.exp(-0.4028 * r_a) + self.c[3] * torch.exp(-0.2016 * r_a))
        v = (14.3996 * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        v = 0.5 * v * self._envelope(x, r_max, self.p)
        return scatter_sum(v, edge_index[1], num_nodes).squeeze(-1)


INTERACTIONS = {
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
}
# GATES is the shared SCALAR_ACTIVATIONS registry (imported above): the gate
# options by their upstream spellings (silu/tanh/abs/ssp/None).


# ===========================================================================
# The MACE model
# ===========================================================================
@register_model("mace")
class MACE(EquivariantGNN):
    """Faithful MACE with a flexible number of interaction layers (T = 0..N).

    Subclasses :class:`~xnns.gnn.models.base.EquivariantGNN`, inheriting species
    bookkeeping, the per-element reference energy ``atom_ref``, and the
    :class:`~xnns.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge
    featurizer, and adds the MACE-specific interaction blocks and the learned
    symmetric-contraction product basis. The number of message-passing layers
    ``T = num_interactions`` is fully flexible (``T = 0`` gives a pure
    ``atom_ref``/pair-repulsion baseline).

    Parameters
    ----------
    species : list of int
        Atomic numbers of the elements the model supports.
    cutoff : float, optional
        Radial cutoff distance ``r_max`` in angstrom, by default 4.0.
    max_ell : int, optional
        Maximum degree ``l`` of the spherical-harmonic edge attributes, by
        default 3.
    max_L : int, optional
        Maximum output irrep order ``L`` of the hidden node features, by
        default 0 (invariant features only).
    num_channels : int, optional
        Number of feature channels (multiplicity) in the hidden irreps, by
        default 32.
    n_rbf : int, optional
        Number of radial basis functions, by default 8.
    num_interactions : int, optional
        Number of message-passing layers ``T``, by default 2. Must be ``>= 0``.
    correlation : int or list of int, optional
        Correlation order (body order minus one) of the symmetric contraction,
        by default 3. An int is broadcast across all interactions.
    MLP_irreps : str, optional
        Hidden irreps of the final non-linear readout MLP, by default
        ``"16x0e"``.
    radial_MLP : list of int or None, optional
        Hidden layer widths of the radial MLP producing tensor-product weights.
        ``None`` defaults to ``[64, 64, 64]``.
    interaction : str, optional
        Interaction-block class name for layers after the first, by default
        ``"RealAgnosticResidualInteractionBlock"``.
    interaction_first : str, optional
        Interaction-block class name for the first layer, by default
        ``"RealAgnosticResidualInteractionBlock"``.
    gate : str or None, optional
        Name of the scalar gate activation for the non-linear readout, by
        default ``"silu"``.
    avg_num_neighbors : float, optional
        Average neighbour count used to normalise messages, by default 1.0.
    hidden_irreps : str or None, optional
        Explicit hidden irreps. ``None`` derives them from ``num_channels`` and
        ``max_L``.
    num_cutoff_basis : int, optional
        Polynomial cutoff degree ``p`` for the edge envelope (and ZBL), by
        default 5.
    radial_type : str, optional
        Radial basis type (``"bessel"`` or ``"gaussian"``), by default
        ``"bessel"``.
    distance_transform : str, optional
        Distance transform; only ``"None"`` is supported.
    pair_repulsion : bool, optional
        If ``True``, add a :class:`_ZBLPairRepulsion` short-range term, by
        default ``False``.
    atomic_energies : torch.Tensor or None, optional
        Per-element reference energies (``E0s``) used to initialise
        ``atom_ref``.

    Raises
    ------
    ValueError
        If ``num_interactions`` is negative.
    NotImplementedError
        If ``distance_transform`` is anything other than ``"None"``.
    """

    pair_repulsion: Final[bool]

    def __init__(
        self,
        species: List[int],
        cutoff: float = 4.0,
        max_ell: int = 3,
        max_L: int = 0,
        num_channels: int = 32,
        n_rbf: int = 8,
        num_interactions: int = 2,
        correlation: Union[int, List[int]] = 3,
        MLP_irreps: str = "16x0e",
        radial_MLP: Optional[List[int]] = None,
        interaction: str = "RealAgnosticResidualInteractionBlock",
        interaction_first: str = "RealAgnosticResidualInteractionBlock",
        gate: Optional[str] = "silu",
        avg_num_neighbors: float = 1.0,
        hidden_irreps: Optional[str] = None,
        num_cutoff_basis: int = 5,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        pair_repulsion: bool = False,
        atomic_energies: Optional[Tensor] = None,
    ):
        if num_interactions < 0:
            raise ValueError("num_interactions (T) must be >= 0")
        if distance_transform not in ("None", None):
            raise NotImplementedError(
                f"distance_transform={distance_transform!r} is not supported; use 'None'"
            )
        # EquivariantGNN gives species/z_to_index/node_attr/atom_ref/edge_feat/
        # irreps_sh; the featurizer uses MACE's cutoff degree + radial type.
        super().__init__(species, cutoff, l_max=max_ell, n_rbf=n_rbf,
                         p=num_cutoff_basis, radial_type=radial_type)
        if atomic_energies is not None:  # per-element reference energy (E0s)
            self.set_atomic_energies(atomic_energies)

        num_elements = len(self.species)
        hid = (o3.Irreps(hidden_irreps) if hidden_irreps is not None
               else _hidden_irreps(num_channels, max_L))
        MLP_irreps = o3.Irreps(MLP_irreps)
        radial_MLP = list(radial_MLP) if radial_MLP is not None else [64, 64, 64]
        if isinstance(correlation, int):
            correlation = [correlation] * max(num_interactions, 1)
        gate_fn = GATES[gate]
        inter_cls = INTERACTIONS[interaction]
        inter_cls_first = INTERACTIONS[interaction_first]

        num_features = hid.count(o3.Irrep(0, 1))
        node_feats_irreps = o3.Irreps([(num_features, (0, 1))])
        # invariant (l=0) channels per layer, concatenated across layers (for
        # e.g. LES latent charges); with T=0 the species embedding itself
        self._n_scalar_features = num_features
        self.node_feature_dim = num_features * max(1, num_interactions)
        sh_irreps = self.irreps_sh                                  # spherical_harmonics(max_ell)
        interaction_irreps = _hidden_irreps(num_features, max_ell)  # C copies of SH(max_ell)
        edge_feats_irreps = o3.Irreps(f"{n_rbf}x0e")

        self.node_embedding = o3.Linear(self.node_attr_irreps, node_feats_irreps)
        self.pair_repulsion = bool(pair_repulsion)
        if pair_repulsion:
            self.pair_repulsion_fn = _ZBLPairRepulsion(p=num_cutoff_basis)

        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        self.readouts = nn.ModuleList()

        if num_interactions >= 1:
            hidden_out = o3.Irreps(str(hid[0])) if num_interactions == 1 else hid
            self.interactions.append(inter_cls_first(
                self.node_attr_irreps, node_feats_irreps, sh_irreps, edge_feats_irreps,
                interaction_irreps, hidden_out, avg_num_neighbors, radial_MLP,
            ))
            self.products.append(_EquivariantProductBasis(
                interaction_irreps, hidden_out, correlation[0], num_elements,
                use_sc="Residual" in interaction_first,
            ))
            self.readouts.append(_LinearReadout(hidden_out))

            for i in range(num_interactions - 1):
                last = i == num_interactions - 2
                hidden_out = o3.Irreps(str(hid[0])) if last else hid
                self.interactions.append(inter_cls(
                    self.node_attr_irreps, hid, sh_irreps, edge_feats_irreps,
                    interaction_irreps, hidden_out, avg_num_neighbors, radial_MLP,
                ))
                self.products.append(_EquivariantProductBasis(
                    interaction_irreps, hidden_out, correlation[i + 1], num_elements, use_sc=True,
                ))
                self.readouts.append(
                    _NonLinearReadout(hidden_out, MLP_irreps, gate_fn) if last
                    else _LinearReadout(hid)
                )

    @torch.jit.export
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: tensors in, features + energies out.

        The single implementation reused by :meth:`node_energy` (the deploy
        entry point) and :meth:`forward`, so it must avoid the
        :class:`~xnns.common.data.AtomicGraph` dataclass and any Python-only
        constructs. Starts from the per-element reference energy, optionally
        adds the ZBL pair-repulsion term, then runs ``T`` rounds of
        interaction + product basis + readout, accumulating each readout into
        the node energy and collecting the invariant (``l = 0``) channels of
        every layer's node features (what
        :class:`~xnns.common.models.les.LatentEwald` consumes).

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : torch.Tensor
            Edge index of shape ``(2, E)``; row 0 is the source (neighbour) and
            row 1 the destination (centre) node of each edge.
        edge_vec : torch.Tensor
            Edge displacement vectors, shape ``(E, 3)`` (already accounting for
            any periodic cell shifts).

        Returns
        -------
        tuple of torch.Tensor
            The concatenated invariant node features
            ``(N, node_feature_dim)`` and the per-atom energy ``(N,)``.
        """
        node_attrs = self.node_attr(atomic_numbers)
        node_energy = self.atom_ref(atomic_numbers).squeeze(-1)
        num_nodes = atomic_numbers.shape[0]
        lengths, edge_sh, edge_radial = self.edge_feat.embed(edge_vec)

        if self.pair_repulsion:
            node_energy = node_energy + self.pair_repulsion_fn(
                lengths[:, None], atomic_numbers, edge_index, num_nodes,
            )

        node_feats = self.node_embedding(node_attrs)
        feats_list: List[Tensor] = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs, node_feats, edge_sh, edge_radial, edge_index
            )
            node_feats = product(node_feats, sc, node_attrs)
            node_energy = node_energy + readout(node_feats).squeeze(-1)
            # scalar (l=0) channels come first in the e3nn irreps layout
            feats_list.append(node_feats[:, :self._n_scalar_features])
        if len(feats_list) == 0:
            feats_list.append(node_feats)
        features = torch.cat(feats_list, dim=-1)

        return features, node_energy

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
    def from_config(cls, cfg) -> "MACE":
        """Construct a :class:`MACE` from a core model config.

        Reads the MACE-specific hyper-parameters from ``cfg.extra`` (falling
        back to defaults) and the shared fields from ``cfg`` directly. Only the
        xnns canonical key names are read here; upstream MACE-CLI spellings
        (``r_max``, ``num_radial_basis``, ``atomic_numbers``, ``E0s``, ...) are
        translated to these names at config-load time by
        :mod:`xnns.common.config.translate`. Values copied from an upstream
        yaml are coerced: ``species`` accepts a ``"[1, 6, 8]"`` string,
        ``radial_MLP`` a ``"[64, 64, 64]"`` string, and ``atomic_energies``
        (MACE ``E0s``) a list aligned with ``species``, a ``{Z: E0}`` dict, or
        the string form of either.

        Parameters
        ----------
        cfg : xnns.common.config.schema.ModelConfig
            The core model config, whose ``extra`` dict carries the MACE
            architecture options.

        Returns
        -------
        MACE
            The instantiated model.
        """
        import ast

        from xnns.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=[1, 6, 8])
        atomic_energies = coerce_per_species(
            extra.get("atomic_energies"), species, "atomic_energies (MACE E0s)")

        radial_MLP = extra.get("radial_MLP")
        if isinstance(radial_MLP, str):
            radial_MLP = ast.literal_eval(radial_MLP)
        return cls(
            species=species,
            cutoff=cfg.cutoff,
            max_ell=extra.get("max_ell", 3),
            max_L=extra.get("max_L", 0),
            num_channels=cfg.n_features,
            n_rbf=cfg.n_rbf,
            num_interactions=cfg.n_interactions,
            correlation=extra.get("correlation", 3),
            MLP_irreps=extra.get("MLP_irreps", "16x0e"),
            radial_MLP=radial_MLP,
            interaction=extra.get("interaction", "RealAgnosticResidualInteractionBlock"),
            interaction_first=extra.get(
                "interaction_first", "RealAgnosticResidualInteractionBlock"
            ),
            gate=extra.get("gate", "silu"),
            avg_num_neighbors=extra.get("avg_num_neighbors", 1.0),
            hidden_irreps=extra.get("hidden_irreps"),
            num_cutoff_basis=extra.get("num_polynomial_cutoff", 5),
            radial_type=extra.get("radial_type", "bessel"),
            distance_transform=extra.get("distance_transform", "None"),
            pair_repulsion=extra.get("pair_repulsion", False),
            atomic_energies=atomic_energies,
        )
