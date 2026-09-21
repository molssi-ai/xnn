"""MACE (Batatia et al. 2022): higher body-order equivariant message passing.

A faithful, self-contained MACE built on the xnn equivariant-GNN abstractions:
it subclasses :class:`~xnn.gnn.models.base.EquivariantGNN` (species bookkeeping,
per-element reference energy ``atom_ref``, and the
:class:`~xnn.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge featurizer) and
adds the genuinely MACE-specific pieces -- the real
``RealAgnostic(Residual)InteractionBlock`` and a *learned symmetric contraction*
over Clebsch-Gordan paths (``correlation`` order).

The CG ``U`` basis (:func:`U_matrix_real`) is bit-identical to ``mace-torch`` and
the symmetric contraction reproduces it to ~1e-16 given the same weights
(see ``tests/test_gnn.py``). Only ``e3nn`` is required -- no ``mace-torch``,
``cuequivariance`` or ``opt_einsum_fx``.

The model is TorchScript-deployable: the tensor-only :meth:`MACE.node_energy`
core compiles under ``torch.jit.script`` (used by the LAMMPS/TorchScript
exporters in :mod:`xnn.common.deploy`) and reproduces the eager model to
machine precision (see ``tests/test_mace.py``).

Difference from upstream MACE: ``num_interactions`` (the number of message-passing
layers *T*) is fully flexible -- ``T = 0`` (a pure ``atom_ref``/pair-repulsion
baseline) through any ``T = N`` -- rather than being fixed to 2. All architecture
options are read from ``ModelConfig.extra`` (see :meth:`MACE.from_config`);
upstream MACE-CLI spellings (``r_max``, ``atomic_numbers``, ``E0s``, ...) are
translated to the xnn names at config-load time by the key-translation registry
in :mod:`xnn.common.config.translate`.

The CG coupling and the symmetric contraction are independent implementations
(plain ``torch.einsum``, built on e3nn's ``o3.wigner_3j``; no codegen/cueq
deps), verified bit-identical against ACEsuit/mace (MIT licence). All
conventions -- CG normalization, coupling-path ordering, parameter and buffer
names -- follow upstream so trained ``mace-torch`` weights transplant directly.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies the
# class-level attribute annotations that TorchScript needs to resolve (e.g.
# `widths: List[int]` on _ReshapeIrreps), breaking `torch.jit.script`.
import itertools
from typing import Final, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3
from e3nn import nn as e3nn_nn

from xnn.common.data import AtomicGraph
from xnn.common.models.ops import scatter_sum
from xnn.common.models.registry import register_model
from .base import EquivariantGNN
from ..featurizers import DISTANCE_TRANSFORMS
from .blocks import SCALAR_ACTIVATIONS as GATES
from .blocks import ScalarActivation as _ScalarActivation
from .blocks import hidden_irreps as _hidden_irreps
from .blocks import tp_out_irreps_with_instructions


# ===========================================================================
# Clebsch-Gordan symmetric coupling basis (the ``U`` tensors)
# ===========================================================================
def _wigner_nj(irrepss, normalization: str = "component", filter_ir_mid=None, dtype=None):
    """Generalized Clebsch-Gordan coupling of ``len(irrepss)`` irreps factors.

    Enumerates every coupling path that combines one irrep occurrence from each
    factor in ``irrepss`` into a single output irrep, together with the coupling
    tensor of that path. The coupling is built as a left fold: the paths for the
    first ``d`` factors are extended by contracting a Wigner-3j symbol with each
    irrep occurrence of factor ``d + 1``, and the path list is re-sorted by
    output irrep after every stage (ties keep enumeration order).

    Parameters
    ----------
    irrepss : list of e3nn.o3.Irreps
        The list of input irreps (one entry per body factor) to couple together.
    normalization : {"component", "norm"}, optional
        Clebsch-Gordan normalization convention. ``"component"`` scales each
        3j block by ``sqrt(dim)`` of the coupled output irrep; ``"norm"``
        scales by ``sqrt(dim)`` of both coupled inputs. Defaults to
        ``"component"``.
    filter_ir_mid : list of e3nn.o3.Irrep or None, optional
        If given, restricts the intermediate/output irreps to this set (used to
        keep the coupling tractable at high correlation order). ``None`` keeps
        every reachable irrep.
    dtype : torch.dtype or None, optional
        Dtype for the coupling tensors. ``None`` uses the tensor default.

    Returns
    -------
    list of tuple
        Entries ``(ir_out, C)``: the coupled output :class:`e3nn.o3.Irrep` and
        the coupling tensor of shape ``(ir_out.dim, in_0.dim, ..., in_last.dim)``
        (each ``in_d.dim`` the full dimension of factor ``d``, zero outside the
        occurrence the path passes through). Sorted by output irrep whenever
        more than one factor is coupled.
    """
    factors = [o3.Irreps(irreps) for irreps in irrepss]
    keep = None
    if filter_ir_mid is not None:
        keep = frozenset(o3.Irrep(ir) for ir in filter_ir_mid)

    # correlation-1 seed: each irrep occurrence of the first factor couples to
    # itself through the identity (its rows of the full-dimension identity)
    seed = factors[0]
    identity = torch.eye(seed.dim, dtype=dtype)
    paths: List[Tuple[o3.Irrep, Tensor]] = []
    row = 0
    for mul, ir in seed:
        for _ in range(mul):
            paths.append((ir, identity[row : row + ir.dim]))
            row += ir.dim

    for depth in range(1, len(factors)):
        factor = factors[depth]
        lead_shape = [f.dim for f in factors[:depth]]
        grown: List[Tuple[o3.Irrep, Tensor]] = []
        for ir_acc, w_acc in paths:
            col = 0  # running offset of the occurrence within `factor`
            for mul, ir_f in factor:
                for ir_tot in ir_acc * ir_f:
                    if keep is not None and ir_tot not in keep:
                        continue
                    w3j = o3.wigner_3j(ir_tot.l, ir_acc.l, ir_f.l, dtype=dtype)
                    if normalization == "component":
                        w3j = w3j * ir_tot.dim**0.5
                    if normalization == "norm":
                        w3j = w3j * (ir_acc.dim**0.5 * ir_f.dim**0.5)
                    coupled = torch.einsum("ap,oaf->opf", w_acc.flatten(1), w3j)
                    coupled = coupled.reshape(ir_tot.dim, *lead_shape, ir_f.dim)
                    for copy in range(mul):
                        lo = col + copy * ir_f.dim
                        embedded = torch.zeros(
                            ir_tot.dim, *lead_shape, factor.dim, dtype=dtype
                        )
                        embedded[..., lo : lo + ir_f.dim] = coupled
                        grown.append((ir_tot, embedded))
                col += mul * ir_f.dim
        grown.sort(key=lambda entry: entry[0])
        paths = grown
    return paths


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
        Alternating ``[ir, U, ir, U, ...]`` pairs, one per contiguous run of
        coupling paths reaching an irrep of ``irreps_out``; each ``U`` has
        shape ``([out.dim,] in.dim, ..., in.dim, n_paths)`` (leading output
        axis squeezed away for scalar outputs, so callers typically take the
        final element). When no coupling path exists, a single
        ``[label, zeros]`` pair with one all-zero path is returned instead.
    """
    irreps_out = o3.Irreps(irreps_out)
    if correlation == 4:
        # tractability restriction inherited from upstream: 4-body
        # intermediates are limited to the natural-parity irreps p = (-1)^l
        # (the spherical-harmonic series) up to l = 11
        filter_ir_mid = [o3.Irrep(l, (-1) ** l) for l in range(12)]
    couplings = _wigner_nj(
        [o3.Irreps(irreps_in)] * correlation, normalization, filter_ir_mid, dtype
    )

    # group the (sorted) paths by output irrep; stack every run reaching a
    # requested irrep along a trailing path axis
    result = []
    for ir, run in itertools.groupby(couplings, key=lambda entry: entry[0]):
        if ir in irreps_out:
            result.append(ir)
            result.append(torch.stack([w.squeeze() for _, w in run], dim=-1))
    if result:
        return result

    # nothing couples into irreps_out: emit one all-zero path so downstream
    # contraction shapes stay well-defined
    shape = [o3.Irreps(irreps_in).dim] * correlation + [1]
    if irreps_out.dim != 1:
        shape.insert(0, irreps_out.dim)
    # upstream labels this placeholder with the target irreps string minus its
    # final two characters; the odd value is kept for bit-compatibility
    text = format(irreps_out)
    return [text[: len(text) - 2], torch.zeros(shape, dtype=dtype)]


# ===========================================================================
# Symmetric contraction (MACE Eq. 10-11): the learned product basis
# ===========================================================================
# free einsum labels for the correlation axes of the U tensors; anything is
# fine as long as none collides with the reserved labels b (batch), c (channel),
# e (element), i (coupling dim), k (path) used in the contraction equations
_EINSUM_AXES = "mnopqrstuvwx"


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
        # precompute the einsum equations (TorchScript cannot build them);
        # non-scalar outputs carry one extra spatial (m) axis on the U tensors,
        # scalar outputs have it squeezed away
        m_axis = 1 if self.lmax_out > 0 else 0
        lead = _EINSUM_AXES[: correlation + m_axis - 1]
        self.eq_main = f"{lead}ik,ekc,bci,be->bc{lead}"
        self.eqs_weighting = []
        self.eqs_contract = []
        for order in range(1, correlation):
            axes_w = _EINSUM_AXES[: order + m_axis]
            axes_f = _EINSUM_AXES[: order + m_axis - 1]
            self.eqs_weighting.append(f"{axes_w}k,ekc,be->bc{axes_w}")
            self.eqs_contract.append(f"bc{axes_f}i,bci->bc{axes_f}")

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
        path_weights: List[Tensor] = []
        for w in self.weights:
            path_weights.append(w)
        bases = [self.U_matrix_1, self.U_matrix_2, self.U_matrix_3, self.U_matrix_4]
        corr = self.correlation
        # Horner evaluation: start at the highest order and repeatedly fold in
        # the next-lower weighted basis before contracting one power of x away
        acc = torch.einsum(self.eq_main, bases[corr - 1], path_weights[corr - 1], x, y)
        for order in range(corr - 1, 0, -1):
            weighted = torch.einsum(
                self.eqs_weighting[order - 1], bases[order - 1], path_weights[order - 1], y
            )
            acc = torch.einsum(self.eqs_contract[order - 1], weighted + acc, x)
        return acc.reshape(acc.shape[0], -1)


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
    widths : list of int
        The flat width ``mul * ir.dim`` of each irrep entry (the split sizes).
    shapes : list of list of int
        The target ``[mul, ir.dim]`` shape of each entry.
    """

    widths: List[int]
    shapes: List[List[int]]

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.widths = [mul * ir.dim for mul, ir in self.irreps]
        self.shapes = [[mul, ir.dim] for mul, ir in self.irreps]

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
        pieces = torch.split(tensor, self.widths, dim=-1)
        unflattened: List[Tensor] = []
        for pos, piece in enumerate(pieces):
            unflattened.append(piece.unflatten(-1, self.shapes[pos]))
        return torch.cat(unflattened, dim=-1)


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
        n_atoms = node_feats.shape[0]
        feats = self.linear_up(node_feats)
        radial_w = self.conv_tp_weights(edge_feats)  # per-edge TP weights
        edge_msg = self.conv_tp(feats[edge_index[0]], edge_attrs, radial_w)
        pooled = scatter_sum(edge_msg, edge_index[1], n_atoms)
        pooled = self.linear(pooled) / self.avg_num_neighbors
        # the self-connection acts on the aggregated message here, not the input
        pooled = self.skip_tp(pooled, node_attrs)
        return self.reshape(pooled), None


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
        n_atoms = node_feats.shape[0]
        # self-connection from the raw input features, returned for the
        # product basis to add downstream
        residual = self.skip_tp(node_feats, node_attrs)
        feats = self.linear_up(node_feats)
        radial_w = self.conv_tp_weights(edge_feats)  # per-edge TP weights
        edge_msg = self.conv_tp(feats[edge_index[0]], edge_attrs, radial_w)
        pooled = scatter_sum(edge_msg, edge_index[1], n_atoms)
        pooled = self.linear(pooled) / self.avg_num_neighbors
        return self.reshape(pooled), residual


class RealAgnosticDensityInteractionBlock(RealAgnosticInteractionBlock):
    """Non-residual interaction with learned density normalization.

    Identical to :class:`RealAgnosticInteractionBlock` except that the
    aggregated message is divided by ``1 + rho_i`` -- a learned, per-node
    neighbor density ``rho_i = sum_j tanh(d(e_ij)^2)`` built from the radial
    edge features -- instead of the global ``avg_num_neighbors`` constant.
    This is the interaction of the MACE-MP "density" foundation generation
    (0b2 / 0b3 / MPA-0 / OMAT-0 / MATPES).
    """

    def _setup(self):
        """Add the per-edge density network to the non-residual setup."""
        super()._setup()
        self.density_fn = e3nn_nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps, 1], F.silu)

    def forward(self, node_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor,
                edge_feats: Tensor, edge_index: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute one density-normalized update (non-residual variant).

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
        n_atoms = node_feats.shape[0]
        feats = self.linear_up(node_feats)
        radial_w = self.conv_tp_weights(edge_feats)  # per-edge TP weights
        density = scatter_sum(torch.tanh(self.density_fn(edge_feats) ** 2),
                              edge_index[1], n_atoms)
        edge_msg = self.conv_tp(feats[edge_index[0]], edge_attrs, radial_w)
        pooled = scatter_sum(edge_msg, edge_index[1], n_atoms)
        pooled = self.linear(pooled) / (density + 1.0)
        # the self-connection acts on the aggregated message here, not the input
        pooled = self.skip_tp(pooled, node_attrs)
        return self.reshape(pooled), None


class RealAgnosticDensityResidualInteractionBlock(RealAgnosticResidualInteractionBlock):
    """Residual interaction with learned density normalization.

    Identical to :class:`RealAgnosticResidualInteractionBlock` except for the
    ``1 + rho_i`` message normalization of
    :class:`RealAgnosticDensityInteractionBlock`.
    """

    def _setup(self):
        """Add the per-edge density network to the residual setup."""
        super()._setup()
        self.density_fn = e3nn_nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps, 1], F.silu)

    def forward(self, node_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor,
                edge_feats: Tensor, edge_index: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute one density-normalized update (residual variant).

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
        n_atoms = node_feats.shape[0]
        residual = self.skip_tp(node_feats, node_attrs)
        feats = self.linear_up(node_feats)
        radial_w = self.conv_tp_weights(edge_feats)  # per-edge TP weights
        density = scatter_sum(torch.tanh(self.density_fn(edge_feats) ** 2),
                              edge_index[1], n_atoms)
        edge_msg = self.conv_tp(feats[edge_index[0]], edge_attrs, radial_w)
        pooled = scatter_sum(edge_msg, edge_index[1], n_atoms)
        pooled = self.linear(pooled) / (density + 1.0)
        return self.reshape(pooled), residual


class _ScaleShift(nn.Module):
    """Affine rescaling of the per-atom interaction energy.

    Implements the upstream ``ScaleShiftMACE`` convention: the *interaction*
    energy (readouts plus pair repulsion, everything except the per-element
    reference ``atom_ref``) is mapped through ``scale * x + shift``, where
    ``scale`` is typically the force RMS of the training set and ``shift``
    the mean interaction energy per atom. Both are registered buffers (named
    as upstream, so checkpoint values carry over); the defaults ``(1, 0)``
    are the identity, which recovers the plain MACE energy expression.

    Parameters
    ----------
    scale, shift : float, optional
        The affine constants, by default 1.0 and 0.0.
    """

    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=dtype))
        self.register_buffer("shift", torch.tensor(float(shift), dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``scale * x + shift`` elementwise.

        Parameters
        ----------
        x : torch.Tensor
            Per-atom interaction energies.

        Returns
        -------
        torch.Tensor
            The rescaled energies.
        """
        return self.scale * x + self.shift


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
            raise ImportError('pair_repulsion needs ase: pip install "xnn[ase]"') from e
        self.register_buffer("c", torch.tensor([0.1818, 0.5099, 0.2802, 0.02817]))
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii", torch.tensor(ase.data.covalent_radii, dtype=torch.get_default_dtype())
        )
        # the universal screening-length constants, as buffers (upstream
        # layout) so trained/quantized checkpoint values carry over
        self.register_buffer("a_exp", torch.tensor(0.300))
        self.register_buffer("a_prefactor", torch.tensor(0.4543))

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
        s = x / r_max
        a0 = (p + 1.0) * (p + 2.0) / 2.0
        a1 = p * (p + 2.0)
        a2 = p * (p + 1.0) / 2.0
        smooth = 1.0 - a0 * s**p + a1 * s ** (p + 1) - a2 * s ** (p + 2)
        return smooth * (x < r_max)

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
        r = lengths  # (E, 1)
        z_src = atomic_numbers[edge_index[0]].to(torch.int64).unsqueeze(-1)
        z_dst = atomic_numbers[edge_index[1]].to(torch.int64).unsqueeze(-1)
        # ZBL universal screening length (angstrom) and screening function
        screen_len = self.a_prefactor * 0.529 / (
            torch.pow(z_src, self.a_exp) + torch.pow(z_dst, self.a_exp))
        d = r / screen_len
        screening = (self.c[0] * torch.exp(-3.2 * d) + self.c[1] * torch.exp(-0.9423 * d)
                     + self.c[2] * torch.exp(-0.4028 * d) + self.c[3] * torch.exp(-0.2016 * d))
        pair_energy = (14.3996 * z_src * z_dst) / r * screening
        r_cut = self.covalent_radii[z_src] + self.covalent_radii[z_dst]
        # halve the double-counted pair sum and taper it to zero at r_cut
        pair_energy = 0.5 * pair_energy * self._envelope(r, r_cut, self.p)
        return scatter_sum(pair_energy, edge_index[1], num_nodes).squeeze(-1)


INTERACTIONS = {
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticDensityInteractionBlock": RealAgnosticDensityInteractionBlock,
    "RealAgnosticDensityResidualInteractionBlock":
        RealAgnosticDensityResidualInteractionBlock,
}
# GATES is the shared SCALAR_ACTIVATIONS registry (imported above): the gate
# options by their upstream spellings (silu/tanh/abs/ssp/None).


# ===========================================================================
# The MACE model
# ===========================================================================
@register_model("mace")
class MACE(EquivariantGNN):
    """Faithful MACE with a flexible number of interaction layers (T = 0..N).

    Subclasses :class:`~xnn.gnn.models.base.EquivariantGNN`, inheriting species
    bookkeeping, the per-element reference energy ``atom_ref``, and the
    :class:`~xnn.gnn.featurizers.SphericalHarmonicEdgeEmbedding` edge
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
        Chemistry-aware warp of the distance fed to the radial basis (the
        cutoff envelope always sees the raw distance): ``"None"`` (default),
        ``"Agnesi"`` or ``"Soft"`` (see
        :mod:`xnn.gnn.featurizers.radial`; ``"Agnesi"`` is what the
        MACE-MP-0b and later foundation models use).
    pair_repulsion : bool, optional
        If ``True``, add a :class:`_ZBLPairRepulsion` short-range term, by
        default ``False``.
    atomic_energies : torch.Tensor or None, optional
        Per-element reference energies (``E0s``) used to initialise
        ``atom_ref``.
    scale, shift : float, optional
        Affine rescaling of the per-atom *interaction* energy (readouts plus
        pair repulsion), ``E_i = E0_i + scale * E_int,i + shift`` -- the
        upstream ``ScaleShiftMACE`` convention. The defaults (1, 0) recover
        the plain MACE energy expression, so one class covers both upstream
        variants.

    Raises
    ------
    ValueError
        If ``num_interactions`` is negative.
    NotImplementedError
        If ``distance_transform`` is not one of the supported options.
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
        scale: float = 1.0,
        shift: float = 0.0,
    ):
        if num_interactions < 0:
            raise ValueError("num_interactions (T) must be >= 0")
        if distance_transform not in DISTANCE_TRANSFORMS:
            raise NotImplementedError(
                f"distance_transform={distance_transform!r} is not supported; "
                f"options: {[k for k in DISTANCE_TRANSFORMS if k]}"
            )
        # EquivariantGNN gives species/z_to_index/node_attr/atom_ref/edge_feat/
        # irreps_sh; the featurizer uses MACE's cutoff degree + radial type.
        super().__init__(species, cutoff, l_max=max_ell, n_rbf=n_rbf,
                         p=num_cutoff_basis, radial_type=radial_type)
        if atomic_energies is not None:  # per-element reference energy (E0s)
            self.set_atomic_energies(atomic_energies)
        self.distance_transform = DISTANCE_TRANSFORMS[distance_transform]()
        self.scale_shift = _ScaleShift(scale, shift)

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
        :class:`~xnn.common.data.AtomicGraph` dataclass and any Python-only
        constructs. Accumulates the *interaction* energy -- the optional ZBL
        pair-repulsion term plus one readout per round of interaction +
        product basis -- maps it through ``scale_shift`` (identity unless
        constructed with ``scale``/``shift``, the ``ScaleShiftMACE``
        convention) and adds the per-element reference energy. The radial
        basis sees the (optionally ``distance_transform``-warped) distance
        while the cutoff envelope always sees the raw one. The invariant
        (``l = 0``) channels of every layer's node features are collected
        alongside (what :class:`~xnn.common.models.les.LatentEwald`
        consumes).

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
        e0 = self.atom_ref(atomic_numbers).squeeze(-1)
        num_nodes = atomic_numbers.shape[0]
        # the cutoff envelope acts on the raw distance; the radial basis on
        # the (optionally chemistry-warped) one -- the upstream convention
        lengths = torch.linalg.norm(edge_vec, dim=-1)
        edge_sh = self.edge_feat.sph(edge_vec)
        r_basis = self.distance_transform(lengths, atomic_numbers, edge_index)
        edge_radial = self.edge_feat.rbf(r_basis) \
            * self.edge_feat.envelope(lengths)[:, None]

        inter_energy = torch.zeros_like(e0)
        if self.pair_repulsion:
            inter_energy = inter_energy + self.pair_repulsion_fn(
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
            inter_energy = inter_energy + readout(node_feats).squeeze(-1)
            # scalar (l=0) channels come first in the e3nn irreps layout
            feats_list.append(node_feats[:, :self._n_scalar_features])
        if len(feats_list) == 0:
            feats_list.append(node_feats)
        features = torch.cat(feats_list, dim=-1)

        node_energy = e0 + self.scale_shift(inter_energy)
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
        data : xnn.common.data.AtomicGraph
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
        xnn canonical key names are read here; upstream MACE-CLI spellings
        (``r_max``, ``num_radial_basis``, ``atomic_numbers``, ``E0s``, ...) are
        translated to these names at config-load time by
        :mod:`xnn.common.config.translate`. Values copied from an upstream
        yaml are coerced: ``species`` accepts a ``"[1, 6, 8]"`` string,
        ``radial_MLP`` a ``"[64, 64, 64]"`` string, and ``atomic_energies``
        (MACE ``E0s``) a list aligned with ``species``, a ``{Z: E0}`` dict, or
        the string form of either.

        Alternatively, ``extra["foundation"]`` names (or points to) a
        pretrained MACE foundation checkpoint: the model is then built by
        :meth:`from_foundation` (optionally with ``extra["head"]`` and
        ``extra["dtype"]``) and every architecture key is taken from the
        checkpoint instead of the config.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config, whose ``extra`` dict carries the MACE
            architecture options.

        Returns
        -------
        MACE
            The instantiated model.
        """
        import ast

        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        foundation = extra.get("foundation")
        if foundation is not None:
            model = cls.from_foundation(foundation, head=extra.get("head"),
                                        dtype=extra.get("dtype"))
            # the neighbor-list cutoff is wired through cfg.cutoff (the data
            # pipeline follows it), so it must equal the checkpoint's r_max
            if abs(float(cfg.cutoff) - float(model.cutoff)) > 1e-9:
                raise ValueError(
                    f"model.cutoff={cfg.cutoff} does not match the "
                    f"foundation checkpoint's r_max={model.cutoff}; set "
                    f"cutoff: {model.cutoff} in the config")
            return model
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
            scale=float(extra.get("scale", 1.0)),
            shift=float(extra.get("shift", 0.0)),
        )

    @classmethod
    def from_foundation(cls, source, head=None, dtype=None) -> "MACE":
        """Load a pretrained MACE foundation model into an xnn :class:`MACE`.

        Downloads (and caches) the requested checkpoint if needed, unpickles
        it with the ``mace-torch`` package, and converts it weight-for-weight
        into this implementation -- covering the ``ScaleShiftMACE`` energy
        expression, the Agnesi distance transform, ZBL pair repulsion, the
        density-normalized interaction generation, and multi-head
        checkpoints (sliced to one head). See
        :mod:`xnn.gnn.models.mace_foundation` for the alias registry and the
        conversion details.

        Parameters
        ----------
        source : str or Path or torch.nn.Module
            A registered alias (e.g. ``"mace-mp-0-medium"``,
            ``"mace-off23-small"``; see
            :data:`~xnn.gnn.models.mace_foundation.FOUNDATION_MODELS`), a
            checkpoint URL or local path, or an already-loaded ``mace-torch``
            model instance.
        head : str, optional
            Which head of a multi-head checkpoint to keep. Defaults to the
            checkpoint's only head; required (with the options listed in the
            error) when there are several.
        dtype : torch.dtype or str, optional
            Final dtype of the converted model; ``None`` keeps the
            checkpoint's (float64 for most foundation models).

        Returns
        -------
        MACE
            The converted model, ready for evaluation or fine-tuning.
        """
        from .mace_foundation import foundation_to_xnn
        return foundation_to_xnn(source, head=head, dtype=dtype)
