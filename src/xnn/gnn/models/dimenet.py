"""DimeNet and DimeNet++ (Gasteiger et al.) -- directional message passing.

Faithful implementations of the architectures described in

* J. Gasteiger, J. Gross, S. Guennemann, "Directional Message Passing for
  Molecular Graphs", ICLR 2020 (arXiv:2003.03123) -- DimeNet; and
* J. Gasteiger, S. Giri, J. T. Margraf, S. Guennemann, "Fast and
  Uncertainty-Aware Directional Message Passing for Non-Equilibrium
  Molecules", NeurIPS-W 2020 (arXiv:2011.14115) -- DimeNet++,

built from the papers' equations on the xnn abstractions
(:class:`~xnn.gnn.featurizers.BesselRBF`,
:class:`~xnn.gnn.featurizers.PolynomialCutoff`,
:class:`~xnn.gnn.featurizers.SphericalBesselBasis`,
:func:`~xnn.common.models.ops.scatter_sum`, ...) and consistent with the
conventions of the authors' reference code (parameter layout, initialization,
basis ordering), without copying it.

Directional message passing (ICLR paper section 4). Instead of atoms, the
network embeds the directed edges ``j -> i`` of the cutoff graph: the message
``m_ji`` carries the direction from ``j`` to ``i``. A message is updated from
the messages ``m_kj`` arriving at ``j`` from its other neighbors
``k != i``, using the distance ``d_ji`` and, jointly, the distance ``d_kj``
and the angle ``alpha_(kj,ji) = angle(x_k, x_j, x_i)`` at ``j`` (eq 4),

    m_ji^(l+1) = f_update(m_ji^(l), sum_k f_int(m_kj^(l), e_RBF(d_ji), a_SBF(d_kj, alpha))).

Representations (section 5): the distances enter through the Bessel radial
basis ``e_RBF,n(d) = sqrt(2/c) sin(n pi d / c) / d`` (eq 7, frequencies
fine-tuned by backpropagation) and distance/angle pairs through the 2D
spherical Fourier-Bessel basis ``a_SBF,ln`` (eq 6), both multiplied by the
polynomial envelope ``u(d)`` of eq 8 (``p = 6``) so that the model is twice
continuously differentiable and the autograd forces of
:class:`~xnn.common.models.outputs.ForceStressOutput` are continuous.

Architecture (ICLR Fig. 4, hyperparameters of its Appendix B):

* embedding block (eq 9): ``m_ji = sigma([h_j || h_i || sigma(W_e e_RBF + b_e)] W + b)``
  from learned atom-type embeddings ``h`` of width ``F``;
* ``T`` interaction blocks: ``m_ji`` and ``m_kj`` each pass a dense layer; the
  ``kj`` branch is gated element-wise by a linear map of ``e_RBF(d_ji)`` and,
  per triplet, contracted with a linear map of ``a_SBF`` through a bilinear
  weight tensor of size ``N_bilinear x F x F``; the triplet terms are summed
  over ``k``, added to the ``ji`` branch, refined by residual blocks (two
  dense layers and a skip), added to the block input through a skip
  connection and refined by further residual blocks;
* output blocks after the embedding and after every interaction block:
  ``m_ji`` gated by a linear map of ``e_RBF(d_ji)``, summed over the
  incoming messages of every atom ``i`` (``h_i = sum_j m_ji``), passed
  through dense layers and a final linear layer to the atom-wise output
  ``t_i^(l)``; the per-atom energy is ``sum_l t_i^(l)`` and the total energy
  the sum over atoms.

DimeNet++ (NeurIPS-W paper Fig. 1 and section 2) keeps the embedding, the
bases and the skip structure and changes the expensive parts: the bilinear
layer becomes a Hadamard product, the basis representations pass through
two-layer linear maps (``N_rbf -> N_basis -> F`` and
``N_sbf -> N_basis -> N_triplet``), the triplet branch is down-projected to
``N_triplet`` features before the aggregation and up-projected after it, the
output block up-projects the atom features to ``N_out`` before its dense
layers, and four blocks replace the six of DimeNet. The activation is the
self-gated swish ``sigma(x) = x sigmoid(x)`` throughout (``torch.nn.SiLU``).

Conventions shared with the reference implementation, for transplantable
weights: the atom embedding is indexed by the atomic number; the columns of
``a_SBF`` run degree-major (``l * N_SRBF + n``); dense layers are initialized
with Glorot-scaled orthogonal matrices and zero biases, the bilinear tensor
from ``N(0, (2/F)^2)``, the atom embeddings uniformly on
``[-sqrt(3), sqrt(3)]``, and the final linear layer of each output block
either to zero or Glorot-orthogonally (``output_init``; the reference uses
the latter for energies). Where the papers and the reference code differ, the model follows the paper
unless an option says otherwise (see :class:`DimeNet`): the angle is the
paper's ``alpha_(kj,ji)`` (the code's angle is its supplement ``pi - alpha``,
a sign flip of the odd-``l`` basis functions that a weight transplant
absorbs); the envelope of ``a_SBF`` is the paper's ``u(d)`` unless
``reference_basis`` is set (the code's carries an extra ``c / d_kj``); and
the radial gate of the incoming messages is the code's ``e_RBF(d_kj)`` by
default (``radial_gate="kj"``, what the published models were trained with),
while eq 4 as printed, ``e_RBF(d_ji)``, is ``radial_gate="ji"``.

The only addition is ``atom_ref``, the learnable per-element reference energy
every xnn potential carries (zero by default, so inert unless set/trained);
the paper trains on atomization energies, i.e. with these references
subtracted. Periodic systems are handled by the edge vectors of the graph
(cell shifts included); a triplet excludes only the exact reverse image of
the message's own edge, so an atom and its periodic images are distinct
neighbors.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve (the scriptable
# `node_features_energy` core is what LAMMPS deployment uses).
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.ops import glorot_orthogonal_, make_activation, scatter_sum
from xnn.common.models.registry import register_model
from ..featurizers import BesselRBF, PolynomialCutoff, SphericalBesselBasis

_MAX_Z = 100


def _dense(n_in: int, n_out: int, bias: bool = True,
           init: str = "glorot_orthogonal") -> nn.Linear:
    """A dense layer with the reference initialization (zero bias)."""
    lin = nn.Linear(n_in, n_out, bias=bias)
    if init == "zeros":
        nn.init.zeros_(lin.weight)
    elif init == "glorot_orthogonal":
        glorot_orthogonal_(lin.weight)
    else:
        raise ValueError(f"init must be 'zeros' or 'glorot_orthogonal', got {init!r}")
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


def directed_triplets(edge_index: Tensor, edge_vec: Tensor,
                      num_nodes: int) -> Tuple[Tensor, Tensor]:
    """Enumerate the message pairs ``(k -> j, j -> i)`` of directional message passing.

    For every directed edge ``j -> i`` (the message ``m_ji``) all edges
    ``k -> j`` arriving at its source are listed, except the reverse of the
    edge itself (``k = i`` in a molecule; in a periodic system only the same
    image of ``i``, found by the exact antisymmetry of the two edge vectors,
    so other images of ``i`` count as neighbors ``k``). Vectorized: the edges
    are grouped by their destination node and every edge ``j -> i`` is paired
    with the whole group of ``j``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``(2, E)``; row 0 is the source ``j`` and row 1 the
        destination ``i`` of each edge.
    edge_vec : Tensor
        Edge vectors ``x_i - x_j`` (cell shifts included), shape ``(E, 3)``.
    num_nodes : int
        Number of atoms.

    Returns
    -------
    tuple of Tensor
        ``(kj, ji)``, long tensors of shape ``(T,)`` indexing the edge
        dimension: ``kj[t]`` is the edge ``k -> j`` whose message feeds the
        update of the edge ``ji[t]`` (``j -> i``). Empty when there is no
        triplet. TorchScript-compatible.
    """
    src, dst = edge_index[0], edge_index[1]
    num_edges = src.shape[0]
    counts = torch.bincount(dst, minlength=num_nodes)          # edges into each node
    order = torch.argsort(dst)                                  # edges grouped by destination
    starts = torch.cumsum(counts, 0) - counts                   # first position of each group
    n_in = counts[src]                                          # candidates k -> j per edge j -> i
    total = int(n_in.sum())
    ji = torch.repeat_interleave(torch.arange(num_edges, device=src.device), n_in)
    first = torch.repeat_interleave(torch.cumsum(n_in, 0) - n_in, n_in)
    local = torch.arange(total, device=src.device) - first
    kj = order[starts[src[ji]] + local]
    # the reverse edge i -> j of the same image: same atom and the exact
    # negative of the edge vector
    reverse = (src[kj] == dst[ji]) & ((edge_vec[kj] + edge_vec[ji]) == 0).all(dim=-1)
    keep = ~reverse
    return kj[keep], ji[keep]


class _Residual(nn.Module):
    """Residual block of the interaction block: two dense layers and a skip.

    ``x + sigma(W_2 sigma(W_1 x + b_1) + b_2)`` (ICLR paper, "Interaction
    block": residual blocks "inspired by ResNet ... consist of two stacked
    dense layers and a skip connection").

    Parameters
    ----------
    n_features : int
        Width ``F`` of the features (kept constant).
    act : torch.nn.Module
        The activation (swish).
    """

    def __init__(self, n_features: int, act: nn.Module):
        super().__init__()
        self.lin1 = _dense(n_features, n_features)
        self.lin2 = _dense(n_features, n_features)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class _Embedding(nn.Module):
    """Embedding block (ICLR paper eq 9 and Fig. 4, left).

    Atom-type embeddings ``h_i`` (indexed by atomic number) and the radial
    basis of the edge, passed through a dense layer, are concatenated as
    ``[h_j || h_i || sigma(W_e e_RBF(d_ji) + b_e)]`` and mapped by a dense
    layer with activation to the first message ``m_ji``.

    Parameters
    ----------
    n_features : int
        Embedding and message width ``F``.
    n_rbf : int
        Number of radial basis functions.
    act : torch.nn.Module
        The activation.
    """

    def __init__(self, n_features: int, n_rbf: int, act: nn.Module):
        super().__init__()
        self.embedding = nn.Embedding(_MAX_Z, n_features)
        nn.init.uniform_(self.embedding.weight, -math.sqrt(3.0), math.sqrt(3.0))
        self.lin_rbf = _dense(n_rbf, n_features)
        self.lin = _dense(3 * n_features, n_features)
        self.act = act

    def forward(self, atomic_numbers: Tensor, edge_index: Tensor, rbf: Tensor) -> Tensor:
        h = self.embedding(atomic_numbers)
        e = self.act(self.lin_rbf(rbf))
        x = torch.cat([h[edge_index[0]], h[edge_index[1]], e], dim=-1)   # [h_j || h_i || e]
        return self.act(self.lin(x))


class _Interaction(nn.Module):
    """DimeNet interaction block (ICLR paper Fig. 4, middle).

    ``x_ji = sigma(W m_ji + b)`` and ``x_kj = sigma(W m_kj + b) o (W_RBF e_RBF(d))``
    (the element-wise radial gate the paper found to work better than a
    second bilinear layer; ``d = d_kj`` in the reference code and by default,
    ``d = d_ji`` as printed in eq 4 with ``radial_gate="ji"``); per triplet
    the message ``x_kj`` and the
    ``N_bilinear``-dimensional map ``W_SBF a_SBF`` of the 2D basis are
    contracted with the bilinear tensor ``W`` (``N_bilinear x F x F``,
    Appendix D eq 12-14) and summed over ``k``. The sum is added to ``x_ji``,
    refined by ``n_before_skip`` residual blocks and a dense layer, added to
    the block input ``m_ji`` (skip connection) and refined by
    ``n_after_skip`` residual blocks.

    Parameters
    ----------
    n_features : int
        Message width ``F``.
    n_rbf, n_sbf : int
        Sizes of the radial and the 2D basis.
    n_bilinear : int
        Inner dimension ``N_bilinear`` of the bilinear layer.
    n_before_skip, n_after_skip : int
        Residual blocks before and after the skip connection.
    act : torch.nn.Module
        The activation.
    """

    def __init__(self, n_features: int, n_rbf: int, n_sbf: int, n_bilinear: int,
                 n_before_skip: int, n_after_skip: int, act: nn.Module,
                 radial_gate: str = "kj"):
        super().__init__()
        self.lin_rbf = _dense(n_rbf, n_features, bias=False)
        self.lin_sbf = _dense(n_sbf, n_bilinear, bias=False)
        self.lin_ji = _dense(n_features, n_features)
        self.lin_kj = _dense(n_features, n_features)
        self.bilinear = nn.Parameter(torch.empty(n_bilinear, n_features, n_features))
        nn.init.normal_(self.bilinear, 0.0, 2.0 / n_features)
        self.before_skip = nn.ModuleList([_Residual(n_features, act) for _ in range(n_before_skip)])
        self.lin_skip = _dense(n_features, n_features)
        self.after_skip = nn.ModuleList([_Residual(n_features, act) for _ in range(n_after_skip)])
        self.act = act
        self.radial_gate = radial_gate

    def forward(self, m: Tensor, rbf: Tensor, sbf: Tensor, kj: Tensor, ji: Tensor) -> Tensor:
        x_ji = self.act(self.lin_ji(m))
        x_kj = self.act(self.lin_kj(m))
        gate = self.lin_rbf(rbf)
        if self.radial_gate == "ji":
            x_kj = x_kj[kj] * gate[ji]           # eq 4 as written: e_RBF(d_ji)
        else:
            x_kj = (x_kj * gate)[kj]             # the reference code: e_RBF(d_kj)
        # bilinear contraction per triplet: sum_{b,f} s_b x_f W_{bfg}
        msg = torch.einsum("tb,tf,bfg->tg", self.lin_sbf(sbf), x_kj, self.bilinear)
        x = x_ji + scatter_sum(msg, ji, m.shape[0])
        for block in self.before_skip:
            x = block(x)
        m = m + self.act(self.lin_skip(x))
        for block in self.after_skip:
            m = block(m)
        return m


class _InteractionPP(nn.Module):
    """DimeNet++ interaction block (NeurIPS-W paper Fig. 1, middle; section 2).

    The bilinear layer of :class:`_Interaction` is replaced by a Hadamard
    product, the basis representations pass through two bias-free linear
    layers each (``N_rbf -> N_basis -> F`` and ``N_sbf -> N_basis -> N_triplet``,
    the "MLPs for the basis representations" that recover the expressiveness),
    and the triplet branch is down-projected to ``N_triplet`` features before
    the gather onto the triplets and up-projected to ``F`` after the sum over
    ``k`` (the "embedding hierarchy"). Skip structure as in DimeNet.

    Parameters
    ----------
    n_features : int
        Message width ``F``.
    n_rbf, n_sbf : int
        Sizes of the radial and the 2D basis.
    n_triplet : int
        Width ``N_triplet`` of the triplet (down-projected) features.
    n_basis : int
        Inner width ``N_basis`` of the basis maps.
    n_before_skip, n_after_skip : int
        Residual blocks before and after the skip connection.
    act : torch.nn.Module
        The activation.
    """

    def __init__(self, n_features: int, n_rbf: int, n_sbf: int, n_triplet: int,
                 n_basis: int, n_before_skip: int, n_after_skip: int, act: nn.Module,
                 radial_gate: str = "kj"):
        super().__init__()
        self.lin_rbf1 = _dense(n_rbf, n_basis, bias=False)
        self.lin_rbf2 = _dense(n_basis, n_features, bias=False)
        self.lin_sbf1 = _dense(n_sbf, n_basis, bias=False)
        self.lin_sbf2 = _dense(n_basis, n_triplet, bias=False)
        self.lin_ji = _dense(n_features, n_features)
        self.lin_kj = _dense(n_features, n_features)
        self.lin_down = _dense(n_features, n_triplet, bias=False)
        self.lin_up = _dense(n_triplet, n_features, bias=False)
        self.before_skip = nn.ModuleList([_Residual(n_features, act) for _ in range(n_before_skip)])
        self.lin_skip = _dense(n_features, n_features)
        self.after_skip = nn.ModuleList([_Residual(n_features, act) for _ in range(n_after_skip)])
        self.act = act
        self.radial_gate = radial_gate

    def forward(self, m: Tensor, rbf: Tensor, sbf: Tensor, kj: Tensor, ji: Tensor) -> Tensor:
        x_ji = self.act(self.lin_ji(m))
        x_kj = self.act(self.lin_kj(m))
        gate = self.lin_rbf2(self.lin_rbf1(rbf))
        if self.radial_gate == "ji":
            x_kj = self.act(self.lin_down(x_kj[kj] * gate[ji]))     # eq 4 as written
        else:
            x_kj = self.act(self.lin_down(x_kj * gate))[kj]         # the reference code
        msg = x_kj * self.lin_sbf2(self.lin_sbf1(sbf))
        agg = self.act(self.lin_up(scatter_sum(msg, ji, m.shape[0])))
        x = x_ji + agg
        for block in self.before_skip:
            x = block(x)
        m = m + self.act(self.lin_skip(x))
        for block in self.after_skip:
            m = block(m)
        return m


class _Output(nn.Module):
    """Output block (ICLR paper Fig. 4, right; NeurIPS-W paper Fig. 1, right).

    The messages are gated by a bias-free linear map of ``e_RBF(d_ji)`` and
    summed onto their destination atom, ``h_i = sum_j m_ji o (W_RBF e_RBF)``;
    DimeNet++ then up-projects ``h_i`` to ``N_out`` features (bias-free, no
    activation). ``n_layers`` dense layers with activation and a bias-free
    linear layer give the atom-wise output ``t_i``.

    Parameters
    ----------
    n_features : int
        Message width ``F``.
    n_rbf : int
        Number of radial basis functions.
    n_out : int or None
        Width of the up-projected atom features (DimeNet++); ``None`` keeps
        ``F`` without an up-projection (DimeNet).
    n_layers : int
        Dense layers before the final linear layer.
    act : torch.nn.Module
        The activation.
    output_init : str
        Initialization of the final layer, ``"zeros"`` or
        ``"glorot_orthogonal"``.
    """

    def __init__(self, n_features: int, n_rbf: int, n_out: Optional[int], n_layers: int,
                 act: nn.Module, output_init: str):
        super().__init__()
        self.lin_rbf = _dense(n_rbf, n_features, bias=False)
        self.up: Optional[nn.Linear] = (None if n_out is None
                                        else _dense(n_features, n_out, bias=False))
        width = n_features if n_out is None else n_out
        self.dense = nn.ModuleList([_dense(width, width) for _ in range(n_layers)])
        self.final = _dense(width, 1, bias=False, init=output_init)
        self.act = act

    def forward(self, m: Tensor, rbf: Tensor, dst: Tensor,
                num_nodes: int) -> Tuple[Tensor, Tensor]:
        h = scatter_sum(m * self.lin_rbf(rbf), dst, num_nodes)
        if self.up is not None:
            h = self.up(h)
        for lin in self.dense:
            h = self.act(lin(h))
        return h, self.final(h).squeeze(-1)


@register_model("dimenet")
class DimeNet(InteratomicPotential):
    """DimeNet: directional message passing with the bilinear interaction (ICLR 2020).

    See the module docstring for the equation-by-equation walk-through. The
    defaults are the paper's QM9/MD17 architecture (Appendix B): ``F = 128``,
    six interaction blocks, ``N_SHBF = 7``, ``N_SRBF = N_RBF = 6``,
    ``N_bilinear = 8``, cutoff 5 Angstrom, envelope exponent ``p = 6``, one
    residual block before and two after the skip connection, three dense
    layers in the output blocks. Works for molecules and periodic systems
    (periodicity enters through the edge vectors).

    Parameters
    ----------
    n_features : int, optional
        Embedding and message width ``F``, by default 128.
    n_interactions : int, optional
        Number of interaction blocks ``T``, by default 6.
    n_rbf : int, optional
        Number of radial Bessel functions ``N_RBF`` (also the zeros per degree
        ``N_SRBF`` of the 2D basis, as in the paper), by default 6.
    n_spherical : int, optional
        Number of spherical-harmonic degrees ``N_SHBF`` of the 2D basis, by
        default 7.
    cutoff : float, optional
        Cutoff ``c`` in Angstrom, by default 5.0.
    n_bilinear : int, optional
        Inner dimension of the bilinear layer, by default 8.
    n_output_features : int or None, optional
        Width of the atom features in the output blocks; ``None`` (default,
        the paper) keeps ``F`` without an up-projection. DimeNet++ sets it.
    p : int, optional
        Exponent of the envelope ``u(d)`` of eq 8, by default 6 (the reference
        code's ``envelope_exponent + 1``).
    n_before_skip, n_after_skip : int, optional
        Residual blocks before and after the skip connection of each
        interaction block, by default 1 and 2.
    n_output_layers : int, optional
        Dense layers of each output block, by default 3.
    activation : str or torch.nn.Module, optional
        Activation, by default ``"silu"`` (the paper's swish); see
        :func:`~xnn.common.models.ops.make_activation`.
    output_init : str, optional
        Initialization of the final layer of the output blocks:
        ``"glorot_orthogonal"`` (default, the reference choice for energies)
        or ``"zeros"`` (the fresh model then predicts exactly ``atom_ref``).
    trainable_rbf : bool, optional
        Fine-tune the Bessel frequencies ``n pi`` by backpropagation (section
        5), by default ``True``.
    radial_gate : str, optional
        Which distance gates the incoming message ``m_kj`` in the interaction
        block: ``"kj"`` (default) is the reference implementation, where the
        gate is ``W e_RBF(d_kj)``; ``"ji"`` is eq 4 and Fig. 4 as printed,
        ``W e_RBF(d_ji)``. The published models were trained with ``"kj"``.
    reference_basis : bool, optional
        Reproduce the 2D basis of the reference implementation, whose
        envelope carries an extra factor ``c / d_kj`` relative to eq 6
        (its radial and 2D bases share one envelope that includes the
        ``1 / d`` of eq 7). ``False`` (default) is the paper's ``u(d) a_SBF``.
    species : list of int or None, optional
        Only used to interpret ``atomic_energies``; the model handles all
        elements up to Z = 99.
    atomic_energies : array-like or None, optional
        Per-species reference energies loaded into ``atom_ref`` (aligned with
        ``species``).

    Attributes
    ----------
    cutoff : float
        Neighbor-list cutoff ``c``.
    rbf : BesselRBF
        Radial basis of eq 7 (trainable frequencies).
    sbf : SphericalBesselBasis
        2D spherical Fourier-Bessel basis of eq 6.
    envelope : PolynomialCutoff
        The envelope ``u(d)`` of eq 8.
    embedding : _Embedding
        The embedding block (holds the atom-type embeddings).
    interactions : torch.nn.ModuleList
        The ``T`` interaction blocks.
    outputs : torch.nn.ModuleList
        The ``T + 1`` output blocks (one after the embedding block).
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference, initialized to zero.
    node_feature_dim : int
        Width of the invariant per-atom features returned as
        ``"node_features"`` (the summed output-block features before their
        final layer; what e.g. the LES charges are read from).
    """

    # one readout head = the output blocks and the reference energies
    head_modules = ("outputs", "atom_ref")

    def __init__(self, n_features: int = 128, n_interactions: int = 6, n_rbf: int = 6,
                 n_spherical: int = 7, cutoff: float = 5.0, n_bilinear: int = 8,
                 n_output_features: Optional[int] = None, p: int = 6,
                 n_before_skip: int = 1, n_after_skip: int = 2, n_output_layers: int = 3,
                 activation="silu", output_init: str = "glorot_orthogonal",
                 trainable_rbf: bool = True, radial_gate: str = "kj",
                 reference_basis: bool = False, species=None, atomic_energies=None):
        super().__init__()
        act = make_activation(activation)
        self._init_trunk(n_features, n_rbf, n_spherical, cutoff, n_output_features, p,
                         n_interactions, n_output_layers, act, output_init, trainable_rbf,
                         radial_gate, reference_basis, species, atomic_energies)
        self.interactions = nn.ModuleList([
            _Interaction(n_features, n_rbf, self.n_sbf, n_bilinear, n_before_skip,
                         n_after_skip, act, radial_gate)
            for _ in range(n_interactions)])

    def _init_trunk(self, n_features: int, n_rbf: int, n_spherical: int, cutoff: float,
                    n_output_features: Optional[int], p: int, n_interactions: int,
                    n_output_layers: int, act: nn.Module, output_init: str,
                    trainable_rbf: bool, radial_gate: str, reference_basis: bool,
                    species, atomic_energies) -> None:
        """Everything but the interaction blocks (shared with DimeNet++)."""
        if output_init not in ("zeros", "glorot_orthogonal"):
            raise ValueError(
                f"output_init must be 'zeros' or 'glorot_orthogonal', got {output_init!r}")
        if radial_gate not in ("kj", "ji"):
            raise ValueError(f"radial_gate must be 'kj' or 'ji', got {radial_gate!r}")
        self.cutoff = cutoff
        self.radial_gate = radial_gate
        self.reference_basis = reference_basis
        if species is not None:          # the elements the references are set for
            self.species = [int(z) for z in species]
        self.node_feature_dim = n_features if n_output_features is None else n_output_features
        self.rbf = BesselRBF(n_rbf, cutoff, trainable=trainable_rbf)
        self.sbf = SphericalBesselBasis(n_spherical, n_rbf, cutoff)
        self.n_sbf = self.sbf.output_dim
        self.envelope = PolynomialCutoff(cutoff, p)
        self.embedding = _Embedding(n_features, n_rbf, act)
        self.outputs = nn.ModuleList([
            _Output(n_features, n_rbf, n_output_features, n_output_layers, act, output_init)
            for _ in range(n_interactions + 1)])
        # per-element energy reference (learnable shift); the paper's targets
        # are atomization energies, i.e. these references subtracted
        self.atom_ref = nn.Embedding(_MAX_Z, 1)
        nn.init.zeros_(self.atom_ref.weight)
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)

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
    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> Tuple[Tensor, Tensor]:
        """TorchScript-compatible core: tensors in, features + energy out.

        The single implementation reused by :meth:`node_energy` (the deploy
        entry point) and :meth:`forward`.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``; row 0 is the source ``j`` and row 1
            the destination ``i`` of each edge (the message ``m_ji``).
        edge_vec : Tensor
            Edge vectors ``x_i - x_j``, shape ``(E, 3)`` (cell shifts included).

        Returns
        -------
        tuple of Tensor
            The invariant per-atom features ``(N, node_feature_dim)`` (the
            output-block features before their final layer, summed over the
            blocks) and the per-atom energy ``(N,)``,
            ``sum_l t_i^(l) + atom_ref[Z_i]``.
        """
        num_nodes = atomic_numbers.shape[0]
        dst = edge_index[1]
        r = torch.linalg.norm(edge_vec, dim=-1)
        env = self.envelope(r)
        rbf = self.rbf(r) * env[:, None]                               # eq 7 x eq 8
        # triplets k -> j -> i and the angle alpha_(kj,ji) at j (paper section 4)
        kj, ji = directed_triplets(edge_index, edge_vec, num_nodes)
        cos_alpha = -(edge_vec[kj] * edge_vec[ji]).sum(dim=-1) / (r[kj] * r[ji])
        cos_alpha = cos_alpha.clamp(-1.0, 1.0)
        env_sbf = env * (self.cutoff / r) if self.reference_basis else env
        radial = self.sbf.radial(r) * env_sbf[:, None, None]           # eq 6 x eq 8, per edge
        sbf = radial[kj] * self.sbf.angular(cos_alpha)[:, :, None]
        sbf = sbf.reshape(kj.shape[0], self.n_sbf)

        m = self.embedding(atomic_numbers, edge_index, rbf)           # eq 9
        messages: List[Tensor] = [m]
        for block in self.interactions:
            m = block(m, rbf, sbf, kj, ji)                             # eq 4
            messages.append(m)
        features = torch.zeros(num_nodes, self.node_feature_dim, dtype=m.dtype, device=m.device)
        node_energy = self.atom_ref(atomic_numbers).squeeze(-1)
        for i, out in enumerate(self.outputs):
            h, t = out(messages[i], rbf, dst, num_nodes)
            features = features + h
            node_energy = node_energy + t                              # t = sum_l t_i^(l)
        return features, node_energy

    @torch.jit.export
    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energy, shape ``(N,)`` (thin wrapper over
        :meth:`node_features_energy`; the deploy wrappers call this)."""
        out = self.node_features_energy(atomic_numbers, edge_index, edge_vec)
        return out[1]

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Compute per-atom and total energies for a batch of structures.

        Parameters
        ----------
        data : AtomicGraph
            Batched atomic graph providing atomic numbers, edge index and
            edge vectors.

        Returns
        -------
        dict[str, Tensor]
            ``"node_energy"`` (per-atom energy, shape ``(N,)``), ``"energy"``
            (per-structure total energy, the sum over atoms) and
            ``"node_features"`` (invariant per-atom features, shape
            ``(N, node_feature_dim)``).
        """
        features, node_energy = self.node_features_energy(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        energy = self.aggregate_energy(node_energy, data)
        return {"node_energy": node_energy, "energy": energy, "node_features": features}

    @staticmethod
    def _common_options(cfg) -> dict:
        """The constructor options shared by DimeNet and DimeNet++ read from a config."""
        from xnn.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        p = extra.get("p")
        if p is None:   # the reference code's spelling: envelope_exponent = p - 1
            p = int(extra.get("envelope_exponent", 5)) + 1
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        n_out = extra.get("n_output_features")
        return dict(
            n_features=cfg.n_features, n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf, cutoff=cfg.cutoff,
            n_spherical=int(extra.get("n_spherical", 7)),
            n_output_features=None if n_out is None else int(n_out),
            p=int(p),
            n_before_skip=int(extra.get("n_before_skip", 1)),
            n_after_skip=int(extra.get("n_after_skip", 2)),
            n_output_layers=int(extra.get("n_output_layers", 3)),
            activation=extra.get("activation", "silu"),
            output_init=extra.get("output_init", "glorot_orthogonal"),
            trainable_rbf=bool(extra.get("trainable_rbf", True)),
            radial_gate=str(extra.get("radial_gate", "kj")),
            reference_basis=bool(extra.get("reference_basis", False)),
            species=species,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species or [], "atomic_energies"),
        )

    @classmethod
    def from_config(cls, cfg) -> "DimeNet":
        """Build a :class:`DimeNet` from a configuration object.

        Core fields: ``cfg.n_features`` -> ``F``, ``cfg.n_interactions`` ->
        ``T``, ``cfg.n_rbf`` -> ``N_RBF = N_SRBF``, ``cfg.cutoff`` -> ``c``.
        ``cfg.extra`` may hold ``n_spherical``, ``n_bilinear``,
        ``n_output_features``, ``p`` (or the reference code's
        ``envelope_exponent = p - 1``), ``n_before_skip``, ``n_after_skip``,
        ``n_output_layers``, ``activation``, ``output_init``,
        ``trainable_rbf``, ``reference_basis``, ``species`` and
        ``atomic_energies``. The reference code's key spellings
        (``emb_size``, ``num_blocks``, ...) are translated by
        :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        DimeNet
            Instantiated model.
        """
        extra = dict(cfg.extra or {})
        return cls(n_bilinear=int(extra.get("n_bilinear", 8)), **cls._common_options(cfg))


@register_model("dimenet++")
class DimeNetPP(DimeNet):
    """DimeNet++: the fast directional message passing of Gasteiger et al. (NeurIPS-W 2020).

    Same trunk as :class:`DimeNet` with the Hadamard interaction, the
    two-layer basis maps, the triplet down/up-projections and the output
    up-projection described in the module docstring. The defaults are the
    paper's (Table 1, last row): ``F = 128``, four blocks, ``N_triplet = 64``,
    ``N_basis = 8``, ``N_out = 256``, ``N_SHBF = 7``, ``N_SRBF = N_RBF = 6``,
    cutoff 5 Angstrom.

    Parameters
    ----------
    n_features : int, optional
        Message width ``F``, by default 128.
    n_interactions : int, optional
        Number of interaction blocks, by default 4.
    n_rbf : int, optional
        Number of radial Bessel functions (and zeros per degree of the 2D
        basis), by default 6.
    n_spherical : int, optional
        Number of spherical-harmonic degrees of the 2D basis, by default 7.
    cutoff : float, optional
        Cutoff in Angstrom, by default 5.0.
    n_triplet_features : int, optional
        Width ``N_triplet`` of the down-projected triplet features, by
        default 64.
    n_basis_features : int, optional
        Inner width ``N_basis`` of the two-layer basis maps, by default 8.
    n_output_features : int, optional
        Width ``N_out`` of the up-projected atom features in the output
        blocks, by default 256.
    **kwargs
        The remaining options of :class:`DimeNet` (``p``, ``n_before_skip``,
        ``n_after_skip``, ``n_output_layers``, ``activation``,
        ``output_init``, ``trainable_rbf``, ``reference_basis``, ``species``,
        ``atomic_energies``).
    """

    def __init__(self, n_features: int = 128, n_interactions: int = 4, n_rbf: int = 6,
                 n_spherical: int = 7, cutoff: float = 5.0, n_triplet_features: int = 64,
                 n_basis_features: int = 8, n_output_features: int = 256, p: int = 6,
                 n_before_skip: int = 1, n_after_skip: int = 2, n_output_layers: int = 3,
                 activation="silu", output_init: str = "glorot_orthogonal",
                 trainable_rbf: bool = True, radial_gate: str = "kj",
                 reference_basis: bool = False, species=None, atomic_energies=None):
        InteratomicPotential.__init__(self)
        act = make_activation(activation)
        self._init_trunk(n_features, n_rbf, n_spherical, cutoff, n_output_features, p,
                         n_interactions, n_output_layers, act, output_init, trainable_rbf,
                         radial_gate, reference_basis, species, atomic_energies)
        self.interactions = nn.ModuleList([
            _InteractionPP(n_features, n_rbf, self.n_sbf, n_triplet_features, n_basis_features,
                           n_before_skip, n_after_skip, act, radial_gate)
            for _ in range(n_interactions)])

    @classmethod
    def from_config(cls, cfg) -> "DimeNetPP":
        """Build a :class:`DimeNetPP` from a configuration object.

        As :meth:`DimeNet.from_config`, with the extras ``n_triplet_features``
        (default 64), ``n_basis_features`` (8) and ``n_output_features``
        (256) in place of ``n_bilinear``; the reference code's
        ``int_emb_size`` / ``basis_emb_size`` / ``out_emb_size`` spellings are
        translated.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        DimeNetPP
            Instantiated model.
        """
        extra = dict(cfg.extra or {})
        options = cls._common_options(cfg)
        if options["n_output_features"] is None:
            options["n_output_features"] = 256
        return cls(n_triplet_features=int(extra.get("n_triplet_features", 64)),
                   n_basis_features=int(extra.get("n_basis_features", 8)), **options)
