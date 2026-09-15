"""CACE (Cheng 2024): Cartesian Atomic Cluster Expansion potential.

A faithful, self-contained re-implementation of the original
BingqingCheng/cace reference code on the xnns abstractions: it subclasses
:class:`~xnns.gnn.models.base.GNNPotential` (species bookkeeping, per-species
energy shift ``atom_ref``), reuses the shared
:class:`~xnns.gnn.featurizers.BesselRBF` /
:class:`~xnns.gnn.featurizers.PolynomialCutoff` radial pieces and the
:class:`~xnns.gnn.featurizers.CartesianAngularBasis`, and adds the genuinely
CACE-specific blocks -- the tensor-product edge-type encoding, the trainable
radial channel coupling, the Cartesian symmetrizer, and the two
message-passing mechanisms. Given the same weights it reproduces the original
package to machine precision (``tests/test_cace.py``), needing neither
spherical harmonics nor e3nn.

Architecture (paper eqs 1-15, npj Comput Mater 10, 157, 2024):

* every element is embedded into a low-dimensional vector ``theta`` (length
  ``n_atom_basis``, typically 1-4); an edge type is the flattened tensor
  product ``theta_i (x) theta_j`` -- ``c = n_atom_basis^2`` channels (eq 1);
* the edge basis ``chi = T_c(s_i, s_j) R_n(r_ji) L_l(r_hat_ji)`` combines the
  edge type with a (trainable) Bessel radial basis times a polynomial cutoff
  and the Cartesian angular monomials ``x^lx y^ly z^lz`` (eq 2);
* summing edges onto nodes gives the atom-centered ``A`` basis (eq 6), whose
  raw radial channels are mixed per ``(l, c)`` by a learned linear map
  (eq 5, the "radial channel coupling");
* products of ``A`` entries whose angular indices pair up with shared
  factors are summed with multinomial prefactors into the polynomially
  independent invariant ``B`` features of body order ``nu`` (eqs 7-10 and
  fig 1i);
* optional message passing (eqs 11-14): per-edge messages
  ``m1 = F(r_ji) A_j`` (learned exponential-decay filter, ``Ar``) and
  ``m2 = H(B_j) chi`` (recursive edge embedding, ``Bchi``) are aggregated,
  normalized by ``1/sqrt(avg_num_neighbors)`` and combined with a per-node
  memory term (``M``) into the next ``A``;
* the ``B`` features of all layers are concatenated and read out by the sum
  of a linear layer and an MLP (eq 15), plus the per-species reference
  energy ``atom_ref``.

Upstream conventions preserved: normalized edge vectors (receiver - sender,
``eps = 1e-9``); trainable Bessel basis with the MACE ``sqrt(2/cutoff)``
prefactor; ``torch.rand`` initialization of the radial-coupling and ``Ar``
filter parameters; the shared radial transform applied both to the initial
``A`` basis and to the aggregated ``Bchi`` messages; readout = linear +
[32, 16] SiLU MLP.

Unlike the other xnns GNNs this model has no TorchScript/LAMMPS export path
-- upstream CACE has none either (MD runs through the ASE calculator, which
uses the regular eager ``forward``).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from xnns.common.data import AtomicGraph
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model
from ..featurizers import BesselRBF, PolynomialCutoff
from ..featurizers.cartesian import (
    CartesianAngularBasis,
    lxlylz_list,
    multinomial_coefficient,
)
from .base import GNNPotential

MESSAGE_TYPES = ("M", "Ar", "Bchi")


# ---------------------------------------------------------------------------
# symmetrizer combination rules (upstream cace.modules.angular_tools)
# ---------------------------------------------------------------------------

def _combos_nu2(l_max: int) -> dict:
    """nu=2 invariants ``B2_l = sum C(l) A_l^2`` (paper eq 7): one group per
    total ``l`` in ``1..l_max``, each pairing every angular entry with itself."""
    combos: dict = {}
    lst = lxlylz_list(l_max)
    for c1 in lst:
        l1 = sum(c1)
        if l1 < 1:
            continue
        combos.setdefault(l1, []).append(
            ((c1, c1), multinomial_coefficient(c1)))
    return combos


def _combos_nu3(l_max: int) -> dict:
    """nu=3 invariants ``B3_l1l2 = sum C(l1) C(l2) A_l1 A_(l1+l2) A_l2``
    (paper eq 9), one group per ``(l1, l2)`` with ``1 <= l1 <= l2`` and
    ``l1 + l2 <= l_max``. Iteration order matches upstream
    ``find_combo_vectors_nu3`` so feature ordering is identical."""
    combos: dict = {}
    r = range(l_max + 1)
    for lx1 in r:
        for ly1 in r:
            for lz1 in r:
                l1 = lx1 + ly1 + lz1
                if not 0 < l1 <= l_max:
                    continue
                for lx2 in r:
                    for ly2 in r:
                        for lz2 in r:
                            l2 = lx2 + ly2 + lz2
                            if not l1 <= l2 <= l_max:
                                continue
                            c3 = (lx1 + lx2, ly1 + ly2, lz1 + lz2)
                            if sum(c3) > l_max:
                                continue
                            c1, c2 = (lx1, ly1, lz1), (lx2, ly2, lz2)
                            pref = (multinomial_coefficient(c1)
                                    * multinomial_coefficient(c2))
                            combos.setdefault((l1, l2), []).append(
                                ((c1, c2, c3), pref))
    return combos


def _combos_nu4(l_max: int) -> dict:
    """nu=4 invariants (paper eq 10), one group per ``(l1, l2, dl)`` with
    ``1 <= l1 < l2`` and shared factor ``dl >= 1``; the four coupled entries
    are ``(c1, c2, c1 + d, c2 + d)``. Matches upstream
    ``find_combo_vectors_nu4`` term-for-term."""
    combos: dict = {}
    r = range(l_max + 1)
    triples = [(x, y, z) for x in r for y in r for z in r]
    for c1 in triples:
        l1 = sum(c1)
        if not 0 < l1 <= l_max:
            continue
        for c2 in triples:
            l2 = sum(c2)
            if not l1 < l2 <= l_max:
                continue
            for d in triples:
                if sum(d) < 1:
                    continue
                c3 = tuple(a + b for a, b in zip(c1, d))
                c4 = tuple(a + b for a, b in zip(c2, d))
                if sum(c3) > l_max or sum(c4) > l_max:
                    continue
                pref = (multinomial_coefficient(c1)
                        * multinomial_coefficient(c2)
                        * multinomial_coefficient(d))
                combos.setdefault((l1, l2, sum(d)), []).append(
                    ((c1, c2, c3, c4), pref))
    return combos


_COMBO_BUILDERS = {2: _combos_nu2, 3: _combos_nu3, 4: _combos_nu4}


class _Symmetrizer(nn.Module):
    """Contract the ``A`` basis into invariant ``B`` features (paper fig 1i).

    For each body order ``nu`` the angular entries listed by the combination
    rules are multiplied together, weighted by their multinomial prefactor and
    summed into one output feature per group -- a fully vectorized version of
    upstream ``cace.modules.Symmetrizer`` (gather -> product -> ``index_add``)
    producing bit-identical features in the same order.

    Parameters
    ----------
    max_nu : int
        Maximum body order of the invariants (1-4; ``nu = 1`` is the bare
        ``l = 0`` channel).
    l_max : int
        Maximum total angular momentum of the ``A`` basis.
    """

    def __init__(self, max_nu: int, l_max: int):
        super().__init__()
        if not 1 <= max_nu <= 4:
            raise ValueError(f"max_nu must be in 1..4, got {max_nu}")
        self.max_nu = max_nu
        self.l_max = l_max
        index = {c: i for i, c in enumerate(lxlylz_list(l_max))}
        self.n_features = 1
        for nu in (2, 3, 4):
            idx: list[list[int]] = []
            pref: list[float] = []
            out: list[int] = []
            if nu <= max_nu:
                groups = _COMBO_BUILDERS[nu](l_max)
                for group, terms in enumerate(groups.values()):
                    for combos, prefactor in terms:
                        idx.append([index[c] for c in combos])
                        pref.append(float(prefactor))
                        out.append(self.n_features + group)
                self.n_features += len(groups)
            self.register_buffer(f"_idx{nu}", torch.tensor(idx, dtype=torch.long),
                                 persistent=False)
            self.register_buffer(f"_pref{nu}", torch.tensor(pref), persistent=False)
            self.register_buffer(f"_out{nu}", torch.tensor(out, dtype=torch.long),
                                 persistent=False)

    def forward(self, node_feat_a: Tensor) -> Tensor:
        """Symmetrize ``(N, R, n_angular, C) -> (N, R, n_features, C)``.

        Parameters
        ----------
        node_feat_a : Tensor
            The atom-centered ``A`` basis.

        Returns
        -------
        Tensor
            The invariant ``B`` features, ordered ``nu = 1`` first, then the
            ``nu = 2..max_nu`` groups.
        """
        n, r, _, c = node_feat_a.shape
        out = node_feat_a.new_zeros(n, r, self.n_features, c)
        out[:, :, 0, :] = node_feat_a[:, :, 0, :]
        for nu in (2, 3, 4):
            idx = getattr(self, f"_idx{nu}")
            if idx.numel() == 0:
                continue
            prods = node_feat_a[:, :, idx, :].prod(dim=3)  # (N, R, terms, C)
            prods = prods * getattr(self, f"_pref{nu}")[None, None, :, None]
            out.index_add_(2, getattr(self, f"_out{nu}"), prods)
        return out


class _SharedRadialTransform(nn.Module):
    """Trainable radial channel coupling ``R_n,cl = sum_n~ R_n~ W_n~n,cl``
    (paper eq 5), applied on the atom-centered basis for efficiency.

    One ``(n_rbf, n_radial_basis, channels)`` weight per total ``l``, shared
    by all angular entries of that ``l`` -- the stacked, single-einsum
    equivalent of upstream ``cace.modules.SharedRadialLinearTransform``.

    Parameters
    ----------
    l_of_entry : Tensor
        Total ``l`` of every angular entry (from
        :class:`~xnns.gnn.featurizers.CartesianAngularBasis`).
    n_rbf : int
        Raw radial basis size (input width).
    n_radial_basis : int
        Mixed radial embedding size (output width).
    channels : int
        Number of edge channels ``c``.
    """

    def __init__(self, l_of_entry: Tensor, n_rbf: int, n_radial_basis: int,
                 channels: int):
        super().__init__()
        l_max = int(l_of_entry.max())
        # upstream initializes with torch.rand (uniform [0, 1))
        self.weight = nn.Parameter(
            torch.rand(l_max + 1, n_rbf, n_radial_basis, channels))
        self.register_buffer("l_of_entry", l_of_entry.clone(), persistent=False)

    def forward(self, node_feat_a: Tensor) -> Tensor:
        """Mix raw radial channels: ``(N, n_rbf, L, C) -> (N, n_radial_basis, L, C)``."""
        return torch.einsum("nrac,armc->nmac",
                            node_feat_a, self.weight[self.l_of_entry])


class _NodeMemory(nn.Module):
    """Per-node memory term of the ``A`` update (message type ``M``): the old
    features scaled by a learned per-``(l, n, c)`` coefficient (init 0.25),
    upstream ``cace.modules.NodeMemory``."""

    def __init__(self, l_of_entry: Tensor, n_radial_basis: int, channels: int):
        super().__init__()
        l_max = int(l_of_entry.max())
        self.memory_coef = nn.Parameter(
            torch.full((l_max + 1, n_radial_basis, channels), 0.25))
        self.register_buffer("l_of_entry", l_of_entry.clone(), persistent=False)

    def forward(self, node_feat_a: Tensor) -> Tensor:
        """Scale ``(N, R, L, C)`` by the memory coefficients (same shape out)."""
        return node_feat_a * self.memory_coef[self.l_of_entry].permute(1, 0, 2)


class _MessageAr(nn.Module):
    """Orientation-dependent message ``m1_ji = F(r_ji) A_j`` (paper eq 11,
    message type ``Ar``): the sender's ``A`` features scaled by a trainable
    exponential-decay filter ``a exp(-r/r0)`` times the cutoff envelope, with
    independent ``(a, 1/r0)`` per ``(l, n, c)``. Upstream
    ``cace.modules.MessageAr`` (``1/r0`` init uniform in
    ``[0.5/cutoff, 1.5/cutoff)``)."""

    def __init__(self, l_of_entry: Tensor, cutoff: float, n_radial_basis: int,
                 channels: int):
        super().__init__()
        l_max = int(l_of_entry.max())
        self.prefactor = nn.Parameter(
            torch.rand(l_max + 1, n_radial_basis, channels))
        self.inv_r0 = nn.Parameter(
            (torch.rand(l_max + 1, n_radial_basis, channels) + 0.5) / cutoff)
        self.register_buffer("l_of_entry", l_of_entry.clone(), persistent=False)

    def forward(self, node_feat_a: Tensor, lengths: Tensor, envelope: Tensor,
                sender: Tensor) -> Tensor:
        """Per-edge messages ``(E, R, L, C)`` from node features ``(N, R, L, C)``.

        Parameters
        ----------
        node_feat_a : Tensor
            Current ``A`` features, shape ``(N, R, L, C)``.
        lengths : Tensor
            Edge lengths, shape ``(E,)``.
        envelope : Tensor
            Cutoff-envelope values at the edge lengths, shape ``(E,)``.
        sender : Tensor
            Sender (neighbor) node index of each edge, shape ``(E,)``.
        """
        decay = (torch.exp(-lengths[:, None, None, None] * self.inv_r0)
                 * self.prefactor * envelope[:, None, None, None])
        return node_feat_a[sender] * decay[:, self.l_of_entry].transpose(1, 2)


class _MessageBchi(nn.Module):
    """Recursive edge-embedding message ``m2_ji = H(B_j) chi`` (paper eq 12,
    message type ``Bchi``): the layer-0 edge basis reweighted by a linear
    function of the sender's invariant ``B`` features -- upstream
    ``cace.modules.MessageBchi`` with its defaults (one shared scalar weight
    across ``l``, ``n`` and ``c``)."""

    def __init__(self, n_b_features: int):
        super().__init__()
        self.h = nn.Linear(n_b_features, 1)

    def forward(self, node_feat_b: Tensor, edge_attr: Tensor,
                sender: Tensor) -> Tensor:
        """Per-edge messages ``(E, n_rbf, L, C)``.

        Parameters
        ----------
        node_feat_b : Tensor
            Current ``B`` features, shape ``(N, R, n_B, C)``.
        edge_attr : Tensor
            The layer-0 edge basis ``chi``, shape ``(E, n_rbf, L, C)``.
        sender : Tensor
            Sender (neighbor) node index of each edge, shape ``(E,)``.
        """
        weight = self.h(node_feat_b.flatten(1))  # (N, 1)
        return edge_attr * weight[sender][:, :, None, None]


class _CaceInteraction(nn.Module):
    """One CACE message-passing layer: aggregates the enabled message types
    into the ``A`` update ``A^(t+1) = (Ar + Bchi) / sqrt(avg_n) + M`` (paper
    eqs 13-14 with the upstream linear ``G``).

    Parameters
    ----------
    l_of_entry : Tensor
        Total ``l`` per angular entry.
    cutoff : float
        Radial cutoff (for the ``Ar`` filter init).
    n_radial_basis, channels : int
        Mixed radial / edge-channel widths of the ``A`` basis.
    n_b_flat : int
        Flattened size of the ``B`` features (input of the ``Bchi`` weight).
    message_types : sequence of str
        Enabled mechanisms, subset of ``{"M", "Ar", "Bchi"}``.
    mp_norm : float
        Message normalization ``1/sqrt(avg_num_neighbors)``.
    """

    def __init__(self, l_of_entry: Tensor, cutoff: float, n_radial_basis: int,
                 channels: int, n_b_flat: int, message_types, mp_norm: float):
        super().__init__()
        self.mp_norm = mp_norm
        self.memory = (_NodeMemory(l_of_entry, n_radial_basis, channels)
                       if "M" in message_types else None)
        self.message_ar = (_MessageAr(l_of_entry, cutoff, n_radial_basis, channels)
                           if "Ar" in message_types else None)
        self.message_bchi = (_MessageBchi(n_b_flat)
                             if "Bchi" in message_types else None)

    def forward(self, node_feat_a: Tensor, node_feat_b: Tensor,
                edge_attr: Tensor, lengths: Tensor, envelope: Tensor,
                edge_index: Tensor, radial_transform: nn.Module) -> Tensor:
        """Compute the updated ``A`` features ``(N, R, L, C)``.

        Parameters
        ----------
        node_feat_a, node_feat_b : Tensor
            Current ``A`` ``(N, R, L, C)`` and ``B`` ``(N, R, n_B, C)``.
        edge_attr : Tensor
            Layer-0 edge basis ``chi`` with raw radial channels,
            ``(E, n_rbf, L, C)``.
        lengths, envelope : Tensor
            Edge lengths and cutoff-envelope values, ``(E,)``.
        edge_index : Tensor
            ``(2, E)`` sender/receiver indices.
        radial_transform : nn.Module
            The model's shared radial coupling, applied to the aggregated
            ``Bchi`` messages (raw radial -> mixed radial channels).
        """
        sender, receiver = edge_index[0], edge_index[1]
        n_nodes = node_feat_a.shape[0]
        new_a = torch.zeros_like(node_feat_a)
        if self.message_ar is not None:
            messages = self.message_ar(node_feat_a, lengths, envelope, sender)
            new_a = new_a + scatter_sum(messages, receiver, n_nodes)
        if self.message_bchi is not None:
            messages = self.message_bchi(node_feat_b, edge_attr, sender)
            new_a = new_a + radial_transform(
                scatter_sum(messages, receiver, n_nodes))
        new_a = new_a * self.mp_norm
        if self.memory is not None:
            new_a = new_a + self.memory(node_feat_a)
        return new_a


@register_model("cace")
class CACE(GNNPotential):
    """Faithful CACE (Cheng 2024): Cartesian atomic cluster expansion.

    Body-ordered invariant features built entirely in Cartesian coordinates
    (see the module docstring for the architecture walk-through), read out by
    a linear + MLP head into per-atom energies. With ``num_message_passing=0``
    this is exactly an (optimized-radial-coupling, element-embedded) ACE; each
    message-passing layer appends one more set of ``B`` features.

    All architecture options are read from ``ModelConfig.extra`` (see
    :meth:`from_config`); upstream constructor spellings are translated to
    the xnns names at config-load time by
    :mod:`xnns.common.config.translate`.

    Parameters
    ----------
    species : list of int
        Atomic numbers of the supported elements, in channel order
        (upstream ``zs``).
    cutoff : float, optional
        Radial cutoff in Angstrom, by default 5.5 (the paper's water model).
    n_atom_basis : int, optional
        Length of the learnable element embedding ``theta`` (paper
        ``N_embedding``, typically 1-4), by default 3. The number of edge
        channels is ``n_atom_basis**2``.
    n_rbf : int, optional
        Number of raw Bessel radial functions, by default 8.
    n_radial_basis : int or None, optional
        Mixed radial channels after the learned coupling (paper ``n``);
        ``None`` (default) keeps ``n_rbf``.
    max_l : int, optional
        Maximum total angular momentum of the Cartesian basis, by default 3.
    max_nu : int, optional
        Maximum body order of the invariant ``B`` features (1-4), by
        default 3.
    num_message_passing : int, optional
        Number of message-passing layers ``T`` (0 = plain Cartesian ACE), by
        default 1.
    message_types : sequence of str, optional
        Enabled message mechanisms per layer, subset of ``("M", "Ar",
        "Bchi")`` (node memory, radial-filter message, recursive edge
        embedding), by default all three (the upstream default; the paper's
        water model uses ``("Bchi",)``).
    embed_receiver_nodes : bool, optional
        Use a separate embedding table for receiver atoms in the edge type
        ``theta_i (x) theta_j`` (upstream flag), by default ``False``
        (sender table shared, as in the upstream constructor default).
    avg_num_neighbors : float, optional
        Message normalization ``1/sqrt(avg_num_neighbors)``, by default 10.0.
    num_polynomial_cutoff : int, optional
        Degree ``p`` of the polynomial cutoff envelope, by default 6.
    trainable_rbf : bool, optional
        Learnable Bessel frequencies, by default ``True`` (the paper's
        "trainable Bessel functions").
    readout_hidden : list of int, optional
        Hidden widths of the readout MLP, by default ``[32, 16]`` (the
        upstream example). The readout is the sum of this MLP and a parallel
        linear layer (paper eq 15).
    atomic_energies : array-like or None, optional
        Per-species reference energies folded into ``atom_ref`` (upstream
        subtracts them from the training data instead).
    """

    def __init__(
        self,
        species: list[int],
        cutoff: float = 5.5,
        n_atom_basis: int = 3,
        n_rbf: int = 8,
        n_radial_basis: int | None = None,
        max_l: int = 3,
        max_nu: int = 3,
        num_message_passing: int = 1,
        message_types=("M", "Ar", "Bchi"),
        embed_receiver_nodes: bool = False,
        avg_num_neighbors: float = 10.0,
        num_polynomial_cutoff: int = 6,
        trainable_rbf: bool = True,
        readout_hidden: list[int] | None = None,
        atomic_energies=None,
    ):
        super().__init__(species, cutoff)
        message_types = tuple(message_types)
        unknown = set(message_types) - set(MESSAGE_TYPES)
        if unknown:
            raise ValueError(
                f"unknown message types {sorted(unknown)}; choose from {MESSAGE_TYPES}")
        if num_message_passing > 0 and not message_types:
            raise ValueError("message_types must not be empty when "
                             "num_message_passing > 0")
        self.n_atom_basis = n_atom_basis
        self.channels = n_atom_basis ** 2
        self.n_rbf = n_rbf
        self.n_radial_basis = n_radial_basis or n_rbf
        self.max_l = max_l
        self.max_nu = max_nu
        self.message_types = message_types
        self.mp_norm = avg_num_neighbors ** -0.5

        # element embeddings theta (xavier-uniform, as upstream NodeEmbedding)
        self.embed_sender = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(len(self.species), n_atom_basis)))
        if embed_receiver_nodes:
            self.embed_receiver = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(len(self.species), n_atom_basis)))
        else:
            self.embed_receiver = self.embed_sender

        self.rbf = BesselRBF(n_rbf, cutoff, trainable=trainable_rbf)
        self.envelope = PolynomialCutoff(cutoff, p=num_polynomial_cutoff)
        self.angular = CartesianAngularBasis(max_l)
        l_of_entry = self.angular.l_of_entry
        self.radial_transform = _SharedRadialTransform(
            l_of_entry, n_rbf, self.n_radial_basis, self.channels)
        self.symmetrizer = _Symmetrizer(max_nu, max_l)
        self.n_b_features = self.symmetrizer.n_features

        n_b_flat = self.n_radial_basis * self.n_b_features * self.channels
        self.interactions = nn.ModuleList([
            _CaceInteraction(l_of_entry, cutoff, self.n_radial_basis,
                             self.channels, n_b_flat, message_types,
                             self.mp_norm)
            for _ in range(num_message_passing)])

        # readout: linear + MLP on the concatenated B features (paper eq 15)
        feat_dim = n_b_flat * (num_message_passing + 1)
        self.node_feature_dim = feat_dim  # invariant features (for e.g. LES)
        readout_hidden = list(readout_hidden or [32, 16])
        layers: list[nn.Module] = []
        widths = [feat_dim] + readout_hidden
        for w_in, w_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(w_in, w_out), nn.SiLU()]
        layers.append(nn.Linear(widths[-1], 1))
        self.readout_mlp = nn.Sequential(*layers)
        self.readout_linear = nn.Linear(feat_dim, 1)

        if atomic_energies is not None:
            self.set_atomic_energies(atomic_energies)

    def node_features_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                             edge_vec: Tensor) -> tuple[Tensor, Tensor]:
        """Tensor core: invariant features and per-atom energies.

        Parameters
        ----------
        atomic_numbers : Tensor
            Per-atom atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index ``(2, E)``; row 0 is the sender (neighbor), row 1 the
            receiver (center) -- the upstream CACE/MACE convention.
        edge_vec : Tensor
            Edge vectors ``pos[receiver] - pos[sender]``, shape ``(E, 3)``.

        Returns
        -------
        tuple of Tensor
            The concatenated invariant ``B`` features
            ``(N, node_feature_dim)`` (what the readout consumes, and what
            :class:`~xnns.common.models.les.LatentEwald` maps to latent
            charges) and the per-atom energies ``(N,)``.
        """
        n_nodes = atomic_numbers.shape[0]
        sender, receiver = edge_index[0], edge_index[1]
        lengths = edge_vec.norm(dim=-1)
        unit_vec = edge_vec / (lengths + 1e-9).unsqueeze(-1)

        # edge type T = theta_i (x) theta_j (paper eq 1)
        one_hot = self.node_attr(atomic_numbers)
        theta_sender = (one_hot @ self.embed_sender)[sender]
        theta_receiver = (one_hot @ self.embed_receiver)[receiver]
        edge_type = (theta_sender.unsqueeze(2)
                     * theta_receiver.unsqueeze(1)).flatten(1)  # (E, C)

        # edge basis chi = T * R(r) * L(r_hat), shape (E, n_rbf, L, C)
        envelope = self.envelope(lengths)
        radial = self.rbf(lengths) * envelope.unsqueeze(-1)
        angular = self.angular(unit_vec)
        edge_attr = (radial.unsqueeze(2).unsqueeze(3)
                     * angular.unsqueeze(1).unsqueeze(3)
                     * edge_type.unsqueeze(1).unsqueeze(2))

        # A basis (eq 6) with mixed radial channels (eq 5), then B (eqs 7-10)
        node_feat_a = self.radial_transform(
            scatter_sum(edge_attr, receiver, n_nodes))
        node_feats = [self.symmetrizer(node_feat_a)]
        for interaction in self.interactions:
            node_feat_a = interaction(node_feat_a, node_feats[-1], edge_attr,
                                      lengths, envelope, edge_index,
                                      self.radial_transform)
            node_feats.append(self.symmetrizer(node_feat_a))

        features = torch.stack(node_feats, dim=-1).flatten(1)
        energy = self.readout_mlp(features) + self.readout_linear(features)
        return features, energy.squeeze(-1) + self.atom_ref(atomic_numbers).squeeze(-1)

    def node_energy(self, atomic_numbers: Tensor, edge_index: Tensor,
                    edge_vec: Tensor) -> Tensor:
        """Per-atom energies from raw graph tensors (thin wrapper over
        :meth:`node_features_energy`)."""
        return self.node_features_energy(atomic_numbers, edge_index, edge_vec)[1]

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict per-node and total energy for an atomic graph.

        Parameters
        ----------
        data : xnns.common.data.AtomicGraph
            The input atomic graph.

        Returns
        -------
        dict of str to torch.Tensor
            ``"node_energy"`` (per-atom energies), ``"energy"``
            (per-structure totals) and ``"node_features"`` (the invariant
            ``B`` features, shape ``(N, node_feature_dim)``).
        """
        features, node_energy = self.node_features_energy(
            data.atomic_numbers, data.edge_index, data.edge_vectors())
        return {"node_energy": node_energy,
                "energy": self.aggregate_energy(node_energy, data),
                "node_features": features}

    @classmethod
    def from_config(cls, cfg) -> "CACE":
        """Construct a :class:`CACE` from a core model config.

        Reads the CACE hyper-parameters from ``cfg.extra``; upstream
        spellings are translated by :mod:`xnns.common.config.translate` and
        value forms coerced by :mod:`xnns.common.config.coerce`. Note that
        ``cfg.n_features`` is not used -- CACE's feature width is set by
        ``n_atom_basis`` / ``n_radial_basis`` / ``max_l`` / ``max_nu``.

        Parameters
        ----------
        cfg : xnns.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        CACE
            The instantiated model.
        """
        import ast

        from xnns.common.config.coerce import coerce_per_species, coerce_species

        extra = dict(cfg.extra or {})
        species = coerce_species(extra.get("species"), default=[1, 6, 8])
        message_types = extra.get("message_types", list(MESSAGE_TYPES))
        if isinstance(message_types, str):
            message_types = ast.literal_eval(message_types)
        readout_hidden = extra.get("readout_hidden")
        if isinstance(readout_hidden, str):
            readout_hidden = ast.literal_eval(readout_hidden)
        n_radial_basis = extra.get("n_radial_basis")
        if n_radial_basis is not None:
            n_radial_basis = int(n_radial_basis)
        return cls(
            species=species,
            cutoff=cfg.cutoff,
            n_atom_basis=int(extra.get("n_atom_basis", 3)),
            n_rbf=cfg.n_rbf,
            n_radial_basis=n_radial_basis,
            max_l=int(extra.get("max_l", 3)),
            max_nu=int(extra.get("max_nu", 3)),
            num_message_passing=cfg.n_interactions,
            message_types=message_types,
            embed_receiver_nodes=bool(extra.get("embed_receiver_nodes", False)),
            avg_num_neighbors=float(extra.get("avg_num_neighbors", 10.0)),
            num_polynomial_cutoff=int(extra.get("num_polynomial_cutoff", 6)),
            trainable_rbf=bool(extra.get("trainable_rbf", True)),
            readout_hidden=readout_hidden,
            atomic_energies=coerce_per_species(
                extra.get("atomic_energies"), species, "atomic_energies"),
        )
