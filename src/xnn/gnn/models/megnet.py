"""MEGNet: materials graph networks with a global state (Chen et al., Chem. Mater. 31, 3564 (2019)).

A MEGNet model predicts one property per structure (a formation energy per
atom, a band gap, a modulus) from a graph of atoms ``v_i``, bonds ``e_k`` and
a global state ``u``. Every block updates the three in turn (eqs. 1-6 of the
paper):

* ``e'_k = phi_e(v_i + v_j + e_k + u)``, for the bond ``k`` from atom ``j`` to
  atom ``i`` (``+`` is concatenation);
* ``v'_i = phi_v(v_i + mean_k e'_k + u)``, the mean over the bonds of atom ``i``;
* ``u' = phi_u(u + mean_k e'_k + mean_i v'_i)``, the means over the structure.

The ``phi`` are dense networks with the shifted softplus ``ln(e^x + 1) - ln 2``
(eq. 7). Each block after the first starts with two dense layers per kind, and
a residual connection spans every block (Fig. 2). The atoms enter as element
embeddings, the bonds as Gaussian expansions of their lengths and the state as
given features. Set2Set readouts (Vinyals et al., 2015) reduce the atoms and
the bonds of every structure, and a dense network maps them and the state to
the property.

Multi-fidelity MEGNet (Chen et al., Nat. Comput. Sci. 1, 46 (2021)) is the
same network with an integer state: the fidelity of each label (PBE, GLLB-SC,
HSE, SCAN, experiment) indexes a trainable fidelity embedding that becomes
``u``. Set ``n_state_types`` and give every structure its label in
``AtomicGraph.state``. A disordered site (``AtomicGraph.site_species`` and
``site_occupancy``) takes the occupancy-weighted sum of its species' element
embeddings, as in that paper.

The defaults are those of MatGL's MEGNet (the crystal models of the paper):
16-wide element embeddings, 100 Gaussians on ``[0, cutoff + 1]`` with
``exp(-0.5 (r - mu)^2)``, a 4 Angstrom cutoff, dense layers of 64 and 32
units, three blocks, two Set2Set steps and a two-dimensional zero state. The
paper's Gaussians, ``exp(-(r - mu)^2 / 0.5^2)``, are ``gamma=4.0``.

The output is ``"properties"``, one value (or ``n_outputs`` values) per
structure, standardized as ``property_scale * y + property_shift``. Train it
with ``optim.property_weight`` on ``properties`` labels. The model predicts no
energy, so it gives no forces or stress.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import make_radial_basis
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.blocks import MLP
from xnn.common.models.constants import ExactFloat64Constants
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import gather_rows, init_linears_, scatter_sum, segment_softmax
from xnn.common.models.registry import register_model


def _mean(src: Tensor, index: Tensor, count: Tensor) -> Tensor:
    """Mean of the rows of ``src`` per bucket of ``index``; an empty bucket gives zeros."""
    return scatter_sum(src, index, count.shape[0]) / count.clamp(min=1).unsqueeze(-1).to(src.dtype)


def _grad(*tensors) -> bool:
    """Whether autograd records an operation on any of ``tensors``."""
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


class Set2Set(nn.Module, FastPathModule):
    """Set2Set readout of a set of vectors per structure (Vinyals, Bengio and Kudlur, 2015).

    An LSTM reads the query ``q*`` of every structure, attention over the
    rows of the structure (a softmax of ``x_r . q``) forms the readout ``r``,
    and ``q* = q + r`` (concatenation) feeds the next step. The output is
    the last ``q*``, twice the input width.

    Parameters
    ----------
    n_features : int
        Width of the rows.
    steps : int
        Number of attention steps.
    layers : int, optional
        Number of LSTM layers, by default 1.
    """

    def __init__(self, n_features: int, steps: int, layers: int = 1):
        super().__init__()
        self.n_features = int(n_features)
        self.steps = int(steps)
        self.layers = int(layers)
        self.lstm = nn.LSTM(2 * self.n_features, self.n_features, self.layers)
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """The fast stage writes out a plain LSTM (biases, no dropout, no projection)."""
        lstm = self.lstm
        return (dtype in (torch.float32, torch.float64) and lstm.bias and lstm.dropout == 0
                and lstm.proj_size == 0 and not lstm.batch_first and not lstm.bidirectional)

    def forward(self, x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """Read out ``x`` ``(R, F)``, whose rows belong to the structures ``index`` ``(R,)``.

        Returns
        -------
        Tensor
            The readout ``(num_segments, 2 F)``.
        """
        if self.fast_active:
            return self._forward_fast(x, index, num_segments)
        h = x.new_zeros((self.layers, num_segments, self.n_features))
        state = (h, h)
        q_star = x.new_zeros((num_segments, 2 * self.n_features))
        for _ in range(self.steps):
            q, state = self.lstm(q_star.unsqueeze(0), state)
            q = q.reshape(num_segments, self.n_features)
            a = segment_softmax((x * gather_rows(q, index)).sum(-1, keepdim=True), index, num_segments)
            q_star = torch.cat([q, scatter_sum(a * x, index, num_segments)], dim=-1)
        return q_star

    @torch.jit.unused
    def _forward_fast(self, x: Tensor, index: Tensor, num_segments: int) -> Tensor:
        """All steps as one stage of :mod:`xnn.gnn.fast.megnet`."""
        from xnn.gnn.fast.megnet import stage
        params = []
        for i in range(self.layers):
            params += [getattr(self.lstm, f"{name}_l{i}") for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
        fn = stage("set2set", (self.steps, self.layers), None, 1 + len(params), 2, x, _grad(x, *params))
        return fn(x, *params, index, x.new_empty((num_segments, 1)))


class _Block(nn.Module, FastPathModule):
    """One MEGNet block: optional dense layers, the edge -> node -> state update, a residual."""

    def __init__(self, width: int, dense: Optional[Sequence[int]], conv: Sequence[int],
                 activation: str, dropout: float):
        super().__init__()
        conv = [int(c) for c in conv]
        if dense:
            sizes = [width, *[int(d) for d in dense]]
            self.edge_dense = MLP(sizes, activation, activate_last=True)
            self.node_dense = MLP(sizes, activation, activate_last=True)
            self.state_dense = MLP(sizes, activation, activate_last=True)
            inner = sizes[-1]
        else:
            self.edge_dense = self.node_dense = self.state_dense = None
            inner = width
        out = conv[-1]
        self.edge_mlp = MLP([4 * inner, *conv], activation, activate_last=True)
        self.node_mlp = MLP([out + 2 * inner, *conv], activation, activate_last=True)
        self.state_mlp = MLP([2 * out + inner, *conv], activation, activate_last=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fast_active = False

    def _mlps(self) -> list:
        dense = [] if self.edge_dense is None else [self.edge_dense, self.node_dense, self.state_dense]
        return dense + [self.edge_mlp, self.node_mlp, self.state_mlp]

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """The fast stage reads plain dense layers (not wrapped, e.g. by LoRA)."""
        from xnn.gnn.fast.megnet import plain_mlp
        return dtype in (torch.float32, torch.float64) and all(plain_mlp(m) for m in self._mlps())

    def forward(self, e: Tensor, x: Tensor, u: Tensor, dst: Tensor, src: Tensor, batch: Tensor,
                edge_batch: Tensor, bonds_per_atom: Tensor, bonds_per_structure: Tensor,
                atoms_per_structure: Tensor):
        """One block; in the fast path the three counts are clamped float columns ``(n, 1)``."""
        if self.fast_active:
            return self._forward_fast(e, x, u, dst, src, batch, edge_batch, bonds_per_atom,
                                      bonds_per_structure, atoms_per_structure)
        e0, x0, u0 = e, x, u
        if self.edge_dense is not None:
            e, x, u = self.edge_dense(e), self.node_dense(x), self.state_dense(u)
        # the receiving atom comes first: its bonds are the ones it averages
        e = self.edge_mlp(torch.cat([gather_rows(x, dst), gather_rows(x, src), e, gather_rows(u, edge_batch)],
                                    dim=-1))
        x = self.node_mlp(torch.cat([x, _mean(e, dst, bonds_per_atom), gather_rows(u, batch)], dim=-1))
        u = self.state_mlp(torch.cat([u, _mean(e, edge_batch, bonds_per_structure),
                                      _mean(x, batch, atoms_per_structure)], dim=-1))
        if self.dropout is not None:
            e, x, u = self.dropout(e), self.dropout(x), self.dropout(u)
        return e + e0, x + x0, u + u0

    @torch.jit.unused
    def _forward_fast(self, e: Tensor, x: Tensor, u: Tensor, dst: Tensor, src: Tensor, batch: Tensor,
                      edge_batch: Tensor, per_atom: Tensor, per_bonds: Tensor, per_atoms: Tensor):
        """The block as one stage of :mod:`xnn.gnn.fast.megnet` (dropout, when on, outside it)."""
        from xnn.gnn.fast.megnet import linear_params, stage
        params = linear_params(self._mlps())
        n_dense = 0 if self.edge_dense is None else len(self.edge_dense.layers)
        drop = self.dropout is not None and self.training
        fn = stage("block", (n_dense, len(self.edge_mlp.layers), not drop), self.edge_mlp.act,
                   3 + len(params), 7, e, _grad(e, x, u, *params))
        out = fn(e, x, u, *params, dst, src, batch, edge_batch, per_atom, per_bonds, per_atoms)
        if not drop:
            return out
        return tuple(self.dropout(t) + t0 for t, t0 in zip(out, (e, x, u)))


@register_model("megnet")
class MEGNet(ExactFloat64Constants, InteratomicPotential, FastPathModule):
    """MEGNet property model with a global state; multi-fidelity with ``n_state_types``.

    See the module docstring for the equations and the defaults.

    Parameters
    ----------
    n_features : int, optional
        Width of the element embeddings, by default 16.
    n_interactions : int, optional
        Number of MEGNet blocks, by default 3.
    n_rbf : int, optional
        Number of Gaussians expanding the bond lengths, by default 100.
    cutoff : float, optional
        Bond cutoff in Angstrom, by default 4.0 (the multi-fidelity paper: 5.0).
    rbf_max : float or None, optional
        Position of the last Gaussian; ``None`` (default) is ``cutoff + 1``.
    gamma : float, optional
        Exponent of the Gaussians, ``exp(-gamma (r - mu)^2)``, by default 0.5
        (the paper: 4.0).
    state_dim : int, optional
        Width of the state: the number of state features, or of the state
        embedding with ``n_state_types``. By default 2.
    n_state_types : int or None, optional
        Number of integer state labels (fidelities). ``None`` (default) reads
        float state features.
    state_names : sequence of str or None, optional
        Names of the state labels in order (e.g. ``["PBE", "GLLB-SC", "HSE",
        "SCAN"]``), for :meth:`state_label`.
    encoder_sizes : sequence of int, optional
        Dense layers before the first block, and at the start of every later
        block, by default ``(64, 32)``.
    conv_sizes : sequence of int, optional
        Layers of the update networks ``phi_e``, ``phi_v``, ``phi_u``, by
        default ``(64, 64, 32)``. The last size equals the last encoder size.
    readout_sizes : sequence of int, optional
        Hidden layers of the output network, by default ``(32, 16)``.
    set2set_steps : int, optional
        Attention steps of the Set2Set readouts, by default 2.
    set2set_layers : int, optional
        LSTM layers of the Set2Set readouts, by default 1.
    activation : str, optional
        Activation of every dense layer, by default ``"ssp"``.
    dropout : float, optional
        Dropout after every block and before the output network, by default 0.
    n_outputs : int, optional
        Number of properties per structure, by default 1.
    classification : bool, optional
        Pass the output through a sigmoid (a metal / nonmetal classifier),
        by default ``False``.
    n_elements : int, optional
        Rows of the element embedding, indexed by atomic number, by default 95
        (row 0 unused, as in the paper).
    property_scale, property_shift : float, optional
        Standardization of the output, by default 1 and 0.
    weight_init : str, optional
        Initialization of the dense layers (a key of
        :data:`~xnn.common.models.ops.WEIGHT_INITS`), by default ``"torch"``.
    """

    energy_output = False

    def __init__(self, n_features: int = 16, n_interactions: int = 3, n_rbf: int = 100,
                 cutoff: float = 4.0, rbf_max: Optional[float] = None, gamma: float = 0.5,
                 state_dim: int = 2, n_state_types: Optional[int] = None,
                 state_names: Optional[Sequence[str]] = None,
                 encoder_sizes: Sequence[int] = (64, 32), conv_sizes: Sequence[int] = (64, 64, 32),
                 readout_sizes: Sequence[int] = (32, 16), set2set_steps: int = 2,
                 set2set_layers: int = 1, activation: str = "ssp", dropout: float = 0.0,
                 n_outputs: int = 1, classification: bool = False, n_elements: int = 95,
                 property_scale: float = 1.0, property_shift: float = 0.0,
                 weight_init: str = "torch"):
        super().__init__()
        encoder_sizes = [int(s) for s in encoder_sizes]
        conv_sizes = [int(s) for s in conv_sizes]
        if encoder_sizes[-1] != conv_sizes[-1]:
            raise ValueError(f"the residual of the first block needs the last encoder size "
                             f"({encoder_sizes[-1]}) to equal the last conv size ({conv_sizes[-1]})")
        if n_state_types is not None and state_names is not None and len(state_names) != n_state_types:
            raise ValueError(f"got {len(state_names)} state names for {n_state_types} state types")
        self.cutoff = float(cutoff)
        self.n_state_types = None if n_state_types is None else int(n_state_types)
        self.state_dim = int(state_dim)
        self.state_names = None if state_names is None else [str(s) for s in state_names]
        self.n_outputs = int(n_outputs)
        self.classification = bool(classification)

        self.rbf = make_radial_basis("gaussian", int(n_rbf),
                                     self.cutoff + 1.0 if rbf_max is None else float(rbf_max),
                                     gamma=float(gamma))
        self.node_embedding = nn.Embedding(int(n_elements), int(n_features))
        self.state_embedding = (nn.Embedding(self.n_state_types, self.state_dim)
                                if self.n_state_types is not None else None)
        self.edge_encoder = MLP([int(n_rbf), *encoder_sizes], activation, activate_last=True)
        self.node_encoder = MLP([int(n_features), *encoder_sizes], activation, activate_last=True)
        self.state_encoder = MLP([self.state_dim, *encoder_sizes], activation, activate_last=True)
        width = conv_sizes[-1]
        self.blocks = nn.ModuleList(
            _Block(width, None if k == 0 else encoder_sizes, conv_sizes, activation, float(dropout))
            for k in range(int(n_interactions)))
        self.node_set2set = Set2Set(width, set2set_steps, set2set_layers)
        self.edge_set2set = Set2Set(width, set2set_steps, set2set_layers)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else None
        self.readout = MLP([5 * width, *[int(s) for s in readout_sizes], self.n_outputs], activation)
        self.register_buffer("property_scale", torch.tensor(float(property_scale)))
        self.register_buffer("property_shift", torch.tensor(float(property_shift)))
        init_linears_(self, weight_init)
        # the encoders' fast stage (this model is a fast-path module too), and the choice of
        # implementation (see set_use_fast)
        self.fast_active = False
        self.use_fast = "auto"

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """The encoders' fast stage reads the Gaussian basis and plain dense layers."""
        from xnn.common.featurizers.radial import GaussianRBF
        from xnn.gnn.fast.megnet import plain_mlp
        return (dtype in (torch.float32, torch.float64) and isinstance(self.rbf, GaussianRBF)
                and all(plain_mlp(m) for m in (self.edge_encoder, self.node_encoder, self.state_encoder)))

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "MEGNet":
        """Choose the implementation of the encoders, blocks and readouts.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the compiled stages of
            :mod:`xnn.gnn.fast.megnet` on a GPU above the size crossover measured
            for evaluation (``AUTO_POLICY``) or training (``AUTO_POLICY_TRAINING``);
            ``True``: always (as written on a CPU); ``False``: the reference.
            The stages agree with the reference to about 1e-6 relative in float32
            and 1e-14 in float64, with the same first and second derivatives.

        Returns
        -------
        MEGNet
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> bool:
        """Switch the fast stages on or off for one evaluation; whether the blocks take them."""
        from xnn.common.models.fast import fast_modules, select
        from xnn.gnn.fast.megnet import AUTO_POLICY, AUTO_POLICY_TRAINING
        modules = self.__dict__.get("_fast_blocks")
        if modules is None:
            modules = self.__dict__["_fast_blocks"] = fast_modules(self)
        like = self.node_embedding.weight
        select(modules, self.use_fast, AUTO_POLICY_TRAINING if self.training else AUTO_POLICY,
               like.device, like.dtype, num_edges)
        # the blocks share the count columns of the fast stage: all of them or none
        fast = bool(self.blocks) and all(b.fast_active for b in self.blocks)
        for b in self.blocks:
            b.fast_active = fast
        return fast

    @torch.jit.unused
    def _encode_fast(self, dist: Tensor, x: Tensor, u: Tensor):
        """The Gaussian expansion and the three encoders as one stage of :mod:`xnn.gnn.fast.megnet`."""
        from xnn.gnn.fast.megnet import linear_params, stage
        mlps = (self.edge_encoder, self.node_encoder, self.state_encoder)
        params = linear_params(mlps)
        fn = stage("encode", tuple(len(m.layers) for m in mlps), self.edge_encoder.act, 3 + len(params), 2,
                   dist, _grad(dist, x, u, *params))
        return fn(dist, x, u, *params, self.rbf.centers.to(dist.dtype), float(self.rbf.gamma))

    def state_label(self, name: str) -> int:
        """The integer state label of a named state (a fidelity such as ``"HSE"``).

        Raises
        ------
        ValueError
            If the model has no state names or does not know ``name``.
        """
        if self.state_names is None or name not in self.state_names:
            raise ValueError(f"unknown state {name!r}; the model knows {self.state_names}")
        return self.state_names.index(name)

    def set_property_scale_shift(self, scale: float, shift: float) -> None:
        """Set the output standardization ``scale * y + shift`` (e.g. the training-set std and mean)."""
        self.property_scale.fill_(float(scale))
        self.property_shift.fill_(float(shift))

    def _sites(self, data: AtomicGraph) -> Tensor:
        """The element embedding of every site, ``(N, F)``: of its species, or the occupancy-weighted sum of its
        species' embeddings for a disordered site."""
        if data.site_species is None:
            return self.node_embedding(data.atomic_numbers)
        weights = data.site_occupancy.to(self.node_embedding.weight.dtype)
        return (self.node_embedding(data.site_species) * weights.unsqueeze(-1)).sum(1)

    def _state(self, data: AtomicGraph, dtype: torch.dtype) -> Tensor:
        """The state vector of every structure, ``(B, state_dim)``."""
        b, device = data.num_graphs, data.pos.device
        if self.state_embedding is not None:
            labels = (torch.zeros(b, dtype=torch.long, device=device) if data.state is None
                      else data.state.reshape(b).long())
            return self.state_embedding(labels)
        if data.state is None:
            return torch.zeros((b, self.state_dim), dtype=dtype, device=device)
        return data.state.reshape(b, self.state_dim).to(dtype)

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Predict the property of every structure.

        Parameters
        ----------
        data : AtomicGraph
            The batch; ``data.state`` holds the state labels or features
            (``None``: label 0 or zero features).

        Returns
        -------
        dict[str, Tensor]
            ``"properties"``, shape ``(B,)`` (or ``(B, n_outputs)``), and
            ``"structure_features"``, the readout ``(B, 5 F)`` the output
            network maps (the latent features of the multi-fidelity paper).
        """
        src, dst = data.edge_index[0], data.edge_index[1]
        fast_blocks = self._select_fast(int(dst.shape[0]))
        if fast_blocks:
            # rows of their own: the compiled stages guard on storage offsets, and the second row's is the edge count
            src, dst = src.clone(), dst.clone()
        b, n = data.num_graphs, data.num_nodes
        batch = data.batch
        edge_batch = batch.index_select(0, dst)
        dist = data.edge_vectors().norm(dim=-1)
        dtype = dist.dtype
        counts = (torch.bincount(dst, minlength=n), torch.bincount(edge_batch, minlength=b), data.n_atoms)
        if fast_blocks:
            counts = tuple(c.clamp(min=1).unsqueeze(-1).to(dtype) for c in counts)
        bonds_per_atom, bonds_per_structure, atoms_per_structure = counts

        x, u = self._sites(data), self._state(data, dtype)
        if self.fast_active:
            e, x, u = self._encode_fast(dist, x, u)
        else:
            e, x, u = self.edge_encoder(self.rbf(dist)), self.node_encoder(x), self.state_encoder(u)
        for block in self.blocks:
            e, x, u = block(e, x, u, dst, src, batch, edge_batch, bonds_per_atom,
                            bonds_per_structure, atoms_per_structure)
        features = torch.cat([self.node_set2set(x, batch, b), self.edge_set2set(e, edge_batch, b), u],
                             dim=-1)
        y = self.readout(features if self.dropout is None else self.dropout(features))
        if self.classification:
            y = torch.sigmoid(y)
        y = self.property_scale * y + self.property_shift
        return {"properties": y.squeeze(-1) if self.n_outputs == 1 else y,
                "structure_features": features}

    @classmethod
    def from_foundation(cls, name: str = "megnet-mp-2018.6.1-eform", dtype=None,
                        cache_dir=None, quiet: bool = False) -> "MEGNet":
        """Load a pretrained MatGL MEGNet model (formation energy or multi-fidelity band gap).

        See :mod:`xnn.gnn.models.megnet_foundation`.

        Parameters
        ----------
        name : str, optional
            ``"megnet-mp-2018.6.1-eform"`` (default, eV/atom) or
            ``"megnet-mp-2019.4.1-bandgap-mfi"`` (eV; fidelities PBE, GLLB-SC,
            HSE and SCAN).
        dtype : torch.dtype or str, optional
            Final dtype; ``None`` keeps the checkpoint's float32.
        cache_dir : str or pathlib.Path, optional
            Model cache directory.
        quiet : bool, optional
            Hide the download progress bars. Default ``False``.

        Returns
        -------
        MEGNet
            The converted model, in evaluation mode.
        """
        from .megnet_foundation import from_foundation
        return from_foundation(name, dtype=dtype, cache_dir=cache_dir, quiet=quiet)

    @classmethod
    def from_config(cls, cfg) -> "MEGNet":
        """Build a :class:`MEGNet` from a configuration object.

        Core fields: ``cfg.n_features`` (element embedding width),
        ``cfg.n_interactions`` (blocks), ``cfg.n_rbf`` and ``cfg.cutoff``.
        Everything else is read from ``cfg.extra`` (``rbf_max``, ``gamma``,
        ``state_dim``, ``n_state_types``, ``state_names``, ``encoder_sizes``,
        ``conv_sizes``, ``readout_sizes``, ``set2set_steps``,
        ``set2set_layers``, ``activation``, ``dropout``, ``n_outputs``,
        ``classification``, ``n_elements``, ``property_scale``,
        ``property_shift``, ``weight_init``); MatGL's spellings are
        translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        MEGNet
            Instantiated model.
        """
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        rbf_max = extra.get("rbf_max")
        n_state_types = extra.get("n_state_types")
        state_names = extra.get("state_names")
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            rbf_max=None if rbf_max is None else float(rbf_max),
            gamma=float(extra.get("gamma", 0.5)),
            state_dim=int(extra.get("state_dim", 2)),
            n_state_types=None if n_state_types is None else int(n_state_types),
            state_names=None if state_names is None else list(state_names),
            encoder_sizes=list(extra.get("encoder_sizes", (64, 32))),
            conv_sizes=list(extra.get("conv_sizes", (64, 64, 32))),
            readout_sizes=list(extra.get("readout_sizes", (32, 16))),
            set2set_steps=int(extra.get("set2set_steps", 2)),
            set2set_layers=int(extra.get("set2set_layers", 1)),
            activation=str(extra.get("activation", "ssp")),
            dropout=float(extra.get("dropout", 0.0) or 0.0),
            n_outputs=int(extra.get("n_outputs", 1)),
            classification=bool(extra.get("classification", False)),
            n_elements=int(extra.get("n_elements", 95)),
            property_scale=float(extra.get("property_scale", 1.0)),
            property_shift=float(extra.get("property_shift", 0.0)),
            weight_init=str(extra.get("weight_init", "torch")),
        )
