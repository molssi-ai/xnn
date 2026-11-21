"""CGCNN: crystal graph convolutional neural networks (Xie and Grossman, Phys. Rev. Lett. 120, 145301 (2018)).

A CGCNN predicts one property per crystal (a formation energy per atom, a band
gap, a modulus) from a graph that connects every atom ``i`` to its nearest
neighbors ``j``. Each atom starts from a fixed feature vector of its element,
mapped by a dense layer to ``v_i``; each bond carries a Gaussian expansion
``u_ij`` of its length. Every convolution concatenates
``z_ij = v_i + v_j + u_ij`` and updates (eq. 5 of the paper)

``v_i' = g(v_i + BN(sum_j sigmoid(BN(z_ij W_f + b_f)) * g(BN(z_ij W_s + b_s))))``,

with ``g`` the softplus and ``BN`` batch normalization (over the bonds inside
the sum, over the atoms outside it). The mean of the ``v_i`` of a crystal
passes through the softplus and dense layers with softplus activations to the
output.

The defaults are those of the authors' code: the 12 nearest neighbors within
8 Angstrom, 41 Gaussians ``exp(-(r - mu)^2 / 0.2^2)`` on ``[0, 8]``, 64 atom
features, three convolutions and one hidden layer of 128. The atom features are
the authors' 92-column element table, downloaded on first use. The paper's
eq. 5 has neither the batch normalizations nor the outer softplus:
``normalization=None`` and ``residual_activation=False``.

The output is ``"properties"``, one value (or ``n_outputs`` values) per
crystal, standardized as ``property_scale * y + property_shift``. With
``classification=True`` the output layer gives two classes and
``"properties"`` is the probability of the second (a metal, for the authors'
metal / semiconductor model). Train it with ``optim.property_weight`` on
``properties`` labels. The model predicts no energy, so it gives no forces or
stress.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xnn.common.data import AtomicGraph
from xnn.common.data.neighborlist import nearest_edges
from xnn.common.featurizers.radial import GaussianRBF
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.constants import ExactFloat64Constants, register_constant
from xnn.common.models.fast import FastPathModule
from xnn.common.models.ops import exact_softplus, gather_rows, init_linears_, scatter_sum
from xnn.common.models.registry import register_model

#: Rows (atomic numbers 0 to 100) and columns of the authors' element feature table.
CGCNN_TABLE_SHAPE = (101, 92)


def read_atom_features(spec, n_elements: int = CGCNN_TABLE_SHAPE[0]) -> Tensor:
    """An element feature table ``(rows, features)`` whose row ``Z`` belongs to atomic number ``Z``.

    Parameters
    ----------
    spec : str, path, mapping or array
        ``"onehot"`` (the identity of ``n_elements`` rows, a learned embedding
        per element), a JSON file in the authors' ``atom_init.json`` format,
        a mapping of atomic numbers to vectors, or an array of rows.
    n_elements : int, optional
        Rows of the ``"onehot"`` table, by default 101.

    Returns
    -------
    Tensor
        The table in float64; elements a file or mapping leaves out have zeros.
    """
    if isinstance(spec, str) and spec == "onehot":
        return torch.eye(int(n_elements), dtype=torch.float64)
    if isinstance(spec, (str, Path)):
        with open(spec) as f:
            spec = json.load(f)
    if isinstance(spec, dict):
        rows = {int(z): torch.as_tensor(v, dtype=torch.float64).reshape(-1) for z, v in spec.items()}
        widths = {v.numel() for v in rows.values()}
        if len(widths) != 1:
            raise ValueError(f"element feature vectors of different lengths: {sorted(widths)}")
        table = torch.zeros(max(rows) + 1, widths.pop(), dtype=torch.float64)
        for z, v in rows.items():
            table[z] = v
        return table
    table = torch.as_tensor(spec, dtype=torch.float64)
    if table.dim() != 2:
        raise ValueError(f"an element feature table has two dimensions, got shape {tuple(table.shape)}")
    return table


class ConvLayer(nn.Module):
    """One CGCNN convolution (eq. 5 of the paper, with the authors' batch normalizations).

    Parameters
    ----------
    n_features : int
        Width of the atom features.
    n_rbf : int
        Width of the bond features.
    normalization : str or None, optional
        ``"batch"`` (default): batch normalization of the gates and of the
        neighbor sums; ``None``: none.
    residual_activation : bool, optional
        Pass the residual sum through the softplus, by default ``True``.
    """

    def __init__(self, n_features: int, n_rbf: int, normalization: Optional[str] = "batch",
                 residual_activation: bool = True):
        super().__init__()
        f = int(n_features)
        self.fc_full = nn.Linear(2 * f + int(n_rbf), 2 * f)
        self.bn1 = nn.BatchNorm1d(2 * f) if normalization == "batch" else None
        self.bn2 = nn.BatchNorm1d(f) if normalization == "batch" else None
        self.residual_activation = bool(residual_activation)

    def forward(self, x: Tensor, u: Tensor, src: Tensor, dst: Tensor, per_atom: Optional[int] = None) -> Tensor:
        """Update the atom features ``x`` ``(N, F)`` from the bond features ``u`` ``(E, R)`` of the edges
        ``src -> dst``; with ``per_atom``, the edges are grouped by ``dst``, ``per_atom`` for every atom."""
        n, f = x.shape
        if per_atom is not None:
            # a table of per_atom neighbors per atom: the atom's own features broadcast, the sum a dense reduction
            m = per_atom
            if x.is_cuda or type(self.fc_full) is not nn.Linear:
                z = torch.cat([x.unsqueeze(1).expand(n, m, f), gather_rows(x, src).view(n, m, f), u.view(n, m, -1)],
                              dim=-1)
                h = self.fc_full(z).view(n * m, 2 * f)
            else:
                # on a CPU the layer split by input is faster: the atom's own and its neighbors' parts once per atom,
                # only the bond part per edge
                w = self.fc_full.weight
                atoms = F.linear(x, torch.cat([w[:, :f], w[:, f:2 * f]]))
                h = (F.linear(u, w[:, 2 * f:], self.fc_full.bias).view(n, m, 2 * f) + atoms[:, :2 * f].unsqueeze(1)
                     + gather_rows(atoms[:, 2 * f:], src).view(n, m, 2 * f)).view(n * m, 2 * f)
        else:
            h = self.fc_full(torch.cat([gather_rows(x, dst), gather_rows(x, src), u], dim=-1))
        if self.bn1 is not None:
            h = self.bn1(h)
        gate, core = h.chunk(2, dim=-1)
        message = torch.sigmoid(gate) * exact_softplus(core)
        s = message.view(n, per_atom, f).sum(1) if per_atom is not None else scatter_sum(message, dst, n)
        if self.bn2 is not None:
            s = self.bn2(s)
        out = x + s
        return exact_softplus(out) if self.residual_activation else out


@register_model("cgcnn")
class CGCNN(ExactFloat64Constants, InteratomicPotential, FastPathModule):
    """CGCNN property model of crystals.

    See the module docstring for the equations and the defaults.

    Parameters
    ----------
    n_features : int, optional
        Width of the atom features, by default 64.
    n_interactions : int, optional
        Number of convolutions, by default 3.
    n_rbf : int, optional
        Number of Gaussians expanding the bond lengths, by default 41.
    cutoff : float, optional
        Neighbor search radius in Angstrom, by default 8.0.
    max_neighbors : int or None, optional
        Nearest neighbors kept per atom, by default 12; ``None`` keeps every
        neighbor within the cutoff.
    rbf_start : float, optional
        Position of the first Gaussian, by default 0.
    rbf_max : float or None, optional
        Position of the last Gaussian; ``None`` (default) is the cutoff.
    gamma : float or None, optional
        Exponent of the Gaussians, ``exp(-gamma (r - mu)^2)``; ``None``
        (default) is the inverse square of the center spacing (25 for the
        default 0.2 Angstrom grid).
    readout_sizes : sequence of int, optional
        Hidden layers after the pooling, by default ``(128,)``.
    normalization : str or None, optional
        ``"batch"`` (default) or ``None`` (the paper's eq. 5).
    residual_activation : bool, optional
        Softplus after every residual sum, by default ``True`` (the paper:
        ``False``).
    n_outputs : int, optional
        Number of properties per crystal, by default 1.
    classification : bool, optional
        Two-class output: ``"properties"`` is the probability of the second
        class and ``"log_probabilities"`` holds both. Default ``False``.
    dropout : float or None, optional
        Dropout after the first hidden layer; ``None`` (default) is 0.5 for a
        classifier and 0 otherwise.
    atom_features : str, path, mapping or array, optional
        The fixed element features: ``"cgcnn"`` (default, the authors' table,
        downloaded on first use), ``"onehot"``, a file in the authors'
        ``atom_init.json`` format, a mapping of atomic numbers to vectors or
        an array of rows (see :func:`read_atom_features`).
    n_elements : int, optional
        Rows of the ``"onehot"`` table, by default 101.
    property_scale, property_shift : float, optional
        Standardization of the output, by default 1 and 0.
    weight_init : str, optional
        Initialization of the dense layers (a key of
        :data:`~xnn.common.models.ops.WEIGHT_INITS`), by default ``"torch"``.
    """

    energy_output = False

    def __init__(self, n_features: int = 64, n_interactions: int = 3, n_rbf: int = 41,
                 cutoff: float = 8.0, max_neighbors: Optional[int] = 12, rbf_start: float = 0.0,
                 rbf_max: Optional[float] = None, gamma: Optional[float] = None,
                 readout_sizes: Sequence[int] = (128,), normalization: Optional[str] = "batch",
                 residual_activation: bool = True, n_outputs: int = 1, classification: bool = False,
                 dropout: Optional[float] = None, atom_features="cgcnn",
                 n_elements: int = CGCNN_TABLE_SHAPE[0], property_scale: float = 1.0,
                 property_shift: float = 0.0, weight_init: str = "torch"):
        super().__init__()
        readout_sizes = [int(s) for s in readout_sizes]
        if not readout_sizes:
            raise ValueError("CGCNN needs at least one hidden layer after the pooling")
        if normalization not in ("batch", None):
            raise ValueError(f"normalization must be 'batch' or None, got {normalization!r}")
        self.cutoff = float(cutoff)
        self.max_neighbors = None if max_neighbors is None else int(max_neighbors)
        self.classification = bool(classification)
        self.n_outputs = 2 if self.classification else int(n_outputs)

        rbf_max = self.cutoff if rbf_max is None else float(rbf_max)
        n_rbf = int(n_rbf)
        if gamma is None:
            gamma = 1.0 / ((rbf_max - float(rbf_start)) / (n_rbf - 1)) ** 2 if n_rbf > 1 else 1.0
        self.rbf = GaussianRBF(n_rbf, rbf_max, gamma=float(gamma), start=float(rbf_start))
        register_constant(self.rbf, "centers", torch.linspace(float(rbf_start), rbf_max, n_rbf, dtype=torch.float64))

        # the authors' table is fetched when first needed, unless a checkpoint brings it
        self._atom_table_pending = isinstance(atom_features, str) and atom_features == "cgcnn"
        table = (torch.zeros(CGCNN_TABLE_SHAPE, dtype=torch.float64) if self._atom_table_pending
                 else read_atom_features(atom_features, n_elements))
        self.register_buffer("atom_features", table.to(torch.get_default_dtype()))
        f = int(n_features)
        self.embedding = nn.Linear(table.shape[1], f)
        self.convs = nn.ModuleList(ConvLayer(f, n_rbf, normalization, residual_activation)
                                   for _ in range(int(n_interactions)))
        self.conv_to_fc = nn.Linear(f, readout_sizes[0])
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(readout_sizes[:-1], readout_sizes[1:]))
        self.fc_out = nn.Linear(readout_sizes[-1], self.n_outputs)
        p = (0.5 if self.classification else 0.0) if dropout is None else float(dropout)
        self.dropout = nn.Dropout(p) if p > 0 else None
        self.register_buffer("property_scale", torch.tensor(float(property_scale)))
        self.register_buffer("property_shift", torch.tensor(float(property_shift)))
        init_linears_(self, weight_init)
        # the choice of implementation (see set_use_fast); the model is the fast-path module of its stage
        self.use_fast = "auto"
        self.fast_active = False

    def fast_supported(self, device: torch.device, dtype: torch.dtype) -> bool:
        """The fast stage reads plain dense layers and plain batch normalizations with running statistics, the
        same in every convolution."""
        layers = [self.embedding] + [c.fc_full for c in self.convs]
        norms = [m for c in self.convs for m in (c.bn1, c.bn2) if m is not None]
        return (dtype in (torch.float32, torch.float64) and len(self.convs) > 0
                and all(type(m) is nn.Linear and m.bias is not None for m in layers)
                and len(norms) in (0, 2 * len(self.convs))
                and len({c.residual_activation for c in self.convs}) == 1
                and all(type(m) is nn.BatchNorm1d and m.affine and m.track_running_stats and m.momentum is not None
                        and m.eps == norms[0].eps and m.training == norms[0].training for m in norms))

    @torch.jit.unused
    def _forward_fast(self, data: AtomicGraph, src: Tensor, keep: Optional[Tensor]) -> Tensor:
        """The network up to the pooling as the stage of :mod:`xnn.gnn.fast.cgcnn`, on edges grouped by atom, the
        same number for every atom; returns the pooled atom features."""
        from xnn.gnn.fast.cgcnn import mark_sizes, stage, update_running_statistics
        n = data.num_nodes
        shifts = data.cell_shifts if keep is None else data.cell_shifts.index_select(0, keep)
        # aliases of the graph's tensors for the stage: the size marks of the compiler stay on them
        pos, src, shifts, z, batch = (t.view_as(t) for t in (data.pos, src, shifts, data.atomic_numbers, data.batch))
        if src.is_cuda:
            mark_sizes([src, shifts], [pos, z, batch])
        norm = self.convs[0].bn1 is not None
        training = norm and self.convs[0].bn1.training
        # the positions are differentiated only for a gradient by them
        pos_input = torch.is_grad_enabled() and pos.requires_grad
        inputs = ([pos] if pos_input else []) + [self.embedding.weight, self.embedding.bias]
        for conv in self.convs:
            inputs += [conv.fc_full.weight, conv.fc_full.bias]
            if norm:
                inputs += [conv.bn1.weight, conv.bn1.bias, conv.bn2.weight, conv.bn2.bias]
        periodic = data.cell is not None
        consts = ([] if pos_input else [pos]) + [
            data.cell if periodic else pos.new_zeros((1, 3, 3)), shifts, src, self.atom_features, z,
            self.rbf.centers, batch, data.n_atoms]
        if norm and not training:
            for conv in self.convs:
                consts += [conv.bn1.running_mean, conv.bn1.running_var, conv.bn2.running_mean, conv.bn2.running_var]
        key = (len(self.convs), norm, training, self.convs[0].residual_activation, float(self.rbf.gamma),
               float(self.convs[0].bn1.eps) if norm else 0.0, pos_input, periodic, data.num_graphs == 1 and n > 0)
        grad = torch.is_grad_enabled() and any(t.requires_grad for t in inputs)
        out = stage(key, src.shape[0], len(inputs), len(consts), self.embedding.weight, grad)(*inputs, *consts)
        if not training:
            return out
        pooled, stats = out[0], out[1:]
        for k, conv in enumerate(self.convs):
            update_running_statistics(conv.bn1, stats[4 * k].detach(), stats[4 * k + 1].detach(), src.shape[0])
            update_running_statistics(conv.bn2, stats[4 * k + 2].detach(), stats[4 * k + 3].detach(), n)
        return pooled

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "CGCNN":
        """Choose the implementation of the network up to the pooling.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the compiled stage of :mod:`xnn.gnn.fast.cgcnn` on a GPU above
            the size crossover measured for evaluation (``AUTO_POLICY``) or training
            (``AUTO_POLICY_TRAINING``); ``True``: always (as written on a CPU); ``False``: the
            reference. The stage serves graphs whose atoms all keep the same number of neighbors
            (crystals) and agree with the reference to about 1e-6 relative in float32 and 1e-14 in
            float64, with the same first and second derivatives.

        Returns
        -------
        CGCNN
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> bool:
        """Switch the fast stage on or off for one evaluation; whether it is on."""
        from xnn.common.models.fast import select
        from xnn.gnn.fast.cgcnn import AUTO_POLICY, AUTO_POLICY_TRAINING
        like = self.embedding.weight
        return select([self], self.use_fast, AUTO_POLICY_TRAINING if self.training else AUTO_POLICY,
                      like.device, like.dtype, num_edges)

    def _ensure_atom_table(self) -> None:
        """Fill the authors' element table the first time it is needed."""
        if self._atom_table_pending:
            from .cgcnn_foundation import atom_init_table
            with torch.no_grad():
                self.atom_features.copy_(atom_init_table().to(self.atom_features))
            self._atom_table_pending = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        if prefix + "atom_features" in state_dict:
            self._atom_table_pending = False
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._ensure_atom_table()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def set_property_scale_shift(self, scale: float, shift: float) -> None:
        """Set the output standardization ``scale * y + shift`` (e.g. the training-set std and mean)."""
        self.property_scale.fill_(float(scale))
        self.property_shift.fill_(float(shift))

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Predict the property of every crystal.

        Parameters
        ----------
        data : AtomicGraph
            The batch, with a neighbor list of at least ``cutoff``.

        Returns
        -------
        dict[str, Tensor]
            ``"properties"``, shape ``(B,)`` (or ``(B, n_outputs)``),
            ``"structure_features"``, the pooled atom features ``(B, F)``, and
            for a classifier ``"log_probabilities"`` ``(B, 2)``.
        """
        self._ensure_atom_table()
        src, dst = data.edge_index[0], data.edge_index[1]
        n, b = data.num_nodes, data.num_graphs
        lengths: List[Tensor] = []

        def bond_lengths() -> Tensor:
            # formed when the selection or the reference needs them, once
            if not lengths:
                lengths.append(data.edge_vectors().norm(dim=-1))
            return lengths[0]

        keep, per_atom = ((None, None) if self.max_neighbors is None
                          else nearest_edges(bond_lengths, dst, n, self.max_neighbors))
        if keep is not None:
            src, dst = src.index_select(0, keep), dst.index_select(0, keep)
        fast = (per_atom is not None and not (data.cell is not None and data.cell.requires_grad)
                and self._select_fast(int(dst.shape[0])))
        if fast:
            pooled = self._forward_fast(data, src, keep)
        else:
            dist = bond_lengths() if keep is None else bond_lengths().index_select(0, keep)
            u = self.rbf(dist)
            x = self.embedding(self.atom_features.index_select(0, data.atomic_numbers))
            for conv in self.convs:
                x = conv(x, u, src, dst, per_atom)
            if b == 1 and n > 0:
                pooled = x.mean(0, keepdim=True)
            else:
                pooled = scatter_sum(x, data.batch, b) / data.n_atoms.clamp(min=1).unsqueeze(-1).to(x.dtype)
        h = exact_softplus(self.conv_to_fc(exact_softplus(pooled)))
        if self.dropout is not None:
            h = self.dropout(h)
        for fc in self.fcs:
            h = exact_softplus(fc(h))
        y = self.fc_out(h)
        out = {"structure_features": pooled}
        if self.classification:
            log_p = torch.log_softmax(y, dim=-1)
            out["log_probabilities"] = log_p
            out["properties"] = log_p[:, 1].exp()
            return out
        y = torch.addcmul(self.property_shift, self.property_scale, y)
        out["properties"] = y.squeeze(-1) if self.n_outputs == 1 else y
        return out

    @classmethod
    def from_foundation(cls, name: str = "cgcnn-formation-energy-per-atom", dtype=None,
                        cache_dir: Optional[Union[str, Path]] = None, quiet: bool = False) -> "CGCNN":
        """Load one of the authors' pretrained models (see :mod:`xnn.gnn.models.cgcnn_foundation`).

        Parameters
        ----------
        name : str, optional
            A key of :data:`~xnn.gnn.models.cgcnn_foundation.FOUNDATION_MODELS`,
            by default the formation energy per atom (eV/atom).
        dtype : torch.dtype or str, optional
            Final dtype; ``None`` keeps the checkpoint's float32.
        cache_dir : str or pathlib.Path, optional
            Model cache directory.
        quiet : bool, optional
            Hide the download progress bars. Default ``False``.

        Returns
        -------
        CGCNN
            The converted model, in evaluation mode.
        """
        from .cgcnn_foundation import from_foundation
        return from_foundation(name, dtype=dtype, cache_dir=cache_dir, quiet=quiet)

    @classmethod
    def from_config(cls, cfg) -> "CGCNN":
        """Build a :class:`CGCNN` from a configuration object.

        Core fields: ``cfg.n_features`` (atom feature width),
        ``cfg.n_interactions`` (convolutions), ``cfg.n_rbf`` and ``cfg.cutoff``.
        Everything else is read from ``cfg.extra`` (``max_neighbors``,
        ``rbf_start``, ``rbf_max``, ``gamma``, ``readout_sizes``,
        ``normalization``, ``residual_activation``, ``n_outputs``,
        ``classification``, ``dropout``, ``atom_features``, ``n_elements``,
        ``property_scale``, ``property_shift``, ``weight_init``); the authors'
        spellings are translated by :mod:`xnn.common.config.translate`.

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        CGCNN
            Instantiated model.
        """
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        max_neighbors = extra.get("max_neighbors", 12)
        rbf_max = extra.get("rbf_max")
        gamma = extra.get("gamma")
        dropout = extra.get("dropout")
        return cls(
            n_features=cfg.n_features,
            n_interactions=cfg.n_interactions,
            n_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            max_neighbors=None if max_neighbors is None else int(max_neighbors),
            rbf_start=float(extra.get("rbf_start", 0.0)),
            rbf_max=None if rbf_max is None else float(rbf_max),
            gamma=None if gamma is None else float(gamma),
            readout_sizes=list(extra.get("readout_sizes", (128,))),
            normalization=extra.get("normalization", "batch"),
            residual_activation=bool(extra.get("residual_activation", True)),
            n_outputs=int(extra.get("n_outputs", 1)),
            classification=bool(extra.get("classification", False)),
            dropout=None if dropout is None else float(dropout),
            atom_features=extra.get("atom_features", "cgcnn"),
            n_elements=int(extra.get("n_elements", CGCNN_TABLE_SHAPE[0])),
            property_scale=float(extra.get("property_scale", 1.0)),
            property_shift=float(extra.get("property_shift", 0.0)),
            weight_init=str(extra.get("weight_init", "torch")),
        )
