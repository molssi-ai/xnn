"""BAMBOO: a graph equivariant transformer force field (Gong et al. 2024).

A faithful, self-contained re-implementation of the BAMBOO model (ByteDance AI
Molecular Simulation Booster, arXiv:2404.07181) built on the xnns
abstractions. BAMBOO is a *hybrid* potential: a graph neural network whose
message-passing layers are transformers (the Graph Equivariant Transformer,
GET), followed by a physics-based split of the atomic energy into three pieces
(Supplementary A.2)::

    E_i = E_i^NN  +  E_i^elec  +  E_i^disp

* **semi-local** ``E^NN`` -- an MLP on the per-atom GET features;
* **electrostatic** ``E^elec`` -- a charge-equilibrium energy built from
  predicted partial charges (a per-atom electronegativity/hardness term plus a
  damped Coulomb sum over all pairs);
* **dispersion** ``E^disp`` -- an optional D3(CSO) correction (off by default,
  matching the paper, which excludes dispersion from the DFT training data and
  only adds it during MD).

Architecture (Supplementary A.1): the atom type ``Z`` is embedded to the scalar
node feature ``x_i`` while the vector node feature ``V_i`` starts at zero. Each
GET layer runs a multi-head QKV attention on the neighbour graph (the shared
:class:`~xnns.transformer.attention.EdgeMultiheadAttention`), scales the
neighbour values by a radial edge feature (from the
:class:`~xnns.transformer.featurizers.ExpNormalSmearing` basis) and the
attention weight, and mixes the scalar and vector channels through inner
products so both stay rotation-equivariant. Two MLPs read the final scalar
features into the per-atom energy and the partial charge.

The model subclasses :class:`~xnns.common.models.base.InteratomicPotential`
directly (like :class:`~xnns.dnn.models.physnet.PhysNet`, the other
physics-split potential in xnns): it needs no ``e3nn`` because equivariance
comes from Cartesian vector channels, not spherical harmonics. Forces and
stress are added uniformly by
:class:`~xnns.common.models.outputs.ForceStressOutput` via autograd -- the
paper's separately damped Coulomb *force* is, on inspection, exactly the
gradient of its Coulomb *energy* (the softplus energy damping differentiates to
the sigmoid force damping), so autograd reproduces it.

Given the same weights this matches the original ``bamboo`` package
(bytedance/bamboo) to machine precision -- see ``tests/test_bamboo.py`` and
``examples/fidelity_checks/bamboo_verification.ipynb``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from xnns.common.data import AtomicGraph
from xnns.common.models.base import InteratomicPotential
from xnns.common.models.ops import scatter_sum
from xnns.common.models.registry import register_model
from xnns.common.featurizers import CosineCutoff
from xnns.transformer.attention import EdgeMultiheadAttention
from xnns.transformer.featurizers import ExpNormalSmearing

from .dispersion import D3CSODispersion

# Physical constants in BAMBOO's native units (energies in kcal/mol, lengths in
# Angstrom), reproduced from bytedance/bamboo ``utils/constant.py``.
#: Coulomb prefactor ``k_e * e^2`` in kcal/mol * Angstrom / e^2.
ELE_FACTOR = 332.06349451357806
#: Dipole conversion (e * Angstrom -> Debye is the reciprocal).
DEBYE_EA = 0.20819427381112157

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU, "swish": nn.SiLU, "gelu": nn.GELU,
    "tanh": nn.Tanh, "relu": nn.ReLU,
}


def _make_act(act: "str | nn.Module") -> nn.Module:
    """Return an activation module from a name or an existing module.

    Parameters
    ----------
    act : str or torch.nn.Module
        Either an activation name (``"silu"``, ``"gelu"``, ...) or an
        already-constructed activation module (returned as-is).

    Returns
    -------
    torch.nn.Module
        The activation module.
    """
    if isinstance(act, nn.Module):
        return act
    key = str(act).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {act!r}; choose from {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


def _energy_mlp(n_layers: int, dim: int, act: str) -> nn.Sequential:
    """Build a BAMBOO read-out MLP ``dim -> dim/2 -> ... -> 1``.

    Mirrors the upstream ``get_mlp_layers``: ``n_layers`` hidden linears (the
    first ``dim -> dim/2``, the rest ``dim/2 -> dim/2``) each followed by the
    activation, then a final ``dim/2 -> 1`` linear.

    Parameters
    ----------
    n_layers : int
        Number of hidden layers.
    dim : int
        Input feature width; the hidden width is ``dim // 2``.
    act : str
        Activation name for the hidden layers.

    Returns
    -------
    torch.nn.Sequential
        The read-out MLP producing a scalar per atom.
    """
    hidden = dim // 2
    layers: list[nn.Module] = []
    for i in range(n_layers):
        layers.append(nn.Linear(dim if i == 0 else hidden, hidden))
        layers.append(_make_act(act))
    layers.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layers)


class GETLayer(nn.Module):
    """One Graph Equivariant Transformer layer (BAMBOO Supplementary A.1).

    A GET layer updates the scalar node feature ``x_i`` and, except in the
    last layer, the vector node feature ``V_i``. It first runs the shared
    multi-head :class:`~xnns.transformer.attention.EdgeMultiheadAttention` to
    get a neighbour value ``v_j`` and attention weight ``a_ij`` per edge, then:

    * forms the **scalar message** ``m_i = sum_j a_ij (v_j * d_ij)`` where
      ``d_ij`` is the per-head radial edge feature, and the **vector message**
      ``u_i = sum_j v_j * e_ij`` where ``e_ij = d_ij * r_hat_ij`` is the
      equivariant edge vector (both aggregated onto the centre atom);
    * mixes the current vector feature through learned projections and an inner
      product ``w_i = <U1 V_i, U2 V_i>`` (a rotation-invariant scalar), and
      combines everything into the scalar/vector updates.

    The first layer has no incoming ``V_i`` (its vector output is just ``u_i``)
    and the last layer produces no ``V_i`` (only the scalar update, which feeds
    the read-outs). Every layer's scalar/vector updates are added residually by
    the parent :class:`BAMBOO`.

    Parameters
    ----------
    dim : int
        Node scalar feature width.
    num_heads : int
        Number of attention heads.
    attn_act : str or torch.nn.Module
        Activation applied to the attention logits (BAMBOO uses GELU).
    is_first : bool, optional
        First layer (no vector input), by default ``False``.
    is_last : bool, optional
        Last layer (no vector output), by default ``False``.
    """

    def __init__(self, dim: int, num_heads: int, attn_act: "str | nn.Module",
                 is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.is_first = is_first
        self.is_last = is_last
        self.attn = EdgeMultiheadAttention(dim, num_heads, _make_act(attn_act))
        if is_first:
            self.output_proj = nn.Linear(dim, dim)
            self.vec_proj: Optional[nn.Linear] = None
        elif is_last:
            self.output_proj = nn.Linear(dim, dim * 2)
            self.vec_proj = nn.Linear(dim, dim * 2, bias=False)
        else:
            self.output_proj = nn.Linear(dim, dim * 3)
            self.vec_proj = nn.Linear(dim, dim * 3, bias=False)

    def forward(self, node_feat: Tensor, edge_feat: Tensor,
                edge_vec: Optional[Tensor], node_vec: Optional[Tensor],
                center: Tensor, neighbor: Tensor, envelope: Tensor,
                n_atoms: int) -> tuple[Tensor, Optional[Tensor]]:
        """Apply one GET layer.

        Parameters
        ----------
        node_feat : Tensor
            Scalar node features ``(N, dim)``.
        edge_feat : Tensor
            Per-head radial edge feature ``(E, num_heads, dim_per_head)``.
        edge_vec : Tensor or None
            Equivariant edge vector ``(E, 3, num_heads, dim_per_head)``; unused
            by the last layer (pass ``None``).
        node_vec : Tensor or None
            Vector node features ``(N, 3, dim)``; ``None`` for the first layer.
        center : Tensor
            Centre (receiver) atom index per edge, shape ``(E,)``.
        neighbor : Tensor
            Neighbour (sender) atom index per edge, shape ``(E,)``.
        envelope : Tensor
            Radial cutoff weight per edge, shape ``(E,)``.
        n_atoms : int
            Number of atoms ``N`` (scatter dimension size).

        Returns
        -------
        tuple of (Tensor, Tensor or None)
            The scalar update ``delta_x`` of shape ``(N, dim)`` and, unless
            this is the last layer, the vector update/output of shape
            ``(N, 3, dim)`` (``None`` for the last layer).
        """
        values, gate = self.attn(node_feat, center, neighbor, envelope)

        # scalar channel: values carry the radial edge feature and the
        # attention gate onto the centre atom, heads then merge back to dim
        weighted = values * edge_feat * gate.unsqueeze(-1)
        scal_msg = scatter_sum(weighted, center, n_atoms).flatten(-2)

        if self.is_first:
            # nothing equivariant exists yet, so the vector output is just
            # the aggregated edge-vector message
            lifted = values.unsqueeze(-3) * edge_vec
            vec_msg = scatter_sum(lifted, center, n_atoms).flatten(-2)
            return self.output_proj(scal_msg), vec_msg

        assert self.vec_proj is not None and node_vec is not None
        # a dot product between two learned projections of V_i is rotation
        # invariant, so it may feed the scalar channel without breaking
        # equivariance
        mixed = self.vec_proj(node_vec)
        proj = self.output_proj(scal_msg)

        if self.is_last:
            left, right = torch.split(mixed, self.dim, dim=-1)
            invariant = (left * right).sum(dim=-2)
            scale, shift = torch.split(proj, self.dim, dim=-1)
            return invariant * scale + shift, None

        left, right, carry = torch.split(mixed, self.dim, dim=-1)
        invariant = (left * right).sum(dim=-2)
        vec_gate, scale, shift = torch.split(proj, self.dim, dim=-1)

        lifted = values.unsqueeze(-3) * edge_vec
        vec_msg = scatter_sum(lifted, center, n_atoms).flatten(-2)

        delta_x = invariant * scale + shift
        delta_v = carry * vec_gate.unsqueeze(-2) + vec_msg
        return delta_x, delta_v


@register_model("bamboo")
class BAMBOO(InteratomicPotential):
    """Graph equivariant transformer force field (Gong et al. 2024).

    See the module docstring for the architecture. All hyper-parameters have
    the upstream defaults; the shared fields (``cutoff``, ``n_features``,
    ``n_rbf``, ``n_interactions``) come from the core
    :class:`~xnns.common.config.schema.ModelConfig` and the rest from its
    ``extra`` dict (see :meth:`from_config`).

    Parameters
    ----------
    dim : int, optional
        Node scalar feature width (``n_features``), by default 64. Must be
        divisible by ``num_heads``.
    num_rbf : int, optional
        Number of exponential-normal radial basis functions (``n_rbf``), by
        default 32.
    cutoff : float, optional
        Semi-local radial cutoff ``r_cut`` in Angstrom, by default 5.0.
    n_layers : int, optional
        Number of GET layers (``n_interactions``); must be >= 2, by default 3.
    num_heads : int, optional
        Number of attention heads, by default 16.
    charge_ub : float, optional
        Upper bound of the per-atom partial charge (a ``tanh`` squashes the raw
        charge into ``[-charge_ub, charge_ub]``), by default 2.0.
    charge_mlp_layers, energy_mlp_layers : int, optional
        Hidden-layer counts of the charge and energy read-out MLPs, by default
        2 each.
    n_elements : int, optional
        Size of the atom-type embedding table (indexed directly by atomic
        number ``Z``), by default 87 (H..Rn, upstream default).
    act_fn : str or torch.nn.Module, optional
        Activation of the read-out MLPs (upstream SiLU), by default ``"silu"``.
    attn_act_fn : str or torch.nn.Module, optional
        Activation of the attention and the radial-projection (upstream GELU),
        by default ``"gelu"``.
    coul_damping_beta : float, optional
        Softplus sharpness of the short-range Coulomb damping, by default 18.7.
    coul_damping_r0 : float, optional
        Onset distance of the Coulomb damping in Angstrom, by default 2.2.
    ele_factor : float, optional
        Coulomb prefactor ``k_e e^2`` in kcal/mol*Angstrom, by default
        :data:`ELE_FACTOR`.
    use_electrostatics : bool, optional
        Include the charge-equilibrium electrostatic energy, by default
        ``True``.
    use_dispersion : bool, optional
        Add the D3(CSO) dispersion energy, by default ``False`` (matching the
        paper's training set).
    disp_cutoff : float, optional
        Dispersion cutoff in Angstrom (used only when ``use_dispersion``), by
        default 10.0.
    species : list of int, optional
        Supported atomic numbers, recorded for bookkeeping/config round-trips
        (the embedding is indexed by ``Z`` regardless). Default ``None``.

    Attributes
    ----------
    node_feature_dim : int
        Width of the invariant per-atom features exposed as ``node_features``
        (equal to ``dim``), consumed e.g. by
        :class:`~xnns.common.models.les.LatentEwald`.

    Raises
    ------
    ValueError
        If ``n_layers < 2`` or ``dim`` is not divisible by ``num_heads``.

    Notes
    -----
    The built-in electrostatics is the molecular / gas-phase-cluster
    charge-equilibrium form: a damped Coulomb summed over *all* intra-structure
    pairs from the raw positions (as in the paper's cluster training and the
    upstream ``predict`` path). It is therefore not minimum-imaged, so for
    genuinely periodic long-range electrostatics use the LAMMPS Ewald route (as
    the original does) or wrap the model with
    :class:`~xnns.common.models.les.LatentEwald` (BAMBOO exposes the required
    ``node_features``); set ``use_electrostatics=False`` to drop the built-in
    term in that case.
    """

    def __init__(
        self,
        dim: int = 64,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        n_layers: int = 3,
        num_heads: int = 16,
        charge_ub: float = 2.0,
        charge_mlp_layers: int = 2,
        energy_mlp_layers: int = 2,
        n_elements: int = 87,
        act_fn: "str | nn.Module" = "silu",
        attn_act_fn: "str | nn.Module" = "gelu",
        coul_damping_beta: float = 18.7,
        coul_damping_r0: float = 2.2,
        ele_factor: float = ELE_FACTOR,
        use_electrostatics: bool = True,
        use_dispersion: bool = False,
        disp_cutoff: float = 10.0,
        species: Optional[list[int]] = None,
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers (n_interactions) must be >= 2")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.cutoff = cutoff
        self.dim = dim
        self.num_rbf = num_rbf
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.charge_ub = charge_ub
        self.n_elements = n_elements
        self.ele_factor = ele_factor
        self.coul_damping_beta = coul_damping_beta
        self.coul_damping_r0 = coul_damping_r0
        self.use_electrostatics = use_electrostatics
        self.use_dispersion = use_dispersion
        self.species = list(species) if species is not None else None
        self.node_feature_dim = dim

        self.atom_emb = nn.Embedding(n_elements, dim)
        self.dis_rbf = ExpNormalSmearing(num_rbf, cutoff, trainable=True)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.rbf_proj = nn.Sequential(
            nn.Linear(num_rbf, dim, bias=False), _make_act(attn_act_fn))

        layers: list[GETLayer] = [
            GETLayer(dim, num_heads, attn_act_fn, is_first=True)]
        for _ in range(n_layers - 2):
            layers.append(GETLayer(dim, num_heads, attn_act_fn))
        layers.append(GETLayer(dim, num_heads, attn_act_fn, is_last=True))
        self.layers = nn.ModuleList(layers)

        self.energy_mlp = _energy_mlp(energy_mlp_layers, dim, act_fn)
        self.charge_mlp = _energy_mlp(charge_mlp_layers, dim, act_fn)
        # electronegativity chi and hardness J: MLPs of the *initial* embedding
        self.electronegativity_mlp = _energy_mlp(charge_mlp_layers, dim, act_fn)
        self.hardness_mlp = _energy_mlp(charge_mlp_layers, dim, act_fn)

        self.coul_softplus = nn.Softplus(beta=coul_damping_beta)
        self.dispersion = (
            D3CSODispersion(disp_cutoff=disp_cutoff) if use_dispersion else None)

    # -- charge-equilibrium electrostatics ---------------------------------
    def coulomb_energy(self, charges: Tensor, edge_vec: Tensor,
                       row: Tensor, col: Tensor) -> Tensor:
        """Per-pair damped Coulomb energy (upstream ``get_coulomb``, no Ewald).

        The short-range damping softens the ``1/r`` singularity smoothly::

            E_ij = ele_factor * q_i q_j / r *  (r / r0) / (1 + softplus((r-r0)/r0))

        Its gradient with respect to ``r`` equals the paper's separately-damped
        Coulomb force (softplus differentiates to sigmoid), so autograd forces
        are the intended ones.

        Parameters
        ----------
        charges : Tensor
            Per-atom partial charges ``(N,)``.
        edge_vec : Tensor
            Displacement vectors of the (all-pairs) list ``(P, 3)``.
        row, col : Tensor
            The two atoms of each pair, shape ``(P,)`` each.

        Returns
        -------
        Tensor
            Per-pair Coulomb energy ``(P,)``.
        """
        dist = edge_vec.norm(dim=-1)
        bare = self.ele_factor * charges[row] * charges[col] / dist
        r0 = self.coul_damping_r0
        damp = self.coul_softplus((dist - r0) / r0)
        return bare * dist / r0 / (1.0 + damp)

    @staticmethod
    def _all_pairs(batch: Tensor) -> tuple[Tensor, Tensor]:
        """All ordered intra-structure atom pairs ``(i, j)``, ``i != j``.

        Parameters
        ----------
        batch : Tensor
            Structure index per atom ``(N,)``.

        Returns
        -------
        tuple of Tensor
            The centre indices ``row`` and neighbour indices ``col`` of every
            ordered same-structure pair.
        """
        same = batch.unsqueeze(0) == batch.unsqueeze(1)
        same = same & ~torch.eye(batch.shape[0], dtype=torch.bool,
                                 device=batch.device)
        row, col = same.nonzero(as_tuple=True)
        return row, col

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Predict energies, partial charges and the dipole for a graph.

        Parameters
        ----------
        data : xnns.common.data.AtomicGraph
            The input atomic graph. If it carries a ``total_charge`` attribute
            (per-structure tensor) the partial charges are conserved to it;
            otherwise neutral structures are assumed.

        Returns
        -------
        dict of str to torch.Tensor
            ``"node_energy"`` ``(N,)`` and ``"energy"`` ``(B,)`` (as every xnns
            model), plus ``"node_features"`` ``(N, dim)`` (invariant features),
            ``"charges"`` ``(N,)`` (conserved partial charges), ``"dipole"``
            ``(B, 3)`` in Debye, and the component energies ``"energy_nn"`` /
            ``"energy_elec"`` ``(B,)``.
        """
        Z = data.atomic_numbers
        batch = data.batch
        N = data.num_nodes
        B = data.num_graphs

        edge_vec = data.edge_vectors()               # (E,3) = pos[dst]-pos[src]
        center = data.edge_index[1]                  # receiver (upstream "row")
        neighbor = data.edge_index[0]                # sender   (upstream "col")
        r = edge_vec.norm(dim=-1)
        unit = edge_vec / r.unsqueeze(-1)
        radial_emb = self.dis_rbf(r)
        envelope = self.cutoff_fn(r)

        # split the projected radial embedding into heads, then lift it onto
        # the unit bond direction to seed the equivariant edge feature
        edge_feat = self.rbf_proj(radial_emb).unflatten(
            -1, (self.num_heads, self.dim // self.num_heads))
        edge_vec_feat = unit[..., None, None] * edge_feat.unsqueeze(-3)

        x0 = self.atom_emb(Z)                        # initial scalar feature
        chi = self.electronegativity_mlp(x0).squeeze(-1)
        hardness = self.hardness_mlp(x0).squeeze(-1)

        # GET message passing
        node_feat = x0
        node_vec: Optional[Tensor] = None
        for layer in self.layers:
            delta_feat, delta_vec = layer(
                node_feat, edge_feat, edge_vec_feat, node_vec,
                center, neighbor, envelope, N)
            node_feat = node_feat + delta_feat
            if delta_vec is not None:
                node_vec = delta_vec if node_vec is None else node_vec + delta_vec

        # partial charges (bounded), conserved to the total charge
        raw_charge = self.charge_mlp(node_feat).squeeze(-1)
        charge = self.charge_ub * torch.tanh(raw_charge / self.charge_ub)
        total_charge = getattr(data, "total_charge", None)
        if total_charge is None:
            total_charge = torch.zeros(B, dtype=charge.dtype, device=charge.device)
        n_per = torch.bincount(batch, minlength=B).to(charge.dtype)
        q_sum = scatter_sum(charge, batch, B)
        charges = charge + ((total_charge - q_sum) / n_per)[batch]

        # semi-local NN energy
        node_energy = self.energy_mlp(node_feat).squeeze(-1)
        energy_nn = self.aggregate_energy(node_energy, data)

        # electrostatics: on-site electronegativity/hardness + damped Coulomb
        energy_elec = torch.zeros(B, dtype=node_energy.dtype, device=node_energy.device)
        if self.use_electrostatics:
            electroneg_atom = chi ** 2 * charges + hardness ** 2 * charges ** 2
            node_energy = node_energy + electroneg_atom
            row, col = self._all_pairs(batch)
            if row.numel() > 0:
                pair_vec = data.pos[row] - data.pos[col]
                ecoul = self.coulomb_energy(charges, pair_vec, row, col)
                node_coul = 0.5 * scatter_sum(ecoul, row, N)
                node_energy = node_energy + node_coul
            energy_elec = self.aggregate_energy(node_energy, data) - energy_nn

        # optional D3(CSO) dispersion (semi-local, from the cutoff graph)
        if self.dispersion is not None:
            disp_atom = self.dispersion(Z, edge_vec, data.edge_index, N)
            node_energy = node_energy + disp_atom

        energy = self.aggregate_energy(node_energy, data)
        dipole = scatter_sum(charges.unsqueeze(-1) * data.pos, batch, B) / DEBYE_EA
        return {
            "node_energy": node_energy,
            "energy": energy,
            "node_features": node_feat,
            "charges": charges,
            "dipole": dipole,
            "energy_nn": energy_nn,
            "energy_elec": energy_elec,
        }

    @classmethod
    def from_config(cls, cfg) -> "BAMBOO":
        """Construct a :class:`BAMBOO` from a core model config.

        Core fields map as ``cfg.cutoff -> cutoff``, ``cfg.n_features -> dim``,
        ``cfg.n_rbf -> num_rbf`` and ``cfg.n_interactions -> n_layers``; every
        other option is read from ``cfg.extra`` (upstream spellings are
        translated by :mod:`xnns.common.config.translate`).

        Parameters
        ----------
        cfg : xnns.common.config.schema.ModelConfig
            The core model config, whose ``extra`` dict carries the BAMBOO
            architecture options.

        Returns
        -------
        BAMBOO
            The instantiated model.
        """
        from xnns.common.config.coerce import coerce_species

        extra = dict(cfg.extra or {})
        species = (coerce_species(extra.get("species"))
                   if extra.get("species") is not None else None)
        return cls(
            dim=cfg.n_features,
            num_rbf=cfg.n_rbf,
            cutoff=cfg.cutoff,
            n_layers=cfg.n_interactions,
            num_heads=extra.get("num_heads", 16),
            charge_ub=extra.get("charge_ub", 2.0),
            charge_mlp_layers=extra.get("charge_mlp_layers", 2),
            energy_mlp_layers=extra.get("energy_mlp_layers", 2),
            n_elements=extra.get("n_elements", 87),
            act_fn=extra.get("act_fn", "silu"),
            attn_act_fn=extra.get("attn_act_fn", "gelu"),
            coul_damping_beta=extra.get("coul_damping_beta", 18.7),
            coul_damping_r0=extra.get("coul_damping_r0", 2.2),
            ele_factor=extra.get("ele_factor", ELE_FACTOR),
            use_electrostatics=extra.get("use_electrostatics", True),
            use_dispersion=extra.get("use_dispersion", False),
            disp_cutoff=extra.get("disp_cutoff", 10.0),
            species=species,
        )
