"""Latent Ewald Summation (LES): long-range interactions for any xnns model.

Implements Cheng, *npj Comput Mater* **11**, 80 (2025): short-range MLIPs miss
long-range physics (electrostatics, dispersion) beyond their receptive field.
LES fixes this generically -- a small MLP maps each atom's **invariant
features** to a low-dimensional hidden variable ``q`` (paper eq 2, analogous to
environment-dependent partial charges, but unconstrained), and an Ewald
summation over the structure factor of ``q`` (eqs 3-4) supplies the long-range
energy

    E_lr = (1/V) sum_{0<k<k_c} exp(-sigma^2 k^2 / 2) / k^2 * abs(S(k))^2 .

Faithful to the reference implementation (``cace.modules.EwaldPotential`` of
https://github.com/BingqingCheng/cace and the training scripts of
https://github.com/BingqingCheng/cace-lr-fit): :class:`EwaldSummation` ports the
reciprocal-space (triclinic-capable) sum, the hemisphere symmetry factors, the
``k = 0`` and self-interaction conventions, the ``1/r^6`` dispersion variant
(paper eq 5), and the real-space ``erf``-converged direct sum used for
non-periodic structures. (One upstream wart is fixed rather than ported: its
k-vector grid is always built in float32, which crashes float64 runs; here it
follows the input dtype.)

Because :class:`LatentEwald` only needs *invariant per-atom features*, it wraps
**any** registered xnns model -- every model exposes its features through the
``"node_features"`` output key and a ``node_feature_dim`` attribute (CACE's
symmetrized B features, the scalar channels of MACE / NequIP node features,
Allegro's environment-aggregated edge latents, SchNet / PhysNet feature vectors,
HDNNP/ANI descriptors). Enable it from a config with
``model.extra["long_range"]`` (see
:func:`~xnns.common.models.registry.build_model`) or wrap directly::

    model = LatentEwald(build_model(cfg.model), n_channels=4, sigma=1.0)
    out = ForceStressOutput(model)(graph)   # forces/stress include E_lr

The wrapped energy cost is roughly twice the short-range cost.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..data import AtomicGraph
from .base import InteratomicPotential


class EwaldSummation(nn.Module):
    """Ewald energy of a (latent) per-atom variable ``q`` (paper eqs 3-5).

    For periodic structures, the reciprocal-space sum over a k-grid limited
    by ``|k| <= 2*pi/dl`` (upstream convention: the paper's ``k_c`` equals
    ``2*pi/dl``); for non-periodic structures (no cell), the equivalent
    ``erf``-converged real-space direct sum. Both handle a multi-dimensional
    ``q``, summing the energies of the channels.

    Parameters
    ----------
    dl : float, optional
        Reciprocal grid resolution; the k-space cutoff is ``2*pi/dl``. By
        default 2.0 (``k_c = pi``, the paper's bulk-water setting; its
        dimer/NaCl runs used ``dl = 3``).
    sigma : float, optional
        Gaussian smearing width in Angstrom, by default 1.0 (paper Methods:
        values between ~0.5 and 2 are reasonable; 1 was best for water).
    exponent : int, optional
        Interaction exponent ``p`` of ``1/r^p``: 1 (electrostatics, default)
        or 6 (London dispersion, paper eq 5).
    remove_self_interaction : bool, optional
        Subtract the Gaussian self energy ``sum q^2 / (sigma (2 pi)^{3/2})``
        from the reciprocal sum, by default ``False`` (the reference training
        scripts keep it; the term is short-ranged and can be absorbed by the
        short-range model either way). Note: for multi-channel ``q`` upstream
        subtracts the total ``sum q^2`` once *per channel* (an
        ``n_channels``-fold over-subtraction); xnns subtracts it once. The
        two agree for 1-dimensional ``q`` and always when the flag is off.

    Notes
    -----
    Charges are in scaled units (upstream ``norm_factor = 1``): a physical
    charge ``Q`` in e corresponds to ``q = Q * sqrt(90.0474)`` for energies
    in eV and distances in Angstrom.
    """

    def __init__(self, dl: float = 2.0, sigma: float = 1.0, exponent: int = 1,
                 remove_self_interaction: bool = False):
        super().__init__()
        if exponent not in (1, 6):
            raise ValueError(f"exponent must be 1 or 6, got {exponent}")
        self.dl = dl
        self.sigma = sigma
        self.exponent = exponent
        self.remove_self_interaction = remove_self_interaction
        self.k_sq_max = (2.0 * math.pi / dl) ** 2

    def _kfac(self, k_sq: Tensor) -> Tensor:
        """Interaction kernel in reciprocal space (paper eqs 4 and 5)."""
        sigma_sq_half = self.sigma ** 2 / 2.0
        if self.exponent == 1:
            return torch.exp(-sigma_sq_half * k_sq) / k_sq
        b_sq = k_sq * sigma_sq_half
        b = torch.sqrt(b_sq)
        return -1.0 * k_sq ** 1.5 * (
            math.sqrt(math.pi) * torch.special.erfc(b)
            + (1 / (2 * b ** 3) - 1 / b) * torch.exp(-b_sq))

    def _self_energy(self, q: Tensor) -> Tensor:
        return torch.sum(q ** 2) / (self.sigma * (2 * math.pi) ** 1.5)

    def reciprocal(self, pos: Tensor, q: Tensor, cell: Tensor) -> Tensor:
        """Reciprocal-space Ewald energy of one periodic structure.

        Parameters
        ----------
        pos : Tensor
            Cartesian positions, shape ``(n, 3)``.
        q : Tensor
            Hidden variable, shape ``(n, n_channels)``.
        cell : Tensor
            Row-vector cell matrix, shape ``(3, 3)`` (triclinic allowed).

        Returns
        -------
        Tensor
            Scalar long-range energy (summed over channels).
        """
        device, dtype = pos.device, pos.dtype
        G = 2 * math.pi * torch.linalg.inv(cell).T  # reciprocal lattice rows
        norms = torch.norm(cell, dim=1)
        # the small relative tolerances below resolve floating-point ties in
        # the grid size and at the |k| = k_c shell consistently, so the energy
        # is exactly rotation-invariant (upstream truncates/compares exactly,
        # which can drop a whole k shell when a rotated cell's row norm or a
        # boundary shell lands an ulp below the cut)
        nk = [max(1, int(n.item() / self.dl + 1e-9)) for n in norms]
        grids = [torch.arange(-k, k + 1, device=device) for k in nk]
        nvec = torch.stack(torch.meshgrid(*grids, indexing="ij"),
                           dim=-1).reshape(-1, 3)
        kvec = nvec.to(dtype) @ G
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > self.k_sq_max * 1e-12) & (k_sq <= self.k_sq_max * (1 + 1e-9))
        kvec, k_sq, nvec = kvec[mask], k_sq[mask], nvec[mask]

        # half-space to avoid double counting: keep k whose first nonzero
        # integer component is positive, with symmetry factor 2
        first_nonzero = torch.argmax((nvec != 0).to(torch.int), dim=1)
        sign = torch.gather(nvec, 1, first_nonzero.unsqueeze(1)).squeeze(1)
        keep = sign > 0
        kvec, k_sq = kvec[keep], k_sq[keep]

        exp_ikr = torch.exp(1j * (pos @ kvec.T))               # (n, M)
        s_k = (q.to(exp_ikr.dtype).unsqueeze(2)
               * exp_ikr.unsqueeze(1)).sum(dim=0)              # (n_q, M)
        pot = (2.0 * self._kfac(k_sq) * torch.abs(s_k) ** 2).sum() \
            / torch.det(cell)
        if self.remove_self_interaction and self.exponent == 1:
            pot = pot - self._self_energy(q)
        return pot

    def realspace(self, pos: Tensor, q: Tensor) -> Tensor:
        """Direct-sum equivalent for a non-periodic structure.

        The pair interaction is ``erf(r / (sqrt(2) sigma)) / r`` -- the
        potential of the Gaussian-smeared charges -- normalized by
        ``1/(4 pi)`` exactly as upstream, so periodic and molecular
        structures share the same energy scale. Only ``exponent = 1`` is
        supported (as upstream).
        """
        if self.exponent != 1:
            raise ValueError("realspace fallback supports exponent=1 only")
        r_ij = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), dim=-1)
        conv = torch.special.erf(r_ij / self.sigma / math.sqrt(2.0))
        inv_r = 1.0 / (r_ij + 1e-6)
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1)
                        * (inv_r * conv).unsqueeze(2)) / (4 * math.pi)
        if not self.remove_self_interaction:
            pot = pot + self._self_energy(q)
        return pot

    def forward(self, q: Tensor, pos: Tensor, batch: Tensor, num_graphs: int,
                cell: Tensor | None, pbc: Tensor | None = None) -> Tensor:
        """Long-range energy per structure for a batched graph.

        Parameters
        ----------
        q : Tensor
            Hidden variable, shape ``(N,)`` or ``(N, n_channels)``.
        pos : Tensor
            Cartesian positions, shape ``(N, 3)``.
        batch : Tensor
            Structure index of each atom, shape ``(N,)``.
        num_graphs : int
            Number of structures ``B`` in the batch.
        cell : Tensor or None
            Cells of shape ``(B, 3, 3)``, or ``None`` for molecular batches.
        pbc : Tensor or None, optional
            Per-structure periodic flags ``(B, 3)``; a structure is treated
            as periodic when its cell is nonzero and any flag is set.

        Returns
        -------
        Tensor
            Long-range energies, shape ``(B,)``.
        """
        if q.dim() == 1:
            q = q.unsqueeze(1)
        out = q.new_zeros(num_graphs)
        for i in range(num_graphs):
            mask = batch == i
            periodic = (cell is not None
                        and bool(cell[i].diagonal().abs().sum() > 1e-6)
                        and (pbc is None or bool(pbc[i].any())))
            if periodic:
                out[i] = self.reciprocal(pos[mask], q[mask], cell[i])
            else:
                out[i] = self.realspace(pos[mask], q[mask])
        return out


class LatentEwald(InteratomicPotential):
    """Wrap any xnns model with a Latent-Ewald long-range energy (CACE-LR).

    The wrapped model must expose invariant per-atom features through the
    ``"node_features"`` key of its output dict and a ``node_feature_dim``
    attribute -- every built-in xnns model does. A bias-free MLP plus a
    parallel bias-free linear layer (the head used by the reference
    ``cace-lr-fit`` scripts) maps the features to the hidden variable ``q``
    (paper eq 2), and :class:`EwaldSummation` turns ``q`` into the long-range
    energy added to the model's short-range prediction.

    Enable from a config with ``model.extra["long_range"]``, e.g.::

        extra: {..., long_range: {n_channels: 4, sigma: 1.0, dl: 2.0}}

    Parameters
    ----------
    model : InteratomicPotential
        The short-range model to wrap.
    n_channels : int, optional
        Dimension of the hidden variable ``q``, by default 4 (the paper's
        bulk-water/NaCl setting; the reference charged-dimer script uses 1).
    hidden : list of int, optional
        Hidden widths of the ``q`` MLP, by default ``[24, 12]`` (upstream).
    q_bias : bool, optional
        Use a bias in the ``q`` MLP, by default ``False`` (the water script's
        head). The reference charged-dimer script uses ``True`` -- the bias
        lets the latent charge carry a per-structure offset, which matters for
        net-charged systems like ionic dimers.
    q_add_linear : bool, optional
        Add a parallel bias-free linear layer to the ``q`` head, by default
        ``True`` (the water script). The charged-dimer script uses ``False``
        (MLP only).
    dl, sigma, exponent, remove_self_interaction
        Passed to :class:`EwaldSummation`.

    Attributes
    ----------
    model : InteratomicPotential
        The wrapped short-range model.
    q_net : torch.nn.Module
        The latent-charge MLP.
    q_linear : torch.nn.Module or None
        The optional parallel linear layer; ``q = q_net(B) [+ q_linear(B)]``.
    ewald : EwaldSummation
        The long-range energy module.
    cutoff : float
        The wrapped model's neighbor-list cutoff (proxied).

    Notes
    -----
    ``forward`` returns the combined ``"energy"``; the long-range energy is
    spread uniformly over the atoms of each structure in ``"node_energy"``
    so it still sums to the total. The additional keys ``"energy_sr"``,
    ``"energy_lr"``, and ``"latent_charges"`` expose the decomposition.
    """

    def __init__(self, model: InteratomicPotential, n_channels: int = 4,
                 hidden=None, q_bias: bool = False, q_add_linear: bool = True,
                 dl: float = 2.0, sigma: float = 1.0,
                 exponent: int = 1, remove_self_interaction: bool = False):
        super().__init__()
        feature_dim = getattr(model, "node_feature_dim", None)
        if feature_dim is None:
            raise TypeError(
                f"{type(model).__name__} does not expose node_feature_dim / "
                "'node_features'; LatentEwald needs invariant per-atom "
                "features from the wrapped model")
        self.model = model
        self.cutoff = model.cutoff
        # q head: an MLP (as the reference fit scripts) optionally plus a
        # parallel linear layer -- the water script uses bias-free + linear,
        # the charged-dimer script an MLP with bias and no parallel linear
        hidden = list(hidden or [24, 12])
        layers: list[nn.Module] = []
        widths = [feature_dim] + hidden
        for w_in, w_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(w_in, w_out, bias=q_bias), nn.SiLU()]
        layers.append(nn.Linear(widths[-1], n_channels, bias=q_bias))
        self.q_net = nn.Sequential(*layers)
        self.q_linear = (nn.Linear(feature_dim, n_channels, bias=False)
                         if q_add_linear else None)
        self.ewald = EwaldSummation(dl=dl, sigma=sigma, exponent=exponent,
                                    remove_self_interaction=remove_self_interaction)

    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Short-range prediction plus the latent-Ewald long-range energy.

        Parameters
        ----------
        data : AtomicGraph
            The batched atomic graph.

        Returns
        -------
        dict of str to torch.Tensor
            The wrapped model's outputs with ``"energy"`` and
            ``"node_energy"`` including the long-range term, plus
            ``"energy_sr"`` ``(B,)``, ``"energy_lr"`` ``(B,)`` and
            ``"latent_charges"`` ``(N, n_channels)``.
        """
        out = self.model(data)
        features = out["node_features"]
        q = self.q_net(features)
        if self.q_linear is not None:
            q = q + self.q_linear(features)
        energy_lr = self.ewald(q, data.pos, data.batch, data.num_graphs,
                               data.cell, data.pbc)
        n_atoms = torch.bincount(data.batch, minlength=data.num_graphs)
        out["energy_sr"] = out["energy"]
        out["energy_lr"] = energy_lr
        out["latent_charges"] = q
        out["energy"] = out["energy"] + energy_lr
        out["node_energy"] = out["node_energy"] + (
            energy_lr / n_atoms.to(energy_lr.dtype))[data.batch]
        return out

    @classmethod
    def from_config(cls, cfg) -> "LatentEwald":
        """Not registered directly; built by ``build_model`` via
        ``model.extra['long_range']``."""
        raise NotImplementedError(
            "enable LES via model.extra['long_range'] in the base model's "
            "config, or wrap explicitly: LatentEwald(build_model(cfg), ...)")
