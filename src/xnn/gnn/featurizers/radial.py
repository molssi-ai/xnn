"""Radial pieces of the equivariant GNN edge embedding.

The Bessel basis (NequIP/MACE/DimeNet-style) plus the chemistry-aware
*distance transforms* some force fields warp their radial coordinate with
before the basis expansion: the Agnesi transform (Batatia *et al.*,
MACE-MP-0, arXiv:2401.00096; the transform itself is from the radial
transformations of ACEpotentials.jl, *J. Chem. Phys.* 159, 164101, 2023) and
a tanh-based soft lower clamp. Both rescale each edge by the covalent radii
of its two elements, so short bonds between small atoms and long bonds
between large atoms map to comparable coordinates.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _covalent_radii_table() -> Tensor:
    """The ASE covalent-radii table (Cordero et al. 2008), indexed by Z.

    Returns
    -------
    Tensor
        Covalent radii in Angstrom, shape ``(n_elements,)``.

    Raises
    ------
    ImportError
        If ASE is not installed.
    """
    try:
        from ase.data import covalent_radii
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ImportError(
            'distance transforms need ase: pip install "xnn[ase]"') from e
    return torch.tensor(covalent_radii, dtype=torch.get_default_dtype())


class IdentityDistanceTransform(nn.Module):
    """The trivial distance transform: distances pass through unchanged.

    Keeps the edge-embedding code path uniform (and TorchScript-friendly)
    for models whose radial basis acts on the raw interatomic distance.
    """

    def forward(self, r: Tensor, atomic_numbers: Tensor,
                edge_index: Tensor) -> Tensor:
        """Return the distances unchanged.

        Parameters
        ----------
        r : Tensor
            Interatomic distances, shape ``(E,)``.
        atomic_numbers : Tensor
            Per-node atomic numbers, shape ``(N,)`` (unused).
        edge_index : Tensor
            Edge index of shape ``(2, E)`` (unused).

        Returns
        -------
        Tensor
            ``r``, unchanged.
        """
        return r


class AgnesiDistanceTransform(nn.Module):
    """Agnesi radial transform, rescaled per edge by covalent radii.

    Maps the interatomic distance ``r`` to

        ``T(r) = 1 / (1 + a (r/r0)^q / (1 + (r/r0)^(q-p)))``

    with ``r0`` the *mean* covalent radius of the two edge elements,
    ``0.5 (rcov_i + rcov_j)``. This is the transform of the MACE-MP
    foundation models (``distance_transform="Agnesi"``); the functional form
    comes from the radial transformations of ACEpotentials.jl
    (*J. Chem. Phys.* 159, 164101, 2023). The constants ``a``, ``q``, ``p``
    and the covalent-radii table are registered buffers, so values stored in
    a trained checkpoint carry over on ``load_state_dict``.

    Parameters
    ----------
    a, q, p : float, optional
        The transform constants, by default the published MACE values
        (1.0805, 0.9183, 4.5791).
    """

    def __init__(self, a: float = 1.0805, q: float = 0.9183,
                 p: float = 4.5791):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.register_buffer("a", torch.tensor(a, dtype=dtype))
        self.register_buffer("q", torch.tensor(q, dtype=dtype))
        self.register_buffer("p", torch.tensor(p, dtype=dtype))
        self.register_buffer("covalent_radii", _covalent_radii_table())

    def forward(self, r: Tensor, atomic_numbers: Tensor,
                edge_index: Tensor) -> Tensor:
        """Apply the Agnesi transform per edge.

        Parameters
        ----------
        r : Tensor
            Interatomic distances, shape ``(E,)``.
        atomic_numbers : Tensor
            Per-node atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``.

        Returns
        -------
        Tensor
            Transformed distances, shape ``(E,)``.
        """
        rcov = self.covalent_radii[atomic_numbers]
        r0 = 0.5 * (rcov[edge_index[0]] + rcov[edge_index[1]])
        u = r / r0
        return 1.0 / (1.0 + self.a * u ** self.q / (1.0 + u ** (self.q - self.p)))


class SoftDistanceTransform(nn.Module):
    """Tanh-based soft lower clamp of the interatomic distance.

    Smoothly interpolates between a floor of ``p0 = 3/4 r0`` at short range
    and the identity at long range,

        ``T(r) = p0 + (r - p0) * 1/2 (1 + tanh(alpha' (r - m)))``

    with ``r0`` the *sum* of the two elements' covalent radii,
    ``p1 = 4/3 r0``, midpoint ``m = (p0 + p1) / 2`` and steepness
    ``alpha' = alpha / (p1 - p0)``. This is the MACE
    ``distance_transform="Soft"`` option.

    Parameters
    ----------
    alpha : float, optional
        Dimensionless steepness of the switch, by default 4.0.
    """

    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.get_default_dtype()))
        self.register_buffer("covalent_radii", _covalent_radii_table())

    def forward(self, r: Tensor, atomic_numbers: Tensor,
                edge_index: Tensor) -> Tensor:
        """Apply the soft clamp per edge.

        Parameters
        ----------
        r : Tensor
            Interatomic distances, shape ``(E,)``.
        atomic_numbers : Tensor
            Per-node atomic numbers, shape ``(N,)``.
        edge_index : Tensor
            Edge index of shape ``(2, E)``.

        Returns
        -------
        Tensor
            Transformed distances, shape ``(E,)``.
        """
        rcov = self.covalent_radii[atomic_numbers]
        r0 = rcov[edge_index[0]] + rcov[edge_index[1]]
        p0 = 0.75 * r0
        p1 = r0 * (4.0 / 3.0)
        mid = 0.5 * (p0 + p1)
        switch = 0.5 * (1.0 + torch.tanh(self.alpha / (p1 - p0) * (r - mid)))
        return p0 + (r - p0) * switch


# Distance-transform registry, by the upstream MACE option spellings.
DISTANCE_TRANSFORMS = {
    "None": IdentityDistanceTransform,
    None: IdentityDistanceTransform,
    "Agnesi": AgnesiDistanceTransform,
    "Soft": SoftDistanceTransform,
}


class BesselRBF(nn.Module):
    """Bessel radial basis (NequIP/DimeNet-style), smoother & fewer functions.

    Expands an interatomic distance ``r`` into a set of ``n_rbf`` invariant
    radial features using the normalized sinc/Bessel functions
    ``prefactor * sin(n*pi*r/rc) / r`` for ``n = 1, ..., n_rbf``. Compared with
    a Gaussian basis this is smoother and needs fewer functions to cover the
    cutoff sphere.

    Parameters
    ----------
    n_rbf : int, optional
        Number of Bessel basis functions (output width). Default is 8.
    cutoff : float, optional
        Cutoff radius ``rc`` used to set the basis frequencies and the
        normalization. Default is 5.0.
    trainable : bool, optional
        If ``True``, the ``n * pi`` frequencies are a learnable
        :class:`torch.nn.Parameter` (the NequIP ``BesselBasis`` default);
        otherwise a fixed buffer (the MACE default). Default is ``False``.
    prefactor : float, optional
        Overall normalization factor. ``None`` (default) uses the
        DimeNet/MACE convention ``sqrt(2 / cutoff)``; the original NequIP
        uses ``2 / cutoff``.

    Attributes
    ----------
    freqs : Tensor
        The angular frequencies ``n * pi`` of shape ``(n_rbf,)`` -- a
        registered buffer, or a :class:`torch.nn.Parameter` when
        ``trainable``.
    norm : float
        The scalar normalization ``prefactor``.
    """

    def __init__(self, n_rbf: int = 8, cutoff: float = 5.0,
                 trainable: bool = False, prefactor: float | None = None):
        super().__init__()
        self.cutoff = cutoff
        freqs = math.pi * torch.arange(1, n_rbf + 1, dtype=torch.get_default_dtype())
        if trainable:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)
        self.norm = math.sqrt(2.0 / cutoff) if prefactor is None else float(prefactor)

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances into the Bessel radial basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``. Values are
            clamped to a small positive minimum to avoid division by zero.

        Returns
        -------
        Tensor
            The radial embedding of shape ``(..., n_rbf)``.
        """
        r = r.clamp(min=1e-8)[..., None]
        return self.norm * torch.sin(self.freqs * r / self.cutoff) / r
