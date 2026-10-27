"""Radial basis expansions shared across families: the Gaussian basis of SchNet and the
equivariant GNN edge embedding, the exponential-normal basis of the TorchMD-Net models
(ET, TensorNet) and BAMBOO, and the Bessel basis of NequIP, MACE, DimeNet and the Equiformers."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .cutoff import CosineCutoff


class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion (SchNet-style).

    Expands scalar interatomic distances onto a set of ``n_rbf`` Gaussians
    ``e_k(r) = exp(-gamma (r - mu_k)^2)`` whose centers ``mu_k`` are fixed and
    evenly spaced on ``[start, cutoff]``. Registered as a non-trainable buffer.

    Parameters
    ----------
    n_rbf : int, optional
        Number of Gaussian basis functions (centers). Defaults to ``50``.
    cutoff : float, optional
        Upper bound of the center range, in the same units as the distances.
        Defaults to ``5.0``.
    gamma : float or None, optional
        Width parameter of the Gaussians. ``None`` (default) sets the standard
        deviation to the spacing between adjacent centers, i.e.
        ``gamma = 0.5 / spacing**2``. SchNet (Schuett et al., NIPS 2017) fixes
        ``gamma = 10`` per Angstrom^2 on a 0.1 Angstrom center grid.
    start : float, optional
        Position of the first center, by default ``0.0``.
    endpoint : bool, optional
        Whether the last center sits at ``cutoff`` (default) or the grid stops
        one spacing short of it, ``mu_k = start + k (cutoff - start) / n_rbf``
        (the AIMNet2 layout, whose Gaussians all keep some weight inside the
        cutoff envelope).

    Notes
    -----
    With ``gamma=None`` the width is the spacing between adjacent centers, or
    ``1.0`` when ``n_rbf == 1``.
    """

    def __init__(self, n_rbf: int = 50, cutoff: float = 5.0,
                 gamma: float | None = None, start: float = 0.0,
                 endpoint: bool = True):
        super().__init__()
        if endpoint:
            centers = torch.linspace(start, cutoff, n_rbf)
        else:
            centers = torch.linspace(start, cutoff, n_rbf + 1)[:n_rbf]
        self.register_buffer("centers", centers)
        self.width = (centers[1] - centers[0]).item() if n_rbf > 1 else 1.0
        self.gamma = float(gamma) if gamma is not None \
            else 0.5 / self.width ** 2

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances onto the Gaussian basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The radial basis expansion of shape ``(..., n_rbf)``.
        """
        diff = r[..., None] - self.centers
        return torch.exp(-self.gamma * diff ** 2)


class ExpNormalSmearing(nn.Module):
    r"""Exponential-normal radial basis (TorchMD-Net ET / TensorNet, BAMBOO).

    Expands a distance ``r`` onto ``n_rbf`` Gaussians placed in exponential
    distance space, dense at short range and sparse near the cutoff, times a
    cosine cutoff envelope::

        phi_k(r) = cos_cutoff(r) *
                   exp(-beta_k * (exp(alpha * (cutoff_lower - r)) - mu_k) ** 2)

    with ``alpha = 5 / (cutoff - cutoff_lower)``. The centers ``mu_k`` run
    linearly from ``exp(cutoff_lower - cutoff)`` to ``1`` and every width
    starts at ``(2 / n_rbf * (1 - exp(cutoff_lower - cutoff))) ** -2``
    (Thoelke & De Fabritiis 2022, eq 1, the PhysNet convention).

    Parameters
    ----------
    n_rbf : int, optional
        Number of basis functions, by default 32.
    cutoff : float, optional
        Upper cutoff in the units of the distances, by default 5.0.
    cutoff_lower : float, optional
        Lower cutoff, by default 0.0.
    trainable : bool, optional
        Learn the centers and widths (``True``, the default) or keep them as
        buffers.

    Attributes
    ----------
    means : Tensor
        The centers ``mu_k``, shape ``(n_rbf,)``.
    betas : Tensor
        The inverse widths ``beta_k``, shape ``(n_rbf,)``.
    alpha : float
        The exponential-space scale ``5 / (cutoff - cutoff_lower)``.
    """

    def __init__(self, n_rbf: int = 32, cutoff: float = 5.0,
                 cutoff_lower: float = 0.0, trainable: bool = True):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff
        self.trainable = trainable
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / (self.cutoff_upper - self.cutoff_lower)

        if trainable:
            self.means = nn.Parameter(torch.empty(n_rbf))
            self.betas = nn.Parameter(torch.empty(n_rbf))
        else:
            self.register_buffer("means", torch.empty(n_rbf))
            self.register_buffer("betas", torch.empty(n_rbf))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the centers and widths to their default values in place."""
        first_center = math.exp(self.cutoff_lower - self.cutoff_upper)
        shared_width = (2.0 / self.n_rbf * (1.0 - first_center)) ** -2
        with torch.no_grad():
            self.means.copy_(torch.linspace(first_center, 1.0, self.n_rbf))
            self.betas.fill_(shared_width)

    def forward(self, r: Tensor) -> Tensor:
        """Expand distances onto the basis.

        Parameters
        ----------
        r : Tensor
            Interatomic distances of arbitrary shape ``(...)``.

        Returns
        -------
        Tensor
            The expansion of shape ``(..., n_rbf)``, zero at and beyond the
            cutoff.
        """
        r = r.unsqueeze(-1)
        # the distance mapped to exponential space, where the centers live
        u = torch.exp(self.alpha * (self.cutoff_lower - r))
        return self.cutoff_fn(r) * torch.exp(-self.betas * (u - self.means) ** 2)


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

    def exact_constants(self) -> bool:
        """Rebuild fixed float64 frequencies as the exact ``n * pi``.

        A float32 build or checkpoint leaves them float32-rounded; trainable
        frequencies are learned values and stay, as do stored ones that are not
        ``n * pi`` up to rounding. See
        :func:`~xnn.common.models.constants.exact_float64_constants`.

        Returns
        -------
        bool
            Whether any value changed.
        """
        from ..models.constants import restore_exact
        freqs = self.freqs
        if isinstance(freqs, nn.Parameter) or freqs.dtype != torch.float64:
            return False
        return restore_exact(freqs, math.pi * torch.arange(1, freqs.numel() + 1,
                                                           dtype=torch.float64))

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


#: The radial bases a model with a choice of basis can draw from (:func:`make_radial_basis`).
RADIAL_BASES = ("gaussian", "bessel", "expnorm")


def make_radial_basis(name: str, n_rbf: int, cutoff: float, trainable: bool = False, *,
                      gamma: float | None = None, prefactor: float | None = None) -> nn.Module:
    """A radial basis by name: the bases shared by every model that lets the basis be chosen.

    An option the chosen basis does not have is ignored with an
    :class:`~xnn.common.config.options.IgnoredOptionWarning`.

    Parameters
    ----------
    name : str
        ``"gaussian"`` (:class:`GaussianRBF`: ``n_rbf`` Gaussians centered on
        ``[0, cutoff]``, the center spacing as width unless ``gamma`` is
        given; the centers stay exact in float64 through casts and loads),
        ``"bessel"`` (:class:`BesselRBF`, ``prefactor * sin(n pi r / cutoff) / r``)
        or ``"expnorm"`` (:class:`ExpNormalSmearing`, Gaussians in
        ``exp(-r)`` with their own cosine envelope).
    n_rbf : int
        Number of basis functions.
    cutoff : float
        The cutoff radius.
    trainable : bool, optional
        Learn the parameters of the basis (the Bessel frequencies, the
        exponential-normal centers and widths); the Gaussian basis has none.
    gamma : float or None, optional
        The exponent of the Gaussians, ``exp(-gamma (r - mu)^2)``.
    prefactor : float or None, optional
        The normalization of the Bessel functions (``None``: ``sqrt(2 / cutoff)``).

    Returns
    -------
    torch.nn.Module
        Maps distances ``(...)`` to ``(..., n_rbf)``.

    Raises
    ------
    ValueError
        For an unknown ``name``.
    """
    if name not in RADIAL_BASES:
        raise ValueError(f"radial_basis must be one of {RADIAL_BASES}, got {name!r}")
    from ..config.options import warn_ignored
    given = {"trainable": trainable and name == "gaussian",
             "gamma": gamma is not None and name != "gaussian",
             "prefactor": prefactor is not None and name != "bessel"}
    warn_ignored(f"radial basis {name!r}", [k for k, ignored in given.items() if ignored])
    if name == "gaussian":
        from ..models.constants import register_constant
        if gamma is None and n_rbf > 1:          # the center spacing as width, in double precision
            gamma = 0.5 * ((n_rbf - 1) / cutoff) ** 2
        basis = GaussianRBF(n_rbf, cutoff, gamma=gamma)
        register_constant(basis, "centers", torch.linspace(0.0, cutoff, n_rbf, dtype=torch.float64))
        return basis
    if name == "bessel":
        return BesselRBF(n_rbf, cutoff, trainable=trainable, prefactor=prefactor)
    return ExpNormalSmearing(n_rbf, cutoff, trainable=trainable)
