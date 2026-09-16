"""Equivariant edge featurization for E(3) GNNs (NequIP / MACE / Allegro).

Computes, per edge, the geometric inputs the equivariant interaction blocks
need: the direction encoded as spherical harmonics Y_l(r_hat) (equivariant edge
attributes) and the length encoded with a smooth radial basis (invariant
weights for the tensor-product paths).

Requires e3nn:  pip install "xnn[gnn]"
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.featurizers import Featurizer, GaussianRBF
from .radial import BesselRBF
from .cutoff import PolynomialCutoff

try:
    from e3nn import o3
    _HAS_E3NN = True
except ModuleNotFoundError:
    o3 = None
    _HAS_E3NN = False


class SphericalHarmonicEdgeEmbedding(Featurizer):
    """Per-edge geometric featurizer for E(3)-equivariant GNNs.

    For every edge it produces two complementary encodings of the relative
    displacement vector: the direction encoded as real spherical harmonics
    ``Y_l(r_hat)`` (equivariant edge attributes feeding the tensor-product
    paths) and the interatomic distance encoded with a smooth radial basis
    multiplied by a polynomial cutoff envelope (invariant scalar weights).

    Parameters
    ----------
    l_max : int, optional
        Maximum spherical-harmonic degree ``l``. The spherical-harmonic
        irreps are ``o3.Irreps.spherical_harmonics(l_max)``. Default is 2.
    n_rbf : int, optional
        Number of radial basis functions (width of the invariant radial
        embedding). Default is 8.
    cutoff : float, optional
        Cutoff radius (in the length units of the coordinates) used by both
        the radial basis and the cutoff envelope. Default is 5.0.
    p : int, optional
        Polynomial degree of the :class:`PolynomialCutoff` envelope. Default
        is 6.
    radial_type : str, optional
        Radial basis to use: ``"bessel"`` for :class:`BesselRBF` or
        ``"gaussian"`` for :class:`~xnn.common.featurizers.GaussianRBF`.
        Default is ``"bessel"``.
    trainable_rbf : bool, optional
        Make the Bessel frequencies learnable (NequIP's ``BesselBasis``
        default). Only meaningful for ``radial_type="bessel"``. Default is
        ``False``.
    rbf_prefactor : float, optional
        Normalization prefactor of the Bessel basis. ``None`` (default) is the
        DimeNet/MACE convention ``sqrt(2/cutoff)``; NequIP uses ``2/cutoff``.
        Only meaningful for ``radial_type="bessel"``.

    Attributes
    ----------
    irreps_sh : o3.Irreps
        The e3nn irreps of the spherical-harmonic edge attributes.
    sph : o3.SphericalHarmonics
        The (normalized, component-normalization) spherical-harmonics module.
    rbf : torch.nn.Module
        The radial basis module (Bessel or Gaussian).
    envelope : PolynomialCutoff
        The smooth cutoff envelope applied to the radial embedding.

    Raises
    ------
    ImportError
        If e3nn is not installed.
    ValueError
        If ``radial_type`` is not ``"bessel"`` or ``"gaussian"``.
    """

    def __init__(self, l_max: int = 2, n_rbf: int = 8, cutoff: float = 5.0,
                 p: int = 6, radial_type: str = "bessel",
                 trainable_rbf: bool = False, rbf_prefactor: float | None = None):
        super().__init__()
        if not _HAS_E3NN:
            raise ImportError('e3nn is required: pip install "xnn[gnn]"')
        self.cutoff = cutoff
        self.l_max = l_max
        self.irreps_sh = o3.Irreps.spherical_harmonics(l_max)
        self.sph = o3.SphericalHarmonics(
            self.irreps_sh, normalize=True, normalization="component")
        if radial_type == "bessel":
            self.rbf = BesselRBF(n_rbf, cutoff, trainable=trainable_rbf,
                                 prefactor=rbf_prefactor)
        elif radial_type == "gaussian":
            self.rbf = GaussianRBF(n_rbf, cutoff)
        else:
            raise ValueError(f"unknown radial_type {radial_type!r}")
        self.envelope = PolynomialCutoff(cutoff, p=p)
        self.n_rbf = n_rbf

    @property
    def output_dim(self) -> int:
        """Width of the invariant scalar radial embedding.

        Returns
        -------
        int
            The number of radial basis functions ``n_rbf``.
        """
        return self.n_rbf  # scalar radial-embedding width

    def embed(self, vec: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """TorchScript-compatible core: edge vectors in, geometric features out.

        This is the tensor-only path used by the scriptable ``node_energy``
        model cores (LAMMPS/TorchScript deployment); :meth:`forward` reuses it.

        Parameters
        ----------
        vec : Tensor
            Edge displacement vectors of shape ``(E, 3)`` (already accounting
            for any periodic cell shifts).

        Returns
        -------
        tuple of (Tensor, Tensor, Tensor)
            ``(edge_length, edge_sh, edge_radial)`` -- the interatomic
            distances ``(E,)``, the spherical-harmonic edge attributes
            ``(E, irreps_sh.dim)`` and the enveloped radial embedding
            ``(E, n_rbf)``.
        """
        length = torch.linalg.norm(vec, dim=-1)
        edge_sh = self.sph(vec)                      # (E, irreps_sh.dim)
        radial = self.rbf(length) * self.envelope(length)[:, None]  # (E, n_rbf)
        return length, edge_sh, radial

    @torch.jit.ignore
    def forward(self, data: AtomicGraph) -> dict[str, Tensor]:
        """Compute the per-edge geometric features.

        Parameters
        ----------
        data : AtomicGraph
            The atomic graph; ``data.edge_vectors()`` supplies the relative
            displacement vector for each of the ``E`` edges.

        Returns
        -------
        dict[str, Tensor]
            Mapping with keys:

            - ``"edge_vec"`` : Tensor of shape ``(E, 3)``, the relative
              displacement vectors.
            - ``"edge_length"`` : Tensor of shape ``(E,)``, the interatomic
              distances.
            - ``"edge_sh"`` : Tensor of shape ``(E, irreps_sh.dim)``, the
              equivariant spherical-harmonic edge attributes ``Y_l(r_hat)``.
            - ``"edge_radial"`` : Tensor of shape ``(E, n_rbf)``, the invariant
              radial embedding scaled by the cutoff envelope.
        """
        vec = data.edge_vectors()
        length, edge_sh, radial = self.embed(vec)
        return {
            "edge_vec": vec,
            "edge_length": length,
            "edge_sh": edge_sh,
            "edge_radial": radial,
        }
