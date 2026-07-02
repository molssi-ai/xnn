"""Shared E(3)-equivariant building blocks (e3nn) for the GNN potentials.

Provides the gate-nonlinearity helper and the NequIP-style equivariant
convolution that NequIP and MACE build on. Requires e3nn.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from e3nn import o3
from e3nn.nn import Gate, FullyConnectedNet

from xnns.common.models.ops import scatter_sum


def make_gate(irreps_out: o3.Irreps) -> Gate:
    """Build a gated nonlinearity producing exactly ``irreps_out``.

    Scalar (``l == 0``) irreps pass through a SiLU nonlinearity, while
    higher-``l`` (equivariant) irreps are gated by additional sigmoid-activated
    scalar gates -- preserving equivariance. The resulting Gate's *input* irreps
    (``gate.irreps_in``), which include those extra gate scalars, is what an
    upstream ``o3.Linear`` must produce.

    Parameters
    ----------
    irreps_out : o3.Irreps
        The desired output irreps of the nonlinearity (what the gate emits).

    Returns
    -------
    e3nn.nn.Gate
        A gate module whose ``irreps_out`` equals ``irreps_out`` and whose
        ``irreps_in`` is the irreps an upstream linear layer must supply.
    """
    scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
    gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
    n_gated = sum(mul for mul, _ in gated)
    gates = o3.Irreps([(n_gated, (0, 1))]) if n_gated > 0 else o3.Irreps([])
    return Gate(
        scalars, [F.silu] * len(scalars),
        gates, [torch.sigmoid] * len(gates),
        gated,
    )


class EquivariantConv(nn.Module):
    """NequIP-style equivariant convolution: (node feat) (x) Y_l(r_ij).

    Forms equivariant messages by taking a tensor product of neighbour node
    features with the edge spherical harmonics, weighted by a radial MLP, then
    aggregates over neighbours, mixes with an equivariant self-connection, and
    applies a gated nonlinearity::

        message_ij = TP(linear(h_j), edge_sh; w(radial_ij))
        h_i' = gate( linear(sum_j message_ij) + self_connection(h_i, node_attr) )

    Parameters
    ----------
    irreps_in : o3.Irreps
        Irreps of the input node features ``h``.
    irreps_sh : o3.Irreps
        Irreps of the spherical-harmonic edge attributes ``Y_l(r_ij)``.
    irreps_out : o3.Irreps
        Desired irreps of the output node features (the gate's output irreps).
    irreps_node_attr : o3.Irreps
        Irreps of the invariant node attributes (one-hot species) feeding the
        self-connection.
    n_radial : int
        Width of the invariant radial embedding input to the radial weight MLP.
    radial_hidden : tuple of int, optional
        Hidden layer widths of the radial MLP that generates the tensor-product
        path weights. Default is ``(64, 64)``.

    Attributes
    ----------
    gate : e3nn.nn.Gate
        The gated nonlinearity; ``gate.irreps_out`` defines ``irreps_out``.
    linear_in : o3.Linear
        Equivariant linear applied to node features before the tensor product.
    tp : o3.TensorProduct
        The path-restricted (``uvu``) tensor product node (x) sh with external,
        non-shared weights.
    irreps_mid : o3.Irreps
        Irreps of the tensor-product output (intermediate messages).
    radial : e3nn.nn.FullyConnectedNet
        MLP mapping the radial embedding to the tensor-product path weights.
    linear_out : o3.Linear
        Equivariant linear from the aggregated messages to the gate input.
    self_connection : o3.FullyConnectedTensorProduct
        Equivariant self-interaction combining node features and node
        attributes.
    irreps_out : o3.Irreps
        The output irreps of the convolution (equal to ``gate.irreps_out``).
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_sh: o3.Irreps,
                 irreps_out: o3.Irreps, irreps_node_attr: o3.Irreps,
                 n_radial: int, radial_hidden=(64, 64)):
        super().__init__()
        self.gate = make_gate(irreps_out)
        irreps_gate_in = self.gate.irreps_in

        self.linear_in = o3.Linear(irreps_in, irreps_in)

        # build tensor-product paths node (x) sh -> intermediate (uvu)
        irreps_mid, instructions = [], []
        for i, (mul, ir_in) in enumerate(irreps_in):
            for j, (_, ir_sh) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_sh:
                    if ir_out in [ir for _, ir in irreps_gate_in]:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        self.irreps_mid = o3.Irreps(irreps_mid)
        self.tp = o3.TensorProduct(
            irreps_in, irreps_sh, self.irreps_mid, instructions,
            shared_weights=False, internal_weights=False)
        self.radial = FullyConnectedNet(
            [n_radial, *radial_hidden, self.tp.weight_numel], F.silu)
        self.linear_out = o3.Linear(self.irreps_mid, irreps_gate_in)
        self.self_connection = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_node_attr, irreps_gate_in)
        self.irreps_out = self.gate.irreps_out

    def forward(self, x: Tensor, node_attr: Tensor, edge_index: Tensor,
                edge_sh: Tensor, edge_radial: Tensor) -> Tensor:
        """Apply one equivariant message-passing update.

        Parameters
        ----------
        x : Tensor
            Node features of shape ``(N, irreps_in.dim)``.
        node_attr : Tensor
            Invariant one-hot species node attributes of shape
            ``(N, irreps_node_attr.dim)``.
        edge_index : Tensor
            Long tensor of shape ``(2, E)``; row 0 is the source (neighbour)
            index and row 1 the destination (centre) index of each edge.
        edge_sh : Tensor
            Spherical-harmonic edge attributes of shape ``(E, irreps_sh.dim)``.
        edge_radial : Tensor
            Invariant radial embedding of shape ``(E, n_radial)`` used to
            generate the tensor-product path weights.

        Returns
        -------
        Tensor
            Updated node features of shape ``(N, irreps_out.dim)``.
        """
        src, dst = edge_index[0], edge_index[1]
        x_in = self.linear_in(x)
        w = self.radial(edge_radial)
        msg = self.tp(x_in[src], edge_sh, w)
        agg = scatter_sum(msg, dst, x.shape[0])
        out = self.linear_out(agg) + self.self_connection(x, node_attr)
        return self.gate(out)


def species_irreps(n_species: int) -> o3.Irreps:
    """Irreps of the one-hot species (node attribute) channels.

    Parameters
    ----------
    n_species : int
        Number of distinct chemical species.

    Returns
    -------
    o3.Irreps
        ``n_species`` even scalar irreps, i.e. ``o3.Irreps([(n_species, (0,
        1))])`` (written ``{n_species}x0e``).
    """
    return o3.Irreps([(n_species, (0, 1))])


def hidden_irreps(mul: int, l_max: int) -> o3.Irreps:
    """Hidden feature irreps: ``mul`` copies of the spherical-harmonic irreps.

    Parameters
    ----------
    mul : int
        Multiplicity (number of channels) per irrep degree.
    l_max : int
        Maximum degree ``l``; the degrees/parities follow
        ``o3.Irreps.spherical_harmonics(l_max)`` (i.e. ``0e, 1o, 2e, ...``).

    Returns
    -------
    o3.Irreps
        ``mul`` copies of each spherical-harmonic irrep up to ``l_max``.
    """
    return o3.Irreps([(mul, ir) for _, ir in o3.Irreps.spherical_harmonics(l_max)])
