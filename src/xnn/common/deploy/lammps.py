"""Export a trained model to TorchScript for LAMMPS.

LAMMPS coupling follows the established pattern used by pair_nequip / pair_mace
/ pair_allegro: the trained model is serialized with ``torch.jit.script`` and a
C++ pair style loads it and feeds it the local neighbor list each timestep.

``LAMMPSWrapper`` exposes a forward whose inputs mirror what such a pair style
provides -- positions, edge list, integer shifts, atom types and the cell -- and
returns total energy, per-atom energy and forces. Pair the produced ``.pt`` file
with the matching C++ pair style for your model family.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# (no dataclass import needed: deploy wrapper is tensor-only)


class LAMMPSWrapper(nn.Module):
    """TorchScript-friendly bridge between LAMMPS tensors and the model.

    Operates purely on tensors (no ``AtomicGraph`` dataclass) so the module is
    ``torch.jit.script``-able, which is a hard requirement for use inside a
    LAMMPS pair style. Models expose a scriptable ``node_energy(...)`` core for
    exactly this purpose.

    This wrapper defines the tensor ABI (application binary interface) that the
    LAMMPS pair style exchanges with the model: the argument order, dtypes and
    shapes of :meth:`forward` are the contract between the C++ pair style and
    the serialized model, so they must not drift from what the pair style
    provides and consumes.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model. Must expose a scriptable
        ``node_energy(atomic_numbers, edge_index, edge_vec)`` method returning
        per-atom (node) energies.
    cutoff : float
        Neighbor-list cutoff radius, stored (as a Python ``float``) for
        serialization alongside the scripted module.

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped model.
    cutoff : float
        The neighbor-list cutoff radius.
    """

    def __init__(self, model: nn.Module, cutoff: float):
        super().__init__()
        self.model = model
        self.cutoff = float(cutoff)

    def forward(self, pos: Tensor, edge_index: Tensor, cell_shifts: Tensor,
                atomic_numbers: Tensor, cell: Tensor) -> dict[str, Tensor]:
        """Compute energy and forces from LAMMPS-provided neighbor data.

        Edge vectors are reconstructed as ``pos[dst] - pos[src]`` plus the
        periodic image shift ``cell_shifts @ cell``. The model's per-atom
        (node) energies are summed to the total energy, and forces are the
        negative gradient of that total energy with respect to the (grad-
        enabled) positions.

        Parameters
        ----------
        pos : Tensor
            Atomic positions, shape ``(n_atoms, 3)``.
        edge_index : Tensor
            Neighbor (edge) list, shape ``(2, n_edges)``; row 0 is the source
            index and row 1 is the destination index of each edge.
        cell_shifts : Tensor
            Integer periodic-image shift vectors per edge, shape
            ``(n_edges, 3)``; combined with ``cell`` to place neighbors in
            their correct periodic images.
        atomic_numbers : Tensor
            Per-atom atomic numbers (species), shape ``(n_atoms,)``.
        cell : Tensor
            Simulation cell (lattice) matrix, shape ``(3, 3)``.

        Returns
        -------
        dict of str to Tensor
            A dictionary with keys ``"total_energy"`` (shape ``(1,)``),
            ``"node_energy"`` (per-atom energies, shape ``(n_atoms,)``) and
            ``"forces"`` (shape ``(n_atoms, 3)``). Forces are zero when the
            autograd gradient is unavailable.
        """
        pos = pos.detach().requires_grad_(True)
        src = edge_index[0]
        dst = edge_index[1]
        edge_vec = pos[dst] - pos[src] + torch.mm(cell_shifts.to(pos.dtype), cell)
        node_energy = self.model.node_energy(atomic_numbers, edge_index, edge_vec)
        energy = node_energy.sum()
        # allow_unused: a pure reference-energy model (e.g. MACE T=0) has no
        # position dependence; the None gradient below then maps to zero forces
        grad = torch.autograd.grad([energy], [pos], create_graph=False,
                                   allow_unused=True)[0]
        forces = -grad if grad is not None else torch.zeros_like(pos)
        return {
            "total_energy": energy.reshape(1),
            "node_energy": node_energy,
            "forces": forces,
        }


def export_to_lammps(model: nn.Module, cutoff: float, path: str,
                     metadata: dict | None = None) -> str:
    """Script the model and save a ``.pt`` usable by a LAMMPS pair style.

    Wraps ``model`` in :class:`LAMMPSWrapper`, compiles it with
    ``torch.jit.script`` and saves the scripted module to ``path``. The cutoff
    (and any provided metadata) is embedded as extra files inside the archive,
    with all values stored as strings.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model to serialize (must provide a scriptable
        ``node_energy`` core, as required by :class:`LAMMPSWrapper`).
    cutoff : float
        Neighbor-list cutoff radius; stored in the archive under the
        ``"cutoff"`` extra-file key.
    path : str
        Destination file path for the scripted ``.pt`` archive.
    metadata : dict or None, optional
        Additional key/value metadata to embed as extra files. Values are
        stringified. Defaults to ``None``.

    Returns
    -------
    str
        The ``path`` the scripted model was saved to.
    """
    wrapper = LAMMPSWrapper(model, cutoff).eval()
    scripted = torch.jit.script(wrapper)
    extra = {"cutoff": str(cutoff)}
    if metadata:
        extra.update({k: str(v) for k, v in metadata.items()})
    scripted.save(path, _extra_files=extra)
    return path


def export_torchscript(model: nn.Module, path: str) -> str:
    """Generic TorchScript export (e.g. for custom C++/Python serving).

    Compiles the model directly with ``torch.jit.script`` (without the LAMMPS
    tensor-ABI wrapper) and saves the scripted module to ``path``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to serialize. It is put in ``eval`` mode before scripting.
    path : str
        Destination file path for the scripted ``.pt`` archive.

    Returns
    -------
    str
        The ``path`` the scripted model was saved to.
    """
    scripted = torch.jit.script(model.eval())
    scripted.save(path)
    return path
