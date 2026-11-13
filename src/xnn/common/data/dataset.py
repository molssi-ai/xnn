"""Datasets and graph batching.

`AtomicDataset` turns a list of plain structure dicts into `AtomicGraph`
single-structure objects (computing the neighbor list once, cached). `collate`
concatenates several `AtomicGraph`s into one batched `AtomicGraph` -- this is
what enables optional *batch training*: set ``batch_size=1`` to disable it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .atomic_data import AtomicGraph
from .neighborlist import build_neighbor_list


def structure_to_graph(
    s: dict, cutoff: float, device: Optional[torch.device] = None
) -> AtomicGraph:
    """Build a single-structure :class:`AtomicGraph` from a dict of arrays.

    Values may be array-likes or tensors; they are coerced to tensors. The
    neighbor list is computed via :func:`build_neighbor_list`. When ``cell`` is
    provided but ``pbc`` is not, full periodicity is assumed.

    Parameters
    ----------
    s : dict
        Structure data. Required keys: ``pos`` ``(N, 3)`` and
        ``atomic_numbers`` ``(N,)``. Optional keys: ``cell`` ``(3, 3)``,
        ``pbc`` ``(3,)``, ``energy`` (scalar), ``forces`` ``(N, 3)``,
        ``stress`` ``(3, 3)``, ``dipole`` ``(3,)`` (the molecular dipole
        moment; ``dipole_moment`` is a synonym), ``polarizability``
        ``(3, 3)``, ``charges`` ``(N,)`` (reference partial charges, the
        targets of the HDNNP charge networks), ``total_charge`` (scalar net
        charge; the key ``charge`` is accepted as a synonym), ``fragment_charges``
        ``(N,)`` (the net charge of the fragment each atom belongs to, for the
        LES charge solve), ``spin_multiplicity``
        (scalar ``2S + 1``; ``multiplicity`` is a synonym), ``weight`` (scalar
        per-structure loss weight; see
        :func:`~xnn.common.train.losses.weighted_loss`), ``head`` (integer
        index of the readout head, for multi-head models; see
        :func:`~xnn.common.finetune.label_head`), ``properties`` (scalar or
        ``(k,)``, an intensive property target such as a band gap),
        ``state`` (an integer label such as the data fidelity, or a ``(d,)``
        feature vector; the global state of MEGNet), ``occupancies`` (one
        ``{element: fraction}`` mapping per site, elements as symbols or atomic
        numbers: disordered sites for MEGNet; ``atomic_numbers`` may then be
        left out and default to each site's largest occupant), the per-structure term
        weights ``energy_weight`` / ``force_weight`` / ``stress_weight``
        (scalars) and ``atom_force_weight`` ``(N,)`` (the weight of every
        atom's force error).
    cutoff : float
        Neighbor cutoff radius passed to the neighbor list builder.
    device : torch.device, optional
        Device to build the graph on. The neighbor list is the expensive part
        of this function, so building it where the model already is avoids
        paying for it on the CPU and copying the result across. Defaults to
        ``None``, which builds on the CPU as before -- the right choice when
        the graph is being cached by a dataset rather than fed straight to a
        model.

    Returns
    -------
    AtomicGraph
        A single-structure graph (``batch`` all zeros, ``n_atoms`` of length
        one), with optional target fields populated when present in ``s``.
        Floating-point fields take the default dtype, except ``energy``, which
        is always float64.
    """
    def t(x, dtype=torch.get_default_dtype()):
        """Coerce ``x`` to a tensor, leaving existing tensors untouched.

        Parameters
        ----------
        x : Tensor or array_like
            Value to convert.
        dtype : torch.dtype, optional
            Target dtype used only when ``x`` is not already a tensor;
            defaults to the current PyTorch default dtype.

        Returns
        -------
        Tensor
            ``x`` as a tensor.
        """
        out = x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=dtype)
        return out if device is None else out.to(device)

    pos = t(s["pos"])
    sites = _site_mixture(s.get("occupancies"), pos.shape[0])
    if s.get("atomic_numbers") is not None:
        z = t(s["atomic_numbers"], dtype=torch.long)
    elif sites is not None:
        z = t(sites[0][np.arange(len(sites[0])), sites[1].argmax(1)], dtype=torch.long)
    else:
        raise KeyError("a structure needs atomic_numbers (or occupancies)")
    cell = t(s["cell"]).reshape(3, 3) if s.get("cell") is not None else None
    pbc = (torch.as_tensor(s["pbc"], dtype=torch.bool)
           if s.get("pbc") is not None else
           (torch.ones(3, dtype=torch.bool) if cell is not None else None))

    edge_index, cell_shifts = build_neighbor_list(pos, cutoff, cell, pbc)

    n = pos.shape[0]
    charge = s.get("total_charge", s.get("charge"))
    mult = s.get("spin_multiplicity", s.get("multiplicity"))
    dipole = s.get("dipole", s.get("dipole_moment"))
    alpha = s.get("polarizability")
    prop = s.get("properties")
    state = s.get("state")
    if state is not None:
        # an integer label stays one (an embedding index); features take the default dtype
        integer = (state.dtype in (torch.int32, torch.int64) if isinstance(state, Tensor)
                   else np.issubdtype(np.asarray(state).dtype, np.integer))
        state = (t(state, dtype=torch.long).reshape(1) if integer
                 else t(state).reshape(1, -1))
    return AtomicGraph(
        pos=pos,
        atomic_numbers=z,
        edge_index=edge_index,
        cell_shifts=cell_shifts,
        batch=torch.zeros(n, dtype=torch.long, device=device),
        n_atoms=torch.tensor([n], dtype=torch.long, device=device),
        cell=cell[None] if cell is not None else None,
        pbc=t(pbc)[None] if pbc is not None else None,
        # a total energy stays float64 whatever the default dtype, like the
        # predicted totals (structure_sum); float32 rounds large totals
        energy=(t(torch.as_tensor(s["energy"], dtype=torch.float64)).reshape(1)
                if s.get("energy") is not None else None),
        forces=t(s["forces"]) if s.get("forces") is not None else None,
        stress=t(s["stress"]).reshape(1, 3, 3) if s.get("stress") is not None else None,
        dipole=t(dipole).reshape(1, 3) if dipole is not None else None,
        polarizability=t(alpha).reshape(1, 3, 3) if alpha is not None else None,
        charges=t(s["charges"]).reshape(n) if s.get("charges") is not None else None,
        fragment_charges=(t(s["fragment_charges"]).reshape(n)
                          if s.get("fragment_charges") is not None else None),
        total_charge=(t([charge]).reshape(1) if charge is not None else None),
        spin_multiplicity=(t([mult]).reshape(1) if mult is not None else None),
        weight=(t([s["weight"]]).reshape(1) if s.get("weight") is not None else None),
        head=(t([int(s["head"])], dtype=torch.long).reshape(1)
              if s.get("head") is not None else None),
        properties=(t(prop).reshape((1,) + tuple(np.shape(prop))) if prop is not None else None),
        state=state,
        site_species=t(sites[0], dtype=torch.long) if sites is not None else None,
        site_occupancy=t(sites[1]) if sites is not None else None,
        energy_weight=(t([s["energy_weight"]]).reshape(1)
                       if s.get("energy_weight") is not None else None),
        force_weight=(t([s["force_weight"]]).reshape(1)
                      if s.get("force_weight") is not None else None),
        stress_weight=(t([s["stress_weight"]]).reshape(1)
                       if s.get("stress_weight") is not None else None),
        atom_force_weight=(t(s["atom_force_weight"]).reshape(n)
                           if s.get("atom_force_weight") is not None else None),
    )


def _site_mixture(occupancies, n: int):
    """The ``(species, occupancy)`` arrays ``(n, K)`` of per-site ``{element: fraction}`` mappings, or ``None``.

    Raises
    ------
    ValueError
        For a site count other than ``n``, negative occupancies or a site whose occupancies sum above 1.
    """
    if occupancies is None:
        return None
    from .elements import SYMBOL_TO_Z
    if len(occupancies) != n:
        raise ValueError(f"got occupancies for {len(occupancies)} sites and {n} positions")
    k = max(len(site) for site in occupancies)
    species, weight = np.zeros((n, k), dtype=np.int64), np.zeros((n, k))
    for i, site in enumerate(occupancies):
        for j, (element, x) in enumerate(site.items()):
            species[i, j] = SYMBOL_TO_Z[element] if isinstance(element, str) else int(element)
            weight[i, j] = float(x)
    if (weight < 0).any() or (weight.sum(1) > 1 + 1e-6).any():
        raise ValueError("site occupancies must be non-negative and sum to at most 1")
    return species, weight


def _site_rows(graphs: list) -> tuple:
    """The batch's ``site_species`` and ``site_occupancy`` padded to its widest site, or ``(None, None)`` when
    every structure is ordered."""
    if all(g.site_species is None for g in graphs):
        return None, None
    width = max(g.site_species.shape[1] for g in graphs if g.site_species is not None)
    species, occupancy = [], []
    for g in graphs:
        if g.site_species is None:
            sp = g.atomic_numbers[:, None]
            oc = torch.ones(sp.shape, dtype=g.pos.dtype, device=sp.device)
        else:
            sp, oc = g.site_species, g.site_occupancy
        pad = width - sp.shape[1]
        species.append(torch.nn.functional.pad(sp, (0, pad)))
        occupancy.append(torch.nn.functional.pad(oc.to(g.pos.dtype), (0, pad)))
    return torch.cat(species, 0), torch.cat(occupancy, 0)


def energy_labelled(structures) -> list[dict]:
    """The structure dicts whose energy enters the loss.

    A structure is kept when it has an ``energy`` and neither its ``weight``
    nor its ``energy_weight`` is zero. Reference-energy (E0) fits use this, so
    energies excluded from the loss do not shift the references either.

    Parameters
    ----------
    structures : iterable of dict
        Structure dicts.

    Returns
    -------
    list of dict
        The energy-labelled structures, in their original order.
    """
    def nonzero(s: dict, key: str) -> bool:
        return s.get(key) is None or float(s[key]) != 0

    return [s for s in structures if s.get("energy") is not None
            and nonzero(s, "weight") and nonzero(s, "energy_weight")]


class AtomicDataset(Dataset):
    """PyTorch ``Dataset`` wrapping a list of structure dicts.

    Each structure is converted to a single-structure :class:`AtomicGraph` via
    :func:`structure_to_graph` on first access and cached, so the neighbor list
    for a given index is only computed once.

    Parameters
    ----------
    structures : list of dict
        Structure dicts, each in the format accepted by
        :func:`structure_to_graph`.
    cutoff : float
        Neighbor cutoff radius used when building each graph.

    Attributes
    ----------
    structures : list of dict
        The wrapped structure dicts.
    cutoff : float
        Neighbor cutoff radius.
    """

    def __init__(self, structures: list[dict], cutoff: float):
        self.structures = structures
        self.cutoff = cutoff
        self._cache: dict[int, AtomicGraph] = {}

    @classmethod
    def from_file(cls, path: str, cutoff: float, index: str = ":",
                  **target_keys) -> "AtomicDataset":
        """Build a dataset from a structure file readable by ASE.

        Any ASE-readable format works (``.xyz`` / ``.extxyz`` / ``.cif`` /
        VASP / ...); frames are converted via
        :func:`~xnn.common.data.ase_io.load_structures`, picking up energy /
        forces / stress targets when the file carries them. Requires the
        ``ase`` extra.

        Parameters
        ----------
        path : str
            Path to the structure file.
        cutoff : float
            Neighbor cutoff radius used when building each graph.
        index : str, optional
            Frame selection passed to :func:`ase.io.read`; the default ``":"``
            loads all frames.
        **target_keys
            ``energy_key`` / ``forces_key`` / ``stress_key`` overrides for
            files that store targets under non-standard names (e.g.
            ``energy_key="REF_energy"``); see
            :func:`~xnn.common.data.ase_io.atoms_to_structure`.

        Returns
        -------
        AtomicDataset
            Dataset over all selected frames.
        """
        from .ase_io import load_structures
        return cls(load_structures(path, index, **target_keys), cutoff)

    @classmethod
    def from_atoms(cls, atoms, cutoff: float, **target_keys) -> "AtomicDataset":
        """Build a dataset from ASE ``Atoms`` object(s) already in memory.

        Parameters
        ----------
        atoms : ase.Atoms or list of ase.Atoms
            Structure(s) to convert, via
            :func:`~xnn.common.data.ase_io.atoms_to_structure`.
        cutoff : float
            Neighbor cutoff radius used when building each graph.
        **target_keys
            ``energy_key`` / ``forces_key`` / ``stress_key`` overrides; see
            :func:`~xnn.common.data.ase_io.atoms_to_structure`.

        Returns
        -------
        AtomicDataset
            Dataset over the given structure(s).
        """
        from .ase_io import atoms_to_structure
        if not isinstance(atoms, (list, tuple)):
            atoms = [atoms]
        return cls([atoms_to_structure(a, **target_keys) for a in atoms], cutoff)

    def __len__(self) -> int:
        """Return the number of structures in the dataset.

        Returns
        -------
        int
            Number of wrapped structures.
        """
        return len(self.structures)

    def __getitem__(self, idx: int) -> AtomicGraph:
        """Return the :class:`AtomicGraph` for structure ``idx``.

        The graph is built on first access and cached for subsequent calls.

        Parameters
        ----------
        idx : int
            Index of the structure.

        Returns
        -------
        AtomicGraph
            The single-structure graph at ``idx``.
        """
        if idx not in self._cache:
            self._cache[idx] = structure_to_graph(self.structures[idx], self.cutoff)
        return self._cache[idx]


def collate(graphs: list[AtomicGraph]) -> AtomicGraph:
    """Concatenate graphs into one batched graph.

    Nodes and edges are concatenated along their respective axes, ``edge_index``
    entries are offset by the running node count, and a ``batch`` vector mapping
    each node to its structure index is built. Using ``batch_size=1``
    effectively disables batching.

    A batch may mix any structures:

    * molecular and periodic ones: when any structure has a cell, the batch
      keeps ``cell`` / ``pbc`` and a molecular structure gets a zero cell and no
      periodic flag (its edges carry no image shift, so every edge keeps its
      length, and its stress is zero);
    * structures with and without energy, force, stress, dipole,
      polarizability, partial-charge or property labels: the missing ones are
      zeros, and ``energy_mask`` / ``forces_mask`` / ``stress_mask`` /
      ``dipole_mask`` / ``polarizability_mask`` / ``charges_mask`` /
      ``properties_mask`` (per structure) mark the labelled ones for the loss
      and the metrics;
    * structures with and without ``state`` (missing = label 0 or zero
      features);
    * ordered structures and structures with disordered sites: every site
      becomes a padded row of ``site_species`` / ``site_occupancy``, an
      ordered one its own species at occupancy 1;
    * structures with and without ``total_charge`` (missing = neutral),
      ``spin_multiplicity`` (missing = 1), ``weight``, ``energy_weight``,
      ``force_weight``, ``stress_weight``, ``atom_force_weight`` (missing =
      1) or ``head`` (missing = 0, the first head).

    Parameters
    ----------
    graphs : list of AtomicGraph
        Graphs (single structures or batches) to concatenate.

    Returns
    -------
    AtomicGraph
        One batched graph with ``num_graphs`` equal to the total number of
        structures.
    """
    pos, z, batch, n_atoms = [], [], [], []
    edge_index, cell_shifts = [], []
    cells, pbcs = [], []
    energies, forces, stresses, charges, mults, weights = [], [], [], [], [], []
    dipoles, alphas, d_masks, p_masks = [], [], [], []
    atom_charges, ac_masks = [], []
    heads = []
    props, pr_masks, states = [], [], []
    frag_charges, fc_masks = [], []
    e_masks, f_masks, s_masks = [], [], []
    term_weights = {k: [] for k in ("energy_weight", "force_weight", "stress_weight")}
    atom_force_weights = []

    dtype, device = graphs[0].pos.dtype, graphs[0].pos.device
    has_cell = any(g.cell is not None for g in graphs)
    has_e = any(g.energy is not None for g in graphs)
    has_f = any(g.forces is not None for g in graphs)
    has_s = any(g.stress is not None for g in graphs)
    has_d = any(g.dipole is not None for g in graphs)
    has_p = any(g.polarizability is not None for g in graphs)
    has_ac = any(g.charges is not None for g in graphs)
    has_q = any(g.total_charge is not None for g in graphs)
    has_fc = any(g.fragment_charges is not None for g in graphs)
    has_m = any(g.spin_multiplicity is not None for g in graphs)
    has_w = any(g.weight is not None for g in graphs)
    has_h = any(g.head is not None for g in graphs)
    # the shape and dtype of a missing property or state follow the first given one
    prop_like = next((g.properties for g in graphs if g.properties is not None), None)
    state_like = next((g.state for g in graphs if g.state is not None), None)
    has_tw = {k: any(getattr(g, k) is not None for g in graphs) for k in term_weights}
    has_aw = any(g.atom_force_weight is not None for g in graphs)

    def label_mask(g: AtomicGraph, value, mask) -> Tensor:
        if mask is not None:
            return mask
        return torch.full((g.num_graphs,), value is not None, dtype=torch.bool, device=device)

    node_offset = 0
    graph_offset = 0
    for g in graphs:
        b = g.num_graphs
        pos.append(g.pos)
        z.append(g.atomic_numbers)
        batch.append(g.batch + graph_offset)
        n_atoms.append(g.n_atoms)
        edge_index.append(g.edge_index + node_offset)
        cell_shifts.append(g.cell_shifts)
        node_offset += g.num_nodes
        graph_offset += b
        if has_cell:
            if g.cell is not None:
                cells.append(g.cell)
                # graphs built on a GPU keep pbc on the CPU (structure_to_graph)
                pbcs.append(g.pbc.to(device) if g.pbc is not None
                            else torch.ones((b, 3), dtype=torch.bool, device=device))
            else:
                cells.append(torch.zeros((b, 3, 3), dtype=dtype, device=device))
                pbcs.append(torch.zeros((b, 3), dtype=torch.bool, device=device))
        if has_e:
            energies.append(g.energy if g.energy is not None
                            else torch.zeros(b, dtype=torch.float64, device=device))
            e_masks.append(label_mask(g, g.energy, g.energy_mask))
        if has_f:
            forces.append(g.forces if g.forces is not None else torch.zeros_like(g.pos))
            f_masks.append(label_mask(g, g.forces, g.forces_mask))
        if has_s:
            stresses.append(g.stress if g.stress is not None
                            else torch.zeros((b, 3, 3), dtype=dtype, device=device))
            s_masks.append(label_mask(g, g.stress, g.stress_mask))
        if has_d:
            dipoles.append(g.dipole if g.dipole is not None
                           else torch.zeros((b, 3), dtype=dtype, device=device))
            d_masks.append(label_mask(g, g.dipole, g.dipole_mask))
        if has_p:
            alphas.append(g.polarizability if g.polarizability is not None
                          else torch.zeros((b, 3, 3), dtype=dtype, device=device))
            p_masks.append(label_mask(g, g.polarizability, g.polarizability_mask))
        if has_ac:
            atom_charges.append(g.charges if g.charges is not None
                                else torch.zeros(g.pos.shape[0], dtype=dtype, device=device))
            ac_masks.append(label_mask(g, g.charges, g.charges_mask))
        if has_q:
            charges.append(g.total_charge if g.total_charge is not None
                           else torch.zeros(b, dtype=dtype, device=device))
        if has_fc:
            frag_charges.append(g.fragment_charges if g.fragment_charges is not None
                                else torch.zeros(g.pos.shape[0], dtype=dtype, device=device))
            fc_masks.append(label_mask(g, g.fragment_charges, g.fragment_charges_mask))
        if has_m:
            mults.append(g.spin_multiplicity if g.spin_multiplicity is not None
                         else torch.ones(b, dtype=dtype, device=device))
        if has_w:
            weights.append(g.weight if g.weight is not None
                           else torch.ones(b, dtype=dtype, device=device))
        if has_h:
            heads.append(g.head if g.head is not None
                         else torch.zeros(b, dtype=torch.long, device=device))
        if prop_like is not None:
            props.append(g.properties if g.properties is not None
                         else prop_like.new_zeros((b,) + tuple(prop_like.shape[1:])))
            pr_masks.append(label_mask(g, g.properties, g.properties_mask))
        if state_like is not None:
            states.append(g.state if g.state is not None
                          else state_like.new_zeros((b,) + tuple(state_like.shape[1:])))
        for k, values in term_weights.items():
            if has_tw[k]:
                v = getattr(g, k)
                values.append(v if v is not None else torch.ones(b, dtype=dtype, device=device))
        if has_aw:
            atom_force_weights.append(
                g.atom_force_weight if g.atom_force_weight is not None
                else torch.ones(g.pos.shape[0], dtype=dtype, device=device))

    e_mask = torch.cat(e_masks, 0) if has_e else None
    f_mask = torch.cat(f_masks, 0) if has_f else None
    s_mask = torch.cat(s_masks, 0) if has_s else None
    d_mask = torch.cat(d_masks, 0) if has_d else None
    p_mask = torch.cat(p_masks, 0) if has_p else None
    ac_mask = torch.cat(ac_masks, 0) if has_ac else None
    fc_mask = torch.cat(fc_masks, 0) if has_fc else None
    pr_mask = torch.cat(pr_masks, 0) if prop_like is not None else None
    site_species, site_occupancy = _site_rows(graphs)
    return AtomicGraph(
        pos=torch.cat(pos, 0),
        atomic_numbers=torch.cat(z, 0),
        edge_index=torch.cat(edge_index, 1),
        cell_shifts=torch.cat(cell_shifts, 0),
        batch=torch.cat(batch, 0),
        n_atoms=torch.cat(n_atoms, 0),
        cell=torch.cat(cells, 0) if has_cell else None,
        pbc=torch.cat(pbcs, 0) if has_cell else None,
        energy=torch.cat(energies, 0) if has_e else None,
        forces=torch.cat(forces, 0) if has_f else None,
        stress=torch.cat(stresses, 0) if has_s else None,
        # a mask only where the batch mixes labelled and unlabelled structures
        energy_mask=None if e_mask is None or bool(e_mask.all()) else e_mask,
        forces_mask=None if f_mask is None or bool(f_mask.all()) else f_mask,
        stress_mask=None if s_mask is None or bool(s_mask.all()) else s_mask,
        dipole=torch.cat(dipoles, 0) if has_d else None,
        polarizability=torch.cat(alphas, 0) if has_p else None,
        dipole_mask=None if d_mask is None or bool(d_mask.all()) else d_mask,
        polarizability_mask=None if p_mask is None or bool(p_mask.all()) else p_mask,
        charges=torch.cat(atom_charges, 0) if has_ac else None,
        charges_mask=None if ac_mask is None or bool(ac_mask.all()) else ac_mask,
        fragment_charges=torch.cat(frag_charges, 0) if has_fc else None,
        fragment_charges_mask=None if fc_mask is None or bool(fc_mask.all()) else fc_mask,
        total_charge=torch.cat(charges, 0) if has_q else None,
        spin_multiplicity=torch.cat(mults, 0) if has_m else None,
        weight=torch.cat(weights, 0) if has_w else None,
        head=torch.cat(heads, 0) if has_h else None,
        properties=torch.cat(props, 0) if prop_like is not None else None,
        properties_mask=None if pr_mask is None or bool(pr_mask.all()) else pr_mask,
        state=torch.cat(states, 0) if state_like is not None else None,
        site_species=site_species,
        site_occupancy=site_occupancy,
        energy_weight=torch.cat(term_weights["energy_weight"], 0) if has_tw["energy_weight"] else None,
        force_weight=torch.cat(term_weights["force_weight"], 0) if has_tw["force_weight"] else None,
        stress_weight=torch.cat(term_weights["stress_weight"], 0) if has_tw["stress_weight"] else None,
        atom_force_weight=torch.cat(atom_force_weights, 0) if has_aw else None,
    )
