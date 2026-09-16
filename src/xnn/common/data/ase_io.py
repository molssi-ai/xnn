"""Read ASE-native structure data (extxyz, CIF, VASP, ...) into xnn.

``ase.io.read`` already understands every common atomistic file format; the
helpers here convert its ``Atoms`` objects into the plain structure dicts that
:func:`~xnn.common.data.dataset.structure_to_graph` and
:class:`~xnn.common.data.dataset.AtomicDataset` consume. Training targets
(energy, forces, stress) are taken from the frame's calculator when present --
``ase.io.read`` attaches a ``SinglePointCalculator`` for formats that store
them, e.g. extxyz -- with ``atoms.info`` / ``atoms.arrays`` as a fallback.

ASE is an optional dependency (``pip install xnn[ase]``), so it is imported
lazily inside the functions.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _full_stress(stress) -> np.ndarray:
    """Return the stress as a full ``(3, 3)`` matrix.

    Parameters
    ----------
    stress : array_like
        Stress in Voigt ``(6,)`` or full ``(3, 3)`` form.

    Returns
    -------
    numpy.ndarray
        The stress as a ``(3, 3)`` matrix.
    """
    stress = np.asarray(stress)
    if stress.shape == (3, 3):
        return stress
    from ase.stress import voigt_6_to_full_3x3_stress
    return voigt_6_to_full_3x3_stress(stress)


def atoms_to_structure(atoms, energy_key: str = "energy",
                       forces_key: str = "forces",
                       stress_key: str = "stress") -> dict[str, Any]:
    """Convert one ASE ``Atoms`` object to an xnn structure dict.

    Geometry (positions, atomic numbers, cell, pbc) is always extracted;
    positions are taken as-is (the neighbor-list builder handles unwrapped
    coordinates). Targets are added when available: with the default key
    names, first from the attached calculator (``get_potential_energy`` /
    ``get_forces`` / ``get_stress``), then from ``atoms.info["energy"]`` /
    ``atoms.arrays["forces"]`` / ``atoms.info["stress"]``. A non-default key
    selects that entry of ``info`` / ``arrays`` instead (e.g. MACE-convention
    ``REF_energy`` / ``REF_forces`` / ``REF_stress``). Stress is converted to
    a full ``(3, 3)`` matrix.

    Parameters
    ----------
    atoms : ase.Atoms
        The structure to convert.
    energy_key : str, optional
        Name of the reference energy in ``atoms.info``.
    forces_key : str, optional
        Name of the reference forces in ``atoms.arrays``.
    stress_key : str, optional
        Name of the reference stress in ``atoms.info``.

    Returns
    -------
    dict
        Structure dict with keys ``"pos"``, ``"atomic_numbers"``, ``"cell"``
        (``None`` for non-periodic frames), ``"pbc"``, and optionally
        ``"energy"``, ``"forces"`` and ``"stress"``.
    """
    d: dict[str, Any] = {
        "pos": atoms.get_positions(),
        "atomic_numbers": atoms.get_atomic_numbers(),
        "cell": atoms.get_cell()[:] if atoms.pbc.any() else None,
        "pbc": atoms.pbc.copy(),
    }
    if atoms.calc is not None:
        # A custom key means "use exactly that entry", so the calculator only
        # supplies properties whose key is left at the default.
        for prop, key, getter in (("energy", energy_key, atoms.get_potential_energy),
                                  ("forces", forces_key, atoms.get_forces),
                                  ("stress", stress_key, atoms.get_stress)):
            if key == prop:
                try:
                    d[prop] = getter()
                except Exception:
                    pass
    if "energy" not in d and energy_key in atoms.info:
        d["energy"] = float(atoms.info[energy_key])
    if "forces" not in d and forces_key in atoms.arrays:
        d["forces"] = atoms.arrays[forces_key]
    if "stress" not in d and stress_key in atoms.info:
        d["stress"] = atoms.info[stress_key]
    if "stress" in d:
        d["stress"] = _full_stress(d["stress"])
    return d


def load_structures(path: str, index: str = ":", **target_keys) -> list[dict[str, Any]]:
    """Load a structure file into the list-of-dicts format the dataset expects.

    Reads with :func:`ase.io.read`, so any ASE-readable format works
    (``.xyz`` / ``.extxyz`` / ``.cif`` / VASP / ...), then converts each frame
    via :func:`atoms_to_structure`.

    Parameters
    ----------
    path : str
        Path to a structure file readable by :func:`ase.io.read`.
    index : str, optional
        Frame selection passed to :func:`ase.io.read`; the default ``":"``
        loads all frames.
    **target_keys
        ``energy_key`` / ``forces_key`` / ``stress_key`` overrides forwarded to
        :func:`atoms_to_structure`.

    Returns
    -------
    list of dict
        One structure dict per frame (see :func:`atoms_to_structure`).
    """
    from ase.io import read
    frames = read(path, index=index)
    if not isinstance(frames, list):
        frames = [frames]
    return [atoms_to_structure(a, **target_keys) for a in frames]
