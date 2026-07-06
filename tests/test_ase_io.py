"""ASE-native data ingestion (extxyz, CIF, in-memory Atoms).

`AtomicDataset.from_file` / `load_structures` must read any ASE-supported
format, pick up energy / forces / stress targets when the file carries them,
and build correct graphs without requiring pre-wrapped coordinates.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.stress import voigt_6_to_full_3x3_stress

from xnns.common.data import AtomicDataset, atoms_to_structure, load_structures


def _argon_frame(seed=0, L=8.0, n=16):
    rng = np.random.default_rng(seed)
    atoms = Atoms(numbers=[18] * n, positions=rng.uniform(0, L, (n, 3)),
                  cell=np.eye(3) * L, pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms, energy=float(rng.normal()), forces=rng.normal(size=(n, 3)),
        stress=rng.normal(size=6))
    return atoms


def test_extxyz_roundtrip_with_targets(tmp_path):
    """extxyz frames come back with energy, forces and (3,3) stress targets."""
    torch.set_default_dtype(torch.float64)
    frames = [_argon_frame(s) for s in range(3)]
    path = tmp_path / "frames.extxyz"
    write(path, frames)

    ds = AtomicDataset.from_file(str(path), cutoff=4.0)
    assert len(ds) == 3
    for i, ref in enumerate(frames):
        g = ds[i]
        assert np.isclose(float(g.energy), ref.get_potential_energy())
        assert np.allclose(g.forces.numpy(), ref.get_forces(), atol=1e-9)
        assert g.stress.shape == (1, 3, 3)
        assert np.allclose(g.stress[0].numpy(),
                           voigt_6_to_full_3x3_stress(ref.get_stress()), atol=1e-9)
        assert bool(g.pbc.all())


def test_cif_geometry(tmp_path):
    """A CIF crystal loads with correct species, periodicity and edges."""
    torch.set_default_dtype(torch.float64)
    atoms = Atoms("NaCl", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                  cell=np.eye(3) * 5.64, pbc=True)
    path = tmp_path / "nacl.cif"
    write(path, atoms)

    ds = AtomicDataset.from_file(str(path), cutoff=5.0)
    assert len(ds) == 1
    g = ds[0]
    assert g.num_nodes == 2
    assert sorted(g.atomic_numbers.tolist()) == [11, 17]
    assert g.energy is None and g.forces is None and g.stress is None
    assert bool(g.pbc.all())
    assert g.num_edges > 0
    assert float(g.edge_vectors().norm(dim=1).max()) <= 5.0 + 1e-8


def test_molecular_xyz(tmp_path):
    """A plain (non-periodic) xyz molecule loads with cell=None."""
    torch.set_default_dtype(torch.float64)
    water = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    path = tmp_path / "water.xyz"
    write(path, water)

    s = load_structures(str(path))
    assert len(s) == 1 and s[0]["cell"] is None
    g = AtomicDataset(s, cutoff=2.0)[0]
    assert g.cell is None
    assert torch.all(g.cell_shifts == 0)
    assert g.num_edges > 0


def test_from_atoms_single_and_list():
    """`from_atoms` accepts a single Atoms or a list, without touching disk."""
    torch.set_default_dtype(torch.float64)
    frames = [_argon_frame(s) for s in range(2)]
    assert len(AtomicDataset.from_atoms(frames[0], cutoff=4.0)) == 1
    ds = AtomicDataset.from_atoms(frames, cutoff=4.0)
    assert len(ds) == 2
    assert np.isclose(float(ds[1].energy), frames[1].get_potential_energy())


def test_unwrapped_coordinates_give_same_graph(tmp_path):
    """Frames with atoms outside the cell need no wrap(): graphs match exactly."""
    torch.set_default_dtype(torch.float64)
    atoms = _argon_frame(seed=4)
    unwrapped = atoms.copy()
    rng = np.random.default_rng(11)
    unwrapped.positions += rng.integers(-3, 4, (len(atoms), 3)) @ atoms.cell[:]

    g_ref = AtomicDataset.from_atoms(atoms, cutoff=4.0)[0]
    g = AtomicDataset.from_atoms(unwrapped, cutoff=4.0)[0]
    assert g.num_edges == g_ref.num_edges
    d, d_ref = (x.edge_vectors().norm(dim=1).numpy() for x in (g, g_ref))
    assert np.allclose(np.sort(d), np.sort(d_ref), atol=1e-10)


def test_custom_target_keys(tmp_path):
    """MACE-style REF_* keys are read via energy_key / forces_key / stress_key."""
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(3)
    atoms = Atoms(numbers=[18] * 4, positions=rng.uniform(0, 5, (4, 3)),
                  cell=np.eye(3) * 5.0, pbc=True)
    atoms.info["REF_energy"] = -2.5
    atoms.info["REF_stress"] = rng.normal(size=6)
    atoms.new_array("REF_forces", rng.normal(size=(4, 3)))
    path = tmp_path / "ref.extxyz"
    write(path, atoms)

    ds = AtomicDataset.from_file(str(path), cutoff=4.0, energy_key="REF_energy",
                                 forces_key="REF_forces", stress_key="REF_stress")
    g = ds[0]
    assert np.isclose(float(g.energy), -2.5)
    assert np.allclose(g.forces.numpy(), atoms.arrays["REF_forces"], atol=1e-9)
    assert np.allclose(g.stress[0].numpy(),
                       voigt_6_to_full_3x3_stress(atoms.info["REF_stress"]), atol=1e-9)

    # default keys must not accidentally pick the REF_* entries up
    assert AtomicDataset.from_file(str(path), cutoff=4.0)[0].energy is None


def test_info_and_arrays_fallback():
    """Targets in atoms.info / atoms.arrays are picked up when no calculator."""
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(2)
    atoms = Atoms(numbers=[18] * 4, positions=rng.uniform(0, 5, (4, 3)),
                  cell=np.eye(3) * 5.0, pbc=True)
    atoms.info["energy"] = -1.5
    atoms.info["stress"] = rng.normal(size=6)          # Voigt
    atoms.new_array("forces", rng.normal(size=(4, 3)))

    s = atoms_to_structure(atoms)
    assert s["energy"] == -1.5
    assert s["forces"].shape == (4, 3)
    assert s["stress"].shape == (3, 3)
    assert np.allclose(s["stress"], voigt_6_to_full_3x3_stress(atoms.info["stress"]))
