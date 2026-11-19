"""GFN-FF deployment: the TorchScript export of molecules and periodic cells and the MDI engine, against the model."""
import numpy as np
import pytest
import torch

from test_gfnff import PERIODIC_FULL, REFERENCE, _crossing
from xnn.common.data import structure_to_graph
from xnn.common.deploy import export_torchscript_potential
from xnn.common.models import ForceStressOutput
from xnn.common.models.dispersion import HARTREE
from xnn.ffnn.models import GFNFF, gfnfflib
from xnn.ffnn.models.gfnff_export import GFNFFCellExport, GFNFFExport


@pytest.fixture(autouse=True)
def _parameter_file():
    try:
        gfnfflib.parameters()
    except FileNotFoundError:
        pytest.skip("the GFN-FF parameter file is not cached and XNN_OFFLINE is set")


def _eager(model, pos, cell=None):
    """Energy (eV), forces (eV/A) and, with a cell, the strain derivative (eV) of the eager terms."""
    x = torch.tensor(np.asarray(pos), dtype=torch.float64, requires_grad=True)
    if cell is None:
        e = sum(model.term_energies(x).values()) * HARTREE
        (g,) = torch.autograd.grad(e, x)
        return float(e), -g.numpy(), None
    c = torch.tensor(np.asarray(cell), dtype=torch.float64, requires_grad=True)
    e = sum(model.term_energies(x, c).values()) * HARTREE
    gx, gc = torch.autograd.grad(e, (x, c))
    virial = x.detach().T @ gx + c.detach().T @ gc
    return float(e), -gx.numpy(), (0.5 * (virial + virial.T)).numpy()


def _moved(pos, seed, scale=0.03):
    pos = np.asarray(pos, dtype=float)
    return pos + np.random.default_rng(seed).normal(0.0, scale, pos.shape)


@pytest.mark.parametrize("name", ["formamide", "chlorobenzene", "water_dimer", "iminoalkoxide"])
def test_torchscript_export_of_a_molecule(tmp_path, name):
    """The exported molecule gives the energy, forces and total charge of the eager terms through both entry
    points; the export of a float32 model computes in float32."""
    r = REFERENCE[name]
    model = GFNFF(r["numbers"], r["positions"], r["charge"], use_fast=False)
    path = export_torchscript_potential(model, 1.0, str(tmp_path / f"{name}.pt"))
    scripted = torch.jit.load(path)
    z = torch.tensor(r["numbers"])
    for seed in range(3):
        pos = _moved(r["positions"], seed)
        e, f, _ = _eager(model, pos)
        got = scripted(torch.tensor(pos), z)
        assert float(got["energy"]) == pytest.approx(e, rel=1e-12, abs=1e-12)
        assert np.abs(got["forces"].numpy() - f).max() < 1e-10
        assert float(got["charges"].sum()) == pytest.approx(r["charge"], abs=1e-10)
        assert float(got["stress"].abs().max()) == 0.0
        empty = torch.zeros((2, 0), dtype=torch.long)
        lmp = scripted.forward_lammps(torch.tensor(pos), empty, torch.zeros((0, 3), dtype=torch.float64), z,
                                     torch.zeros((3, 3), dtype=torch.float64))
        assert torch.equal(lmp["energy"], got["energy"]) and torch.equal(lmp["forces"], got["forces"])
    single = GFNFF(r["numbers"], r["positions"], r["charge"], use_fast=False).float()
    path = export_torchscript_potential(single, 1.0, str(tmp_path / f"{name}32.pt"))
    got = torch.jit.load(path)(torch.tensor(pos, dtype=torch.float32), z)
    assert got["forces"].dtype == torch.float32
    assert float(got["energy"]) == pytest.approx(e, rel=1e-6, abs=1e-5)


@pytest.mark.parametrize("system", ["dimer", "trimer", "halogen"])
def test_torchscript_export_follows_the_hydrogen_and_halogen_bonds(tmp_path, system):
    """A molecule exported once gives the energies and forces of models built at each conformation while hydrogen-
    and halogen-bond partners cross the distance cuts; an export with fixed lists keeps the rows of the build."""
    z, path = _crossing(system)
    scripted = torch.jit.load(export_torchscript_potential(GFNFF(z, path[0], use_fast=False), 1.0,
                                                           str(tmp_path / "once.pt")))
    fixed = GFNFF(z, path[0], use_fast=False, fixed_lists=True)
    pinned = torch.jit.load(export_torchscript_potential(fixed, 1.0, str(tmp_path / "fixed.pt")))
    for pos in path:
        e, f, _ = _eager(GFNFF(z, pos, use_fast=False), pos)
        got = scripted(torch.tensor(pos), torch.tensor(z))
        assert float(got["energy"]) == pytest.approx(e, rel=1e-12, abs=1e-12)
        assert np.abs(got["forces"].numpy() - f).max() < 1e-10
        e, f, _ = _eager(fixed, pos)
        got = pinned(torch.tensor(pos), torch.tensor(z))
        assert float(got["energy"]) == pytest.approx(e, rel=1e-12, abs=1e-12)
        assert np.abs(got["forces"].numpy() - f).max() < 1e-10


@pytest.mark.parametrize("name", sorted(PERIODIC_FULL))
def test_torchscript_export_of_a_cell(tmp_path, name):
    """The exported cell gives the energy, forces and stress of the eager terms (the stress also from the reference
    program's lattice gradient), in a strained cell and with an atom wrapped back into the cell; an atom beyond the
    rows is refused."""
    r = PERIODIC_FULL[name]
    model = GFNFF(r["numbers"], r["positions"], r["charge"], cell=r["cell"], use_fast=False)
    scripted = torch.jit.load(export_torchscript_potential(model, 1.0, str(tmp_path / f"{name}.pt")))
    z, pbc = torch.tensor(r["numbers"]), torch.ones(3, dtype=torch.bool)
    cell = np.asarray(r["cell"])
    got = scripted(torch.tensor(r["positions"], dtype=torch.float64), z, torch.tensor(cell), pbc)
    volume = abs(np.linalg.det(cell))
    bohr = gfnfflib.BOHR
    virial = cell.T @ np.asarray(r["gradlatt"]) / bohr * HARTREE
    assert np.abs(got["stress"].numpy() * volume - 0.5 * (virial + virial.T)).max() < 1e-6
    strained = cell @ (np.eye(3) + np.array([[0.004, 0.002, 0.0], [0.002, -0.003, 0.001], [0.0, 0.001, 0.002]]))
    pos = _moved(r["positions"], 1, 0.02) @ np.linalg.solve(cell, strained)
    e, f, w = _eager(model, pos, strained)
    wrapped = pos.copy()
    wrapped[0] += strained[0] + strained[2]
    for x in (pos, wrapped):
        got = scripted(torch.tensor(x), z, torch.tensor(strained), pbc)
        assert float(got["energy"]) == pytest.approx(e, rel=1e-12, abs=1e-11)
        assert np.abs(got["forces"].numpy() - f).max() < 1e-9
        assert np.abs(got["stress"].numpy() * abs(np.linalg.det(strained)) - w).max() < 1e-9
        assert float(got["charges"].sum()) == pytest.approx(r["charge"], abs=1e-10)
    far = pos.copy()
    far[0] += 0.6
    with pytest.raises(Exception, match="skin"):
        scripted(torch.tensor(far), z, torch.tensor(strained), pbc)
    with pytest.raises(Exception, match="three directions"):
        scripted(torch.tensor(pos), z)


def test_export_errors(tmp_path):
    """Each core takes its kind of model, and the scripted molecule refuses other atoms and cells."""
    r = REFERENCE["formamide"]
    model = GFNFF(r["numbers"], r["positions"], use_fast=False)
    c = PERIODIC_FULL["formic_acid_chain"]
    cell = GFNFF(c["numbers"], c["positions"], cell=c["cell"], use_fast=False)
    with pytest.raises(ValueError, match="GFNFFCellExport"):
        GFNFFExport(cell)
    with pytest.raises(ValueError, match="GFNFFExport"):
        GFNFFCellExport(model)
    assert isinstance(model.export_core(), GFNFFExport) and isinstance(cell.export_core(), GFNFFCellExport)
    scripted = torch.jit.load(export_torchscript_potential(model, 1.0, str(tmp_path / "m.pt")))
    pos = torch.tensor(r["positions"], dtype=torch.float64)
    with pytest.raises(Exception, match="bound"):
        scripted(pos, torch.tensor(r["numbers"][::-1]))
    with pytest.raises(Exception, match="cells"):
        scripted(pos, torch.tensor(r["numbers"]), torch.eye(3, dtype=torch.float64) * 20.0,
                 torch.ones(3, dtype=torch.bool))
    scripted = torch.jit.load(export_torchscript_potential(cell, 1.0, str(tmp_path / "c.pt")))
    with pytest.raises(Exception, match="bound"):
        scripted(torch.tensor(c["positions"], dtype=torch.float64), torch.tensor(c["numbers"][::-1]),
                 torch.tensor(c["cell"], dtype=torch.float64), torch.ones(3, dtype=torch.bool))
    graph = structure_to_graph({"pos": pos, "atomic_numbers": torch.tensor(r["numbers"][::-1])}, 1.0)
    with pytest.raises(ValueError, match="conformation"):
        model.forward_direct(graph)
    graph = structure_to_graph({"pos": torch.tensor(c["positions"], dtype=torch.float64),
                                "atomic_numbers": torch.tensor(c["numbers"])}, 1.0)
    with pytest.raises(ValueError, match="cell of every structure"):
        cell.forward_direct(graph)


@pytest.mark.parametrize("fast", [False, "auto"])
def test_forces_and_stress_of_an_evaluation(fast):
    """In evaluation the wrapped model forms its forces and stress from ``energy_and_gradient`` (the fused kernels
    where they run): the eager values for a molecule and a periodic cell, in a batch of conformations."""
    from xnn.common.data import collate

    r = REFERENCE["water_dimer"]
    model = GFNFF(r["numbers"], r["positions"], use_fast=fast).eval()
    confs = [_moved(r["positions"], s) for s in range(2)]
    graphs = [structure_to_graph({"pos": torch.tensor(p), "atomic_numbers": torch.tensor(r["numbers"])}, 1.0)
              for p in confs]
    assert model.forms_forces(graphs[0])
    out = ForceStressOutput(model).eval()(collate(graphs))
    for k, pos in enumerate(confs):
        e, f, _ = _eager(GFNFF(r["numbers"], r["positions"], use_fast=False), pos)
        assert float(out["energy"][k]) == pytest.approx(e, rel=1e-10)
        assert np.abs(out["forces"][6 * k:6 * k + 6].numpy() - f).max() < 1e-8
    c = PERIODIC_FULL["water_dimer_across_cell"]
    model = GFNFF(c["numbers"], c["positions"], cell=c["cell"], use_fast=fast).eval()
    pos = _moved(c["positions"], 3, 0.02)
    s = {"pos": torch.tensor(pos), "atomic_numbers": torch.tensor(c["numbers"]),
         "cell": torch.tensor(c["cell"], dtype=torch.float64), "pbc": torch.ones(3, dtype=torch.bool)}
    out = ForceStressOutput(model, compute_stress=True).eval()(structure_to_graph(s, 1.0))
    e, f, w = _eager(GFNFF(c["numbers"], c["positions"], cell=c["cell"], use_fast=False), pos, c["cell"])
    assert float(out["energy"]) == pytest.approx(e, rel=1e-10)
    assert np.abs(out["forces"].numpy() - f).max() < 1e-8
    assert np.abs(out["stress"][0].numpy() * abs(np.linalg.det(c["cell"])) - w).max() < 1e-8
    model.train()
    assert not model.forms_forces(structure_to_graph(s, 1.0))


def test_mdi_engine_serves_a_periodic_checkpoint(tmp_path):
    """A checkpoint whose config binds a cell serves the model's energy, forces and stress through the MDI engine,
    also when the driver wraps an atom back into the cell."""
    from xnn.common.config import from_dict
    from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine

    c = PERIODIC_FULL["trifluoroiodomethane_ammonia"]
    cfg = from_dict({"model": {"name": "gfnff", "cutoff": 1.0, "atomic_numbers": c["numbers"],
                               "positions": c["positions"], "cell": c["cell"]}})
    model = GFNFF(c["numbers"], c["positions"], cell=c["cell"], use_fast=False)
    path = tmp_path / "gfnff_cell.pt"
    torch.save({"model": ForceStressOutput(model).state_dict(), "cfg": cfg}, path)
    engine = MDIEngine.from_checkpoint(str(path), dtype=torch.float64)
    cell = np.asarray(c["cell"])
    engine.natoms = len(c["numbers"])
    engine.atomic_numbers = np.asarray(c["numbers"])
    engine.cell_bohr = cell / BOHR_TO_ANGSTROM
    pos = _moved(c["positions"], 5, 0.02)
    e, f, w = _eager(model, pos, cell)
    wrapped = pos.copy()
    wrapped[4] -= cell[1]
    for x in (pos, wrapped):
        engine.coords_bohr = x / BOHR_TO_ANGSTROM
        engine.calculate()
        assert engine.energy * HARTREE_TO_EV == pytest.approx(e, rel=1e-10)
        forces = np.asarray(engine.forces).reshape(-1, 3) * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        assert np.abs(forces - f).max() < 1e-8
        stress = -np.asarray(engine.stress).reshape(3, 3) * HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3
        assert np.abs(stress * abs(np.linalg.det(cell)) - w).max() < 1e-8


def _core_outputs(core, pos, z, cell=None):
    """Energy (eV), forces and strain derivative of an exported core run eagerly (the scripted module's code)."""
    x = torch.tensor(np.asarray(pos), dtype=torch.float64, requires_grad=True)
    strain = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    h = torch.zeros((3, 3), dtype=torch.float64) if cell is None else torch.tensor(np.asarray(cell))
    pbc = torch.full((3,), cell is not None)
    _, node, q = core.node_features_energy_charges(torch.tensor(z), torch.zeros((2, 0), dtype=torch.long),
                                                   torch.zeros((0, 3)), x + x @ strain, h + h @ strain, pbc, 0.0, 1.0)
    gx, gs = torch.autograd.grad(node.sum(), (x, strain))
    return float(node.sum()), -gx.numpy(), (0.5 * (gs + gs.T)).numpy(), q


@pytest.mark.parametrize("name", ["formamide", "formamidine", "chlorobenzene", "water_dimer", "aziridine"])
def test_export_core_of_a_molecule_eager(name):
    """The exported molecule core, run eagerly, gives the energy and forces of the eager terms."""
    r = REFERENCE[name]
    model = GFNFF(r["numbers"], r["positions"], r["charge"], use_fast=False)
    core = model.export_core()
    for seed in range(2):
        pos = _moved(r["positions"], seed)
        e, f, _ = _eager(model, pos)
        got_e, got_f, _, q = _core_outputs(core, pos, r["numbers"])
        assert got_e == pytest.approx(e, rel=1e-12, abs=1e-12) and np.abs(got_f - f).max() < 1e-10
        assert float(q.sum()) == pytest.approx(r["charge"], abs=1e-10)
    with pytest.raises(RuntimeError, match="bound"):
        _core_outputs(core, pos[::-1].copy(), r["numbers"][::-1])
    with pytest.raises(RuntimeError, match="cells"):
        _core_outputs(core, pos, r["numbers"], np.eye(3) * 20.0)


@pytest.mark.parametrize("system", ["dimer", "trimer", "halogen"])
def test_export_core_follows_the_rows_eager(system):
    """The molecule core, run eagerly, chooses the rows of each conformation, and keeps those of a model with
    fixed lists."""
    z, path = _crossing(system)
    core = GFNFF(z, path[0], use_fast=False).export_core()
    fixed = GFNFF(z, path[0], use_fast=False, fixed_lists=True)
    pinned = fixed.export_core()
    for pos in path:
        e, f, _ = _eager(GFNFF(z, pos, use_fast=False), pos)
        got_e, got_f, _, _ = _core_outputs(core, pos, z)
        assert got_e == pytest.approx(e, rel=1e-12, abs=1e-12) and np.abs(got_f - f).max() < 1e-10
        e, f, _ = _eager(fixed, pos)
        got_e, got_f, _, _ = _core_outputs(pinned, pos, z)
        assert got_e == pytest.approx(e, rel=1e-12, abs=1e-12) and np.abs(got_f - f).max() < 1e-10


#: formamidine in a box: a hydrogen bond to an sp2 N acceptor with two neighbours
_BOX = dict(numbers=REFERENCE["formamidine"]["numbers"], charge=0,
            positions=(np.asarray(REFERENCE["formamidine"]["positions"]) + 3.0).tolist(),
            cell=[[7.0, 0.0, 0.0], [0.0, 7.5, 0.0], [0.5, 0.0, 8.0]])


@pytest.mark.parametrize("name", sorted(PERIODIC_FULL) + ["formamidine_box"])
def test_export_core_of_a_cell_eager(name):
    """The exported cell core, run eagerly, gives the energy, forces and strain derivative of the eager terms, also
    with an atom wrapped back into the cell; it refuses a structure that is not periodic and an atom beyond the rows,
    unless the model keeps fixed lists."""
    r = PERIODIC_FULL.get(name, _BOX)
    model = GFNFF(r["numbers"], r["positions"], r["charge"], cell=r["cell"], use_fast=False)
    core = model.export_core()
    cell = np.asarray(r["cell"])
    pos = _moved(r["positions"], 2, 0.02)
    e, f, w = _eager(model, pos, cell)
    wrapped = pos.copy()
    wrapped[-1] -= cell[0]
    for x in (pos, wrapped):
        got_e, got_f, got_w, q = _core_outputs(core, x, r["numbers"], cell)
        assert got_e == pytest.approx(e, rel=1e-12, abs=1e-11)
        assert np.abs(got_f - f).max() < 1e-9 and np.abs(got_w - w).max() < 1e-9
        assert float(q.sum()) == pytest.approx(r["charge"], abs=1e-10)
    far = pos.copy()
    far[0] += 0.6
    with pytest.raises(RuntimeError, match="skin"):
        _core_outputs(core, far, r["numbers"], cell)
    with pytest.raises(RuntimeError, match="three directions"):
        _core_outputs(core, pos, r["numbers"])
    with pytest.raises(RuntimeError, match="bound"):
        _core_outputs(core, pos[::-1].copy(), r["numbers"][::-1], cell)
    fixed = GFNFF(r["numbers"], r["positions"], r["charge"], cell=r["cell"], use_fast=False, fixed_lists=True)
    e, f, w = _eager(fixed, far, cell)
    got_e, got_f, got_w, _ = _core_outputs(fixed.export_core(), far, r["numbers"], cell)
    assert got_e == pytest.approx(e, rel=1e-12, abs=1e-11) and np.abs(got_f - f).max() < 1e-9
