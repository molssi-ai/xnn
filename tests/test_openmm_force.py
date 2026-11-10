"""xnn potentials in OpenMM (xnn.common.deploy.openmm_force): the TorchForce module's units, atom subsets and
periodic boxes against ForceStressOutput, and, where OpenMM is installed, a context and openmm-ml's ML/MM."""
import numpy as np
import pytest
import torch

from xnn.common.config import from_dict
from xnn.common.data import structure_to_graph
from xnn.common.deploy import openmm_potential
from xnn.common.deploy.openmm_force import KJ_PER_MOL_PER_EV, XNNPotentialImpl, XNNPotentialImplFactory
from xnn.common.models import ForceStressOutput, build_model

Z = [8, 1, 1, 6, 1, 1]
CELL = 7.5


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _model():
    torch.manual_seed(0)
    cfg = from_dict({"model": {"name": "schnet", "cutoff": 4.0, "n_features": 16, "n_interactions": 2,
                               "n_rbf": 8}})
    return build_model(cfg.model).eval()


def _positions(n=len(Z), seed=0):
    return np.random.default_rng(seed).uniform(0.0, 4.0, (n, 3))


def _reference(model, pos, periodic):
    s = {"pos": pos, "atomic_numbers": Z}
    if periodic:
        s["cell"], s["pbc"] = np.eye(3) * CELL, [True] * 3
    out = ForceStressOutput(model)(structure_to_graph(s, model.cutoff))
    return float(out["energy"]), out["forces"].detach()


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("subset", [False, True])
def test_module_units_subset_and_box(periodic, subset):
    """Energies in kJ/mol and forces in kJ/mol/nm from positions in nm, on the model's atoms only."""
    model = _model()
    pos = _positions()
    atoms = [2, 3, 5, 7, 8, 9] if subset else None
    module = openmm_potential(model, Z, atoms=atoms, periodic=periodic)
    n = 11 if subset else len(Z)
    everything = torch.as_tensor(_positions(n, seed=1)) / 10.0
    index = torch.tensor(atoms) if subset else torch.arange(len(Z))
    everything[index] = torch.as_tensor(pos) / 10.0
    box = torch.eye(3) * CELL / 10.0 if periodic else None
    energy, forces = module(everything, box)
    ref_e, ref_f = _reference(model, pos, periodic)
    assert float(energy) == pytest.approx(ref_e * KJ_PER_MOL_PER_EV, rel=1e-12)
    assert torch.allclose(forces[index], ref_f * KJ_PER_MOL_PER_EV * 10.0, atol=1e-9)
    others = torch.ones(n, dtype=torch.bool)
    others[index] = False
    assert forces.shape == (n, 3) and float(forces[others].abs().sum()) == 0.0


def test_module_answers_in_the_callers_precision():
    """float32 positions (OpenMM's mixed precision) give float32 results for a float64 model."""
    module = openmm_potential(_model(), Z)
    energy, forces = module(torch.as_tensor(_positions(), dtype=torch.float32) / 10.0)
    assert energy.dtype == torch.float32 and forces.dtype == torch.float32


def test_factory_and_long_range_report():
    """openmm-ml's factory needs a model; a model without a stated receptive field reports None, a local
    one False."""
    with pytest.raises(ValueError, match="model="):
        XNNPotentialImplFactory().createImpl("xnn")
    impl = XNNPotentialImplFactory().createImpl("xnn", model=_model())
    assert isinstance(impl, XNNPotentialImpl) and impl.getSupportedEmbeddings() == []
    assert impl.getMLLongRange() is None
    pytest.importorskip("e3nn")
    mace = build_model(from_dict({"model": {
        "name": "mace", "cutoff": 4.0, "n_features": 8, "n_interactions": 1,
        "extra": {"species": [1, 6, 8], "max_ell": 2, "correlation": 2, "num_channels": 8, "max_L": 1}}}).model)
    assert XNNPotentialImplFactory().createImpl("xnn", model=mace).getMLLongRange() is False


def _topology(n_extra=0, periodic=False):
    app = pytest.importorskip("openmm.app")
    import openmm
    from openmm import unit
    top = app.Topology()
    residue = top.addResidue("MOL", top.addChain())
    for i, z in enumerate(Z):
        top.addAtom(f"A{i}", app.Element.getByAtomicNumber(z), residue)
    solvent = top.addResidue("ION", top.addChain())
    for i in range(n_extra):
        top.addAtom(f"N{i}", app.Element.getBySymbol("Ne"), solvent)
    if periodic:
        top.setPeriodicBoxVectors([openmm.Vec3(CELL / 10.0, 0, 0), openmm.Vec3(0, CELL / 10.0, 0),
                                   openmm.Vec3(0, 0, CELL / 10.0)] * unit.nanometer)
    return top


def _context(system, positions):
    import openmm
    from openmm import unit
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


@pytest.mark.parametrize("periodic", [False, True])
def test_torch_force_in_an_openmm_context(periodic):
    """The TorchForce in an OpenMM context gives the xnn energy and forces in OpenMM's units."""
    openmm = pytest.importorskip("openmm")
    pytest.importorskip("openmmtorch")
    from xnn.common.deploy import create_torch_force
    model = _model()
    top = _topology(periodic=periodic)
    system = openmm.System()
    for atom in top.atoms():
        system.addParticle(atom.element.mass)
    if periodic:
        system.setDefaultPeriodicBoxVectors(*top.getPeriodicBoxVectors())
    system.addForce(create_torch_force(model, top))
    pos = _positions()
    energy, forces = _context(system, pos / 10.0)
    ref_e, ref_f = _reference(model, pos, periodic)
    assert energy == pytest.approx(ref_e * KJ_PER_MOL_PER_EV, rel=1e-6)
    assert np.allclose(forces, ref_f.numpy() * KJ_PER_MOL_PER_EV * 10.0, atol=1e-3)


def test_openmm_ml_create_system_and_mixed_system():
    """MLPotential("xnn") builds an ML system, and an ML/MM system whose energy is the MM energy without the
    interactions among the model's atoms plus the model's energy."""
    openmm = pytest.importorskip("openmm")
    pytest.importorskip("openmmtorch")
    openmmml = pytest.importorskip("openmmml")
    if "xnn" not in openmmml.MLPotential._implFactories:
        openmmml.MLPotential.registerImplFactory("xnn", XNNPotentialImplFactory())
    model = _model()
    potential = openmmml.MLPotential("xnn", model=model)
    pos = _positions()
    energy, _ = _context(potential.createSystem(_topology()), pos / 10.0)
    ref_e, _ = _reference(model, pos, False)
    assert energy == pytest.approx(ref_e * KJ_PER_MOL_PER_EV, rel=1e-6)

    # MM: Lennard-Jones and charges on every atom of the molecule and three neon atoms
    top = _topology(n_extra=3)
    everything = np.concatenate([pos, [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]]) / 10.0
    n = len(everything)

    def mm(exclude_ml_pairs):
        system = openmm.System()
        nb = openmm.NonbondedForce()
        for i, atom in enumerate(top.atoms()):
            system.addParticle(atom.element.mass)
            nb.addParticle(0.1 * (-1) ** i if i < len(Z) else 0.0, 0.3, 0.2)
        if exclude_ml_pairs:
            for i in range(len(Z)):
                for j in range(i + 1, len(Z)):
                    nb.addException(i, j, 0.0, 1.0, 0.0)
        system.addForce(nb)
        return system

    mixed = potential.createMixedSystem(top, mm(False), list(range(len(Z))))
    energy, _ = _context(mixed, everything)
    mm_rest, _ = _context(mm(True), everything)
    assert n == mixed.getNumParticles()
    assert energy == pytest.approx(mm_rest + ref_e * KJ_PER_MOL_PER_EV, rel=1e-6, abs=1e-6)
