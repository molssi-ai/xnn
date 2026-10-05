"""Tests for AIMNet2 (registered as ``aimnet2``).

Covers the architecture against a dense all-pairs evaluation of the paper's
equations on a tiny artifact in the reference layout, charge conservation
(neutral, charged, two spin channels), invariances and conservative forces,
the two Coulomb sums (all pairs, damped shifted-force on periodic cells), the
float64 energy shifts, the conversion of reference artifacts through the model
hub (``.pt``, safetensors directory, aliases, cache) and the published
checkpoints against reference values where the cache holds them.
"""
import json
import math
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")   # the gnn family (where AIMNet2 lives) registers with e3nn

from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.models import (ForceStressOutput, available_models, build_model,  # noqa: E402
                               from_pretrained, list_models, model_card, register_pretrained)
from xnn.common.models.d3 import D3Dispersion  # noqa: E402
from xnn.common.models.electrostatics import COULOMB_CONSTANT, all_pairs  # noqa: E402
from xnn.common.models.hub import registry  # noqa: E402
from xnn.gnn.featurizers import MollifierCutoff  # noqa: E402
from xnn.gnn.models import aimnet2_foundation as foundation  # noqa: E402
from xnn.gnn.models.aimnet2 import AIMNet2, ShellConvolution  # noqa: E402

# the model cache of the environment, read before the fixture clears the variables
_CACHE_AT_IMPORT = os.environ.get("XNN_MODELS")

SPECIES = [1, 6, 7, 8]
TINY = dict(n_features=4, n_rbf=6, hidden=[[16, 12], [16, 12], [16, 12, 12]], aim_size=10,
            readout_hidden=[8], n_vector_combinations=3)


@pytest.fixture(autouse=True)
def _f64_and_registry(monkeypatch):
    # no cache or offline settings from the environment (the hub tests download
    # from file:// URLs); float64 for the tiny models; the registry restored
    for var in ("XNN_MODELS", "XNN_CACHE", "XNN_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    saved, saved_aliases = dict(registry._REGISTRY), dict(registry._ALIASES)
    yield
    torch.set_default_dtype(old)
    registry._REGISTRY.clear(); registry._REGISTRY.update(saved)
    registry._ALIASES.clear(); registry._ALIASES.update(saved_aliases)


def _molecule(n=8, seed=1, R=None, shift=0.0, charge=None, mult=None, cell=None,
              cutoff=5.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 4.0, (n, 3)) + shift
    if R is not None:
        pos = pos @ R.T
    s = {"pos": pos, "atomic_numbers": ([8, 1, 1, 6, 7, 1, 6, 8, 1, 1] * n)[:n]}
    if charge is not None:
        s["total_charge"] = charge
    if mult is not None:
        s["spin_multiplicity"] = mult
    if cell is not None:
        s["cell"] = np.eye(3) * cell
        s["pbc"] = [True, True, True]
    return structure_to_graph(s, cutoff)


def _build(seed=0, **extra):
    opts = {**TINY, "species": SPECIES, **extra}
    hidden = opts.pop("hidden")
    cfg = from_dict({"model": {"name": "aimnet2", "cutoff": 5.0,
                               "n_features": opts.pop("n_features"),
                               "n_rbf": opts.pop("n_rbf"), "n_interactions": len(hidden),
                               "extra": {"hidden": hidden, **opts}}})
    torch.manual_seed(seed)
    return build_model(cfg.model)


def _proper_rotation(seed=3):
    rng = np.random.default_rng(seed)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def test_registered():
    assert "aimnet2" in available_models()


def test_outputs_and_charge_conservation():
    model = _build()
    for charge in (None, -1.0, 2.0):
        g = _molecule(charge=charge)
        out = model(g)
        n = g.num_nodes
        assert out["energy"].shape == (1,) and out["energy"].dtype == torch.float64
        assert out["node_energy"].shape == (n,) and out["node_features"].shape == (n, 10)
        assert out["dipole"].shape == (1, 3) and out["energy_coulomb"].shape == (1,)
        # the 1e-6 regularization of the weight sum leaves a residual of that order
        # relative to the sum of the (here random, small) weights
        assert abs(float(out["charges"].sum()) - (charge or 0.0)) < 1e-4
        assert "spin_charges" not in out


def test_two_charge_channels_follow_charge_and_multiplicity():
    model = _build(charge_channels=2)
    out = model(_molecule(charge=1.0, mult=2.0))
    assert abs(float(out["charges"].sum()) - 1.0) < 1e-4
    assert abs(float(out["spin_charges"].sum()) - 1.0) < 1e-4      # 2S = M - 1
    out = model(_molecule(charge=0.0))                              # no multiplicity: singlet
    assert abs(float(out["spin_charges"].sum())) < 1e-4
    assert model.conv_q.weight.shape == (2, 6, 3)


def test_invariance_equivariance_and_permutation():
    model = ForceStressOutput(_build())
    R = _proper_rotation()
    Rt = torch.tensor(R, dtype=torch.float64)
    o0 = model(_molecule(charge=-1.0))
    o1 = model(_molecule(R=R, shift=2.5, charge=-1.0))
    assert abs(float(o0["energy"]) - float(o1["energy"])) < 1e-9
    assert torch.allclose(o1["forces"], o0["forces"] @ Rt.T, atol=1e-8)
    assert torch.allclose(o1["charges"], o0["charges"], atol=1e-10)
    # the shift is applied before the rotation in _molecule: d1 = R (d0 + Q s)
    assert torch.allclose(o1["dipole"], (o0["dipole"] + o1["charges"].sum() * 2.5) @ Rt.T,
                          atol=1e-8)
    g = _molecule(charge=-1.0)
    perm = torch.randperm(g.num_nodes, generator=torch.Generator().manual_seed(5))
    gp = structure_to_graph({"pos": g.pos[perm], "atomic_numbers": g.atomic_numbers[perm],
                             "total_charge": -1.0}, 5.0)
    op = model(gp)
    assert abs(float(op["energy"]) - float(o0["energy"])) < 1e-9
    assert torch.allclose(op["charges"], o0["charges"][perm], atol=1e-10)


def test_batch_matches_single_structures():
    model = _build()
    a, b = _molecule(seed=1, charge=-1.0), _molecule(n=6, seed=2, charge=1.0)
    out = model(collate([a, b]))
    ea, eb = model(a), model(b)
    assert abs(float(out["energy"][0]) - float(ea["energy"])) < 1e-10
    assert abs(float(out["energy"][1]) - float(eb["energy"])) < 1e-10
    assert torch.allclose(out["charges"], torch.cat([ea["charges"], eb["charges"]]), atol=1e-10)


@pytest.mark.parametrize("coulomb", ["simple", "dsf", None])
def test_forces_are_conservative(coulomb):
    model = ForceStressOutput(_build(coulomb=coulomb, lr_cutoff=6.0))
    g = _molecule(charge=-1.0, cutoff=model.model.cutoff)
    forces = model(g)["forces"]
    pos = g.pos.detach().clone()
    h = 1e-4
    for atom, axis in ((0, 0), (3, 2), (5, 1)):
        plus, minus = pos.clone(), pos.clone()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        e_plus = float(model(structure_to_graph(
            {"pos": plus, "atomic_numbers": g.atomic_numbers, "total_charge": -1.0},
            model.model.cutoff))["energy"])
        e_minus = float(model(structure_to_graph(
            {"pos": minus, "atomic_numbers": g.atomic_numbers, "total_charge": -1.0},
            model.model.cutoff))["energy"])
        assert abs(float(forces[atom, axis]) + (e_plus - e_minus) / (2 * h)) < 1e-6


def test_dsf_on_a_cell_matches_the_isolated_molecule_and_gives_stress():
    model = ForceStressOutput(_build(coulomb="dsf", lr_cutoff=6.0), compute_stress=True)
    assert model.model.cutoff == 6.0
    free = model(_molecule(charge=-1.0, cutoff=6.0))
    # a box so large that no periodic image lies within the Coulomb cutoff
    boxed = model(_molecule(charge=-1.0, cutoff=6.0, cell=40.0))
    assert abs(float(free["energy"]) - float(boxed["energy"])) < 1e-9
    assert torch.allclose(free["forces"], boxed["forces"], atol=1e-9)
    assert boxed["stress"].shape == (1, 3, 3)
    # the Coulomb energy differs between the two truncations, nothing else
    simple = _build(coulomb="simple")
    simple.load_state_dict(model.model.state_dict())
    out_s, out_d = simple(_molecule(charge=-1.0)), model.model(_molecule(charge=-1.0, cutoff=6.0))
    assert torch.allclose(out_s["charges"], out_d["charges"], atol=1e-12)
    assert abs(float((out_s["energy"] - out_s["energy_coulomb"])
                     - (out_d["energy"] - out_d["energy_coulomb"]))) < 1e-9


def test_simple_coulomb_rejects_periodic_structures():
    model = _build()
    with pytest.raises(ValueError, match="dsf"):
        model(_molecule(cell=12.0))


def test_unsupported_element_raises():
    model = _build()
    g = _molecule()
    g.atomic_numbers[0] = 17
    with pytest.raises(ValueError, match=r"\[17\]"):
        model(g)


def test_upstream_key_translation():
    cfg = from_dict({"model": {"name": "aimnet2", "rc_s": 4.5, "nfeature": 3, "nshifts_s": 5,
                               "ncomb_v": 2, "num_charge_channels": 2, "rmin": 0.5,
                               "implemented_species": SPECIES,
                               "hidden": "[[8, 6], [8, 6, 6]]", "aim_size": 7}})
    assert cfg.model.cutoff == 4.5 and cfg.model.n_features == 3 and cfg.model.n_rbf == 5
    m = build_model(cfg.model)
    assert m.conv_a.weight.shape == (3, 5, 2) and m.charge_channels == 2
    assert len(m.passes) == 2 and m.node_feature_dim == 7
    assert torch.allclose(m.rbf.centers, 0.5 + torch.arange(5) * (4.0 / 5))


def test_gaussian_basis_layout_and_mollifier():
    m = _build(n_rbf=16, n_features=2)
    assert torch.allclose(m.rbf.centers, 0.8 + torch.arange(16) * (4.2 / 16))
    assert abs(m.rbf.gamma - (16 / 4.2) ** 2) < 1e-12
    fc = MollifierCutoff(4.6)
    r = torch.tensor([0.0, 1.0, 4.59, 4.6, 5.0])
    v = fc(r)
    assert abs(float(v[0]) - 1.0) < 1e-12 and float(v[3]) == 0.0 and float(v[4]) == 0.0
    assert bool((v[:-1][1:] <= v[:-1][:-1]).all())


def test_atom_ref_shifts_stay_float64_and_add_up():
    m = _build()
    shifts = [-13.6, -1029.3, -1485.1, -70045.56521]        # not representable in float32
    m.set_atomic_energies(shifts)
    exact = m.atom_ref.weight.detach().clone()
    m = m.float().double().float()
    assert m.atom_ref.weight.dtype == torch.float64
    assert m.embedding.weight.dtype == torch.float32
    # the casts leave the shifts exact (a float32 pass would round iodine's by 3e-3 eV)
    assert torch.equal(m.atom_ref.weight, exact)
    m = _build()
    g = _molecule()
    e0 = float(m(g)["energy"])
    shifts = [-13.6, -1029.3, -1485.1, -2041.7]
    m.set_atomic_energies(shifts)
    e1 = float(m(g)["energy"])
    expected = sum(shifts[SPECIES.index(int(z))] for z in g.atomic_numbers)
    assert abs((e1 - e0) - expected) < 1e-9


def test_shell_convolution_matches_dense_equations():
    torch.manual_seed(0)
    conv = ShellConvolution(n_channels=3, n_shells=4, n_combinations=2)
    g = _molecule()
    src, dst = g.edge_index
    vec = g.edge_vectors()
    r = vec.norm(dim=-1)
    basis = torch.randn(g.num_edges, 4)
    unit = -vec / r[:, None]
    feats = torch.randn(g.num_nodes, 3, 4)
    out = conv(feats[src], basis, basis[:, :, None] * unit[:, None, :], dst, g.num_nodes)
    # paper eqs 4-5 written atom by atom
    for i in range(g.num_nodes):
        edges = (dst == i).nonzero().flatten()
        scalar = (feats[src[edges]] * basis[edges][:, None, :]).sum(0)          # (C, S)
        vector = (feats[src[edges]][..., None] * basis[edges][:, None, :, None]
                  * unit[edges][:, None, None, :]).sum(0)                       # (C, S, 3)
        mixed = torch.einsum("csh,csd->chd", conv.weight, vector)
        ref = torch.cat([scalar.flatten(), mixed.square().sum(-1).flatten()])
        assert torch.allclose(out[i], ref, atol=1e-12)


def test_equilibration_formula():
    m = _build()
    torch.manual_seed(1)
    q = torch.randn(7, 1)
    f = torch.rand(7, 1)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    total = torch.tensor([[1.0], [-2.0]])
    out = m.equilibrate(q, f, total, batch, 2)
    for b in range(2):
        sel = batch == b
        dq = float(total[b]) - float(q[sel].sum())
        ref = q[sel] + f[sel] / (f[sel].sum() + 1e-6) * dq
        assert torch.allclose(out[sel], ref, atol=1e-12)


def test_collate_and_ase_io_carry_spin_multiplicity():
    a, b = _molecule(seed=1, mult=3.0), _molecule(n=5, seed=2)
    batch = collate([a, b])
    assert torch.allclose(batch.spin_multiplicity, torch.tensor([3.0, 1.0]))
    assert collate([b, b]).spin_multiplicity is None
    ase = pytest.importorskip("ase")
    from xnn.common.data import atoms_to_structure
    atoms = ase.Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    atoms.info["charge"] = -1
    atoms.info["multiplicity"] = 1
    s = atoms_to_structure(atoms)
    assert s["total_charge"] == -1.0 and s["spin_multiplicity"] == 1.0


def test_ase_calculator_passes_charge_and_returns_charges():
    pytest.importorskip("ase")
    from ase import Atoms
    from xnn.common.deploy import XNNCalculator
    model = ForceStressOutput(_build())
    atoms = Atoms("OHH", positions=[[0, 0, 0.12], [0, 0.76, -0.48], [0, -0.76, -0.48]])
    atoms.info["charge"] = -1
    atoms.calc = XNNCalculator(model, cutoff=5.0)
    e_anion = atoms.get_potential_energy()
    assert abs(atoms.calc.results["charges"].sum() + 1.0) < 1e-4
    assert atoms.get_dipole_moment().shape == (3,)
    # the same geometry as a neutral molecule is a new state, not the cached one
    atoms.info["charge"] = 0
    assert atoms.get_potential_energy() != e_anion
    assert abs(atoms.calc.results["charges"].sum()) < 1e-4


# conversion of reference artifacts
def _tiny_artifact(seed=0, charge_channels=1, coulomb=True, d3=True, species=SPECIES):
    """A random artifact in the layout of the published ``.pt`` files."""
    import yaml
    torch.manual_seed(seed)
    f, s, h, aim = TINY["n_features"], TINY["n_rbf"], TINY["n_vector_combinations"], TINY["aim_size"]
    hidden = TINY["hidden"]
    c = charge_channels
    outputs = {
        "energy_mlp": {"class": "aimnet.modules.Output",
                       "kwargs": {"n_in": aim, "n_out": 1, "key_in": "aim", "key_out": "energy",
                                  "mlp": {"activation_fn": "torch.nn.GELU", "last_linear": True,
                                          "hidden": list(TINY["readout_hidden"])}}},
        "atomic_shift": {"class": "aimnet.modules.AtomicShift",
                         "kwargs": {"key_in": "energy", "key_out": "energy"}},
        "atomic_sum": {"class": "aimnet.modules.AtomicSum",
                       "kwargs": {"key_in": "energy", "key_out": "energy"}},
    }
    if coulomb:
        outputs["srcoulomb"] = {"class": "aimnet.modules.SRCoulomb",
                                "kwargs": {"rc": 4.6, "key_in": "charges", "key_out": "energy",
                                           "envelope": "exp"}}
    kwargs = {"nfeature": f, "d2features": True, "ncomb_v": h, "hidden": hidden, "aim_size": aim,
              "aev": {"rc_s": 5.0, "nshifts_s": s}, "outputs": outputs}
    if c == 2:
        kwargs["num_charge_channels"] = 2
    model_yaml = yaml.safe_dump({"class": "aimnet.models.aimnet2.AIMNet2", "kwargs": kwargs})

    def linear(n_in, n_out):
        return {"weight": torch.randn(n_out, n_in, dtype=torch.float32) / math.sqrt(n_in),
                "bias": 0.1 * torch.randn(n_out, dtype=torch.float32)}

    sd = {}
    rc = torch.tensor(5.0, dtype=torch.float32)
    eta = torch.tensor((s / 4.2) ** 2, dtype=torch.float32)
    shifts = torch.linspace(0.8, 5.0, s + 1)[:s].to(torch.float32)
    for mod in ("_s", "_v"):
        sd[f"aev.rc{mod}"], sd[f"aev.eta{mod}"], sd[f"aev.shifts{mod}"] = rc, eta, shifts
    sd["afv.weight"] = torch.randn(64, f * s, dtype=torch.float32)
    sd["afv.weight"][0] = 0.0
    sd["conv_a.agh"] = torch.randn(f, s, h, dtype=torch.float32)
    sd["conv_q.agh"] = torch.randn(c, s, h, dtype=torch.float32)
    conv_a_out, conv_q_out = f * (s + h), c * (s + h)
    n_embed = f * s
    sizes = [[n_embed + conv_a_out, *hidden[0], n_embed + 2 * c]]
    for widths in hidden[1:-1]:
        sizes.append([n_embed + conv_a_out + c + conv_q_out, *widths, n_embed + 2 * c])
    sizes.append([n_embed + conv_a_out + c + conv_q_out, *hidden[-1], aim])
    for i, dims in enumerate(sizes):
        for k in range(len(dims) - 1):
            for name, t in linear(dims[k], dims[k + 1]).items():
                sd[f"mlps.{i}.{2 * k}.{name}"] = t
    dims = [aim, *TINY["readout_hidden"], 1]
    for k in range(len(dims) - 1):
        for name, t in linear(dims[k], dims[k + 1]).items():
            sd[f"outputs.energy_mlp.mlp.{2 * k}.{name}"] = t
    shift = torch.zeros(64, 1, dtype=torch.float64)
    shift[species, 0] = torch.tensor([-13.6, -1029.3, -1485.1, -2041.7])[:len(species)]
    sd["outputs.atomic_shift.shifts.weight"] = shift
    if coulomb:
        sd["outputs.srcoulomb.rc"] = torch.tensor(4.6, dtype=torch.float32)
    return {
        "format_version": 2, "model_yaml": model_yaml, "cutoff": 5.0,
        "needs_coulomb": coulomb, "needs_dispersion": d3,
        "coulomb_mode": "sr_embedded" if coulomb else "none",
        "coulomb_sr_rc": 4.6 if coulomb else None, "coulomb_sr_envelope": "exp" if coulomb else None,
        "d3_params": {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128} if d3 else None,
        "has_embedded_lr": coulomb, "implemented_species": list(species), "state_dict": sd,
    }


def _dense_reference(artifact, g, charge=0.0, mult=1.0):
    """The paper's equations on every pair, straight off the artifact tensors (float64)."""
    import yaml
    kw = yaml.safe_load(artifact["model_yaml"])["kwargs"]
    sd = {k: v.to(torch.float64) for k, v in artifact["state_dict"].items()}
    f, s, c = kw["nfeature"], kw["aev"]["nshifts_s"], kw.get("num_charge_channels", 1)
    z, pos = g.atomic_numbers, g.pos.to(torch.float64)
    n = len(z)
    rc, eta, centers = sd["aev.rc_s"], sd["aev.eta_s"], sd["aev.shifts_s"]
    d = torch.cdist(pos, pos)
    pair = ~torch.eye(n, dtype=torch.bool) & (d < rc)
    fc = 0.5 * (torch.cos(d.clamp(1e-6, float(rc)) * math.pi / rc) + 1.0)
    gs = torch.exp(-eta * (d[..., None] - centers) ** 2) * fc[..., None]
    gs = torch.where(pair[..., None], gs, torch.zeros_like(gs))                  # (N, N, S)
    u = (pos[None, :, :] - pos[:, None, :]) / d.clamp(min=1e-12)[..., None]     # r_j - r_i
    gv = gs[..., None] * u[:, :, None, :]                                        # (N, N, S, 3)

    def conv(feat, agh):                     # feat (N, C, S) indexed by the neighbor j
        xs = torch.einsum("jcs,ijs->ics", feat, gs)
        xv = torch.einsum("jcs,ijsd->icsd", feat, gv)
        v = torch.einsum("csh,icsd->ichd", agh, xv)
        return torch.cat([xs.flatten(1), v.square().sum(-1).flatten(1)], -1)

    def mlp(prefix, x, final_act):
        k = 0
        while f"{prefix}.{k}.weight" in sd:
            x = x @ sd[f"{prefix}.{k}.weight"].T + sd[f"{prefix}.{k}.bias"]
            if f"{prefix}.{k + 2}.weight" in sd or final_act:
                x = torch.nn.functional.gelu(x)
            k += 2
        return x

    half_spin = 0.5 * (mult - 1.0)
    totals = torch.tensor([0.5 * charge + half_spin, 0.5 * charge - half_spin]) if c == 2 \
        else torch.tensor([charge])
    a = sd["afv.weight"][z].view(n, f, s)
    q = None
    n_pass = len(kw["hidden"])
    for i in range(n_pass):
        x = [a.flatten(1), conv(a, sd["conv_a.agh"])]
        if q is not None:
            x += [q, conv(q[:, :, None].expand(n, c, s), sd["conv_q.agh"])]
        out = mlp(f"mlps.{i}", torch.cat(x, -1), final_act=i > 0)
        if i == n_pass - 1:
            aim = out
        else:
            dq, fw, da = out.split([c, c, f * s], -1)
            q = dq if q is None else q + dq
            fw = fw.square()
            q = q + fw / (fw.sum(0) + 1e-6) * (totals - q.sum(0))
            a = a + da.view(n, f, s)
    charges = q.sum(-1)
    energy = mlp("outputs.energy_mlp.mlp", aim, final_act=False).squeeze(-1)
    energy = energy.sum() + sd["outputs.atomic_shift.shifts.weight"][z, 0].sum()
    if "outputs.srcoulomb.rc" in sd:
        rsr = float(sd["outputs.srcoulomb.rc"])
        off = ~torch.eye(n, dtype=torch.bool)
        qq = charges[:, None] * charges[None, :]
        x = (d / rsr).clamp(0.0, 1.0 - 1e-6)
        env = torch.exp(1.0 - 1.0 / (1.0 - x * x))
        e_sr = 0.5 * COULOMB_CONSTANT * (env * qq / d.clamp(min=1e-12))[off].sum()
        e_full = 0.5 * COULOMB_CONSTANT * (qq / d.clamp(min=1e-12))[off].sum()
        energy = energy - e_sr + e_full
    return float(energy), charges


@pytest.mark.parametrize("charge_channels,coulomb", [(1, True), (2, True), (1, False)])
def test_converted_artifact_matches_dense_reference(charge_channels, coulomb):
    art = _tiny_artifact(charge_channels=charge_channels, coulomb=coulomb)
    model, cfg = foundation.convert_artifact(art)
    assert cfg.model.name == "aimnet2" and cfg.subtracted_dispersion["name"] == "d3"
    assert cfg.subtracted_dispersion["s8"] == 0.3908 and cfg.subtracted_dispersion["cutoff_pair"] == 15.0
    assert (cfg.model.extra["coulomb"] == "simple") is coulomb
    model = model.double()
    # weights transplanted row for row
    assert torch.equal(model.embedding.weight, art["state_dict"]["afv.weight"][SPECIES].double())
    assert torch.equal(model.atom_ref.weight[SPECIES],
                       art["state_dict"]["outputs.atomic_shift.shifts.weight"][SPECIES])
    for charge, mult in ((0.0, 1.0), (-1.0, 2.0 if charge_channels == 2 else 1.0)):
        g = _molecule(charge=charge, mult=mult)
        out = model(g)
        e_ref, q_ref = _dense_reference(art, g, charge=charge, mult=mult)
        assert abs(float(out["energy"]) - e_ref) < 1e-9
        assert torch.allclose(out["charges"], q_ref, atol=1e-10)


def test_converter_rejects_unknown_pieces():
    art = _tiny_artifact()
    bad = dict(art, model_yaml=art["model_yaml"].replace("d2features: true", "d2features: false"))
    with pytest.raises(NotImplementedError, match="shared-shell"):
        foundation.convert_artifact(bad)
    extra = dict(art, state_dict={**art["state_dict"], "outputs.lrcoulomb.rc": torch.tensor(4.6)})
    with pytest.raises(NotImplementedError, match="outputs.lrcoulomb.rc"):
        foundation.convert_artifact(extra)
    # the rxn models carry an (unused) atomic-mass table under their dipole heads
    masses = dict(art, state_dict={**art["state_dict"], "outputs.dipole.mass": torch.zeros(119)})
    foundation.convert_artifact(masses)
    legacy = dict(art, coulomb_mode="full_embedded")
    with pytest.raises(NotImplementedError, match="Coulomb mode"):
        foundation.convert_artifact(legacy)


def _write_safetensors(tensors, path):
    header, blobs, offset = {}, [], 0
    codes = {torch.float32: "F32", torch.float64: "F64"}
    for name, t in tensors.items():
        raw = t.contiguous().numpy().tobytes()
        header[name] = {"dtype": codes[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(head)))
        fh.write(head)
        for b in blobs:
            fh.write(b)


def test_hub_loads_pt_artifact_safetensors_directory_and_aliases(tmp_path):
    art = _tiny_artifact()
    direct = foundation.from_aimnet_artifact(art).double()
    g = _molecule(charge=-1.0, cutoff=15.0)
    e_direct = float(direct(_molecule(charge=-1.0))["energy"])

    # the .pt artifact: detected without a format hint, served with its D3 term
    torch.save(art, tmp_path / "tiny.pt")
    served = from_pretrained(tmp_path / "tiny.pt", dtype=torch.float64)
    assert isinstance(served.model, D3Dispersion) and isinstance(served.model.model, AIMNet2)
    assert served.cutoff == 15.0
    out = served(g)
    assert abs(float(out["energy_sr"]) - e_direct) < 1e-9 and float(out["energy_disp"]) < 0.0
    bare = from_pretrained(tmp_path / "tiny.pt", wrap=False, dispersion=False, dtype=torch.float64)
    assert isinstance(bare, AIMNet2) and bare.cutoff == 5.0
    assert abs(float(bare(_molecule(charge=-1.0))["energy"]) - e_direct) < 1e-9
    dsf = from_pretrained(tmp_path / "tiny.pt", wrap=False, dispersion=False,
                          model_options={"coulomb": "dsf", "lr_cutoff": 12.0})
    assert dsf.coulomb == "dsf" and dsf.cutoff == 12.0

    # the same model as a config.json + ensemble_<k>.safetensors directory
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({k: v for k, v in art.items() if k != "state_dict"}))
    _write_safetensors(art["state_dict"], hf / "ensemble_0.safetensors")
    other = dict(art["state_dict"])
    other["afv.weight"] = other["afv.weight"] + 1.0
    _write_safetensors(other, hf / "ensemble_1.safetensors")
    loaded = foundation.load_artifact(hf)
    assert all(torch.equal(loaded["state_dict"][k], v) for k, v in art["state_dict"].items())
    from_dir = from_pretrained(hf, wrap=False, dispersion=False, dtype=torch.float64)
    assert abs(float(from_dir(_molecule(charge=-1.0))["energy"]) - e_direct) < 1e-9
    member1 = from_pretrained(hf, wrap=False, dispersion=False, head="1")
    assert torch.allclose(member1.embedding.weight.double(),
                          direct.embedding.weight + 1.0)
    with pytest.raises(FileNotFoundError, match="member 2"):
        foundation.load_artifact(hf, member=2)

    # a registered name with aliases: converted once into the cache
    register_pretrained(name="tiny-aimnet2-0", url=(tmp_path / "tiny.pt").as_uri(),
                        format="aimnet2", architecture="aimnet2", license="MIT",
                        aliases=["tiny-aimnet2"])
    assert model_card("tiny-aimnet2").name == "tiny-aimnet2-0"
    cache = tmp_path / "cache"
    via_alias = from_pretrained("tiny-aimnet2", cache_dir=cache, quiet=True, dtype=torch.float64)
    slot = cache / "tiny-aimnet2-0"
    assert sorted(p.name for p in slot.iterdir()) == ["card.json", "config.yaml", "model.pt"]
    assert "subtracted_dispersion" in (slot / "config.yaml").read_text()
    again = from_pretrained("tiny-aimnet2-0", cache_dir=cache, local_files_only=True,
                            dtype=torch.float64)
    for m in (via_alias, again):
        assert isinstance(m.model, D3Dispersion)
        assert abs(float(m(g)["energy_sr"]) - e_direct) < 1e-9
    assert "tiny-aimnet2-0" in list_models(cache)
    assert "tiny-aimnet2" not in list_models(cache)
    with pytest.raises(ValueError, match="registered model name"):
        register_pretrained(name="other", url="file:///x", aliases=["tiny-aimnet2-0"])


def test_foundation_registry():
    names = foundation.FOUNDATION_MODELS
    assert len(names) == 24
    for family in ("wb97m-d3", "b973c-d3", "b973c-2025-d3", "nse", "pd", "rxn"):
        for k in range(4):
            assert f"aimnet2-{family}-{k}" in names
    for url, license_ in names.values():
        assert url.startswith("https://") and license_ == "MIT"
    for alias, target in (("aimnet2", "aimnet2-wb97m-d3-0"), ("aimnet2-b973c", "aimnet2-b973c-d3-0"),
                          ("aimnet2-2025", "aimnet2-b973c-2025-d3-0"), ("aimnet2-nse", "aimnet2-nse-0"),
                          ("aimnet2-pd", "aimnet2-pd-0"), ("aimnet2-rxn", "aimnet2-rxn-0")):
        assert foundation.ALIASES[alias] == target
        assert model_card(alias).name == target
    assert model_card("aimnet2-pd").species == [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 46, 53]
    assert set(list_models(architecture="aimnet2")) == set(names)


# the published checkpoints (served, with their D3 term) against the reference
# package: energies in eV, forces in eV/A, charges in e, as computed by the aimnet
# package (0.2.0, float32, GPU) on the perturbed geometries below; skipped unless the
# model is in the cache. The tolerances are float32 round-off on energies of
# thousands of eV.
REFERENCE = {
    'aimnet2-wb97m-d3-0': [
        {"name": "water", "numbers": [8, 1, 1], "pos": [[0.017279, 0.041081, 0.135784], [-0.065158, 0.808507, -0.454728], [-0.026848, -0.734183, -0.458818]], "energy": -2081.03248, "forces": [[-0.09562, -0.23245, -0.94852], [0.05435, -0.52156, 0.38569], [0.04126, 0.75401, 0.56283]], "charges": [-0.68884, 0.34495, 0.34389]},
        {"name": "hydronium", "numbers": [8, 1, 1, 1], "pos": [[0.102046, -0.127783, 0.020905], [0.951612, -0.022632, 0.08922], [-0.590999, 0.837103, 0.056739], [-0.32385, -0.837411, 0.082368]], "charge": 1.0, "energy": -2086.63135, "forces": [[-5.86675, 13.35743, -1.5444], [7.78128, 0.8323, 0.74455], [3.42347, -4.3385, -0.12827], [-5.33801, -9.85122, 0.92812]], "charges": [-0.33389, 0.42393, 0.49595, 0.41402]},
        {"name": "acetate", "numbers": [6, 6, 8, 8, 1, 1, 1], "pos": [[-0.03259, -0.008736, 0.083186], [1.552957, -0.08207, -0.00026], [2.088827, 1.107432, -0.080409], [2.132089, -1.088231, 0.078781], [-0.354168, 0.545527, 0.805344], [-0.257364, 0.424218, -0.82491], [-0.386495, -1.074032, -0.032814]], "charge": -1.0, "energy": -6221.64989, "forces": [[3.35124, -5.72053, -3.90701], [-1.54798, 7.92146, 0.55721], [-1.18789, -1.49867, -0.13054], [2.40443, -6.00334, 0.17596], [-2.01075, 2.93936, 4.50479], [-0.92041, 1.31122, -1.98792], [-0.08864, 1.0505, 0.78751]], "charges": [-0.36853, 0.48641, -0.72264, -0.61562, 0.09539, 0.07741, 0.04759]},
    ],
    'aimnet2-nse-0': [
        {"name": "methyl-radical", "numbers": [6, 1, 1, 1], "pos": [[-0.040097, -0.066218, -0.012418], [0.021022, 1.135212, 0.005485], [0.906298, -0.578444, 0.037437], [-0.852191, -0.525567, -0.061666]], "mult": 2.0, "energy": -1083.87794, "forces": [[6.83242, 6.93386, 0.45721], [-0.27729, -3.30448, -0.05682], [0.01368, -0.03878, -0.0014], [-6.56881, -3.5906, -0.39899]], "charges": [-0.06779, 0.04175, 0.02056, 0.00546]},
        {"name": "hydroxyl-radical", "numbers": [8, 1], "pos": [[0.052656, 0.088825, -0.018879], [-0.006898, 0.050686, -0.802677]], "mult": 2.0, "energy": -2060.85441, "forces": [[1.25733, 0.80521, 16.54793], [-1.25733, -0.80521, -16.54793]], "charges": [-0.1242, 0.12418]},
    ],
    'aimnet2-pd-0': [
        {"name": "pd-complex", "numbers": [46, 17, 17, 7, 7, 1, 1, 1, 1, 1, 1], "pos": [[-0.040142, 0.012142, -0.082817], [2.332805, 0.057173, -0.022631], [-2.278476, 0.012547, -0.019718], [-0.04312, 1.948372, 0.070521], [-0.002382, -1.923884, 0.041311], [0.013889, 2.367128, 1.019623], [0.794686, 2.478497, -0.489918], [-0.810702, 2.323866, -0.352838], [-0.004701, -2.419256, 0.990542], [0.775431, -2.36162, -0.528562], [-0.792738, -2.452206, -0.561853]], "energy": -31611.134, "forces": [[1.39, -0.0986, 0.64216], [-0.02584, -0.02186, -0.06632], [-1.26441, -0.07019, -0.03089], [5.26799, 3.25867, -0.29844], [-1.21907, -5.49731, -2.4552], [-0.17495, -0.61115, -0.74004], [-2.42703, -2.08166, 1.88118], [-2.81108, 1.26781, -1.09159], [0.10182, 0.95324, -1.5941], [-0.3888, 0.97164, 1.50391], [1.55139, 1.92941, 2.24933]], "charges": [0.22501, -0.50473, -0.45404, -0.545, -0.52533, 0.31418, 0.28297, 0.31181, 0.31594, 0.29521, 0.28397]},
    ],
    'aimnet2-rxn-0': [
        {"name": "ethanol", "numbers": [6, 6, 8, 1, 1, 1, 1, 1, 1], "pos": [[1.177634, -0.426519, -0.020653], [-0.122073, 0.649447, 0.057208], [-1.206354, -0.188979, 0.014061], [-1.974314, 0.430403, -0.015528], [0.026116, 1.167901, 0.909681], [0.037597, 1.234772, -0.917292], [2.122232, 0.100186, 0.042073], [1.138001, -1.020705, 0.906406], [1.078061, -0.998075, -0.783046]], "energy": -0.06104, "forces": [[-1.37492, 5.07142, 6.34343], [4.58879, -3.42518, -4.82186], [-2.6276, 0.99173, 0.13014], [0.57965, -1.22185, 0.03429], [0.00935, 2.10442, 2.68174], [-0.61769, -0.19295, 1.55706], [0.11473, 0.16945, -0.43167], [-0.14848, 0.05152, -0.40346], [-0.52384, -3.54856, -5.08967]], "charges": [-0.28925, 0.01996, -0.51117, 0.37406, 0.08332, 0.04769, 0.06294, 0.08658, 0.12585]},
    ],
}



def _cached_pretrained(name, **kwargs):
    from xnn.common.models.hub.cache import is_ready
    cache = os.environ.get("XNN_MODELS") or _CACHE_AT_IMPORT
    slot = model_card(name).name                   # an alias resolves to its card
    if cache is None or not is_ready(Path(cache) / slot):
        pytest.skip(f"{slot} is not in the model cache (tests never download)")
    return from_pretrained(name, cache_dir=cache, local_files_only=True, **kwargs)


def test_published_model_structure():
    model = _cached_pretrained("aimnet2")
    core = model.model.model
    assert isinstance(model.model, D3Dispersion) and isinstance(core, AIMNet2)
    assert core.species == [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]
    assert core.n_features == 16 and core.n_rbf == 16 and len(core.passes) == 3
    assert core.aim_size == 256 and abs(core.coulomb_sr_cutoff - 4.6) < 1e-6
    assert core.atom_ref.weight.dtype == torch.float64 and core.embedding.weight.dtype == torch.float32
    assert model.cutoff == 15.0
    water = structure_to_graph({"pos": np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                                                 [0.0, -0.763239, -0.477047]]),
                                "atomic_numbers": [8, 1, 1]}, 15.0)
    water.compute_dtype = torch.float32
    out = model(water)
    assert abs(float(out["charges"].sum())) < 1e-5
    assert float(out["charges"][0]) < -0.5                      # oxygen carries the negative charge
    assert -2200.0 < float(out["energy"]) < -1900.0             # a wB97M total energy in eV


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_published_models_reproduce_reference_values(name):
    model = _cached_pretrained(name)
    for case in REFERENCE[name]:
        g = structure_to_graph({"pos": np.array(case["pos"]), "atomic_numbers": case["numbers"],
                                "total_charge": case.get("charge", 0.0),
                                "spin_multiplicity": case.get("mult", 1.0)}, model.cutoff)
        g.compute_dtype = torch.float32
        out = model(g)
        assert abs(float(out["energy"]) - case["energy"]) < 1e-3
        assert np.abs(out["forces"].detach().numpy() - np.array(case["forces"])).max() < 2e-3
        assert np.abs(out["charges"].detach().numpy() - np.array(case["charges"])).max() < 1e-4


def test_stress_matches_finite_difference_of_the_energy():
    """The strain derivative of a periodic DSF model equals the energy change under
    an isotropic scaling of positions and cell: ``dE/d eps = V tr(sigma)``."""
    model = ForceStressOutput(_build(coulomb="dsf", lr_cutoff=6.0), compute_stress=True)
    # a dense periodic cluster: images within both the local and the Coulomb cutoff
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.0, 7.0, (10, 3))
    z = ([8, 1, 1, 6, 7, 1, 6, 8, 1, 1])

    def energy(scale):
        g = structure_to_graph({"pos": pos * scale, "atomic_numbers": z, "total_charge": -1.0,
                                "cell": np.eye(3) * 7.0 * scale, "pbc": [True] * 3}, 6.0)
        return model(g)

    out = energy(1.0)
    volume = 7.0 ** 3
    trace = float(out["stress"][0].trace())
    h = 1e-4
    de = (float(energy(1.0 + h)["energy"]) - float(energy(1.0 - h)["energy"])) / (2 * h)
    assert abs(volume * trace - de) < 1e-5 * max(1.0, abs(de))


# Ewald and particle-mesh Ewald electrostatics
MADELUNG_NACL = 1.747564594633182       # rock salt, per ion pair, nearest-neighbor distance unit


def _rock_salt(a=5.64):
    """The conventional NaCl cell: four +1 and four -1 point charges."""
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float)
    pos = np.concatenate([base, base + [0.5, 0, 0]]) * a
    charges = torch.tensor([1.0] * 4 + [-1.0] * 4, dtype=torch.float64)
    return torch.tensor(pos), charges, torch.eye(3, dtype=torch.float64) * a


def _periodic_pairs(pos, cell, cutoff):
    from xnn.common.data.neighborlist import build_neighbor_list
    edge_index, shifts = build_neighbor_list(pos, cutoff, cell, torch.ones(3, dtype=torch.bool))
    vec = pos[edge_index[1]] - pos[edge_index[0]] + shifts.to(pos.dtype) @ cell
    return edge_index, torch.linalg.norm(vec, dim=-1)


@pytest.mark.parametrize("method, accuracy, tol", [("ewald", 1e-8, 1e-7), ("pme", 1e-6, 2e-5)])
def test_lattice_sums_reproduce_the_madelung_constant(method, accuracy, tol):
    """The Ewald and PME kernels on rock salt give the Madelung energy ``-M k_e / d`` per pair."""
    from xnn.common.models.electrostatics import (coulomb_ewald, coulomb_pme, ewald_mesh_sizes,
                                                  ewald_parameters)
    pos, charges, cell = _rock_salt()
    cutoff = 8.0
    alpha, k_cutoff = ewald_parameters(cutoff, accuracy)
    edge_index, r = _periodic_pairs(pos, cell, cutoff)
    if method == "ewald":
        node = coulomb_ewald(charges, pos, cell, edge_index, r, alpha, cutoff, k_cutoff)
    else:
        node = coulomb_pme(charges, pos, cell, edge_index, r, alpha, cutoff, accuracy, 4)
        assert ewald_mesh_sizes(cell, alpha, accuracy) == [32, 32, 32]
    exact = -4.0 * MADELUNG_NACL * COULOMB_CONSTANT / (5.64 / 2)
    assert node.dtype == torch.float64 and node.shape == (8,)
    assert abs(float(node.sum()) - exact) < tol * abs(exact)
    # every ion sees the same lattice: equal per-atom shares
    assert torch.allclose(node, node.mean() * torch.ones_like(node), atol=tol * abs(exact))


def test_lattice_sums_do_not_depend_on_the_splitting_and_agree_with_each_other():
    """A charged, random cell: the Ewald energy at two real-space cutoffs (two splitting
    parameters), the PME energy and the neutralizing background all agree to the accuracy."""
    from xnn.common.models.electrostatics import coulomb_ewald, coulomb_pme, ewald_parameters
    rng = np.random.default_rng(3)
    pos = torch.tensor(rng.uniform(0.0, 9.0, (20, 3)))
    charges = torch.tensor(rng.normal(size=20))
    charges[0] += 1.0                                  # a net charge, so the background term matters
    cell = torch.tensor(np.diag([9.0, 10.0, 11.0]) + rng.normal(scale=0.4, size=(3, 3)))
    energies = {}
    for method, cutoff in (("ewald", 6.0), ("ewald", 9.0), ("pme", 7.0)):
        alpha, k_cutoff = ewald_parameters(cutoff, 1e-7)
        edge_index, r = _periodic_pairs(pos, cell, cutoff)
        if method == "ewald":
            node = coulomb_ewald(charges, pos, cell, edge_index, r, alpha, cutoff, k_cutoff)
        else:
            node = coulomb_pme(charges, pos, cell, edge_index, r, alpha, cutoff, 1e-7, 5)
        energies[(method, cutoff)] = float(node.sum())
    values = list(energies.values())
    scale = max(1.0, abs(values[0]))
    assert abs(values[0] - values[1]) < 1e-6 * scale
    assert abs(values[0] - values[2]) < 1e-5 * scale


@pytest.mark.parametrize("coulomb", ["ewald", "pme"])
def test_lattice_sum_forces_and_stress_match_finite_differences(coulomb):
    model = ForceStressOutput(_build(coulomb=coulomb, lr_cutoff=6.0), compute_stress=True)
    assert model.model.cutoff == 6.0
    rng = np.random.default_rng(7)
    pos = rng.uniform(0.0, 7.0, (10, 3))
    z = [8, 1, 1, 6, 7, 1, 6, 8, 1, 1]

    def energy(p, scale=1.0):
        g = structure_to_graph({"pos": p * scale, "atomic_numbers": z, "total_charge": -1.0,
                                "cell": np.eye(3) * 7.0 * scale, "pbc": [True] * 3}, 6.0)
        return model(g)

    out = energy(pos)
    assert abs(float(out["charges"].sum()) + 1.0) < 1e-4
    h = 1e-4
    for atom, axis in ((0, 0), (3, 2), (7, 1)):
        plus, minus = pos.copy(), pos.copy()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        de = (float(energy(plus)["energy"]) - float(energy(minus)["energy"])) / (2 * h)
        assert abs(float(out["forces"][atom, axis]) + de) < 1e-6
    de = (float(energy(pos, 1.0 + h)["energy"]) - float(energy(pos, 1.0 - h)["energy"])) / (2 * h)
    assert abs(7.0 ** 3 * float(out["stress"][0].trace()) - de) < 1e-5 * max(1.0, abs(de))


def test_lattice_sum_models_mix_periodic_and_free_structures():
    """Under ``ewald``/``pme`` a molecule of the batch gets the all-pairs sum (the same
    energy as ``coulomb="simple"``), a periodic cell the lattice sum, and both agree with
    their single-structure evaluations; a slab is refused."""
    ewald = _build(coulomb="ewald", lr_cutoff=6.0)
    simple = _build(coulomb="simple")
    simple.load_state_dict(ewald.state_dict())
    free = _molecule(charge=-1.0, cutoff=6.0)
    cell = _molecule(n=6, seed=2, charge=1.0, cutoff=6.0, cell=6.5)
    both = ewald(collate([free, cell]))
    e_free, e_cell = ewald(free), ewald(cell)
    assert abs(float(both["energy"][0]) - float(e_free["energy"])) < 1e-9
    assert abs(float(both["energy"][1]) - float(e_cell["energy"])) < 1e-9
    assert abs(float(e_free["energy"]) - float(simple(_molecule(charge=-1.0))["energy"])) < 1e-9
    # the periodic structure's Coulomb energy differs from the DSF truncation's, nothing else
    dsf = _build(coulomb="dsf", lr_cutoff=6.0)
    dsf.load_state_dict(ewald.state_dict())
    o = dsf(cell)
    assert torch.allclose(o["charges"], e_cell["charges"], atol=1e-10)
    assert abs(float((o["energy"] - o["energy_coulomb"])
                     - (e_cell["energy"] - e_cell["energy_coulomb"]))) < 1e-9
    assert abs(float(o["energy_coulomb"] - e_cell["energy_coulomb"])) > 1e-6
    slab = _molecule(n=6, seed=2, cutoff=6.0, cell=6.5)
    slab.pbc[0, 2] = False
    with pytest.raises(ValueError, match="three directions"):
        ewald(slab)
    with pytest.raises(ValueError, match="ewald_accuracy"):
        _build(coulomb="pme", ewald_accuracy=2.0)


# TorchScript export
@pytest.mark.parametrize("coulomb, periodic", [("simple", False), ("dsf", True), ("ewald", True),
                                               ("pme", True), (None, False)])
def test_scripted_potential_matches_eager(coulomb, periodic, tmp_path):
    from xnn.common.data.neighborlist import build_neighbor_list
    from xnn.common.deploy import TorchScriptPotential, export_torchscript_potential
    model = _build(coulomb=coulomb, lr_cutoff=6.0).eval()
    rng = np.random.default_rng(5)
    pos = rng.uniform(0.0, 6.0, (9, 3))
    z = [8, 1, 1, 6, 7, 1, 6, 8, 1]
    s = {"pos": pos, "atomic_numbers": z, "total_charge": -1.0}
    if periodic:
        s["cell"], s["pbc"] = np.eye(3) * 6.5, [True] * 3
    eager = ForceStressOutput(model, compute_stress=True)(structure_to_graph(s, model.cutoff))

    wrapper = TorchScriptPotential(model, model.cutoff, total_charge=-1.0).eval()
    assert wrapper.has_charges
    scripted = torch.jit.script(wrapper)
    cell = torch.eye(3) * 6.5 if periodic else None
    pbc = torch.tensor([periodic] * 3)
    out = scripted(torch.tensor(pos), torch.tensor(z), cell, pbc)
    assert abs(float(out["energy"]) - float(eager["energy"])) < 1e-9
    assert torch.allclose(out["forces"], eager["forces"], atol=1e-8)
    assert torch.allclose(out["charges"], eager["charges"], atol=1e-10)
    assert abs(float(out["node_energy"].sum()) - float(out["energy"])) < 1e-9
    if periodic:
        assert torch.allclose(out["stress"], eager["stress"][0], atol=1e-8)
        ei, cs = build_neighbor_list(torch.tensor(pos), model.cutoff, cell, pbc)
        pair = scripted.forward_lammps(torch.tensor(pos), ei, cs, torch.tensor(z), cell)
        assert abs(float(pair["energy"]) - float(out["energy"])) < 1e-9
    # the artifact round trip, with its metadata
    path = str(tmp_path / "aimnet2.pt")
    export_torchscript_potential(model, model.cutoff, path, total_charge=-1.0)
    extra = {"charges": "", "total_charge": "", "spin_multiplicity": ""}
    loaded = torch.jit.load(path, _extra_files=extra)
    again = loaded(torch.tensor(pos), torch.tensor(z), cell, pbc)
    assert abs(float(again["energy"]) - float(out["energy"])) < 1e-9
    assert extra["charges"].decode() == "True" and extra["total_charge"].decode() == "-1.0"


def test_scripted_two_channel_model_reads_the_spin_multiplicity():
    from xnn.common.deploy import TorchScriptPotential
    model = _build(charge_channels=2).eval()
    g = _molecule(charge=1.0, mult=3.0)
    eager = model(g)
    scripted = torch.jit.script(TorchScriptPotential(model, model.cutoff, total_charge=1.0,
                                                     spin_multiplicity=3.0).eval())
    out = scripted(g.pos, g.atomic_numbers)
    assert abs(float(out["energy"]) - float(eager["energy"])) < 1e-9
    assert torch.allclose(out["charges"], eager["charges"], atol=1e-10)
    singlet = torch.jit.script(TorchScriptPotential(model, model.cutoff, total_charge=1.0).eval())
    assert abs(float(singlet(g.pos, g.atomic_numbers)["energy"]) - float(eager["energy"])) > 1e-6


# the reference package's Coulomb kernels (nvalchemiops 0.4.1, the backend of aimnet 0.2.0) on fixed
# point charges, in eV, with xnn's parameters (alpha = sqrt(-ln eps) / r_c, the reciprocal sphere handed
# over as explicit lattice vectors, the same mesh and spline order), both precisions; "ewald_box" is the
# kernel with its own lattice box, "dsf" its damped shifted-force sum at alpha 0.2
KERNEL_REFERENCE = {
    'nacl': {
        'cell': [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        ('float32', 8.0, 1e-06): {"ewald": -35.69404395, "ewald_box": -35.69403729, "pme4": -35.69402408, "pme5": -35.69403438, "dsf": -34.9703879},
        ('float32', 8.0, 1e-08): {"ewald": -35.69402371, "ewald_box": -35.69402614, "pme4": -35.69402609, "pme5": -35.69402609, "dsf": -34.9703879},
        ('float64', 8.0, 1e-06): {"ewald": -35.69404375, "ewald_box": -35.69403603, "pme4": -35.6940244, "pme5": -35.69403598, "dsf": -34.97041344},
        ('float64', 8.0, 1e-08): {"ewald": -35.69402448, "ewald_box": -35.69402444, "pme4": -35.69402431, "pme5": -35.69402444, "dsf": -34.97041344},
    },
    'random-neutral': {
        'cell': [[8.312726, 0.67273, 0.301111], [0.301426, 10.455153, 0.139691], [-0.255699, -0.320096, 10.67992]],
        ('float32', 6.0, 1e-06): {"ewald": -67.99476008, "ewald_box": -67.99475317, "pme4": -67.99491002, "pme5": -67.99503394, "dsf": -67.55845372},
        ('float32', 9.0, 1e-08): {"ewald": -67.99474403, "ewald_box": -67.994755, "pme4": -67.99474405, "pme5": -67.99429245, "dsf": -69.97877632},
        ('float64', 6.0, 1e-06): {"ewald": -67.99475803, "ewald_box": -67.99473679, "pme4": -67.99474059, "pme5": -67.99473851, "dsf": -67.55844741},
        ('float64', 9.0, 1e-08): {"ewald": -67.99474281, "ewald_box": -67.99474269, "pme4": -67.99474301, "pme5": -67.99474385, "dsf": -69.9787683},
    },
    'random-charged': {
        'cell': [[8.312726, 0.67273, 0.301111], [0.301426, 10.455153, 0.139691], [-0.255699, -0.320096, 10.67992]],
        ('float32', 6.0, 1e-06): {"ewald": -70.89267392, "ewald_box": -70.89266698, "pme4": -70.89280117, "pme5": -70.89289259, "dsf": -67.3132733},
        ('float32', 9.0, 1e-08): {"ewald": -70.89264941, "ewald_box": -70.89266216, "pme4": -70.89265227, "pme5": -70.89224882, "dsf": -66.71257396},
        ('float64', 6.0, 1e-06): {"ewald": -70.8926677, "ewald_box": -70.89264386, "pme4": -70.892646, "pme5": -70.89264577, "dsf": -67.31326013},
        ('float64', 9.0, 1e-08): {"ewald": -70.89264394, "ewald_box": -70.89264382, "pme4": -70.89264416, "pme5": -70.89264514, "dsf": -66.71256186},
    },
    'co2-small': {
        'cell': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ('float32', 6.0, 1e-06): {"ewald": -41.95251716, "ewald_box": -41.95251143, "pme4": -41.95251303, "pme5": -41.95252521, "dsf": -39.42147029},
        ('float32', 9.0, 1e-06): {"ewald": -41.95250726, "ewald_box": -41.95250036, "pme4": -41.95250003, "pme5": -41.95249885, "dsf": -41.56916324},
        ('float64', 6.0, 1e-06): {"ewald": -41.95251957, "ewald_box": -41.95251384, "pme4": -41.95251388, "pme5": -41.95251387, "dsf": -39.4214689},
        ('float64', 9.0, 1e-06): {"ewald": -41.95250974, "ewald_box": -41.95250285, "pme4": -41.95250314, "pme5": -41.95250285, "dsf": -41.56916527},
    },
}

# the served general model (member 0, float32, GPU) on a 24-atom CO2 cell with the DSF, Ewald and PME
# sums of the reference calculator (its own Kolafa-Perram parameters: alpha 0.301, real-space cutoff
# 12.35 A, 32^3 mesh), with and without the D3 term; energies in eV, forces in eV/A, charges in e
PERIODIC_REFERENCE = {
    'numbers': [6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8, 6, 8, 8],
    'pos': [[2.907924, 2.867416, 2.346908], [2.504486, 1.912303, 2.907471], [3.311362, 3.82253, 1.786345], [2.410609, 2.341785, 7.670918], [3.035991, 1.472848, 7.177887], [1.785227, 3.210722, 8.163949], [2.724066, 6.945803, 2.969965], [2.131218, 6.193014, 3.656314], [3.316913, 7.698591, 2.283615], [2.47107, 7.704114, 7.45903], [1.295398, 7.621047, 7.447712], [3.646743, 7.78718, 7.470348], [7.638933, 2.747354, 2.439241], [7.781451, 3.494088, 3.33997], [7.496416, 2.00062, 1.538512], [7.454164, 2.70571, 7.238898], [6.874856, 2.342176, 6.27896], [8.033473, 3.069243, 8.198835], [7.618495, 7.29883, 1.923898], [8.425006, 6.524429, 1.550968], [6.811983, 8.073231, 2.296827], [7.255784, 7.359721, 7.142039], [7.636471, 6.296651, 6.804111], [6.875097, 8.42279, 7.479967]],
    'cell': [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    'runs': {
        'float32/dsf15': {"energy": -41081.13486, "forces_max": 2.26017, "forces": [[-0.06502, -0.09485, 0.0961], [0.76859, 1.94179, -1.1398], [-0.74177, -1.70687, 1.04335], [-0.09566, 0.14881, 0.07024], [-1.23902, 1.58923, 0.96722], [1.26933, -1.75807, -0.97953], [0.19622, 0.19095, -0.29981], [1.08786, 1.30189, -1.21981], [-1.27018, -1.64148, 1.41683], [0.10223, 0.02435, 0.04227], [2.20904, 0.19439, 0.02729], [-2.26017, -0.20832, -0.04094], [0.00658, -0.09653, -0.04952], [-0.27391, -1.41935, -1.69219], [0.26952, 1.45176, 1.7275], [0.13157, 0.03656, 0.21061], [1.15194, 0.72484, 1.83484], [-1.2257, -0.78893, -2.04636], [-0.14865, 0.17958, 0.06218], [-1.46749, 1.44304, 0.6971], [1.64057, -1.58173, -0.71715], [-0.1184, 0.23055, 0.05805], [-0.7336, 2.01262, 0.63425], [0.80612, -2.17423, -0.70272]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "dsf", "cutoff": 15.0, "ewald_accuracy": 1e-06, "d3": False}},
        'float32/dsf15/d3': {"energy": -41081.73979, "forces_max": 2.25462, "forces": [[-0.06461, -0.08628, 0.09613], [0.77026, 1.94619, -1.14052], [-0.74479, -1.6991, 1.04811], [-0.09546, 0.14777, 0.07137], [-1.2387, 1.58551, 0.9681], [1.26857, -1.75909, -0.97837], [0.19854, 0.1821, -0.30016], [1.09179, 1.29666, -1.22331], [-1.26691, -1.64629, 1.41854], [0.10464, 0.02666, 0.03839], [2.20936, 0.19374, 0.02491], [-2.25462, -0.20584, -0.0433], [0.00447, -0.09162, -0.04855], [-0.27617, -1.41723, -1.69008], [0.26886, 1.45527, 1.72639], [0.13209, 0.04153, 0.20863], [1.15375, 0.72839, 1.83165], [-1.2253, -0.78632, -2.04635], [-0.15077, 0.17517, 0.06235], [-1.47016, 1.43848, 0.69987], [1.63789, -1.58446, -0.718], [-0.12084, 0.22838, 0.06009], [-0.73381, 2.0068, 0.63698], [0.80193, -2.17641, -0.70287]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "dsf", "cutoff": 15.0, "ewald_accuracy": 1e-06, "d3": True}},
        'float32/dsf9': {"energy": -41080.89184, "forces_max": 2.23516, "forces": [[-0.07143, -0.10175, 0.0998], [0.76054, 1.92371, -1.12995], [-0.72752, -1.68059, 1.02914], [-0.09909, 0.15445, 0.07038], [-1.22398, 1.56725, 0.95728], [1.25191, -1.73833, -0.96724], [0.1947, 0.19067, -0.304], [1.07686, 1.28408, -1.20703], [-1.25154, -1.62631, 1.3995], [0.09898, 0.02344, 0.04307], [2.18461, 0.19091, 0.02972], [-2.23516, -0.2061, -0.03604], [0.00639, -0.09701, -0.05265], [-0.27331, -1.39918, -1.67103], [0.26809, 1.44348, 1.70892], [0.12942, 0.03136, 0.20588], [1.14531, 0.72001, 1.81391], [-1.21084, -0.77781, -2.02247], [-0.14792, 0.17811, 0.06735], [-1.45121, 1.41838, 0.68387], [1.61795, -1.5672, -0.71], [-0.11456, 0.22991, 0.05515], [-0.72465, 1.99041, 0.6294], [0.79645, -2.1519, -0.69296]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "dsf", "cutoff": 9.0, "ewald_accuracy": 1e-06, "d3": False}},
        'float32/dsf9/d3': {"energy": -41081.47939, "forces_max": 2.22965, "forces": [[-0.07101, -0.0932, 0.09991], [0.76231, 1.92819, -1.13068], [-0.73062, -1.6729, 1.03399], [-0.09885, 0.15338, 0.07161], [-1.22374, 1.5636, 0.9583], [1.25124, -1.73944, -0.96609], [0.19693, 0.18187, -0.30446], [1.0808, 1.27898, -1.21064], [-1.24837, -1.6312, 1.40121], [0.10134, 0.0257, 0.03919], [2.18499, 0.19021, 0.02738], [-2.22965, -0.20364, -0.03843], [0.00437, -0.09221, -0.05169], [-0.27561, -1.39714, -1.66899], [0.2675, 1.44699, 1.70788], [0.12992, 0.03637, 0.20385], [1.14716, 0.72356, 1.81076], [-1.21054, -0.77512, -2.02259], [-0.15008, 0.17382, 0.06744], [-1.45391, 1.41394, 0.68665], [1.61536, -1.56996, -0.71089], [-0.11701, 0.22768, 0.05718], [-0.72478, 1.98468, 0.63215], [0.79224, -2.15416, -0.69304]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "dsf", "cutoff": 9.0, "ewald_accuracy": 1e-06, "d3": True}},
        'float32/ewald': {"energy": -41081.13328, "forces_max": 2.26242, "forces": [[-0.06627, -0.10233, 0.09893], [0.76856, 1.94396, -1.14001], [-0.74069, -1.70125, 1.04128], [-0.09381, 0.14326, 0.06826], [-1.2407, 1.59104, 0.96737], [1.26865, -1.75495, -0.97804], [0.20201, 0.19982, -0.30535], [1.08438, 1.29566, -1.2153], [-1.27244, -1.64435, 1.41834], [0.10508, 0.02662, 0.04304], [2.20868, 0.19352, 0.02638], [-2.26242, -0.20957, -0.04149], [0.0055, -0.10387, -0.05565], [-0.27321, -1.41672, -1.69084], [0.27013, 1.45647, 1.73163], [0.13138, 0.03383, 0.21146], [1.15327, 0.72739, 1.83668], [-1.22659, -0.78874, -2.04831], [-0.15298, 0.18577, 0.06222], [-1.46493, 1.44095, 0.69772], [1.64236, -1.58556, -0.71787], [-0.12048, 0.239, 0.06164], [-0.73298, 2.0112, 0.63282], [0.80749, -2.18113, -0.70492]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "ewald", "cutoff": None, "ewald_accuracy": 1e-06, "d3": False, "alpha": 0.3010302527893431, "real_space_cutoff": 12.347337699147769, "k_cutoff": 2.237812052215571}},
        'float32/ewald/d3': {"energy": -41081.7382, "forces_max": 2.25688, "forces": [[-0.06585, -0.09376, 0.09896], [0.77023, 1.94835, -1.14072], [-0.74372, -1.69348, 1.04604], [-0.09362, 0.14223, 0.06939], [-1.24039, 1.58732, 0.96824], [1.2679, -1.75597, -0.97687], [0.20433, 0.19097, -0.30571], [1.08831, 1.29044, -1.2188], [-1.26917, -1.64915, 1.42005], [0.10749, 0.02892, 0.03917], [2.209, 0.19287, 0.02401], [-2.25688, -0.20709, -0.04385], [0.0034, -0.09897, -0.05468], [-0.27547, -1.41461, -1.68873], [0.26947, 1.45998, 1.73052], [0.13191, 0.0388, 0.20948], [1.15508, 0.73093, 1.83349], [-1.22619, -0.78614, -2.0483], [-0.1551, 0.18135, 0.06239], [-1.4676, 1.43639, 0.70049], [1.63969, -1.58828, -0.71872], [-0.12292, 0.23682, 0.06369], [-0.7332, 2.00539, 0.63554], [0.80331, -2.18331, -0.70506]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "ewald", "cutoff": None, "ewald_accuracy": 1e-06, "d3": True, "alpha": 0.3010302527893431, "real_space_cutoff": 12.347337699147769, "k_cutoff": 2.237812052215571}},
        'float32/pme': {"energy": -41081.13328, "forces_max": 2.26242, "forces": [[-0.06626, -0.10233, 0.09893], [0.76856, 1.94396, -1.14001], [-0.74069, -1.70126, 1.04128], [-0.09381, 0.14326, 0.06826], [-1.2407, 1.59104, 0.96737], [1.26865, -1.75495, -0.97804], [0.20201, 0.19982, -0.30535], [1.08438, 1.29566, -1.2153], [-1.27244, -1.64435, 1.41834], [0.10508, 0.02661, 0.04304], [2.20868, 0.19352, 0.02638], [-2.26242, -0.20957, -0.04149], [0.0055, -0.10388, -0.05565], [-0.27321, -1.41672, -1.69084], [0.27013, 1.45648, 1.73164], [0.13138, 0.03383, 0.21146], [1.15327, 0.72739, 1.83668], [-1.22659, -0.78874, -2.04831], [-0.15297, 0.18577, 0.06222], [-1.46493, 1.44095, 0.69772], [1.64236, -1.58556, -0.71787], [-0.12048, 0.239, 0.06164], [-0.73298, 2.0112, 0.63282], [0.80749, -2.18114, -0.70492]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "pme", "cutoff": None, "ewald_accuracy": 1e-06, "d3": False, "alpha": 0.3010302527893431, "real_space_cutoff": 12.347337699147769, "mesh": [32, 32, 32]}},
        'float32/pme/d3': {"energy": -41081.7382, "forces_max": 2.25688, "forces": [[-0.06585, -0.09376, 0.09896], [0.77023, 1.94835, -1.14073], [-0.74371, -1.69348, 1.04604], [-0.09362, 0.14223, 0.06939], [-1.24039, 1.58732, 0.96824], [1.2679, -1.75597, -0.97687], [0.20432, 0.19097, -0.30571], [1.08831, 1.29044, -1.2188], [-1.26917, -1.64915, 1.42005], [0.10749, 0.02892, 0.03917], [2.209, 0.19287, 0.02401], [-2.25688, -0.20709, -0.04385], [0.0034, -0.09897, -0.05468], [-0.27547, -1.41461, -1.68873], [0.26947, 1.45998, 1.73052], [0.13191, 0.0388, 0.20948], [1.15508, 0.73093, 1.83349], [-1.22619, -0.78614, -2.0483], [-0.1551, 0.18136, 0.06239], [-1.4676, 1.43639, 0.7005], [1.63969, -1.58828, -0.71872], [-0.12292, 0.23682, 0.06369], [-0.7332, 2.00539, 0.63554], [0.80331, -2.18331, -0.70506]], "charges": [0.57061, -0.29408, -0.27417, 0.56581, -0.27499, -0.28608, 0.5506, -0.26683, -0.28444, 0.55053, -0.2677, -0.27489, 0.55665, -0.27029, -0.28238, 0.53903, -0.27499, -0.28128, 0.57565, -0.27911, -0.28868, 0.53942, -0.26501, -0.2834], "settings": {"method": "pme", "cutoff": None, "ewald_accuracy": 1e-06, "d3": True, "alpha": 0.3010302527893431, "real_space_cutoff": 12.347337699147769, "mesh": [32, 32, 32]}},
    },
}


def _kernel_case(name):
    """The fixed point charges of ``KERNEL_REFERENCE[name]`` (the recipes of the reference script)."""
    if name == "nacl":
        pos, charges, _ = _rock_salt()
        return pos, charges
    if name.startswith("random"):
        rng = np.random.default_rng(3)
        pos = rng.uniform(0.0, 9.0, (20, 3))
        charges = rng.normal(size=20)
        if name == "random-charged":
            charges[0] += 1.0
        else:
            charges -= charges.mean()
        return torch.tensor(pos), torch.tensor(charges)
    pos = np.array(PERIODIC_REFERENCE["pos"])
    numbers = np.array(PERIODIC_REFERENCE["numbers"])
    return torch.tensor(pos), torch.tensor(np.where(numbers == 6, 0.7, -0.35))


@pytest.mark.parametrize("name", sorted(KERNEL_REFERENCE))
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_kernels_match_the_reference_package_kernels(name, precision):
    """xnn's Ewald and PME kernels against the reference's on the same charges, parameters and
    lattice vectors: the Ewald sums to about 1e-6 relative (the reference's own float64 result
    sits that far from the exact Madelung energy of rock salt, which xnn reproduces to 1e-7),
    PME within the mesh interpolation error of two different spline deconvolutions."""
    from xnn.common.models.electrostatics import coulomb_ewald, coulomb_pme, ewald_parameters
    pos, charges = _kernel_case(name)
    cell = torch.tensor(KERNEL_REFERENCE[name]["cell"])
    for key, ref in KERNEL_REFERENCE[name].items():
        if key == "cell" or key[0] != precision:
            continue
        dtype_name, cutoff, accuracy = key
        alpha, k_cutoff = ewald_parameters(cutoff, accuracy)
        edge_index, r = _periodic_pairs(pos, cell, cutoff)
        e_ewald = float(coulomb_ewald(charges, pos, cell, edge_index, r, alpha, cutoff, k_cutoff).sum())
        e_pme = float(coulomb_pme(charges, pos, cell, edge_index, r, alpha, cutoff, accuracy, 4).sum())
        scale = abs(ref["ewald"])
        assert abs(e_ewald - ref["ewald"]) < 5e-6 * scale
        assert abs(e_pme - ref["pme4"]) < 3e-5 * scale
        assert abs(e_ewald - e_pme) < 3e-5 * scale


@pytest.mark.parametrize("tag", ["float32/dsf15", "float32/ewald", "float32/pme", "float32/ewald/d3",
                                 "float32/pme/d3"])
def test_published_model_reproduces_periodic_reference_values(tag):
    """The served general model on a 24-atom CO2 cell against the reference calculator's DSF,
    Ewald and PME paths (float32 network; the lattice sums split at different real-space
    cutoffs, so they agree to the target accuracy of the Coulomb energy on top of the float32
    round-off of the network)."""
    method = tag.split("/")[1].rstrip("0123456789")
    cutoff = 15.0
    model = _cached_pretrained("aimnet2", model_options={"coulomb": method, "lr_cutoff": cutoff},
                               dispersion=None if tag.endswith("/d3") else False)
    ref = PERIODIC_REFERENCE["runs"][tag]
    g = structure_to_graph({"pos": np.array(PERIODIC_REFERENCE["pos"]),
                            "atomic_numbers": PERIODIC_REFERENCE["numbers"],
                            "cell": np.array(PERIODIC_REFERENCE["cell"]), "pbc": [True] * 3}, model.cutoff)
    g.compute_dtype = torch.float32
    out = model(g)
    assert abs(float(out["energy"]) - ref["energy"]) < 2e-3
    assert np.abs(out["forces"].detach().numpy() - np.array(ref["forces"])).max() < 2e-3
    assert np.abs(out["charges"].detach().numpy() - np.array(ref["charges"])).max() < 1e-4
