"""Load RuNNer models (``input.nn``, weights and scaling files) into :class:`~xnn.dnn.models.hdnnp.HDNNP`.

RuNNer (Behler group) stores a high-dimensional neural network potential as
a directory with:

* ``input.nn``: the settings (elements, generation, cutoff functions,
  symmetry functions, network architectures, scaling flags, free-atom
  energies, Gaussian widths, screening);
* ``scaling.data`` (or ``scaling_<suffix>.data`` per network type): ``element
  feature min max avg`` per line, the elements numbered in ascending atomic
  number;
* ``weights_<suffix>.<ZZZ>.data``: one value per line, layer by layer, each
  layer's weight matrix (input index outer, output index inner) followed by
  its biases; ``weights_hardness.data`` for element hardnesses, one per
  element in ascending atomic number;
* ``qeq_scaling.data`` (4G): the statistics of the charge input of the
  short-range networks.

RuNNer orders the symmetry functions of every element not by their order in
``input.nn`` but by type, cutoff radius, cutoff kind and parameters (rounded
to five decimals), then the neighbour elements; :func:`runner_feature_key`
reproduces that order so the weights and scaling rows line up. Everything is
converted to the xnn units (Angstrom, eV); the networks keep their atomic-unit
outputs, scaled by ``energy_scale``, ``chi_scale`` and ``hardness_scale``.

Example::

    from xnn.dnn.common.runner import load_runner_model

    model = load_runner_model("path/to/runner_model")   # an HDNNP, eV and Angstrom
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from xnn.common.data.elements import atomic_number
from xnn.common.data.hub.units import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from xnn.dnn.featurizers.acsf import AtomCenteredSymmetryFunctions

#: RuNNer activation codes -> :func:`~xnn.common.models.ops.make_activation` names
ACTIVATIONS = {"t": "tanh", "tanh": "tanh", "st": "scaled_tanh", "stanh": "scaled_tanh",
               "r": "relu", "relu": "relu", "l": "linear", "linear": "linear",
               "p": "softplus", "softplus": "softplus", "s": "sigmoid", "sigmoid": "sigmoid",
               "sq": "square", "square": "square"}
#: RuNNer cutoff keywords -> :data:`~xnn.dnn.featurizers.acsf.CUTOFF_KINDS`
CUTOFFS = {"fc_cosine": "cosine", "fc_hypertangent": "tanh",
           "fc_hypertangent_approx": "tanh_approx", "fc_polynomial": "polynomial",
           "fc_hard": "hard"}
_CUTOFF_NAMES = {"cosine": "Cosine", "hard": "Cutoff", "tanh": "Hypert",
                 "tanh_approx": "Hypert", "polynomial": "Polyno"}
_SUFFIXES = ("short", "charge", "chi", "hardness", "hirshv")


def runner_feature_key(row: dict, cutoff: dict) -> tuple:
    """The sort key RuNNer orders the features of an element by.

    Parameters
    ----------
    row : dict
        A symmetry-function row in RuNNer units (``type``, ``neighbors``,
        ``eta``, ``rs``, ``lambda``, ``zeta``, ``theta_s``).
    cutoff : dict
        Its cutoff function (``kind``, ``r_cut``).

    Returns
    -------
    tuple
        Type, cutoff radius, cutoff kind, the parameters rounded to five
        decimals, then the neighbour atomic numbers (fluorine last, as in
        RuNNer's string comparison).
    """
    def z(v: int) -> float:
        return float("inf") if v == 9 else float(v)

    t = row["type"]
    head = (t, round(cutoff["r_cut"], 5), _CUTOFF_NAMES[cutoff["kind"]])
    nb = tuple(z(v) for v in sorted(row["neighbors"]))
    if t == 1:
        return head + nb
    if t == 2:
        return head + (round(row["eta"], 5), round(row["rs"], 5)) + nb
    if t in (3, 9):
        return head + (round(row["eta"], 5), round(row["zeta"], 5), round(row["lambda"], 5)) + nb
    return head + (round(row["eta"], 5), round(row["theta_s"], 5)) + nb


def parse_input_nn(path: Union[str, Path]) -> dict:
    """The settings of a RuNNer ``input.nn`` that define the model (RuNNer units).

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    dict
        ``elements`` (atomic numbers), ``generation``, ``cutoffs`` (index ->
        spec, bohr), ``functions`` (suffix -> list of rows; ``"default"`` for
        the ``symfunction`` lines), ``feature_map_default``, ``nodes`` and
        ``activations`` (suffix -> {``"default"`` or Z: list}), ``scaling``
        (suffix -> mode), ``scale_range``, ``atom_energies``,
        ``gaussian_widths``, ``screening`` (cutoff index or ``None``),
        ``hardness_model``, ``hardness_activation``, ``charge_neuron``,
        ``charge_input`` (``(scale, center)`` flags) and ``committee``.
    """
    s = {"elements": [], "generation": 2, "cutoffs": {}, "functions": {}, "feature_map_default": False,
         "nodes": {}, "activations": {}, "flags": set(), "scale_range": {}, "atom_energies": {},
         "gaussian_widths": {}, "screening": None, "hardness_model": "hdnn",
         "hardness_activation": "l", "committee": 1}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#")[0].split("!")[0].strip()
        if not line:
            continue
        tok = line.split()
        key = tok[0].lower()
        args = tok[1:]
        if key == "elements":
            s["elements"] = sorted(atomic_number(e) for e in args)
        elif key == "nnp_generation":
            s["generation"] = int(args[0])
        elif key in CUTOFFS:
            kind = CUTOFFS[key]
            idx = int(args[0])
            vals = [float(v) for v in args[1:]]
            if kind == "polynomial":
                spec = {"kind": kind, "exponent": int(vals[0]), "r_cut": vals[1]}
            elif kind == "hard":
                spec = {"kind": kind, "r_cut": vals[0]}
            else:
                spec = {"kind": kind, "r_inner": vals[0] if len(vals) == 2 else 0.0, "r_cut": vals[-1]}
            s["cutoffs"][idx] = spec
        elif key.startswith("symfunction"):
            suffix = key[len("symfunction"):].lstrip("_") or "default"
            s["functions"].setdefault(suffix, []).append(_parse_symfunction(args))
        elif key == "feature_map_default":
            s["feature_map_default"] = True
        elif "nodes" in key:
            _architecture(s["nodes"], key, "nodes", args, int)
        elif "activation" in key and "elemental" in key:
            s["hardness_activation"] = args[-1].lower()
        elif "activation_nn" in key:
            _architecture(s["activations"], key, "activation_nn", args, str.lower)
        elif key in ("scale_feature_maps", "center_feature_maps") or any(
                key == f"{base}_{sfx}" for base in ("scale_feature_maps", "center_feature_maps")
                for sfx in _SUFFIXES):
            s["flags"].add(key)
        elif key.startswith("scale_feature_maps_range"):
            suffix = key[len("scale_feature_maps_range"):].lstrip("_") or "default"
            s["scale_range"][suffix] = (float(args[0]), float(args[1]))
        elif key == "atom_energy":
            s["atom_energies"][atomic_number(args[0])] = float(args[1])
        elif key == "fixed_gausswidth":
            s["gaussian_widths"][atomic_number(args[0])] = float(args[1])
        elif key == "screening_function":
            s["screening"] = int(args[0])
        elif key == "model_type_hardness":
            s["hardness_model"] = args[0].lower()
        elif key in ("no_charge_neuron", "scale_global_feature_maps_q", "center_global_feature_maps_q"):
            s["flags"].add(key)
        elif key == "num_committee_members":
            s["committee"] = int(args[0])
    return s


def _parse_symfunction(args: list[str]) -> dict:
    """One ``symfunction`` line (after the keyword) as a row in RuNNer units."""
    element = atomic_number(args[0])
    t = int(args[1])
    rest = args[2:]
    if any(a.startswith("modifiers") for a in rest):
        raise NotImplementedError("symmetry-function modifiers (spin, weighted) are not supported")
    if t == 1:
        nfeat = int(rest[1]) if len(rest) == 3 else 1
        return {"element": element, "type": 1, "neighbors": [atomic_number(rest[0])],
                "n_features": nfeat, "cutoff": int(rest[-1])}
    if t == 2:
        return {"element": element, "type": 2, "neighbors": [atomic_number(rest[0])],
                "eta": float(rest[1]), "rs": float(rest[2]), "cutoff": int(rest[3])}
    if t in (3, 9):
        return {"element": element, "type": t, "neighbors": [atomic_number(rest[0]), atomic_number(rest[1])],
                "eta": float(rest[2]), "lambda": float(rest[3]), "zeta": float(rest[4]),
                "cutoff": int(rest[5])}
    if t == 8:
        return {"element": element, "type": 8, "neighbors": [atomic_number(rest[0]), atomic_number(rest[1])],
                "theta_s": float(rest[2]), "eta": float(rest[3]), "cutoff": int(rest[4])}
    raise NotImplementedError(f"symmetry function type {t}")


def _architecture(table: dict, key: str, word: str, args: list[str], cast) -> None:
    """Fill ``table[suffix][element or 'default']`` from a nodes / activation keyword."""
    if key.endswith("_comm") or "_comm_" in key:
        raise NotImplementedError("per-committee-member architectures are not supported")
    per_default = key.startswith("default_")
    body = key[len("default_"):] if per_default else key
    suffix = body[len(word):].lstrip("_") or "default"
    if per_default:
        table.setdefault(suffix, {})["default"] = [cast(a) for a in args]
    else:
        table.setdefault(suffix, {})[atomic_number(args[0])] = [cast(a) for a in args[1:]]


def _resolve(table: dict, suffix: str, z: int):
    """The value of ``suffix`` for element ``z``, falling back to the generic keyword."""
    for sfx in (suffix, "default"):
        block = table.get(sfx, {})
        if z in block:
            return block[z]
        if "default" in block:
            return block["default"]
    raise ValueError(f"input.nn gives no {suffix} architecture for Z={z}")


def _scaling_mode(s: dict, suffix: str) -> tuple[str, tuple]:
    def has(base: str) -> bool:
        return base in s["flags"] or f"{base}_{suffix}" in s["flags"]

    rng = s["scale_range"].get(suffix, s["scale_range"].get("default"))
    if rng is not None:
        return "range", rng
    scale, center = has("scale_feature_maps"), has("center_feature_maps")
    mode = {(True, True): "center_scale", (True, False): "scale",
            (False, True): "center", (False, False): "none"}[(scale, center)]
    return mode, (0.0, 1.0)


def _featurizer(s: dict, suffix: str, directory: Path) -> AtomCenteredSymmetryFunctions:
    """The symmetry functions of one network type in xnn units, in RuNNer's order, with its scaling."""
    rows = s["functions"].get("default" if s["feature_map_default"] else suffix)
    if rows is None:
        rows = s["functions"].get("default")
    if rows is None:
        raise ValueError(f"input.nn declares no symmetry functions for the {suffix} networks")
    used = sorted({r["cutoff"] for r in rows})
    cut_pos = {idx: i for i, idx in enumerate(used)}
    cutoffs = []
    for idx in used:
        c = dict(s["cutoffs"][idx])
        c["r_cut"] *= BOHR_TO_ANGSTROM
        c["r_inner"] = c.get("r_inner", 0.0) * BOHR_TO_ANGSTROM
        cutoffs.append(c)
    ordered = sorted(rows, key=lambda r: (r["element"],
                                          runner_feature_key(_defaults(r), s["cutoffs"][r["cutoff"]])))
    functions = []
    for r in ordered:
        f = dict(_defaults(r), cutoff=cut_pos[r["cutoff"]])
        if f["type"] in (2, 3, 9):
            f["eta"] = f["eta"] / BOHR_TO_ANGSTROM ** 2
        if f["type"] == 2:
            f["rs"] = f["rs"] * BOHR_TO_ANGSTROM
        functions.append(f)
    mode, rng = _scaling_mode(s, suffix)
    feat = AtomCenteredSymmetryFunctions(s["elements"], cutoffs, functions, mode=mode, scale_range=rng)
    if mode != "none":
        name = "scaling.data" if s["feature_map_default"] else f"scaling_{suffix}.data"
        path = directory / name
        if not path.exists() and (directory / "scaling.data").exists():
            path = directory / "scaling.data"
        _read_scaling(feat, path, s["elements"])
    return feat


def _defaults(r: dict) -> dict:
    out = {"eta": 0.0, "rs": 0.0, "lambda": 1.0, "zeta": 1.0, "theta_s": 0.0}
    out.update(r)
    return out


def _read_scaling(feat: AtomCenteredSymmetryFunctions, path: Path, elements: list[int]) -> None:
    """Fill ``stat_min/max/avg`` from a ``scaling.data`` file (elements in ascending Z)."""
    lo, hi, avg = feat.stat_min.clone(), feat.stat_max.clone(), feat.stat_avg.clone()
    sp = {z: i for i, z in enumerate(feat.species)}
    for line in path.read_text().splitlines():
        tok = line.split()
        if len(tok) < 5:
            continue
        z = elements[int(tok[0]) - 1]
        col = int(tok[1]) - 1
        if col >= feat.n_features[z]:
            raise ValueError(f"{path}: feature {col + 1} of Z={z} beyond its {feat.n_features[z]} features")
        lo[sp[z], col], hi[sp[z], col], avg[sp[z], col] = float(tok[2]), float(tok[3]), float(tok[4])
    feat.stat_min.copy_(lo)
    feat.stat_max.copy_(hi)
    feat.stat_avg.copy_(avg)


def _read_values(path: Path) -> list[float]:
    values = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].split("!")[0].strip()
        if line:
            values.append(float(line.split()[0].replace("D", "E").replace("d", "e")))
    return values


def _load_networks(nets, directory: Path, basename: str) -> None:
    """Copy ``<basename>.<ZZZ>.data`` into the per-element networks of ``nets``."""
    for z in nets.species:
        path = directory / f"{basename}.{z:03d}.data"
        values = _read_values(path)
        linears = [m for m in nets.nets[str(z)] if isinstance(m, torch.nn.Linear)]
        need = sum(m.weight.numel() + m.bias.numel() for m in linears)
        if len(values) != need:
            raise ValueError(f"{path}: {len(values)} values, the network has {need} parameters")
        k = 0
        with torch.no_grad():
            for m in linears:
                n_out, n_in = m.weight.shape
                w = torch.tensor(values[k:k + n_out * n_in], dtype=torch.float64)
                m.weight.copy_(w.reshape(n_in, n_out).t())
                k += n_out * n_in
                m.bias.copy_(torch.tensor(values[k:k + n_out], dtype=torch.float64))
                k += n_out


def load_runner_model(directory: Union[str, Path], input_nn: str = "input.nn",
                      dtype: Optional[torch.dtype] = torch.float64):
    """Build an :class:`~xnn.dnn.models.hdnnp.HDNNP` from a RuNNer model directory.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory with ``input.nn``, the weights and the scaling files (one
        committee member).
    input_nn : str, optional
        Name of the settings file, by default ``"input.nn"``.
    dtype : torch.dtype, optional
        Parameter dtype, by default float64.

    Returns
    -------
    HDNNP
        The model in xnn units (Angstrom, eV), in eval mode.
    """
    from xnn.dnn.models.hdnnp import HDNNP

    directory = Path(directory)
    s = parse_input_nn(directory / input_nn)
    gen = s["generation"]
    if gen not in (2, 3, 4):
        raise ValueError(f"nnp_generation {gen} is not a RuNNer HDNNP generation")
    if s["committee"] != 1:
        raise NotImplementedError("load committee members one directory at a time")
    elements = s["elements"]
    short = _featurizer(s, "short", directory)
    charge_suffix = {3: "charge", 4: "chi"}.get(gen)
    charge_feat = None
    if charge_suffix is not None:
        charge_feat = _featurizer(s, charge_suffix, directory)
        if (s["feature_map_default"]
                or s["functions"].get(charge_suffix) is None) and _same_features(charge_feat, short):
            charge_feat = None

    def acts(suffix: str) -> dict:
        return {z: [ACTIVATIONS[a] for a in _resolve(s["activations"], suffix, z)] for z in elements}

    def nodes(suffix: str) -> dict:
        return {z: _resolve(s["nodes"], suffix, z) for z in elements}

    hard_model = s["hardness_model"]
    sigma = {z: s["gaussian_widths"].get(z) for z in elements}
    widths = None
    if any(v is not None for v in sigma.values()):
        from xnn.dnn.models.hdnnp import default_gaussian_widths
        default = dict(zip(elements, default_gaussian_widths(elements)))
        widths = {z: (sigma[z] * BOHR_TO_ANGSTROM if sigma[z] is not None else default[z]) for z in elements}
    screening = None
    if s["screening"] is not None and gen >= 3:
        c = dict(s["cutoffs"][s["screening"]])
        c["r_cut"] *= BOHR_TO_ANGSTROM
        c["r_inner"] = c.get("r_inner", 0.0) * BOHR_TO_ANGSTROM
        screening = c
    model = HDNNP(
        elements, short, generation=gen, hidden=nodes("short"), activation=acts("short"),
        atomic_energies=[s["atom_energies"].get(z, 0.0) * HARTREE_TO_EV for z in elements],
        charge_featurizer=charge_feat,
        charge_hidden=nodes(charge_suffix) if charge_suffix else None,
        charge_activation=acts(charge_suffix) if charge_suffix else None,
        gaussian_widths=widths, screening=screening,
        hardness="element" if hard_model == "elemental" else "network",
        hardness_activation=ACTIVATIONS[s["hardness_activation"]],
        hardness_hidden=nodes("hardness") if gen == 4 and hard_model != "elemental" else None,
        hardness_activation_nn=acts("hardness") if gen == 4 and hard_model != "elemental" else None,
        charge_neuron="no_charge_neuron" not in s["flags"],
        coulomb_constant=HARTREE_TO_EV * BOHR_TO_ANGSTROM,
        energy_scale=HARTREE_TO_EV, chi_scale=HARTREE_TO_EV, hardness_scale=HARTREE_TO_EV)
    if dtype is not None:
        model = model.to(dtype)
    _load_networks(model.element_nets, directory, "weights_short")
    if gen == 3:
        _load_networks(model.charge_nets, directory, "weights_charge")
    if gen == 4:
        _load_networks(model.charge_nets, directory, "weights_chi")
        if model.hardness_nets is not None:
            _load_networks(model.hardness_nets, directory, "weights_hardness")
        else:
            values = _read_values(directory / "weights_hardness.data")
            if len(values) != len(elements):
                raise ValueError(f"weights_hardness.data: {len(values)} values for {len(elements)} elements")
            with torch.no_grad():
                model.hardness_raw.copy_(torch.tensor(values, dtype=model.hardness_raw.dtype))
        if model.charge_neuron:
            _read_charge_input(model, directory, s)
    return model.eval()


def _same_features(a: AtomCenteredSymmetryFunctions, b: AtomCenteredSymmetryFunctions) -> bool:
    return (a.functions == b.functions and a.cutoffs == b.cutoffs and a.mode == b.mode
            and torch.equal(a.stat_min, b.stat_min) and torch.equal(a.stat_max, b.stat_max)
            and torch.equal(a.stat_avg, b.stat_avg))


def _read_charge_input(model, directory: Path, s: dict) -> None:
    """The scaling ``(q - c_Z) s_Z`` of the 4G charge input from ``qeq_scaling.data``."""
    scale = "scale_global_feature_maps_q" in s["flags"]
    center = "center_global_feature_maps_q" in s["flags"]
    if not (scale or center):
        return
    shift, factor = model.charge_input_shift.clone(), model.charge_input_factor.clone()
    for line in (directory / "qeq_scaling.data").read_text().splitlines():
        tok = line.split()
        if len(tok) < 7:
            continue
        z, qmax, qmin, qavg = int(tok[1]), float(tok[2]), float(tok[3]), float(tok[4])
        shift[z] = qavg if center else qmin
        factor[z] = 1.0 / (qmax - qmin) if scale and qmax != qmin else 1.0
    model.charge_input_shift.copy_(shift)
    model.charge_input_factor.copy_(factor)
