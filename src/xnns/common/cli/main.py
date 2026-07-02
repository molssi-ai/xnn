"""Command-line entrypoint: train / evaluate / export.

    xnns train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0
    xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps

Hydra users can instead write a tiny @hydra.main wrapper that calls
`xnns.common.config.from_hydra(cfg)` and hands the Config to the same routines.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _load_structures(path: str):
    """Load structures into the list-of-dicts format the dataset expects.

    Uses ASE (handles ``.xyz`` / ``.extxyz`` / ``.cif`` / ... and periodic
    cells). Energy and forces are included when the frame carries a calculator
    or an ``"energy"`` entry in its ``info`` and they can be read successfully.

    Parameters
    ----------
    path : str
        Path to a structure file readable by :func:`ase.io.read`. All frames in
        the file (``index=":"``) are loaded.

    Returns
    -------
    list of dict
        One dict per frame with keys ``"pos"``, ``"atomic_numbers"``,
        ``"cell"`` (``None`` for non-periodic frames), ``"pbc"``, and optionally
        ``"energy"`` and ``"forces"`` when available.
    """
    from ase.io import read
    frames = read(path, index=":")
    out = []
    for a in frames:
        d = {
            "pos": a.get_positions(),
            "atomic_numbers": a.get_atomic_numbers(),
            "cell": a.get_cell()[:] if a.pbc.any() else None,
            "pbc": a.pbc,
        }
        if a.calc is not None or "energy" in a.info:
            try:
                d["energy"] = a.get_potential_energy()
                d["forces"] = a.get_forces()
            except Exception:
                pass
        out.append(d)
    return out


def main(argv=None):
    """Command-line entry point dispatching the ``train`` and ``export`` commands.

    The first argument selects the command; the rest are that command's options.
    ``train`` builds a :class:`Config` from the arguments, constructs the
    training/validation datasets, and runs the trainer. ``export`` loads a
    checkpoint into a :class:`ForceStressOutput`-wrapped model and writes it out
    for LAMMPS or as TorchScript. With no arguments a usage line is printed;
    an unknown command prints an error message.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector excluding the program name. When ``None`` (the default),
        ``sys.argv[1:]`` is used.

    Returns
    -------
    None
        This function prints results and has no return value.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: xnns {train,export} [options]"); return
    cmd, rest = argv[0], argv[1:]

    from .. import config as cfgmod

    if cmd == "train":
        cfg = cfgmod.from_argparse(rest)
        from ..data import AtomicDataset
        from ..train import Trainer
        train = AtomicDataset(_load_structures(cfg.data.train_path), cfg.data.cutoff)
        val = (AtomicDataset(_load_structures(cfg.data.val_path), cfg.data.cutoff)
               if cfg.data.val_path else None)
        Trainer(cfg, train, val).fit()

    elif cmd == "export":
        p = argparse.ArgumentParser()
        p.add_argument("--config", required=True)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--to", choices=["lammps", "torchscript"], default="lammps")
        p.add_argument("--out", default="model_deployed.pt")
        args, _ = p.parse_known_args(rest)
        cfg = (cfgmod.from_toml(args.config) if args.config.endswith(".toml")
               else cfgmod.from_yaml(args.config))
        from ..models import build_model, ForceStressOutput
        from ..deploy import export_to_lammps, export_torchscript
        base = build_model(cfg.model)
        wrapped = ForceStressOutput(base)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        wrapped.load_state_dict(state["model"])
        if args.to == "lammps":
            print("wrote", export_to_lammps(base, cfg.model.cutoff, args.out))
        else:
            print("wrote", export_torchscript(base, args.out))
    else:
        print(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
