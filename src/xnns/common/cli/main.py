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
        keys = dict(energy_key=cfg.data.energy_key, forces_key=cfg.data.forces_key,
                    stress_key=cfg.data.stress_key)
        train = AtomicDataset.from_file(cfg.data.train_path, cfg.data.cutoff, **keys)
        val = (AtomicDataset.from_file(cfg.data.val_path, cfg.data.cutoff, **keys)
               if cfg.data.val_path else None)
        test = (AtomicDataset.from_file(cfg.data.test_path, cfg.data.cutoff, **keys)
                if cfg.data.test_path else None)
        Trainer(cfg, train, val, test).fit()

    elif cmd == "export":
        p = argparse.ArgumentParser()
        p.add_argument("--config", required=True)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--to", choices=["lammps", "torchscript"], default="lammps")
        p.add_argument("--out", default="model_deployed.pt")
        args, _ = p.parse_known_args(rest)
        cfg = cfgmod.from_yaml(args.config)
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
