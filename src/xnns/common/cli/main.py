"""Command-line entrypoint: train / benchmark / export.

    xnns train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0
    xnns benchmark --config configs/benchmark.yaml
    xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps

Hydra users can instead write a tiny @hydra.main wrapper that calls
`xnns.common.config.from_hydra(cfg)` and hands the Config to the same routines.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _apply_dict_overrides(d: dict, overrides: list[str]) -> dict:
    """Apply ``a.b=c`` dotted overrides to a plain (pre-schema) config dict.

    Used by the ``benchmark`` command, whose config is a free-form nested dict
    (with lists of model entries) rather than a fixed dataclass tree, so the
    dataclass-oriented :func:`~xnns.common.config.apply_overrides` does not
    apply. Intermediate dicts are created as needed; each value is parsed with
    :func:`ast.literal_eval`, falling back to the raw string. Entries without
    ``=`` are skipped.

    Parameters
    ----------
    d : dict
        The config dict to mutate in place.
    overrides : list of str
        Override strings of the form ``"output.dir=runs/bench"`` or
        ``"phases=['benchmark']"``.

    Returns
    -------
    dict
        The same ``d`` instance, mutated.
    """
    import ast
    for ov in overrides:
        if "=" not in ov:
            continue
        path, raw = ov.split("=", 1)
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        obj = d
        parts = path.split(".")
        for p in parts[:-1]:
            obj = obj.setdefault(p, {})
        obj[parts[-1]] = val
    return d


def main(argv=None):
    """Command-line entry point dispatching the ``train`` and ``export`` commands.

    The first argument selects the command; the rest are that command's options.
    ``train`` builds a :class:`Config` from the arguments, constructs the
    training/validation datasets, and runs the trainer. ``benchmark`` loads a
    :class:`~xnns.common.benchmark.BenchmarkConfig` and runs several models
    through the train/evaluate/benchmark phases, writing a comparison table.
    ``export`` loads a checkpoint into a :class:`ForceStressOutput`-wrapped
    model and writes it out for LAMMPS or as TorchScript. With no arguments a
    usage line is printed; an unknown command prints an error message.

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
        print("usage: xnns {train,benchmark,export} [options]"); return
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

    elif cmd == "benchmark":
        p = argparse.ArgumentParser(prog="xnns benchmark")
        p.add_argument("--config", required=True,
                       help="YAML benchmark config file")
        p.add_argument("--set", dest="overrides", action="extend", nargs="+",
                       default=[], metavar="KEY=VALUE",
                       help="dotted override(s) applied to the config dict; "
                            "repeatable, several per flag")
        args, _ = p.parse_known_args(rest)
        import yaml
        from ..benchmark import from_dict, run_benchmark
        with open(args.config) as f:
            raw = yaml.safe_load(f) or {}
        _apply_dict_overrides(raw, args.overrides)
        run_benchmark(from_dict(raw))

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
