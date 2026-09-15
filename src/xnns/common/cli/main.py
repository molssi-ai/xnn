"""Command-line entrypoint: train / benchmark / export / mdi.

    xnns train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0
    xnns benchmark --config configs/benchmark.yaml
    xnns export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps
    xnns mdi --ckpt runs/exp/best.pt -mdi "-role ENGINE -name xnns -method TCP ..."

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
        ``"metrics={'energy': ['mae']}"``.

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
    """Command-line entry point dispatching the ``train``, ``benchmark``, ``export`` and ``mdi`` commands.

    The first argument selects the command; the rest are that command's options.
    ``train`` builds a :class:`Config` from the arguments, constructs the
    training/validation datasets, and runs the trainer. ``benchmark`` loads a
    :class:`~xnns.common.benchmark.BenchmarkConfig` and scores the listed
    pre-trained models on the dataset, writing a comparison table. ``export``
    loads a checkpoint into a :class:`ForceStressOutput`-wrapped model and
    writes it out for LAMMPS or as TorchScript. ``mdi`` serves a checkpoint as
    an MDI engine (see :mod:`xnns.common.deploy.mdi_engine`). With no
    arguments a usage line
    is printed; an unknown command prints an error message.

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
        print("usage: xnns {train,benchmark,export,mdi} [options]"); return
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
        p = argparse.ArgumentParser(prog="xnns export")
        p.add_argument("--config", default=None,
                       help="YAML config describing the architecture; "
                            "optional, and only needed for checkpoints that "
                            "do not embed their own Config (xnns-trained ones "
                            "do)")
        p.add_argument("--ckpt", required=True, help="checkpoint to export")
        p.add_argument("--to", choices=["lammps", "torchscript"],
                       default="lammps",
                       help="both write the same self-contained artifact, "
                            "which exposes the whole-system entry point "
                            "'forward' and the pair-style 'forward_lammps'")
        p.add_argument("--out", default="model_deployed.pt")
        args, _ = p.parse_known_args(rest)
        from ..models import build_model, ForceStressOutput
        from ..deploy import export_torchscript_potential
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        # prefer the architecture embedded in the checkpoint, as `benchmark`
        # does, so exporting a trained run needs nothing but the .pt
        cfg = cfgmod.from_yaml(args.config) if args.config else state.get("cfg")
        if cfg is None:
            raise SystemExit(
                f"{args.ckpt} embeds no config; pass --config with the "
                f"architecture it was trained with")
        base = build_model(cfg.model)
        ForceStressOutput(base).load_state_dict(state["model"])
        meta = {"model": cfg.model.name,
                "species": (cfg.model.extra or {}).get("species"),
                "source_checkpoint": args.ckpt}
        print("wrote", export_torchscript_potential(
            base, cfg.model.cutoff, args.out, meta))

    elif cmd == "mdi":
        from ..deploy.mdi_engine import main as mdi_main
        mdi_main(rest)

    else:
        print(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
