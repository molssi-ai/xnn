"""Command-line entrypoint: train / benchmark / export / mdi.

    xnn train --config configs/train.yaml --set optim.epochs=50 model.cutoff=6.0
    xnn benchmark --config configs/benchmark.yaml
    xnn export --config configs/train.yaml --ckpt runs/exp/best.pt --to lammps
    xnn mdi --ckpt runs/exp/best.pt -mdi "-role ENGINE -name xnn -method TCP ..."

Hydra users can instead write a tiny @hydra.main wrapper that calls
`xnn.common.config.from_hydra(cfg)` and hands the Config to the same routines.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _apply_dict_overrides(d: dict, overrides: list[str]) -> dict:
    """Apply ``a.b=c`` dotted overrides to a plain (pre-schema) config dict.

    Used by the ``benchmark`` command, whose config is a free-form nested dict
    (with lists of model entries) rather than a fixed dataclass tree, so the
    dataclass-oriented :func:`~xnn.common.config.apply_overrides` does not
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
    """Command-line entry point dispatching the ``train``, ``benchmark``, ``export``, ``mdi`` and ``models`` commands.

    The first argument selects the command; the rest are that command's options.
    ``train`` builds a :class:`Config` from the arguments, constructs the
    training/validation datasets, and runs the trainer. ``benchmark`` loads a
    :class:`~xnn.common.benchmark.BenchmarkConfig` and scores the listed
    pre-trained models on the dataset, writing a comparison table. ``export``
    loads a checkpoint into a :class:`ForceStressOutput`-wrapped model and
    writes it out for LAMMPS or as TorchScript. ``mdi`` serves a checkpoint as
    an MDI engine (see :mod:`xnn.common.deploy.mdi_engine`). ``models``
    lists, inspects, downloads and packages pre-trained models (see
    :mod:`xnn.common.models.hub.cli`). With no
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
        print("usage: xnn {train,benchmark,export,mdi,models} [options]"); return
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
        # the replay set of a multi-head fine-tuning run (model.heads)
        replay = (AtomicDataset.from_file(cfg.data.replay_path, cfg.data.cutoff, **keys)
                  if cfg.data.replay_path else None)
        Trainer(cfg, train, val, test, replay_set=replay).fit()

    elif cmd == "benchmark":
        p = argparse.ArgumentParser(prog="xnn benchmark")
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
        p = argparse.ArgumentParser(prog="xnn export")
        p.add_argument("--config", default=None,
                       help="YAML config describing the architecture; "
                            "optional, and only needed for checkpoints that "
                            "do not embed their own Config (xnn-trained ones "
                            "do)")
        p.add_argument("--ckpt", required=True,
                       help="checkpoint to export: a trainer best.pt, a portable "
                            "model directory, or a pre-trained model name, URL "
                            "or Zenodo DOI (see `xnn models list`)")
        p.add_argument("--cache-dir", default=None,
                       help="model hub cache for a name, URL or DOI")
        p.add_argument("--to", choices=["lammps", "torchscript"],
                       default="lammps",
                       help="both write the same self-contained artifact, "
                            "which exposes the whole-system entry point "
                            "'forward' and the pair-style 'forward_lammps'")
        p.add_argument("--out", default="model_deployed.pt")
        p.add_argument("--no-dispersion", action="store_true",
                       help="export the checkpoint as is, ignoring a recorded "
                            "subtracted_dispersion")
        p.add_argument("--head", default=None,
                       help="head of a multi-head (replay fine-tuned) checkpoint "
                            "to export; LoRA updates are always folded in")
        p.add_argument("--total-charge", type=float, default=0.0,
                       help="net charge of the deployed system (D4 EEQ charges, "
                            "charge-predicting models such as AIMNet2)")
        p.add_argument("--spin-multiplicity", type=float, default=1.0,
                       help="spin multiplicity of the deployed system (the "
                            "two-channel AIMNet2 models)")
        args, _ = p.parse_known_args(rest)
        from ..models.hub import build_potential, fetch_model, load_checkpoint
        from ..models.registry import recorded_dispersion
        from ..deploy import export_torchscript_potential
        ck = load_checkpoint(fetch_model(args.ckpt, cache_dir=args.cache_dir))
        # prefer the architecture embedded in the checkpoint, as `benchmark`
        # does, so exporting a trained run needs nothing but the .pt
        cfg = cfgmod.from_yaml(args.config) if args.config else ck.config
        if cfg is None:
            raise SystemExit(
                f"{args.ckpt} embeds no config; pass --config with the "
                f"architecture it was trained with")
        # a route-B checkpoint records what its labels had subtracted; the
        # export adds it back unless told not to
        recorded = recorded_dispersion(cfg)
        base = build_potential(cfg, ck.state_dict, label=args.ckpt, head=args.head,
                               dispersion=False if args.no_dispersion else None).model
        if recorded is not None and not args.no_dispersion:
            print("adding back the dispersion recorded as subtracted from the "
                  f"training labels: {recorded}")
        elif recorded is not None:
            print(f"exporting WITHOUT the recorded subtracted dispersion ({recorded})")
        meta = {"model": cfg.model.name,
                "species": (cfg.model.extra or {}).get("species"),
                "source_checkpoint": args.ckpt}
        # the model's own cutoff (a dispersion wrapper widens it beyond the
        # config's core-model radius)
        cutoff = float(getattr(base, "cutoff", cfg.model.cutoff))
        print("wrote", export_torchscript_potential(
            base, cutoff, args.out, meta, total_charge=args.total_charge,
            spin_multiplicity=args.spin_multiplicity))

    elif cmd == "mdi":
        from ..deploy.mdi_engine import main as mdi_main
        mdi_main(rest)

    elif cmd == "models":
        from ..models.hub.cli import main as models_main
        models_main(rest)

    else:
        print(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
