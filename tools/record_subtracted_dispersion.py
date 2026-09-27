#!/usr/bin/env python
"""Record in an existing checkpoint what was subtracted from its training labels.

A route-B checkpoint (trained on labels with the dispersion removed) should carry
``subtracted_dispersion`` in its config, so that ``xnn mdi`` and ``xnn export``
add the term back with the right parameters. New trainings set it in the training
config; this script adds it to a checkpoint trained before the field existed.

Example::

    python tools/record_subtracted_dispersion.py runs/water/best.pt \\
        --spec "{name: d4, s6: 1.0, s8: 1.20065498, a1: 0.40085597, a2: 5.02928789, s9: 1.0,
                 cutoff_pair: 12.0, switch_width_pair: 2.0, cutoff_triple: 10.0,
                 switch_width_triple: 1.0, cutoff_eeq: 16.0, regime: auto,
                 dataset: water_clusters_minusD4}"

The checkpoint is rewritten in place (a ``.bak`` copy is kept) unless ``--out``
names another file. The record is validated like a training config: the term must
be d3 or d4, the keys must be options of that term (plus ``dataset`` / ``note``),
and a model that already includes dispersion is refused.
"""
import argparse
import shutil

import torch
import yaml

from xnn.common.config import normalize_subtracted_dispersion

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("ckpt", help="trainer checkpoint (best.pt)")
p.add_argument("--spec", default="d4",
               help="YAML mapping (or a bare name) of what was subtracted, with the settings "
                    "to add it back (default: 'd4', the PBE0-D4 defaults, uncut)")
p.add_argument("--out", default=None, help="write here instead of in place")
p.add_argument("--force", action="store_true", help="replace an existing record")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
existing = getattr(cfg, "subtracted_dispersion", None)
if existing is not None and not args.force:
    raise SystemExit(f"{args.ckpt} already records {existing}; use --force to replace it")
cfg.subtracted_dispersion = normalize_subtracted_dispersion(yaml.safe_load(args.spec), cfg.model)
out = args.out or args.ckpt
if out == args.ckpt:
    shutil.copy2(args.ckpt, args.ckpt + ".bak")
torch.save(ckpt, out)
print(f"{out}: subtracted_dispersion = {cfg.subtracted_dispersion}")
