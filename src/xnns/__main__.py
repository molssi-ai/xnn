"""Run the CLI as ``python -m xnns``.

This is what lets distributed launchers drive the same entry point, e.g.
``torchrun --nproc-per-node 2 -m xnns train --config train.yaml`` or an
equivalently configured ``accelerate launch -m xnns train ...``.
"""
from .common.cli import main

if __name__ == "__main__":
    main()
