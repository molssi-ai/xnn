"""Run the CLI as ``python -m xnn``.

This is what lets distributed launchers drive the same entry point, e.g.
``torchrun --nproc-per-node 2 -m xnn train --config train.yaml``.
"""
from .common.cli import main

if __name__ == "__main__":
    main()
