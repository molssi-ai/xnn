# tools/

Maintenance and benchmarking scripts that are not part of the package.

| script | purpose |
|---|---|
| `build_d3_reference.py` | regenerate `xnn/common/models/d3_reference.npz` from the simple-dftd3 sources (both reference sets) |
| `build_d4_reference.py` | regenerate `xnn/common/models/d4_reference.npz` from the dftd4 / multicharge / mctc-lib sources |
| `d4_bench.py` | where the time goes in D4 (energy + forces + stress) on a periodic water box: neighbor list, forward / backward split, a table of the D4 stages with synchronized timers, optional CUDA kernel table; `--frames` for MD-like sequences with the EEQ reuse and the triplet cache |
| `d4_md_step.py` | the full MD step through the MDI engine (a checkpoint plus D4 added at run time) on consecutive frames, comparing the D4 MD options: `cutoff_eeq` 16 A, the EEQ reuse, the triplet cache |

Both D4 scripts change nothing in xnn: every tunable is a command-line option of
the real `DFTD4` term or of the engine. Examples:

```bash
python tools/d4_bench.py --n-side 12 --rt 8 --dtype float32 --kernels
python tools/d4_bench.py --n-side 12 --rt 8 --frames 10 --eeq-reuse --json bench.json
python tools/d4_md_step.py --ckpt runs/water/best.pt --n-side 12 --rt 8 --n-frames 20
python tools/d4_md_step.py --ckpt runs/water/best.pt --frames traj.npz --dtype float64
```

`d4_md_step.py` accepts a trajectory as an `.npz` with `frames (F, N, 3)` in
Angstrom, `z (N,)` and `cell (3, 3)`; without one it displaces a generated water
box by small random steps, which exercises the same code paths but is not an
equilibrated liquid.
