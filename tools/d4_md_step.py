#!/usr/bin/env python
"""The full MD step through the MDI engine, with and without the D4 MD options.

Runs ``MDIEngine.calculate()`` (neighbor list, model, forces, stress: what an MD
driver waits for every step) on consecutive frames, for a checkpoint served as is
and with a D4 correction added at run time (route B), in several variants:

  1  the checkpoint alone
  2  + D4, cutoff_eeq as given by ``--cutoff-eeq-base`` (12 A, the pre-2026-09 default)
  3  + cutoff_eeq 16 A (D4's default real-space range of the Ewald split)
  4  + EEQ reuse between steps (``--eeq-reuse``)
  5  + the three-body triplet cache (default budget)

and compares the energies and forces of 3-5 with 2 (same dtype). Frames come from an
``.npz`` with ``frames (F, N, 3)`` in Angstrom, ``z (N,)`` and ``cell (3, 3)``, or,
without one, from a jittered water lattice displaced by small random steps.

Example::

    python tools/d4_md_step.py --ckpt runs/water/best.pt --frames traj.npz --n-frames 20
    python tools/d4_md_step.py --ckpt runs/water/best.pt --n-side 12 --rt 8 --dtype float32
"""
import argparse
import time

import numpy as np
import torch

from xnn.common.deploy.mdi_engine import BOHR_TO_ANGSTROM, HARTREE_TO_EV, MDIEngine

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt", required=True, help="trainer checkpoint (best.pt) of the short-range model")
p.add_argument("--frames", default=None, help=".npz with frames, z, cell (default: a generated water box)")
p.add_argument("--n-side", type=int, default=12, help="waters per box edge of the generated box")
p.add_argument("--n-frames", type=int, default=20, help="frames after the warm-up frame")
p.add_argument("--step", type=float, default=0.01, help="rms displacement per generated frame, A")
p.add_argument("--rp", type=float, default=12.0, help="D4 pair cutoff, A")
p.add_argument("--rt", type=float, default=8.0, help="three-body cutoff, A")
p.add_argument("--cn-cutoff", type=float, default=12.0, help="CN and EEQ-CN cutoffs, A")
p.add_argument("--cutoff-eeq-base", type=float, default=12.0, help="cutoff_eeq of variant 2, A")
p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--seed", type=int, default=7)
args = p.parse_args()

dtype = getattr(torch, args.dtype)
cuda = args.device.startswith("cuda")


def sync():
    if cuda:
        torch.cuda.synchronize()


if args.frames:
    f = np.load(args.frames)
    frames, z, cell = f["frames"][: args.n_frames + 1], f["z"], f["cell"]
else:
    WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                      [0.0, -0.763239, -0.477047]])
    rng = np.random.default_rng(args.seed)
    a = 3.104
    pos = []
    for i in range(args.n_side):
        for j in range(args.n_side):
            for k in range(args.n_side):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q *= np.sign(np.linalg.det(q))
                pos.extend((WATER - WATER[0]) @ q.T + np.array([i, j, k]) * a
                           + rng.normal(scale=0.1, size=3))
    frames = [np.array(pos)]
    for _ in range(args.n_frames):
        frames.append(frames[-1] + rng.normal(scale=args.step, size=frames[-1].shape))
    frames = np.array(frames)
    z = np.array([8, 1, 1] * args.n_side ** 3)
    cell = np.eye(3) * a * args.n_side
n = len(z)


def d4_spec(cutoff_eeq, triplet_cache=0.0):
    return {"name": "d4", "cutoff_pair": args.rp, "switch_width_pair": 2.0,
            "cutoff_triple": args.rt, "switch_width_triple": 1.0, "s9": 1.0,
            "cutoff_cn": args.cn_cutoff, "cutoff_eeq_cn": args.cn_cutoff,
            "cutoff_eeq": cutoff_eeq, "regime": "large", "triplet_cache": triplet_cache}


variants = [("1 checkpoint alone", None, False, 0.0),
            (f"2 + D4 (cutoff_eeq {args.cutoff_eeq_base:g})", args.cutoff_eeq_base, False, 0.0),
            ("3 + cutoff_eeq 16", 16.0, False, 0.0),
            ("4 + EEQ reuse", 16.0, True, 0.0),
            ("5 + triplet cache", 16.0, True, None)]

print(f"{n} atoms, {args.dtype}, r_t = {args.rt} A, {args.n_frames} consecutive frames "
      f"after a warm-up frame\n", flush=True)
results = {}
for label, ce, reuse, cache in variants:
    engine = MDIEngine.from_checkpoint(args.ckpt, device=args.device, dtype=dtype,
                                       dispersion=d4_spec(ce, cache) if ce else None,
                                       eeq_reuse=reuse)
    engine.natoms, engine.atomic_numbers = n, z.astype(np.int64)
    engine.cell_bohr = cell / BOHR_TO_ANGSTROM
    times, energies, forces = [], [], []
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for k in range(args.n_frames + 1):
        engine.coords_bohr = frames[k] / BOHR_TO_ANGSTROM
        engine._needs_calculation = True
        sync()
        t0 = time.perf_counter()
        engine.calculate()
        sync()
        if k > 0:
            times.append(time.perf_counter() - t0)
            energies.append(engine.energy * HARTREE_TO_EV)
            forces.append(engine.forces * HARTREE_TO_EV / BOHR_TO_ANGSTROM)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else float("nan")
    results[label] = (np.array(energies), np.array(forces))
    extra = ""
    if reuse:
        from xnn.common.models.d4 import DFTD4
        for term in (m for m in engine.model.modules() if isinstance(m, DFTD4)):
            st = term.__dict__.get("_eeq_reuse")
            if st is not None:
                extra += (f"  EEQ: {st.stats['iterations'] / max(st.stats['solves'], 1):.1f} it/solve, "
                          f"preconditioner {st.stats['preconditioners']}x, {st.stats['fallbacks']} fallbacks")
    cmp_ = ""
    if ce and not label.startswith("2"):
        e_ref, f_ref = results[variants[1][0]]
        e_, f_ = results[label]
        cmp_ = (f"  vs 2: max|dE| {np.abs(e_ - e_ref).max():.1e} eV, "
                f"max|dF| {np.abs(f_ - f_ref).max():.1e} eV/A")
    print(f"{label:28s} {np.mean(times) * 1000:7.1f} ms/step (min {np.min(times) * 1000:.1f})  "
          f"peak {peak:.1f} GB{cmp_}{extra}", flush=True)
    del engine
    if cuda:
        torch.cuda.empty_cache()
