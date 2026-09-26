#!/usr/bin/env python
"""Where the time goes in xnn's D4 for a periodic water box: energy + forces + stress.

Builds a jittered lattice of randomly oriented waters at about 1 g/cm^3, evaluates
``ForceStressOutput(D4Dispersion(...))`` once for warm-up and once for the measurement,
and reports

* the neighbor-list build (cold and warm; an MD engine rebuilds it every step),
* the total time, split into the forward pass and the backward pass
  (``torch.autograd.grad`` inside ``ForceStressOutput``),
* a table of the D4 stages with inclusive and self time, number of calls, and the
  pass they ran in (the three-body recompute blocks and the EEQ implicit
  differentiation run code again in the backward pass, so their labels appear there
  too),
* optionally the top CUDA kernels of one step (``--kernels``).

Every tunable of the D4 term is a command-line option; nothing in xnn is patched.
With ``--frames N`` the box is displaced by ``N`` small random steps and every frame is
evaluated, which is what the EEQ reuse (``--eeq-reuse``) and the three-body triplet
cache act on; the per-frame time and the agreement with the first frame's fresh
solve are reported.

Timers call ``torch.cuda.synchronize()`` at every scope boundary, so the stage times
are attributable; the total is measured separately without the timers.

Examples::

    python tools/d4_bench.py --n-side 12 --rt 8 --dtype float32 --kernels
    python tools/d4_bench.py --n-side 12 --rt 8 --frames 10 --eeq-reuse --json out.json
"""
import argparse
import functools
import json
import os
import platform
import sys
import time

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--n-side", type=int, default=12, help="waters per box edge (12 -> 1728 H2O, 5184 atoms)")
p.add_argument("--rp", type=float, default=12.0, help="D4 pair cutoff, A")
p.add_argument("--switch", type=float, default=2.0, help="pair switching width, A")
p.add_argument("--rt", type=float, default=8.0, help="three-body cutoff, A")
p.add_argument("--switch-triple", type=float, default=1.0, help="three-body switching width, A")
p.add_argument("--s9", type=float, default=1.0, help="three-body scaling (0 = no three-body term)")
p.add_argument("--cn-cutoff", type=float, default=12.0, help="CN and EEQ-CN cutoffs, A")
p.add_argument("--cutoff-eeq", type=float, default=None,
               help="large-regime EEQ real-space range, A (default: D4's own, 16 A or the largest cutoff)")
p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
p.add_argument("--regime", default="auto", help="EEQ regime: auto, dense or large")
p.add_argument("--eeq-solver", default="auto", help="large-regime EEQ solver: auto, lu or cg")
p.add_argument("--chunk", type=int, default=None, help="triplets per three-body recompute block (default: from free memory)")
p.add_argument("--triplet-cache", type=float, default=None,
               help="triple cache budget in GB (default: a quarter of the free memory; 0 disables it)")
p.add_argument("--eeq-reuse", action="store_true", help="carry the EEQ solve between frames (needs --frames)")
p.add_argument("--frames", type=int, default=0, help="additional displaced frames to evaluate (MD-like)")
p.add_argument("--step", type=float, default=0.01, help="rms displacement per frame, A")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--kernels", action="store_true", help="also print the top CUDA kernels")
p.add_argument("--json", default=None, help="write the results to this JSON file")
p.add_argument("--save-forces", default=None, help="save the forces and stress (.npz) of the measured call")
args = p.parse_args()

import xnn  # noqa: E402
from xnn.common.data import atomic_data, structure_to_graph  # noqa: E402
from xnn.common.models import D4Dispersion, ForceStressOutput  # noqa: E402
import xnn.common.models.d4 as d4mod  # noqa: E402
import xnn.common.models.dispersion as dispmod  # noqa: E402
import xnn.common.models.eeq as eeqmod  # noqa: E402
import xnn.common.models.recompute as recmod  # noqa: E402

DTYPE = getattr(torch, args.dtype)
torch.set_default_dtype(DTYPE)
dev = torch.device(args.device)
cuda = dev.type == "cuda"


def sync():
    if cuda:
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# labeled, synchronized timers around the real functions
# ----------------------------------------------------------------------------
PHASE = ["forward"]
STACK = []
STATS = {}          # (phase, label) -> [inclusive, self, calls]
ENABLED = [False]


def timed(label, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        if not ENABLED[0]:
            return fn(*a, **k)
        sync()
        STACK.append(0.0)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sync()
            dt = time.perf_counter() - t0
            children = STACK.pop()
            if STACK:
                STACK[-1] += dt
            s = STATS.setdefault((PHASE[0], label), [0.0, 0.0, 0])
            s[0] += dt
            s[1] += dt - children
            s[2] += 1
    return wrapper


def wrap_attr(owner, name, label, static=False):
    if not hasattr(owner, name):
        return
    new = timed(label, getattr(owner, name))
    setattr(owner, name, staticmethod(new) if static else new)


wrap_attr(atomic_data.AtomicGraph, "edge_vectors", "edge_vectors")
for name, label in [("coordination_numbers", "d4.coordination_numbers"),
                    ("eeq_charges", "d4.eeq_dense"),
                    ("_eeq_charges_large", "d4.eeq_large"),
                    ("reference_weights", "d4.reference_weights"),
                    ("dynamic_polarizabilities", "d4.dynamic_polarizabilities"),
                    ("two_body_energy", "d4.two_body"),
                    ("_two_body_chunked", "d4.two_body_blocks"),
                    ("_three_body_chunked", "d4.three_body")]:
    wrap_attr(d4mod.DFTD4, name, label)
for name, label in [("__init__", "eeq.setup"), ("assemble", "eeq.assemble"),
                    ("_solve_lu", "eeq.solve_lu"), ("_solve_cg", "eeq.solve_cg"),
                    ("matvec", "eeq.matvec"), ("apply_differentiable", "eeq.apply_differentiable")]:
    wrap_attr(eeqmod.EEQSystem, name, label)
wrap_attr(eeqmod.EEQReuse, "solve", "eeq.reuse_solve")
wrap_attr(eeqmod.EEQReuse, "_precondition", "eeq.reuse_preconditioner")
wrap_attr(eeqmod, "reciprocal_vectors", "eeq.reciprocal_vectors")
wrap_attr(eeqmod._AugmentedSolve, "backward", "eeq.implicit_backward", static=True)
for name, label in [("_triplet_block", "atm.block"),
                    ("build_triplets", "atm.enumerate_triplets"),
                    ("triplet_energy", "atm.triplet_energy"),
                    ("_block_triplets", "atm.enumerate_block"),
                    ("_third_side", "atm.third_side_lookup"),
                    ("_triplet_block_grad", "atm.block_grad (closed form)"),
                    ("_center_sum", "atm.center_sum"),
                    ("edge_cell_shifts", "atm.edge_cell_shifts"),
                    ("pair_edge_keys", "atm.pair_edge_keys")]:
    wrap_attr(dispmod, name, label)
wrap_attr(recmod._RecomputeBlock, "backward", "recompute.block_backward", static=True)
wrap_attr(recmod._RecomputeGrad, "backward", "recompute.grad_backward", static=True)

CHUNKS = []
_auto = dispmod.auto_triplet_chunk


def _auto_logged(*a, **k):
    n = _auto(*a, **k)
    CHUNKS.append(n)
    return n


dispmod.auto_triplet_chunk = _auto_logged

_autograd_grad = torch.autograd.grad
_DEPTH = [0]


def grad_with_phase(*a, **k):
    # nested calls (the recompute blocks differentiate inside the backward
    # pass) keep the phase and get their own label
    label = ("autograd.grad (forces + stress)" if _DEPTH[0] == 0
             else "autograd.grad (nested, inside recompute)")
    _DEPTH[0] += 1
    PHASE[0] = "backward"
    try:
        return timed(label, _autograd_grad)(*a, **k)
    finally:
        _DEPTH[0] -= 1
        if _DEPTH[0] == 0:
            PHASE[0] = "forward"


torch.autograd.grad = grad_with_phase

# ----------------------------------------------------------------------------
# the structure: a jittered lattice of randomly oriented waters, ~1.0 g/cm^3
# ----------------------------------------------------------------------------
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
pos = np.array(pos)
z = [8, 1, 1] * args.n_side ** 3
cell = np.eye(3) * a * args.n_side

d4_options = dict(cutoff_pair=args.rp, switch_width_pair=args.switch,
                  cutoff_triple=args.rt, switch_width_triple=args.switch_triple, s9=args.s9,
                  cutoff_cn=args.cn_cutoff, cutoff_eeq_cn=args.cn_cutoff,
                  regime=args.regime, eeq_solver=args.eeq_solver,
                  triplet_chunk=args.chunk, triplet_cache=args.triplet_cache)
if args.cutoff_eeq is not None:
    d4_options["cutoff_eeq"] = args.cutoff_eeq
model = ForceStressOutput(D4Dispersion(**d4_options), compute_stress=True).to(dev)
d4 = model.model.d4
if args.eeq_reuse:
    d4.enable_eeq_reuse()


def graph_of(positions):
    s = {"pos": torch.tensor(positions, dtype=DTYPE, device=dev),
         "atomic_numbers": torch.tensor(z, device=dev),
         "cell": torch.tensor(cell, dtype=DTYPE, device=dev),
         "pbc": torch.ones(3, dtype=torch.bool, device=dev)}
    return structure_to_graph(s, model.model.cutoff, device=dev)


try:
    import vesin
    vesin_version = getattr(vesin, "__version__", "yes")
except ImportError:
    vesin_version = "not installed"
env = {"python": platform.python_version(), "torch": torch.__version__,
       "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
       "xnn": xnn.__version__, "xnn_path": os.path.dirname(xnn.__file__),
       "vesin": vesin_version}
print("environment:", json.dumps(env))
print("options:", json.dumps(vars(args)))
regime = d4.select_regime(len(z), True)
print(f"system: {len(z)} atoms, L = {cell[0, 0]:.2f} A, regime = {regime}, "
      f"neighbor-list radius {model.model.cutoff:.2f} A (cutoff_eeq {d4.cutoff_eeq:.1f} A)", flush=True)

# ----------------------------------------------------------------------------
# neighbor list, then the evaluation
# ----------------------------------------------------------------------------
graph_times = []
for _ in range(3):
    sync()
    t0 = time.perf_counter()
    g = graph_of(pos)
    sync()
    graph_times.append(time.perf_counter() - t0)
n_edges = int(g.edge_index.shape[1])
print(f"neighbor list: {n_edges} edges; cold {graph_times[0]:.3f} s, "
      f"warm {min(graph_times[1:]):.3f} s", flush=True)

if cuda:
    torch.cuda.reset_peak_memory_stats()
model(g)                                  # warm-up (compilation, caches, a first reuse step)
sync()
t0 = time.perf_counter()
out = model(g)
sync()
total = time.perf_counter() - t0
energy = float(out["energy"].detach())
forces0 = out["forces"].detach().double().cpu().numpy()
stress0 = out["stress"].detach().double().cpu().numpy()
if args.save_forces:
    np.savez(args.save_forces, forces=forces0, stress=stress0, energy=energy)
del out
peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
print(f"E + F + S (warm, no timers): {total:.3f} s   E = {energy:.8f} eV   "
      f"peak GPU {peak:.1f} GB", flush=True)

ENABLED[0] = True
sync()
t0 = time.perf_counter()
out = model(graph_of(pos))
sync()
total_timed = time.perf_counter() - t0
del out
ENABLED[0] = False
fwd = sum(v[1] for (ph, _), v in STATS.items() if ph == "forward")
bwd = STATS.get(("backward", "autograd.grad (forces + stress)"), [0, 0, 0])[0]
print(f"with timers: {total_timed:.3f} s = forward {total_timed - bwd:.3f} s + backward (autograd.grad) {bwd:.3f} s")
if CHUNKS:
    print(f"three-body block size (auto): {CHUNKS[-1]}")
rows = sorted(STATS.items(), key=lambda kv: -kv[1][0])
print(f"\n{'phase':9s} {'stage':44s} {'incl. s':>9s} {'self s':>9s} {'calls':>6s}")
for (ph, label), (incl, self_, calls) in rows:
    if incl >= 0.001:
        print(f"{ph:9s} {label:44s} {incl:9.3f} {self_:9.3f} {calls:6d}")

# ----------------------------------------------------------------------------
# frames: an MD-like sequence of small displacements
# ----------------------------------------------------------------------------
frame_times, frame_dq, frame_df = [], [], []
if args.frames > 0:
    fresh = ForceStressOutput(D4Dispersion(**d4_options), compute_stress=True).to(dev)
    x = pos.copy()
    for k in range(args.frames):
        x = x + rng.normal(scale=args.step, size=x.shape)
        gk = graph_of(x)
        sync()
        t0 = time.perf_counter()
        out = model(gk)
        sync()
        frame_times.append(time.perf_counter() - t0)
        ref = fresh(graph_of(x))
        frame_dq.append(float((out["eeq_charges"] - ref["eeq_charges"]).abs().max()))
        frame_df.append(float((out["forces"] - ref["forces"]).abs().max()))
        del out, ref
    print(f"\n{args.frames} displaced frames ({args.step} A rms per frame): "
          f"{1000 * np.mean(frame_times):.1f} ms per frame (min {1000 * np.min(frame_times):.1f}), "
          f"vs a fresh solve: max |dq| {max(frame_dq):.1e} e, max |dF| {max(frame_df):.1e} eV/A")
    if args.eeq_reuse:
        st = d4.__dict__["_eeq_reuse"].stats
        print(f"EEQ reuse: {st['solves']} solves, {st['iterations'] / max(st['solves'], 1):.1f} iterations "
              f"per solve, preconditioner formed {st['preconditioners']}x, {st['fallbacks']} fallbacks")

# ----------------------------------------------------------------------------
# kernels
# ----------------------------------------------------------------------------
if args.kernels and cuda:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(graph_of(pos))
        sync()
    print("\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if args.json:
    result = {"environment": env, "options": vars(args), "atoms": len(z), "edges": n_edges,
              "graph_cold_s": graph_times[0], "graph_warm_s": min(graph_times[1:]),
              "total_s": total, "backward_s": bwd, "energy_eV": energy, "peak_gb": peak,
              "block_size": CHUNKS[-1] if CHUNKS else None,
              "stages": {f"{ph}:{label}": {"inclusive": v[0], "self": v[1], "calls": v[2]}
                         for (ph, label), v in STATS.items()},
              "frames_ms": [1000 * t for t in frame_times], "frames_dq": frame_dq, "frames_df": frame_df}
    with open(args.json, "w") as f:
        json.dump(result, f, indent=1)
    print("wrote", args.json)
