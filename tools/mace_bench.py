#!/usr/bin/env python
"""Where the time goes in xnn's MACE on the GPU: energy + forces (+ stress).

Loads a model through the model hub (a registry name such as ``mace-off23-small``,
a trainer checkpoint or a model directory), builds a realistic periodic system, and
reports for one molecular-dynamics step

* the neighbor-list / graph build,
* the total time, split into the forward pass and the backward pass (the forces and
  stress are ``torch.autograd.grad`` of the energy),
* a table of the MACE stages (edge features, each interaction, product and readout)
  with synchronized timers in the forward pass,
* the forward and backward cost of every block on its real inputs (``--blocks``):
  the inputs a block saw in the step are captured and the block is re-run with a
  random cotangent, which is the work the force computation does in it,
* the aten operators with the most CUDA time, forward and backward (``--ops``),
* the peak memory.

With ``--sizes`` the step is timed for several supercell sizes (the scaling table).
``--train B`` times a training step instead: a batch of ``B`` copies of the system,
forces with ``create_graph=True`` and the backward pass of an energy + force loss.

Systems: ``water`` (a jittered lattice of waters at 1 g/cm^3, or a real liquid frame
given with ``--frame file.extxyz``, tiled ``--reps`` times along each axis) and
``si`` (diamond silicon, thermally rattled). Tiling a real frame keeps its neighbor
statistics at every size.

Nothing in xnn is patched: the timers wrap module forwards through hooks. Timers
call ``torch.cuda.synchronize()`` at every boundary, so the stage table is
attributable; the totals are measured without them, as the median of
``--repeats`` steps.

Examples::

    python tools/mace_bench.py --model mace-off23-small --system water --reps 4 --blocks --ops
    python tools/mace_bench.py --model runs/water/best.pt --core --frame water64.extxyz --sizes 1 2 3 4 6
    python tools/mace_bench.py --model mace-mp-0-medium --system si --reps 10 --dtype float64
    python tools/mace_bench.py --model mace-off23-medium --train 8 --reps 1 --json out.json
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import warnings

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", required=True, help="registry name, checkpoint or model directory")
p.add_argument("--head", default=None, help="head of a multi-head model")
p.add_argument("--core", action="store_true",
               help="profile the bare MACE inside any LES / dispersion wrappers")
p.add_argument("--no-dispersion", action="store_true",
               help="do not add a dispersion term recorded in the checkpoint")
p.add_argument("--system", choices=["water", "si"], default="water")
p.add_argument("--frame", default=None, help="extxyz frame (periodic) to tile instead of the lattice")
p.add_argument("--reps", type=int, default=2, help="supercell repeats along each axis")
p.add_argument("--sizes", type=int, nargs="*", default=None, help="repeat counts for a scaling table")
p.add_argument("--dtype", choices=["float32", "float64", "checkpoint"], default="float32")
p.add_argument("--stress", action="store_true", help="also compute the stress (NPT)")
p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (float32 only)")
p.add_argument("--compile", choices=["static", "dynamic", "auto"], default=None,
               help="torch.compile the MACE core: static shapes (recompiles for every new edge "
                    "count, an upper bound), dynamic, or auto (static first, dynamic on change)")
p.add_argument("--train", type=int, default=0, help="time a training step on a batch of this many copies")
p.add_argument("--repeats", type=int, default=10)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--blocks", action="store_true", help="forward/backward cost of every block")
p.add_argument("--ops", action="store_true", help="top aten operators by CUDA time")
p.add_argument("--fast", choices=["off", "on", "auto"], default="off",
               help="fast paths (cuEquivariance MACE, factorized LES / dispersion): off (the reference, "
                    "default), on, or auto")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--cache-dir", default=None, help="model hub cache")
p.add_argument("--json", default=None, help="write the results to this JSON file")
args = p.parse_args()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The TorchScript type system")

from xnn.common.data import collate, structure_to_graph  # noqa: E402
from xnn.common.models import ForceStressOutput  # noqa: E402
from xnn.common.models.hub import load_pretrained  # noqa: E402
from xnn.common.models.hub.checkpoint import core_model  # noqa: E402

dev = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
torch.backends.cudnn.allow_tf32 = bool(args.tf32)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


# the model
dtype = None if args.dtype == "checkpoint" else getattr(torch, args.dtype)
loaded = load_pretrained(args.model, head=args.head, cache_dir=args.cache_dir, dtype=dtype,
                         dispersion=False if args.no_dispersion else None,
                         compute_stress=args.stress, device=str(dev), quiet=True,
                         use_fast={"off": False, "on": True, "auto": "auto"}[args.fast])
model = loaded.model
if args.core:
    model = ForceStressOutput(core_model(model.model), compute_stress=args.stress)
    from xnn.common.models.fast import set_use_fast
    set_use_fast(model, {"off": False, "on": True, "auto": "auto"}[args.fast])
DTYPE = next(model.parameters()).dtype
torch.set_default_dtype(DTYPE)
mace = core_model(model.model)
cutoff = float(getattr(model.model, "cutoff", loaded.cutoff))
model.requires_grad_(args.train > 0)
if args.compile:
    # the tensor-only core; an MD engine changes the edge count every step
    mace.node_features_energy = torch.compile(
        mace.node_features_energy, dynamic={"static": False, "dynamic": True, "auto": None}[args.compile])


# the system
WATER = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047], [0.0, -0.763239, -0.477047]])


def unit_cell():
    """Positions, atomic numbers and cell of the unit to tile."""
    rng = np.random.default_rng(args.seed)
    if args.frame:
        from ase.io import read
        atoms = read(args.frame, 0)
        return atoms.positions, atoms.numbers, np.asarray(atoms.cell)
    if args.system == "si":
        from ase.build import bulk
        atoms = bulk("Si", "diamond", a=5.431, cubic=True)
        return atoms.positions, atoms.numbers, np.asarray(atoms.cell)
    n, a = 4, 3.104                       # 64 waters at about 1 g/cm^3
    pos = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                q *= np.sign(np.linalg.det(q))
                pos.extend((WATER - WATER[0]) @ q.T + np.array([i, j, k]) * a
                           + rng.normal(scale=0.1, size=3))
    return np.array(pos), np.array([8, 1, 1] * n ** 3), np.eye(3) * a * n


def supercell(reps):
    pos, z, cell = unit_cell()
    shifts = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
    big = (pos[None] + (shifts @ cell)[:, None]).reshape(-1, 3)
    rng = np.random.default_rng(args.seed + reps)
    if args.system == "si" and not args.frame:
        big = big + rng.normal(scale=0.05, size=big.shape)    # ~300 K displacements
    return big, np.tile(z, len(shifts)), cell * reps


def graph_of(pos, z, cell):
    s = {"pos": torch.tensor(pos, dtype=DTYPE, device=dev),
         "atomic_numbers": torch.tensor(z, device=dev),
         "cell": torch.tensor(cell, dtype=DTYPE, device=dev),
         "pbc": torch.ones(3, dtype=torch.bool, device=dev)}
    return structure_to_graph(s, cutoff, device=dev)


def median_time(fn, repeats, warmup):
    """Median and all wall times of ``fn`` (synchronized)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def best_time(fn, repeats, warmup):
    """Minimum wall time of ``fn``: the least contended estimate on a shared GPU."""
    return min(median_time(fn, repeats, warmup)[1])


# the step
def md_step(g):
    out = model(g)
    return out["energy"].sum().item()


def split_step(g):
    """Forward and backward of one MD step, timed separately."""
    sync()
    t0 = time.perf_counter()
    g = g.clone() if hasattr(g, "clone") else g
    pos = g.pos.requires_grad_(True)
    out = model.model(g)
    e = out["energy"].sum()
    sync()
    t1 = time.perf_counter()
    torch.autograd.grad(e, pos)
    sync()
    return t1 - t0, time.perf_counter() - t1


def train_step(batch):
    out = model(batch)                              # forces with create_graph in train mode
    loss = out["energy"].pow(2).mean() + out["forces"].pow(2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    return loss.item()


def gpu_state():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return q.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


env = {"python": platform.python_version(), "torch": torch.__version__,
       "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
       "gpus_before": gpu_state(), "model": args.model, "dtype": str(DTYPE).replace("torch.", ""),
       "tf32": args.tf32, "compile": args.compile, "cutoff": cutoff,
       "mace": {"channels": mace._n_scalar_features, "hidden_irreps": str(mace.interactions[0].hidden_irreps),
                "max_ell": int(mace.irreps_sh.lmax), "layers": len(mace.interactions),
                "elements": len(mace.species), "params": sum(q.numel() for q in mace.parameters())}}
print(json.dumps(env, indent=1))
results = {"env": env}


def save():
    """Write the results so far (after every size, so a later failure keeps them)."""
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if args.train:
    model.train()
    pos, z, cell = supercell(args.reps)
    structs = []
    rng = np.random.default_rng(args.seed)
    for _ in range(args.train):
        structs.append({"pos": pos + rng.normal(scale=0.02, size=pos.shape), "atomic_numbers": z,
                        "cell": cell, "pbc": [True] * 3})
    batch = collate([structure_to_graph(s, cutoff) for s in structs]).to(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t, ts = median_time(lambda: train_step(batch), args.repeats, args.warmup)
    res = {"atoms_per_structure": len(z), "batch": args.train, "edges": int(batch.edge_index.shape[1]),
           "step_s": t, "structures_per_s": args.train / t,
           "peak_GB": torch.cuda.max_memory_allocated(dev) / 1e9}
    print(f"\ntraining step: {args.train} x {len(z)} atoms, {res['edges']} edges | {t * 1e3:.1f} ms | "
          f"{res['structures_per_s']:.1f} structures/s | peak {res['peak_GB']:.2f} GB")
    results["train"] = res

sizes = args.sizes if args.sizes else ([] if args.train else [args.reps])
results["scaling"] = []
for reps in sizes:
    pos, z, cell = supercell(reps)
    try:
        sync()
        t0 = time.perf_counter()
        g = graph_of(pos, z, cell)
        sync()
        t_graph_cold = time.perf_counter() - t0
        t_graph, _ = median_time(lambda: graph_of(pos, z, cell), max(3, args.repeats // 2), 1)
        torch.cuda.reset_peak_memory_stats(dev)
        t_step, ts = median_time(lambda: md_step(g), args.repeats, args.warmup)
        peak = torch.cuda.max_memory_allocated(dev) / 1e9
        splits = [split_step(g) for _ in range(max(3, args.repeats // 2))]
        t_fwd = statistics.median(s[0] for s in splits)
        t_bwd = statistics.median(s[1] for s in splits)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        # e3nn's scripted code raises a plain RuntimeError for out-of-memory
        if "out of memory" not in str(exc):
            raise
        print(f"{len(z):8d} atoms: out of memory")
        results["scaling"].append({"atoms": len(z), "oom": True})
        torch.cuda.empty_cache()
        break
    n, e = len(z), int(g.edge_index.shape[1])
    row = {"reps": reps, "atoms": n, "edges": e, "edges_per_atom": e / n, "graph_cold_s": t_graph_cold,
           "graph_s": t_graph, "step_s": t_step, "forward_s": t_fwd, "backward_s": t_bwd,
           "us_per_atom": t_step / n * 1e6, "atoms_steps_per_s": n / t_step, "peak_GB": peak}
    results["scaling"].append(row)
    save()
    print(f"{n:8d} atoms {e / n:6.1f} edges/atom | graph {t_graph * 1e3:7.2f} ms | step {t_step * 1e3:8.2f} ms "
          f"(fwd {t_fwd * 1e3:7.2f}, bwd {t_bwd * 1e3:7.2f}) | {row['us_per_atom']:6.2f} us/atom | "
          f"peak {peak:6.2f} GB")

if not args.train and (args.blocks or args.ops):
    # the block profile keeps every block's inputs alive, so it runs at --reps only
    # when a sweep showed that size fits with half the GPU to spare (else the
    # largest swept size that does)
    reps = args.reps
    budget = 0.5 * torch.cuda.get_device_properties(dev).total_memory / 1e9
    fits = [r for r in results["scaling"] if not r.get("oom")]
    if fits and not any(r["reps"] == reps and r["peak_GB"] <= budget for r in fits):
        small = [r for r in fits if r["peak_GB"] <= budget] or fits[:1]
        reps = max(small, key=lambda r: r["atoms"])["reps"]
        print(f"\nprofiling at --reps {reps} ({args.reps} does not fit the block profile)")
    results["profile_reps"] = reps
    g = None
    torch.cuda.empty_cache()
    pos, z, cell = supercell(reps)
    g = graph_of(pos, z, cell)

if args.blocks and not args.train:
    # Per-block forward and backward time inside the real step, from CUDA events
    # (no host synchronization). Forward: events in the pre / post hooks. Backward:
    # the pre-hook routes every float input through a view whose gradient hook
    # records "this block's input gradients are done", and the post-hook puts a
    # hook on the output that records "this block's backward starts".
    def named_blocks():
        yield "edge: spherical harmonics", mace.edge_feat.sph
        yield "edge: radial basis", mace.edge_feat.rbf
        yield "node embedding", mace.node_embedding
        for i, (inter, prod, read) in enumerate(zip(mace.interactions, mace.products, mace.readouts)):
            yield f"L{i} skip_tp (element linear)", inter.skip_tp
            yield f"L{i} linear_up", inter.linear_up
            yield f"L{i} radial MLP", inter.conv_tp_weights
            yield f"L{i} conv_tp (tensor product)", inter.conv_tp
            yield f"L{i} linear (post-aggregation)", inter.linear
            yield f"L{i} interaction (total)", inter
            yield f"L{i} symmetric contraction", prod.symmetric_contractions
            yield f"L{i} product linear", prod.linear
            yield f"L{i} product (total)", prod
            yield f"L{i} readout", read

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    marks = {}           # label -> {"f0", "f1", "b0", "b1": [events]}
    captured = {}

    def make_hooks(label):
        def pre(m, inp):
            if label not in captured:
                captured[label] = tuple(x.detach().clone() if torch.is_tensor(x) else x for x in inp)
            rec = marks.setdefault(label, {"f0": [], "f1": [], "b0": [], "b1": []})
            rec["f0"].append(ev())
            new = []
            for x in inp:
                if torch.is_tensor(x) and x.requires_grad:
                    x = x.view_as(x)
                    x.register_hook(lambda g, rec=rec: rec["b1"].append(ev()))
                new.append(x)
            return tuple(new)

        def post(m, inp, out):
            rec = marks[label]
            rec["f1"].append(ev())
            outs = out if isinstance(out, tuple) else (out,)
            for o in outs:
                if torch.is_tensor(o) and o.requires_grad:
                    o.register_hook(lambda g, rec=rec: rec["b0"].append(ev()))
                    break
        return pre, post

    def profiled_step():
        gg = g.clone() if hasattr(g, "clone") else g
        pos = gg.pos.requires_grad_(True)
        start = ev()
        e = model.model(gg)["energy"].sum()
        mid = ev()
        torch.autograd.grad(e, pos)
        end = ev()
        return start, mid, end

    hooks = []
    for label, mod in named_blocks():
        pre, post = make_hooks(label)
        hooks += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    n_rep = max(5, args.repeats)
    for _ in range(2):
        profiled_step()
    sync()
    marks.clear()
    totals = []
    for _ in range(n_rep):
        totals.append(profiled_step())
    sync()
    for h in hooks:
        h.remove()
    t_f = min(a.elapsed_time(b) for a, b, c in totals) / 1e3
    t_b = min(b.elapsed_time(c) for a, b, c in totals) / 1e3

    def span(rec, k0, k1):
        # per repeat: forward has one f0/f1 pair; backward ends at the last input gradient
        vals = []
        n0, n1 = len(rec[k0]), len(rec[k1])
        if n0 == 0 or n1 == 0:
            return float("nan")
        per = n1 // n0 if k1 == "b1" else 1
        for r in range(n0):
            a = rec[k0][r]
            b = rec[k1][min((r + 1) * per - 1, n1 - 1)] if k1 == "b1" else rec[k1][r]
            vals.append(a.elapsed_time(b) / 1e3)
        return min(vals)

    def block_cost(mod, inputs):
        xs = [t.clone().requires_grad_(t.is_floating_point()) if torch.is_tensor(t) else t for t in inputs]
        need = [x for x in xs if torch.is_tensor(x) and x.requires_grad]

        def fwd():
            with torch.no_grad():
                mod(*xs)

        def fwd_bwd():
            out = mod(*xs)
            outs = [o for o in (out if isinstance(out, tuple) else (out,)) if torch.is_tensor(o) and o.requires_grad]
            torch.autograd.grad(outs, need, [torch.ones_like(o) for o in outs], allow_unused=True)
        return best_time(fwd, 15, 3), best_time(fwd_bwd, 15, 3)

    print(f"\nstep (CUDA events, min of {n_rep}): forward {t_f * 1e3:.2f} ms, backward {t_b * 1e3:.2f} ms")
    print(f"{'block':34s} {'fwd':>9s} {'bwd':>9s} {'share':>6s} | {'alone: fwd':>10s} {'fwd+bwd':>9s}")
    rows = []
    for label, mod in named_blocks():
        rec = marks.get(label)
        if rec is None:
            continue
        tf_in, tb_in = span(rec, "f0", "f1"), span(rec, "b0", "b1")
        try:
            af, afb = block_cost(mod, captured[label])
        except Exception as exc:
            af = afb = float("nan")
        share = (tf_in + (tb_in if tb_in == tb_in else 0.0)) / (t_f + t_b)
        rows.append({"block": label, "forward_s": tf_in, "backward_s": tb_in, "share_of_step": share,
                     "alone_forward_s": af, "alone_forward_backward_s": afb})
        print(f"{label:34s} {tf_in * 1e3:7.3f}ms {tb_in * 1e3:7.3f}ms {share:6.1%} | "
              f"{af * 1e3:8.3f}ms {afb * 1e3:7.3f}ms")
    results["blocks"] = {"forward_s": t_f, "backward_s": t_b, "rows": rows}

if args.ops and not args.train:
    from torch.profiler import ProfilerActivity, profile
    for _ in range(2):
        md_step(g)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            md_step(g)
        sync()
    ka = prof.key_averages()
    attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
    from torch.autograd import DeviceType
    events = [e for e in ka if getattr(e, attr) > 0]
    kernels = [e for e in events if e.device_type == DeviceType.CUDA]
    ops = [e for e in events if e.device_type != DeviceType.CUDA]
    out = {}
    for name, rows in (("aten operators", ops), ("CUDA kernels", kernels)):
        rows = sorted(rows, key=lambda e: -getattr(e, attr))
        total = sum(getattr(e, attr) for e in rows)
        print(f"\ntop {name} by self CUDA time ({total / 3e3:.2f} ms per step):")
        out[name] = []
        for e in rows[:18]:
            t = getattr(e, attr)
            out[name].append({"name": e.key, "cuda_ms_per_step": t / 3e3, "share": t / total,
                              "calls_per_step": e.count / 3})
            print(f"  {t / total:6.1%} {t / 3e3:8.3f} ms  x{e.count / 3:5.0f}  {e.key[:95]}")
    results["ops"] = out

results["gpus_after"] = gpu_state()
save()
if args.json:
    print("wrote", args.json)
