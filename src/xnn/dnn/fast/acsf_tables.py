"""The lane tables of the HDNNP symmetry-function kernels (GPU: :mod:`xnn.dnn.fast.acsf_kernels`, CPU:
:mod:`xnn.dnn.fast.acsf_cpu`): for every center element and neighbor species (radial) or species pair
(angular), the functions as lanes with their output column and parameters. Free of ``triton``."""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import Tensor

#: The cutoff kinds the kernels evaluate (their codes in the tables).
KERNEL_CUTOFFS = {"cosine": 0, "tanh": 1}


class Tables:
    """The lane tables of one term: ``(species * Q * K,)`` arrays (row ``s * Q + q`` holds the functions of
    center species ``s`` and neighbor species / species pair ``q``, ``K`` lanes, output column ``-1`` for an
    empty lane), the declared cutoffs ``(2 * 5,)`` and their count, the width ``D`` and whether type 1 is
    present. Built by :func:`tables`."""

    def __init__(self, arrays: List[Tensor], cut: Tensor, n_cut: int, q: int, k: int, d: int, has_type1: bool):
        self.arrays = [a.contiguous() for a in arrays] + [cut.contiguous()]
        self.nc, self.q, self.k, self.d, self.has1 = int(n_cut), int(q), int(k), int(d), bool(has_type1)


def pair_index(a: Tensor, b: Tensor, n_sp: int) -> Tensor:
    """The index of the unordered species pair ``(a, b)`` among the ``n_sp (n_sp + 1) / 2`` pairs."""
    lo, hi = torch.minimum(a, b), torch.maximum(a, b)
    return lo * n_sp - lo * (lo - 1) // 2 + (hi - lo)


def tables(functions: List[dict], species: List[int], cutoffs: List[dict], d: int, dtype,
           device) -> Tuple[Tables, Tables]:
    """The radial and angular lane tables of the featurizer's function rows (each row with ``element``,
    ``type``, ``neighbors``, ``column``, ``cutoff`` and its parameters; at most two cutoffs)."""
    index = {z: i for i, z in enumerate(species)}
    n_sp = len(species)
    n_pairs = n_sp * (n_sp + 1) // 2
    rad_rows: dict = {}
    ang_rows: dict = {}
    for row in functions:
        nbs = sorted(index[z] for z in row["neighbors"])
        s = index[row["element"]]
        if row["type"] in (1, 2):
            rad_rows.setdefault(s * n_sp + nbs[0], []).append(row)
        elif row["type"] in (3, 9):
            p = int(pair_index(torch.tensor(nbs[0]), torch.tensor(nbs[1]), n_sp))
            ang_rows.setdefault(s * n_pairs + p, []).append(row)
    cut = []
    for c in (cutoffs + cutoffs)[:2]:
        rc, rin = float(c["r_cut"]), float(c["r_inner"])
        cut += [float(KERNEL_CUTOFFS[c["kind"]]), rc, rin, 1.0 / (rc - rin), math.pi / (rc - rin)]
    cut_t = torch.tensor(cut, dtype=torch.float64).to(dtype).to(device)
    has1 = any(r["type"] == 1 for r in functions)

    def build(rows: dict, n_rows: int, keys) -> Tuple[List[Tensor], int]:
        k = max((len(v) for v in rows.values()), default=1)
        size = n_rows * k
        col = torch.full((size,), -1, dtype=torch.int32)
        out = {name: torch.zeros(size, dtype=torch.float64) for name, _ in keys}
        typ = torch.zeros(size, dtype=torch.int32)
        ci = torch.zeros(size, dtype=torch.int32)
        for key, lanes in rows.items():
            for j, row in enumerate(lanes):
                at = key * k + j
                col[at] = row["column"]
                typ[at] = row["type"]
                ci[at] = row["cutoff"]
                for name, fn in keys:
                    out[name][at] = fn(row)
        arrays = [col.to(device), typ.to(device)] + [out[name].to(dtype).to(device) for name, _ in keys]
        return arrays + [ci.to(device)], k

    rad_arrays, kr = build(rad_rows, n_sp * n_sp, [("eta", lambda r: r["eta"]), ("rs", lambda r: r["rs"]),
                                                   ("power", lambda r: r["power"])])
    ang_arrays, ka = build(ang_rows, n_sp * n_pairs, [("eta", lambda r: r["eta"]), ("lambda", lambda r: r["lambda"]),
                                                      ("zeta", lambda r: r["zeta"]),
                                                      ("pref", lambda r: 2.0 ** (1.0 - r["zeta"]))])
    return (Tables(rad_arrays, cut_t, len(cutoffs), n_sp, kr, d, has1),
            Tables(ang_arrays, cut_t, len(cutoffs), n_pairs, ka, d, has1))


