"""Nonbonded terms of a periodic GFN-FF model: EEQ electrostatics by an Ewald sum, dispersion over the images, and
the hydrogen- and halogen-bond rows over the images. Positions and cells enter in Angstrom and are used in Bohr; every
energy is differentiable in the positions and the cell."""
import functools
import itertools
import math

import numpy as np
import torch
from torch import Tensor

from . import gfnfflib as lib

BOHR = lib.BOHR
#: pair cutoff (Bohr) of the dispersion over the images
DISP_CUT = 60.0
#: squared radii (Bohr^2) of the translation lists of the dispersion and of the hydrogen bonds
DISP_LIST2 = 500.0
HB_LIST2 = 450.0
_CUBE2 = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=float)
_CUBE1 = np.array(list(itertools.product(range(-1, 2), repeat=3)), dtype=float)
_SHORT = np.array([v for v in itertools.product(range(-3, 4), repeat=3) if any(v)], dtype=float)
#: images of a pair within this distance (Bohr) of its minimum image are all centres of the real-space sum
NEAR_IMAGE = 0.01


#: Section ratio, bracket (1/Bohr) and bracket tolerance of the search for the Ewald splitting parameter
ETA_RATIO = (1.0 + math.sqrt(2.0)) / 2.0
ETA_BRACKET = (1.0e-8, 2.0)
ETA_TOL = 1.0e-4


def ewald_eta(cellb: np.ndarray, gamma: float = math.inf) -> float:
    """Ewald splitting parameter (1/Bohr): a section search with the ratio ``ETA_RATIO`` over ``ETA_BRACKET`` for the
    smallest absolute difference of ``(erfc(eta r) - erfc(gamma r))/r`` and ``4 pi/V exp(-g^2/(4 eta^2))/g^2``, with
    ``r`` and ``g`` the shortest lattice and reciprocal vectors (over integer combinations up to three of each basis
    vector) and ``gamma`` the Gaussian-charge exponent of the mean squared width of the atoms (:func:`mean_gamma`); it
    stops when the bracket is shorter than ``ETA_TOL`` and returns its midpoint, so the parameter is piecewise constant
    in the cell."""
    c = np.asarray(cellb, dtype=float)
    vol = abs(float(np.linalg.det(c)))
    rmin = float(np.linalg.norm(_SHORT @ c, axis=1).min())
    gmin = float(np.linalg.norm(_SHORT @ (2.0 * math.pi * np.linalg.inv(c).T), axis=1).min())

    width = math.erfc(gamma * rmin) / rmin if math.isfinite(gamma) else 0.0

    def f(a):
        rec = 4.0 * math.pi / vol * math.exp(-gmin * gmin / (4.0 * a * a)) / gmin ** 2
        return abs(math.erfc(a * rmin) / rmin - width - rec)

    lo, hi = ETA_BRACKET
    while abs(hi - lo) >= ETA_TOL:
        left = hi - (hi - lo) / ETA_RATIO
        right = lo + (hi - lo) / ETA_RATIO
        if f(left) < f(right):
            hi = right
        else:
            lo = left
    return 0.5 * (lo + hi)


def mean_gamma(model) -> float:
    """Gaussian-charge exponent ``1/sqrt(2 <alpha^2>)`` of the mean squared charge width of the atoms (1/Bohr)."""
    g = torch.diagonal(model.e_g).detach().double()
    return float(1.0 / torch.sqrt((1.0 / g ** 2).mean()))


def counts(model, x: Tensor, cellb: Tensor) -> Tensor:
    """Log-damped erf coordination numbers over the images; an atom and its own images enter from both sides."""
    n = x.shape[0]
    tr = model.cn_shift.to(x.dtype) @ cellb
    dv = x[None, :, None, :] + tr[None, None, :, :] - x[:, None, None, :]
    own = (torch.eye(n, dtype=torch.bool, device=x.device)[:, :, None]
           & (model.cn_shift.abs().sum(-1) == 0)[None, None, :])
    d = torch.sqrt((dv ** 2).sum(-1) + own.to(x.dtype))
    rc = ((4.0 / 3.0) * (model.rcov[:, None] + model.rcov[None, :])).to(x.dtype)[:, :, None]
    on = (d.detach() < 2.0 * rc) & ~own
    f = torch.where(on, 0.5 * (1.0 + torch.erf(-7.5 * (d / rc - 1.0))), torch.zeros_like(d))
    f = f * (1.0 + torch.eye(n, dtype=f.dtype, device=f.device)[:, :, None])
    count = f.sum((1, 2))
    return math.log(1.0 + math.exp(4.4)) - torch.log(1.0 + torch.exp(4.4 - count))


def _ewald_matrix(x: Tensor, cellb: Tensor, gam: Tensor, eta: float) -> Tensor:
    """Real space (the fixed +-2 cube around the minimum images of a pair, averaged over every image within
    ``NEAR_IMAGE`` of the shortest),
    reciprocal space (the fixed +-2 cube) and the self term, without the hardness."""
    n = x.shape[0]
    vol = torch.abs(torch.det(cellb))
    # minimum images among the raw difference of the wrapped coordinates and its +-1 neighbours
    f = (x[None, :, :] - x[:, None, :]) @ torch.linalg.inv(cellb)
    cand = torch.as_tensor(_CUBE1, dtype=x.dtype, device=x.device)
    v = (f[:, :, None, :] + cand[None, None]) @ cellb
    l2 = (v.detach() ** 2).sum(-1)
    ln = torch.sqrt(l2)
    tied = ln <= ln.min(-1, keepdim=True).values + NEAR_IMAGE
    i, j, k = torch.nonzero(tied, as_tuple=True)
    tr = torch.as_tensor(_CUBE2, dtype=x.dtype, device=x.device) @ cellb
    d = v[i, j, k][:, None, :] + tr[None]
    r2 = (d ** 2).sum(-1)
    small = r2.detach() < 1e-20
    r = torch.sqrt(torch.where(small, torch.ones_like(r2), r2))
    pair = (torch.erfc(eta * r) - torch.erfc(gam[i, j][:, None] * r)) / r
    per = torch.where(small, torch.zeros_like(r), pair).sum(-1)
    real = torch.zeros((n, n), dtype=x.dtype, device=x.device).index_put((i, j), per, accumulate=True)
    real = real / tied.sum(-1).to(x.dtype)
    rec_vec = 2.0 * math.pi * torch.linalg.inv(cellb).T
    g = torch.as_tensor(_CUBE2[np.any(_CUBE2 != 0, axis=1)], dtype=x.dtype, device=x.device) @ rec_vec
    k2 = (g ** 2).sum(-1)
    w = 4.0 * math.pi / vol * torch.exp(-k2 / (4.0 * eta ** 2)) / k2
    ph = x @ g.T
    c, s = torch.cos(ph), torch.sin(ph)
    rec = (c * w) @ c.T + (s * w) @ s.T
    return real + rec - torch.diag(torch.full((n,), 2.0 * eta / math.sqrt(math.pi), dtype=x.dtype, device=x.device))


def _constraints(model, dtype, device):
    """One charge constraint per fragment of the bonds found in both neighbour passes; the total charge sits on the
    fragment of the first atom."""
    t = model.topology
    n = len(t.z)
    fr = t.frags if len(t.frags) > 1 else [list(range(n))]
    con = torch.zeros((len(fr), n), dtype=dtype, device=device)
    for k, members in enumerate(fr):
        con[k, list(members)] = 1.0
    tot = torch.tensor([float(t.charge) if 0 in members else 0.0 for members in fr], dtype=dtype, device=device)
    return con, tot


def electrostatics(model, x: Tensor, cellb: Tensor, logcn: Tensor) -> Tensor:
    """EEQ charges and their energy over the images (Hartree); ``x`` and ``cellb`` in Bohr."""
    n = x.shape[0]
    eta = ewald_eta(cellb.detach().cpu().double().numpy(), mean_gamma(model))
    a = _ewald_matrix(x, cellb, model.e_g.to(x.dtype), eta) + torch.diag(model.e_diag.to(x.dtype))
    pos = logcn > 0
    xx = model.e_x0.to(x.dtype) + model.e_kappa.to(x.dtype) * torch.where(
        pos, torch.sqrt(torch.where(pos, logcn, torch.ones_like(logcn))), torch.zeros_like(logcn))
    con, tot = _constraints(model, x.dtype, x.device)
    nc = tot.shape[0]
    mat = torch.cat([torch.cat([a, con.T], 1), torch.cat([con, torch.zeros((nc, nc), dtype=x.dtype, device=x.device)],
                                                         1)], 0)
    sol = torch.linalg.solve(mat, torch.cat([xx, tot]))
    q, lam = sol[:n], sol[n:]
    # -q.x/2 + lambda.Q/2: the value at the solution, as xtb evaluates it for charged cells
    return -0.5 * (q * xx).sum() + 0.5 * (lam * tot).sum()


def dispersion(model, x: Tensor, cellb: Tensor, logcn: Tensor) -> Tensor:
    """Dispersion energy over the capped translation list (Hartree); an atom and its own images at full weight."""
    n = x.shape[0]
    g = torch.exp(-4.0 * (logcn[:, None] - model.d_ref.to(x.dtype)) ** 2)
    g = torch.where(model.d_valid, g, torch.zeros_like(g))
    alpha = torch.einsum("nr,krn->nk", g / g.sum(1, keepdim=True), model.d_alpha.to(x.dtype))
    c6 = (3.0 / math.pi) * (alpha * model.d_w.to(x.dtype)) @ alpha.T
    cell = cellb.detach().cpu().double().numpy() * BOHR
    # every translation of the capped list; the cutoff applies to the pair distances only
    sh = np.asarray(lib.translation_list(cell, DISP_LIST2, cap=True), dtype=float)
    tr = torch.as_tensor(sh, dtype=x.dtype, device=x.device) @ cellb
    d = x[None, :, None, :] - x[:, None, None, :] + tr[None, None, :, :]
    r2 = (d ** 2).sum(-1)
    ok = (r2.detach() > 1e-20) & (r2.detach() < DISP_CUT ** 2)
    r2s = torch.where(ok, r2, torch.ones_like(r2))
    r6 = r2s ** 3
    e = 1.0 / (r6 + model.d_r06.to(x.dtype)[:, :, None]) + 2.0 * model.d_q.to(x.dtype)[:, :, None] / (
        r6 * r2s + model.d_r08.to(x.dtype)[:, :, None])
    e = torch.where(ok, (model.d_zz.to(x.dtype) * c6)[:, :, None] * e, torch.zeros_like(e))
    w = torch.full((n, n), 0.5, dtype=x.dtype, device=x.device) + 0.5 * torch.eye(n, dtype=x.dtype, device=x.device)
    return -(w[:, :, None] * e).sum()


def _set_hugepage(flag):
    """Switch numpy's request for transparent huge pages on large arrays; returns the previous setting (None when
    this numpy has no such switch)."""
    core = getattr(np, "_core", None) or getattr(np, "core", None)
    switch = getattr(getattr(core, "multiarray", None), "_set_madvise_hugepage", None)
    return None if switch is None else switch(flag)


def _plain_pages(fn):
    """Run ``fn`` without huge-page requests for its arrays: the row searches allocate and free many large short-lived
    arrays, and on a host with fragmented memory every such request can stall in page compaction."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        before = _set_hugepage(False)
        try:
            return fn(*args, **kwargs)
        finally:
            if before:
                _set_hugepage(True)
    return run


def _lists(t):
    """Ligand neighbour lists with their images (an entry per image)."""
    return [sorted(v, key=lambda j: (int(j), j.s)) for v in t.lig]


#: elements of the largest temporary arrays of the row searches (blocks stay in the cache)
_BLOCK = 1 << 18

#: offset and width of a lattice shift component in the integer keys of (atom, atom, shift) triples
_KEY_OFF, _KEY_W = 64, 129

#: size of a dense table of image keys beyond four entries per key (larger key ranges are sorted instead)
_DENSE = 1 << 16


def _keys(i, j, s, n):
    """Integer keys of the triples (atom i, atom j, lattice shift s of j)."""
    s = np.asarray(s, dtype=np.int64).reshape(-1, 3) + _KEY_OFF
    return ((np.asarray(i, dtype=np.int64) * n + np.asarray(j, dtype=np.int64)) * _KEY_W + s[:, 0]) * _KEY_W ** 2 \
        + s[:, 1] * _KEY_W + s[:, 2]


def _hop_table(t):
    """The bond counts of :func:`~xnn.ffnn.models.gfnfflib.near_hops` without eta bonds as arrays, in its order: atom,
    partner, lattice shift of the partner and count (kept on the topology)."""
    tab = getattr(t, "_hop_table", None)
    if tab is None:
        hops = lib.near_hops(t, eta=False)
        a = np.repeat(np.arange(len(hops), dtype=np.int64), [len(h) for h in hops])
        flat = [(b, *sh, d) for h in hops for (b, sh), d in h.items()]
        v = np.asarray(flat, dtype=np.int64).reshape(-1, 5)
        tab = t._hop_table = (a, v[:, 0], v[:, 1:4], v[:, 4])
    return tab


def _neighbour_tables(nbl, n):
    """Padded neighbour lists of the atoms: index ``(n, w)``, lattice shift ``(n, w, 3)`` and mask."""
    w = max([len(v) for v in nbl] + [1])
    idx = np.zeros((n, w), dtype=np.int64)
    sh = np.zeros((n, w, 3), dtype=np.int64)
    mask = np.zeros((n, w), dtype=bool)
    for i, v in enumerate(nbl):
        for m, j in enumerate(v):
            idx[i, m], sh[i, m], mask[i, m] = int(j), j.s, True
    return idx, sh, mask


@_plain_pages
def bind(model, x: np.ndarray, skin: float = 0.0) -> dict:
    """Hydrogen- and halogen-bond rows of a periodic model at the wrapped positions ``x`` (Bohr); ``skin`` (Bohr)
    widens the distance cuts so that the rows stay valid for small moves."""
    t = model.topology
    z, q, n = t.z, t.qt, len(t.z)
    cell = np.asarray(model.cell, dtype=float)
    cellb = cell / BOHR
    nbl = _lists(t)
    nbi, nbs, nbm = _neighbour_tables(nbl, n)

    def widen(cut, k):
        return (math.sqrt(cut) + math.sqrt(k) * skin) ** 2

    cut, cut3, xcut = widen(lib.HB_CUT, 1), widen(lib.HB_CUT3, 3), widen(lib.XB_CUT, 1)
    shifts = lib.translation_list(cell, HB_LIST2)
    act = [lib.hb_active(z, nbl, q, i) for i in range(n)]
    hs = [h for h in range(n) if z[h] == 1 and len(nbl[h]) == 1 and (q[h] > lib.HB_H_CUT or z[int(nbl[h][0])] > 10)]
    # the H atoms of each donor, in order, with the shift of the H as seen from the donor
    dlist = [[] for _ in range(n)]
    for h in hs:
        d = nbl[h][0]
        dlist[int(d)].append((h, tuple(-v for v in d.s)))
    dptr = np.zeros(n + 1, dtype=np.int64)
    dptr[1:] = np.cumsum([len(v) for v in dlist])
    dh = np.array([h for v in dlist for h, _ in v], dtype=np.int64)
    dsh = np.array([s for v in dlist for _, s in v], dtype=np.int64).reshape(-1, 3)
    loose = np.array([h for h in hs if not (z[int(nbl[h][0])] == 6 and len(nbl[int(nbl[h][0])]) >= 4 and q[h] <= 0.06)],
                     dtype=np.int64)
    bonded_keys = np.unique(_keys(np.repeat(np.arange(n), nbm.sum(1)), nbi[nbm], nbs[nbm], n))
    on = np.array([i for i in range(n) if act[i]], dtype=np.int64)
    ti = np.asarray(shifts, dtype=np.int64)
    tv = ti.astype(float) @ cellb
    actv = np.asarray(act, dtype=np.int64)
    ci = cj = np.zeros(0, dtype=np.int64)
    cs = np.zeros((0, 3), dtype=np.int64)
    cd = np.zeros(0)
    if len(on):
        # all (i, j, image) in blocks of i (small arrays); an atom and its own images form no pair
        parts = []
        step = max(1, _BLOCK // (3 * len(on) * len(tv)))
        for lo in range(0, len(on), step):
            oa = on[lo:lo + step]
            d = x[on][None, :, None, :] + tv[None, None, :, :] - x[oa][:, None, None, :]
            d2 = np.einsum("ijkc,ijkc->ijk", d, d)
            ok = (d2 < cut) & (oa[:, None, None] != on[None, :, None]) \
                & (actv[oa][:, None, None] + actv[on][None, :, None] >= 3)
            a, b, k = np.nonzero(ok)
            parts.append((oa[a], on[b], ti[k], d2[a, b, k]))
        ci, cj, cs, cd = (np.concatenate([v[m] for v in parts]) for m in range(4))
    cb = np.isin(_keys(ci, cj, cs, n), bonded_keys)
    # bonded-H rows: the H atoms of the donor of every close pair that is no bond
    sel = np.flatnonzero(~cb & (dptr[ci + 1] > dptr[ci]))
    cnt = (dptr[ci + 1] - dptr[ci])[sel]
    rp = np.repeat(sel, cnt)
    off = np.arange(len(rp)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    hk = dptr[ci[rp]] + off
    bb = (ci[rp], cj[rp], cs[rp], dh[hk], dsh[hk])
    # loose rows: the loose H atoms near every close pair with j < i
    la, lb, ls, lh, lsh = ([] for _ in range(5))
    pl = np.flatnonzero(cj < ci)
    if len(loose) and len(pl):
        xh = (x[loose][:, None, :] + tv[None, :, :]).reshape(-1, 3)
        nk = len(tv)
        hd = np.array([int(nbl[h][0]) for h in loose], dtype=np.int64)
        hds = np.array([nbl[h][0].s for h in loose], dtype=np.int64).reshape(-1, 3)
        # the H images within the triple cut of each atom, nearest first
        near, nd = [], []
        for i in range(n):
            if act[i]:
                dd = x[i] - xh
                d2h = np.einsum("fc,fc->f", dd, dd)
                f = np.flatnonzero(d2h < cut3)
                o = np.argsort(d2h[f], kind="stable")
                near.append(f[o])
                nd.append(d2h[f][o])
            else:
                near.append(np.zeros(0, dtype=np.int64))
                nd.append(np.zeros(0))
        nptr = np.zeros(n + 1, dtype=np.int64)
        nptr[1:] = np.cumsum([len(v) for v in near])
        nflat = np.concatenate(near) if n else np.zeros(0, dtype=np.int64)
        span = 2.0 * cut3 + 1.0
        gkey = np.concatenate(nd) + np.repeat(np.arange(n), np.diff(nptr)) * span if n else np.zeros(0)
        # r_AH^2 + r_BH^2 = 2 r_MH^2 + r_AB^2 / 2 (M the midpoint of AB): a row has r_AH below r_AM plus the radius
        # of that sphere, and r_AH^2 below cut3 - r_AB^2
        c2 = cd[pl]
        lim = np.minimum(cut3 - c2, (0.5 * np.sqrt(c2) + np.sqrt(np.maximum(cut3 - 1.5 * c2, 0.0) / 2.0)) ** 2)
        lim = lim * (1.0 + 1e-9) + 1e-6
        cnt_all = np.maximum(np.searchsorted(gkey, ci[pl] * span + lim) - nptr[ci[pl]], 0)
        # blocks of pairs with a bounded number of candidates (small arrays)
        ends = np.cumsum(cnt_all)
        lo = 0
        while lo < len(pl):
            hi = max(lo + 1, int(np.searchsorted(ends, (ends[lo - 1] if lo else 0) + _BLOCK // 8, side="right")))
            sel, cnt = pl[lo:hi], cnt_all[lo:hi]
            lo = hi
            rp = np.repeat(sel, cnt)
            f = nflat[nptr[ci[rp]] + np.arange(len(rp)) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
            xb = x[cj[rp]] + cs[rp].astype(float) @ cellb
            da, db = x[ci[rp]] - xh[f], xb - xh[f]
            s3 = cd[rp] + np.einsum("fc,fc->f", da, da) + np.einsum("fc,fc->f", db, db)
            hit = np.flatnonzero(s3 < cut3)
            # pair-major rows in (H, image) order
            hit = hit[np.lexsort((f[hit], rp[hit]))]
            pp, v, w = rp[hit], f[hit] // nk, f[hit] % nk
            img = ti[w] + hds[v]
            own_i = (hd[v] == ci[pp]) & ~img.any(1)
            own_j = (hd[v] == cj[pp]) & (img == cs[pp]).all(1)
            keep = cb[pp] | (~own_i & ~own_j)
            la.append(ci[pp][keep])
            lb.append(cj[pp][keep])
            ls.append(cs[pp][keep])
            lh.append(loose[v][keep])
            lsh.append(ti[w][keep])
    cat = lambda v, shape: np.concatenate(v) if v else np.zeros(shape, dtype=np.int64)  # noqa: E731
    bl = (cat(la, 0), cat(lb, 0), cat(ls, (0, 3)), cat(lh, 0), cat(lsh, (0, 3)))
    # halogen bonds: the halogen in the cell, every neighbour image of it a donor, the acceptor within one cell
    ys = [i for i in range(n) if (z[i] == 6 and len(nbl[i]) < 4 and q[i] < lib.XB_CARBON_CUT)
          or (z[i] not in (6, 14) and z[i] in lib.HB_STRONG | lib.HB_WEAK)]
    xs = [xa for xa in range(n) if z[xa] in lib.XB_X and not (z[xa] == 16 and len(nbl[xa]) > 2)]
    xb = (np.zeros(0, dtype=np.int64), np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64),
          np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64))
    if xs and ys:
        ha, hb, hs, hd = _hop_table(t)
        sel = np.isin(ha, np.asarray(xs, dtype=np.int64)) & (hd < 4)
        hk = np.unique(_keys(ha[sel], hb[sel], hs[sel], n))
        # every (halogen, donor image) with every acceptor image next to the cell: acceptor-major, then shift
        dx = np.array([(xa, int(d), *d.s) for xa in xs for d in nbl[xa]], dtype=np.int64).reshape(-1, 5)
        ya = np.asarray(ys, dtype=np.int64)
        sr = np.asarray(shifts[:27], dtype=np.int64)
        # candidates from the distances of the donor atoms to the acceptor images (the donor's own image shift cancels),
        # with a margin; the exact tests below decide
        ud, inv = np.unique(dx[:, 1], return_inverse=True)
        sc = sr.astype(float) @ cellb
        found = [(np.zeros(0, dtype=np.int64),) * 3]
        step = max(1, _BLOCK // max(1, 3 * len(ya) * len(sr)))
        for lo in range(0, len(ud), step):
            dv = x[ud[lo:lo + step]][:, None, None, :] - x[ya][None, :, None, :] - sc[None, None, :, :]
            u, yv, kv = np.nonzero(np.einsum("uykc,uykc->uyk", dv, dv) < xcut * (1.0 + 1e-6) + 1e-9)
            found.append((u + lo, yv, kv))
        cu, cy, ck = (np.concatenate([f[m] for f in found]) for m in range(3))
        # every row of the donor atom of a candidate: row-major, then acceptor, then shift, as an exhaustive search
        rows = np.argsort(inv, kind="stable")
        first = np.searchsorted(inv[rows], np.arange(len(ud)))
        cnt = np.bincount(inv, minlength=len(ud))[cu]
        rr = rows[np.repeat(first[cu], cnt) + np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)]
        yi, kk = np.repeat(cy, cnt), np.repeat(ck, cnt)
        o = np.lexsort((kk, yi, rr))
        rr, yi, kk = rr[o], yi[o], kk[o]
        xa, dd, ds = dx[rr, 0], dx[rr, 1], dx[rr, 2:]
        yy = ya[yi]
        sy = ds + sr[kk]
        a1 = np.abs(sy) <= 1
        ok = (yy != xa) & ~((yy == dd) & ~sr.any(1)[kk]) & a1[:, 0] & a1[:, 1] & a1[:, 2]
        rr, yy, sy = rr[ok], yy[ok], sy[ok]
        r2 = (((x[dx[rr, 1]] + dx[rr, 2:].astype(float) @ cellb) - (x[yy] + sy.astype(float) @ cellb)) ** 2).sum(1)
        w = np.flatnonzero(r2 < xcut)
        w = w[~np.isin(_keys(dx[rr[w], 0], yy[w], sy[w], n), hk)]
        xb = (dx[rr[w], 1], dx[rr[w], 2:], yy[w], sy[w], dx[rr[w], 0])
    return _tables(t, nbl, (nbi, nbs, nbm), bb, bl, xb)


def _hb_atoms(t, nbl):
    z, q, n = t.z, t.qt, len(t.z)
    act = [lib.hb_active(z, nbl, q, i) for i in range(n)]

    def sig(v):
        return 1.0 / (1.0 + math.exp(v))

    def per(fn):
        return np.array([fn(i) for i in range(n)], dtype=float)

    o1 = np.array([lib.o1_class(z, nbl, i) for i in range(n)], dtype=int)
    return dict(rad=per(lambda i: lib.HB_RAD.get(z[i], 0.0)),
                ac=per(lambda i: lib.acidity(z, nbl, i) if act[i] else 0.0),
                bs=per(lambda i: lib.basicity(z, nbl, i) if act[i] else 0.0),
                sq=per(lambda i: sig(15.0 * q[i])), sh=per(lambda i: sig(-15.0 * q[i])), o1=o1,
                cnb=per(lambda i: 2.0 if z[i] == 7 and len(nbl[i]) == 1 else 11.2),
                n2=np.array([z[i] == 7 and len(nbl[i]) == 2 and not o1[i] for i in range(n)]))


def _shifts3(s0, s1, s2):
    """Image shifts of the three atoms of a row as one float array (rows, 3, 3), filled in place (no int64 copy);
    None stands for the home cell."""
    m = len(s1 if s1 is not None else s0)
    out = np.empty((m, 3, 3))
    for c, v in enumerate((s0, s1, s2)):
        if v is None:
            out[:, c] = 0.0
        else:
            out[:, c] = v
    return out


def _tables(t, nbl, tabs, bb, bl, xb):
    z, q = t.z, t.qt
    nbi, nbs, nbm = tabs
    at = _hb_atoms(t, nbl)
    a, b, sb, h, sh = bb
    rt = {"nbb": len(a), "nbl": len(bl[0]), "nxb": len(xb[0]), "bb_b": b, "bb_sb": sb, "bb_h": h, "bb_hs": sh}
    rt["bb_idx"] = np.stack([a, b, h], 1)
    rt["bb_sh"] = _shifts3(None, sb, sh)
    rr = at["rad"][a] + at["rad"][b]
    o1 = at["o1"][b]
    rt["bb_rr"] = rr
    rt["bb_p"] = np.where(o1 > 0, 1.75, 1.34) * at["ac"][a] * at["bs"][b] * at["sq"][a] * at["sq"][b] * at["sh"][h]
    rt["bb_cnb"] = at["cnb"][b] / rr if len(a) else np.zeros(0)
    rt["bb_nb"], rt["bb_nbsh"], rt["bb_nbm"] = nbi[b], (nbs[b] + sb[:, None, :]).astype(float), nbm[b]
    r1 = np.flatnonzero(o1 > 0)
    rt["o1_rows"] = r1
    xi, xs = nbi[b[r1], 0], nbs[b[r1], 0] + sb[r1]
    rt["o1_x"], rt["o1_xsh"] = xi, xs.astype(float).reshape(-1, 3)
    yi, ysh = nbi[xi], nbs[xi] + xs[:, None, :]
    ym = nbm[xi] & ~((yi == b[r1][:, None]) & (ysh == sb[r1][:, None, :]).all(-1))
    rt["o1_y"], rt["o1_ysh"], rt["o1_ym"] = yi, ysh.astype(float), ym
    r2 = np.flatnonzero(at["n2"][b]) if len(a) else np.zeros(0, np.int64)
    rt["n2_rows"] = r2
    rt["n2_nb"], rt["n2_nbsh"] = nbi[b[r2], :2], (nbs[b[r2], :2] + sb[r2][:, None, :]).astype(float)
    la, lb, ls, lh, lsh = bl
    rt["bl_idx"] = np.stack([la, lb, lh], 1)
    rt["bl_sh"] = _shifts3(None, ls, lsh)
    ba = at["bs"] * at["sq"]
    rt["bl_rr"] = at["rad"][la] + at["rad"][lb]
    rt["bl_ca"], rt["bl_cb"], rt["bl_ba"], rt["bl_bb"], rt["bl_qh"] = (at["ac"][la], at["ac"][lb], ba[la], ba[lb],
                                                                      at["sh"][lh])
    xd, xds, xy, xys, xx = xb
    rt["xb_idx"] = np.stack([xd, xy, xx], 1)
    rt["xb_sh"] = _shifts3(xds, xys, None)
    topo = np.array([lib.TOPO_RAD[v] for v in z], dtype=float)
    rt["xb_rad"] = topo[xd] + topo[xy]
    cc = np.array([lib.XB_C.get(v, 0.0) for v in z], dtype=float)
    cx = np.array([1.0 + 0.03 * math.exp(-15.0 * qv) for qv in q])
    cy = np.array([1.0 + 0.03 * math.exp(15.0 * qv) for qv in q])
    rt["xb_p"] = cc[xx] / (cx[xx] * cy[xy])
    return rt


@_plain_pages
def bond_acceptors(model, rows: dict):
    """Acceptors of the H-N and H-O bond softening over the images: the N and O acceptors of the H in the bonded-H
    rows, each image once; a bond whose H lies in another image than its partner takes none. Returns the rows
    ``(bond, H, acceptor)`` in extended indices and their reference distances (Bohr)."""
    z = np.asarray(model.topology.z)
    b, sb, h, sh = rows["bb_b"], rows["bb_sb"], rows["bb_h"], rows["bb_hs"]
    acc = np.isin(z[b], (7, 8))
    accs, dd = [], []
    n = len(z)
    for ib, hh in model._bond_h:
        if any(getattr(hh, "s", (0, 0, 0))):
            continue
        ks = np.flatnonzero(acc & (h == int(hh)))
        img = sb[ks] - sh[ks]
        # each (acceptor, image) once, in the order of its first row
        _, first = np.unique(_keys(b[ks], b[ks], img, n), return_index=True)
        for k, im in zip(ks[np.sort(first)].tolist(), img[np.sort(first)].tolist()):
            accs.append((ib, model._e(hh), model._e(lib.Img(int(b[k]), im))))
            dd.append(1.78 * (4.0 / 3.0) * (lib.RCOV[1] + lib.RCOV[z[b[k]]]))
    order = sorted(range(len(accs)), key=lambda k: accs[k])
    return [accs[k] for k in order], [dd[k] for k in order]


def _place(x, cellb, idx, sh):
    return x[idx] + sh @ cellb


def hb_xb(rows: dict, x: Tensor, cellb: Tensor) -> dict:
    """Hydrogen- and halogen-bond energies (Hartree) of the bound rows; ``x`` and ``cellb`` in Bohr."""
    dt, dev = x.dtype, x.device

    def tt(v):
        return torch.as_tensor(v, dtype=dt, device=dev)

    def ti(v):
        return torch.as_tensor(v, dtype=torch.long, device=dev)

    e = torch.zeros((), dtype=dt, device=dev)
    if rows["nbb"]:
        idx, sh = ti(rows["bb_idx"]), tt(rows["bb_sh"])
        xa, xb, xh = (_place(x, cellb, idx[:, k], sh[:, k]) for k in range(3))
        rr = tt(rows["bb_rr"])
        rab, rah, rbh = (xa - xb).norm(dim=-1), (xa - xh).norm(dim=-1), (xb - xh).norm(dim=-1)
        arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
        on = (arg < lib.HB_SWITCH_CUT) & (rab.detach() ** 2 < lib.HB_CUT)
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(lib.HB_SWITCH_CUT))), torch.zeros_like(arg))
        xn = x[ti(rows["bb_nb"])] + tt(rows["bb_nbsh"]) @ cellb
        # the padding of the neighbour rows at a constant vector: no zero length in the second derivatives
        nbm = torch.as_tensor(rows["bb_nbm"], device=dev)
        pad = torch.ones_like(xn)
        arm = (torch.where(nbm[..., None], xa[:, None] - xn, pad).norm(dim=-1)
               + torch.where(nbm[..., None], xb[:, None] - xn, pad).norm(dim=-1)) / rab[:, None]
        fac = 2.0 / (1.0 + torch.exp(-tt(rows["bb_cnb"])[:, None] * (arm - 1.0))) - 1.0
        o = o * torch.where(nbm, fac, torch.ones_like(fac)).prod(1)
        if len(rows["o1_rows"]):
            r = ti(rows["o1_rows"])
            xx1 = x[ti(rows["o1_x"])] + tt(rows["o1_xsh"]) @ cellb
            u, v = xx1 - xb[r], xh[r] - xb[r]
            cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
            f1 = 1.0 - 16.0 / 45.0 * (cos + 0.5) ** 2
            b1 = (u / u.norm(dim=-1, keepdim=True))[:, None, :]
            b2 = x[ti(rows["o1_y"])] + tt(rows["o1_ysh"]) @ cellb - xx1[:, None, :]
            vv = (v[:, None, :] - (v[:, None, :] * b1).sum(-1, keepdim=True) * b1).expand_as(b2)
            ww = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
            s = (torch.linalg.cross(b1.expand_as(b2), vv) * ww).sum(-1)
            c = (vv * ww).sum(-1)
            den = s ** 2 + c ** 2
            # a neighbour on the acceptor axis counts as perpendicular
            sin2 = torch.where(den > 0, s ** 2 / torch.where(den > 0, den, torch.ones_like(den)), torch.ones_like(den))
            f1 = f1 * torch.where(torch.as_tensor(rows["o1_ym"], device=dev), 1.0 - 0.06 * sin2,
                                  torch.ones_like(sin2)).prod(1)
            o = o * torch.ones_like(o).index_copy(0, r, f1)
        if len(rows["n2_rows"]):
            r = ti(rows["n2_rows"])
            nb2 = x[ti(rows["n2_nb"])] + tt(rows["n2_nbsh"]) @ cellb
            u = xb[r] - 0.5 * (nb2[:, 0] + nb2[:, 1])
            lp = xb[r] + 0.41 * u / u.norm(dim=-1, keepdim=True)
            arg2 = 56.0 / rr[r] * ((0.41 + (xa[r] - lp).norm(dim=-1)) / rab[r] - 1.0)
            o = o * torch.ones_like(o).index_copy(0, r, 2.0 / (1.0 + torch.exp(arg2.clamp_max(300.0))))
        damp = 1.0 / (1.0 + (22.0 * rr / rab ** 2) ** 6) / (1.0 + (rab ** 2 / 85.0) ** 6)
        e = e + (-tt(rows["bb_p"]) * o * damp * (1.8 / rbh ** 3 - 0.8 / rab ** 3)).sum()
    if rows["nbl"]:
        idx, sh = ti(rows["bl_idx"]), tt(rows["bl_sh"])
        xa, xb, xh = (_place(x, cellb, idx[:, k], sh[:, k]) for k in range(3))
        rr = tt(rows["bl_rr"])
        rab, rah, rbh = (xa - xb).norm(dim=-1), (xa - xh).norm(dim=-1), (xb - xh).norm(dim=-1)
        arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
        r2 = rab.detach() ** 2
        on = (arg < lib.HB_SWITCH_CUT) & (r2 < lib.HB_CUT) & (r2 + rah.detach() ** 2 + rbh.detach() ** 2 < lib.HB_CUT3)
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(lib.HB_SWITCH_CUT))), torch.zeros_like(arg))
        wa, wb = rbh ** 4, rah ** 4
        acid = (tt(rows["bl_ca"]) * wa + tt(rows["bl_cb"]) * wb) / (wa + wb)
        base = (tt(rows["bl_ba"]) * wb + tt(rows["bl_bb"]) * wa) / (wa + wb)
        damp = 1.0 / (1.0 + (22.0 * rr / rab ** 2) ** 6) / (1.0 + (rab ** 2 / 85.0) ** 6)
        e = e + (-5.0 * o * damp * acid * base * tt(rows["bl_qh"]) / rab ** 3).sum()
    ex = torch.zeros((), dtype=dt, device=dev)
    if rows["nxb"]:
        idx, sh = ti(rows["xb_idx"]), tt(rows["xb_sh"])
        xd, xy, xx = (_place(x, cellb, idx[:, k], sh[:, k]) for k in range(3))
        rdx, rxy, rdy = (xd - xx).norm(dim=-1), (xx - xy).norm(dim=-1), (xd - xy).norm(dim=-1)
        arg = 70.0 * ((rdx + rxy) / rdy - 1.0)
        on = (arg < lib.HB_SWITCH_CUT) & (rdy.detach() ** 2 < lib.XB_CUT)
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(lib.HB_SWITCH_CUT))), torch.zeros_like(arg))
        srt = 1.0 / (1.0 + (5.0 * tt(rows["xb_rad"]) / rxy ** 2) ** 6)
        ex = (-tt(rows["xb_p"]) * o * srt / (1.0 + (rxy ** 2 / 70.0) ** 6) / rxy ** 3).sum()
    return {"hb": e, "xb": ex}


def _extended(model, idx, sh):
    """Extended atom indices of the atoms ``idx`` in the images of lattice shifts ``sh`` (images new to the model are
    added to its extended atoms)."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    sh = np.rint(np.asarray(sh, dtype=float).reshape(-1, 3)).astype(np.int64)
    out = idx.copy()
    img = (sh[:, 0] != 0) | (sh[:, 1] != 0) | (sh[:, 2] != 0)
    if img.any():
        ii, ss = idx[img], sh[img]
        # keys in the order of (atom, shift), each shift component in [-r, r]
        r = int(np.abs(ss).max())
        w = 2 * r + 1
        keys = ((ii * w + ss[:, 0] + r) * w + ss[:, 1] + r) * w + ss[:, 2] + r
        size = (int(ii.max()) + 1) * w ** 3
        if size <= 4 * len(keys) + _DENSE:
            seen = np.zeros(size, dtype=bool)
            seen[keys] = True
            uk = np.flatnonzero(seen)
            at = np.empty(size, dtype=np.int64)
            at[uk] = np.arange(len(uk))
            inv = at[keys]
        else:
            uk, inv = np.unique(keys, return_inverse=True)
        rest = uk % w ** 3
        tk = list(zip((uk // w ** 3).tolist(), zip((rest // (w * w) - r).tolist(), ((rest // w) % w - r).tolist(),
                                                    (rest % w - r).tolist())))
        # as model._e: a new image of an atom gets the next extended index (here in key order)
        table, base = model._ext, model.num_atoms
        new = [k for k in tk if k not in table]
        table.update(zip(new, range(base + len(table), base + len(table) + len(new))))
        out[img] = np.array([table[k] for k in tk], dtype=np.int64)[inv.reshape(-1)]
    return out


@_plain_pages
def kernel_tables(model) -> dict:
    """The hydrogen- and halogen-bond rows of the bound lists in the layout of the molecular kernels, every atom an
    index of the atoms of the cell followed by the images the rows reach (images new to the model are added to its
    extended atoms)."""
    rows = model._hbxb_rows
    dev = model.rcov.device

    def li(v, w):
        return torch.as_tensor(np.asarray(v, dtype=np.int64).reshape(-1, w), dtype=torch.long, device=dev)

    def lf(*cols):
        return torch.as_tensor(np.stack([np.asarray(c, dtype=float).reshape(-1) for c in cols], 1), dtype=torch.float64,
                               device=dev).contiguous()

    def ext3(idx, sh):
        return _extended(model, np.asarray(idx).reshape(-1), np.asarray(sh).reshape(-1, 3)).reshape(-1, 3)

    nbb = rows["nbb"]
    bb = ext3(rows["bb_idx"], rows["bb_sh"])
    m = np.asarray(rows["bb_nbm"], dtype=bool)
    nbs = _extended(model, np.asarray(rows["bb_nb"])[m], np.asarray(rows["bb_nbsh"])[m])
    nbp = np.zeros(nbb + 1, dtype=np.int64)
    nbp[1:] = np.cumsum(m.sum(1))
    aux = np.zeros((nbb, 3), dtype=np.int64)
    r1 = np.asarray(rows["o1_rows"], dtype=np.int64)
    aux[r1, 0] = 1
    aux[r1, 1] = _extended(model, rows["o1_x"], rows["o1_xsh"])
    ym = np.asarray(rows["o1_ym"], dtype=bool)
    yv = _extended(model, np.asarray(rows["o1_y"])[ym], np.asarray(rows["o1_ysh"])[ym])
    ycount = np.zeros(nbb, dtype=np.int64)
    ycount[r1] = ym.sum(1)
    yp = np.zeros(nbb + 1, dtype=np.int64)
    yp[1:] = np.cumsum(ycount)
    # the y entries in row order (o1 rows are increasing)
    ys = yv
    r2 = np.asarray(rows["n2_rows"], dtype=np.int64)
    aux[r2, 0] = 2
    aux[r2, 1:] = _extended(model, np.asarray(rows["n2_nb"]).reshape(-1),
                            np.asarray(rows["n2_nbsh"]).reshape(-1, 3)).reshape(-1, 2)
    bl = ext3(rows["bl_idx"], rows["bl_sh"])
    xb = ext3(rows["xb_idx"], rows["xb_sh"])
    # the images the rows added
    keys = list(model._ext)  # in the order of their extended indices
    model._set("x_base", [k[0] for k in keys], torch.long)
    model._set("x_shift", np.array([k[1] for k in keys], dtype=float).reshape(-1, 3))
    return {"bb": li(bb, 3), "bb_par": lf(rows["bb_rr"], rows["bb_p"], rows["bb_cnb"]), "nbp": li(nbp, 1).reshape(-1),
            "nbs": li(nbs, 1).reshape(-1), "aux": li(aux, 3), "yp": li(yp, 1).reshape(-1), "ys": li(ys, 1).reshape(-1),
            "bl": li(bl, 3), "bl_par": lf(rows["bl_rr"], rows["bl_ca"], rows["bl_cb"], rows["bl_ba"], rows["bl_bb"],
                                          rows["bl_qh"]),
            "xb": li(xb, 3), "xb_par": lf(rows["xb_rad"], rows["xb_p"])}


@_plain_pages
def repulsion_rows(t):
    """The repulsion pairs of a periodic topology over the images of its translation list (each unordered pair once,
    an atom with its own images counted from both sides): atoms ``i`` and ``j`` of the cell, the lattice shift of
    ``j``, prefactors and exponents."""
    z = np.asarray(t.z)
    n = len(z)
    nn = np.array([len(v) for v in t.lig])
    expo = lib.PAR[z - 1, 6] * (1.0 - 0.127 / (1.0 + nn ** 2)) * (1.0 + 0.348 * t.qt)
    val = np.array([lib.VALENCE[int(v)] for v in z], dtype=float)
    x, cell = t.x.x, t.cell
    cut = lib.repulsion_reach(t)
    allowed = set(lib.translation_list(cell, lib.TRANS_ENERGY2, cap=True))
    ii, jj, ss = [], [], []
    shs = np.array([sh for sh in lib.shifts_within(cell, cut) if sh in allowed], dtype=np.int64).reshape(-1, 3)
    # positive shifts (lexicographic): an atom pairs with its own image from that side only
    pos = (shs[:, 0] > 0) | ((shs[:, 0] == 0) & ((shs[:, 1] > 0) | ((shs[:, 1] == 0) & (shs[:, 2] > 0))))
    tb = max(1, (1 << 20) // max(1, n * n))  # the translations in blocks of 2^20 / n^2 shifts
    tr = [shs[lo:lo + tb].astype(float) @ cell for lo in range(0, len(shs), tb)]
    tr = np.concatenate(tr) if tr else np.zeros((0, 3))
    # the pairs i <= j of the cell (the others are the same pairs from the other side), in blocks of shifts
    iu, ju = np.triu_indices(n)
    xi, xj = x[iu], x[ju]
    step = max(1, _BLOCK // max(1, 3 * len(iu)))  # small temporaries (cache-sized blocks of shifts)
    for lo in range(0, len(shs), step):
        v = xj[None, :, :] + tr[lo:lo + step, None, :] - xi[None, :, :]
        d = np.sqrt(np.add.reduce(v * v, axis=-1))  # as np.linalg.norm
        k, p = np.nonzero(d < cut)
        i, j = iu[p], ju[p]
        keep = (i < j) | pos[lo + k]
        ii.append(i[keep])
        jj.append(j[keep])
        ss.append(shs[lo + k[keep]])
    i = np.concatenate(ii) if ii else np.zeros(0, np.int64)
    j = np.concatenate(jj) if jj else np.zeros(0, np.int64)
    s = np.concatenate(ss) if ss else np.zeros((0, 3), np.int64)
    zi, zj = z[i], z[j]
    zz = val[i] * val[j]
    base = np.sqrt(expo[i] * expo[j])
    both, one = (zi == 1) & (zj == 1), (zi == 1) != (zj == 1)
    other = np.where(zi == 1, zj, zi)
    fac = np.where(both, 0.629, 1.0)
    fac = np.where(one & np.isin(other, sorted(lib.H_METAL)), 0.85, fac)
    fac = np.where(one & (other == 6), 0.91, fac)
    fac = np.where(one & (other == 8), 1.04, fac)
    c, m = 0.427 * zz, base * fac
    # bond counts of the pairs within three bonds
    ha, hb, hs, hd = _hop_table(t)
    d = np.zeros(len(i), dtype=np.int64)
    if len(ha):
        keys = _keys(ha, hb, hs, n)
        order = np.argsort(keys)
        keys, hd = keys[order], hd[order]
        pk = _keys(i, j, s, n)
        at = np.clip(np.searchsorted(keys, pk), 0, len(keys) - 1)
        hit = keys[at] == pk
        d[hit] = hd[at[hit]]
    one_bond = d == 1
    c = np.where(one_bond, 1.7583 * zz, c)
    m = np.where(one_bond, np.sqrt(lib.PAR[zi - 1, 5] * lib.PAR[zj - 1, 5]), m)
    m = np.where(both & ((d == 2) | (d == 3)), base * np.where(d == 2, 0.917082, 0.445332), m)
    # an atom and its own image enter from both sides
    c = np.where((i == j) & ~one_bond, 2.0 * c, c)
    return i, j, s, c, m
