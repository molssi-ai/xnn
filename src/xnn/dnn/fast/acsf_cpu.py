"""CPU kernels of the HDNNP symmetry functions (C++ and OpenMP, built on first use).

RuNNer evaluates the symmetry functions in loops over the atoms, their neighbors and their neighbor
pairs, without storing anything per triplet. These kernels do the same: one OpenMP loop over the
centers, each walking its own edges (radial functions) and the pairs of its edges inside the angular
cutoff (angular functions), evaluating only the functions of its element and the pair's neighbor
species (the lane tables of :func:`xnn.dnn.fast.acsf_kernels.tables`) and summing them into its own
row. The derivative with respect to the edge vectors (the forces) walks the same loops and adds into
the center's own edges, so no two threads write the same memory. Types 1, 2, 3 and 9 with at most two
cosine or tanh cutoffs, as the GPU kernels; the derivative is not differentiable again, so the
kernels serve evaluation.

The source is compiled with :func:`torch.utils.cpp_extension.load_inline` on first use (a C++
compiler with OpenMP is needed; :func:`available` reports whether the build works) and kept in
PyTorch's extension cache.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_SOURCE = r"""
#include <torch/extension.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T> struct Cut { int kind; T rc, rin, inv, pinv; };

template <typename T>
static inline void cutoff(T r, const Cut<T>& c, T& f, T& d) {
  if (r >= c.rc) { f = T(0); d = T(0); return; }
  T x = (r - c.rin) * c.inv;
  bool moving = r > c.rin;
  if (!moving) x = T(0);
  if (c.kind == 1) {
    T t = std::tanh(T(1) - x);
    f = t * t * t;
    d = moving ? T(-3) * t * t * (T(1) - t * t) * c.inv : T(0);
  } else {
    T a = moving ? c.pinv * (r - c.rin) : T(0);
    f = T(0.5) * std::cos(a) + T(0.5);
    d = moving ? T(-0.5) * c.pinv * std::sin(a) : T(0);
  }
}

template <typename T>
static inline T power(T base, T zeta) {
  // integer exponents (the usual 1, 2, 4, ... 16) by repeated squaring, others by pow
  if (zeta == std::floor(zeta) && zeta >= T(0) && zeta <= T(64)) {
    long k = static_cast<long>(zeta);
    T out = T(1), b = base;
    while (k) { if (k & 1) out *= b; b *= b; k >>= 1; }
    return out;
  }
  return std::pow(base, zeta);
}

template <typename T>
static void run(const T* vec, const int64_t* ptr, const int32_t* sn, const int32_t* sp, int64_t n,
                const int32_t* rcol, const int32_t* rtyp, const T* reta, const T* rrs, const T* rpw,
                const int32_t* rci, int64_t kr,
                const int32_t* acol, const int32_t* atyp, const T* aeta, const T* alam, const T* azeta,
                const T* apref, const int32_t* aci, int64_t ka,
                const int32_t* aes, const int32_t* aps, const T* rek, const T* rpk, int64_t ke, int64_t kp,
                const int32_t* has3,
                const Cut<T>* cuts, int64_t ncut, int64_t n_sp, int64_t d, T ang_cut,
                T* out, const T* w, T* grad) {
  const int64_t n_pairs = n_sp * (n_sp + 1) / 2;
  const bool backward = grad != nullptr;
#pragma omp parallel for schedule(dynamic, 4)
  for (int64_t i = 0; i < n; ++i) {
    const int64_t sc = sp[i];
    const int64_t e0 = ptr[i], e1 = ptr[i + 1];
    std::vector<T> rr(e1 - e0), fe(2 * (e1 - e0)), de(2 * (e1 - e0));
    for (int64_t e = e0; e < e1; ++e) {
      const T* v = vec + 3 * e;
      const T r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      rr[e - e0] = r;
      // the cutoff values of every edge, once (the radial terms and both sides of every triplet use them)
      for (int64_t c = 0; c < ncut; ++c) cutoff(r, cuts[c], fe[2 * (e - e0) + c], de[2 * (e - e0) + c]);
    }
    T* gi = out ? out + i * d : nullptr;
    const T* wi = w ? w + i * d : nullptr;
    // radial
    if (kr > 0) {
      for (int64_t e = e0; e < e1; ++e) {
        const T r = rr[e - e0];
        const T* fc = &fe[2 * (e - e0)];
        const T* dc = &de[2 * (e - e0)];
        const int64_t row = (sc * n_sp + sn[e]) * kr;
        T dr = T(0);
        for (int64_t l = 0; l < kr; ++l) {
          const int col = rcol[row + l];
          if (col < 0) continue;
          const int ci = rci[row + l];
          const T f = fc[ci], fd = dc[ci];
          T g, g1;
          if (rtyp[row + l] == 1) {
            const T p = rpw[row + l];
            g = f > T(0) ? std::pow(f, p) : T(0);
            g1 = f > T(0) ? p * g / f * fd : T(0);
          } else {
            const T u = r - rrs[row + l], eta = reta[row + l];
            const T ex = std::exp(-eta * u * u);
            g = ex * f;
            g1 = ex * (fd - T(2) * eta * u * f);
          }
          if (backward) dr += wi[col] * g1; else gi[col] += g;
        }
        if (backward && r > T(0)) {
          const T* v = vec + 3 * e;
          T* gv = grad + 3 * e;
          for (int k = 0; k < 3; ++k) gv[k] += dr * v[k] / r;
        }
      }
    }
    // angular
    if (ka > 0) {
      for (int64_t a = e0; a < e1; ++a) {
        const T ra = rr[a - e0];
        if (!(ra < ang_cut)) continue;
        const T* va = vec + 3 * a;
        const T* fa = &fe[2 * (a - e0)];
        const T* da = &de[2 * (a - e0)];
        for (int64_t b = a + 1; b < e1; ++b) {
          const T rb = rr[b - e0];
          if (!(rb < ang_cut)) continue;
          const T* vb = vec + 3 * b;
          const T dx = va[0] - vb[0], dy = va[1] - vb[1], dz = va[2] - vb[2];
          const T r3 = std::sqrt(dx * dx + dy * dy + dz * dz);
          const T cs = (va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]) / (ra * rb);
          const T* fb = &fe[2 * (b - e0)];
          const T* db = &de[2 * (b - e0)];
          int64_t lo = sn[a], hi = sn[b];
          if (lo > hi) { int64_t t = lo; lo = hi; hi = t; }
          const int64_t pair = lo * n_sp - lo * (lo - 1) / 2 + (hi - lo);
          const int64_t rowi = sc * n_pairs + pair;
          // the third distance's cutoff only for rows with type-3 functions
          T f3[2] = {T(1), T(1)}, d3[2] = {T(0), T(0)};
          if (has3[rowi])
            for (int64_t c = 0; c < ncut; ++c) cutoff(r3, cuts[c], f3[c], d3[c]);
          const int64_t row = rowi * ka;
          // the distinct exponentials (eta, type) and powers (lambda, zeta) of the row, once per triplet
          T ev[16], pv[16];
          const T s2 = ra * ra + rb * rb, s3 = s2 + r3 * r3;
          for (int64_t k = 0; k < ke; ++k) {
            const T* key = rek + (rowi * ke + k) * 2;
            ev[k] = key[0] < T(0) ? T(0) : std::exp(-key[0] * (key[1] > T(0) ? s3 : s2));
          }
          for (int64_t k = 0; k < kp; ++k) {
            const T* key = rpk + (rowi * kp + k) * 2;
            const T base = T(1) + key[0] * cs;
            pv[k] = base > T(0) ? power(base, key[1]) : T(0);
          }
          T g1s = T(0), g2s = T(0), g3s = T(0), gcs = T(0);
          for (int64_t l = 0; l < ka; ++l) {
            const int col = acol[row + l];
            if (col < 0) continue;
            const int ci = aci[row + l];
            const bool three = atyp[row + l] == 3;
            const T eta = aeta[row + l], lam = alam[row + l], zeta = azeta[row + l];
            const T base = T(1) + lam * cs;
            if (!(base > T(0))) continue;
            const T ex = ev[aes[row + l]];
            const T pw = pv[aps[row + l]];
            const T ae = apref[row + l] * pw * ex;
            const T F1 = fa[ci], F2 = fb[ci], F3 = three ? f3[ci] : T(1);
            if (!backward) { gi[col] += ae * F1 * F2 * F3; continue; }
            const T wl = wi[col];
            if (wl == T(0)) continue;
            const T q1 = da[ci] - T(2) * eta * ra * F1;
            const T q2 = db[ci] - T(2) * eta * rb * F2;
            const T q3 = three ? d3[ci] - T(2) * eta * r3 * F3 : T(0);
            g1s += wl * ae * q1 * F2 * F3;
            g2s += wl * ae * F1 * q2 * F3;
            g3s += wl * ae * F1 * F2 * q3;
            gcs += wl * zeta * lam / base * ae * F1 * F2 * F3;
          }
          if (backward) {
            T* ga = grad + 3 * a;
            T* gb = grad + 3 * b;
            const T dd[3] = {dx, dy, dz};
            for (int k = 0; k < 3; ++k) {
              const T dca = vb[k] / (ra * rb) - cs * va[k] / (ra * ra);
              const T dcb = va[k] / (ra * rb) - cs * vb[k] / (rb * rb);
              const T t3 = r3 > T(0) ? g3s * dd[k] / r3 : T(0);
              ga[k] += g1s * va[k] / ra + t3 + gcs * dca;
              gb[k] += g2s * vb[k] / rb - t3 + gcs * dcb;
            }
          }
        }
      }
    }
  }
}

template <typename T>
static std::vector<Cut<T>> cuts_of(const torch::Tensor& cut, int64_t ncut) {
  auto c = cut.contiguous();
  const T* p = c.data_ptr<T>();
  std::vector<Cut<T>> out(ncut);
  for (int64_t k = 0; k < ncut; ++k)
    out[k] = Cut<T>{static_cast<int>(p[5 * k]), p[5 * k + 1], p[5 * k + 2], p[5 * k + 3], p[5 * k + 4]};
  return out;
}

torch::Tensor apply(torch::Tensor vec, torch::Tensor ptr, torch::Tensor sn, torch::Tensor sp,
                    std::vector<torch::Tensor> rad, std::vector<torch::Tensor> ang, torch::Tensor cut, int64_t ncut,
                    int64_t n_sp, int64_t d, double ang_cut, c10::optional<torch::Tensor> w) {
  const int64_t n = sp.size(0);
  const bool backward = w.has_value();
  torch::Tensor out = backward ? torch::zeros_like(vec) : torch::zeros({n, d}, vec.options());
  const int64_t kr = rad.empty() ? 0 : rad[0].size(0) / (n_sp * n_sp);
  const int64_t n_rows_a = n_sp * n_sp * (n_sp + 1) / 2;
  const int64_t ka = ang.empty() ? 0 : ang[0].size(0) / n_rows_a;
  const int64_t ke = ang.empty() ? 0 : ang[9].size(0) / (2 * n_rows_a);
  const int64_t kp = ang.empty() ? 0 : ang[10].size(0) / (2 * n_rows_a);
  AT_DISPATCH_FLOATING_TYPES(vec.scalar_type(), "acsf_cpu", [&] {
    auto cuts = cuts_of<scalar_t>(cut, ncut);
    const scalar_t* wp = backward ? w->data_ptr<scalar_t>() : nullptr;
    run<scalar_t>(vec.data_ptr<scalar_t>(), ptr.data_ptr<int64_t>(), sn.data_ptr<int32_t>(), sp.data_ptr<int32_t>(), n,
                  kr ? rad[0].data_ptr<int32_t>() : nullptr, kr ? rad[1].data_ptr<int32_t>() : nullptr,
                  kr ? rad[2].data_ptr<scalar_t>() : nullptr, kr ? rad[3].data_ptr<scalar_t>() : nullptr,
                  kr ? rad[4].data_ptr<scalar_t>() : nullptr, kr ? rad[5].data_ptr<int32_t>() : nullptr, kr,
                  ka ? ang[0].data_ptr<int32_t>() : nullptr, ka ? ang[1].data_ptr<int32_t>() : nullptr,
                  ka ? ang[2].data_ptr<scalar_t>() : nullptr, ka ? ang[3].data_ptr<scalar_t>() : nullptr,
                  ka ? ang[4].data_ptr<scalar_t>() : nullptr, ka ? ang[5].data_ptr<scalar_t>() : nullptr,
                  ka ? ang[6].data_ptr<int32_t>() : nullptr, ka,
                  ka ? ang[7].data_ptr<int32_t>() : nullptr, ka ? ang[8].data_ptr<int32_t>() : nullptr,
                  ka ? ang[9].data_ptr<scalar_t>() : nullptr, ka ? ang[10].data_ptr<scalar_t>() : nullptr, ke, kp,
                  ka ? ang[11].data_ptr<int32_t>() : nullptr,
                  cuts.data(), ncut, n_sp, d, static_cast<scalar_t>(ang_cut),
                  backward ? nullptr : out.data_ptr<scalar_t>(), wp, backward ? out.data_ptr<scalar_t>() : nullptr);
  });
  return out;
}
"""

def _module():
    from xnn.common.models.cpu_extension import load
    return load("xnn_acsf_cpu", _SOURCE, ["apply"])


def available() -> bool:
    """Whether the C++ kernels build here (checked once)."""
    return _module() is not None


def _slots(ang: list, n_sp: int) -> list:
    """Per angular lane the slot of its exponential ``(eta, type)`` and power ``(lambda, zeta)`` among the
    row's distinct ones, per row those keys (``(rows, KE, 2)`` and ``(rows, KP, 2)``, padded with
    ``eta = -1``) and whether the row has type-3 functions."""
    col, typ, eta, lam, zeta = ang[0], ang[1], ang[2], ang[3], ang[4]
    n_rows = n_sp * n_sp * (n_sp + 1) // 2
    k = col.numel() // n_rows
    es = torch.zeros_like(col)
    ps = torch.zeros_like(col)
    e_rows, p_rows = [], []
    for r in range(n_rows):
        ek, pk = [], []
        for j in range(r * k, (r + 1) * k):
            if int(col[j]) < 0:
                continue
            e = (float(eta[j]), 1.0 if int(typ[j]) == 3 else 0.0)
            p = (float(lam[j]), float(zeta[j]))
            if e not in ek:
                ek.append(e)
            if p not in pk:
                pk.append(p)
            es[j], ps[j] = ek.index(e), pk.index(p)
        e_rows.append(ek)
        p_rows.append(pk)
    ke = max(1, max(len(x) for x in e_rows))
    kp = max(1, max(len(x) for x in p_rows))
    if ke > 16 or kp > 16:
        raise ValueError("more than 16 distinct exponentials or powers per element pair")
    rek = torch.full((n_rows, ke, 2), -1.0, dtype=eta.dtype)
    rpk = torch.zeros((n_rows, kp, 2), dtype=eta.dtype)
    for r in range(n_rows):
        for j, e in enumerate(e_rows[r]):
            rek[r, j] = torch.tensor(e, dtype=eta.dtype)
        for j, p in enumerate(p_rows[r]):
            rpk[r, j] = torch.tensor(p, dtype=eta.dtype)
    has3 = ((typ.reshape(n_rows, k) == 3) & (col.reshape(n_rows, k) >= 0)).any(dim=1).to(torch.int32)
    return [es.contiguous(), ps.contiguous(), rek.reshape(-1).contiguous(), rpk.reshape(-1).contiguous(),
            has3.contiguous()]


class _Tables:
    """The radial and angular lane tables as CPU tensors in the kernels' layout."""

    def __init__(self, rad, ang, n_sp: int):
        def lanes(tab) -> list:
            if tab is None:
                return []
            arrays = tab.arrays[:-1]
            return [a.contiguous() for a in arrays]
        self.rad, self.ang = lanes(rad), lanes(ang)
        if self.ang:
            self.ang += _slots(self.ang, n_sp)
        tab = rad if rad is not None else ang
        self.cut, self.ncut, self.n_sp = tab.arrays[-1].contiguous(), tab.nc, n_sp


class ACSFCpuFunction(torch.autograd.Function):
    """The symmetry functions ``(N, D)`` from the edge vectors in center order (``ptr`` the centers' edge
    ranges, ``sn`` each edge's neighbor species, ``sp`` the atoms' species); the gradient is not
    differentiable again."""

    @staticmethod
    def forward(ctx, vec, ptr, sn, sp, tables: _Tables, d: int, ang_cut: float):
        vec = vec.contiguous()
        ctx.save_for_backward(vec)
        ctx.args = (ptr, sn, sp, tables, d, ang_cut)
        return _module().apply(vec, ptr, sn, sp, tables.rad, tables.ang, tables.cut, tables.ncut, tables.n_sp, d,
                               ang_cut, None)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, w):
        (vec,) = ctx.saved_tensors
        ptr, sn, sp, tables, d, ang_cut = ctx.args
        g = _module().apply(vec, ptr, sn, sp, tables.rad, tables.ang, tables.cut, tables.ncut, tables.n_sp, d,
                            ang_cut, w.contiguous())
        return g, None, None, None, None, None, None


def symmetry_functions(vec: Tensor, edge_index: Tensor, sp: Tensor, n: int, tables: Optional[_Tables], d: int,
                       ang_cut: float) -> Tensor:
    """The raw symmetry functions ``(n, d)`` of a graph on the C++ kernels (``sp`` the atoms' species
    indices; ``tables`` from :class:`_Tables`)."""
    dst = edge_index[1]
    order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=n)
    ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    sn = sp[edge_index[0]].index_select(0, order).to(torch.int32).contiguous()
    return ACSFCpuFunction.apply(vec.index_select(0, order), ptr, sn, sp.to(torch.int32).contiguous(), tables, d,
                                 float(ang_cut))
