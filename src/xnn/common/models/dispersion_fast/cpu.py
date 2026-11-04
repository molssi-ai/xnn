"""D3 and D4 on CPUs: C++ and OpenMP kernels (built on first use).

simple-dftd3 and dftd4 evaluate the dispersion energy in loops over the atoms and their neighbors,
energy and gradient in one pass, without tensors per pair or triangle. The reference
(:class:`~xnn.common.models.d3.DFTD3`, :class:`~xnn.common.models.d4.DFTD4`) evaluates the same
formulas as tensor operations; on a CPU its many operations over gathered copies take longer than
those loops. These kernels do what the Fortran codes do:

* the edges are sorted by center, then neighbor (a counting sort and a sort per center);
* D3: one loop over the centers forms the coordination numbers, the Gaussian reference weights and
  their derivatives, and with a three-body term the pair C6 and their derivatives with respect to
  both coordination numbers fill dense ``(N, N)`` arrays inside each structure;
* one loop over the centers sums the two-body energies of their edges (D3: C6 from the weights;
  D4: from the dynamic polarizabilities), and one sums the three-body energies of their neighbor
  pairs: a triangle of three distinct atoms once, from its smallest-index corner, with a third of
  its energy to each atom (the reference's bookkeeping), and a triangle with a repeated atom
  (periodic images) from every corner, a third each;
* the gradient with respect to the edge vectors adds into the center's own edges (no two threads
  write the same memory); the derivatives through the C6 collect per thread on the coordination
  numbers. D3 carries them to the edges in a last loop over the centers; D4 returns them, with
  the derivatives in the dynamic polarizabilities, to the charges, weights and coordination numbers
  that the reference code forms.

The forward pass forms the gradient for unit weights on every atom's energy, which is what a sum
of energies (forces, stress) asks for; another weighting evaluates again with those weights. The
gradient is not differentiable again, so the kernels serve evaluation. They compute in float64 for
float32 inputs as well (scalar code is no slower in it, and the sums of a periodic three-body term
cancel to a few digits). The source is compiled with :func:`torch.utils.cpp_extension.load_inline`
on first use (:mod:`xnn.common.models.cpu_extension`).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from ..cpu_extension import load

#: Atom count up to which the three-body term runs here: it reads the pair C6 and their derivatives
#: from dense ``(N, N)`` arrays (four of them, 2 GB at this size).
MAX_DENSE_ATOMS = 8192

_DAMPINGS = {"bj": 0, "zero": 1, "mzero": 2, "op": 3}

_SOURCE = r"""
#include <torch/extension.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using T = double;

static inline int n_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static inline int thread_id() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

struct Prm {
  int damping;
  T s6, s8, s9, a1, a2, rs6, rs8, bet, alp, rs9, kcn, wf, alp3;
  T cut_pair, cut_triple, cut_cn, w_pair, w_triple, bohr, hartree;
};

// the quintic switch of dispersion.switching_function and its slope in r
static inline void switching(T r, T cutoff, T width, T& s, T& ds) {
  if (width <= T(0)) { s = T(1); ds = T(0); return; }
  const T w = std::min(width, cutoff);
  const T x = (cutoff - r) / w;
  if (x <= T(0)) { s = T(0); ds = T(0); return; }
  if (x >= T(1)) { s = T(1); ds = T(0); return; }
  s = x * x * x * (T(10) + x * (T(6) * x - T(15)));
  ds = -T(30) * x * x * (T(1) - x) * (T(1) - x) / w;
}

// s6 t6 + s8 rr t8 of the selected damping and its slope in r
static inline void damped(const Prm& p, T r, T r0, T rr, T& g, T& dg) {
  const T r2 = r * r, r6 = r2 * r2 * r2, r8 = r6 * r2;
  T t6, t8, d6, d8;
  if (p.damping == 0) {
    const T q2 = r0 * r0, q6 = q2 * q2 * q2, q8 = q6 * q2;
    t6 = T(1) / (r6 + q6);
    t8 = T(1) / (r8 + q8);
    d6 = -T(6) * r6 / r * t6 * t6;
    d8 = -T(8) * r8 / r * t8 * t8;
  } else if (p.damping == 1) {
    const T f6 = T(6) * std::pow(p.rs6 * r0 / r, p.alp), f8 = T(6) * std::pow(p.rs8 * r0 / r, p.alp + T(2));
    t6 = T(1) / (T(1) + f6) / r6;
    t8 = T(1) / (T(1) + f8) / r8;
    d6 = t6 * (p.alp * f6 / (T(1) + f6) - T(6)) / r;
    d8 = t8 * ((p.alp + T(2)) * f8 / (T(1) + f8) - T(8)) / r;
  } else if (p.damping == 2) {
    const T u6 = r / (p.rs6 * r0) + p.bet * r0, u8 = r / (p.rs8 * r0) + p.bet * r0;
    const T f6 = T(6) * std::pow(u6, -p.alp), f8 = T(6) * std::pow(u8, -(p.alp + T(2)));
    t6 = T(1) / (T(1) + f6) / r6;
    t8 = T(1) / (T(1) + f8) / r8;
    d6 = t6 * (p.alp * f6 / (u6 * p.rs6 * r0 * (T(1) + f6)) - T(6) / r);
    d8 = t8 * ((p.alp + T(2)) * f8 / (u8 * p.rs8 * r0 * (T(1) + f8)) - T(8) / r);
  } else {
    const T rb = std::pow(r, p.bet), ab = std::pow(r0, p.bet), drb = p.bet * rb / r;
    const T q2 = r0 * r0, q6 = q2 * q2 * q2, q8 = q6 * q2;
    const T D6 = rb * r6 + ab * q6, D8 = rb * r8 + ab * q8;
    t6 = rb / D6;
    t8 = rb / D8;
    d6 = (drb * D6 - rb * (drb * r6 + rb * T(6) * r6 / r)) / (D6 * D6);
    d8 = (drb * D8 - rb * (drb * r8 + rb * T(8) * r8 / r)) / (D8 * D8);
  }
  g = p.s6 * t6 + p.s8 * rr * t8;
  dg = p.s6 * d6 + p.s8 * rr * d8;
}

// Gaussian reference weights (seven slots) and their derivatives in the coordination number (D3)
static inline void ref_weights(T cn, int64_t zz, const T* refcn, const int64_t* nref, int64_t nz, T wf,
                               T* w, T* dw) {
  const int nr = static_cast<int>(nref[zz]);
  T gw[7], dg[7];
  T norm = T(0), dsum = T(0);
  for (int a = 0; a < 7; ++a) { w[a] = T(0); dw[a] = T(0); }
  for (int a = 0; a < nr; ++a) {
    const T d = cn - refcn[a * nz + zz];
    gw[a] = std::exp(-wf * (d * d));
    dg[a] = -T(2) * wf * d * gw[a];
    norm += gw[a];
    dsum += dg[a];
  }
  if (norm > T(0)) {
    for (int a = 0; a < nr; ++a) {
      w[a] = gw[a] / norm;
      dw[a] = (dg[a] - w[a] * dsum) / norm;
    }
  } else {
    T top = -std::numeric_limits<T>::infinity();
    for (int a = 0; a < nr; ++a) top = std::max(top, refcn[a * nz + zz]);
    for (int a = 0; a < nr; ++a) w[a] = std::abs(refcn[a * nz + zz] - top) < T(1e-12) ? T(1) : T(0);
  }
}

// C6 of a pair from the weights, with its derivatives in the first and second coordination number (D3)
static inline void pair_c6(const T* ref, const T* wi, const T* dwi, int ni, const T* wj, const T* dwj, int nj,
                           T& c6, T& di, T& dj) {
  c6 = T(0); di = T(0); dj = T(0);
  for (int b = 0; b < nj; ++b) {
    T v = T(0), dv = T(0);
    for (int a = 0; a < ni; ++a) {
      v += wi[a] * ref[a * 7 + b];
      dv += dwi[a] * ref[a * 7 + b];
    }
    c6 += wj[b] * v;
    di += wj[b] * dv;
    dj += dwj[b] * v;
  }
}

// x^(-1/3) for x > 0: a first guess from the exponent bits, then four Newton steps (3 ulp)
static inline T inv_cbrt(T x) {
  uint64_t i;
  std::memcpy(&i, &x, sizeof i);
  i = 0x553ef00000000000ULL - i / 3;
  T y;
  std::memcpy(&y, &i, sizeof y);
  for (int k = 0; k < 4; ++k) y = y * (T(4) - x * y * y * y) * (T(1) / T(3));
  return y;
}

// the edges by center, then neighbor, in bohr
struct Edges {
  int64_t n, ne;
  bool par;
  std::vector<int64_t> ptr, order, js;
  std::vector<T> v, r;
};

static Edges sorted_edges(const T* vec, const int64_t* src, const int64_t* dst, int64_t n, int64_t ne, T bohr) {
  Edges E;
  E.n = n; E.ne = ne; E.par = ne >= 2048;
  E.ptr.assign(n + 1, 0);
  E.order.resize(ne);
  for (int64_t e = 0; e < ne; ++e) E.ptr[dst[e] + 1] += 1;
  for (int64_t c = 0; c < n; ++c) E.ptr[c + 1] += E.ptr[c];
  {
    std::vector<int64_t> fill(E.ptr.begin(), E.ptr.end() - 1);
    for (int64_t e = 0; e < ne; ++e) E.order[fill[dst[e]]++] = e;
  }
#pragma omp parallel for schedule(dynamic, 16) if (E.par)
  for (int64_t c = 0; c < n; ++c) {
    auto lo = E.order.begin() + E.ptr[c], hi = E.order.begin() + E.ptr[c + 1];
    auto less = [&](int64_t a, int64_t b) { return src[a] < src[b] || (src[a] == src[b] && a < b); };
    if (!std::is_sorted(lo, hi, less)) std::sort(lo, hi, less);
  }
  E.v.resize(3 * ne); E.r.resize(ne); E.js.resize(ne);
#pragma omp parallel for schedule(static) if (E.par)
  for (int64_t k = 0; k < ne; ++k) {
    const int64_t e = E.order[k];
    const T x = vec[3 * e] / bohr, y = vec[3 * e + 1] / bohr, w = vec[3 * e + 2] / bohr;
    E.v[3 * k] = x; E.v[3 * k + 1] = y; E.v[3 * k + 2] = w;
    E.r[k] = std::sqrt(x * x + y * y + w * w);
    E.js[k] = src[e];
  }
  return E;
}

// the gradient of the sorted edges (bohr) into the input order (Angstrom, eV)
static void scatter_gradient(const Edges& E, const std::vector<T>& gb, T scale, T* grad) {
#pragma omp parallel for schedule(static) if (E.par)
  for (int64_t k = 0; k < E.ne; ++k) {
    const int64_t e = E.order[k];
    grad[3 * e] = gb[3 * k] * scale; grad[3 * e + 1] = gb[3 * k + 1] * scale; grad[3 * e + 2] = gb[3 * k + 2] * scale;
  }
}

struct Loc {
  int64_t k, j, sj;
  T vx, vy, vz, r2, r, sw, dsw, c6, sq, dcc, dcj, r0, r0p, q, wj;
  T gx, gy, gz, m, ej, dn;
};

// the sums of one neighbor over its partners in the inner loop, and its rows of the dense arrays
struct Acc {
  T ec = T(0), ej = T(0), gx = T(0), gy = T(0), gz = T(0), m = T(0), dn = T(0);
  const T *c6row = nullptr, *dmrow = nullptr, *dnrow = nullptr, *qrow = nullptr;
};

// the three-body (ATM) energies (hartree) into the per-thread rows e3_t and, with GRAD, the gradient into gb
// and the coordination-number derivatives into the per-thread rows dcn_t; s6m[i, j] = sqrt(C6_ij), dc6m[i, j] = d C6_ij /
// d CN_i, dc6n[i, j] = d C6_ij / d CN_j, the pair radii scale * r0tab[Z_i, Z_j]
template <bool GRAD>
static void atm(const Edges& E, const int64_t* z, int64_t nz, const T* s6m, const T* dc6m, const T* dc6n,
                const T* r0tab, T scale, T s9, T alp3, T cut, T width, const T* w3, T* e3_t, T* dcn_t, T* gb) {
  const int64_t n = E.n;
  const T cut2 = cut * cut, eps = T(2.220446049250313e-16);
  const bool sw_on = width > T(0);
  // the damping power (r0 / sqrt(r2ij r2ik r2jk))^alp3: for alp3 = k / 3 an inverse cube root and integer
  // powers, with the radii powers per species pair
  const T k3r = std::round(T(3) * alp3);
  const bool thirds = std::abs(T(3) * alp3 - k3r) < T(1e-9) && k3r > T(0) && k3r < T(64);
  const int k3 = static_cast<int>(k3r);
  const bool whole = thirds && k3 % 3 == 0;
  const int kp = whole ? k3 / 3 : k3;
  std::vector<int64_t> spec_of(nz, -1), sidx(n), species;
  for (int64_t i = 0; i < n; ++i) {
    if (spec_of[z[i]] < 0) { spec_of[z[i]] = static_cast<int64_t>(species.size()); species.push_back(z[i]); }
    sidx[i] = spec_of[z[i]];
  }
  const int64_t ns = static_cast<int64_t>(species.size());
  std::vector<T> r0s(ns * ns), r0ps(ns * ns);
  for (int64_t a = 0; a < ns; ++a)
    for (int64_t b = 0; b < ns; ++b) {
      r0s[a * ns + b] = scale * r0tab[species[a] * nz + species[b]];
      r0ps[a * ns + b] = thirds ? std::pow(r0s[a * ns + b], alp3) : T(0);
    }
  auto ipow = [](T x, int k) {
    T out = T(1);
    while (k > 0) { if (k & 1) out *= x; x *= x; k >>= 1; }
    return out;
  };
  // without two images of one atom pair inside the cutoff (molecules, large cells) every side of a triangle is
  // an edge, and the damping power splits into per-edge factors (R0 / r)^alp3
  int unique = 1;
#pragma omp parallel for schedule(static) reduction(min : unique) if (E.par)
  for (int64_t c = 0; c < n; ++c) {
    int64_t prev = -1;
    for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k) {
      if (E.r[k] > cut) continue;
      if (E.js[k] == c || E.js[k] == prev) unique = 0;
      prev = E.js[k];
    }
  }
  std::vector<T> qm;
  if (unique) {
    qm.assign(static_cast<size_t>(n) * n, T(0));
#pragma omp parallel for schedule(static) if (E.par)
    for (int64_t c = 0; c < n; ++c)
      for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k)
        if (E.r[k] <= cut) {
          const int64_t sp = sidx[c] * ns + sidx[E.js[k]];
          qm[c * n + E.js[k]] = thirds ? r0ps[sp] * ipow(whole ? T(1) / E.r[k] : inv_cbrt(E.r[k]), whole ? kp : k3)
                                       : std::pow(r0s[sp] / E.r[k], alp3);
        }
  }
#pragma omp parallel if (E.par)
  {
    const size_t t = static_cast<size_t>(thread_id());
    T* dcn = GRAD ? dcn_t + t * n : nullptr;
    T* e3l = e3_t + t * n;
    std::vector<Loc> loc;
#pragma omp for schedule(dynamic, 1)
    for (int64_t c = 0; c < n; ++c) {
      const T wc = w3 ? w3[c] : T(1);
      loc.clear();
      for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k) {
        if (E.r[k] > cut) continue;
        Loc l;
        l.k = k; l.j = E.js[k]; l.sj = sidx[l.j];
        l.vx = E.v[3 * k]; l.vy = E.v[3 * k + 1]; l.vz = E.v[3 * k + 2];
        l.r = E.r[k]; l.r2 = E.r[k] * E.r[k];
        switching(E.r[k], cut, width, l.sw, l.dsw);
        l.sq = s6m[c * n + l.j]; l.c6 = l.sq * l.sq; l.dcc = dc6m[c * n + l.j]; l.dcj = dc6n[c * n + l.j];
        l.r0 = r0s[sidx[c] * ns + l.sj];
        l.r0p = r0ps[sidx[c] * ns + l.sj];
        l.q = unique ? qm[c * n + l.j] : T(0);
        l.wj = w3 ? w3[l.j] : T(1);
        l.gx = l.gy = l.gz = l.m = l.ej = l.dn = T(0);
        loc.push_back(l);
      }
      const int64_t m = static_cast<int64_t>(loc.size());
      int64_t i_s = 0, i_g = 0;
      while (i_s < m && loc[i_s].j < c) ++i_s;
      i_g = i_s;
      while (i_g < m && loc[i_g].j == c) ++i_g;
      // a triangle of the neighbors a and b; a's sums go to aa (registers in the inner loop), b's to b; UQ and SW
      // fix the damping form and the switch at compile time
      auto tri = [&](auto uq, auto swc, const Loc& a, Acc& aa, Loc& b, bool distinct) {
        constexpr bool UQ = decltype(uq)::value, SW = decltype(swc)::value;
        const T dx = a.vx - b.vx, dy = a.vy - b.vy, dz = a.vz - b.vz;
        const T w = dx * dx + dy * dy + dz * dz;
        if (w > cut2 || w <= eps) return;
        const T x = a.r2, y = b.r2;
        const T sjk = aa.c6row[b.j], c6jk = sjk * sjk;
        const T c9 = s9 * a.sq * b.sq * sjk;
        // 1 / sqrt(P), P = r2ij r2ik r2jk, its powers, the damping and 1 / C6_jk
        const T pr2 = x * y * w, pr1 = std::sqrt(pr2);
        T ip1, tt, damp, ic6;
        if constexpr (UQ) {
          tt = T(6) * a.q * b.q * aa.qrow[b.j];
          const T opt = T(1) + tt, inv = T(1) / (pr1 * opt * c6jk), d = inv * c6jk;
          ip1 = d * opt;
          damp = d * pr1;
          ic6 = inv * pr1 * opt;
        } else {
          const T inv = T(1) / (pr1 * c6jk);
          ip1 = inv * c6jk;
          ic6 = inv * pr1;
          if (thirds) {
            tt = T(6) * (a.r0p * b.r0p * r0ps[a.sj * ns + b.sj]) * ipow(whole ? ip1 : inv_cbrt(pr1), kp);
          } else {
            tt = T(6) * std::pow(a.r0 * b.r0 * r0s[a.sj * ns + b.sj] * ip1, alp3);
          }
          damp = T(1) / (T(1) + tt);
        }
        const T ip2 = ip1 * ip1, ip3 = ip2 * ip1, ip5 = ip3 * ip2;
        const T A = x + w - y, B = x - w + y, C = -x + w + y;
        const T abc = A * B * C;
        const T ang = T(0.375) * abc * ip5 + ip3;
        T swk = T(1), dswk = T(0), rk = T(0);
        if constexpr (SW) { rk = std::sqrt(w); switching(rk, cut, width, swk, dswk); }
        const T swt = SW ? a.sw * b.sw * swk : T(1);
        const T et = c9 * ang * damp * swt;
        const T third = et * (T(1) / T(3));
        aa.ec += third;
        if (distinct) { aa.ej += third; b.ej += third; }
        if constexpr (GRAD) {
          const T mult = distinct ? (wc + a.wj + b.wj) * (T(1) / T(3)) : wc * (T(1) / T(3));
          if (mult == T(0)) return;
          // 1 / x, 1 / y, 1 / w from 1 / P
          const T iw = x * y * ip2, ix = y * w * ip2, iy = x * w * ip2;
          const T K = T(0.375) * T(2.5) * abc * ip5 + T(1.5) * ip3;
          const T BC = B * C, AC = A * C, AB = A * B;
          const T dax = T(0.375) * (BC + AC - AB) * ip5 - K * ix;
          const T day = T(0.375) * (-BC + AC + AB) * ip5 - K * iy;
          const T daw = T(0.375) * (BC - AC + AB) * ip5 - K * iw;
          const T hd = damp * damp * (alp3 * T(0.5)) * tt;
          const T base = c9 * swt;
          T ex = base * (dax * damp + ang * hd * ix);
          T ey = base * (day * damp + ang * hd * iy);
          T ew = base * (daw * damp + ang * hd * iw);
          if constexpr (SW) {
            const T ad = c9 * ang * damp;
            ex += ad * b.sw * swk * a.dsw / (T(2) * a.r);
            ey += ad * a.sw * swk * b.dsw / (T(2) * b.r);
            ew += ad * a.sw * b.sw * dswk / (T(2) * rk);
          }
          const T m2 = T(2) * mult;
          ex *= m2; ey *= m2; ew *= m2;
          aa.gx += ex * a.vx + ew * dx; aa.gy += ex * a.vy + ew * dy; aa.gz += ex * a.vz + ew * dz;
          b.gx += ey * b.vx - ew * dx; b.gy += ey * b.vy - ew * dy; b.gz += ey * b.vz - ew * dz;
          const T me = mult * et;
          aa.m += me;
          b.m += me;
          const T bk = T(0.5) * me * ic6;
          aa.dn += bk * aa.dmrow[b.j];
          b.dn += bk * aa.dnrow[b.j];
        }
      };
      T ec = T(0), dcc = T(0);
      auto pass = [&](auto uq, auto swc) {
        // the neighbor q's own sums, in registers over the inner loop
        auto run_q = [&](int64_t q, int64_t s0, int64_t s1, bool same_atom_only, bool repeated) {
          const Loc a = loc[q];
          Acc aa;
          aa.c6row = s6m + a.j * n; aa.dmrow = dc6m + a.j * n; aa.dnrow = dc6n + a.j * n;
          aa.qrow = unique ? qm.data() + a.j * n : nullptr;
          for (int64_t s = s0; s < s1; ++s) {
            if (same_atom_only && loc[s].j != a.j) break;
            tri(uq, swc, a, aa, loc[s], !repeated && loc[s].j != a.j);
          }
          Loc& l = loc[q];
          ec += aa.ec;
          l.ej += aa.ej; l.gx += aa.gx; l.gy += aa.gy; l.gz += aa.gz; l.m += aa.m; l.dn += aa.dn;
        };
        // three distinct atoms, from the smallest-index corner
        for (int64_t q = i_g; q < m; ++q) run_q(q, q + 1, m, false, false);
        // two images of one smaller-index atom
        for (int64_t q = 0; q < i_s; ++q) run_q(q, q + 1, i_s, true, true);
        // images of the center, among themselves and with every other neighbor
        for (int64_t q = i_s; q < i_g; ++q) {
          run_q(q, q + 1, i_g, false, true);
          run_q(q, 0, i_s, false, true);
          run_q(q, i_g, m, false, true);
        }
      };
      using yes = std::true_type;
      using no = std::false_type;
      if (unique) { if (sw_on) pass(yes{}, yes{}); else pass(yes{}, no{}); }
      else { if (sw_on) pass(no{}, yes{}); else pass(no{}, no{}); }
      e3l[c] += ec;
      for (int64_t q = 0; q < m; ++q) {
        const Loc& l = loc[q];
        if (l.ej != T(0)) e3l[l.j] += l.ej;
        if (GRAD) {
          gb[3 * l.k] += l.gx; gb[3 * l.k + 1] += l.gy; gb[3 * l.k + 2] += l.gz;
          T dj = l.dn;
          if (l.m != T(0)) {
            const T a6 = l.m / (T(2) * l.c6);
            dcc += a6 * l.dcc;
            dj += a6 * l.dcj;
          }
          if (dj != T(0)) dcn[l.j] += dj;
        }
      }
      if (GRAD) dcn[c] += dcc;
    }
  }
}

// the sum over the thread rows
static void reduce_rows(const std::vector<T>& rows, int nt, int64_t n, T scale, T* out, bool par) {
#pragma omp parallel for schedule(static) if (par)
  for (int64_t c = 0; c < n; ++c) {
    T s = T(0);
    for (int t = 0; t < nt; ++t) s += rows[static_cast<size_t>(t) * n + c];
    out[c] = s * scale;
  }
}

template <bool GRAD>
static void run_d3(const T* vec, const int64_t* src, const int64_t* dst, const int64_t* z, const int64_t* batch,
                   int64_t n, int64_t ne, const T* rcov, const T* r4r2, const T* rvdw, const T* refcn,
                   const int64_t* nref, const T* c6r, int64_t nz, const Prm& p, const T* w2, const T* w3,
                   const T* seed, T* e2, T* e3, T* cn, T* c6self, T* dself, T* grad) {
  const bool three = p.s9 != T(0);
  const int nt = n_threads();
  const Edges E = sorted_edges(vec, src, dst, n, ne, p.bohr);
  const bool par = E.par;
  std::vector<T> gb;
  if (GRAD) gb.assign(3 * ne, T(0));
  // coordination numbers, reference weights
  std::vector<T> wt(7 * n), dwt(7 * n);
#pragma omp parallel for schedule(static) if (par)
  for (int64_t c = 0; c < n; ++c) {
    T s = T(0);
    for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k) {
      if (E.r[k] > p.cut_cn) continue;
      const T rc = rcov[z[c]] + rcov[z[E.js[k]]];
      s += T(1) / (T(1) + std::exp(-p.kcn * (rc / E.r[k] - T(1))));
    }
    cn[c] = s;
    ref_weights(s, z[c], refcn, nref, nz, p.wf, &wt[7 * c], &dwt[7 * c]);
  }
  auto ref_of = [&](int64_t zi, int64_t zj) { return c6r + (zi * nz + zj) * 49; };
  auto nr_of = [&](int64_t zi) { return static_cast<int>(nref[zi]); };
  // dense pair sqrt(C6) and C6 derivatives inside each structure (three-body term)
  std::vector<T> s6m, dc6m, dc6n;
  if (three) {
    std::vector<int64_t> hi(n);
    for (int64_t i = n - 1; i >= 0; --i) hi[i] = (i == n - 1 || batch[i + 1] != batch[i]) ? i + 1 : hi[i + 1];
    s6m.assign(static_cast<size_t>(n) * n, T(0));
    dc6m.assign(static_cast<size_t>(n) * n, T(0));
    dc6n.assign(static_cast<size_t>(n) * n, T(0));
#pragma omp parallel for schedule(dynamic, 8) if (par)
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = i; j < hi[i]; ++j) {
        T c6, di, dj;
        pair_c6(ref_of(z[i], z[j]), &wt[7 * i], &dwt[7 * i], nr_of(z[i]), &wt[7 * j], &dwt[7 * j], nr_of(z[j]),
                c6, di, dj);
        const T sq = std::sqrt(c6);
        s6m[i * n + j] = sq; s6m[j * n + i] = sq;
        dc6m[i * n + j] = di; dc6m[j * n + i] = dj;
        dc6n[i * n + j] = dj; dc6n[j * n + i] = di;
        if (j == i) { c6self[i] = c6; dself[i] = di + dj; }
      }
    }
  } else {
#pragma omp parallel for schedule(static) if (par)
    for (int64_t i = 0; i < n; ++i) {
      T c6, di, dj;
      pair_c6(ref_of(z[i], z[i]), &wt[7 * i], &dwt[7 * i], nr_of(z[i]), &wt[7 * i], &dwt[7 * i], nr_of(z[i]),
              c6, di, dj);
      c6self[i] = c6;
      dself[i] = di + dj;
    }
  }
  const size_t per = static_cast<size_t>(nt) * n;
  std::vector<T> dcn_t(GRAD ? per : 0, T(0)), e3_t(three ? per : 0, T(0));
  const bool rational = p.damping == 0 || p.damping == 3;
  // two-body sum: every directed edge gives half its pair energy to its center
#pragma omp parallel if (par)
  {
    T* dcn = GRAD ? dcn_t.data() + static_cast<size_t>(thread_id()) * n : nullptr;
#pragma omp for schedule(dynamic, 16)
    for (int64_t c = 0; c < n; ++c) {
      const int64_t zc = z[c];
      const T wc = w2 ? w2[c] : T(1);
      T ec = T(0);
      for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k) {
        const T rk = E.r[k];
        if (rk > p.cut_pair) continue;
        const int64_t j = E.js[k], zj = z[j];
        T c6, dci, dcj;
        if (three) {
          c6 = s6m[c * n + j] * s6m[c * n + j]; dci = dc6m[c * n + j]; dcj = dc6n[c * n + j];
        } else {
          pair_c6(ref_of(zc, zj), &wt[7 * c], &dwt[7 * c], nr_of(zc), &wt[7 * j], &dwt[7 * j], nr_of(zj), c6, dci, dcj);
        }
        const T rr = T(3) * r4r2[zc] * r4r2[zj];
        const T r0 = rational ? p.a1 * std::sqrt(rr) + p.a2 : rvdw[zc * nz + zj];
        T g, dg, s, ds;
        damped(p, rk, r0, rr, g, dg);
        switching(rk, p.cut_pair, p.w_pair, s, ds);
        ec += -T(0.5) * c6 * s * g;
        if (GRAD) {
          const T f = -T(0.5) * c6 * (ds * g + s * dg) * wc / rk;
          gb[3 * k] += f * E.v[3 * k]; gb[3 * k + 1] += f * E.v[3 * k + 1]; gb[3 * k + 2] += f * E.v[3 * k + 2];
          const T de = -T(0.5) * s * g * wc;
          dcn[c] += de * dci;
          dcn[j] += de * dcj;
        }
      }
      e2[c] = ec * p.hartree;
    }
  }
  if (three)
    atm<GRAD>(E, z, nz, s6m.data(), dc6m.data(), dc6n.data(), rvdw, p.rs9, p.s9, p.alp3, p.cut_triple, p.w_triple,
              w3, e3_t.data(), GRAD ? dcn_t.data() : nullptr, GRAD ? gb.data() : nullptr);
  if (three) reduce_rows(e3_t, nt, n, p.hartree, e3, par);
  if (!GRAD) return;
  // the coordination-number chain, into each center's own edges
#pragma omp parallel for schedule(dynamic, 16) if (par)
  for (int64_t c = 0; c < n; ++c) {
    T g = seed ? seed[c] / p.hartree : T(0);
    for (int t = 0; t < nt; ++t) g += dcn_t[static_cast<size_t>(t) * n + c];
    if (g != T(0)) {
      for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k) {
        const T rk = E.r[k];
        if (rk > p.cut_cn) continue;
        const T rc = rcov[z[c]] + rcov[z[E.js[k]]];
        const T ex = std::exp(-p.kcn * (rc / rk - T(1)));
        const T cnt = T(1) / (T(1) + ex);
        const T f = -g * cnt * cnt * ex * p.kcn * rc / (rk * rk * rk);
        gb[3 * k] += f * E.v[3 * k]; gb[3 * k + 1] += f * E.v[3 * k + 1]; gb[3 * k + 2] += f * E.v[3 * k + 2];
      }
    }
  }
  scatter_gradient(E, gb, p.hartree / p.bohr, grad);
}

static Prm params_of(const std::vector<double>& prm) {
  Prm p;
  p.damping = static_cast<int>(prm[0]);
  T* f[] = {&p.s6, &p.s8, &p.s9, &p.a1, &p.a2, &p.rs6, &p.rs8, &p.bet, &p.alp, &p.rs9, &p.kcn, &p.wf,
            &p.alp3, &p.cut_pair, &p.cut_triple, &p.cut_cn, &p.w_pair, &p.w_triple, &p.bohr, &p.hartree};
  for (int i = 0; i < 20; ++i) *f[i] = prm[i + 1];
  return p;
}

static void evaluate_d3(const torch::Tensor& vec, const torch::Tensor& src, const torch::Tensor& dst,
                        const torch::Tensor& z, const torch::Tensor& batch, const std::vector<torch::Tensor>& tabs,
                        const Prm& p, bool grad, const T* w2, const T* w3, const T* seed, torch::Tensor& e2,
                        torch::Tensor& e3, torch::Tensor& cn, torch::Tensor& c6self, torch::Tensor& dself,
                        torch::Tensor& g) {
  const int64_t n = z.size(0), ne = src.size(0);
  auto call = [&](auto grad_tag) {
    constexpr bool G = decltype(grad_tag)::value;
    run_d3<G>(vec.data_ptr<T>(), src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(), z.data_ptr<int64_t>(),
              batch.data_ptr<int64_t>(), n, ne, tabs[0].data_ptr<T>(), tabs[1].data_ptr<T>(), tabs[2].data_ptr<T>(),
              tabs[3].data_ptr<T>(), tabs[4].data_ptr<int64_t>(), tabs[5].data_ptr<T>(), tabs[0].size(0), p, w2, w3,
              seed, e2.data_ptr<T>(), e3.data_ptr<T>(), cn.data_ptr<T>(), c6self.data_ptr<T>(), dself.data_ptr<T>(),
              G ? g.data_ptr<T>() : nullptr);
  };
  if (grad) call(std::true_type{}); else call(std::false_type{});
}

// D3: the energies per atom (total, two-body, three-body), the features (coordination number, homoatomic C6),
// d C6_ii / d CN_i and, with grad, the gradient of the summed energies in the edge vectors
std::vector<torch::Tensor> d3(torch::Tensor vec, torch::Tensor src, torch::Tensor dst, torch::Tensor z,
                              torch::Tensor batch, std::vector<torch::Tensor> tabs, std::vector<double> prm,
                              bool grad) {
  TORCH_CHECK(vec.scalar_type() == torch::kFloat64, "the dispersion CPU kernels compute in float64");
  check_d3(z, tabs, prm);
  const int64_t n = z.size(0), ne = src.size(0);
  auto opt = vec.options();
  torch::Tensor e2 = torch::zeros({n}, opt), e3 = torch::zeros({n}, opt), cn = torch::zeros({n}, opt);
  torch::Tensor c6self = torch::zeros({n}, opt), dself = torch::zeros({n}, opt);
  torch::Tensor g = grad ? torch::zeros({ne, 3}, opt) : torch::zeros({0, 3}, opt);
  if (n > 0) evaluate_d3(vec, src, dst, z, batch, tabs, params_of(prm), grad, nullptr, nullptr, nullptr, e2, e3, cn,
                         c6self, dself, g);
  torch::Tensor node = e2 + e3, feats = torch::stack({cn, c6self}, 1);
  return {node, e2, e3, feats, dself, g};
}

// output weights per atom from the gradients of the total, two-body and three-body energies; whether they weight
// every atom alike (then the forward pass's gradient, scaled by the returned value, is the answer)
static bool weights_of(int64_t n, const c10::optional<torch::Tensor>& gn, const c10::optional<torch::Tensor>& g2,
                       const c10::optional<torch::Tensor>& g3, bool three, std::vector<T>& w2, std::vector<T>& w3,
                       T& c0) {
  w2.assign(n, T(0)); w3.assign(n, T(0));
  auto add = [&](const c10::optional<torch::Tensor>& t, std::vector<T>& w) {
    if (!t.has_value()) return;
    auto c = t->contiguous();
    const T* q = c.data_ptr<T>();
    for (int64_t i = 0; i < n; ++i) w[i] += q[i];
  };
  add(gn, w2); add(gn, w3); add(g2, w2); add(g3, w3);
  if (n == 0) return false;
  c0 = w2[0];
  for (int64_t i = 0; i < n; ++i)
    if (w2[i] != c0 || (three && w3[i] != c0)) return false;
  return true;
}

// D3: the gradient in the edge vectors for the output gradients gn (total), g2, g3 (two- and three-body) and gf
// (features)
torch::Tensor d3_backward(torch::Tensor vec, torch::Tensor src, torch::Tensor dst, torch::Tensor z,
                          torch::Tensor batch, std::vector<torch::Tensor> tabs, std::vector<double> prm,
                          torch::Tensor cached, torch::Tensor dself, c10::optional<torch::Tensor> gn,
                          c10::optional<torch::Tensor> g2, c10::optional<torch::Tensor> g3,
                          c10::optional<torch::Tensor> gf) {
  const int64_t n = z.size(0), ne = src.size(0);
  const Prm p = params_of(prm);
  std::vector<T> w2, w3, seed(n, T(0));
  T c0 = T(0);
  const bool uniform = weights_of(n, gn, g2, g3, p.s9 != T(0), w2, w3, c0);
  bool any_seed = false;
  if (gf.has_value()) {
    auto c = gf->contiguous();
    const T* q = c.data_ptr<T>();
    const T* ds = dself.data_ptr<T>();
    for (int64_t i = 0; i < n; ++i) {
      seed[i] = q[2 * i] + q[2 * i + 1] * ds[i];
      any_seed = any_seed || seed[i] != T(0);
    }
  }
  if (cached.numel() > 0 && !any_seed && uniform) return cached * c0;
  auto opt = vec.options();
  torch::Tensor e2 = torch::zeros({n}, opt), e3 = torch::zeros({n}, opt), cn = torch::zeros({n}, opt);
  torch::Tensor c6self = torch::zeros({n}, opt), ds = torch::zeros({n}, opt), g = torch::zeros({ne, 3}, opt);
  if (n > 0) evaluate_d3(vec, src, dst, z, batch, tabs, p, true, w2.data(), w3.data(), any_seed ? seed.data() : nullptr,
                         e2, e3, cn, c6self, ds, g);
  return g;
}

static void check_d3(const torch::Tensor& z, const std::vector<torch::Tensor>& tabs, const std::vector<double>& prm) {
  const int64_t n = z.size(0), max_z = static_cast<int64_t>(prm[21]);
  const int64_t* zp = z.data_ptr<int64_t>();
  const int64_t* nr = tabs[4].data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i)
    if (zp[i] < 1 || zp[i] > max_z) throw std::invalid_argument("D3 reference data cover atomic numbers 1..103");
  for (int64_t i = 0; i < n; ++i)
    if (nr[zp[i]] == 0)
      throw std::invalid_argument("no D3 reference systems for an element of the input (Z > 94 with "
                                  "references='2010'?)");
}

// D3 with the forces and the strain derivative, no autograd: the edge vectors from the positions, the cells (B, 3, 3;
// empty for molecules) and the integer shifts; returns the energies per atom, the per-structure energies (B, 3:
// total, two-body, three-body), the features (N, 2), the forces (N, 3) and dE / d(strain) (B, 3, 3), symmetric, in eV
std::vector<torch::Tensor> d3_direct(torch::Tensor pos, torch::Tensor cell, torch::Tensor shifts, torch::Tensor src,
                                     torch::Tensor dst, torch::Tensor z, torch::Tensor batch, int64_t num_graphs,
                                     std::vector<torch::Tensor> tabs, std::vector<double> prm) {
  TORCH_CHECK(pos.scalar_type() == torch::kFloat64, "the dispersion CPU kernels compute in float64");
  check_d3(z, tabs, prm);
  const int64_t n = z.size(0), ne = src.size(0);
  const T* p = pos.data_ptr<T>();
  const int64_t* s = src.data_ptr<int64_t>();
  const int64_t* d = dst.data_ptr<int64_t>();
  const int64_t* b = batch.data_ptr<int64_t>();
  const bool periodic = cell.numel() > 0;
  const T* cp = periodic ? cell.data_ptr<T>() : nullptr;
  const T* sh = periodic ? shifts.data_ptr<T>() : nullptr;
  std::vector<T> v(3 * ne);
#pragma omp parallel for schedule(static) if (ne >= 2048)
  for (int64_t e = 0; e < ne; ++e) {
    for (int a = 0; a < 3; ++a) v[3 * e + a] = p[3 * d[e] + a] - p[3 * s[e] + a];
    if (periodic) {
      const T* C = cp + 9 * b[s[e]];
      for (int a = 0; a < 3; ++a)
        v[3 * e + a] += sh[3 * e] * C[a] + sh[3 * e + 1] * C[3 + a] + sh[3 * e + 2] * C[6 + a];
    }
  }
  auto opt = pos.options();
  torch::Tensor node = torch::empty({n}, opt), energy = torch::zeros({num_graphs, 3}, opt);
  torch::Tensor feats = torch::empty({n, 2}, opt), forces = torch::zeros({n, 3}, opt);
  torch::Tensor strain = torch::zeros({num_graphs, 3, 3}, opt);
  if (n == 0) return {node, energy, feats, forces, strain};
  std::vector<T> e2(n), e3(n, T(0)), cn(n), c6s(n), ds(n), g(3 * ne);
  const Prm pp = params_of(prm);
  run_d3<true>(v.data(), s, d, z.data_ptr<int64_t>(), b, n, ne, tabs[0].data_ptr<T>(), tabs[1].data_ptr<T>(),
               tabs[2].data_ptr<T>(), tabs[3].data_ptr<T>(), tabs[4].data_ptr<int64_t>(), tabs[5].data_ptr<T>(),
               tabs[0].size(0), pp, nullptr, nullptr, nullptr, e2.data(), e3.data(), cn.data(), c6s.data(), ds.data(),
               g.data());
  T* np_ = node.data_ptr<T>();
  T* en = energy.data_ptr<T>();
  T* ft = feats.data_ptr<T>();
  for (int64_t i = 0; i < n; ++i) {
    np_[i] = e2[i] + e3[i];
    T* E = en + 3 * b[i];
    E[0] += np_[i]; E[1] += e2[i]; E[2] += e3[i];
    ft[2 * i] = cn[i]; ft[2 * i + 1] = c6s[i];
  }
  T* f = forces.data_ptr<T>();
  T* m = strain.data_ptr<T>();
  for (int64_t e = 0; e < ne; ++e) {
    T* M = m + 9 * b[d[e]];
    for (int a = 0; a < 3; ++a) {
      f[3 * d[e] + a] -= g[3 * e + a];
      f[3 * s[e] + a] += g[3 * e + a];
      for (int c = 0; c < 3; ++c) M[3 * a + c] += v[3 * e + a] * g[3 * e + c];
    }
  }
  for (int64_t k = 0; k < num_graphs; ++k) {
    T* M = m + 9 * k;
    for (int a = 0; a < 3; ++a)
      for (int c = a + 1; c < 3; ++c) {
        const T sym = T(0.5) * (M[3 * a + c] + M[3 * c + a]);
        M[3 * a + c] = sym; M[3 * c + a] = sym;
      }
  }
  return {node, energy, feats, forces, strain};
}

// D4 dynamic polarizabilities (N, 23) at the charges q and their derivatives in the coordination number and the
// charge (each atom's own); with neutral, also those of the neutral atom and its coordination-number derivative
struct D4Ref {
  const T *refcn, *refq, *zeff, *gi, *refalpha;  // refalpha (Z, 7, 23)
  const int64_t *ngw, *nref;
  int64_t nz;
  T ga, wf, eps_norm;
};

static void d4_alpha(int64_t i, int64_t zz, T cn, T q, const D4Ref& R, T* a, T* dacn, T* daq, T* an, T* dan) {
  const int nr = static_cast<int>(R.nref[zz]);
  T w[7], dw[7], zeta[7], dzeta[7], zeta0[7];
  T norm = T(0), dnorm = T(0);
  for (int r = 0; r < nr; ++r) {
    const T d = cn - R.refcn[r * R.nz + zz];
    const T d2 = d * d;
    const int64_t ng = R.ngw[r * R.nz + zz];
    T g = T(0), dg = T(0);
    for (int64_t j = 1; j <= ng; ++j) {
      const T e = std::exp(-R.wf * T(j) * d2);
      g += e;
      dg += e * (-T(2) * R.wf * T(j) * d);
    }
    w[r] = g; dw[r] = dg;
    norm += g; dnorm += dg;
  }
  if (norm > R.eps_norm) {
    for (int r = 0; r < nr; ++r) {
      const T wr = w[r] / norm;
      dw[r] = (dw[r] - wr * dnorm) / norm;
      w[r] = wr;
    }
  } else {
    T top = -std::numeric_limits<T>::infinity();
    for (int r = 0; r < nr; ++r) top = std::max(top, R.refcn[r * R.nz + zz]);
    for (int r = 0; r < nr; ++r) {
      w[r] = std::abs(R.refcn[r * R.nz + zz] - top) < T(1e-12) ? T(1) : T(0);
      dw[r] = T(0);
    }
  }
  const T zi = R.zeff[zz], gi = R.gi[zz];
  for (int r = 0; r < nr; ++r) {
    const T qref = R.refq[r * R.nz + zz] + zi;
    const T qmod = q + zi;
    if (qmod < T(0)) {
      zeta[r] = std::exp(R.ga); dzeta[r] = T(0);
    } else {
      const T eu = std::exp(gi * (T(1) - qref / qmod));
      zeta[r] = std::exp(R.ga * (T(1) - eu));
      dzeta[r] = zeta[r] * (-R.ga * eu) * gi * qref / (qmod * qmod);
    }
    if (an) zeta0[r] = zi < T(0) ? std::exp(R.ga) : std::exp(R.ga * (T(1) - std::exp(gi * (T(1) - qref / zi))));
  }
  for (int k = 0; k < 23; ++k) { a[k] = T(0); dacn[k] = T(0); daq[k] = T(0); }
  if (an) for (int k = 0; k < 23; ++k) { an[k] = T(0); dan[k] = T(0); }
  for (int r = 0; r < nr; ++r) {
    const T* ra = R.refalpha + (zz * 7 + r) * 23;
    const T wz = w[r] * zeta[r], dwz = dw[r] * zeta[r], wdz = w[r] * dzeta[r];
    for (int k = 0; k < 23; ++k) { a[k] += wz * ra[k]; dacn[k] += dwz * ra[k]; daq[k] += wdz * ra[k]; }
    if (an) {
      const T w0 = w[r] * zeta0[r], dw0 = dw[r] * zeta0[r];
      for (int k = 0; k < 23; ++k) { an[k] += w0 * ra[k]; dan[k] += dw0 * ra[k]; }
    }
  }
}

// D4: the polarizabilities, the two-body energies from them and the three-body energies from the dense C6 of the
// neutral atoms; with GRAD the gradients in the edge vectors and the coordination numbers and charges
template <bool GRAD>
static void run_d4(const Edges& E, const int64_t* z, const int64_t* batch, const T* cn, const T* q, const D4Ref& R,
                   const T* cpw, const T* r4r2, const T* r0tab, const std::vector<double>& prm, const T* w2,
                   const T* w3, const T* ga_out, T* e2, T* e3, T* alpha, T* gvec, T* gcn, T* gq) {
  const int64_t n = E.n;
  const int nt = n_threads();
  Prm p;
  p.damping = 0;
  p.s6 = prm[0]; p.s8 = prm[1]; p.a1 = prm[2]; p.a2 = prm[3]; p.s9 = prm[4]; p.alp3 = prm[5];
  p.cut_pair = prm[6]; p.cut_triple = prm[7]; p.w_pair = prm[8]; p.w_triple = prm[9]; p.bohr = prm[10];
  p.hartree = prm[11];
  const bool three = p.s9 != T(0);
  const T k6 = T(3) / T(3.141592653589793);
  std::vector<T> dacn(23 * n), daq(23 * n), an(three ? 23 * n : 0), dan(three ? 23 * n : 0);
#pragma omp parallel for schedule(static) if (E.par)
  for (int64_t i = 0; i < n; ++i)
    d4_alpha(i, z[i], cn[i], q[i], R, alpha + 23 * i, &dacn[23 * i], &daq[23 * i], three ? &an[23 * i] : nullptr,
             three ? &dan[23 * i] : nullptr);
  std::vector<T> gb, galpha;
  if (GRAD) { gb.assign(3 * E.ne, T(0)); galpha.assign(23 * n, T(0)); }
#pragma omp parallel for schedule(dynamic, 16) if (E.par)
  for (int64_t c = 0; c < n; ++c) {
    const int64_t zc = z[c];
    const T wc = w2 ? w2[c] : T(1);
    const T* ac = alpha + 23 * c;
    T acc[23];
    for (int w = 0; w < 23; ++w) acc[w] = T(0);
    T ec = T(0);
    for (int64_t k = E.ptr[c]; k < E.ptr[c + 1]; ++k) {
      const T rk = E.r[k];
      if (rk > p.cut_pair) continue;
      const int64_t j = E.js[k], zj = z[j];
      const T* aj = alpha + 23 * j;
      T c6 = T(0);
      for (int w = 0; w < 23; ++w) c6 += ac[w] * aj[w] * cpw[w];
      c6 *= k6;
      const T rr = T(3) * r4r2[zc] * r4r2[zj];
      const T r0 = p.a1 * std::sqrt(rr) + p.a2;
      T g, dg, s, ds;
      damped(p, rk, r0, rr, g, dg);
      switching(rk, p.cut_pair, p.w_pair, s, ds);
      ec += -T(0.5) * c6 * s * g;
      if (GRAD) {
        const T f = -T(0.5) * c6 * (ds * g + s * dg) * wc / rk;
        gb[3 * k] += f * E.v[3 * k]; gb[3 * k + 1] += f * E.v[3 * k + 1]; gb[3 * k + 2] += f * E.v[3 * k + 2];
        // this edge's energy (weight wc) and its reverse's, at the neighbor (weight w2[j]), in alpha_c
        const T de = -T(0.5) * s * g * (wc + (w2 ? w2[j] : T(1))) * k6;
        for (int w = 0; w < 23; ++w) acc[w] += de * cpw[w] * aj[w];
      }
    }
    e2[c] = ec * p.hartree;
    if (GRAD)
      for (int w = 0; w < 23; ++w) galpha[23 * c + w] = acc[w];
  }
  const size_t per = static_cast<size_t>(nt) * n;
  std::vector<T> e3_t(three ? per : 0, T(0)), dcn_t(GRAD && three ? per : 0, T(0));
  if (three) {
    // the neutral atoms' C6 and their derivatives in either coordination number, inside each structure
    std::vector<int64_t> hi(n);
    for (int64_t i = n - 1; i >= 0; --i) hi[i] = (i == n - 1 || batch[i + 1] != batch[i]) ? i + 1 : hi[i + 1];
    std::vector<T> s6m(static_cast<size_t>(n) * n, T(0)), dc6m(static_cast<size_t>(n) * n, T(0));
    std::vector<T> dc6n(static_cast<size_t>(n) * n, T(0));
#pragma omp parallel for schedule(dynamic, 8) if (E.par)
    for (int64_t i = 0; i < n; ++i) {
      const T* ai = &an[23 * i];
      const T* di = &dan[23 * i];
      for (int64_t j = i; j < hi[i]; ++j) {
        const T* aj = &an[23 * j];
        const T* dj = &dan[23 * j];
        T c6 = T(0), dci = T(0), dcj = T(0);
        for (int w = 0; w < 23; ++w) {
          c6 += ai[w] * cpw[w] * aj[w];
          dci += di[w] * cpw[w] * aj[w];
          dcj += ai[w] * cpw[w] * dj[w];
        }
        c6 *= k6; dci *= k6; dcj *= k6;
        const T sq = std::sqrt(c6);
        s6m[i * n + j] = sq; s6m[j * n + i] = sq;
        dc6m[i * n + j] = dci; dc6m[j * n + i] = dcj;
        dc6n[i * n + j] = dcj; dc6n[j * n + i] = dci;
      }
    }
    atm<GRAD>(E, z, R.nz, s6m.data(), dc6m.data(), dc6n.data(), r0tab, T(1), p.s9, p.alp3, p.cut_triple,
              p.w_triple, w3, e3_t.data(), GRAD ? dcn_t.data() : nullptr, GRAD ? gb.data() : nullptr);
    reduce_rows(e3_t, nt, n, p.hartree, e3, E.par);
  }
  if (!GRAD) return;
  // the polarizability gradient (pair term and the caller's) into the coordination numbers and charges
#pragma omp parallel for schedule(static) if (E.par)
  for (int64_t i = 0; i < n; ++i) {
    T sc = T(0), sq = T(0);
    for (int w = 0; w < 23; ++w) {
      const T g = galpha[23 * i + w] + (ga_out ? ga_out[23 * i + w] / p.hartree : T(0));
      sc += g * dacn[23 * i + w];
      sq += g * daq[23 * i + w];
    }
    if (three)
      for (int t = 0; t < nt; ++t) sc += dcn_t[static_cast<size_t>(t) * n + i];
    gcn[i] = sc * p.hartree;
    gq[i] = sq * p.hartree;
  }
  scatter_gradient(E, gb, p.hartree / p.bohr, gvec);
}

static D4Ref d4_ref(const std::vector<torch::Tensor>& tabs, const std::vector<double>& prm) {
  D4Ref R;
  R.refcn = tabs[0].data_ptr<T>(); R.refq = tabs[1].data_ptr<T>(); R.zeff = tabs[2].data_ptr<T>();
  R.gi = tabs[3].data_ptr<T>(); R.refalpha = tabs[4].data_ptr<T>(); R.ngw = tabs[5].data_ptr<int64_t>();
  R.nref = tabs[6].data_ptr<int64_t>(); R.nz = tabs[2].size(0);
  R.ga = prm[12]; R.wf = prm[13]; R.eps_norm = prm[14];
  return R;
}

// D4: tabs = [refcn (7, Z), refq (7, Z), zeff (Z), hardness * gc (Z), refalpha (Z, 7, 23), ngw (7, Z), nref (Z),
// Casimir-Polder weights (23), r4r2 (Z), BJ radii (Z, Z)]; returns the two- and three-body energies per atom, the
// polarizabilities (N, 23) and, with grad, the gradients of the summed energies in the edge vectors, the
// coordination numbers and the charges
static std::vector<torch::Tensor> evaluate_d4(const torch::Tensor& vec, const torch::Tensor& src,
                                              const torch::Tensor& dst, const torch::Tensor& z,
                                              const torch::Tensor& batch, const torch::Tensor& cn,
                                              const torch::Tensor& q, const std::vector<torch::Tensor>& tabs,
                                              const std::vector<double>& prm, bool grad, const T* w2, const T* w3,
                                              const T* ga_out) {
  const int64_t n = z.size(0), ne = src.size(0);
  auto opt = vec.options();
  torch::Tensor e2 = torch::zeros({n}, opt), e3 = torch::zeros({n}, opt), alpha = torch::zeros({n, 23}, opt);
  torch::Tensor gv = grad ? torch::zeros({ne, 3}, opt) : torch::zeros({0, 3}, opt);
  torch::Tensor gc = grad ? torch::zeros({n}, opt) : torch::zeros({0}, opt);
  torch::Tensor gq = grad ? torch::zeros({n}, opt) : torch::zeros({0}, opt);
  if (n == 0) return {e2, e3, alpha, gv, gc, gq};
  const Edges E = sorted_edges(vec.data_ptr<T>(), src.data_ptr<int64_t>(), dst.data_ptr<int64_t>(), n, ne, prm[10]);
  const D4Ref R = d4_ref(tabs, prm);
  auto call = [&](auto grad_tag) {
    constexpr bool G = decltype(grad_tag)::value;
    run_d4<G>(E, z.data_ptr<int64_t>(), batch.data_ptr<int64_t>(), cn.data_ptr<T>(), q.data_ptr<T>(), R,
              tabs[7].data_ptr<T>(), tabs[8].data_ptr<T>(), tabs[9].data_ptr<T>(), prm, w2, w3, ga_out,
              e2.data_ptr<T>(), e3.data_ptr<T>(), alpha.data_ptr<T>(), G ? gv.data_ptr<T>() : nullptr,
              G ? gc.data_ptr<T>() : nullptr, G ? gq.data_ptr<T>() : nullptr);
  };
  if (grad) call(std::true_type{}); else call(std::false_type{});
  return {e2, e3, alpha, gv, gc, gq};
}

std::vector<torch::Tensor> d4(torch::Tensor vec, torch::Tensor src, torch::Tensor dst, torch::Tensor z,
                              torch::Tensor batch, torch::Tensor cn, torch::Tensor q, std::vector<torch::Tensor> tabs,
                              std::vector<double> prm, bool grad) {
  TORCH_CHECK(vec.scalar_type() == torch::kFloat64, "the dispersion CPU kernels compute in float64");
  return evaluate_d4(vec, src, dst, z, batch, cn, q, tabs, prm, grad, nullptr, nullptr, nullptr);
}

// D4: the gradients in the edge vectors, coordination numbers and charges for the output gradients g2, g3 (two- and
// three-body energies) and galpha (polarizabilities)
std::vector<torch::Tensor> d4_backward(torch::Tensor vec, torch::Tensor src, torch::Tensor dst, torch::Tensor z,
                                       torch::Tensor batch, torch::Tensor cn, torch::Tensor q,
                                       std::vector<torch::Tensor> tabs, std::vector<double> prm,
                                       std::vector<torch::Tensor> cached, c10::optional<torch::Tensor> g2,
                                       c10::optional<torch::Tensor> g3, c10::optional<torch::Tensor> galpha) {
  const int64_t n = z.size(0);
  std::vector<T> w2, w3;
  T c0 = T(0);
  const bool uniform = weights_of(n, c10::nullopt, g2, g3, prm[4] != 0.0, w2, w3, c0);
  if (cached[0].numel() > 0 && uniform && !galpha.has_value())
    return {cached[0] * c0, cached[1] * c0, cached[2] * c0};
  torch::Tensor ga;
  if (galpha.has_value()) ga = galpha->contiguous();
  auto out = evaluate_d4(vec, src, dst, z, batch, cn, q, tabs, prm, true, w2.data(), w3.data(),
                         galpha.has_value() ? ga.data_ptr<T>() : nullptr);
  return {out[3], out[4], out[5]};
}
"""


def _module():
    return load("xnn_dispersion_cpu", _SOURCE, ["d3", "d3_backward", "d3_direct", "d4", "d4_backward"])


def available() -> bool:
    """Whether the C++ kernels build here (checked once per process)."""
    return _module() is not None


def _d3_tables(term) -> List[Tensor]:
    """D3's reference data in the kernels' layout (cached on the term): covalent radii, ``r4r2``, van der
    Waals radii, reference coordination numbers ``(7, Z)``, reference counts and the reference C6 as
    ``(Z, Z, 7, 7)``."""
    key = term.c6ref.data_ptr()
    cached = term.__dict__.get("_cpu_tables")
    if cached is None or cached[0] != key:
        f64 = torch.float64
        cached = (key, [term.rcov.to(f64).contiguous(), term.r4r2.to(f64).contiguous(),
                        term.rvdw.to(f64).contiguous(), term.refcn.to(f64).contiguous(),
                        term.nref.to(torch.int64).contiguous(),
                        term.c6ref.to(f64).permute(2, 3, 0, 1).contiguous()])
        term.__dict__["_cpu_tables"] = cached
    return cached[1]


def _d3_parameters(term) -> List[float]:
    """D3's damping code, parameters and constants, in bohr and hartree (cached on the term while its
    parameter buffers are unchanged)."""
    names = ("s6", "s8", "s9", "a1", "a2", "rs6", "rs8", "bet")
    cached = term.__dict__.get("_cpu_parameters")
    if cached is not None:
        tensors, versions, prm = cached
        if all(term._buffers.get(k) is t and t._version == v for k, t, v in zip(names, tensors, versions)):
            return prm
    tensors = [term._buffers[k] for k in names]
    s6, s8, s9, a1, a2, rs6, rs8, bet = torch.stack(tensors).tolist()
    b = term.bohr
    prm = [float(_DAMPINGS[term.damping]), s6, s8, s9, a1, a2, rs6, rs8, bet, term.alp, term.rs9, term.kcn,
           term.wf, (term.alp + 2.0) / 3.0, term.cutoff_pair / b, term.cutoff_triple / b, term.cutoff_cn / b,
           term.switch_width_pair / b, term.switch_width_triple / b, b, term.hartree, float(term.max_z)]
    term.__dict__["_cpu_parameters"] = (tensors, [t._version for t in tensors], prm)
    return prm


class D3CpuFunction(torch.autograd.Function):
    """Energies per atom (total, two-body, three-body; eV) and the features (coordination number,
    homoatomic C6) from the edge vectors (Angstrom); the gradient is not differentiable again."""

    @staticmethod
    def forward(ctx, vec, src, dst, z, batch, tabs, prm):
        ctx.set_materialize_grads(False)
        node, e2, e3, feats, dself, grad = _module().d3(vec, src, dst, z, batch, tabs, prm,
                                                        bool(ctx.needs_input_grad[0]))
        ctx.save_for_backward(vec, src, dst, z, batch, dself, grad)
        ctx.args = (tabs, prm)
        return node, e2, e3, feats

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gn, g2, g3, gf):
        vec, src, dst, z, batch, dself, grad = ctx.saved_tensors
        tabs, prm = ctx.args
        g = _module().d3_backward(vec, src, dst, z, batch, tabs, prm, grad, dself, gn, g2, g3, gf)
        return g, None, None, None, None, None, None


def evaluate_d3(term, z: Tensor, edge_index: Tensor, edge_vec: Tensor, batch: Tensor) -> Tuple[Tensor, ...]:
    """The energies per atom (total, two-body, three-body; eV) and the features ``(N, 2)`` (coordination
    number, homoatomic C6) of :class:`~xnn.common.models.d3.DFTD3` ``term`` on the C++ kernels (the
    reference's values), in the dtype of ``edge_vec``; raises ``ValueError`` for elements without D3
    reference data."""
    dtype = edge_vec.dtype
    try:
        out = D3CpuFunction.apply(edge_vec.to(torch.float64), edge_index[0].contiguous(),
                                  edge_index[1].contiguous(), z.contiguous(), batch.contiguous(),
                                  _d3_tables(term), _d3_parameters(term))
    except RuntimeError as err:
        if "D3 reference" in str(err):
            raise ValueError(str(err)) from None
        raise
    return out if dtype == torch.float64 else tuple(t.to(dtype) for t in out)


def d3_direct(term, data) -> Tuple[Tensor, ...]:
    """D3 of a batched graph on the C++ kernels with its forces, without autograd: the energies per atom and
    per structure ``(B, 3)`` (total, two-body, three-body; eV) and the features in the model dtype, the forces
    ``(N, 3)`` (eV / A) and ``dE / d(strain)`` ``(B, 3, 3)`` (eV) in the dtype of the positions."""
    f64 = torch.float64
    pos = data.pos.detach()
    pos = (pos if pos.dtype == f64 else pos.to(f64)).contiguous()
    if data.cell is not None:
        cell, shifts = data.cell.detach().to(f64).contiguous(), data.cell_shifts.to(f64).contiguous()
    else:
        cell = shifts = pos.new_zeros(0)
    try:
        out = _module().d3_direct(pos, cell, shifts, data.edge_index[0].contiguous(), data.edge_index[1].contiguous(),
                                  data.atomic_numbers.contiguous(), data.batch.contiguous(), int(data.num_graphs),
                                  _d3_tables(term), _d3_parameters(term))
    except RuntimeError as err:
        if "D3 reference" in str(err):
            raise ValueError(str(err)) from None
        raise
    dtype, pdtype = data.model_dtype, data.pos.dtype
    if dtype == f64 and pdtype == f64:
        return tuple(out)
    return tuple(t.to(dtype) for t in out[:3]) + tuple(t.to(pdtype) for t in out[3:])


class D4CpuFunction(torch.autograd.Function):
    """D4's two-body and three-body energies per atom (hartree) and dynamic polarizabilities ``(N, 23)``
    from the edge vectors (Angstrom), the coordination numbers and the charges; the gradient is not
    differentiable again."""

    @staticmethod
    def forward(ctx, vec, cn, q, src, dst, z, batch, tabs, prm):
        ctx.set_materialize_grads(False)
        e2, e3, alpha, gv, gc, gq = _module().d4(vec, src, dst, z, batch, cn, q, tabs, prm,
                                                 bool(any(ctx.needs_input_grad[:3])))
        ctx.save_for_backward(vec, cn, q, src, dst, z, batch, gv, gc, gq)
        ctx.args = (tabs, prm)
        return e2, e3, alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g2, g3, galpha):
        vec, cn, q, src, dst, z, batch, gv, gc, gq = ctx.saved_tensors
        tabs, prm = ctx.args
        gv, gc, gq = _module().d4_backward(vec, src, dst, z, batch, cn, q, tabs, prm, [gv, gc, gq], g2, g3, galpha)
        return gv, gc, gq, None, None, None, None, None, None


def _d4_tables(term) -> List[Tensor]:
    """D4's reference data in the kernels' layout (float64, cached on the term)."""
    key = term.refalpha.data_ptr()
    cached = term.__dict__.get("_cpu_tables")
    if cached is None or cached[0] != key:
        f64 = torch.float64
        cached = (key, [term.refcn.to(f64).contiguous(), term.refq.to(f64).contiguous(),
                        term.zeff.to(f64).contiguous(), (term.hardness * term.gc).to(f64).contiguous(),
                        term.refalpha.to(f64).permute(2, 1, 0).contiguous(), term.ngw.to(torch.int64).contiguous(),
                        term.nref.to(torch.int64).contiguous(), term.cp_weights.to(f64).contiguous(),
                        term.r4r2.to(f64).contiguous()])
        term.__dict__["_cpu_tables"] = cached
    return cached[1]


def d4_terms(term, z: Tensor, edge_index: Tensor, edge_vec: Tensor, batch: Tensor, cn: Tensor,
             q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """The two-body and three-body energies per atom (hartree) and the dynamic polarizabilities ``(N, 23)``
    of :class:`~xnn.common.models.d4.DFTD4` ``term`` on the C++ kernels (the reference's values),
    differentiable once in ``edge_vec``, ``cn`` (the D4 coordination numbers) and ``q`` (the charges)."""
    dtype, f64 = edge_vec.dtype, torch.float64
    b = term.bohr
    s6, s8, a1, a2, s9 = torch.stack([term.s6, term.s8, term.a1, term.a2, term.s9]).tolist()
    # sqrt(tiny) of the working precision, as the reference's weights
    eps_norm = 1.0842021724855044e-19 if dtype == torch.float32 else 1.4916681462400413e-154
    # energies in hartree, as the reference's terms
    prm = [s6, s8, a1, a2, s9, term.alp / 3.0, term.cutoff_pair / b, term.cutoff_triple / b,
           term.switch_width_pair / b, term.switch_width_triple / b, b, 1.0, term.ga, term.wf, eps_norm]
    tabs = _d4_tables(term) + [term.pair_radius_table().to(f64).contiguous()]
    out = D4CpuFunction.apply(edge_vec.to(f64), cn.to(f64), q.to(f64), edge_index[0].contiguous(),
                              edge_index[1].contiguous(), z.contiguous(), batch.contiguous(), tabs, prm)
    return out if dtype == f64 else tuple(t.to(dtype) for t in out)
