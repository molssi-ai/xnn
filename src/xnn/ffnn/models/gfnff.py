"""GFN-FF: the general force field for elements, built from the structure alone.

GFN-FF (Spicher & Grimme, *Angew. Chem. Int. Ed.* 59, 15665, 2020) needs no
atom types and no input topology: neighbours, hybridization, pi systems and
all parameters follow from the elements, the positions and the total charge.
The energy is

    E = E_bond + E_angle + E_torsion + E_repulsion + E_bonded-ATM
        + E_dispersion + E_HB + E_XB + E_electrostatic

* **Bonds**: Gaussians ``k exp(-alpha (R - R0)^2)`` whose reference length
  follows the coordination numbers and whose exponent is softened for
  hydrogens in hydrogen bonds.
* **Angles**: ``k (cos t - cos t0)^2`` (``k (t - pi)^2`` for linear
  centres), damped with the two bond lengths.
* **Torsions and inversions**: ``k [1 + cos(n (phi - phi0) + pi)]`` and the
  out-of-plane forms, damped with the three bond lengths.
* **Repulsion**: ``c / R exp(-m R^1.5)`` over all pairs, with separate
  bonded and non-bonded constants.
* **Bonded ATM**: the three-body dispersion term over 1,4 pairs and their
  neighbours.
* **Dispersion**: D4-type pairwise ``C6``/``C8`` terms with charge-scaled
  reference polarizabilities.
* **Hydrogen and halogen bonds**: charge-scaled, angle-damped three-body
  terms over donor, hydrogen (or halogen) and acceptor.
* **Electrostatics**: an electronegativity-equilibration model of Gaussian
  charges solved at every geometry.

The topology (neighbours, charges of the topology, rows of the bonded
terms) is generated once from the structure the model is built with
(:mod:`~xnn.ffnn.models.gfnfflib`); ``forward`` evaluates conformations of
that molecule, and forces follow by autograd through
:class:`~xnn.common.models.outputs.ForceStressOutput`. The hydrogen- and
halogen-bond rows depend on distances and follow each conformation. The implementation
reproduces the energies and gradients of the xtb program (release 6.8.0)
for molecules of the elements in :data:`~xnn.ffnn.models.gfnfflib.SUPPORTED`.
"""
import gc
import math
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from xnn.common.data.atomic_data import AtomicGraph
from xnn.common.models import InteratomicPotential, register_model
from xnn.common.models.dispersion import HARTREE
from xnn.common.models.fast import resolve_use_fast

from . import gfnfflib as lib
from .gfnff_periodic import _plain_pages

#: Energy terms in the order of the model's ``e_<term>`` outputs.
TERMS = ("bond", "angle", "torsion", "repulsion", "batm", "dispersion", "hb", "xb", "electrostat")
#: Pair distance (Bohr) beyond which the reference program drops dispersion pairs.
DISPERSION_CUT = 50.0
#: Repulsion pairs with an exponent above this value (terms below exp(-60) of their prefactor) are skipped.
REPULSION_DROP = 60.0


def _padded(lists):
    """``(R, K)`` index tensor and mask from ragged index lists."""
    k = max(max((len(v) for v in lists), default=0), 1)
    idx = np.zeros((len(lists), k), dtype=np.int64)
    mask = np.zeros((len(lists), k), dtype=bool)
    for r, v in enumerate(lists):
        idx[r, :len(v)] = list(v)
        mask[r, :len(v)] = True
    return torch.from_numpy(idx), torch.from_numpy(mask)


class _PairDistance:
    """Distances of index pairs of a set of positions, taken as ``dist[i, j]`` like a distance matrix."""

    def __init__(self, x: Tensor):
        self.x = x

    def __getitem__(self, ij):
        i, j = ij
        return torch.sqrt(((self.x[j] - self.x[i]) ** 2).sum(-1))

    def detach(self):
        return _PairDistance(self.x.detach())


class _FusedEnergy(torch.autograd.Function):
    """The total energy from the fused kernels; its backward returns their gradient through :class:`_FusedGradient`,
    so that a force loss differentiates it again."""

    @staticmethod
    def forward(ctx, positions: Tensor, plan):
        terms, grad = plan(positions)
        ctx.save_for_backward(positions)
        ctx.grad, ctx.plan = grad.to(positions.dtype), plan
        ctx.mark_non_differentiable(terms)
        return terms.sum(), terms

    @staticmethod
    def backward(ctx, g_total, _):
        (positions,) = ctx.saved_tensors
        return g_total * _FusedGradient.apply(positions, ctx.grad, ctx.plan), None


class _FusedGradient(torch.autograd.Function):
    """The gradient of the fused kernels as a function of the positions: its backward is the Hessian-vector product
    of the tangent kernels."""

    @staticmethod
    def forward(ctx, positions: Tensor, grad: Tensor, plan):
        ctx.save_for_backward(positions)
        ctx.plan = plan
        return grad.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, u):
        (positions,) = ctx.saved_tensors
        return ctx.plan.hvp(positions, u).to(positions.dtype), None, None


class _FusedPeriodicEnergy(torch.autograd.Function):
    """The total energy of a cell from the fused kernels, differentiable twice in the positions and the cell."""

    @staticmethod
    def forward(ctx, positions: Tensor, cell: Tensor, plan):
        terms, grad, gcell = plan(positions, cell)
        ctx.save_for_backward(positions, cell)
        ctx.grads = (grad.to(positions.dtype), gcell.to(cell.dtype))
        ctx.plan = plan
        ctx.mark_non_differentiable(terms)
        return terms.sum(), terms

    @staticmethod
    def backward(ctx, g_total, _):
        positions, cell = ctx.saved_tensors
        grad, gcell = _FusedPeriodicGradient.apply(positions, cell, *ctx.grads, ctx.plan)
        return g_total * grad, g_total * gcell, None


class _FusedPeriodicGradient(torch.autograd.Function):
    """The gradients of the fused periodic kernels in the positions and the cell as a function of both."""

    @staticmethod
    def forward(ctx, positions: Tensor, cell: Tensor, grad: Tensor, gcell: Tensor, plan):
        ctx.save_for_backward(positions, cell)
        ctx.plan = plan
        return grad.clone(), gcell.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, u, w):
        positions, cell = ctx.saved_tensors
        u = torch.zeros_like(positions) if u is None else u
        w = torch.zeros_like(cell) if w is None else w
        hp, hc = ctx.plan.hvp(positions, cell, u, w)
        return hp.to(positions.dtype), hc.to(cell.dtype), None, None, None


def _tangents(plan) -> bool:
    """Whether the tangent kernels of ``plan`` (the second derivatives of training) can run, made once."""
    from xnn.common.cpu_kernels import Unsupported

    if getattr(plan, "no_tangents", False):
        return False
    try:
        plan.tangents()
    except (ImportError, Unsupported):
        plan.no_tangents = True
        return False
    return True


def _structure(structure):
    """Atomic numbers and positions of an ``ase.Atoms`` or a ``(positions, atomic_numbers)`` pair."""
    if hasattr(structure, "get_atomic_numbers"):
        return structure.get_atomic_numbers(), structure.get_positions()
    positions, numbers = structure
    return numbers, positions


_D4 = []


def _d4_tables():
    """The D4 reference tables in double precision, whatever the default dtype (built once)."""
    if not _D4:
        from xnn.common.models.d4 import DFTD4

        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            _D4.append(DFTD4(ga=0.0))
        finally:
            torch.set_default_dtype(default)
    return _D4[0]


@register_model("gfnff")
class GFNFF(InteratomicPotential):
    """The GFN-FF force field of one molecule.

    The topology is generated at construction from the given structure; every
    structure passed to ``forward`` must be a conformation of that molecule
    (same atoms in the same order). The hydrogen- and halogen-bond rows are
    those of each conformation: candidates within ``skin`` of the distance
    cuts are kept, each evaluation applies the cuts, and the candidates are
    regenerated once an atom has moved by more than half the skin. The energy is in eV. It has no unique
    atomic decomposition (the charges come from a global solve), so
    ``node_energy`` carries an equal share per atom.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic numbers of the molecule, ``(N,)``.
    positions : array-like
        Positions in Angstrom, ``(N, 3)``, that define the topology.
    charge : int, optional
        Total charge.
    cutoff : float, optional
        Radius of the neighbor list of the input graphs. The model evaluates
        all pairs and does not read the list.
    cache_dir : str or Path, optional
        Model cache holding the downloaded element parameters.
    use_fast : bool or "auto", optional
        ``False``: the eager terms, differentiable to any order. ``True``: the
        fused kernels of :mod:`~xnn.ffnn.models.gfnff_kernels` on CPUs and
        GPUs, the same energies, forces and stress to rounding, differentiable
        twice (their tangent kernels give the second derivatives of a force or
        stress loss). ``"auto"`` (the default): the kernels in evaluation and
        training mode where they can run, the eager terms otherwise.
    skin : float, optional
        Margin in Angstrom of the hydrogen- and halogen-bond candidates.
    fixed_lists : bool, optional
        Keep the hydrogen- and halogen-bond rows of the given structure for
        every conformation instead.
    cell : array-like, optional
        Lattice vectors in Angstrom, ``(3, 3)`` rows, for a structure periodic
        in three dimensions; the topology is that of the unit cell with its
        images.

    Raises
    ------
    NotImplementedError
        For an element or a bonding situation the rules do not cover.
    """

    @_plain_pages
    def __init__(self, atomic_numbers: Sequence[int], positions, charge: int = 0, *, cutoff: float = 1.0,
                 cache_dir: Optional[Union[str, Path]] = None, use_fast: Union[bool, str] = "auto",
                 skin: float = 1.0, fixed_lists: bool = False, cell=None):
        super().__init__()
        self.cutoff, self.skin, self.fixed_lists = float(cutoff), float(skin), bool(fixed_lists)
        self.use_fast = resolve_use_fast(use_fast)
        self._plan = None
        z = [int(v) for v in np.asarray(atomic_numbers).reshape(-1)]
        x = np.asarray(torch.as_tensor(positions).detach().cpu().numpy() if torch.is_tensor(positions) else positions,
                       dtype=float).reshape(-1, 3)
        missing = sorted(set(z) - lib.SUPPORTED)
        if missing:
            raise NotImplementedError(f"GFN-FF: no rules for the elements {missing}")
        lib.parameters(cache_dir)
        self.charge, self.num_atoms = int(charge), len(z)
        if torch.is_tensor(cell):
            cell = cell.detach().cpu().numpy()
        self.cell = None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3)
        # the collector pauses: the rows make many small long-lived objects that it would rescan again and again
        collect = gc.isenabled()
        gc.disable()
        try:
            self.topology = t = lib.Topology(z, x, self.charge, cell=self.cell)
            self._bind(t)
            self._bind_lists(x / lib.BOHR)
        except (KeyError, StopIteration) as err:
            raise NotImplementedError(f"GFN-FF: no rule for {err} in this structure") from err
        finally:
            if collect:
                gc.enable()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_plan"] = None
        return state

    def fused(self, positions: Tensor):
        """The launches of the fused kernels when this evaluation takes the fast path, else ``None``."""
        if self.use_fast is False:
            return None
        if positions.device.type not in ("cpu", "cuda"):
            return None
        if self._plan and self._plan.device != positions.device:
            self._plan = None
        if self._plan is None:
            from xnn.common.cpu_kernels import Unsupported

            self._plan = False
            try:
                from . import gfnff_kernels

                if gfnff_kernels.available(positions.device) and self.rcov.device == positions.device:
                    plan = gfnff_kernels.Plan if self.cell is None else gfnff_kernels.PeriodicPlan
                    self._plan = _plain_pages(plan)(self)
            except (ImportError, Unsupported):
                pass
        plan = self._plan or None
        if plan is not None and self.training and not _tangents(plan):
            return None
        return plan

    def energy_and_gradient(self, positions: Tensor, cell: Optional[Tensor] = None) -> tuple:
        """Term energies and the gradient of their sum for one conformation, by the fused kernels where
        :meth:`fused` selects them and by the eager terms otherwise.

        Parameters
        ----------
        positions : Tensor
            Positions in Angstrom, ``(N, 3)``.
        cell : Tensor, optional
            Lattice vectors in Angstrom of a periodic model, ``(3, 3)`` rows; the bound cell by default.

        Returns
        -------
        tuple of Tensor
            The energies ``(9,)`` in Hartree in the order of :data:`TERMS`, and the gradient ``(N, 3)`` in Hartree
            per Angstrom; a periodic model adds the gradient in the cell at fixed positions ``(3, 3)`` in Hartree per
            Angstrom.
        """
        self.follow(positions)
        plan = self.fused(positions)
        if self.cell is not None:
            c = self.cell_ref if cell is None else cell
            if plan is not None:
                return plan(positions, c)
            with torch.enable_grad():
                x = positions.detach().requires_grad_(True)
                h = c.detach().to(self.rcov.dtype).requires_grad_(True)
                terms = self.term_energies(x, h)
                grad, gcell = torch.autograd.grad(sum(terms.values()), (x, h))
            return torch.stack([terms[k].detach() for k in TERMS]), grad, gcell
        if plan is not None:
            return plan(positions)
        with torch.enable_grad():
            x = positions.detach().requires_grad_(True)
            terms = self.term_energies(x)
            (grad,) = torch.autograd.grad(sum(terms.values()), x)
        return torch.stack([terms[k].detach() for k in TERMS]), grad

    def _set(self, name: str, value, dtype=torch.float64, width: int = 0) -> Tensor:
        t = torch.as_tensor(np.asarray(value), dtype=dtype) if not torch.is_tensor(value) else value.to(dtype)
        t = t.reshape(-1, width) if width else t
        ref = getattr(self, "rcov", None)
        if ref is not None:
            # buffers regenerated after a cast or a move follow the model
            t = t.to(device=ref.device, dtype=ref.dtype if t.is_floating_point() else t.dtype)
        self.register_buffer(name, t, persistent=False)
        return t

    def _bind(self, t: "lib.Topology") -> None:
        """Turn the rows of every term into buffers."""
        z, q, n = t.z, t.qt, len(t.z)
        self._ext = {}
        zt = self._set("numbers", z, torch.long)
        long, f = torch.long, lambda vals: np.asarray(vals, dtype=float).reshape(-1)  # noqa: E731
        self._set("rcov", lib.RCOV[np.asarray(z)])
        rows = lib.bond_rows(t, [])
        self._set("b_ij", self._e_rows(rows, 2), long, 2)
        for name, c in (("b_sh", 2), ("b_fp", 3), ("b_al", 4), ("b_k", 5)):
            self._set(name, f([r[c] for r in rows]))

        self._set("b_ra", f([lib.BOND_RAD[z[r[0]]][0] + lib.BOND_RAD[z[r[1]]][0] for r in rows]))
        self._set("b_ci", f([lib.BOND_RAD[z[r[0]]][1] for r in rows]))
        self._set("b_cj", f([lib.BOND_RAD[z[r[1]]][1] for r in rows]))
        # bonds of an H to N or O, whose exponent softens with the acceptors of that H
        self._bond_h = [(ib, r[6]) for ib, r in enumerate(rows)
                        if r[6] >= 0 and z[r[0] + r[1] - r[6]] in (7, 8)]
        ang = lib.angle_params(t)
        for c, a, b, t0, _ in ang:
            if t0 is None:
                raise KeyError(f"the angle {z[a]}-{z[c]}-{z[b]}")
        self._set("a_idx", self._e_rows(ang, 3), long, 3)
        self._set("a_t0", f([r[3] for r in ang]))
        self._set("a_k", f([r[4] for r in ang]))
        self._set("arad", f([lib.ANGLE_RAD[a] / lib.BOHR for a in z]))
        tor = lib.torsion_rows(t)
        for i, j, k, l, _, _, kt in tor:
            if math.isnan(kt):
                raise KeyError(f"the torsion {z[i]}-{z[j]}-{z[k]}-{z[l]}")
        self._set("t_idx", self._e_rows(tor, 4), long, 4)
        self._set("t_n", f([r[4] for r in tor]))
        self._set("t_p0", f([math.radians(r[5]) for r in tor]))
        self._set("t_k", f([r[6] for r in tor]))
        inv = lib.inversion_rows(t)
        self._set("i_idx", self._e_rows(inv, 4), long, 4)
        self._set("i_neg", [r[4] < 0 for r in inv], torch.bool)
        self._set("i_c0", f([math.cos(math.radians(r[5])) for r in inv]))
        self._set("i_k", f([r[6] for r in inv]))
        if self.cell is None:
            pairs, rep_c, rep_m = lib.repulsion_rows(t)
        else:
            from . import gfnff_periodic as per

            # image pairs out to the largest distance at which a pair still enters (its exponent drops below 60),
            # an atom of the cell and the lattice shift of its partner
            ri, rj, rs, rep_c, rep_m = per.repulsion_rows(t)
            pairs = np.stack([ri, rj], 1)
            self._set("r_sh", rs.astype(float).reshape(-1, 3))
        self._set("r_ij", pairs, long, 2)
        self._set("r_c", rep_c)
        self._set("r_m", rep_m)
        self._set("r_cut", (REPULSION_DROP / rep_m) ** (2.0 / 3.0))
        atm = lib.batm_rows(t)
        self._set("m_idx", self._e_rows(atm, 3), long, 3)
        self._set("m_c", f([r[3] for r in atm]))
        self._bind_dispersion(t, zt)
        if self.cell is not None:
            keys = list(self._ext)  # in the order of their extended indices
            self._set("x_base", [k[0] for k in keys], long)
            self._set("x_shift", np.array([k[1] for k in keys], dtype=float).reshape(-1, 3))
            self._set("x_wrap", t.wrap)
            self._set("cell_ref", self.cell)
            # the coordination numbers count the images within twice the largest pair radius
            cut = 2.0 * (4.0 / 3.0) * 2.0 * float(lib.RCOV[np.asarray(z)].max()) * lib.BOHR
            self._set("cn_shift", np.array(lib.shifts_within(self.cell, cut), dtype=float).reshape(-1, 3))
        za = np.asarray(z)
        en, jj, kappa, alp = (lib.PAR[za - 1, c] for c in (0, 1, 2, 3))
        alpha = alp + np.array([lib.CA[13 if lib.atom_class(t, i) == 2 else a] for i, a in enumerate(z)]) * q
        self._set("e_diag", jj + lib.hardness_shift(t) * q + math.sqrt(2.0 / math.pi) / alpha)
        a2 = torch.as_tensor(alpha, dtype=torch.float64) ** 2
        self._set("e_g", 1.0 / torch.sqrt(a2[:, None] + a2[None, :]))
        self._set("e_x0", -(en + lib.en_shifts(z, t.nb, t.hyb)))
        self._set("e_kappa", kappa)
        _, comps = lib.components(n, t.edges)
        con = np.zeros((len(comps), n))
        for k, frag in enumerate(comps):
            con[k, frag] = 1.0
        self._set("e_con", con)
        self._set("e_tot", lib.fragment_charges(comps, t.charge))

    def _e(self, a) -> int:
        """Index of an atom of a row: the atom of the unit cell, or an extended atom for an image."""
        s = getattr(a, "s", (0, 0, 0))
        if self.cell is None or not any(s):
            return int(a)
        key = (int(a), s)
        if key not in self._ext:
            self._ext[key] = self.num_atoms + len(self._ext)
        return self._ext[key]

    def _e_rows(self, rows, width: int):
        """The indices of :meth:`_e` for the first ``width`` atoms of every row (lists of a molecule, an integer
        array ``(rows, width)`` of a periodic structure)."""
        if self.cell is None:
            return [[int(v) for v in r[:width]] for r in rows]
        if not len(rows):
            return np.zeros((0, width), dtype=np.int64)
        # the distinct atom objects of the rows (the rows share them) in the order they first appear
        cols, m = list(zip(*rows))[:width], len(rows)
        ids = np.stack([np.fromiter(map(id, c), dtype=np.int64, count=m) for c in cols], 1).reshape(-1)
        _, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        ext = np.empty(len(first), dtype=np.int64)
        table, base, img_t = self._ext, self.num_atoms, lib.Img
        order = np.argsort(first)
        for u, f in zip(order.tolist(), first[order].tolist()):
            v = cols[f % width][f // width]
            s = v.s if v.__class__ is img_t else None
            if s is None or not any(s):
                ext[u] = int(v)
                continue
            # an image: the next extended atom when new
            key = (int(v), s)
            e = table.get(key)
            if e is None:
                e = table[key] = base + len(table)
            ext[u] = e
        return ext[inv.reshape(-1)].reshape(-1, width)

    def _bind_dispersion(self, t: "lib.Topology", zt: Tensor) -> None:
        d4 = _d4_tables()
        z = np.asarray(t.z)
        hardness = d4.hardness[zt].double().numpy()
        # effective nuclear charges of the def2 core potentials (28, 46 and 60 core electrons; 28 for Ce to Lu)
        zeff = np.where(z > 71, z - 60, np.where(z > 57, z - 28, np.where(z > 54, z - 46, np.where(z > 36, z - 28, z))))
        zeff = zeff.astype(float)
        zeta = torch.as_tensor(np.exp(3.0 * (1.0 - np.exp(hardness * (1.0 - zeff / (zeff + t.qt))))))
        self._set("d_zz", zeta[:, None] * zeta[None, :])
        ref = self._set("d_ref", lib.REFCN[:, z].T)
        self._set("d_valid", torch.arange(ref.shape[1])[None, :] < d4.nref[zt][:, None], torch.bool)
        self._set("d_alpha", d4.refalpha[:, :, zt])
        self._set("d_w", d4.cp_weights.clone())
        r4 = d4.r4r2[zt].double()
        dq = self._set("d_q", 3.0 * r4[:, None] * r4[None, :])
        r0 = 0.58 * torch.sqrt(dq) + 4.8
        self._set("d_r06", r0 ** 6)
        self._set("d_r08", r0 ** 8)

    def follow(self, positions: Tensor) -> None:
        """Regenerate the hydrogen- and halogen-bond candidates when an atom of ``positions`` (Angstrom) has moved
        by more than half the skin since they were generated."""
        if self.fixed_lists:
            return
        # numpy is cheaper than torch for this one reduction
        x = positions.detach().cpu().double().numpy()
        d = x - self._ref
        if 4.0 * float((d * d).sum(1).max()) <= self.skin ** 2:
            return
        self._bind_lists(x / lib.BOHR)
        self._plan = None

    @_plain_pages
    def _bind_lists(self, x: np.ndarray) -> None:
        """The hydrogen- and halogen-bond rows within the skin of their cuts at the positions ``x`` (Bohr), or at
        their cuts with fixed lists."""
        if self.cell is not None:
            self._bind_periodic_lists(x)
            return
        t, z, q = self.topology, self.topology.z, self.topology.qt
        long, f = torch.long, lambda vals: np.asarray(vals, dtype=float).reshape(-1)  # noqa: E731
        s = 0.0 if self.fixed_lists else self.skin / lib.BOHR
        # a displacement below s / 2 per atom changes a distance by less than s and the root of the sum of three
        # squared distances by less than sqrt(3) s
        widen = lambda cut, k: (math.sqrt(cut) + math.sqrt(k) * s) ** 2  # noqa: E731
        cuts = (lib.HB_CUT, lib.HB_CUT3, lib.XB_CUT)
        self.list_cuts = (1e30,) * 3 if self.fixed_lists else cuts
        hbb = lib.hbb_list(t, x, widen(lib.HB_CUT, 1))
        accs = [(ib, h, b) for ib, h in self._bond_h
                for b in sorted({b_ for _, b_, h_ in hbb if h_ == h and z[b_] in (7, 8)})]
        self._set("b_acc", accs, long, 3)
        self._set("b_accd", f([1.78 * (4.0 / 3.0) * (lib.RCOV[1] + lib.RCOV[z[b]]) for _, _, b in accs]))
        self._bind_hb(x, widen(lib.HB_CUT, 1), widen(lib.HB_CUT3, 3))
        xb = lib.xb_rows(t, x, widen(lib.XB_CUT, 1))
        self._set("x_idx", xb, long, 3)
        self._set("x_rad", f([lib.TOPO_RAD[z[d]] + lib.TOPO_RAD[z[y]] for d, y, xa in xb]))
        self._set("x_p", f([lib.XB_C[z[xa]] / ((1.0 + 0.03 * math.exp(-15.0 * q[xa]))
                                              * (1.0 + 0.03 * math.exp(15.0 * q[y]))) for d, y, xa in xb]))
        self._ref = x * lib.BOHR

    def _bind_periodic_lists(self, x: np.ndarray) -> None:
        """The hydrogen- and halogen-bond rows over the images and the acceptors of the H-N and H-O bonds at the
        positions ``x`` (Bohr) of a periodic structure."""
        from . import gfnff_periodic as per

        cellb = self.cell / lib.BOHR
        xw = x - np.asarray(self.topology.wrap, dtype=float) @ cellb
        self._hbxb_rows = per.bind(self, xw, 0.0 if self.fixed_lists else self.skin / lib.BOHR)
        accs, dd = per.bond_acceptors(self, self._hbxb_rows)
        # the acceptors may add images to the extended atoms
        keys = list(self._ext)  # in the order of their extended indices
        self._set("x_base", [k[0] for k in keys], torch.long)
        self._set("x_shift", np.array([k[1] for k in keys], dtype=float).reshape(-1, 3))
        self._set("b_acc", accs, torch.long, 3)
        self._set("b_accd", np.asarray(dd, dtype=float).reshape(-1))
        self._ref = x * lib.BOHR

    def _hb_atoms(self) -> dict:
        """Per-atom tables of the hydrogen-bond rows (cached): radius, acidity, basicity, charge factors, acceptor
        class, neighbour tables."""
        if getattr(self, "_hb_tables", None) is None:
            t = self.topology
            z, q, nb, n = t.z, t.qt, lib.hb_neighbours(t), len(t.z)
            act = [lib.hb_active(z, nb, q, i) for i in range(n)]
            sig = lambda v: 1.0 / (1.0 + math.exp(v))  # noqa: E731
            per_atom = lambda fn: np.array([fn(i) for i in range(n)], dtype=float)  # noqa: E731
            o1 = np.array([lib.o1_class(z, nb, i) for i in range(n)], dtype=int)
            # the neighbours of the one neighbour of an O1 acceptor, without the acceptor
            x1 = [next(iter(nb[i])) if o1[i] else -1 for i in range(n)]
            y1, ym = _padded([[y for y in sorted(nb[x1[i]]) if y != i] if o1[i] else [] for i in range(n)])
            nbi, nbm = _padded([sorted(v) for v in nb])
            self._hb_tables = {
                "rad": per_atom(lambda i: lib.HB_RAD.get(z[i], 0.0)),
                "ac": per_atom(lambda i: lib.acidity(z, nb, i) if act[i] else 0.0),
                "bs": per_atom(lambda i: lib.basicity(z, nb, i) if act[i] else 0.0),
                "sq": per_atom(lambda i: sig(15.0 * q[i])), "sh": per_atom(lambda i: sig(-15.0 * q[i])),
                "o1": o1, "x1": np.array(x1, dtype=np.int64), "y1": y1, "y1m": ym, "nb": nbi, "nbm": nbm,
                "cnb": per_atom(lambda i: 2.0 if z[i] == 7 and len(nb[i]) == 1 else 11.2),
                "n2": np.array([z[i] == 7 and len(nb[i]) == 2 and not o1[i] for i in range(n)]),
            }
        return self._hb_tables

    def _bind_hb(self, x: np.ndarray, cut: float, cut3: float) -> None:
        t = self.topology
        z, q, nb = t.z, t.qt, lib.hb_neighbours(t)
        bb, bl = lib.hb_rows(z, nb, q, x, cut, cut3)
        at, long = self._hb_atoms(), torch.long
        a, b, h = np.asarray(bb, dtype=np.int64).reshape(-1, 3).T
        self._set("h_bb", np.stack([a, b, h], axis=1), long, 3)
        rr = at["rad"][a] + at["rad"][b]
        self._set("h_rr", rr)
        o1 = at["o1"][b]
        self._set("h_p", np.where(o1 > 0, 1.75, 1.34) * at["ac"][a] * at["bs"][b] * at["sq"][a] * at["sq"][b]
                  * at["sh"][h])
        self._set("h_cnb", at["cnb"][b] / rr)
        self._set("h_nb", at["nb"][b], long)
        self._set("h_nbm", at["nbm"][b], torch.bool)
        r1 = np.flatnonzero(o1 > 0)
        self._set("h_o1", r1, long)
        self._set("h_o1x", at["x1"][b[r1]], long)
        self._set("h_o1y", at["y1"][b[r1]], long)
        self._set("h_o1ym", at["y1m"][b[r1]], torch.bool)
        r2 = np.flatnonzero(at["n2"][b])
        self._set("h_n2", r2, long)
        self._set("h_n2nb", at["nb"][b[r2], :2], long, 2)
        la, lb, lh = np.asarray(bl, dtype=np.int64).reshape(-1, 3).T
        self._set("h_bl", np.stack([la, lb, lh], axis=1), long, 3)
        rad, ac, ba = at["rad"], at["ac"], at["bs"] * at["sq"]
        self._set("l_rr", rad[la] + rad[lb])
        self._set("l_ca", ac[la])
        self._set("l_cb", ac[lb])
        self._set("l_ba", ba[la])
        self._set("l_bb", ba[lb])
        self._set("l_qh", at["sh"][lh])

    @classmethod
    def from_atoms(cls, structure, *, charge: int = 0, **kwargs) -> "GFNFF":
        """Build the force field of a structure.

        Parameters
        ----------
        structure : object
            An ``ase.Atoms`` or a ``(positions, atomic_numbers)`` pair. The cell of an ``ase.Atoms`` that is periodic
            in all three directions is used unless ``cell`` is given.
        charge : int, optional
            Total charge.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        GFNFF
            The model.
        """
        numbers, positions = _structure(structure)
        pbc = getattr(structure, "pbc", None)
        if "cell" not in kwargs and pbc is not None and all(bool(p) for p in pbc):
            kwargs["cell"] = np.asarray(structure.get_cell(), dtype=float)
        return cls(numbers, positions, charge, **kwargs)

    @classmethod
    def from_config(cls, cfg) -> "GFNFF":
        """Construct the model from a core model config.

        ``cfg.extra`` holds ``atomic_numbers`` and ``positions`` (the
        structure that defines the topology) and optionally ``charge``,
        ``use_fast``, ``skin``, ``fixed_lists`` and ``cell`` (the lattice
        vectors of a periodic structure, one per row).

        Parameters
        ----------
        cfg : xnn.common.config.schema.ModelConfig
            The core model config.

        Returns
        -------
        GFNFF
            The model.

        Raises
        ------
        ValueError
            If the structure is missing.
        """
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        if extra.get("atomic_numbers") is None or extra.get("positions") is None:
            raise ValueError("GFN-FF needs 'atomic_numbers' and 'positions' of the molecule that defines its "
                             "topology in the model options")
        return cls(extra["atomic_numbers"], extra["positions"], int(extra.get("charge", 0)), cutoff=float(cfg.cutoff),
                   use_fast=extra.get("use_fast", "auto"), skin=float(extra.get("skin", 1.0)),
                   fixed_lists=bool(extra.get("fixed_lists", False)), cell=extra.get("cell"))

    def forms_forces(self, data: AtomicGraph) -> bool:
        """Whether :meth:`forward_direct` serves this evaluation: always in evaluation mode, unless the positions
        carry an autograd graph that the outputs must continue."""
        return not self.training and not (torch.is_grad_enabled() and data.pos.requires_grad)

    def forward_direct(self, data: AtomicGraph) -> dict:
        """:meth:`forward` with the forces ``"forces"`` and the strain derivative ``"virial"`` of
        :meth:`energy_and_gradient` (see :func:`~xnn.ffnn.models.gfnff_export.direct_outputs`); the cell of a
        periodic structure is the one of the graph. The outputs carry no autograd graph."""
        from .gfnff_export import direct_outputs

        return direct_outputs(self, data)

    def export_core(self) -> torch.nn.Module:
        """The scriptable core of :func:`~xnn.common.deploy.export_torchscript_potential`: the bound molecule or
        cluster (:class:`~xnn.ffnn.models.gfnff_export.GFNFFExport`) or periodic cell
        (:class:`~xnn.ffnn.models.gfnff_export.GFNFFCellExport`)."""
        from .gfnff_export import GFNFFCellExport, GFNFFExport

        return GFNFFExport(self) if self.cell is None else GFNFFCellExport(self)

    def term_energies(self, positions: Tensor, cell: Optional[Tensor] = None) -> dict:
        """Energy terms (Hartree) of one conformation.

        Parameters
        ----------
        positions : Tensor
            Positions in Angstrom, ``(N, 3)``.
        cell : Tensor, optional
            Lattice vectors in Angstrom of a periodic model, ``(3, 3)`` rows; the bound cell by default.

        Returns
        -------
        dict of str to Tensor
            One scalar per entry of :data:`TERMS`.
        """
        if self.cell is not None:
            return self._periodic_energies(positions, cell)
        self.follow(positions)
        # the precision of the model's buffers (float64 unless the model was cast)
        x = positions.to(self.rcov.dtype) / lib.BOHR
        n = self.num_atoms
        eye = torch.eye(n, dtype=x.dtype, device=x.device)
        dist = torch.sqrt(sum((x[:, c, None] - x[None, :, c]) ** 2 for c in range(3)) + eye)
        rc = (4.0 / 3.0) * (self.rcov[:, None] + self.rcov[None, :])
        i, j = torch.nonzero((dist.detach() < 2.0 * rc) & (eye == 0), as_tuple=True)
        count = torch.zeros(n, dtype=x.dtype, device=x.device).index_add(
            0, i, 0.5 * (1.0 + torch.erf(-7.5 * (dist[i, j] / rc[i, j] - 1.0))))
        logcn = math.log(1.0 + math.exp(4.4)) - torch.log(1.0 + torch.exp(4.4 - count))
        out = {"bond": self._e_bond(dist, logcn), "angle": self._e_angle(x, dist), "torsion": self._e_torsion(x, dist)}
        i, j = self.r_ij.T
        keep = dist.detach()[i, j] < self.r_cut
        rr, rep_m = dist[i[keep], j[keep]], self.r_m[keep]
        out["repulsion"] = (self.r_c[keep] / rr * torch.exp(-rep_m * rr * torch.sqrt(rr))).sum()
        i, j, k = self.m_idx.T
        rij, rik, rjk = dist[i, j], dist[i, k], dist[j, k]
        ci = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik)
        cj = (rij ** 2 + rjk ** 2 - rik ** 2) / (2 * rij * rjk)
        ck = (rik ** 2 + rjk ** 2 - rij ** 2) / (2 * rik * rjk)
        out["batm"] = (self.m_c * (3.0 * ci * cj * ck + 1.0) / (rij * rik * rjk) ** 3).sum()
        g = torch.exp(-4.0 * (logcn[:, None] - self.d_ref) ** 2)
        g = torch.where(self.d_valid, g, torch.zeros_like(g))
        alpha = torch.einsum("nr,krn->nk", g / g.sum(1, keepdim=True), self.d_alpha)
        c6 = (3.0 / math.pi) * (alpha * self.d_w) @ alpha.T
        i, j = torch.nonzero(torch.triu(dist < DISPERSION_CUT, 1), as_tuple=True)
        r2 = dist[i, j] ** 2
        r6 = r2 * r2 * r2
        pair = 1.0 / (r6 + self.d_r06[i, j]) + 2.0 * self.d_q[i, j] / (r6 * r2 + self.d_r08[i, j])
        pair = self.d_zz[i, j] * c6[i, j] * pair
        out["dispersion"] = -pair.sum()
        out["hb"] = self._e_hbb(x, dist) + self._e_hbl(dist)
        d, y, xa = self.x_idx.T
        rdx, rxy, rdy = dist[d, xa], dist[xa, y], dist[d, y]
        arg = 70.0 * ((rdx + rxy) / rdy - 1.0)
        on = (arg < lib.HB_SWITCH_CUT) & (rdy.detach() ** 2 < self.list_cuts[2])
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(lib.HB_SWITCH_CUT))), torch.zeros_like(arg))
        srt = 1.0 / (1.0 + (5.0 * self.x_rad / rxy ** 2) ** 6)
        out["xb"] = (-self.x_p * o * srt / (1.0 + (rxy ** 2 / 70.0) ** 6) / rxy ** 3).sum()
        amat = torch.erf(self.e_g * dist) / dist * (1.0 - eye) + torch.diag(self.e_diag)
        # an isolated ion has a count of exactly zero: its square root enters with a zero derivative, as in the kernels
        pos = logcn > 0
        xx = self.e_x0 + self.e_kappa * torch.where(pos, torch.sqrt(torch.where(pos, logcn, torch.ones_like(logcn))),
                                                    torch.zeros_like(logcn))
        nc = self.e_tot.shape[0]
        zero = torch.zeros((nc, nc), dtype=x.dtype, device=x.device)
        mat = torch.cat([torch.cat([amat, self.e_con.T], 1), torch.cat([self.e_con, zero], 1)], 0)
        sol = torch.linalg.solve(mat, torch.cat([xx, self.e_tot]))
        q, lam = sol[:n], sol[n:]
        # the Lagrangian, stationary in the charges and multipliers: errors of the solve enter only to second order
        out["electrostat"] = -(q * xx).sum() + 0.5 * q @ amat @ q + lam @ (self.e_con @ q - self.e_tot)
        return out

    def _periodic_energies(self, positions: Tensor, cell: Optional[Tensor]) -> dict:
        """Every energy term (Hartree) of a periodic structure; the nonbonded terms run over the images."""
        from . import gfnff_periodic as per

        self.follow(positions)
        out = self._periodic_terms(positions, cell)
        cellb = (self.cell_ref if cell is None else cell.to(self.rcov.dtype)) / lib.BOHR
        x = positions.to(self.rcov.dtype) / lib.BOHR - self.x_wrap @ cellb
        logcn = per.counts(self, x, cellb)
        out["dispersion"] = per.dispersion(self, x, cellb, logcn)
        out.update(per.hb_xb(self._hbxb_rows, x, cellb))
        out["electrostat"] = per.electrostatics(self, x, cellb, logcn)
        return out

    def _periodic_terms(self, positions: Tensor, cell: Optional[Tensor]) -> dict:
        """Bonded terms and repulsion (Hartree) of a periodic structure: bond, angle, torsion, repulsion and the bonded
        three-body term."""
        cellb = (self.cell_ref if cell is None else cell.to(self.rcov.dtype)) / lib.BOHR
        # the atoms wrapped into the cell as the topology saw them
        x = positions.to(self.rcov.dtype) / lib.BOHR - self.x_wrap @ cellb
        n = self.num_atoms
        # coordination numbers over the images
        tr = self.cn_shift @ cellb
        dv = x[None, :, None, :] + tr[None, None, :, :] - x[:, None, None, :]
        own = (torch.eye(n, dtype=torch.bool, device=x.device)[:, :, None]
               & (self.cn_shift.abs().sum(-1) == 0)[None, None, :])
        d = torch.sqrt((dv ** 2).sum(-1) + own.to(x.dtype))
        rc = ((4.0 / 3.0) * (self.rcov[:, None] + self.rcov[None, :]))[:, :, None]
        on = (d.detach() < 2.0 * rc) & ~own
        f = torch.where(on, 0.5 * (1.0 + torch.erf(-7.5 * (d / rc - 1.0))), torch.zeros_like(d))
        # an atom and its own images enter from both sides
        f = f * (1.0 + torch.eye(n, dtype=f.dtype, device=f.device)[:, :, None])
        count = f.sum((1, 2))
        logcn = math.log(1.0 + math.exp(4.4)) - torch.log(1.0 + torch.exp(4.4 - count))
        # the atoms of the cell and the images the rows reach
        xe = torch.cat([x, x[self.x_base] + self.x_shift @ cellb])
        base = torch.cat([torch.arange(n, device=x.device), self.x_base])
        dist = _PairDistance(xe)
        rad = self.arad[base]
        out = {"bond": self._e_bond(dist, logcn[base]), "angle": self._e_angle(xe, dist, rad),
               "torsion": self._e_torsion(xe, dist, rad)}
        i, j = self.r_ij.T
        rr = (x[j] + self.r_sh @ cellb - x[i]).norm(dim=-1)
        keep = rr.detach() < self.r_cut
        rr, rep_m = rr[keep], self.r_m[keep]
        out["repulsion"] = (self.r_c[keep] / rr * torch.exp(-rep_m * rr * torch.sqrt(rr))).sum()
        i, j, k = self.m_idx.T
        rij, rik, rjk = dist[i, j], dist[i, k], dist[j, k]
        ci = (rij ** 2 + rik ** 2 - rjk ** 2) / (2 * rij * rik)
        cj = (rij ** 2 + rjk ** 2 - rik ** 2) / (2 * rij * rjk)
        ck = (rik ** 2 + rjk ** 2 - rij ** 2) / (2 * rik * rjk)
        out["batm"] = (self.m_c * (3.0 * ci * cj * ck + 1.0) / (rij * rik * rjk) ** 3).sum()
        return out

    def _e_bond(self, dist: Tensor, logcn: Tensor) -> Tensor:
        i, j = self.b_ij.T
        r0 = self.b_fp * (self.b_ra + self.b_ci * logcn[i] + self.b_cj * logcn[j] + self.b_sh)
        ib, h, b = self.b_acc.T
        m = torch.zeros_like(r0).index_add(0, ib, 0.5 * (1.0 + torch.erf(-27.5 * (dist[h, b] / self.b_accd - 1.0))))
        return (self.b_k * torch.exp(-self.b_al * (1.0 - 0.1 * m) * (dist[i, j] - r0) ** 2)).sum()

    def _e_angle(self, x: Tensor, dist: Tensor, rad: Optional[Tensor] = None) -> Tensor:
        c, a, b = self.a_idx.T
        cos = ((x[a] - x[c]) * (x[b] - x[c])).sum(-1) / (dist[c, a] * dist[c, b])
        rad = self.arad if rad is None else rad
        damp = 1.0 / (1.0 + lib.ANGLE_DAMP * (dist[c, a] / (rad[c] + rad[a])) ** 4) \
            / (1.0 + lib.ANGLE_DAMP * (dist[c, b] / (rad[c] + rad[b])) ** 4)
        lin = (self.a_t0 - math.pi).abs() < 1e-6
        u, v = x[a[lin]] - x[c[lin]], x[b[lin]] - x[c[lin]]
        # pi minus the angle from the cross and dot products, well conditioned near 180 degrees
        e_lin = torch.atan2(torch.linalg.cross(u, v).norm(dim=-1), -(u * v).sum(-1)) ** 2
        e_ang = torch.where(lin, torch.zeros_like(cos), (cos - torch.cos(self.a_t0)) ** 2).masked_scatter(lin, e_lin)
        return (self.a_k * e_ang * damp).sum()

    def _e_torsion(self, x: Tensor, dist: Tensor, rad: Optional[Tensor] = None) -> Tensor:
        rad = self.arad if rad is None else rad
        td = lambda p, q: 1.0 / (1.0 + lib.TORS_DAMP * (dist[p, q] / (rad[p] + rad[q])) ** 4)  # noqa: E731
        # the dihedral and the damping pairs follow the stored order (a, c, b, d) of a row
        a, b, c, l = self.t_idx.T
        b0, b1, b2 = x[c] - x[a], x[b] - x[c], x[l] - x[b]
        n1, n2 = torch.linalg.cross(b0, b1), torch.linalg.cross(b1, b2)
        mm = torch.linalg.cross(n1, b1 / b1.norm(dim=-1, keepdim=True))
        phi = torch.atan2((mm * n2).sum(-1), (n1 * n2).sum(-1))
        e_t = self.t_k * td(a, c) * td(c, b) * td(b, l) * (1.0 + torch.cos(self.t_n * (phi - self.t_p0) + math.pi))
        c, j, k, l = self.i_idx.T
        nrm, v = torch.linalg.cross(x[j] - x[c], x[k] - x[c]), x[l] - x[c]
        sw = (nrm * v).sum(-1) / (nrm.norm(dim=-1) * v.norm(dim=-1))
        cw = torch.sqrt((1.0 - sw ** 2).clamp_min(0.0))
        e_i = self.i_k * td(c, j) * td(j, k) * td(j, l) * torch.where(self.i_neg, (cw - self.i_c0) ** 2, 1.0 - cw)
        return e_t.sum() + e_i.sum()

    @staticmethod
    def _hb_damp(rr: Tensor, rab: Tensor) -> Tensor:
        return 1.0 / (1.0 + (22.0 * rr / rab ** 2) ** 6) / (1.0 + (rab ** 2 / 85.0) ** 6)

    def _e_hbb(self, x: Tensor, dist: Tensor) -> Tensor:
        """Hydrogen bonds of H atoms bonded to the donor."""
        a, b, h = self.h_bb.T
        rr, rab, rah, rbh = self.h_rr, dist[a, b], dist[a, h], dist[b, h]
        arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
        # rows beyond the distance cut or whose switch argument reaches HB_SWITCH_CUT contribute nothing
        on = (arg < lib.HB_SWITCH_CUT) & (rab.detach() ** 2 < self.list_cuts[0])
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(lib.HB_SWITCH_CUT))), torch.zeros_like(arg))
        arm = (dist[a[:, None], self.h_nb] + dist[b[:, None], self.h_nb]) / rab[:, None]
        fac = 2.0 / (1.0 + torch.exp(-self.h_cnb[:, None] * (arm - 1.0))) - 1.0
        o = o * torch.where(self.h_nbm, fac, torch.ones_like(fac)).prod(1)
        if self.h_o1.shape[0]:
            r, xb = self.h_o1, self.h_o1x
            u, v = x[xb] - x[b[r]], x[h[r]] - x[b[r]]
            cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
            f1 = 1.0 - 16.0 / 45.0 * (cos + 0.5) ** 2
            b1 = (u / u.norm(dim=-1, keepdim=True))[:, None, :]
            b2 = x[self.h_o1y] - x[xb][:, None, :]
            vv = (v[:, None, :] - (v[:, None, :] * b1).sum(-1, keepdim=True) * b1).expand_as(b2)
            ww = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
            s = (torch.linalg.cross(b1.expand_as(b2), vv) * ww).sum(-1)
            c = (vv * ww).sum(-1)
            den = s ** 2 + c ** 2
            # a neighbour on the acceptor axis counts as perpendicular
            sin2 = torch.where(den > 0, s ** 2 / torch.where(den > 0, den, torch.ones_like(den)), torch.ones_like(den))
            f1 = f1 * torch.where(self.h_o1ym, 1.0 - 0.06 * sin2, torch.ones_like(sin2)).prod(1)
            o = o * torch.ones_like(o).index_copy(0, r, f1)
        if self.h_n2.shape[0]:
            r = self.h_n2
            u = x[b[r]] - 0.5 * (x[self.h_n2nb[:, 0]] + x[self.h_n2nb[:, 1]])
            lp = x[b[r]] + 0.41 * u / u.norm(dim=-1, keepdim=True)
            arg = 56.0 / rr[r] * ((0.41 + (x[a[r]] - lp).norm(dim=-1)) / rab[r] - 1.0)
            o = o * torch.ones_like(o).index_copy(0, r, 2.0 / (1.0 + torch.exp(arg.clamp_max(300.0))))
        return (-self.h_p * o * self._hb_damp(rr, rab) * (1.8 / rbh ** 3 - 0.8 / rab ** 3)).sum()

    def _e_hbl(self, dist: Tensor) -> Tensor:
        """Hydrogen bonds of H atoms bonded to neither or to both partners."""
        a, b, h = self.h_bl.T
        rr, rab, rah, rbh = self.l_rr, dist[a, b], dist[a, h], dist[b, h]
        arg = 49.0 / rr * ((rah + rbh) / rab - 1.0)
        # rows beyond the distance cuts or whose switch argument reaches HB_SWITCH_CUT contribute nothing
        r2 = rab.detach() ** 2
        on = (arg < lib.HB_SWITCH_CUT) & (r2 < self.list_cuts[0]) \
            & (r2 + rah.detach() ** 2 + rbh.detach() ** 2 < self.list_cuts[1])
        o = torch.where(on, 2.0 / (1.0 + torch.exp(arg.clamp_max(lib.HB_SWITCH_CUT))), torch.zeros_like(arg))
        wa, wb = rbh ** 4, rah ** 4
        acid = (self.l_ca * wa + self.l_cb * wb) / (wa + wb)
        base = (self.l_ba * wb + self.l_bb * wa) / (wa + wb)
        return (-5.0 * o * self._hb_damp(rr, rab) * acid * base * self.l_qh / rab ** 3).sum()

    def forward(self, data: AtomicGraph) -> dict:
        """Evaluate the energy of a batch of conformations.

        Parameters
        ----------
        data : AtomicGraph
            The batched graph; every structure is a conformation of the
            molecule the model was built for.

        Returns
        -------
        dict of str to Tensor
            ``energy`` ``(B,)`` and ``node_energy`` ``(N,)`` in eV, and one
            ``e_<term>`` ``(B,)`` per entry of :data:`TERMS`.

        Raises
        ------
        ValueError
            If a structure is not a conformation of the bound molecule.
        """
        n, batch = self.num_atoms, data.batch
        nb = int(data.num_graphs)
        if data.pos.shape[0] != n * nb or not bool((data.atomic_numbers.reshape(nb, n) == self.numbers).all()):
            raise ValueError("every structure must be a conformation of the molecule the GFN-FF model was built for")
        if not bool((batch.reshape(nb, n) == torch.arange(nb, device=batch.device)[:, None]).all()):
            raise ValueError("the atoms of each structure must be contiguous and in the order of the bound molecule")
        pos = data.pos.reshape(nb, n, 3)
        cells = [None] * nb
        if self.cell is not None:
            # the cell of each structure (stress through its strain), the bound cell without one
            c = getattr(data, "cell", None)
            c = None if c is None else c.reshape(-1, 3, 3)
            cells = [self.cell_ref.to(pos.dtype) if c is None or not bool(c[g].any()) else c[g] for g in range(nb)]
        if self.fused(pos) is not None:
            # per-term energies come without a graph; the total carries the kernels' gradient
            fused = []
            for g in range(nb):
                self.follow(pos[g])
                plan = self.fused(pos[g])
                if cells[g] is None:
                    fused.append(_FusedEnergy.apply(pos[g], plan))
                else:
                    fused.append(_FusedPeriodicEnergy.apply(pos[g], cells[g], plan))
            energy = torch.stack([f[0] for f in fused]) * HARTREE
            parts = {k: torch.stack([f[1][j] for f in fused]) * HARTREE for j, k in enumerate(TERMS)}
        else:
            terms = [self.term_energies(pos[g], cells[g]) for g in range(nb)]
            parts = {k: torch.stack([t[k] for t in terms]) * HARTREE for k in TERMS}
            energy = sum(parts.values())
        out = {"energy": energy, "node_energy": (energy / n).repeat_interleave(n)}
        out.update({f"e_{k}": v for k, v in parts.items()})
        return out
