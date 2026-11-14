"""Gradient-domain machine learning (GDML) and its symmetrized form (sGDML).

A global force field for one molecule in a fixed atom order (Chmiela et al.,
Sci. Adv. 3, e1603015, 2017; Nat. Commun. 9, 3887, 2018; Comput. Phys.
Commun. 240, 38, 2019). The molecule is described by its inverse interatomic
distances ``x``; the energy is a kernel expansion whose gradient interpolates the
training forces:

``E(x) = c + s sum_j sum_q [ g(|D|) (D . P_q w_j) + k(|D|) a_j ]``,  ``D = x - P_q x_j``

with ``k`` a Matern kernel, ``g = -k'/r``, the training descriptors ``x_j``, their
weights ``w_j = J_j alpha_j`` (the force coefficients through the descriptor
Jacobian), the optional energy coefficients ``a_j``, the atom permutations
``P_q`` of the molecule's symmetries (sGDML; the identity alone is GDML), the
scale ``s`` of the training forces and the integration constant ``c``. Forces
are the negative gradient of this energy, so the model is conservative.

The coefficients cancel to many digits, so the model keeps them, and evaluates
the kernel expansion, in float64 whatever its dtype: a float32 model rounds only
its inputs and outputs.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

from xnn.common.data import AtomicGraph
from xnn.common.data.elements import atomic_number
from xnn.common.models.base import InteratomicPotential
from xnn.common.models.registry import register_model

from ..featurizers.inverse_distance import InverseDistances
from ..kernels import Matern, safe_norm

#: the buffers whose sizes follow the training set and the symmetry group
_SIZED = ("train_desc", "train_weights", "energy_coef", "perms", "perm_index", "lattice")

#: elements of the ``(B, M, Q, D)`` differences formed at once on a CPU, per thread (2 MB in float64:
#: the passes over them stay in the cache)
CHUNK = 1 << 18
#: the same on a GPU (128 MB in float64), where small chunks leave the device waiting on launches
CHUNK_GPU = 1 << 24


def _chunks(n_train: int, per_train: int, device: torch.device = torch.device("cpu")):
    """Slices of the training points such that each holds about :data:`CHUNK` elements per CPU thread
    (:data:`CHUNK_GPU` on a GPU)."""
    size = CHUNK * torch.get_num_threads() if device.type == "cpu" else CHUNK_GPU
    step = max(1, size // max(1, per_train))
    return [slice(s, min(s + step, n_train)) for s in range(0, n_train, step)]


def _rowdot(a: Tensor, b: Tensor) -> Tensor:
    """Row-wise dot products over the last axis."""
    return (a * b).sum(-1)


def kernel_energy(x: Tensor, xp: Tensor, wp: Tensor, ec: Tensor, kernel: Matern,
                  gradient: bool = False):
    """``sum_{j,q} g(r) (D.w) + k(r) a_j`` for descriptors ``x`` ``(B, D)`` (paper eq. 5).

    ``xp`` and ``wp`` are the ``(M, Q, D)`` permuted training descriptors and
    weights, ``ec`` the ``(M,)`` energy coefficients (possibly empty). With
    ``gradient`` the ``(B, D)`` gradient (:func:`kernel_energy_gradient`) is formed
    in the same pass over the differences and returned as well.
    """
    b, (m, q, d) = x.shape[0], xp.shape
    e = x.new_zeros(b)
    grad = x.new_zeros(b, d) if gradient else None
    for c in _chunks(m, b * q * d, x.device):
        w = wp[c].reshape(-1, d)                          # (mQ, D)
        diff = x[:, None, :] - xp[c].reshape(1, -1, d)    # (B, mQ, D)
        r = torch.linalg.vector_norm(diff, dim=-1)
        g, t = kernel.g_and_t(r) if gradient else (kernel.g(r), None)
        dot = _rowdot(diff, w)
        e = e + (g * dot).sum(-1)
        if ec.numel() > 0:
            e = e + kernel.k(r) @ ec[c].repeat_interleave(q)
        if t is not None:
            coef = t * dot
            if ec.numel() > 0:
                coef = coef - g * ec[c].repeat_interleave(q)
            grad = grad + g @ w + torch.bmm(coef.unsqueeze(1), diff).squeeze(1)
    return (e, grad) if gradient else e


def kernel_energy_gradient(x: Tensor, xp: Tensor, wp: Tensor, ec: Tensor, kernel: Matern) -> Tensor:
    """``(B, D)`` gradient of :func:`kernel_energy` by the descriptors (paper eq. 4).

    ``sum_{j,q} [g(r) w + t(r) (D . w) D - g(r) a_j D]`` with ``t = g'/r``, built from
    differentiable operations, so it can be differentiated again.
    """
    b, (m, q, d) = x.shape[0], xp.shape
    grad = x.new_zeros(b, d)
    for c in _chunks(m, b * q * d, x.device):
        w = wp[c].reshape(-1, d)
        diff = x[:, None, :] - xp[c].reshape(1, -1, d)
        r = safe_norm(diff)
        g = kernel.g(r)
        coef = kernel.t_of(r) * _rowdot(diff, w)
        if ec.numel() > 0:
            coef = coef - g * ec[c].repeat_interleave(q)
        grad = grad + g @ w + torch.bmm(coef.unsqueeze(1), diff).squeeze(1)
    return grad


class _KernelEnergy(torch.autograd.Function):
    """:func:`kernel_energy` with :func:`kernel_energy_gradient` as its derivative.

    The gradient is formed with the energy, in the same pass. A derivative of the
    gradient itself (``create_graph``) forms it again from differentiable operations.
    """

    @staticmethod
    def forward(ctx, x, xp, wp, ec, kernel):
        ctx.kernel = kernel
        if not ctx.needs_input_grad[0]:
            return kernel_energy(x, xp, wp, ec, kernel)
        e, grad = kernel_energy(x, xp, wp, ec, kernel, gradient=True)
        ctx.save_for_backward(x, xp, wp, ec, grad)
        return e

    @staticmethod
    def backward(ctx, grad_e):
        x, xp, wp, ec, grad = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad = kernel_energy_gradient(x, xp, wp, ec, ctx.kernel)
        return grad_e[:, None] * grad, None, None, None, None


@register_model("sgdml")
class SGDML(InteratomicPotential):
    """Symmetrized gradient-domain machine-learning force field.

    The model is tied to its molecule: every structure must list the elements of
    ``species`` in that order. It is filled by :func:`~xnn.krnn.common.sgdml_fit.fit_sgdml`
    or loaded from an sGDML model file (:func:`~xnn.krnn.common.sgdml_io.load_sgdml_model`).

    Parameters
    ----------
    species : sequence of int or str
        The elements of the atoms, in order.
    sigma : float, optional
        Length scale of the Matern kernel, by default 10.0.
    kernel_order : int, optional
        ``n`` of the Matern order ``n + 1/2``, by default 2.
    use_symmetries : bool, optional
        Find the permutational symmetries of the molecule when fitting (sGDML),
        by default ``True``; ``False`` gives GDML.
    cutoff : float, optional
        Neighbour-list radius of the data pipeline; the model reads positions
        only. By default 1.0.

    Attributes
    ----------
    train_desc : Tensor
        ``(M, D)`` descriptors of the training geometries.
    train_weights : Tensor
        ``(M, D)`` force coefficients through the descriptor Jacobians.
    energy_coef : Tensor
        ``(M,)`` energy coefficients (empty unless energies were part of the fit).
    perms : Tensor
        ``(Q, N)`` atom permutations of the symmetry group.
    std, c : Tensor
        Scale of the training forces and the integration constant.
    lattice : Tensor
        ``(3, 3)`` cell (rows as vectors) of a periodic model, empty otherwise.
    """

    head_modules = ("train_weights", "energy_coef", "c")

    def __init__(self, species: Sequence, sigma: float = 10.0, kernel_order: int = 2,
                 use_symmetries: bool = True, cutoff: float = 1.0):
        super().__init__()
        self.species = [atomic_number(z) if isinstance(z, str) else int(z) for z in species]
        self.n_atoms = len(self.species)
        if self.n_atoms < 2:
            raise ValueError("a GDML model needs at least two atoms")
        self.sigma = float(sigma)
        self.kernel_order = int(kernel_order)
        self.kernel = Matern(self.sigma, self.kernel_order)
        self.use_symmetries = bool(use_symmetries)
        self.cutoff = float(cutoff)
        self.descriptor = InverseDistances(self.n_atoms)
        d = self.descriptor.dim
        self.register_buffer("_species", torch.tensor(self.species, dtype=torch.long), persistent=False)
        self.register_buffer("train_desc", torch.zeros(0, d, dtype=torch.float64))
        self.register_buffer("train_weights", torch.zeros(0, d, dtype=torch.float64))
        self.register_buffer("energy_coef", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("perms", torch.arange(self.n_atoms).unsqueeze(0))
        self.register_buffer("perm_index", torch.arange(d).unsqueeze(0))
        self.register_buffer("std", torch.ones((), dtype=torch.float64))
        self.register_buffer("c", torch.zeros((), dtype=torch.float64))
        self.register_buffer("lattice", torch.zeros(0, 3, dtype=torch.float64))
        self.register_buffer("length_scale", torch.tensor(self.sigma, dtype=torch.float64))
        #: options of :meth:`fit_closed_form` (from the config)
        self.fit_options: Dict[str, object] = {}
        #: how the last fit was solved (the solver and, for the iterative one, its iterations and residual)
        self.fit_info: Dict[str, object] = {}
        # the fused kernels of the expansion (see set_use_fast); "auto" takes them where measured faster
        self.use_fast = "auto"

    def set_kernel(self, sigma: float, kernel_order: Optional[int] = None) -> None:
        """Change the kernel (before fitting)."""
        self.sigma = float(sigma)
        if kernel_order is not None:
            self.kernel_order = int(kernel_order)
        self.kernel = Matern(self.sigma, self.kernel_order)
        self.length_scale.fill_(self.sigma)

    def set_permutations(self, perms) -> None:
        """Use the atom permutations ``perms`` ``(Q, N)`` as the symmetry group."""
        p = torch.as_tensor(perms, dtype=torch.long).reshape(-1, self.n_atoms)
        self.perms = p.to(self.perms.device)
        self.perm_index = self.descriptor.permutation_index(p.tolist()).to(self.perms.device)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        for name in _SIZED:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
        if prefix + "length_scale" in state_dict:
            self.set_kernel(float(self.length_scale))

    def receptive_field(self) -> Optional[float]:
        """Unbounded: the energy reads every pair of atoms."""
        return None

    def geometry(self, data: AtomicGraph):
        """``(B, N, 3)`` positions and ``(B, 3, 3)`` cells of the batch, in float64.

        The cell is the structure's when it is periodic, otherwise the model's own
        (a model fitted in a cell); ``None`` for molecules.
        """
        b = data.num_graphs
        if not bool((data.n_atoms == self.n_atoms).all()):
            raise ValueError(f"the model is built for {self.n_atoms} atoms; got structures of "
                             f"{data.n_atoms.tolist()} atoms")
        if not torch.equal(data.atomic_numbers.reshape(b, self.n_atoms), self._species.expand(b, self.n_atoms)):
            raise ValueError("the atoms are not the elements of the model in its order")
        pos = data.pos.reshape(b, self.n_atoms, 3).to(torch.float64)
        cell = None
        if data.cell is not None and data.pbc is not None and bool(data.pbc.any()):
            cell = data.cell.reshape(b, 3, 3).to(torch.float64)
        elif self.lattice.numel() > 0:
            cell = self.lattice.expand(b, 3, 3)
        return pos, cell

    def permuted_training_set(self):
        """``(M, Q, D)`` training descriptors and weights under every permutation (cached)."""
        key = tuple((t.data_ptr(), t._version, t.device) for t in (self.train_desc, self.train_weights,
                                                                    self.perm_index))
        cache = self.__dict__.get("_permuted")
        if cache is None or cache[0] != key:
            cache = (key, self.train_desc[:, self.perm_index].contiguous(),
                     self.train_weights[:, self.perm_index].contiguous())
            self.__dict__["_permuted"] = cache
        return cache[1], cache[2]

    def set_use_fast(self, use_fast) -> "SGDML":
        """Choose the implementation of the kernel expansion.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the fused kernels (:mod:`xnn.krnn.fast.sgdml`) where
            they are available and measured to be faster (a CUDA device, Triton, a batch
            above the size crossover of ``AUTO_POLICY``, or ``AUTO_POLICY_TRAINING`` in
            training mode); ``True``: wherever they are available; ``False``: always the
            reference. The kernels give Hessians through the forces as the reference does.

        Returns
        -------
        SGDML
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    def uses_fast_path(self, x: Tensor) -> bool:
        """Whether the expansion of the descriptors ``x`` runs on the fused kernels."""
        if self.use_fast is False:
            return False
        from ..fast import sgdml as fast
        if not fast.supported(x.device, x.dtype):
            return False
        if self.use_fast is True:
            return True
        policy = fast.AUTO_POLICY_TRAINING if self.training else fast.AUTO_POLICY
        size = x.shape[0] * self.train_desc.shape[0] * self.perm_index.shape[0]
        return policy.decide(x.device, x.dtype, size)

    def energy_of_descriptors(self, x: Tensor) -> Tensor:
        """Energies ``(B,)`` of float64 descriptors ``(B, D)``."""
        xp, wp = self.permuted_training_set()
        if self.uses_fast_path(x):
            from ..fast.sgdml_kernels import KernelEnergy
            e = KernelEnergy.apply(x, xp, wp, self.energy_coef, self.kernel)
        else:
            e = _KernelEnergy.apply(x, xp, wp, self.energy_coef, self.kernel)
        return self.c + self.std * e

    def _outputs(self, energy: Tensor, data: AtomicGraph) -> Dict[str, Tensor]:
        node_energy = (energy / self.n_atoms).index_select(0, data.batch)
        return {"node_energy": node_energy.to(data.model_dtype), "energy": self.aggregate_energy(node_energy, data)}

    def forward(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """Energy of each structure; the per-atom energies are its equal shares."""
        pos, cell = self.geometry(data)
        return self._outputs(self.energy_of_descriptors(self.descriptor(pos, cell)), data)

    def forms_forces(self, data: AtomicGraph) -> bool:
        """Whether :meth:`forward_direct` serves this evaluation: always in evaluation mode, unless the
        positions carry an autograd graph that the outputs must continue."""
        return not self.training and not (torch.is_grad_enabled() and data.pos.requires_grad)

    def forward_direct(self, data: AtomicGraph) -> Dict[str, Tensor]:
        """:meth:`forward` with the forces ``"forces"`` and ``"virial"`` (``dE / d(strain)``) of the
        paper's force predictor: the gradient of the expansion by the descriptors, formed with
        the energy, through the descriptor Jacobian. The outputs carry no autograd graph."""
        with torch.no_grad():
            pos, cell = self.geometry(data)
            d = self.descriptor.differences(pos, cell)                   # (B, D, 3)
            r = torch.linalg.vector_norm(d, dim=-1)
            x = 1.0 / r
            xp, wp = self.permuted_training_set()
            if self.uses_fast_path(x):
                from ..fast.sgdml_kernels import energy_gradient
                e, gx = energy_gradient(x, xp, wp, self.energy_coef, self.kernel)
            else:
                e, gx = kernel_energy(x, xp, wp, self.energy_coef, self.kernel, gradient=True)
            # dE/d(r_a - r_b) of every pair: the gradient times d x / d d = -d / r^3
            pair = d * (-self.std * gx / (r * r * r)).unsqueeze(-1)
            grad = torch.zeros_like(pos).index_add_(1, self.descriptor._first, pair)
            grad.index_add_(1, self.descriptor._second, pair, alpha=-1)
            out = self._outputs(self.c + self.std * e, data)
            out["forces"] = (-grad).reshape(-1, 3).to(data.pos.dtype)
            out["virial"] = torch.einsum("bdi,bdk->bik", d, pair).to(data.pos.dtype)
        return out

    def _apply(self, fn, recurse=True):
        # the coefficients cancel to many digits: they stay float64 through every cast
        kept = {name: buf for name, buf in self.named_buffers(recurse=False) if buf.is_floating_point()}
        out = super()._apply(fn, recurse)
        for name, old in kept.items():
            new = getattr(self, name)
            setattr(self, name, old.to(device=new.device, dtype=torch.float64))
        return out

    def fit(self, graphs, **kwargs) -> "SGDML":
        """Fit to labelled structures (see :func:`~xnn.krnn.common.sgdml_fit.fit_sgdml`)."""
        from ..common.sgdml_fit import fit_sgdml
        return fit_sgdml(self, graphs, **kwargs)

    def fit_closed_form(self, train, validation=None) -> "SGDML":
        """The trainer's fit: :meth:`fit` with the options of the config (:attr:`fit_options`)."""
        return self.fit(train, validation=validation, **self.fit_options)

    @classmethod
    def from_config(cls, cfg) -> "SGDML":
        """Build from ``cfg.extra``: ``species`` (one entry per atom, in order),
        ``sigma`` (a length scale, or several to choose from on the validation
        set when the trainer fits the model; by default 10 to 100 in steps of 10),
        ``kernel_order``, ``use_symmetries``, the fit options ``lam``,
        ``energy_constraint``, ``use_energy``, ``solver``, ``solver_tol``, ``n_inducing`` and
        ``max_memory``, or ``model_file`` (an sGDML model
        file to load); ``cfg.cutoff`` sizes the data pipeline's graph."""
        from xnn.common.config.options import model_options

        extra = model_options(cfg)
        if extra.get("model_file"):
            from ..common.sgdml_io import load_sgdml_model
            return load_sgdml_model(extra["model_file"], cutoff=float(cfg.cutoff))
        sigma = extra.get("sigma", [float(s) for s in range(10, 101, 10)])
        sigmas = [float(s) for s in sigma] if isinstance(sigma, (list, tuple)) else [float(sigma)]
        model = cls(extra["species"], sigma=sigmas[0], kernel_order=int(extra.get("kernel_order", 2)),
                    use_symmetries=bool(extra.get("use_symmetries", cls._default_symmetries)),
                    cutoff=float(cfg.cutoff))
        model.fit_options = {"sigma": sigmas if len(sigmas) > 1 else sigmas[0],
                             "lam": float(extra.get("lam", 1e-10)),
                             "energy_constraint": bool(extra.get("energy_constraint", False)),
                             "use_energy": bool(extra.get("use_energy", True)),
                             "solver": str(extra.get("solver", "auto")),
                             "tol": float(extra.get("solver_tol", 1e-4))}
        for key in ("n_inducing", "max_memory"):
            if extra.get(key) is not None:
                model.fit_options[key] = int(extra[key]) if key == "n_inducing" else float(extra[key])
        return model

    _default_symmetries = True


@register_model("gdml")
class GDML(SGDML):
    """Gradient-domain machine learning without symmetries (Chmiela et al., 2017).

    The same model as :class:`SGDML` with the identity as its only permutation.
    """

    _default_symmetries = False

    def __init__(self, species: Sequence, sigma: float = 10.0, kernel_order: int = 2,
                 use_symmetries: bool = False, cutoff: float = 1.0):
        super().__init__(species, sigma, kernel_order, use_symmetries, cutoff)
