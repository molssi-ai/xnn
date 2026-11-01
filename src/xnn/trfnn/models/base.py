"""Energy bookkeeping shared by the transformer potentials (ET, Equiformer, EquiformerV2).

Every model of the family maps its per-atom readout ``y_i`` to an energy
``E_i = energy_scale * y_i + energy_shift + atom_ref[Z_i]``: a per-atom
standardization from the training data and a learnable per-element reference.
"""
# NOTE: no `from __future__ import annotations` here -- PEP 563 stringifies
# the annotations that TorchScript needs to resolve.
import torch
from torch import nn

from xnn.common.models.base import InteratomicPotential


class TransformerPotential(InteratomicPotential):
    """Base of the transformer potentials: energy standardization and per-element references.

    Subclasses call :meth:`_init_energy_terms` in their ``__init__``. Models
    whose constants (coupling coefficients, grid matrices) must stay exact in
    float64 set ``exact_float64 = True``.

    Attributes
    ----------
    energy_scale, energy_shift : Tensor
        The per-atom standardization buffers.
    atom_ref : torch.nn.Embedding
        Learnable per-element energy reference indexed by ``Z``, initialized to zero.
    """

    head_modules = ("readout", "energy_scale", "energy_shift", "atom_ref")
    exact_float64 = False
    #: the module of the model's fast path (its ``AUTO_POLICY`` and ``AUTO_POLICY_TRAINING``)
    fast_path = ""

    def _init_energy_terms(self, n_elements: int, energy_scale: float, energy_shift: float,
                           species=None, atomic_energies=None) -> None:
        """Register the standardization buffers and the per-element references."""
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("energy_shift", torch.tensor(float(energy_shift)))
        self.atom_ref = nn.Embedding(n_elements, 1)
        nn.init.zeros_(self.atom_ref.weight)
        if atomic_energies is not None:
            if species is None:
                raise ValueError("species is required to map atomic_energies")
            self.set_atomic_energies(species, atomic_energies)
        if self.exact_float64:
            from xnn.common.models.constants import exact_constants_after_load
            self.register_load_state_dict_post_hook(exact_constants_after_load)
        # the fused kernels (see set_use_fast); "auto" takes them where measured faster
        self.use_fast = "auto"

    @torch.jit.unused
    def set_use_fast(self, use_fast) -> "TransformerPotential":
        """Choose between the reference layers and the fused kernels of the fast path.

        Parameters
        ----------
        use_fast : bool or str
            ``"auto"`` (the default): the fused GPU kernels (see the model's
            fast-path module in :mod:`xnn.trfnn.fast`) where they are available
            and faster for the input (a CUDA device, a graph above the size
            crossover measured for evaluation, ``AUTO_POLICY``, or for
            training, ``AUTO_POLICY_TRAINING``); ``True``: wherever they are
            available; ``False``: always the reference implementation. The
            kernels are twice differentiable, so force training takes them as
            well.

        Returns
        -------
        TransformerPotential
            ``self``.
        """
        from xnn.common.models.fast import resolve_use_fast
        self.use_fast = resolve_use_fast(use_fast)
        return self

    @torch.jit.unused
    def _select_fast(self, num_edges: int) -> None:
        """Switch the fused kernels on or off for one evaluation."""
        if not self.fast_path:
            return
        import importlib

        from xnn.common.models.fast import fast_modules, select
        modules = self.__dict__.get("_fast_modules")
        if modules is None:
            modules = fast_modules(self)
            self.__dict__["_fast_modules"] = modules
        if not modules:
            return
        policies = importlib.import_module(self.fast_path)
        like = self.atom_ref.weight
        policy = policies.AUTO_POLICY_TRAINING if self.training else policies.AUTO_POLICY
        select(modules, self.use_fast, policy, like.device, like.dtype, num_edges)

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse)
        if self.exact_float64:
            from xnn.common.models.constants import exact_float64_constants
            exact_float64_constants(self)
        return out

    @torch.jit.ignore
    def set_energy_scale_shift(self, scale: float, shift: float) -> None:
        """Set the per-atom energy standardization from training statistics.

        Parameters
        ----------
        scale : float
            Standard deviation of the training-set energy per atom.
        shift : float
            Mean training-set energy per atom.
        """
        with torch.no_grad():
            self.energy_scale.fill_(float(scale))
            self.energy_shift.fill_(float(shift))

    @torch.jit.ignore
    def set_atomic_energies(self, species, values) -> None:
        """Initialize the per-element reference energies ``atom_ref``.

        Parameters
        ----------
        species : list of int
            Atomic numbers the values refer to, in order.
        values : array-like
            One reference energy per entry of ``species``.

        Raises
        ------
        ValueError
            If the number of values does not match the number of species.
        """
        ae = torch.as_tensor(values, dtype=self.atom_ref.weight.dtype)
        if ae.numel() != len(list(species)):
            raise ValueError(
                f"got {ae.numel()} atomic energies for {len(list(species))} species")
        with torch.no_grad():
            self.atom_ref.weight[torch.tensor(list(species)), 0] = ae
