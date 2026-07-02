"""Typed configuration schema.

A single nested dataclass tree is the *one* internal representation. Every
frontend (YAML, TOML, argparse, Hydra) is just a different loader that produces
this same `Config`. That is what makes the formats interchangeable -- see
`loaders.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _default_mace_extra() -> dict[str, Any]:
    """MACE architecture options carried in :attr:`ModelConfig.extra` by default.

    Mirrors the canonical MACE defaults (see ``configs/model/mace.yaml``); every
    value matches what ``MACE.from_config`` would otherwise fall back to, so
    building the default model reproduces stock MACE.

    Returns
    -------
    dict[str, Any]
        The default ``extra`` payload for a MACE model.
    """
    return {
        "species": [1, 6, 8],          # H, C, O
        "max_ell": 3,                  # spherical-harmonic degree on edges
        "max_L": 0,                    # max angular momentum of hidden node features
        "correlation": 3,              # symmetric-contraction body order
        "num_polynomial_cutoff": 5,
        "MLP_irreps": "16x0e",
        "radial_MLP": [64, 64, 64],
        "gate": "silu",
        "avg_num_neighbors": 1.0,
    }


@dataclass
class ModelConfig:
    """Model architecture and hyperparameters.

    Attributes
    ----------
    name : str
        Registry key selecting the model family (``schnet``/``hdnnp``/``ani``/
        ``nequip``/``mace``/``allegro``). Defaults to ``"mace"``.
    cutoff : float
        Interaction/neighbor-list radius in the same length units as the data.
        MACE ``r_max``. Defaults to ``4.0``.
    n_features : int
        Width of the per-atom feature (embedding) channels. MACE
        ``num_channels``. Defaults to ``32``.
    n_interactions : int
        Number of message-passing / interaction blocks. MACE
        ``num_interactions``. Defaults to ``2``.
    n_rbf : int
        Number of radial basis functions used to expand interatomic distances.
        MACE ``num_bessel``/``num_radial_basis``. Defaults to ``8``.
    extra : dict[str, Any]
        Free-form, model-specific options (e.g. the species list, correlation
        order, and irreps for MACE). Defaults to the canonical MACE architecture
        options (see :func:`_default_mace_extra`).
    """

    name: str = "mace"             # registry key: schnet/hdnnp/ani/nequip/mace/allegro
    cutoff: float = 4.0            # MACE r_max
    n_features: int = 32           # MACE num_channels
    n_interactions: int = 2        # MACE num_interactions
    n_rbf: int = 8                 # MACE num_bessel / num_radial_basis
    # free-form, model-specific options (MACE architecture defaults)
    extra: dict[str, Any] = field(default_factory=_default_mace_extra)


@dataclass
class DataConfig:
    """Dataset paths and data-loading options.

    Attributes
    ----------
    train_path : Optional[str]
        Path to the training set (``.xyz`` / ``.extxyz`` / ``.npz`` / any
        ASE-readable file). Defaults to ``None``.
    val_path : Optional[str]
        Path to the validation set. If ``None``, a validation split is carved
        out of the training set using ``val_fraction``. Defaults to ``None``.
    cutoff : float
        Neighbor-list cutoff radius; must match ``model.cutoff`` and is kept in
        lockstep by :meth:`Config.__post_init__`. Defaults to ``4.0``.
    batch_size : int
        Mini-batch size; set to ``1`` to disable batch training. Defaults to ``16``.
    num_workers : int
        Number of dataloader worker processes. Defaults to ``0``.
    val_fraction : float
        Fraction of the training set held out for validation when ``val_path``
        is ``None``. Defaults to ``0.1``.
    """

    train_path: Optional[str] = None   # .xyz / .extxyz / .npz / ASE-readable
    val_path: Optional[str] = None
    cutoff: float = 4.0                # must match model.cutoff for neighbor lists
    batch_size: int = 16               # set to 1 to disable batch training
    num_workers: int = 0
    val_fraction: float = 0.1          # used if val_path is None


@dataclass
class OptimConfig:
    """Optimizer, schedule, and loss-weighting settings.

    Attributes
    ----------
    lr : float
        Learning rate. Defaults to ``1e-3``.
    weight_decay : float
        L2 weight-decay coefficient. Defaults to ``0.0``.
    epochs : int
        Number of training epochs. MACE ``max_num_epochs``. Defaults to ``100``.
    energy_weight : float
        Weight of the energy term in the loss. MACE ``energy_weight``. Defaults
        to ``1.0``.
    force_weight : float
        Weight of the force term in the loss. Defaults to ``10.0``.
    stress_weight : float
        Weight of the stress term; values ``> 0`` enable stress training for
        periodic systems. Defaults to ``0.0``.
    scheduler : str
        Learning-rate scheduler (``none`` / ``cosine`` / ``plateau``). MACE uses
        ``ReduceLROnPlateau``. Defaults to ``"plateau"``.
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100                  # MACE max_num_epochs
    energy_weight: float = 1.0
    force_weight: float = 10.0
    stress_weight: float = 0.0         # > 0 enables stress training (periodic)
    scheduler: str = "plateau"         # none / cosine / plateau (MACE: ReduceLROnPlateau)


@dataclass
class Config:
    """Top-level experiment configuration.

    This nested dataclass tree is the single internal representation that every
    frontend loader (YAML, TOML, argparse, Hydra) produces; see ``loaders.py``.

    Attributes
    ----------
    model : ModelConfig
        Model architecture and hyperparameters. Defaults to a fresh
        :class:`ModelConfig`.
    data : DataConfig
        Dataset paths and data-loading options. Defaults to a fresh
        :class:`DataConfig`.
    optim : OptimConfig
        Optimizer and loss settings. Defaults to a fresh :class:`OptimConfig`.
    device : str
        Compute device (``auto`` / ``cpu`` / ``cuda`` / ``cuda:0`` ...).
        Defaults to ``"auto"``.
    seed : int
        Random seed for reproducibility. MACE ``seed``. Defaults to ``1234``.
    output_dir : str
        Directory for run outputs (checkpoints, logs). Defaults to ``"runs/exp"``.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    device: str = "auto"               # auto / cpu / cuda / cuda:0 ...
    seed: int = 1234                   # MACE seed
    output_dir: str = "runs/exp"

    def __post_init__(self):
        """Synchronize the data cutoff with the model cutoff.

        Overwrites ``data.cutoff`` with ``model.cutoff`` so the neighbor-list
        radius and the model interaction radius stay in lockstep.
        """
        # keep the neighbor-list cutoff and the model cutoff in lockstep
        self.data.cutoff = self.model.cutoff
