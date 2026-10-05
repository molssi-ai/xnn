"""Fine-tuning pretrained potentials: multi-head replay, LoRA, references.

The strategies of the MACE fine-tuning protocols (Batatia *et al.*,
arXiv:2401.00096; Tompa *et al.*, arXiv:2606.12704; Wang *et al.*, ELoRA,
ICML 2025), available to every xnn model that supports them:

* **naive fine-tuning**: ``model: {pretrained: <any hub model>}`` in a
  training config continues training every weight;
* **layer freezing**: :func:`freeze_parameters`, or ``optim.freeze`` /
  ``optim.train_only`` patterns;
* **LoRA**: :func:`inject_lora` / :func:`merge_lora`, or
  ``model.extra["lora"]``;
* **multi-head replay**: :class:`MultiHead`, :func:`select_replay`,
  :func:`pseudolabel`, or ``model.extra["heads"]`` with ``data.replay_path``;
* **reference energies**: :func:`estimate_atomic_energies` (model-aware
  reestimation) and :func:`average_atomic_energies`, or
  ``atomic_energies: estimated`` / ``average`` in the config.

See the how-to guide on fine-tuning for the recipes.
"""
from .freeze import count_parameters, freeze_parameters
from .heads import MultiHead, find_multihead, head_module_names, label_head
from .lora import (LoRAAdapter, LoRAEquivariantLinear, has_lora, inject_lora, lora_modules,
                   lora_parameters, merge_lora, register_lora_target)
from .reference import (REFERENCE_MARKERS, average_atomic_energies, estimate_atomic_energies,
                        get_atomic_energies, predict_energies, reference_markers,
                        set_atomic_energies, species_of)
from .replay import REPLAY_FILTERS, element_filter, pseudolabel, select_replay

__all__ = [
    "MultiHead", "find_multihead", "head_module_names", "label_head",
    "LoRAAdapter", "LoRAEquivariantLinear", "inject_lora", "merge_lora", "has_lora",
    "lora_modules", "lora_parameters", "register_lora_target",
    "estimate_atomic_energies", "average_atomic_energies", "get_atomic_energies",
    "set_atomic_energies", "predict_energies", "species_of", "reference_markers",
    "REFERENCE_MARKERS",
    "select_replay", "element_filter", "pseudolabel", "REPLAY_FILTERS",
    "freeze_parameters", "count_parameters",
]
