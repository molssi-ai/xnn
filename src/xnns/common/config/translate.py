"""Key-name translation registry: upstream GNN-code configs -> xnns names.

xnns keeps *one* canonical spelling for every option (the typed
:class:`~xnns.common.config.schema.ModelConfig` fields plus each model's
documented ``extra`` keys). Other codes spell the same knobs differently
(MACE-CLI ``r_max``/``num_radial_basis``/``E0s``, NequIP ``num_layers``, ...).
Rather than teaching every ``Model.from_config`` a pile of aliases, a per-model
translation table rewrites the foreign spellings to the canonical names once,
inside the :func:`~xnns.common.config.loaders.from_dict` funnel -- so keys
copied verbatim from an upstream MACE / NequIP / Allegro yaml just work, in
every frontend (YAML / argparse / Hydra).

Rules
-----
- Translation applies to the raw ``model:`` section only, selected by its
  ``name`` (models without a registered table pass through untouched).
- The xnns canonical spelling always wins when both spellings are present.
- Untranslated unknown keys are left as-is (they fold into
  ``ModelConfig.extra`` downstream).
- Only key *names* are rewritten; values pass through verbatim (each
  ``from_config`` coerces upstream value forms such as MACE's
  ``E0s: '{1: -13.6}'`` strings).

Add a table for a new model family (or extend an existing one) with
:func:`register_key_translation`.
"""
from __future__ import annotations

from typing import Any

# upstream spelling -> xnns canonical spelling, per model-registry name.
# Targets may be typed core fields (cutoff/n_features/n_interactions/n_rbf)
# or the model's documented `extra` keys.
_KEY_TRANSLATIONS: dict[str, dict[str, str]] = {
    # original MACE: CLI/yaml spellings plus mace.modules.MACE constructor ones
    "mace": {
        "r_max": "cutoff",
        "num_channels": "n_features",
        "num_interactions": "n_interactions",
        "num_radial_basis": "n_rbf",
        "num_bessel": "n_rbf",
        "num_cutoff_basis": "num_polynomial_cutoff",
        "atomic_numbers": "species",
        "E0s": "atomic_energies",
    },
    # NequIP yaml spellings
    "nequip": {
        "r_max": "cutoff",
        "num_layers": "n_interactions",
        "num_features": "n_features",
        "num_basis": "n_rbf",
        "PolynomialCutoff_p": "num_polynomial_cutoff",
        "BesselBasis_trainable": "trainable_rbf",
        "conv_to_output_hidden_irreps_out": "conv_to_output_hidden",
        "chemical_symbols": "species",
        "per_species_rescale_shifts": "atomic_energies",
        "per_species_rescale_scales": "atomic_scales",
    },
    # CACE constructor spellings (BingqingCheng/cace `Cace(...)` kwargs),
    # plus the MACE/NequIP "l_max" spelling of CACE's max_l
    "cace": {
        "r_max": "cutoff",
        "zs": "species",
        "num_message_passing": "n_interactions",
        "type_message_passing": "message_types",
        "atomic_numbers": "species",
        "l_max": "max_l",
    },
    # schnetpack spellings (SchNet's reference code base) -- key names only;
    # the xnns SchNet itself is built from the papers, not from schnetpack
    "schnet": {
        "n_atom_basis": "n_features",
        "n_gaussians": "n_rbf",
        "n_radial_basis": "n_rbf",
        "atomref": "atomic_energies",
    },
    # PhysNet train.py argument spellings (MMunibas/PhysNet)
    "physnet": {
        "num_features": "n_features",
        "num_basis": "n_rbf",
        "num_blocks": "n_interactions",
        "sr_cut": "cutoff",
        "lr_cut": "lr_cutoff",
        "use_electrostatic": "use_electrostatics",
        "use_dispersion": "use_dispersion",
        "grimme_s6": "s6",
        "grimme_s8": "s8",
        "grimme_a1": "a1",
        "grimme_a2": "a2",
        "Eshift": "energy_shift",
        "Escale": "energy_scale",
        "Qshift": "charge_shift",
        "Qscale": "charge_scale",
    },
    # ANI / torchani / NeuroChem spellings (torchani.AEVComputer kwargs and the
    # ANI-1 potential's config knobs).
    "ani": {
        "Rcr": "radial_cutoff",
        "Rca": "angular_cutoff",
        "atomic_numbers": "species",
        "self_energies": "atomic_energies",
        "sae": "atomic_energies",
        "network_dims": "hidden",
    },
    # BAMBOO (bytedance/bamboo) nn_params / gnn_params spellings
    "bamboo": {
        "rcut": "cutoff",
        "dim": "n_features",
        "num_rbf": "n_rbf",
        "n_layers": "n_interactions",
    },
    # alternative ReaxFF spellings used by ReaxFF-nn training tools
    "reaxff": {
        "libfile": "ffield",
        "vdwcut": "cutoff",
        "hbshort": "hb_short",
        "hblong": "hb_long",
    },
    # OPLS spellings used by GROMACS / OpenMM-style inputs
    "opls": {
        "itp": "library",
        "prm": "library",
        "parameter_file": "library",
        "ffield": "library",
        "rvdw": "cutoff",
        "pair_cutoff": "cutoff",
        "fudgeLJ": "fudge_lj",
        "fudgeQQ": "fudge_qq",
    },
    # Allegro yaml spellings
    "allegro": {
        "r_max": "cutoff",
        "num_layers": "n_interactions",
        "num_tensor_features": "n_features",
        "env_embed_multiplicity": "n_features",
        "num_bessels_per_basis": "n_rbf",
        "num_basis": "n_rbf",
        "PolynomialCutoff_p": "num_polynomial_cutoff",
        "two_body_latent_mlp_latent_dimensions": "two_body_latent",
        "latent_mlp_latent_dimensions": "latent",
        "env_embed_mlp_latent_dimensions": "env_embed",
        "edge_eng_mlp_latent_dimensions": "edge_eng",
        "chemical_symbols": "species",
        "per_species_rescale_shifts": "atomic_energies",
        "per_species_rescale_scales": "atomic_scales",
    },
}


def register_key_translation(name: str, table: dict[str, str]) -> None:
    """Register (or extend) the upstream -> xnns key table for one model family.

    Parameters
    ----------
    name : str
        Model-registry name the table applies to (e.g. ``"mace"``).
    table : dict[str, str]
        Mapping of upstream key spellings to xnns canonical spellings; merged
        into any existing table for ``name`` (new entries win).
    """
    _KEY_TRANSLATIONS.setdefault(name, {}).update(table)


def translate_model_keys(section: dict[str, Any]) -> dict[str, Any]:
    """Rewrite upstream key spellings in a raw ``model:`` section to xnns names.

    Looks up the translation table for ``section["name"]`` (defaulting to
    ``"mace"``, matching :class:`ModelConfig`) and returns a flat copy of the
    section with every foreign key renamed to its canonical spelling. An
    explicit ``extra:`` sub-dict is flattened first so its keys are translated
    too -- ``from_dict`` re-splits typed fields from ``extra`` afterwards.
    When both spellings of the same option are present, the canonical one wins.

    Parameters
    ----------
    section : dict[str, Any]
        The raw ``model:`` section as produced by any config frontend.

    Returns
    -------
    dict[str, Any]
        A new section dict with canonical key names only.
    """
    table = _KEY_TRANSLATIONS.get(str(section.get("name", "mace")))
    if not table:
        return dict(section)
    merged = {**(section.get("extra") or {}),
              **{k: v for k, v in section.items() if k != "extra"}}
    out = {k: v for k, v in merged.items() if k not in table}
    for k, v in merged.items():
        if k in table:
            out.setdefault(table[k], v)  # canonical spelling wins
    return out
