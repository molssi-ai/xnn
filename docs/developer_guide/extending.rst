.. _developer-guide-extending:

**************
Extending xnns
**************

The guiding rules (see :ref:`design`): one canonical implementation per
component, reuse the existing abstractions, and place code with the family
that uses it — or in ``common`` when more than one family needs it.

Adding a model
==============
1. **Pick the family package** — ``gnn``, ``cnn``, or ``dnn`` (or add a new
   family) — and add a module under ``src/xnns/<family>/models/``.

2. **Subclass the right base.**
   :class:`~xnns.common.models.base.InteratomicPotential` is the minimal
   interface; the family bases give you more for free —
   :class:`~xnns.gnn.models.base.GNNPotential` handles species bookkeeping
   and per-element reference energies (CACE builds on it directly);
   :class:`~xnns.gnn.models.base.EquivariantGNN` adds the spherical-harmonic
   edge embedding on top (NequIP/MACE/Allegro);
   :class:`~xnns.dnn.models.base.DescriptorPotential` composes a
   featurizer with per-element MLPs.

3. **Implement** ``forward(data)`` taking an
   :class:`~xnns.common.data.atomic_data.AtomicGraph` and returning
   ``{"node_energy": (N,), "energy": (B,)}`` — use
   ``self.aggregate_energy(node_energy, data)`` for the per-structure sum.
   Do *not* compute forces or stress;
   :class:`~xnns.common.models.outputs.ForceStressOutput` does that for
   every model.

4. **Register and configure.**

   .. code-block:: python

      from xnns.common.models import InteratomicPotential, register_model

      @register_model("mymodel")
      class MyModel(InteratomicPotential):
          @classmethod
          def from_config(cls, cfg):          # cfg is a ModelConfig
              return cls(cutoff=cfg.cutoff, **cfg.extra)

   Add a ``configs/model/mymodel.yaml`` template, and make sure the family
   package imports your module so the registration runs on import.

5. **(Optional) make it deployable.** Expose a scriptable
   ``node_energy(atomic_numbers, edge_index, edge_vec)`` core — SchNet shows
   the pattern; e3nn-based models need e3nn's JIT support for this. That is
   all :func:`~xnns.common.deploy.lammps.export_to_lammps` needs.

If your model should accept config keys from an upstream code, register a
key-translation table with
:func:`~xnns.common.config.translate.register_key_translation` rather than
adding aliases in ``from_config`` — translations live at the loader level.

Adding a dataset
================
The dataset hub (:func:`~xnns.common.data.hub.base.load_dataset`) is extensible
in the same register-by-name way as models. Subclass
:class:`~xnns.common.data.hub.base.DatasetBuilder`, set its ``name``, and
implement ``load`` to download (via
:func:`~xnns.common.data.hub._download.download_file`, which caches and verifies
by MD5) and return xnns :ref:`structure dictionaries <structure-dicts>` —
``{split: [structure_dict, ...]}`` when ``split`` is ``None``, else a single
list:

.. code-block:: python

   from xnns.common.data.hub import DatasetBuilder, register_dataset

   class MyDataset(DatasetBuilder):
       name = "mydataset"
       description = "One-line summary shown by list_datasets()."

       def load(self, *, split=None, cache_dir, **kwargs):
           # download_file(url, cache_dir / self.name / "raw" / fname, md5)
           structures = [...]                       # list of structure dicts
           splits = {"train": structures}
           return splits if split is None else splits[split]

   register_dataset(MyDataset())

Put the builder module under ``src/xnns/common/data/hub/`` and import it from
``hub/__init__.py`` so the registration runs on import (as ``rmd17`` and
``lode_dimers`` do). ``load_dataset`` then handles the ``cutoff=`` wrapping into
an :class:`~xnns.common.data.dataset.AtomicDataset` for you, so builders only
produce structure dicts. Reuse :func:`~xnns.common.data.ase_io.atoms_to_structure`
for any ASE-readable source, and show progress with ``tqdm`` (respect a
``quiet`` flag).

Adding a featurizer
===================
Subclass :class:`~xnns.common.featurizers.base.Featurizer`, implement the
``output_dim`` property and ``forward(data)``. Put it in
``common/featurizers/`` if it is shared, otherwise under the using family's
``featurizers/`` package, and compose it into models.

Swapping the neighbor list
==========================
The reference :func:`~xnns.common.data.neighborlist.build_neighbor_list` is
correct but brute-force. For large periodic systems, swap in a cell-list or
`matscipy <https://github.com/libAtoms/matscipy>`_ builder — as long as it
returns the same ``edge_index`` / ``cell_shifts`` pair, nothing else
changes.
