.. _developer-guide-extending:

**************
Extending xnn
**************

The rules (see :ref:`design`): one implementation per component, reuse the
existing abstractions, and place code with the family that uses it, or in
``common`` when more than one family needs it.

Adding a model
==============
1. **Pick the family** (``gnn``, ``cnn``, ``dnn``, ``ffnn``, ``hybrid``, or a
   new one) and add a module under ``src/xnn/<family>/models/``.
2. **Subclass the right base.**
   :class:`~xnn.common.models.base.InteratomicPotential` is the minimal
   interface. :class:`~xnn.gnn.models.base.GNNPotential` adds species
   bookkeeping and per-element reference energies,
   :class:`~xnn.gnn.models.base.EquivariantGNN` the spherical-harmonic edge
   embedding, :class:`~xnn.dnn.models.base.DescriptorPotential` a
   featurizer with per-element MLPs.
3. **Implement** ``forward(data)``: take an
   :class:`~xnn.common.data.atomic_data.AtomicGraph`, return
   ``{"node_energy": (N,), "energy": (B,)}`` using
   ``self.aggregate_energy(node_energy, data)``. Never compute forces or
   stress; :class:`~xnn.common.models.outputs.ForceStressOutput` does that.
   Expose the invariant per-atom features as ``"node_features"`` (with
   ``node_feature_dim``) and the LES long-range term works on top of it.
4. **Register.**

   .. code-block:: python

      from xnn.common.models import InteratomicPotential, register_model

      @register_model("mymodel")
      class MyModel(InteratomicPotential):
          @classmethod
          def from_config(cls, cfg):          # cfg is a ModelConfig
              return cls(cutoff=cfg.cutoff, **cfg.extra)

   Add a ``configs/model/mymodel.yaml`` template and import the module from
   the family package so the registration runs.
5. **Optionally make it exportable.** A scriptable
   ``node_features_energy(atomic_numbers, edge_index, edge_vec, ...)`` core,
   as in SchNet, is all :func:`~xnn.common.deploy.export_torchscript_potential`
   needs.

Upstream config spellings go into a key-translation table
(:func:`~xnn.common.config.translate.register_key_translation`), never into
``from_config`` aliases.

Adding a dataset
================
Subclass :class:`~xnn.common.data.hub.base.DatasetBuilder` and return
structure dictionaries, ``{split: [...]}`` when ``split`` is ``None``:

.. code-block:: python

   from xnn.common.data.hub import DatasetBuilder, register_dataset

   class MyDataset(DatasetBuilder):
       name = "mydataset"
       description = "One-line summary shown by list_datasets()."

       def load(self, *, split=None, cache_dir, **kwargs):
           # download_file(url, cache_dir / self.name / "raw" / fname, md5)
           splits = {"train": structures}
           return splits if split is None else splits[split]

   register_dataset(MyDataset())

:func:`~xnn.common.data.hub._download.download_file` caches and verifies by
MD5; :func:`~xnn.common.data.ase_io.atoms_to_structure` converts ASE
frames. Put the module under ``src/xnn/common/data/hub/`` and import it
from the package. ``load_dataset`` handles the ``cutoff=`` wrapping. A
builder need not download: ``argon_md`` reads files bundled under
``datasets/``.

Adding a pre-trained model
==========================
Write the model as a portable directory (``save_pretrained`` or ``xnn models
pack``), upload it, and add a card with
:func:`~xnn.common.models.hub.register_pretrained` or an entry in
``src/xnn/common/models/hub/models.json``. A new foreign checkpoint format
needs a converter registered in :mod:`xnn.common.models.hub.formats`.

Adding a featurizer
===================
Subclass :class:`~xnn.common.featurizers.base.Featurizer`, implement
``output_dim`` and ``forward(data)``, and put it under
``common/featurizers/`` when shared, otherwise with the family that uses it.

Neighbor lists
==============
:func:`~xnn.common.data.neighborlist.build_neighbor_list` returns
``edge_index`` and ``cell_shifts``; that pair is all a model sees. The
reference implementation is brute force, and the ``vesin`` cell list is used
when installed. Another backend only has to return the same pair.
