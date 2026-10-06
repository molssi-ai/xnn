.. _developer-guide-extending:

**************
Extending xnn
**************

One implementation per component; put code with the family that uses it
(:ref:`design`).

Adding a model
==============

.. code-block:: python

   from xnn.common.models import InteratomicPotential, register_model

   @register_model("mymodel")
   class MyModel(InteratomicPotential):
       def forward(self, data):                      # AtomicGraph in
           node_energy = ...                         # (N,)
           return {"node_energy": node_energy,
                   "energy": self.aggregate_energy(node_energy, data)}

       @classmethod
       def from_config(cls, cfg):                    # ModelConfig in
           return cls(cutoff=cfg.cutoff, **cfg.extra)

- Bases: :class:`~xnn.common.models.base.InteratomicPotential`,
  :class:`~xnn.gnn.models.base.GNNPotential`,
  :class:`~xnn.gnn.models.base.EquivariantGNN`,
  :class:`~xnn.dnn.models.base.DescriptorPotential`,
  :class:`~xnn.cnn.models.base.VoxelPotential`.
- No forces or stress in the model;
  :class:`~xnn.common.models.outputs.ForceStressOutput` adds them.
- Import the module from its family package and add
  ``configs/model/mymodel.yaml``.
- Upstream key spellings:
  :func:`~xnn.common.config.translate.register_key_translation`.
- Return ``"node_features"`` for the LES term; a scriptable
  ``node_features_energy(...)`` core makes the model exportable.

Adding a dataset
================

.. code-block:: python

   from xnn.common.data.hub import DatasetBuilder, register_dataset

   class MyDataset(DatasetBuilder):
       name = "mydataset"
       description = "One line, shown by list_datasets()."

       def load(self, *, split=None, cache_dir, **kwargs):
           splits = {"train": structures}            # structure dictionaries
           return splits if split is None else splits[split]

   register_dataset(MyDataset())

Place it under ``src/xnn/common/data/hub/`` and import it from the package.
:func:`~xnn.common.data.hub._download.download_file` caches and verifies by
MD5; ``load_dataset`` handles the ``cutoff=`` wrapping.

Adding a pre-trained model
==========================
Write a portable directory (``trainer.save_pretrained()`` or ``xnn models
pack``), upload it, and register a card with
:func:`~xnn.common.models.hub.register_pretrained` or in ``models.json``.
A new checkpoint format is a converter registered with
:func:`~xnn.common.models.hub.formats.register_format`.

Adding a featurizer
===================
Subclass :class:`~xnn.common.featurizers.base.Featurizer` with
``output_dim`` and ``forward(data)``.

Neighbor lists
==============
A model only sees ``edge_index`` and ``cell_shifts`` from
:func:`~xnn.common.data.neighborlist.build_neighbor_list`; any backend that
returns the same pair works (``vesin`` is used when installed).
