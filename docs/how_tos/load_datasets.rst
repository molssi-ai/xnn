.. _howto-load-datasets:

******************************
Download an Upstream Dataset
******************************

The dataset hub downloads and preprocesses standard benchmark datasets in a
single line, HuggingFace ``load_dataset()``-style. It handles the download,
caching, MD5 verification, unit conversion, and conversion into xnns structure
dictionaries, so you can go straight to training. See :ref:`data` for the full
reference and ``examples/data/load_dataset_tutorial.ipynb`` for a runnable
walkthrough.

List what is available
======================

.. code-block:: python

   from xnns.common.data import load_dataset, list_datasets

   list_datasets()          # ['ani1', 'ani1ccx', 'ani1x', 'ani2x', 'argon_md', 'lode_dimers', 'rmd17']

Load a dataset
==============
With no ``split`` you get every split as a mapping of structure-dict lists; name
a ``split`` to get just that one. Dataset-specific options are passed as keyword
arguments (here rMD17's ``molecule``):

.. code-block:: python

   splits = load_dataset("rmd17", molecule="aspirin")            # {"train", "test"}
   train = load_dataset("rmd17", molecule="aspirin", split="train")

Get a training-ready ``AtomicDataset``
======================================
Pass ``cutoff=`` and each split comes back as an
:class:`~xnns.common.data.dataset.AtomicDataset` (neighbour graphs built on
demand), ready for a ``DataLoader`` or the :class:`~xnns.common.train.trainer.Trainer`:

.. code-block:: python

   from torch.utils.data import DataLoader
   from xnns.common.data import collate

   ds = load_dataset("rmd17", molecule="aspirin", cutoff=5.0)
   loader = DataLoader(ds["train"], batch_size=16, shuffle=True, collate_fn=collate)

Choose splits, units, and subsets
=================================
rMD17 ships five official folds and is converted to eV by default:

.. code-block:: python

   train = load_dataset("rmd17", molecule="ethanol", split="train",
                        fold=3, units="kcal/mol", n_train=500)

The ``ani1`` set (the 20 M-conformation ANI-1 training data, Smith *et al.*
2017) is distributed as one 4.8 GB pyanitools archive; select heavy-atom
subsets ``ani_gdb_s0X.h5`` and cap the amount materialised for tractable
experiments. ``split`` in ``{"train", "val", "test"}`` gives the paper's
per-molecule 80/10/10 partition:

.. code-block:: python

   # molecules with 2-4 heavy atoms, capped; energies converted to eV
   data = load_dataset("ani1", heavy_atoms=[2, 3, 4],
                       max_molecules=60, max_conformations=60)   # {"all": [...]}
   train = load_dataset("ani1", heavy_atoms=2, split="train")

The ``ani1x`` set is the active-learning training data behind the
:meth:`~xnns.dnn.models.ani.ANI.ani1x` preset (Smith *et al.* 2018/2020): ~5 M
conformations with wB97X **energies and forces** in one 5.6 GB HDF5 file.
``level`` selects the level of theory (``wb97x_dz`` by default -- what the model
was trained on), and per-conformation NaN entries are dropped automatically:

.. code-block:: python

   # capped demo subset with forces, energies converted to eV
   data = load_dataset("ani1x", max_molecules=50)               # {"all": [...]}
   train = load_dataset("ani1x", split="train")                 # 80/10/10 split
   ccx = load_dataset("ani1x", level="ccsd(t)_cbs", forces=False)  # energy-only

The ``ani1ccx`` set is the coupled-cluster companion behind the
:meth:`~xnns.dnn.models.ani.ANI.ani1ccx` preset (Smith *et al.* 2019/2020):
the ~500 k-conformation subset of ANI-1x recomputed at the CCSD(T)*/CBS level
(energy-only; no coupled-cluster forces). It lives inside the same release
file as ``ani1x``, so the download and cache are shared; loading it is
equivalent to the ``level="ccsd(t)_cbs"`` call above, under its own name:

.. code-block:: python

   data = load_dataset("ani1ccx", max_molecules=50)             # {"all": [...]}
   train = load_dataset("ani1ccx", split="train")               # 80/10/10 split

The ``ani2x`` set is the seven-element training data behind the
:meth:`~xnns.dnn.models.ani.ANI.ani2x` preset (Devereux *et al.* 2020): ~9.6 M
conformations with wB97X/6-31G* **energies and forces** for H/C/N/O/S/F/Cl. It
is a separate 3.7 GB pyanitools HDF5 download from Zenodo (it shares nothing
with the ``ani1x`` file), whose top-level groups are keyed by atom count;
``n_atoms`` selects those groups. ``split`` in ``{"train", "val", "test"}``
gives a reproducible 80/10/10 partition, and energies/forces convert to eV by
default:

.. code-block:: python

   # capped subset (3- and 4-atom groups) with forces, energies in eV
   data = load_dataset("ani2x", n_atoms=[3, 4], max_conformations=200)  # {"all": [...]}
   train = load_dataset("ani2x", split="train")                 # 80/10/10 split
   # the S/F/Cl elements (16, 9, 17) and forces are present in the structures
   assert "forces" in data["all"][0]

The ``argon_md`` set (periodic argon configurations with energies, forces and
stress, bundled with the repository and used by the ``*_argon_*`` example
notebooks) needs no download; the ``config_type=IsolatedAtom`` reference frame is
dropped automatically:

.. code-block:: python

   splits = load_dataset("argon_md")                         # {"train", "test"}
   train = load_dataset("argon_md", split="train", cutoff=6.0)

The ``lode_dimers`` set (molecular dimers for long-range interactions) selects a
``subset`` and, for the biomolecular dimers, filters by fragment-polarity
``label``; ``return_info=True`` attaches per-frame metadata for binding-energy
analysis:

.. code-block:: python

   cc = load_dataset("lode_dimers", subset="bio", label="CC",
                     split="all", return_info=True)
   binding = cc[0]["energy"] - cc[0]["info"]["energyA"] - cc[0]["info"]["energyB"]

The bundled ``subset="bio_scan"`` (a curated charged/polar dimer distance scan
used by the long-range example notebooks) loads offline with the same
``label`` / ``return_info`` options:

.. code-block:: python

   scan = load_dataset("lode_dimers", subset="bio_scan", split="all",
                       return_info=True)

Where files are cached
======================
Downloads land under ``datasets/<name>/`` in the repository by default. Point
them elsewhere per call with ``cache_dir=``, or globally with the
``XNNS_DATASETS`` (or ``XNNS_CACHE``) environment variable:

.. code-block:: python

   splits = load_dataset("rmd17", molecule="aspirin", cache_dir="/scratch/data")

Re-running is instant and offline once cached (files are verified by MD5). To add
your own dataset to the hub, see :ref:`developer-guide-extending`.
