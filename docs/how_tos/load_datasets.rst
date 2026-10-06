.. _howto-load-datasets:

******************************
Download an Upstream Dataset
******************************

``load_dataset()`` downloads, verifies, converts and caches a benchmark
dataset in one call. ``examples/data/load_dataset_tutorial.ipynb`` is the
runnable version of this page.

.. code-block:: python

   from xnn.common.data import load_dataset, list_datasets

   list_datasets()   # ['ani1', 'ani1ccx', 'ani1x', 'ani2x', 'argon_md', 'lode_dimers', 'rmd17']

   splits = load_dataset("rmd17", molecule="aspirin")                 # {"train": [...], "test": [...]}
   train = load_dataset("rmd17", molecule="aspirin", split="train")   # one list of structures
   ds = load_dataset("rmd17", molecule="aspirin", cutoff=5.0)         # AtomicDataset per split

Without ``split`` you get every split as a mapping of structure lists. With
``cutoff`` each split comes back as an
:class:`~xnn.common.data.dataset.AtomicDataset`, ready for the
:class:`~xnn.common.train.trainer.Trainer` or a ``DataLoader`` with
:func:`~xnn.common.data.dataset.collate`.

Dataset options
===============
Each dataset takes its own keyword arguments (full list in :ref:`data`):

.. code-block:: python

   # rMD17: ten molecules, five official folds, eV by default
   load_dataset("rmd17", molecule="ethanol", split="train", fold=3, n_train=500)

   # ANI-1: 20 M conformations in one archive; pick heavy-atom subsets and cap the size
   load_dataset("ani1", heavy_atoms=[2, 3, 4], max_molecules=60, max_conformations=60)

   # ANI-1x (energies and forces), ANI-1ccx (coupled-cluster energies), ANI-2x (seven elements)
   load_dataset("ani1x", split="train")
   load_dataset("ani1ccx", max_molecules=50)
   load_dataset("ani2x", n_atoms=[3, 4], max_conformations=200)

   # Argon MD: periodic frames with stress, bundled with the repository
   load_dataset("argon_md", split="train", cutoff=6.0)

   # LODE dimers: binding curves with per-frame metadata
   cc = load_dataset("lode_dimers", subset="bio", label="CC", split="all", return_info=True)
   binding = cc[0]["energy"] - cc[0]["info"]["energyA"] - cc[0]["info"]["energyB"]

The ANI sets need the ``hub`` extra (``h5py``). The ``split`` of the ANI sets
(``train`` / ``val`` / ``test``) is the paper's per-molecule 80/10/10
partition.

Where the files go
==================
Downloads land under ``datasets/<name>/`` in the repository. Change that per
call with ``cache_dir`` or globally with the ``XNN_DATASETS`` (or
``XNN_CACHE``) environment variable:

.. code-block:: python

   load_dataset("rmd17", molecule="aspirin", cache_dir="/scratch/data")

A cached dataset loads again instantly and offline. To register a dataset of
your own, see :ref:`developer-guide-extending`.
