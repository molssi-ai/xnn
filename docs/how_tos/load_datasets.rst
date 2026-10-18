.. _howto-load-datasets:

******************************
Download an Upstream Dataset
******************************

xnn offers a dataset hub which is a collection of popular benchamrk datasets in
Machine Learning Interatomic Potential (MLIP) research. You can get a list of
the available datasets with :func:`~xnn.common.data.list_datasets`. The
``load_dataset()`` downloads, verifies, converts and caches a benchmark dataset,
listed above in one call. The ``examples/data/load_dataset_tutorial.ipynb``
notebook offers a runnable version of this page.

.. code-block:: python

   from xnn.common.data import load_dataset, list_datasets

   # List the available datasets in the hub
   list_datasets()   # ['ani1', 'ani1ccx', 'ani1x', 'ani2x', 'lode_dimers', 'qm7', 'qm9', 'rmd17', ...]

   # Load a dataset by name and optional keyword arguments
   splits = load_dataset("rmd17", molecule="aspirin")                 # {"train": [...], "test": [...]}
   train = load_dataset("rmd17", molecule="aspirin", split="train")   # One list of structures
   ds = load_dataset("rmd17", molecule="aspirin", cutoff=5.0)         # AtomicDataset per split

Without the ``split`` argument, you get every split available in the data as a
mapping of structure lists. With ``cutoff``, each split comes back as an
:class:`~xnn.common.data.dataset.AtomicDataset`, ready for the
:class:`~xnn.common.train.trainer.Trainer` or a ``DataLoader`` with
:func:`~xnn.common.data.dataset.collate` to be subsequently used in training or
evaluation.

Dataset options
===============
Each dataset offers its own keyword arguments for more control during
the pre-processing (see the full list in :ref:`data`):

.. code-block:: python

   # rMD17: ten molecules, five official folds, energies are in eV by default
   load_dataset("rmd17", molecule="ethanol", split="train", fold=3, n_train=500)

   # QM7: 7165 molecules with atomization energies, test fold 1 of the stratified five
   load_dataset("qm7", split="train", fold=1)

   # QM9: 130,831 molecules, U0 in eV, the 110,000 / 10,000 / 10,831 split of seed 0
   load_dataset("qm9", split="train", label="U0")

   # Ethanol with dipoles and polarizabilities: seeded 8000 / 1000 / 1000 split
   load_dataset("ethanol_response", split="train")

   # ANI-1: 20 M conformations in one archive; pick heavy-atom subsets and cap the size
   load_dataset("ani1", heavy_atoms=[2, 3, 4], max_molecules=60, max_conformations=60)

   # ANI-1x (energies and forces), ANI-1ccx (coupled-cluster energies),
   # ANI-2x (seven elements)
   load_dataset("ani1x", split="train")
   load_dataset("ani1ccx", max_molecules=50)
   load_dataset("ani2x", n_atoms=[3, 4], max_conformations=200)

   # The 4G-HDNNP benchmarks with Hirshfeld charges: carbon_chain, ag_clusters, 
   # nacl_clusters, au2_mgo
   load_dataset("hdnnp4g", system="ag_clusters", split="all")

   # The same geometries in two charge or spin states (the SpookyNet set): ag3 or carbene
   load_dataset("spookynet_states", system="carbene", split="train")

   # Argon MD: periodic frames with stress, bundled with the repository
   load_dataset("argon_md", split="train", cutoff=6.0)

   # LODE dimers: binding energy curves with per-frame metadata
   cc = load_dataset("lode_dimers", subset="bio", label="CC", split="all", return_info=True)
   binding = cc[0]["energy"] - cc[0]["info"]["energyA"] - cc[0]["info"]["energyB"]

   # S22 / S66 noncovalent dimers: fragment labels and reference interaction energies
   dimers = load_dataset("s66", split="all")
   dimers[0]["fragment"], dimers[0]["interaction_energy"]       # per-atom 0 / 1, kcal/mol

Accessing the ANI sets requires installing the ``hub`` extra optional dependency
(``h5py``). The ``split`` of the ANI sets (``train`` / ``val`` / ``test``) is
based on their original paper's per-molecule 80/10/10 partition.

Where the files go?
===================
The downloaded files land under ``datasets/<name>/`` within the repository. You
can change this behavior per call by setting the ``cache_dir`` argument or
system-wide by setting the environment variables ``XNN_DATASETS`` or ``XNN_CACHE``:

.. code-block:: python

   load_dataset("rmd17", molecule="aspirin", cache_dir="/scratch/data")

Subsequent calls will read the cached dataset instantly and in the offline mode.
To register a dataset of your own, see the :ref:`developer-guide-extending`
guide.
