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

   list_datasets()     # ['ani1', 'ani1ccx', 'ani1x', 'ani2x', 'argon_md', 'lode_dimers', 'rmd17']

Load a dataset
==============
With no ``split`` argument in the ``load_dataset()`` function, you get every
split as a mapping of structure-dict lists. If a specific ``split`` is specified
(often set to ``"train"`` and ``"test"`` ), you get just that split.
Dataset-specific options are passed as keyword arguments (here, we use rMD17's
``molecule`` argument):

.. code-block:: python

   splits = load_dataset("rmd17", molecule="aspirin")    # {"train", "test"}
   train = load_dataset("rmd17", molecule="aspirin", split="train")

Get a training-ready ``AtomicDataset``
======================================
Passing the ``cutoff`` argument to ``load_dataset()`` allows it to return each
data split as an :class:`~xnns.common.data.dataset.AtomicDataset` (neighbour
graphs built on demand), ready for a PyTorch ``DataLoader`` or the
:class:`~xnns.common.train.trainer.Trainer`:

.. code-block:: python

   from torch.utils.data import DataLoader
   from xnns.common.data import collate

   ds = load_dataset("rmd17", molecule="aspirin", cutoff=5.0)
   loader = DataLoader(ds["train"], batch_size=16, shuffle=True, collate_fn=collate)

Splits, units, and subsets
==========================
The keyword arguments for each dataset, documented in the :ref:`data` reference,
enable fine-grained control over what is returned.

rMD17 dataset
--------------
rMD17 ships five official folds which can be selected using the ``fold``
argument. The ``n_train`` argument caps the number of training structures
returned for tractable experiments. Although the units are converted to eV by
default, it can be overridden with ``units`` argument as shown below. 

.. code-block:: python

   train = load_dataset("rmd17", molecule="ethanol", split="train",
                        fold=3, units="kcal/mol", n_train=500)

ANI-1 dataset
--------------
The ``ani1`` set the 20 M-conformation ANI-1 training data, `Smith et al., Chem.
Sci. 8, 3192 (2017) <https://doi.org/10.1039/c6sc05720a>`_ is distributed as one
4.8 GB pyanitools archive. One can select heavy-atom subsets ``ani_gdb_s0X.h5``
and cap the amount returned molecules and conformers for tractable experiments.
The ``split`` argument can be set to ``{"train", "val", "test"}`` which gives
the paper's per-molecule 80/10/10 partition:

.. code-block:: python

   # molecules with 2-4 heavy atoms, capped; energies converted to eV
   data = load_dataset("ani1", heavy_atoms=[2, 3, 4],
                       max_molecules=60, max_conformations=60)   # {"all": [...]}
   train = load_dataset("ani1", heavy_atoms=2, split="train")

ANI-1x dataset
--------------
The ``ani1x`` set is the active-learning training data behind the
:meth:`~xnns.dnn.models.ani.ANI.ani1x` preset (see `Smith, J.; et al. Chem.
Phys. 148, 241733 (2018) <https://doi.org/10.1063/1.5023802>`_ and `Smith, J.;
et al. Sci. Data 7, 134 (2020) <https://doi.org/10.1038/s41597-020-0473-z>`_ for
details). The dataset involves ~5 M conformations with wB97X **energies and
forces** in one 5.6 GB HDF5 file. The argument, ``level``, selects the level of
theory which is set to ``wb97x_dz`` by default (what the model was trained on).
The per-conformation NaN entries are dropped automatically:

.. code-block:: python

   # capped demo subset with forces and energies
   # the energies are converted to eV
   data = load_dataset("ani1x", max_molecules=50)  # {"all": [...]}
   train = load_dataset("ani1x", split="train")    # 80/10/10 split
   ccx = load_dataset("ani1x", level="ccsd(t)_cbs", forces=False)  # energy-only

ANI-1ccx dataset
----------------
The ``ani1ccx`` set is the coupled-cluster companion behind the
:meth:`~xnns.dnn.models.ani.ANI.ani1ccx` preset (see `Smith, J.; et al.
chemrxiv.6744440.v1 (2018) <https://doi.org/10.26434/chemrxiv.6744440.v1>`_ and
`Smith, J.; et al. Sci. Data 7, 134 (2020)
<https://doi.org/10.1038/s41597-020-0473-z>`_ for details). The dataset contains
~500k conformations as a subset of ANI-1x, recomputed at the CCSD(T)*/CBS level
of theory. This subset involves energies only (no coupled-cluster forces). The
dataset lives inside the same release file as ``ani1x``. As such, the download
and cache directories are shared between ANI-1x and ANI-1ccx. Loading the
dataset is equivalent to setting the ``level="ccsd(t)_cbs"`` in the function
call above but under its own name:

.. code-block:: python

   data = load_dataset("ani1ccx", max_molecules=50)   # {"all": [...]}
   train = load_dataset("ani1ccx", split="train")     # 80/10/10 split

ANI-2x dataset
--------------
The ``ani2x`` set is the seven-element training data behind the
:meth:`~xnns.dnn.models.ani.ANI.ani2x` preset (see `Devereux C.; et al. JCTC 16,
4192 (2020) <https://doi.org/10.1021/acs.jctc.0c00121>`_). The dataset contains
~9.6 M conformations with wB97X/6-31G* **energies and forces** for
H/C/N/O/S/F/Cl. It is a separate 3.7 GB pyanitools HDF5 download from Zenodo (it
shares nothing with the ``ani1x`` file), whose top-level groups are keyed by
atom count. The ``n_atoms`` argument selects the data groups. The ``split``
option in ``{"train", "val", "test"}`` gives a reproducible 80/10/10 partition,
and the energies and forces are converted to eV and eV/A by default:

.. code-block:: python

   # capped subset (3- and 4-atom groups) with forces
   data = load_dataset("ani2x", n_atoms=[3, 4], max_conformations=200)  # {"all": [...]}
   train = load_dataset("ani2x", split="train")    # 80/10/10 split
   # the S/F/Cl elements (16, 9, 17) and forces are present in the structures
   assert "forces" in data["all"][0]

Argon MD dataset
----------------
The ``argon_md`` set (periodic argon configurations with energies, forces and
stress, bundled with the repository and used by the ``*_argon_*`` example
notebooks) needs no download; the ``config_type=IsolatedAtom`` reference frame is
dropped automatically:

.. code-block:: python

   splits = load_dataset("argon_md")                         # {"train", "test"}
   train = load_dataset("argon_md", split="train", cutoff=6.0)

LODE dimers dataset
-------------------
The ``lode_dimers`` set  focuses on `molecular dimers for long-range
interactions <https://archive.materialscloud.org/records/405an-d8183>`_. One can
select a ``subset`` and, for the biomolecular dimers, filter the entities by
fragment-polarity ``label``. Setting ``return_info=True`` attaches per-frame
metadata to the entries for binding-energy analysis:

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

Data caching
============
Data downloads land under ``datasets/<name>/`` in the repository by default. You
can direct the data storage to another location per call by setting the
``cache_dir`` argument or globally with the ``XNNS_DATASETS`` (or
``XNNS_CACHE``) environment variable:

.. code-block:: python

   splits = load_dataset("rmd17", molecule="aspirin", cache_dir="/scratch/data")

Re-calling the ``load_dataset()`` is instant and offline once the dataset is
cached (files are verified by MD5). To register your own dataset to the hub, see
:ref:`developer-guide-extending`.
