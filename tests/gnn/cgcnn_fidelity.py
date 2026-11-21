"""Fidelity of the xnn CGCNN to the authors' code (txie-93/cgcnn, MIT).

    python cgcnn_fidelity.py <checkout of txie-93/cgcnn> [float64|float32]

Rattled crystals are written as CIF files and read through the authors' own
data pipeline (pymatgen neighbors, their Gaussian expansion and element
table); xnn reads the same files. The checks:

* random weights and batch-normalization statistics transplanted into xnn: the
  property of every crystal alone and in one batch in evaluation mode, and a
  classifier's log-probabilities;
* training mode on the batch: the output, the gradient of a squared loss for
  every parameter and the updated running statistics;
* every published checkpoint in ``pre-trained/``, converted by
  :func:`xnn.gnn.models.cgcnn_foundation.convert_checkpoint`.

Errors are relative to ``max(|reference|, 1)``; the last printed line is the
worst. Two patches concern the harness only: the authors' pipeline makes
float32 tensors, which are made in the run's dtype instead, and
``torch.nn.Softplus`` drops the tail ``log(1 + e^-x)`` above ``x = 20``
(below float32 rounding, 2e-9 in float64), so float64 runs evaluate it exactly.
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

from xnn.common.data import collate, structure_to_graph
from xnn.gnn.models.cgcnn import CGCNN
from xnn.gnn.models.cgcnn_foundation import convert_checkpoint

DTYPE = torch.float64


class _Torch:
    """The ``torch`` the authors' data module sees: its float tensors in the run's dtype."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def Tensor(x):  # noqa: N802 - the name the authors' module calls
        return torch.as_tensor(np.asarray(x), dtype=DTYPE)


def setup(repo, dtype):
    """Import the authors' modules from ``repo`` with the harness patches; returns ``(data, model)``."""
    global DTYPE
    DTYPE = dtype
    sys.path.insert(0, str(repo))
    import torch.nn.functional as F
    from cgcnn import data, model

    def softplus(self, x):
        return F.softplus(x, threshold=40.0 if x.dtype == torch.float64 else 20.0)
    torch.nn.Softplus.forward = softplus
    data.torch = _Torch()
    return data, model


def crystals(seed=0):
    """Rattled crystals as pymatgen structures; every atom has at least 12 neighbors within 8 Angstrom."""
    from pymatgen.core import Lattice, Structure
    rng = np.random.default_rng(seed)
    out = [
        Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]),
        Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
                  [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * (2, 1, 1),
        Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    ]
    for n, elements, a in ((16, ["Li", "Fe", "P", "O"], 6.5), (10, ["Ga", "N", "Al"], 5.5)):
        lattice = Lattice(np.eye(3) * a + rng.uniform(-0.4, 0.4, (3, 3)))
        s = Structure(lattice, [], [])
        while len(s) < n:
            frac = rng.uniform(0, 1, 3)
            if len(s) == 0 or min(s.lattice.get_all_distances(frac, s.frac_coords)[0]) > 1.6:
                s.append(elements[len(s) % len(elements)], frac)
        out.append(s)
    for s in out:
        for i in range(len(s)):
            s.translate_sites([i], rng.normal(0.0, 0.05, 3), frac_coords=False)
    return out


def read_both(data, files, root, atom_init):
    """The authors' inputs (one tuple per crystal) and xnn graphs of the same CIF files."""
    from pymatgen.core import Structure
    with open(Path(root) / "id_prop.csv", "w") as f:
        f.write("".join(f"{Path(p).stem},0\n" for p in files))
    os.symlink(atom_init, Path(root) / "atom_init.json")
    dataset = data.CIFData(str(root))
    by_id = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not find enough neighbors")
        for i in range(len(dataset)):
            item = dataset[i]
            by_id[item[2]] = item
    upstream, graphs = [], []
    for p in files:
        upstream.append(by_id[Path(p).stem])
        s = Structure.from_file(p)
        g = {"pos": s.cart_coords, "atomic_numbers": s.atomic_numbers, "cell": s.lattice.matrix, "pbc": [True] * 3}
        default = torch.get_default_dtype()
        torch.set_default_dtype(DTYPE)
        try:
            graphs.append(structure_to_graph(g, 8.0))
        finally:
            torch.set_default_dtype(default)
    return upstream, graphs


def randomize_batch_norms(net, seed):
    """Running statistics and affine parameters away from their identity values."""
    gen = torch.Generator().manual_seed(seed)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            n = m.num_features
            m.running_mean.copy_(0.5 * torch.randn(n, generator=gen))
            m.running_var.copy_(0.5 + 1.5 * torch.rand(n, generator=gen))
            with torch.no_grad():
                m.weight.copy_(0.5 + torch.rand(n, generator=gen))
                m.bias.copy_(0.1 * torch.randn(n, generator=gen))


def rel(mine, ref):
    mine, ref = torch.as_tensor(mine).double(), torch.as_tensor(ref).double()
    return float(((mine - ref).abs() / ref.abs().clamp(min=1.0)).max())


def main():
    args = sys.argv[1:]
    dtype = torch.float64
    if args and args[-1] in ("float64", "float32"):
        dtype = getattr(torch, args.pop())
    repo = Path(args[0]).resolve()
    data, model = setup(repo, dtype)
    atom_init = repo / "data" / "sample-regression" / "atom_init.json"
    worst = 0.0

    def report(label, err):
        nonlocal worst
        worst = max(worst, err)
        print(f"{label}: relative error {err:.3e}")

    with tempfile.TemporaryDirectory() as root:
        files = []
        for k, s in enumerate(crystals()):
            path = Path(root) / f"c{k}.cif"
            s.to(filename=str(path))
            files.append(str(path))
        upstream, graphs = read_both(data, files, root, atom_init)
        print(f"{len(files)} crystals, {data.GaussianDistance(0, 8, 0.2).filter.size} Gaussians")
        batch_in, _, _ = data.collate_pool(upstream)
        batch = collate(graphs)

        for classification in (False, True):
            torch.manual_seed(1)
            up = model.CrystalGraphConvNet(92, 41, atom_fea_len=32, n_conv=3, h_fea_len=48, n_h=2,
                                           classification=classification)
            randomize_batch_norms(up, 2)
            up = up.to(dtype).eval()
            ours = CGCNN(n_features=32, n_interactions=3, n_rbf=41, readout_sizes=[48, 48],
                         classification=classification, atom_features=str(atom_init)).to(dtype).eval()
            missing, unexpected = ours.load_state_dict(up.state_dict(), strict=False)
            assert not unexpected and set(missing) <= {"atom_features", "rbf.centers", "property_scale",
                                                       "property_shift"}, (missing, unexpected)
            key = "log_probabilities" if classification else "properties"
            kind = "classifier" if classification else "regression"
            with torch.no_grad():
                for (inputs, _, cif_id), g in zip(upstream, graphs):
                    ref = up(*inputs, [torch.arange(inputs[0].shape[0])])
                    mine = ours(g)[key].reshape(ref.shape)
                    report(f"{kind} {cif_id}", rel(mine, ref))
                ref = up(*batch_in)
                report(f"{kind} batch", rel(ours(batch)[key].reshape(ref.shape), ref))
            if classification:
                continue

            # training mode: batch statistics, gradients and running statistics
            up.train()
            ours.train()
            target = torch.linspace(-1.0, 1.0, len(graphs), dtype=dtype)
            ref = up(*batch_in).reshape(-1)
            mine = ours(batch)["properties"]
            report("training output", rel(mine.detach(), ref.detach()))
            ((ref - target) ** 2).sum().backward()
            ((mine - target) ** 2).sum().backward()
            theirs = dict(up.named_parameters())
            scale = max(float(p.grad.abs().max()) for p in theirs.values())
            err = max(float((p.grad - theirs[n].grad).abs().max()) / max(scale, 1.0)
                      for n, p in ours.named_parameters() if n in theirs)
            report("training gradients", err)
            buffers = dict(up.named_buffers())
            err = max(rel(b, buffers[n]) for n, b in ours.named_buffers() if n in buffers)
            report("running statistics", err)

        # the published checkpoints
        for path in sorted((repo / "pre-trained").glob("*.pth.tar")):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            a = checkpoint["args"]
            classification = a["task"] == "classification"
            up = model.CrystalGraphConvNet(92, 41, atom_fea_len=a["atom_fea_len"], n_conv=a["n_conv"],
                                           h_fea_len=a["h_fea_len"], n_h=a["n_h"], classification=classification)
            up.load_state_dict(checkpoint["state_dict"])
            up = up.to(dtype).eval()
            ours = convert_checkpoint(path, dtype, atom_features=str(atom_init))
            with torch.no_grad():
                ref = up(*batch_in)
                if classification:
                    ref = ref.exp()[:, 1]
                else:
                    norm = checkpoint["normalizer"]
                    std, mean = (torch.as_tensor(norm[k], dtype=dtype) for k in ("std", "mean"))
                    ref = ref.reshape(-1) * std + mean
                mine = ours(batch)["properties"]
            print(f"{path.name} args: data_options {a.get('data_options')}, atom_fea_len {a['atom_fea_len']}, "
                  f"n_conv {a['n_conv']}, h_fea_len {a['h_fea_len']}, n_h {a['n_h']}; values {ref.tolist()}")
            report(f"pretrained {path.name}", rel(mine, ref))
    print(worst)


if __name__ == "__main__":
    main()
