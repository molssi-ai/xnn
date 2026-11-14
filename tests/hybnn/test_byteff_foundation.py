"""The released ByteFF and ByteFF-Pol checkpoints: parameter names, architecture, loading and the model hub."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
from test_byteff import GRAPH, SMALL_POL, _coords, _graphs, _mol, _z  # noqa: E402
from xnn.hybnn.models import ByteFF, ByteFFPol  # noqa: E402
from xnn.hybnn.models import byteff_foundation as bf  # noqa: E402

PLAIN = dict(graph=GRAPH, bonded=dict(pre=(16, 16, 2), out=(16, 2)), nonbonded=dict(pre=(16, 16, 2), out=(16, 2)))

# the released names of xnn's modules
_EMBED = {"element": "atom_embedding", "degree": "connectivity_embedding", "formal_charge": "fm_chg_embedding",
          "ring_degree": "ring_con_embedding", "min_ring": "min_ring_size_embedding",
          "bond_ring": "bond_ring_embedding", "bond_order": "bond_order_embedding"}
_LAYER = {"key": "lin_key", "query": "lin_query", "value": "lin_value", "edge_proj": "lin_proj_edge",
          "proj": "lin_proj", "gate": "lin_gate", "edge_bias": "lin_edge", "edge_norm1": "norm1_edge",
          "edge_norm2": "norm2_edge", "norm1": "norm1", "norm2": "norm2", "edge_ffn": "ffn_edge", "ffn": "ffn"}
_HEADS = {"sigma": "sigma_mlp", "epsilon": "epsilon_mlp", "charge.pre": "charge_pre_mlp",
          "charge.out": "charge_out_mlp", "volume": "volume_mlp", "lam": "lamb_mlp", "eps": "eps_mlp",
          "ct_eps": "ct_eps_mlp", "ct_lam": "ct_lamb_mlp"}
_FREE = {"v_free": "v_free", "c6_free": "c6_free", "r_free": "rvdw_free", "alpha_free": "alpha_free"}


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _released(model) -> dict:
    """The weights of ``model``'s first network under the released parameter names."""
    member = model.members[0]
    bonded = "PreMMBondedConj" if member.bonded.form == "two_well" else "PreMMBonded"
    pre = "preff_block.pre_layers"
    out = {}

    def mlp(parts):
        return ".".join(parts).replace("layers.", "lins.")

    for key, value in member.state_dict().items():
        p = key.split(".")
        if p[0] == "graph" and p[1] == "embedding":
            name = f"{_EMBED[p[2]]}.{p[3]}" if p[2] in _EMBED else f"{p[2]}.{mlp(p[3:])}"
            key = f"graph_block.feature_layer.{name}"
        elif p[0] == "graph" and p[1] in ("edge_in", "edge_out"):
            key = f"graph_block.gnn_layer.{p[1]}_lin.{p[2]}"
        elif p[0] == "graph" and p[1] == "out":
            key = f"graph_block.gnn_layer.gnn.lin.{p[2]}"
        elif p[0] == "graph":
            key = f"graph_block.gnn_layer.gnn.convs.{p[2]}.{_LAYER[p[3]]}.{mlp(p[4:])}"
        elif p[0] == "bonded":
            stage = {"pool": "pre", "post": "post", "out": "out"}[p[1]]
            key = f"{pre}.{bonded}.{stage}_mlp.{p[2]}.{mlp(p[3:])}"
        elif p[1] in _FREE:
            key, value = f"{pre}.PreExp6Pol.{_FREE[p[1]]}.weight", value.reshape(-1, 1)
        else:
            head, rest = (".".join(p[1:3]), p[3:]) if p[1] == "charge" else (p[1], p[2:])
            owner = ("PreChargeVolume" if head in ("charge.pre", "charge.out", "volume") else "PreExp6Pol") \
                if model.polarizable else "PreLJEs"
            key = f"{pre}.{owner}.{_HEADS[head]}.{mlp(rest)}"
        out[key] = value.detach().clone()
    return out


def _parameters_equal(a, b):
    pa, pb = a.predicted_parameters(), b.predicted_parameters()
    assert pa.keys() == pb.keys()
    for k in pa:
        assert torch.allclose(pa[k], pb[k], rtol=1e-14, atol=1e-14), k


@pytest.mark.parametrize("polarizable", [False, True])
def test_released_checkpoint_round_trip(tmp_path, polarizable):
    """A checkpoint under the released names maps back onto the network, its architecture is read from the
    shapes, and the loaded model predicts the original's parameters and energies."""
    torch.manual_seed(1)
    original = ByteFFPol(**SMALL_POL) if polarizable else ByteFF(**PLAIN)
    with torch.no_grad():
        for name, p in original.named_parameters():
            if not name.endswith("_free"):
                p.add_(0.01 * torch.randn_like(p))
    state = _released(original)
    converted = bf.convert_state_dict(state)
    reference = original.members[0].state_dict()
    assert converted.keys() == reference.keys()
    assert all(torch.equal(converted[k], reference[k]) for k in reference)
    cls, arch = bf.architecture(state)
    assert cls is (ByteFFPol if polarizable else ByteFF)
    assert arch == dict(graph={**GRAPH, "jumping_knowledge": "cat"},
                        bonded=dict(pre=(16, 16, 2), out=(16, 2), form="two_well" if polarizable else "harmonic"),
                        nonbonded=dict(pre=(16, 16, 2), out=(16, 2)))
    path = tmp_path / "optimal.pt"
    torch.save({"model_state_dict": state}, path)
    mols = [_mol("CC(=O)NC"), _mol("O")] if polarizable else [_mol("CC(=O)NC")]
    loaded = bf.from_byteff2(path, structure=mols, dtype=torch.float64)
    assert type(loaded) is cls and not loaded.training
    original.bind(mols)
    _parameters_equal(loaded, original)
    coords = np.concatenate([_coords(m)[0] + 4.0 * k for k, m in enumerate(mols)])[None]
    graph = _graphs(_z(mols), coords, original)
    assert torch.allclose(loaded(graph)["energy"], original.eval()(graph)["energy"], rtol=1e-12, atol=1e-12)


def test_unknown_parameters_are_rejected():
    """A parameter without an xnn counterpart raises a KeyError."""
    for key in ("graph_block.feature_layer.other_embedding.weight", "model.extra.weight"):
        with pytest.raises(KeyError, match="no xnn counterpart"):
            bf.convert_state_dict({key: torch.zeros(2, 2)})


def test_hub_converts_a_released_checkpoint(tmp_path):
    """The model hub recognizes a released checkpoint and converts it with the config that rebuilds it."""
    from xnn.common.models.hub.checkpoint import load_checkpoint
    from xnn.common.models.hub.formats import detect_format
    torch.manual_seed(2)
    original = ByteFF(**PLAIN)
    path = tmp_path / "optimal.pt"
    torch.save({"model_state_dict": _released(original)}, path)
    assert detect_format(path) == "byteff2"
    (tmp_path / "other.pt").write_bytes(b"not a checkpoint")
    assert not bf._detect(tmp_path / "other.pt")
    ckpt = load_checkpoint(path)
    assert ckpt.config.model.name == "byteff"
    assert ckpt.config.model.extra == bf.architecture(_released(original))[1]
    weights = {k.split("members.0.", 1)[1]: v for k, v in ckpt.state_dict.items() if "members.0." in k}
    assert all(torch.equal(weights[k], v) for k, v in original.members[0].state_dict().items())


def test_from_foundation_downloads_once_then_reads_the_cache(tmp_path, monkeypatch):
    """A released name is fetched into the hub cache once, takes the card's elements and loads from the cache
    afterwards; unknown names and missing files offline raise."""
    from xnn.common.models.hub.registry import get_card
    torch.manual_seed(3)
    original = ByteFFPol(**SMALL_POL)
    source = tmp_path / "released.pt"
    torch.save({"model_state_dict": _released(original)}, source)
    calls = []

    def download(url, dest, md5=None, **kwargs):
        calls.append((url, md5))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)
        return Path(dest)

    monkeypatch.setattr("xnn.common.data.hub._download.download_file", download)
    card = get_card("byteff-pol-25")
    cache = tmp_path / "cache"
    model = bf.from_foundation("byteff-pol-25", structure=_mol("O"), cache_dir=cache)
    assert calls == [(card.url, card.md5)]
    assert isinstance(model, ByteFFPol) and model.supported_elements == tuple(card.species)
    original.bind(_mol("O"))
    _parameters_equal(model, original)
    cached = bf.from_foundation("byteff-pol-25", cache_dir=cache, local_files_only=True)
    assert len(calls) == 1 and not cached.bound
    with pytest.raises(ValueError, match="unknown released model"):
        bf.from_foundation("byteff-0", cache_dir=cache)
    with pytest.raises(FileNotFoundError, match="not cached"):
        bf.from_foundation("byteff-26-member-2", cache_dir=cache, local_files_only=True)
