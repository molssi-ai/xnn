"""Molecular graphs (standardization, resonance, linear torsion centers) and the edge-augmented graph
transformer (degree scaling, read-outs, option checks)."""
import pytest
import torch

from xnn.gnn.models.egt import EGTLayer, MolecularGraphTransformer

SMALL = dict(atom_dim=8, feature_dim=4, node_mlp=(16, 16, 2), edge_mlp=(16, 16, 2), hidden=16, layers=2, heads=2,
             head_dim=4, ffn=(16, 2))


def _chem():
    return pytest.importorskip("rdkit.Chem")


def _features():
    """Ethanol's heavy atoms and one hydrogen: atom and bond features, edges and classes."""
    nodes = torch.tensor([[5, 2, 0, 0, 0], [5, 2, 0, 0, 0], [7, 2, 0, 0, 0], [0, 1, 0, 0, 0]])
    bonds = [(0, 1), (1, 2), (2, 3)]
    edge_index = torch.tensor([[a for i, j in bonds for a in (i, j)], [b for i, j in bonds for b in (j, i)]])
    edges = torch.zeros(2 * len(bonds), 2, dtype=torch.long)
    return nodes, edges, edge_index, torch.arange(4), torch.arange(2 * len(bonds))


def test_preparation_rejects_implicit_hydrogens_and_fragments():
    """Hydrogens must be explicit atoms and a structure one molecule."""
    from xnn.gnn.featurizers.molgraph import prepare_molecule
    Chem = _chem()
    with pytest.raises(ValueError, match="explicit atoms"):
        prepare_molecule(Chem.MolFromSmiles("CO"))
    with pytest.raises(ValueError, match="several fragments"):
        prepare_molecule(Chem.AddHs(Chem.MolFromSmiles("C.O")))


def test_unsupported_bond_type():
    """A quadruple bond has no bond-order feature."""
    from xnn.gnn.featurizers.molgraph import molecular_graph
    Chem = _chem()
    with pytest.raises(ValueError, match="unsupported bond type"):
        molecular_graph(Chem.MolFromSmiles("[C]$[C]"), prepare=False)


def test_resonance_forms_give_one_graph():
    """The charge-separated form of acetamide standardizes to the neutral one: the same structure, bond orders
    and charges."""
    from xnn.gnn.featurizers.molgraph import canonical_resonance, molecular_graph
    Chem = _chem()
    neutral = Chem.AddHs(Chem.MolFromSmiles("CC(=O)N"))
    separated = Chem.RWMol(neutral)
    separated.GetBondBetweenAtoms(1, 2).SetBondType(Chem.BondType.SINGLE)
    separated.GetBondBetweenAtoms(1, 3).SetBondType(Chem.BondType.DOUBLE)
    separated.GetAtomWithIdx(2).SetFormalCharge(-1)
    separated.GetAtomWithIdx(3).SetFormalCharge(1)
    separated = separated.GetMol()
    Chem.SanitizeMol(separated)
    assert Chem.MolToSmiles(separated) != Chem.MolToSmiles(neutral)
    assert Chem.MolToSmiles(canonical_resonance(separated)) == Chem.MolToSmiles(canonical_resonance(neutral))
    ga, gb = molecular_graph(neutral), molecular_graph(separated)
    assert (ga.bond_order == gb.bond_order).all() and not gb.formal_charges.any()
    assert ga.total_charge == gb.total_charge == 0


def test_linear_torsions_about_charged_nitrogen():
    """The two-coordinate cationic nitrogen of an azide is a linear torsion center, unless a neighbor is a
    nitrogen with one hydrogen (hydrazoic acid, its hydrogen an atom of the graph); chains are oriented with
    the smaller first differing end atom first."""
    from xnn.gnn.featurizers.molgraph import _canonical, molecular_graph
    Chem = _chem()
    azide = molecular_graph(Chem.AddHs(Chem.MolFromSmiles("CN=[N+]=[N-]")))
    assert azide.proper_linear.tolist() == [tuple(p) == (0, 1, 2, 3) for p in azide.propers.tolist()]
    acid = molecular_graph(Chem.AddHs(Chem.MolFromSmiles("N=[N+]=[N-]")))
    assert acid.propers.shape[0] > 0 and not acid.proper_linear.any()
    assert _canonical((3, 1, 0)) == (0, 1, 3) and _canonical((2, 5, 2)) == (2, 5, 2)


def test_degree_scaling_of_the_attention():
    """With scale_by_degree the attention-weighted messages are multiplied by the receiving atom's in-degree."""
    torch.manual_seed(0)
    layer = EGTLayer(8, heads=2, head_dim=4, ffn=(8, 2), scale_by_degree=True).double()
    src = torch.tensor([0, 1, 1, 2, 3, 3, 3, 0])
    dst = torch.tensor([1, 0, 2, 1, 0, 1, 2, 3])
    x, e = torch.randn(4, 8, dtype=torch.float64), torch.randn(8, 8, dtype=torch.float64)
    seen = []
    hook = layer.proj.register_forward_hook(lambda module, inputs, output: seen.append(inputs[0]))
    layer(x, e, src, dst)
    layer.scale_by_degree = False
    layer(x, e, src, dst)
    hook.remove()
    degree = torch.bincount(dst, minlength=4).to(torch.float64)
    assert torch.allclose(seen[0], degree[:, None] * seen[1], rtol=1e-12, atol=1e-14)


def test_last_layer_read_out():
    """jumping_knowledge 'last' maps the activated last layer through one linear read-out."""
    torch.manual_seed(1)
    last = MolecularGraphTransformer(**SMALL, out_dim=12, jumping_knowledge="last").double()
    bare = MolecularGraphTransformer(**SMALL, out_dim=16, jumping_knowledge=None).double()
    bare.load_state_dict({k: v for k, v in last.state_dict().items() if not k.startswith("out.")})
    x_last, e_last = last(*_features())
    x_bare, e_bare = bare(*_features())
    assert x_last.shape == (4, 12)
    assert torch.allclose(x_last, last.out(last.act(x_bare)), rtol=1e-12, atol=1e-14)
    assert torch.allclose(e_last, e_bare, rtol=1e-12, atol=1e-14)


def test_transformer_option_checks():
    """Unknown read-outs, an attention width other than the embedding's and a mismatched output width raise."""
    with pytest.raises(ValueError, match="jumping_knowledge"):
        MolecularGraphTransformer(jumping_knowledge="max")
    with pytest.raises(ValueError, match="attention width"):
        MolecularGraphTransformer(**{**SMALL, "hidden": 8})
    with pytest.raises(ValueError, match="out_dim must equal hidden"):
        MolecularGraphTransformer(**SMALL, out_dim=12, jumping_knowledge=None)
