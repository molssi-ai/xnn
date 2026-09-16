"""Helper for ``test_physnet.py::test_parity_vs_original_physnet``.

Builds the original TF1 PhysNet graph (from a MMunibas/PhysNet clone, path in
``sys.argv[1]``), transplants every variable into the xnn PhysNet, and
compares energies, forces, corrected charges, and the non-hierarchicality
penalty on a toy molecule. Prints the worst absolute difference on the last
stdout line; run in a subprocess because it patches ``sys.modules`` and
disables TF eager mode.
"""
import sys

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
sys.modules["tensorflow"] = tf  # upstream modules do `import tensorflow as tf`
# upstream applies dropout with keep_prob = 1.0 (identity); its float32
# placeholder trips TF2's dtype check in float64 graphs
tf.nn.dropout = lambda x, keep_prob=None, **kw: x

sys.path.insert(0, sys.argv[1])
import neural_network.NeuralNetwork as _nnmod  # noqa: E402
from neural_network.NeuralNetwork import NeuralNetwork  # noqa: E402

# upstream forgets to forward dtype to RBFLayer (float32 hard-coded)
_OrigRBF = _nnmod.RBFLayer
_nnmod.RBFLayer = lambda K, cutoff, scope=None: _OrigRBF(
    K, cutoff, scope=scope, dtype=tf.float64)

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from xnn.common.config import from_dict  # noqa: E402
from xnn.common.data import AtomicGraph  # noqa: E402
from xnn.common.models import ForceStressOutput, build_model  # noqa: E402

F_DIM, K, SR_CUT, NB, NRA, NRI, NRO = 24, 16, 4.0, 3, 2, 3, 1


def run_case(lr_cut, use_ele, use_disp, q_tot, seed):
    tf.reset_default_graph()
    # float64-exact shifted softplus (upstream's subtracts a float32 log(2))
    act = lambda x: tf.nn.softplus(x) - np.log(2.0)  # noqa: E731
    nn = NeuralNetwork(F=F_DIM, K=K, sr_cut=SR_CUT, lr_cut=lr_cut,
                       num_blocks=NB, num_residual_atomic=NRA,
                       num_residual_interaction=NRI, num_residual_output=NRO,
                       use_electrostatic=use_ele, use_dispersion=use_disp,
                       Eshift=0.1, Escale=1.3, Qshift=0.01, Qscale=0.9,
                       activation_fn=act, dtype=tf.float64, scope="nn",
                       seed=seed)

    rng = np.random.default_rng(seed)
    N = 8
    R_np = rng.uniform(0, 3.5, (N, 3))
    Z_np = np.array([8, 1, 1, 6, 1, 1, 1, 1])
    idx_i = np.repeat(np.arange(N), N - 1)
    idx_j = np.concatenate([[j for j in range(N) if j != i] for i in range(N)])

    Z = tf.constant(Z_np, dtype=tf.int32)
    R = tf.constant(R_np, dtype=tf.float64)
    ii = tf.constant(idx_i, dtype=tf.int32)
    jj = tf.constant(idx_j, dtype=tf.int32)
    Qt = tf.constant([q_tot], dtype=tf.float64)
    energy_op, forces_op = nn.energy_and_forces(Z, R, ii, jj, Q_tot=Qt)
    Ea_op, Qa_raw_op, Dij_op, nh_op = nn.atomic_properties(Z, R, ii, jj)
    Qa_op = nn.scaled_charges(Z, Qa_raw_op, Q_tot=Qt)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # the k2f and output heads are zero-initialized upstream, which would
        # leave the interaction blocks unexercised -- randomize them
        rng_w = np.random.default_rng(seed + 100)
        for v in tf.global_variables():
            if "k2f/W" in v.name or "dense_layer/W" in v.name:
                sess.run(v.assign(0.2 * rng_w.standard_normal(
                    v.shape.as_list())))
        vals = {v.name: sess.run(v) for v in tf.global_variables()}
        E_tf, F_tf, Qa_tf, nh_tf = sess.run(
            [energy_op, forces_op, Qa_op, nh_op])

    cfg = from_dict({"model": {
        "name": "physnet", "cutoff": SR_CUT, "n_features": F_DIM, "n_rbf": K,
        "n_interactions": NB,
        "extra": {"lr_cutoff": lr_cut, "num_residual_atomic": NRA,
                  "num_residual_interaction": NRI, "num_residual_output": NRO,
                  "use_electrostatics": use_ele, "use_dispersion": use_disp}}})
    x = build_model(cfg.model)

    def g(name):
        return torch.tensor(vals[f"nn/{name}:0"])

    with torch.no_grad():
        x.embeddings.copy_(g("embeddings"))
        x.rbf_layer.centers.copy_(g("rbf_layer/centers"))
        x.rbf_layer.widths.copy_(g("rbf_layer/widths"))
        x.Eshift.copy_(g("Eshift")); x.Escale.copy_(g("Escale"))
        x.Qshift.copy_(g("Qshift")); x.Qscale.copy_(g("Qscale"))
        x._s6.copy_(g("s6")); x._s8.copy_(g("s8"))
        x._a1.copy_(g("a1")); x._a2.copy_(g("a2"))

        def copy_dense(dst, scope, bias=True):
            dst.weight.copy_(g(f"{scope}/W"))
            if bias:
                dst.bias.copy_(g(f"{scope}/b"))

        def copy_res(dst, scope):
            copy_dense(dst.dense, f"{scope}/dense")
            copy_dense(dst.residual, f"{scope}/residual")

        for b in range(NB):
            ib, sc = x.interaction_blocks[b], f"interaction_block{b}"
            il = ib.interaction
            copy_dense(il.k2f, f"{sc}/interaction_layer/k2f", bias=False)
            copy_dense(il.dense_i, f"{sc}/interaction_layer/dense_i")
            copy_dense(il.dense_j, f"{sc}/interaction_layer/dense_j")
            for k, res in enumerate(il.residuals):
                copy_res(res, f"{sc}/interaction_layer/residual_layer{k}")
            copy_dense(il.dense, f"{sc}/interaction_layer/dense")
            il.u.copy_(g(f"{sc}/interaction_layer/u"))
            for k, res in enumerate(ib.residuals):
                copy_res(res, f"{sc}/residual_layer{k}")
            ob, osc = x.output_blocks[b], f"output_block{b}"
            for k, res in enumerate(ob.residuals):
                copy_res(res, f"{osc}/residual_layer{k}")
            ob.dense.weight.copy_(g(f"{osc}/dense_layer/W"))

    graph = AtomicGraph(
        pos=torch.tensor(R_np), atomic_numbers=torch.tensor(Z_np),
        edge_index=torch.tensor(np.stack([idx_j, idx_i])),
        cell_shifts=torch.zeros(len(idx_i), 3, dtype=torch.long),
        batch=torch.zeros(N, dtype=torch.long),
        n_atoms=torch.tensor([N]), cell=None, pbc=None)
    graph.total_charge = torch.tensor([q_tot])
    out = ForceStressOutput(x)(graph)

    return max(
        abs(float(out["energy"]) - float(E_tf)),
        float(np.abs(out["forces"].detach().numpy() - F_tf).max()),
        float(np.abs(out["charges"].detach().numpy() - Qa_tf).max()),
        abs(float(out["nh_loss"]) - float(nh_tf)))


worst = 0.0
for lr, ele, disp, qt, seed in [(None, True, True, 0.0, 7),
                                (None, True, False, -1.0, 8),
                                (6.0, True, True, 1.0, 9),
                                (6.0, False, True, 0.0, 10),
                                (None, False, False, 0.0, 11)]:
    d = run_case(lr, ele, disp, qt, seed)
    print(f"lr_cut={lr} ele={ele} disp={disp} Qtot={qt}: worst diff {d:.3e}")
    worst = max(worst, d)
print(worst)
