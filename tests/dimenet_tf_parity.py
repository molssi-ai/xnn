"""Helper for ``test_dimenet.py::test_parity_vs_reference_*``.

    python dimenet_tf_parity.py <checkout of gasteigerjo/dimenet> random|pretrained

``random``: builds the authors' TensorFlow DimeNet and DimeNet++ with random
weights in float64, transplants every variable into the xnn models and
compares energies and forces on toy molecules. The last stdout line is the
worst relative error (energy relative to ``|E|``, forces relative to ``max|F|``).

``pretrained``: restores the published DimeNet++ ``U0`` checkpoint (float32,
the only precision it exists in) into the reference model and into xnn and
compares the atomization energies and forces of a few molecules. The last
stdout line is the worst relative error.

Run in a subprocess: :func:`setup` patches module attributes of TensorFlow,
NumPy and the reference code before importing the latter (the fidelity
notebook imports this module and calls :func:`setup` itself). The patches
concern the harness only, not the models:

* ``TF_USE_LEGACY_KERAS=1`` keeps ``tf.keras`` at the Keras 2 API the
  reference code was written for;
* ``np.math`` (dropped in NumPy 2) is the reference basis utilities' spelling
  of ``math``;
* in float64 mode, ``tf.float32`` is rebound to ``tf.float64`` because the
  reference code hard-codes float32 for its own weights and constants, and
  its spherical Bessel zeros (float32 in the reference) are recomputed in
  float64;
* the original DimeNet computes its angle at atom ``i`` (the authors' comment
  in ``dimenet.py`` calls this a known mistake kept for the pretrained
  models); the harness gives it the DimeNet++ angle, as that comment
  recommends for new models.

Conventions absorbed by the transplant (see ``xnn.gnn.models.dimenet``): the
reference radial basis is ``c^1.5 / sqrt 2`` times eq 7 and its 2D basis
``c^1.5`` times eq 6 with an extra ``c / d`` (``reference_basis=True`` in
xnn); its angle is ``pi - alpha``, a sign flip of the odd-degree basis
functions; its embedding block concatenates ``[h_i, h_j, e]``.
"""
import math
import os
import sys

import numpy as np
import torch

CUTOFF = 5.0
tf = None            # the TensorFlow module, set by setup()
TFDimeNet = TFDimeNetPP = swish = basis_utils = None
F64 = True


def setup(upstream: str, mode: str):
    """Import TensorFlow and the reference code with the harness patches.

    Parameters
    ----------
    upstream : str
        Checkout of gasteigerjo/dimenet (the directory holding ``dimenet/``).
    mode : str
        ``"random"`` (float64 everywhere) or ``"pretrained"`` (float32).
    """
    global tf, TFDimeNet, TFDimeNetPP, swish, basis_utils, F64
    assert mode in ("random", "pretrained")
    F64 = mode == "random"
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    np.math = math
    import tensorflow as _tf
    tf = _tf
    try:                       # the reference runs on the CPU (the models are small; the float64
        tf.config.set_visible_devices([], "GPU")   # patches are not exercised on GPU kernels)
    except Exception:          # noqa: BLE001 - already initialized or no GPU
        pass
    if F64:
        tf.keras.backend.set_floatx("float64")
        tf.float32 = tf.float64
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    from dimenet.model.layers import basis_utils as _bu
    basis_utils = _bu
    if F64:
        from scipy.optimize import brentq

        def zeros_f64(n_orders, n_zeros):
            """First ``n_zeros`` zeros of j_0 .. j_{n_orders-1} in float64 (the
            reference computes them in float32), by root bracketing between
            the zeros of the previous order."""
            out = np.zeros((n_orders, n_zeros))
            previous = np.arange(1, n_zeros + n_orders + 1) * np.pi
            out[0] = previous[:n_zeros]
            for order in range(1, n_orders):
                roots = [brentq(_bu.Jn, previous[i], previous[i + 1], (order,),
                                xtol=1e-15, rtol=4 * np.finfo(float).eps)
                         for i in range(len(previous) - 1)]
                previous = np.array(roots)
                out[order] = previous[:n_zeros]
            return out

        _bu.Jn_zeros = zeros_f64
    from dimenet.model.activations import swish as _swish
    from dimenet.model.dimenet import DimeNet as _TFDimeNet
    from dimenet.model.dimenet_pp import DimeNetPP as _TFDimeNetPP
    swish, TFDimeNet, TFDimeNetPP = _swish, _TFDimeNet, _TFDimeNetPP
    # the DimeNet++ angle (between x_j - x_i and x_k - x_j) for the original model too
    TFDimeNet.calculate_neighbor_angles = TFDimeNetPP.calculate_neighbor_angles
    torch.set_default_dtype(torch.float64 if F64 else torch.float32)


def tf_inputs(graph):
    """The reference model's index arrays from an xnn graph, by brute force."""
    src = graph.edge_index[0].tolist()
    dst = graph.edge_index[1].tolist()
    edges = list(zip(src, dst))                           # (j, i): the message j -> i
    kj, ji, t_i, t_j, t_k = [], [], [], [], []
    for e1, (j, i) in enumerate(edges):
        for e2, (k, jj) in enumerate(edges):
            if jj == j and k != i:
                kj.append(e2)
                ji.append(e1)
                t_i.append(i)
                t_j.append(j)
                t_k.append(k)
    as_int = lambda a: tf.constant(np.asarray(a, dtype=np.int32))  # noqa: E731
    np_float = np.float64 if F64 else np.float32
    return {
        "Z": as_int(graph.atomic_numbers.numpy()),
        "R": tf.Variable(graph.pos.numpy().astype(np_float)),
        "batch_seg": as_int(np.zeros(graph.num_nodes)),
        "idnb_i": as_int(dst), "idnb_j": as_int(src),
        "id_expand_kj": as_int(kj), "id_reduce_ji": as_int(ji),
        "id3dnb_i": as_int(t_i), "id3dnb_j": as_int(t_j), "id3dnb_k": as_int(t_k),
    }


def tf_energy_forces(model, inputs):
    """Energy and forces (minus the position gradient) of the reference model."""
    with tf.GradientTape() as tape:
        energy = model(inputs)
    grad = tf.convert_to_tensor(tape.gradient(energy, inputs["R"]))   # gathers give IndexedSlices
    return float(energy.numpy().reshape(-1)[0]), -grad.numpy()


def _kernel(dense):
    return torch.tensor(dense.kernel.numpy().T)            # (out, in)


def _bias(dense):
    return torch.tensor(dense.bias.numpy())


def transplant(tf_model, model, n_spherical, n_radial):
    """Copy the reference variables into the xnn model (built with ``reference_basis=True``)."""
    from xnn.gnn.models.dimenet import DimeNetPP

    c = CUTOFF
    rbf_scale = c ** 1.5 / math.sqrt(2.0)
    sbf_scale = torch.tensor(np.repeat(c ** 1.5 * (-1.0) ** np.arange(n_spherical), n_radial))
    pp = isinstance(model, DimeNetPP)
    F = model.embedding.embedding.weight.shape[1]
    sd = {}
    emb = tf_model.emb_block
    weight = model.embedding.embedding.weight.detach().clone()
    weight[:95] = torch.tensor(emb.embeddings.numpy())
    sd["embedding.embedding.weight"] = weight
    sd["embedding.lin_rbf.weight"] = _kernel(emb.dense_rbf) * rbf_scale
    sd["embedding.lin_rbf.bias"] = _bias(emb.dense_rbf)
    w = _kernel(emb.dense)                                  # columns [x_i | x_j | rbf]
    sd["embedding.lin.weight"] = torch.cat([w[:, F:2 * F], w[:, :F], w[:, 2 * F:]], dim=1)
    sd["embedding.lin.bias"] = _bias(emb.dense)
    sd["rbf.freqs"] = torch.tensor(tf_model.rbf_layer.frequencies.numpy())
    sd["sbf.zeros"] = model.sbf.zeros.detach().clone()
    sd["sbf.norm"] = model.sbf.norm.detach().clone()

    def dense(prefix, layer):
        sd[prefix + ".weight"] = _kernel(layer)
        if layer.bias is not None:
            sd[prefix + ".bias"] = _bias(layer)

    for b, blk in enumerate(tf_model.int_blocks):
        pre = f"interactions.{b}."
        if pp:
            sd[pre + "lin_rbf1.weight"] = _kernel(blk.dense_rbf1) * rbf_scale
            dense(pre + "lin_rbf2", blk.dense_rbf2)
            sd[pre + "lin_sbf1.weight"] = _kernel(blk.dense_sbf1) * sbf_scale[None, :]
            dense(pre + "lin_sbf2", blk.dense_sbf2)
            dense(pre + "lin_down", blk.down_projection)
            dense(pre + "lin_up", blk.up_projection)
        else:
            sd[pre + "lin_rbf.weight"] = _kernel(blk.dense_rbf) * rbf_scale
            sd[pre + "lin_sbf.weight"] = _kernel(blk.dense_sbf) * sbf_scale[None, :]
            # reference W[i_out, b, l_in] contracted over b and l; xnn W[b, f_in, g_out]
            sd[pre + "bilinear"] = torch.tensor(np.transpose(blk.W_bilin.numpy(), (1, 2, 0)))
        dense(pre + "lin_ji", blk.dense_ji)
        dense(pre + "lin_kj", blk.dense_kj)
        for r, res in enumerate(blk.layers_before_skip):
            dense(pre + f"before_skip.{r}.lin1", res.dense_1)
            dense(pre + f"before_skip.{r}.lin2", res.dense_2)
        dense(pre + "lin_skip", blk.final_before_skip)
        for r, res in enumerate(blk.layers_after_skip):
            dense(pre + f"after_skip.{r}.lin1", res.dense_1)
            dense(pre + f"after_skip.{r}.lin2", res.dense_2)
    for o, out in enumerate(tf_model.output_blocks):
        pre = f"outputs.{o}."
        sd[pre + "lin_rbf.weight"] = _kernel(out.dense_rbf) * rbf_scale
        if pp:
            dense(pre + "up", out.up_projection)
        for d, layer in enumerate(out.dense_layers):
            dense(pre + f"dense.{d}", layer)
        dense(pre + "final", out.dense_final)
    sd["atom_ref.weight"] = model.atom_ref.weight.detach().clone()
    model.load_state_dict(sd, strict=True)


def xnn_energy_forces(model, graph):
    from xnn.common.models import ForceStressOutput
    out = ForceStressOutput(model)(graph)
    return float(out["energy"]), out["forces"].detach().numpy()


def compare(label, tf_model, model, n_spherical, n_radial, structures, forces=True):
    """Transplant and compare on every structure; return the worst relative error
    (energy relative to ``|E|``, forces relative to the largest force component)."""
    from xnn.common.data import structure_to_graph
    worst = 0.0
    for s in structures:
        g = structure_to_graph(s, CUTOFF)
        inputs = tf_inputs(g)
        tf_model(inputs)                                    # builds the variables
        transplant(tf_model, model, n_spherical, n_radial)
        e_tf, f_tf = tf_energy_forces(tf_model, inputs)
        e_x, f_x = xnn_energy_forces(model, g)
        de = abs(e_tf - e_x)
        df = float(np.abs(f_tf - f_x).max())
        rel_e = de / abs(e_tf)
        rel_f = df / float(np.abs(f_tf).max())
        print(f"{label}: E_tf={e_tf:+.10f} E_xnn={e_x:+.10f} |dE|={de:.3e} ({rel_e:.1e} relative) "
              f"max|dF|={df:.3e} ({rel_f:.1e} of max|F|={float(np.abs(f_tf).max()):.3g}) "
              f"(edges {g.num_edges}, triplets {len(inputs['id_expand_kj'])})")
        worst = max(worst, rel_e, rel_f if forces else 0.0)
    return worst


def random_structure(n, seed):
    rng = np.random.default_rng(seed)
    return {"pos": rng.uniform(0, 3.6, (n, 3)), "atomic_numbers": [1, 6, 8, 7, 9, 1, 6][:n]}


#: QM9-like molecules for the pretrained check (approximate geometries, Angstrom).
MOLECULES = {
    "methane": {"atomic_numbers": [6, 1, 1, 1, 1],
                "pos": [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                        [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]},
    "water": {"atomic_numbers": [8, 1, 1],
              "pos": [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]},
    "methanol": {"atomic_numbers": [6, 8, 1, 1, 1, 1],
                 "pos": [[-0.0466, 0.6636, 0.0], [-0.0466, -0.7569, 0.0], [-1.0881, 0.9775, 0.0],
                         [0.4379, 1.0714, 0.8901], [0.4379, 1.0714, -0.8901], [0.8606, -1.0608, 0.0]]},
    "formamide": {"atomic_numbers": [6, 8, 7, 1, 1, 1],
                  "pos": [[0.0, 0.3907, 0.0], [1.1587, 0.7595, 0.0], [-1.0493, 1.2541, 0.0],
                          [-0.2735, -0.6753, 0.0], [-0.8619, 2.2385, 0.0], [-1.9949, 0.9223, 0.0]]},
}

#: Hyperparameters of the published DimeNet++ models (config_pp.yaml of the reference).
PP_PUBLISHED = dict(emb_size=128, out_emb_size=256, int_emb_size=64, basis_emb_size=8,
                    num_blocks=4, num_spherical=7, num_radial=6, cutoff=CUTOFF,
                    envelope_exponent=5, num_before_skip=1, num_after_skip=2,
                    num_dense_output=3, num_targets=1, extensive=True,
                    output_init="GlorotOrthogonal")


def pretrained_pp(upstream, target="U0"):
    """The published DimeNet++ model of one QM9 target, restored from the checkpoint."""
    from xnn.common.data import structure_to_graph
    model = TFDimeNetPP(activation=swish, **PP_PUBLISHED)
    model(tf_inputs(structure_to_graph(MOLECULES["methane"], CUTOFF)))   # builds the variables
    # the published files are Keras weight checkpoints with the model as root object
    status = model.load_weights(os.path.join(upstream, "pretrained", "dimenet_pp", target, "ckpt"))
    status.assert_existing_objects_matched()
    return model


def main():
    upstream, mode = sys.argv[1], sys.argv[2]
    setup(upstream, mode)
    from xnn.gnn.models.dimenet import DimeNet, DimeNetPP

    if mode == "random":
        tf.random.set_seed(0)
        torch.manual_seed(0)
        tf_model = TFDimeNet(emb_size=16, num_blocks=2, num_bilinear=4, num_spherical=4,
                             num_radial=3, cutoff=CUTOFF, envelope_exponent=5, num_before_skip=1,
                             num_after_skip=2, num_dense_output=2, num_targets=1,
                             activation=swish, output_init="GlorotOrthogonal")
        model = DimeNet(n_features=16, n_interactions=2, n_rbf=3, n_spherical=4, cutoff=CUTOFF,
                        n_bilinear=4, p=6, n_output_layers=2, reference_basis=True).double()
        worst = compare("DimeNet", tf_model, model, 4, 3,
                        [random_structure(7, 0), random_structure(5, 1)])
        tf_pp = TFDimeNetPP(emb_size=16, out_emb_size=12, int_emb_size=8, basis_emb_size=4,
                            num_blocks=2, num_spherical=4, num_radial=3, cutoff=CUTOFF,
                            envelope_exponent=5, num_before_skip=1, num_after_skip=2,
                            num_dense_output=2, num_targets=1, activation=swish,
                            output_init="GlorotOrthogonal")
        pp = DimeNetPP(n_features=16, n_interactions=2, n_rbf=3, n_spherical=4, cutoff=CUTOFF,
                       n_triplet_features=8, n_basis_features=4, n_output_features=12, p=6,
                       n_output_layers=2, reference_basis=True).double()
        worst = max(worst, compare("DimeNet++", tf_pp, pp, 4, 3,
                                   [random_structure(7, 2), random_structure(6, 3)]))
    else:
        tf_pp = pretrained_pp(upstream)
        pp = DimeNetPP(reference_basis=True).float()
        worst = compare("pretrained DimeNet++ U0", tf_pp, pp, 7, 6, list(MOLECULES.values()))
    print(worst)


if __name__ == "__main__":
    main()
