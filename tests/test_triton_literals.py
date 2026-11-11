"""Float literals in the Triton kernels.

Compiled Triton may turn a Python float literal into a float32 constant, also inside float64 arithmetic,
while the CPU interpreter of the tests keeps it exact. A literal that float32 cannot represent (0.6, 0.04)
therefore puts a 1e-8 error into float64 results on the GPU that no interpreted test sees. Inexact values
belong in a float64 table, or are written with exact literals (``x / 25.0``). Thresholds, floors and tiny
offsets are left as they are: their rounding changes nothing beyond round-off.
"""
import ast
import struct
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "xnn"

# deliberate margins whose float32 rounding does not matter
ALLOWED = {("common/data/grid_pairs.py", "cut2 * 1.000001")}


def _float32_exact(value: float) -> bool:
    return struct.unpack("f", struct.pack("f", value))[0] == value


def _harmless(node, parents) -> bool:
    """Whether an inexact literal is a threshold, a floor or ceiling, or an offset of at most 1e-6."""
    parent = parents[node]
    if isinstance(parent, ast.UnaryOp):
        node, parent = parent, parents[parent]
    if isinstance(parent, ast.Compare):
        return True
    if isinstance(parent, ast.Call) and ast.unparse(parent.func).split(".")[-1] in ("maximum", "minimum", "where"):
        return True
    value = node.value if isinstance(node, ast.Constant) else node.operand.value
    return isinstance(parent, ast.BinOp) and isinstance(parent.op, (ast.Add, ast.Sub)) and abs(value) <= 1e-6


def _jit_functions(tree):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and any(ast.unparse(d).startswith(("triton.jit", "jit"))
                                                   for d in fn.decorator_list):
            yield fn


def test_no_inexact_float_literals_in_triton_arithmetic():
    """No Triton kernel puts a float literal that float32 cannot represent into its arithmetic."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "triton" not in source:
            continue
        rel = path.relative_to(SRC).as_posix()
        for fn in _jit_functions(ast.parse(source)):
            parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, float)):
                    continue
                if _float32_exact(node.value) or _harmless(node, parents):
                    continue
                if (rel, ast.unparse(parents[node])) not in ALLOWED:
                    found.append(f"{rel}:{node.lineno} {fn.name}: {node.value!r}")
    assert not found, "\n".join(found)


def test_the_check_flags_inexact_arithmetic():
    """The check flags an inexact literal in arithmetic and passes thresholds, floors and tiny offsets."""
    fn = next(_jit_functions(ast.parse(
        "@triton.jit\n"
        "def k(x, d):\n"
        "    a = 0.6 * d\n"
        "    b = tl.where(x > 0.9999999, 1.0, x)\n"
        "    c = tl.maximum(x, 1e-12) + 1.0e-9\n"
        "    e = -0.04 * d\n")))
    parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
    flagged = sorted(n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, float)
                     and not _float32_exact(n.value) and not _harmless(n, parents))
    assert flagged == [0.04, 0.6]
