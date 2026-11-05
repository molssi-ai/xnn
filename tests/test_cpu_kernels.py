"""Triton kernels on the CPU (xnn.common.cpu_kernels) against the same operations in PyTorch: masked loads and
stores with Triton's literal typing, scattered and block-summed atomic adds, a uniform add once per program,
per-program sums of a row loop, runtime branches and loops, runtime arguments passed by name, and the refusal
of a block reduction it does not cover."""
import math
import sys

import pytest
import torch
import triton
import triton.language as tl

from xnn.common import cpu_kernels

pytestmark = pytest.mark.skipif(not cpu_kernels.available(), reason="no C++ compiler or Triton")


@triton.jit
def _affine(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < n
    x = tl.load(x_ptr + e, mask=m, other=0.0)
    tl.store(y_ptr + e, x * 0.1 + 1.0, mask=m)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_loads_stores_and_literals(dtype):
    x = torch.randn(1000, dtype=dtype)
    y = torch.zeros_like(x)
    cpu_kernels.launch(_affine, (triton.cdiv(1000, 128),), x, y, 1000, BLOCK=128)
    # a literal takes the value's precision (0.1 rounded to float32 against a float32 value); the product and the
    # sum may contract to one fused multiply-add, as Triton's do
    want = x * torch.tensor(0.1, dtype=dtype) + 1.0
    assert float((y - want).abs().max()) <= 2 * torch.finfo(dtype).eps
    if dtype == torch.float32:
        assert float((y.double() - (x.double() * 0.1 + 1.0)).abs().max()) > 1e-9


@triton.jit
def _scatter(v_ptr, idx_ptr, out_ptr, tot_ptr, cnt_ptr, n, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < n
    v = tl.load(v_ptr + e, mask=m, other=0.0)
    i = tl.load(idx_ptr + e, mask=m, other=0)
    tl.atomic_add(out_ptr + i, v, mask=m, sem="relaxed")
    tl.atomic_add(tot_ptr, tl.sum(tl.where(m, v, 0.0), 0), sem="relaxed")
    tl.atomic_add(cnt_ptr, 1.0, sem="relaxed")


def test_atomic_adds():
    torch.manual_seed(0)
    n, N = 5000, 37
    v = torch.randn(n, dtype=torch.float64)
    idx = torch.randint(0, N, (n,))
    out = torch.full((N,), 2.0, dtype=torch.float64)
    tot = torch.zeros(1, dtype=torch.float64)
    cnt = torch.zeros(1, dtype=torch.float64)
    grid = (triton.cdiv(n, 64),)
    cpu_kernels.launch(_scatter, grid, v, idx, out, tot, cnt, n, BLOCK=64)
    want = torch.full((N,), 2.0, dtype=torch.float64).index_add_(0, idx, v)
    assert torch.allclose(out, want, rtol=1e-13, atol=1e-13)
    assert torch.allclose(tot, v.sum().reshape(1), rtol=1e-13)
    # an add of a uniform value at a uniform address runs once per program
    assert float(cnt) == grid[0]
    # the buffers are left zeroed: a second launch adds the same again
    cpu_kernels.launch(_scatter, grid, v, idx, out, tot, cnt, n, BLOCK=64)
    assert torch.allclose(out - 2.0, 2.0 * (want - 2.0), rtol=1e-13, atol=1e-13)


@triton.jit
def _rows(ptr_ptr, val_ptr, y_ptr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    lo = tl.load(ptr_ptr + r)
    hi = tl.load(ptr_ptr + r + 1)
    acc = tl.zeros([BLOCK], dtype=tl.float64)
    start = lo
    while start < hi:
        e = start + tl.arange(0, BLOCK)
        acc += tl.load(val_ptr + e, mask=e < hi, other=0.0)
        start += BLOCK
    tl.store(y_ptr + r, tl.sum(acc, 0))


def test_a_row_loop_with_a_sum_per_program():
    counts = torch.tensor([0, 3, 17, 1, 40, 8])
    ptr = torch.zeros(len(counts) + 1, dtype=torch.int64)
    ptr[1:] = torch.cumsum(counts, 0)
    val = torch.randn(int(ptr[-1]), dtype=torch.float64)
    y = torch.full((len(counts),), -1.0, dtype=torch.float64)
    cpu_kernels.launch(_rows, (len(counts),), ptr, val, y, BLOCK=8)
    want = torch.stack([val[int(a):int(b)].sum() for a, b in zip(ptr[:-1], ptr[1:])])
    assert torch.allclose(y, want, rtol=1e-13, atol=1e-14)


@triton.jit
def _branches(x_ptr, y_ptr, n, k, scale_ptr, NTERMS: tl.constexpr, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = e < n
    x = tl.load(x_ptr + e, mask=m, other=0.0)
    s = tl.load(scale_ptr)
    acc = x * 0.0
    for j in tl.static_range(NTERMS):
        acc += x * (j + 1)
    t = x * 0.0
    for j in range(k):
        t += s * (j + 1)
    if k > 2:
        t = t * 2.0
    tl.store(y_ptr + e, acc + t, mask=m)


def test_runtime_branches_loops_and_arguments_by_name():
    x = torch.randn(300, dtype=torch.float64)
    y = torch.zeros_like(x)
    scale = torch.tensor([0.25], dtype=torch.float64)
    cpu_kernels.launch(_branches, (3,), x, y, 300, 4, scale_ptr=scale, NTERMS=3, BLOCK=128)
    assert torch.allclose(y, 6.0 * x + 2.0 * 0.25 * (1 + 2 + 3 + 4), rtol=1e-14)


@triton.jit
def _carried_copy(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    w = n * 0
    for j in range(0, n):
        at = w
        w += 1
        tl.store(y_ptr + at, tl.load(x_ptr + j))


def test_a_name_bound_to_a_carried_value_keeps_it():
    """``at = w`` then ``w += 1``: ``at`` keeps the value ``w`` had."""
    x = torch.arange(10, dtype=torch.float64)
    y = torch.full((11,), -1.0, dtype=torch.float64)
    cpu_kernels.launch(_carried_copy, (1,), x, y, 10, BLOCK=1)
    assert torch.equal(y[:10], x)


@triton.jit
def _block_max(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e, mask=e < n, other=0.0)
    tl.store(y_ptr + tl.program_id(0), tl.max(x, 0))


def test_an_uncovered_block_reduction_is_refused():
    x = torch.randn(10, dtype=torch.float64)
    with pytest.raises(cpu_kernels.Unsupported):
        cpu_kernels.launch(_block_max, (1,), x, torch.zeros(1, dtype=torch.float64), 10, BLOCK=16)


@triton.jit
def _scaled(v, s: tl.constexpr = 3.0):
    return v * s


@triton.jit
def _put(p, v, m):
    tl.store(p, v, mask=m)
    return


@triton.jit
def _ops(x_ptr, i_ptr, f_ptr, g_ptr, s_ptr, tot_ptr, n, k, flag, BLOCK: tl.constexpr, NEG: tl.constexpr):
    tl.static_assert(BLOCK > 0)
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    e = tl.max_contiguous(e, BLOCK)
    m = e < n
    x = tl.load(x_ptr + e, mask=m, other=1.0)
    i = tl.load(i_ptr + e, mask=m, other=0)
    tl.store(f_ptr + 0 * n + e, x // 2.0, mask=m)
    tl.store(f_ptr + 1 * n + e, x % 2.0, mask=m)
    tl.store(f_ptr + 2 * n + e, x ** 2.0, mask=m)
    tl.store(f_ptr + 3 * n + e, tl.rsqrt(x), mask=m)
    tl.store(f_ptr + 4 * n + e, tl.sigmoid(x), mask=m)
    tl.store(f_ptr + 5 * n + e, +x, mask=m)
    tl.store(f_ptr + 6 * n + e, x if NEG else -x, mask=m)
    tl.store(f_ptr + 7 * n + e, x if flag else 2.0 * x, mask=m)
    tl.store(f_ptr + 8 * n + e, tl.where(m, 1.0, 2), mask=m)
    tl.store(f_ptr + 9 * n + e, tl.where(x > 2.5, float("inf"), x), mask=m)
    tl.store(f_ptr + 10 * n + e, tl.where(x < 0.5, float("nan"), x), mask=m)
    tl.store(f_ptr + 11 * n + e, tl.full([BLOCK], 1.5, dtype=(x + x).dtype), mask=m)
    tl.store(f_ptr + 12 * n + e, i / 2, mask=m)
    tl.store(f_ptr + 13 * n + e, (x > 1.0) + (x > 2.0), mask=m)
    tl.store(f_ptr + 14 * n + e, x * 0.0 + tl.cdiv(n, 4), mask=m)
    y: tl.float64 = x * 3.0
    tl.store(f_ptr + 15 * n + e, y, mask=m)
    tl.store(f_ptr + 16 * n + e, x * min(BLOCK, 4), mask=m)
    if BLOCK in (16, 32):
        c = x + 1.0
    else:
        c = x - 1.0
    tl.store(f_ptr + 17 * n + e, c, mask=m)
    acc = 0.0
    for _j in range(k):
        acc += x
    tl.store(f_ptr + 18 * n + e, acc, mask=m)
    w = x * 0.0
    while w < x:
        w += 1.0
    tl.store(f_ptr + 19 * n + e, w, mask=m)
    tl.store(f_ptr + 20 * n + e, tl.where((x > 1.0) and (x < 2.0), 1.0, 0.0), mask=m)
    tl.store(f_ptr + 21 * n + e, tl.where((x < 1.0) or (x > 2.0), 1.0, 0.0), mask=m)
    tl.store(f_ptr + 22 * n + e, tl.load(x_ptr + tl.minimum(e, n - 1), mask=None), mask=m)
    tl.store(f_ptr + 23 * n + e, _scaled(x), mask=m)
    _put(f_ptr + 24 * n + e, x * 4.0, m)
    tl.store(g_ptr + 0 * n + e, ~i, mask=m)
    tl.store(g_ptr + 1 * n + e, i & 3, mask=m)
    tl.store(g_ptr + 2 * n + e, i | 4, mask=m)
    tl.store(g_ptr + 3 * n + e, i ^ 1, mask=m)
    tl.store(g_ptr + 4 * n + e, i << 2, mask=m)
    tl.store(g_ptr + 5 * n + e, i >> 1, mask=m)
    tl.store(g_ptr + 6 * n + e, tl.abs(i - 5), mask=m)
    tl.store(s_ptr + pid, tl.sum(tl.where(m, x, 0.0), 0))
    tl.atomic_add(tot_ptr, tl.sum(-tl.where(m, x, 0.0), 0), sem="relaxed")


@pytest.mark.parametrize("flag", [True, False])
def test_operators_constants_and_control_flow(flag):
    """Each operator, math function, literal and statement form the translation covers, against PyTorch."""
    gen = torch.Generator().manual_seed(1)
    n, block = 70, 16
    x = 0.1 + 2.9 * torch.rand(n, dtype=torch.float64, generator=gen)
    x[3] = 2.0
    i = torch.randint(0, 20, (n,), generator=gen)
    f = torch.full((25, n), -7.0, dtype=torch.float64)
    g = torch.full((7, n), -7, dtype=torch.int64)
    programs = triton.cdiv(n, block)
    s = torch.zeros(programs, dtype=torch.float64)
    tot = torch.zeros(1, dtype=torch.float64)
    cpu_kernels.launch(_ops, (programs,), x, i, f, g, s, tot, n, 3, flag, BLOCK=block, NEG=False)
    want = [torch.floor(x / 2.0), torch.fmod(x, 2.0), x * x, 1.0 / torch.sqrt(x), torch.sigmoid(x), x, -x,
            x if flag else 2.0 * x, torch.ones_like(x), torch.where(x > 2.5, torch.inf, x),
            torch.where(x < 0.5, torch.nan, x), torch.full_like(x, 1.5), i.double() / 2,
            (x > 1.0).double() + (x > 2.0).double(), torch.full_like(x, triton.cdiv(n, 4)), 3.0 * x, 4.0 * x,
            x + 1.0, 3.0 * x, torch.ceil(x), ((x > 1.0) & (x < 2.0)).double(), ((x < 1.0) | (x > 2.0)).double(),
            x, 3.0 * x, 4.0 * x]
    for row, (got, ref) in enumerate(zip(f, want)):
        assert torch.allclose(got, ref, rtol=1e-14, atol=0, equal_nan=True), row
    assert float(f[19, 3]) == 2.0
    want_int = [torch.bitwise_not(i), i & 3, i | 4, i ^ 1, i << 2, i >> 1, (i - 5).abs()]
    for row, (got, ref) in enumerate(zip(g, want_int)):
        assert torch.equal(got, ref), row
    pad = torch.zeros(programs * block - n, dtype=torch.float64)
    assert torch.allclose(s, torch.cat([x, pad]).reshape(programs, block).sum(1), rtol=1e-14)
    assert torch.allclose(tot, -x.sum().reshape(1), rtol=1e-14)


@triton.jit
def _first_programs(x_ptr, y_ptr, limit, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ok = pid < limit
    e = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr + e) * 2.0, mask=ok)


def test_a_mask_uniform_across_a_program_guards_its_lanes():
    """A store under a mask that holds for whole programs writes the programs where it holds only."""
    x = torch.randn(64, dtype=torch.float64)
    y = torch.zeros_like(x)
    cpu_kernels.launch(_first_programs, (4,), x, y, 2, BLOCK=16)
    assert torch.equal(y[:32], 2.0 * x[:32]) and not bool(y[32:].any())


@triton.jit
def _two(a, b):
    return a + b


@triton.jit
def _pair(a):
    return a, a


@triton.jit
def _three(a):
    return a, a, a


@triton.jit
def _after_return(a):
    return a
    a = a + 1.0


@triton.jit
def _branch_return(a):
    if a > 0.0:
        return a
    return -a


@triton.jit
def _renamed(a):
    return a


@triton.jit
def _bad_f16(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.full([BLOCK], 1.0, dtype=tl.float16))


@triton.jit
def _bad_sum_value(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.sum(tl.load(x_ptr + e), 0) * 2.0)


@triton.jit
def _bad_pointer_value(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, x_ptr)


@triton.jit
def _bad_name(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr + e) + not_defined_anywhere)  # noqa: F821


@triton.jit
def _bad_subscript(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr + e)[0])


@triton.jit
def _bad_sum_address(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.sum(tl.load(x_ptr + e), 0))


@triton.jit
def _bad_pointer_product(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr * 2 + e))


@triton.jit
def _bad_matmul(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    tl.store(y_ptr + e, x @ x)


@triton.jit
def _bad_bitcast(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr + e).to(tl.int64, bitcast=True))


@triton.jit
def _bad_method(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr + e).abs())


@triton.jit
def _bad_python_call(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, math.sqrt(tl.load(x_ptr + e)))


@triton.jit
def _bad_scalar_load(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(n))


@triton.jit
def _bad_axis(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + e, tl.load(x_ptr + e))


@triton.jit
def _bad_helper_argument(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, _two(tl.load(x_ptr + e)))


@triton.jit
def _bad_bind_none(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    r = tl.store(y_ptr + e, tl.load(x_ptr + e))
    tl.store(y_ptr + e, r)


@triton.jit
def _bad_bind_tuple(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    t = _pair(tl.load(x_ptr + e))
    tl.store(y_ptr + e, t)


@triton.jit
def _bad_unpack_value(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    a, b = x
    tl.store(y_ptr + e, a + b)


@triton.jit
def _bad_unpack_length(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    a, b = _three(tl.load(x_ptr + e))
    tl.store(y_ptr + e, a + b)


@triton.jit
def _bad_item_assignment(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    x[0] = 1.0
    tl.store(y_ptr + e, x)


@triton.jit
def _bad_after_return(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, _after_return(tl.load(x_ptr + e)))


@triton.jit
def _bad_expression_statement(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    x  # noqa: B018
    tl.store(y_ptr + e, x)


@triton.jit
def _bad_chained_assignment(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    a = b = tl.load(x_ptr + e)
    tl.store(y_ptr + e, a + b)


@triton.jit
def _bad_statement(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    tl.store(y_ptr + e, x)
    del x


@triton.jit
def _bad_kind_change(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    q = n * 0
    for _j in range(n):
        q = x_ptr
    tl.store(y_ptr + e, tl.load(q + e))


@triton.jit
def _bad_branch_return(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, _branch_return(tl.load(x_ptr + e)))


@triton.jit
def _bad_for_tuple(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    for j in (1, 2):
        x = x * j
    tl.store(y_ptr + e, x)


@triton.jit
def _bad_for_iterator(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    for _j in enumerate(n):
        x = x * 2.0
    tl.store(y_ptr + e, x)


@triton.jit
def _bad_while_else(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    while x < 1.0:
        x = x + 1.0
    else:
        x = x * 2.0
    tl.store(y_ptr + e, x)


@triton.jit
def _bad_return_before_sum(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    e = pid * BLOCK + tl.arange(0, BLOCK)
    if pid > 0:
        return
    tl.store(y_ptr + pid, tl.sum(tl.load(x_ptr + e), 0))


@triton.jit
def _bad_override(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    tl.store(y_ptr + e, _renamed(tl.load(x_ptr + e)))


@triton.jit
def _bad_add_and_store(x_ptr, y_ptr, BLOCK: tl.constexpr):
    e = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + e)
    tl.store(y_ptr + e, x)
    tl.atomic_add(y_ptr + e, x, sem="relaxed")


@triton.jit
def _no_block(x_ptr, y_ptr, SIZE: tl.constexpr):
    e = tl.arange(0, SIZE)
    tl.store(y_ptr + e, tl.load(x_ptr + e))


_REFUSED = [_bad_f16, _bad_sum_value, _bad_pointer_value, _bad_name, _bad_subscript, _bad_sum_address,
            _bad_pointer_product, _bad_matmul, _bad_bitcast, _bad_method, _bad_python_call, _bad_axis,
            _bad_helper_argument, _bad_bind_none, _bad_bind_tuple, _bad_unpack_value, _bad_unpack_length,
            _bad_item_assignment, _bad_after_return, _bad_expression_statement, _bad_chained_assignment,
            _bad_statement, _bad_branch_return, _bad_for_tuple, _bad_while_else, _bad_return_before_sum,
            _bad_add_and_store]


@pytest.mark.parametrize("kernel", _REFUSED, ids=lambda k: k.fn.__name__)
def test_constructs_outside_the_lane_rules_are_refused(kernel):
    """A construct the translation does not cover raises Unsupported before anything is compiled."""
    x = torch.randn(16, dtype=torch.float64)
    with pytest.raises(cpu_kernels.Unsupported):
        cpu_kernels.launch(kernel, (1,), x, torch.zeros(16, dtype=torch.float64), BLOCK=16)


@pytest.mark.parametrize("kernel", [_bad_scalar_load, _bad_kind_change, _bad_for_iterator],
                         ids=lambda k: k.fn.__name__)
def test_constructs_with_a_size_argument_are_refused(kernel):
    """The refused constructs that read a runtime size."""
    x = torch.randn(16, dtype=torch.float64)
    with pytest.raises(cpu_kernels.Unsupported):
        cpu_kernels.launch(kernel, (1,), x, torch.zeros(16, dtype=torch.float64), 4, BLOCK=16)


def test_arguments_and_launches_outside_the_rules_are_refused(monkeypatch):
    """Unsupported argument types, missing arguments or constexprs, a second grid axis and a helper override
    without the helper's name raise before a launch."""
    x = torch.randn(16, dtype=torch.float64)
    y = torch.zeros(16, dtype=torch.float64)
    with pytest.raises(cpu_kernels.Unsupported, match="tensor dtype"):
        cpu_kernels.launch(_affine, (1,), x.half(), y, 16, BLOCK=16)
    with pytest.raises(cpu_kernels.Unsupported, match="argument n"):
        cpu_kernels.launch(_affine, (1,), x, y, "16", BLOCK=16)
    with pytest.raises(cpu_kernels.Unsupported, match="not a triton.jit"):
        cpu_kernels.translate(lambda: None, (), {})
    with pytest.raises(cpu_kernels.Unsupported, match="missing constexpr"):
        cpu_kernels.launch(_affine, (1,), x, y, 16)
    with pytest.raises(cpu_kernels.Unsupported, match="no BLOCK"):
        cpu_kernels.launch(_no_block, (1,), x, y, SIZE=16)
    with pytest.raises(cpu_kernels.Unsupported, match="grid"):
        cpu_kernels.prepare(_affine, (1, 2), x, y, 16, BLOCK=16)
    with pytest.raises(TypeError, match="missing argument n"):
        cpu_kernels.prepare(_affine, (1,), x, y, BLOCK=16)
    monkeypatch.setattr(cpu_kernels, "_OVERRIDES", {})
    cpu_kernels.override(_renamed, "def another_name(a):\n    return a\n")
    with pytest.raises(cpu_kernels.Unsupported, match="no source"):
        cpu_kernels.launch(_bad_override, (1,), x, y, BLOCK=16)


def test_launches_check_their_tensors():
    """Contiguity, the shared memory of two atomic-add targets, a rebound argument, the balance prefix and the
    precision of deferred adds are checked; a rebound argument is read by the next call."""
    x = torch.randn(64, dtype=torch.float64)
    y = torch.zeros_like(x)
    with pytest.raises(ValueError, match="contiguous"):
        cpu_kernels.prepare(_affine, (1,), torch.randn(128, dtype=torch.float64)[::2], y, 64, BLOCK=64)
    with pytest.raises(ValueError, match="balance"):
        cpu_kernels.prepare(_affine, (1,), x, y, 64, BLOCK=64, balance=torch.zeros(5, dtype=torch.int64))
    run = cpu_kernels.prepare(_affine, (1,), x, y, 64, BLOCK=64)
    with pytest.raises(ValueError, match="dtype and size"):
        run.bind(0, torch.zeros(32, dtype=torch.float64))
    x2 = torch.randn(64, dtype=torch.float64)
    run.bind(0, x2)
    run()
    assert torch.allclose(y, x2 * 0.1 + 1.0, rtol=1e-15, atol=1e-15)
    v = torch.randn(8, dtype=torch.float64)
    idx = torch.zeros(8, dtype=torch.int64)
    shared = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="share"):
        cpu_kernels.prepare(_scatter, (1,), v, idx, shared, shared[:1], torch.zeros(1, dtype=torch.float64), 8,
                            BLOCK=64)
    v32 = v.float()
    run = cpu_kernels.prepare(_scatter, (1,), v32, idx, torch.zeros(4), torch.zeros(1), torch.zeros(1), 8, BLOCK=64)
    with pytest.raises(ValueError, match="float64"):
        cpu_kernels.Deferred([run])()


def test_the_build_environment_fallbacks(monkeypatch, tmp_path):
    """Without /proc the build links its own OpenMP and the CPU tag still names the machine; no compiler or no
    Triton turns the kernels off; the cache directory follows XNN_CPU_KERNELS_DIR; a source that does not
    compile raises; a compiled source is loaded once."""
    class NoProc:
        def __init__(self, *_):
            pass

        def read_text(self):
            raise OSError("no /proc")

    class Maps(NoProc):
        def read_text(self):
            return "7f00-7f01 r-xp 0 0 0 /usr/lib/libc.so.6\n"

    monkeypatch.setattr(cpu_kernels, "Path", NoProc)
    assert cpu_kernels._openmp_library.__wrapped__() is None
    tag = cpu_kernels._cpu_tag.__wrapped__()
    assert len(tag) == 8 and int(tag, 16) >= 0
    monkeypatch.setattr(cpu_kernels, "Path", Maps)
    assert cpu_kernels._openmp_library.__wrapped__() is None
    monkeypatch.undo()
    monkeypatch.setenv("XNN_CPU_KERNELS_DIR", str(tmp_path))
    assert cpu_kernels._cache_dir() == tmp_path
    with pytest.raises(RuntimeError, match="compiling a CPU kernel failed"):
        cpu_kernels._compile("this is not C++", "run")
    lib = cpu_kernels._compile(cpu_kernels._DRIVER, "run_sequence")
    assert cpu_kernels._compile(cpu_kernels._DRIVER, "run_sequence") is lib
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", "")
    assert cpu_kernels._compiler() is None
    assert not cpu_kernels.available.__wrapped__()
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "triton", None)
    assert not cpu_kernels.available.__wrapped__()
