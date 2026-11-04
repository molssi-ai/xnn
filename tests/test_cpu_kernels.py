"""Triton kernels on the CPU (xnn.common.cpu_kernels) against the same operations in PyTorch: masked loads and
stores with Triton's literal typing, scattered and block-summed atomic adds, a uniform add once per program,
per-program sums of a row loop, runtime branches and loops, runtime arguments passed by name, and the refusal
of a block reduction it does not cover."""
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
