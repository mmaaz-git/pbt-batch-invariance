# uncomment if running a notebook
#%%writefile test_batch_invariance.py

import numpy as np
from hypothesis import given, settings, assume, note, strategies as st
from hypothesis.extra import numpy as hnp
import pytest
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")


# MAMTMUL


@st.composite
def matmul_strategy(
    draw,
    *,
    max_dim=2048,
    min_dim=1,
    min_value=-1e3,
    max_value=1e3,
):
    # Couple dimensions so (B, D) @ (D, N) is valid
    B = draw(st.integers(min_value=min_dim, max_value=max_dim))
    D = draw(st.integers(min_value=min_dim, max_value=max_dim))
    N = draw(st.integers(min_value=min_dim, max_value=max_dim))

    elems = st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=64,
        min_value=min_value,
        max_value=max_value,
    )

    a = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(B, D), elements=elems))).to(DEVICE)
    b = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(D, N), elements=elems))).to(DEVICE)

    # 0 <= m < n <= B
    m = draw(st.integers(min_value=0, max_value=B-1))
    n = draw(st.integers(min_value=m+1, max_value=B))

    return a, b, m, n

def matmul_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Built-in matmul.
    """
    return a @ b

def matmul_rowwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute each row then stack them.
    """
    result = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype)
    for i in range(a.shape[0]):
        result[i] = a[i] @ b
    return result

@pytest.mark.parametrize("matmul_fn", [matmul_batched, matmul_rowwise])
@given(inputs=matmul_strategy())
@settings(deadline=None, max_examples=1000)
def test_matmul(inputs, matmul_fn):
    a, b, m, n = inputs

    # Property: slicing before vs after batch matmul yields same first row
    out1 = matmul_fn(a[m:n], b)    # (n-m,D) @ (D,N) -> (n-m,N)
    out2 = matmul_fn(a, b)[m:n]    # (B,D) @ (D,N) -> (B,N) then slice -> (n-m,N)

    diff = torch.abs(out1 - out2).max()

    # guard against nan, inf, etc.
    assume(torch.isfinite(diff))

    diff = diff.item()
    torch.set_printoptions(precision=17)
    note(f"diff: {diff}\nout1: {out1}\nout2: {out2}")

    # assert diff
    assert diff == 0


# RMSNORM


@st.composite
def rmsnorm_strategy(
    draw,
    *,
    max_dim=2048,
    min_dim=1,
    min_value=-1e3,
    max_value=1e3,
):
    B = draw(st.integers(min_value=min_dim, max_value=max_dim))
    D = draw(st.integers(min_value=min_dim, max_value=max_dim))

    elems = st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=64,
        min_value=min_value,
        max_value=max_value,
    )

    x = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(B, D), elements=elems))).to(DEVICE)
    gamma = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(D,), elements=elems))).to(DEVICE)

    # Generate slice indices: 0 <= m < n <= B
    m = draw(st.integers(min_value=0, max_value = B-1))
    n = draw(st.integers(min_value=m + 1, max_value=B))

    return x, gamma, m, n

def rmsnorm_batched(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Built-in rmsnorm.
    """
    return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) * gamma

def rmsnorm_rowwise(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Compute each row then stack them.
    """
    result = torch.empty_like(x)
    for i in range(x.shape[0]):
        result[i] = x[i] * torch.rsqrt(torch.mean(x[i] ** 2)) * gamma
    return result

@pytest.mark.parametrize("rmsnorm_fn", [rmsnorm_batched, rmsnorm_rowwise])
@given(inputs=rmsnorm_strategy())
@settings(deadline=None, max_examples=1000)
def test_rmsnorm(inputs, rmsnorm_fn):
    x, gamma, m, n = inputs

    # Property: slicing before vs after batch matmul yields same first row
    out1 = rmsnorm_fn(x[m:n], gamma)    # (n-m,D) @ (D,N) -> (n-m,N)
    out2 = rmsnorm_fn(x, gamma)[m:n]    # (B,D) @ (D,N) -> (B,N) then slice -> (n-m,N)

    diff = torch.abs(out1 - out2).max()

    # guard against nan, inf, etc.
    assume(torch.isfinite(diff))

    diff = diff.item()
    torch.set_printoptions(precision=17)
    note(f"diff: {diff}\nout1: {out1}\nout2: {out2}")

    # assert diff
    assert diff == 0


# ATTENTION


@st.composite
def attn_strategy(
    draw,
    *,
    max_batch=128,
    min_batch=1,
    max_seq_len=2048,
    min_seq_len=1,
    min_num_heads=1,
    max_num_heads=8,
    min_head_dim=1,
    max_head_dim=64,
    min_value=-1e3,
    max_value=1e3,
):
    B = draw(st.integers(min_value=min_batch, max_value=max_batch))
    seq_len = draw(st.integers(min_value=min_seq_len, max_value=max_seq_len))
    num_heads = draw(st.integers(min_value=min_num_heads, max_value=max_num_heads))
    head_dim = draw(st.integers(min_value=min_head_dim, max_value=max_head_dim))

    elems = st.floats(
        allow_nan=False,
        allow_infinity=False,
        width=32,
        min_value=min_value,
        max_value=max_value,
    )

    # Generate Q, K, V tensors: [B, num_heads, seq_len, head_dim]
    Q = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(B, num_heads, seq_len, head_dim), elements=elems))).to(DEVICE)
    K = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(B, num_heads, seq_len, head_dim), elements=elems))).to(DEVICE)
    V = torch.from_numpy(draw(hnp.arrays(dtype=np.float64, shape=(B, num_heads, seq_len, head_dim), elements=elems))).to(DEVICE)

    # Generate slice indices: 0 <= m < n <= B
    m = draw(st.integers(min_value=0, max_value=B-1))
    n = draw(st.integers(min_value=m+1, max_value=B))

    return Q, K, V, m, n


def attn_batched(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Standard batched attention.
    """
    return F.scaled_dot_product_attention(Q, K, V)


def attn_rowwise(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Process rowwise attention.
    """
    result = torch.empty_like(Q)

    for i in range(Q.shape[0]):
        result[i] = F.scaled_dot_product_attention(
            Q[i:i+1],
            K[i:i+1],
            V[i:i+1]
        ).squeeze(0)

    return result

@pytest.mark.parametrize("attn_fn", [attn_batched, attn_rowwise])
@given(inputs=attn_strategy())
@settings(deadline=None, max_examples=1000)
def test_attn(inputs, attn_fn):
    Q, K, V, m, n = inputs

    out1 = attn_fn(Q[m:n], K[m:n], V[m:n])
    out2 = attn_fn(Q, K, V)[m:n]

    diff = torch.abs(out1 - out2).max()

    # guard against nan, inf, etc.
    assume(torch.isfinite(diff))

    diff = diff.item()
    torch.set_printoptions(precision=17)
    note(f"diff: {diff}\nout1: {out1}\nout2: {out2}")

    # assert diff
    assert diff == 0