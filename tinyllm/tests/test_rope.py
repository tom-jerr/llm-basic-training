import pytest
import torch
import numpy as np
from typing import Tuple
from src.position_encoding import (
    RotaryEmbedding,
)  # 假设你有对应的 PyTorch RotaryEmbedding 实现
from tests.utils import *


# llama的实现
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    seq_len = xq.shape[1]
    freqs_cis = freqs_cis[:seq_len]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def apply_rotary_emb_qwen2(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    half_dim = xq.shape[-1] // 2

    # 分为前半部分和后半部分
    x1 = xq[..., :half_dim]  # 前半部分
    x2 = xq[..., half_dim:]  # 后半部分

    # 截取与输入序列长度匹配的频率张量
    seq_len = xq.shape[1]
    freqs_cis = freqs_cis[:seq_len]

    # 获取 cos 和 sin 分量
    cos_freqs = freqs_cis.real  # [seq_len, half_dim]
    sin_freqs = freqs_cis.imag  # [seq_len, half_dim]

    # reshape for broadcast: [1, seq_len, 1, half_dim]
    cos_freqs = cos_freqs.view(1, seq_len, 1, half_dim)
    sin_freqs = sin_freqs.view(1, seq_len, 1, half_dim)

    # 应用旋转
    # output[0:half_dim] = x1 * cos - x2 * sin
    # output[half_dim:dim] = x1 * sin + x2 * cos
    real = x1 * cos_freqs - x2 * sin_freqs
    imag = x1 * sin_freqs + x2 * cos_freqs

    # 重新组合
    xq_out = torch.cat([real, imag], dim=-1)
    return xq_out.type_as(xq)


def RotaryEmbedding_helper(
    deviceStr: str, traditional: bool, precision: torch.dtype, with_offset: bool
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000

    device = torch.device("cuda") if deviceStr == "cuda" else torch.device("cpu")
    for _ in range(100):
        user_layer = RotaryEmbedding(
            HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional
        ).to(device)
        x = torch.rand(
            BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=precision
        )

        if with_offset:
            input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
            input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
        else:
            input_pos_user = None
            input_pos = 0

        # 根据传统/非传统模式选择不同的参考实现
        freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN, BASE).to(device)
        if input_pos > 0:
            freqs_cis = freqs_cis[input_pos : input_pos + SEQ_LEN]
        else:
            freqs_cis = freqs_cis[:SEQ_LEN]
        if traditional:
            reference_output = apply_rotary_emb(x, freqs_cis)
        else:
            reference_output = apply_rotary_emb_qwen2(x, freqs_cis)
        user_output = user_layer(x, input_pos_user).to(device)

        atol = 5e-6 if precision == torch.float32 else 1e-3
        assert_allclose(user_output, reference_output, precision, atol=atol)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_torch_traditional(device, with_offset, precision):
    RotaryEmbedding_helper(device, True, precision, with_offset)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_rope_torch_non_traditional(device, with_offset, precision):
    RotaryEmbedding_helper(device, False, precision, with_offset)
