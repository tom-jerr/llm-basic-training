import pytest
import torch
from .utils import *
from src.attention import scaled_dot_product_attention_grouped, causal_mask
import src.qwen2 as qwen2
import math


# Efficient implementation from pytorch
# Warning: PyTorch 官方实现的 causal mask 与我们这里的实现不同
def ref_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=S - L
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def grouped_attention_helper(
    device: torch.device,
    dtype: torch.dtype,
    batch_dimension: int,
    scale: float | None,
    is_causal_mask: bool,
):
    H_q = 18
    H = 6
    L = 3
    D = 5
    S = 7
    BATCH = 10
    BATCH_2 = 2

    if batch_dimension == 0:
        q_shape = (H_q, L, D)
        kv_shape = (H, S, D)
        mask_shape = (H_q, L, S)
    elif batch_dimension == 1:
        q_shape = (BATCH, H_q, L, D)
        kv_shape = (BATCH, H, S, D)
        mask_shape = (BATCH, H_q, L, S)
    elif batch_dimension == 2:
        q_shape = (BATCH_2, BATCH, H_q, L, D)
        kv_shape = (BATCH_2, BATCH, H, S, D)
        mask_shape = (BATCH_2, BATCH, H_q, L, S)

    for _ in range(100):
        query = torch.rand(q_shape, dtype=dtype, device=device)
        key = torch.rand(kv_shape, dtype=dtype, device=device)
        value = torch.rand(kv_shape, dtype=dtype, device=device)
        mask = torch.rand(mask_shape, dtype=dtype, device=device)

        # PyTorch's scaled_dot_product_attention expects shape: (batch, num_heads, seq_len, head_dim)
        # Need to reshape and transpose appropriately
        query_reshaped = query.reshape(-1, H_q, L, D)
        key_reshaped = key.reshape(-1, H, S, D)
        value_reshaped = value.reshape(-1, H, S, D)
        mask_reshaped = mask.reshape(-1, H_q, L, S) if not is_causal_mask else None

        scale_factor = scale if scale is not None else (1.0 / (D**0.5))

        # Convert mask to attention mask format (additive mask with -inf for masked positions)
        if not is_causal_mask:
            attn_mask = mask_reshaped
        else:
            attn_mask = None

        reference_output = ref_scaled_dot_product_attention(
            query_reshaped,
            key_reshaped,
            value_reshaped,
            enable_gqa=True,
            attn_mask=attn_mask,
            scale=scale_factor,
            is_causal=is_causal_mask,
        )

        # Reshape reference output back to original shape
        reference_output = reference_output.reshape(query.shape)

        user_output = scaled_dot_product_attention_grouped(
            query,
            key,
            value,
            scale=scale,
            mask=mask if not is_causal_mask else "causal",
        )

        assert_allclose(user_output, reference_output, precision=dtype)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    device: str, dtype: torch.dtype, batch_dimension: int, scale: float | None
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    grouped_attention_helper(dev, dtype, batch_dimension, scale, False)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_task_2_mask_only_same_dim(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    L = 3
    S = 3
    user_output = causal_mask(L, S, torch.float32, device=dev)

    expected = torch.tensor(
        [
            [0, -float("inf"), -float("inf")],
            [0, 0, -float("inf")],
            [0, 0, 0],
        ],
        dtype=torch.float32,
        device=dev,
    )

    assert_allclose(user_output, expected, precision=torch.float32)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_task_2_mask_only_different_dim(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    L = 3
    S = 5
    user_output = causal_mask(L, S, torch.float32, device=dev)

    expected = torch.tensor(
        [
            [0, 0, 0, -float("inf"), -float("inf")],
            [0, 0, 0, 0, -float("inf")],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
        device=dev,
    )

    assert_allclose(user_output, expected, precision=torch.float32)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    device: str, dtype: torch.dtype, batch_dimension: int, scale: float | None
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    grouped_attention_helper(dev, dtype, batch_dimension, scale, True)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_3_qwen2_grouped_query_attention(
    device: str, dtype: torch.dtype, mask: str | None
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)

    batch_size = 1
    seq_len = 4
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    max_seq_len = 64
    theta = 10000

    # Note: You'll need to adapt the Qwen2 model import for PyTorch
    # This is a placeholder - adjust based on your actual implementation
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2RotaryEmbedding,
    )

    config = Qwen2Config(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        rope_theta=theta,
        max_position_embeddings=max_seq_len,
    )

    config._attn_implementation = "sdpa"  # 这里可以使用causal mask
    rotary_emb = Qwen2RotaryEmbedding(config)
    torch_attention = Qwen2Attention(config, layer_idx=0).to(device=dev, dtype=dtype)

    torch.manual_seed(42)
    x = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype, device=dev) * 2 - 1
    position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)
    position_embeddings = rotary_emb(x, position_ids)

    torch_output = torch_attention(
        x,
        position_embeddings,
        attention_mask=None,
        is_causal=True if mask == "causal" else False,
    )[0].to(device=dev, dtype=dtype)

    # Extract weights and biases
    wq = torch_attention.q_proj.weight
    wk = torch_attention.k_proj.weight
    wv = torch_attention.v_proj.weight
    wo = torch_attention.o_proj.weight
    bq = (
        torch_attention.q_proj.bias if hasattr(torch_attention.q_proj, "bias") else None
    )
    bk = (
        torch_attention.k_proj.bias if hasattr(torch_attention.k_proj, "bias") else None
    )
    bv = (
        torch_attention.v_proj.bias if hasattr(torch_attention.v_proj, "bias") else None
    )

    user_attention = qwen2.Qwen2MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        bq=bq,
        bk=bk,
        bv=bv,
        max_seq_len=max_seq_len,
        theta=theta,
    )

    user_output = user_attention(x, mask=mask).to(device=dev, dtype=dtype)

    assert_allclose(user_output, torch_output, precision=dtype)
