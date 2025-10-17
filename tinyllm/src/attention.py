import torch
import torch.nn as nn
from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the scaled dot-product attention.

    q dot k表示query和key的相似度，除以sqrt(dk)是为了防止点积过大导致softmax梯度过小
    每个query对所有key的相似度转换为一个概率分布，再用这个概率分布对value加权求和，得到最终的注意力输出

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, depth).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, depth).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, depth_v).
        scale (float, optional): Scaling factor. If None, defaults to sqrt(depth). Default is None.
        mask (torch.Tensor, optional): Mask tensor broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k). Default is None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len_q, depth_v).
    """
    q_k = torch.matmul(
        query, key.transpose(-2, -1)
    )  # (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = q_k / (
        scale
        if scale is not None
        else torch.sqrt(torch.tensor(query.shape[-1], dtype=query.dtype))
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    # 每个 query 对所有 key 的相似度 转换为一个概率分布
    attn_weights = softmax(
        scores, axis=-1
    )  # (batch_size, num_heads, seq_len_q, seq_len_k)
    output = torch.matmul(
        attn_weights, value
    )  # (batch_size, num_heads, seq_len_q, depth_v)
    return output


class SimpleMultiHeadAttention(nn.Module):
    """A simple implementation of Multi-Head Attention mechanism.

    E is hidden_size or embed_dim or dims or model_dim
    H is num_heads
    D is head_dim
    L is seq_len, in PyTorch API it's S (source len)

    w_q/w_k/w_v: (H x D) x E
    output/input: N x L x E
    w_o: E x (H x D)
    拼接后的 (N, L, H*D) 还要再映射回原始维度 (N, L, E)，这样才能跟残差连接 (residual connection) 对齐。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**0.5
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)  # (N, L, H, D)
            .transpose(1, 2)  # (N, H, L, D), 每个头单独计算注意力
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        x = scaled_dot_product_attention_simple(
            projection_q, projection_k, projection_v, self.scale, mask
        )  # (N, H, L, D)
        x = x.transpose(1, 2).reshape(
            N, L, self.num_heads * self.head_dim
        )  # (N, L, H*D)
        return linear(x, self.wo)  # (N, L, E)


def causal_mask(L: int, S: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """
    PyTorch: 如果 S > L(即在推理阶段KV cache 累积的 key 更多),
    那么 query 只能看到前 L 个 key, 这是不正确的
    [[1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0]]

    Ours:
    [[1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1]]
    """
    offset = S - L
    mask = torch.tril(torch.ones((L, S), device=device), diagonal=offset)
    mask = torch.where(mask.bool(), 0.0, -torch.inf).to(dtype=dtype, device=device)
    return mask


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """GQA: num_kv_heads << num_query_heads

    Args:
        query (torch.Tensor): [B, H_q, S_q, D]
        key (torch.Tensor): [B, H_kv, S_k, D]
        value (torch.Tensor): [B, H_kv, S_k, D]
        scale (float, optional): Scaling factor. If None, defaults to sqrt(depth). Default is None.
        mask (torch.Tensor, optional): [B, H_q, S_q, D]

    Returns:
        torch.Tensor: [B, H_q, S, D]
    """
    expect_shape = query.shape
    H_q, S_q, D = query.shape[-3:]
    H_kv, S_k, _ = key.shape[-3:]
    scale_factor = scale if scale is not None else 1.0 / (D**0.5)
    B = query.shape[:-3]
    assert H_q % H_kv == 0
    n_repeats = H_q // H_kv
    # -1 保证 query 的 batch 维度是任意多维
    query = query.reshape(*B, -1, H_kv, n_repeats, S_q, D)  # [B, H_kv, repeats, S_q, D]
    insert_dim = len(B)
    key = key.reshape(*B, -1, H_kv, 1, S_k, D)
    value = value.reshape(*B, -1, H_kv, 1, S_k, D)
    q_k = torch.matmul(
        query, key.transpose(-2, -1)
    )  # (batch_size, H_kv, repeats, seq_len_q, seq_len_k)

    scores = q_k * scale_factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(S_q, S_k, scores.dtype, scores.device)
            scores = scores + mask
        else:
            mask = mask.broadcast_to((*B, H_q, S_q, S_k))
            mask = mask.reshape(*B, 1, H_kv, n_repeats, S_q, S_k)
            scores = scores + mask
    output = torch.matmul(
        softmax(scores, axis=-1), value
    )  # (batch_size, h_kv, n_repeat, seq_len_q, depth_v)
    return output.reshape(expect_shape)  # [B, H_q, S, D]
