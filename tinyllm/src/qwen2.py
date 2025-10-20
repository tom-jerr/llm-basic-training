import torch
import torch.nn as nn
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .position_encoding import RotaryEmbedding
from typing import Any, Optional
from .embedding import EmbeddingLayer

# from .quantize import dequantize_linear


class Qwen2MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert (
            hidden_size % num_heads == 0
        ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        assert (
            num_heads % num_kv_heads == 0
        ), f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / (self.head_dim**0.5)
        # Trainable parameters
        self.register_buffer("wq", wq)
        self.register_buffer("wk", wk)
        self.register_buffer("wv", wv)
        self.register_buffer("wo", wo)
        self.register_buffer("bq", bq)
        self.register_buffer("bk", bk)
        self.register_buffer("bv", bv)

        self.rope = RotaryEmbedding(
            self.head_dim, max_seq_len, theta, traditional=False
        )  # Qwen2 uses nontraditional RoPE

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor | str] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, E)
            mask: (B, 1, L, S) or "causal" or None
        """
        B, L, _ = x.shape

        projection_q = linear(x, self.wq, bias=self.bq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        projection_k = linear(x, self.wk, bias=self.bk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_v = linear(x, self.wv, bias=self.bv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )

        projection_q = self.rope(projection_q, offset=slice(0, L))
        projection_k = self.rope(projection_k, offset=slice(0, L))

        projection_q = projection_q.transpose(1, 2)  # (B, num_heads, L, head_dim)
        projection_k = projection_k.transpose(1, 2)  # (B, num_kv_heads, L, head_dim)
        projection_v = projection_v.transpose(1, 2)  # (B, num_kv_heads, L, head_dim)

        x = scaled_dot_product_attention_grouped(
            projection_q.float(),
            projection_k.float(),
            projection_v.float(),
            scale=self.scale,
            mask=mask,
        ).to(x.dtype)

        x = x.transpose(1, 2).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.register_buffer("w_gate", w_gate)
        self.register_buffer("w_up", w_up)
        self.register_buffer("w_down", w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP(x)=(SiLU(W_gate(x))⊙W_up(x))W_down"""
        return linear(silu(linear(x, self.w_gate)) * linear(x, self.w_up), self.w_down)


class Qwen2TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        # super().__init__()
        # self.num_attention_heads = num_attention_heads
        # self.hidden_size = hidden_size

        # self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        # self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm(
        #     hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        # )
        # self.self_attn = Qwen2MultiHeadAttention(
        #     num_heads=num_attention_heads,
        #     hidden_size=hidden_size,
        #     num_kv_heads=num_kv_heads,
        #     wq=wq,
        #     wk=wk,
        #     wv=wv,
        #     wo=wo,
        #     bq=bq,
        #     bk=bk,
        #     bv=bv,
        #     max_seq_len=max_seq_len,
        #     theta=theta,
        # )
        pass

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor | str] = None,
    ) -> torch.Tensor:
        # r = self.self_attn(self.input_layernorm(x), mask)
        # h = x + r
        # r = self.mlp(self.post_attention_layernorm(h))
        # out = h + r
        # return out
        pass


class Qwen2ModelWeek1(nn.Module):
    def __init__(
        self,
        torch_model: Any,
    ):
        # super().__init__()
        # self.num_hidden_layers = torch_model.config.num_hidden_layers
        # self.hidden_size = torch_model.config.hidden_size
        # self.vocab_size = torch_model.config.vocab_size
        # precision = torch.float16
        # self.precision = precision

        # self.embedding = Embedding(
        #     vocab_size=self.vocab_size,
        #     embedding_dim=self.hidden_size,
        #     weight=dequantize_linear(torch_model.model.embed_tokens).to(precision),
        # )

        # self.layers_inner = nn.ModuleList()

        # for i in range(torch_model.config.num_hidden_layers):
        #     wq = dequantize_linear(torch_model.model.layers[i].self_attn.q_proj)
        #     wk = dequantize_linear(torch_model.model.layers[i].self_attn.k_proj)
        #     wv = dequantize_linear(torch_model.model.layers[i].self_attn.v_proj)
        #     wo = dequantize_linear(torch_model.model.layers[i].self_attn.o_proj)
        #     w_gate = dequantize_linear(torch_model.model.layers[i].mlp.gate_proj)
        #     w_up = dequantize_linear(torch_model.model.layers[i].mlp.up_proj)
        #     w_down = dequantize_linear(torch_model.model.layers[i].mlp.down_proj)

        #     layer = Qwen2TransformerBlock(
        #         num_attention_heads=torch_model.config.num_attention_heads,
        #         num_kv_heads=torch_model.config.num_key_value_heads,
        #         hidden_size=torch_model.config.hidden_size,
        #         intermediate_size=torch_model.config.intermediate_size,
        #         rms_norm_eps=torch_model.config.rms_norm_eps,
        #         wq=wq.to(precision),
        #         wk=wk.to(precision),
        #         wv=wv.to(precision),
        #         wo=wo.to(precision),
        #         bq=torch_model.model.layers[i].self_attn.q_proj.bias.to(precision),
        #         bk=torch_model.model.layers[i].self_attn.k_proj.bias.to(precision),
        #         bv=torch_model.model.layers[i].self_attn.v_proj.bias.to(precision),
        #         w_gate=w_gate.to(precision),
        #         w_up=w_up.to(precision),
        #         w_down=w_down.to(precision),
        #         w_input_layernorm=torch_model.model.layers[
        #             i
        #         ].input_layernorm.weight.to(precision),
        #         w_post_attention_layernorm=torch_model.model.layers[
        #             i
        #         ].post_attention_layernorm.weight.to(precision),
        #         max_seq_len=torch_model.config.max_position_embeddings,
        #         theta=torch_model.config.rope_theta,
        #     )
        #     self.layers_inner.append(layer)

        # self.norm = RMSNorm(
        #     torch_model.config.hidden_size,
        #     weight=torch_model.model.norm.weight.to(precision),
        #     eps=torch_model.config.rms_norm_eps,
        # )

        # if not torch_model.config.tie_word_embeddings:
        #     self.register_buffer('w_lm_head', dequantize_linear(torch_model.lm_head))
        # else:
        #     self.w_lm_head = None

        # self.torch_model = torch_model
        pass

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # h = self.embedding(inputs)
        # for layer in self.layers_inner:
        #     h = layer(h, mask="causal")
        # h = self.norm(h)
        # if self.w_lm_head is not None:
        #     return linear(h, self.w_lm_head)
        # else:
        #     return self.embedding.as_linear(h)
        pass

    def to(self, device):
        """Override to method to move all components to device"""
        super().to(device)
        if self.w_lm_head is not None:
            self.w_lm_head = self.w_lm_head.to(device)
        return self
