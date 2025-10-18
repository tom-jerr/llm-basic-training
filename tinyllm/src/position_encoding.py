import torch
import torch.nn as nn
from functools import lru_cache


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        max_seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        super().__init__()
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.half_dims = dims // 2
        self.max_seq_len = max_seq_len
        self.traditional = traditional
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs: torch.Tensor = torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)  # [seq_len, head_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        B, S, H, D = x.shape
        if offset is None:
            # from 0 to s-1th: [S, D] -> [1, S, D]
            cos_sin = self.cos_sin_cache[:S].unsqueeze(0).to(x.device)
        elif isinstance(offset, slice):
            # just one slice: [S, D] -> [1, S, D]
            assert (
                offset.stop - offset.start == S
            ), "Offset slice length must match sequence length"
            cos_sin = self.cos_sin_cache[offset].unsqueeze(0).to(x.device)

        elif isinstance(offset, list):
            # slice list, collase all cos and sin to one tensor: [S,D]->[B,S,D]
            assert (
                len(offset) == B
            ), "Number of slices in offset list must match batch size"
            cos_sin = torch.stack([self.cos_sin_cache[s] for s in offset], dim=0)

        else:
            raise TypeError(f"Unsupported type for offset: {type(offset)}")

        # [B, S, D/2] or [1, S, D/2]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.traditional:
            x = x.reshape(B, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            # Qwen2 style
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]
        # [B, S, D/2] -> [B, S, 1, D/2]
        cos = cos.reshape(-1, S, 1, self.half_dims)
        sin = sin.reshape(-1, S, 1, self.half_dims)
        # [B, S, H, D/2]
        real = x1 * cos - x2 * sin
        imag = x1 * sin + x2 * cos
        if self.traditional:
            y = torch.stack([real, imag], dim=-1)
            y = y.reshape(B, S, H, D)
        else:
            y = torch.cat((real, imag), dim=-1)
            y = y.reshape(B, S, H, D)
        return y.type_as(x)
