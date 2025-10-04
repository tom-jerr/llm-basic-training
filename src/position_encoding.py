import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        super().__init__()
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len
        half_dims = dims // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, half_dims, dtype=torch.float32) / half_dims)
        )
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        self.cos_freqs = torch.cos(angles)  # (seq_len, half_dims)
        self.sin_freqs = torch.sin(angles)
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional

    def forward(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        N, S, H, D = x.shape
        if offset is not None:
            if isinstance(offset, slice):
                assert (
                    offset.stop - offset.start == S
                ), "offset length must match sequence length"

            elif isinstance(offset, list):
                assert (
                    len(offset) == N
                ), f"offsets must have the same length as batch size {N}"
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"
                offset = torch.tensor([list(range(i.start, i.stop)) for i in offset])  # type: ignore
        cos_basis = (
            self.cos_freqs[:S] if offset is None else self.cos_freqs[offset, :]
        ).to(x.device)
        sin_basis = (
            self.sin_freqs[:S] if offset is None else self.sin_freqs[offset, :]
        ).to(x.device)

        # reshape x: (b, s, n_heads, head_dim // 2, 2)
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            # 非传统模式：头嵌入维度分为两半 (Qwen2 style)
            x1 = x[..., 0 : self.half_dims]  # 前半部分
            x2 = x[..., self.half_dims : self.dims]  # 后半部分
        # reshape basis: (1, s, 1, dims // 2)
        cos_basis = cos_basis.reshape(-1, S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, S, 1, self.half_dims)
        # (b, s, n_heads, head_dim // 2)
        real = x1 * cos_basis - x2 * sin_basis
        imag = x1 * sin_basis + x2 * cos_basis
        if self.traditional:
            y = torch.stack([real, imag], dim=-1)
            y = y.reshape(N, S, H, D)
        else:
            # 非传统模式：按照 Qwen2 的方式重新组合
            # output[0:half_dims] = real, output[half_dims:dims] = imag
            y = torch.cat((real, imag), dim=-1)
            y = y.reshape(N, S, H, D)
        return y.type_as(x)
