import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x * rms
        return x_normed * self.weight
