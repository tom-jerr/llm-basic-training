import torch
import numpy as np


def swapping_tiles(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    """
    Swaps the tiles of a 2D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (H, W).
        tile_size (int): Size of the tiles to be swapped.

    Returns:
        torch.Tensor: Tensor with swapped tiles.
    """
    H, W = x.shape
    assert (
        H % tile_size == 0 and W % tile_size == 0
    ), "Height and Width must be divisible by tile_size"

    # Reshape to (H//tile_size, tile_size, W//tile_size, tile_size)
    x_reshaped = x.reshape(H // tile_size, tile_size, W // tile_size, tile_size)
    print(x_reshaped)

    # Swap the tiles
    # permute之后的reshape会拷贝内存，在一个不连续的张量上执行 reshape 操作，就会强制触发一次数据拷贝
    x_swapped = x_reshaped.permute(2, 1, 0, 3).reshape(H, W)

    return x_swapped


if __name__ == "__main__":
    x = torch.arange(16).reshape(4, 4)
    print("Original Tensor:")
    print(x)

    tile_size = 2
    x_swapped = swapping_tiles(x, tile_size)
    print(f"\nTensor after swapping tiles of size {tile_size}:")
    print(x_swapped)
