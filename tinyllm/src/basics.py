import torch
import math


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Compute the softmax of a tensor along a specified axis.
    softmax(x_i) = exp(x_i) / sum(exp(x_i))

    Args:
        x (torch.Tensor): Input tensor.
        axis (int): Axis along which to compute the softmax.

    Returns:
        torch.Tensor: Softmax of the input tensor along the specified axis.
    """
    x_max = torch.max(x, dim=axis, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=axis, keepdim=True)
    return x_exp / x_sum


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply a linear transformation to the incoming data: y = xW^T + b

    Args:
        x (torch.Tensor): Input tensor of shape (N, *, in_features).
        weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
        bias (torch.Tensor, optional): Bias tensor of shape (out_features,). Default is None.

    Returns:
        torch.Tensor: Output tensor of shape (N, *, out_features).
    """
    y = torch.matmul(x, weight.t())
    if bias is not None:
        y += bias
    return y


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function, also known as the swish function.
    silu(x) = x * sigmoid(x)
    平滑且连续可导，非单调性

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the SiLU activation function.
    """
    return x * torch.sigmoid(x)
