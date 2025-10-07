from typing import Any, Dict, List
import torch
from auto_diff import *


class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, node_A: Node, node_B: Node, normalized_shape: List[int], eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        A, B = input_values
        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]

        # First perform matrix multiplication
        matmul_result = torch.matmul(A, B)

        # Then perform layer normalization
        dims = tuple(range(-len(normalized_shape), 0))  # 要归一化的维度
        mean = matmul_result.mean(dim=dims, keepdim=True)
        var = matmul_result.var(dim=dims, unbiased=False, keepdim=True)
        normalized_result = (matmul_result - mean) / torch.sqrt(var + eps)

        return normalized_result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs
        matmul_result = matmul(A, B)
        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]

        # 创建一个layernorm节点来利用其梯度计算
        layernorm_node = layernorm(matmul_result, normalized_shape, eps)
        layernorm_grad = layernorm_node.op.gradient(layernorm_node, output_grad)[0]

        # 然后计算matmul对A和B的梯度
        matmul_node = matmul(A, B)
        matmul_grads = matmul_node.op.gradient(matmul_node, layernorm_grad)

        return matmul_grads


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(self, node_A: Node, node_B: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"dim": dim},
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        A, B = input_values
        dim = node.attrs["dim"]
        matmul_result = A @ B
        x_shifted = matmul_result - matmul_result.max(dim=dim, keepdim=True).values
        exp_x = torch.exp(x_shifted)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        A, B = node.inputs
        dim = node.attrs["dim"]
        matmul_result = matmul(A, B)

        # 计算softmax对其输入的梯度
        # 创建一个softmax节点来利用其梯度计算
        softmax_node = softmax(matmul_result, dim=dim)
        softmax_grad = softmax_node.op.gradient(softmax_node, output_grad)[0]

        # 然后计算matmul对A和B的梯度
        matmul_node = matmul(A, B)
        matmul_grads = matmul_node.op.gradient(matmul_node, softmax_grad)

        return matmul_grads


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
