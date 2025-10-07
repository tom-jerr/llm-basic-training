from typing import Any, Dict, List

import torch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]


class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]


class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]


class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class SumOp(Op):
    """
    Op to compute sum along specified dimensions.

    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        if keepdim:
            return [output_grad]
        else:
            reshape_grad = expand_as_3d(output_grad, node.inputs[0])
            return [reshape_grad]


class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""

        return [sum_op(output_grad, dim=0), zeros_like(output_grad)]


class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values

        # Handle scalar tensor case
        if input_tensor.dim() == 0:
            # For scalar, expand to match target shape
            return input_tensor.expand_as(target_tensor)
        else:
            return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""

        return [sum_op(output_grad, dim=(0, 1)), zeros_like(output_grad)]


class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(
        self, node_A: Node, input_shape: List[int], target_shape: List[int]
    ) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.

        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError(
                "Input shape is not set. Make sure compute() is called before gradient()"
            )

        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]

        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(
            zip(input_shape[::-1], output_shape[::-1])
        ):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)

        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)

        if len(output_shape) > len(input_shape):
            grad = sum_op(
                grad,
                dim=list(range(len(output_shape) - len(input_shape))),
                keepdim=False,
            )

        return [grad]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        """TODO: your code here"""
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        a, b = node.inputs[0], node.inputs[1]
        grad_a = output_grad / b
        grad_b = output_grad * (mul_by_const(a, -1) / (b * b))
        return [grad_a, grad_b]


class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad / node.constant]


class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.

        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        """TODO: your code here"""
        dim0, dim1 = node.attrs["dim0"], node.attrs["dim1"]
        return input_values[0].transpose(dim0, dim1)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input."""
        """TODO: your code here"""
        dim0, dim1 = node.attrs["dim0"], node.attrs["dim1"]
        grad = transpose(output_grad, dim0, dim1)
        return [grad]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        """TODO: your code here"""
        return input_values[0] @ input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input."""
        a, b = node.inputs[0], node.inputs[1]
        grad_a = matmul(output_grad, transpose(b, -1, -2))
        grad_b = matmul(transpose(a, -1, -2), output_grad)
        return [grad_a, grad_b]


class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        """TODO: your code here"""
        dim = node.attrs["dim"]
        x = input_values[0]
        x_shifted = x - x.max(dim=dim, keepdim=True).values
        exp_x = torch.exp(x_shifted)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input."""
        # grad = y∗(g−sum(g∗y))
        dim = node.attrs["dim"]
        if isinstance(dim, int):
            dim = (dim,)
        grad = node * (output_grad - sum_op(output_grad * node, dim=dim, keepdim=True))
        return [grad]


class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(
        self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5
    ) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1
        """TODO: your code here"""
        x = input_values[0]
        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]
        dims = tuple(range(-len(normalized_shape), 0))  # 要归一化的维度

        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        return y

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Given gradient of the LayerNorm node wrt its output, return partial
        adjoint (gradient) wrt the input x.
        """
        x = node.inputs[0]
        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]
        dims = tuple(range(-len(normalized_shape), 0))

        # Forward 计算中间值（均值、方差、归一化）
        x_mean = mean(x, dims, keepdim=True)
        var_val = mean((x - x_mean) * (x - x_mean), dims, keepdim=True)
        std = sqrt(var_val + eps)
        x_hat = (x - x_mean) / std

        g = output_grad
        mean_g = mean(g, dims, keepdim=True)
        mean_gx = mean(g * x_hat, dims, keepdim=True)

        grad_x = (g - mean_g - x_hat * mean_gx) / std
        return [grad_x]


class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        return torch.relu(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        # ReLU gradient: 1 if input > 0, else 0
        input_node = node.inputs[0]
        zero_node = add_by_const(
            zeros_like(input_node), 0
        )  # Create a zero tensor with same shape
        mask = greater(input_node, zero_node)  # input > 0
        return [output_grad * mask]


class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.sqrt(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """sqrt gradient: d/dx(sqrt(x)) = 1/(2*sqrt(x))"""
        input_node = node.inputs[0]
        grad = output_grad / (mul_by_const(sqrt(input_node), 2))
        return [grad]


class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        exponent = node.attrs["exponent"]
        return torch.pow(input_values[0], exponent)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """power gradient: d/dx(x^n) = n*x^(n-1)"""
        input_node = node.inputs[0]
        exponent = node.attrs["exponent"]
        if exponent == 0:
            grad = zeros_like(input_node)
        else:
            grad = output_grad * mul_by_const(power(input_node, exponent - 1), exponent)
        return [grad]


class MeanOp(Op):
    """Op to compute mean along specified dimensions."""

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]
        return input_values[0].mean(dim=dim, keepdim=keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """mean gradient: equally distribute gradient to all elements"""
        input_node = node.inputs[0]
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]

        # For mean, the gradient needs to be divided by the number of elements
        # that were averaged over, and broadcast back to the original shape
        if keepdim:
            # If keepdim is True, we can directly use the gradient
            return [output_grad]
        else:
            # If keepdim is False, we need to expand the gradient back
            # Use the same approach as SumOp
            reshape_grad = expand_as_3d(output_grad, input_node)
            return [reshape_grad]


# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()


def topological_sort(nodes):
    """Helper function to perform topological sort on nodes.

    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort

    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    if isinstance(nodes, Node):
        nodes = [nodes]

    visited = set()
    temp_mark = set()
    result = []

    def visit(node):
        if node in temp_mark:
            raise ValueError("Cycle detected in computational graph")
        if node in visited:
            return

        temp_mark.add(node)
        for input_node in node.inputs:
            visit(input_node)
        temp_mark.remove(node)
        visited.add(node)
        result.append(node)

    # Visit all nodes in the list
    for node in nodes:
        visit(node)

    return result


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        # Get all nodes in topological order
        sorted_nodes = topological_sort(self.eval_nodes)

        # Dictionary to store computed values
        node_values = {}

        # Initialize with input values
        for node, value in input_values.items():
            node_values[node] = value

        # Compute values in topological order
        for node in sorted_nodes:
            if node in node_values:
                continue  # Already computed (input node)

            # Get input values for this node
            input_vals = []
            for input_node in node.inputs:
                if input_node not in node_values:
                    raise ValueError(
                        f"Input value for node {input_node.name} not provided"
                    )
                input_vals.append(node_values[input_node])

            # Compute the value using the node's operation
            node_values[node] = node.op.compute(node, input_vals)

        # Return values for eval_nodes in the same order
        result = []
        for node in self.eval_nodes:
            if node not in node_values:
                raise ValueError(f"Could not compute value for node {node.name}")
            result.append(node_values[node])

        return result


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """
    # Dictionary to store gradients for each node
    node_to_grad = {}

    # Initialize output gradient to 1 (ones_like the output)
    node_to_grad[output_node] = ones_like(output_node)

    # Get all nodes in reverse topological order (for backward pass)
    sorted_nodes = topological_sort([output_node])

    # Process nodes in reverse order
    for node in reversed(sorted_nodes):
        if node not in node_to_grad:
            continue

        output_grad = node_to_grad[node]

        # Skip placeholder nodes (they have no inputs)
        if isinstance(node.op, PlaceholderOp):
            continue

        # Get gradients for input nodes
        input_grads = node.op.gradient(node, output_grad)

        # Accumulate gradients for input nodes
        for input_node, grad in zip(node.inputs, input_grads):
            if input_node in node_to_grad:
                # Add to existing gradient
                node_to_grad[input_node] = node_to_grad[input_node] + grad
            else:
                # Initialize gradient
                node_to_grad[input_node] = grad

    # Return gradients for requested nodes
    result = []
    for node in nodes:
        if node in node_to_grad:
            result.append(node_to_grad[node])
        else:
            # If no gradient computed, return zeros
            result.append(zeros_like(node))

    return result
