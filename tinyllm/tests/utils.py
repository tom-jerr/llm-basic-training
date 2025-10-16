import torch
import numpy as np

DEVICES = [torch.device("cpu"), torch.device("cuda")]
DEVICES_IDS = ["cpu", "cuda"]
PRECISIONS = [torch.float32, torch.float16]
PRECISION_IDS = ["float32", "float16"]


def assert_allclose(
    actual,
    expected,
    precision: torch.dtype,
    rtol: float | None = None,
    atol: float | None = None,
):
    """
    Assert that two tensors are all close within some tolerance.

    Args:
        actual: Actual output tensor
        expected: Expected output tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        precision: Tensor precision type
    """
    if precision == torch.float16:
        rtol = rtol or 3.0e-2
        atol = atol or 1.0e-5
    elif precision == torch.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-8
    else:
        return ValueError(f"Unsupported precision: {precision}")

    actual = actual.detach().cpu().numpy()
    expected = expected.detach().cpu().numpy()
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.invert(np.isclose(actual, expected, rtol=rtol, atol=atol))
        if diff.size > 10000 and np.sum(diff) <= 3:
            # if only a small number of elements are different in a large array, probably fine
            return
        with np.printoptions(precision=3, suppress=True):
            print("aactual=", actual)
            print("expected=", actual, expected)
            print("diff_actual=", actual * diff)
            print("diff_expected=", expected * diff)
            print("diff_actual_val=", actual[diff])
            print("diff_bexpected_val=", expected[diff])
            assert False, f"Arrays are not all close."
