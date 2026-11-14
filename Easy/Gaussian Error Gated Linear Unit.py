"""Implement the Gaussian Error Gated Linear Unit (GEGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula. The input and output tensor must be of type float32.

GEGLU is defined as:

    Split input into two halves: and
    Compute GELU on the second half:
    Compute the GEGLU output:

Implementation Requirements

    Use only native features (external libraries are not permitted)
    The solve function signature must remain unchanged
    The final result must be stored in the output tensor

Example 1:

Input:  [1.0, 1.0]  (N=2)
Output: [0.8413447]

Example 2:

Input:  [2.0, -1.0, 1.0, 0.5]  (N=4)
Output: [1.6826895, -0.3457312]

Constraints

    1 ≤ N ≤ 1,000,000
    N is an even number
    -100.0 ≤ input values ≤ 100.0
    Performance is measured with N = 1,000,000
"""

import torch


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):

    x1, x2 = torch.chunk(input, 2)

    output.copy_(x1 * (1 / 2) * x2 * (1 + torch.erf(x2 / 2 ** (1 / 2))))
