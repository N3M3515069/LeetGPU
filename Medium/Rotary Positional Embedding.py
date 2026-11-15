"""Implement a GPU program that computes the Rotary Positional Embedding (RoPE) for a batch of query vectors. RoPE is a method for encoding positional information in transformer models by rotating the query and key vectors using precomputed cosine and sine components.

Mathematically, given a query vector and corresponding cosine and sine vectors, the operation is defined as:

Where denotes element-wise multiplication. The operation swaps the first and second halves of the vector and negates the first half. For a vector of dimension :
Implementation Requirements

    External libraries are not permitted
    The solve function signature must remain unchanged
    The input tensors Q, cos, and sin have shape (M, D), where M is the number of tokens and D is the head dimension
    cos and sin use half-split layout: values at dimensions j and j + D/2 are identical for each row
    D (head dimension) is guaranteed to be an even number
    The final result must be stored in the output variable with the same shape (M, D)

Example 1:

Input:  Q   = [[1.0, 2.0, 3.0, 4.0],
               [1.0, 1.0, 1.0, 1.0]]
        Cos = [[1.0, 1.0, 1.0, 1.0],
               [0.0, 0.0, 0.0, 0.0]]
        Sin = [[0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0]]
Output: result = [[1.0, 2.0, 3.0, 4.0],
                  [-1.0, -1.0, 1.0, 1.0]]
        (Row 0 is identity via Cos; Row 1 is rotated via Sin)

Constraints

    Q, cos, and sin have identical dimensions
    D % 2 == 0
    1 ≤ M, D ≤ 10,000
    Performance is measured with D = 128, M = 1,048,576
"""

import torch


# Q, cos, sin, output are tensors on the GPU
def solve(
    Q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    output: torch.Tensor,
    M: int,
    D: int,
):

    mid = Q.shape[-1] // 2

    rotate_half = torch.cat((-Q[..., mid:], Q[..., :mid]), dim=-1)

    output.copy_(Q * cos + rotate_half * sin)
