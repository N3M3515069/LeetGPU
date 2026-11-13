"""Implement a program that performs R rounds of parallel hashing on an array of 32-bit integers using the provided hash function. The hash should be applied R times iteratively (the output of one round becomes the input to the next).
Implementation Requirements

    External libraries are not permitted
    The solve function signature must remain unchanged
    The final result must be stored in array output

Example 1:

Input:  numbers = [123, 456, 789], R = 2
Output: hashes = [1636807824, 1273011621, 2193987222]

Example 2:

Input:  numbers = [0, 1, 2147483647], R = 3
Output: hashes = [96754810, 3571711400, 2006156166]

Constraints

    1 ≤ N ≤ 10,000,000
    1 ≤ R ≤ 100
    0 ≤ input[i] ≤ 2147483647
    Performance is measured with N = 5,000,000
"""

import torch


def fnv1a_hash(x: torch.Tensor) -> torch.Tensor:
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    x_int = x.to(torch.int64)
    hash_val = torch.full_like(x_int, OFFSET_BASIS, dtype=torch.int64)

    for byte_pos in range(4):
        byte = (x_int >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME
        hash_val = hash_val & 0xFFFFFFFF

    return hash_val.to(torch.int32)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):

    for _ in range(R):
        input = fnv1a_hash(input)
    output.copy_(input)
