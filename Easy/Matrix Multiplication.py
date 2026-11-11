'''Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix 
 of dimensions 
 and matrix 
 of dimensions 
, compute the product matrix 
, which will have dimensions 
. All matrices are stored in row-major format.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in matrix C
Example 1:
Input:
Matrix 
 (
):
 
Matrix 
 (
):
 
Output:
Matrix 
 (
):
 

Example 2:
Input:
Matrix 
 (
):
 
Matrix 
 (
):
 
Output:
Matrix 
 (
):
 

Constraints
1 ≤ M, N, K ≤ 8192
Performance is measured with M = 8192, N = 6144, K = 4096'''

import torch

# A, B, C are tensors on the GPU
def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, M: int, N: int, K: int):

    torch.matmul(A, B, out= C)
