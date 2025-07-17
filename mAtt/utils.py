import torch
import numpy as np
from torch import Tensor

def symmetric(A: Tensor) -> Tensor:
   
    size = list(range(len(A.shape)))
    temp = size[-1]
    size.pop()
    size.insert(-1, temp)
    return 0.5 * (A + A.permute(*size))


def is_nan_or_inf(A: Tensor) -> bool:
  
    # Use torch.isnan and torch.isinf to check for invalid values.
    return torch.isnan(A).any().item() or torch.isinf(A).any().item()


def is_pos_def(x: Tensor) -> bool:
   
    eigenvalues = torch.linalg.eigvals(x)
    # In case of complex eigenvalues, compare their real parts.
    return torch.all(torch.real(eigenvalues) > 0).item()


def matrix_operator(A: Tensor, operator: str) -> Tensor:
   
    # Use torch.linalg.svd for numerical stability.
    u, s, _ = torch.linalg.svd(A, full_matrices=False)
    if operator == 'sqrtm':
        s = torch.sqrt(s)
    elif operator == 'rsqrtm':
        s = torch.rsqrt(s)
    elif operator == 'logm':
        s = torch.log(s)
    elif operator == 'expm':
        s = torch.exp(s)
    else:
        raise ValueError(f"operator {operator} is not implemented")
    
    return u @ torch.diag_embed(s) @ u.transpose(-2, -1)


def tangent_space(A: Tensor, ref: Tensor, inverse_transform: bool = False) -> Tensor:
   
    ref_sqrt = matrix_operator(ref, 'sqrtm')
    ref_sqrt_inv = matrix_operator(ref, 'rsqrtm')
    middle = ref_sqrt_inv @ A @ ref_sqrt_inv
    if inverse_transform:
        middle = matrix_operator(middle, 'logm')
    else:
        middle = matrix_operator(middle, 'expm')
    out = ref_sqrt @ middle @ ref_sqrt
    return out


def untangent_space(A: Tensor, ref: Tensor) -> Tensor:
   
    return tangent_space(A, ref, inverse_transform=True)


def parallel_transform(A: Tensor, ref1: Tensor, ref2: Tensor) -> Tensor:
 
    # Debug print; remove or replace with logging in production.
    print(A.size(), ref1.size(), ref2.size())
    out = untangent_space(A, ref1)
    out = tangent_space(out, ref2)
    return out


def orthogonal_projection(A: Tensor, B: Tensor) -> Tensor:
   
    return A - B @ symmetric(B.transpose(0, 1) @ A)


def retraction(A: Tensor, ref: Tensor) -> Tensor:
   
    data = A + ref
    # Use torch.linalg.qr which is more stable than torch.Tensor.qr().
    Q, R = torch.linalg.qr(data)
    # Adjust the sign for stability: ensure diagonal entries of R are positive.
    diag = R.diagonal(offset=0, dim1=-2, dim2=-1)
    sign = torch.diag_embed(torch.sign(diag))
    out = Q @ sign
    return out