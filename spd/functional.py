import torch as th


def trace(A):
    """"
    compute the batch trace of A [...,n,n]
    """
    # trace_vec = th.diagonal(A, dim1=-2, dim2=-1).sum(dim=-1)
    r_trace = th.einsum("...ii->...", A)
    return r_trace

def inner_product(A, B):
    """"
    compute the batch inner product of A and B, with [...,n,n] [...,n,n]
    """
    r_inner_prod = th.einsum("...ij,...ij->...", A, B)
    return r_inner_prod


def tril_half_diag(A):
    """"[...n,n] A, strictly lower part + 1/2 * half of diagonal part"""
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    half_diag_A = str_tril_A + 0.5 * th.diag_embed(diag_A_vec)
    return half_diag_A
