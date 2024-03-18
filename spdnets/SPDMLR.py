"""
    Official Implementation of the SPD MLR presented in
        @inproceedings{chen2024spdmlr,
        title={Riemannian Multinomial Logistics Regression for SPD Neural Networks},
        author={Chen, Ziheng and Song, Yue and Liu, Gaowen and Kompella, Ramana Rao and Wu, Xiaojun and Sebe, Nicu},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2024}
        }
"""

import math
import geoopt
import torch as th
import torch.nn as nn

import spd.spd_matrices as spd_matrices

class SPDRMLR(nn.Module):
    def __init__(self,n,c,metric='SPDLogEuclideanMetric',power=1.0,alpha=1.0,beta=0.):
        """
            Input X: (N,h,n,n) SPD matrices
            Output P: (N,dim_vec) vectors
            SPD parameter of size (c,n,n), where c denotes the number of classes
            Sym parameters (c,n,n)
        """
        super(__class__, self).__init__()
        self.n = n;self.c = c;
        self.metric = metric;self.power = power;self.alpha = alpha;self.beta = beta;

        self.P = geoopt.ManifoldParameter(th.empty(c, n, n), manifold=geoopt.manifolds.SymmetricPositiveDefinite())
        init_3Didentity(self.P)
        self.A = nn.Parameter(th.zeros_like(self.P))
        init_matrix_uniform(self.A, int(n ** 2))

        if self.metric == 'SPDLogEuclideanMetric':
            self.spd = spd_matrices.SPDLogEuclideanMetric(n=self.n,alpha=self.alpha,beta=self.beta)
        elif self.metric=='SPDLogCholeskyMetric':
            self.power = power;
            self.spd = spd_matrices.SPDLogCholeskyMetric(n=self.n,power=self.power)
        else:
            raise Exception('unknown metric {}'.format(metric))

    def forward(self,X):
        A_sym = symmetrize_by_tril(self.A)
        X_new = self.spd.RMLR(X, self.P, A_sym)
        return X_new

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n},c={self.c},metric={self.metric},power={self.power},alpha={self.alpha},beta={self.beta})"

def symmetrize_by_tril(A):
    """"
    symmetrize A by the lower part of A, with [...,n,n]
    """
    str_tril_A = A.tril(-1)
    diag_A_vec = th.diagonal(A, dim1=-2, dim2=-1)
    tmp_A_sym = str_tril_A + str_tril_A.transpose(-1, -2) + th.diag_embed(diag_A_vec)
    return tmp_A_sym

def init_matrix_uniform(A,fan_in,factor=6):
    bound = math.sqrt(factor / fan_in) if fan_in > 0 else 0
    nn.init.uniform_(A, -bound, bound)

def init_3Didentity(S):
    """ initializes to identity a (h,ni,no) 3D-SPDParameter"""
    h,n1,n2=S.shape
    for i in range(h):
        S.data[i] = th.eye(n1, n2)