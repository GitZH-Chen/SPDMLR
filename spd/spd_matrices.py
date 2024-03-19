import math
import torch as th
import torch.nn as nn

from spdnets.functionals import sym_powm,sym_logm
from spd.functional import inner_product,trace,tril_half_diag

class SPDMatrices(nn.Module):
    """Computation for SPD data with [...,n,n]"""
    def __init__(self, n,power=1.):
        super().__init__()
        self.n=n; self.dim = int(n * (n + 1) / 2)
        self.register_buffer('power', th.tensor(power))
        self.register_buffer('I', th.eye(n))

        if power == 0:
            raise Exception('power should not be zero with power={:.4f}'.format(power))
        self.sgn_power = -1 if self.power < 0 else 1

    def spd_pow(self, S):
        if self.power == 1.:
            Power_S = S;
        else:
            Power_S = sym_powm.apply(S, self.power)
        return Power_S

    def RMLR(self, S, P, A):
        """
        RMLR based on margin distance, generating A by parallel transportation
        Inputs:
        S: [b,c,n,n] SPD
        P: [class,n,n] SPD matrices
        A: [class,n,n] symmetric matrices
        """
        raise NotImplementedError

class SPDOnInvariantMetric(SPDMatrices):
    """
    Computation for SPD data with [b,c,n,n], the base class of (\theta,\alpha,\beta)-EM/LEM/AIM/
    """
    def __init__(self, n, alpha=1.0, beta=0.,power=1.):
        super(__class__, self).__init__(n,power)
        if alpha <= 0 or beta <= -alpha / n:
            raise Exception('wrong alpha or beta with alpha={:.4f},beta={:.4f}'.format(alpha, beta))
        self.alpha = alpha;self.beta = beta;
        self.p = (self.alpha + self.n * self.beta) ** 0.5
        self.q = self.alpha ** 0.5

    def alpha_beta_Euc_inner_product(self, tangent_vector1, tangent_vector2):
        """"computing the O(n)-invariant Euclidean inner product"""
        if self.alpha==1. and self.beta==0.:
            X_new = inner_product(tangent_vector1, tangent_vector2)
        else:
            item1 = inner_product(tangent_vector1, tangent_vector2)
            trace_vec1 = trace(tangent_vector1)
            trace_vec2 = trace(tangent_vector2)
            item2 = th.mul(trace_vec1, trace_vec2)
            X_new = self.alpha * item1 + self.beta * item2
        return X_new

class SPDLogEuclideanMetric(SPDOnInvariantMetric):
    """ (\alpha,\beta)-LEM """
    def __init__(self,n,alpha=1.0, beta=0.):
        super(__class__, self).__init__(n,alpha, beta)

    def RMLR(self, S, P, A):
        P_phi = sym_logm.apply(P)
        S_phi = sym_logm.apply(S)
        X_new = self.alpha_beta_Euc_inner_product(S_phi - P_phi, A)

        return X_new

class SPDLogCholeskyMetric(SPDMatrices):
    """ \theta-LCM """
    def __init__(self, n,power=1.):
        super(__class__, self).__init__(n,power)

    def RMLR(self, S, P, A):
        Power_S = self.spd_pow(S)
        Power_P = self.spd_pow(P)

        Chol_of_Power_S = th.linalg.cholesky(Power_S)
        Chol_of_Power_P = th.linalg.cholesky(Power_P)

        item1_diag_vec = th.log(th.diagonal(Chol_of_Power_S, dim1=-2, dim2=-1)) - th.log(th.diagonal(Chol_of_Power_P, dim1=-2, dim2=-1))
        item1 = Chol_of_Power_S.tril(-1) - Chol_of_Power_P.tril(-1) + th.diag_embed(item1_diag_vec)
        X_new = (1 / self.power) * inner_product(item1, tril_half_diag(A))

        return X_new