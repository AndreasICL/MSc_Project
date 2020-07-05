from scipy.stats import rv_continuous
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import covariances as cov
from gpflow.base import TensorLike
from gpflow.config import default_float, default_jitter

b = 6
a = 0
intervalLen = b - a
resolution = 1e-3

x = np.arange(0, intervalLen, resolution)

class RVFF_ND(gpflow.inducing_variables.InducingVariables):
    def __init__(self, a, b, M, D, jitter=None):
        self.length = M * D
        self.dim = D
        self.M = M
        # [a, b] defining the interval of the Fourier representation:
        self.a = gpflow.Parameter(a, dtype=gpflow.default_float())
        self.b = gpflow.Parameter(b, dtype=gpflow.default_float())

        self.phis = gpflow.Parameter(np.random.uniform(0, 2 * np.pi, size=(D, M) ))
        self.omegas = gpflow.Parameter(np.random.uniform(0, 0.5 * M, size=(D, M) ))
        # self.omegas = gpflow.Parameter( p.rvs( size=M ) + default_jitter() )

    def __len__(self):
        """ number of inducing variables (defines dimensionality of q(u)) """
        return self.M * self.dim  # M * D sine components

@cov.Kuf.register(RVFF_ND, gpflow.kernels.Product, TensorLike)
def Kuf_matern12_RVFF_ND(inducing_variable, kernel: gpflow.kernels.Product, X):
    # X = tf.squeeze(X, axis=1)
    a, omegas, phis = (lambda u: (u.a, u.omegas, u.phis))(inducing_variable)

    result = tf.ones( shape=(inducing_variable.M, X.shape[0]), dtype=default_float() )
    for d in range(inducing_variable.dim):
        result = result * tf.sin( omegas[d][:, None] * ( tf.transpose(X)[d][None, :] - a + phis[d][:, None]) )

    return result

@cov.Kuu.register(RVFF_ND, gpflow.kernels.Product)
def Kuu_matern12_RVFF_ND(inducing_variable, kernel, jitter=None):
    a, b, omegas, phis = (lambda u: (u.a, u.b, u.omegas, u.phis))(inducing_variable)

    lambda_ = 1.0 / kernel.lengthscales

    def innerProduct( intervalLen, omegas, phis, kernelVar, lambda_ ):
        pass

    return innerProduct( b - a, omegas, phis, kernel.variance, lambda_ )
