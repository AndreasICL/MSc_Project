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
    def __init__(self, a, b, M, D):
        self.length = tf.math.pow(M, D)
        self.dim = D
        # [a, b] defining the interval of the Fourier representation:
        self.a = gpflow.Parameter(a, dtype=gpflow.default_float())
        self.b = gpflow.Parameter(b, dtype=gpflow.default_float())

        self.phis = gpflow.Parameter(np.random.uniform(0, 2 * np.pi, size=M))
        self.omegas = gpflow.Parameter(np.random.uniform(0, 0.5 * M, size=M))
        # self.omegas = gpflow.Parameter( p.rvs( size=M ) + default_jitter() )

    def __len__(self):
        """ number of inducing variables (defines dimensionality of q(u)) """
        return self.length  # M^D sine components

@cov.Kuf.register(RVFF_ND, gpflow.kernels.Product, TensorLike)
def Kuf_matern12_RVFF_ND(inducing_variable, kernel: gpflow.kernels.Product, X):
    # X = tf.squeeze(X, axis=1)
    a, omegas, phis = (lambda u: (u.a, u.omegas, u.phis))(inducing_variable)

    results = []

    print(X[:, 0].shape)

    for i in range(inducing_variable.dim):
        results.append( tf.sin( omegas[:, None] * ( X[:, i][None, :] - a + phis[:, None] ) ) )

    print(results[0].shape)

    # phis = tf.tile(tf.reshape(phis, (-1,1)), [1, len(X)])
    # omegas = tf.tile(tf.reshape(omegas, (-1,1)), [1, len(X)])
    # zeroOmegas = tf.sin( phis )

    # res = tf.where( omegas != 0, nonZeroOmegas, zeroOmegas )
    finalResult = results[0]

    for i in range( len(results) - 1 ):
        finalResult = np.kron(finalResult, results[i + 1])

    print(finalResult.shape)

    return finalResult
