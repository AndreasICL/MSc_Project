import tensorflow as tf
import numpy as np
import gpflow
from gpflow.inducing_variables import InducingVariables
from gpflow.base import TensorLike
from gpflow.utilities import to_default_float
from gpflow import covariances as cov
from gpflow import kullback_leiblers as kl
from gpflow.ci_utils import ci_niter

BlockDiag = tf.linalg.LinearOperatorBlockDiag
Diag = tf.linalg.LinearOperatorDiag
LowRank = tf.linalg.LinearOperatorLowRankUpdate

class FourierFeatures1D(InducingVariables):
    def __init__(self, a, b, M):
        # [a, b] defining the interval of the Fourier representation:
        self.a = gpflow.Parameter(a, dtype=gpflow.default_float())
        self.b = gpflow.Parameter(b, dtype=gpflow.default_float())
        # integer array defining the frequencies, ω_m = 2π (b - a)/m:
        self.ms = np.arange(M)
        self.omegas = 2.0 * np.pi * self.ms / (b - a)

    def __len__(self):
        """ number of inducing variables (defines dimensionality of q(u)) """
        return 2 * len(self.ms) - 1  # M cosine and M-1 sine components

@cov.Kuu.register(FourierFeatures1D, gpflow.kernels.Matern12)
def Kuu_matern12_fourierfeatures1d(inducing_variable, kernel, jitter=None):
    a, b, ms = (lambda u: (u.a, u.b, u.ms))(inducing_variable)
    omegas = 2.0 * np.pi * ms / (b - a)

    # Cosine block:
    lamb = 1.0 / kernel.lengthscales
    two_or_four = to_default_float(tf.where(omegas == 0, 2.0, 4.0))
    d_cos = (
        (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kernel.variance / two_or_four
    )  # eq. (111)
    v_cos = tf.ones_like(d_cos) / tf.sqrt(kernel.variance)  # eq. (110)
    cosine_block = LowRank(Diag(d_cos), v_cos[:, None])

    # Sine block:
    omegas = omegas[tf.not_equal(omegas, 0)]  # the sine block does not include omega=0
    d_sin = (
        (b - a) * (tf.square(lamb) + tf.square(omegas)) / lamb / kernel.variance / 4.0
    )  # eq. (113)
    sine_block = Diag(d_sin)

    return BlockDiag([cosine_block, sine_block]).to_dense()


@cov.Kuf.register(FourierFeatures1D, gpflow.kernels.Matern12, TensorLike)
def Kuf_matern12_fourierfeatures1d(inducing_variable, kernel, X):
    X = tf.squeeze(X, axis=1)
    a, b, ms = (lambda u: (u.a, u.b, u.ms))(inducing_variable)

    omegas = 2.0 * np.pi * ms / (b - a)
    Kuf_cos = tf.cos(omegas[:, None] * (X[None, :] - a))
    omegas_sin = omegas[tf.not_equal(omegas, 0)]  # don't compute zero frequency
    Kuf_sin = tf.sin(omegas_sin[:, None] * (X[None, :] - a))

    # correct Kuf outside [a, b] -- see Table 1
    Kuf_sin = tf.where((X < a) | (X > b), tf.zeros_like(Kuf_sin), Kuf_sin)  # just zero

    left_tail = tf.exp(-tf.abs(X - a) / kernel.lengthscales)[None, :]
    right_tail = tf.exp(-tf.abs(X - b) / kernel.lengthscales)[None, :]
    Kuf_cos = tf.where(X < a, left_tail, Kuf_cos)  # replace with left tail
    Kuf_cos = tf.where(X > b, right_tail, Kuf_cos)  # replace with right tail

    return tf.concat([Kuf_cos, Kuf_sin], axis=0)
