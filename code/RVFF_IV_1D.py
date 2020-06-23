from scipy.stats import rv_continuous
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import covariances as cov
from gpflow.base import TensorLike

b = 6
a = 0
intervalLen = b - a
resolution = 1e-3

x = np.arange(0, intervalLen, resolution)

var = 0.51
l = 2.87
mu = var * np.exp(-x/l)

mu = mu / tf.reduce_sum(mu) # (intervalLen * tf.reduce_sum(mu)/len(x) ) / 10.0

class matern12PowerSpectrum(rv_continuous):
  def __init__(self, mu, a, b, resolution):
    rv_continuous.__init__(self, a=a, b=b)
    self.cumulativeMu = np.cumsum(mu)
    self.res = resolution

  def _cdf(self, x):
    x = int(x / self.res)
    return self.cumulativeMu[x]

p = matern12PowerSpectrum(mu, a, b, resolution)

class RVFF_1D(gpflow.inducing_variables.InducingVariables):
    def __init__(self, a, b, M):
        self.length = M
        # [a, b] defining the interval of the Fourier representation:
        self.a = gpflow.Parameter(a, dtype=gpflow.default_float())
        self.b = gpflow.Parameter(b, dtype=gpflow.default_float())

        self.phis = gpflow.Parameter(np.random.uniform(0, 2 * np.pi, size=M))
        self.omegas = gpflow.Parameter(np.random.uniform(0, 0.5 * M, size=M))
        # self.omegas = gpflow.Parameter( p.rvs( size=M ) + default_jitter() )

    def __len__(self):
        """ number of inducing variables (defines dimensionality of q(u)) """
        return self.length  # M sine components

@cov.Kuu.register(RVFF_1D, gpflow.kernels.Matern12)
def Kuu_matern12_RVFF_1D(inducing_variable, kernel, jitter=None):
    a, b, omegas, phis = (lambda u: (u.a, u.b, u.omegas, u.phis))(inducing_variable)

    lambda_ = 1.0 / kernel.lengthscales

    def innerProduct( intervalLen, omegas, phis, kernelVar, lambda_):
      angle1 = intervalLen * ( omegas[:, None] - omegas[None, :] ) + omegas[:, None] * phis[:, None] - omegas[None, :] * phis[None, :]
      angle2 = intervalLen * ( omegas[:, None] + omegas[None, :] ) + omegas[:, None] * phis[:, None] + omegas[None, :] * phis[None, :]
      angle3 = omegas[:, None] * phis[:, None] - omegas[None, :] * phis[None, :]
      angle4 = omegas[:, None] * phis[:, None] + omegas[None, :] * phis[None, :]
      angle5 = omegas * ( 2 * intervalLen + phis[:, None] + phis[None, :] )
      angle6 = omegas * (phis[:, None] + phis[None, :])

      denom1 = tf.Variable(omegas[:, None] - omegas[None, :])
      denom2 = tf.Variable(omegas[:, None] + omegas[None, :])
      denom3 = tf.Variable(2 * omegas)

      denom1 = tf.where(denom1 == 0, 1.0, denom1)
      denom2 = tf.where(denom2 == 0, 1.0, denom2)
      denom3 = tf.where(denom3 == 0, 1.0, denom3)

      coeff1 = ( omegas[:, None] * omegas[None, :] + lambda_ * lambda_ ) / denom1
      coeff2 = ( omegas[:, None] * omegas[None, :] - lambda_ * lambda_ ) / denom2 #( omegas[:, None] + omegas[None, :] )
      coeff3 = ( omegas * omegas - lambda_ * lambda_ ) / denom3
      coeff4 = -coeff3
      coeff5 = lambda_ * lambda_ + omegas * omegas

      denom = 4  * kernelVar * lambda_

      firstTerm = ( coeff1 * tf.sin( angle1 ) +
                    coeff2 * tf.sin( angle2 ) -
                    lambda_ * tf.cos( angle2 ) +
                    lambda_ * tf.cos( angle1 ) -
                    coeff1 * tf.sin( angle3 ) -
                    coeff2 * tf.sin( angle4 ) +
                    lambda_ * tf.cos( angle4 ) -
                    lambda_ * tf.cos( angle3 ) ) / denom

      firstTermForEqualOmegas = ( ( omegas * omegas + lambda_ * lambda_ ) * tf.cos(omegas * (phis[:, None] - phis[None, :]) ) * intervalLen +
                                coeff3 * tf.sin( angle5 ) -
                                ( lambda_ * tf.cos( angle5 ) ) -
                                ( coeff3 * tf.sin( angle6 ) ) +
                                lambda_ * tf.cos( angle6 ) ) / denom

      firstTermForZeroOmegas = tf.reshape(tf.Variable(intervalLen * lambda_ / kernelVar / 2), shape=(-1))

      firstTerm = tf.where( denom1 == 1.0, firstTermForEqualOmegas, firstTerm )
      firstTerm = tf.where( denom3 == 1.0, 0.0, firstTerm )
      firstTerm = tf.where( denom2 == 1.0, firstTermForZeroOmegas, firstTerm )

      secondTermfactors = tf.where( denom3 != 1.0, tf.sin( omegas * phis ), tf.sin( phis ) )

      secondTerm = secondTermfactors[:, None] * secondTermfactors[None, :] / kernelVar

      res = firstTerm + secondTerm
      res = 0.5 * (res + tf.transpose(res))

      if jitter != None:
        res = res + tf.cast(tf.linalg.diag( jitter * tf.ones( res.shape[0] ) ), default_float())

      return res

    return innerProduct( b - a, omegas, phis, kernel.variance, lambda_ )

@cov.Kuf.register(RVFF_1D, gpflow.kernels.Matern12, TensorLike)
def Kuf_matern12_RVFF_1D(inducing_variable, kernel, X):
    X = tf.squeeze(X, axis=1)
    a, omegas, phis = (lambda u: (u.a, u.omegas, u.phis))(inducing_variable)

    nonZeroOmegas = tf.sin( omegas[:, None] * ( X[None, :] - a + phis[:, None]) )
    phis = tf.tile(tf.reshape(phis, (-1,1)), [1, len(X)])
    omegas = tf.tile(tf.reshape(omegas, (-1,1)), [1, len(X)])
    zeroOmegas = tf.sin( phis )

    res = tf.where( omegas != 0, nonZeroOmegas, zeroOmegas )

    return res
