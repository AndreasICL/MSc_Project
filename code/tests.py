from gpflow.config import default_float, default_jitter
import matplotlib.pyplot as plt
from shutil import copyfile, rmtree

import numpy as np
import tensorflow as tf
import gpflow
import pickle
import os

import VFF_IV
from RVFF_IV_1D import RVFF_1D

# ===================================================================
# Testing whether the model is the equivalent to Variational Fourier
# Features when the frequencies and phases are set manually to
# correspond to multiples of the base frequency, and not optimised as
# variational parameters.
# ===================================================================

def testFeatureequivalence(tol):
  VFFlength = 10
  RVFFlength = 2 * VFFlength - 1

  kernel = gpflow.kernels.Matern12(variance=1.0, lengthscales=0.7)

  VFFinducingVar = VFF_IV.FourierFeatures1D(0, 6, VFFlength)
  VFFmodel = gpflow.models.SGPR((X, Y), kernel, VFFinducingVar)

  RVFFinducingVar = RVFF_1D(0, 6, RVFFlength)

  RVFFfrequencies = tf.concat( ( VFFinducingVar.omegas, tf.reshape( tf.gather( VFFinducingVar.omegas, tf.where( VFFinducingVar.omegas != 0.0 ) ), [-1] ) ), axis=0 )
  RVFFphases = tf.concat( ( np.pi / ( 2 * VFFinducingVar.omegas[1:VFFlength] ), tf.zeros(RVFFlength - VFFlength, dtype=default_float()) ), axis=0 )
  RVFFphases = tf.concat( ( tf.constant([ np.pi / 2 ], dtype=default_float()), RVFFphases ), axis=0 )

  RVFFinducingVar.omegas.assign( RVFFfrequencies )
  RVFFinducingVar.phis.assign( RVFFphases )

  RVFFmodel = gpflow.models.SGPR((X, Y), kernel, RVFFinducingVar)

  meanVFF, covVFF = VFFmodel.predict_f(Xtest, True, False)
  meanRVFF, covRVFF = RVFFmodel.predict_f(Xtest, True, False)

  plot(Xtest, [(meanVFF, tf.transpose(tf.linalg.diag_part(covVFF)), 'r', 'c'), (meanRVFF, tf.transpose(tf.linalg.diag_part(covRVFF)), 'b', 'g')])
  print("If you can only see one mean function and one error bar, VFF and RVFF perfectly coincide.")

  print("RVFF model ELBO is %2.2f" % RVFFmodel.elbo())
  print("VFF model ELBO is %2.2f" % VFFmodel.elbo())

  return tf.math.reduce_sum(tf.math.abs(covRVFF-covVFF)) < tol and tf.math.reduce_sum(tf.math.abs(meanRVFF-meanVFF)) < tol

# ===================================================================
# Testing whether $q(u) = p(u)$ when we fix q_sqrt = cholesky($K_{uu}$)
# ===================================================================

def testPriorEqualsPosterior():

  kernel = gpflow.kernels.Matern12(variance=1.0, lengthscales=0.7)
  likelihood = gpflow.likelihoods.Gaussian()

  inducing_variable = RVFF_1D( a=0, b=6, M=100 )

  model = gpflow.models.SVGP(
      kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable
  )
  q_sqrt = tf.linalg.cholesky( gpflow.covariances.Kuu(model.inducing_variable, model.kernel) )

  return model.prior_kl()

# ===================================================================
# Testing whether $K_{uu}$ is computed correctly with Monte-Carlo integration:
# ===================================================================

def grid_search_integrate(f, bound1, bound2, numOfSamples):
  samples = tf.range(bound1, bound2, ( bound2 - bound1 ) / float(numOfSamples))
  V = tf.math.reduce_prod( bound2 - bound1 )

  return V * tf.math.reduce_sum( f( samples ) ) / numOfSamples

def mattern_half_inner_product_integrand(f, g, df, dg, lambda_):
  def evaluate_on(x):
    return ( lambda_ * f(x) + df(x) ) * ( lambda_ * g(x) + dg(x) )
  return evaluate_on

def grid_search_mattern_half_inner_prod( sigma, lambda_, a, b, omega1, phi1, omega2, phi2 ):
  f = sin(omega1, a, phi1)
  g = sin(omega2, a, phi2)

  df = dsin(omega1, a, phi1)
  dg = dsin(omega2, a, phi2)

  integral = grid_search_integrate(
      mattern_half_inner_product_integrand(f, g, df, dg, lambda_),
      a,
      b,
      10e6
  )

  term1 = integral / ( 2 * lambda_ * sigma * sigma )
  term2 = f(a) * g(a) / (sigma * sigma)

  return term1 + term2

def sin( omega, a, phi ):
  def compute_sin(x):
    return tf.sin( omega * ( x - a + phi ) )
  return compute_sin

def dsin( omega, a, phi ):
  def compute_dsin(x):
    return omega * tf.cos( omega * ( x - a + phi ) )
  return compute_dsin

def test_Kuu_evaluation(inducing_variable, kernel, tol):
  analyticKuu = gpflow.covariances.Kuu(inducing_variable, kernel)
  numericalKuu = np.empty((len(inducing_variable), len(inducing_variable)))

  a, b, omegas, phis = (lambda u: (u.a, u.b, u.omegas, u.phis)) (inducing_variable)
  lambda_, sigma = (lambda k: (1/k.lengthscales, k.variance))(kernel)

  for i in range(len(inducing_variable)):
    for j in range(len(inducing_variable)):
      numericalKuu[i, j] = grid_search_mattern_half_inner_prod(sigma, lambda_, a, b, omegas[i], phis[i], omegas[j], phis[j])

  print(analyticKuu)
  print(numericalKuu)

  return True if tf.reduce_all( abs(analyticKuu - numericalKuu) < tol ) else False

# ===================================================================
# Testing whether the predictive covariance $K_{ff}$ is positive semidefinite.
# ===================================================================

def is_psd(matrix, tol):
  return tf.reduce_all(tf.linalg.eigh(matrix)[0] > tol)

def test_predictive_cov_is_psd(model):
  Xtest = np.arange(0, 6, 0.05).reshape(-1, 1)
  predictive_cov = tf.squeeze( model.predict_f(X, True, False)[1] )
  return is_psd(predictive_cov, -1e-1)

# ===================================================================
# ===================================================================

def loadModel(path):
    with open(path, 'rb') as fp:
        # param_dict = pickle.load(fp)
        omegas = pickle.load(fp)
        phis = pickle.load(fp)
        likelihoodVariance = pickle.load(fp)
        param_dict = [omegas, phis, likelihoodVariance]

    inducing_variable = RVFF_1D( a=0, b=6, M=len(omegas.numpy()) )
    inducing_variable.omegas = omegas
    inducing_variable.phis = phis

    kernel = gpflow.kernels.Matern12(variance=1.0, lengthscales=0.7)

    model = gpflow.models.SGPR((X, Y), kernel=kernel, inducing_variable=inducing_variable)
    model.likelihood.variance = likelihoodVariance

    gpflow.utilities.set_trainable(kernel.lengthscales, False)
    gpflow.utilities.set_trainable(kernel.variance, False)
    gpflow.utilities.set_trainable(inducing_variable.a, False)
    gpflow.utilities.set_trainable(inducing_variable.b, False)

    if len(model.trainable_parameters) == len(param_dict):
      for i in range(len(model.trainable_parameters)):
        model.trainable_parameters[i].assign(param_dict[i])

    return model

def plot(Xtest, models):

  for (mean, std, meanColour, stdColour) in models:
    plt.plot(Xtest, mean, color=meanColour)

    plt.plot(Xtest, mean + std, color=stdColour)
    plt.plot(Xtest, mean - std, color=stdColour)
    plt.fill_between(np.squeeze(Xtest), np.squeeze(mean + std), np.squeeze(mean - std), color=stdColour, alpha=0.2)


  plt.scatter(X, Y, s=15)

  plt.show()

'''
  This code was copied from https://github.com/edwardlib/observations/blob/master/observations/snelson1d.py
  on 01/05/2020
'''

from observations.util import maybe_download_and_extract


def snelson1d(path):
  """Load Edward Snelson's 1d regression data set [@snelson2006fitc].
  It contains 200 examples of a few oscillations of an example function. It has
  seen extensive use as a toy dataset for illustrating qualitative behaviour of
  Gaussian process approximations.
  Args:
    path: str.
      Path to directory which either stores file or otherwise file will be
      downloaded and extracted there. Filenames are `snelson_train_*`.
  Returns:
    Tuple of two np.darray `inputs` and `outputs` with 200 rows and 1 column.
  """
  path = os.path.expanduser(path)
  inputs_path = os.path.join(path, 'snelson_train_inputs')
  outputs_path = os.path.join(path, 'snelson_train_outputs')

  # Contains all source as well. We just need the data.
  url = 'http://www.gatsby.ucl.ac.uk/~snelson/SPGP_dist.zip'

  if not (os.path.exists(inputs_path) and os.path.exists(outputs_path)):
    maybe_download_and_extract(path, url)

    # Copy the required data
    copyfile(os.path.join(path, "SPGP_dist", "train_inputs"), inputs_path)
    copyfile(os.path.join(path, "SPGP_dist", "train_outputs"), outputs_path)

    # Clean up everything else
    rmtree(os.path.join(path, "SPGP_dist"))
    os.remove(os.path.join(path, "SPGP_dist.zip"))

  X = np.loadtxt(os.path.join(inputs_path))[:, None]
  Y = np.loadtxt(os.path.join(outputs_path))[:, None]

  return X, Y

Xtest = np.arange(0.0, 6.0, 0.01).reshape(-1, 1)
X, Y = snelson1d(".")

model = loadModel('./model')

str = "True" if test_predictive_cov_is_psd(model) else "False"
print( "predictive_cov is psd: " + str )

str = "Test passed!" if test_Kuu_evaluation(RVFF_1D(0, 6, 2), gpflow.kernels.Matern12(), 10e-3) else "Test failed."
print(str)

str = "Test passed!" if testFeatureequivalence(10e-2) else "Test failed."
print(str)

print( "KL[ q(u) | p(u) ] = %2.3f" % testPriorEqualsPosterior() )
