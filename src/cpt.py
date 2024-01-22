"""
Reference
code: https://github.com/pni-lab/mlconfound/blob/master/mlconfound/stats.py
paper: Tamas Spisak, Statistical quantification of confounding bias in machine learning models, GigaScience, Volume 11, 2022, giac082
link: https://doi.org/10.1093/gigascience/giac082
"""
import torch
import numpy as np
import pandas as pd
from pygam import LinearGAM
from scipy.stats import norm
from joblib import Parallel, delayed
from mlconfound.stats import partial_confound_test
from mlconfound.simulate import simulate_y_c_yhat

# NOTE
# 1. Original function combined
#   I.   density estimation
#   II.  MCMC (Markov Chain Monte Carlo) Sampling,
#   III. p-value calculation
# in one single function. It also contains a bunch of wrapper function that is not necessarily
# needed. Therefore, the modification here is to separate out I. and II. The last step is done
# inside the loss function of neural networks in able to utilize autograd.
#
# 2. All function are implemented based on notation for
# H0: X ⟂ Y|C
# but when these functions are called, the input arguments are given as
# H0: C ⟂ Ŷ|Y

# I. density estimation Q(X|C)
# xdtype = 'categorical' - X is a categorical variable
# xdtype = 'numerical'   - X is a continuous variable
# Each element [i,j] in the result matrix is the log-likelihood of observing X[j] given C[i]
def conditional_log_likelihood(X, C, xdtype='categorical'):
  # https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html
  default_kwargs = {'n_splines': 8, 'dtype': [xdtype]}
  fit = LinearGAM(**default_kwargs).gridsearch(y=X, X=C.reshape(-1, 1), progress=False)  # todo: multivariate case
  mu = np.array(fit.predict(C))
  sigma = np.repeat(np.std(X - mu), len(C))

  # TODO If C is continuous variable, it makes to track based on the entire len(C)
  # However, if C is categorical, in theory it's only necessarily to track each category
  return np.array([norm.logpdf(X, loc=m, scale=sigma) for m in mu]).T

# II. MCMC permutation sampling for [C_{pi_1}, C_{pi_2}, ..., C_{pi_m}]
def generate_X_CPT_MC(nstep, log_likelihood_mat, Pi, random_state=None):
  n = len(Pi)
  npair = np.floor(n / 2).astype(int)
  rng = np.random.default_rng(random_state)
  for _ in range(nstep):
    perm = rng.choice(n, n, replace=False)
    inds_i = perm[0:npair]
    inds_j = perm[npair:(2 * npair)]
    # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
    log_odds = log_likelihood_mat[Pi[inds_i], inds_j] + log_likelihood_mat[Pi[inds_j], inds_i] - \
               log_likelihood_mat[Pi[inds_i], inds_i] - log_likelihood_mat[Pi[inds_j], inds_j]
    swaps = rng.binomial(1, 1 / (1 + np.exp(-np.maximum(-500, log_odds))))
    Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps * (Pi[inds_j] - Pi[inds_i]), Pi[inds_j] - swaps * (Pi[inds_j] - Pi[inds_i])
  # Pi is a permutations of array indices
  return Pi

def cpt_p_pearson(c, yhat, yt, cond_like_mat=None, mcmc_steps=50, random_state=None, num_perm=1000, dtype='numerical'):
  # fully confounder test    H0: X ⟂ Y|C
  # partical confounder test H0: C ⟂ Ŷ|Y
  x, y, c = c, yhat, yt

  # 1. density estimation
  if cond_like_mat is None: 
    cond_log_like_mat = conditional_log_likelihood(X=x, C=c, xdtype=dtype)
  else:
    cond_log_like_mat = cond_like_mat

  # 2. permutation sampling
  Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(len(x), dtype=int), random_state)
  def workhorse(_random_state):
    # batched os job_batch for efficient parallelization
    Pi = generate_X_CPT_MC(mcmc_steps, cond_log_like_mat, Pi_init, random_state=_random_state)
    return x[Pi]
  rng = np.random.default_rng(random_state)
  random_states = rng.integers(np.iinfo(np.int32).max, size=num_perm)
  x_perm = np.array(Parallel(n_jobs=-1)(delayed(workhorse)(i) for i in random_states))

  # 3. p-value calculation
  # compute t_xy which is just Pearson correlation in this case but is replaced with a
  # different metric in the neural networks loss function
  t_x_y    = np.corrcoef(x, y)[0,1] ** 2
  t_xpi_y = np.zeros(num_perm)
  y_tile   = np.tile(y, (num_perm,1))
  for i in range(num_perm):
    t_xpi_y[i] = np.corrcoef(x_perm[i,:], y_tile[i,:])[0,1] ** 2
  p = np.sum(t_xpi_y >= t_x_y) / len(t_xpi_y)
  return p, t_xpi_y

def cpt_p_pearson_torch(x, y, cond_log_like_mat, mcmc_steps=50, num_perm=1000, random_state=None, dtype='numerical'):
  # fully confounder test (notation referred to in this implementation) - H0: X ⟂ Y|C
  # partical confounder test                                            - H0: C ⟂ Ŷ|Y
  # x, y - c, yhat

  # 1. density estimation is done ahead of time to then conert to torch tensors 

  # 2. permutation sampling. can be done without torch, all that matters was generated permutation is converted to torch
  Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(len(x), dtype=int), random_state)
  def workhorse(_random_state):
    Pi = generate_X_CPT_MC(mcmc_steps, cond_log_like_mat, Pi_init, random_state=_random_state)
    return x[Pi]
  rng = np.random.default_rng(random_state)
  random_states = rng.integers(np.iinfo(np.int32).max, size=num_perm)
  x_perm = np.array(Parallel(n_jobs=-1)(delayed(workhorse)(i) for i in random_states))

  # 3. p-value calculation
  # compute t_xy which is just Pearson correlation in this case but is replaced with a
  # different metric in the neural networks loss function
  t_x_y    = np.corrcoef(x, y)[0,1] ** 2
  t_xpi_y = np.zeros(num_perm)
  y_tile   = np.tile(y, (num_perm,1))
  for i in range(num_perm):
    t_xpi_y[i] = np.corrcoef(x_perm[i,:], y_tile[i,:])[0,1] ** 2
  p = np.sum(t_xpi_y >= t_x_y) / len(t_xpi_y)
  return p, t_xpi_y

def verify_implementation(random_state, num_perm, H1_y, H1_c, H1_yhat):
  # original function
  ret = partial_confound_test(H1_y, H1_yhat, H1_c, num_perms=num_perm, return_null_dist=True, random_state=random_state, n_jobs=-1)
  print(pd.DataFrame({'p' : [ret.p],
                      'ci lower' : [ret.p_ci[0]],
                      'ci upper' : [ret.p_ci[1]],
                      'R2(y,c)' : [ret.r2_y_c],
                      'R2(y,y^)' : [ret.r2_y_yhat],
                      'Expected R2(y^,c)': [np.round(ret.expected_r2_yhat_c, 3)],
                      'R2(y^,c)' : [ret.r2_yhat_c]}))

  p, t_xpi_y = cpt_p_pearson(H1_c, H1_yhat, H1_y, random_state=random_state, num_perm=num_perm)

  # verify the results to make sure that they match with the original implementation
  print(f"original implementation   p-value: {ret.p}")
  print(f"simplified implementation p-value: {p}")

  assert np.allclose(ret.p, p), "p-value does not match with original implementation"
  assert np.allclose(ret.null_distribution, t_xpi_y), "null distribution does not match with original implementation"

def verify_np_vs_torch(random_state, num_perm, H1_y, H1_c, H1_yhat):
  # original function
  ret = partial_confound_test(H1_y, H1_yhat, H1_c, num_perms=num_perm, return_null_dist=True, random_state=random_state, n_jobs=-1)
  print(pd.DataFrame({'p' : [ret.p],
                      'ci lower' : [ret.p_ci[0]],
                      'ci upper' : [ret.p_ci[1]],
                      'R2(y,c)' : [ret.r2_y_c],
                      'R2(y,y^)' : [ret.r2_y_yhat],
                      'Expected R2(y^,c)': [np.round(ret.expected_r2_yhat_c, 3)],
                      'R2(y^,c)' : [ret.r2_yhat_c]}))

if __name__ == "__main__":
  # exampled from https://github.com/pni-lab/mlconfound/blob/master/notebooks/quickstart.ipynb used to verify
  H1_y, H1_c, H1_yhat = simulate_y_c_yhat(w_yc=0.5, w_yyhat=0.5, w_cyhat=0.1, n=1000, random_state=42)

  # verify the original implementation vs. the simplified numpy implementation
  # verify_implementation(num_perm=25,   random_state=25,  H1_y=H1_y, H1_c=H1_c, H1_yhat=H1_yhat)
  # verify_implementation(num_perm=50,   random_state=5,   H1_y=H1_y, H1_c=H1_c, H1_yhat=H1_yhat)
  # verify_implementation(num_perm=100,  random_state=30,  H1_y=H1_y, H1_c=H1_c, H1_yhat=H1_yhat)
  # verify_implementation(num_perm=250,  random_state=2,   H1_y=H1_y, H1_c=H1_c, H1_yhat=H1_yhat)
  # verify_implementation(num_perm=500,  random_state=130, H1_y=H1_y, H1_c=H1_c, H1_yhat=H1_yhat)
  # verify_implementation(num_perm=1000, random_state=421, H1_y=H1_y, H1_c=H1_c, H1_yhat=H1_yhat)

  # verify the numpy implementation vs. the torch implementation
  cond_like_mat = conditional_log_likelihood(X=H1_c, C=H1_y, xdtype='numerical')
  print(cond_like_mat)
  print(cond_like_mat.shape)