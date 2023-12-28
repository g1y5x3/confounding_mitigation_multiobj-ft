

"""
Reference
code: https://github.com/pni-lab/mlconfound/blob/master/mlconfound/stats.py
paper: Tamas Spisak, Statistical quantification of confounding bias in machine learning models, GigaScience, Volume 11, 2022, giac082
link: https://doi.org/10.1093/gigascience/giac082
"""
import numpy as np
import pandas as pd
from pygam import LinearGAM
from scipy.stats import norm
from mlconfound.stats import partial_confound_test
from mlconfound.simulate import simulate_y_c_yhat

# NOTE
# 1. Original function combined I. density estimation, II. MCMC (Markov Chain Monte Carlo) Sampling, and III. p-value calculation in one
# single function, this implementation separate these 3 stages to for better utilization of computations.
# 2. All function implementation uses the original notation for
# H0: X ⟂ Y|C
# but when these functions are called, the input arguments are given as
# H0: C ⟂ Ŷ|Y

# I. density estimation q(X|C)
# xdtype = 'categorical' - X is a categorical variable
# xdtype = 'numerical'   - X is a continuous variable
def conditional_log_likelihood(X, C, xdtype='categorical'):
  default_kwargs = {'n_splines': 8, 'dtype': [xdtype]}
  fit = LinearGAM(**default_kwargs).gridsearch(y=X, X=C.reshape(-1, 1), progress=False)  # todo: multivariate case
  mu = np.array(fit.predict(C))
  sigma = np.repeat(np.std(X - mu), len(C))
  # X | C = C_i ~ N(mu[i], sig2[i])
  return np.array([norm.logpdf(X, loc=m, scale=sigma) for m in mu]).T

# II. MCMC permutation sampling for [C_{pi_1}, C_{pi_2}, ..., C_{pi_m}]
def generate_X_CPT_MC(nstep, log_likelihood_mat, Pi, random_state=None):
  n = len(Pi)
  npair = np.floor(n / 2).astype(int)
  rng = np.random.default_rng(random_state)
  for istep in range(nstep):
    perm = rng.choice(n, n, replace=False)
    inds_i = perm[0:npair]
    inds_j = perm[npair:(2 * npair)]
    # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
    log_odds = log_likelihood_mat[Pi[inds_i], inds_j] + log_likelihood_mat[Pi[inds_j], inds_i] - \
               log_likelihood_mat[Pi[inds_i], inds_i] - log_likelihood_mat[Pi[inds_j], inds_j]
    swaps = rng.binomial(1, 1 / (1 + np.exp(-np.maximum(-500, log_odds))))
    Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps * (Pi[inds_j] - Pi[inds_i]), Pi[inds_j] - swaps * (Pi[inds_j] - Pi[inds_i])
  return Pi

if __name__ == "__main__":
  H1_y, H1_c, H1_yhat = simulate_y_c_yhat(w_yc=0.5, w_yyhat=0.5, w_cyhat=0.1, n=1000, random_state=42)
  ret = partial_confound_test(H1_y, H1_yhat, H1_c, num_perms=1000, return_null_dist=True, random_state=42, n_jobs=-1)

  print(pd.DataFrame({'p' : [ret.p],
                      'ci lower' : [ret.p_ci[0]],
                      'ci upper' : [ret.p_ci[1]],
                      'R2(y,c)' : [ret.r2_y_c],
                      'R2(y,y^)' : [ret.r2_y_yhat],
                      'Expected R2(y^,c)': [np.round(ret.expected_r2_yhat_c, 3)],
                      'R2(y^,c)' : [ret.r2_yhat_c]}))

  random_state = 42
  cond_log_like_mat = conditional_log_likelihood(X=H1_c, C=H1_y, xdtype='numerical')
  print(cond_log_like_mat)

  mcmc_steps = 50
  Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(len(H1_c), dtype=int), random_state)
  print(Pi_init)

  # rng = np.random.default_rng(random_state)
  # random_sates = rng.integers(np.iinfo(np.int32).max, size=1)
  # print(np.arange(len(H1_c), dtype=int))
  # print(random_state)