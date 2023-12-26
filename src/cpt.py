

"""
Reference
code: https://github.com/pni-lab/mlconfound/blob/master/mlconfound/stats.py
paper: Tamas Spisak, Statistical quantification of confounding bias in machine learning models, GigaScience, Volume 11, 2022, giac082
linkL https://doi.org/10.1093/gigascience/giac082
y   : prediction target
yhat: prediction
c   : confounder

from mlconfound.stats import partial_confound_test
partial_confound_test(x=y, y=yhat, c=c)

def partial_confound_test(y, yhat, c, num_perms=1000,
                          cat_y=False, cat_yhat=False, cat_c=False,
                          mcmc_steps=50, cond_dist_method="gam",
                          return_null_dist=False, random_state=None, progress=True, n_jobs=-1):

return ResultsPartiallyConfounded(
  *cpt(x=c, y=yhat, z=y, num_perms=num_perms, t_xy=r2_c_yhat, t_xz=r2_c_y, t_yz=r2_yhat_y, condlike_f=condlike_f,
       mcmc_steps=mcmc_steps, return_null_dist=return_null_dist, random_state=random_state, progress=progress, n_jobs=n_jobs))

condlike_f = _conditional_log_likelihood_factory(cat_c, cat_y, cond_dist_method)

def _conditional_log_likelihood_factory(cat_x, cat_y, cond_dist_method):
                                        cat_c, cat_y



def cpt(x, y, z, t_xy, t_xz, t_yz, condlike_f, condlike_model_args=None, num_perms=1000, mcmc_steps=50,
        return_null_dist=False, random_state=None, progress=True, n_jobs=-1):

def _conditional_log_likelihood_gaussian_gam_cont_cat(X0, Z, **model_kwargs):
  df = pd.DataFrame({
      'Z': Z,
      'X': X0
  })
  default_kwargs = {'n_splines': 8, 'dtype': ['categorical']}
  model_kwargs = {**default_kwargs, **model_kwargs}
  fit = LinearGAM(**model_kwargs).gridsearch(y=df.X, X=df.Z.values.reshape(-1, 1),
                                              progress=False)  # todo: multivariate case
  return _gauss_cdf(fit, df)

def _conditional_log_likelihood_gaussian_gam_cont_cont(X0, Z, **model_kwargs):
  df = pd.DataFrame({
      'Z': Z,
      'X': X0
  })
  default_kwargs = {'n_splines': 8, 'dtype': ['numerical']}
  model_kwargs = {**default_kwargs, **model_kwargs}
  fit = LinearGAM(**model_kwargs).gridsearch(y=df.X, X=df.Z.values.reshape(-1, 1),
                                              progress=False)  # todo: multivariate case
  return _gauss_cdf(fit, df)

Pi_init = _generate_X_CPT_MC(mcmc_steps * 5, cond_log_lik_mat, np.arange(len(x), dtype=int),
                             random_state=random_state)

def _generate_X_CPT_MC(nstep, log_lik_mat, Pi, random_state=None):
  # modified version of: http: // www.stat.uchicago.edu / ~rina / cpt / Bikeshare1.html
  # Berrett, T.B., Wang, Y., Barber, R.F. and Samworth, R.J., 2020. The conditional permutation test
  # for independence while controlling for confounders.
  # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1), pp.175 - 197.
  n = len(Pi)
  npair = np.floor(n / 2).astype(int)
  rng = np.random.default_rng(random_state)
  for istep in range(nstep):
    perm = rng.choice(n, n, replace=False)
    inds_i = perm[0:npair]
    inds_j = perm[npair:(2 * npair)]
    # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
    log_odds = log_lik_mat[Pi[inds_i], inds_j] + log_lik_mat[Pi[inds_j], inds_i] \
                - log_lik_mat[Pi[inds_i], inds_i] - log_lik_mat[Pi[inds_j], inds_j]
    swaps = rng.binomial(1, 1 / (1 + np.exp(-np.maximum(-500, log_odds))))
    Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps * (Pi[inds_j] - Pi[inds_i]), Pi[inds_j] - \
                              swaps * (Pi[inds_j] - Pi[inds_i])
  return Pi

def workhorse(_random_state):
  # batched os job_batch for efficient parallelization
  Pi = _generate_X_CPT_MC(mcmc_steps, cond_log_lik_mat, Pi_init, random_state=_random_state)
  return t_xy(x[Pi], y)

with tqdm_joblib(tqdm(desc='Permuting', total=num_perms, disable=not progress)):
  r2_xpi_y = np.array(Parallel(n_jobs=n_jobs)(delayed(workhorse)(i) for i in random_sates))
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
# 2. All function implementation uses the original notation for H0: X ⟂ Y|C but when these functions are called, the input arguments are
# given as H0: C ⟂ Ŷ|Y

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


def _generate_X_CPT_MC(nstep, log_lik_mat, Pi, random_state=None):
  # modified version of: http: // www.stat.uchicago.edu / ~rina / cpt / Bikeshare1.html
  # Berrett, T.B., Wang, Y., Barber, R.F. and Samworth, R.J., 2020. The conditional permutation test
  # for independence while controlling for confounders.
  # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1), pp.175 - 197.
  print(nstep)
  print(log_lik_mat)
  print(Pi)
  print(random_state)
  n = len(Pi)
  npair = np.floor(n / 2).astype(int)
  rng = np.random.default_rng(random_state)
  print(rng)
  for istep in range(nstep):
    perm = rng.choice(n, n, replace=False)
    inds_i = perm[0:npair]
    inds_j = perm[npair:(2 * npair)]
    # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
    log_odds = log_lik_mat[Pi[inds_i], inds_j] + log_lik_mat[Pi[inds_j], inds_i] \
             - log_lik_mat[Pi[inds_i], inds_i] - log_lik_mat[Pi[inds_j], inds_j]
    swaps = rng.binomial(1, 1 / (1 + np.exp(-np.maximum(-500, log_odds))))
    Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps * (Pi[inds_j] - Pi[inds_i]), Pi[inds_j] - \
                              swaps * (Pi[inds_j] - Pi[inds_i])
  print(Pi)
  return Pi

# II. MCMC permutation sampling for (C_pi_1, C_pi_2, ..., C_pi_m)
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

# 3. Compute p-value for the hypothesis


if __name__ == "__main__":
  H1_y, H1_c, H1_yhat = simulate_y_c_yhat(w_yc=0.5, w_yyhat=0.5, w_cyhat=0.1, n=1000, random_state=42)
  ret=partial_confound_test(H1_y, H1_yhat, H1_c, num_perms=1, return_null_dist=True, random_state=42, n_jobs=-1)

  print(pd.DataFrame({'p' : [ret.p],
                      'ci lower' : [ret.p_ci[0]],
                      'ci upper' : [ret.p_ci[1]],
                      'R2(y,c)' : [ret.r2_y_c],
                      'R2(y,y^)' : [ret.r2_y_yhat],
                      'Expected R2(y^,c)': [np.round(ret.expected_r2_yhat_c, 3)],
                      'R2(y^,c)' : [ret.r2_yhat_c]}))

  random_state = 42
  # rng = np.random.default_rng(random_state)
  # random_sates = rng.integers(np.iinfo(np.int32).max, size=1)

  cond_log_like_mat = conditional_log_likelihood(X=H1_c, C=H1_y, xdtype='numerical')

  mcmc_steps = 50
  # print(cond_log_like_mat)
  # print(np.arange(len(H1_c), dtype=int))
  # print(random_state)
  Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(len(H1_c), dtype=int), random_state)
  print(Pi_init)