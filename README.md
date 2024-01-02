# Confounding Mitigation in Deep Learning
## TO-DO
- [] fix CPT algorithm for the loss function
  - currently it uses the batch to estimate the initial density function which is not working. In order to fix it, the density has to be estimated
  from the whole set. However, to do so, all the indices of samples during training w.r.t the original dataloader needs to be tracked in order to
  retrieve the correct log-likelihood value which is very messy.
  - distance correlation is slow to compute when the size for calculating pair-wise distances get too large.
- [] GA-SVM for fMRI data (basically copy the workflow from [sEMG_GA-SVM.py](src/sEMG_GA-SVM.py) script to [train.py](src/train.py) which uses CNN to
predict subject ages with fMRI)
- [] Implement MLP for fMRI data, will be extremely similar workflow to [train.py](src/train.py)
- [] Maybe use the voice acoustic dataset for depression prediction as the third dataset since the GA-SVM workflow already exists. But need to make
sure all the previous are completed.

## Implementation details for emmbedding CPT into the Loss function
The original library contains a bunch of functions for utility which makes it harder to read and understand. There are essentially 3 steps for the CPT
algorithm to compute the p-value:

1. estimate the probably density $q(c|y)$

    Note that we condition the probability density estimation on $\textbf{y}$ which can be either **continuous** or **categorical**.
    ```python
    def conditional_log_likelihood(X, C, xdtype='categorical'):
      default_kwargs = {'n_splines': 8, 'dtype': [xdtype]}
      fit = LinearGAM(**default_kwargs).gridsearch(y=X, X=C.reshape(-1, 1), progress=False)
      mu = np.array(fit.predict(C))
      sigma = np.repeat(np.std(X - mu), len(C))

      return np.array([norm.logpdf(X, loc=m, scale=sigma) for m in mu]).T
    ```
    GAM model is used to estimate
    $$ c = \alpha + \beta f(y) + e $$
    where the feature function $f$ is built using penalized B-splines. If we write $\mu=\alpha+\beta f(y)$ and $\sigma$ denotes the standard deviation of the residual $e$

2. sample permutations of $c^{(i)}$ using Markov Chain Monte Carlo sampler (**the slowest part of the algorithm**)

    [generate_X_CPT_MC](src/cpt.py)
    ```python
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
      # Pi is a permutations of array indices
      return Pi
    ```
    It draws disjoint pairs in parallel and decides whether or not to swap them randomly, according to the odds ratio calculated from the conditional
    densities belonging to the original and swapped data.
    $$
    ln\frac{q(c_j|y_i)q(c_i|y_j)}{q(c_i|y_i)q(c_j|y_j)} = \ell(c_j|y_i)+\ell(c_i|y_j)-\ell(c_i|y_i)-\ell(c_j|y_j)
    $$
    where $\ell$ denotes the log-likelihood from 1.

3. calculate p-value
    ```python
    def cpt_p_dcor(c, yhat, y, mcmc_steps=50, random_state=123, num_perm=1000):
      # sampling permutations of c
      bs = y.shape[0]
      cond_log_like_mat = conditional_log_likelihood(X=c.numpy(), C=y.numpy(), xdtype='categorical')
      Pi_init = generate_X_CPT_MC(mcmc_steps*5, cond_log_like_mat, np.arange(bs, dtype=int), random_state)

      def workhorse(c, _random_state):
        # batched os job_batch for efficient parallelization
        Pi = generate_X_CPT_MC(mcmc_steps, cond_log_like_mat, Pi_init, random_state=_random_state)
        return c[Pi]
      rng = np.random.default_rng(random_state)
      random_states = rng.integers(np.iinfo(np.int32).max, size=num_perm)
      c_pi = torch.tensor(np.array(Parallel(n_jobs=-1)(delayed(workhorse)(c, i) for i in random_states)), dtype=torch.float32)
      # compute p-value
      t_yhat_c = distance_correlation(yhat.reshape([bs, -1]), c.reshape([bs, -1])).repeat(num_perm)
      t_yhat_cpi = torch.zeros(num_perm)
      for i in range(num_perm):
        t_yhat_cpi[i] = distance_correlation(yhat.reshape([bs, -1]), c_pi[i,:].reshape([bs, -1]))

      return torch.sigmoid(t_yhat_cpi - t_yhat_c).mean()
    ```
    $$
    p = \frac{\sum^m_{i=1}{1\{T(c^{(i)}, \hat{y}, y) \geq T(c, \hat{y}, y)\}}}{m}
    $$
    There are a few prerequisits here in order to be able to apply this metric into the loss function.
    1. function $T$ needs to be non-parametric and diffientable - distance correlation (correlation of pairwise distances within the two variables)
    2. $\geq$ is essentially a step function which is not diffientable, so it is replaced with a **sigmoid** function of
    $T(c^{(i)}, \hat{y}, y) - T(c, \hat{y}, y)$.