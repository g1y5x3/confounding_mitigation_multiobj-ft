# Confounding Mitigation in Machine Learning

## Implementation details for CPT
The experiments use the library [mlconfound](https://github.com/pni-lab/mlconfound). However, the original library contains a bunch of functions for utility 
which makes it harder to read and understand. There are essentially 3 steps for the CPT algorithm to compute the p-value:

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
    $$c = \alpha + \beta f(y) + e$$
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
    $$ln\frac{q(c_j|y_i)q(c_i|y_j)}{q(c_i|y_i)q(c_j|y_j)} = \ell(c_j|y_i)+\ell(c_i|y_j)-\ell(c_i|y_i)-\ell(c_j|y_j)$$
    where $\ell$ denotes the log-likelihood from 1.

3. calculate p-value
    ```python
    t_x_y   = np.corrcoef(x, y)[0,1]
    t_xpi_y = np.zeros(num_perm)
    y_tile  = np.tile(y, (num_perm,1))
    for i in range(num_perm):
      t_xpi_y[i] = np.corrcoef(x_perm[i,:], y_tile[i,:])[0,1]
    p = np.sum(t_xpi_y >= t_x_y) / len(t_xpi_y)
    ```
    $$p = \frac{\sum^m_{i=1}{1\{T(c^{(i)}, \hat{y}, y) \geq T(c, \hat{y}, y)\}}}{m}$$