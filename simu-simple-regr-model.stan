functions {
  int poisson_safe_rng(real eta) {
    real eta2 = (log(eta) < 20.79) ? log(eta) : 20.79;
    return poisson_log_rng(eta2);
  }
}
data {
  int<lower=2> J;
  int<lower=0> N;
  int<lower=0> K;
  matrix[N,J] E;
  matrix[N, K] X;
  vector<lower=0>[K] prior_scales_beta;
  vector<lower=0>[K] prior_scales_theta;
  vector[K] prior_mean_beta;
  vector[K] prior_mean_theta;
  vector<lower=0>[J] prior_scales_alpha;
  vector<lower=0>[J] prior_scales_gamma;
  vector[J] prior_mean_alpha;
  vector[J] prior_mean_gamma;
}
transformed data {
  real tot_pop;
  vector[J] pop_by_cat;
  for (j in 1:J) {
    pop_by_cat[j] = sum(E[,j]);
  }
  tot_pop = sum(pop_by_cat);
}
generated quantities {
  // Poisson parameters
  vector[J] alpha;
  vector[K] beta;
  // Bernoulli parameters
  vector[J] gamma;
  vector[K] theta;
  int y_obs[N,J];
  int y_tot[N];
  int y_miss[N];
  int N_miss;
  int idx_miss[N] = rep_array(0,N);
  int miss_count;
  real miss_prop;
  real incidence;
  vector[J] incidence_by_cat;
  matrix[N,J] mu_pois;
  matrix[N,J] mu_bern;
  {
    int y_latent[N,J];
    for (k in 1:K) {
      beta[k] = normal_rng(prior_mean_beta[k], prior_scales_beta[k]);
      theta[k] = normal_rng(prior_mean_theta[k], prior_scales_theta[k]);
    }
    for (j in 1:J) {
      alpha[j] = normal_rng(prior_mean_alpha[j], prior_scales_alpha[j]);
      gamma[j] = normal_rng(prior_mean_gamma[j], prior_scales_gamma[j]);
    }
    for (j in 1:J) { 
      mu_pois[,j] = X * beta + alpha[j];
      for (n in 1:N)
        if (E[n,j] > 0)
          y_latent[n,j] = poisson_safe_rng(E[n,j] * exp(mu_pois[n,j]));
        else
          y_latent[n,j] = 0;
    }
    for (j in 1:J) {
      mu_bern[,j] = X * theta + gamma[j];
      for (n in 1:N)
        y_obs[n,j] = binomial_rng(y_latent[n,j],inv_logit(mu_bern[n,j]));
    }
    for (n in 1:N)
      y_tot[n] = sum(y_latent[n,]);
    for (n in 1:N)
      y_miss[n] = y_tot[n] - sum(y_obs[n,]);
    {
      int idx = 1;
      for (n in 1:N)
        if (y_miss[n] > 0) {
          idx_miss[idx] = n;
          idx = idx + 1;
        }
      N_miss = idx - 1;
      miss_count = 0;
      for (j in 1:J) {
        miss_count += sum(y_latent[,j]) - sum(y_obs[,j]);
        incidence_by_cat[j] = sum(y_latent[,j]) / pop_by_cat[j];
      }
      miss_prop = miss_count / (1.0 * sum(y_tot));
      incidence = (1.0 * sum(y_tot)) / tot_pop;
    }
  }
}

