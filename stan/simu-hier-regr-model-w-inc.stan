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
  int<lower=0> N_geo;
  matrix[N,J] E;
  matrix[N, K] X;
  int<lower=1, upper = N_geo> geo_idx[N];
  // Hierarchical mean parameters
  vector<lower=0>[K] prior_scales_alpha_beta;
  vector<lower=0>[K] prior_scales_alpha_gamma;
  vector[K] prior_mean_alpha_beta;
  vector[K] prior_mean_alpha_gamma;
  vector<lower=0>[J] prior_scales_alpha_lambda;
  vector<lower=0>[J] prior_scales_alpha_eta;
  vector[J] prior_mean_alpha_lambda;
  vector[J] prior_mean_alpha_eta;
  // Hierarchical variance parameters
  vector<lower=0>[J] prior_mean_sigma_lambda;
  vector<lower=0>[J] prior_scales_sigma_lambda;
  vector[J] prior_mean_sigma_eta;
  vector[J] prior_scales_sigma_eta;
  vector[K] prior_mean_sigma_beta;
  vector[K] prior_scales_sigma_beta;
  vector[K] prior_mean_sigma_gamma;
  vector[K] prior_scales_sigma_gamma;
  // stratum information
  int<lower=1> N_age_sex;
  int<lower=1, upper=N_age_sex> age_sex_idx[N];
}
transformed data {
  real tot_pop;
  vector[J] pop_by_cat;
  real age_sex_denom[N_age_sex] = rep_array(0.0, N_age_sex);
  for (n in 1:N) {
    age_sex_denom[age_sex_idx[n]] += sum(E[n,]);
  }
  for (j in 1:J) {
    pop_by_cat[j] = sum(E[,j]);
  }
  tot_pop = sum(pop_by_cat);
}
generated quantities {
  // Poisson parameters
  vector[J] alpha_lambda;
  vector[K] alpha_beta;
  matrix[N_geo, J] log_lambda;
  matrix[N_geo, K] beta;
  vector[J] sigma_lambda;
  vector[K] sigma_beta;
  // Bernoulli parameters
  vector[J] alpha_eta;
  vector[K] alpha_gamma;
  matrix[N_geo, J] eta;
  matrix[N_geo, K] gamma;
  vector[J] sigma_eta;
  vector[K] sigma_gamma;
  int y_obs[N,J];
  int y_tot[N];
  int y_miss[N];
  int N_miss;
  int idx_miss[N] = rep_array(0,N);
  int miss_count;
  real miss_prop;
  real incidence;
  matrix[N,J] mu_pois;
  matrix[N,J] mu_bern;
  vector[J] incidence_by_race;
  vector[J] filtered_incidence_by_race;
  real age_sex_inc[N_age_sex] = rep_array(0.0,N_age_sex);
  real standard_incidence_by_race[J] = rep_array(0.0,J);
  {
    int y_latent[N,J];
    for (k in 1:K) {
      alpha_beta[k] = normal_rng(prior_mean_alpha_beta[k], prior_scales_alpha_beta[k]);
      alpha_gamma[k] = normal_rng(prior_mean_alpha_gamma[k], prior_scales_alpha_gamma[k]);
      sigma_beta[k] = fabs(normal_rng(prior_mean_sigma_beta[k],
                                      prior_scales_sigma_beta[k]));
      sigma_gamma[k] = fabs(normal_rng(prior_mean_sigma_gamma[k],
                                      prior_scales_sigma_gamma[k]));
    }
    for (j in 1:J) {
      alpha_lambda[j] = normal_rng(prior_mean_alpha_lambda[j], prior_scales_alpha_lambda[j]);
      alpha_eta[j] = normal_rng(prior_mean_alpha_eta[j], prior_scales_alpha_eta[j]);
      sigma_lambda[j] = fabs(normal_rng(prior_mean_sigma_lambda[j],
                                        prior_scales_sigma_lambda[j]));
      sigma_eta[j] = fabs(normal_rng(prior_mean_sigma_eta[j],
                                        prior_scales_sigma_eta[j]));
    }
    for (n in 1:N_geo) {
      for (j in 1:J) {
        log_lambda[n,j] = alpha_lambda[j] + normal_rng(0, sigma_lambda[j]);
        eta[n,j] = alpha_eta[j] + normal_rng(0, sigma_eta[j]);
      }
      for (k in 1:K) {
        beta[n,k] = alpha_beta[k] + normal_rng(0, sigma_beta[k]);
        gamma[n,k] = alpha_gamma[k] + normal_rng(0, sigma_gamma[k]);
      }
    }
    for (j in 1:J) { 
      for (n in 1:N) {
        mu_pois[n,j] = X[n,] * beta[geo_idx[n],]' + log_lambda[geo_idx[n], j];
        if (E[n,j] > 0)
          y_latent[n,j] = poisson_safe_rng(E[n,j] * exp(mu_pois[n,j]));
        else
          y_latent[n,j] = 0;
      }
    }
    for (j in 1:J) {
      for (n in 1:N) {
        mu_bern[n,j] = X[n,] * gamma[geo_idx[n], ]' + eta[geo_idx[n],j];
        y_obs[n,j] = binomial_rng(y_latent[n,j],inv_logit(mu_bern[n,j]));
      }
    }
    for (n in 1:N)
      y_tot[n] = sum(y_latent[n,]);
    for (n in 1:N)
      y_miss[n] = y_tot[n] - sum(y_obs[n,]);
    for (n in 1:N)
      age_sex_inc[age_sex_idx[n]] += dot_product(exp(mu_pois[n,]), E[n,]) / age_sex_denom[age_sex_idx[n]];
    for (n in 1:N)
      for (j in 1:J)
        standard_incidence_by_race[j] += E[n,j] * age_sex_inc[age_sex_idx[n]]
          / pop_by_cat[j];
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
        incidence_by_race[j] = dot_product(E[,j], exp(mu_pois[,j])) / pop_by_cat[j];
        filtered_incidence_by_race[j] = dot_product(E[,j],
                                                    inv_logit(mu_bern[,j])
                                                    .* exp(mu_pois[,j]))
          / pop_by_cat[j];
      }
      miss_prop = miss_count / (1.0 * sum(y_tot));
      incidence = (1.0 * sum(y_tot)) / tot_pop;
    }
  }
}

