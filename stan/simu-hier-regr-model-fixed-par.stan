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
  vector[J] alpha_lambda_fixed;
  vector[J] alpha_eta_fixed;
  vector[K] alpha_gamma_fixed;
  vector[K] alpha_beta_fixed;
  // Hierarchical variance parameters
  vector[J] sigma_lambda_fixed;
  vector[J] sigma_eta_fixed;
  vector[K] sigma_gamma_fixed;
  vector[K] sigma_beta_fixed;
  //
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
  matrix[N_geo, J] log_lambda;
  matrix[N_geo, K] beta;
  // Bernoulli parameters
  matrix[N_geo, J] eta;
  matrix[N_geo, K] gamma;
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
  vector[J] alpha_lambda = alpha_lambda_fixed;
  vector[J] alpha_eta = alpha_eta_fixed;
  vector[K] alpha_gamma = alpha_gamma_fixed;
  vector[K] alpha_beta =  alpha_beta_fixed;
  vector[J] sigma_lambda = sigma_lambda_fixed;
  vector[J] sigma_eta =  sigma_eta_fixed;
  vector[K] sigma_gamma = sigma_gamma_fixed;
  vector[K] sigma_beta = sigma_beta_fixed;
  vector[J] prop_obs_by_race;
  vector[J] obs_inc_by_race;
  vector[J] true_inc_by_race;
  {
    int y_latent[N,J];
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
    for (j in 1:J) {
      prop_obs_by_race[j] = (1.0 * sum(y_obs[,j])) / sum(y_latent[,j]);
      true_inc_by_race[j] = (1.0 * sum(y_latent[,j]) / pop_by_cat[j]);
      obs_inc_by_race[j] = (1.0 * sum(y_obs[,j]) / pop_by_cat[j]);
    }
  }
}

