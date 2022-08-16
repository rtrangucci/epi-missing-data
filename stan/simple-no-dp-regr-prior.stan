functions {
  int poisson_safe_rng(real eta) {
    real eta2 = (log(eta) < 20.79) ? log(eta) : 20.79;
    return poisson_log_rng(eta2);
  }
}
data {
  int<lower=2> J; // Number of categories in variable with missingness (e.g. race)
  int<lower=0> N; // Number of strata observed (e.g. number of age-sex strata)
  int<lower=0> K; // Dimensions of stratum-specific predictors
  matrix[N,J] E; // Population counts for stratum by category 
  matrix[N, K] X; // Predictor matrix
  vector<lower=0>[K] prior_scales_beta_fit; // Prior scales for disease-log-rate coefficients
  vector<lower=0>[K] prior_scales_gamma_fit; // Prior scales for log-odds-of-observation coefficients
  vector[K] prior_mean_beta_fit; // Prior means for disease-log-rate coefficients
  vector[K] prior_mean_gamma_fit; // Prior means for log-odds-of-observation coefficients
  vector<lower=0>[J] prior_scales_log_lambda_fit; // Prior scales for disease-log-rate by category 
  vector<lower=0>[J] prior_scales_eta_fit; // Prior scales for log-odds-of-observation by category
  vector[J] prior_mean_log_lambda_fit;// Prior means for disease-log-rate by category 
  vector[J] prior_mean_eta_fit;// Prior means for log-odds-of-observation by category

  int<lower=0> y_miss[N]; // Number of cases missing category information by stratum
  int<lower=0> y_obs[N,J]; // Number of cases observed by stratum and cateogry
}
transformed data {
  real tot_pop;
  vector[J] pop_by_cat;
  for (j in 1:J) {
    pop_by_cat[j] = sum(E[,j]);
  }
  tot_pop = sum(pop_by_cat);
}
parameters {
  vector[J] log_lambda;
  vector[K] beta;
  // Bernoulli parameters
  vector[J] eta;
  vector[K] gamma;
}
model {
  matrix[N,J] mu_pois;
  matrix[N,J] mu_bern;
  // Poisson parameters
  beta ~ normal(prior_mean_beta_fit, prior_scales_beta_fit);
  log_lambda ~ normal(prior_mean_log_lambda_fit, prior_scales_log_lambda_fit);

  // Binomial parameters
  eta ~ normal(prior_mean_eta_fit, prior_scales_eta_fit);
  gamma ~ normal(prior_mean_gamma_fit, prior_scales_gamma_fit);
  for (j in 1:J) {
    mu_pois[,j] = X * beta + log_lambda[j];
    for (n in 1:N)
      if (E[n,j] > 0)
        y_obs[n,j] ~ poisson_log(log(E[n,j]) + mu_pois[n,j]);
  }
  for (j in 1:J) {
    mu_bern[,j] = X * gamma + eta[j];
    for (n in 1:N)
      y_obs[n,j] ~ binomial_logit(y_obs[n,j],mu_bern[n,j]);
  }
  for (n in 1:N) {
    if (y_miss[n] > 0) {
      target += y_miss[n] *
        log(dot_product(E[n,], exp(mu_pois[n,]) .* (1 - inv_logit(mu_bern[n,]))));
    }
  }
}
// generated quantities {
//   int y_obs[N,J];
//   int y_tot[N];
//   int y_miss[N];
//   int N_miss;
//   int idx_miss[N] = rep_array(0,N);
//   int miss_count;
//   real miss_prop;
//   real incidence;
//   vector[J] incidence_by_cat;
//   {
//     int y_latent[N,J];
//     matrix[N,J] mu_pois;
//     matrix[N,J] mu_bern;
//     for (j in 1:J) { 
//       mu_pois[,j] = X * beta + log_lambda[j];
//       for (n in 1:N)
//         if (E[n,j] > 0)
//           y_latent[n,j] = poisson_safe_rng(E[n,j] * exp(mu_pois[n,j]));
//         else
//           y_latent[n,j] = 0;
//     }
//     for (j in 1:J) {
//       mu_bern[,j] = X * gamma + eta[j];
//       for (n in 1:N)
//         y_obs[n,j] = binomial_rng(y_latent[n,j],inv_logit(mu_bern[n,j]));
//     }
//     for (n in 1:N)
//       y_tot[n] = sum(y_latent[n,]);
//     for (n in 1:N)
//       y_miss[n] = y_tot[n] - sum(y_obs[n,]);
//     {
//       int idx = 1;
//       for (n in 1:N)
//         if (y_miss[n] > 0) {
//           idx_miss[idx] = n;
//           idx = idx + 1;
//         }
//       N_miss = idx - 1;
//       miss_count = 0;
//       for (j in 1:J) {
//         miss_count += sum(y_latent[,j]) - sum(y_obs[,j]);
//         incidence_by_cat[j] = sum(y_latent[,j]) / pop_by_cat[j];
//       }
//       miss_prop = miss_count / (1.0 * sum(y_tot));
//       incidence = (1.0 * sum(y_tot)) / tot_pop;
//     }
//   }
// }

