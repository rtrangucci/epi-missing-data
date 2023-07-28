functions {
  // compensate for numerical over/underflow
  int poisson_log_safe_rng(real eta) {
    real eta2 = (eta < 20.79) ? eta : 20.79;
    return poisson_log_rng(eta2);
  }
  // following Carpenter on forums
  real binomial_pois_lpmf(int y_obs, real log_theta, real logit_p) {
    return binomial_logit_lpmf(y_obs | y_obs, logit_p)
           + poisson_log_lpmf(y_obs | log_theta);
  }
  // different than appendix I
  real miss_lpmf(int n_miss, vector log_theta_j, vector logit_p,
                 array[] real E) {
    int J = rows(logit_p);
    
    vector[J] filt_lambda = (1 - inv_logit(logit_p)) .* exp(log_theta_j);
    return n_miss * log(dot_product(to_vector(E), filt_lambda));
  }
  // structure of y: First 4 elements:
  // {
  //  # of race/ethnicity categories,
  //  # of observations per geographic unit (18 age x sex categories),
  //  # of data points per observation (J counts of cases by race/ethnicity,
  //                       1 count of cases without race/ethnicity info,
  //                       1 age-sex category),
  //  Dimension of predictors
  // }
  vector log_lik(vector globals, vector locals, array[] real E, array[] int y) {
    real ll = 0;
    int J = y[1];
    int obs_per_geo = y[2];
    int data_per_obs = y[3];
    int K = y[4];
    int tot_pred = 2 * K;
    for (i in 1 : obs_per_geo) {
      int start = 4 + (i - 1) * data_per_obs;
      int end = 4 + i * data_per_obs;
      int E_start = (i - 1) * (J + K) + 1;
      int E_end = E_start + J - 1;
      int pred_start = E_end + 1;
      int pred_end = pred_start + K - 1;
      array[J] real E_ig = E[E_start : E_end];
      vector[K] z_ig = to_vector(E[pred_start : pred_end]);
      array[J] int y_i = y[(start + 1) : (start + J)];
      int n_miss = y[start + J + 1];
      int age_sex_idx = y[start + J + 2];
      int locals_start = 1;
      int locals_end = K;
      int locals_start_p = K + 1;
      int locals_end_p = 2 * K;
      vector[J] logit_p;
      vector[J] log_rate;
      for (j in 1 : J) {
        log_rate[j] = dot_product(locals[locals_start : locals_end], z_ig)
                        + locals[tot_pred + j];
        logit_p[j] = dot_product(locals[locals_start_p : locals_end_p], z_ig)
                     + locals[tot_pred + J + j];
        if (abs(E_ig[j]) > 1e-16) {
          ll += binomial_pois_lpmf(y_i[j] | log_rate[j] + log(E_ig[j]), logit_p[j]);
        }
      }
      if (n_miss > 0) {
        ll += miss_lpmf(n_miss | log_rate, logit_p, E_ig);
      }
    }
    return [ll]';
  }
}
data {
  int<lower=0> N;
  int<lower=0> J;
  array[N, J] int y;
  array[N, J] real E;
  array[N] int<lower=0> n_miss;

  int<lower=1> K;
  int<lower=1> N_age;
  int<lower=1> N_age_sex;
  matrix[N, K] Z;
  array[N] int<lower=1, upper=N_age_sex> age_sex_idx;

  array[N] int<lower=1, upper=2> sex_idx;
  array[N] int<lower=1, upper=N_age> age_idx;

  int<lower=1> N_geo;
  array[N] int<lower=1, upper=N_geo> geo_idx;
  array[N] int<lower=1, upper=N_age_sex> obs_per_geo;
  row_vector[K] X_means;
  vector<lower=0>[K] prior_scales_alpha_beta;
  vector<lower=0>[K] prior_scales_alpha_gamma;
  vector[K] prior_mean_alpha_gamma;
  vector<lower=0>[J] prior_scales_alpha_lambda;
  vector<lower=0>[J] prior_scales_alpha_eta;
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
  // end hierarchical variance parameters
  
  vector[K] prior_mean_alpha_beta;
  vector[J] prior_mean_alpha_lambda;
  
  int<lower=1> N_county;
  array[N] int<lower=1, upper=N_county> county_idx;
  array[N_geo] int<lower=1, upper=N_county> county_idx_by_geo;
}
transformed data {
  vector<lower=0>[K] shared_scale = rep_vector(1, K);
  vector<lower=0>[J] shared_scale_race = rep_vector(1, J);
  vector<lower=0>[J] sd_geo_race_scale = rep_vector(1, J);
  vector<lower=0>[K] sd_geo_age_sex_scale = rep_vector(1, K);
  vector<lower=0>[K] t_shared_prior_scale =
               prior_scales_alpha_beta ./ shared_scale;
  vector<lower=0>[J] t_shared_race_prior_scale =
               prior_scales_alpha_lambda ./ shared_scale_race;
  vector<lower=0>[K] t_sd_geo_age_sex_prior_scale =
               prior_scales_sigma_beta ./ sd_geo_age_sex_scale;
  vector<lower=0>[J] t_sd_geo_race_prior_scale =
               prior_scales_sigma_lambda ./ sd_geo_race_scale;
  array[N_age_sex] real age_sex_denom = rep_array(0.0, N_age_sex);
  array[J] real race_denom = rep_array(0.0, J);
  array[N_geo] real geo_denom = rep_array(0.0, N_geo);

  // J races per geo x age x sex cell, 1 observations per race
  // 1 observation per geo x age x sex cell (n_miss),
  // 1 indices per geo-obs (age_sex_idx)
  int<lower=1> data_per_obs = J + 1 + 1;

  // 4 meta-data integers (J, obs_p_geo, data_per_obs, N_age_sex)
  array[N_geo, max(obs_per_geo) * data_per_obs + 4] int geo_data =
      rep_array(0, N_geo, max(obs_per_geo) * data_per_obs + 4);

  // J races per geo x age x sex population counts
  // K predictors per geo x age x sex
  array[N_geo, max(obs_per_geo) * (J + K)] real E_geo =
      rep_array(0.0, N_geo, max(obs_per_geo) * (J + K));
  {
    /*
      Loop through the data and build the array for observations by geographic unit of analysis
      Structure of elements of geo_data array:
      {
      First 4 elements:
        {
          # of race/ethnicity categories (J),
          # of observations per geographic unit (18 age x sex categories),
          # of data points per observation (J counts of cases by race/ethnicity,
                              1 count of cases without race/ethnicity info,
                              1 age-sex category),
          Dimension of predictors

          }
          For each age-sex category within a geographic unit:
          (Count of cases observed with race 1, Count of cases observed with race 2, \dots,
           Count of cases observed with race J, Count of cases missing race, index for age-sex category)
        }
       Structure of the elements of E_geo: For each age-sex category within geographic unit
       (E_ig, z_ig): Vector of population counts by race, vector predictors
    */
    array[N_geo] int idx_ind = rep_array(1, N_geo);
    for (n in 1 : N) {
      int idx = geo_idx[n];
      int n_obs = idx_ind[idx];
      int start = 4 + (n_obs - 1) * data_per_obs;
      int end = 4 + n_obs * data_per_obs;
      int E_start = (n_obs - 1) * (J + K) + 1;
      int E_end = E_start + J - 1;
      int pred_start = E_end + 1;
      int pred_end = pred_start + K - 1;
      geo_data[idx, 1 : 4] = {J, obs_per_geo[n], data_per_obs, K};
      for (j in 1 : J) {
        geo_data[idx, start + j] = y[n, j];
      }
      geo_data[idx, start + J + 1 : end] = {n_miss[n], age_sex_idx[n]};
      E_geo[idx, E_start : E_end] = to_array_1d(E[n,  : ]);
      E_geo[idx, pred_start : pred_end] = to_array_1d(Z[n,  : ]);
      idx_ind[idx] += 1;
    }
  }
  for (n in 1 : N) {
    age_sex_denom[age_sex_idx[n]] += sum(E[n,  : ]);
    geo_denom[geo_idx[n]] += sum(E[n,  : ]);
    for (j in 1 : J) {
      race_denom[j] += E[n, j];
    }
  }
}
parameters {
  vector[K] alpha_beta_raw;
  vector<lower=0>[K] sigma_beta;
  matrix[N_geo, K] beta_raw;
  vector[K] alpha_gamma;
  vector<lower=0>[K] sigma_gamma;
  matrix[N_geo, K] gamma_raw;
  vector[J] alpha_lambda_raw;
  vector<lower=0>[J] sigma_lambda;
  matrix[N_geo, J] log_lambda_raw;
  vector[J] alpha_eta;
  vector<lower=0>[J] sigma_eta;
  matrix[N_geo, J] eta_raw;
}
transformed parameters {
  vector[K] alpha_beta = prior_mean_alpha_beta + alpha_beta_raw .* shared_scale;
  vector[J] alpha_lambda = prior_mean_alpha_lambda
                           + alpha_lambda_raw .* shared_scale_race;
  matrix[N_geo, J] log_lambda;
  matrix[N_geo, J] eta;
  matrix[N_geo, K] beta;
  matrix[N_geo, K] gamma;
  for (j in 1 : J) {
      log_lambda[ : , j] =
        alpha_lambda[j] + sigma_lambda[j] * log_lambda_raw[ : , j];
  }
  for (j in 1 : J) {
    eta[ : , j] =
      alpha_eta[j] + sigma_eta[j] * eta_raw[ : , j];
  }
  for (k in 1 : K) {
    beta[ : , k] =
      alpha_beta[k] + sigma_beta[k] * beta_raw[ : , k];
  }
  for (k in 1 : K) {
    gamma[ : , k] =
      alpha_gamma[k] + sigma_gamma[k] * gamma_raw[ : , k];
  }
}
model {
  array[N_geo] vector[2 * K + 2 * J] theta;
  vector[0] dummy_vec;
  for (n in 1 : N_geo) {
    theta[n] = append_col(append_col(beta[n,  : ],
                                         gamma[n,  : ]),
                              append_col(log_lambda[n,  : ],
                                         eta[n,  : ]))';
  }
  to_vector(log_lambda_raw) ~ std_normal();
  to_vector(eta_raw) ~ std_normal();
  to_vector(beta_raw) ~ std_normal();
  to_vector(gamma_raw) ~ std_normal();
  sigma_lambda ~ normal(prior_mean_sigma_lambda,
                           prior_scales_sigma_lambda);
  sigma_eta ~ normal(prior_mean_sigma_eta, prior_scales_sigma_eta);
  sigma_beta ~ normal(prior_mean_sigma_beta, prior_scales_sigma_beta);
  sigma_gamma ~ normal(prior_mean_sigma_gamma, prior_scales_sigma_gamma);
  alpha_beta_raw ~ normal(0, t_shared_prior_scale);
  alpha_lambda_raw ~ normal(0, t_shared_race_prior_scale);
  alpha_gamma ~ normal(prior_mean_alpha_gamma, prior_scales_alpha_gamma);
  alpha_eta ~ normal(prior_mean_alpha_eta, prior_scales_alpha_eta);
  target += sum(map_rect(log_lik, dummy_vec, theta, E_geo, geo_data));
}
generated quantities {
  array[N_age_sex] real age_sex_inc = rep_array(0.0, N_age_sex);
  array[J] real standard_incidence_by_race = rep_array(0.0, J);
  array[J] real incidence_by_race = rep_array(0.0, J);
  array[J] real filtered_incidence_by_race = rep_array(0.0, J);
  {
    array[N, J] int y_obs_rep;
    array[N, J] int y_latent_obs_rep;
    array[N, J] real rate;
    array[N_county] int dummy_arr = rep_array(0, N_county);
    array[N, J] real filtered_rate = rep_array(0.0, N, J);
    for (n in 1 : N) {
      real zbeta = Z[n,  : ] * beta[geo_idx[n],  : ]';
      real zgamma = Z[n,  : ] * gamma[geo_idx[n],  : ]';
      for (j in 1 : J) {
        real log_rate_j = zbeta + log_lambda[geo_idx[n], j];
        real p_j = inv_logit(zgamma + eta[geo_idx[n], j]);
        int y_latent_rep;
        rate[n, j] = exp(log_rate_j);
        filtered_rate[n, j] = p_j * rate[n, j];
        if (E[n, j] > 0) {
          y_latent_rep = poisson_log_safe_rng(log_rate_j + log(E[n, j]));
        } else {
          y_latent_rep = 0;
        }
        y_latent_obs_rep[n, j] = y_latent_rep;
        if (y_latent_obs_rep[n, j] > 0) {
          y_obs_rep[n, j] = binomial_rng(y_latent_rep, p_j);
        } else {
          y_obs_rep[n, j] = 0;
        }
        age_sex_inc[age_sex_idx[n]] +=
            rate[n, j] * E[n, j] / age_sex_denom[age_sex_idx[n]];
        incidence_by_race[j] += rate[n, j] * E[n, j] / race_denom[j];
        filtered_incidence_by_race[j] +=
            filtered_rate[n, j] * E[n, j] / race_denom[j];
      }
    }
    for (n in 1 : N) {
      for (j in 1 : J) {
        real s_inc = E[n, j] * age_sex_inc[age_sex_idx[n]];
        standard_incidence_by_race[j] += s_inc / race_denom[j];
      }
    }
  }
}

