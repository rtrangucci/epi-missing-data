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
  // structure of y:  4-tuple:  category, ??
  vector log_lik(vector globals, vector locals, array[] real E, array[] int y) {
    real ll = 0;
    int J = y[1];
    int obs_per_geo = y[2];
    int data_per_obs = y[3];
    int K_full = y[4];
    int tot_pred = 2 * K_full;
    for (i in 1 : obs_per_geo) {
      int start = 4 + (i - 1) * data_per_obs;
      int end = 4 + i * data_per_obs;
      int E_start = (i - 1) * (J + K_full) + 1;
      int E_end = E_start + J - 1;
      int pred_start = E_end + 1;
      int pred_end = pred_start + K_full - 1;
      array[J] real E_i = E[E_start : E_end];
      vector[K_full] Q_i = to_vector(E[pred_start : pred_end]);
      array[J] int y_i = y[(start + 1) : (start + J)];
      int n_miss = y[start + J + 1];
      int age_sex_idx = y[start + J + 2];
      int locals_start = 1;
      int locals_end = K_full;
      int locals_start_p = K_full + 1;
      int locals_end_p = 2 * K_full;
      vector[J] logit_p;
      vector[J] log_lambda;
      for (j in 1 : J) {
        log_lambda[j] = dot_product(locals[locals_start : locals_end], Q_i)
                        + locals[tot_pred + j];
        logit_p[j] = dot_product(locals[locals_start_p : locals_end_p], Q_i)
                     + locals[tot_pred + J + j];
        if (abs(E_i[j]) > 1e-16) {
          ll += binomial_pois_lpmf(y_i[j] | log_lambda[j] + log(E_i[j]), logit_p[j]);
        }
      }
      if (n_miss > 0) {
        ll += miss_lpmf(n_miss | log_lambda, logit_p, E_i);
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
  matrix[N, K] Q;
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
  real<lower=0> sd_sd_geo = 1;
  real<lower=0> sd_sd_geo_p = 1;
  real<lower=0> sd_sd_geo_age_sex = 1;
  real<lower=0> sd_sd_geo_age_sex_p = 1;
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
      E_geo[idx, pred_start : pred_end] = to_array_1d(Q[n,  : ]);
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
  vector[K] eta_shared;
  vector[J] eta_shared_race;
  vector<lower=0>[J] eta_sd_geo_race;
  vector<lower=0>[K] eta_sd_geo_age_sex;
  vector[K] eta_shared_p;
  vector<lower=0>[J] eta_sd_geo_p_race;
  vector<lower=0>[K] sigma_gamma;
  vector[J] eta_shared_p_race;
  matrix[N_geo, J] eta_geo_e_race;
  matrix[N_geo, J] eta_geo_e_p_race;
  matrix[N_geo, K] eta_geo_e_age_sex;
  matrix[N_geo, K] eta_geo_e_p_age_sex;
}
transformed parameters {
  vector[K] alpha_beta = prior_mean_alpha_beta + eta_shared .* shared_scale;
  vector[J] alpha_lambda = prior_mean_alpha_lambda
                           + eta_shared_race .* shared_scale_race;
  vector[K] alpha_gamma = eta_shared_p;
  vector[J] alpha_eta = eta_shared_p_race;
  vector[J] sigma_lambda = eta_sd_geo_race;
  vector[K] sigma_beta = eta_sd_geo_age_sex;
  vector[J] sigma_eta = eta_sd_geo_p_race;
  matrix[N_geo, J] geo_e_race;
  matrix[N_geo, J] geo_e_p_race;
  matrix[N_geo, K] geo_e_age_sex;
  matrix[N_geo, K] geo_e_p_age_sex;
  for (j in 1 : J) {
      geo_e_race[ : , j] =
          alpha_lambda[j] + sd_sd_geo * sigma_lambda[j] * eta_geo_e_race[ : , j];
  }
  for (j in 1 : J) {
    geo_e_p_race[ : , j] =
        alpha_eta[j] + sd_sd_geo_p * sigma_eta[j] * eta_geo_e_p_race[ : , j];
  }
  for (k in 1 : K) {
    geo_e_age_sex[ : , k] =
        alpha_beta[k] + sd_sd_geo_age_sex * sigma_beta[k] * eta_geo_e_age_sex[ : , k];
  }
  for (k in 1 : K) {
    geo_e_p_age_sex[ : , k] =
        alpha_gamma[k] + sd_sd_geo_age_sex_p * sigma_gamma[k] * eta_geo_e_p_age_sex[ : , k];
  }
}
model {
  array[N_geo] vector[2 * K + 2 * J] theta_geo;
  vector[0] dummy_vec;
  for (n in 1 : N_geo) {
    theta_geo[n] = append_col(append_col(geo_e_age_sex[n,  : ],
                                         geo_e_p_age_sex[n,  : ]),
                              append_col(geo_e_race[n,  : ],
                                         geo_e_p_race[n,  : ]))';
  }
  to_vector(eta_geo_e_race) ~ std_normal();
  to_vector(eta_geo_e_p_race) ~ std_normal();
  to_vector(eta_geo_e_age_sex) ~ normal(0, 1);
  to_vector(eta_geo_e_p_age_sex) ~ normal(0, 1);
  eta_sd_geo_race ~ normal(prior_mean_sigma_lambda,
                           prior_scales_sigma_lambda);
  eta_sd_geo_p_race ~ normal(prior_mean_sigma_eta, prior_scales_sigma_eta);
  eta_sd_geo_age_sex ~ normal(prior_mean_sigma_beta, prior_scales_sigma_beta);
  sigma_gamma ~ normal(prior_mean_sigma_gamma, prior_scales_sigma_gamma);
  eta_shared ~ normal(0, t_shared_prior_scale);
  eta_shared_race ~ normal(0, t_shared_race_prior_scale);
  eta_shared_p ~ normal(prior_mean_alpha_gamma, prior_scales_alpha_gamma);
  eta_shared_p_race ~ normal(prior_mean_alpha_eta, prior_scales_alpha_eta);
  target += sum(map_rect(log_lik, dummy_vec, theta_geo, E_geo, geo_data));
}
generated quantities {
  array[N_age_sex] real age_sex_inc = rep_array(0.0, N_age_sex);
  array[J] real standard_incidence_by_race = rep_array(0.0, J);
  array[J] real incidence_by_race = rep_array(0.0, J);
  array[J] real filtered_incidence_by_race = rep_array(0.0, J);
  {
    array[N, J] int y_obs_rep;
    array[N, J] int y_latent_obs_rep;
    array[N, J] real lambda;
    array[N_county] int dummy_arr = rep_array(0, N_county);
    array[N, J] real filtered_lambda = rep_array(0.0, N, J);
    for (n in 1 : N) {
      real eff = Q[n,  : ] * geo_e_age_sex[geo_idx[n],  : ]';
      real eff_p = Q[n,  : ] * geo_e_p_age_sex[geo_idx[n],  : ]';
      for (j in 1 : J) {
        real log_lambda_j = eff + geo_e_race[geo_idx[n], j];
        real p_j = inv_logit(eff_p + geo_e_p_race[geo_idx[n], j]);
        int y_latent_rep;
        lambda[n, j] = exp(log_lambda_j);
        filtered_lambda[n, j] = p_j * lambda[n, j];
        if (E[n, j] > 0) {
          y_latent_rep = poisson_log_safe_rng(log_lambda_j + log(E[n, j]));
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
            lambda[n, j] * E[n, j] / age_sex_denom[age_sex_idx[n]];
        incidence_by_race[j] += lambda[n, j] * E[n, j] / race_denom[j];
        filtered_incidence_by_race[j] +=
            filtered_lambda[n, j] * E[n, j] / race_denom[j];
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

