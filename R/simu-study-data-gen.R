library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)
library(posterior)

gen_fixed_data <- function(obs_prop) {
  puma_inc <- read_csv("data/nhgis0005_csv/nhgis0005_ds244_20195_2019_puma.csv") %>%
    filter(STATE == "Michigan")

  all <- read_csv("data/base_population_data.csv") %>%
    filter(
      phase == "2020-03-01"
    ) %>%
    mutate(
      age_fac = as.factor(coarse_bins),
      gender_fac = as.factor(gender),
      age_sex_fac = interaction(age_fac, gender_fac),
      age_sex_idx = as.integer(age_sex_fac),
      gender_idx = as.integer(gender_fac),
      age_idx = as.integer(age_fac)
    ) %>%
    filter(county %in% c("wayne")) %>%
    #   ) %>% filter(county %in% c('oakland','macomb')) %>%
    ## ) %>% filter(county %in% c("wayne")) %>%
    left_join(
      puma_inc[, c("PUMA5A", "ALW1E001")],
      by = c("PUMA" = "PUMA5A")
    ) %>%
    tidyr::replace_na(list(beds = 0)) %>%
    mutate(
      geo_ind = PUMA,
      geo_idx = as.integer(as.factor(PUMA)),
      med_inc_10k = ALW1E001 / 1e4,
      beds_100 = beds / 1e2
    ) %>%
    rename(
      pop_black = pop_scale_black,
      pop_white = pop_scale_white,
      pop_latino = pop_scale_latino,
      pop_asian = pop_scale_asian,
      pop_native = pop_scale_native,
      pop_other = pop_scale_other
    ) %>%
    mutate(
      county_idx = as.integer(as.factor(county))
    )

  obs_p_geo_by_geo <- all %>%
    group_by(geo_ind) %>%
    summarise(obs_p_geo = n())
  all <- all %>% left_join(obs_p_geo_by_geo, by = "geo_ind")

  set.seed(123)
  N_obs <- data.matrix(all[, c(
    "pop_black", "pop_latino", "pop_other",
    "pop_asian", "pop_native", "pop_white"
  )])
  N_obs[, "pop_other"] <- N_obs[, "pop_other"] +
    N_obs[, "pop_native"] +
    N_obs[, "pop_asian"] 
  N_obs <- N_obs[, c("pop_black", "pop_latino", "pop_other", "pop_white")]
  N_obs <- N_obs * 365 / as.integer(lubridate::ymd("2020-07-01") - lubridate::ymd("2020-03-01"))
  pop_tots <- colSums(N_obs)
  tot_pop <- sum(pop_tots)
  prop_obs_ratios <- c(0.75 / 0.9, 1, 0.6 / 0.9, 1)
  weights <- sum(prop_obs_ratios * pop_tots / tot_pop)
  prop_obs_by_group <- obs_prop / weights * prop_obs_ratios

  X <- model.matrix(~ 1 + gender_fac + age_fac,
                    all,
                    contrasts = list(age_fac = "contr.sum")
    )[, -1] %>% as.matrix()
  K <- ncol(X)
  X_cent <- scale(X, scale = FALSE)

  J <- ncol(N_obs)
  N <- nrow(N_obs)
  geo_map <- all %>%
    select(PUMA, geo_idx, county_idx, beds_100, med_inc_10k) %>%
    rename(beds = beds_100, inc = med_inc_10k) %>%
    mutate(
      beds = beds - mean(beds),
      inc = inc - mean(inc)
    ) %>%
      unique() %>%
    arrange(geo_idx)
  geo_pred <- geo_map %>%
    select(
      beds, inc
    ) %>%
    as.matrix
  stan_dat <-
    list(
      J = J,
      N = N,
      E = N_obs,
      K = K,
      X = X_cent,
      Q = X_cent,
      geo_idx = all$geo_idx,
      N_geo = length(unique(all$geo_idx)),
      prior_scales_alpha_beta = c(1, rep(1, 8)),
      prior_scales_alpha_gamma = c(1, rep(1, 8)),
      prior_mean_alpha_beta = c(0, c(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)),
      prior_mean_alpha_gamma = c(0, rep(0, 8)),
      prior_scales_alpha_lambda = rep(1, J),
      prior_scales_alpha_eta = rep(1.0, J),
      prior_mean_alpha_lambda = rep(-5.0, J),
      prior_mean_alpha_eta = rep(2, J), ## Hierarchical variance pars follow
      prior_scales_sigma_beta = c(1, rep(1, 8)),
      prior_scales_sigma_gamma = c(1, rep(1, 8)),
      prior_mean_sigma_beta = c(0, rep(0, 8)),
      prior_mean_sigma_gamma = c(0, rep(0, 8)),
      prior_scales_sigma_lambda = rep(0.5, J),
      prior_scales_sigma_eta = rep(1, J),
      prior_mean_sigma_lambda = rep(0, J),
      prior_mean_sigma_eta = rep(0, J),
      N_county = length(unique(all$county_idx)),
      county_idx = all$county_idx,
      county_idx_by_geo = geo_map$county_idx,
      N_age = 9,
      N_sex = 2,
      age_idx = all$age_idx,
      sex_idx = all$gender_idx,
      N_age_sex = 18,
      age_sex_idx = all$age_sex_idx,
      geo_idx = all$geo_idx,
      N_geo = length(unique(all$geo_ind)),
      obs_per_geo = all$obs_p_geo,
      X_means = colMeans(X),
      alpha_beta_fixed_old = c(-0.5, c(-2, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.0)),
      alpha_beta_fixed = c(-0.05, c(-2.5, -2.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0)),
      alpha_gamma_fixed_old = c(-0.5, c(1, 0.5, 0, -0.5, -0.5, -0.5, 0, 1.0)),
      alpha_gamma_fixed = c(-0.1, c(-0.3, -0.3, -0.2, -0.2, -0.2, -0.1, 0.1, 0.4)),
      alpha_lambda_fixed = rep(-4,J),
      alpha_eta_fixed = qlogis(prop_obs_by_group),
      sigma_beta_fixed = c(0.5, rep(0.5, 8)),
      sigma_gamma_fixed = c(0.3, rep(0.3, 8)),
      sigma_eta_fixed = rep(0.3, J),
      sigma_lambda_fixed = rep(0.5, J)
    )
  return(
    list(
      stan_data = stan_dat,
      all = all,
      E_all = N_obs
    )
  )
}



find_tract <- function(tbl) {
  gpd <- tbl %>%
    group_by(tract) %>%
    summarise(
      pop_black = sum(pop_black),
      pop_white = sum(pop_white),
      pop_latino = sum(pop_latino),
      pop_other = sum(pop_other),
      pop_asian = sum(pop_asian),
      pop_native = sum(pop_native)
    ) %>%
    mutate(
      z_black = scale(pop_black),
      z_white = scale(pop_white),
      z_asian = scale(pop_asian),
      z_latino = scale(pop_latino),
      z_native = scale(pop_native),
      z_other = scale(pop_other)
    )
  return(gpd)
}


gen_data_random <- function(simu_model, S, fixed_data) {
  fake_samps <- simu_model$sample(
    fixed_param = T,
    data = fixed_data,
    iter_warmup = 0,
    iter_sampling = S,
    chains = 1,
    parallel_chains = 1,
    seed = 407009584
  )
  return(fake_samps)
}

simu_curate <- function(simu_data, fixed_data) {
  N <- fixed_data$stan_data$N
  J <- fixed_data$stan_data$J
  pars <- posterior::as_draws_df(
    simu_data$draws(
                variables = c(
                  "alpha_lambda", "alpha_beta",
                  "alpha_eta", "alpha_gamma",
                  "sigma_lambda", "sigma_beta",
                  "sigma_eta", "sigma_gamma",
                  "incidence_by_race",
                  "filtered_incidence_by_race",
                  "standard_incidence_by_race",
                  "age_sex_inc"
                )
    )
  )
  y_obs <- posterior::as_draws_df(
    simu_data$draws(
      variables = c("y_obs")
    )
  )
  y_miss <- posterior::as_draws_df(
    simu_data$draws(
      variables = c("y_miss")
    )
  )
  S <- nrow(pars)
  simu_list <- list()
  for (s in 1:S) {
    stan_dat <- fixed_data$stan_data
    y_obs_mat <- matrix(NA_integer_, N, J)
    y_obs_s <- y_obs[s, , drop = TRUE]
    y_miss_s <- y_miss[s, , drop = TRUE]
    miss_nm_sel <- paste0("y_miss[", 1:N, "]")
    stan_dat$n_miss <- y_miss_s[, miss_nm_sel] %>%
      as.data.frame() %>%
      t() %>%
      drop()
    for (j in 1:J) {
      obs_nm_sel <- paste0("y_obs[", 1:N, ",", j, "]")
      y_obs_mat[, j] <- y_obs_s[obs_nm_sel] %>%
        as.data.frame() %>%
        t() %>%
        drop()
    }
    stan_dat$y <- y_obs_mat
    simu_list[[s]] <- list(
      stan_data = stan_dat,
      true_pars = pars[s, , drop = TRUE]
    )
  }
  return(simu_list)
}

fit_mod <- function(model, data,
                    vars_to_pull) {
  fit <- model$sample(
    data = data$stan_data,
    iter_warmup = 100,
    iter_sampling = 100,
    refresh = 3000,
    chains = 4, parallel_chains = 4,
    adapt_delta = 0.9
  )
  pars <- posterior::as_draws_df(
    fit$draws(
                variables = vars_to_pull
    )
  )
  true <- data$true_pars
  samp_pars <- fit$sampler_diagnostics() %>%
    as_draws_df()
  return(
    list(
      samps = pars,
      true = true,
      diags = samp_pars,
      metadata = fit$metadata(),
      sum_fit = fit$summary()
    )
  )
}

fit_ensemble <- function(model, datasets, vars_to_pull) {
  fits <- list()
  S <- length(datasets)
  for (i in 1:1) {
    if (i %% 10 == 0) {
      print(i)
    }
    fits[[i]] <- fit_mod(model, datasets[[i]], vars_to_pull)
  }
  return(fits)
}

cover_quantiles <- function(draws, true, p) {
  lower <- (1 - p) / 2
  upper <- 1 - lower
  quants <- apply(draws, 2, quantile, c(lower, upper))
  n_pars <- ncol(draws)
  cover <- rep(NA_integer_, n_pars)
  for (idx_par in 1:n_pars) {
    cover_i <- 0
    if (true[idx_par] >= quants[1, idx_par] &&
      true[idx_par] <= quants[2, idx_par]) {
      cover_i <- 1
    }
    cover[idx_par] <- cover_i
  }
  return(cover)
}

z_score_pars <- function(draws, true, true_prior_var) {
  post_means <- colMeans(draws)
  post_vars <- apply(draws, 2, var)
  zs <- (post_means - true) / sqrt(post_vars)
  shrink <- 1 - post_vars / true_prior_var
  return(list(
    z_score = zs,
    shrink = shrink
  ))
}

parse_fits <- function(ensemble_fits, fixed_data) {
  alpha_lambda_prior_var <- fixed_data$stan_data$prior_scales_alpha_lambda^2
  alpha_beta_prior_var <- fixed_data$stan_data$prior_scales_alpha_beta^2
  alpha_eta_prior_var <- fixed_data$stan_data$prior_scales_alpha_eta^2
  alpha_gamma_prior_var <- fixed_data$stan_data$prior_scales_alpha_gamma^2

  scale_factor <- 1 - 2 / pi
  sigma_lambda_prior_var <- fixed_data$stan_data$prior_scales_sigma_lambda^2 *
    scale_factor
  sigma_beta_prior_var <- fixed_data$stan_data$prior_scales_sigma_beta^2 *
    scale_factor
  sigma_eta_prior_var <- fixed_data$stan_data$prior_scales_sigma_eta^2 *
    scale_factor
  sigma_gamma_prior_var <- fixed_data$stan_data$prior_scales_sigma_gamma^2 *
    scale_factor
  prior_vars <- c(
    alpha_lambda_prior_var,
    alpha_beta_prior_var,
    alpha_eta_prior_var,
    alpha_gamma_prior_var,
    sigma_lambda_prior_var,
    sigma_beta_prior_var,
    sigma_eta_prior_var,
    sigma_gamma_prior_var
  )
  S <- length(ensemble_fits)
  n_vars <- length(prior_vars)
  z_mat <- matrix(NA_real_, S, n_vars)
  diags <- rep(NA_character_, S)

  colnames(z_mat) <- ensemble_fits[[1]]$samps %>%
    select(
      -`.chain`,
      -`.iteration`,
      -`.draw`
    ) %>%
    names()
  rank_mat <- shrink_mat <- cover_mat_80 <- cover_mat_50 <- z_mat
  for (idx_fit in seq_along(ensemble_fits)) {
    fit <- ensemble_fits[[idx_fit]]
    samps <- fit$samps %>%
      subset_draws(
        iteration = seq(1, 1000, by = 4)
      ) %>%
      select(
        -`.chain`,
        -`.iteration`,
        -`.draw`
      ) %>%
      as.data.frame()
    true <- fit$true %>%
      select(
        -`.chain`,
        -`.iteration`,
        -`.draw`
      ) %>%
      as.data.frame() %>%
      t() %>%
      drop()
    stopifnot(all(names(samps) == names(true)))
    zs <- z_score_pars(samps, true, prior_vars)
    cover_50 <- cover_quantiles(samps, true, 0.5)
    cover_80 <- cover_quantiles(samps, true, 0.8)
    z_mat[idx_fit, ] <- zs$z_score
    shrink_mat[idx_fit, ] <- zs$shrink
    cover_mat_50[idx_fit, ] <- cover_50
    cover_mat_80[idx_fit, ] <- cover_80
    ranks <- sapply(
      seq(1, ncol(samps), by = 1),
      function(i) sum(true[i] < samps[, i])
    )
    rank_mat[idx_fit, ] <- ranks
    diags_s <- fit$diags
    div <- any(diags_s$divergent__ == 1)
    treedepth <- any(diags_s$treedepth__ == 10)
    problem <- "no treedepth"
    if (treedepth) {
      problem <- "treedepth"
    }
    if (div) {
      problem <- paste0(problem, ", divergence")
    } else {
      problem <- paste0(problem, ", no divergence")
    }
    diags[idx_fit] <- problem
  }
  return(
    list(
      cover_50 = cover_mat_50,
      cover_80 = cover_mat_80,
      shrink = shrink_mat,
      z_scores = z_mat,
      ranks = rank_mat,
      diagnostics = diags
    )
  )
}

make_regr_shrink_zscore_plot <- function(data, post_data) {
  pdf("posterior-shrinkage-regr.pdf")
  K <- data$K
  J <- data$J
  for (k in 1:K) {
    nm <- paste0("beta[",k,"]")
    plot(post_data$shrink[, nm],
      post_data$z_scores[, nm],
      xlim = c(0, 1),
      ylim = c(-4, 4),
      main = nm,
      xlab = "Variance shrinkage",
      ylab = "z-score",
    )
    abline(h = 2)
    abline(h = -2)
  }
  for (k in 1:K) {
    nm <- paste0("theta[", k, "]")
    plot(post_data$shrink[, nm],
      post_data$z_scores[, nm],
      xlim = c(0, 1),
      ylim = c(-4, 4),
      main = nm,
      xlab = "Variance shrinkage",
      ylab = "z-score",
    )
    abline(h = 2)
    abline(h = -2)
  }
  dev.off()
}

make_rate_shrink_zscore_plot <- function(data, post_data) {
  pdf("posterior-shrinkage-rates.pdf")
  K <- data$K
  J <- data$J
  for (k in 1:J) {
    nm <- paste0("alpha_lambda[", k, "]")
    plot(post_data$shrink[, nm],
      post_data$z_scores[, nm],
      xlim = c(0, 1),
      ylim = c(-4, 4),
      main = nm,
      xlab = "Variance shrinkage",
      ylab = "z-score",
    )
    abline(h = 2)
    abline(h = -2)
  }
  for (k in 1:J) {
    nm <- paste0("alpha_eta[", k, "]")
    plot(post_data$shrink[, nm],
      post_data$z_scores[, nm],
      xlim = c(0, 1),
      ylim = c(-4, 4),
      main = nm,
      xlab = "Variance shrinkage",
      ylab = "z-score",
    )
    abline(h = 2)
    abline(h = -2)
  }
  dev.off()
}

beta_solver <- function(X, y) {
  qr_X <- qr(X)
  Q <- qr.Q(qr_X)
  R <- qr.R(qr_X)
  if (qr_X$rank < ncol(X)) {
    return(rep(0, ncol(X)))
  }
  beta_hat <- backsolve(R, t(Q) %*% y)
  return(beta_hat)
}

tract_id_vec_1 <- function(E, Z, add_col = 0) {
  J <- ncol(E)
  K <- ncol(Z)
  I <- nrow(E)
  target <- E[, add_col + 1]
  obj <- target - diag(rowSums(E)) %*% Z %*% rep(1, ncol(Z))
  obj <- sum(obj^2)
  s_max <- 1000
  lambda <- rep(1, J)
  beta_hat <- rep(1, K)
  last_obj <- obj
  for (s in 1:s_max) {
    if (add_col > 0) {
      if (any(lambda == 0)) {
        lambda[lambda == 0] <- 1e-8
      }
      Z_update <- cbind(diag(as.vector(E %*% lambda)) %*% Z,
                        E[, 1:add_col])
    } else {
      Z_update <- diag(as.vector(E %*% lambda)) %*% Z
    }
    beta_hat <- beta_solver(Z_update, target)
    E_update <- sweep(E, 1, as.vector(Z %*% beta_hat[1:K]), FUN = "*")
    ret <- glmnet::glmnet(E_update, target,lambda = 0, lower.limits = 0,
                          intercept = FALSE)
    lambda <- coef(ret) %>% as.vector %>% tail(-1)
    pred <- diag(as.vector(E %*% lambda)) %*% Z %*% beta_hat[1:K]
    if (add_col > 0) {
      pred <- pred + E[, 1:add_col] %*% tail(beta_hat, -K)
    }
    obj <- sum((target - pred)^2)
    if (abs(obj - last_obj) < 1e-8) {
      break
    }
    last_obj <- obj
  }
  return(list(
    obj = obj,
    beta = beta_hat,
    lambda = lambda
  ))
}

tract_id <- function(E, Z) {
  if (ncol(E) + ncol(Z) > nrow(E)) {
    return(list())
  }
  id_res <- list()
  id_1 <- tract_id_vec_1(E, Z)
  id_res[[1]] <- id_1
  if (id_1$obj > 1e-8) {
    n_rem <- ncol(E) - 1
    for (col_i in 1:n_rem) {
      id_i <- tract_id_vec_1(E, Z, col_i)
      id_res[[col_i + 1]] <- id_i
    }
    return(id_res)
  }
  return(id_res)
}

system.time(tt <- tract_id(
  fixed_data$stan_data$E,
  fixed_data$stan_data$X
)
)
tt <- sim_coefs(
  fixed_data$stan_data$E,
  fixed_data$stan_data$X
)

## fixed_data_simple <- gen_fixed_data_simple(6)
sim_model <- cmdstan_model("stan/simu-hier-regr-model-fixed-par.stan")
sim_prior_model <- cmdstan_model("stan/simu-hier-regr-model-w-inc.stan")
prior_fixed_data <- gen_fixed_data(0.9)
prior_pred <- gen_data_random(sim_prior_model, 1000, prior_fixed_data$stan_data)
prior_summary <- prior_pred$summary(variables = c(
  "incidence",
  "miss_prop",
  "incidence_by_race",
  "filtered_incidence_by_race",
  "standard_incidence_by_race",
  "age_sex_inc",
  "alpha_lambda"
))
sds <- prior_summary %>%
  filter(grepl("incidence_by_race",variable) | grepl("age_sex_inc",variable)) %>%
  select(variable,sd)
vars_to_save <- sds$sd
names(vars_to_save) <- sds$variable
saveRDS(vars_to_save, file = "great-lakes-simu-study/prior_vars_simu.RDS")
prop_miss_sim <- prior_pred$draws(
  variables = c("miss_prop", "incidence"),
  format = "draws_matrix"
)
vars_to_pull <- c(
  "alpha_lambda",
  "alpha_beta",
  "sigma_lambda",
  "sigma_beta",
  "incidence_by_race",
  "filtered_incidence_by_race",
  "standard_incidence_by_race",
  "age_sex_inc"
)
names_suffix <- function(N) {
  return(paste0("[", 1:N, "]"))
}
schedule_nms <- c("10p","20p","40p","80p")
# schedule <- c(0.91, 0.836, 0.64, 0.24)
schedule <- c(0.89, 0.79, 0.56, 0.15)
sims_sum <- list() 
for (i in seq_along(schedule_nms)) {
  nm_i <- schedule_nms[i]
  p_i <- schedule[i]
  fixed_data <- gen_fixed_data(p_i)
  fake_data <- gen_data_random(sim_model, 200, fixed_data$stan_data)
  fake_data_summary <- fake_data$summary(variables = c(
    "incidence",
    "miss_prop",
    "prop_obs_by_race",
    "obs_inc_by_race",
    "true_inc_by_race",
    "standard_incidence_by_race",
    "filtered_incidence_by_race"
  ))
  prop_obs_by_race <- fake_data_summary %>%
    filter(grepl("prop_obs_by_race",variable)) %>%
    select(variable, mean, sd) %>%
    mutate(
      gp_num = readr::parse_number(variable)
    ) %>% select(-variable) %>%
    rename(prop_obs = mean,
           sd_prop_obs = sd)
  obs_inc_by_race <- fake_data_summary %>%
    filter(grepl("obs_inc_by_race", variable)) %>%
    select(mean, variable, sd) %>%
      mutate(
        gp_num = readr::parse_number(variable)
      ) %>%
    select(-variable) %>%
    rename(obs_inc = mean,
           sd_obs_inc = sd)
  true_inc_by_race <- fake_data_summary %>%
    filter(grepl("true_inc_by_race", variable)) %>%
    select(mean, variable, sd) %>%
      mutate(
        gp_num = readr::parse_number(variable)
      ) %>%
    select(-variable) %>%
    rename(true_inc = mean,
           sd_true_inc = sd)
  df_sim_sum <- prop_obs_by_race %>%
    left_join(obs_inc_by_race, by = "gp_num") %>%
    left_join(true_inc_by_race, by = "gp_num")
  df_sim_sum$prop_sim <- nm_i
  sims_sum[[i]] <- df_sim_sum

  tt <- simu_curate(fake_data, fixed_data)
  saveRDS(tt, file = paste0("great-lakes-simu-study/","data_",nm_i,".RDS"))
}
sims_sum <- bind_rows(sims_sum) 
gp_nms <- data.frame(
  group = c("Black", "Hispanic/Latino", "Other", "White"),
  gp_num = c(1, 2, 3, 4)
)
sims_sum <- sims_sum %>%
  left_join(gp_nms, by = "gp_num")
saveRDS(sims_sum, file = "great-lakes-simu-study/simulated-data-summary.RDS")
prop_miss_sim <- fake_data$draws(variables = c("miss_prop", "incidence"),
                                 format = "draws_matrix")
saveRDS(fixed_data, file = "great-lakes-simu-study/fixed_data_simu.RDS")

