library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)
library(posterior)

## Generates regression design matrix and population totals
## for model, to be fixed across simulation runs
## This function defines the prior hyperparameters
gen_fixed_data <- function(J) {
  N <- 18
  if (J < 2 || J > 6) {
    stop("J must be in [2,6]")
  }
  fl <- "~/state-work/mi-covid-maps/all_tracts_mi_all_21_04_23_expand_other.csv"
  all <-
    read_csv(fl) %>%
    mutate(
      age_fac = as.factor(coarse_bins),
      gender_fac = as.factor(gender),
      age_sex_fac = interaction(age_fac, gender_fac),
      age_sex_idx = as.integer(age_sex_fac),
      gender_idx = as.integer(gender_fac),
      age_idx = as.integer(age_fac),
      tract = paste0(
        stringr::str_sub(GISJOIN, 2, 3),
        stringr::str_sub(GISJOIN, 5, 7),
        stringr::str_sub(GISJOIN, 9, 14)
      )
    ) %>%
    filter(county %in% c("wayne")) %>%
    mutate(
      tract_idx = as.integer(as.factor(tract))
    )

  set.seed(123)
  tbl <- all[all$tract == "26163565300", ]
  N_obs <- data.matrix(tbl[, c(
    "pop_black", "pop_latino", "pop_other",
    "pop_asian", "pop_native", "pop_white"
  )])

  X <- model.matrix(~ 1 + gender_fac + age_fac,
    tbl,
    ## contrasts = list(age_fac = "contr.sum")
    )[, -1] %>% as.matrix()
  ## X[1:14,1] <- 0
  ## X[15,1] <- 1
  K <- ncol(X)

  stan_dat <-
    list(
      J = J,
      N = N,
      E = N_obs[1:N, ],
      K = K,
      X = X[1:N, ],
      prior_scales_beta = c(1, rep(2, 8) / sqrt(8)),
      prior_scales_gamma = c(1, rep(2, 8)),
      prior_mean_beta = c(0, c(-2, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.0)),
      prior_mean_gamma = c(0, rep(0, 8)),
      prior_scales_log_lambda = rep(1, J),
      prior_scales_eta = rep(2, J),
      prior_mean_log_lambda = rep(-4.5, J),
      prior_mean_eta = rep(1, J)
    )
  return(
    list(
      stan_data = stan_dat
    )
  )
}

## @param simu_model pre-compiled Stan simulation model
## @param S Number of simulation runs
## @param fixed_data Static data generated from gen_fixed_data
## @return S simulated datasets from simu_model
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

## @param simu_data simulated datasets 
## @param fixed_data Static data generated from gen_fixed_data
## @return List of S simulated datasets, each element ready to pass
##         to a Stan model
simu_curate <- function(simu_data, fixed_data) {
  N <- fixed_data$stan_data$N
  J <- fixed_data$stan_data$J
  pars <- posterior::as_draws_df(
    simu_data$draws(
      variables = c("gamma", "eta", "beta", "log_lambda")
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
    stan_dat$y_miss <- y_miss_s[, miss_nm_sel] %>%
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
    true_pars <- stan_dat$y_obs <- y_obs_mat
    simu_list[[s]] <- list(
      stan_data = stan_dat,
      true_pars = pars[s, , drop = TRUE]
    )
  }
  return(simu_list)
}

## @param model precompiled Stan model
## @param data single dataset for Stan model
## @return Posterior samples for parameters of interest,
##         parameters that generated simulated dataset,
##         and sampling diagnostics
fit_mod <- function(model, data) {
  fit <- model$sample(
    data = data$stan_data,
    iter_warmup = 1000,
    iter_sampling = 1000,
    refresh = 3000,
    chains = 4, parallel_chains = 4,
    adapt_delta = 0.9
  )
  pars <- posterior::as_draws_df(
    fit$draws(
      variables = c("gamma", "eta", "beta", "log_lambda")
    )
  )
  true <- data$true_pars
  samp_pars <- fit$sampler_diagnostics() %>%
    as_draws_df()
  return(
    list(
      samps = pars,
      true = true,
      diags = samp_pars
    )
  )
}

## @param model precompiled Stan model
## @param datasets List of simulated datasets to fit model to
## @return List of model fits associated with list of simulated datasets
fit_ensemble <- function(model, datasets) {
  fits <- list()
  S <- length(datasets)
  for (i in 1:S) {
    if (i %% 10 == 0) {
      print(i)
    }
    fits[[i]] <- fit_mod(model, datasets[[i]])
  }
  return(fits)
}

## @param draws S x K matrix of S posterior draws for K quantities
## @param true True quantities against which to measure posterior draws
## @param p posterior probability interval 
## @return Binary vector for each quantity indicating if p-posterior
##         interval contained the true quantity
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

## @param draws S x K matrix of S posterior draws for K quantities
## @param true True quantities against which to measure posterior draws
## @param true_prior_var prior variance for K quantities
## @return posterior z-score and posterior shrinkage for K quantities
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

## @param ensemble_fits list of model fits to simulated datasets
## @param fixed_data static data for simulations
## @return matrices of coverage, shrinkage, z-scores and sampling diagnostics
##         across all model fits
parse_fits <- function(ensemble_fits, fixed_data) {
  gamma_prior_var <- fixed_data$stan_data$prior_scales_gamma^2
  eta_prior_var <- fixed_data$stan_data$prior_scales_eta^2
  beta_prior_var <- fixed_data$stan_data$prior_scales_beta^2
  log_lambda_prior_var <- fixed_data$stan_data$prior_scales_log_lambda^2
  prior_vars <- c(
    gamma_prior_var,
    eta_prior_var,
    beta_prior_var,
    log_lambda_prior_var
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

## @param data data on number of groups/predictors
## @param post_data summarized fits from parse_fits
## @return NULL creates bivariate z-score vs. shrinkage plots
##         for regression parameters
make_regr_shrink_zscore_plot <- function(data, post_data) {
  pdf("posterior-shrinkage-regr-dummy-run.pdf")
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
    nm <- paste0("gamma[", k, "]")
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

## @param data Data on number of groups/predictors
## @param post_data summarized fits from parse_fits
## @return NULL creates bivariate z-score vs. shrinkage plots
##         for missingness and rate parameters by race
make_rate_shrink_zscore_plot <- function(data, post_data) {
  pdf("posterior-shrinkage-rates-dummy-run.pdf")
  K <- data$K
  J <- data$J
  for (k in 1:J) {
    nm <- paste0("log_lambda[", k, "]")
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
    nm <- paste0("eta[", k, "]")
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

## compile simulation model
sim_model <- cmdstan_model("stan/simu-simple-regr-model.stan")
## compile inferential model
inf_model <- cmdstan_model("stan/simple-no-dp-regr-prior.stan")
## Read fixed data
fixed_data <  readRDS("data/fixed-data.RDS")

## Generate 2 simulated datasets
fake_data <- gen_data_random(sim_model, 2, fixed_data$stan_data)
## Transform simulated datasets into lists ready to pass to Stan
tt <- simu_curate(fake_data, fixed_data)
## Fit Stan model to simulated datsets and summarize the results
ensemble <- fit_ensemble(inf_model, tt)
parsed <- parse_fits(ensemble, fixed_data = fixed_data)

plt_data <- list(K = 9, J = 6)

## Make plots
make_regr_shrink_zscore_plot(plt_data, parsed)
make_rate_shrink_zscore_plot(plt_data, parsed)
