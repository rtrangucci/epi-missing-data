args <- commandArgs(trailingOnly = TRUE)
data_fl <- args[1]
data_idx <- as.numeric(args[2])
mod_nm <-  args[3]
prep <- args[4]
vars_to_pull_fl <- args[5]
print(args)
data_fit <- readRDS(data_fl)[[data_idx]]
inf_model <- readRDS(mod_nm)
vars_to_pull <- readRDS(vars_to_pull_fl)

options(CMDSTANR_NO_VER_CHECK = TRUE)
print("loading cmdstanr")
library(cmdstanr, quietly = T)
print("loaded cmdstanr")
library(dplyr)
library(posterior)

print("loading fit_mod")
fit_mod <- function(model, data) {
  fit <- model$sample(
    data = data$stan_data,
    iter_warmup = 2000,
    iter_sampling = 1500,
    refresh = 3000,
    chains = 4, parallel_chains = 4,
    adapt_delta = 0.95,
    max_treedepth = 14
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
print("loaded fit_mod")

print("fitting-model")
fit <- fit_mod(inf_model, data_fit)

saveRDS(fit, paste0(prep,"_",data_idx,".RDS"))
