library(cmdstanr)

gen_data_multi_cat <- function(N, J, theta, p) {
  E <- matrix(1 + rpois(N * J, lambda = 5), N, J)
  Y <- X <- matrix(NA_real_, N, J)
  for (j in 1:J) {
    Y[, j] <- rbinom(N, E[, j], theta[j])
    X[, j] <- sapply(Y[, j], \(x) rbinom(1, x, p[j]))
  }
  M <- rowSums(Y - X)
  return(
    list(
      E = E,
      X = X,
      M = M,
      n = N,
      J = J
    )
  )
}

mod_miss <- cmdstan_model("binomial-binomial-model.stan")

set.seed(3326)
dat <- gen_data_multi_cat(1000, 3, c(0.8,0.3,0.2), c(0.8, 0.95, 0.85))
mle_cat <- mod_miss$optimize(data = dat)
mle_cat$mle()
fit_multi_cat <- mod_miss$sample(data = dat, parallel_chains = 4, init = f_start)
fit_multi_cat$summary()

mean(fit_multi_cat$draws("p[3]") > fit_multi_cat$draws("p[1]"))
