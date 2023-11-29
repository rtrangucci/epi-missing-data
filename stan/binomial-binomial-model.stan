functions {
  real binomial_2_lpmf(int y_obs, int y_tot,
                              real p, real theta, int E) {
    if (y_tot > E)
      return negative_infinity();
    return binomial_lpmf(y_obs | y_tot, p)
      + binomial_lpmf(y_tot | E, theta) - binomial_lpmf(y_obs | E, p * theta);
  }
  real miss_lpmf(array[] int y, int n_miss, 
                 vector p, vector theta,
                 array[] int E) {
    int N = rows(theta);
    array[N + 1, n_miss + 1] real alpha;
  
  // alpha[n + 1, tot + 1] = log p of tot missing cases
  // distributed among first n categories 
    alpha[1, 1:(n_miss + 1)] = rep_array(0, n_miss+1);
    for (n in 1:N) {
      // tot = 0
      alpha[n + 1, 1] = alpha[n, 1] 
          + binomial_2_lpmf(y[n]|y[n],p[n],theta[n], E[n]);
      
      // 0 < tot < n
        
        for (tot in 1:n_miss) {
          if (n > 1) {
            vector[tot + 1] vec;
              for (i in 1:(tot + 1)) {
                vec[i] = alpha[n,i] 
                + binomial_2_lpmf(y[n] | 
                    y[n] + tot - (i - 1), 
                    p[n],theta[n], E[n]);
              }
              alpha[n + 1, tot + 1] = log_sum_exp(vec);
          } else {
              alpha[n + 1,tot + 1] 
               = binomial_2_lpmf(y[n]| y[n] 
               + tot,p[n],theta[n], E[n]);
          }
        }
    }
    return alpha[N + 1, n_miss + 1];
  }
}
data {
  int n;
  int J;
  array[n, J] int E;
  array[n, J] int X;
  array[n] int M;
}
parameters {
  vector<lower=0, upper=1>[J] p;
  vector<lower=0, upper=1>[J] theta;
}
transformed parameters {
  vector[J] filtered = p .* theta;
}
model {
  for (i in 1:n)
    X[i,] ~ binomial(E[i,], filtered);
  for (i in 1:n) 
    target += miss_lpmf(X[i,] | M[i], p, theta, E[i,]);
}
