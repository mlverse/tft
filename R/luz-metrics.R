luz_quantile_loss <- luz::luz_metric(
  abbrev = "q",
  initialize = function(quantile, q) {
    self$qloss <- 0
    self$n <- 0
    self$quantile <- torch:::torch_scalar_tensor(quantile)
    self$q <- q
    self$abbrev <- sprintf("q%02d", as.integer(100*quantile))
  },
  update = function(preds, target) {
    preds <- preds[,,self$q, drop=FALSE]
    other <- torch::torch_zeros_like(preds)

    low_res <- torch::torch_max(target - preds, other = other)
    up_res <- torch::torch_max(preds - target, other = other)

    quantiles <- self$quantile$to(device = target$device)

    loss <- torch::torch_mean(quantiles * low_res + (1 - quantiles) * up_res)
    self$qloss <- weighted.mean(
      c(self$qloss, loss$item()),
      c(self$n, 1)
    )
    self$n <- self$n + 1
  },
  compute = function() {
    self$qloss
  }
)
