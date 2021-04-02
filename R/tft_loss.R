pinball_loss <- torch::nn_module(
    "pinball_loss",
    initialize = function(training_tau, device){
      self$training_tau <- training_tau
      self$device <- device

    },
    forward = function(predictions, actuals) {
      cond <- torch::torch_zeros_like(predictions, device=self$device)
      loss <- torch::torch_sub(actuals, predictions)$to(self$device)

      less_than <- torch::torch_mul(loss,
                                    torch::torch_mul(torch::torch_gt(loss, cond)$type_as(torch::torch_float())$to(self$device),
                                                     self$training_tau))

      greater_than <- torch::torch_mul(loss,
                                       torch::torch_mul(torch::torch_lt(loss, cond)$type(torch::torch_float)$to(self$device),
                                                        (self$training_tau - 1)))
      final_loss <- torch::torch_add(less_than, greater_than)
      return(torch::torch_sum(final_loss) / (final_loss$shape[1] * final_loss$shape[2]) * 2)
    }
)
quantile_loss <- torch::nn_module(
  "quantile_loss",
  initialize = function(quantiles){
    self$quantiles = quantiles
  },
  forward = function( predictions, actuals) {
    stopifnot(!actuals$requires_grad)
    stopifnot(predictions$size(1) == actuals$size(1))
    losses = list()
    for (i in self$quantiles) {
      errors <- actuals - predictions[, i]
      losses <- c(losses, torch::torch_max( (q-1) * errors, q * errors)$unsqueeze(2) )

    }
    loss = torch::torch_mean(torch::torch_sum(torch::torch_cat(losses, dim=2), dim=2))
    return(loss)


  }
)
rmsse_loss <- torch::nn_module(
  "rmsse_loss",
  initialize = function(device){
    self$device <- device
  },
  forward = function(predictions, actuals) {
    sequence_length <- predictions$shape[2]
    numerator <-  torch::torch_sum( torch::torch_pow(predictions - actuals, 2), dim=2)
    loss <-  torch::torch_div(numerator, sequence_length)
    loss <-  torch::torch_sqrt(torch::torch_mean(loss))
    return(loss)

  }
)

smape_loss <- torch::nn_module(
  "smape_loss",
  initialize = function(device){
    self$device <- device
  },
  forward = function(predictions, actuals) {
    sequence_length <- predictions$shape[2]
    predictions <- predictions$type_as(torch::torch_float())
    actuals <- actuals$type_as(torch::torch_float())
    sumf <- torch::torch_sum(torch::torch_abs(predictions - actuals) / (torch::torch_abs(predictions) + torch::torch_abs(actuals)), dim=2)

    return(torch::torch_mean((2 * sumf) / sequence_length))
  }
)
