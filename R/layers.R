gated_linear_unit <- torch::nn_module(
  "gated_linear_unit",
  initialize = function(input_size) {
    self$fc1 <- torch::nn_linear(input_size, input_size)
    self$fc2 <- torch::nn_linear(input_size, input_size)
    self$sigmoid <- torch::nn_sigmoid()
  },
  forward = function(x) {
    sig <- x %>%
      self$fc1() %>%
      self$sigmoid()
    x %>%
      self$fc2() %>%
      torch::torch_mul(sig, .)
  }
)

# see https://github.com/mattsherar/Temporal_Fusion_Transform/blob/master/tft_model.py#L34
time_distributed <- torch::nn_module(
  "time_distributed",
  initialize = function(module, dim = 2) {
    self$module <- module
    self$dim <- dim
  },
  forward = function(x) {

    if (x$ndim <= 2)
      return(self$module(x))

    torch::torch_unbind(x, self$dim) %>%
      lapply(self$module) %>%
      torch::torch_stack(self$dim)
  }
)

gated_residual_network <- torch::nn_module(
  "gated_residual_network",
  initialize = function() {

  },
  forward = function(x, context = torch::torch_zeros_like(x)) {

  }
)
