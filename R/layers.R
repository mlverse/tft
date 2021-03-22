gated_linear_unit <- torch::nn_module(
  "gated_linear_unit",
  # TODO add the use_time_distributed option
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
  ## Takes any module and stacks the time dimension with the batch dimension of inputs before apply the module
  ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
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
  initialize = function(input_size, hidden_state_size) {
    self$hidden1 <- torch::nn_linear(2*input_size, hidden_state_size)
    self$hidden2 <- torch::nn_linear(hidden_state_size, hidden_state_size)
    self$elu <- torch::nn_elu()
    self$layer_norm <- torch::nn_layer_norm(hidden_state_size)
    self$glu <- gated_linear_unit(hidden_state_size)
    self$skip_connection <- torch::nn_linear(input_size, hidden_state_size)
  },
  forward = function(x, context = torch::torch_zeros_like(x)) {
    hidden_state <- list(x, context) %>%
      torch::torch_cat(dim = -1) %>%
      self$hidden1() %>%
      self$elu() %>%
      self$hidden2()

    self$layer_norm(self$skip_connection(x) + self$glu(hidden_state))
  }
)
scaled_dot_product_attention <- torch::nn_module(
  "scaled_dot_product_attention",
  initialize = function(attn_dropout=0) {
    self$dropout <- torch::nn_dropout(attn_dropout)
    self$activation <- torch::nn_softmax(dim=-1)
    self$device <- torch::torch_device(if (torch::cuda_is_available()) "cuda" else "cpu")
  },
  forward = function(query, key, value, mask) {
    # applies scaled dot product attention
    temper <- torch::torch_sqrt(torch::torch_tensor(tail(dim(key),1), dtype = torch::torch_float, device = self.device) )
    attn <- torch::torch_bmm(query, torch::torch_transpose(key, 2,3) )
    if (!is.null(mask)) {
      mmask <- -1e-9 * (1 - torch::torch_tensor(mask, dtype = torch::torch_float, device = self.device))
      attn <- torch::torch_add(attn, mmask)
    }
    attn <- self$activation(attn)
    attn <- self$dropout(attn)
    output <- torch::torch_bmm(attn, value)
  }
)
