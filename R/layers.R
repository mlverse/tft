gated_linear_unit <- torch::nn_module(
  "gated_linear_unit",
  # TODO add the use_time_distributed option
  initialize = function(input_size, hidden_layer_size,
                        dropout_rate=NULL,
                        use_time_distributed=TRUE,
                        batch_first=FALSE) {
    self$hidden_layer_size <- hidden_layer_size
    self$dropout_rate <- dropout_rate
    self$use_time_distributed <- use_time_distributed

    if (!is.null(self$dropout_rate)) {
      self$dropout <- torch::nn_dropout(self$dropout_rate)

    }

    self$activation_layer <- linear_layer(input_size, hidden_layer_size, use_time_distributed, batch_first)
    self$gated_layer <- linear_layer(input_size, hidden_layer_size, use_time_distributed, batch_first)
    self$sigmoid <- torch::nn_sigmoid()
  },

  forward = function(x) {
    if (!is.null(self$dropout_rate)) {
      x <- self$dropout(x)

    }
    gated <- x %>%
      self$gated_layer() %>%
      self$sigmoid()

    return(list(x %>% self$activation_layer() %>% torch::torch_mul(., gated), gated))
  }
)

# see https://github.com/akeskiner/Temporal_Fusion_Transform/blob/master/tft_model.py#L34
time_distributed <- torch::nn_module(
  "time_distributed",
  ## Takes any module and stacks the time dimension with the batch dimension of inputs before apply the module
  ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
  # currently without batch_first
  initialize = function(module, batch_first=FALSE) {
    self$module <- module
    self$batch_first <- batch_first
  },
  forward = function(x) {

    if (x$ndim <= 2)
      return(self$module(x))

    # Squash samples and timesteps into a single axis
    # TODO BUG
    x_reshape <- x$contiguous()$view(c(-1, x$size(x$ndim)))  # (samples * timesteps, input_size)

    y <- self$module(x_reshape)

    # We have to reshape Y
    if (self$batch_first){
      y <- y$contiguous()$view(c(x$size(1), -1, y$size(y$ndim)))  # (samples, timesteps, output_size)
    } else{
      y <- y$view(c(-1, x$size(2), y$size(y$ndim)))  # (timesteps, samples, output_size)
    }

    return(y)
  }
)

linear_layer <- torch::nn_module(
  "linear_layer",
  initialize = function(input_size, size,  use_time_distributed=TRUE,  batch_first=FALSE) {
    self$use_time_distributed <- use_time_distributed
    self$input_size <-input_size
    self$size <- size
    if (use_time_distributed) {
      self$layer <- time_distributed(torch::nn_linear(input_size, size), batch_first=batch_first)
    } else {
      self$layer <- torch::nn_linear(input_size, size)
    }
  },
  forward = function(x) {
      return(self$layer(x))
  }
)

gated_residual_network <- torch::nn_module(
  "gated_residual_network",
  initialize = function(input_size, hidden_layer_size, output_size=NULL,  dropout_rate=NULL,
                        use_time_distributed=TRUE, return_gate=FALSE, batch_first=FALSE ) {

    if (is.null(output_size)) {
      output_size <- hidden_layer_size
    }

    self$input_size <- input_size
    self$output_size <- output_size
    self$hidden_layer_size <- hidden_layer_size
    self$return_gate <- return_gate

    self$skip_linear_layer <- linear_layer(input_size, output_size, use_time_distributed, batch_first)

    self$hidden_linear_layer1 <- linear_layer(input_size, hidden_layer_size, use_time_distributed, batch_first)
    self$hidden_context_layer <- linear_layer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
    self$hidden_linear_layer2 <- linear_layer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)

    self$elu <- torch::nn_elu()
    self$glu <- gated_linear_unit(hidden_layer_size, output_size, dropout_rate, use_time_distributed, batch_first)
    self$add_and_norm <- add_and_norm(hidden_layer_size = output_size)
    self$skip_connection <- torch::nn_linear(input_size, hidden_layer_size)
  },
  forward = function(x, context = NULL) {
    # Setup skip connection
    if (is.null(self$output_size)) {
      skip <- x
    } else {
      skip <- self$skip_linear_layer(x)
    }
    # Apply feedforward network
    hidden <- self$hidden_linear_layer1(x)

    if (!is.null(context))
      hidden <- hidden + self$hidden_context_layer(context)

    hidden <- self$elu(hidden)
    hidden <- self$hidden_linear_layer2(hidden)

    gating_layer_gate <- self$glu(hidden)
    if (self$return_gate) {
      list(self$add_and_norm(skip, gating_layer_gate[[1]]), gating_layer_gate[[2]])
    } else {
      self$add_and_norm(skip, gating_layer_gate[[1]])
    }
  }
)

scaled_dot_product_attention <- torch::nn_module(
  "scaled_dot_product_attention",
  initialize = function(attn_dropout=0) {
    self$dropout <- torch::nn_dropout(attn_dropout)
    self$activation <- torch::nn_softmax(dim=-1)
  },
  forward = function(query, key, value, mask) {
    # applies scaled dot product attention (unused)
    # query
    # temper <- torch::torch_sqrt(torch::torch_tensor(key$shape[key$ndim], dtype = torch::torch_float, device = self$device) )
    attn <- torch::torch_bmm(query, key$permute(c(1, 3, 2)) )
    if (!is.null(mask)) {
      mmask <- -1e-9 * (1 - mask)
      attn <- torch::torch_add(attn, mmask)
    }
    attn <- self$activation(attn)
    attn <- self$dropout(attn)
    output <- torch::torch_bmm(attn, value)
    return(list(output, attn))
  }
)

interpretable_multihead_attention <- torch::nn_module(
  "interpretable_multihead_attention",
  initialize = function(n_head, d_model, dropout_rate) {
    self$n_head <- n_head
    self$d_k <- self$d_v <- d_k <- d_v <- d_model %/% n_head
    self$dropout <- torch::nn_dropout(dropout_rate)

    self$qs_layers <- torch::nn_module_list()
    self$ks_layers <- torch::nn_module_list()
    self$vs_layers <- torch::nn_module_list()

    # use same value layer to facilitate interpretability
    vs_layer <- torch::nn_linear(d_model, d_v, bias=FALSE)
    qs_layer <- torch::nn_linear(d_model, d_k, bias=FALSE)
    ks_layer <- torch::nn_linear(d_model, d_k, bias=FALSE)

    for (head in seq_len(n_head)) {
      self$qs_layers$append(qs_layer)
      self$ks_layers$append(ks_layer)
      self$vs_layers$append(vs_layer)
    }

    self$attention <- scaled_dot_product_attention()
    self$w_o <- torch::nn_linear(self$d_k, d_model, bias=FALSE)
  },
  forward = function( query, key, value, mask) {
    # Applies interpretable multihead attention.
    #
    # Using T to denote the number of time steps fed into the transformer.
    # query :Query tensor of dim (?, T, d_model)
    # key: Key of dim (?, T, d_model)
    # value: Values of dim (?, T, d_model)
    # mask: Masking if required with dim (?, T, T)
    # returns a list( layer_outputs, attention_weights)

    n_head <- self$n_head
    heads <- list()
    attns <- list()
    for (head in seq_len(n_head)) {
      qs <- self$qs_layers[[head]](query)
      ks <- self$ks_layers[[head]](key)
      vs <- self$vs_layers[[head]](value)
      attn_lst <- self$attention(qs, ks, vs, mask)

      head_dropout <- self$dropout(attn_lst[[1]])
      heads <- c(heads, head_dropout)
      attns <- c(attns, attn_lst[[2]])
    }
    attn <- torch::torch_stack(attns)
    outputs <- heads %>%
      torch::torch_stack(dim=1) %>%
      torch::torch_mean(dim=1) %>%
      self$w_o() %>%
      self$dropout()

    return(list(outputs, attn))
    }
)

add_and_norm <- torch::nn_module(
  "add_and_norm",
  initialize = function(hidden_layer_size){
    self$normalize <- torch::nn_layer_norm(hidden_layer_size)
  },
  forward = function(x1, x2) {
    x <- torch::torch_add(x1, x2)
    return(self$normalize(x))
  }
)

static_combine_and_mask <- torch::nn_module(
  "static_combine_and_mask",
  # Applies variable selection network to static inputs.
  initialize = function(input_size, num_static, hidden_layer_size, dropout_rate, additional_context=NULL, use_time_distributed=FALSE, batch_first=TRUE){
    self$hidden_layer_size <- hidden_layer_size
    self$input_size <- input_size
    self$num_static <- num_static
    self$dropout_rate <- dropout_rate
    self$additional_context <- additional_context

    self$flattened_grn <- gated_residual_network(self$num_static*self$hidden_layer_size, self$hidden_layer_size,
                                                   self$num_static, self$dropout_rate, use_time_distributed=use_time_distributed,
                                                   return_gate=FALSE, batch_first=batch_first)

    self$single_variable_grns <- torch::nn_module_list()
    for (i in seq_len(self$num_static)) {
      self$single_variable_grns$append(gated_residual_network(self$hidden_layer_size, self$hidden_layer_size,
                                                              NULL, self$dropout_rate, use_time_distributed=use_time_distributed,
                                                              return_gate=FALSE,
                                                              batch_first=batch_first))
    }

    self$softmax <- torch::nn_softmax(dim=2)

  },
  forward = function( embedding, additional_context=NULL) {
    # Add temporal features
    flattened_embedding <- torch::torch_flatten(embedding, start_dim=2)
    if (!is.null(additional_context)) {
      sparse_weights <- self$flattened_grn(flattened_embedding, additional_context)
    } else {
      sparse_weights <- self$flattened_grn(flattened_embedding)
    }

    sparse_weights <- self$softmax(sparse_weights)$unsqueeze(3)

    transformed_embedding <- purrr::map(seq_len(self$num_static),
                                         ~self$single_variable_grns[[.x]](torch::torch_flatten(embedding[, .x, ,drop = FALSE], start_dim=2))) %>%
                                           torch::torch_stack(dim=2)

    combined <- transformed_embedding*sparse_weights

    static_vec <- combined$sum(dim=2)

    return(list(static_vec, sparse_weights))
  }
)

lstm_combine_and_mask <- torch::nn_module(
  "lstm_combine_and_mask",
  initialize = function(input_size, num_inputs, hidden_layer_size, dropout_rate, use_time_distributed=FALSE, batch_first=TRUE){
    # self$hidden_layer_size <- hidden_layer_size
    # self$input_size <- input_size
    # self$num_inputs <- num_inputs
    self$dropout_rate <- dropout_rate

    self$flattened_grn <- gated_residual_network(num_inputs*hidden_layer_size, hidden_layer_size,
                                                 num_inputs, self$dropout_rate, use_time_distributed=use_time_distributed,
                                                 return_gate=TRUE, batch_first=batch_first)

    self$single_variable_grns <- torch::nn_module_list()
    for (i in seq_len(num_inputs)) {
      self$single_variable_grns$append(gated_residual_network(hidden_layer_size, hidden_layer_size,
                                                              NULL, self$dropout_rate,
                                                              use_time_distributed=use_time_distributed,
                                                              return_gate=FALSE, batch_first=batch_first))
    }
    self$softmax <- torch::nn_softmax(dim=3)
  },
  forward = function( embedding, additional_context=NULL) {
    # Add temporal features
    dim_embedding <- embedding$shape
    batch <- dim_embedding[1]
    time_steps <- dim_embedding[2]
    embedding_dim <- dim_embedding[3]
    num_inputs <- dim_embedding[4]

    flattened_embedding <- torch::torch_reshape(embedding, list(batch, time_steps, embedding_dim * num_inputs))

    if (!is.null(additional_context)) {
      expanded_static_context <- additional_context$unsqueeze(2)
      sparse_weights_staticgate <- self$flattened_grn(flattened_embedding, expanded_static_context)
      sparse_weights <- sparse_weights_staticgate[[1]]
      static_gate <- sparse_weights_staticgate[[2]]
    } else {
      sparse_weights <- self$flattened_grn(flattened_embedding)[[1]]
      static_gate <- NULL
    }

    sparse_weights <- self$softmax(sparse_weights)$unsqueeze(3)

    transformed_embedding <- seq_len(num_inputs) %>%
      purrr::map(~self$single_variable_grns[[.x]](embedding[.., .x])) %>%
      torch::torch_stack( dim=-1)

    combined <- transformed_embedding*sparse_weights

    temporal_ctx <- combined$sum(dim=-1)

    return(list(temporal_ctx, sparse_weights, static_gate))
  }
)
