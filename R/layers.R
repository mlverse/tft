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

# see https://github.com/akeskiner/Temporal_Fusion_Transform/blob/master/tft_model.py#L34
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

linear_layer <- torch::nn_module(
  "linear_layer",
  initialize = function(input_size, size,  use_time_distributed=TRUE,  batch_first=FALSE){
    self$use_time_distributed <- use_time_distributed
    self$input_size <-input_size
    self$size <- size
    if (use_time_distributed) {
      self$layer <- time_distributed(torch::nn_linear(input_size, size), batch_first=batch_first)
    } else {
      self$layer = torch::nn_linear(input_size, size)
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
    if (!is.null(output_size)) {
      output <- hidden_layer_size
    } else {
      output <- output_size
    }
    self$output <- output
    self$input_size <- input_size
    self$output_size <- output_size
    self$hidden_layer_size <- hidden_layer_size
    self$return_gate <- return_gate

    self$linear_layer <- linear_layer(input_size, output, use_time_distributed, batch_first)

    self$hidden_linear_layer1 <- linear_layer(input_size, hidden_layer_size, use_time_distributed, batch_first)
    self$hidden_context_layer <- linear_layer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
    self$hidden_linear_layer2 <- linear_layer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)

    self$elu <- torch::nn_elu()
    self$glu <- gated_linear_unit(hidden_state_size, output, dropout_rate, use_time_distributed, batch_first)
    self$add_and_norm <- add_and_norm(hidden_layer_size = output)
    self$skip_connection <- torch::nn_linear(input_size, hidden_layer_size)
  },
  forward = function(x, context = torch::torch_zeros_like(x)) {
    # Setup skip connection
    if (!is.null(self$output_size)) {
      skip = x
    } else {
      skip = self$linear_layer(x)
    }
    # Apply feedforward network
    hidden <- self$hidden_linear_layer1(x)
    if (!is.null(context)) {
      hidden <- hidden + self$hidden_context_layer(context)

    }
    hidden <- self$elu1(hidden)
    hidden <- self$hidden_linear_layer2(hidden)

    gating_layer_gate <- self$glu(hidden)
    if (self$return_gate) {
          return(list(self$add_and_norm(skip, gating_layer_gate[[1]]), gating_layer_gate[[2]]))
    } else {
          return(self$add_and_norm(skip, gating_layer_gate[[1]]))
    }
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
    # query
    temper <- torch::torch_sqrt(torch::torch_tensor(tail(dim(key),1), dtype = torch::torch_float, device = self.device) )
    attn <- torch::torch_bmm(query, torch::torch_transpose(key, 2,3) )
    if (!is.null(mask)) {
      mmask <- -1e-9 * (1 - torch::torch_tensor(mask, dtype = torch::torch_float, device = self.device))
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
    # query :Query tensor of dim (?, T, d_model)
    # key: Key of dim (?, T, d_model)
    # value: Values of dim (?, T, d_model)
    # mask: Masking if required with dim (?, T, T)

    head <- self$n_head
    heads <- list()
    attns <- list()
    for (head in seq_len(n_head)) {
      qs <- self$qs_layers[head](query)
      ks <- self$ks_layers[head](key)
      vs <- self$vs_layers[head](value)
      attn_lst <- self$attention(qs, ks, vs, mask)

      head_dropout <- self$dropout(attn_lst[[1]])
      heads$append(head_dropout)
      attns$append(attn_lst[[2]])
    }
    if (n_head>1) {
      head <- torch::torch_stack(heads)
    } else {
      head <- heads[1]
    }
    attn <- torch::torch_stack(attns)

    if (n_head>1) {
      outputs <- torch::torch_mean(head, dim=1)
    } else {
      outputs <- head
    }
    outputs <- self$w_o(outputs)
    outputs <- self$dropout(outputs)

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
    return(self$normalize)
  }
)

static_combine_and_mask <- torch::nn_module(
  "static_combine_and_mask",
  initialize = function(input_size, num_static, hidden_layer_size, dropout_rate, additional_context=NULL, use_time_distributed=FALSE, batch_first=TRUE){
    self$hidden_layer_size <- hidden_layer_size
    self$input_size <- input_size
    self$num_static <- num_static
    self$dropout_rate <- dropout_rate
    self$additional_context <- additional_context

    if (!is.null(self$additional_context)) {
      self$flattened_grn <- gated_residual_network(self$num_static*self$hidden_layer_size, self$hidden_layer_size, self$num_static, self$dropout_rate, use_time_distributed=FALSE, return_gate=FALSE, batch_first=batch_first)
    } else {
      self$flattened_grn <- gated_residual_network(self$num_static*self$hidden_layer_size, self$hidden_layer_size, self$num_static, self$dropout_rate, use_time_distributed=FALSE, return_gate=FALSE, batch_first=batch_first)
    }

    self$single_variable_grns <- torch::nn_module_list()
    for (i in seq_len(self$num_static)) {
      self$single_variable_grns$append(gated_residual_network(self$hidden_layer_size, self$hidden_layer_size,
                                                              NULL, self$dropout_rate,
                                                              use_time_distributed=FALSE, return_gate=FALSE, batch_first=batch_first))
    }

    self$softmax <- torch::nn_softmax(dim=2)

  },
  forward = function( embedding, additional_context=NULL) {
    # Add temporal features
    num_static <- dim(embedding)[2]
    flattened_embedding <- torch::torch_flatten(embedding, start_dim=1)
    if (!is.null(additional_context)) {
      sparse_weights <- self$flattened_grn(flattened_embedding, additional_context)
    } else {
      sparse_weights <- self$flattened_grn(flattened_embedding)
    }

    sparse_weights <- self$softmax(sparse_weights)$unsqueeze(3)

    # TODO make it R friendly
    trans_emb_list <- list()
    for (i in seq_len(self$num_static)){
      ##select slice of embedding belonging to a single input
      trans_emb_list <- c(trans_emb_list,
                          self$single_variable_grns[i](torch::torch_flatten(embedding[, i, ], start_dim=2))
      )
    }

    transformed_embedding <- torch::torch_stack(trans_emb_list, dim=0)

    combined <- transformed_embedding*sparse_weights

    static_vec <- combined$sum(dim=0)

    return(list(static_vec, sparse_weights))
  }
)

lstm_combine_and_mask <- torch::nn_module(
  "lstm_combine_and_mask",
  initialize = function(input_size, num_inputs, hidden_layer_size, dropout_rate, use_time_distributed=FALSE, batch_first=TRUE){
    self$hidden_layer_size <- hidden_layer_size
    self$input_size <- input_size
    self$num_inputs <- num_inputs
    self$dropout_rate <- dropout_rate

    self$flattened_grn <- gated_residual_network(self$num_inputs*self$hidden_layer_size, self$hidden_layer_size,
                                                 self$num_inputs, self$dropout_rate, use_time_distributed=use_time_distributed,
                                                 return_gate=TRUE, batch_first=batch_first)

    self$single_variable_grns <- torch::nn_module_list()
    for (i in seq_len(self$num_inputs)) {
      self$single_variable_grns$append(gated_residual_network(self$hidden_layer_size, self$hidden_layer_size,
                                                              NULL, self$dropout_rate,
                                                              use_time_distributed=use_time_distributed,
                                                              return_gate=FALSE, batch_first=batch_first))
    }

    self$softmax <- torch::nn_softmax(dim=3)

  },
  forward = function( embedding, additional_context=NULL) {
    # Add temporal features
    dim_embedding <- dim(embedding)
    time_steps <- dim_embedding[2]
    embedding_dim <- dim_embedding[3]
    num_inputs <- dim_embedding[4]

    flattened_embedding <- torch::torch_reshape(embedding, list(-1, time_steps, embedding_dim * num_inputs))
    expanded_static_context <- additional_context$unsqueeze(2)

    if (!is.null(additional_context)) {
      sparse_weights_staticgate <- self$flattened_grn(flattened_embedding, expanded_static_context)
      sparse_weights <- sparse_weights_staticgate[[1]]
      static_gate <- sparse_weights_staticgate[[2]]
    } else {
      sparse_weights <- self$flattened_grn(flattened_embedding)
    }

    sparse_weights <- self$softmax(sparse_weights)$unsqueeze(3)

    # TODO make it R friendly
    trans_emb_list <- list()
    for (i in seq_len(self$num_inputs)){
      ##select slice of embedding belonging to a single input
      trans_emb_list <- c(trans_emb_list,
                          self$single_variable_grns[i](embedding[, i])
      )
    }

    transformed_embedding <- torch::torch_stack(trans_emb_list, dim=0)

    combined <- transformed_embedding*sparse_weights

    temporal_ctx <- combined$sum(dim=0)

    return(list(temporal_ctx, sparse_weights, static_gate))
  }
)
