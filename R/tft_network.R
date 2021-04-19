tft_nn <- torch::nn_module(
  "tft",
  initialize = function( input_dim, output_dim, cat_idxs , cat_dims, cat_emb_dim, static_idx, known_idx, input_idx,
                         total_time_steps = 252 + 5, num_encoder_steps = 252,
                         minibatch_size = 256, quantiles = list(0.5),
                         hidden_layer_size = 160, dropout_rate, stack_size = 1, num_heads) {
    self$cat_idxs <- cat_idxs  #  _known_categorical_input_idx
    self$cat_dims <- cat_dims # category_counts
    # broadcast cat_emb_dim if needed
    if (length(cat_emb_dim)==length(cat_dims)) {
        self$cat_emb_dims <- cat_emb_dim
    } else {
        self$cat_emb_dims <- rep(cat_emb_dim[[1]],length(cat_dims)) # num_categorical_variables
    }
    self$static_idx <- static_idx  # _static_input_loc # the grouping variable like shop_id
    self$known_idx <- known_idx  # _known_regular_input_idx # like day_of_week from date
    self$input_idx <- input_idx  # _input_obs_loc # time-dependant covariate like oil_price


    # a check par, just to easily find out when we need to
    # reload the model
    self$.check <- torch::nn_parameter(torch::torch_tensor(1, requires_grad = TRUE))

    self$time_steps <- total_time_steps
    self$num_encoder_steps <- num_encoder_steps
    self$input_dim <- input_dim # input_size
    self$output_dim <- output_dim # output_size
    self$quantiles <- quantiles
    self$hidden_layer_size <- hidden_layer_size
    self$dropout_rate <- dropout_rate
    self$num_stacks <- stack_size
    self$num_heads <- num_heads
    self$batch_first <- TRUE
    self$num_static <- length(self$static_idx)
    self$num_inputs <- length(self$known_idx) + self$output_dim
    self$num_inputs_decoder <- length(self$known_idx)

    self$minibatch_size <- minibatch_size
    self$input_placeholder <- NULL
    self$attention_components <- NULL
    self$prediction_parts <- NULL

#   num_regular_variables <- self$input_dim - cat_idxs
#
#   embedding_sizes <- [
#     self$hidden_layer_size for i, size in enumerate(self$cat_dims)
#   ]
#
    ### Categorical embeddings. May be improved via tabnet::embedding_generator
    self$embeddings <- purrr::map(
      seq_along(cat_idxs), ~torch::nn_embedding(
        self$cat_dims[[.x]],
        self$cat_emb_dims[[.x]])
      ) %>%
      torch::nn_module_list()

    ### Static inputs
    self$static_input_layer <- torch::nn_linear(self$hidden_layer_size, self$hidden_layer_size)
    ### Time_varying inputs
    self$time_varying_embedding_layer <- linear_layer(input_size=1, size=self$hidden_layer_size,
                                                      use_time_distributed=TRUE, batch_first=self$batch_first)
    self$static_combine_and_mask <- static_combine_and_mask(
      input_size=self$input_dim,
      num_static=self$num_static,
      hidden_layer_size=self$hidden_layer_size,
      dropout_rate=self$dropout_rate,
      additional_context=NULL,
      use_time_distributed=FALSE,
      batch_first=self$batch_first)
    self$static_context_variable_selection_grn <- gated_residual_network(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      output_size=NULL,
      dropout_rate=self$dropout_rate,
      use_time_distributed=FALSE,
      return_gate=FALSE,
      batch_first=self$batch_first)
    self$static_context_enrichment_grn <- gated_residual_network(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      output_size=NULL,
      dropout_rate=self$dropout_rate,
      use_time_distributed=FALSE,
      return_gate=FALSE,
      batch_first=self$batch_first)
    self$static_context_state_h_grn <- gated_residual_network(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      output_size=NULL,
      dropout_rate=self$dropout_rate,
      use_time_distributed=FALSE,
      return_gate=FALSE,
      batch_first=self$batch_first)
    self$static_context_state_c_grn <- gated_residual_network(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      output_size=NULL,
      dropout_rate=self$dropout_rate,
      use_time_distributed=FALSE,
      return_gate=FALSE,
      batch_first=self$batch_first)
    self$historical_lstm_combine_and_mask <- lstm_combine_and_mask(
      input_size=self$num_encoder_steps,
      num_inputs=self$num_inputs,
      hidden_layer_size=self$hidden_layer_size,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      batch_first=self$batch_first)
    self$future_lstm_combine_and_mask <- lstm_combine_and_mask(
      input_size=self$num_encoder_steps,
      num_inputs=self$num_inputs_decoder,
      hidden_layer_size=self$hidden_layer_size,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      batch_first=self$batch_first)

    self$lstm_encoder <- torch::nn_lstm(input_size=self$hidden_layer_size, hidden_size=self$hidden_layer_size, batch_first=self$batch_first)
    self$lstm_decoder <- torch::nn_lstm(input_size=self$hidden_layer_size, hidden_size=self$hidden_layer_size, batch_first=self$batch_first)

    self$lstm_glu <- gated_linear_unit(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      batch_first=self$batch_first)
    self$lstm_glu_add_and_norm <- add_and_norm(hidden_layer_size=self$hidden_layer_size)

    self$static_enrichment_grn <- gated_residual_network(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      output_size=NULL,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      return_gate=TRUE,
      batch_first=self$batch_first)

    self$self_attn_layer <- interpretable_multihead_attention(self$num_heads, self$hidden_layer_size, dropout_rate=self$dropout_rate)

    self$self_attention_glu <- gated_linear_unit(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      batch_first=self$batch_first)
    self$self_attention_glu_add_and_norm <- add_and_norm(hidden_layer_size=self$hidden_layer_size)

    self$decoder_grn <- gated_residual_network(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      output_size=NULL,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      return_gate=FALSE,
      batch_first=self$batch_first)

    self$final_glu <- gated_linear_unit(
      input_size=self$hidden_layer_size,
      hidden_layer_size=self$hidden_layer_size,
      dropout_rate=self$dropout_rate,
      use_time_distributed=TRUE,
      batch_first=self$batch_first)
    self$final_glu_add_and_norm <- add_and_norm(hidden_layer_size=self$hidden_layer_size)

    self$output_layer <- linear_layer(
      input_size=self$hidden_layer_size,
      size=self$output_dim * length(self$quantiles),
      use_time_distributed=TRUE,
      batch_first=self$batch_first)
  },

  get_decoder_mask = function(self_attn_inputs) {
        # """Returns causal mask to apply for self-attention layer.
        #
        #   Args:
        #     self_attn_inputs: Inputs to self attention layer to determine mask shape
        #   """
  len_s <- self_attn_inputs$shape[[2]] # 192
  bs <- tail(self_attn_inputs$shape,1) # [64]
  # create batch_size identity matrices
  mask <- torch::torch_cumsum(torch::torch_eye(len_s, device = self$device)$reshape(c(1, len_s, len_s))$torch_repeat_interleave(bs, 1, 1), 1)
  return(mask)
  },

  get_tft_embeddings = function(known_numerics, known_categorical,
                                obsvd_numerics, obsvd_categorical,
                                statc_numerics, statc_categorical,
                                target_numerics, target_categorical) {

    time_steps <- self$time_steps

    embedded_inputs <- list(
      map(seq_len(known_categorical$shape[3]), ~self$embeddings[[.x]](known_categorical[,,.x]$long())),
      map(seq_len(obsvd_categorical$shape[3]), ~self$embeddings[[.x]](obsvd_categorical[,,.x]$long())),
      map(seq_len(statc_categorical$shape[3]), ~self$embeddings[[.x]](statc_categorical[,,.x]$long())),
    )


    # Static inputs , we keep only the first time-step (by nature)
    if (self$static_idx) {
      static_inputs <- list(
        map(seq_len(statc_numerics$shape[3]), ~self$static_input_layer(statc_numerics[,1,.x]$long())),
        map(seq_len(statc_categorical$shape[3]), ~self$embeddings[[.x]](statc_categorical[,1,.x]$long()))
      ) %>%
        torch::torch_stack(dim=-1)

    } else {
      static_inputs <- NULL

    }

    # Targets
    obs_input <- seq_len(self$input_idx) %>%
      map(~self$time_varying_embedding_layer(regular_inputs[.., .x]$float())) %>%
      torch::torch_stack(dim=-1)


    # Target (Observed & a priori unknown) inputs
    # TODO maybe not prone to zero entries
    unknown_inputs <-  list(
      map(seq_len(target_numerics$shape[3]), ~self$time_varying_embedding_layer(target_numerics[..,.x])$float()),
      map(seq_len(target_categorical$shape[3]), ~self$embeddings[[.x]](target_categorical[..,.x]$long()))
    ) %>%
      torch::torch_stack(dim=-1)

    # A priori known inputs
    known_regular_inputs <- list(
      map(seq_len(obsvd_numerics$shape[3]), ~self$time_varying_embedding_layer(obsvd_numerics[..,.x]$float())),
      map(seq_len(obsvd_categorical$shape[3]), ~self$embeddings[[.x]](obsvd_categorical[,,.x]$long())),
    ) %>%
      torch::torch_stack(dim=-1)

    return( list(unknown_inputs, known_combined_layer, obs_inputs, static_inputs))
},

  forward = function(known_numerics, known_categorical,
                     observed_numerics, observed_categorical,
                     static_numerics, static_categorical,
                     target_numerics, target_categorical) {
    # Size definitions.
    time_steps <- self$time_steps
    combined_input_size <- self$input_size
    encoder_steps <- self$num_encoder_steps

    input_lst <- self$get_tft_embeddings(known_numerics, known_categorica,
                                         observed_numerics, observed_categorical,
                                         static_numerics, static_categorical,
                                         target_numerics, target_categorical)
    unknown_inputs <- input_lst[[1]]
    known_combined_layer <- input_lst[[2]]
    obs_inputs <- input_lst[[3]]
    static_inputs <- input_lst[[4]]

    # Isolate known and observed historical inputs.
    if (!is.null(unknown_inputs)) {
      historical_inputs <- torch.cat(list(unknown_inputs[ , 1:encoder_steps, ],
                                          known_combined_layer[ , 1:encoder_steps, ],
                                          obs_inputs[ , 1:encoder_steps, ]), dim=-1)

    } else {
      historical_inputs <- torch.cat(list(known_combined_layer[ , 1:encoder_steps, ],
                                          obs_inputs[ , 1:encoder_steps, ]), dim=-1)
    }
    # Isolate only known future inputs.
    future_inputs <- known_combined_layer[ , encoder_steps+1:-1, ]

    static_encoder_weights <- self$static_combine_and_mask(static_inputs)
    static_weights <- static_encoder_weights[[2]]
    static_context_variable_selection <- self$static_context_variable_selection_grn(static_encoder_weights[[1]])
    static_context_enrichment <- self$static_context_enrichment_grn(static_encoder_weights[[1]])
    static_context_state_h <- self$static_context_state_h_grn(static_encoder_weights[[1]])
    static_context_state_c <- self$static_context_state_c_grn(static_encoder_weights[[1]])
    historical_features_flags <- self$historical_lstm_combine_and_mask(historical_inputs, static_context_variable_selection)
    historical_flags <- historical_features_flags[[2]]
    future_features_flags <- self$future_lstm_combine_and_mask(future_inputs, static_context_variable_selection)
    future_flags <- future_features_flags[[2]]

    history_lstm_state_h_state_c <- self$lstm_encoder(historical_features_flags[[1]], c(static_context_state_h$unsqueeze(1), static_context_state_c$unsqueeze(1)))
    future_lstm <- self$lstm_decoder(future_features_flags[[1]], c(history_lstm_state_h_state_c[[2]], history_lstm_state_h_state_c[[3]]))

    lstm_layer <- torch.cat(c(history_lstm_state_h_state_c[[1]], future_lstm[[1]]), dim=1)
    # Apply gated skip connection
    input_embeddings <- torch.cat(c(historical_features_flags[[1]], future_features_flags[[1]]), dim=1)

    lstm_layer <- self$lstm_glu(lstm_layer)
    temporal_feature_layer <- self$lstm_glu_add_and_norm(lstm_layer[[1]], input_embeddings)

    # Static enrichment layers
    expanded_static_context <- static_context_enrichment$unsqueeze(2)
    enriched <- self$static_enrichment_grn(temporal_feature_layer, expanded_static_context)[[1]]

    # Decoder self attention
    mask <- self$get_decoder_mask(enriched)
    x_self_att <- self$self_attn_layer(enriched, enriched, enriched, mask)#, attn_mask=mask.repeat(self$num_heads, 1, 1))

    x <- self$self_attention_glu(x_self_att[[1]])
    x <- self$self_attention_glu_add_and_norm(x, enriched)

    # Nonlinear processing on outputs
    decoder <- self$decoder_grn(x)
    # Final skip connection
    decoder <- self$final_glu(decoder)[[1]]
    transformer_layer <- self$final_glu_add_and_norm(decoder, temporal_feature_layer)
    # Attention components for explainability
    attention_components <- list(
      # Temporal attention weights
      decoder_self_attn = x_self_att[[2]],
      # Static variable selection weights
      static_flags = static_weights[.., 1],
      # Variable selection weights of past inputs
      historical_flags = historical_flags[.., 1, ],
      # Variable selection weights of future inputs
      future_flags = future_flags[.., 1, ]
    )

    outputs <- self$output_layer(transformer_layer[ , self$num_encoder_steps+1:-1, ])
  return(list(outputs, all_inputs, attention_components))
 }
)
