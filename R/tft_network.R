tft_nn <- torch::nn_module(
  "tft",
  initialize = function( input_dim, output_dim, cat_idxs, cat_dims,
                         known_idx, observed_idx, static_idx, target_idx,
                         total_time_steps = 252 + 5, num_encoder_steps = 252,
                         minibatch_size = 256, quantiles = list(0.5),
                         hidden_layer_size = 160, dropout_rate, stack_size = 1, num_heads, device) {
    self$cat_idxs <- cat_idxs  #  _known_categorical_input_idx
    self$cat_dims <- cat_dims # category_counts, list of nlevels along categories
    self$static_idx <- static_idx  # _static_input_loc # the grouping variable like shop_id
    self$known_idx <- known_idx  # _known_regular_input_idx # like day_of_week from date
    self$target_idx <- target_idx  # _input_obs_loc # target

    # a check par, just to easily find out when we need to
    # reload the model
    self$.check <- torch::nn_parameter(torch::torch_tensor(1, requires_grad = TRUE))

    self$time_steps <- total_time_steps
    self$num_encoder_steps <- num_encoder_steps
    self$input_dim <- input_dim # input_size # is c(time, id, known, observed, static)
    self$output_dim <- output_dim # output_size
    self$quantiles <- quantiles
    self$hidden_layer_size <- hidden_layer_size
    self$dropout_rate <- dropout_rate
    self$num_stacks <- stack_size
    self$num_heads <- num_heads
    self$device <- device
    self$batch_first <- TRUE
    self$num_static <- length(self$static_idx)
    self$num_inputs <- length(known_idx) +length(static_idx) + self$output_dim
    self$num_inputs_decoder <- length(known_idx) +length(static_idx)

    self$minibatch_size <- minibatch_size
    self$input_placeholder <- NULL
    self$attention_components <- NULL
    self$prediction_parts <- NULL

    ### TODO should Be Splitted Categorical embeddings. Currently stacks kwn_cat + obs_cat + stc_cat + target in that order
    ### TODO May be improved via tabnet::embedding_generator
    self$embeddings <- purrr::map(
      seq_along(cat_idxs), ~torch::nn_embedding(
        self$cat_dims[[.x]],
        self$hidden_layer_size)
      ) %>%
      torch::nn_module_list()

    ### Static inputs
    self$static_input_layer <- linear_layer(input_size=1, size=self$hidden_layer_size,
                                            use_time_distributed=TRUE, batch_first=self$batch_first)
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
    len_s <- self_attn_inputs$shape[2] # 192
    bs <- self_attn_inputs$shape[1] # 64
    # create batch_size lower triangular matrices
    mask <- purrr::map(seq_len(bs), ~torch::torch_ones(len_s, len_s, device = self$device)$tril()) %>% torch::torch_stack( dim=1)
    return(mask)
  },


  forward = function(known_numerics, known_categorical,
                     observed_numerics, observed_categorical,
                     static_numerics, static_categorical,
                     target_numerics, target_categorical) {
    # Size definitions.
    encoder_steps <- self$num_encoder_steps

    #### used to be in  get_tft_embeddings
    kwn_cat <- known_categorical$shape[3]
    obs_cat <- observed_categorical$shape[3]
    stc_cat <- static_categorical$shape[3]
    tgt_cat <- target_categorical$shape[3]

    kwn_num <- known_numerics$shape[3]
    obs_num <- observed_numerics$shape[3]
    stc_num <- static_numerics$shape[3]
    tgt_num <- target_numerics$shape[3]


    # Static inputs
    if (!is.null(self$static_idx)) {
      static_inputs <- c(
        if (stc_num>0)
          purrr::map(seq_len(stc_num), ~self$static_input_layer(static_numerics[,,.x:.x]$to(dtype=torch::torch_float()))),
        if (stc_cat>0)
          purrr::map(seq_len(stc_cat), ~self$embeddings[[kwn_cat + obs_cat + .x]](static_categorical[,,.x]$to(dtype=torch::torch_long())))
      ) %>%
        torch::torch_stack(dim=-1)

    } else {
      static_inputs <- NULL

    }

    # Targets ( should be numerical only ?)
    target_inputs <- c(
      if (tgt_num>0)
        purrr::map(seq_len(tgt_num), ~self$time_varying_embedding_layer(target_numerics[.., .x:.x]$to(dtype=torch::torch_float()))),
      if (tgt_cat>0)
        purrr::map(seq_len(tgt_cat), ~self$embeddings[[kwn_cat + obs_cat + stc_cat + .x]](target_categorical[..,.x]$to(dtype=torch::torch_long())))

      ) %>%
      torch::torch_stack(dim=-1)


    # Observed covariates are unknown inputs
    if (obs_num+obs_cat>0) {
      unknown_inputs <-  c(
        if (obs_num>0)
          purrr::map(seq_len(obs_num), ~self$time_varying_embedding_layer(observed_numerics[..,.x:.x]$to(dtype=torch::torch_float()))),
        if (obs_cat>0)
          purrr::map(seq_len(obs_cat), ~self$embeddings[[kwn_cat + .x]](observed_categorical[..,.x]$to(dtype=torch::torch_long())))
      ) %>%
        torch::torch_stack(dim=-1)
    } else {
      unknown_inputs <- NULL

    }
    # Known inputs
    if (kwn_num+kwn_cat>0) {
      known_inputs <- c(
        if (kwn_num>0)
          purrr::map(seq_len(kwn_num), ~self$time_varying_embedding_layer(known_numerics[..,.x:.x]$to(dtype=torch::torch_float()))),
        if (kwn_cat>0)
          purrr::map(seq_len(kwn_cat), ~self$embeddings[[.x]](known_categorical[..,.x]$to(dtype=torch::torch_long())))
    ) %>%
        torch::torch_stack(dim=-1)
    } else {
      known_inputs <- NULL

    }
    #### ended get_tft_embeddings


    # Isolate known and observed historical inputs.
    if (!is.null(unknown_inputs)) {
      historical_inputs <- torch::torch_cat(c(unknown_inputs[ , 1:encoder_steps, ],
                                          known_inputs[ , 1:encoder_steps, ],
                                          target_inputs[ , 1:encoder_steps, ]), dim=-1)

    } else {
      historical_inputs <- torch::torch_cat(c(known_inputs[ , 1:encoder_steps, ],
                                          target_inputs[ , 1:encoder_steps, ]), dim=-1)
    }
    # Isolate only known future inputs.
    future_inputs <- torch::torch_cat(c(known_inputs[ ,(encoder_steps+1):-1, ],
                                        static_inputs[ ,(encoder_steps+1):-1, ]), dim=-1)

    #Static_input first time-step (as constant by nature)
    static_encoder_weights <- self$static_combine_and_mask(static_inputs[,1:1,])
    static_weights <- static_encoder_weights[[2]]
    static_encoder <- static_encoder_weights[[1]]
    static_context_variable_selection <- self$static_context_variable_selection_grn(static_encoder)
    static_context_enrichment <- self$static_context_enrichment_grn(static_encoder)
    static_context_state_h <- self$static_context_state_h_grn(static_encoder)
    static_context_state_c <- self$static_context_state_c_grn(static_encoder)
    historical_features_flags <- self$historical_lstm_combine_and_mask(historical_inputs, static_context_variable_selection)
    historical_flags <- historical_features_flags[[2]]
    future_features_flags <- self$future_lstm_combine_and_mask(future_inputs, static_context_variable_selection)
    future_flags <- future_features_flags[[2]]

    history_lstm_state_h_state_c <- self$lstm_encoder(historical_features_flags[[1]], c(static_context_state_h$unsqueeze(1), static_context_state_c$unsqueeze(1)))
    future_lstm <- self$lstm_decoder(future_features_flags[[1]], c(history_lstm_state_h_state_c[[2]][[1]], history_lstm_state_h_state_c[[2]][[2]]))

    lstm_layer <- torch::torch_cat(c(history_lstm_state_h_state_c[[1]], future_lstm[[1]]), dim=2)
    # Apply gated skip connection
    input_embeddings <- torch::torch_cat(c(historical_features_flags[[1]], future_features_flags[[1]]), dim=2)

    lstm_layer <- self$lstm_glu(lstm_layer)
    temporal_feature_layer <- self$lstm_glu_add_and_norm(lstm_layer[[1]], input_embeddings)

    # Static enrichment layers
    expanded_static_context <- static_context_enrichment$unsqueeze(2)
    enriched <- self$static_enrichment_grn(temporal_feature_layer, expanded_static_context)[[1]]

    # Decoder self attention
    mask <- self$get_decoder_mask(enriched)
    x_self_att <- self$self_attn_layer(enriched, enriched, enriched, mask)#, attn_mask=mask.repeat(self$num_heads, 1, 1))

    x <- self$self_attention_glu(x_self_att[[1]])
    x <- self$self_attention_glu_add_and_norm(x[[1]], enriched)

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

    outputs <- self$output_layer(transformer_layer[ ,(encoder_steps+1):-1, ])
  return(list(outputs, NULL, attention_components))
 }
)
