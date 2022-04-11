
#' Temporal Fusion Transformer Module
#'
#'
#' @param n_features a list containing the shapes for all necessary information
#'        to define the size of layers, including:
#'                   - `$encoder$past$(num|cat)`: shape of past features
#'                   - `$encoder$static$(num|cat)`: shape of the static features
#'                   - `$decoder$target`: shape of the target variable
#'        We exclude the batch dimension.
#' @param feature_sizes The number of unique elements for each categorical
#'        variable in the dataset.
#' @param hidden_state_size The size of the model shared accross multiple parts
#'        of the architecture.
#' @param dropout Dropout rate used in many different places in the network
#' @param num_heads Number of heads in the attention layer.
#' @param num_lstm_layers Number of LSTM layers used in the Locality Enhancement
#'   Layer. Usually 2 is good enough.
temporal_fusion_transformer <- torch::nn_module(
  "temporal_fusion_transformer",
  initialize = function(num_features, feature_sizes, hidden_state_size = 100,
                        dropout = 0.1, num_heads = 4, num_lstm_layers = 2,
                        num_quantiles = 3) {
    self$preprocessing <- preprocessing(
      n_features = num_features,
      feature_sizes = feature_sizes,
      hidden_state_size = hidden_state_size
    )
    self$context <- static_context(
      n_features = num_features$encoder$static,
      hidden_state_size = hidden_state_size
    )
    self$temporal_selection <- temporal_selection(
      n_features = num_features,
      hidden_state_size = hidden_state_size
    )
    self$locality_enhancement <- locality_enhancement_layer(
      hidden_state_size = hidden_state_size,
      num_layers =  num_lstm_layers,
      dropout = dropout
    )
    self$temporal_attn <- temporal_self_attention(
      n_heads = num_heads,
      hidden_state_size = hidden_state_size,
      dropout = dropout
    )
    self$position_wise <- position_wise_feedforward(
      hidden_state_size = hidden_state_size,
      dropout = dropout
    )
    self$output_layer <- quantile_output_layer(
      n_quantiles = num_quantiles,
      hidden_state_size = hidden_state_size
    )
  },
  forward = function(x) {
    # We use entity embeddings [31] for categorical variables as feature representations,
    # and linear transformations for continuous variables – transforming each
    # input variable into a (dmodel)-dimensional vector which matches the dimensions
    # in subsequent layers for skip connections.
    transformed <- self$preprocessing(x)

    # In contrast with other time series forecasting architectures, the TFT is carefully
    # designed to integrate information from static metadata, using separate
    # GRN encoders to produce four different context vectors, cs, ce, cc, and ch.
    # These contect vectors are wired into various locations in the temporal fusion
    # decoder (Sec. 4.5) where static variables play an important role in processing.
    context <- self$context(transformed$encoder$static)

    # TFT is designed to provide instance-wise variable selection through the use
    # of variable selection networks applied to both static covariates and time-dependent
    # covariates. Beyond providing insights into which variables are most significant
    # for the prediction problem, variable selection also allows TFT to remove any
    # unnecessary noisy inputs which could negatively impact performance.
    transformed <- self$temporal_selection(transformed, context)

    # For instance, [12] adopts a single convolutional layer for locality enhancement
    # – extracting local patterns using the same filter across all time. However,
    # this might not be suitable for cases when observed inputs exist, due to the
    # differing number of past and future inputs. As such, we propose the application
    # of a sequence-to-sequence model to naturally handle these differences
    transformed <- self$locality_enhancement(transformed, context)

    # Besides preserving causal information flow via masking, the self-attention layer
    # allows TFT to pick up long-range dependencies that may be challenging for RNN-based
    # architectures to learn. Following the self-attention layer, an additional gating
    # layer is also applied to facilitate training
    attn_output <- self$temporal_attn(transformed)

    # we also apply a gated residual connection which skips over the entire transformer
    # block, providing a direct path to the sequence-to-sequence layer – yielding a
    # simpler model if additional complexity is not required, as shown below
    output <- self$position_wise(attn_output, transformed$decoder$known)

    # TFT also generates prediction intervals on top of point forecasts. This is
    # achieved by the simultaneous prediction of various percentiles (e.g. 10th,
    # 50th and 90th) at each time step. Quantile forecasts are generated using linear
    # transformation of the output from the temporal fusion decoder
    self$output_layer(output)
  }
)

quantile_loss <- torch::nn_module(
  initialize = function(quantiles) {
    self$quantiles <- torch::torch_tensor(sort(quantiles))$unsqueeze(1)$unsqueeze(1)
  },
  forward = function(y_pred, y_true) {
    low_res <- torch::torch_max(y_true - y_pred, other = torch::torch_zeros_like(y_pred))
    up_res <- torch::torch_max(y_pred - y_true, other = torch::torch_zeros_like(y_pred))

    quantiles <- self$quantiles$to(device = y_true$device)
    torch::torch_mean(quantiles * low_res + (1 - quantiles) * up_res)
  }
)


quantile_output_layer <- torch::nn_module(
  initialize = function(n_quantiles, hidden_state_size) {
    self$linear <- torch::nn_linear(hidden_state_size, n_quantiles)
  },
  forward = function(x) {
    self$linear(x)
  }
)

position_wise_feedforward <- torch::nn_module(
  initialize = function(hidden_state_size, dropout = 0) {
    self$grn <- time_distributed(gated_residual_network(
      input_size = hidden_state_size,
      output_size = hidden_state_size,
      hidden_state_size = hidden_state_size,
      dropout = dropout
    ))
    self$layer_norm <- torch::nn_layer_norm(
      normalized_shape = hidden_state_size
    )
    self$glu <- gated_linear_unit(
      input_size = hidden_state_size,
      output_size = hidden_state_size
    )
  },
  forward = function(x, known) {
    output <- self$grn(x)
    self$layer_norm(known + self$glu(output))
  }
)

temporal_self_attention <- torch::nn_module(
  initialize = function(n_heads, hidden_state_size, dropout) {
    self$multihead_attn <- interpretable_multihead_attention(
      n_heads = 3, hidden_state_size = hidden_state_size,
      dropout = dropout
    )
    self$glu <- gated_linear_unit(
      input_size = hidden_state_size,
      output_size = hidden_state_size
    )
    self$layer_norm <- torch::nn_layer_norm(
      normalized_shape = hidden_state_size
    )
  },
  forward = function(x) {
    full_seq <- torch::torch_cat(list(
      x$encoder$past,
      x$decoder$known
    ), dim = 2)

    attn_output <- self$multihead_attn(x$decoder$known, full_seq, full_seq)
    attn_output <- attn_output[,-x$decoder$known$size(2):N,]

    self$layer_norm(self$glu(attn_output) + x$decoder$known)
  }
)

interpretable_multihead_attention <- torch::nn_module(
  initialize = function(n_heads, hidden_state_size, dropout) {
    attn_size <- trunc(hidden_state_size / n_heads)

    self$query_layers <- seq_len(n_heads) %>%
      purrr::map(~torch::nn_linear(hidden_state_size, attn_size)) %>%
      torch::nn_module_list()
    self$key_layers <- seq_len(n_heads) %>%
      purrr::map(~torch::nn_linear(hidden_state_size, attn_size)) %>%
      torch::nn_module_list()
    self$value_layer <- torch::nn_linear(hidden_state_size, attn_size)
    self$output_layer <- torch::nn_linear(attn_size, hidden_state_size)

    self$attention <- scaled_dot_product_attention(dropout = dropout)

  },
  forward = function(q, k, v) {
    queries <- purrr::map(as.list(self$query_layers), ~.x(q))
    keys <- purrr::map(as.list(self$key_layers), ~.x(k))
    value <- self$value_layer(v)

    outputs <- purrr::map2(queries, keys, ~self$attention(.x, .y, value))
    outputs %>%
      torch::torch_stack(dim = 3) %>%
      torch::torch_mean(dim = 3) %>%
      self$output_layer()
  }
)

scaled_dot_product_attention <- torch::nn_module(
  initialize = function(dropout = 0) {
    self$dropout <- torch::nn_dropout(p = dropout)
    self$softmax <- torch::nn_softmax(dim = 3)
  },
  forward = function(q, k, v, mask = TRUE) {

    scaling_factor <- sqrt(k$size(3))
    attn <- q %>%
      torch::torch_bmm(k$permute(c(1,3,2))) %>%
      torch::torch_divide(scaling_factor)

    if (mask) {
      m <- attn %>%
        torch::torch_ones_like() %>%
        torch::torch_triu(diagonal = 1 + (attn$size(3) - attn$size(2)))
      attn <- attn$masked_fill(m$to(dtype = torch::torch_bool()), -1e9)
    }

    attn %>%
      self$softmax() %>%
      self$dropout() %>%
      torch::torch_bmm(v)
  }
)

static_enrichment_layer <- torch::nn_module(
  "static_enrichment_layer",
  initialize = function(hidden_state_size, dropout = 0) {
    self$grn <- time_distributed(gated_residual_network(
      input_size = hidden_state_size,
      output_size = hidden_state_size,
      hidden_state_size = hidden_state_size,
      dropout = dropout
    ))
  },
  forward = function(x, context) {
    list(
      encoder = list(
        past = self$grn(x$encoder$past, context$static_enrichment)
      ),
      decoder = list(
        known = self$grn(x$decoder$known, context$static_enrichment)
      )
    )
  }
)

locality_enhancement_layer <- torch::nn_module(
  "locality_enhancement_layer",
  initialize = function(hidden_state_size, num_layers, dropout = 0) {

    dropout <- if (num_layers > 1) dropout else 0

    self$encoder <- torch::nn_lstm(
      input_size = hidden_state_size,
      hidden_size = hidden_state_size,
      num_layers = num_layers,
      dropout = dropout,
      batch_first = TRUE
    )
    self$decoder <- torch::nn_lstm(
      input_size = hidden_state_size,
      hidden_size = hidden_state_size,
      num_layers = num_layers,
      dropout = dropout,
      batch_first = TRUE
    )
    self$encoder_gate <- gated_linear_unit(
      input_size = hidden_state_size,
      output_size = hidden_state_size
    )
    self$decoder_gate <- gated_linear_unit(
      input_size = hidden_state_size,
      output_size = hidden_state_size
    )
    self$encoder_norm <- torch::nn_layer_norm(
      normalized_shape = hidden_state_size
    )
    self$decoder_norm <- torch::nn_layer_norm(
      normalized_shape = hidden_state_size
    )
    self$num_layers <- num_layers
  },
  forward = function(x, context) {
    c(encoder_output, states) %<-% self$encoder(
      input = x$encoder$past,
      hx = self$expand_context(context$seq2seq_initial_state)
    )
    c(decoder_output, .) %<-% self$decoder(
      input = x$decoder$known,
      hx = states
    )

    list(
      encoder = list(
        past = encoder_output %>%
          self$encoder_gate() %>%
          magrittr::add(x$encoder$past) %>%
          self$encoder_norm()
      ),
      decoder = list(
        known = decoder_output %>%
          self$decoder_gate() %>%
          magrittr::add(x$decoder$known) %>%
          self$decoder_norm()
      )
    )
  },
  expand_context = function(context) {
    purrr::map(context, ~.x$expand(c(self$num_layers, -1, -1)))
  }
)

temporal_selection <- torch::nn_module(
  "selection",
  initialize = function(n_features, hidden_state_size) {
    self$known <- variable_selection_network(
      n_features = sum(as.numeric(n_features$decoder$known)),
      hidden_state_size = hidden_state_size
    )
    self$past <- variable_selection_network(
      n_features = sum(as.numeric(n_features$encoder$past)),
      hidden_state_size = hidden_state_size
    )
  },
  forward = function(x, context) {
    x$encoder$past <- x$encoder$past %>%
      self$past(context = context$temporal_variable_selection)
    x$decoder$known <- x$decoder$known %>%
      self$known(context = context$temporal_variable_selection)
    x
  }
)

static_context <- torch::nn_module(
  "static_context",
  initialize = function(n_features, hidden_state_size) {
    self$static <- variable_selection_network(
      n_features = sum(as.numeric(n_features)),
      hidden_state_size = hidden_state_size
    )

    self$temporal_variable_selection <- gated_residual_network(
      input_size = hidden_state_size,
      output_size = hidden_state_size,
      hidden_state_size = hidden_state_size
    )

    self$cell_state <- gated_residual_network(
      input_size = hidden_state_size,
      output_size = hidden_state_size,
      hidden_state_size = hidden_state_size
    )
    self$hidden_state <- gated_residual_network(
      input_size = hidden_state_size,
      output_size = hidden_state_size,
      hidden_state_size = hidden_state_size
    )
  },
  forward = function(x) {
    selected <- x %>%
      torch::torch_unsqueeze(dim = 2) %>%
      self$static() %>%
      torch::torch_squeeze(dim = 2)

    list(
      temporal_variable_selection = self$temporal_variable_selection(selected),
      seq2seq_initial_state = list(
        cell_state = self$cell_state(selected),
        hidden_state = self$hidden_state(selected)
      )
    )
  }
)

variable_selection_network <- torch::nn_module(
  "variable_selection_network",
  initialize = function(n_features, hidden_state_size) {
    self$global <- time_distributed(gated_residual_network(
      input_size = n_features*hidden_state_size,
      output_size = n_features,
      hidden_state_size = hidden_state_size
    ))
    self$local <- seq_len(n_features) %>%
      purrr::map(~time_distributed(gated_residual_network(
        input_size = hidden_state_size,
        output_size = hidden_state_size,
        hidden_state_size = hidden_state_size
      ))) %>%
      torch::nn_module_list()
  },
  forward = function(x, context = NULL) {
    v <- self$global(x, context = context) %>%
      torch::nnf_softmax(dim = 3) %>%
      torch::torch_unsqueeze(dim = 4)

    x <- x %>%
      torch::torch_unbind(dim = 3) %>%
      purrr::map2(as.list(self$local), ~.y(.x)) %>%
      torch::torch_stack(dim = 3)

    torch::torch_sum(v*x, dim = 3)
  }
)

gated_residual_network <- torch::nn_module(
  "gated_residual_network",
  initialize = function(input_size, output_size, hidden_state_size, dropout = 0.1) {
    self$input <- torch::nn_linear(input_size, hidden_state_size)
    self$context <- torch::nn_linear(hidden_state_size, hidden_state_size, bias = FALSE)
    self$hidden <- torch::nn_linear(hidden_state_size, hidden_state_size)
    self$dropout <- torch::nn_dropout(dropout)
    self$gate <- gated_linear_unit(hidden_state_size, output_size)
    self$norm <- torch::nn_layer_norm(output_size)
    self$elu <- torch::nn_elu()
    if (input_size == output_size) {
      self$skip <- torch::nn_identity()
    } else {
      self$skip <- torch::nn_linear(input_size, output_size)
    }
  },
  forward = function(x, context = NULL) {

    if (x$ndim > 2) {
      x <- torch::torch_flatten(x, start_dim = 2)
    }

    skip <- self$skip(x)
    x <- self$input(x)

    if (!is.null(context)) {
      x <- x + self$context(context)
    }

    hidden <- x %>%
      self$elu() %>%
      self$hidden() %>%
      self$dropout()

    self$norm(skip + self$gate(hidden))
  }
)

gated_linear_unit <- torch::nn_module(
  "gated_linear_unit",
  initialize = function(input_size, output_size) {
    self$gate <- torch::nn_sequential(
      torch::nn_linear(input_size, output_size),
      torch::nn_sigmoid()
    )
    self$activation <- torch::nn_linear(input_size, output_size)
  },
  forward = function(x) {
    self$gate(x) * self$activation(x)
  }
)

preprocessing <- torch::nn_module(
  "preprocessing",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    self$past <- preprocessing_group(
      n_features = n_features$encoder$past,
      feature_sizes = feature_sizes$past,
      hidden_state_size = hidden_state_size
    )
    self$known <- preprocessing_group(
      n_features = n_features$decoder$known,
      feature_sizes = feature_sizes$known,
      hidden_state_size = hidden_state_size
    )
    self$static <- preprocessing_group(
      n_features = n_features$encoder$static,
      feature_sizes = feature_sizes$static,
      hidden_state_size = hidden_state_size
    )
  },
  forward = function(x) {
    list(
      encoder = list(
        past = self$past(x$encoder$past),
        static = x$encoder$static %>%
          purrr::map(~torch::torch_unsqueeze(.x, dim = 2)) %>%
          self$static() %>%
          torch::torch_squeeze(dim = 2)
      ),
      decoder = list(
        known = self$known(x$decoder$known)
      )
    )
  }
)


# Preprocess a group of time-varying variables
#
# Handles both numeric and categorical variables.
# Each numeric variables passes trough a linear transformation that is shared
# accross every time step.
# Categorical variables are represented trough embeddings.
preprocessing_group <- torch::nn_module(
  "preprocessing_group",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    self$num <- linear_preprocessing(n_features$num, hidden_state_size)
    self$cat <- embedding_preprocessing(n_features$cat, feature_sizes, hidden_state_size)
  },
  forward = function(x) {
    x$num <- self$num(x$num)
    x$cat <- self$cat(x$cat)
    torch::torch_cat(x, dim = 3)
  }
)

linear_preprocessing <- torch::nn_module(
  "linear_preprocessing",
  initialize = function(n_features, hidden_state_size) {
    self$linears <- seq_len(n_features) %>%
      purrr::map(~time_distributed(torch::nn_linear(1, hidden_state_size))) %>%
      torch::nn_module_list()
  },
  # @param x `Tensor[batch, time_steps, n_features]`
  forward = function(x) {
    if (x$size(3) == 0) return(NULL)
    x %>%
      torch::torch_unsqueeze(dim = 4) %>%
      torch::torch_unbind(dim = 3) %>%
      purrr::imap(~self$linears[[.y]](.x)) %>%
      torch::torch_stack(dim = 3)
  }
)

embedding_preprocessing <- torch::nn_module(
  "embedding_preprocessing",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    self$embeddings <- feature_sizes %>%
      purrr::map(~time_distributed(torch::nn_embedding(.x, hidden_state_size))) %>%
      torch::nn_module_list()
  },
  # @param x `Tensor[batch, time_steps, n_features]`
  forward = function(x) {
    if (x$size(3) == 0) return(NULL)
    x %>%
      torch::torch_unbind(dim = 3) %>%
      purrr::imap(~self$embeddings[[.y]](.x)) %>%
      torch::torch_stack(dim = 3)
  }
)

time_distributed <- torch::nn_module(
  "time_distributed",
  initialize = function(module) {
    self$module <- module
  },
  forward = function(x, ...) {
    extra_args <- list(...)
    x %>%
      torch::torch_unbind(dim = 2) %>%
      purrr::map(~rlang::exec(self$module, .x, !!!extra_args)) %>%
      torch::torch_stack(dim = 2)
  }
)
