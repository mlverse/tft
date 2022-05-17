#' Temporal Fusion transformer
#'
#' @param spec A spec created with [tft_dataset_spec()].
#'  This is required because the model depends on some information that is
#'  created/defined in the dataset.
#' @param ... Additional parameters passed to [tft_config2()].
#'
#' @describeIn temporal_fusion_transformer Create the tft module
#'
#' @export
temporal_fusion_transformer <- function(spec, ...) {

  if (!inherits(spec, "prepared_tft_dataset_spec")) {
    cli::cli_abort(c(
      "{.var spec} must be an object created with {.var tft_dataset_spec()}",
      x = "Got an object with {.cls {class(dataset)}}."
    ))
  }

  dataset <- spec$dataset
  config <- tft_config2(...)

  n_features <- get_n_features(dataset[1][[1]])
  feature_sizes <- dataset$feature_sizes

  module <- temporal_fusion_transformer_model %>%
    luz::setup(
      loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
      optimizer = config$optimizer,
      metrics = list(
        luz_quantile_loss(quantile = 0.1,1),
        luz_quantile_loss(quantile = 0.5,2),
        luz_quantile_loss(quantile = 0.9,3)
      )
    ) %>%
    luz::set_hparams(
      num_features = n_features,
      feature_sizes = feature_sizes,
      hidden_state_size = config$hidden_state_size,
      dropout = config$dropout,
      num_quantiles = 3,
      num_heads = config$num_attention_heads,
      num_lstm_layers = config$num_lstm_layers
    ) %>%
    luz::set_opt_hparams(
      lr = config$learn_rate
    )

  class(module) <- c("tft_module", class(module))
  module
}

#' Fit the Temporal Fusion Transformer module
#'
#' @param object a TFT module created with [temporal_fusion_transformer()].
#' @param ... Arguments passed to [luz::fit.luz_module_generator()].
#'
#' @export
fit.tft_module <- function(object, ...) {
  out <- NextMethod()
  class(out) <- c("tft_result", class(out))
  out
}

#' Configuration for the tft model
#'
#' @param hidden_state_size Hidden size of network which is its main hyperparameter
#'   and can range from 8 to 512. It's also known as `d_model` across the paper.
#' @param num_attention_heads Number of attention heads in the Multi-head attention layer.
#'   The paper refer to it as `m_H`. `4` is a good default.
#' @param num_lstm_layers Number of LSTM layers used in the Locality Enhancement
#'   Layer. Usually 2 is good enough.
#' @param dropout Dropout rate used in many places in the architecture.
#' @param optimizer Optimizer used for training. Can be a string with 'adam', 'sgd',
#'   or 'adagrad'. Can also be a [torch::optimizer()].
#' @param learn_rate Leaning rate used by the optimizer.
#' @param quantiles A numeric vector with 3 quantiles for the quantile loss.
#'   The first is treated as lower bound of the interval, the second as the
#'   point prediction and the thir as the upper bound.
#'
#' @describeIn temporal_fusion_transformer Configuration for the Temporal Fusion Transformer
#' @export
tft_config2 <- function(hidden_state_size = 16,
                        num_attention_heads = 4,
                        num_lstm_layers = 2, dropout = 0.1,
                        optimizer = "adam",
                        learn_rate = 0.01,
                        quantiles = c(0.1, 0.5, 0.9)) {

  if (rlang::is_scalar_character(optimizer)) {
    optimizer <- switch (optimizer,
                         "adam" = torch::optim_adam,
                         "sgd" = torch::optim_sgd,
                         "adagrad" = torch::optim_adagrad
    )
  }

  list(
    hidden_state_size = hidden_state_size,
    num_attention_heads = num_attention_heads,
    num_lstm_layers = num_lstm_layers,
    dropout = dropout,
    optimizer = optimizer,
    learn_rate = learn_rate,
    quantiles = quantiles
  )
}
