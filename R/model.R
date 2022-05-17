#' Temporal Fusion transformer
#'
#' @param dataset A [torch::dataset()] created with [time_series_dataset()].
#'  This is required because the model depends on some information that is
#'  created/defined in the dataset.
#'
#' @describeIn tft Create the tft module
#' @inheritParams tft
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
