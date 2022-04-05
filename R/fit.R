
#' Temporal Fusion Transformer
#'
#' @export
tft <- function(x, ...) {
  UseMethod("tft")
}

#' @export
tft.default <- function(x, ...) {
  cli::cli_abort(
    "{.var tft} is not defined for objects with class {.cls {class(x)}}.")
}

#' @export
tft.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  tft_bridge(processed)
}

#' @export
tft.recipe <- function(x, data, ...) {
  processed <- hardhat::mold(x, data)
  tft_bridge(processed)
}

tft_bridge <- function(processed) {

  if (is.null(attr(processed$outcomes, "roles"))) {
    cli::cli_warn(c(
      "The provided {.cls {class(processed$outcomes)}} doesn't include role information.",
      "i" = "You should use {.var step_include_roles} in the {.cls recipe} so the role information is correctly passed to the model."
    ))
  }

  tft_impl(
    dplyr::bind_cols(processed$predictors, processed$outcomes),
    attr(processed$predictors, "roles")
  )
}

tft_impl <- function(x, roles) {
  dataset <- time_series_dataset(x, roles, lookback = 120, assess_stop = 4)

  n_features <- get_n_features(dataset[1][[1]])
  hidden_state_size <- 16
  feature_sizes <- dataset$feature_sizes

  result <- temporal_fusion_transformer %>%
    luz::setup(
      loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
      optimizer = torch::optim_adam
    ) %>%
    luz::set_hparams(
      num_features = n_features,
      feature_sizes = feature_sizes,
      hidden_state_size = 16,
      dropout = 0.1,
      num_quantiles = 3,
      num_heads = 1
    ) %>%
    luz::set_opt_hparams(
      lr = 0.03
    ) %>%
    fit(dataset, epochs = 5, dataloader_options = list(
      batch_size = 256, num_workers = 0
    ))

  result
}


