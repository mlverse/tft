
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
  config <- tft_config(...)
  tft_bridge(processed, config)
}

#' @export
tft.recipe <- function(x, data, ...) {
  config <- tft_config(...)
  processed <- hardhat::mold(x, tibble::as_tibble(data))
  tft_bridge(processed, config)
}

tft_bridge <- function(processed, config) {

  if (is.null(attr(processed$outcomes, "roles"))) {
    cli::cli_abort(c(
      "The provided {.cls {class(processed$outcomes)}} doesn't include role information.",
      "i" = "You should use {.var step_include_roles} in the {.cls recipe} so the role information is correctly passed to the model."
    ))
  }

  data <- dplyr::bind_cols(processed$predictors, processed$outcomes)

  result <- tft_impl(
    x = data,
    recipe = attr(processed$predictors, "recipe"),
    config = config
  )

  new_tft(
    module = result$module,
    past_data = result$past_data,
    normalization = result$normalization,
    recipe = attr(processed$predictors, "recipe"),
    config = config,
    blueprint = processed$blueprint
  )
}

new_tft <- function(module, past_data, normalization, recipe, config, blueprint) {
  hardhat::new_model(
    module = module,
    past_data = past_data,
    normalization = normalization,
    recipe = recipe,
    config = config,
    blueprint = blueprint,
    class = "tft"
  )
}

tft_impl <- function(x, recipe, config) {

  normalization <- normalize_outcome(
    x = x,
    keys = get_variables_with_role(recipe$term_info, "key"),
    outcome = get_variables_with_role(recipe$term_info, "outcome")
  )

  x <- normalization$x
  normalization <- normalization$constant

  dataset <- time_series_dataset(
    x, recipe$term_info,
    lookback = config$lookback,
    assess_stop = config$horizon
  )

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
    generics::fit(dataset, epochs = config$epochs, verbose = config$verbose, dataloader_options = list(
      batch_size = config$batch_size, num_workers = 0
    ))

  list(past_data = dataset$df, module = result, normalization = normalization)
}

# by default we normalize the outcomes per group.
normalize_outcome <- function(x, keys, outcome) {
  outcome <- rlang::sym(outcome)
  constants <- x %>%
    tibble::as_tibble() %>%
    dplyr::ungroup() %>%
    dplyr::group_by(!!!rlang::syms(keys)) %>%
    dplyr::summarise(.groups = "drop",
      ..mean := mean({{outcome}}),
      ..sd := sd({{outcome}})
    )

  x <- x %>%
    dplyr::left_join(constants, by = keys) %>%
    dplyr::mutate({{outcome}} := ({{outcome}} - ..mean)/..sd) %>%
    dplyr::select(-..mean, -..sd)

  list(constants = constants, x = x)
}

unnormalize_outcome <- function(x, constants, outcome) {
  keys <- names(constants)
  keys <- keys[!keys %in% c("..mean", "..sd")]

  outcome <- rlang::sym(outcome)

  x %>%
    dplyr::left_join(constants, by = keys) %>%
    dplyr::mutate({{outcome}} := {{outcome}} *..sd + ..mean) %>%
    dplyr::select(-..mean, -..sd)
}

tft_config <- function(lookback, horizon, batch_size = 256, epochs = 5,
                       verbose = FALSE) {
  list(
    lookback = lookback,
    horizon = horizon,
    batch_size = batch_size,
    epochs = epochs,
    verbose = verbose
  )
}
