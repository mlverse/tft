
#' Temporal Fusion Transformer
#'
#' @param x A recipe containing [step_include_roles()] as the last step. Can
#'  also be a data.frame, but expect it to have a `recipe` attribute attribute
#'  containing the `recipe` that generated it via [recipes::bake()] or
#'  [recipes::juice()].
#' @param y A data.frame containing the response variable.
#' @param data Dataset used for training the model.
#' @param ... Additional arguments passed to [tft_config()].
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
  feature_sizes <- dataset$feature_sizes

  callbacks <- list()
  if (config$gradient_clip_norm > 0) {
    callbacks[["gradient_clip"]] <- luz::luz_callback_gradient_clip(
      max_norm = config$gradient_clip_norm
    )
  }

  if (config$learn_rate_decay[1] > 0) {
    callbacks[["lr_scheduler"]] <- luz::luz_callback_lr_scheduler(
      torch::lr_step, step_size = config$learn_rate_decay[2],
      gamma = config$learn_rate_decay[1]
    )
  }

  result <- temporal_fusion_transformer %>%
    luz::setup(
      loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
      optimizer = config$optimizer
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
    ) %>%
    generics::fit(
      dataset, epochs = config$epochs, verbose = config$verbose,
      callbacks = callbacks,
      dataloader_options = list(
        batch_size = config$batch_size, num_workers = 0
      )
    )

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

#' Configuration for the Temporal Fusion Transformer network
#'
#' @returns A list with the configuration parameters.
#'
#' @param lookback Number of timesteps that are used as historic data for
#'  prediction.
#' @param horizon Number of timesteps ahead that will be predicted by the
#'  model.
#' @param hidden_state_size Hidden size of network which is its main hyperparameter
#'   and can range from 8 to 512. It's also known as `d_model` across the paper.
#' @param num_attention_heads Number of attention heads in the Multi-head attention layer.
#'   The paper refer to it as `m_H`. `4` is a good default.
#' @param num_lstm_layers Number of LSTM layers used in the Locality Enhancement
#'   Layer. Usually 2 is good enough.
#' @param dropout Dropout rate used in many places in the architecture.
#' @param batch_size How many samples per batch to load.
#' @param epochs Maximum number of epochs for training the model.
#' @param optimizer Optimizer used for training. Can be a string with 'adam', 'sgd',
#'   or 'adagrad'. Can also be a [torch::optimizer()].
#' @param learn_rate Leaning rate used by the optimizer.
#' @param learn_rate_decay Decrease the learning rate by this factor each epoch.
#'  Can also be a vector with 2 elements. In this case we decrease the learning by
#'  the `x[1]` every `x[2]` epochs - (where `x` is the `learn_rate_decay` vector.)
#'  Use `FALSE` or any negative number to disable.
#' @param gradient_clip_norm Maximum norm of the gradients. Passed on to
#'   [luz::luz_callback_gradient_clip()]. If <= 0 or `FALSE` then no gradient
#'   clipping is performed.
#' @param quantiles A numeric vector with 3 quantiles for the quantile loss.
#'   The first is treated as lower bound of the interval, the second as the
#'   point prediction and the thir as the upper bound.
#' @param verbose Logical value stating if the model should produce status
#'   outputs, like a progress bar, during training.
#'
#' @describeIn tft Configuration configuration options for tft.
#'
#' @export
tft_config <- function(lookback, horizon, hidden_state_size = 16, num_attention_heads = 4,
                       num_lstm_layers = 2, dropout = 0.1, batch_size = 256,
                       epochs = 5, optimizer = "adam", learn_rate = 0.01,
                       learn_rate_decay = c(0.1, 5), gradient_clip_norm = 0.1,
                       quantiles = c(0.1, 0.5, 0.9), verbose = FALSE) {

  if (rlang::is_false(learn_rate_decay)) {
    learn_rate_decay <- -1
  }

  if (rlang::is_scalar_double(learn_rate_decay)) {
    learn_rate_decay <- c(learn_rate_decay, 1)
  }

  if (rlang::is_false(gradient_clip_norm)) {
    gradient_clip_norm <- -1
  }

  if (rlang::is_scalar_character(optimizer)) {
    optimizer <- switch (optimizer,
      "adam" = torch::optim_adam,
      "sgd" = torch::optim_sgd,
      "adagrad" = torch::optim_adagrad
    )
  }

  list(
    lookback = lookback,
    horizon = horizon,
    hidden_state_size = hidden_state_size,
    num_attention_heads = num_attention_heads,
    num_lstm_layers = num_lstm_layers,
    dropout = dropout,
    gradient_clip_norm = gradient_clip_norm,
    optimizer = optimizer,
    learn_rate = learn_rate,
    learn_rate_decay = learn_rate_decay,
    batch_size = batch_size,
    epochs = epochs,
    quantiles = quantiles,
    verbose = verbose
  )
}
