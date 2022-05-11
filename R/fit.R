
#' Temporal Fusion Transformer
#'
#' @param x A recipe or data.frame that will be used to train the model.
#' @inheritParams parsnip::linear_reg
#' @param ... Additional arguments passed to [tft_config()].
#'
#' @seealso [predict.tft()] for how to create predictions.
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
  data <- tibble::as_tibble(data)
  processed <- hardhat::mold(x, data)
  tft_bridge(processed, config)
}

tft_bridge <- function(processed, config) {

  config$input_types <- evaluate_types(processed$predictors, config$input_types)
  config$input_types[["outcome"]] <- names(processed$outcome)

  processed_data <- dplyr::bind_cols(processed$predictors, processed$outcomes)

  result <- tft_impl(
    x = processed_data,
    config = config
  )

  new_tft(
    module = result$module,
    past_data = processed_data,
    normalization = result$normalization,
    config = config,
    blueprint = processed$blueprint
  )
}

new_tft <- function(module, past_data, normalization, config, blueprint) {
  hardhat::new_model(
    module = module,
    past_data = past_data,
    normalization = normalization,
    config = config,
    blueprint = blueprint,
    class = "tft",
    .serialized_model = model_to_raw(module)
  )
}

tft_impl <- function(x, config) {

  normalization <- normalize_outcome(
    x = x,
    keys = get_variables_with_role(config$input_types, "keys"),
    outcome = get_variables_with_role(config$input_types, "outcome")
  )

  x <- normalization$x
  normalization <- normalization$constant

  dataset <- time_series_dataset(
    x, config$input_types,
    lookback = config$lookback,
    assess_stop = config$horizon,
    subsample = config$subsample
  )

  n_features <- get_n_features(dataset[1][[1]])
  feature_sizes <- dataset$feature_sizes

  callbacks <- config$callbacks
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

  result <- temporal_fusion_transformer_model %>%
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
    ) %>%
    generics::fit(
      dataset, epochs = config$epochs, verbose = config$verbose,
      callbacks = callbacks,
      dataloader_options = list(
        batch_size = config$batch_size, num_workers = config$num_workers
      )
    )

  list(module = result, normalization = normalization)
}

# by default we normalize the outcomes per group.
#' @importFrom stats sd
#' @importFrom utils tail
normalize_outcome <- function(x, keys, outcome, constants = NULL) {
  outcome <- rlang::sym(outcome)

  if (is.null(constants)) {
    constants <- x %>%
      tibble::as_tibble() %>%
      dplyr::ungroup() %>%
      dplyr::group_by(!!!rlang::syms(keys)) %>%
      dplyr::summarise(.groups = "drop",
                       ..mean := mean({{outcome}}),
                       ..sd := sd({{outcome}})
      )
  }

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
#' @param input_types A list with the elements `index`, `keys`, `static`, `known` and
#'  `unknown`. It's recommended to use [covariates_spec()] to create it. Each element
#'  should be a character vector containing the names of
#'  the columns that are used for each role in the TFT model. `index` must be a date
#'  column and `keys` are columns that allow one to identidy each time-series.
#'  `index` and `keys` must be specified. If a column that exists in the data.frame
#'  doesn't appear in this list, then it's considered `unknown`.
#' @param subsample Subsample from all possible slices. An integer with the number
#'  of samples or a proportion.
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
#' @param num_workers Number of parallel workers for preprocessing data.
#' @param callbacks Additional callbacks passed when fitting the module with
#'   luz.
#' @param verbose Logical value stating if the model should produce status
#'   outputs, like a progress bar, during training.
#'
#' @describeIn tft Configuration configuration options for tft.
#'
#' @export
tft_config <- function(lookback, horizon, input_types, subsample = 1,
                       hidden_state_size = 16, num_attention_heads = 4,
                       num_lstm_layers = 2, dropout = 0.1, batch_size = 256,
                       epochs = 5, optimizer = "adam", learn_rate = 0.01,
                       learn_rate_decay = c(0.1, 5), gradient_clip_norm = 0.1,
                       quantiles = c(0.1, 0.5, 0.9), num_workers = 0,
                       callbacks = list(),
                       verbose = FALSE) {

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
    subsample = subsample,
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
    num_workers = num_workers,
    callbacks = callbacks,
    verbose = verbose,
    input_types = input_types
  )
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  luz::luz_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

is_null_external_pointer <- function(pointer) {
  a <- attributes(pointer)
  attributes(pointer) <- NULL
  out <- identical(pointer, methods::new("externalptr"))
  attributes(pointer) <- a
  out
}

reload_model <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- luz::luz_load(con)
  module
}

#' Create a specification of covariates types
#'
#' @param index A column that identifies the time variable. Usually a date column.
#' @param keys A set of colums that uniquely identify each time series.
#' @param static A set of colums that will be treated as static predictors by the
#'  model. Ie, those are variables that don't vary in time.
#' @param known A set of columns that are known for every possible input `index`.
#'  Examples are 'day of the week' or 'is_holiday' flags that are always known even
#'  for future timesteps.
#' @param unknown A set of columns that are considered to be unknown variables.
#'  By default, all columns that don't fit in any of the previous parameters are
#'  considered unknown, so you don't need to specify it manually.
#'
#' @export
covariates_spec <- function(index, keys, static = NULL, known = NULL, unknown = NULL) {
  make_input_types(
    index = {{index}},
    keys = {{keys}},
    static = {{static}},
    known = {{known}},
    unknown = {{unknown}}
  )
}

make_input_types <- function(index, keys, static = NULL, known = NULL,
                             unknown = NULL) {
  output <- list(
    index = rlang::enexpr(index),
    keys = rlang::enexpr(keys),
    static = rlang::enexpr(static),
    known = rlang::enexpr(known),
    unknown = rlang::enexpr(unknown)
  )
  output
}

evaluate_types <- function(data, types) {
  types <- lapply(types, function(x) {
    colnames(dplyr::select(data, !!x))
  })
  # Non-specified variables are considered unknown.
  unknown <- names(data)[!names(data) %in% unlist(types)]
  types[["unknown"]] <- c(types[["unknown"]], unknown)
  types
}
