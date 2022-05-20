
#' Predict for TFT
#'
#' @importFrom stats predict
#' @inheritParams stats::predict
#'
#' @param new_data A [data.frame()] containing a dataset to generate predictions
#'   for. In general it's used to pass static and known information to generate
#'   forecasts.
#' @param past_data A [data.frame()] with past information for creating the
#'  predictions. It should include at least `lookback` values - but can be more.
#'  It's concatenated with `new_data` before passing forward. If `NULL`, the
#'  data used to train the model is used.
#' @param ... other arguments passed to the [predict.luz_module_fitted()].
#'
#' @export
predict.tft_result <- function(object, new_data = NULL, ..., past_data = NULL) {

  if (!is.null(new_data))
    new_data <- tibble::as_tibble(new_data)
  if (!is.null(past_data))
    past_data <- tibble::as_tibble(past_data)

  if (is_null_external_pointer(object$model$.check)) {
    reloaded <- reload_model(object$.serialized)
    object$model$load_state_dict(reloaded$model$state_dict())
  }

  dataset <- transform(object$spec, past_data = past_data, new_data = new_data,
                       .verify = TRUE)
  preds <- NextMethod("predict", object, dataset, ...)

  predictions <- (preds$cpu()) %>%
    torch::torch_unbind(dim = 1) %>%
    torch::torch_cat(dim =  1) %>%
    as.matrix() %>%
    tibble::as_tibble(.name_repair = "minimal") %>%
    rlang::set_names(c(".pred_lower", ".pred", ".pred_upper"))

  input_types <- object$spec$config$input_types

  obs <- dataset$slices %>%
    purrr::map_dfr(function(.x) {
      res <- tibble::as_tibble(dataset$df[.x$decoder,]) %>%
        dplyr::select(dplyr::all_of(
          get_variables_with_role(input_types, c("keys", "index"))
        ))
      res
    })

  out <- dplyr::bind_cols(predictions, tibble::as_tibble(obs))
  out <- new_data %>%
    dplyr::left_join(
      out,
      by = c(get_variables_with_role(input_types, c("keys", "index")))
    )

  for (var in names(predictions)) {
    out <- unnormalize_outcome(out, object$spec$normalization, outcome = var)
  }

  dplyr::select(out, dplyr::starts_with(".pred"))
}

adjust_past_data <- function(past_data, blueprint) {
  past_data <- hardhat::forge(past_data, blueprint, outcomes = TRUE)
  past_data <- dplyr::bind_cols(past_data$predictors, past_data$outcomes)
  past_data
}

adjust_new_data <- function(new_data, input_types, blueprint, outcomes = FALSE) {

  ptypes <- blueprint$ptypes
  new_data <- tibble::as_tibble(new_data)

  keys <- get_variables_with_role(input_types, "keys")
  index <- get_variables_with_role(input_types, "index")

  # we make sure that new data includes at least all index and keys.
  if (!all(c(keys, index) %in% names(new_data))) {
    cli::cli_abort(c(
      "New data does not include all the {.var keys} and {.var index}.",
      "x" = "Missing {.var {setdiff(c(keys, index), names(new_data))}}"
    ))
  }

  known <- dplyr::intersect(
    get_variables_with_role(input_types, c("static", "known")),
    colnames(ptypes$predictors)
  )
  for (var in known) {
    if (is.null(new_data[[var]])) {
      cli::cli_abort(c(
        "Known or static variable is missing from {.var new_data}.",
        "x" = "Check for {.var {var}}."
      ))
    }
  }

  # we now add all `predictors` to the new_obs dataset
  predictors <- dplyr::intersect(
    get_variables_with_role(input_types, c("unknown", "static", "known", "keys")),
    colnames(ptypes$predictors)
  )

  for (var in predictors) {
    if (is.null(new_data[[var]])) {
      new_data[[var]] <- NA
    }
  }

  # in this case we need to add the outcome variable if it's not present in the
  # dataset
  if (outcomes) {
    outcome <- get_variables_with_role(input_types, "outcome")
    if (is.null(new_data[[outcome]])) {
      new_data[[outcome]] <- NA
    }
  }

  out <- hardhat::forge(new_data, blueprint, outcomes = outcomes)
  dplyr::bind_cols(out$predictors, out$outcomes)
}

verify_new_data <- function(new_data, past_data, object) {

  data <- list(new_data = new_data, past_data = past_data)
  input_types <- object$config$input_types

  # we also have to make sure that all static and known predictors exist
  # in the new datatset
  #term_info <- recipe$term_info
  known <- get_variables_with_role(input_types, c("keys", "index", "static", "known"))

  for (var in known) {
    for (d in c("new_data", "past_data")) {
      if (any(is.na(data[[d]][[var]]))) {
        cli::cli_abort(c(
          "Found missing values in at least one known in variable in the {.var {d}}.",
          "i" = "This kind of variables should be fully known for future inputs.",
          "x" = "Check variable {.var {var}}."
        ))
      }
    }
  }


  possible_dates <- future_data(past_data, horizon = object$config$horizon,
                                input_types = input_types)
  not_allowed <- new_data %>%
    dplyr::anti_join(
      possible_dates,
      by = c(tsibble::index_var(possible_dates), tsibble::key_vars(possible_dates))
    )
  if (nrow(not_allowed) > 0) {
    cli::cli_abort(c(
      "{.var new_data} includes obs that we can't generate predictions.",
      "x" = "Found {.var {nrow(not_allowed)}} observations."
    ))
  }

  # we verify that all groups have the entire prediction range

  counts <- new_data %>%
    dplyr::group_by(!!!rlang::syms(get_variables_with_role(input_types, "keys"))) %>%
    dplyr::count()

  if (any(counts$n != object$config$horizon)) {
    n_groups <- sum(counts$n != object$config$horizon)
    h <- object$config$horizon
    cli::cli_abort(c(
      "At least one group doesn't have an entire range for prediction.",
      "x" = "Found {.var {n_groups}} group that don't have {.var {h}} dates."
    ))
  }

  invisible(NULL)
}


#' Generate forecasts for TFT models
#'
#' `forecast` can only be used if the model object doesn't include `known`
#' predictors that must exist in the data. It's fine if a `recipe` passed to
#' [tft_dataset_spec()] computes `known` predictors though.
#'
#' @param object The `tft_result` object that will be used to create predictions.
#' @param horizon Number of time steps ahead to generate predictions.
#'
#' @importFrom generics forecast
#' @export
forecast.tft_result <- function(object, horizon = NULL) {

  if (is.null(horizon)) {
    horizon <- object$spec$config$horizon
  }

  if (horizon > object$spec$config$horizon) {
    cli::cli_abort(c(
      "{.var horizon} is larger than the maximum allowed.",
      "x" = "Got {horizon}, max allowed is {object$horizon}."
    ))
  }

  f_data <- future_data(object$spec$past_data, horizon = object$spec$config$horizon,
                        input_types = object$spec$config$input_types)
  f_data <- tibble::as_tibble(f_data)

  pred <- dplyr::bind_cols(
    f_data,
    predict(object, new_data = f_data)
  )

  future_data <- future_data(object$spec$past_data, horizon = horizon,
                             input_types = object$spec$config$input_types) %>%
    dplyr::left_join(pred, by = names(.)) %>%
    tibble::as_tibble() %>%
    structure(class = c("tft_forecast", class(.)))
}

future_data <- function(past_data, horizon, input_types = NULL) {
  if (!tsibble::is_tsibble(past_data))
    past_data <- make_tsibble(past_data, input_types)

  x <- tsibble::new_data(past_data, horizon)
  index <- tsibble::index(x)
  x %>%
    tsibble::group_by_key() %>%
    dplyr::mutate(
      ".min_index" = min({{ index }})
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(.min_index >= max(.min_index)) %>%
    dplyr::select(-.min_index)
}

#' Defines rolling slices
#'
#' Sometimes your validation or testing data has more values than the `horizon`
#' of your model but you still want to create predictions for each time step on
#' them.
#'
#' This function will combine your `past_data` (that can also include your training data)
#' and create slices so you create predictions for each value in `new_data`.
#'
#' @inheritParams predict.tft_result
#' @param step Default is the step to be the same as the horizon o the model,
#'  that way we have one prediction per slice.
#'
#' @export
rolling_predict <- function(object, past_data, new_data, step = NULL) {

  if (!inherits(object, "tft_result")) {
    cli::cli_abort(c(
      "{.var object} must be a {.cls tft_result}.",
      x = "Got an object with class {.cls {class(object)}}"
    ))
  }

  if (is.null(step)) {
    step <- object$spec$config$horizon
  }

  # make sure past_data only includes groups that exist in `new_data`
  past_data <- past_data %>%
    tibble::as_tibble() %>%
    dplyr::semi_join(
      tibble::as_tibble(new_data),
      by = get_variables_with_role(object$spec$config$input_types, "keys")
    )

  past_data <- get_last_lookback_interval(
    tibble::as_tibble(past_data),
    lookback = object$spec$config$lookback,
    input_types = object$spec$config$input_types
  )

  data <- dplyr::bind_rows(past_data, tibble::as_tibble(new_data))
  index_col <- get_variables_with_role(object$spec$config$input_types, "index")

  slices <- rolling_slices(
    data[[index_col]],
    lookback = object$spec$config$lookback,
    horizon = object$spec$config$horizon,
    step = step,
    start_date = min(new_data[[index_col]]),
    period = get_period(past_data, input_types = object$spec$config$input_types)
  )

  predictions <- list()
  for (s in slices) {
    p_data <- data[s$encoder,]
    n_data <- data[s$decoder,]

    nm <- as.character(max(p_data[[index_col]]))
    pred <- predict(object, new_data = n_data, past_data = p_data)

    predictions[[nm]] <- tibble::tibble(
      past_data = list(p_data),
      new_data = list(n_data),
      .pred = list(pred)
    )
  }
  dplyr::bind_rows(predictions)
}

get_last_lookback_interval <- function(past_data, lookback, input_types) {
  index_col <- get_variables_with_role(input_types, "index")
  # we only need the last `lookback` interval from the past_data.
  last_index <- max(past_data[[index_col]])
  interval <- get_period(past_data, input_types)
  last_index <- last_index - lookback*interval

  # now filter the past_data
  past_data <- past_data %>%
    dplyr::filter(.data[[index_col]] > last_index)
}

rolling_slices <- function(index, lookback, horizon, step, start_date,
                           period) {
  # list start and ending dates for past and pred
  slices <- list()
  max_date <- max(index)
  while (!start_date > max_date) {
    slices[[length(slices) + 1]] <- list(
      encoder = which(index > (start_date - lookback*period - 1) &
                        index < (start_date)),
      decoder = which(index >= start_date &
                        index < (start_date + horizon*period))
    )
    start_date <- start_date + step*period
  }
  slices
}

get_period <- function(data, input_types) {
  data %>%
    make_tsibble(input_types) %>%
    tsibble::interval() %>%
    lubridate::as.period()
}

make_prediction_dataset <- function(new_data, past_data, config) {
  input_types <- config$input_types
  key_cols <- get_variables_with_role(input_types, "keys")
  index_col <- get_variables_with_role(input_types, "index")

  # only grab past data for keys that exist in the new data
  past_data <- new_data %>%
    dplyr::select(!!!rlang::syms(key_cols)) %>%
    dplyr::distinct() %>%
    dplyr::left_join(
      tibble::as_tibble(past_data),
      by = key_cols
    )

  if (sum(is.na(past_data[[index_col]]))) {
    cli::cli_warn(c(
      "No past information for a few groups in `new_data`. They will be dropped.",
      i = "{{sum(is.na(past_data[[index_col]]))}} groups will be dropped."
    ))
    past_data <- past_data[!is.na(past_data[[index_col]]),]
  }

  # now filter the past_data
  past_data <- get_last_lookback_interval(
    past_data,
    lookback = config$lookback,
    input_types = input_types
  )

  time_series_dataset_generator(
    dplyr::bind_rows(past_data, new_data),
    input_types,
    lookback = config$lookback,
    assess_stop = config$horizon,
    step = 1L,
    subsample = config$subsample
  )
}
