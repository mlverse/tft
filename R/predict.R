
#' Create predictions for TFT models
#'
#' @importFrom stats predict
#' @inheritParams stats::predict
#' @param new_data A [data.frame()] containing a dataset to generate predictions
#'   for. In general it's used to pass static and known information to generate
#'   forecasts.
#' @param type Currently only `'numeric'` is accepted but this might change in
#'  the future if we end up supporting classification.
#'
#' @param step Step for predictions when using `mode='full'`.
#' @export
predict.tft <- function(object, new_data, type = "numeric", ...,
                        past_data = NULL) {

  if (is_null_external_pointer(object$module$model$.check)) {
    object$module <- reload_model(object$.serialized_model)
  }

  new_data <- adjust_new_data(
    new_data,
    object$config$input_types,
    object$blueprint$ptypes
  )

  new_data <- hardhat::forge(new_data, object$blueprint)$predictors

  # if past data includes a recipe, it means it has already been
  # preprocessed, thus we don't need to reapply the preprocessing.
  if (is.null(past_data)) {
    past_data <- object$past_data
  } else {
    past_data <- hardhat::forge(past_data, object$blueprint)
    past_data <- dplyr::bind_cols(past_data$predictors, past_data$outcome)
  }

  verify_new_data(new_data, past_data, object)
  out <- predict_impl(object, new_data, past_data, step)
  out
}

predict_impl <- function(object, new_data, past_data, step) {

  input_types <- object$config$input_types
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

  # we only need the last `lookback` interval from the past_data.
  last_index <- max(past_data[[index_col]])
  interval <- lubridate::as.period(tsibble::interval(make_tsibble(past_data, input_types)))
  last_index <- last_index - object$config$lookback*interval

  # now filter the past_data
  past_data <- past_data %>%
    dplyr::filter(.data[[index_col]] > last_index)

  past_data <- normalize_outcome(
    x = past_data,
    keys = get_variables_with_role(object$config$input_types, "keys"),
    outcome = get_variables_with_role(object$config$input_types, "outcome"),
    constants = object$normalization
  )$x

  dataset <- time_series_dataset(
    dplyr::bind_rows(past_data, new_data),
    input_types,
    lookback = object$config$lookback,
    assess_stop = object$config$horizon,
    step = 1L
  )

  res <- predict(object$module, dataset)

  predictions <- (res$cpu()) %>%
    torch::torch_unbind(dim = 1) %>%
    torch::torch_cat(dim =  1) %>%
    as.matrix() %>%
    tibble::as_tibble(.name_repair = "minimal") %>%
    rlang::set_names(c(".pred_lower", ".pred", ".pred_upper"))

  obs <- dataset$slices %>%
    purrr::map_dfr(function(.x) {
      res <- tibble::as_tibble(dataset$df[.x$decoder,]) %>%
        dplyr::select(dplyr::all_of(
          get_variables_with_role(
            object$config$input_types, c("keys", "index")
          )
        ))
      res
    })

  out <- dplyr::bind_cols(predictions, tibble::as_tibble(obs))
  out <- new_data %>%
    dplyr::left_join(
      out,
      by = c(get_variables_with_role(object$config$input_types, c("keys", "index")))
    )

  for (var in names(predictions)) {
    out <- unnormalize_outcome(out, object$normalization, outcome = var)
  }

  dplyr::select(out, dplyr::starts_with(".pred"))
}

adjust_new_data <- function(new_data, input_types, ptypes) {

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

  new_data
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


#' @importFrom generics forecast
#' @export
forecast.tft <- function(object, horizon = NULL) {

  if (is.null(horizon)) {
    horizon <- object$config$horizon
  }

  if (horizon > object$config$horizon) {
    cli::cli_abort(c(
      "{.var horizon} is larger than the maximum allowed.",
      "x" = "Got {horizon}, max allowed is {object$horizon}."
    ))
  }

  f_data <- future_data(object$past_data, horizon = object$config$horizon,
                        input_types = object$config$input_types)
  f_data <- tibble::as_tibble(f_data)

  pred <- dplyr::bind_cols(
    f_data,
    predict(object, new_data = f_data)
  )

  future_data <- future_data(object$past_data, horizon = horizon,
                             input_types = object$config$input_types) %>%
    dplyr::left_join(pred, by = names(.)) %>%
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
