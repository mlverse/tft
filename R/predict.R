
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
                        step = NULL,
                        past_data = object$past_data) {

  if (is_null_external_pointer(object$module$model$.check)) {
    object$module <- reload_model(object$.serialized_model)
  }

  new_data <- adjust_new_data(new_data, object$recipe)

  new_data <- recipes::bake(object$recipe, new_data)
  past_data <- recipes::bake(object$recipe, past_data)

  verify_new_data(new_data, object)
  out <- predict_impl(object, new_data, past_data, step)
  out
}

predict_impl <- function(object, new_data, past_data, step) {


  key_cols <- get_variables_with_role(object$recipe$term_info, "key")
  index_col <- get_variables_with_role(object$recipe$term_info, "index")

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
  period <- lubridate::as.period(tsibble::interval(object$past_data))
  last_index <- last_index - object$config$lookback*period

  # now filter the past_data
  past_data <- past_data %>%
    dplyr::filter(.data[[index_col]] > last_index)

  if (mode == "full") {
    # In the full mode we generate multi horizon forecasts starting from multiple
    # steps.
    # We need to figure if new_data is right after past data or not, to decide
    # if we can bind them together or not.
    interval <- lubridate::as.period(tsibble::interval(object$past_data))
    min_new_date <- min(new_data[[index_col]])
    max_old_date <- max(object$past_data[[index_col]])

    # in this scenario, new_data also contains the target, so we need to
    # normalize it.
    new_data_normalized <- normalize_outcome(
      x = new_data,
      keys = get_variables_with_role(object$recipe$term_info, "key"),
      outcome = get_variables_with_role(object$recipe$term_info, "outcome"),
      constants = object$normalization
    )$x

    if (min_new_date != (max_old_date + interval)) {
      data <- new_data_normalized
    } else {
      data <- dplyr::bind_rows(past_data, new_data_normalized)
    }
    if (is.null(step)){
      step <- 1
    }

  } else {
    if (!is.null(step)) {
      cli::cli_warn(c("Step is not {.var NULL} but won't be used."))
    }
    step <- 1
    data <- dplyr::bind_rows(past_data, new_data)
  }

  dataset <- time_series_dataset(
    data,
    object$recipe$term_info,
    lookback = object$config$lookback,
    assess_stop = object$config$horizon,
    step = step
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
            object$recipe$term_info, c("key", "index")
          )
        ))
      res[[".pred_at"]] <-
        max(tibble::as_tibble(dataset$df[.x$encoder,])[[index_col]])
      res
    })

  out <- dplyr::bind_cols(predictions, tibble::as_tibble(obs))
  out <- new_data %>%
    dplyr::left_join(
      out,
      by = c(get_variables_with_role(object$recipe$term_info, c("key", "index")))
    )

  for (var in names(predictions)) {
    out <- unnormalize_outcome(out, object$normalization, outcome = var)
  }

  if (mode == "horizon") {
    dplyr::select(out, dplyr::starts_with(".pred"))
  } else {
    out
  }
}

adjust_new_data <- function(new_data, recipe) {
  var_info <- recipe$var_info

  new_data <- tibble::as_tibble(new_data)

  keys <- var_info$variable[var_info$role %in% c("key")]
  index <- var_info$variable[var_info$role %in% c("index")]

  # we make sure that new data includes at least all index and keys.
  if (!all(c(keys, index) %in% names(new_data))) {
    cli::cli_abort(c(
      "New data does not include all the {.var keys} and {.var index}.",
      "x" = "Missing {.var {setdiff(c(keys, index), names(new_data))}}"
    ))
  }

  known <- var_info$variable[var_info$role %in% c("static", "known")]
  for (var in known) {
    if (is.null(new_data[[var]])) {
      cli::cli_abort(c(
        "Known or static variable is missing from {.var new_data}.",
        "x" = "Check for {.var {var}}."
      ))
    }
  }

  # we now add all `predictors` to the new_obs dataset
  predictors <- var_info$variable[var_info$role == "predictor"]
  for (var in predictors) {
    if (is.null(new_data[[var]])) {
      new_data[[var]] <- NA
    }
  }

  new_data
}

verify_new_data <- function(new_data, object, mode) {

  past_data <- object$past_data
  recipe <- object$recipe

  # we also have to make sure that all static and known predictors exist
  # in the new datatset
  term_info <- recipe$term_info
  known <- term_info$variable[term_info$tft_role %in% c("key", "index", "static", "known")]
  for (var in known) {
    if (any(is.na(new_data[[var]]))) {
      cli::cli_abort(c(
        "Found missing values in at least one known variable.",
        "i" = "This kind of variables should be fully known for future inputs.",
        "x" = "Check variable {.var {var}}."
      ))
    }
  }

  if (mode == "horizon") {
    possible_dates <- future_data(past_data, horizon = object$config$horizon)
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
  }

  # we verify that all groups have the entire prediction range
  if (mode == "horizon") {
    counts <- new_data %>%
      dplyr::group_by(!!!tsibble::key(past_data)) %>%
      dplyr::count()

    if (any(counts$n != object$config$horizon)) {
      n_groups <- sum(counts$n != object$config$horizon)
      h <- object$config$horizon
      cli::cli_abort(c(
        "At least one group doesn't have an entire range for prediction.",
        "x" = "Found {.var {n_groups}} group that don't have {.var {h}} dates."
      ))
    }
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

  future_data <- future_data(object$past_data, horizon = object$config$horizon,
                             roles = object$recipe$term_info)
  pred <- dplyr::bind_cols(
    tibble::as_tibble(future_data),
    predict(object, new_data = future_data)
  )

  future_data(object$past_data, horizon) %>%
    dplyr::left_join(pred, by = names(.)) %>%
    structure(class = c("tft_forecast", class(.)))
}

future_data <- function(past_data, horizon, roles = NULL) {
  if (!tsibble::is_tsibble(past_data))
    past_data <- make_tsibble(past_data, roles)

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
