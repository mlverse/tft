#' @export
predict.tft <- function(object, new_data, type = "numeric", ...) {
  new_data <- adjust_new_data(new_data, object$recipe)
  new_data <- recipes::bake(object$recipe, new_data)
  verify_new_data(new_data, object)
  out <- predict_impl(object, new_data)
  out
}

predict_impl <- function(object, new_data) {

  past_data <- new_data %>%
    dplyr::select(!!!tsibble::key(object$past_data)) %>%
    dplyr::distinct() %>%
    dplyr::left_join(
      tibble::as_tibble(object$past_data),
      by = tsibble::key_vars(object$past_data)
    )

  dataset <- time_series_dataset(
    dplyr::bind_rows(past_data, new_data),
    object$blueprint$recipe$term_info,
    lookback = object$config$lookback,
    assess_stop = object$config$horizon,
    mode = "predict"
  )

  res <- predict(object$module, dataset)

  predictions <- res %>%
    torch::torch_unbind(dim = 1) %>%
    torch::torch_cat(dim =  1) %>%
    as.matrix() %>%
    tibble::as_tibble(.name_repair = "minimal") %>%
    rlang::set_names(c(".pred_lower", ".pred", ".pred_upper"))

  obs <- dataset$slices %>%
    purrr::map_dfr(~dataset$df[.x$decoder,])

  out <- dplyr::bind_cols(predictions, tibble::as_tibble(obs))
  out <- new_data %>%
    dplyr::left_join(
      out,
      by = c(get_variables_with_role(object$recipe$term_info, c("key", "index")))
    )

  for (var in names(predictions)) {
    out <- unnormalize_outcome(out, object$normalization, outcome = var)
  }
  dplyr::select(out, dplyr::starts_with(".pred"))
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
    new_data[[var]] <- NA
  }

  new_data
}

verify_new_data <- function(new_data, object) {

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

  # we verify that al groups have the entire prediction range
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

  future_data <- future_data(object$past_data, object$config$horizon)
  pred <- dplyr::bind_cols(
    tibble::as_tibble(future_data),
    predict(object, new_data = future_data)
  )

  future_data(object$past_data, horizon) %>%
    dplyr::left_join(pred, by = names(.)) %>%
    structure(class = c("tft_forecast", class(.)))
}

future_data <- function(past_data, horizon) {
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
