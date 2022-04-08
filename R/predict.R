#' @export
predict.tft <- function(object, new_data, type = "numeric", ...) {

  if (missing(new_data) || is.null(new_data)) {
    new_data <- object$future_data
  } else {
    processed <- forge(new_data, object$blueprint)
    new_data <- processed$predictors
  }

  out <- predict_impl(object, new_data)

  out
}

predict_impl <- function(object, new_data) {
  dataset <- time_series_dataset(
    dplyr::bind_rows(object$past_data, new_data),
    object$blueprint$recipe$term_info,
    lookback = 120,
    assess_stop = 4,
    mode = "predict"
  )
  res <- predict(object$module, dataset)

  predictions <- res %>%
    torch::torch_unbind(dim = 1) %>%
    torch::torch_cat(dim =  1) %>%
    as.matrix() %>%
    tibble::as_tibble(.name_repair = "minimal") %>%
    rlang::set_names(c(".pred_lower", ".pred", ".pred_upper"))

  out <- dplyr::bind_cols(predictions, new_data)
  for (var in names(predictions)) {
    out <- unnormalize_outcome(out, object$normalization, outcome = var)
  }
  dplyr::select(out, dplyr::starts_with(".pred"))
}
