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


object <- result
new_data <- dplyr::bind_rows(object$past_data, object$future_data)

predict_impl <- function(object, new_data) {

  dataset <- time_series_dataset(
    dplyr::bind_rows(object$past_data, new_data),
    object$blueprint$recipe$term_info,
    lookback = 120,
    assess_stop = 4,
    mode = "predict"
  )

  res <- predict(object$module, dataset)
}
