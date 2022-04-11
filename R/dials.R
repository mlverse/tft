#' Parameters related to data preparation in TFT
#'
#' @inheritParams dials::window_size
#' @examples
#' lookback()
#' @rdname dataprep_parameters
#' @name dataprep_parameters
NULL


#' @describeIn dataprep_parameters lookback Lookback from the history
#' @export
lookback <- function(range = c(3L, 365L), trans = NULL) {
  new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(window_size = "Lookback"),
    finalize = NULL
  )
}

#' @describeIn dataprep_parameters horizon Number of steps in the multi-horizon forecast
#' @export
horizon <- function(range = c(1L, 365L), trans = NULL) {
  new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(window_size = "Horizon"),
    finalize = NULL
  )
}
