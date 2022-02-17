
add_known_input_cols <- function(.data, rules) {
  calls <- lapply(rules, function(f) {
    rlang::call2(f, tsibble::index(.data), !!!tsibble::key(.data))
  })
  out <- dplyr::mutate(.data, !!!calls)
  tsibble::new_tsibble(x = out, rules = rules, class = "ts_rules_tbl")
}

#' Provide rules to define variables that are known for future timesteps.
#'
#' The known input rules are automatically apllied when calling [new_data()]
#' on the `ts_rules_tbl`. Making it easy to generate forecasts.
#'
#' @param .data a tsibble
#' @param ... functions that are used to define new variables based only on
#'  the index and the keys of `.data` tsibble.
#' @examples
#'
#' library(dplyr)
#' library(tsibble)
#' weather <- nycflights13::weather %>%
#'   select(origin, time_hour, temp, humid, precip)
#' weather_tsbl <- as_tsibble(weather, key = origin, index = time_hour)
#' weather_tsbl
#'
#' weather_tsbl %>%
#'   known_input_rules(
#'     hour = function(x, ...) lubridate::hour(x)
#'   ) %>%
#'   new_data()
#'
#' @export
known_input_rules <- function(.data, ...) {
  rlang:::inherits_all(.data, "tbl_ts")
  rules <- rlang::dots_list(..., .named = NULL, .homonyms = "error")
  add_known_input_cols(.data, rules)
}

#' @importFrom tsibble new_data
#' @export
new_data.ts_rules_tbl <- function(.data, n = 1L, ...) {
  add_known_input_cols(NextMethod(), attr(.data, "rules"))
}

