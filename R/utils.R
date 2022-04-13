success <- function(...) {
  cli::cli_alert_success(paste(...))
}

data_types <- list(
  real = "real",
  categorical = "categorical",
  date = "date"
)

input_types <- list(
  target = "target",
  observed_input = "observerd_input",
  known_input = "known_input",
  static_input = "static_input",
  id = "id",
  time = "time"
)


get_n_features <- function(x) {
  f <- function(x) {
    if (inherits(x, "torch_tensor"))
      tail(x$shape, 1)
    else
      lapply(x, f)
  }
  f(x)
}
