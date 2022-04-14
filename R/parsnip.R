
set_tft_arg <- function(name, func, has_submodel = FALSE, original = name) {
  parsnip::set_model_arg(
    model        = "temporal_fusion_transformer",
    eng          = "torch",
    parsnip      = name,
    original     = original,
    func         = func,
    has_submodel = FALSE
  )
}


parsnip::set_new_model("temporal_fusion_transformer")
parsnip::set_model_mode(model = "temporal_fusion_transformer", mode = "regression")
parsnip::set_model_engine(model = "temporal_fusion_transformer", mode = "regression", eng = "torch")
parsnip::set_dependency(model = "temporal_fusion_transformer", eng = "torch", pkg = "tft")

set_tft_arg(
  name = "lookback",
  func = list(pkg = "tft", fun = "lookback")
)

set_tft_arg(
  name = "horizon",
  func = list(pkg = "tft", fun = "horizon")
)

set_tft_arg(
  name = "hidden_state_size",
  func = list(pkg = "tft", fun = "hidden_state_size")
)

set_tft_arg(
  name = "dropout",
  func = list(pkg = "dials", fun = "dropout")
)

set_tft_arg(
  name = "learn_rate",
  func = list(pkg = "dials", fun = "learn_rate")
)

set_tft_arg(
  name = "batch_size",
  func = list(pkg = "dials", fun = "batch_size")
)

set_tft_arg(
  name = "epochs",
  func = list(pkg = "dials", fun = "epochs")
)

#' @describeIn tft Parsnip wrappers for TFT.
#' @export
temporal_fusion_transformer <- function(mode = "regression", lookback = NULL,
                                        horizon = NULL, hidden_state_size = NULL,
                                        dropout = NULL, learn_rate = NULL,
                                        batch_size = NULL, epochs = NULL) {
  args <- list(
    lookback = rlang::enquo(lookback),
    horizon = rlang::enquo(horizon),
    hidden_state_size = rlang::enquo(hidden_state_size),
    dropout = rlang::enquo(dropout),
    learn_rate = rlang::enquo(learn_rate),
    batch_size = rlang::enquo(batch_size),
    epochs = rlang::enquo(epochs)
  )

  parsnip::new_model_spec(
    "temporal_fusion_transformer",
    args     = args,
    eng_args = NULL,
    mode     = mode,
    method   = NULL,
    engine   = NULL
  )
}

#' @export
#' @importFrom stats update
update.temporal_fusion_transformer <- function(object, parameters = NULL, lookback = NULL,
                                               horizon = NULL, hidden_state_size = NULL,
                                               dropout = NULL, learn_rate = NULL,
                                               batch_size = NULL, epochs = NULL, ...) {
  rlang::check_installed("parsnip")
  eng_args <- parsnip::update_engine_parameters(object$eng_args, ...)
  args <- list(
    lookback = current_or_value(object$args$lookback, lookback),
    horizon = current_or_value(object$args$horizon, horizon),
    hidden_state_size = current_or_value(object$args$hidden_state_size, hidden_state_size),
    dropout = current_or_value(object$args$dropout, dropout),
    learn_rate = current_or_value(object$args$learn_rate, learn_rate),
    batch_size = current_or_value(object$args$batch_size, batch_size),
    epochs = current_or_value(object$args$epochs, epochs)
  )
  args <- parsnip::update_main_parameters(args, parameters)
  parsnip::new_model_spec(
    "temporal_fusion_transformer",
    args = args,
    eng_args = eng_args,
    mode = object$mode,
    method = NULL,
    engine = object$engine
  )
}

current_or_value <- function(x, y) {
  if (is.null(y))
    x
  else
    rlang::enquo(y)
}

parsnip::set_fit(
  model  = "temporal_fusion_transformer",
  eng    = "torch",
  mode   = "regression",
  value  = list(
    interface = "data.frame",
    protect   = c("x", "y"),
    func      = c(fun = "tft"),
    defaults  = list()
  )
)

parsnip::set_encoding(
  model   = "temporal_fusion_transformer",
  eng     = "torch",
  mode    = "regression",
  options = list(
    predictor_indicators = "none",
    compute_intercept = FALSE,
    remove_intercept = FALSE,
    allow_sparse_x = FALSE
  )
)

parsnip::set_pred(
  model = "temporal_fusion_transformer",
  eng  = "torch",
  mode = "regression",
  type = "numeric",
  value = list(
    pre  = NULL,
    post = NULL,
    func = c(fun = "predict"),
    args =
      list(
        object   = rlang::expr(object$fit),
        new_data = rlang::expr(new_data)
      )
  )
)

