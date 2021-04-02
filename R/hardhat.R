#' Temporal Fusion Transformer model
#'
#' Fits the [Temporal Fusion Transformer for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1908.07442) model
#'
#' @param x Depending on the context:
#'
#'   * A __data frame__ of predictors.
#'   * A __matrix__ of predictors.
#'   * A __recipe__ specifying a set of preprocessing steps
#'     created from [recipes::recipe()].
#'
#'  The predictor data should be standardized (e.g. centered or scaled).
#'  The model treats categorical predictors internally thus, you don't need to
#'  make any treatment.
#'
#' @param y When `x` is a __data frame__ or __matrix__, `y` is the outcome
#' specified as:
#'
#'   * A __data frame__ with 1 numeric column.
#'   * A __matrix__ with 1 numeric column.
#'   * A numeric __vector__.
#'
#' @param data When a __recipe__ or __formula__ is used, `data` is specified as:
#'
#'   * A __data frame__ containing both the predictors and the outcome.
#'
#' @param formula A formula specifying the outcome terms on the left-hand side,
#'  and the predictor terms on the right-hand side.
#' @param tft_model A previously fitted TFT model object to continue the fitting on.
#'  if `NULL` (the default) a brand new model is initialized.
#' @param from_epoch When a `tft_model` is provided, restore the network weights from a specific epoch.
#'  Default is last available checkpoint for restored model, or last epoch for in-memory model.
#' @param ... Model hyperparameters. See [tft_config()] for a list of
#'  all possible hyperparameters.
#'
#' @section Fitting a pre-trained model:
#'
#' When providing a parent `tft_model` parameter, the model fitting resumes from that model weights
#' at the following epoch:
#'    * last fitted epoch for a model already in torch context
#'    * Last model checkpoint epoch for a model loaded from file
#'    * the epoch related to a checkpoint matching or preceding the `from_epoch` value if provided
#' The model fitting metrics append on top of the parent metrics in the returned TFT model.
#'
#' @section Threading:
#'
#' TFT uses `torch` as its backend for computation and `torch` uses all
#' available threads by default.
#'
#' You can control the number of threads used by `torch` with:
#'
#' ```
#' torch::torch_set_num_threads(1)
#' torch::torch_set_num_interop_threads(1)
#' ```
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' data("ames", package = "modeldata")
#' fit <- tft_fit(Sale_Price ~ ., data = ames, epochs = 1)
#' }
#'
#' @return A TFT model object. It can be used for serialization, predictions, or further fitting.
#'
#' @export
tft_fit <- function(x, ...) {
  UseMethod("tft_fit")
}

#' @export
#' @rdname tft_fit
tft_fit.default <- function(x, ...) {
  stop(
    "`tft_fit()` is not defined for a '", class(x)[1], "'.",
    call. = FALSE
  )
}

#' @export
#' @rdname tft_fit
tft_fit.recipe <- function(x, data, tft_model = NULL, ..., from_epoch = NULL) {
  processed <- hardhat::mold(x, data)
  config <- do.call(tft_config, list(...))
  tft_bridge(processed, config = config, tft_model, from_epoch)
}

new_tft_fit <- function(fit, blueprint) {

  serialized_net <- model_to_raw(fit$network)

  hardhat::new_model(
    fit = fit,
    serialized_net = serialized_net,
    blueprint = blueprint,
    class = "tft_fit"
  )
}


tft_bridge <- function(processed, config = tft_config(), tft_model, from_epoch, task="supervised") {
  predictors <- processed$predictors
  outcomes <- processed$outcomes
  if (!(is.null(tft_model) || inherits(tft_model, "tft_fit") || inherits(tft_model, "tft_pretrain")))
    rlang::abort(paste0(tft_model," is not recognised as a proper TFT model"))
  if (task == "supervised") {
    if (is.null(tft_model)) {
      # new supervised model needs network initialization
      tft_model_lst <- tft_initialize(predictors, outcomes, config = config)
      tft_model <-  new_tft_fit(tft_model_lst, blueprint = processed$blueprint)
      epoch_shift <- 0L

    } else if (!is.null(from_epoch)) {
      # model must be loaded from checkpoint

      if (from_epoch > (length(tft_model$fit$checkpoints) * tft_model$fit$config$checkpoint_epoch))
        rlang::abort(paste0("The model was trained for less than ", from_epoch, " epochs"))

      # find closest checkpoint for that epoch
      closest_checkpoint <- from_epoch %/% tft_model$fit$config$checkpoint_epoch

      tft_model$fit$network <- reload_model(tft_model$fit$checkpoints[[closest_checkpoint]])
      epoch_shift <- closest_checkpoint * tft_model$fit$config$checkpoint_epoch

    } else if (!check_net_is_empty_ptr(tft_model) && inherits(tft_model, "tft_fit")) {

      if (!identical(processed$blueprint, tft_model$blueprint))
        rlang::abort("Model dimensions don't match.")

      # model is available from tft_model$serialized_net
      m <- reload_model(tft_model$serialized_net)

      # this modifies 'tft_model' in-place so subsequent predicts won't
      # need to reload.
      tft_model$fit$network$load_state_dict(m$state_dict())
      epoch_shift <- length(tft_model$fit$metrics)


    } else if (inherits(tft_model, "tft_pretrain")) {
      # pretrain_model after reload

      tft_model_lst <- model_pretrain_to_fit(tft_model, predictors, outcomes, config)
      tft_model <-  new_tft_fit(tft_model_lst, blueprint = processed$blueprint)
      epoch_shift <- 0L


    }  else if (length(tft_model$fit$checkpoints)) {
      # model is loaded from the last available checkpoint

      last_checkpoint <- length(tft_model$fit$checkpoints)

      tft_model$fit$network <- reload_model(tft_model$fit$checkpoints[[last_checkpoint]])
      epoch_shift <- last_checkpoint * tft_model$fit$config$checkpoint_epoch

    } else rlang::abort(paste0("No model serialized weight can be found in ", tft_model, ", check the model history"))

    fit_lst <- tft_train_supervised(tft_model, predictors, outcomes, config = config, epoch_shift)
    return(new_tft_fit(fit_lst, blueprint = processed$blueprint))

  } else if (task == "unsupervised") {

    pretrain_lst <- tft_train_unsupervised( predictors, config = config)
    return(new_tft_pretrain(pretrain_lst, blueprint = processed$blueprint))

  }
}


#' @importFrom stats predict
#' @export
predict.tft_fit <- function(object, new_data, type = NULL, ..., epoch = NULL) {
  # Enforces column order, type, column names, etc
  processed <- hardhat::forge(new_data, object$blueprint)
  out <- predict_tft_bridge(type, object, processed$predictors, epoch)
  hardhat::validate_prediction_size(out, new_data)
  out
}

check_type <- function(model, type) {

  outcome_ptype <- model$blueprint$ptypes$outcomes[[1]]

  if (is.null(type)) {
    if (is.factor(outcome_ptype))
      type <- "class"
    else if (is.numeric(outcome_ptype))
      type <- "numeric"
    else
      rlang::abort(glue::glue("Unknown outcome type '{class(outcome_ptype)}'"))
  }

  type <- rlang::arg_match(type, c("numeric", "prob", "class"))

  if (is.factor(outcome_ptype)) {
    if (!type %in% c("prob", "class"))
      rlang::abort(glue::glue("Outcome is factor and the prediction type is '{type}'."))
  } else if (is.numeric(outcome_ptype)) {
    if (type != "numeric")
      rlang::abort(glue::glue("Outcome is numeric and the prediction type is '{type}'."))
  }

  type
}



predict_tft_bridge <- function(type, object, predictors, epoch) {

  type <- check_type(object, type)

  if (!is.null(epoch)) {

    if (epoch > (length(object$fit$checkpoints) * object$fit$config$checkpoint_epoch))
      rlang::abort(paste0("The model was trained for less than ", epoch, " epochs"))

    # find closest checkpoint for that epoch
    ind <- epoch %/% object$fit$config$checkpoint_epoch

    object$fit$network <- reload_model(object$fit$checkpoints[[ind]])
  }

  if (check_net_is_empty_ptr(object)) {
    m <- reload_model(object$serialized_net)
    # this modifies 'object' in-place so subsequent predicts won't
    # need to reload.
    object$fit$network$load_state_dict(m$state_dict())
  }

  switch(
    type,
    numeric = predict_impl_numeric(object, predictors),
    prob    = predict_impl_prob(object, predictors),
    class   = predict_impl_class(object, predictors)
  )
}

model_to_raw <- function(model) {
  con <- rawConnection(raw(), open = "wr")
  torch::torch_save(model, con)
  on.exit({close(con)}, add = TRUE)
  r <- rawConnectionValue(con)
  r
}

model_pretrain_to_fit <- function(obj, x, y, config = tft_config()) {

  tft_model_lst <- tft_initialize(x, y, config)


  # do not restore previous metrics as loss function return non comparable
  # values, nor checkpoints
  m <- reload_model(obj$serialized_net)

  if (m$input_dim != tft_model_lst$network$input_dim)
    rlang::abort("Model dimensions don't match.")

  # perform update of selected weights into new tft_model
  m_stat_dict <- m$state_dict()
  tft_state_dict <- tft_model_lst$network$state_dict()
  for (param in names(m_stat_dict)) {
    if (grepl("^encoder", param)) {
      # Convert encoder's layers name to match
      new_param <- paste0("tft.", param)
    } else {
      new_param <- param
    }
    if (!is.null(tft_state_dict[new_param])) {
      tft_state_dict[[new_param]] <- m_stat_dict[[param]]
    }
  }
  tft_model_lst$network$load_state_dict(tft_state_dict)
  tft_model_lst
}


check_net_is_empty_ptr <- function(object) {
  is_null_external_pointer(object$fit$network$.check$ptr)
}

# https://stackoverflow.com/a/27350487/3297472
is_null_external_pointer <- function(pointer) {
  a <- attributes(pointer)
  attributes(pointer) <- NULL
  out <- identical(pointer, methods::new("externalptr"))
  attributes(pointer) <- a
  out
}

reload_model <- function(object) {
  con <- rawConnection(object)
  on.exit({close(con)}, add = TRUE)
  module <- torch::torch_load(con)
  module
}

#' @export
print.tft_fit <- function(x, ...) {
  if (check_net_is_empty_ptr(x)) {
    print(reload_model(x$serialized_net))
  } else {
    print(x$fit$network)
  }
  invisible(x)
}
#' @export
print.tft_pretrain <- print.tft_fit
