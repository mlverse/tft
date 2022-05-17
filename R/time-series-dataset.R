# input data must be a single data frame with all time series: this makes it easier
# to detect levels of categorical variables, etc. checks that should be done:
# - data is contigous for all groups. We could relax this at some point, but, probably
# is a mistake so we would at least need to warn people.
# - the number of possible slices (step=1) is N + 1 - (lookback + horizon). it makes
# sense to subsample this slices, otherwise there might be too many.

# roles: should be a S3 object defining what's the role role of each column in `df`.
# we should be able to create it from a recipe, but users should be able to specify
# it manually.

# lookback & horizon: should be in units of time. we currently rely on tsibble
# auto-detecting the time interval. Series must have a constant time interval.

# splits are generated per group, so groups can have different number of observations.
# users should be able to provide the splits they want to use manually. Splits
# can be represented as a tuple (ids_lookback, ids_horizon).

#' Creates a TFT data specification
#'
#' This is used to create [torch::dataset()]s for training the model,
#' take care of target normalization and allow initializing the
#' [temporal_fusion_transformer()] model, that requires a specification
#' to be passed as its first argument.
#'
#' @param x A recipe or data.frame that will be used to obtain statiscs for
#' preparing the recipe and preparing the dataset.
#'
#' @returns
#' A `tft_dataset_spec` that you can add `spec_` functions using the `|>` (pipe)
#' [prep()] when done and [transform()] to obtain [torch::dataset()]s.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#' sales <- walmartdata::walmart_sales %>%
#'   dplyr::filter(Store == 1, Dept %in% c(1,2))
#'
#' rec <- recipes::recipe(Weekly_Sales ~ ., sales)
#'
#' spec <- tft_dataset_spec(rec, sales) %>%
#'   spec_time_splits(lookback = 52, horizon = 4) %>%
#'   spec_covariate_index(Date) %>%
#'   spec_covariate_keys(Store, Dept) %>%
#'   spec_covariate_static(Type, Size) %>%
#'   spec_covariate_known(starts_with("MarkDown"))
#'
#' print(spec)
#'
#' spec <- prep(spec)
#' dataset <- transform(spec) # this is a torch dataset.
#' str(dataset[1])
#' }
#' @export
tft_dataset_spec <- function(x, ...) {
  UseMethod("tft_dataset_spec")
}

#' @export
tft_dataset_spec.data.frame <- function(x, y, ...) {
  ellipsis::check_dots_empty()
  if (is.data.frame(x) && is.data.frame(y)) {
    cli::cli_abort(c(
      "{.var x} and {.var y} must be {.cls data.frame}s.",
      "x" = "{.var x} is {.cls {class(x)}} and {.var y} is {.cls {class(y)}}."
    ))
  }

  new_tft_dataset_spec(inputs = list(x, y))
}

#' @export
tft_dataset_spec.recipe <- function(x, data, ...) {
  ellipsis::check_dots_empty()
  if (!is.data.frame(data)) {
    cli::cli_abort(c(
      "{.var data} must be a {.cls data.frame}.",
      x = "Got a {.cls {class(data}}"
    ))
  }
  new_tft_dataset_spec(inputs = list(x, data))
}


#' @importFrom recipes prep
#' @export
prep.tft_dataset_spec <- function(x, ...) {
  ellipsis::check_dots_empty()
  inputs <- append(x$inputs, x["input_types"])
  inputs <- append(inputs, x[c("lookback", "horizon", "subsample")])
  do.call(time_series_dataset, inputs)
}

#' @export
print.tft_dataset_spec <- function(x, ...) {
  cat(sep="\n", cli::cli_format_method({
    cli::cli_text("A {.cls tft_dataset_spec} with:")
    cli::cat_line()

    if (is.null(x$lookback) || is.null(x$horizon)) {
      cli::cli_alert_danger(
        "{.var lookback} and {.var horizon} are not set. Use {.var spec_time_splits()} ",
        "to set them."
      )
    } else {
      cli::cli_alert_success(
        "lookback = {x$lookback} and horizon = {x$horizon}."
      )
    }

    cli::cli_h3("Covariates:")
    for (name in c("index", "keys", "static", "known", "unknown")) {
      cat_covariates_set(x$input_types, name)
    }

    cli::cat_line()

    cli::cli_alert_info("Call {.var prep()} to prepare the specification.")
  }))
  invisible(x)
}

#' @export
print.prepared_tft_dataset_spec <- function(x, ...) {
  config <- x$config
  cat(sep="\n", cli::cli_format_method({
    cli::cli_text("A {.cls prepared_tft_dataset_spec} with:")
    cli::cat_line()

    cli::cli_alert_success(
      "lookback = {config$lookback} and horizon = {config$horizon}."
    )
    cli::cli_alert_success(
      "The number of possible slices is {scales::comma(length(x$dataset))}"
    )

    cli::cli_h3("Covariates:")
    for (name in c("index", "keys", "static", "known", "unknown")) {
      cat_covariates_prepared(config$input_types, name)
    }
    cli::cli_alert_info("Variables that are not specified in other types are considered {.var unknown}.")

    cli::cat_line()

    cli::cli_alert_info("Call {.var transform()} to apply this spec to a different dataset.")
  }))
  invisible(x)
}

cat_covariates_prepared <- function(input_types, name) {
  cli::cli_alert_success(
    "{.var {name}}: {input_types[[name]]}"
  )
}

cat_covariates_set <- function(input_types, name) {
  if (is.null(input_types[[name]])) {
    if (name != "unknown") {
      alert <- if(name %in% c("index", "keys")) cli::cli_alert_danger else cli::cli_alert_warning
      alert(
        "{.var {name}} is not set. Use {.var {paste0('spec_covariate_', name, '()')}} to set it."
      )
    } else {
      cli::cli_alert_warning(
        "{.var unknown} is not set. Covariates that are not listed as other types are considered {.var unknown}."
      )
    }
  } else {
    cli::cli_alert_success(
      "{.var {name}}: {rlang::expr_deparse(input_types[[name]])}"
    )
  }
}

new_tft_dataset_spec <- function(inputs) {
  structure(list(
    inputs = inputs,
    input_types = list(),
    lookback = NULL,
    horizon = NULL,
    subsample = 1
  ), class = "tft_dataset_spec")
}

#' Time splits setting
#'
#'
#' @param lookback Number of timesteps that are used as historic data for
#'  prediction.
#' @param horizon Number of timesteps ahead that will be predicted by the
#'  model.
#'
#' @describeIn tft_dataset_spec Sets `lookback` and `horizon` parameters.
#'
#' @export
spec_time_splits <- function(spec, lookback, horizon) {
  spec$lookback <- lookback
  spec$horizon <- horizon
  spec
}

#' @param index A column name that indexes the data. Usually a date column.
#' @param spec A spec created with `tft_dataset_spec()`.
#' @describeIn tft_dataset_spec Sets the `index` column.
#' @export
spec_covariate_index <- function(spec, index) {
  spec$input_types$index <- rlang::enexpr(index)
  spec
}

#' @param ... Column names, selected using tidyselect. See <[`tidy-select`][dplyr_tidy_select]>
#'  for more information.
#' @describeIn tft_dataset_spec Sets the `keys` - variables that define each time series
#' @export
spec_covariate_keys <- function(spec, ...) {
  spec$input_types$keys <- rlang::enexprs(...)
  spec
}

#' @describeIn tft_dataset_spec Sets `known` time varying covariates.
#' @export
spec_covariate_known <- function(spec, ...) {
  spec$input_types$known <- rlang::enexprs(...)
  spec
}

#' @describeIn tft_dataset_spec Sets `unknown` time varying covariates.
#' @export
spec_covariate_unknown <- function(spec, ...) {
  spec$input_types$unknown <- rlang::enexprs(...)
  spec
}

#' @describeIn tft_dataset_spec Sets `static` covariates.
#' @export
spec_covariate_static <- function(spec, ...) {
  spec$input_types$static <- rlang::enexprs(...)
  spec
}

#' @export
time_series_dataset <- function(x, ...) {
  UseMethod("time_series_dataset")
}

#' @export
time_series_dataset.default <- function(x, ...) {
  cli::cli_abort(
    "{.var time_series_dataset} is not defined for objects with class {.cls {class(x)}}.")
}

#' @export
time_series_dataset.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  config <- ts_dataset_config(...)
  ts_dataset_bridge(processed, config)
}

#' @export
time_series_dataset.recipe <- function(x, data, ...) {
  config <- ts_dataset_config(...)
  data <- tibble::as_tibble(data)
  processed <- hardhat::mold(x, data)
  ts_dataset_bridge(processed, config)
}

ts_dataset_bridge <- function(processed, config) {
  config$input_types <- evaluate_types(processed$predictors, config$input_types)
  config$input_types[["outcome"]] <- names(processed$outcome)

  processed_data <- dplyr::bind_cols(processed$predictors, processed$outcomes)

  normalization <- normalize_outcome(
    x = processed_data,
    keys = get_variables_with_role(config$input_types, "keys"),
    outcome = get_variables_with_role(config$input_types, "outcome")
  )

  dataset <- time_series_dataset_generator(
    normalization$x,
    config$input_types,
    lookback = config$lookback,
    assess_stop = config$horizon,
    subsample = config$subsample
  )

  hardhat::new_model(
    past_data = processed_data,
    dataset = dataset,
    normalization = normalization$constants,
    config = config,
    blueprint = processed$blueprint,
    class = "prepared_tft_dataset_spec"
  )
}

#' @export
transform.prepared_tft_dataset_spec <- function(`_data`, past_data = NULL, ...,
                                                new_data = NULL, .verify = FALSE) {
  object <- `_data`

  if (is.null(past_data) && is.null(new_data))
    return(object$dataset)

  # here we apply the normalization and can create the dataset directly.
  if (is.null(new_data)) {
    past_data <- adjust_past_data(past_data, object$blueprint)
    past_data <- normalize_outcome(
      x = past_data,
      keys = get_variables_with_role(object$config$input_types, "keys"),
      outcome = get_variables_with_role(object$config$input_types, "outcome"),
      constants = object$normalization
    )$x
    dataset <- time_series_dataset_generator(
      past_data,
      object$config$input_types,
      lookback = object$config$lookback,
      assess_stop = object$config$horizon,
      subsample = object$config$subsample
    )
    return(dataset)
  }

  # when we also have a `new_data`, we will prepare a 'validation dataset'.
  # it differs because only observations in `new_data` are iterated as targets.
  config <- object$config
  input_types <- object$config$input_types

  if (is.null(past_data)) {
    past_data <- object$past_data
  } else {
    past_data <- adjust_past_data(past_data, object$blueprint)
  }
  past_data <- normalize_outcome(
    x = past_data,
    keys = get_variables_with_role(input_types, "keys"),
    outcome = get_variables_with_role(input_types, "outcome"),
    constants = object$normalization
  )$x

  new_data <- adjust_new_data(
    new_data,
    input_types,
    object$blueprint,
    outcomes = TRUE
  )
  new_data <- normalize_outcome(
    x = new_data,
    keys = get_variables_with_role(input_types, "keys"),
    outcome = get_variables_with_role(input_types, "outcome"),
    constants = object$normalization
  )$x

  if (.verify)
    verify_new_data(new_data = new_data, past_data = past_data, object = object)

  make_prediction_dataset(
    new_data = new_data,
    past_data = past_data,
    config = config
  )
}

ts_dataset_config <- function(...) {
  args <- list(...)

  if (is.null(args$input_types)) {
    cli::cli_abort("Please provide {.var input_types}.")
  }

  if (is.null(args$lookback)) {
    cli::cli_abort("Please provide a {.var lookback}.")
  }

  if (is.null(args$horizon)) {
    cli::cli_abort("Please provide a {.var horizon}.")
  }

  if (is.null(args$step)) {
    args$step <- 1
  }

  if (is.null(args$subsample)) {
    args$subsample <- 1
  }

  args
}

time_series_dataset_generator <- torch::dataset(
  "time_series_dataset",
  initialize = function(df, roles, lookback = 2L,
                        assess_stop = 1L, step = 1L,
                        subsample = 1) {

    self$roles <- roles
    # create a tsibble using information from the recipe
    keys <- get_variables_with_role(roles, "keys")
    index <- get_variables_with_role(roles, "index")

    if (length(index) != 1) {
      cli::cli_abort(c(
        "A sigle index must be provided in {.var roles}",
        "x" = "Got {.var {length(index)}}"
      ))
    }

    self$df <- df <- make_tsibble(df, roles)

    # Slices are lists with 2 elements: encoder and decoder that are both indexes
    # for rows of `df` to be used for a data frame.
    self$slices <- slice_df(
      df,
      lookback = lookback,
      horizon = assess_stop,
      step = step,
      keys = rlang::syms(keys)
    )

    if (subsample < 1) {
      self$slices <- self$slices[sample.int(length(self$slices), size = subsample*length(self$slices))]
    } else if (subsample > 1) {
      self$slices <- self$slices[sample.int(length(self$slices), size = subsample)]
    }

    # figure out variables of each type
    terms <- self$roles

    self$known <- terms %>%
      pull_term_names(df, "known")
    self$observed <- terms %>%
      pull_term_names(df, c("unknown", "outcome"))

    self$past <- purrr::map2(self$known, self$observed, ~c(.x, .y))

    self$static <- terms %>%
      pull_term_names(df, c("static", "keys"))

    self$target <- terms %>%
      pull_term_names(df, "outcome")

    # compute feature sizes
    dictionary <- df %>%
      purrr::keep(is.factor) %>%
      purrr::map(~length(levels(.x)))

    self$feature_sizes <- list()
    self$feature_sizes$known <- dictionary[self$known$cat]
    self$feature_sizes$observed <- dictionary[self$observed$cat]
    self$feature_sizes$past <- dictionary[self$past$cat]
    self$feature_sizes$static <- dictionary[self$static$cat]
    self$feature_sizes$target <- dictionary[self$target$cat]
  },
  .getitem = function(i) {
    split <- self$slices[[i]]
    x <- tibble::as_tibble(self$df[split$encoder,])
    y <- tibble::as_tibble(self$df[split$decoder,])

    past <- list(
      num = self$to_tensor(x[,self[["past"]][["num"]]]),
      cat = self$to_cat_tensor(x[,self[["past"]][["cat"]]])
    )

    static <- list(
      num = self$to_tensor(x[1,self[["static"]][["num"]]])[1,],
      cat = self$to_cat_tensor(x[1,self[["static"]][["cat"]]])[1,]
    )

    known <- list(
      num = self$to_tensor(y[,self[["known"]][["num"]]]),
      cat = self$to_cat_tensor(y[,self[["known"]][["cat"]]])
    )

    target <- list(
      num = self$to_tensor(y[,self[["target"]][["num"]]]),
      cat = self$to_cat_tensor(y[,self[["target"]][["cat"]]])
    )

    list(
      list(
        encoder = list(past = past, static = static),
        decoder = list(known = known, target = target)
      ),
      target$num
    )
  },
  .length = function() {
    length(self$slices)
  },
  to_cat_tensor = function(df) {
    if (length(df) == 0) {
      return(torch::torch_tensor(matrix(numeric(), ncol = 0, nrow = 0)))
    }

    df %>%
      do.call(cbind, .) %>%
      torch::torch_tensor()
  },
  to_tensor = function(df) {
    df %>%
      as.matrix() %>%
      torch::torch_tensor()
  }
)

pull_term_names <- function(input_types, df, terms) {
  factors <- names(purrr::keep(df, is.factor))
  numerics <- names(purrr::discard(df, is.factor))

  vars <- get_variables_with_role(input_types, terms)

  output <- list()
  output$cat <- dplyr::intersect(vars, factors)
  output$num <- dplyr::intersect(vars, numerics)

  lapply(output, unique)
}

get_variables_with_role <- function(roles, role) {
  unname(unlist(roles[role]))
}

# Assumes that observations are ordered by date and that there are no implicit
# missing obs.
make_slices <- function(x, lookback, horizon, step = 1) {
  len <- length(x)

  if (len < (lookback + horizon)) {
    return(list())
  }

  start_lookback <- seq(
    from = 1,
    to = len - (lookback + horizon - 1),
    by = step
  )
  end_lookback <- start_lookback + lookback - 1
  start_horizon <- end_lookback + 1
  end_horizon <- start_horizon + horizon - 1

  purrr::pmap(
    list(
      start_lookback,
      end_lookback,
      start_horizon,
      end_horizon
    ),
    function(sl, el, sh, eh) {
      list(
        encoder = x[seq(from = sl, to = el)],
        decoder = x[seq(from = sh, to = eh)]
      )
    }
  )
}

slice_df <- function(df, lookback, horizon, step, keys) {
  slices <- df %>%
    dplyr::ungroup() %>%
    dplyr::mutate(.row = dplyr::row_number()) %>%
    dplyr::group_split(!!!keys) %>%
    purrr::map(function(.x) {
      make_slices(
        .x$.row,
        lookback = lookback,
        horizon = horizon,
        step = step
      )
    }) %>%
    purrr::compact()

  if (length(slices) == 0) {
    cli::cli_abort(c(
      "No group has enough observations to statisfy the requested {.var lookback}."
    ))
  }

  rlang::flatten_if(slices, function(.x) {!rlang::is_named(.x)})
}

make_tsibble <- function(df, roles) {
  keys <- get_variables_with_role(roles, "keys")
  index <- get_variables_with_role(roles, "index")

  df %>%
    tsibble::as_tsibble(key = dplyr::all_of(keys), index = dplyr::all_of(index)) %>%
    dplyr::arrange(!!!tsibble::index_var(.))
}
