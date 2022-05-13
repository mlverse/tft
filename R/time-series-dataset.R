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
    class = "time_series_dataset"
  )
}

#' @export
transform.time_series_dataset <- function(`_data`, past_data = NULL, new_data = NULL, ...) {
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
