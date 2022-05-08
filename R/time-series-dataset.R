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


time_series_dataset <- torch::dataset(
  "time_series_dataset",
  initialize = function(df, roles, lookback = 2L,
                        assess_stop = 1L, step = 1L,
                        subsample = 1) {

    self$roles <- roles
    # create a tsibble using information from the recipe
    keys <- get_variables_with_role(roles, "key")
    index <- get_variables_with_role(roles, "index")

    if (length(index) != 1) {
      cli::cli_abort(c(
        "A sigle index must be provided in {.var roles}",
        "x" = "Got {.var {length(index)}}"
      ))
    }

    self$df <- df <- make_tsibble(df, roles)

    # we create rsample `split` objects that don't materialize the data until
    # `training` or `testing` allowing us to compute the number of slices that
    # we are able to create.
    # since data is already split between future and past values it's also easier
    # to reason when implementing the network.

    self$slices <- self$df %>%
      dplyr::ungroup() %>%
      dplyr::mutate(.row = dplyr::row_number()) %>%
      dplyr::group_split(!!!tsibble::key(df)) %>%
      purrr::map(function(.x) {
        make_slices(
          .x$.row,
          lookback = lookback,
          horizon = assess_stop,
          step = step
        )
      }) %>%
      purrr::compact()

    if (length(self$slices) == 0) {
      cli::cli_abort(c(
        "No group has enough observations to statisfy the requested {.var lookback}."
      ))
    }

    self$slices <- rlang::flatten_if(self$slices, function(.x) {!rlang::is_named(.x)})

    if (subsample < 1) {
      self$slices <- self$slices[sample.int(length(self$slices), size = subsample*length(self$slices))]
    } else if (subsample > 1) {
      self$slices <- self$slices[sample.int(length(self$slices), size = subsample)]
    }

    # figure out variables of each type
    terms <- self$roles

    self$known <- terms %>%
      pull_term_names(tft_role %in% c("known"), !(variable %in% tsibble::key_vars(df)))
    self$observed <- terms %>%
      pull_term_names(tft_role %in% c("predictor", "outcome"), !(variable %in% tsibble::key_vars(df)))

    self$past <- purrr::map2(self$known, self$observed, ~c(.x, .y))

    self$static <- terms %>%
      pull_term_names(tft_role %in% c("static", "predictor", "key"), variable %in% tsibble::key_vars(df))

    self$target <- terms %>%
      pull_term_names(tft_role == "outcome")

    # compute feature sizes
    dictionary <- purrr::set_names(roles$size, roles$variable)

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

pull_term_names <- function(terms, ...) {
  output <- list()
  terms <- terms %>%
    dplyr::filter(...)

  output$cat <- terms %>% dplyr::filter(type == "nominal") %>% dplyr::pull(variable)
  output$num <- terms %>% dplyr::filter(type == "numeric") %>% dplyr::pull(variable)

  lapply(output, unique)
}

get_variables_with_role <- function(roles, role) {
  roles$variable[roles$tft_role %in% role]
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

make_tsibble <- function(df, roles) {
  keys <- get_variables_with_role(roles, "key")
  index <- get_variables_with_role(roles, "index")

  df %>%
    tsibble::as_tsibble(key = dplyr::all_of(keys), index = dplyr::all_of(index)) %>%
    dplyr::arrange(!!!tsibble::index_var(.))
}
