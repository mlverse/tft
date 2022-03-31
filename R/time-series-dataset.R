
time_series_dataset <- torch::dataset(
  "time_series_dataset",
  initialize = function(df, recipe, lookback = 2L, assess_start = 1L,
                        assess_stop = 1L, complete = TRUE, step = 1L,
                        skip = 0L) {

    if (!tsibble::is_tsibble(df)) {
      cli::cli_abort(c(
        "{.var df} must be {.cls tbl_ts}",
        "x" = "Got a {.cls {class(df)}}"
      ))
    }

    if (!inherits(recipe, "recipe")) {
      cli::cli_abort(c(
        "{.var recipe} must be a {.cls recipe}",
        "x" = "Got a {.cls {class(recipe)}}"
      ))
    }

    # TODO we will probably want to take a prepared recipe so we are able to
    # use this also for the validation data.
    self$recipe <- recipes::prep(recipe, df)

    self$df <-  self$recipe %>%
      recipes::juice() %>%
      dplyr::arrange(!!!tsibble::index_var(df))

    # we create rsample `split` objects that don't materialize the data until
    # `training` or `testing` allowing us to compute the number of slices that
    # we are able to create.
    # since data is already split between future and past values it's also easier
    # to reason when implementing the network.
    self$slices <- self$df %>%
      dplyr::ungroup() %>%
      dplyr::mutate(.row = dplyr::row_number()) %>%
      dplyr::group_split(!!!tsibble::key(df)) %>%
      purrr::discard(~nrow(.x) < (lookback + 1 + assess_stop)) %>%
      purrr::map_dfr(~rsample::sliding_index(
        dplyr::arrange(.x, !!!tsibble::index_var(df)),
        index = tsibble::index_var(df),
        lookback = lookback * lubridate::as.period(tsibble::interval(df)),
        assess_stop = assess_stop * lubridate::as.period(tsibble::interval(df)),
        assess_start = assess_start * lubridate::as.period(tsibble::interval(df)),
        skip = skip,
        complete = complete,
        step = step
      ))

    self$slices <- self$slices$splits %>% purrr::map(
      ~list(
        encoder = rsample::training(.x)$.row,
        decoder = rsample::testing(.x)$.row
      )
    )

    # figure out variables of each type
    terms <- self$recipe$term_info

    self$known <- terms %>%
      pull_term_names(role %in% c("known"), !(variable %in% tsibble::key_vars(df)))
    self$observed <- terms %>%
      pull_term_names(role %in% c("predictor", "outcome"), !(variable %in% tsibble::key_vars(df)))

    self$past <- purrr::map2(self$known, self$observed, ~c(.x, .y))

    self$static <- terms %>%
      pull_term_names(role %in% c("static", "predictor"), variable %in% tsibble::key_vars(df))

    self$target <- terms %>%
      pull_term_names(role == "outcome")

    # compute feature sizes
    dictionary <- self$recipe$levels %>%
      purrr::keep(~isTRUE(.x$factor)) %>%
      purrr::map(~length(.x$values))

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

