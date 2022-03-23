
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
    split <- self$slices$splits[[i]]
    x <- rsample::training(split)
    y <- rsample::testing(split)

    encoder <- list()
    for (type in c("past", "static")) {
      encoder[[type]] <- list()
      for (dtype in c("num", "cat")) {
        vars <- x[,self[[type]][[dtype]]] %>%
          self$to_tensor()
        if (type == "static") {
          vars <- vars[1,]
        }
        encoder[[type]][[dtype]] <- vars
      }
    }

    decoder <- list()
    for (type in c("known", "target")) {
      decoder[[type]] <- list()
      for(dtype in c("num", "cat")) {
        decoder[[type]][[dtype]] <- y[,self[[type]][[dtype]]] %>%
          self$to_tensor()
      }
    }

    list(list(encoder = encoder, decoder = decoder), decoder$target$num)
  },
  .length = function() {
    nrow(self$slices)
  },
  to_tensor = function(df) {
    df %>%
      dplyr::mutate(dplyr::across(where(is.factor), as.integer)) %>%
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

