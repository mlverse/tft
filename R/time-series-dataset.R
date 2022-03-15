library(recipes)

data(walmart_sales, package = "walmartdata")
df <- walmart_sales %>%
  mutate(
    Store = as.factor(Store),
    Dept = as.factor(Dept)
  ) %>%
  tsibble::tsibble(
    key = c(Store, Dept, Type, Size),
    index = Date
  )

recipe <- recipe(Weekly_Sales ~ ., data = df) %>%
  update_role(IsHoliday, new_role = "known") %>%
  step_date(Date, role = "known", features = c("year", "month", "doy")) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_indicate_na(starts_with("MarkDown")) %>%
  step_impute_mean(starts_with("Markdown"))

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
      pull_term_names(role %in% c("predictor", "target"), !(variable %in% tsibble::key_vars(df)))
    self$static <- terms %>%
      pull_term_names(role %in% c("static", "predictor"), variable %in% tsibble::key_vars(df))
    self$target <- terms %>%
      pull_term_names(role == "outcome")

  },
  .getitem = function(i) {
    split <- self$slices$splits[[i]]
    x <- rsample::training(split)
    y <- rsample::testing(split)

    encoder <- list()
    for (type in c("known", "static", "observed")) {
      encoder[[type]] <- list()
      for (dtype in c("num", "cat")) {
        vars <- x %>%
          dplyr::select(!!!self[[type]][[dtype]]) %>%
          to_tensor()
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
        decoder[[type]][[dtype]] <- y %>%
          dplyr::select(!!!self[[type]][[dtype]]) %>%
          to_tensor()
      }
    }

    list(encoder = encoder, decoder = decoder)
  },
  .length = function() {
    nrow(self$slices)
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

to_tensor <- function(df) {
  df %>%
    dplyr::mutate(dplyr::across(where(is.factor), as.integer)) %>%
    as.matrix() %>%
    torch::torch_tensor(dtype = torch::torch_float())
}

