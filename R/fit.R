
#' Temporal Fusion Transformer
#'
#' @export
tft <- function(x, ...) {
  UseMethod("tft")
}

#' @export
tft.default <- function(x, ...) {
  cli::cli_abort(
    "{.var tft} is not defined for objects with class {.cls {class(x)}}.")
}

#' @export
tft.data.frame <- function(x, y, ...) {
  processed <- hardhat::mold(x, y)
  tft_bridge(processed)
}

#' @export
tft.recipe <- function(x, data, ...) {
  processed <- hardhat::mold(x, data)
  tft_bridge(processed)
}

tft_bridge <- function(processed) {

  if (is.null(attr(processed$outcomes, "roles"))) {
    cli::cli_abort(c(
      "The provided {.cls {class(processed$outcomes)}} doesn't include role information.",
      "i" = "You should use {.var step_include_roles} in the {.cls recipe} so the role information is correctly passed to the model."
    ))
  }

  data <- dplyr::bind_cols(processed$predictors, processed$outcomes)

  result <- tft_impl(
    x = data,
    recipe = attr(processed$predictors, "recipe")
  )

  new_tft(
    module = result$module,
    future_data = result$future_data,
    past_data = result$past_data,
    blueprint = processed$blueprint
  )
}

new_tft <- function(module, future_data, past_data, blueprint) {
  hardhat::new_model(
    module = module,
    future_data = future_data,
    past_data = past_data,
    blueprint = blueprint,
    class = "tft"
  )
}

tft_impl <- function(x, recipe) {
  dataset <- time_series_dataset(x, recipe$term_info, lookback = 120, assess_stop = 4)

  n_features <- get_n_features(dataset[1][[1]])
  hidden_state_size <- 16
  feature_sizes <- dataset$feature_sizes

  future_data <- prepare_future_data(
    df = dataset$df,
    recipe = recipe,
    horizon = 4
  )

  result <- temporal_fusion_transformer %>%
    luz::setup(
      loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
      optimizer = torch::optim_adam
    ) %>%
    luz::set_hparams(
      num_features = n_features,
      feature_sizes = feature_sizes,
      hidden_state_size = 16,
      dropout = 0.1,
      num_quantiles = 3,
      num_heads = 1
    ) %>%
    luz::set_opt_hparams(
      lr = 0.03
    ) %>%
    fit(dataset, epochs = 5, dataloader_options = list(
      batch_size = 256, num_workers = 0
    ))

  list(future_data = future_data, past_data = dataset$df, module = result)
}

prepare_future_data <- function(df, recipe, horizon = 4) {

  if (is.null(recipe)) {
    cli::cli_abort(c(
      "The recipe describing data preprocessing must be use {.var step_include_roles()}",
      "x" = "Use {.var step_include_roles()} as the last step of the recipe."
    ))
  }

  new_obs <- tsibble::new_data(df, n = horizon)

  var_info <- recipe$var_info
  keys <- var_info$variable[var_info$role %in% c("key")]
  index <- var_info$variable[var_info$role %in% c("index")]

  if (!all(names(new_obs) %in% c(keys, index))) {
    cli::cli_abort(c(
      "New data does not include all the {.var keys} and {.var index}.",
      "x" = "Missing {.var {setdiff(names(new_obs), c(keys, index))}}"
    ))
  }

  # we now add all `predictors` to the new_obs dataset.
  predictors <- var_info$variable[var_info$role == "predictor"]
  for (var in predictors) {
    new_obs[[var]] <- NA
  }

  # we should now `bake` the recipe for the `new_obs` dataset and make sure we
  # could compute all `static` and `known` variables.
  # We will make sure no `NA` is found in `static`, `keys` `index` and `known`.
  new_obs <- recipes::bake(recipe, new_obs)

  term_info <- recipe$term_info
  known <- term_info$variable[term_info$tft_role %in% c("key", "index", "static", "known")]
  for (var in known) {
    if (any(is.na(new_obs[[var]]))) {
      cli::cli_abort(c(
        "Found missing values in at least opne known variable.",
        "i" = "This kind of variables should be fully known for future inputs.",
        "x" = "Check variable {.var {var}}."
      ))
    }
  }

  new_obs
}
