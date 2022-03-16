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

dataset <- time_series_dataset(df, recipe)
n_features <- get_n_features(dataset[1])
hidden_state_size <- 128
feature_sizes <- dataset$feature_sizes

x <- coro::collect(torch::dataloader(dataset, batch_size = 32), 1)[[1]]


#' Temporal Fusion Transformer Module
#'
#'
#' @param n_features a list containing the shapes for all necessary information
#'        to define the size of layers, including:
#'                   - `$encoder$known$(num|cat)`: shape of known features
#'                   - `$encoder$unknown$(num|cat)`: shape of unknown features
#'                   - `$encoder$static$(num|cat)`: shape of the static features
#'                   - `$decoder$target$(num|cat)`: shape of the targets.
#'        We exclude the batch dimension.
#' @param hidden_state_size The size of the model shared accross multiple parts
#'        of the architecture.
temporal_fusion_transformer <- torch::nn_module(
  "temporal_fusion_transformer",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    self$preprocessing <- preprocessing(n_features, feature_sizes, hidden_state_size)
  },
  forward = function(x) {
    x <- self$preprocessing(x)
  }
)

temporal_selection <- torch::nn_module(
  "selection",
  initialize = function(n_features, hidden_state_size) {

    self$static <- variable_selection_network(
      n_features = sum(as.numeric(n_features$encoder$static)),
      hidden_state_size = hidden_state_size
    )

    self$temporal_variable_selection_context <- gated_residual_network(
      input_size = hidden_state_size,
      output_size = hidden_state_size,
      hidden_state_size = hidden_state_size
    )

    self$known <- variable_selection_network(
      n_features = sum(as.numeric(n_features$encoder$known)),
      hidden_state_size = hidden_state_size
    )

  },
  forward = function(x) {

    x$encoder$static <- x$encoder$static %>%
      torch::torch_unsqueeze(dim = 2) %>%
      self$static() %>%
      torch::torch_squeeze(dim = 2)

    x$static_context <- list(
      temporal_variable_selection = self$temporal_variable_selection_context(
        x$encoder$static
      )
    )

    x$encoder$known <- x$encoder$known %>%
      self$known(context = x$static_context$temporal_variable_selection)


    x
  }
)

variable_selection_network <- torch::nn_module(
  "variable_selection_network",
  initialize = function(n_features, hidden_state_size) {
    self$global <- time_distributed(gated_residual_network(
      input_size = n_features*hidden_state_size,
      output_size = n_features,
      hidden_state_size = hidden_state_size
    ))
    self$local <- seq_len(n_features) %>%
      purrr::map(~time_distributed(gated_residual_network(
        input_size = hidden_state_size,
        output_size = hidden_state_size,
        hidden_state_size = hidden_state_size
      )))
  },
  forward = function(x, context = NULL) {
    v <- self$global(x, context = context) %>%
      torch::nnf_softmax(dim = 3) %>%
      torch::torch_unsqueeze(dim = 4)

    x <- x %>%
      torch::torch_unbind(dim = 3) %>%
      purrr::map2(self$local, ~.y(.x)) %>%
      torch::torch_stack(dim = 3)

    torch::torch_sum(v*x, dim = 3)
  }
)

gated_residual_network <- torch::nn_module(
  "gated_residual_network",
  initialize = function(input_size, output_size, hidden_state_size, dropout = 0.1) {
    self$input <- torch::nn_linear(input_size, hidden_state_size)
    self$context <- torch::nn_linear(hidden_state_size, hidden_state_size, bias = FALSE)
    self$hidden <- torch::nn_linear(hidden_state_size, hidden_state_size)
    self$dropout <- torch::nn_dropout(dropout)
    self$gate <- gated_linear_unit(hidden_state_size, output_size)
    self$norm <- torch::nn_layer_norm(output_size)
    self$elu <- torch::nn_elu()
    if (input_size == output_size) {
      self$skip <- torch::nn_identity()
    } else {
      self$skip <- torch::nn_linear(input_size, output_size)
    }
  },
  forward = function(x, context = NULL) {

    if (x$ndim > 2) {
      x <- torch::torch_flatten(x, start_dim = 2)
    }

    skip <- self$skip(x)
    x <- self$input(x)

    if (!is.null(context)) {
      x <- x + self$context(context)
    }

    hidden <- x %>%
      self$elu() %>%
      self$hidden() %>%
      self$dropout()

    self$norm(skip + self$gate(hidden))
  }
)

gated_linear_unit <- torch::nn_module(
  "gated_linear_unit",
  initialize = function(input_size, output_size) {
    self$gate <- torch::nn_sequential(
      torch::nn_linear(input_size, output_size),
      torch::nn_sigmoid()
    )
    self$activation <- torch::nn_linear(input_size, output_size)
  },
  forward = function(x) {
    self$gate(x) * self$activation(x)
  }
)

preprocessing <- torch::nn_module(
  "preprocessing",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    for (type in c("known", "static", "observed")) {
      self[[type]] <- preprocessing_group(
        n_features = n_features$encoder[[type]],
        feature_sizes = feature_sizes[[type]],
        hidden_state_size = hidden_state_size
      )
    }
  },
  forward = function(x) {
    x$encoder$known <- self$known(x$encoder$known)
    x$decoder$known <- self$known(x$decoder$known)
    # add a time dimension temporarily
    x$encoder$static <- x$encoder$static %>%
      purrr::map(~torch::torch_unsqueeze(.x, dim = 2)) %>%
      self$static() %>%
      torch::torch_squeeze(dim = 2)
    x$encoder$observed <- self$observed(x$encoder$observed)
    x
  }
)


#' Preprocess a group of time-varying variables
#'
#' Handles both numeric and categorical variables.
#' Each numeric variables passes trough a linear transformation that is shared
#' accross every time step.
#' Categorical variables are represented trough embeddings.
#'
preprocessing_group <- torch::nn_module(
  "preprocessing_group",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    self$num <- linear_preprocessing(n_features$num, hidden_state_size)
    self$cat <- embedding_preprocessing(n_features$cat, feature_sizes, hidden_state_size)
  },
  forward = function(x) {
    x$num <- self$num(x$num)
    x$cat <- self$cat(x$cat)
    torch::torch_cat(x, dim = 3)
  }
)

linear_preprocessing <- torch::nn_module(
  "linear_preprocessing",
  initialize = function(n_features, hidden_state_size) {
    self$linears <- seq_len(n_features) %>%
      purrr::map(~time_distributed(torch::nn_linear(1, hidden_state_size))) %>%
      torch::nn_module_list()
  },
  #' @param x `Tensor[batch, time_steps, n_features]`
  forward = function(x) {
    if (x$size(3) == 0) return(NULL)
    x %>%
      torch::torch_unsqueeze(dim = 4) %>%
      torch::torch_unbind(dim = 3) %>%
      purrr::imap(~self$linears[[.y]](.x)) %>%
      torch::torch_stack(dim = 3)
  }
)

embedding_preprocessing <- torch::nn_module(
  "embedding_preprocessing",
  initialize = function(n_features, feature_sizes, hidden_state_size) {
    self$embeddings <- feature_sizes %>%
      purrr::map(~time_distributed(torch::nn_embedding(.x, hidden_state_size))) %>%
      torch::nn_module_list()
  },
  #' @param x `Tensor[batch, time_steps, n_features]`
  forward = function(x) {
    if (x$size(3) == 0) return(NULL)
    x %>%
      torch::torch_unbind(dim = 3) %>%
      purrr::imap(~self$embeddings[[.y]](.x)) %>%
      torch::torch_stack(dim = 3)
  }
)

time_distributed <- torch::nn_module(
  "time_distributed",
  initialize = function(module) {
    self$module <- module
  },
  forward = function(x, ...) {
    extra_args <- list(...)
    x %>%
      torch::torch_unbind(dim = 2) %>%
      purrr::map(~rlang::exec(self$module, .x, !!!extra_args)) %>%
      torch::torch_stack(dim = 2)
  }
)
