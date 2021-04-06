#' conditionning input data into tensors according to tft variable roles
#'
#' @param df a data frame
#' @param recipe a recipe affecting tft roles to df
#' @param total_time_steps time_step value (default 48)
batch_data <- function(recipe, df, total_time_steps = 12, device) {
  if (device == "auto") {
    if (torch::cuda_is_available())
      device <- "cuda"
    else
      device <- "cpu"
  }

  var_type_role <- summary(recipe)
  id <- recipes::terms_select(var_type_role, term=quos(recipes::has_role("id")))
  time <- recipes::terms_select(var_type_role, term=quos(recipes::has_role("time")))
  all_numeric <- recipes::terms_select(var_type_role, term=quos(recipes::all_numeric()))
  all_nominal <- recipes::terms_select(var_type_role, term=quos(recipes::all_nominal()))
  known <- recipes::terms_select(var_type_role, term=quos(recipes::has_role("known_input")))
  observed <- recipes::terms_select(var_type_role, term=quos(recipes::has_role("observed_input")))
  static <- recipes::terms_select(var_type_role, term=quos(recipes::has_role("static_input")))
  known_numeric <- intersect(known, all_numeric)
  known_categorical <- intersect(known, all_nominal)
  observed_numeric <- intersect(observed, all_numeric)
  observed_categorical <- intersect(observed, all_nominal)
  target_numeric <- intersect(terms_select(var_type_role, term=quos(recipes::all_outcomes())), all_numeric)
  target_categorical <- intersect(terms_select(var_type_role, term=quos(recipes::all_outcomes())), all_nominal)
  static_numeric <- intersect(static, all_numeric)
  static_categorical <- intersect(static, all_nominal)


  # TODO use tsibble function and interval attribute to reach the same result
  # TODO get rid or the remaining hardcoded $id
  positions <- df %>%
    dplyr::group_by(dplyr::across(id)) %>%
    dplyr::filter(dplyr::n() >= (total_time_steps*2+1)) %>%
    dplyr::group_split(.keep = TRUE) %>%
    purrr::map_dfr(
      ~tibble::tibble(
        id = dplyr::first(.x$id),
        start_time = seq(
          # TODO don't hardcode Time
          from = min(.x$Time),
          # TODO don't hardcode steps to be hours here
          to   = max(.x$Time) - lubridate::hours(total_time_steps),
          by   = "hours"
        ),
        end_time = start_time + lubridate::hours(total_time_steps)
      )
    )
  if (nrow(positions)<500) {
    rlang::warn(glue::glue("total_time_steps={total_time_steps} hours does not allow to extract 500 samples, you should lower its value"))
  }
  if (nrow(positions)<2) {
    rlang::abort(glue::glue("total_time_steps={total_time_steps} hours does not allow to extract samples, you should lower its value"))
  }

  # positions <- positions %>% dplyr::sample_n(500)
  positions <- positions %>% dplyr::sample_n(50)

  output <- positions %>%
    dplyr::group_split(id, start_time) %>%
    # TODO don't hardcode Time
    purrr::map(
      ~df %>%
        dplyr::filter(
          id == .x$id,
          Time >= .x$start_time,
          Time < .x$end_time
        )
    )
# TODO BUG 10 groups do not size total_time_steps*2 and should be removed or torch_stack will fail
  output <-output[output %>% purrr::map_lgl(~nrow(.x)==24)]

  known_t <- list(
    numerics = output %>%
        purrr::map(~.x %>% dplyr::select(dplyr::all_of(known_numeric)) %>% df_to_tensor(device = device)) %>%
        torch::torch_stack(),
    categorical = output %>%
        purrr::map(~.x %>% dplyr::select(dplyr::all_of(known_categorical)) %>% df_to_tensor(device = device)) %>%
        torch::torch_stack()
  )

  observed_t <- list(
    numerics = output %>%
      purrr::map(~.x %>% dplyr::select(dplyr::all_of(observed_numeric)) %>% df_to_tensor(device = device)) %>%
      torch::torch_stack(),
    categorical = output %>%
      purrr::map(~.x %>% dplyr::select(dplyr::all_of(observed_categorical)) %>% df_to_tensor(device = device)) %>%
      torch::torch_stack()
  )

  target_t <- list(
    numerics = output %>%
      purrr::map(~.x %>% dplyr::select(dplyr::all_of(target_numeric)) %>% df_to_tensor(device = device)) %>%
      torch::torch_stack(),
    categorical = output %>%
      purrr::map(~.x %>% dplyr::select(dplyr::all_of(target_categorical)) %>% df_to_tensor(device = device)) %>%
      torch::torch_stack()
  )

  static_t <- list(
    numerics = output %>%
      purrr::map(~.x %>% dplyr::select(dplyr::all_of(static_numeric)) %>% df_to_tensor(device = device)) %>%
      torch::torch_stack(),
    categorical = output %>%
      purrr::map(~.x %>% dplyr::select(dplyr::all_of(static_categorical)) %>% df_to_tensor(device = device)) %>%
      torch::torch_stack()
  )
  cat_idxs = which(names(df) %in% c(known_categorical, observed_categorical, target_categorical, static_categorical))
  list(known = known_t,
       observed = observed_t,
       static = static_t,
       target = target_t,
       input_dim = sum(length(c(time, id, known, observed, static))),
       cat_idxs = cat_idxs ,
       cat_dims = purrr::map(cat_idxs, ~length(unique(df[[.x]]))),
       known_idx = which(names(df) %in% c(known_numeric, known_categorical)),
       observed_idx = which(names(df) %in%  c(observed_numeric, observed_categorical)),
       static_idx = which(names(df) %in% c(static_numeric, static_categorical)),
       output_dim = length(target_categorical) + length(target_numeric)
  )
}

df_to_tensor <- function(df, device) {
  df %>%
    dplyr::mutate(dplyr::across(where(is.factor), as.integer)) %>%
    dplyr::mutate(dplyr::across(where(lubridate::is.Date), as.integer)) %>%
    dplyr::mutate(dplyr::across(where(lubridate::is.POSIXt), as.integer)) %>%
    as.matrix() %>%
    torch::torch_tensor(device = device,dtype = torch::torch_float())
}

#' Configuration for Tft models
#'
#' @param batch_size (int) Number of examples per batch, large batch sizes are
#'   recommended. (default: 1024)
#' @param clip_value If a float is given this will clip the gradient at
#'   clip_value. Pass `NULL` (default) to not clip.
#' @param loss (character or function) Loss function for training within
#' ["quantile_loss", "pinball_loss", "rmsse_loss", "smape_loss"] (default to ["quantile_loss"])
#' @param epochs (int) Number of training epochs.
#' @param drop_last (bool) Whether to drop last batch if not complete during
#'   training
#' @param total_time_steps (int) Size of the look-back time window + forecast horizon in steps. .
#' @param num_encoder_steps (int) Size of the look-back time window in steps.
#' @param num_steps (int) Number of steps in the architecture
#'   (usually between 3 and 10)
#' @param quantiles (list) list of quantiles forcasts. (default = [list(0.5)]).
#' @param virtual_batch_size (int) Size of the mini batches used for
#'   Batch Normalization (default=256)
#' @param learn_rate initial learning rate for the optimizer.
#' @param optimizer the optimization method. currently only 'adam' is supported,
#'   you can also pass any torch optimizer function.
#' @param valid_split (float) The fraction of the dataset used for validation.
#' @param hidden_layer_size (int)size of the hidden layer (default=160).
#' @param dropout_rate dropout rate applied to each nn block (default=0.3)
#' @param stack_size (int) size of the stack (default=3)
#' @param num_heads (int) number of attention head (default=1)
#' @param verbose (bool) wether to print progress and loss values during
#'   training.
#' @param lr_scheduler if `NULL`, (default) no learning rate decay is used. if ["step"]
#'   decays the learning rate by `lr_decay` every `step_size` epochs. It can
#'   also be a [torch::lr_scheduler] function that only takes the optimizer
#'   as parameter. The `step` method is called once per epoch.
#' @param lr_decay multiplies the initial learning rate by `lr_decay` every
#'   `step_size` epochs. Unused if `lr_scheduler` is a `torch::lr_scheduler`
#'   or `NULL`.
#' @param step_size number of epoch before modifying learning rate by `lr_decay`.
#'   Unused if `lr_scheduler` is a `torch::lr_scheduler` or `NULL`.
#' @param cat_emb_dim (int or list) Embedding size for categorial features,
#'   broadcasted to each categorical feature, or per categorical feature
#'   when a list of the same size as the categorical features  (default=1)
#' @param momentum Momentum for batch normalization, typically ranges from 0.01
#'   to 0.4 (default=0.02)
#' @param checkpoint_epochs checkpoint model weights and architecture every
#'   `checkpoint_epochs`. (default is 10). This may cause large memory usage.
#'   Use `0` to disable checkpoints.
#' @param device the device to use for training. "cpu" or "cuda". The default ("auto")
#'   uses  to "cuda" if it's available, otherwise uses "cpu".
#' @param importance_sample_size sample of the dataset to compute importance metrics.
#'   If the dataset is larger than 1e5 obs we will use a sample of size 1e5 and
#'   display a warning.
#'
#' @return A named list with all hyperparameters of the TabNet implementation.
#'
#' @export
tft_config <- function(batch_size = 256,
                       clip_value = NULL,
                       loss = "quantile_loss",
                       epochs = 5,
                       drop_last = FALSE,
                       total_time_steps = NULL,
                       num_encoder_steps = NULL,
                       quantiles = list(0.5),
                       virtual_batch_size = 256,
                       valid_split = 0,
                       learn_rate = 2e-2,
                       optimizer = "adam",
                       lr_scheduler = NULL,
                       lr_decay = 0.1,
                       step_size = 30,
                       checkpoint_epochs = 10,
                       cat_emb_dim = 1,
                       hidden_layer_size = 160,
                       dropout_rate = 0.3,
                       stack_size = 3,
                       num_heads = 1,
                       verbose = FALSE,
                       device = "auto") {

# TODO add assert parameters consistency
# assert forecast horizon shall be 1 or more
if (is.null(total_time_steps)) {
  rlang::abort("total_time_steps is missing in tft_config() and cannot be guessed")
}
if (is.null(num_encoder_steps)) {
  rlang::abort("num_encoder_steps is missing in tft_config() and cannot be guessed")
}
if ((total_time_steps - num_encoder_steps) < 2) {
  rlang::abort("The forecast horizon (total_time_steps - num_encoder_steps) shall be at least 1")
}

  list(
    batch_size = batch_size,
    clip_value = clip_value,
    loss = loss,
    epochs = epochs,
    drop_last = drop_last,
    total_time_steps = total_time_steps,
    num_encoder_steps = num_encoder_steps,
    quantiles = quantiles,
    minibatch_size = virtual_batch_size,
    valid_split = valid_split,
    verbose = verbose,
    learn_rate = learn_rate,
    optimizer = optimizer,
    lr_scheduler = lr_scheduler,
    lr_decay = lr_decay,
    step_size = step_size,
    cat_emb_dim = cat_emb_dim,
    hidden_layer_size = hidden_layer_size,
    dropout_rate = dropout_rate,
    stack_size = stack_size,
    checkpoint_epochs = checkpoint_epochs,
    num_heads = num_heads,
    device = device
  )
}

batch_to_device <- function(batch, device) {
  batch <- list(x = batch$x, y  = batch$y)
  lapply(batch, function(x) {
    x$to(device = device)
  })
}

train_batch <- function(network, optimizer, batch, config) {
  # forward pass
  output <- network(batch$x)
  loss <- config$loss_fn(output[[1]], batch$y)

  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * output[[2]]

  # step of the optimization
  optimizer$zero_grad()
  loss$backward()
  if (!is.null(config$clip_value)) {
    torch::nn_utils_clip_grad_norm_(network$parameters, config$clip_value)
  }
  optimizer$step()

  list(
    loss = loss$item()
  )
}

valid_batch <- function(network, batch, config) {
  # forward pass
  output <- network(batch$x)
  loss <- config$loss_fn(output[[1]], batch$y)

  # Add the overall sparsity loss
  loss <- loss - config$lambda_sparse * output[[2]]

  list(
    loss = loss$item()
  )
}

transpose_metrics <- function(metrics) {
  nms <- names(metrics[1])
  out <- vector(mode = "list", length = length(nms))
  for (nm in nms) {
    out[[nm]] <- vector("numeric", length = length(metrics))
  }

  for (i in seq_along(metrics)) {
    for (nm in nms) {
      out[[nm]][i] <- metrics[i][[nm]]
    }
  }

  out
}

tft_initialize <- function(batch_data, config = tft_config()) {

  torch::torch_manual_seed(sample.int(1e6, 1))
  has_valid <- config$valid_split > 0

  if (config$device == "auto") {
    if (torch::cuda_is_available())
      device <- "cuda"
    else
      device <- "cpu"
  } else {
    device <- config$device
  }

  # TODO processed data shall be splitted upfront if needed
  # if (has_valid) {
  #   n <- nrow(df)
  #   valid_idx <- sample.int(n, n*config$valid_split)
  #
  #   valid_data <- df[valid_idx, ]
  #   df <- df[-valid_idx, ]
  # }

  # create network
  network <- tft_nn(
    input_dim = data$input_dim,
    output_dim = data$output_dim,
    cat_idxs = data$cat_idx,
    cat_dims = data$cat_dims,
    cat_emb_dim = config$cat_emb_dim,
    static_idx = data$static_idx,
    known_idx = data$known_idx,
    input_idx = data$input_idx,
    total_time_steps = config$total_time_steps,
    num_encoder_steps = config$num_encoder_steps,
    quantiles = config$quantiles,
    minibatch_size = config$minibatch_size,
    hidden_layer_size = config$hidden_layer_size,
    dropout_rate = config$dropout_rate,
    stack_size = config$stack_size,
    num_heads = config$num_heads
  )

  # main loop
  metrics <- list()
  checkpoints <- list()


  importances <- tibble::tibble(
    variables = colnames(x),
    importance = NA
  )

  list(
    network = network,
    metrics = metrics,
    config = config,
    checkpoints = checkpoints,
    importances = importances
  )
}

tft_train <- function(obj, df, transform, config = tft_config(), epoch_shift=0L) {
  stopifnot("tft_model shall be initialised"= (length(obj$fit$network) > 0))
  torch::torch_manual_seed(sample.int(1e6, 1))
  has_valid <- config$valid_split > 0

  if (config$device == "auto") {
    if (torch::cuda_is_available())
      device <- "cuda"
    else
      device <- "cpu"
  } else {
    device <- config$device
  }

  if (has_valid) {
    n <- nrow(df)
    valid_idx <- sample.int(n, n*config$valid_split)

    valid_data <- df[valid_idx, ]
    df <- df[-valid_idx, ]
  }

  # training data (could have changed)
  data <- batch_data(df, transform)
  dl <- torch::dataloader(
    torch::tensor_dataset(x = data$x, y = data$y),
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE
  )

  # validation data (could have changed)
  if (has_valid) {
    valid_data <- batch_data(valid_data, transform)
    valid_dl <- torch::dataloader(
      torch::tensor_dataset(x = valid_data$x, y = valid_data$y),
      batch_size = config$batch_size,
      drop_last = FALSE,
      shuffle = FALSE
    )
  }

  # resolve loss (could have changed)
  if (config$loss == "quantile_loss")
    config$loss_fn <- quantile_loss()
  else if (config$loss == "pinball_loss")
    config$loss_fn <- pinball_loss()
  else if (config$loss == "rmsse_loss")
    config$loss_fn <- rmsse_loss()
  else if (config$loss == "smape_loss")
    config$loss_fn <- smape_loss()

  # restore network from model and send it to device
  network <- obj$fit$network

  network$to(device = device)

  # define optimizer

  if (rlang::is_function(config$optimizer)) {

    optimizer <- config$optimizer(network$parameters, config$learn_rate)

  } else if (rlang::is_scalar_character(config$optimizer)) {

    if (config$optimizer == "adam")
      optimizer <- torch::optim_adam(network$parameters, lr = config$learn_rate)
    else
      rlang::abort("Currently only the 'adam' optimizer is supported.")

  }

  # define scheduler
  if (is.null(config$lr_scheduler)) {
    scheduler <- list(step = function() {})
  } else if (rlang::is_function(config$lr_scheduler)) {
    scheduler <- config$lr_scheduler(optimizer)
  } else if (config$lr_scheduler == "step") {
    scheduler <- torch::lr_step(optimizer, config$step_size, config$lr_decay)
  }

  # restore previous metrics & checkpoints
  metrics <- obj$fit$metrics
  checkpoints <- obj$fit$checkpoints

  # main loop
  for (epoch in seq_len(config$epochs)+epoch_shift) {

    metrics[[epoch]] <- list(train = NULL, valid = NULL)
    train_metrics <- c()
    valid_metrics <- c()

    network$train()

    if (config$verbose)
      pb <- progress::progress_bar$new(
        total = length(dl),
        format = "[:bar] loss= :loss"
      )

    coro::loop(for (batch in dl) {
      m <- train_batch(network, optimizer, batch_to_device(batch, device), config)
      if (config$verbose) pb$tick(tokens = m)
      train_metrics <- c(train_metrics, m)
    })
    metrics[[epoch]][["train"]] <- transpose_metrics(train_metrics)

    if (config$checkpoint_epochs > 0 && epoch %% config$checkpoint_epochs == 0) {
      network$to(device = "cpu")
      checkpoints[[length(checkpoints) + 1]] <- model_to_raw(network)
      network$to(device = device)
    }

    network$eval()
    if (has_valid) {
      for (batch in torch::enumerate(valid_dl)) {
        m <- valid_batch(network, batch_to_device(batch, device), config)
        valid_metrics <- c(valid_metrics, m)
      }
      metrics[[epoch]][["valid"]] <- transpose_metrics(valid_metrics)
    }

    message <- sprintf("[Epoch %03d] Loss: %3f", epoch, mean(metrics[[epoch]]$train$loss))
    if (has_valid)
      message <- paste0(message, sprintf(" Valid loss: %3f", mean(metrics[[epoch]]$valid$loss)))

    if (config$verbose)
      rlang::inform(message)

    scheduler$step()
  }

  network$to(device = "cpu")

  importance_sample_size <- config$importance_sample_size
  if (is.null(config$importance_sample_size) && data$x$shape[1] > 1e5) {
    rlang::warn(c(glue::glue("Computing importances for a dataset with size {data$x$shape[1]}."),
                  "This can consume too much memory. We are going to use a sample of size 1e5",
                  "You can disable this message by using the `importance_sample_size` argument."))
    importance_sample_size <- 1e5
  }
  indexes <- torch::torch_randint(
    1, data$x$shape[1], min(importance_sample_size, data$x$shape[1]),
    dtype = torch::torch_long()
  )
  importances <- tibble::tibble(
    variables = colnames(x),
    importance = compute_feature_importance(network, data$x[indexes,..])
  )

  list(
    network = network,
    metrics = metrics,
    config = config,
    checkpoints = checkpoints,
    importances = importances
  )
}

predict_impl <- function(obj, df, transform, batch_size = 1e5) {
  data <- batch_data(df, transform)

  network <- obj$fit$network
  network$eval()

  splits <- torch::torch_split(data$x, split_size = 10000)
  splits <- lapply(splits, function(x) network(x)[[1]])
  torch::torch_cat(splits)
}

predict_impl_numeric <- function(obj, x, batch_size) {
  p <- as.numeric(predict_impl(obj, x, batch_size))
  hardhat::spruce_numeric(p)
}

get_blueprint_levels <- function(obj) {
  levels(obj$blueprint$ptypes$outcomes[[1]])
}

predict_impl_prob <- function(obj, x, batch_size) {
  p <- predict_impl(obj, x, batch_size)
  p <- torch::nnf_softmax(p, dim = 2)
  p <- as.matrix(p)
  hardhat::spruce_prob(get_blueprint_levels(obj), p)
}

predict_impl_class <- function(obj, x, batch_size) {
  p <- predict_impl(obj, x, batch_size)
  p <- torch::torch_max(p, dim = 2)
  p <- as.integer(p[[2]])
  p <- get_blueprint_levels(obj)[p]
  p <- factor(p, levels = get_blueprint_levels(obj))
  hardhat::spruce_class(p)

}