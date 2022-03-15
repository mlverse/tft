#' conditionning input data into tensors according to tft variable roles
#'
#' @param df a data frame
#' @param recipe a recipe affecting tft roles to df
#' @param total_time_steps time_step value (default 48)
#' @param device the device to use for training. `cpu` or `cuda`. The default (`auto`)
#'   uses `cuda` if it's available, otherwise uses `cpu`.
batch_data <- function(recipe, df, total_time_steps = 12, device) {
  if (device == "auto") {
    if (torch::cuda_is_available())
      device <- "cuda"
    else
      device <- "cpu"
  }

  var_type_role <- summary(recipe)
  id <- recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::has_role("id")))
  time <- recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::has_role("time")))
  all_numeric <- c(recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::all_numeric())),
                   var_type_role[var_type_role$type == "date", "variable"] %>% unlist ) %>%
    as.character()
  all_nominal <- c(recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::all_nominal())),
                   var_type_role[var_type_role$type %in% c("logical", "other"), "variable"] %>% unlist) %>%
    as.character()
  known <- recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::has_role("known_input")))
  observed <- recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::has_role("observed_input")))
  static <- recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::has_role("static_input")))
  known_numeric <- intersect(known, all_numeric)
  known_categorical <- intersect(known, all_nominal)
  observed_numeric <- intersect(observed, all_numeric)
  observed_categorical <- intersect(observed, all_nominal)
  target_numeric <- intersect(recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::all_outcomes())), all_numeric)
  target_categorical <- intersect(recipes::recipes_eval_select(df, info=var_type_role, quos=rlang::quos(recipes::all_outcomes())), all_nominal)
  static_numeric <- intersect(static, all_numeric)
  static_categorical <- intersect(static, all_nominal)


  processed_roles <- hardhat::mold(recipe, df)

  # the as.numeric(as.factor) trick is required to prevent logicals to become tensors in [0,1] unexpected by nn_embedding
  output <- df %>%
    dplyr::mutate(dplyr::across(dplyr::all_of(all_nominal), ~as.numeric(as.factor(.x)))) %>%
    dplyr::group_by(rlang::eval_tidy(static)) %>%
    slider::slide(~.x, .before=total_time_steps-1, .complete = TRUE) %>%
    purrr::compact() %>%
    purrr::map(dplyr::ungroup)


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
  cat_idxs = c(which(names(df) %in% known_categorical),
               which(names(df) %in% observed_categorical),
               which(names(df) %in% static_categorical),
               which(names(df) %in% target_categorical))
  known_idx = which(names(df) %in% c(known_numeric, known_categorical))
  observed_idx = which(names(df) %in%  c(observed_numeric, observed_categorical))
  static_idx = which(names(df) %in% c(static_numeric, static_categorical))
  target_idx =  which(names(df) %in% c(target_numeric, target_categorical))
  list(known = known_t,
       observed = observed_t,
       static = static_t,
       target = target_t,
       input_dim = sum(length(c(time, id, known, observed, static))),
       cat_idxs = cat_idxs,
       known_idx = known_idx,
       observed_idx = observed_idx,
       static_idx = static_idx,
       target_idx =  target_idx,
       cat_dims = purrr::map(cat_idxs, ~length(unique(df[[.x]]))),
       output_dim = length(target_categorical) + length(target_numeric),
       blueprint = processed_roles$blueprint
  )
}

df_to_tensor <- function(df, device) {
  df %>%
    dplyr::mutate_if(is.factor, as.integer) %>%
    dplyr::mutate_if(lubridate::is.Date, as.integer) %>%
    dplyr::mutate_if(lubridate::is.POSIXt, as.integer) %>%
    as.matrix() %>%
    torch::torch_tensor(device = device,dtype = torch::torch_float())
}

#' Configuration for Tft models
#'
#' @param total_time_steps (int) Size of the look-back time window + forecast horizon in steps.
#'   This is the width of Temporal fusion decoder N.
#' @param num_encoder_steps (int) Size of the look-back time window in steps. This is the size
#'   of LSTM encoder.
#' @param hidden_layer_size (int)size of the Internal state layer (default=160).
#' @param dropout_rate dropout rate applied to each nn block (default=0.3)
#' @param stack_size (int) Number of self-attention layers to apply (default=3). Use 1 for
#'   basic TFT.
#' @param num_heads (int) number of interpretable multi-attention head (default=1)
#' @param loss (character or function) Loss function for training within
#'   `"quantile_loss"`, `"pinball_loss"`, `"rmsse_loss"`, `"smape_loss"`
#'   (default to `quantile_loss`)
#' @param quantiles (list) list of quantiles forcasts to be used in quantile loss. (default = `list(0.5)`).
#' @param training_tau (float) training_tau value to be used in pinball loss. (default = 0.3).
#' @param batch_size (int) Number of examples per batch, large batch sizes are
#'   recommended. (default: 1024)
#' @param clip_value If a float is given this will clip the gradient at
#'   clip_value. Pass `NULL` (default) to not clip.
#' @param epochs (int) Number of training epochs.
#' @param drop_last (bool) Whether to drop last batch if not complete during
#'   training
#' @param virtual_batch_size (int) Size of the mini batches used for
#'   Batch Normalization (default=256)
#' @param learn_rate initial learning rate for the optimizer.
#' @param optimizer the optimization method. currently only 'adam' is supported,
#'   you can also pass any torch optimizer function.
#' @param valid_split (float) The fraction of the dataset used for validation.
#' @param verbose (bool) wether to print progress and loss values during
#'   training.
#' @param lr_scheduler if `NULL`, (default) no learning rate decay is used. if `step`
#'   decays the learning rate by `lr_decay` every `step_size` epochs. It can
#'   also be a `torch::lr_scheduler` function that only takes the optimizer
#'   as parameter. The `step` method is called once per epoch.
#' @param lr_decay multiplies the initial learning rate by `lr_decay` every
#'   `step_size` epochs. Unused if `lr_scheduler` is a `torch::lr_scheduler`
#'   or `NULL`.
#' @param step_size number of epoch before modifying learning rate by `lr_decay`.
#'   Unused if `lr_scheduler` is a `torch::lr_scheduler` or `NULL`.
#' @param cat_emb_dim (int or list) Embedding size for categorial features,
#'   broadcasted to each categorical feature, or per categorical feature
#'   when a list of the same size as the categorical features  (default=1)
#' @param checkpoint_epochs checkpoint model weights and architecture every
#'   `checkpoint_epochs`. (default is 10). This may cause large memory usage.
#'   Use `0` to disable checkpoints.
#' @param device the device to use for training. `cpu` or `cuda`. The default (`auto`)
#'   uses `cuda`` if it's available, otherwise uses `cpu`.
#'
#' @return A named list with all hyperparameters of the TabNet implementation.
#'
#' @export
tft_config <- function(batch_size = 256^2,
                       clip_value = NULL,
                       loss = "quantile_loss",
                       epochs = 5,
                       drop_last = FALSE,
                       total_time_steps = NULL,
                       num_encoder_steps = NULL,
                       quantiles = list(0.5),
                       training_tau = 0.3,
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
    training_tau = training_tau,
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
  batch <- list(known_numerics = batch$known_numerics, known_categorical = batch$known_categorical,
                observed_numerics = batch$observed_numerics, observed_categorical = batch$observed_categorical,
                static_numerics = batch$static_numerics, static_categorical = batch$static_categorical,
                target_numerics = batch$target_numerics, target_categorical = batch$target_categorical)
  lapply(batch, function(x) {
    x$to(device = device)
  })
}

train_batch <- function(network, optimizer, batch, config) {
  # forward pass
  output <- network(batch$known_numerics, batch$known_categorical,
                    batch$observed_numerics, batch$observed_categorical,
                    batch$static_numerics, batch$static_categorical,
                    batch$target_numerics, batch$target_categorical)
  actuals <- torch::torch_cat(c(batch$target_numerics[,(config$num_encoder_steps+1):-1,],
                        batch$target_categorical[,(config$num_encoder_steps+1):-1,]), dim=-1)
  loss <- config$loss_fn(output[[1]], actuals)

  # Add the overall sparsity loss (currently lambda_sparse is not in config qnd output[[2]] is NULL)
  # loss <- loss - config$lambda_sparse * output[[2]]

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
  output <- network(batch$known_numerics, batch$known_categorical,
                    batch$observed_numerics, batch$observed_categorical,
                    batch$static_numerics, batch$static_categorical,
                    batch$target_numerics, batch$target_categorical)
  actuals <- torch::torch_cat(c(batch$target_numerics[,(config$num_encoder_steps+1):-1,],
                                batch$target_categorical[,(config$num_encoder_steps+1):-1,]), dim=-1)
  loss <- config$loss_fn(output[[1]], actuals)

  # Add the overall sparsity loss
  # loss <- loss - config$lambda_sparse * output[[2]]

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

tft_initialize <- function(data, config = tft_config()) {

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
    known_idx = data$known_idx,
    observed_idx = data$observed_idx,
    static_idx = data$static_idx,
    target_idx = data$target_idx,
    total_time_steps = config$total_time_steps,
    num_encoder_steps = config$num_encoder_steps,
    quantiles = config$quantiles,
    minibatch_size = config$minibatch_size,
    hidden_layer_size = config$hidden_layer_size,
    dropout_rate = config$dropout_rate,
    stack_size = config$stack_size,
    num_heads = config$num_heads,
    device = device
  )

  # main loop
  metrics <- list()
  checkpoints <- list()


  importances <- tibble::tibble(
    variables = names(data)[1:3],
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

tft_train <- function(obj, data, config = tft_config(), epoch_shift=0L) {
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
    rlang::abort("'valid_split=TRUE' is not implemented yet")
  }

  # training data (can't have changed yet)
  dl <- torch::dataloader(
    torch::tensor_dataset(known_numerics = data$known$numerics, known_categorical = data$known$categorical,
                          observed_numerics = data$observed$numerics, observed_categorical = data$observed$categorical,
                          static_numerics = data$static$numerics, static_categorical = data$static$categorical,
                          target_numerics = data$target$numerics, target_categorical = data$target$categorical),
    batch_size = config$batch_size,
    drop_last = config$drop_last,
    shuffle = TRUE
  )

  # validation data (can't have changed yet)
  # if (has_valid) {
  #   valid_data <- batch_data(valid_data, transform)
  #   valid_dl <- torch::dataloader(
  #     torch::tensor_dataset(x = valid_data$x, y = valid_data$y),
  #     batch_size = config$batch_size,
  #     drop_last = FALSE,
  #     shuffle = FALSE
  #   )
  # }

  # resolve loss (could have changed)
  if (config$loss == "quantile_loss") {
    config$loss_fn <- quantile_loss(config$quantiles)
  } else if (config$loss == "pinball_loss") {
    config$loss_fn <- pinball_loss(config$training_tau)
  } else if (config$loss == "rmsse_loss") {
    config$loss_fn <- rmsse_loss()
  } else if (config$loss == "smape_loss") {
    config$loss_fn <- smape_loss()
  }

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
      coro::loop(for (batch in valid_dl) {
        m <- valid_batch(network, batch_to_device(batch, device), config)
        valid_metrics <- c(valid_metrics, m)
      })
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

  # TODO extract feature importance
  # importance_sample_size <- config$importance_sample_size
  # if (is.null(config$importance_sample_size) && data$target$numerics$shape[1] > 1e5) {
  #   rlang::warn(c(glue::glue("Computing importances for a dataset with size {data$x$shape[1]}."),
  #                 "This can consume too much memory. We are going to use a sample of size 1e5",
  #                 "You can disable this message by using the `importance_sample_size` argument."))
  #   importance_sample_size <- 1e5
  # }
  # indexes <- torch::torch_randint(
  #   1, data$target$numerics$shape[1], min(importance_sample_size, data$target$numerics$shape[1]),
  #   dtype = torch::torch_long()
  # )
  # importances <- tibble::tibble(
  #   variables = colnames(x),
  #   importance = compute_feature_importance(network, data$x[indexes,..])
  # )

  list(
    network = network,
    metrics = metrics,
    config = config,
    checkpoints = checkpoints,
    importances = obj$fit$importances
  )
}

predict_impl <- function(obj, recipe, processed) {

  network <- obj$fit$network
  network$eval()
  if (obj$fit$config$device %in% c("auto", "cuda")) {
    if (torch::cuda_is_available())
      network$to(device="cuda")
  }
  dl <- torch::dataloader(
    torch::tensor_dataset(known_numerics = processed$known$numerics, known_categorical = processed$known$categorical,
                          observed_numerics = processed$observed$numerics, observed_categorical = processed$observed$categorical,
                          static_numerics = processed$static$numerics, static_categorical = processed$static$categorical,
                          target_numerics = processed$target$numerics, target_categorical = processed$target$categorical),
    batch_size = obj$fit$config$batch_size,
    drop_last = obj$fit$config$drop_last)

  batch_outputs <- c()
  coro::loop(for (batch in dl){
    batch_output <- network(batch$known_numerics, batch$known_categorical,
                       batch$observed_numerics, batch$observed_categorical,
                       batch$static_numerics, batch$static_categorical,
                       batch$target_numerics, batch$target_categorical)
    batch_outputs <- c(batch_outputs,batch_output[[1]])
  })
  torch::torch_cat(batch_outputs)
}

predict_impl_numeric <- function(obj, recipe, processed) {
  p <- predict_impl(obj, recipe, processed)
  p <- p$to(device="cpu") %>% as.array
  # spruce_numeric is not compliant with multi-horizon forecast
  #hardhat::spruce_numeric(p)
  tibble::tibble(.pred=p)
}

get_blueprint_levels <- function(obj) {
  levels(obj$blueprint$ptypes$outcomes[[1]])
}

predict_impl_prob <- function(obj, recipe, processed) {
  p <- predict_impl(obj, recipe, processed)
  p <- torch::nnf_softmax(p, dim = 2)
  p <- p$to(device="cpu") %>% as.array
  # TODO spurce_prob is not compliant with multi-horizon forecast
  hardhat::spruce_prob(get_blueprint_levels(obj), p)
}

predict_impl_class <- function(obj, recipe, processed) {
  p <- predict_impl(obj, recipe, processed)
  p <- torch::torch_max(p, dim = 2)
  p <- as.integer(p[[2]]$to(device="cpu"))
  p <- get_blueprint_levels(obj)[p]
  p <- factor(p, levels = get_blueprint_levels(obj))
  # TODO spruce_class is not compliant with multi-horizon forecast
  hardhat::spruce_class(p)

}

