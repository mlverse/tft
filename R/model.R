
temporal_fusion_transformer <- function(dataset, ...) {

  config <- tft_config(...)

  n_features <- get_n_features(dataset[1][[1]])
  feature_sizes <- dataset$feature_sizes

  module <- temporal_fusion_transformer_model %>%
    luz::setup(
      loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
      optimizer = config$optimizer,
      metrics = list(
        luz_quantile_loss(quantile = 0.1,1),
        luz_quantile_loss(quantile = 0.5,2),
        luz_quantile_loss(quantile = 0.9,3)
      )
    ) %>%
    luz::set_hparams(
      num_features = n_features,
      feature_sizes = feature_sizes,
      hidden_state_size = config$hidden_state_size,
      dropout = config$dropout,
      num_quantiles = 3,
      num_heads = config$num_attention_heads,
      num_lstm_layers = config$num_lstm_layers
    ) %>%
    luz::setup(
      loss = quantile_loss(quantiles = c(0.1, 0.5, 0.9)),
      optimizer = config$optimizer,
      metrics = list(
        luz_quantile_loss(quantile = 0.1,1),
        luz_quantile_loss(quantile = 0.5,2),
        luz_quantile_loss(quantile = 0.9,3)
      )
    )

  module
}
